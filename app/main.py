"""Ask a question, get two answers with the page each one came from."""

import asyncio
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config as cfg
from app.corpus import paragraphs
from app.rerank import rerank
from app.search import BM25, Vector, hybrid, tokens

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
OLLAMA = cfg.CHAT_URL
MODEL = cfg.MODEL
RESULTS = ROOT / "eval" / "results.json"
# The model runs on one GPU; two generations at a time is all it can hold.
GENERATIONS = threading.Semaphore(2)
STREAMS = threading.Semaphore(
    1
)  # one stream at a time: it already holds two generations
CACHE = {}

app = FastAPI(title="manual-rag-evals")
chunks = list(paragraphs())
bm25 = BM25(chunks)
vector = Vector(chunks)

SYSTEM = (
    "You answer questions about aircraft inspection and repair using only the "
    "passages given. Two sentences at most. End with the page you used, written "
    "exactly as it appears in brackets, for example (Page 6-30). If the passages "
    "do not answer the question, say that instead of guessing."
)


class Ask(BaseModel):
    q: str = Field(min_length=3, max_length=400)


def passages(ranked):
    """One passage per page: two chunks of the same page crowd out the rest."""
    seen, rows = set(), []
    for _, c in ranked:
        if c["page"] in seen:
            continue
        seen.add(c["page"])
        rows.append(f"[Page {c['page']}] {c['text']}")
    return "\n\n".join(rows)


def answer(question, ranked):
    prompt = f"Passages:\n{passages(ranked)}\n\nQuestion: {question}"
    try:
        reply = httpx.post(
            OLLAMA,
            json={
                "model": MODEL,
                "stream": False,
                # Thinking burns the whole token budget before any answer appears,
                # and 8k context keeps the 27B on the GPU instead of the CPU.
                "think": False,
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 160},
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=60,
        )
        reply.raise_for_status()
        text = reply.json()["message"]["content"].strip()
        if not text:
            raise ValueError("empty reply")
        return text, False
    except Exception:
        return (
            "The model did not answer, so here is the passage itself: "
            f"{ranked[0][1]['text'][:400]} (Page {ranked[0][1]['page']})",
            True,
        )


SENTENCE = re.compile(r"(?<=[.!?])\s+")


def evidence(question, ranked):
    """The retrieved passage whose sentence best matches the question, with those
    sentences marked so a reader can see where the answer came from."""
    asked = set(tokens(question))
    best = None
    for rank, (_, chunk) in enumerate(ranked):
        parts = [s.strip() for s in SENTENCE.split(chunk["text"]) if s.strip()]
        # a chunk can start mid-sentence; a lowercase fragment is not worth reading
        if (
            parts
            and not parts[0][:1].isupper()
            and not re.match(r"[a-z]\.\s", parts[0])
        ):
            parts = parts[1:]
        scores = [len(asked & set(tokens(s))) for s in parts]
        peak = max(scores, default=0)
        if best is None or peak > best[0]:
            best = (peak, chunk, parts, scores)
    peak, chunk, parts, scores = best
    floor = max(2, peak - 1)
    return {
        "page": chunk["page"],
        "sentences": [
            {"text": s, "hit": score >= floor} for s, score in zip(parts, scores)
        ],
    }


REFUSE_BELOW = cfg.REFUSE_BELOW
log = logging.getLogger("rag")
logging.basicConfig(level=cfg.LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
CITED = re.compile(r"\(Page (\d+-\d+)\)")


def grounded(text, ranked):
    """True when the answer names a page the model was actually shown."""
    found = CITED.search(text)
    return bool(found and found.group(1) in {c["page"] for _, c in ranked})


REFUSAL = "The manual does not cover this question: nothing in AC 43.13-1B answers it."


def confidence(question):
    """Cosine of the closest paragraph. The manual's own vocabulary scores higher."""
    hit = vector.search(question, 1)
    return hit[0][0] if hit else 0.0


def column(question, ranked, retrieval_ms, top=1.0):
    log.info("ask %r top=%.3f", question[:80], top)
    if top < REFUSE_BELOW:
        return {
            "answer": REFUSAL,
            "pages": list(dict.fromkeys(c["page"] for _, c in ranked[:3])),
            "evidence": evidence(question, ranked[:3]),
            "retrieval_ms": retrieval_ms,
            "answer_ms": 0,
            "fallback": False,
            "refused": True,
            "grounded": False,
        }
    started = time.perf_counter()
    key = (question, tuple(c["page"] for _, c in ranked))
    if key in CACHE:
        text, fallback = CACHE[key]
    else:
        text, fallback = answer(question, ranked)
        if len(CACHE) >= 32:
            CACHE.clear()
        CACHE[key] = (text, fallback)
    return {
        "answer": text,
        "pages": list(dict.fromkeys(c["page"] for _, c in ranked)),
        "evidence": evidence(question, ranked),
        "retrieval_ms": retrieval_ms,
        "answer_ms": round((time.perf_counter() - started) * 1000),
        "fallback": fallback,
        "refused": False,
        "grounded": grounded(text, ranked),
    }


def search_timed(search, *args):
    started = time.perf_counter()
    ranked = search(*args, 5)
    return ranked, round((time.perf_counter() - started) * 1000)


def reranked(q, k=5):
    """Third arm: the same hybrid list, read by the model before it is kept."""
    return rerank(q, hybrid(bm25, vector, q, 30), keep=k)


STREAM_OPTIONS = {"temperature": 0, "num_ctx": 8192, "num_predict": 160}


def stream_answer(question, ranked, emit):
    """Tokens as they arrive, so a column fills while the others are still thinking."""
    text = ""
    try:
        with GENERATIONS:
            payload = {
                "model": MODEL,
                "stream": True,
                "think": False,
                "options": STREAM_OPTIONS,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"Passages:\n{passages(ranked)}\n\nQuestion: {question}",
                    },
                ],
            }
            with httpx.stream("POST", OLLAMA, json=payload, timeout=90) as response:
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    piece = json.loads(line)["message"]["content"]
                    if piece:
                        text += piece
                        emit({"kind": "delta", "text": piece})
    except Exception:
        text = ""
    if text.strip():
        return text, False
    # The model did not answer: show the passage it would have used, and say so.
    return (
        f"The model did not answer, so here is the passage itself: "
        f"{ranked[0][1]['text'][:400]} (Page {ranked[0][1]['page']})",
        True,
    )


@app.post("/api/ask/stream")
async def ask_stream(body: Ask):
    """One event stream for all three arms: meta, then tokens, then done."""
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    if not STREAMS.acquire(blocking=False):
        raise HTTPException(429, "two answers at a time; try again in a few seconds")
    top = confidence(body.q)

    def emit(payload):
        loop.call_soon_threadsafe(events.put_nowait, payload)

    def arm(name, ranked, retrieval_ms, top):
        emit(
            {
                "arm": name,
                "kind": "meta",
                "retrieval_ms": retrieval_ms,
                "pages": list(dict.fromkeys(c["page"] for _, c in ranked)),
                "evidence": evidence(body.q, ranked),
                "refused": top < REFUSE_BELOW,
            }
        )
        if top < REFUSE_BELOW:
            emit(
                {
                    "arm": name,
                    "kind": "done",
                    "answer": REFUSAL,
                    "answer_ms": 0,
                    "fallback": False,
                    "refused": True,
                    "grounded": False,
                }
            )
            return
        key = (body.q, tuple(c["page"] for _, c in ranked))
        if key in CACHE:
            text, fallback = CACHE[key]
            emit(
                {
                    "arm": name,
                    "kind": "done",
                    "answer": text,
                    "answer_ms": 0,
                    "fallback": fallback,
                    "refused": False,
                    "grounded": grounded(text, ranked),
                }
            )
            return
        started = time.perf_counter()
        text, fallback = stream_answer(
            body.q, ranked, lambda payload: emit({"arm": name, **payload})
        )
        if len(CACHE) >= 32:
            CACHE.clear()
        CACHE[key] = (text, fallback)
        emit(
            {
                "arm": name,
                "kind": "done",
                "answer": text,
                "answer_ms": round((time.perf_counter() - started) * 1000),
                "fallback": fallback,
                "refused": False,
                "grounded": grounded(text, ranked),
            }
        )

    def run(name, search, *args):
        try:
            ranked, ms = search_timed(search, *args)
            arm(name, ranked, ms, top)
        except Exception as error:  # never leave a column spinning
            emit(
                {
                    "arm": name,
                    "kind": "done",
                    "answer": f"failed: {error}",
                    "answer_ms": 0,
                    "fallback": True,
                }
            )

    def run_rerank():
        if top < REFUSE_BELOW:
            emit(
                {
                    "arm": "C",
                    "kind": "meta",
                    "retrieval_ms": 0,
                    "pages": [],
                    "evidence": [],
                    "refused": True,
                }
            )
            emit(
                {
                    "arm": "C",
                    "kind": "done",
                    "answer": REFUSAL,
                    "answer_ms": 0,
                    "fallback": False,
                    "refused": True,
                    "grounded": False,
                }
            )
            return
        emit({"arm": "C", "kind": "status", "text": "reading the top 30"})
        run("C", reranked, body.q)

    for name, args in (
        ("A", (vector.search, body.q)),
        ("B", (hybrid, bm25, vector, body.q)),
    ):
        threading.Thread(target=run, args=(name, *args), daemon=True).start()
    threading.Thread(target=run_rerank, daemon=True).start()

    async def events_out():
        finished = 0
        try:
            while finished < 3:
                payload = await events.get()
                if payload["kind"] == "done":
                    finished += 1
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            STREAMS.release()

    return StreamingResponse(events_out(), media_type="text/event-stream")


@app.post("/api/ask")
def ask(body: Ask):
    # Two arms share nothing after retrieval, so they run side by side: 2 s
    # instead of 4. One GPU holds two generations; beyond that, say no.
    if not GENERATIONS.acquire(blocking=False):
        raise HTTPException(429, "two answers at a time; try again in a few seconds")
    try:
        started = time.perf_counter()
        top = confidence(body.q)

        with ThreadPoolExecutor(2) as pool:
            left = pool.submit(
                lambda: column(body.q, *search_timed(vector.search, body.q), top=top)
            )
            right = pool.submit(
                lambda: column(
                    body.q, *search_timed(hybrid, bm25, vector, body.q), top=top
                )
            )
            third = pool.submit(lambda: search_timed(reranked, body.q))
            first, second = left.result(), right.result()
            ranked_c, rerank_ms = third.result()
        # C is answered after A and B, so three answers never share the GPU
        return {
            "question": body.q,
            "vector": first,
            "hybrid": second,
            "rerank": column(body.q, ranked_c, rerank_ms, top=top),
            "total_ms": round((time.perf_counter() - started) * 1000),
        }
    finally:
        GENERATIONS.release()


@app.get("/health")
def health():
    """Is the app able to answer: corpus loaded, model reachable, threshold set."""
    try:
        tags = httpx.get(f"{cfg.OLLAMA_HOST}/api/tags", timeout=3).json()
        models = [m.get("name", "") for m in tags.get("models", [])]
        reachable = True
    except Exception:
        models, reachable = [], False
    base = MODEL.split(":")[0]
    present = any(name.split(":")[0] == base for name in models)
    return {
        "ok": reachable and present,
        "chunks": len(chunks),
        "model": MODEL,
        "model_present": present,
        "refuse_below": REFUSE_BELOW,
    }


@app.get("/api/results")
def results():
    """Nine aggregates for the bars. The per-question rows stay in the file."""
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    fields = ("hit_at_5", "hit_at_5_ci", "mrr", "citation", "citation_ci")
    return {
        "questions": data.get("questions"),
        **{
            arm: {field: values.get(field) for field in fields}
            for arm, values in data.items()
            if isinstance(values, dict) and "hit_at_5" in values
        },
    }


@app.get("/")
def index():
    return FileResponse(PUBLIC / "index.html")


app.mount("/", StaticFiles(directory=PUBLIC), name="static")
