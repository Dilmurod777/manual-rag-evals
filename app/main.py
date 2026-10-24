"""Ask a question, get two answers with the page each one came from."""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.corpus import paragraphs
from app.rerank import rerank
from app.search import BM25, Vector, hybrid, tokens

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("RAG_MODEL", "qwen3.8-27b")
RESULTS = ROOT / "eval" / "results.json"
# The model runs on one GPU; two generations at a time is all it can hold.
GENERATIONS = threading.Semaphore(2)

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
    rerank: bool = False


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


def column(question, ranked, retrieval_ms):
    started = time.perf_counter()
    text, fallback = answer(question, ranked)
    return {
        "answer": text,
        "pages": list(dict.fromkeys(c["page"] for _, c in ranked)),
        "evidence": evidence(question, ranked),
        "retrieval_ms": retrieval_ms,
        "answer_ms": round((time.perf_counter() - started) * 1000),
        "fallback": fallback,
    }


def search_timed(search, *args):
    started = time.perf_counter()
    ranked = search(*args, 5)
    return ranked, round((time.perf_counter() - started) * 1000)


def reranked(q, k=5):
    """Third arm: the same hybrid list, read by the model before it is kept."""
    return rerank(q, hybrid(bm25, vector, q, 30), keep=k)


@app.post("/api/ask")
def ask(body: Ask):
    # Two arms share nothing after retrieval, so they run side by side: 2 s
    # instead of 4. One GPU holds two generations; beyond that, say no.
    if not GENERATIONS.acquire(blocking=False):
        raise HTTPException(429, "two answers at a time; try again in a few seconds")
    try:
        started = time.perf_counter()

        with ThreadPoolExecutor(2) as pool:
            left = pool.submit(
                lambda: column(body.q, *search_timed(vector.search, body.q))
            )
            right = pool.submit(
                lambda: column(body.q, *search_timed(hybrid, bm25, vector, body.q))
            )
            third = pool.submit(lambda: search_timed(reranked, body.q))
            first, second = left.result(), right.result()
            ranked_c, rerank_ms = third.result()
        # C is answered after A and B, so three answers never share the GPU
        return {
            "question": body.q,
            "vector": first,
            "hybrid": second,
            "rerank": column(body.q, ranked_c, rerank_ms),
            "total_ms": round((time.perf_counter() - started) * 1000),
        }
    finally:
        GENERATIONS.release()


@app.get("/api/results")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@app.get("/")
def index():
    return FileResponse(PUBLIC / "index.html")


app.mount("/", StaticFiles(directory=PUBLIC), name="static")
