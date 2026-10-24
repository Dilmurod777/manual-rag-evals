"""Ask a question, get two answers with the page each one came from."""

import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.corpus import paragraphs
from app.search import BM25, Vector, hybrid

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("RAG_MODEL", "qwen3.8-27b")
RESULTS = ROOT / "eval" / "results.json"

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
    q: str


def passages(ranked):
    return "\n\n".join(f"[Page {c['page']}] {c['text']}" for _, c in ranked)


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
    except Exception:
        text = f"No model running, so here is the passage itself: {ranked[0][1]['text'][:400]} (Page {ranked[0][1]['page']})"
    return text


def column(question, ranked):
    return {
        "answer": answer(question, ranked),
        "pages": list(dict.fromkeys(c["page"] for _, c in ranked)),
        "top": ranked[0][1]["text"][:420],
    }


@app.post("/api/ask")
def ask(body: Ask):
    return {
        "question": body.q,
        "vector": column(body.q, vector.search(body.q, 5)),
        "hybrid": column(body.q, hybrid(bm25, vector, body.q, 5)),
    }


@app.get("/api/results")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@app.get("/")
def index():
    return FileResponse(PUBLIC / "index.html")


app.mount("/", StaticFiles(directory=PUBLIC), name="static")
