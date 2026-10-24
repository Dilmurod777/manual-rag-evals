"""Listwise rerank with the model that writes the answers.

No cross-encoder is installed, and a 27B judge over short snippets is the honest
substitute: it sees the question and twelve candidate paragraphs and picks five.
"""

import os
import re

import httpx

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("RAG_MODEL", "qwen3.8-27b")

PROMPT = """A mechanic asks: {q}

Below are {n} paragraphs from an aircraft repair manual. Pick the five that answer the
question, best first. Use only paragraphs that answer it; do not pick one just because it
mentions the same part.

{candidates}

Reply with five numbers separated by commas, nothing else."""


def rerank(question, ranked, depth=12, keep=5):
    candidates = ranked[:depth]
    if len(candidates) <= keep:
        return ranked
    listing = "\n\n".join(
        f"[{i}] Page {chunk['page']}: {chunk['text'][:300]}"
        for i, (_, chunk) in enumerate(candidates)
    )
    payload = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 24},
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(
                    q=question, n=len(candidates), candidates=listing
                ),
            }
        ],
    }
    try:
        reply = httpx.post(OLLAMA, json=payload, timeout=60).json()
        order = [int(n) for n in re.findall(r"\d+", reply["message"]["content"])]
    except Exception:
        return ranked[:keep]
    picked, seen, pages = [], set(), set()
    for i in order:
        if 0 <= i < len(candidates) and i not in seen:
            chunk = candidates[i][1]
            if chunk["page"] in pages:
                continue  # five passages means five pages, not five slices of one
            seen.add(i)
            pages.add(chunk["page"])
            picked.append(candidates[i])
    # pad from the original order so every arm still shows five passages
    for pair in candidates:
        if len(picked) == keep:
            break
        if pair not in picked and pair[1]["page"] not in pages:
            pages.add(pair[1]["page"])
            picked.append(pair)
    return picked[:keep]
