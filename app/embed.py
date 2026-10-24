"""Embed every chunk with the local Ollama and cache it in data/embeddings.npz."""

import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from app.corpus import paragraphs

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "embeddings.npz"
MODEL = "qwen3-embedding:0.6b"
URL = "http://127.0.0.1:11434/api/embed"


def fingerprint(chunks):
    """The ids are positional, so a cache from different text must be refused."""
    return hashlib.sha256("".join(c["text"] for c in chunks).encode()).hexdigest()


def embed(texts):
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    request = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        vectors = np.array(json.loads(response.read())["embeddings"], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def main():
    chunks = list(paragraphs())
    started = time.time()
    parts = []
    for start in range(0, len(chunks), 32):
        batch = chunks[start : start + 32]
        parts.append(embed([c["text"] for c in batch]))
        print(f"{start + len(batch)}/{len(chunks)}", flush=True)
    vectors = np.vstack(parts)
    np.savez_compressed(
        OUT,
        vec=vectors,
        ids=np.array([c["id"] for c in chunks]),
        fingerprint=fingerprint(chunks),
    )
    print(f"{vectors.shape} in {time.time() - started:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
