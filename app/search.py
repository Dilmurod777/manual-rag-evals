"""BM25, vectors, and hybrid retrieval over the manual's paragraphs."""

import argparse
import math
import re
from collections import Counter

import numpy as np

from app.corpus import paragraphs
from app.embed import OUT, embed, fingerprint

STOP = set(
    "the a an and or of to in for on with is are be by as at from that this it its not "
    "any all if when which who what how do does can may should must will shall".split()
)


def tokens(text):
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 1 and t not in STOP
    ]


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.docs = [Counter(tokens(c["text"])) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        self.df = Counter()
        for doc in self.docs:
            self.df.update(doc.keys())

    def search(self, query, k=5):
        terms = tokens(query)
        total = len(self.docs)
        idfs = {
            term: math.log(1 + (total - self.df[term] + 0.5) / (self.df[term] + 0.5))
            for term in terms
            if self.df[term]
        }
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for term, idf in idfs.items():
                tf = doc.get(term)
                if not tf:
                    continue
                score += (
                    idf
                    * tf
                    * (self.k1 + 1)
                    / (
                        tf
                        + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                    )
                )
            if score:
                scored.append((score, self.chunks[i]))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]


class Vector:
    def __init__(self, chunks):
        # Loaded on first search, so importing the package does not need the cache.
        self.source = chunks
        self.vec = None
        self.chunks = None

    def load(self):
        if self.vec is not None:
            return
        saved = np.load(OUT)
        if str(saved["fingerprint"]) != fingerprint(self.source):
            raise RuntimeError(
                "data/embeddings.npz was built from different text; "
                "run `python app/embed.py` before searching"
            )
        by_id = {c["id"]: c for c in self.source}
        self.vec = saved["vec"]
        self.chunks = [by_id[str(i)] for i in saved["ids"]]

    def search(self, query, k=5):
        self.load()
        scores = self.vec @ embed([query])[0]
        order = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in order]


def hybrid(bm25, vector, query, k=5, depth=30):
    """Reciprocal rank fusion of the two ranked lists."""
    fused, chunks = {}, {}
    for ranked in (bm25.search(query, depth), vector.search(query, depth)):
        for rank, (_, chunk) in enumerate(ranked):
            fused[chunk["id"]] = fused.get(chunk["id"], 0.0) + 1 / (60 + rank + 1)
            chunks[chunk["id"]] = chunk
    top = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:k]
    return [(score, chunks[chunk_id]) for chunk_id, score in top]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--mode", default="bm25", choices=["bm25", "vec", "hybrid"])
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--chars", type=int, default=110)
    args = parser.parse_args()

    chunks = list(paragraphs())
    bm25 = BM25(chunks)
    vector = Vector(chunks) if args.mode != "bm25" else None
    for query in args.queries:
        print(f"\n{query}")
        if args.mode == "vec":
            ranked = vector.search(query, args.k)
        elif args.mode == "hybrid":
            ranked = hybrid(bm25, vector, query, args.k)
        else:
            ranked = bm25.search(query, args.k)
        for score, chunk in ranked:
            print(f"  {chunk['page']:>7}  {score:6.3f}  {chunk['text'][: args.chars]}")


if __name__ == "__main__":
    main()
