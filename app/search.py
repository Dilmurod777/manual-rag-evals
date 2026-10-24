"""BM25 over the manual's paragraphs. No dependencies."""

import math
import re
import sys
from collections import Counter

from app.corpus import paragraphs

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
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for term in terms:
                tf = doc.get(term)
                if not tf:
                    continue
                idf = math.log(
                    1 + (len(self.docs) - self.df[term] + 0.5) / (self.df[term] + 0.5)
                )
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


if __name__ == "__main__":
    index = BM25(list(paragraphs()))
    for query in sys.argv[1:]:
        print(f"\n{query}")
        for score, chunk in index.search(query, 5):
            print(f"  {chunk['page']:>7}  {score:6.2f}  {chunk['text'][:110]}")
