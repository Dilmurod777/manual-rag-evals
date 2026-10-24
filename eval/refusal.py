"""Does the manual cover the question? Score the gate, and what the model said before it."""

import json
import re
import sys
from pathlib import Path

from app.corpus import paragraphs
from app.main import answer
from app.search import BM25, Vector, hybrid

HERE = Path(__file__).resolve().parent
REFUSE_BELOW = 0.58
REFUSED = re.compile(
    r"(do|does) not (cover|contain|answer|specify|address|describe)|no passage|cannot find|cannot answer",
    re.I,
)


def load(name):
    return [
        json.loads(line)
        for line in (HERE / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    chunks = list(paragraphs())
    bm25 = BM25(chunks)
    vector = Vector(chunks)

    def tops(rows):
        return [max(score for score, _ in vector.search(row["q"], 5)) for row in rows]

    gold, blind, unknown = (
        load("gold.jsonl"),
        load("blind.jsonl"),
        load("unanswerable.jsonl"),
    )
    for label, rows in (("gold", gold), ("blind", blind), ("unanswerable", unknown)):
        scores = tops(rows)
        below = sum(1 for s in scores if s < REFUSE_BELOW)
        print(
            f"gate  {label:<13} {below:>2}/{len(rows):<3} below {REFUSE_BELOW}  min {min(scores):.3f}"
        )

    if "--model" in sys.argv:
        right = 0
        for row in unknown:
            text, _ = answer(row["q"], hybrid(bm25, vector, row["q"], 5))
            if REFUSED.search(text):
                right += 1
            else:
                print(f"  invented: {row['q'][:50]} -> {text[:90]}")
        print(f"model refused {right}/{len(unknown)} without the gate")


if __name__ == "__main__":
    sys.exit(main())
