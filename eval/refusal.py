"""Fit the refusal threshold on half the unanswerable questions, report on the other half."""

import json
import sys
from pathlib import Path

from app.corpus import paragraphs
from app.search import BM25, Vector

HERE = Path(__file__).resolve().parent
REFUSE_BELOW = 0.58


def load(name):
    return [
        json.loads(line)
        for line in (HERE / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    chunks = list(paragraphs())
    bm25, vector = BM25(chunks), Vector(chunks)

    def top(row):
        return max(score for score, _ in vector.search(row["q"], 5))

    gold = [top(row) for row in load("gold.jsonl")]
    blind = [top(row) for row in load("blind.jsonl")]
    unknown = [top(row) for row in load("unanswerable.jsonl")]

    fit, held = unknown[:20], unknown[20:]
    ceiling = min(gold + blind)  # no answerable question may fall below the line
    fitted = round((max(fit) + ceiling) / 2, 3) if max(fit) < ceiling else round(ceiling - 0.005, 3)

    caught = sum(1 for score in held if score < fitted)
    print(f"answerable  min {ceiling:.3f} over {len(gold) + len(blind)} questions")
    print(f"fit half    max {max(fit):.3f}  -> threshold {fitted}")
    print(
        f"held half   {caught}/{len(held)} below {fitted}  (scores {[round(s, 3) for s in held]})"
    )
    return 0 if caught else 1


if __name__ == "__main__":
    sys.exit(main())
