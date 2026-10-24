"""Score vector against hybrid: does retrieval find the page, and does the
answer cite it?"""

import json
import re
import time
from pathlib import Path

from app.corpus import paragraphs
from app.main import answer
from app.search import BM25, Vector, hybrid

GOLD = Path(__file__).with_name("gold.jsonl")
OUT = Path(__file__).with_name("results.json")
CITED = re.compile(r"\(Page (\d+-\d+)\)")


def main():
    gold = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = list(paragraphs())
    bm25 = BM25(chunks)
    vector = Vector(chunks)

    results = {"questions": len(gold)}
    for mode in ("vector", "hybrid"):
        hits, rr, cited, misses, rows = 0, 0.0, 0, [], []
        for row in gold:
            started = time.perf_counter()
            if mode == "vector":
                ranked = vector.search(row["q"], 5)
            else:
                ranked = hybrid(bm25, vector, row["q"], 5)
            retrieval_ms = round((time.perf_counter() - started) * 1000)
            pages = [chunk["page"] for _, chunk in ranked]
            hit = row["page"] in pages
            if hit:
                hits += 1
                rr += 1 / (pages.index(row["page"]) + 1)
            else:
                misses.append({"q": row["q"], "page": row["page"], "got": pages})
            started = time.perf_counter()
            text, _ = answer(row["q"], ranked)
            answer_ms = round((time.perf_counter() - started) * 1000)
            found = CITED.search(text)
            right = bool(found and found.group(1) == row["page"])
            cited += right
            rows.append(
                {
                    "q": row["q"],
                    "gold": row["page"],
                    "rank": pages.index(row["page"]) + 1 if hit else 0,
                    "cited": found.group(1) if found else None,
                    "right": right,
                    "retrieval_ms": retrieval_ms,
                    "answer_ms": answer_ms,
                }
            )
        results[mode] = {
            "hit_at_5": round(hits / len(gold), 4),
            "mrr": round(rr / len(gold), 4),
            "citation": round(cited / len(gold), 4),
            "misses": misses,
            "rows": rows,
        }
        print(
            f"{mode:<7} hit@5 {hits}/{len(gold)}  mrr {rr / len(gold):.3f}  "
            f"cited {cited}/{len(gold)}"
        )

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for mode, data in results.items():
        if mode == "questions":
            continue
        for miss in data["misses"]:
            print(
                f"  {mode} miss  {miss['page']:>6}  {miss['q'][:60]}  -> {', '.join(miss['got'])}"
            )


if __name__ == "__main__":
    main()
