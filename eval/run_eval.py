"""Score vector against hybrid: does retrieval find the page, and does the
answer cite it?"""

import json
import re
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
        hits, rr, cited, misses = 0, 0.0, 0, []
        for row in gold:
            if mode == "vector":
                ranked = vector.search(row["q"], 5)
            else:
                ranked = hybrid(bm25, vector, row["q"], 5)
            pages = [chunk["page"] for _, chunk in ranked]
            if row["page"] in pages:
                hits += 1
                rr += 1 / (pages.index(row["page"]) + 1)
            else:
                misses.append({"q": row["q"], "page": row["page"], "got": pages})
            text, _ = answer(row["q"], ranked)
            found = CITED.search(text)
            cited += bool(found and found.group(1) == row["page"])
        results[mode] = {
            "hit_at_5": round(hits / len(gold), 4),
            "mrr": round(rr / len(gold), 4),
            "citation": round(cited / len(gold), 4),
            "misses": misses,
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
