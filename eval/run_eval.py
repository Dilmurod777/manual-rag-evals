"""Score vector against hybrid retrieval on the gold set."""

import json
from pathlib import Path

from app.corpus import paragraphs
from app.search import BM25, Vector, hybrid

GOLD = Path(__file__).with_name("gold.jsonl")
OUT = Path(__file__).with_name("results.json")


def ranked_pages(mode, query, bm25, vector, k=5):
    if mode == "vector":
        ranked = vector.search(query, k)
    else:
        ranked = hybrid(bm25, vector, query, k)
    return [chunk["page"] for _, chunk in ranked]


def main():
    gold = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = list(paragraphs())
    bm25 = BM25(chunks)
    vector = Vector(chunks)

    results = {}
    for mode in ("vector", "hybrid"):
        hits, rr, misses = 0, 0.0, []
        for row in gold:
            pages = ranked_pages(mode, row["q"], bm25, vector)
            if row["page"] in pages:
                hits += 1
                rr += 1 / (pages.index(row["page"]) + 1)
            else:
                misses.append({"q": row["q"], "page": row["page"], "got": pages})
        results[mode] = {
            "hit_at_5": round(hits / len(gold), 4),
            "mrr": round(rr / len(gold), 4),
            "misses": misses,
        }
        print(f"{mode:<7} hit@5 {hits}/{len(gold)}  mrr {rr / len(gold):.3f}")

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for mode, data in results.items():
        for miss in data["misses"]:
            print(
                f"  {mode} miss  {miss['page']:>6}  {miss['q'][:60]}  -> {', '.join(miss['got'])}"
            )


if __name__ == "__main__":
    main()
