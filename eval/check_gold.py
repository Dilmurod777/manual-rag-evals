"""Every gold row must be provable: its evidence has to exist on its own page."""

import json
from collections import defaultdict
from pathlib import Path

from app.corpus import paragraphs

GOLD = Path(__file__).resolve().parent / "gold.jsonl"


def flat(text):
    return " ".join(text.split())


def main():
    rows = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_page = defaultdict(list)
    for chunk in paragraphs():
        by_page[chunk["page"]].append(chunk["text"])
    pages = {page: flat(" ".join(parts)) for page, parts in by_page.items()}

    failed = 0
    for row in rows:
        text = pages.get(row["page"], "")
        if flat(row["evidence"]) in text:
            print(f"ok   {row['page']:>7}  {row['q'][:70]}")
        else:
            failed += 1
            print(f"FAIL {row['page']:>7}  {row['q'][:70]}")
    print(f"\n{len(rows) - failed}/{len(rows)} verified, {len(rows)} rows")
    chapters = sorted({row["page"].split("-")[0] for row in rows})
    print(f"chapters: {', '.join(chapters)}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
