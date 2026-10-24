"""Read the AC 43.13-1B text into citable paragraphs.

Every paragraph carries the manual's own footer label, so an answer can cite
"Page 6-30" instead of a PDF index.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = ROOT / "data" / "ac43-13-1b.txt"

PAGE_LABEL = re.compile(r"Page\s+(\d+-\d+)")
PAR_START = re.compile(r"^\s*(\d+-\d+)\.\s")
RUNNING = re.compile(
    r"^\s*(AC 43\.13-1B\b.*|9/8/98\s*|Page\s+\d+-\d+(\s+Par\s+\d+-\d+)?|Par\s+\d+-\d+(\s+Page\s+\d+-\d+)?)\s*$"
)
HEADER = re.compile(
    r"(?m)^\s*(?:\d{1,2}/\d{1,2}/\d{2,4}\s+)?AC 43\.13-1B(?:\s+CHG\s*\d)?"
    r"(?:\s+\d{1,2}/\d{1,2}/\d{2,4})?\s*"
)


def pages():
    raw = TEXT.read_text(encoding="utf-8", errors="replace").split("\f")
    for index, text in enumerate(raw, start=1):
        if len(text.strip()) < 40:
            continue
        # the footer is the last non-empty line
        lines = [line for line in text.splitlines() if not RUNNING.match(line)]
        body = "\n".join(lines).strip()
        label = PAGE_LABEL.search(text)
        yield {
            "pdf_page": index,
            "page": label.group(1) if label else f"pdf-{index}",
            "body": body,
        }


def paragraphs():
    n = 0
    for page in pages():
        for block in re.split(r"(?m)(?=^\s*\d+-\d+\.\s)", page["body"]):
            text = re.sub(
                r"[ \t]+", " ", HEADER.sub("", block).replace("\n", " ")
            ).strip()
            if len(text) < 80 or re.search(r"\.{4,}", text):
                continue
            par = PAR_START.match(text)
            for piece in split_long(text):
                n += 1
                yield {
                    "id": f"{page['page']}#{n}",
                    "page": page["page"],
                    "pdf_page": page["pdf_page"],
                    "par": par.group(1) if par else None,
                    "text": piece,
                }


def split_long(text, limit=1100):
    """A paragraph that runs past ~1100 characters is split at sentence ends."""
    while len(text) > limit:
        cut = text.rfind(". ", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        yield text[: cut + 1].strip()
        text = text[cut + 1 :].strip()
    if text:
        yield text


def main():
    chunks = list(paragraphs())
    pages_n = len({c["pdf_page"] for c in chunks})
    print(
        json.dumps(
            {
                "chunks": len(chunks),
                "pages_with_text": pages_n,
                "chars": sum(len(c["text"]) for c in chunks),
                "with_par": sum(1 for c in chunks if c["par"]),
                "median_chars": sorted(len(c["text"]) for c in chunks)[
                    len(chunks) // 2
                ],
            },
            indent=2,
        )
    )
    if "--sample" in sys.argv:
        for c in chunks[:3] + chunks[len(chunks) // 2 : len(chunks) // 2 + 2]:
            print(f"\n[{c['page']} par={c['par']}] {c['text'][:220]}")


if __name__ == "__main__":
    main()
