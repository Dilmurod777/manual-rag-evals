"""Ask a second model pass whether the answer is supported by the page it cites.

The retrieval score says the right page was found. This says the sentence written about it
is in the manual, which is a different claim and the one a reader cares about.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

from app.corpus import paragraphs
from app.main import answer
from app.search import BM25, Vector, hybrid

HERE = Path(__file__).resolve().parent
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "gold.jsonl"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "judgement.json"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = os.environ.get("RAG_JUDGE_MODEL", "qwen3.8-27b")

RUBRIC = """You are checking a mechanic's answer against a repair manual.

Question: {q}

Answer given: {a}

Text of the page the answer cites (Page {page}):
{page_text}

Is every factual claim in the answer stated in that page text? Ignore wording, judge meaning.
Reply in exactly this form, two lines:
VERDICT: SUPPORTED or PARTIAL or UNSUPPORTED
WHY: one sentence naming what is or is not there."""


def page_text(page, chunks):
    return " ".join(c["text"] for c in chunks if c["page"] == page)[:3500]


def judge(question, reply, page, chunks):
    payload = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 120},
        "messages": [
            {
                "role": "user",
                "content": RUBRIC.format(
                    q=question, a=reply, page=page, page_text=page_text(page, chunks)
                ),
            }
        ],
    }
    try:
        text = httpx.post(OLLAMA, json=payload, timeout=90).json()["message"]["content"]
    except Exception:
        return None, "judge did not answer"
    found = re.search(r"VERDICT:\s*(SUPPORTED|PARTIAL|UNSUPPORTED)", text)
    why = re.search(r"WHY:\s*(.+)", text)
    return (found.group(1) if found else "UNCLEAR"), (why.group(1).strip() if why else text.strip())


def main():
    chunks = list(paragraphs())
    bm25, vector = BM25(chunks), Vector(chunks)
    gold = [
        json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = []
    for row in gold:
        ranked = hybrid(bm25, vector, row["q"], 5)
        reply, _ = answer(row["q"], ranked)
        cited = re.search(r"\(Page (\d+-\d+)\)", reply)
        page = cited.group(1) if cited else None
        verdict, why = (
            judge(row["q"], reply, page, chunks)
            if page
            else ("NO-CITATION", "answer named no page")
        )
        rows.append(
            {
                "q": row["q"],
                "gold": row["page"],
                "answer": reply,
                "cited": page,
                "verdict": verdict,
                "why": why,
            }
        )
        print(f"{verdict:<11} {page or '-':>7}  {row['q'][:50]}")
    counts = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    OUT.write_text(
        json.dumps(
            {"model": MODEL, "measured": time.strftime("%Y-%m-%d"), "counts": counts, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(counts)


if __name__ == "__main__":
    main()
