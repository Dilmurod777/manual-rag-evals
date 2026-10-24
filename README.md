# manual-rag-evals

Ask an aircraft-maintenance manual a question; two retrievers answer, and every answer
names the page it came from.

![the app](docs/app.png)

## What this is

FAA AC 43.13-1B, *Acceptable Methods, Techniques, and Practices — Aircraft Inspection
and Repair* (646 pages, public domain), cut into **1,758 paragraphs**. Each paragraph
keeps the manual's own footer label, so a citation is `Page 6-30` — the page a mechanic
would open — and not a PDF index.

Two pipelines over the same paragraphs:

| | retrieval | embedding |
|---|---|---|
| **A** | vector only, cosine | `qwen3-embedding:0.6b`, 1024-d, local |
| **B** | BM25 + vectors, reciprocal rank fusion | same |

Answers are written by a local model (`qwen3.8-27b` through Ollama) from the five
retrieved paragraphs, and it is told to end with the page it used. If Ollama is not
running, the app shows the passage and its page instead of inventing an answer.

## The measurement

**60 questions**, each with the printed page that answers it, every page verified by
re-reading it (see `eval/check_gold.py`). Five retrieved paragraphs per question.

| pipeline | hit@5 | MRR |
|---|---|---|
| A · vector only | 0.98 | 0.894 |
| B · hybrid | **1.00** | **0.939** |

A misses exactly one: "how is pressure applied over a laminated plastic repair while it
cures" (Page 3-4) returns Pages 1-25, 3-6, 3-1 and 3-7 instead. B finds it.

The caveat, stated plainly: the questions were written after reading passages that
retrieval had already returned, so both pipelines are flattered and hit@5 is close to
its ceiling. The honest signal here is MRR — B puts the right page first more often —
plus the one question A cannot reach at all. A harder, paraphrase-only set is the next
thing this repo needs; its absence is the weakest part of the result.

## Three commands

Needs `pdftotext` (Poppler) and a running Ollama with `qwen3-embedding:0.6b`.

```bash
uv venv && uv pip install -r requirements.txt
python -m app.setup            # download the manual, extract, embed (about a minute)
python -m uvicorn app.main:app --port 8041
```

Open <http://127.0.0.1:8041>. `python -m eval.run_eval` re-scores both pipelines and
rewrites `eval/results.json`; `python -m eval.check_gold` proves all 60 gold pages.

## Layout

```
app/corpus.py   646 pages -> citable paragraphs
app/embed.py    embeddings, cached in data/embeddings.npz
app/search.py   BM25, vector search, fusion
app/main.py     FastAPI: /api/ask, /api/results, the page
eval/gold.jsonl 60 questions with page + the sentence that proves it
eval/run_eval.py hit@5 and MRR for both pipelines
public/index.html  the page (no build step)
```

## Non-goals

No fine-tuning, no OCR, no second manual, and no answer without a citation.
