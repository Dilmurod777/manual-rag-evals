# manual-rag-evals — design

Status: approved 24 Oct 2026. Third portfolio project.

## Intent

Show that retrieval on a real maintenance manual can be measured, not asserted.
A reader asks a question, gets an answer with the manual's own page number, and can
compare two retrieval pipelines on the same 60 questions.

## Corpus

FAA AC 43.13-1B with Change 1, *Acceptable Methods, Techniques, and Practices —
Aircraft Inspection and Repair* (US government work, public domain). 646 PDF pages,
2.0M characters, digital text, no OCR needed. Every page carries its own citation in
the footer: `Page 6-30`, and most paragraphs carry `Par 6-151`.

Citations use the manual's printed labels, never PDF indices, so a reader can open a
paper copy and land on the same page.

## Chunks

One paragraph, one chunk. Each chunk keeps the printed page label and the paragraph
number when the paragraph has one. Running headers and footers are stripped.

## Pipelines

A. **Vector only** — `qwen3-embedding:0.6b` through the local Ollama, cosine top-5.
B. **Hybrid** — the same vectors plus BM25 over the same chunks, fused with reciprocal
   rank fusion.
C. **Hybrid + rerank** — B's top 30 reranked by the local model, top-5 kept.

C is opt-in: it costs a generation call per query.

## Gold set

60 questions written by hand against the manual, each with the page that answers it.
Drafts are generated from retrieved passages, then checked against the page text
before they enter the set; every question names the sentence that proves its page.

## Measures

- **Hit@5** — the gold page appears in the top 5 chunks.
- **Citation precision** — of the answers the model produced, the share whose cited
  page is the gold page (or within the same paragraph family).
- **Failure gallery** — the questions the best pipeline still misses, with the page it
  returned instead.

## App

FastAPI plus one static page: a question box, two answer columns (A and B) with their
citations, and the bars underneath. No account, no database, JSON on disk.

## Non-goals

- No fine-tuning.
- No OCR or scanned manuals.
- No second document; 60 questions on one manual beat 10 on five.
- No answers without a citation. A pipeline that cannot cite does not answer.

## Anti-goal

An answer that sounds right and cites nothing. That is the failure this project exists
to make visible.
