# manual-rag-evals

[![tests](https://github.com/Dilmurod777/manual-rag-evals/actions/workflows/tests.yml/badge.svg)](https://github.com/Dilmurod777/manual-rag-evals/actions/workflows/tests.yml)

Ask an aircraft-maintenance manual a question; three retrievers answer, and every answer
names the page it came from.

https://github.com/user-attachments/assets/7bf278e7-fa71-44d0-b283-1e31c9d7139b

The clip runs three questions end to end. Each one sends five paragraphs to every pipeline;
the page shows what each arm answered, which page it used, and the sentence that matched. The
first asks about a circuit breaker (all three cite Page 11-16); the second is the one A misses;
the third is a flutter question whose citation lands on the wrong page.

FAA AC 43.13-1B, *Acceptable Methods, Techniques, and Practices â€” Aircraft Inspection
and Repair*: 646 pages, public domain, 1,758 paragraphs. Ask "why should a circuit breaker
not be used as a switch?" and every pipeline answers from the same passage on **Page
11-16**.

The point is not that a model can quote a manual. The point is that the quote is checkable:
the page number is the manual's own, and the sentence that matched is marked under the
answer.

---

## What it does

One question goes to three retrieval pipelines at once. Each returns five paragraphs, at most
one per page. A local 27B model writes a two-sentence answer from those paragraphs and is
told to end with the page it used. The page shows the three answers side by side with their
timings, the passage behind each one, and the highlighted sentence that matched.

When the closest paragraph is too far from the question, the app refuses before the model
is called: three retriever answers would be three guesses.

| | retrieval | embedding |
|---|---|---|
| **A** | vector only, cosine | `qwen3-embedding:0.6b`, 1024-d, local |
| **B** | BM25 + vectors, reciprocal rank fusion | same |
| **C** | B's top 30, read again by the model, five kept | same |

Generation is `qwen3.8-27b` through Ollama. If Ollama is down, the app shows the passage and
its page instead of inventing an answer, and says so in italics.

---

## The measurement

60 questions, five retrieved paragraphs each, every gold page verified by re-reading it.

| pipeline | hit@5 | MRR | answer cites the gold page |
|---|---|---|---|
| A Â· vector only | 59/60 | 0.894 | 55/60 |
| B Â· hybrid | **60/60** | **0.939** | **55/60** |
| C Â· hybrid + rerank | **60/60** | **0.944** | 54/60 |

One question is worth 1.7 points at this size. The 95% intervals are wide enough to swallow
every gap in that table except the miss itself â€” hit@5 is 0.91â€“1.00 for A and 0.94â€“1.00 for
B and C; citation is 0.82â€“0.96 for A and B, 0.80â€“0.95 for C. That is why the counts are
written as counts. These are single runs: re-running the same questions moves one or two
between ranks, which is what those intervals are telling you. C is also not a better writer.
All three arms use the same model and the same prompt; C only changes the order of the five
pages it reads, so its edge is in retrieval, not in prose.

A misses exactly one page: "how is pressure applied over a laminated plastic repair while
it cures" (Page 3-4) returns Pages 1-25, 3-6, 3-1 and 3-7 instead. B and C find it.

### Question by question

![which rank the gold page got](docs/rank.svg)

Half the story is in the ranking: B puts the gold page first 53 times out of 60, A 49. A's
single miss is the only question where the right page never appears. The chart shows all
three arms; C's per-question ranks are in `eval/results.json`.

![hit@5 by chapter](docs/by_chapter.svg)

Every chapter is perfect except chapter 3, where the question about curing pressure is the
one A loses. The chapters with two questions are thin evidence; the spread is there to show
that the misses are not hiding in one section.

The five answers that cited the wrong page are worth reading, because four of them are one
page off â€” the answer is on the neighbouring page of a two-page spread:

| question | gold page | page the answer cited |
|---|---|---|
| Which instruments can test bonded structure for disbonds from one side? | 5-46 | 5-48 |
| If the original rib lace spacing is not known, what maximum spacing may be used on the wings? | 2-21 | 2-19 |
| Can a copper-sleeve splice replace a five-tuck splice in a control cable? | 7-32 | 7-33 |
| How is pressure applied over a laminated plastic repair while it cures? | 3-4 | 3-6 |
| What precaution keeps a repaired flight control surface from fluttering? | 4-9 | 1-15 |

The last one is the only real confusion: both pages discuss flutter, and the model picked
the chapter 1 discussion over the chapter 4 rule. A reader would notice; that is why the
page shows the sentence, not just the citation.

All three arms answered all 60 questions and named the gold page 55, 55 and 54 times.
Retrieval is not the limit: five pages were in front of the model and in five or six cases
it quoted a different one. That is the cost of generation, and it is why the page marks the
sentence instead of asking for trust.

The caveat, stated plainly: the questions were written after reading passages that
retrieval had already returned, so every pipeline is flattered and hit@5 sits near its
ceiling. That is why there is a second set.


## How it was built

1. **Pick a document a reader can check.** AC 43.13-1B is the FAA's handbook of acceptable
   repair methods. It is public domain, it is text (no OCR), and every page carries its own
   label in the footer: `Page 6-30`, plus a paragraph number `Par 6-151` on most pages.
   That label is the product. A citation has to lead back to the page a mechanic opens, not
   to a PDF index.
2. **Extract without destroying the reading order.** `pdftotext -raw`, not `-layout`: the
   manual is two columns, and layout mode interleaves them into sentences like "IN-PLACE
   TREATMENT OF c. Strip paint if required. MAGNESIUM CASTINGS". The PDF also splits words
   across lines ("reaÂ­ son", "de- vices"); `app/corpus.py` repairs both.
3. **Cut on the manual's own boundaries.** Paragraphs start with `6-151.`; the splitter
   uses those markers, drops table-of-contents rows (dot leaders), running headers and
   footers, and cuts anything past ~1,100 characters at a sentence end. Each chunk keeps
   `page`, `pdf_page`, `par` and `text`.
4. **Embed once, cache with a fingerprint.** `app/embed.py` writes `data/embeddings.npz`
   with a SHA-256 of the joined chunk text. The first search compares fingerprints and raises
   on a mismatch, because chunk ids are positional: edit the text, forget to re-embed, and
   every later id points at a different paragraph. That failure is silent, so it is checked.
   Loading is lazy so the package can be imported without the cache.
5. **Write the questions before the scores.** 60 questions, each with the page that answers
   it and the sentence that proves it (`eval/gold.jsonl`). `eval/check_gold.py` re-reads every
   cited page and fails if the evidence sentence is not there. Questions were written after
   reading the page, never from memory. Three more sets keep that one honest: 40 blind
   questions written from the table of contents, 12 paraphrases that avoid the manual's words,
   and 12 questions the manual cannot answer.
6. **Score retrieval and generation separately.** `eval/run_eval.py` reports hit@5, MRR, and
   whether the generated answer cites the gold page. The third number is the one a reader
   cares about and the one most demos skip.

---

## How it works

```
question
   |
   +--> A: embed -> cosine over 1,758 vectors -> top 5 (one per page)
   |
   +--> B: BM25 top 30 + cosine top 30 -> reciprocal rank fusion -> top 5
   |
   +--> C: B's top 30 -> the model reads all 30 -> top 5
   |
   +--> gate: is the closest paragraph within 0.58? if not, refuse before any model call
   |
   +--> three prompts to qwen3.8-27b, streamed token by token into three columns
   |
   +--> {answer, page, grounded, highlighted sentence, retrieval ms, answer ms}
```

- `POST /api/ask {"q": "..."}` returns all three arms at once. `q` is 3â€“400 characters;
  anything else is a 422 before a model is touched.
- `POST /api/ask/stream` is the same question as server-sent events, which is what the page
  uses: `status`, then `meta` (timing, pages, passage), `delta` per token, `done` with the
  answer and the `refused` / `grounded` flags.
- `GET /api/results` returns `eval/results.json` for the bars at the bottom of the page.
- `GET /health` is described under "Running it like a service".
- `GET /` serves `public/index.html`, one file, no build step.
- Two requests are served at a time (`threading.Semaphore(2)`); a third gets a 429 instead of
  a queue. Inside one request, A and B answer together and C answers after them, because the
  27B is already holding the GPU. This is a local demo by design.
- Context is capped at 8k and `think: false` is set. Without those, Qwen3 spends its whole
  token budget on hidden reasoning and returns an empty answer â€” the bug that made the
  first demo seem broken.

---

## A blind set

Forty more questions were written from the manual's table of contents alone â€” pick a
section, read its opening page, write the question â€” with no retrieval result in view.
Nothing was tuned against them. `eval/blind.jsonl`, checked by the same reader.

| pipeline | hit@5 | MRR | answer cites the gold page |
|---|---|---|---|
| A Â· vector only | 39/40 | 0.852 | 38/40 |
| B Â· hybrid | 38/40 | 0.921 | 34/40 |
| C Â· hybrid + rerank | 39/40 | **0.975** | 38/40 |

This is the more interesting table. With 40 questions the 95% interval is about Â±9 points,
so the pipelines are not distinguishable on hit@5, and the citation gap sits inside its
own interval as well. What survives is MRR: on questions nobody tuned for, C puts the
right page first most often (0.975), then B (0.921), then A (0.852). C also restores the
citations B loses (38 against B's 34), so the extra model call earns its 1.7 s here. Both
of B's misses are honest: "what is the first step before repairing a metal structure"
(Page 4-1, the chapter's opening instruction) is not a paragraph a retriever can match to a
question, and B's other miss is the equipment-change question on Page 10-1, answered one
section later on 10-14.

---

## When the words change

Twelve of the tuned questions rewritten the way a person asks without the manual in front
of them ("How do I know if a piece of wood has soaked up too much water?" instead of "How
can excessive moisture in a wood member be detected?"). Same pages, same evidence.

| pipeline | hit@5 | MRR | answer cites the gold page |
|---|---|---|---|
| A Â· vector only | 11/12 | 0.812 | 8/12 |
| B Â· hybrid | 9/12 | 0.569 | 8/12 |
| C Â· hybrid + rerank | 11/12 | **0.833** | 10/12 |

Hybrid loses the most, which is the predictable cost of keyword scoring: BM25 rewards the
manual's own vocabulary, and a paraphrase has none of it. C recovers what B drops. Twelve
questions is a pilot, not a result â€” the intervals are enormous â€” but the direction is
consistent with why C exists.

## Refusing

Twelve questions the manual cannot answer: type-specific torque values, avionics tests,
another aircraft's service ceiling. Before the gate, the prompt's "say so if the passages do
not answer" instruction was obeyed 12 of 12 times; the model did not invent a page.

The gate was fitted on six of those questions and checked on the other six, because a
threshold chosen and scored on the same rows proves nothing. Fitted line: 0.579, set halfway
between the highest unanswerable score in the fit half and the lowest answerable score over
all 100 tuned and blind questions. On the six it never saw it catches **2 of 6** before the
model is called, which is the honest number: the gate skips a three-second generation on the
obvious misses, and the model still refuses the rest. False refusals: 0 of the 60 tuned and 0
of the 40 blind questions, whose lowest score is 0.584.

The gate is not a better reader than the model. It is a guarantee that does not depend on
prompt compliance. `python -m eval.refusal` re-fits and re-checks; six fit questions is a
small sample, and the split is there to keep the claim from being circular, not to make it
precise.

## The three approaches, and what each is good at

**A â€” vector only.** Embeds the question and compares it to paragraph embeddings.

- Good: finds the right passage when the words differ ("how tight should a turnbuckle be"
  against "cable tension"). Ignores spelling and word order. One index, one comparison, 20 ms.
- Bad: weak on rare tokens that matter â€” part numbers, `AN470`, `2117`, `4130`. A paragraph
  that is *about* the topic but does not answer the question ranks well. It cannot tell you
  the page says "do not" rather than "do".
- Failure here: Page 3-4, the laminate cure pressure, buried in a chapter full of similar
  repair text.

**B â€” BM25 + vectors, fused.** Keyword scoring and vector scoring, combined by reciprocal
rank fusion.

- Good: keeps the exact-token strength of BM25 (part numbers, "circuit breaker", "2117")
  without giving up semantic matching. Fusion needs no score calibration â€” it combines
  ranks, so a BM25 score of 30 and a cosine of 0.8 never have to be made comparable.
- Bad: two indexes to build and keep. Fusion adds a knob (how deep each list goes) that is
  tuned, not derived. A question whose answer has no exact keyword overlap gains nothing
  over A.
- Result here: 60/60 hit@5, better MRR, and the one page A misses.

**C â€” hybrid + rerank.** B's top 30, listed to the same local model with the question, which
returns the five that answer it.

- Good: ranking is the job a language model is good at here. It reads meaning, not token
  overlap, and it never returns two slices of one page.
- Bad: one more model call per question, 1.7 s warm and up to 3 s cold, which is most of the
  cost of writing the answer itself. It is also a second place the model can be wrong: a bad
  list loses a page retrieval had already found.
- Result: the best ranker on both sets (MRR 0.944 tuned, 0.975 blind), and it recovers the
  citations hybrid loses on questions nobody tuned for (38/40 against 34/40). On the tuned set
  it costs one citation (54 against 55), which is inside noise.

| questions nobody tuned for | hit@5 | MRR | answer cites gold page |
|---|---|---|---|
| B Â· hybrid | 38/40 | 0.921 | 34/40 |
| C Â· hybrid + rerank | 39/40 | **0.975** | **38/40** |

A cross-encoder would be the cheaper form of C â€” a small model instead of a 27B one â€” and is
the obvious next step if this ever left a laptop.

**Why no vector database.** Retrieval is a quarter of a second of a request whose first words
appear in half a second. The corpus is 1,758
paragraphs; the vectors fit in memory and a cosine sweep is 20 ms. A vector database earns
its place at millions of chunks, with metadata filters, or with writes during queries â€”
none of which this has. Adding one would move 2% of the wait and add a service to run.

---

## Where the time goes

| step | time |
|---|---|
| retrieval, one pipeline (including the query embedding) | 210â€“250 ms |
| the cosine or BM25 comparison alone | 20â€“45 ms |
| one answer | 2.9 s |
| first words of an answer, streamed | 0.5â€“0.8 s |
| A and B, side by side | 2.5â€“3.6 s |
| all three arms, C answered after A and B | 6.5â€“7 s |
| the rerank call alone | 1.7 s warm, up to 3 s cold |

Medians over the 60 gold questions: 252 ms for A, 209 ms for B, 2.9 s per answer. Retrieval
is about 8% of the wait, and most of that is the HTTP call to the embedding model, not the
comparison. The 1,758 paragraphs and their 1,024-dimension vectors fit in memory, and a
cosine sweep is 20 ms. A vector database earns its place at millions of chunks, metadata
filters, or writes during queries â€” none of which this corpus has. The levers that mattered
were an 8k context (keeps the 27B on the GPU instead of spilling to the CPU), `think:
false`, and running the arms in parallel instead of one after the other. Answers stream, so
the first words appear at about 0.5 s and the page fills as each arm finishes â€” a 3 s answer
that starts in half a second feels like a different product.

---

## Before and after: streaming and parallel arms

| | before | after |
|---|---|---|
| first words on screen | 5.9 s, when all three had answered | **1.5 s** |
| first complete answer | 5.9 s | **2.5 s** |
| all three complete | 5.9 s | 5.2 s |
| how the columns fill | all at once | A, then B, then C, each as it finishes |

The three answers still do not decode at the same instant. Ollama runs one generation per
model at a time on this GPU, so B's request is in flight after 0.23 s but starts decoding
when A releases the model. What changed is that nothing waits for the slowest arm: the
request leaves immediately, the text appears as it is produced, and the first answer is
readable at 1.5 s instead of 5.9 s. Real overlap needs the server restarted with
`OLLAMA_NUM_PARALLEL=2`, which halves each arm's tokens per second and was not worth the
trade: a two-slot attempt still serialised.

## Running it like a service

- `GET /health` answers `{"ok": true, "chunks": 1758, "model_present": true, "refuse_below": 0.58}`.
  It is the check a deploy would poll, and it is false when Ollama is down.
- Each answer carries `grounded`: false when the page it named was not among the five it
  was shown. The page does not print it yet; the API does.
- Repeated questions are cached on the passages they retrieved: an answer already produced
  for the same five pages returns in 0 ms and skips the model, streaming included.
- One log line per arm, with the question and the closeness of the best paragraph.
  `RAG_LOG=DEBUG` for more.
- `docker build -t manual-rag . && docker run -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 manual-rag`.
  The container builds the corpus on first start; the manual is not stored in the image.
- Every knob lives in `app/config.py` and reads the environment: `OLLAMA_HOST`, `RAG_MODEL`,
  `RAG_EMBED_MODEL`, `RAG_TOP_K`, `RAG_REFUSE_BELOW`, `RAG_NUM_CTX`, `RAG_CORPUS_URL`.
- CI (`.github/workflows/tests.yml`) runs the unit tests, downloads the manual, and re-proves
  all 60 gold and 40 blind pages against it on every push.

The page is usable without a mouse and without motion: arm A's answer is an `aria-live`
region and B and C stay silent, so a screen reader hears one answer instead of three
overlapping token streams; focus is visible, and `prefers-reduced-motion` turns off the
transitions.

## Run it

Needs `pdftotext` (Poppler) and Ollama with `qwen3-embedding:0.6b`. The generator is a
local name, so build it once from the published tag (`make` is optional, every target is
one command):

```bash
ollama pull qwen3-embedding:0.6b
ollama create qwen3.8-27b -f Modelfile   # FROM the unsloth Q4_K_XL 27B, 8k context
```

```bash
uv venv && uv pip install -r requirements.txt
python -m app.setup            # download the manual, extract, embed (about a minute)
python -m uvicorn app.main:app --port 8041
```

Open <http://127.0.0.1:8041>. Re-score with `python -m eval.run_eval`; prove the gold pages
with `python -m eval.check_gold`; run the tests with `pytest -q`.

---

## Make it yours

- **Another manual.** Set `RAG_CORPUS_URL` (or edit `CORPUS_URL` in `app/config.py`). If the
  pages do not print `Page X-Y` and `Par X-Y` footers, change `PAGE_LABEL` and `PAR_START` in
  `app/corpus.py`; the rest of the pipeline does not care where the label comes from.
- **After any text change**: `python -m app.corpus`, then `python -m app.embed`. The first
  search then refuses with a fingerprint error until the vectors match the text â€” on purpose.
  Finish with `python -m eval.run_eval`.
- **Another model.** `RAG_MODEL=qwen3.6-27b python -m uvicorn app.main:app`. Anything that
  Ollama serves over `/api/chat` works.
- **Your own questions.** Add lines to `eval/gold.jsonl` (`{"q", "page", "evidence"}`) and
  run the checker; it will tell you which evidence sentences are not on the page you named.

---

## Layout

```
app/corpus.py    646 pages -> citable paragraphs
app/embed.py     embeddings, cached in data/embeddings.npz with a fingerprint
app/search.py    BM25, vector search, fusion
app/rerank.py    listwise rerank: twelve candidates in, five out
app/config.py    every knob, read from the environment
app/main.py      FastAPI: /api/ask, /api/ask/stream, /health, the page
eval/gold.jsonl  60 questions with page + the sentence that proves it
eval/blind.jsonl 40 questions written from the table of contents
eval/paraphrase.jsonl 12 questions that avoid the manual's wording
eval/unanswerable.jsonl 12 questions the manual does not cover
eval/run_eval.py hit@5, MRR and citation accuracy for the three pipelines
eval/refusal.py  score distributions and the gate's decisions
eval/charts.py   the two SVG charts
public/index.html the page (no build step)
tests/           three asserts on the parts that would break silently
Dockerfile       build, corpus on first start, uvicorn on 8000
.github/workflows/tests.yml  tests + both gold sets re-proved
```

---

## Limits

- One manual, one machine. The numbers describe this corpus.
- Four question sets, not one: 60 tuned, 40 blind, 12 paraphrases, 12 unanswerable. The last
  two are pilots â€” wide intervals, useful direction, not enough to rank the pipelines.
- The blind set is 40 questions: enough to keep the tuned set honest, not enough to rank the
  pipelines.
- Local only. A public deploy would need a queue, a quota and a GPU that is not a laptop.
- No fine-tuning, no OCR, no answer without a citation.
