# RAGAS evaluation layer (optional, fully removable)

This folder is a self-contained add-on for scoring RagForge's answers with
[RAGAS](https://github.com/explodinggradients/ragas). It does not modify
any file outside `evaluation/`, and does not touch `pyproject.toml` or the
project's main `.venv`. Delete this whole folder at any time and the rest
of the project is completely unaffected.

## Why two scripts, two environments

- **`collect_predictions.py`** runs inside your normal RagForge `.venv`
  (the same one `cli.py` uses). It imports the existing `ragforge` package
  as-is -- no new dependencies -- indexes a document, runs every question
  in `eval_questions.json` through the real pipeline, and writes the raw
  results (question, answer, retrieved chunks) to
  `evaluation/reports/predictions.json`.

- **`run_ragas_eval.py`** runs in its own throwaway virtual environment
  (`evaluation/.ragas_venv`), built from the pinned versions in
  `requirements-ragas.txt`. It only reads the plain-data
  `predictions.json` -- it never imports `ragforge` -- so its dependencies
  (deliberately pinned slightly old; see the comment at the top of
  `requirements-ragas.txt` for why) can never conflict with or downgrade
  anything the main pipeline relies on.

This split is what makes the whole thing removable: nothing about RAGAS
ever gets installed into the environment your actual pipeline runs in.

## Usage

```bash
# 0. (optional but recommended) List a document's chunks + chunk_ids, to
#    pick reference_chunk_ids for eval_questions.json (needed for hitRate).
#    Read-only: no Milvus, no Ollama, no Anthropic calls.
uv run python evaluation/list_chunks.py --document /path/to/your/document.md

# 1. Collect predictions (uses your existing .venv, e.g. via `uv run`)
uv run python evaluation/collect_predictions.py --document /path/to/your/document.md

# 2. One-time setup of the isolated RAGAS environment
python3 -m venv evaluation/.ragas_venv
evaluation/.ragas_venv/bin/pip install -r evaluation/requirements-ragas.txt

# 3. Score the predictions
evaluation/.ragas_venv/bin/python evaluation/run_ragas_eval.py
```

## Incremental ablation study across Stage 2 features

`collect_predictions.py` takes a `--variant` flag, one per stage of the
pipeline (baseline, then each Stage 2 feature turned on cumulatively). It
builds each variant with the exact same component-selection logic
`cli_stage2.py` uses, so the numbers reflect what the real pipeline does,
not a hand-picked guess:

```
baseline      Stage 1: fixed_size chunking, DenseRetriever, SimpleGenerator
chunking      + ParentChild chunking
hybrid        + Hybrid (dense+sparse) retrieval
reranking     + Cross-Encoder reranking
query_engine  + query engine (rewrite + HyDE + multi-query/decomposition)
generation    + citation generation + hallucination detection
```

Two things are collapsed here on purpose rather than tracked separately:
HyDE and multi-query generation can't be isolated from each other, because
`QueryEngine.process_query` doesn't actually branch on its own
`enable_rewrite`/`enable_hyde`/`enable_multi_query`/`enable_decomposition`
config flags -- it always runs rewrite+HyDE together, then always runs
multi-query or decomposition depending on the classified intent. So
there's no way to isolate HyDE from the rest of the query engine with the
codebase as it stands; they're bundled into one `query_engine` variant.
Likewise, context compression has no on/off switch in `GeneratorConfig` at
all (it's always built once you're past `baseline`) -- it's a constant
background factor in every non-`baseline` variant, not something this
ablation isolates.

Run each variant in turn, then score each one separately:

```bash
for variant in baseline chunking hybrid reranking query_engine generation; do
  uv run python evaluation/collect_predictions.py \
      --document /path/to/your/document.md --variant "$variant"
done

for variant in baseline chunking hybrid reranking query_engine generation; do
  pred="evaluation/reports/predictions.json"
  [ "$variant" != "baseline" ] && pred="evaluation/reports/predictions_${variant}.json"
  evaluation/.ragas_venv/bin/python evaluation/run_ragas_eval.py \
      --predictions "$pred" \
      --out "evaluation/reports/ragas_report_${variant}.json"
done
```

That gives you six `ragas_report_*.json` files (`faithfulness`,
`answer_relevancy`, `hit_rate` once `reference_chunk_ids` are filled in)
you can line up side by side -- the point isn't hitting any particular
target number, it's confirming that each feature moves the scores in the
direction it's supposed to, row by row.

Results land in `evaluation/reports/ragas_report.json` (averaged scores +
a per-question breakdown), and are also printed to the console.
`collect_predictions.py` also prints hitRate directly (see below) since it
doesn't need the RAGAS judge at all.

## What it measures

- **faithfulness** and **answer_relevancy** run out of the box, no ground
  truth needed (reference-free, LLM-as-judge).
- **context_precision** and **context_recall** need a human-written
  reference answer per question. Fill in the `"reference"` field in
  `eval_questions.json` to unlock them -- until every question has one,
  `run_ragas_eval.py` skips these two and tells you why.
- **hitRate** is a deterministic IR metric (no LLM judge, no RAGAS): for
  each question with `"reference_chunk_ids"` filled in, it's 1.0 if
  retrieval surfaced at least one of them and 0.0 if it surfaced none.
  `collect_predictions.py` computes it per-question and prints the
  average; `run_ragas_eval.py` folds that same average into
  `ragas_report.json`'s `averaged_scores.hit_rate` so it lives alongside
  the RAGAS metrics in one report. Until every question has
  `reference_chunk_ids`, both scripts skip questions that don't and tell
  you so -- hitRate is computed only over the questions that do.

The judge LLM reuses your existing `ANTHROPIC_API_KEY` (from `.env`) --
same account, no new signup. The embedding step (needed for
`answer_relevancy`) reuses your existing local Ollama setup.

## Picking `reference_chunk_ids` with `list_chunks.py`

hitRate needs to know, per question, which chunk_id(s) actually contain
the answer. `list_chunks.py` parses and chunks a document exactly the way
indexing would (using the real `ragforge` parser/chunker when your
checkout has them; otherwise a byte-for-byte copy of
`FixedSizeChunker`'s splitting + id logic, clearly labeled when it's used
-- see the comment at the top of that file) and writes every chunk's
`chunk_id` + full content to `evaluation/reports/chunks.json`, plus a
compact preview table to the console. Open that file, find the chunk(s)
that answer a given question, and copy their `chunk_id`s into that
question's `"reference_chunk_ids"` array in `eval_questions.json`. A
cross-paragraph question can legitimately have more than one id.

## Editing `eval_questions.json`

The file now ships with 54 questions covering three types on purpose --
tagged via `"type"`: `"summary"` (e.g. "what is this document mainly
about"), `"detail"` (a specific technical choice, config value, or
number -- best at revealing imprecise retrieval), and `"cross_paragraph"`
(needs multiple chunks combined -- tests whether a single chunk is enough
or `topK` needs to pull in more). Each question already has a
human-written `"reference"` answer so `context_precision`/`context_recall`
are unlocked out of the box; `"reference_chunk_ids"` is intentionally left
empty for you to fill in per-document via `list_chunks.py` (see above) --
they can't be filled in generically since they depend on exactly how your
document gets chunked. Swap in questions about your own document(s) the
same way: mix the three types so the scores actually reflect known
failure modes (like the `scoreThreshold` under-retrieval issue) instead of
just a single averaged number.

## Testing across multiple documents (cross-domain generalization)

Every script that touches a document (`list_chunks.py`, `collect_predictions.py`)
takes an independent `--document` and `--questions` pair, so nothing about the
harness is tied to a single document -- running it against a different document
is just a different pair of flags, not a code change.

`evaluation/documents/` holds a few additional test documents, picked to be
public/permissively-licensed and structurally different from the primary
Clinical Note Structuring Tool document (narrative prose, heavy configuration
detail): a legal license text (numbered clauses), a public-health program
evaluation report (long-form with many named lists and sub-steps), and a
research paper (abstract/method/results structure). Each has its own question
set, built the same way as `eval_questions.json` -- summary/detail/cross_paragraph
questions with hand-written `"reference"` answers, `"reference_chunk_ids"` left
empty for you to fill in via `list_chunks.py`:

```
evaluation/documents/ccby_4.0_legalcode.pdf                    <-> evaluation/eval_questions_ccby.json
evaluation/documents/cdc_program_evaluation_framework_2024.pdf <-> evaluation/eval_questions_cdc_program_eval.json
evaluation/documents/ragbench_2407.11005v2.pdf                 <-> evaluation/eval_questions_ragbench.json
```

To run the full ablation across one of these instead of the primary document,
substitute both flags in the loop from the section above:

```bash
for variant in baseline chunking hybrid reranking query_engine generation; do
  uv run python evaluation/collect_predictions.py \
      --document evaluation/documents/ragbench_2407.11005v2.pdf \
      --questions evaluation/eval_questions_ragbench.json \
      --variant "$variant" \
      --out "evaluation/reports/predictions_ragbench_${variant}.json"
done
```

Score each one with `run_ragas_eval.py --predictions ... --out
evaluation/reports/ragas_report_ragbench_<variant>.json` the same way, then
compare the per-document tables (e.g. via `summarize_reports.py`, pointed at
the right report prefix) to see whether a feature's effect direction holds up
across document types, not just on the primary one.

## Switching to your own custom evaluation instead

Nothing here is wired into `RAGPipeline`, `cli.py`, or `stage1.yml` --
`collect_predictions.py` is the only file that touches the real pipeline,
and it does so exactly the way a normal `index` + `query` session would
via `cli.py`. To swap to your own hand-rolled scoring logic later, you can
either write a different consumer of `predictions.json` (same input
format, your own scoring), or ignore this folder's format entirely and
build your own `eval_set.json` + evaluator from scratch -- both are
independent of anything else in the project.
