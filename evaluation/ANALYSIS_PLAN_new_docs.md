# Analysis plan: strategies on 7 new documents (Experiment 4)

Written 2026-10-12, **before** running `run_new_docs.sh`. Do not change the contrasts, the predictions or the decision rule after seeing results. Changes go in a dated note at the bottom, and anything found after looking at the data is labelled post hoc.

## Question
The paper's claims were found on 3 documents and 50 questions. Do they hold on documents of a different kind (a research paper, a framework, a report summary, a novel, a clinical guideline excerpt, a product manual, an annual report)?

## Data
- 7 new documents, 174 answerable questions (attention 20, nist_ai_rmf 28, ipcc_spm 28, alice 28, who_handhygiene 28, roomba 14, tootsie 28). Questions and reference answers were drafted by a model, checked by independent model reviewers, and spot-checked by hand (21 of 217 items; 5 problems found, then a stricter second review of 3 documents). They are not independently annotated.
- Arms: Stage 1 (baseline), Stage 2 (generation_v2), Self-RAG, CRAG, Adaptive, Agentic with RRF merge, Agentic with dense merge (threshold 0.65, fixed in Finding 2 on the original documents, **not re-tuned**). One run per cell.
- Same code, same models, same judge (claude-haiku-4-5) as the main experiment.
- Vector store during the run: the 3 original documents plus the 7 new ones = **10 collections** (all real documents). Agentic searches all of them. The main experiment's store had 5 collections (the 3 documents plus two miscellaneous ones, readme and test_doc, which are no longer in the store and are not recreated). By the per-collection quota, the unrelated share of Agentic's candidates is therefore expected near 9/10 (90%), not 4/5 (80%). So absolute Agentic precision is **not** comparable with the main table; only direction and contrasts are. (10 collections is also the largest size in the collection-count experiment.)

## Metrics
faithfulness, answer_relevancy, context_precision (cp), context_recall (cr). cp and cr are primary because the findings are about them.

## Contrasts, fixed now
Difference a − b over questions (n = 174, paired by question, 2000 bootstrap resamples, seed 0, 95% CI). A per-document table is shown next to each contrast.

| id | contrast | metric | prediction (from the main experiment, experiments 2 and 3) |
|---|---|---|---|
| E1 | Agentic(RRF) − Stage 2 | cp | negative |
| E2 | Agentic(RRF) − Stage 2 | cr | positive |
| E3 | Agentic(dense) − Agentic(RRF) | cp | positive |
| E4 | Agentic(dense) − Agentic(RRF) | cr | none (the repeated runs gave a CI that included 0) |
| E5 | CRAG − Stage 2 | cr | positive |
| E6 | CRAG − Stage 2 | faithfulness | none (not robust to a second judge) |
| E7 | Stage 2 − Stage 1 | cp | positive |
| E8 | Agentic(RRF) − CRAG | cp | negative |

## Decision rules
- **Replicates**: the CI excludes 0 and the sign matches the prediction.
- **Does not replicate**: the CI includes 0. Report "no detectable difference on the new documents".
- **Contradicts**: the CI excludes 0 with the opposite sign. Report it and weaken the claim in the paper; do not drop the contrast.
- E4 and E6 have no prediction and are reported as they come.
- Per-document sign counts are reported for E1, E2 and E3 (how many of 7 documents have the predicted sign). A contrast that replicates overall but has the opposite sign in 3 or more documents is called "heterogeneous".
- No multiple-comparison correction (8 pre-specified contrasts). Say so.

## Descriptives (no tests)
- Per-arm means: pooled over questions, and mean of the 7 per-document means; per-document table.
- Agentic behaviour: stuck rate (round 5 still RETRIEVE), share of questions stopping in round 1, mean rounds, chunks retrieved per question, and the share of final chunks that come from the target collection.
- Arms not in a contrast (Self-RAG, Adaptive) are described only.

## Limitations to state
- Single run per cell. Experiment 2 gave run-to-run SDs of about 0.01–0.04 on the metrics; contrasts smaller than that should not be read.
- Questions and references are model-drafted; the judge and the generator are from the same model family.
- Document set chosen by me for variety, not sampled. 10-collection store (see above).
- The dense threshold was tuned on the original 50 questions; here it is out of sample, which is a fairer test of Finding 2's fix but may be too high or too low for these documents.

## Notes added after the run
- 2026-10-12 (before any results were looked at, written while the run was in progress): the descriptive "share of final chunks that come from the target collection" cannot be computed as written, because the final chunks in the traces carry no collection label. `analyze_new_docs.py` instead reports the share of chunks KEPT per retrieval call (from `retrieval_log`) that come from the question's own collection. Under RRF merge every collection keeps the same quota, so this is about 1/10 by construction; it is informative only under dense merge. This changes a descriptive only, not a contrast or a decision rule.
- 2026-10-12 (written after a test run of `analyze_new_docs.py` on the first 2-3 finished documents, where I saw partial contrast output and noticed from the small n of E1 (12 of 48) that many context_precision values are NaN; the partial results were all in the predicted direction and are NOT used for any decision here): in the 10-collection store Agentic (RRF) retrieves about 130-155 chunks per question, and RAGAS context_precision (one judge call per chunk) hits its default 180 s timeout for many of them (attention 11 of 20, nist_ai_rmf 25 of 28 NaN). A contrast computed only on the questions that did not time out would be biased toward questions with fewer chunks. Rule: before analysis, run `evaluation/rescore_nan_cp.py`, which re-scores only the NaN context_precision values with the same judge and a 900 s timeout. `analyze_new_docs.py` prints how many values are still missing and the n of every contrast. If a contrast (E1, E3, E8 are the ones affected) still has more than 10% of its questions missing, it is reported with that caveat and not as a clean replication test. This is a measurement repair applied to every arm alike; no contrast, prediction or decision rule is changed.
