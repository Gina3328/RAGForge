# Analysis plan: refusals, false refusals and unsupported answers per strategy (Experiment 7)

Written 2026-10-08, **before** running `run_refusal.sh`. Do not change the definitions, the contrasts or the decision rules after seeing results. Changes go in a dated note at the bottom.

## Question
Finding 1 says CRAG's confidence threshold declines questions it should answer. That is a claim about answerable questions. The other half is what a refusal buys: does a strategy that refuses more also refuse the questions that really have no answer in the document? And what do the other strategies do with an unanswerable question: refuse, or write a specific answer?

## Data
One mixed question file per document: 50 answerable questions in total (16 / 17 / 17), 45 set-A unanswerable questions (15 per document: 10 on-topic, 5 unrelated general-knowledge), 24 set-B unanswerable questions (8 per document, on-topic, length-matched, written after seeing Experiment 1's first results). All strategies answer the same file in one run each, same code, same vector store (the 5 original collections).
Groups: **answerable** (50); **near** = on-topic unanswerable, set A (30) + set B (24) = 54; **far** = unrelated (15).
Arms: Stage 1, Stage 2, Self-RAG, CRAG, Adaptive, Agentic (RRF merge), Agentic (dense merge, threshold 0.65).

## Labelling
Each response is labelled REFUSAL (says the information is not available / not found / declines; related background is allowed if no specific answer is given) or ANSWER (a specific answer, right or wrong) by an LLM judge that sees only the question and the response. The fixed CRAG decline message is labelled REFUSAL by a rule. The judge does not decide whether an answer is correct.
Checks: (1) 40 answers (20 REFUSAL, 20 ANSWER, random) are checked by hand; if the hand labels disagree with the judge on more than 10% of them, the judge labels are reported with that caveat. (2) A second judge from another model family labels everything; agreement and Cohen's kappa are reported.

## Outcomes (all reported)
- **False refusal** = share of answerable questions refused. Lower is better.
- **Near refusal** = share of near questions refused. Higher is better.
- **Unsupported answer** = 1 - near refusal: a specific answer to a question the document cannot answer.
- **Far refusal**: reported, not interpreted (answering from general knowledge may be legitimate).
- **J = near refusal - false refusal**: how much more the strategy refuses questions without an answer than questions with one. J = 0 means it refuses both alike. 95% stratified bootstrap CI.
- Set A and set B near-refusal rates are also shown separately (B is post hoc).
- Rates carry 95% Wilson intervals.

## Contrasts, fixed now (difference in rate, paired by question, 95% bootstrap CI, 2000 resamples, seed 0)
- K1 false refusal: CRAG - Stage 2
- K2 near refusal: CRAG - Stage 2
- K3 false refusal: Agentic(RRF) - Stage 2
- K4 near refusal: Agentic(RRF) - Stage 2
- K5 near refusal: Agentic(dense) - Agentic(RRF)
- K6 near refusal: Stage 2 - Stage 1

## Hypotheses, written before the data
- H1 (from Finding 1): CRAG refuses more answerable questions than Stage 2 (K1 > 0, CI excludes 0).
- H2: CRAG also refuses more near questions than Stage 2 (K2 > 0). I expect the gain to be smaller than the false-refusal cost, so that CRAG's J is not clearly larger than Stage 2's. I am unsure about this; CRAG's confidence uses the raw cosine score, which separated the classes reasonably in Experiment 1 (AUROC about 0.8), so J may well be larger.
- H3 (exploratory): Agentic (RRF merge) refuses fewer near questions than Stage 2 (K4 < 0), because the extra, unrelated chunks give the generator something to build an answer on. No directional prediction for K5.
- H4 (exploratory): Stage 1 has the lowest near refusal, because it has no instruction to decline (K6 > 0).

## Decision rules
- Report every contrast whatever it shows. A contrast whose CI includes 0 is called "no detectable difference".
- If the second judge's labels change the sign or the significance of a contrast, report both and say the result depends on the judge.
- Do not call a strategy better because J is higher unless false refusal is also reported next to it. A strategy that refuses everything has J = 0 and 100% false refusal.

## Limitations to state
- The unanswerable questions were written by hand and checked for absence by keyword search, not by an independent annotator; set B was written after the first Experiment 1 results.
- REFUSAL / ANSWER is judged by an LLM; the hand check covers 40 answers only.
- One run per cell; the generator's own randomness is not measured here (Experiment 2 measured it for answerable questions only).
- 3 documents, 54 near questions: the intervals are wide. The far questions are general knowledge and not the main result.
- A specific answer to a near question is called "unsupported" because the document has no answer; the response may still be partly right from general knowledge.

## Notes added after the run
(none yet)
