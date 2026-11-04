# Analysis plan: repeated runs of the key arms

Written 2026-10-08, **before running** `evaluation/run_repeats.sh`. Do not change the arms, metrics or contrasts after seeing results; if something has to change, add a dated note at the bottom.

## Question
All the main comparisons so far come from one run each, and one infrastructure fix already moved some numbers a lot. How large is run-to-run variation, and which of the paper's comparisons survive it?

## Design
- Four arms: Stage 2 (`generation`), CRAG, Agentic with RRF merge, Agentic with dense-cosine merge (threshold 0.65).
- Same three documents, same 50 answerable questions, same code, same config, same vector store (5 collections). Three repetitions. The only source of difference is the LLM's own randomness.
- Judge: claude-haiku-4-5, as in the earlier runs. (A second judge is a separate experiment.)
- Run this before any new document is added to the vector store, because Agentic RAG searches every collection.

## Reported
1. Per arm and metric: the 50-question mean in each repetition, mean ± standard deviation across repetitions, min and max. The SD is the measured run-to-run noise and replaces the rough "±0.04 / ±0.015" figures used in earlier drafts.
2. Eight contrasts, fixed now, A − B with a paired bootstrap 95% CI over questions (5000 resamples; each question's score first averaged over the repetitions):
   - Agentic (RRF) − Stage 2: context_precision, context_recall
   - Agentic (dense) − Agentic (RRF): context_precision, context_recall
   - CRAG − Stage 2: context_recall, faithfulness
   - Agentic (RRF) − CRAG: context_precision, context_recall

## Expectations written before the data
- Agentic (RRF) has much lower context_precision than Stage 2 and CRAG (the earlier single runs: 0.185 vs 0.519 and 0.410).
- Agentic (dense) recovers part of the precision and loses some recall (the earlier A/B: +0.122 and −0.080).
- CRAG and Agentic have similar context_recall.
- Differences smaller than the run-to-run SD are not interpreted.

## Interpretation rules
- A contrast is called a real difference only if its CI excludes 0 **and** its size is larger than the run-to-run SD of the arms involved.
- Report every contrast, including those that do not hold. If a conclusion in an earlier finding does not survive, change the finding rather than the analysis.

## Limitations to state in the paper
- Three repetitions give a rough SD, not a precise one.
- The questions are the same in every repetition, so the CIs describe variation over these 50 questions and over LLM randomness, not over other documents or other question sets.
- One judge model.

## Notes added after the run
(none yet)
