# Analysis plan: do the paper's conclusions survive a change of judge model? (Experiment 3)

Written 2026-10-08, **before** any re-scoring with a new judge. Do not change the contrasts, the decision rule or the interpretation after seeing results. Changes go in a dated note at the bottom.

## Why
Every RAGAS number so far comes from one judge, claude-haiku-4-5, and the answers being judged were most likely written by a Claude model too. A reviewer can ask whether the judge prefers its own family's answers, and whether the conclusions depend on one small judge.

## What is re-scored
The 24 main reports, unchanged otherwise: 6 arms (baseline, generation_v2, self_rag, crag, adaptive, agentic; the `_post_flush_fix` runs) x 3 documents, plus `agentic_merge_rrf` and `agentic_merge_dense` x 3 documents. 50 questions per arm. The same retrieved contexts, answers and references are given to the new judge (`rescore_judge.py` reads them from `per_sample`). Nothing is retrieved or generated again, so any change in scores comes from the judge alone.

## Judges
1. Original: claude-haiku-4-5 (already done; default temperature).
2. Cross-family: an OpenAI chat model (non-reasoning). Answers the self-preference question.
3. Stronger same-family: a larger Claude model. Answers "is the result an artefact of using a small judge".
Exact model names are written into the output files (`judge` field) and into the paper.

Settings for the new judges: temperature 0, max_tokens 4096, the same embedding model (Ollama) for answer_relevancy. The original judge used the default temperature, so a small part of any difference can come from temperature; this is stated in the paper. The between-run noise of the original judge is measured separately in Experiment 2 (repeats).

## Outcomes (all reported)
1. Per-arm means per judge (mean of the per-document means).
2. Spearman correlation between the original judge and each new judge of (a) the six Stage 1-4 arm means, per metric, and (b) the per-question scores over all re-scored questions.
3. Seven contrasts, fixed now. Difference = first arm minus second arm, paired by question, 95% bootstrap CI (2000 resamples, seed 0):
   - C1 generation_v2 - baseline, context_precision
   - C2 agentic - generation_v2, context_precision
   - C3 agentic - generation_v2, context_recall
   - C4 crag - generation_v2, context_recall
   - C5 crag - generation_v2, faithfulness
   - C6 self_rag - generation_v2, faithfulness
   - C7 agentic_merge_dense - agentic_merge_rrf, context_precision

## Decision rule
- A contrast is **robust to the judge** if its CI excludes 0 under the original judge and, under every new judge, the CI excludes 0 with the same sign.
- If the sign is the same but the CI includes 0 under a new judge, write "same direction, not significant under judge X".
- If the sign flips under a new judge, the conclusion depends on the judge. Say so in the paper and weaken the claim; do not drop the contrast.
- A contrast whose CI includes 0 under the original judge is reported, not interpreted.
- Arm-rank agreement: Spearman >= 0.8 is described as "the ordering of the strategies is largely preserved"; lower values are reported as they are.

## Expectations (written before the data, so they can fail)
- Absolute scores will differ between judges (the means will not be equal). Only the ordering and the contrasts are tested.
- I expect C2 (Agentic has much lower context_precision) to be robust, because it is a large effect and also follows from the retrieval design, not only from the judge.
- I expect the faithfulness contrasts (C5, C6) to be the least robust, because faithfulness depends most on how strict the judge is about each claim.
- I do not expect the judges to agree on answer_relevancy as closely as on the context metrics.

## Limitations to state
- Still 3 documents and 50 questions per arm; the larger document set (Experiment 4) can also be re-scored.
- A new judge is not ground truth. Agreement between judges shows robustness to the judge, not correctness.
- Contexts and answers were produced while Claude was the generator; the cross-family judge checks self-preference only for judging, not for generating.

## Notes added after the run
(none yet)
