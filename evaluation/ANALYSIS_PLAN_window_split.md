# Analysis plan: chunks or characters? (Experiment 6, Finding 3 follow-up)

Written 2026-10-08, **before** running `run_window_split.sh`. Do not change the contrasts, the metrics or the decision rules after seeing results. Changes go in a dated note at the bottom.

## Question
Finding 3: a 20 x 500 decision window stopped the Agent's loop (0/50 stuck, 1.34 rounds on average) while the original 10 x 200 window did not (24/50 stuck, 3.46 rounds), but the bigger window also lowered answer quality (recall -0.10). The window changed in two ways at once. Which one matters, the number of chunks shown or the number of characters per chunk?

## Design
2 x 2 grid, same code state and same vector store (the 5 original collections), same 50 questions, RRF merge, window ordered by RRF score, no score shown:
`w10x200` (original), `w10x500`, `w20x200`, `w20x500`. Each arm is run twice (REPS = 2) and scored with the same RAGAS judge as before. The two corner arms (`w10x200`, `w20x500`) are re-runs of Finding 3's win_rrf and win_big under today's code, so the four cells are comparable with each other. The earlier Finding 3 runs are not pooled with these.

## Outcomes
1. Behaviour: stuck questions (round 5 still RETRIEVE), questions that stop in round 1, mean number of rounds.
2. Quality: faithfulness, answer_relevancy, context_precision, context_recall (RAGAS).
3. Effects on each outcome, per question (value averaged over repetitions), 95% bootstrap CI, 2000 resamples, seed 0:
   - chunks effect = mean(w20x*) - mean(w10x*)
   - characters effect = mean(w*x500) - mean(w*x200)
   - interaction = (w20x500 - w20x200) - (w10x500 - w10x200)

## Decision rules (primary outcome: mean rounds, secondary: stuck questions)
- A factor "drives stopping" if its effect on mean rounds has a CI that excludes 0 and is at least twice as large in absolute value as the other factor's effect.
- If both effects exclude 0 and neither is twice the other, write that both matter and report the interaction.
- If neither CI excludes 0, but the corners reproduce Finding 3 (w20x500 vs w10x200), write that the two changes cannot be separated at this sample size.
- If w10x200 does not reproduce Finding 3's original behaviour (stuck far from 19-26 or mean rounds far from 3.4-3.5), say so first. Then the comparison with Finding 3 is weaker and the 2 x 2 is read on its own.
- Quality outcomes are reported for every cell whatever the behaviour result is. A setting that stops earlier is not called better unless recall and answer_relevancy are not lower.

## Expectations (written before the data, so they can fail)
- I expect characters per chunk to matter more than the number of chunks: a 200-character snippet is often the start of a legal clause or a heading, so the Agent may judge it insufficient, whereas 500 characters shows enough to decide. I am not sure of this; the number of chunks could matter more, because more chunks make it likelier that one clearly answers the question.
- I expect any setting that stops the loop early to show lower recall than w10x200, as in Finding 3.

## Limitations to state
- Two repetitions per cell; the earlier same-configuration noise was about +-3 stuck questions out of 50.
- One Agent model, one prompt, three documents, the RRF merge only.
- The decision prompt is not shown to the model with scores here (that was a separate Finding 3 experiment).

## Notes added after the run
(none yet)
