# Experiment 5 analysis plan: effect of the number of collections N on Agentic cross-collection retrieval

Written before the runs (2026-10-13). The predictions and decision rules must not be changed after the runs; any analysis added later must be labeled "post hoc".

## Question
Finding 2 found, in a store with 5 collections, that Agentic takes the top 5 chunks from every collection. An RRF score only expresses relative rank, so the top 5 of an irrelevant collection also enter the merged result, and the noise grows with the number of collections. Experiment 4 reproduced this with 10 collections (130.7 chunks per question, and the target collection makes up about 1/10 of them). But "grows with N" is so far only a cross-experiment comparison between two values of N (5 and 10), and the two experiments used different document sets and different questions. Experiment 5 changes only the number of collections that Agentic can see, on the **same questions**.

## Design
- Questions: the 50 questions on the original 3 documents (ccby, cdc, ragbench). These are the target documents, and they are in the store for every N.
- N = 3, 6, 10. The sets are nested, and the order is fixed before the runs:
  - N=3: ccby, cdc, ragbench
  - N=6: add attention, nist_ai_100_1, ipcc_ar6_syr_spm (the first three in the order of `new_docs.tsv`)
  - N=10: add alice, who_hand_hygiene, roomba, tootsie (the full store of Experiment 4)
- Implementation: the store itself does not change. Only the `list_collections()` that Agentic sees is limited to the first N.
- Two arms: Agentic (RRF merge) and Agentic (dense merge, threshold 0.65, not re-tuned). All other settings are the same as in Experiment 4 (`config/stage4.yml`).
- Only the decision traces are collected. RAGAS is not run (this experiment does not compare answer quality; it looks only at the number of chunks, the share of target chunks, and stopping behavior).
- 6 runs in total (2 arms × 3 values of N), 50 questions each.

## Metrics (one value per question)
- chunks: the number of chunks finally returned.
- target share: among the chunks that were kept ("kept") in all retrieval calls for the question, the share that come from the target document's collection (averaged over the question's retrieval calls).
- stuck: still RETRIEVE at round 5 (the same definition as in Finding 3 and Experiment 4); stopping at round 1; number of rounds.
- check: the number of collections actually searched in each retrieval call must equal N (the script reports violations).

## Predictions and decision rules (paired bootstrap, 2000 resamples, seed 0, 95% CI; paired by question)
- **P1 (mechanism check)**: in the RRF arm, the target share is about 1/N (|mean − 1/N| ≤ 0.05). This follows from taking 5 chunks from every collection. It is not a finding; it is only used to confirm that the implementation is correct.
- **P2**: in the RRF arm, the number of chunks per question grows with N: the difference N=10 minus N=3 has a CI lower bound > 0. I also report 6−3 and 10−6 to see whether the growth is roughly linear.
- **P3**: dense merge suppresses this growth. Both parts must hold:
  - at every N, the CI lower bound of the target share difference (dense − RRF) is > 0;
  - for the number of chunks, the CI upper bound of the difference-in-differences [(dense: N10−N3) − (RRF: N10−N3)] is < 0.
- **P4 (description only, no decision rule)**: how the stuck rate, the round-1 rate and the number of rounds change with N. I give the paired difference N=10 minus N=3 and its CI, but make no prediction. In Experiment 4 the stuck rate differed a lot between documents (10%–82%), so I do not expect a common direction here.

## Limits on interpretation (to carry into the paper)
1. There are only 3 target documents, with different numbers of questions. The conclusion is about behavior on these 50 questions, not an estimate for all documents.
2. The documents added at N=6 and N=10 are fixed and nested, and which documents are added was not randomized. The similarity between documents affects the result. If the result at some N is clearly caused by one added document, split it by collection using `retrieval_log` and label it post hoc.
3. Only traces are used and answer quality is not scored, so this experiment cannot show that "answers get worse as N grows". For answer quality see Experiment 4.
4. The dense threshold 0.65 was set in Finding 2 and was not re-tuned for different N.
5. P1 is a mechanism check and must not be written as a finding. P2 is also expected from the mechanism. Its value is that it gives a quantitative relation between N and the amount of noise, and removes the confound of the cross-experiment comparison.

## Running and analysis
```
uv run python evaluation/index_new_docs.py          # first reset the store (Stage 1 of Experiment 4 modifies it), and confirm there are 10 collections
caffeinate -i bash evaluation/run_collection_count.sh
python3 evaluation/analyze_collection_count.py
```
Output: `evaluation/reports/decision_traces_exp5_<doc>_<arm>_n<N>.json` and `collection_count_summary.json`.
The script can be interrupted and re-run; finished files are skipped.
