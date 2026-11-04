# Analysis plan: can a retrieval score separate answerable from unanswerable questions?

Written 2026-10-08, **before running** `evaluation/score_separation.py` on the real index. Do not change the metrics, the thresholds or the interpretation rules below after seeing results. If something has to change, add a dated note at the bottom saying what and why.

## Question
The paper claims that an RRF score only guarantees a relative ranking and so cannot support decisions that need an absolute judgment. A decision such as "is there anything relevant here at all?" is exactly that kind of judgment. This experiment tests the claim directly: how well does the top-1 retrieval score tell an answerable question from an unanswerable one?

## Data
- Answerable: the 50 questions in `eval_questions_{ccby,cdc_program_eval,ragbench}.json` (16 / 17 / 17).
- Unanswerable: 45 questions in `eval_unanswerable_*.json`, 15 per document.
  - 10 per document `unanswerable_near`: on-topic, but the document does not contain the answer. Absence was checked by keyword search of the extracted document text.
  - 5 per document `unanswerable_far`: general-knowledge questions unrelated to the document.
- Each question is retrieved from its own document's collection only (one collection, no merging).

## Scores
- `rrf_top1_raw`: top-1 score of `HybridRetriever` with `score_threshold` set to 0 (nothing filtered).
- `dense_top1_raw`: top-1 raw dense cosine score, threshold 0.
- Both are measured without a threshold on purpose: the question is how much information the score carries, not where a cutoff happens to sit.
- Secondary: mean of the top-3; and top-1 under the system's real threshold (empty result counted as 0).

## Primary outcome
AUROC of top-1 score, answerable = positive class, pooled over the three documents. 0.5 means no information. 95% CI from a stratified bootstrap (2000 resamples; answerable and unanswerable questions resampled separately; seed 0). The difference AUROC(dense) − AUROC(RRF) uses paired resamples.

## Secondary outcomes (all reported, none dropped)
1. AUROC for near-topic only and far-only unanswerable questions.
2. AUROC per document.
3. Top-3 mean and in-system (thresholded) variants.
4. Operating point: the threshold that keeps at least 95% of the answerable questions, and the share of unanswerable questions it rejects.
5. Correlation of each top-1 score with question length in words. `HybridRetriever` uses dynamic weights that depend on query length, so an RRF score could track length instead of relevance.
6. Score distributions: min / median / max and the number of distinct values per group.

## Hypotheses, written before the data
- H1: RRF top-1 carries little information. Expected AUROC close to 0.5 (I would call it "little" if the upper end of its CI is below 0.65).
- H2: dense cosine top-1 is better than RRF top-1. Supported if the CI of the paired difference excludes 0.
- H3 (exploratory): near-topic unanswerable questions are harder than far ones for dense cosine, so AUROC(near) < AUROC(far).

## Interpretation rules
- Report all results whatever they are.
- If H1 fails (RRF AUROC clearly above 0.5), say so, and look at the length correlation before explaining it.
- If dense AUROC on near-topic questions is below 0.75, write that even a raw cosine score cannot reliably detect near-topic unanswerable questions. The claim of the paper is that RRF is unsuitable for absolute judgments, not that cosine is sufficient.
- Question sets are small (50 answerable, 45 unanswerable). Differences smaller than the bootstrap CI are not interpreted.

## Limitations to state in the paper
- Unanswerable questions were written by hand and checked by keyword search, not by an independent annotator.
- Near-topic questions are few (30); results on them are less certain than the pooled result.
- One embedding model, one retriever configuration, three documents.

## Notes added after the run

**2026-10-08, results of the pre-registered run (set A: 50 answerable, 45 unanswerable).**
- H1 (RRF carries little information, upper CI end below 0.65): **not supported.** AUROC(RRF top-1) = 0.70, 95% CI [0.59, 0.80]. At the operating point that keeps 96% of answerable questions, RRF rejects 13% of the unanswerable ones.
- H2 (dense better than RRF): **supported on the pooled data.** AUROC(dense top-1) = 0.85 [0.77, 0.92]; paired difference +0.16 [0.04, 0.28]. At the same operating point dense rejects 56%. The difference is not significant on near-topic questions alone (+0.12 [-0.03, 0.27]) or in ccby and cdc separately (small samples).
- H3 (near harder than far for dense): **supported.** Dense AUROC 0.80 [0.69, 0.89] on near-topic vs 0.97 [0.91, 1.00] on far. The rule "write that cosine cannot reliably detect near-topic questions if its AUROC is below 0.75" was not triggered by the point estimate, but the CI reaches below 0.75.
- The planned length check: correlation of top-1 score with question length is +0.28 (RRF) and +0.43 (dense).

**2026-10-08, a problem found after the run (post hoc, not in the plan above).** The set-A unanswerable questions are shorter than the answerable ones (mean 10.3 vs 15.6 words; the far questions mean 7.8). Question length alone separates the two classes with AUROC 0.81, so part of both scores' AUROC may come from length, not from relevance. This is a design flaw in set A. Exploratory length bins (set A): 11-14 words (22 answerable / 17 unanswerable): AUROC RRF 0.66, dense 0.79; up to 10 words (6 / 24): RRF 0.67, dense 0.84. The ordering dense > RRF holds inside the bins, but the bins are small.

**Response.** Set A stays the primary, pre-registered result and is reported as it is, together with the length problem. As a robustness check I wrote set B (`eval_unanswerable_long_*.json`): 24 near-topic unanswerable questions (8 per document, mean 17.5 words, about 2 words longer than the answerable questions, so any length leak now works against the hypothesis), written after seeing the first results and checked for absence by keyword search in the same way. Run `uv run python evaluation/score_separation.py --set-b`. Both results go into the paper; set B is labelled post hoc.

**Wording for the paper.** Do not write that RRF has no information about answerability. Write that it separates the two classes much worse than raw cosine (0.70 vs 0.85; 13% vs 56% rejected at 96% retention), that even cosine is weak on near-topic questions, and that neither score gives a calibrated absolute judgment.

**2026-10-08, set B result (post hoc robustness check, 50 answerable vs 24 length-matched near-topic unanswerable).**
- Set B is slightly longer than the answerable questions (mean 17.5 vs 15.6 words; length alone gives AUROC 0.32), so any remaining length leak works against the hypothesis for both scores.
- AUROC RRF top-1 = 0.55 [0.40, 0.68]; dense top-1 = 0.84 [0.75, 0.92]; paired difference +0.29 [0.13, 0.46]. Keeping 96% of answerable questions: RRF rejects 8%, dense 50%. Per document: RRF 0.62 / 0.62 / 0.40, dense 0.76 / 0.91 / 0.89 (ccby / cdc / ragbench).
- Caveat: set B asks about many of the same missing facts as set A (rephrased, longer), so A and B are not independent samples, and B was written after seeing the first results.

**2026-10-08, length-adjusted analysis (post hoc, exploratory; `--length-adjusted`).** Fit a line of score on question length over all questions, take residuals, compute AUROC of the residuals. Answerable (50) vs all unanswerable A+B (69): RRF 0.64 raw -> 0.61 adjusted [0.51, 0.70]; dense 0.85 raw -> 0.80 adjusted [0.72, 0.87]. Near-topic only (54): RRF 0.62 -> 0.61 [0.49, 0.71]; dense 0.82 -> 0.80 [0.72, 0.88]. A and B share topics, so treat the intervals as rough.

**Summary of the evidence for the paper.** In every variant dense cosine separates answerable from unanswerable questions clearly better than the RRF score. Dense stays at about 0.80-0.85 (0.80 on near-topic questions after length adjustment). RRF falls from 0.70 (set A, where length leaks) to 0.55-0.61 once length is controlled, with intervals that reach down to about 0.4-0.5; the pre-registered H1 (upper CI end below 0.65) is still not met on the primary data, so state the primary 0.70 and then the length-controlled values, and do not claim that RRF has no information at all. Better wording: after controlling for question length, the RRF top-1 score is close to uninformative about answerability (AUROC about 0.55-0.61), while cosine is informative but not strong on near-topic questions (about 0.80).

