#!/usr/bin/env bash
# Agentic RAG cross-collection merge: RRF (A) vs dense cosine (B), on the same
# 3 documents, back to back, against the same vector store.
#
#   A: --agentic-merge-source rrf    (original behavior)
#   B: --agentic-merge-source dense --agentic-dense-min-score <THRESHOLD>
#
# Both arms are re-run now (not reusing the older *_post_flush_fix files) so
# that store contents, config and code are identical except for the one switch.
#
# Usage (project root):
#   1) uv run python evaluation/calibrate_dense_threshold.py     # picks THRESHOLD
#   2) bash evaluation/run_agentic_merge_comparison.sh <THRESHOLD>
# Output files:
#   evaluation/reports/{decision_traces,predictions,ragas_report}_<doc>_agentic_merge_{rrf,dense}.json
# Each traces file records which collections were in the store and the settings used.
# Safe to Ctrl-C and re-run: finished arms (traces file exists) are skipped.

set -e
THRESHOLD="${1:?usage: bash evaluation/run_agentic_merge_comparison.sh <dense_min_score, from calibrate_dense_threshold.py>}"
RAGAS_PY="evaluation/.ragas_venv/bin/python"

DOCS=(
  "ccby:evaluation/documents/ccby_4.0_legalcode.pdf:evaluation/eval_questions_ccby.json"
  "cdc:evaluation/documents/cdc_program_evaluation_framework_2024.pdf:evaluation/eval_questions_cdc_program_eval.json"
  "ragbench:evaluation/documents/ragbench_2407.11005v2.pdf:evaluation/eval_questions_ragbench.json"
)

for entry in "${DOCS[@]}"; do
  doc_key="${entry%%:*}"
  rest="${entry#*:}"
  doc_path="${rest%%:*}"
  questions_path="${rest#*:}"

  for arm in rrf dense; do
    traces_out="evaluation/reports/decision_traces_${doc_key}_agentic_merge_${arm}.json"
    pred_out="evaluation/reports/predictions_${doc_key}_agentic_merge_${arm}.json"
    ragas_out="evaluation/reports/ragas_report_${doc_key}_agentic_merge_${arm}.json"

    if [ -f "$traces_out" ]; then
      echo "--- ${doc_key} / ${arm} : traces already exist, skipping collection ---"
    else
      echo "--- ${doc_key} / ${arm} : collecting agentic traces ---"
      if [ "$arm" = "dense" ]; then
        uv run python evaluation/collect_decision_traces.py \
          --strategy agentic --document "$doc_path" --questions "$questions_path" \
          --agentic-merge-source dense --agentic-dense-min-score "$THRESHOLD" --out "$traces_out"
      else
        uv run python evaluation/collect_decision_traces.py \
          --strategy agentic --document "$doc_path" --questions "$questions_path" \
          --agentic-merge-source rrf --out "$traces_out"
      fi
    fi

    echo "--- ${doc_key} / ${arm} : converting to predictions ---"
    uv run python evaluation/convert_traces_to_predictions.py --traces "$traces_out" --out "$pred_out"

    echo "--- ${doc_key} / ${arm} : RAGAS scoring ---"
    "$RAGAS_PY" evaluation/run_ragas_eval.py --predictions "$pred_out" --out "$ragas_out"
  done
done

echo "Done. Compare with: uv run python evaluation/analyze_agentic_merge.py"
