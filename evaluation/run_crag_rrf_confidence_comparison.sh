#!/usr/bin/env bash
# RRF-vs-dense CRAG confidence comparison. Runs CRAG with
# --crag-confidence-source hybrid (RRF-fused score) on the same 3 clean
# (post-flush-fix) documents, so the results can sit side by side with
# the existing dense-based decision_traces_<doc>_crag_post_flush_fix.json.
set -e
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

  traces_out="evaluation/reports/decision_traces_${doc_key}_crag_rrf_confidence.json"
  pred_out="evaluation/reports/predictions_${doc_key}_crag_rrf_confidence.json"
  ragas_out="evaluation/reports/ragas_report_${doc_key}_crag_rrf_confidence.json"

  echo "--- ${doc_key} : collecting CRAG traces with hybrid (RRF) confidence ---"
  uv run python evaluation/collect_decision_traces.py \
    --strategy crag --document "$doc_path" --questions "$questions_path" \
    --crag-confidence-source hybrid --out "$traces_out"

  echo "--- ${doc_key} : converting to predictions ---"
  uv run python evaluation/convert_traces_to_predictions.py \
    --traces "$traces_out" --out "$pred_out"

  echo "--- ${doc_key} : RAGAS scoring ---"
  "$RAGAS_PY" evaluation/run_ragas_eval.py \
    --predictions "$pred_out" --out "$ragas_out"
done

echo "Done. RRF-confidence CRAG results: decision_traces_*_crag_rrf_confidence.json"
