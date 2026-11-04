#!/usr/bin/env bash
# Full re-run of the 3-document x (baseline, generation, self_rag, crag,
# adaptive, agentic) evaluation grid, after the milvus_vector_store.py
# upsert() flush() fix -- see the confidence=0.0 race-condition finding.
# Every output file gets a _post_flush_fix suffix so nothing here
# clobbers the earlier (pre-fix) reports already in evaluation/reports/.
#
# Usage: bash evaluation/rerun_all_post_flush_fix.sh
# Run from the project root. Takes a while (indexes + queries + RAGAS
# judging, 3 docs x 6 pipelines) -- safe to Ctrl-C and resume by commenting
# out the blocks that already finished; each command's output file name
# tells you what's done.

set -e  # stop on first real error, so a bad run doesn't silently continue

RAGAS_PY="evaluation/.ragas_venv/bin/python"

# doc_key : document_path : questions_path
DOCS=(
  "ccby:evaluation/documents/ccby_4.0_legalcode.pdf:evaluation/eval_questions_ccby.json"
  "cdc:evaluation/documents/cdc_program_evaluation_framework_2024.pdf:evaluation/eval_questions_cdc_program_eval.json"
  "ragbench:evaluation/documents/ragbench_2407.11005v2.pdf:evaluation/eval_questions_ragbench.json"
)

echo "=================================================================="
echo "STAGE 1/2: baseline + generation (collect_predictions.py)"
echo "=================================================================="
for entry in "${DOCS[@]}"; do
  doc_key="${entry%%:*}"
  rest="${entry#*:}"
  doc_path="${rest%%:*}"
  questions_path="${rest#*:}"

  for variant in baseline generation; do
    out_label="$variant"
    [ "$variant" = "generation" ] && out_label="generation_v2"
    pred_out="evaluation/reports/predictions_${doc_key}_${out_label}_post_flush_fix.json"
    ragas_out="evaluation/reports/ragas_report_${doc_key}_${out_label}_post_flush_fix.json"

    echo "--- ${doc_key} / ${variant} : predictions ---"
    uv run python evaluation/collect_predictions.py \
      --document "$doc_path" --variant "$variant" \
      --questions "$questions_path" --out "$pred_out"

    echo "--- ${doc_key} / ${variant} : RAGAS scoring ---"
    "$RAGAS_PY" evaluation/run_ragas_eval.py \
      --predictions "$pred_out" --out "$ragas_out"
  done
done

echo "=================================================================="
echo "STAGE 3/4: self_rag, crag, adaptive, agentic (decision traces)"
echo "=================================================================="
for entry in "${DOCS[@]}"; do
  doc_key="${entry%%:*}"
  rest="${entry#*:}"
  doc_path="${rest%%:*}"
  questions_path="${rest#*:}"

  for strategy in self_rag crag adaptive agentic; do
    traces_out="evaluation/reports/decision_traces_${doc_key}_${strategy}_post_flush_fix.json"
    pred_out="evaluation/reports/predictions_${doc_key}_${strategy}_post_flush_fix.json"
    ragas_out="evaluation/reports/ragas_report_${doc_key}_${strategy}_post_flush_fix.json"

    if [ -f "$traces_out" ]; then
      echo "--- ${doc_key} / ${strategy} : traces already exist, skipping collection ---"
    else
      echo "--- ${doc_key} / ${strategy} : collecting decision traces ---"
      uv run python evaluation/collect_decision_traces.py \
        --strategy "$strategy" --document "$doc_path" \
        --questions "$questions_path" --out "$traces_out"
    fi

    echo "--- ${doc_key} / ${strategy} : converting to predictions ---"
    uv run python evaluation/convert_traces_to_predictions.py \
      --traces "$traces_out" --out "$pred_out"

    echo "--- ${doc_key} / ${strategy} : RAGAS scoring ---"
    "$RAGAS_PY" evaluation/run_ragas_eval.py \
      --predictions "$pred_out" --out "$ragas_out"
  done
done

echo "=================================================================="
echo "Done. All outputs suffixed _post_flush_fix.json in evaluation/reports/"
echo "=================================================================="
