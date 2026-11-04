#!/usr/bin/env bash
# Experiment 6: split the Agentic decision window into "how many chunks" and "how many characters per chunk".
#
# Finding 3 compared a 10 x 200 window (original) with a 20 x 500 window. Both numbers changed at once, so the
# result cannot say which one stops the Agent from looping. This script runs the full 2 x 2 grid, all in the same
# code state and the same vector store, with traces + answers + RAGAS:
#
#   w10x200   10 chunks x 200 chars   (original window, re-run now)
#   w10x500   10 chunks x 500 chars   (more text per chunk)
#   w20x200   20 chunks x 200 chars   (more chunks)
#   w20x500   20 chunks x 500 chars   (the earlier "win_big", re-run now)
#
# Everything else is the original Agentic setup: RRF cross-collection merge, window ordered by the RRF score,
# no score shown, same prompt. Same 50 questions. Repeated REPS times (default 2) because a single run of the
# same configuration varies by about +-3 stuck questions.
#
# IMPORTANT: run this BEFORE adding the new documents (documents_new/) to the vector store. Agentic searches
# every collection in the store, so the results are only comparable to Finding 3 while the store is unchanged.
#
# Usage (project root, Docker/Milvus + Ollama running, unattended is fine):
#   bash evaluation/run_window_split.sh          # 2 repetitions
#   bash evaluation/run_window_split.sh 1        # quick: 1 repetition
# Safe to Ctrl-C and re-run: finished runs (RAGAS report exists) are skipped.
# Output: evaluation/reports/{decision_traces,predictions,ragas_report}_<doc>_agentic_<arm>_rep<i>.json
# Then:   python3 evaluation/analyze_window_split.py
# Plan:   evaluation/ANALYSIS_PLAN_window_split.md

set -e
REPS="${1:-2}"
RAGAS_PY="evaluation/.ragas_venv/bin/python"
R=evaluation/reports

DOCS=(
  "ccby:evaluation/documents/ccby_4.0_legalcode.pdf:evaluation/eval_questions_ccby.json"
  "cdc:evaluation/documents/cdc_program_evaluation_framework_2024.pdf:evaluation/eval_questions_cdc_program_eval.json"
  "ragbench:evaluation/documents/ragbench_2407.11005v2.pdf:evaluation/eval_questions_ragbench.json"
)
ARMS=(w10x200 w10x500 w20x200 w20x500)

arm_flags() {
  case "$1" in
    w10x200) echo "--agentic-window-chunks 10 --agentic-window-chars 200" ;;
    w10x500) echo "--agentic-window-chunks 10 --agentic-window-chars 500" ;;
    w20x200) echo "--agentic-window-chunks 20 --agentic-window-chars 200" ;;
    w20x500) echo "--agentic-window-chunks 20 --agentic-window-chars 500" ;;
    *) echo "unknown arm: $1" >&2; exit 1 ;;
  esac
}

for rep in $(seq 1 "$REPS"); do
  echo "##################################################################"
  echo "# Repetition ${rep} of ${REPS}"
  echo "##################################################################"
  for entry in "${DOCS[@]}"; do
    doc_key="${entry%%:*}"; rest="${entry#*:}"; doc_path="${rest%%:*}"; questions_path="${rest#*:}"
    for arm in "${ARMS[@]}"; do
      tag="${doc_key}_agentic_${arm}_rep${rep}"
      traces_out="$R/decision_traces_${tag}.json"
      pred_out="$R/predictions_${tag}.json"
      ragas_out="$R/ragas_report_${tag}.json"
      if [ -f "$ragas_out" ]; then
        echo "--- ${tag} : already done, skipping ---"
        continue
      fi
      FLAGS="$(arm_flags "$arm")"
      echo "--- ${tag} : collecting ($FLAGS) ---"
      if [ ! -f "$traces_out" ]; then
        # shellcheck disable=SC2086
        uv run python evaluation/collect_decision_traces.py \
          --strategy agentic --document "$doc_path" --questions "$questions_path" \
          --agentic-merge-source rrf --agentic-window-order score $FLAGS --out "$traces_out"
      fi
      uv run python evaluation/convert_traces_to_predictions.py --traces "$traces_out" --out "$pred_out"
      echo "--- ${tag} : RAGAS scoring ---"
      "$RAGAS_PY" evaluation/run_ragas_eval.py --predictions "$pred_out" --out "$ragas_out"
    done
  done
done

echo "Done. Now run: python3 evaluation/analyze_window_split.py"
