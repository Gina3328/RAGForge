#!/usr/bin/env bash
# Agentic RAG stop-decision window experiment (Finding 3).
#
# Question: is the Agent's GENERATE/RETRIEVE decision limited by WHICH 10 chunks
# it is shown (RRF ordering) or by HOW MUCH it is shown (10 x 200 chars)?
# Everything else (RRF cross-collection merge, max_iterations, prompt text) is identical.
#
#   win_rrf    window ordered by .score (RRF)      10 chunks x 200 chars   <- original behaviour, re-run with logging
#   win_dense  window ordered by raw dense cosine  10 chunks x 200 chars   <- same candidates, ordering only
#   win_big    window ordered by .score (RRF)      20 chunks x 500 chars   <- size only
#   win_score_rrf    same window as win_rrf, each chunk prefixed with its RRF score (4 dp)
#   win_score_dense  same window as win_rrf, each chunk prefixed with its raw cosine (2 dp)
#   (the "no score" control for these two is win_rrf; merge stays RRF, only the shown number differs)
#
# Traces only: no predictions / RAGAS (the question is about stopping behaviour),
# so this is much cheaper than the merge comparison.
#
# Usage (project root, with Docker/Milvus and Ollama running):
#   bash evaluation/run_agentic_window_experiment.sh                 # all 3 arms x 3 docs
#   bash evaluation/run_agentic_window_experiment.sh win_rrf         # only one arm (comma-separate several)
# Output: evaluation/reports/decision_traces_<doc>_agentic_<arm>.json
# Safe to Ctrl-C and re-run: finished runs (file exists) are skipped.
# Then:   uv run python evaluation/analyze_agentic_window.py

set -e
ARMS="${1:-win_rrf,win_dense,win_big}"

DOCS=(
  "ccby:evaluation/documents/ccby_4.0_legalcode.pdf:evaluation/eval_questions_ccby.json"
  "cdc:evaluation/documents/cdc_program_evaluation_framework_2024.pdf:evaluation/eval_questions_cdc_program_eval.json"
  "ragbench:evaluation/documents/ragbench_2407.11005v2.pdf:evaluation/eval_questions_ragbench.json"
)

arm_flags() {
  case "$1" in
    win_rrf)   echo "--agentic-window-order score --agentic-window-chunks 10 --agentic-window-chars 200" ;;
    win_dense) echo "--agentic-window-order dense --agentic-window-chunks 10 --agentic-window-chars 200" ;;
    win_big)   echo "--agentic-window-order score --agentic-window-chunks 20 --agentic-window-chars 500" ;;
    win_score_rrf)   echo "--agentic-window-order score --agentic-window-chunks 10 --agentic-window-chars 200 --agentic-window-show-score rrf" ;;
    win_score_dense) echo "--agentic-window-order score --agentic-window-chunks 10 --agentic-window-chars 200 --agentic-window-show-score dense" ;;
    *) echo "unknown arm: $1" >&2; exit 1 ;;
  esac
}

IFS=',' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
  FLAGS="$(arm_flags "$arm")"
  for entry in "${DOCS[@]}"; do
    doc_key="${entry%%:*}"; rest="${entry#*:}"; doc_path="${rest%%:*}"; questions_path="${rest#*:}"
    out="evaluation/reports/decision_traces_${doc_key}_agentic_${arm}.json"
    if [ -f "$out" ]; then
      echo "--- ${doc_key} / ${arm} : traces already exist, skipping ---"
      continue
    fi
    echo "--- ${doc_key} / ${arm} : collecting agentic traces ($FLAGS) ---"
    # shellcheck disable=SC2086
    uv run python evaluation/collect_decision_traces.py \
      --strategy agentic --document "$doc_path" --questions "$questions_path" \
      --agentic-merge-source rrf $FLAGS --out "$out"
  done
done
echo "Done. Analyze with: uv run python evaluation/analyze_agentic_window.py"
