#!/usr/bin/env bash
# Repeat the four key arms N times on the same 3 documents, so the paper can
# report run-to-run variation and confidence intervals instead of single runs.
#
#   generation_v2  Stage 2 (collect_predictions.py --variant generation)
#   crag           Stage 3 CRAG
#   agentic_rrf    Stage 4 Agentic, RRF merge (the default behavior)
#   agentic_dense  Stage 4 Agentic, dense-cosine merge (Finding 2's fix)
#
# Nothing is tuned between repetitions: same code, same config, same vector
# store, same questions. The only thing that differs is the LLM's own
# randomness, which is exactly what we want to measure.
#
# Usage (project root; unattended is fine, needs Docker/Milvus, Ollama and the
# Anthropic key like every other evaluation script):
#   bash evaluation/run_repeats.sh            # 3 repetitions
#   bash evaluation/run_repeats.sh 3 0.65     # reps, dense min score
# Safe to Ctrl-C and re-run: any output file that already exists is skipped,
# so a re-run continues where it stopped.
# Output files: evaluation/reports/{decision_traces,predictions,ragas_report}_<doc>_<arm>_rep<i>.json
# Afterwards:   python3 evaluation/analyze_repeats.py
#
# IMPORTANT: run this BEFORE adding new documents to the vector store.
# Agentic RAG searches every collection in the store, so the repetitions are
# only comparable with each other while the store contents stay the same.

set -e
REPS="${1:-3}"
THRESHOLD="${2:-0.65}"
RAGAS_PY="evaluation/.ragas_venv/bin/python"

DOCS=(
  "ccby:evaluation/documents/ccby_4.0_legalcode.pdf:evaluation/eval_questions_ccby.json"
  "cdc:evaluation/documents/cdc_program_evaluation_framework_2024.pdf:evaluation/eval_questions_cdc_program_eval.json"
  "ragbench:evaluation/documents/ragbench_2407.11005v2.pdf:evaluation/eval_questions_ragbench.json"
)
ARMS=(generation_v2 crag agentic_rrf agentic_dense)

R=evaluation/reports

for rep in $(seq 1 "$REPS"); do
  echo "##################################################################"
  echo "# Repetition ${rep} of ${REPS}"
  echo "##################################################################"
  for entry in "${DOCS[@]}"; do
    doc_key="${entry%%:*}"
    rest="${entry#*:}"
    doc_path="${rest%%:*}"
    questions_path="${rest#*:}"

    for arm in "${ARMS[@]}"; do
      tag="${doc_key}_${arm}_rep${rep}"
      traces_out="$R/decision_traces_${tag}.json"
      pred_out="$R/predictions_${tag}.json"
      ragas_out="$R/ragas_report_${tag}.json"

      if [ -f "$ragas_out" ]; then
        echo "--- ${tag} : already done, skipping ---"
        continue
      fi

      echo "--- ${tag} : collecting ---"
      case "$arm" in
        generation_v2)
          [ -f "$pred_out" ] || uv run python evaluation/collect_predictions.py \
            --document "$doc_path" --variant generation \
            --questions "$questions_path" --out "$pred_out"
          ;;
        crag)
          [ -f "$traces_out" ] || uv run python evaluation/collect_decision_traces.py \
            --strategy crag --document "$doc_path" \
            --questions "$questions_path" --out "$traces_out"
          uv run python evaluation/convert_traces_to_predictions.py --traces "$traces_out" --out "$pred_out"
          ;;
        agentic_rrf)
          [ -f "$traces_out" ] || uv run python evaluation/collect_decision_traces.py \
            --strategy agentic --document "$doc_path" --questions "$questions_path" \
            --agentic-merge-source rrf --out "$traces_out"
          uv run python evaluation/convert_traces_to_predictions.py --traces "$traces_out" --out "$pred_out"
          ;;
        agentic_dense)
          [ -f "$traces_out" ] || uv run python evaluation/collect_decision_traces.py \
            --strategy agentic --document "$doc_path" --questions "$questions_path" \
            --agentic-merge-source dense --agentic-dense-min-score "$THRESHOLD" --out "$traces_out"
          uv run python evaluation/convert_traces_to_predictions.py --traces "$traces_out" --out "$pred_out"
          ;;
      esac

      echo "--- ${tag} : RAGAS scoring ---"
      "$RAGAS_PY" evaluation/run_ragas_eval.py --predictions "$pred_out" --out "$ragas_out"
    done
  done
done

echo "Done. Now run: python3 evaluation/analyze_repeats.py"
