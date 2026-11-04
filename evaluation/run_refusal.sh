#!/usr/bin/env bash
# Experiment 7, step 1: collect the answers every strategy gives to answerable AND unanswerable questions.
#
# For each document one mixed question file is built: the 50 answerable questions of that document, the set-A
# unanswerable questions (15 per document) and the set-B unanswerable questions (8 per document). Every strategy
# answers the same mixed file in the same run, so the false-refusal rate (answerable questions that get refused)
# and the refusal rate on unanswerable questions come from the same code state and the same vector store.
# No RAGAS here: step 2 (classify_refusals.py) labels each answer as REFUSAL or ANSWER.
#
# Arms (order = priority; Ctrl-C after the first four if time is short, then analyse what exists):
#   generation_v2  Stage 2        crag           Stage 3 CRAG
#   agentic_rrf    Stage 4, RRF   agentic_dense  Stage 4, dense merge (threshold 0.65)
#   baseline       Stage 1        self_rag       Stage 3 Self-RAG        adaptive  Stage 3 Adaptive
#
# IMPORTANT: run this BEFORE adding the new documents to the vector store (Agentic searches all collections).
#
# Usage (project root, Docker/Milvus + Ollama running, unattended is fine):
#   bash evaluation/run_refusal.sh                         # all seven arms
#   bash evaluation/run_refusal.sh generation_v2,crag      # chosen arms (comma-separated)
# Safe to Ctrl-C and re-run: finished outputs are skipped.
# Output: evaluation/reports/refusal_mixed_<doc>.json (the questions) and refusal_<doc>_<arm>.json (the answers).
# Then:   evaluation/.ragas_venv/bin/python evaluation/classify_refusals.py --provider anthropic --judge-model <model>
#         python3 evaluation/analyze_refusals.py --labels <tag>
# Plan:   evaluation/ANALYSIS_PLAN_refusal.md

set -e
ARMS="${1:-generation_v2,crag,agentic_rrf,agentic_dense,baseline,self_rag,adaptive}"
THRESHOLD="${2:-0.65}"
R=evaluation/reports

DOCS=(
  "ccby|evaluation/documents/ccby_4.0_legalcode.pdf|evaluation/eval_questions_ccby.json|evaluation/eval_unanswerable_ccby.json|evaluation/eval_unanswerable_long_ccby.json"
  "cdc|evaluation/documents/cdc_program_evaluation_framework_2024.pdf|evaluation/eval_questions_cdc_program_eval.json|evaluation/eval_unanswerable_cdc_program_eval.json|evaluation/eval_unanswerable_long_cdc_program_eval.json"
  "ragbench|evaluation/documents/ragbench_2407.11005v2.pdf|evaluation/eval_questions_ragbench.json|evaluation/eval_unanswerable_ragbench.json|evaluation/eval_unanswerable_long_ragbench.json"
)

mkdir -p "$R"
# 1) Build the mixed question files.
for entry in "${DOCS[@]}"; do
  IFS='|' read -r key doc ans u1 u2 <<< "$entry"
  mixed="$R/refusal_mixed_${key}.json"
  [ -f "$mixed" ] && continue
  python3 - "$ans" "$u1" "$u2" "$mixed" <<'PY'
import json, sys
ans, u1, u2, out = sys.argv[1:5]
items = []
for q in json.load(open(ans, encoding="utf-8")):
    items.append({**q, "set": "answerable"})
for q in json.load(open(u1, encoding="utf-8")):
    items.append({**q, "set": "unans_A"})
for q in json.load(open(u2, encoding="utf-8")):
    items.append({**q, "set": "unans_B"})
json.dump(items, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{out}: {len(items)} questions")
PY
done

# 2) Collect the answers.
IFS=',' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
  for entry in "${DOCS[@]}"; do
    IFS='|' read -r key doc ans u1 u2 <<< "$entry"
    mixed="$R/refusal_mixed_${key}.json"
    out="$R/refusal_${key}_${arm}.json"
    if [ -f "$out" ]; then
      echo "--- ${key} / ${arm} : already done, skipping ---"
      continue
    fi
    echo "--- ${key} / ${arm} : collecting ---"
    case "$arm" in
      baseline)
        uv run python evaluation/collect_predictions.py --document "$doc" --variant baseline --questions "$mixed" --out "$out" ;;
      generation_v2)
        uv run python evaluation/collect_predictions.py --document "$doc" --variant generation --questions "$mixed" --out "$out" ;;
      self_rag|crag|adaptive)
        uv run python evaluation/collect_decision_traces.py --strategy "$arm" --document "$doc" --questions "$mixed" --out "$out" ;;
      agentic_rrf)
        uv run python evaluation/collect_decision_traces.py --strategy agentic --document "$doc" --questions "$mixed" \
          --agentic-merge-source rrf --out "$out" ;;
      agentic_dense)
        uv run python evaluation/collect_decision_traces.py --strategy agentic --document "$doc" --questions "$mixed" \
          --agentic-merge-source dense --agentic-dense-min-score "$THRESHOLD" --out "$out" ;;
      *) echo "unknown arm: $arm" >&2; exit 1 ;;
    esac
  done
done
echo "Done. Next: classify_refusals.py (see the header of this script)."
