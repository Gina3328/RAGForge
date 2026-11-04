#!/usr/bin/env bash
# Experiment 4: run the strategies on the 7 NEW documents (174 answerable questions) and score them with RAGAS.
# Analysis plan (fixed before running): evaluation/ANALYSIS_PLAN_new_docs.md
#
# Order of operations (project root; Docker/Milvus + Ollama running; unattended is fine):
#   1) uv run python evaluation/index_new_docs.py        # once: the store must hold 10 collections (3 old + 7 new documents)
#   2) bash evaluation/run_new_docs.sh key               # 4 key arms: generation_v2, crag, agentic_rrf, agentic_dense
#   3) bash evaluation/run_new_docs.sh rest              # 3 more arms: self_rag, adaptive, baseline (Stage 1)
#   4) python3 evaluation/analyze_new_docs.py            # (written separately, after the run has started)
#
# Why two phases: Agentic searches EVERY collection in the store. Stage 1 (baseline) rebuilds its own document's
# collection with the fixed-size chunker, which would change what a later Agentic run sees. So all Agentic runs are
# done in phase "key", and Stage 1 comes last in phase "rest". The script keeps a marker file after Stage 1 has run;
# if you ever need to run phase "key" again, re-run index_new_docs.py first (it removes the marker).
#
# Safe to Ctrl-C and re-run: a finished RAGAS report is skipped; unfinished steps are redone.
# Usage: bash evaluation/run_new_docs.sh key|rest [dense-min-score=0.65]
# Output: evaluation/reports/{decision_traces,predictions,ragas_report}_exp4_<doc>_<arm>.json
# The dense threshold 0.65 is the one fixed in Finding 2 on the original documents; it is NOT re-tuned here.

set -e
PHASE="${1:-}"
THRESHOLD="${2:-0.65}"
RAGAS_PY="evaluation/.ragas_venv/bin/python"
R=evaluation/reports
MARKER="$R/.exp4_store_dirty"

case "$PHASE" in
  key)  ARMS=(generation_v2 crag agentic_rrf agentic_dense) ;;
  rest) ARMS=(self_rag adaptive baseline) ;;
  *) echo "usage: bash evaluation/run_new_docs.sh key|rest [dense-min-score]" >&2; exit 1 ;;
esac

if [ "$PHASE" = "key" ] && [ -f "$MARKER" ]; then
  echo "Stage 1 (baseline) has already rebuilt some collections with the fixed-size chunker." >&2
  echo "Run 'uv run python evaluation/index_new_docs.py' first (it resets the store state), then re-run this." >&2
  exit 1
fi

# Read the document list (evaluation/new_docs.tsv): key <TAB> document glob <TAB> questions file.
DOC_KEYS=(); DOC_PATHS=(); DOC_QS=()
while IFS=$'\t' read -r key doc_glob questions; do
  [ -z "$key" ] && continue
  case "$key" in \#*) continue ;; esac
  doc_path="$(compgen -G "$doc_glob" | head -1)"
  [ -n "$doc_path" ] || { echo "no file matches: $doc_glob" >&2; exit 1; }
  [ -f "$questions" ] || { echo "missing questions file: $questions" >&2; exit 1; }
  DOC_KEYS+=("$key"); DOC_PATHS+=("$doc_path"); DOC_QS+=("$questions")
done < evaluation/new_docs.tsv
echo "Phase: $PHASE   arms: ${ARMS[*]}   documents: ${DOC_KEYS[*]}"

for i in "${!DOC_KEYS[@]}"; do
  key="${DOC_KEYS[$i]}"; doc="${DOC_PATHS[$i]}"; qs="${DOC_QS[$i]}"
  for arm in "${ARMS[@]}"; do
    tag="exp4_${key}_${arm}"
    traces_out="$R/decision_traces_${tag}.json"
    pred_out="$R/predictions_${tag}.json"
    ragas_out="$R/ragas_report_${tag}.json"
    if [ -f "$ragas_out" ]; then
      echo "--- ${tag} : already done, skipping ---"
      continue
    fi
    echo "--- ${tag} : collecting ---"
    case "$arm" in
      baseline)
        touch "$MARKER"
        [ -f "$pred_out" ] || uv run python evaluation/collect_predictions.py \
          --document "$doc" --variant baseline --questions "$qs" --out "$pred_out" ;;
      generation_v2)
        [ -f "$pred_out" ] || uv run python evaluation/collect_predictions.py \
          --document "$doc" --variant generation --questions "$qs" --out "$pred_out" ;;
      self_rag|crag|adaptive)
        [ -f "$traces_out" ] || uv run python evaluation/collect_decision_traces.py \
          --strategy "$arm" --document "$doc" --questions "$qs" --out "$traces_out"
        uv run python evaluation/convert_traces_to_predictions.py --traces "$traces_out" --out "$pred_out" ;;
      agentic_rrf)
        [ -f "$traces_out" ] || uv run python evaluation/collect_decision_traces.py \
          --strategy agentic --document "$doc" --questions "$qs" --agentic-merge-source rrf --out "$traces_out"
        uv run python evaluation/convert_traces_to_predictions.py --traces "$traces_out" --out "$pred_out" ;;
      agentic_dense)
        [ -f "$traces_out" ] || uv run python evaluation/collect_decision_traces.py \
          --strategy agentic --document "$doc" --questions "$qs" \
          --agentic-merge-source dense --agentic-dense-min-score "$THRESHOLD" --out "$traces_out"
        uv run python evaluation/convert_traces_to_predictions.py --traces "$traces_out" --out "$pred_out" ;;
    esac
    echo "--- ${tag} : RAGAS scoring ---"
    "$RAGAS_PY" evaluation/run_ragas_eval.py --predictions "$pred_out" --out "$ragas_out"
  done
done
echo "Phase $PHASE done."
[ "$PHASE" = "key" ] && echo "Next: bash evaluation/run_new_docs.sh rest" || echo "Next: python3 evaluation/analyze_new_docs.py"
