#!/usr/bin/env bash
# Experiment 5: Agentic cross-collection retrieval vs the NUMBER OF COLLECTIONS (N = 3, 6, 10), RRF merge vs dense merge.
# Plan (fixed before running): evaluation/ANALYSIS_PLAN_collection_count.md
#
# Before the first run (project root; Docker/Milvus + Ollama running):
#   uv run python evaluation/index_new_docs.py      # resets the store (experiment 4's Stage 1 run changed it); must show 10 collections
# Then:
#   caffeinate -i bash evaluation/run_collection_count.sh
#   python3 evaluation/analyze_collection_count.py
#
# 6 runs (2 arms x 3 sizes), each over the 50 questions of ccby / cdc / ragbench. Traces only, no RAGAS.
# Safe to Ctrl-C and re-run: a finished document file is skipped. Same dense threshold (0.65) as experiment 4.
set -e
for n in 3 6 10; do
  for arm in rrf dense; do
    echo "################ arm=${arm}  N=${n} ################"
    uv run python evaluation/run_collection_count.py --arm "$arm" --n "$n"
  done
done
echo "Done. Next: python3 evaluation/analyze_collection_count.py"
