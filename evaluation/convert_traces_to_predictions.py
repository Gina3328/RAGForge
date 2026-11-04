"""Phase 1c of the evaluation add-on: convert a Stage 3/4 decision-trace
file (collect_decision_traces.py's output) into a RAGAS-scorable
predictions file that run_ragas_eval.py already knows how to read.

Why this exists: collect_decision_traces.py records question / type /
reference / answerable / answer / trace / retrieved_contexts per question --
everything needed to compute RAGAS's faithfulness / answer_relevancy /
context_precision / context_recall, just under different key names and
wrapped in a {"strategy": ..., "records": [...]} envelope instead of the
flat list run_ragas_eval.py expects. This script does nothing but reshape:
no new pipeline call is made, no dependency is added, no data is
generated -- it is pure data massaging, same spirit as the rest of
evaluation/.

Usage (from the project root, e.g. via `uv run` -- runs in the normal
RagForge .venv, no ragas import needed here, this file only uses stdlib):

    uv run python evaluation/convert_traces_to_predictions.py \\
        --traces evaluation/reports/decision_traces_crag.json \\
        --out evaluation/reports/predictions_ccby_crag.json

Then score it exactly like any Stage 1/2 predictions file, in the separate
ragas venv:

    evaluation/.ragas_venv/bin/python evaluation/run_ragas_eval.py \\
        --predictions evaluation/reports/predictions_ccby_crag.json \\
        --out evaluation/reports/ragas_report_ccby_crag.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a decision_traces_<strategy>.json file into a RAGAS-scorable predictions file."
    )
    parser.add_argument("--traces", required=True, help="Path to a decision_traces_<strategy>[_<label>].json file.")
    parser.add_argument("--out", required=True, help="Where to write the converted predictions file.")
    args = parser.parse_args()

    with open(args.traces, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["records"]
    predictions = []
    missing_contexts = 0
    for r in records:
        if "retrieved_contexts" not in r:
            # Trace was collected before collect_decision_traces.py started
            # recording this field -- re-run collect_decision_traces.py
            # rather than silently scoring with an empty context list (that
            # would tank faithfulness/context_precision for the wrong
            # reason and look like a real pipeline regression).
            missing_contexts += 1
            continue
        predictions.append(
            {
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["retrieved_contexts"],
                "reference": r.get("reference"),
            }
        )

    if missing_contexts:
        raise SystemExit(
            f'{missing_contexts}/{len(records)} record(s) in {args.traces} have no '
            '"retrieved_contexts" field -- this trace file was collected with an '
            "older version of collect_decision_traces.py. Re-run it to regenerate."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    strategy = data.get("strategy", "?")
    print(f"Converted {len(predictions)} {strategy} records from {args.traces} -> {out_path}")


if __name__ == "__main__":
    main()
