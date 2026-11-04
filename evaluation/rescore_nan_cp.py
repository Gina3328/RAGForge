"""Experiment 4: re-score the context_precision values that came back NaN (RAGAS timeouts) in the exp4 reports.

Why: context_precision asks the judge about every retrieved chunk one by one. Agentic (RRF) retrieves 130-155 chunks
per question in the 10-collection store, so many of those calls hit RAGAS's default 180 s timeout and the value is NaN.
Analysing only the questions that did NOT time out would be biased (the ones left are the questions with fewer chunks).

What it does (per report evaluation/reports/ragas_report_exp4_*.json):
  - finds the per-question context_precision values that are NaN / missing,
  - re-scores ONLY those questions, ONLY the context_precision metric, with the SAME judge (claude-haiku-4-5) and a
    longer timeout, fewer parallel workers,
  - keeps a copy of the original report once (*.json.orig), patches the NaN values in place, adds "cp_rescored": true
    to the patched questions, and recomputes averaged_scores.
Nothing else in a report changes. Values that are still NaN stay NaN (analyze_new_docs.py reports the counts).

Run it with the RAGAS venv, after run_new_docs.sh has produced the reports (any time after the `key` phase is fine):
    evaluation/.ragas_venv/bin/python evaluation/rescore_nan_cp.py --dry-run     # only lists what would be re-scored
    evaluation/.ragas_venv/bin/python evaluation/rescore_nan_cp.py               # does it (safe to Ctrl-C and re-run)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "evaluation" / "reports"


def is_nan(x) -> bool:
    if x is None:
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def keyed(questions):
    seen, out = {}, []
    for q in questions:
        seen[q] = seen.get(q, 0) + 1
        out.append((q, seen[q]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="ragas_report_exp4_*.json")
    ap.add_argument("--judge-model", default="claude-haiku-4-5")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per RAGAS job (default RAGAS value is 180)")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = []  # (report path, predictions path, [indexes of NaN context_precision])
    for rf in sorted(REPORTS.glob(args.glob)):
        if rf.name.endswith(".orig.json"):
            continue
        pf = REPORTS / rf.name.replace("ragas_report_", "predictions_")
        report = json.loads(rf.read_text(encoding="utf-8"))
        idx = [i for i, s in enumerate(report["per_sample"]) if is_nan(s.get("context_precision"))]
        if idx:
            todo.append((rf, pf, idx))
    total = sum(len(i) for _, _, i in todo)
    print(f"{total} NaN context_precision values in {len(todo)} reports:")
    for rf, _, idx in todo:
        print(f"   {rf.name}: {len(idx)}")
    if args.dry_run or not todo:
        return

    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_precision
    from ragas.run_config import RunConfig

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not found in the project's .env")
    judge = LangchainLLMWrapper(ChatAnthropic(model=args.judge_model, max_tokens=4096))
    run_config = RunConfig(timeout=args.timeout, max_workers=args.max_workers)

    for rf, pf, idx in todo:
        report = json.loads(rf.read_text(encoding="utf-8"))
        preds = json.loads(pf.read_text(encoding="utf-8"))
        if len(preds) != len(report["per_sample"]):
            print(f"SKIP {rf.name}: {len(preds)} predictions vs {len(report['per_sample'])} report rows")
            continue
        # make sure row i of the report is prediction i (same question text)
        if [p["user_input"] for p in preds] != [s["user_input"] for s in report["per_sample"]]:
            print(f"SKIP {rf.name}: question order differs between predictions and report")
            continue
        cols = {"user_input", "response", "retrieved_contexts", "reference"}
        items = [{k: v for k, v in preds[i].items() if k in cols and v is not None} for i in idx]
        print(f"--- {rf.name}: re-scoring {len(items)} questions (timeout {args.timeout}s, {args.max_workers} workers) ---")
        result = evaluate(dataset=EvaluationDataset.from_list(items), metrics=[context_precision], llm=judge,
                          run_config=run_config)
        df = result.to_pandas()
        orig = rf.with_name(rf.name + ".orig")
        if not orig.exists():
            shutil.copyfile(rf, orig)
        fixed = 0
        for j, i in enumerate(idx):
            v = df["context_precision"].iloc[j]
            if not is_nan(v):
                report["per_sample"][i]["context_precision"] = float(v)
                report["per_sample"][i]["cp_rescored"] = True
                fixed += 1
        vals = [s["context_precision"] for s in report["per_sample"] if not is_nan(s.get("context_precision"))]
        report["averaged_scores"]["context_precision"] = sum(vals) / len(vals) if vals else float("nan")
        rf.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"    {fixed} of {len(idx)} now have a value; still NaN: {len(idx) - fixed}")


if __name__ == "__main__":
    main()
