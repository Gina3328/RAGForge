"""Summarize the Agentic RAG stop-decision window experiment (Finding 3).

Reads evaluation/reports/decision_traces_<doc>_agentic_<arm>.json (JSON only,
no Milvus / LLM). For every arm prints, per document and in total:
  - how the questions ended: stopped in round 1 / rounds 2-4 / chose GENERATE in
    round 5 / "stuck" (round 5 still RETRIEVE, cut off by max_iterations)
  - average number of decision rounds
  - what the decision window contained: mean share of the window's chunks that
    come from the question's own (target) collection
Reference rows (older runs, no window logging) show the stop distribution only.

Usage: uv run python evaluation/analyze_agentic_window.py [--reports-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DOCS = ["ccby", "cdc", "ragbench"]
ARMS = [
    ("win_rrf", "RRF order, 10x200 (original)"),
    ("win_dense", "dense order, 10x200"),
    ("win_big", "RRF order, 20x500"),
    ("win_score_rrf", "RRF order, 10x200, RRF score shown"),
    ("win_score_dense", "RRF order, 10x200, cosine shown"),
]
REFERENCE = [
    ("agentic", "ref: Sep 25 run (no window log)"),
    ("agentic_post_flush_fix", "ref: Sep 28 run (no window log)"),
    ("agentic_merge_rrf", "ref: Oct 5 run (no window log)"),
]


def norm(name: str) -> str:
    return (name or "").removeprefix("ragforge_")


def classify(rounds: list[dict]) -> str:
    last, n = rounds[-1]["action"], len(rounds)
    if last == "GENERATE":
        return "r1" if n == 1 else ("r2_4" if n < 5 else "g5")
    return "stuck"


def load(reports: Path, doc: str, suffix: str):
    f = reports / f"decision_traces_{doc}_{suffix}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def summarize(data: dict) -> dict:
    target = norm(data.get("collection", ""))
    c = dict(n=0, r1=0, r2_4=0, g5=0, stuck=0, rounds=0)
    shares: list[float] = []
    shares_r1: list[float] = []
    shares_stuck: list[float] = []
    for rec in data["records"]:
        rounds = rec["trace"]["rounds"]
        kind = classify(rounds)
        c["n"] += 1
        c[kind] += 1
        c["rounds"] += len(rounds)
        for r in rounds:
            w = r.get("window") or {}
            size = w.get("size") or 0
            if not size:
                continue
            share = norm_get(w["by_collection"], target) / size
            shares.append(share)
            if r["iteration"] == 1:
                shares_r1.append(share)
            if kind == "stuck":
                shares_stuck.append(share)
    c["shares"], c["shares_r1"], c["shares_stuck"] = shares, shares_r1, shares_stuck
    return c


def norm_get(by_collection: dict, target: str) -> int:
    return sum(v for k, v in by_collection.items() if norm(k) == target)


def mean(v: list[float]) -> str:
    return f"{100 * sum(v) / len(v):.0f}%" if v else "-"


def row(label: str, c: dict) -> str:
    avg = f"{c['rounds'] / c['n']:.2f}" if c["n"] else "-"
    stuck = f"{c['stuck']}/{c['n']}"
    return (
        f"{label:34s} n={c['n']:2d}  r1={c['r1']:2d}  r2-4={c['r2_4']:2d}  g5={c['g5']:2d}  "
        f"stuck={stuck:6s} avg_rounds={avg:5s} window_target_share: all={mean(c['shares']):>4s} "
        f"round1={mean(c['shares_r1']):>4s} stuck_q={mean(c['shares_stuck']):>4s}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default=str(Path(__file__).parent / "reports"))
    reports = Path(ap.parse_args().reports_dir)

    for suffix_prefix, items in (("agentic_", ARMS), ("", REFERENCE)):
        for key, desc in items:
            suffix = f"{suffix_prefix}{key}"
            print(f"\n## {desc}   [{suffix}]")
            tot = dict(n=0, r1=0, r2_4=0, g5=0, stuck=0, rounds=0, shares=[], shares_r1=[], shares_stuck=[])
            found = False
            for doc in DOCS:
                data = load(reports, doc, suffix)
                if data is None:
                    print(f"  {doc:9s} (file missing)")
                    continue
                found = True
                c = summarize(data)
                print("  " + row(doc, c))
                for k in tot:
                    tot[k] = tot[k] + c[k]
            if found:
                print("  " + row("TOTAL", tot))
    print(
        "\nstuck = round 5 still RETRIEVE (cut off by max_iterations); g5 = chose GENERATE in round 5; "
        "window_target_share = share of the decision window's chunks that come from the question's own collection."
    )


if __name__ == "__main__":
    main()
