"""Summarise the repeated runs of run_repeats.sh: run-to-run variation and
paired confidence intervals for a small set of contrasts that are fixed in
advance (see ANALYSIS_PLAN_repeats.md). Standard library only.

Reads evaluation/reports/ragas_report_<doc>_<arm>_rep<i>.json, which hold the
per-question RAGAS scores.

Usage:
    python3 evaluation/analyze_repeats.py
    python3 evaluation/analyze_repeats.py --reports-dir some/dir --boot 5000
Writes evaluation/reports/repeats_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS = PROJECT_ROOT / "evaluation" / "reports"

DOCS = ["ccby", "cdc", "ragbench"]
ARMS = ["generation_v2", "crag", "agentic_rrf", "agentic_dense"]
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
LABEL = {
    "generation_v2": "Stage 2",
    "crag": "CRAG",
    "agentic_rrf": "Agentic (RRF merge)",
    "agentic_dense": "Agentic (dense merge)",
}
# Contrasts fixed in advance: (arm A, arm B, metric); reported as A - B.
CONTRASTS = [
    ("agentic_rrf", "generation_v2", "context_precision"),
    ("agentic_rrf", "generation_v2", "context_recall"),
    ("agentic_dense", "agentic_rrf", "context_precision"),
    ("agentic_dense", "agentic_rrf", "context_recall"),
    ("crag", "generation_v2", "context_recall"),
    ("crag", "generation_v2", "faithfulness"),
    ("agentic_rrf", "crag", "context_precision"),
    ("agentic_rrf", "crag", "context_recall"),
]


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def load(reports_dir: Path):
    """scores[arm][rep][(doc, question)] = {metric: value or None}"""
    scores = {a: defaultdict(dict) for a in ARMS}
    pat = re.compile(r"ragas_report_(?P<doc>[a-z]+)_(?P<arm>[a-z_0-9]+)_rep(?P<rep>\d+)\.json$")
    for p in sorted(reports_dir.glob("ragas_report_*_rep*.json")):
        m = pat.match(p.name)
        if not m or m["arm"] not in ARMS or m["doc"] not in DOCS:
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for s in data["per_sample"]:
            scores[m["arm"]][int(m["rep"])][(m["doc"], s["user_input"])] = {
                k: fnum(s.get(k)) for k in METRICS
            }
    return scores


def complete_reps(scores):
    """Repetition numbers for which every arm has all three documents."""
    reps = None
    for a in ARMS:
        have = set(scores[a].keys())
        reps = have if reps is None else reps & have
    # a rep is complete only if each arm covers every document
    ok = []
    for r in sorted(reps or []):
        if all(
            {d for (d, _q) in scores[a][r]} == set(DOCS) for a in ARMS
        ):
            ok.append(r)
    return ok


def rep_mean(per_q: dict, metric: str):
    xs = [v[metric] for v in per_q.values() if v[metric] is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def question_means(scores, arm, reps, metric):
    """Per-question score averaged over repetitions (less noisy than one run)."""
    acc = defaultdict(list)
    for r in reps:
        for key, v in scores[arm][r].items():
            if v[metric] is not None:
                acc[key].append(v[metric])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def paired_boot(a: dict, b: dict, boot: int, seed: int = 0):
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    means = []
    for _ in range(boot):
        means.append(sum(rng.choice(d) for _ in d) / len(d))
    return {
        "n_questions": len(d),
        "diff": sum(d) / len(d),
        "ci95": (pct(means, 0.025), pct(means, 0.975)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default=str(DEFAULT_REPORTS))
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    rdir = Path(args.reports_dir)

    scores = load(rdir)
    reps = complete_reps(scores)
    if not reps:
        print("No complete repetition found (every arm x every document). Has run_repeats.sh finished a full repetition?")
        return
    print(f"Complete repetitions: {reps}  (n = {len(reps)})")
    if len(reps) < 3:
        print("  note: fewer than 3 repetitions, SD is not reliable yet\n")

    out: dict = {"repetitions": reps, "arms": {}, "contrasts": []}

    print("\nMean over all questions, per repetition and across repetitions")
    header = f"{'arm':24s}" + "".join(f"{m[:16]:>20s}" for m in METRICS)
    print(header)
    for arm in ARMS:
        row = f"{LABEL[arm]:24s}"
        out["arms"][arm] = {}
        for m in METRICS:
            vals = [rep_mean(scores[arm][r], m) for r in reps]
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            out["arms"][arm][m] = {
                "per_rep": vals,
                "mean": mean,
                "sd": sd,
                "min": min(vals),
                "max": max(vals),
            }
            row += f"{mean:>10.3f} ±{sd:5.3f}  " if len(vals) > 1 else f"{mean:>10.3f}          "
        print(row)
    print("(± = standard deviation across repetitions of the 50-question mean = run-to-run noise)")

    print("\nContrasts fixed in advance, A - B, paired bootstrap over questions")
    print("(each question's score is first averaged over the repetitions)")
    for a, b, m in CONTRASTS:
        res = paired_boot(
            question_means(scores, a, reps, m), question_means(scores, b, reps, m), args.boot
        )
        sig = "CI excludes 0" if res["ci95"][0] > 0 or res["ci95"][1] < 0 else "CI includes 0"
        print(
            f"  {LABEL[a]:22s} - {LABEL[b]:22s} {m:18s} "
            f"{res['diff']:+.3f}  [{res['ci95'][0]:+.3f}, {res['ci95'][1]:+.3f}]  {sig}  (n={res['n_questions']})"
        )
        out["contrasts"].append({"a": a, "b": b, "metric": m, **res, "ci_excludes_zero": sig == "CI excludes 0"})

    (rdir / "repeats_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {rdir / 'repeats_summary.json'}")


if __name__ == "__main__":
    main()
