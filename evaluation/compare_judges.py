"""Experiment 3, analysis: do the paper's conclusions survive a change of judge model?

Reads the original RAGAS reports (judge = claude-haiku-4-5) and the re-scored ones in
evaluation/reports/rescored/<tag>/, and reports, for every judge:
  1. per-arm means (mean of the three per-document means; this is how the cross-doc averages in the paper are built),
  2. rank agreement of the six Stage 1-4 arms between the original judge and each new judge (Spearman),
  3. per-question agreement between judges (Spearman),
  4. the pre-registered contrasts (paired bootstrap over questions, 95% CI), with a robustness verdict.
Standard library only:

    python3 evaluation/compare_judges.py --judges openai_<model> anthropic_<model>

Writes evaluation/reports/judge_comparison.json. Rules for reading the result: evaluation/ANALYSIS_PLAN_judges.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "evaluation" / "reports"

DOCS = ["ccby", "cdc", "ragbench"]
MAIN_ARMS = ["baseline", "generation_v2", "self_rag", "crag", "adaptive", "agentic"]
ALL_ARMS = MAIN_ARMS + ["agentic_merge_rrf", "agentic_merge_dense"]
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
SHORT = {"faithfulness": "faith", "answer_relevancy": "ans_rel", "context_precision": "cp", "context_recall": "cr"}

# Fixed before the new judges were run (see ANALYSIS_PLAN_judges.md). (name, arm_a, arm_b, metric); difference = a - b.
CONTRASTS = [
    ("C1 Stage 2 vs Stage 1: context_precision", "generation_v2", "baseline", "context_precision"),
    ("C2 Agentic vs Stage 2: context_precision", "agentic", "generation_v2", "context_precision"),
    ("C3 Agentic vs Stage 2: context_recall", "agentic", "generation_v2", "context_recall"),
    ("C4 CRAG vs Stage 2: context_recall", "crag", "generation_v2", "context_recall"),
    ("C5 CRAG vs Stage 2: faithfulness", "crag", "generation_v2", "faithfulness"),
    ("C6 Self-RAG vs Stage 2: faithfulness", "self_rag", "generation_v2", "faithfulness"),
    ("C7 Agentic(dense merge) vs Agentic(RRF merge): context_precision", "agentic_merge_dense", "agentic_merge_rrf", "context_precision"),
]


def fname(doc: str, arm: str) -> str:
    if arm.startswith("agentic_merge_"):
        return f"ragas_report_{doc}_{arm}.json"
    return f"ragas_report_{doc}_{arm}_post_flush_fix.json"


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def load_judge(folder: Path) -> dict:
    """arm -> {(doc, question): {metric: value or None}}; only arms with all three docs present."""
    data: dict[str, dict] = {}
    for arm in ALL_ARMS:
        per_arm = {}
        ok = True
        for doc in DOCS:
            p = folder / fname(doc, arm)
            if not p.exists():
                ok = False
                break
            for s in json.loads(p.read_text(encoding="utf-8"))["per_sample"]:
                per_arm[(doc, s["user_input"])] = {m: num(s.get(m)) for m in METRICS}
        if ok:
            data[arm] = per_arm
    return data


def arm_mean(per_arm: dict, metric: str):
    """Mean of per-document means (documents weighted equally)."""
    means = []
    for doc in DOCS:
        vals = [v[metric] for (d, _), v in per_arm.items() if d == doc and v[metric] is not None]
        if not vals:
            return None
        means.append(sum(vals) / len(vals))
    return sum(means) / len(means)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def paired_diffs(data: dict, a: str, b: str, metric: str):
    if a not in data or b not in data:
        return None
    out = []
    for key, va in data[a].items():
        vb = data[b].get(key)
        if vb is None or va[metric] is None or vb[metric] is None:
            continue
        out.append(va[metric] - vb[metric])
    return out


def boot_ci(diffs, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def f(x, nd=3):
    return "  n/a " if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="*", default=[], help="Folder names inside evaluation/reports/rescored/.")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    judges = {"original (claude-haiku-4-5)": load_judge(REPORTS)}
    for tag in args.judges:
        folder = REPORTS / "rescored" / tag
        if not folder.exists():
            raise SystemExit(f"{folder} not found")
        judges[tag] = load_judge(folder)

    summary: dict = {"arm_means": {}, "rank_agreement": {}, "question_agreement": {}, "contrasts": {}}

    print("== 1. Per-arm means (mean of the 3 per-document means) ==")
    print(f"{'judge':34s}{'arm':22s}" + "".join(f"{SHORT[m]:>9s}" for m in METRICS))
    for jn, data in judges.items():
        summary["arm_means"][jn] = {}
        for arm in ALL_ARMS:
            if arm not in data:
                continue
            row = {m: arm_mean(data[arm], m) for m in METRICS}
            summary["arm_means"][jn][arm] = row
            print(f"{jn:34s}{arm:22s}" + "".join(f"{f(row[m]):>9s}" for m in METRICS))
        print()

    orig = judges["original (claude-haiku-4-5)"]
    print("== 2. Agreement with the original judge ==")
    print("Rank = Spearman of the six Stage 1-4 arm means; Question = Spearman over all re-scored questions.")
    print(f"{'judge':34s}" + "".join(f"{SHORT[m] + ' rank':>12s}{SHORT[m] + ' quest':>12s}" for m in METRICS))
    for jn, data in judges.items():
        if jn == "original (claude-haiku-4-5)":
            continue
        cells = []
        summary["rank_agreement"][jn] = {}
        summary["question_agreement"][jn] = {}
        for m in METRICS:
            arms = [a for a in MAIN_ARMS if a in data and a in orig]
            ra = spearman([arm_mean(orig[a], m) for a in arms], [arm_mean(data[a], m) for a in arms]) if len(arms) >= 4 else None
            xs, ys = [], []
            for a in ALL_ARMS:
                if a in data and a in orig:
                    for key, v in data[a].items():
                        o = orig[a].get(key)
                        if o and o[m] is not None and v[m] is not None:
                            xs.append(o[m])
                            ys.append(v[m])
            qa = spearman(xs, ys) if xs else None
            summary["rank_agreement"][jn][m] = ra
            summary["question_agreement"][jn][m] = {"spearman": qa, "n": len(xs)}
            cells += [f"{f(ra):>12s}", f"{f(qa):>12s}"]
        print(f"{jn:34s}" + "".join(cells))
    print()

    print("== 3. Pre-registered contrasts (difference a - b, 95% paired bootstrap CI over questions) ==")
    for name, a, b, m in CONTRASTS:
        print(name)
        summary["contrasts"][name] = {}
        for jn, data in judges.items():
            d = paired_diffs(data, a, b, m)
            if not d:
                print(f"   {jn:34s} (arms not available)")
                continue
            lo, hi = boot_ci(d, args.boot)
            mean = sum(d) / len(d)
            sig = lo > 0 or hi < 0
            summary["contrasts"][name][jn] = {"diff": mean, "lo": lo, "hi": hi, "n": len(d), "ci_excludes_0": sig}
            print(f"   {jn:34s} diff {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={len(d)}  {'CI excludes 0' if sig else 'CI includes 0'}")
        # Robustness verdict (rule fixed in the analysis plan).
        res = summary["contrasts"][name]
        o = res.get("original (claude-haiku-4-5)")
        others = [v for k, v in res.items() if k != "original (claude-haiku-4-5)"]
        if o and others:
            if o["ci_excludes_0"]:
                same = all(v["ci_excludes_0"] and (v["diff"] > 0) == (o["diff"] > 0) for v in others)
                weak = all((v["diff"] > 0) == (o["diff"] > 0) for v in others)
                verdict = "ROBUST" if same else ("SAME SIGN, NOT SIGNIFICANT UNDER A NEW JUDGE" if weak else "NOT ROBUST (sign flips)")
            else:
                verdict = "n/a (original CI includes 0)"
            res["verdict"] = verdict
            print(f"   -> {verdict}")
        print()

    out = REPORTS / "judge_comparison.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Written {out}")


if __name__ == "__main__":
    main()
