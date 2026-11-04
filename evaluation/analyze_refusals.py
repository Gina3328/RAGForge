"""Experiment 7, step 3: refusal / false-refusal / unsupported-answer rates per strategy.

Uses the labels written by classify_refusals.py. Standard library only:

    python3 evaluation/analyze_refusals.py --labels anthropic_<model> [openai_<model>]

If a second label set is given, the agreement between the two judges is printed, and the tables use the FIRST
label set. Definitions (also in evaluation/ANALYSIS_PLAN_refusal.md):
  false refusal      = share of ANSWERABLE questions whose response is a REFUSAL            (lower is better)
  near refusal       = share of on-topic UNANSWERABLE questions (set A + set B) refused      (higher is better)
  unsupported answer = 1 - near refusal (a specific answer to a question the document cannot answer)
  far refusal        = share of the unrelated general-knowledge questions refused (reported, not interpreted: a model
                       may legitimately answer these from general knowledge)
  discrimination J   = near refusal - false refusal. 0 = refuses answerable and unanswerable questions alike.
Writes evaluation/reports/refusal_summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

REPORTS = Path(__file__).resolve().parent / "reports"
DOCS = ["ccby", "cdc", "ragbench"]
ARMS = ["baseline", "generation_v2", "self_rag", "crag", "adaptive", "agentic_rrf", "agentic_dense"]
NAME = {"baseline": "Stage 1", "generation_v2": "Stage 2", "self_rag": "Self-RAG", "crag": "CRAG",
        "adaptive": "Adaptive", "agentic_rrf": "Agentic (RRF merge)", "agentic_dense": "Agentic (dense merge)"}
# Fixed in advance (see the plan). (name, arm_a, arm_b, group, what is compared)  difference = a - b
CONTRASTS = [
    ("K1 false refusal: CRAG - Stage 2", "crag", "generation_v2", "answerable"),
    ("K2 near refusal: CRAG - Stage 2", "crag", "generation_v2", "near"),
    ("K3 false refusal: Agentic(RRF) - Stage 2", "agentic_rrf", "generation_v2", "answerable"),
    ("K4 near refusal: Agentic(RRF) - Stage 2", "agentic_rrf", "generation_v2", "near"),
    ("K5 near refusal: Agentic(dense) - Agentic(RRF)", "agentic_dense", "agentic_rrf", "near"),
    ("K6 near refusal: Stage 2 - Stage 1", "generation_v2", "baseline", "near"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def group_of(meta: dict) -> str:
    if meta["set"] == "answerable":
        return "answerable"
    return "far" if meta.get("type") == "unanswerable_far" else "near"


def load(label_tag: str):
    labels = json.loads((REPORTS / f"refusal_labels_{label_tag}.json").read_text(encoding="utf-8"))["labels"]
    meta = {}
    for doc in DOCS:
        p = REPORTS / f"refusal_mixed_{doc}.json"
        if p.exists():
            for q in json.loads(p.read_text(encoding="utf-8")):
                meta[(doc, q["question"])] = q
    # arm -> {(doc, question): 1 if REFUSAL, 0 if ANSWER}
    table: dict[str, dict] = {}
    for key, v in labels.items():
        doc, arm, q = key.split("|", 2)
        if v["label"] not in ("REFUSAL", "ANSWER") or (doc, q) not in meta:
            continue
        table.setdefault(arm, {})[(doc, q)] = 1 if v["label"] == "REFUSAL" else 0
    return table, meta


def rate(table, meta, arm, group, set_name=None):
    vals = [v for k, v in table.get(arm, {}).items()
            if group_of(meta[k]) == group and (set_name is None or meta[k]["set"] == set_name)]
    return sum(vals), len(vals)


def boot_diff(a, b, keys, n_boot=2000, seed=0):
    rng = random.Random(seed)
    diffs = [a[k] - b[k] for k in keys]
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(diffs) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1], n


def boot_j(table, meta, arm, n_boot=2000, seed=0):
    """Stratified bootstrap of J = near refusal - false refusal."""
    rng = random.Random(seed)
    ans = [v for k, v in table[arm].items() if group_of(meta[k]) == "answerable"]
    near = [v for k, v in table[arm].items() if group_of(meta[k]) == "near"]
    if not ans or not near:
        return None
    js = sorted(sum(near[rng.randrange(len(near))] for _ in near) / len(near)
                - sum(ans[rng.randrange(len(ans))] for _ in ans) / len(ans) for _ in range(n_boot))
    j = sum(near) / len(near) - sum(ans) / len(ans)
    return j, js[int(0.025 * n_boot)], js[int(0.975 * n_boot) - 1]


def fmt(k, n):
    if n == 0:
        return "   n/a        "
    lo, hi = wilson(k, n)
    return f"{k:3d}/{n:<3d} {100 * k / n:3.0f}% [{100 * lo:2.0f},{100 * hi:3.0f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True, help="Tag(s) of refusal_labels_<tag>.json")
    args = ap.parse_args()
    table, meta = load(args.labels[0])
    arms = [a for a in ARMS if a in table]
    summary: dict = {"labels": args.labels[0], "rates": {}, "contrasts": {}}

    print(f"Label set: {args.labels[0]}\n")
    print("== 1. Rates (count/total, share, 95% Wilson CI in %) ==")
    print(f"{'arm':24s}{'false refusal (answerable)':>30s}{'near refusal (A+B)':>26s}{'  set A':>22s}{'  set B':>22s}{'far refusal':>24s}{'J':>8s}")
    for arm in arms:
        fa = rate(table, meta, arm, "answerable")
        nr = rate(table, meta, arm, "near")
        na = rate(table, meta, arm, "near", "unans_A")
        nb = rate(table, meta, arm, "near", "unans_B")
        fr = rate(table, meta, arm, "far")
        jj = boot_j(table, meta, arm)
        jtxt = f"{jj[0]:+.2f}" if jj else "n/a"
        print(f"{NAME[arm]:24s}{fmt(*fa):>30s}{fmt(*nr):>26s}{fmt(*na):>22s}{fmt(*nb):>22s}{fmt(*fr):>24s}{jtxt:>8s}")
        summary["rates"][arm] = {"answerable": fa, "near": nr, "near_A": na, "near_B": nb, "far": fr,
                                 "J": list(jj) if jj else None}
    print("\nJ = near refusal - false refusal (95% bootstrap CI):")
    for arm in arms:
        jj = boot_j(table, meta, arm)
        if jj:
            print(f"  {NAME[arm]:24s} {jj[0]:+.3f}  [{jj[1]:+.3f}, {jj[2]:+.3f}]")

    print("\n== 2. Pre-registered contrasts (difference a - b in refusal rate, paired by question, 95% bootstrap CI) ==")
    for name, a, b, group in CONTRASTS:
        if a not in table or b not in table:
            print(f"{name}: arms not available")
            continue
        keys = [k for k in table[a] if k in table[b] and group_of(meta[k]) == group]
        if not keys:
            continue
        d, lo, hi, n = boot_diff(table[a], table[b], keys)
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
        print(f"{name:52s} {100 * d:+6.1f} pts  [{100 * lo:+.1f}, {100 * hi:+.1f}]  n={n}  {sig}")
        summary["contrasts"][name] = {"diff": d, "lo": lo, "hi": hi, "n": n, "ci_excludes_0": lo > 0 or hi < 0}

    if len(args.labels) > 1:
        t2, _ = load(args.labels[1])
        both = [(table[a][k], t2[a][k]) for a in arms if a in t2 for k in table[a] if k in t2[a]]
        n = len(both)
        agree = sum(1 for x, y in both if x == y)
        p1 = sum(x for x, _ in both) / n
        p2 = sum(y for _, y in both) / n
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        po = agree / n
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        print(f"\n== 3. Two judges ({args.labels[0]} vs {args.labels[1]}) ==")
        print(f"agreement {agree}/{n} = {100 * po:.1f}%, Cohen's kappa = {kappa:.2f}")
        summary["judge_agreement"] = {"n": n, "agreement": po, "kappa": kappa}

    out = REPORTS / "refusal_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten {out}")


if __name__ == "__main__":
    main()
