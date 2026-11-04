"""Experiment 5, analysis: Agentic cross-collection retrieval vs the number of collections N (3, 6, 10), RRF vs dense merge.
Stdlib only. Plan and decision rules (written before running): evaluation/ANALYSIS_PLAN_collection_count.md

    python3 evaluation/analyze_collection_count.py [--reports-dir evaluation/reports]

Reads decision_traces_exp5_<doc>_<arm>_n<N>.json (docs: ccby, cdc, ragbench; arms: rrf, dense; N: 3, 6, 10).
Per question: chunks returned, target share (share of chunks KEPT per retrieval call that come from the question's own
document's collection, averaged over the question's retrieval calls), stuck (round 5 still RETRIEVE), round-1 stop, rounds.
Questions are paired across N (and across arms) by their position in the document's question file.
Writes reports/collection_count_summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

DOCS = ["ccby", "cdc", "ragbench"]
ARMS = ["rrf", "dense"]
NS = [3, 6, 10]
SHORT = {"ccby": "ccby_4_0_legalcode", "cdc": "cdc_program_evaluation_framework_2024", "ragbench": "ragbench_2407_11005v2"}


def norm(name):
    n = (name or "").lower()
    return n[len("ragforge_"):] if n.startswith("ragforge_") else n


def classify(rounds):  # same definition of "stuck" as Finding 3 / experiment 4
    last, n = rounds[-1]["action"], len(rounds)
    if last == "GENERATE":
        return "r1" if n == 1 else ("r2_4" if n < 5 else "g5")
    return "stuck"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def boot_ci(vals, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def fmt(x, nd=3):
    return "  n/a" if x is None else f"{x:.{nd}f}"


def load(reports, doc, arm, n):
    f = reports / f"decision_traces_exp5_{doc}_{arm}_n{n}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def per_question(data, doc):
    """list of dicts, one per question, in file order"""
    target = SHORT[doc]
    out = []
    for rec in data["records"]:
        tr = rec["trace"]
        rs = tr["rounds"]
        kept_t = kept_all = 0
        shares, ncols = [], 0
        for call in tr.get("retrieval_log", []):
            cols = call.get("collections", [])
            ncols = max(ncols, len(cols))
            kt = sum(c.get("kept", 0) for c in cols if norm(c.get("collection")) == target)
            ka = sum(c.get("kept", 0) for c in cols)
            kept_t += kt
            kept_all += ka
            if ka:
                shares.append(kt / ka)
        out.append({"q": rec["question"], "chunks": len(rec.get("retrieved_contexts", [])),
                    "share": mean(shares),                       # mean over the question's retrieval calls
                    "kind": classify(rs), "rounds": len(rs), "ncols": ncols,
                    "target_kept": kept_t, "all_kept": kept_all})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default=str(Path(__file__).resolve().parent / "reports"))
    args = ap.parse_args()
    reports = Path(args.reports_dir)

    # data[arm][n][doc] = list of per-question dicts
    data = {a: {n: {} for n in NS} for a in ARMS}
    for a in ARMS:
        for n in NS:
            for d in DOCS:
                t = load(reports, d, a, n)
                if t:
                    data[a][n][d] = per_question(t, d)

    print("== 1. Completeness (questions per arm x N; every cell should hold the same questions of the 3 documents) ==")
    cells = {}
    for a in ARMS:
        for n in NS:
            qs = [q for d in DOCS for q in data[a][n].get(d, [])]
            cells[(a, n)] = qs
            docs_done = [d for d in DOCS if d in data[a][n]]
            print(f"   {a:6s} N={n:<3d} questions: {len(qs):3d}   documents: {','.join(docs_done) or '-'}")
    complete = len({len(v) for v in cells.values()}) == 1 and all(len(data[a][n]) == 3 for a in ARMS for n in NS)
    if not complete:
        print("   NOTE: cells differ in size or documents are missing; contrasts use only questions present in BOTH cells compared.")

    bad = [(a, n, d) for a in ARMS for n in NS for d in data[a][n]
           if any(q["ncols"] and q["ncols"] != n for q in data[a][n][d])]
    print("\n== 2. Sanity: every retrieval call searched exactly N collections ==")
    print("   OK" if not bad else f"   VIOLATION in {bad}: the restriction to N collections did not work as intended")

    print("\n== 3. Descriptives per arm and N (pooled over the questions of the 3 documents) ==")
    print(f"{'arm':7s}{'N':>4s}{'n':>5s}{'chunks/q':>10s}{'target share':>14s}{'1/N':>7s}{'stuck':>12s}{'round-1':>12s}{'rounds':>8s}")
    summary = {"cells": {}, "contrasts": {}}
    for a in ARMS:
        for n in NS:
            qs = cells[(a, n)]
            if not qs:
                continue
            m = len(qs)
            stuck = sum(q["kind"] == "stuck" for q in qs)
            r1 = sum(q["kind"] == "r1" for q in qs)
            sh = mean([q["share"] for q in qs])
            print(f"{a:7s}{n:>4d}{m:>5d}{mean([q['chunks'] for q in qs]):>10.1f}{fmt(sh):>14s}{1 / n:>7.3f}"
                  f"{f'{stuck} ({stuck / m:.0%})':>12s}{f'{r1} ({r1 / m:.0%})':>12s}{mean([q['rounds'] for q in qs]):>8.2f}")
            summary["cells"][f"{a}_n{n}"] = {"n": m, "chunks": mean([q["chunks"] for q in qs]), "target_share": sh,
                                             "stuck": stuck, "round1": r1, "mean_rounds": mean([q["rounds"] for q in qs])}

    print("\n   per document (chunks/q | target share | stuck):")
    for a in ARMS:
        for d in DOCS:
            row = []
            for n in NS:
                qs = data[a][n].get(d, [])
                if qs:
                    row.append(f"N={n}: {mean([q['chunks'] for q in qs]):6.1f} | {fmt(mean([q['share'] for q in qs]), 2)} | "
                               f"{sum(q['kind'] == 'stuck' for q in qs):2d}/{len(qs)}")
            print(f"   {a:6s} {d:9s} " + "   ".join(row))

    def paired(a1, n1, a2, n2, key):
        """per-question differences cell1 - cell2 (paired by document and question position)"""
        diffs = []
        for d in DOCS:
            x, y = data[a1][n1].get(d), data[a2][n2].get(d)
            if not x or not y or len(x) != len(y):
                continue
            for p, q in zip(x, y):
                if p["q"] != q["q"]:
                    continue
                if p[key] is None or q[key] is None:
                    continue
                diffs.append(p[key] - q[key])
        return diffs

    def report(name, diffs, nd=3):
        if len(diffs) < 5:
            print(f"   {name}: not enough paired questions ({len(diffs)})")
            return None
        mu = sum(diffs) / len(diffs)
        lo, hi = boot_ci(diffs)
        print(f"   {name}: {mu:+.{nd}f}  [{lo:+.{nd}f}, {hi:+.{nd}f}]  (n={len(diffs)})")
        summary["contrasts"][name] = {"mean": mu, "ci": [lo, hi], "n": len(diffs)}
        return mu, lo, hi

    print("\n== 4. Pre-registered predictions (95% paired-bootstrap CIs, 2000 resamples, seed 0) ==")
    print("\nP1 (mechanical check): under RRF merge the target share is about 1/N (|mean - 1/N| <= 0.05).")
    for n in NS:
        qs = cells[("rrf", n)]
        if qs:
            sh = mean([q["share"] for q in qs])
            ok = abs(sh - 1 / n) <= 0.05
            print(f"   N={n}: share {sh:.3f} vs 1/N {1 / n:.3f} -> {'holds' if ok else 'DOES NOT HOLD'}")

    print("\nP2: under RRF merge, chunks per question grow with N (N=10 minus N=3 > 0, CI excludes 0).")
    for a, b in [(6, 3), (10, 6), (10, 3)]:
        r = report(f"rrf chunks/q  N={a} - N={b}", paired("rrf", a, "rrf", b, "chunks"), 1)
        if r and (a, b) == (10, 3):
            print(f"      -> P2 {'replicates' if r[1] > 0 else 'DOES NOT REPLICATE'}")

    print("\nP3: dense merge keeps a higher target share than RRF at every N (CI above 0), and its chunk growth N=3 -> 10 is smaller (CI below 0).")
    ok3 = []
    for n in NS:
        r = report(f"target share  dense - rrf  at N={n}", paired("dense", n, "rrf", n, "share"))
        ok3.append(bool(r and r[1] > 0))
    print(f"      -> share part {'holds at every N' if all(ok3) and len(ok3) == 3 else 'does NOT hold at every N'} (CI above 0)")
    # difference in differences: (dense N10 - dense N3) - (rrf N10 - rrf N3), per question
    dd = []
    for d in DOCS:
        a10, a3 = data["dense"][10].get(d), data["dense"][3].get(d)
        b10, b3 = data["rrf"][10].get(d), data["rrf"][3].get(d)
        if all([a10, a3, b10, b3]) and len({len(a10), len(a3), len(b10), len(b3)}) == 1:
            for i in range(len(a10)):
                dd.append((a10[i]["chunks"] - a3[i]["chunks"]) - (b10[i]["chunks"] - b3[i]["chunks"]))
    r = report("chunk growth (N=10 - N=3): dense minus rrf", dd, 1)
    if r:
        print(f"      -> growth part {'holds (dense grows less)' if r[2] < 0 else 'DOES NOT HOLD (CI does not exclude 0 or is positive)'}")

    print("\nP4 (descriptive only, no decision rule): stuck rate and rounds as N grows - see the table in section 3.")
    for a in ARMS:
        for x, y in [(10, 3)]:
            report(f"{a} stuck (0/1)  N={x} - N={y}",
                   [(1.0 if p == "stuck" else 0.0) - (1.0 if q == "stuck" else 0.0) for p, q in _kind_pairs(data, a, x, y)], 3)
            report(f"{a} rounds      N={x} - N={y}", paired(a, x, a, y, "rounds"), 2)

    out = reports / "collection_count_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {out}")


def _kind_pairs(data, a, n1, n2):
    res = []
    for d in DOCS:
        x, y = data[a][n1].get(d), data[a][n2].get(d)
        if x and y and len(x) == len(y):
            res += [(p["kind"], q["kind"]) for p, q in zip(x, y) if p["q"] == q["q"]]
    return res


if __name__ == "__main__":
    main()
