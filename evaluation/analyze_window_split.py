"""Experiment 6, analysis: does the number of chunks or the number of characters per chunk decide whether
the Agentic loop stops?

Reads (JSON only, no Milvus / LLM), for arms w10x200, w10x500, w20x200, w20x500 and repetitions 1..N:
  evaluation/reports/decision_traces_<doc>_agentic_<arm>_rep<i>.json
  evaluation/reports/ragas_report_<doc>_agentic_<arm>_rep<i>.json
Only repetitions that are complete for an arm (all three documents, traces and RAGAS) are used.

Prints, per arm: stuck questions, round-1 stops, mean rounds, and the four RAGAS means (mean over repetitions, with
the standard deviation across repetitions). Then the 2 x 2 effects, each as a paired bootstrap over questions
(each question's value is first averaged over the repetitions):
  chunks effect       = mean(w20x*) - mean(w10x*)
  characters effect   = mean(w*x500) - mean(w*x200)
  interaction         = (w20x500 - w20x200) - (w10x500 - w10x200)

    python3 evaluation/analyze_window_split.py
Plan and decision rules: evaluation/ANALYSIS_PLAN_window_split.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

from analyze_agentic_window import classify  # same folder; same definition of "stuck" as Finding 3

DOCS = ["ccby", "cdc", "ragbench"]
ARMS = ["w10x200", "w10x500", "w20x200", "w20x500"]
LABEL = {"w10x200": "10 chunks x 200 chars", "w10x500": "10 chunks x 500 chars",
         "w20x200": "20 chunks x 200 chars", "w20x500": "20 chunks x 500 chars"}
RAGAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
SHORT = {"faithfulness": "faith", "answer_relevancy": "ans_rel", "context_precision": "cp", "context_recall": "cr"}
EFFECT_METRICS = ["rounds", "stuck", "round1", "context_recall", "answer_relevancy", "context_precision", "faithfulness"]


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def load_rep(reports: Path, arm: str, rep: int):
    """-> {(doc, question): {metric: value}} or None if the repetition is incomplete."""
    out = {}
    for doc in DOCS:
        tf = reports / f"decision_traces_{doc}_agentic_{arm}_rep{rep}.json"
        rf = reports / f"ragas_report_{doc}_agentic_{arm}_rep{rep}.json"
        if not (tf.exists() and rf.exists()):
            return None
        traces = json.loads(tf.read_text(encoding="utf-8"))["records"]
        ragas = {s["user_input"]: s for s in json.loads(rf.read_text(encoding="utf-8"))["per_sample"]}
        for rec in traces:
            rounds = rec["trace"]["rounds"]
            kind = classify(rounds)
            row = {"rounds": float(len(rounds)), "stuck": float(kind == "stuck"), "round1": float(kind == "r1")}
            s = ragas.get(rec["question"], {})
            for m in RAGAS:
                row[m] = num(s.get(m))
            out[(doc, rec["question"])] = row
    return out


def boot(vals, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default=str(Path(__file__).parent / "reports"))
    ap.add_argument("--max-reps", type=int, default=10)
    args = ap.parse_args()
    reports = Path(args.reports_dir)

    reps: dict[str, list[dict]] = {a: [] for a in ARMS}
    for a in ARMS:
        for r in range(1, args.max_reps + 1):
            d = load_rep(reports, a, r)
            if d is not None:
                reps[a].append(d)
    n_reps = {a: len(v) for a, v in reps.items()}
    print("Complete repetitions per arm:", n_reps)
    if not all(n_reps.values()):
        raise SystemExit("At least one arm has no complete repetition yet; run evaluation/run_window_split.sh first.")

    print("\n== 1. Per arm (mean over repetitions; +- = SD of the repetition means) ==")
    print(f"{'arm':26s}{'stuck/50':>14s}{'round1/50':>14s}{'mean rounds':>16s}" + "".join(f"{SHORT[m]:>14s}" for m in RAGAS))
    qvals: dict[str, dict] = {}   # arm -> {(doc,q): {metric: mean over reps}}
    summary = {"arms": {}, "effects": {}}
    for a in ARMS:
        per_rep = {m: [] for m in EFFECT_METRICS}
        for d in reps[a]:
            for m in EFFECT_METRICS:
                vals = [v[m] for v in d.values() if v[m] is not None]
                per_rep[m].append(sum(vals) / len(vals) if vals else None)
        qvals[a] = {}
        keys = reps[a][0].keys()
        for k in keys:
            qvals[a][k] = {}
            for m in EFFECT_METRICS:
                vs = [d[k][m] for d in reps[a] if k in d and d[k][m] is not None]
                qvals[a][k][m] = sum(vs) / len(vs) if vs else None

        def fmt(m, scale=1.0, nd=1):
            xs = [x * scale for x in per_rep[m] if x is not None]
            if not xs:
                return "n/a"
            sd = st.stdev(xs) if len(xs) > 1 else 0.0
            return f"{st.mean(xs):.{nd}f}±{sd:.{nd}f}"

        row = f"{a + ' ' + LABEL[a]:26s}{fmt('stuck', 50):>14s}{fmt('round1', 50):>14s}{fmt('rounds', 1, 2):>16s}"
        for m in RAGAS:
            row += f"{fmt(m, 1, 3):>14s}"
        print(row)
        summary["arms"][a] = {m: per_rep[m] for m in EFFECT_METRICS}
    print("(stuck = round 5 still RETRIEVE, cut off by max_iterations; same definition as Finding 3)")

    print("\n== 2. Effects (paired bootstrap over questions, 95% CI) ==")
    common = set(qvals["w10x200"])
    for a in ARMS:
        common &= set(qvals[a])
    common = sorted(common)

    def contrast(coefs):
        out = {}
        for m in EFFECT_METRICS:
            diffs = []
            for k in common:
                vals = [qvals[a][k][m] for a in coefs]
                if any(v is None for v in vals):
                    continue
                diffs.append(sum(c * qvals[a][k][m] for a, c in coefs.items()))
            if diffs:
                lo, hi = boot(diffs)
                out[m] = (sum(diffs) / len(diffs), lo, hi, len(diffs))
        return out

    effects = {
        "chunks (20 vs 10)": {"w20x200": .5, "w20x500": .5, "w10x200": -.5, "w10x500": -.5},
        "characters (500 vs 200)": {"w10x500": .5, "w20x500": .5, "w10x200": -.5, "w20x200": -.5},
        "interaction": {"w20x500": 1, "w20x200": -1, "w10x500": -1, "w10x200": 1},
    }
    for name, coefs in effects.items():
        res = contrast(coefs)
        print(f"\n{name}")
        summary["effects"][name] = {}
        for m in EFFECT_METRICS:
            if m not in res:
                continue
            d, lo, hi, n = res[m]
            scale = 50 if m in ("stuck", "round1") else 1
            unit = " (questions per 50)" if scale == 50 else ""
            sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
            print(f"   {m:18s} {d * scale:+8.3f}  [{lo * scale:+.3f}, {hi * scale:+.3f}]  n={n}  {sig}{unit}")
            summary["effects"][name][m] = {"diff": d, "lo": lo, "hi": hi, "n": n}

    out = reports / "window_split_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten {out}")


if __name__ == "__main__":
    main()
