"""Experiment 4, analysis: do the paper's findings hold on 7 new documents (174 answerable questions)?

Reads JSON only (no Milvus, no LLM). For every document in evaluation/new_docs.tsv and every arm it finds:
  evaluation/reports/ragas_report_exp4_<doc>_<arm>.json
  evaluation/reports/decision_traces_exp4_<doc>_<arm>.json      (self_rag, crag, adaptive, agentic_rrf, agentic_dense)

Arms: baseline (= Stage 1), generation_v2 (= Stage 2), self_rag, crag, adaptive, agentic_rrf, agentic_dense.

Prints
  0. which (document, arm) cells are complete (a partial run is analysed on the documents that are complete for
     BOTH arms of a contrast, and is marked PARTIAL),
  1. per-arm means (pooled over questions; mean of the per-document means),
  2. the 8 pre-specified contrasts E1-E8 (paired bootstrap over questions, 2000 resamples, seed 0, 95% CI),
     with the decision rule applied and the per-document sign counts,
  3. per-document tables for the four RAGAS metrics,
  4. descriptives: Agentic behaviour (stuck, round-1 stops, rounds, chunks, target share) and CRAG declines.
Writes evaluation/reports/new_docs_summary.json.

    python3 evaluation/analyze_new_docs.py [--reports-dir DIR] [--docs-tsv FILE]
Plan and decision rules (fixed before the run): evaluation/ANALYSIS_PLAN_new_docs.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).parent
ARMS = ["baseline", "generation_v2", "self_rag", "crag", "adaptive", "agentic_rrf", "agentic_dense"]
NAME = {"baseline": "Stage 1", "generation_v2": "Stage 2", "self_rag": "Self-RAG", "crag": "CRAG",
        "adaptive": "Adaptive", "agentic_rrf": "Agentic (RRF)", "agentic_dense": "Agentic (dense)"}
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
SHORT = {"faithfulness": "faith", "answer_relevancy": "ans_rel", "context_precision": "cp", "context_recall": "cr"}
DECLINE_PREFIX = "Sorry, I couldn't find enough relevant information"

# id, a, b, metric, prediction (+1 positive, -1 negative, 0 none), prediction text
CONTRASTS = [
    ("E1", "agentic_rrf", "generation_v2", "context_precision", -1, "negative"),
    ("E2", "agentic_rrf", "generation_v2", "context_recall", +1, "positive"),
    ("E3", "agentic_dense", "agentic_rrf", "context_precision", +1, "positive"),
    ("E4", "agentic_dense", "agentic_rrf", "context_recall", 0, "none"),
    ("E5", "crag", "generation_v2", "context_recall", +1, "positive"),
    ("E6", "crag", "generation_v2", "faithfulness", 0, "none"),
    ("E7", "generation_v2", "baseline", "context_precision", +1, "positive"),
    ("E8", "agentic_rrf", "crag", "context_precision", -1, "negative"),
]
SIGN_COUNT_FOR = {"E1", "E2", "E3"}


def norm(name: str) -> str:
    n = (name or "").lower()
    return n[len("ragforge_"):] if n.startswith("ragforge_") else n


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def classify(rounds: list[dict]) -> str:  # same definition of "stuck" as Finding 3 (analyze_agentic_window.py)
    last, n = rounds[-1]["action"], len(rounds)
    if last == "GENERATE":
        return "r1" if n == 1 else ("r2_4" if n < 5 else "g5")
    return "stuck"


def read_docs(tsv: Path) -> list[str]:
    keys = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        keys.append(line.split("\t")[0].strip())
    return keys


def keyed(questions: list[str]) -> list[tuple[str, int]]:
    """Question text plus its occurrence number, so duplicated question texts still pair up correctly."""
    seen: dict[str, int] = {}
    out = []
    for q in questions:
        seen[q] = seen.get(q, 0) + 1
        out.append((q, seen[q]))
    return out


def load_ragas(reports: Path, doc: str, arm: str):
    f = reports / f"ragas_report_exp4_{doc}_{arm}.json"
    if not f.exists():
        return None
    samples = json.loads(f.read_text(encoding="utf-8"))["per_sample"]
    keys = keyed([s["user_input"] for s in samples])
    return {k: {m: num(s.get(m)) for m in METRICS} for k, s in zip(keys, samples)}


def load_traces(reports: Path, doc: str, arm: str):
    f = reports / f"decision_traces_exp4_{doc}_{arm}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", default=str(HERE / "reports"))
    ap.add_argument("--docs-tsv", default=str(HERE / "new_docs.tsv"))
    args = ap.parse_args()
    reports = Path(args.reports_dir)
    docs = read_docs(Path(args.docs_tsv))

    ragas: dict[tuple[str, str], dict] = {}
    for d in docs:
        for a in ARMS:
            r = load_ragas(reports, d, a)
            if r is not None:
                ragas[(d, a)] = r

    summary: dict = {"docs": docs, "complete": {}, "arms": {}, "contrasts": {}, "agentic": {}, "crag": {}}

    # ---- 0. completeness
    print("== 0. Complete (document, arm) cells (RAGAS report present; n = questions) ==")
    print(f"{'document':14s}" + "".join(f"{NAME[a]:>17s}" for a in ARMS))
    for d in docs:
        row = f"{d:14s}"
        for a in ARMS:
            row += f"{(len(ragas[(d, a)]) if (d, a) in ragas else '-'):>17}"
            summary["complete"][f"{d}|{a}"] = len(ragas[(d, a)]) if (d, a) in ragas else 0
        print(row)

    # missing RAGAS values (timeouts) matter: a metric computed only on the questions that did not time out is biased
    missing = []
    for (d, a), r in ragas.items():
        for m in METRICS:
            k = sum(1 for v in r.values() if v[m] is None)
            if k:
                missing.append(f"{d}/{NAME[a]}/{SHORT[m]}: {k} of {len(r)}")
    print("\nMissing (NaN) RAGAS values:" + (" none" if not missing else ""))
    for line in missing:
        print("   " + line)
    summary["missing"] = missing

    # ---- 1. per-arm means
    print("\n== 1. Per-arm means (pooled over questions | mean of per-document means), documents available for the arm ==")
    print(f"{'arm':18s}{'docs':>5s}{'n':>5s}" + "".join(f"{SHORT[m] + ' pooled':>17s}{SHORT[m] + ' macro':>16s}" for m in METRICS))
    for a in ARMS:
        have = [d for d in docs if (d, a) in ragas]
        if not have:
            continue
        n = sum(len(ragas[(d, a)]) for d in have)
        row = f"{NAME[a]:18s}{len(have):>5d}{n:>5d}"
        summary["arms"][a] = {"docs": have, "n": n}
        for m in METRICS:
            pooled = mean([v[m] for d in have for v in ragas[(d, a)].values()])
            macro = mean([mean([v[m] for v in ragas[(d, a)].values()]) for d in have])
            row += f"{fmt(pooled):>17s}{fmt(macro):>16s}"
            summary["arms"][a][m] = {"pooled": pooled, "macro": macro}
        print(row)

    # ---- 2. contrasts
    print("\n== 2. Pre-specified contrasts (a - b; paired bootstrap over questions, 2000 resamples, seed 0, 95% CI) ==")
    for cid, a, b, m, pred, predtxt in CONTRASTS:
        have = [d for d in docs if (d, a) in ragas and (d, b) in ragas]
        title = f"{cid}  {NAME[a]} - {NAME[b]}  [{SHORT[m]}]   prediction: {predtxt}"
        if not have:
            print(f"\n{title}\n   no document is complete for both arms yet")
            continue
        diffs, per_doc = [], {}
        for d in have:
            dd = []
            for k, va in ragas[(d, a)].items():
                vb = ragas[(d, b)].get(k)
                if vb is None or va[m] is None or vb[m] is None:
                    continue
                dd.append(va[m] - vb[m])
            per_doc[d] = dd
            diffs += dd
        if not diffs:
            print(f"\n{title}\n   no paired values")
            continue
        est = sum(diffs) / len(diffs)
        lo, hi = boot_ci(diffs)
        excl = lo > 0 or hi < 0
        sign = 0 if not excl else (1 if est > 0 else -1)
        if pred == 0:
            verdict = "reported as it comes (no prediction)" + ("; CI excludes 0" if excl else "; CI includes 0")
        elif not excl:
            verdict = "DOES NOT REPLICATE (CI includes 0: no detectable difference on the new documents)"
        elif sign == pred:
            verdict = "REPLICATES (CI excludes 0, predicted sign)"
        else:
            verdict = "CONTRADICTS (CI excludes 0, opposite sign): weaken the claim in the paper"
        partial = "" if len(have) == len(docs) else f"   PARTIAL: {len(have)} of {len(docs)} documents"
        print(f"\n{title}{partial}")
        print(f"   diff {est:+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={len(diffs)}   -> {verdict}")
        rec = {"a": a, "b": b, "metric": m, "prediction": predtxt, "diff": est, "lo": lo, "hi": hi,
               "n": len(diffs), "docs": have, "verdict": verdict, "per_doc": {}}
        parts = []
        opposite = 0
        for d in have:
            dm = mean(per_doc[d])
            parts.append(f"{d} {dm:+.3f}")
            rec["per_doc"][d] = dm
            if pred != 0 and dm is not None and dm != 0 and (dm > 0) != (pred > 0):
                opposite += 1
        print("   per document: " + ", ".join(parts))
        if cid in SIGN_COUNT_FOR:
            matching = len(have) - opposite
            msg = f"   predicted sign in {matching} of {len(have)} documents"
            if verdict.startswith("REPLICATES") and opposite >= 3:
                msg += "   -> HETEROGENEOUS (opposite sign in 3 or more documents)"
            print(msg)
            rec["predicted_sign_docs"] = [matching, len(have)]
            rec["heterogeneous"] = bool(verdict.startswith("REPLICATES") and opposite >= 3)
        summary["contrasts"][cid] = rec
    print("\n(No multiple-comparison correction: 8 pre-specified contrasts. One run per cell; run-to-run SDs in "
          "experiment 2 were about 0.01-0.04, so contrasts that small should not be read.)")

    # ---- 3. per-document tables
    print("\n== 3. Per-document means ==")
    for m in METRICS:
        print(f"\n{m}")
        print(f"{'arm':18s}" + "".join(f"{d[:15]:>16s}" for d in docs))
        for a in ARMS:
            if not any((d, a) in ragas for d in docs):
                continue
            row = f"{NAME[a]:18s}"
            for d in docs:
                row += f"{fmt(mean([v[m] for v in ragas[(d, a)].values()]) if (d, a) in ragas else None):>16s}"
            print(row)

    # ---- 4. descriptives
    print("\n== 4. Agentic behaviour (stuck = round 5 still RETRIEVE; target share = share of the chunks KEPT per retrieval "
          "call that come from the question's own collection) ==")
    print("   (Under RRF merge every collection keeps the same quota, so the target share is about 1/N_collections by "
          "construction, 0.10 here; under dense merge it varies. The final chunks carry no collection label, hence 'kept'.)")
    for a in ("agentic_rrf", "agentic_dense"):
        rows = []
        for d in docs:
            t = load_traces(reports, d, a)
            if not t:
                continue
            target = norm(t.get("collection", ""))
            n = stuck = r1 = rounds = chunks = 0
            shares = []
            for rec in t["records"]:
                rs = rec["trace"]["rounds"]
                kind = classify(rs)
                n += 1
                stuck += kind == "stuck"
                r1 += kind == "r1"
                rounds += len(rs)
                chunks += len(rec.get("retrieved_contexts", []))
                kept_t = kept_all = 0
                for call in rec["trace"].get("retrieval_log", []):
                    for c in call.get("collections", []):
                        kept_all += c.get("kept", 0)
                        if norm(c.get("collection", "")) == target:
                            kept_t += c.get("kept", 0)
                if kept_all:
                    shares.append(kept_t / kept_all)
            rows.append((d, n, stuck, r1, rounds / n, chunks / n, mean(shares)))
        if not rows:
            continue
        print(f"\n{NAME[a]}")
        print(f"{'document':16s}{'n':>5s}{'stuck':>14s}{'round-1 stop':>16s}{'mean rounds':>13s}{'chunks/q':>10s}{'target share':>14s}")
        tn = ts = tr = trounds = tchunks = 0
        allshares = []
        for d, n, s, r, mr, mc, sh in rows:
            print(f"{d:16s}{n:>5d}{f'{s} ({s / n:.0%})':>14s}{f'{r} ({r / n:.0%})':>16s}{mr:>13.2f}{mc:>10.1f}{fmt(sh, 2):>14s}")
            tn += n; ts += s; tr += r; trounds += mr * n; tchunks += mc * n
            if sh is not None:
                allshares.append((sh, n))
        pooled_share = sum(s * n for s, n in allshares) / sum(n for _, n in allshares) if allshares else None
        print(f"{'pooled':16s}{tn:>5d}{f'{ts} ({ts / tn:.0%})':>14s}{f'{tr} ({tr / tn:.0%})':>16s}{trounds / tn:>13.2f}"
              f"{tchunks / tn:>10.1f}{fmt(pooled_share, 2):>14s}")
        summary["agentic"][a] = {"n": tn, "stuck": ts, "round1": tr, "mean_rounds": trounds / tn,
                                 "chunks_per_question": tchunks / tn, "target_share": pooled_share,
                                 "per_doc": {d: {"n": n, "stuck": s, "round1": r, "mean_rounds": mr,
                                                 "chunks": mc, "target_share": sh} for d, n, s, r, mr, mc, sh in rows}}

    print("\nCRAG on answerable questions (a decline = the canned 'couldn't find enough relevant information' message)")
    tn = td = 0
    for d in docs:
        t = load_traces(reports, d, "crag")
        if not t:
            continue
        n = len(t["records"])
        dec = sum(1 for r in t["records"] if str(r.get("answer", "")).startswith(DECLINE_PREFIX))
        routes: dict[str, int] = {}
        for r in t["records"]:
            k = r["trace"].get("route", "?")
            routes[k] = routes.get(k, 0) + 1
        print(f"   {d:16s} {dec:>2d} / {n:<3d} declined   routes: {routes}")
        summary["crag"][d] = {"n": n, "declined": dec, "routes": routes}
        tn += n; td += dec
    if tn:
        print(f"   {'pooled':16s} {td:>2d} / {tn:<3d} declined")

    out = reports / "new_docs_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten {out}")


if __name__ == "__main__":
    main()
