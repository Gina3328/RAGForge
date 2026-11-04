"""Summarize the Agentic RAG cross-collection merge comparison (RRF vs dense).

Reads, for each of the 3 documents and each arm, the files written by
run_agentic_merge_comparison.sh:
    decision_traces_<doc>_<suffix>.json, predictions_<doc>_<suffix>.json,
    ragas_report_<doc>_<suffix>.json
plus the Stage 2 (generation_v2) predictions as the reference "what does a
tightly-scoped retriever surface" pool. Plain JSON in, no Milvus/LLM needed.

Per arm and document it prints:
  * avg retrieved chunks per question, and the ratio to Stage 2
  * match rate: share of the arm's retrieved chunks that Stage 2 also surfaced
    for that document (same proxy as in the Finding 2 write-up)
  * foreign share: share of the chunks each retrieval round pulled in that came
    from collections other than the question's own (needs trace["retrieval_log"],
    so only available for runs made after the merge switch was added)
  * avg rounds / % of questions hitting the 5-round cap / % still choosing RETRIEVE
    on the last round (the Finding 3 numbers, since the merge change may move them)
  * RAGAS averages (faithfulness, answer_relevancy, context_precision, context_recall)

Usage:
    uv run python evaluation/analyze_agentic_merge.py
    # sanity-check against the old data (both arms = same old files):
    uv run python evaluation/analyze_agentic_merge.py --rrf-suffix agentic_post_flush_fix --dense-suffix agentic_post_flush_fix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPORTS = Path(__file__).resolve().parent / "reports"
DOCS = ["ccby", "cdc", "ragbench"]
RAGAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def analyze_arm(doc: str, suffix: str, stage2_pool: set[str], stage2_avg: float) -> dict | None:
    traces = load(REPORTS / f"decision_traces_{doc}_{suffix}.json")
    preds = load(REPORTS / f"predictions_{doc}_{suffix}.json")
    if traces is None or preds is None:
        return None
    ragas = load(REPORTS / f"ragas_report_{doc}_{suffix}.json")

    ctxs = [[c.strip() for c in r["retrieved_contexts"]] for r in preds]
    n_chunks = [len(c) for c in ctxs]
    flat = [c for cs in ctxs for c in cs]
    avg_chunks = sum(n_chunks) / len(n_chunks)
    matches = sum(1 for c in flat if c in stage2_pool)

    recs = traces["records"]
    rounds = [r["trace"]["num_rounds"] for r in recs]
    capped = [r for r in recs if r["trace"]["num_rounds"] == 5]
    stuck = [r for r in capped if r["trace"]["rounds"][-1]["action"] == "RETRIEVE"]

    # foreign share from the per-call retrieval log (newer runs only)
    target = traces.get("collection")
    kept_total = kept_foreign = 0
    has_log = False
    for r in recs:
        for call in r["trace"].get("retrieval_log", []):
            has_log = True
            for col in call["collections"]:
                kept_total += col["kept"]
                if not col["collection"].endswith(target):
                    kept_foreign += col["kept"]

    return {
        "n_questions": len(recs),
        "avg_chunks": avg_chunks,
        "ratio_vs_stage2": avg_chunks / stage2_avg,
        "match_n": matches,
        "match_total": len(flat),
        "foreign_share": (kept_foreign / kept_total) if has_log and kept_total else None,
        "avg_rounds": sum(rounds) / len(rounds),
        "capped_n": len(capped),
        "stuck_n": len(stuck),
        "ragas": (ragas or {}).get("averaged_scores", {}),
        "settings": traces.get("settings"),
        "store_collections": traces.get("vector_store_collections"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrf-suffix", default="agentic_merge_rrf")
    ap.add_argument("--dense-suffix", default="agentic_merge_dense")
    ap.add_argument("--stage2-suffix", default="generation_v2_post_flush_fix")
    args = ap.parse_args()

    results: dict[str, dict[str, dict]] = {"rrf": {}, "dense": {}}
    for doc in DOCS:
        s2 = load(REPORTS / f"predictions_{doc}_{args.stage2_suffix}.json")
        if s2 is None:
            print(f"(skip {doc}: no Stage 2 predictions)")
            continue
        s2_pool = {c.strip() for r in s2 for c in r["retrieved_contexts"]}
        s2_avg = sum(len(r["retrieved_contexts"]) for r in s2) / len(s2)
        for arm, suffix in (("rrf", args.rrf_suffix), ("dense", args.dense_suffix)):
            res = analyze_arm(doc, suffix, s2_pool, s2_avg)
            if res is None:
                print(f"(skip {doc}/{arm}: files for suffix {suffix!r} not found yet)")
                continue
            results[arm][doc] = res

    def f(v, spec):
        return "-" if v is None else format(v, spec)

    print(f"\n{'doc':9} {'arm':6} {'chunks/q':>9} {'x Stage2':>9} {'match':>14} {'foreign':>8} {'rounds':>7} {'cap5':>6} {'stuck':>6}  " + "  ".join(m[:12] for m in RAGAS))
    for doc in DOCS:
        for arm in ("rrf", "dense"):
            r = results[arm].get(doc)
            if not r:
                continue
            match = f"{r['match_n']}/{r['match_total']} ({100 * r['match_n'] / r['match_total']:.1f}%)"
            foreign = "-" if r["foreign_share"] is None else f"{100 * r['foreign_share']:.0f}%"
            ragas = "  ".join(f(r["ragas"].get(m), ".3f").rjust(12) for m in RAGAS)
            print(
                f"{doc:9} {arm:6} {r['avg_chunks']:9.2f} {r['ratio_vs_stage2']:8.1f}x {match:>14} {foreign:>8} "
                f"{r['avg_rounds']:7.2f} {r['capped_n']:>3}/{r['n_questions']:<2} {r['stuck_n']:>3}/{r['n_questions']:<2}  {ragas}"
            )

    # cross-document averages (mean of per-document values, as in the Stage 1-4 comparison)
    print("\ncross-document average (mean of the three per-document values):")
    for arm in ("rrf", "dense"):
        rs = list(results[arm].values())
        if len(rs) != len(DOCS):
            continue
        avg = lambda key: sum(r[key] for r in rs) / len(rs)  # noqa: E731
        rag = {m: sum(r["ragas"].get(m, float("nan")) for r in rs) / len(rs) for m in RAGAS}
        print(
            f"  {arm:6} chunks/q={avg('avg_chunks'):.1f}  x Stage2={avg('ratio_vs_stage2'):.1f}  rounds={avg('avg_rounds'):.2f}  "
            + "  ".join(f"{m}={v:.3f}" for m, v in rag.items())
        )

    for arm in ("rrf", "dense"):
        any_doc = next(iter(results[arm].values()), None)
        if any_doc and any_doc.get("store_collections") is not None:
            print(f"\nVector store at run time ({arm}): {any_doc['store_collections']}")
            print(f"Settings ({arm}): {any_doc['settings']}")


if __name__ == "__main__":
    main()
