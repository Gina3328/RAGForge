"""Can a retrieval score tell an answerable question from an unanswerable one?

Direct test of the paper's central claim: an RRF score only guarantees a
relative ranking, so it should carry (almost) no information about whether a
question can be answered at all, while a raw dense cosine score should carry
some. Read-only: no indexing, no writes to Milvus, no LLM calls -- only query
embeddings, vector searches and keyword re-scoring. Needs Milvus and the
embedding model running, and the 3 eval documents already indexed.

The analysis is fixed in advance in evaluation/ANALYSIS_PLAN_unanswerable.md.
Please read that file first; do not change the metrics after seeing results.

What it does
------------
For every question of one document (the answerable ones from
eval_questions_<doc>.json, the unanswerable ones from
eval_unanswerable_<doc>.json) it retrieves from THAT document's collection
and records, per question:

    rrf_top1_raw    top-1 RRF score of HybridRetriever with the score
                    threshold switched off (so no question is filtered to
                    "empty" -- we measure the information in the score)
    dense_top1_raw  top-1 raw dense cosine score, threshold 0
    *_top3_raw      mean of the top-3 scores (secondary)
    *_sys           the same top-1 values with the system's real threshold
                    (empty result -> 0.0), for transparency
    n_words         question length (HybridRetriever's dynamic weighting
                    depends on it, so RRF could leak length, not relevance)

Then it computes AUROC (answerable = positive class) with stratified
bootstrap confidence intervals, overall / near-topic-only / far-only / per
document, plus the "accept 95% of answerable questions" operating point.

Usage (from the project root):
    uv run python evaluation/score_separation.py
    # re-run only the analysis on saved rows (no Milvus needed):
    python3 evaluation/score_separation.py --analyze-only evaluation/reports/score_separation_rows.json
    # post-hoc length-matched check (see ANALYSIS_PLAN_unanswerable.md, notes at the end):
    uv run python evaluation/score_separation.py --set-b
Writes evaluation/reports/score_separation_rows.json and
evaluation/reports/score_separation_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "evaluation" / "reports"
CONFIG_PATH = PROJECT_ROOT / "config" / "stage4.yml"

DOCS = {
    # doc_key: (answerable questions, unanswerable questions, trace file that records the
    #           collection name, set-B unanswerable questions)
    "ccby": (
        "eval_questions_ccby.json",
        "eval_unanswerable_ccby.json",
        "decision_traces_ccby_agentic_post_flush_fix.json",
        "eval_unanswerable_long_ccby.json",
    ),
    "cdc": (
        "eval_questions_cdc_program_eval.json",
        "eval_unanswerable_cdc_program_eval.json",
        "decision_traces_cdc_agentic_post_flush_fix.json",
        "eval_unanswerable_long_cdc_program_eval.json",
    ),
    "ragbench": (
        "eval_questions_ragbench.json",
        "eval_unanswerable_ragbench.json",
        "decision_traces_ragbench_agentic_post_flush_fix.json",
        "eval_unanswerable_long_ragbench.json",
    ),
}

SCORES = ["rrf_top1_raw", "dense_top1_raw"]  # primary
SECONDARY = ["rrf_top3_raw", "dense_top3_raw", "rrf_top1_sys", "dense_top1_sys"]
BOOT = 2000
SEED = 0


# --------------------------------------------------------------------------
# Collection (needs the ragforge package, Milvus and the embedding model)
# --------------------------------------------------------------------------
def collect_rows(set_b: bool = False) -> list[dict]:
    """set_b=False: the pre-registered run (answerable + unanswerable set A).
    set_b=True: only the set-B near-topic unanswerable questions (post hoc)."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from dotenv import load_dotenv

    from ragforge.config import RagForgeConfig
    from ragforge.embedding.embedding_service import EmbeddingService
    from ragforge.retriever.hybrid_retriever import HybridRetriever
    from ragforge.store.milvus_vector_store import MilvusVectorStore

    load_dotenv(PROJECT_ROOT / ".env")
    config = RagForgeConfig.load(str(CONFIG_PATH))
    embedding_service = EmbeddingService(config.embedding)
    store = MilvusVectorStore(config.vector_store)
    retriever_sys = HybridRetriever(embedding_service, store, config.retriever)
    retriever_raw = HybridRetriever(embedding_service, store, config.retriever)
    retriever_raw.score_threshold = 0.0  # no filtering: measure the score itself
    top_k_raw = max(retriever_raw.top_k * 3, 30)

    prefix = config.vector_store.collection_prefix
    shorts = [
        c[len(prefix):] if prefix and c.startswith(prefix) else c
        for c in store.list_collections()
    ]

    rows: list[dict] = []
    for doc, (q_ans, q_unans, trace_file, q_long) in DOCS.items():
        target = json.loads((REPORTS / trace_file).read_text(encoding="utf-8"))["collection"]
        if target not in shorts:
            print(f"!! {doc}: collection {target!r} not in the store; index it first. Skipping.")
            continue
        items = []
        if set_b:
            for item in json.loads((PROJECT_ROOT / "evaluation" / q_long).read_text(encoding="utf-8")):
                items.append((item["type"], item["question"]))  # unanswerable_near
        else:
            for item in json.loads((PROJECT_ROOT / "evaluation" / q_ans).read_text(encoding="utf-8")):
                items.append(("answerable", item["question"]))
            for item in json.loads((PROJECT_ROOT / "evaluation" / q_unans).read_text(encoding="utf-8")):
                items.append((item["type"], item["question"]))  # unanswerable_near / unanswerable_far

        for qtype, q in items:
            qvec = embedding_service.embed(q)
            raw_dense = store.search_dense(target, qvec, top_k_raw, 0.0)
            dense_scores = sorted((r.score for r in raw_dense), reverse=True)

            res_raw = retriever_raw.retrieve(q, target).results
            rrf_scores = sorted((r.score for r in res_raw), reverse=True)

            res_sys = retriever_sys.retrieve(q, target).results
            rrf_sys = max((r.score for r in res_sys), default=0.0)
            dense_sys = max(
                (r.metadata["dense_score"] for r in res_sys if "dense_score" in r.metadata),
                default=0.0,
            )

            def top3(xs: list[float]) -> float:
                xs = xs[:3] + [0.0] * (3 - len(xs[:3]))
                return sum(xs) / 3

            rows.append(
                {
                    "doc": doc,
                    "qtype": qtype,
                    "question": q,
                    "n_words": len(q.split()),
                    "rrf_top1_raw": rrf_scores[0] if rrf_scores else 0.0,
                    "dense_top1_raw": dense_scores[0] if dense_scores else 0.0,
                    "rrf_top3_raw": top3(rrf_scores),
                    "dense_top3_raw": top3(dense_scores),
                    "rrf_top1_sys": rrf_sys,
                    "dense_top1_sys": dense_sys,
                    "n_cands_sys": len(res_sys),
                }
            )
        print(f"{doc}: {len(items)} questions scored")
    return rows


# --------------------------------------------------------------------------
# Analysis (standard library only)
# --------------------------------------------------------------------------
def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score_pos > score_neg) + 0.5 * P(tie); positives = answerable."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def boot_auroc(pos_rows, neg_rows, key_a, key_b=None, boot=BOOT, seed=SEED):
    """Stratified bootstrap (resample answerable and unanswerable questions
    separately). Returns CI of AUROC(key_a) and, if key_b is given, CI of
    AUROC(key_a) - AUROC(key_b) computed on the SAME resamples (paired)."""
    rng = random.Random(seed)
    a_vals, d_vals = [], []
    for _ in range(boot):
        ps = [rng.choice(pos_rows) for _ in pos_rows]
        ns = [rng.choice(neg_rows) for _ in neg_rows]
        a = auroc([r[key_a] for r in ps], [r[key_a] for r in ns])
        a_vals.append(a)
        if key_b:
            b = auroc([r[key_b] for r in ps], [r[key_b] for r in ns])
            d_vals.append(a - b)
    ci = (pct(a_vals, 0.025), pct(a_vals, 0.975))
    dci = (pct(d_vals, 0.025), pct(d_vals, 0.975)) if key_b else None
    return ci, dci


def accept95(pos: list[float], neg: list[float]) -> dict:
    """Threshold that keeps >= 95% of answerable questions; share of
    unanswerable questions it rejects."""
    s = sorted(pos)
    thr = s[int(math.floor(0.05 * len(s)))]
    kept = sum(1 for x in pos if x >= thr) / len(pos)
    rejected = sum(1 for x in neg if x < thr) / len(neg)
    return {"threshold": thr, "answerable_kept": kept, "unanswerable_rejected": rejected}


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or statistics.pstdev(x) == 0 or statistics.pstdev(y) == 0:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den


def subset(rows, doc=None, kind=None):
    pos = [r for r in rows if r["qtype"] == "answerable" and (doc is None or r["doc"] == doc)]
    if kind == "near":
        neg = [r for r in rows if r["qtype"] == "unanswerable_near" and (doc is None or r["doc"] == doc)]
    elif kind == "far":
        neg = [r for r in rows if r["qtype"] == "unanswerable_far" and (doc is None or r["doc"] == doc)]
    else:
        neg = [r for r in rows if r["qtype"].startswith("unanswerable") and (doc is None or r["doc"] == doc)]
    return pos, neg


def analyze(rows: list[dict], boot: int = BOOT) -> dict:
    out: dict = {"n_rows": len(rows), "settings": {"boot": boot, "seed": SEED}}

    # 1. descriptive: how much does each score move?
    desc = {}
    for key in SCORES:
        desc[key] = {}
        for grp in ("answerable", "unanswerable_near", "unanswerable_far"):
            xs = [r[key] for r in rows if r["qtype"] == grp]
            if xs:
                desc[key][grp] = {
                    "n": len(xs),
                    "min": min(xs),
                    "median": statistics.median(xs),
                    "max": max(xs),
                    "distinct_values_6dp": len({round(x, 6) for x in xs}),
                }
    out["descriptive"] = desc

    # 2. AUROC tables
    def block(pos, neg):
        if not pos or not neg:
            return None
        res = {"n_answerable": len(pos), "n_unanswerable": len(neg)}
        for key in SCORES + SECONDARY:
            a = auroc([r[key] for r in pos], [r[key] for r in neg])
            res[key] = {"auroc": a}
        for key in SCORES:
            ci, _ = boot_auroc(pos, neg, key, None, boot)
            res[key]["ci95"] = ci
        _, dci = boot_auroc(pos, neg, "dense_top1_raw", "rrf_top1_raw", boot)
        res["dense_minus_rrf_top1"] = {
            "diff": res["dense_top1_raw"]["auroc"] - res["rrf_top1_raw"]["auroc"],
            "ci95": dci,
        }
        res["accept95"] = {
            key: accept95([r[key] for r in pos], [r[key] for r in neg]) for key in SCORES
        }
        return res

    out["pooled_all"] = block(*subset(rows))
    out["pooled_near"] = block(*subset(rows, kind="near"))
    out["pooled_far"] = block(*subset(rows, kind="far"))
    out["per_doc"] = {}
    for d in sorted({r["doc"] for r in rows}):
        out["per_doc"][d] = block(*subset(rows, doc=d))

    # 3. length confound for RRF (dynamic weights depend on word count)
    out["length_corr"] = {
        key: pearson([r["n_words"] for r in rows], [r[key] for r in rows]) for key in SCORES
    }
    return out


def analyze_set_b(rows_a: list[dict], rows_b: list[dict], boot: int = BOOT) -> dict:
    """Post-hoc robustness check (NOT part of the pre-registered plan).

    The first run showed that the set-A unanswerable questions are much
    shorter than the answerable ones, and question length alone separates the
    two classes. Set B is a second batch of near-topic unanswerable questions
    written to be about as long as the answerable ones. This compares the
    answerable questions against set B only."""
    pos = [r for r in rows_a if r["qtype"] == "answerable"]
    neg = rows_b
    res = {"n_answerable": len(pos), "n_unanswerable": len(neg)}
    res["words"] = {
        "answerable_mean": statistics.mean(r["n_words"] for r in pos),
        "setB_mean": statistics.mean(r["n_words"] for r in neg),
        "length_only_auroc": auroc([r["n_words"] for r in pos], [r["n_words"] for r in neg]),
    }
    for key in SCORES:
        res[key] = {"auroc": auroc([r[key] for r in pos], [r[key] for r in neg])}
        ci, _ = boot_auroc(pos, neg, key, None, boot)
        res[key]["ci95"] = ci
    _, dci = boot_auroc(pos, neg, "dense_top1_raw", "rrf_top1_raw", boot)
    res["dense_minus_rrf_top1"] = {
        "diff": res["dense_top1_raw"]["auroc"] - res["rrf_top1_raw"]["auroc"],
        "ci95": dci,
    }
    res["accept95"] = {key: accept95([r[key] for r in pos], [r[key] for r in neg]) for key in SCORES}
    res["per_doc"] = {}
    for d in sorted({r["doc"] for r in neg}):
        p = [r for r in pos if r["doc"] == d]
        n = [r for r in neg if r["doc"] == d]
        res["per_doc"][d] = {
            key: auroc([r[key] for r in p], [r[key] for r in n]) for key in SCORES
        }
    return res


def print_set_b(res: dict) -> None:
    print("=" * 96)
    print("POST-HOC check, set B: near-topic unanswerable questions written to match answerable length")
    w = res["words"]
    print(
        f"  question length (words): answerable mean {w['answerable_mean']:.1f}, set B mean {w['setB_mean']:.1f}; "
        f"AUROC of length alone = {w['length_only_auroc']:.2f}  (0.5 = no leak)"
    )
    print(f"  n answerable / set B = {res['n_answerable']} / {res['n_unanswerable']}")
    for key in SCORES:
        r = res[key]
        print(f"  {key:16s} AUROC {r['auroc']:.2f} {fmt_ci(r['ci95'])}")
    df = res["dense_minus_rrf_top1"]
    print(f"  dense - RRF      {df['diff']:+.2f} {fmt_ci(df['ci95'])}")
    for key in SCORES:
        a = res["accept95"][key]
        print(
            f"  {key:16s} keep {a['answerable_kept']:.0%} of answerable -> reject "
            f"{a['unanswerable_rejected']:.0%} of set B (threshold {a['threshold']:.4f})"
        )
    for d, v in res["per_doc"].items():
        print(f"  doc {d:9s} AUROC rrf={v['rrf_top1_raw']:.2f} dense={v['dense_top1_raw']:.2f}")


def _resid_auroc(pos: list[dict], neg: list[dict], key: str) -> float:
    """AUROC after removing the linear effect of question length (words)
    from the score. The line is fitted on all questions of the comparison."""
    rows = pos + neg
    x = [r["n_words"] for r in rows]
    y = [r[key] for r in rows]
    mx, my = statistics.mean(x), statistics.mean(y)
    denom = sum((a - mx) ** 2 for a in x)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / denom if denom else 0.0
    a0 = my - b * mx
    return auroc(
        [r[key] - (a0 + b * r["n_words"]) for r in pos],
        [r[key] - (a0 + b * r["n_words"]) for r in neg],
    )


def length_adjusted(rows_a: list[dict], rows_b: list[dict], boot: int = 1000) -> dict:
    """Post-hoc, exploratory: all answerable questions against all unanswerable
    questions (set A + set B), with the linear effect of question length removed
    from each score. Set A and set B near-topic questions share topics, so
    treat the intervals as rough."""
    pos = [r for r in rows_a if r["qtype"] == "answerable"]
    negs = {
        "all unanswerable (A+B)": [r for r in rows_a if r["qtype"] != "answerable"] + rows_b,
        "near-topic only (A+B)": [r for r in rows_a if r["qtype"] == "unanswerable_near"] + rows_b,
    }
    out = {}
    for name, neg in negs.items():
        out[name] = {"n_answerable": len(pos), "n_unanswerable": len(neg)}
        for key in SCORES:
            rng = random.Random(SEED)
            vals = []
            for _ in range(boot):
                p = [rng.choice(pos) for _ in pos]
                n = [rng.choice(neg) for _ in neg]
                vals.append(_resid_auroc(p, n, key))
            out[name][key] = {
                "raw_auroc": auroc([r[key] for r in pos], [r[key] for r in neg]),
                "length_adjusted_auroc": _resid_auroc(pos, neg, key),
                "ci95": (pct(vals, 0.025), pct(vals, 0.975)),
            }
    return out


def print_length_adjusted(res: dict) -> None:
    print("=" * 96)
    print("POST-HOC, exploratory: AUROC with the linear effect of question length removed from the score")
    for name, d in res.items():
        print(f"  {name}  (answerable {d['n_answerable']} / unanswerable {d['n_unanswerable']})")
        for key in SCORES:
            v = d[key]
            print(
                f"    {key:16s} raw {v['raw_auroc']:.2f}   length-adjusted {v['length_adjusted_auroc']:.2f} "
                f"{fmt_ci(v['ci95'])}"
            )


def fmt_ci(ci):
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "-"


def print_report(out: dict) -> None:
    print("=" * 96)
    print("Score distributions (top-1)")
    for key, groups in out["descriptive"].items():
        print(f"  {key}")
        for grp, s in groups.items():
            print(
                f"    {grp:18s} n={s['n']:3d} min={s['min']:.4f} median={s['median']:.4f} "
                f"max={s['max']:.4f} distinct(6dp)={s['distinct_values_6dp']}"
            )
    print("\nAUROC, answerable = positive class (0.5 = no information, 1.0 = perfect)")
    print(f"{'subset':14s} {'n(a/u)':8s} {'RRF top-1':>22s} {'dense top-1':>22s} {'dense - RRF':>24s}")
    sets = [("pooled all", out["pooled_all"]), ("pooled near", out["pooled_near"]), ("pooled far", out["pooled_far"])]
    sets += [(f"doc: {d}", b) for d, b in out["per_doc"].items()]
    for name, b in sets:
        if not b:
            continue
        r, d, df = b["rrf_top1_raw"], b["dense_top1_raw"], b["dense_minus_rrf_top1"]
        print(
            f"{name:14s} {b['n_answerable']:3d}/{b['n_unanswerable']:<3d}  "
            f"{r['auroc']:.2f} {fmt_ci(r['ci95']):>16s}  {d['auroc']:.2f} {fmt_ci(d['ci95']):>16s}  "
            f"{df['diff']:+.2f} {fmt_ci(df['ci95']):>16s}"
        )
    print("\nSecondary (pooled all): top-3 mean and in-system (thresholded) top-1")
    b = out["pooled_all"]
    for key in SECONDARY:
        print(f"  {key:16s} AUROC {b[key]['auroc']:.2f}")
    print("\nOperating point: keep >= 95% of answerable questions, how many unanswerable are rejected?")
    for key in SCORES:
        a = b["accept95"][key]
        print(
            f"  {key:16s} threshold={a['threshold']:.4f}  answerable kept={a['answerable_kept']:.0%}  "
            f"unanswerable rejected={a['unanswerable_rejected']:.0%}"
        )
    print("\nCorrelation of top-1 score with question length (words):")
    for key, v in out["length_corr"].items():
        print(f"  {key:16s} r={v:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", help="path to a saved score_separation_rows.json")
    ap.add_argument(
        "--set-b",
        action="store_true",
        help="post-hoc: score only the set-B near-topic unanswerable questions "
        "(eval_unanswerable_long_*.json) and compare them with the answerable questions "
        "already saved in score_separation_rows.json",
    )
    ap.add_argument(
        "--length-adjusted",
        action="store_true",
        help="post-hoc: AUROC with the effect of question length removed; needs the saved rows "
        "of both the main run and --set-b (no Milvus needed)",
    )
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args()

    if args.length_adjusted:
        rows_a = json.loads((REPORTS / "score_separation_rows.json").read_text(encoding="utf-8"))
        rows_b = json.loads((REPORTS / "score_separation_rows_setB.json").read_text(encoding="utf-8"))
        res = length_adjusted(rows_a, rows_b, min(args.boot, 1000))
        print_length_adjusted(res)
        (REPORTS / "score_separation_length_adjusted.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return

    if args.set_b:
        rows_a_path = REPORTS / "score_separation_rows.json"
        if not rows_a_path.exists():
            print("Run the main analysis first (score_separation_rows.json is missing).")
            return
        rows_a = json.loads(rows_a_path.read_text(encoding="utf-8"))
        b_path = REPORTS / "score_separation_rows_setB.json"
        if b_path.exists() and args.analyze_only:
            rows_b = json.loads(b_path.read_text(encoding="utf-8"))
        else:
            rows_b = collect_rows(set_b=True)
            if not rows_b:
                print("Nothing scored -- is Milvus up and are the documents indexed?")
                return
            b_path.write_text(json.dumps(rows_b, indent=2, ensure_ascii=False), encoding="utf-8")
        res = analyze_set_b(rows_a, rows_b, args.boot)
        print_set_b(res)
        (REPORTS / "score_separation_summary_setB.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return

    if args.analyze_only:
        rows = json.loads(Path(args.analyze_only).read_text(encoding="utf-8"))
    else:
        rows = collect_rows()
        if not rows:
            print("Nothing scored -- is Milvus up and are the documents indexed?")
            return
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "score_separation_rows.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    out = analyze(rows, args.boot)
    print_report(out)
    if not args.analyze_only:
        (REPORTS / "score_separation_summary.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote {REPORTS / 'score_separation_rows.json'} and score_separation_summary.json")


if __name__ == "__main__":
    main()
