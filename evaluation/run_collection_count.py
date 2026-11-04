"""Experiment 5, collector: how does Agentic's cross-collection retrieval change with the NUMBER OF COLLECTIONS?

For one Agentic merge arm (rrf or dense) and one store size N (3, 6 or 10), this runs the Agentic strategy on the 50
questions of the three ORIGINAL documents (ccby, cdc, ragbench) while it can only "see" N collections. The store itself
is not changed: `list_collections()` of the vector store is restricted to the first N collections of a FIXED nested
order (fixed before running, see COLLECTION_ORDER):

    N = 3 : ccby, cdc, ragbench                                  (the three target documents only)
    N = 6 : + attention, nist_ai_rmf, ipcc_spm                   (first three new documents in new_docs.tsv order)
    N = 10: + alice, who_handhygiene, roomba, tootsie            (the full store of experiment 4)

Nothing is indexed here; the 10 collections must already be in the store (run evaluation/index_new_docs.py first).
Only decision traces are collected (no RAGAS): the measures are chunk counts, target share and stopping behaviour.

    uv run python evaluation/run_collection_count.py --arm rrf --n 6
Output: evaluation/reports/decision_traces_exp5_<doc>_<arm>_n<N>.json (skipped if it exists).
Plan and decision rules: evaluation/ANALYSIS_PLAN_collection_count.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import collect_decision_traces as cdt  # noqa: E402  (same folder; also puts src/ on sys.path)

PROJECT_ROOT = cdt.PROJECT_ROOT
REPORTS = HERE / "reports"
PREFIX = "ragforge_"

# (key, collection short name, questions file or None). Targets first, then the new documents in new_docs.tsv order.
COLLECTION_ORDER = [
    ("ccby", "ccby_4_0_legalcode", "evaluation/eval_questions_ccby.json"),
    ("cdc", "cdc_program_evaluation_framework_2024", "evaluation/eval_questions_cdc_program_eval.json"),
    ("ragbench", "ragbench_2407_11005v2", "evaluation/eval_questions_ragbench.json"),
    ("attention", "attention_is_all_you_need", None),
    ("nist_ai_rmf", "nist_ai_100_1", None),
    ("ipcc_spm", "ipcc_ar6_syr_spm", None),
    ("alice", "alice_in_wonderland", None),
    ("who_handhygiene", "who_hand_hygiene_guidelines_excerpt_parts_ii_iv", None),
    ("roomba", "roomba__600_series_owner_s_guide", None),
    ("tootsie", "tootsie_roll_industries_2024_annual_report", None),
]
TARGETS = COLLECTION_ORDER[:3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["rrf", "dense"])
    ap.add_argument("--n", required=True, type=int, choices=[3, 6, 10])
    ap.add_argument("--dense-min-score", type=float, default=0.65, help="same threshold as Finding 2 / experiment 4")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config" / "stage4.yml"))
    args = ap.parse_args()

    marker = REPORTS / ".exp4_store_dirty"
    if marker.exists():
        raise SystemExit("Stage 1 (baseline) of experiment 4 rebuilt some collections with the fixed-size chunker.\n"
                         "Run `uv run python evaluation/index_new_docs.py` first (it resets the store), then re-run this.")

    cdt.load_dotenv(PROJECT_ROOT / ".env")
    config = cdt.RagForgeConfig.load(args.config)
    built = cdt.build_everything(config, agentic_merge_source=args.arm,
                                 agentic_dense_min_score=args.dense_min_score if args.arm == "dense" else 0.0)
    store = built["vector_store"]
    present = set(store.list_collections())
    missing = [PREFIX + short for _, short, _ in COLLECTION_ORDER if PREFIX + short not in present]
    if missing:
        raise SystemExit(f"These collections are not in the store: {missing}\nRun evaluation/index_new_docs.py first.")

    allowed = [PREFIX + short for _, short, _ in COLLECTION_ORDER[: args.n]]
    real_list = store.list_collections
    store.list_collections = lambda: [c for c in real_list() if c in allowed]   # Agentic reads this on every search
    assert store.list_collections() and len(store.list_collections()) == args.n, store.list_collections()
    print(f"arm={args.arm}  N={args.n}  collections visible to Agentic: {store.list_collections()}", flush=True)

    strategy = built["agentic"]
    for key, short, qfile in TARGETS:
        out = REPORTS / f"decision_traces_exp5_{key}_{args.arm}_n{args.n}.json"
        if out.exists():
            print(f"--- {out.name}: already done, skipping ---")
            continue
        items = json.loads((PROJECT_ROOT / qfile).read_text(encoding="utf-8"))
        print(f"--- {key}: {len(items)} questions ---", flush=True)
        records = []
        for i, item in enumerate(items, start=1):
            print(f"[{key} {i}/{len(items)}] {item['question']}", flush=True)
            gen, trace = cdt.run_agentic(strategy, item["question"], short)
            records.append({
                "question": item["question"], "type": item.get("type"), "reference": item.get("reference"),
                "answerable": item.get("reference") is not None, "answer": gen.answer,
                "retrieved_contexts": [c.content for c in gen.retrieved_chunks], "trace": trace,
            })
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "strategy": "agentic", "experiment": 5, "arm": args.arm, "n_collections": args.n, "collection": short,
            "vector_store_collections": store.list_collections(),
            "settings": {"agentic_merge_source": args.arm,
                         "agentic_dense_min_score": args.dense_min_score if args.arm == "dense" else 0.0},
            "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(records)} traces to {out}", flush=True)


if __name__ == "__main__":
    main()
