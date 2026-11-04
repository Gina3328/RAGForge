"""Experiment 4, step 0: put the 7 new documents into the vector store, with the Stage 4 settings
(parent-child chunker), so that every Agentic run in experiment 4 sees the same set of collections.

Run once, BEFORE evaluation/run_new_docs.sh, with Docker/Milvus and Ollama running:

    uv run python evaluation/index_new_docs.py

It indexes the 3 original documents (background: they are the other collections Agentic searches) and the 7 new
documents listed in evaluation/new_docs.tsv (the WHO file is the 41-page excerpt, NOT the 270-page PDF), then
prints the collections in the store. Expected: 10 collections = the 3 original + 7 new documents.
(The two miscellaneous collections of the main experiment, readme and test_doc, are not recreated.)
Re-running is safe: each document's collection is dropped and rebuilt.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

import collect_decision_traces as cdt  # noqa: E402
from ragforge.config import RagForgeConfig  # noqa: E402


BACKGROUND = [
    "evaluation/documents/ccby_4.0_legalcode.pdf",
    "evaluation/documents/cdc_program_evaluation_framework_2024.pdf",
    "evaluation/documents/ragbench_2407.11005v2.pdf",
]


def read_docs() -> list[tuple[str, Path, Path]]:
    rows = []
    for line in (HERE / "new_docs.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, doc_glob, questions = line.split("\t")
        hits = sorted((PROJECT_ROOT).glob(doc_glob))
        if len(hits) != 1:
            sys.exit(f"{key}: pattern {doc_glob!r} matched {len(hits)} files: {[h.name for h in hits]}")
        rows.append((key, hits[0], PROJECT_ROOT / questions))
    return rows


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    config = RagForgeConfig.load(str(PROJECT_ROOT / "config" / "stage4.yml"))
    built = cdt.build_everything(config)
    pipe = built["naive_pipeline"]
    for rel in BACKGROUND:
        doc = PROJECT_ROOT / rel
        if not doc.exists():
            sys.exit(f"missing background document: {doc}")
        print(f"Indexing background document {doc.name} ...", flush=True)
        print(f"  -> collection {pipe.index_document(str(doc))}", flush=True)
    for key, doc, _ in read_docs():
        print(f"Indexing {key}: {doc.name} ...", flush=True)
        name = pipe.index_document(str(doc))
        print(f"  -> collection {name}", flush=True)
    (PROJECT_ROOT / "evaluation" / "reports" / ".exp4_store_dirty").unlink(missing_ok=True)
    cols = built["vector_store"].list_collections()
    print(f"\n{len(cols)} collections in the store:")
    for c in sorted(cols):
        print("  ", c)
    if len(cols) != 10:
        print("\nWARNING: expected 10 collections (3 original + 7 new documents). Check the list above before running experiment 4.")


if __name__ == "__main__":
    main()
