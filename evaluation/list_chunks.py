"""Dev tool: list every chunk RagForge would produce for a document, together
with its chunk_id, so you can eyeball them and pick the correct
`reference_chunk_ids` for evaluation/eval_questions.json (needed for the
hitRate metric -- see collect_predictions.py / run_ragas_eval.py). Also
doubles as the comparison tool for the 5 chunking strategies: pass
`--strategy` to preview any of them (fixed_size / recursive / semantic /
parent_child / structure_aware) against the same document.

This does NOT touch Milvus or the Anthropic API. `--strategy semantic` DOES
call Ollama's embedding endpoint (SemanticChunker needs real embeddings to
group sentences) -- every other strategy stays purely local, cheap to
re-run over and over.

Runs inside the normal RagForge `.venv` (same one `collect_predictions.py`
uses). Two code paths:

  1. Preferred: import the real `ragforge.parser.ParserRouter` +
     `ragforge.chunker.ChunkerFactory.create_chunker` and run the document
     through them exactly as the indexing pipeline would, using whichever
     strategy `--strategy` selects. Used automatically whenever those
     modules import cleanly in your environment.
  2. Fallback: if the real modules aren't importable, read the file
     directly (UTF-8 text for .md, pypdf for .pdf) and chunk it with a
     verbatim copy of FixedSizeChunker's chunk()/_generate_id() logic. This
     fallback only implements fixed_size -- if `--strategy` selects
     anything else and the real modules can't be imported, the tool exits
     with an error instead of guessing. Because the id algorithm
     (md5(f"{filename}_{index}")[:12]) and the chunk() splitting logic are
     copied unchanged from src/ragforge/chunker/FixedSizeChunker.py,
     fallback chunk_ids match the real pipeline's ids exactly as long as
     the parsed text is the same -- true for markdown (no real "parsing"
     happens beyond reading the file), and true for PDFs to the extent the
     real PdfParser's extraction agrees with pypdf's. The tool always prints
     which path it used so you know which guarantee applies.

Usage:
    uv run python evaluation/list_chunks.py --document /path/to/document.md
    uv run python evaluation/list_chunks.py --document doc.md --strategy semantic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402

STRATEGIES = ["fixed_size", "recursive", "semantic", "parent_child", "structure_aware"]


def _load_config(config_path: Path) -> dict:
    """Read the whole yaml file as a plain dict, independent of
    RagForgeConfig, so this still works even if that dataclass changes
    shape (the fallback path needs to keep working without it)."""
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _fallback_read_text(document: Path) -> str:
    ext = document.suffix.lower().lstrip(".")
    if ext in ("md", "markdown", "txt"):
        return document.read_text(encoding="utf-8")
    if ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "pypdf is required to fall back to PDF text extraction "
                "(it's already a project dependency; if this fails your "
                ".venv may not be active -- try `uv run ...`)."
            ) from e
        reader = PdfReader(str(document))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise SystemExit(
        f"Don't know how to read '.{ext}' files without the real ragforge "
        "parser. Supported fallback extensions: .md, .markdown, .txt, .pdf"
    )


def _fallback_chunk(text: str, source: str, max_size: int, overlap: int) -> list[dict]:
    """Verbatim copy of FixedSizeChunker.chunk() + _generate_id(), minus the
    Chunk/ParseResult dataclasses this module doesn't have access to. Kept
    byte-for-byte in sync with
    src/ragforge/chunker/FixedSizeChunker.py -- if that algorithm changes,
    update this copy too so fallback ids keep matching the real pipeline.
    """
    chunks: list[dict] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + max_size, len(text))

        if end < len(text):
            last_period_cn = text.rfind("。", 0, end + 1)
            last_period_en = text.rfind(".", 0, end + 1)
            last_newline = text.rfind("\n", 0, end + 1)
            split_point = max(last_period_cn, last_period_en, last_newline)
            if split_point > start:
                end = split_point + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            raw = f"{Path(source).name}_{index}"
            chunk_id = hashlib.md5(raw.encode()).hexdigest()[:12]
            chunks.append({"index": index, "chunk_id": chunk_id, "content": chunk_text})

        next_start = end - overlap
        min_advance = max(max_size - overlap, 1)
        if next_start < start + min_advance:
            next_start = start + min_advance
        if next_start <= start:
            next_start = end
        start = next_start
        index += 1

    return chunks


def _try_real_pipeline(
    document: Path, strategy: str, chunker_raw: dict, embedding_raw: dict
) -> list[dict] | None:
    """Run `document` through the real ParserRouter + ChunkerFactory, using
    whichever strategy `--strategy` selected. Returns None (triggering the
    fallback path) if the real modules aren't importable in this
    environment. `chunker_raw`/`embedding_raw` are the raw yaml dicts for
    the `chunker`/`embedding` config sections -- passed through as-is so
    this stays in sync with ChunkerConfig/EmbeddingConfig automatically.
    """
    try:
        from ragforge.chunker.chunker_factory import create_chunker
        from ragforge.config import ChunkerConfig, EmbeddingConfig
        from ragforge.embedding.embedding_service import EmbeddingService
        from ragforge.parser.parser_router import ParserRouter
    except ImportError:
        return None

    router = ParserRouter()
    if not router.supports(str(document)):
        raise SystemExit(f"ParserRouter doesn't support '{document.suffix}' files.")

    parse_result = router.parse(str(document))

    config = ChunkerConfig(**{**chunker_raw, "strategy": strategy})
    # Constructing EmbeddingService makes no network call by itself -- only
    # SemanticChunker's later .embed_batch() call does -- so it's cheap and
    # safe to always build one, even for strategies that never touch it.
    embedding_service = EmbeddingService(EmbeddingConfig(**embedding_raw))

    chunker = create_chunker(config, embedding_service)
    real_chunks = chunker.chunk(parse_result)
    return [
        {"index": i, "chunk_id": c.id, "content": c.content}
        for i, c in enumerate(real_chunks)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List chunk_id + content for every chunk of a document, "
        "for picking reference_chunk_ids in eval_questions.json."
    )
    parser.add_argument("--document", required=True, help="Path to the document to chunk.")
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="fixed_size",
        help="Which chunking strategy to preview (default: fixed_size). "
        "'semantic' calls Ollama's embedding endpoint; the rest stay local.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "stage1.yml"),
        help="Yaml config to read the chunker/embedding sections from (e.g. stage1.yml or stage2.yml).",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "evaluation" / "reports" / "chunks.json"),
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=100,
        help="How many characters of each chunk to show in the console table.",
    )
    args = parser.parse_args()

    document = Path(args.document)
    if not document.exists():
        raise SystemExit(f"Document not found: {document}")

    raw = _load_config(Path(args.config))
    chunker_raw = raw.get("chunker") or {}
    embedding_raw = raw.get("embedding") or {}
    max_size = chunker_raw.get("maxSize", 500)
    overlap = chunker_raw.get("overlap", 50)

    # The field that actually bounds chunk size differs by strategy
    # (structure_aware uses structureMaxSize, not maxSize) -- report the
    # one that's actually in effect so the printed config isn't misleading.
    effective_size_field = "structureMaxSize" if args.strategy == "structure_aware" else "maxSize"
    effective_size = chunker_raw.get(effective_size_field, max_size if effective_size_field == "maxSize" else 800)

    chunks = _try_real_pipeline(document, args.strategy, chunker_raw, embedding_raw)
    used_path = f"real ragforge pipeline (ParserRouter + ChunkerFactory, strategy={args.strategy!r})"
    if chunks is None:
        if args.strategy != "fixed_size":
            raise SystemExit(
                f"--strategy {args.strategy!r} requires the real ragforge "
                "chunker modules, but they could not be imported (this "
                "checkout may be missing Chunk.py / ParseResult.py / the "
                "chunker implementations at the time of writing). The "
                "fallback path only implements 'fixed_size' -- fix the "
                "import, or re-run with --strategy fixed_size."
            )
        print(
            "Note: could not import ragforge.parser.ParserRouter / "
            "ragforge.chunker.ChunkerFactory (this checkout may be missing "
            "Chunk.py / ParseResult.py / the parser implementations at the "
            "time of writing). Falling back to a direct-read + copied-logic "
            "chunker -- see this file's module docstring for exactly what "
            "guarantee that gives you.\n"
        )
        text = _fallback_read_text(document)
        chunks = _fallback_chunk(text, str(document), max_size, overlap)
        used_path = "fallback (direct file read + copied FixedSizeChunker logic, strategy=fixed_size)"

    print(f"Chunked '{document}' via: {used_path}")
    print(
        f"chunker config: strategy={args.strategy}, "
        f"{effective_size_field}={effective_size}, overlap={overlap}"
    )
    print(f"{len(chunks)} chunks\n")

    n = args.preview_chars
    for c in chunks:
        preview = c["content"][:n].replace("\n", " ")
        ellipsis = "..." if len(c["content"]) > n else ""
        print(f"[{c['index']:>3}] {c['chunk_id']}  ({len(c['content'])} chars)  {preview}{ellipsis}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "document": str(document),
                "chunker": {
                    "strategy": args.strategy,
                    effective_size_field: effective_size,
                    "overlap": overlap,
                },
                "chunks": chunks,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nFull chunk text written to {out_path} -- open it, pick the chunk_id(s) "
          f"that answer each question, and paste them into "
          f"eval_questions.json's \"reference_chunk_ids\".")


if __name__ == "__main__":
    main()
