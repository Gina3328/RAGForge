"""Markdown document parser."""

from __future__ import annotations

from pathlib import Path

from .parse_result import Metadata, ParseResult


class MarkdownParser:
    """Parses Markdown files, extracting full text and section structure.

    Sections are identified by top-level headings (lines starting with
    "# "), which downstream structure-aware chunking relies on. The first
    top-level heading also becomes the document title.
    """

    def parse(self, file_path: str) -> ParseResult:
        """Read and parse a Markdown file.

        Args:
            file_path: Path to the Markdown file.

        Returns:
            A ParseResult with the full text, title, and section list.
        """
        try:
            path = Path(file_path)
            content = path.read_text(encoding="utf-8")

            # Collect every top-level heading as a section.
            sections: list[str] = []
            for line in content.split("\n"):
                trimmed = line.strip()
                if trimmed.startswith("# "):
                    sections.append(trimmed[2:].strip())

            # Use the first top-level heading as the title, falling back
            # to the file name if the document has none.
            title = sections[0] if sections else path.name

            metadata = Metadata(title=title, source=file_path, sections=sections)
            return ParseResult(content=content, metadata=metadata)
        except OSError as e:
            raise RuntimeError(f"Failed to parse Markdown: {file_path}") from e
