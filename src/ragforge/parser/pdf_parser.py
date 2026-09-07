"""PDF document parser."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .parse_result import Metadata, ParseResult


class PdfParser:
    """Parses PDF files, extracting plain text page by page.

    Note: PDF parsing generally cannot recover section/heading structure,
    so `sections` is always left empty. Downstream chunking falls back to
    a fixed-size or recursive strategy for PDF-sourced content.
    """

    def parse(self, file_path: str) -> ParseResult:
        """Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A ParseResult with the full extracted text and metadata
            (no section structure).
        """
        try:
            reader = PdfReader(file_path)
            pages_text = [page.extract_text() or "" for page in reader.pages]
            content = "\n".join(pages_text)

            # PDFs rarely expose a clean title, so fall back to the file name.
            title = Path(file_path).name
            metadata = Metadata(title=title, source=file_path, sections=[])
            return ParseResult(content=content, metadata=metadata)
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF: {file_path}") from e
