"""Routes document parsing requests to the parser matching the file
extension.
"""

from __future__ import annotations

from pathlib import Path

from .markdown_parser import MarkdownParser
from .parse_result import ParseResult
from .pdf_parser import PdfParser


class ParserRouter:
    """Dispatches a parsing request to the parser for the matching file
    extension (simple factory / router pattern).
    """

    def __init__(self) -> None:
        self._markdown_parser = MarkdownParser()
        self._pdf_parser = PdfParser()

    def parse(self, file_path: str) -> ParseResult:
        """Parse a document using the parser for its file extension.

        Args:
            file_path: Path to the document.

        Returns:
            The parsed result.

        Raises:
            ValueError: If the file extension is not supported.
        """
        ext = self._get_extension(file_path).lower()
        if ext in ("md", "markdown"):
            return self._markdown_parser.parse(file_path)
        if ext == "pdf":
            return self._pdf_parser.parse(file_path)
        raise ValueError(f"Unsupported file format: {ext}")

    def supports(self, file_path: str) -> bool:
        """Check whether the file extension is supported."""
        ext = self._get_extension(file_path).lower()
        return ext in ("md", "markdown", "pdf")

    @staticmethod
    def _get_extension(file_path: str) -> str:
        """Extract the file extension, without the leading dot."""
        return Path(file_path).suffix.lstrip(".")
