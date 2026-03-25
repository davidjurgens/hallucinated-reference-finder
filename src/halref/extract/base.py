"""Abstract interfaces for text extraction and field parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from halref.models import Reference


class TextExtractor(ABC):
    """Extracts raw text from PDF reference pages."""

    name: str = "base"

    @abstractmethod
    def extract_text(
        self, pdf_path: Path, page_range: tuple[int, int] | None = None
    ) -> str:
        """Extract reference section text from PDF.

        Args:
            pdf_path: Path to the PDF file.
            page_range: Optional (start, end) 0-indexed page range.
                       If None, auto-detect reference section.

        Returns:
            Raw text of the reference section.
        """
        ...

    def is_available(self) -> bool:
        """Check if this extractor's dependencies are installed."""
        return True


class FieldParser(ABC):
    """Parses a raw reference string into structured fields."""

    name: str = "base"

    @abstractmethod
    def parse(self, raw_text: str) -> Reference:
        """Parse a single reference string into a Reference object.

        Args:
            raw_text: A single reference entry as a string.

        Returns:
            Parsed Reference with fields populated.
        """
        ...

    def parse_confidence(self, ref: Reference) -> float:
        """Estimate confidence of the parse (0-1).

        A valid academic reference should have a title, authors, and year.
        Missing year is a strong signal that this isn't a real reference.
        """
        import re

        score = 0.0
        if ref.title and len(ref.title) > 10:
            score += 0.3
        if ref.authors and len(ref.authors) >= 1:
            score += 0.3
        if ref.year and 1900 <= ref.year <= 2030:
            score += 0.3
        if ref.venue:
            score += 0.1

        # Penalty: raw text has no comma or "and" (unlikely multi-author ref)
        raw = ref.raw_text
        if raw and "," not in raw and " and " not in raw.lower():
            score -= 0.2

        # Penalty: raw text looks like a list item or table cell
        if raw and re.match(r"^\d+\.\s+\w+:", raw):
            score -= 0.3
        if raw and re.match(r"^[•\-\*]\s", raw):
            score -= 0.3

        return max(0.0, min(1.0, score))

    def is_available(self) -> bool:
        """Check if this parser's dependencies are installed."""
        return True
