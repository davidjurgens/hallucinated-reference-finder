"""pypdf-based text extraction — good quality, already installed."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from halref.extract.base import TextExtractor

logger = logging.getLogger(__name__)

# Reuse patterns from pdfminer extractor
_STOP_PATTERNS = re.compile(
    r"\n\s*(?:"
    r"Appendix|Appendices|Supplementary|Supplemental|Checklist"
    r"|[A-H]\s+[A-Z][a-z]"
    r"|\d+\s+(?:Appendix|Supplementary|Additional)"
    r"|[A-H]\.\d+\s+[A-Z]"
    r"|Acknowledgment|Acknowledgement"
    r"|Ethics\s+Statement|Limitations"
    r"|Broader\s+Impact|Impact\s+Statement"
    r")\b",
    re.IGNORECASE,
)

_REF_HEADING = re.compile(
    r"(?:^|\n)\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n",
)


class PypdfExtractor(TextExtractor):
    """Extract text using pypdf.

    Good word spacing and two-column handling.
    Does not insert blank lines between references (unlike pdfminer).
    """

    name = "pypdf"

    def extract_text(
        self, pdf_path: Path, page_range: tuple[int, int] | None = None
    ) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))

        if page_range:
            start, end = page_range
            pages = range(start, min(end, len(reader.pages)))
            texts = [reader.pages[i].extract_text() or "" for i in pages]
        else:
            texts = self._extract_auto(reader)

        full_text = "\n\n".join(texts)
        full_text = self._strip_line_numbers(full_text)
        return self._find_references_section(full_text)

    def _extract_auto(self, reader) -> list[str]:
        """Scan pages backwards to find References, extract from there."""
        total = len(reader.pages)
        ref_page = None

        for i in range(total - 1, -1, -1):
            text = reader.pages[i].extract_text() or ""
            text = self._strip_line_numbers(text)
            if _REF_HEADING.search(text):
                ref_page = i
                break

        if ref_page is None:
            logger.warning("No References heading found")
            start = max(0, total - 3)
            return [reader.pages[i].extract_text() or "" for i in range(start, total)]

        end_page = min(ref_page + 5, total)
        return [reader.pages[i].extract_text() or "" for i in range(ref_page, end_page)]

    @staticmethod
    def _strip_line_numbers(text: str) -> str:
        """Remove line numbers from ACL review-mode PDFs."""
        lines = text.split("\n")
        filtered = [line for line in lines if not re.match(r"^\s*\d{1,4}\s*$", line)]
        return "\n".join(filtered)

    def _find_references_section(self, text: str) -> str:
        """Extract text between References heading and next section."""
        match = _REF_HEADING.search(text)
        if match:
            after = text[match.end():]
            stop = _STOP_PATTERNS.search(after)
            if stop:
                after = after[:stop.start()]
            return after.strip()

        return text.strip()
