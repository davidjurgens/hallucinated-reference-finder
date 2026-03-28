"""pdfminer.six-based text extraction — best quality for two-column ACL PDFs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from halref.extract.base import TextExtractor

logger = logging.getLogger(__name__)

# Patterns that signal the end of the references section.
# Must handle various appendix heading formats:
#   "Appendix", "A Appendix", "8 Appendix", "A.1 Details",
#   "A Illustrating Narrative", "Limitations", "Ethics Statement"
_STOP_PATTERNS = re.compile(
    r"\n\s*(?:"
    # Direct appendix keywords
    r"Appendix|Appendices|Supplementary|Supplemental|Checklist"
    # "A Title" or "B Title" (single letter section = appendix)
    r"|[A-H]\s+[A-Z][a-z]"
    # "8 Appendix" or "9 Supplementary" (numbered appendix)
    r"|\d+\s+(?:Appendix|Supplementary|Additional)"
    # "A.1 Details" or "B.2 Examples"
    r"|[A-H]\.\d+\s+[A-Z]"
    # Post-references sections
    r"|Acknowledgment|Acknowledgement"
    r"|Ethics\s+Statement|Limitations"
    r"|Broader\s+Impact|Impact\s+Statement"
    # Additional appendix indicators
    r"|Additional\s+(?:Experiments|Details|Results|Analysis|Examples)"
    r"|Supplementary\s+Material"
    r"|Reproducibility"
    # Appendix content markers (figures/tables as standalone headings)
    r"|(?:Figure|Table)\s+\d+\s*:"
    # Common appendix headings
    r"|Prompt\s+Template|Evaluation\s+(?:Form|Rubric|Criteria)"
    r"|Implementation\s+Details|Hyperparameter"
    r"|Dataset\s+(?:Details|Statistics|Description)"
    r")\b",
    re.IGNORECASE,
)

# Patterns that match a References/Bibliography heading
_REF_HEADING = re.compile(
    r"(?:^|\n)\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n",
)


class PdfminerExtractor(TextExtractor):
    """Extract text using pdfminer.six.

    Advantages over pdfplumber:
    - Correctly handles two-column layouts without manual cropping
    - Preserves word spacing (no missing spaces)
    - Inserts blank lines between references, making splitting trivial
    """

    name = "pdfminer"

    def extract_text(
        self, pdf_path: Path, page_range: tuple[int, int] | None = None
    ) -> str:
        from pdfminer.high_level import extract_text

        if page_range:
            start, end = page_range
            page_numbers = list(range(start, end))
            text = extract_text(str(pdf_path), page_numbers=page_numbers)
        else:
            text = self._extract_auto(pdf_path)

        text = self._strip_line_numbers(text)
        return self._find_references_section(text)

    @staticmethod
    def _strip_line_numbers(text: str) -> str:
        """Remove line numbers from ACL review-mode PDFs."""
        lines = text.split("\n")
        filtered = [line for line in lines if not re.match(r"^\s*\d{1,4}\s*$", line)]
        return "\n".join(filtered)

    def _extract_auto(self, pdf_path: Path) -> str:
        """Find the References section by scanning pages backwards.

        Scans from the last page backwards to find the "References" heading,
        then extracts from that page through the end of the references
        (stopping at Appendix/Acknowledgments).

        This handles papers of any length, including those with appendices
        after the references.
        """
        from pdfminer.high_level import extract_text
        from pdfminer.pdfpage import PDFPage

        with open(pdf_path, "rb") as f:
            total_pages = sum(1 for _ in PDFPage.get_pages(f))

        # Scan pages backwards to find the References heading
        ref_page = None
        for page_num in range(total_pages - 1, -1, -1):
            text = extract_text(str(pdf_path), page_numbers=[page_num])
            text = self._strip_line_numbers(text)
            if _REF_HEADING.search(text):
                ref_page = page_num
                break

        if ref_page is None:
            logger.warning(f"No References heading found in {pdf_path.name}")
            # Last resort: return last 3 pages
            start = max(0, total_pages - 3)
            fallback = extract_text(str(pdf_path), page_numbers=list(range(start, total_pages)))
            # Sanity check: if the fallback text has no whitespace it's garbled
            # (pathological PDF with no layout info — pdfminer can't parse it)
            if fallback and fallback.count(" ") + fallback.count("\n") < len(fallback) * 0.05:
                logger.warning(
                    f"{pdf_path.name}: extracted text has no whitespace — "
                    f"PDF may be unreadable by pdfminer (try pypdf or pdfplumber)"
                )
                return ""
            return fallback

        # Extract from the References page through ALL remaining pages.
        # _find_references_section() will stop at Appendix/Limitations/etc.
        pages = list(range(ref_page, total_pages))
        logger.debug(
            f"References heading on page {ref_page + 1}, "
            f"extracting pages {ref_page + 1}-{total_pages}"
        )
        return extract_text(str(pdf_path), page_numbers=pages)

    def _find_references_section(self, text: str) -> str:
        """Extract text between References heading and next section."""
        match = _REF_HEADING.search(text)
        if match:
            after = text[match.end():]
            # Stop at appendix, acknowledgments, etc.
            stop = _STOP_PATTERNS.search(after)
            if stop:
                after = after[:stop.start()]
            return after.strip()

        # No heading found — return all text (user specified page range)
        return text.strip()
