"""pdfplumber-based text extraction with targeted page support."""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from halref.extract.base import TextExtractor


class PdfplumberExtractor(TextExtractor):
    name = "pdfplumber"

    def extract_text(
        self, pdf_path: Path, page_range: tuple[int, int] | None = None
    ) -> str:
        if page_range:
            return self._extract_pages(pdf_path, page_range)
        return self._extract_auto(pdf_path)

    def _extract_pages(self, pdf_path: Path, page_range: tuple[int, int]) -> str:
        """Extract text from specific pages (0-indexed).

        When a page range is specified, the user has indicated these pages
        contain references. We extract all text and try to find the References
        heading. If found, we return text after it. If not (e.g., references
        span multiple pages and the heading is only on the first), we return
        all extracted text, trusting the user's page range.
        """
        start, end = page_range
        pages_idx = list(range(start, end))
        all_page_texts = []
        with pdfplumber.open(pdf_path, pages=pages_idx) as pdf:
            for page in pdf.pages:
                text = self._extract_page_columns(page)
                if text:
                    all_page_texts.append(text)

        # Strategy: For each page, try to find the References heading.
        # Once found, take everything from that point onward across all pages.
        ref_text_parts = []
        found_heading = False

        for page_text in all_page_texts:
            if not found_heading:
                ref_match = re.search(
                    r"(?:^|\n)\s*References\s*\n",
                    page_text,
                    re.IGNORECASE,
                )
                if ref_match:
                    found_heading = True
                    ref_text_parts.append(page_text[ref_match.end():].strip())
                # Also check if the page is mostly references (no heading but
                # looks like reference content - starts with author names)
                elif self._looks_like_references(page_text):
                    ref_text_parts.append(page_text.strip())
            else:
                ref_text_parts.append(page_text.strip())

        if ref_text_parts:
            return "\n\n".join(ref_text_parts)

        # Fallback: return all text (trust the user's page range)
        return "\n\n".join(all_page_texts)

    def _looks_like_references(self, text: str) -> bool:
        """Heuristic: does this text look like it's mostly references?"""
        # Count how many year patterns (19xx or 20xx) appear
        years = re.findall(r"\b(?:19|20)\d{2}\b", text)
        # If there are multiple years and names, it's likely references
        return len(years) >= 3

    def _extract_page_columns(self, page) -> str:
        """Extract text from a page, handling two-column layout.

        Splits the page at the midpoint and reads left column then right column.
        """
        width = page.width
        height = page.height
        mid = width / 2

        # Try column-based extraction
        left_bbox = (0, 0, mid, height)
        right_bbox = (mid, 0, width, height)

        left_crop = page.crop(left_bbox)
        right_crop = page.crop(right_bbox)

        left_text = left_crop.extract_text() or ""
        right_text = right_crop.extract_text() or ""

        # Combine: left column first, then right column
        return left_text + "\n\n" + right_text

    def _extract_auto(self, pdf_path: Path) -> str:
        """Auto-detect reference section by scanning for heading.

        Scans pages from the end backwards looking for the References heading.
        For ACL papers, references are typically in the last 1-3 pages.
        """
        with pdfplumber.open(pdf_path) as pdf:
            # Scan from the end to find references start
            all_text = []
            found_refs = False

            # Check last 5 pages (references rarely span more)
            start_page = max(0, len(pdf.pages) - 5)
            for i in range(start_page, len(pdf.pages)):
                page = pdf.pages[i]
                text = self._extract_page_columns(page)
                if not text:
                    continue

                if not found_refs:
                    # Look for "References" as a standalone heading
                    ref_match = re.search(
                        r"(?:^|\n)\s*References\s*\n",
                        text,
                        re.IGNORECASE,
                    )
                    if ref_match:
                        found_refs = True
                        all_text.append(text[ref_match.end():].strip())
                else:
                    # Check for appendix
                    appendix_match = re.match(
                        r"\s*(?:Appendix|Appendices|Supplementary)\b",
                        text.strip(),
                        re.IGNORECASE,
                    )
                    if appendix_match:
                        break
                    all_text.append(text)

            return "\n\n".join(all_text)

    def _extract_refs_from_text(self, text: str) -> str:
        """If the text contains a References heading, return only the text after it."""
        ref_match = re.search(
            r"(?:^|\n)\s*References\s*\n",
            text,
            re.IGNORECASE,
        )
        if ref_match:
            return text[ref_match.end():].strip()
        return text
