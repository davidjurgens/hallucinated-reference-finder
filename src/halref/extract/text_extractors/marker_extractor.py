"""Marker-based text extraction with ML layout understanding."""

from __future__ import annotations

import re
from pathlib import Path

from halref.extract.base import TextExtractor


class MarkerExtractor(TextExtractor):
    name = "marker"

    def is_available(self) -> bool:
        try:
            import marker  # noqa: F401
            return True
        except ImportError:
            return False

    def extract_text(
        self, pdf_path: Path, page_range: tuple[int, int] | None = None
    ) -> str:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        models = create_model_dict()
        converter = PdfConverter(artifact_dict=models)
        rendered = converter(str(pdf_path))
        md = rendered.markdown

        return self._extract_references(md)

    def _extract_references(self, md: str) -> str:
        """Find references section in Marker's markdown output."""
        pattern = re.compile(
            r"^#{1,3}\s*(?:References|Bibliography|Works Cited)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        match = pattern.search(md)
        if not match:
            return ""

        text_after = md[match.end():].strip()

        # Stop at next major heading
        next_heading = re.search(r"^#{1,2}\s+\w", text_after, re.MULTILINE)
        if next_heading:
            text_after = text_after[:next_heading.start()].strip()

        return text_after
