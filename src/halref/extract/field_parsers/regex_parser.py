"""Regex-based field parser tuned for ACL reference format."""

from __future__ import annotations

import re

from nameparser import HumanName

from halref.extract.base import FieldParser
from halref.models import Author, Reference


class RegexFieldParser(FieldParser):
    """Parse ACL-format references using regex patterns.

    Expected format:
        Last, First and First Last. Year. Title. In Venue, pages X-Y.
    """

    name = "regex"

    # Pattern for the year (4 digits, optionally followed by a/b/c for multi-paper years)
    YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})[a-z]?\b")

    # Pattern for "In Proceedings of..." or "In ..." venue block
    VENUE_PATTERN = re.compile(
        r"\b[Ii]n\s+"
        r"(Proceedings\s+of\s+.*?"
        r"|Advances\s+in\s+.*?"
        r"|Transactions\s+of\s+.*?"
        r"|(?:the\s+)?(?:\d+(?:st|nd|rd|th)\s+)?"
        r"[A-Z].*?)"
        r"(?:\.\s*$|,\s*pages\b|,\s*pp\b|\.$)",
        re.DOTALL,
    )

    # Pattern for DOI
    DOI_PATTERN = re.compile(r"\b(10\.\d{4,}/[^\s]+)")

    # Pattern for URL
    URL_PATTERN = re.compile(r"(https?://[^\s]+)")

    def parse(self, raw_text: str) -> Reference:
        text = raw_text.strip()
        # Strip leading [N] reference numbers
        text = re.sub(r"^\[\d+\]\s*", "", text)
        # Dehyphenate and normalize whitespace
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        text = re.sub(r"\s+", " ", text)
        ref = Reference(raw_text=raw_text)

        # Extract DOI and URL first (remove from text for cleaner parsing)
        doi_match = self.DOI_PATTERN.search(text)
        if doi_match:
            ref.doi = doi_match.group(1).rstrip(".")
            text = text[:doi_match.start()] + text[doi_match.end():]

        url_match = self.URL_PATTERN.search(text)
        if url_match:
            ref.url = url_match.group(1).rstrip(".")
            text = text[:url_match.start()] + text[url_match.end():]

        text = text.strip()

        # Find year — in ACL format, year appears after authors: "Authors. Year."
        year_match = self.YEAR_PATTERN.search(text)
        if year_match:
            ref.year = int(year_match.group(1))
            year_pos = year_match.start()

            # Everything before year is authors
            authors_text = text[:year_pos].strip().rstrip(".")
            ref.authors = self._parse_authors(authors_text)

            # Everything after year is title + venue
            after_year = text[year_match.end():].strip().lstrip(".")
            ref.title, ref.venue, ref.pages = self._parse_title_venue(after_year)
        else:
            # No year found — try to at least get a title
            # Assume first sentence is title
            parts = text.split(". ", 1)
            if len(parts) > 1:
                ref.title = parts[0].strip()

        ref.extraction_confidence = self.parse_confidence(ref)
        return ref

    def _parse_authors(self, text: str) -> list[Author]:
        """Parse author string into Author objects.

        Handles ACL natbib format: "First Last, First Last, and First Last"
        Also handles: "Last, First and Last, First" (less common in natbib output)
        """
        if not text:
            return []

        # First, replace " and " with a separator
        text = re.sub(r",?\s+and\s+", " && ", text)
        text = text.replace(";", " && ")

        # Split on "&&" first
        and_parts = text.split(" && ")

        # Each and-part may contain comma-separated "First Last" authors
        # Detect format: if parts look like "First Last, First Last" (no "Last, First" pattern)
        # then split on commas too
        all_parts = []
        for part in and_parts:
            part = part.strip().strip(",").strip()
            if not part:
                continue
            # Check if this looks like multiple "First Last" entries separated by commas
            # Heuristic: if there are commas and the text before first comma
            # doesn't look like "Last, First" (i.e., the part after comma starts with uppercase)
            sub_parts = [p.strip() for p in part.split(",") if p.strip()]
            if len(sub_parts) > 1 and self._looks_like_first_last_list(sub_parts):
                all_parts.extend(sub_parts)
            else:
                all_parts.append(part)

        authors = []
        for part in all_parts:
            part = part.strip().strip(",").strip()
            if not part or part.lower() in ("et al", "et al.", "others"):
                continue
            # Skip numeric artifacts like "1 others"
            if re.match(r"^\d+\s", part):
                continue
            author = self._parse_single_author(part)
            if author.last:
                authors.append(author)

        return authors

    @staticmethod
    def _looks_like_first_last_list(parts: list[str]) -> bool:
        """Check if comma-separated parts look like 'First Last' entries.

        Returns True if most parts start with a capitalized word followed by
        another capitalized word (First Last pattern), rather than being
        'Last, First' format.
        """
        first_last_count = 0
        for part in parts:
            part = part.strip()
            # "First Last" pattern: starts with Capital word, has space, another Capital
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z]|\s+[a-z]+\s+[A-Z])", part):
                first_last_count += 1
            # Also match initials like "Jared D. Kaplan"
            elif re.match(r"^[A-Z][a-z]+\s+[A-Z]\.", part):
                first_last_count += 1
        return first_last_count >= len(parts) * 0.5

    def _parse_single_author(self, name_str: str) -> Author:
        """Parse a single author name."""
        name_str = name_str.strip().strip(",").strip()
        parsed = HumanName(name_str)
        return Author(
            first=parsed.first or "",
            last=parsed.last or "",
            full=str(parsed),
        )

    def _parse_title_venue(self, text: str) -> tuple[str, str, str]:
        """Parse title and venue from text after the year.

        Returns (title, venue, pages).
        """
        text = text.strip()
        pages = ""

        # Extract pages
        pages_match = re.search(r"(?:pages?|pp\.?)\s*([\d]+\s*[-–]\s*[\d]+)", text, re.IGNORECASE)
        if pages_match:
            pages = pages_match.group(1)

        # Look for venue pattern "In ..."
        venue_match = self.VENUE_PATTERN.search(text)
        if venue_match:
            venue_start = venue_match.start()
            title = text[:venue_start].strip().rstrip(",").rstrip(".")
            venue = text[venue_start:].strip()
            # Clean up venue
            venue = re.sub(r"^[Ii]n\s+", "", venue).strip().rstrip(".")
            venue = re.sub(r",\s*pages?\s*[\d–-]+", "", venue).strip().rstrip(".")
            venue = re.sub(r",\s*pp\.?\s*[\d–-]+", "", venue).strip().rstrip(".")
            return title, venue, pages

        # No venue found — split on periods, first part is title
        parts = text.split(". ")
        if parts:
            title = parts[0].strip()
            venue = ". ".join(parts[1:]).strip().rstrip(".") if len(parts) > 1 else ""
            return title, venue, pages

        return text, "", pages
