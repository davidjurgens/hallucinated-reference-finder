"""Heuristic field parser using punctuation and position-based rules."""

from __future__ import annotations

import re

from nameparser import HumanName

from halref.extract.base import FieldParser
from halref.models import Author, Reference


class HeuristicFieldParser(FieldParser):
    """Parse references using position and punctuation heuristics.

    More flexible than regex parser — handles non-standard orderings
    by finding the year first, then working outward.
    """

    name = "heuristic"

    def parse(self, raw_text: str) -> Reference:
        text = raw_text.strip()
        # Strip leading [N] reference numbers
        text = re.sub(r"^\[\d+\]\s*", "", text)
        # Dehyphenate and normalize whitespace
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        text = re.sub(r"\s+", " ", text)
        ref = Reference(raw_text=raw_text)

        # Step 1: Find all 4-digit year candidates, excluding those in arXiv IDs/URLs
        year_matches = [
            m for m in re.finditer(r"\b((?:19|20)\d{2})[a-z]?\b", text)
            if not self._is_year_in_identifier(text, m)
        ]
        if not year_matches:
            # No year found — try basic period-splitting
            return self._parse_no_year(text, ref)

        # Step 2: Pick the most likely year position
        # In ACL format, year comes early (after authors). Pick the first one
        # that's followed by a period or within the first half of the string.
        year_match = self._pick_best_year(year_matches, text)
        ref.year = int(year_match.group(1))
        year_pos = year_match.start()
        year_end = year_match.end()

        # Step 3: Authors are before the year
        before_year = text[:year_pos].strip()
        # Remove trailing period and whitespace
        before_year = before_year.rstrip(". ")
        ref.authors = self._extract_authors(before_year)

        # Step 4: Title and venue are after the year
        after_year = text[year_end:].strip()
        after_year = after_year.lstrip(". ")
        ref.title, ref.venue = self._extract_title_venue(after_year)

        # Step 5: Extract DOI/URL
        doi_match = re.search(r"\b(10\.\d{4,}/[^\s]+)", text)
        if doi_match:
            ref.doi = doi_match.group(1).rstrip(".")

        url_match = re.search(r"(https?://[^\s]+)", text)
        if url_match:
            ref.url = url_match.group(1).rstrip(".")

        ref.extraction_confidence = self.parse_confidence(ref)
        return ref

    @staticmethod
    def _is_year_in_identifier(text: str, match: re.Match) -> bool:
        """Check if a year match is part of an arXiv ID, URL, or DOI."""
        start = max(0, match.start() - 15)
        before = text[start:match.start()].lower()
        return any(kw in before for kw in ("arxiv:", "arxiv.", "http", "doi", "/"))

    def _pick_best_year(self, matches: list, text: str) -> re.Match:
        """Pick the year match most likely to be the publication year."""
        # Prefer year that's followed by a period (ACL format: "Authors. Year.")
        for m in matches:
            after = text[m.end():m.end() + 3]
            if after.startswith(".") or after.startswith("a.") or after.startswith("b."):
                return m

        # Otherwise, prefer the first year in the first 40% of the text
        cutoff = len(text) * 0.4
        for m in matches:
            if m.start() < cutoff:
                return m

        return matches[0]

    @staticmethod
    def _looks_like_first_last_list(parts: list[str]) -> bool:
        """Check if comma-separated parts look like 'First Last' entries."""
        first_last_count = 0
        for part in parts:
            part = part.strip()
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z]|\s+[a-z]+\s+[A-Z])", part):
                first_last_count += 1
            elif re.match(r"^[A-Z][a-z]+\s+[A-Z]\.", part):
                first_last_count += 1
        return first_last_count >= len(parts) * 0.5

    def _extract_authors(self, text: str) -> list[Author]:
        """Extract authors from text before the year."""
        if not text:
            return []

        # Normalize separators
        text = re.sub(r",?\s+and\s+", " && ", text)
        text = text.replace(";", " && ")

        # Split on "&&" first, then handle comma-separated "First Last" entries
        and_parts = text.split(" && ")
        all_parts = []
        for part in and_parts:
            part = part.strip().strip(",").strip()
            if not part:
                continue
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
            if re.match(r"^\d+\s", part):
                continue

            # Try nameparser
            parsed = HumanName(part)
            if parsed.last:
                authors.append(Author(
                    first=parsed.first or "",
                    last=parsed.last or "",
                    full=str(parsed),
                ))
            elif len(part.split()) >= 2:
                # Fallback: assume "First Last" or "Last, First"
                words = part.split()
                if "," in part:
                    sub = part.split(",", 1)
                    authors.append(Author(
                        first=sub[1].strip() if len(sub) > 1 else "",
                        last=sub[0].strip(),
                        full=part,
                    ))
                else:
                    authors.append(Author(
                        first=" ".join(words[:-1]),
                        last=words[-1],
                        full=part,
                    ))

        return authors

    def _extract_title_venue(self, text: str) -> tuple[str, str]:
        """Extract title and venue from text after the year."""
        if not text:
            return "", ""

        # Look for "In ..." which signals start of venue
        in_match = re.search(r"\.\s+[Ii]n\s+", text)
        if in_match:
            title = text[:in_match.start()].strip()
            venue = text[in_match.end():].strip()
            # Clean up venue
            venue = re.sub(r",\s*pages?\s*[\d–-]+.*$", "", venue, flags=re.IGNORECASE)
            venue = venue.rstrip(".")
            return title, venue

        # No "In" found — split on periods
        # Title is typically the first sentence after the year
        parts = text.split(". ")
        if len(parts) >= 2:
            title = parts[0].strip()
            venue = ". ".join(parts[1:]).strip().rstrip(".")
            return title, venue

        return text.rstrip("."), ""

    def _parse_no_year(self, text: str, ref: Reference) -> Reference:
        """Handle references with no detectable year."""
        parts = text.split(". ")
        if len(parts) >= 3:
            ref.authors = self._extract_authors(parts[0])
            ref.title = parts[1].strip()
            ref.venue = ". ".join(parts[2:]).strip().rstrip(".")
        elif len(parts) == 2:
            ref.title = parts[0].strip()
            ref.venue = parts[1].strip().rstrip(".")
        else:
            ref.title = text
        ref.extraction_confidence = self.parse_confidence(ref)
        return ref
