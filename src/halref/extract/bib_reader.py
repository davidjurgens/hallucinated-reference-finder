"""Read .bib files into Reference objects for verification."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from halref.models import Author, Reference

logger = logging.getLogger(__name__)


def read_bib(bib_path: Path) -> list[Reference]:
    """Parse a .bib file into a list of Reference objects.

    Handles standard BibTeX format with @type{key, field = {value}, ...}.
    """
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    entries = []

    # Split on @ entries (skip @string, @comment, @preamble)
    for i, block in enumerate(re.split(r"\n(?=@(?!string|comment|preamble))", text, flags=re.IGNORECASE)):
        ref = _parse_bib_entry(block)
        if ref:
            ref.source_index = len(entries) + 1
            entries.append(ref)

    logger.info(f"Read {len(entries)} references from {bib_path.name}")
    return entries


def _parse_bib_entry(block: str) -> Reference | None:
    """Parse a single BibTeX entry block into a Reference."""
    # Extract entry type and key
    header = re.match(r"@(\w+)\{([^,]*),", block)
    if not header:
        return None

    entry_type = header.group(1).lower()
    cite_key = header.group(2).strip()

    # Skip non-reference entries
    if entry_type in ("string", "comment", "preamble"):
        return None

    # Extract fields
    title = _extract_field(block, "title")
    year_str = _extract_field(block, "year")
    author_str = _extract_field(block, "author")
    venue = _extract_field(block, "booktitle") or _extract_field(block, "journal")
    doi = _extract_field(block, "doi")
    url = _extract_field(block, "url")
    pages = _extract_field(block, "pages")

    if not title:
        return None

    # Clean LaTeX markup from all fields
    title = _strip_latex(title)
    if venue:
        venue = _strip_latex(venue)

    # Parse year
    year = None
    if year_str:
        year_match = re.search(r"(\d{4})", year_str)
        if year_match:
            year = int(year_match.group(1))

    # Parse authors
    authors = _parse_authors(author_str) if author_str else []

    # Build raw text for display/repair
    parts = []
    if author_str:
        parts.append(_strip_latex(author_str).replace(" and ", ", "))
    if year:
        parts.append(str(year))
    if title:
        parts.append(title)
    if venue:
        parts.append(venue)
    raw_text = ". ".join(parts)

    ref = Reference(
        raw_text=raw_text,
        title=title,
        authors=authors,
        year=year,
        venue=venue or "",
        doi=doi or "",
        url=url or "",
        pages=pages or "",
        extraction_confidence=1.0,  # Bib entries are pre-structured, high confidence
    )
    return ref


def _extract_field(block: str, field_name: str) -> str | None:
    """Extract a field value from a BibTeX entry.

    Handles both {value} and "value" delimiters, including nested braces.
    """
    # Match field = {value} or field = "value"
    pattern = rf"{field_name}\s*=\s*"
    match = re.search(pattern, block, re.IGNORECASE)
    if not match:
        return None

    rest = block[match.end():].lstrip()

    if rest.startswith("{"):
        # Find matching closing brace (handling nesting)
        depth = 0
        for i, ch in enumerate(rest):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return rest[1:i].strip()
    elif rest.startswith('"'):
        # Find closing quote
        end = rest.find('"', 1)
        if end > 0:
            return rest[1:end].strip()
    else:
        # Bare value (e.g., year = 2023)
        end = rest.find(",")
        if end < 0:
            end = rest.find("}")
        if end > 0:
            return rest[:end].strip().strip('"').strip("{").strip("}")

    return None


def _strip_latex(text: str) -> str:
    """Remove LaTeX markup from a string."""
    # Remove \command{content} but keep content
    text = re.sub(r"\\(?:texttt|textbf|textit|emph|url)\{([^}]*)\}", r"\1", text)
    # Accent commands
    text = re.sub(r"\\'([aeiouncAEIOUNC])", r"\1", text)
    text = re.sub(r"\\`([aeiouncAEIOUNC])", r"\1", text)
    text = re.sub(r'\\"([aeiouncAEIOUNC])', r"\1", text)
    text = re.sub(r"\\~([nN])", r"\1", text)
    text = re.sub(r"\\c\{([cC])\}", r"\1", text)
    text = re.sub(r"\\L", "L", text)
    text = re.sub(r"\\l", "l", text)
    # Remove remaining braces and unknown commands
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\\[a-zA-Z]+\s*", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_authors(author_str: str) -> list[Author]:
    """Parse a BibTeX author string into Author objects.

    BibTeX uses "and" to separate authors, with names in either
    "First Last" or "Last, First" format.
    """
    if not author_str:
        return []

    author_str = _strip_latex(author_str)
    parts = re.split(r"\s+and\s+", author_str)

    authors = []
    for part in parts:
        part = part.strip()
        if not part or part.lower() in ("others", "et al", "et al."):
            continue
        # Skip numeric artifacts
        if re.match(r"^\d+\s", part):
            continue

        if "," in part:
            # "Last, First" format
            sub = part.split(",", 1)
            last = sub[0].strip()
            first = sub[1].strip() if len(sub) > 1 else ""
        else:
            # "First Last" format
            words = part.split()
            if len(words) >= 2:
                first = " ".join(words[:-1])
                last = words[-1]
            else:
                first = ""
                last = part

        authors.append(Author(
            first=first,
            last=last,
            full=f"{first} {last}".strip(),
        ))

    return authors
