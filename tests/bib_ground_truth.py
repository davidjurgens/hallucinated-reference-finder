"""Parse ground truth .bib and .bbl files into a common structure for comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GroundTruthRef:
    """A reference from the ground truth bib/bbl file."""

    title: str
    year: int | None
    first_author: str
    raw_key: str = ""


def strip_latex(text: str) -> str:
    """Remove LaTeX markup from a string."""
    # Remove \command{...} but keep content
    text = re.sub(r"\\texttt\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", text)
    # Common accent commands
    text = re.sub(r"\\'([aeiouncAEIOUNC])", r"\1", text)  # \'e -> e
    text = re.sub(r"\\`([aeiouncAEIOUNC])", r"\1", text)
    text = re.sub(r'\\"([aeiouncAEIOUNC])', r"\1", text)
    text = re.sub(r"\\~([nN])", r"\1", text)
    text = re.sub(r"\\c\{([cC])\}", r"\1", text)
    # Remove remaining braces and backslashes
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\\[a-zA-Z]+\s*", "", text)  # Remove remaining commands
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_bib_file(path: Path) -> list[GroundTruthRef]:
    """Parse a .bib file into ground truth entries."""
    text = path.read_text(encoding="utf-8", errors="replace")
    entries = []

    # Split on @ entries (but not @string, @comment, @preamble)
    for block in re.split(r"\n(?=@(?!string|comment|preamble))", text, flags=re.IGNORECASE):
        # Extract key
        key_m = re.match(r"@\w+\{([^,]+),", block)
        key = key_m.group(1).strip() if key_m else ""

        # Extract title
        title_m = re.search(
            r"title\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]+)\")",
            block,
            re.DOTALL | re.IGNORECASE,
        )
        if not title_m:
            continue
        title = strip_latex(title_m.group(1) or title_m.group(2) or "")
        if len(title) < 5:
            continue

        # Extract year
        year_m = re.search(r"year\s*=\s*[{\"]?(\d{4})[}\"]?", block, re.IGNORECASE)
        year = int(year_m.group(1)) if year_m else None

        # Extract first author
        author_m = re.search(
            r"author\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]+)\")",
            block,
            re.DOTALL | re.IGNORECASE,
        )
        first_author = ""
        if author_m:
            raw_authors = strip_latex(author_m.group(1) or author_m.group(2) or "")
            # Get first author's last name
            first_part = re.split(r"\s+and\s+", raw_authors, maxsplit=1)[0].strip()
            if "," in first_part:
                first_author = first_part.split(",")[0].strip()
            else:
                words = first_part.split()
                first_author = words[-1] if words else ""

        entries.append(GroundTruthRef(
            title=title, year=year, first_author=first_author, raw_key=key,
        ))

    return entries


def parse_bbl_file(path: Path) -> list[GroundTruthRef]:
    """Parse a compiled .bbl file into ground truth entries.

    BBL format (natbib) looks like:
    \\bibitem[Author(Year)]{key}
    Author names. Year. Title. Venue...
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    entries = []

    # Split on \bibitem
    items = re.split(r"\\bibitem", text)
    for item in items[1:]:  # Skip preamble before first bibitem
        # Extract key
        key_m = re.search(r"\{([^}]+)\}\s*\n", item)
        key = key_m.group(1).strip() if key_m else ""

        # The reference text follows the key line
        # Remove the bibitem header
        ref_text = re.sub(r"^.*?\}\s*\n", "", item, count=1).strip()

        # Remove \newblock commands
        ref_text = ref_text.replace("\\newblock", "")
        ref_text = strip_latex(ref_text)
        ref_text = re.sub(r"\s+", " ", ref_text).strip()

        if len(ref_text) < 20:
            continue

        # Parse: "Authors. Year. Title. Venue..."
        # Try to extract year
        year_m = re.search(r"\b((?:19|20)\d{2})[a-z]?\b", ref_text)
        year = int(year_m.group(1)) if year_m else None

        # Extract title — typically the text after "Year." and before the next period-terminated sentence
        title = ""
        if year_m:
            after_year = ref_text[year_m.end():].strip().lstrip(".")
            # Title is usually the first sentence after the year
            # It ends with a period followed by venue info ("In ...", journal name, etc.)
            title_m = re.match(r"\s*(.+?)(?:\.\s+(?:In\s|Proceedings|Journal|Trans|arXiv|http|doi)|\.\s*$)", after_year, re.DOTALL)
            if title_m:
                title = title_m.group(1).strip().rstrip(".")
            else:
                # Fallback: take first sentence
                parts = after_year.split(". ", 1)
                title = parts[0].strip().rstrip(".") if parts else ""

        if not title or len(title) < 5:
            # Try alternative: look for italic/emphasized text (often title in .bbl)
            em_m = re.search(r"(?:em |it )(.*?)(?:\.|,)", ref_text)
            if em_m:
                title = em_m.group(1).strip()

        # Extract first author
        first_author = ""
        if year_m:
            before_year = ref_text[:year_m.start()].strip().rstrip(".")
        else:
            before_year = ref_text.split(".")[0] if "." in ref_text else ref_text[:50]

        # Get first author last name
        parts = re.split(r",|\s+and\s+", before_year, maxsplit=1)
        if parts:
            words = parts[0].strip().split()
            first_author = words[-1] if words else ""

        if title and len(title) >= 5:
            entries.append(GroundTruthRef(
                title=title, year=year, first_author=first_author, raw_key=key,
            ))

    return entries


def load_ground_truth(source_dir: Path) -> list[GroundTruthRef]:
    """Load ground truth from .bib or .bbl files in the directory.

    Strategy:
    1. Try .bib files first (more structured, easier to parse accurately)
    2. If .bib has < 10 entries, supplement with .bbl (which has all compiled refs)
    3. Skip anthology.bib (too large, not paper-specific)
    """
    bib_entries = []
    bbl_entries = []

    # Parse .bib files (excluding anthology.bib)
    for bib in sorted(source_dir.glob("*.bib")):
        if "anthology" in bib.name.lower():
            continue
        parsed = parse_bib_file(bib)
        if parsed:
            bib_entries.extend(parsed)

    # Parse .bbl files
    for bbl in sorted(source_dir.glob("*.bbl")):
        parsed = parse_bbl_file(bbl)
        if parsed:
            bbl_entries.extend(parsed)

    # Use .bib if it has enough entries (>= 10), otherwise use .bbl
    if len(bib_entries) >= 10:
        return bib_entries
    elif bbl_entries:
        return bbl_entries
    else:
        return bib_entries  # Return whatever we have
