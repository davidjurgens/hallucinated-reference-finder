"""Write Reference objects to BibTeX format."""

from __future__ import annotations

import re
from pathlib import Path

from halref.models import Reference


def write_bib(references: list[Reference], output_path: Path, quiet: bool = False) -> None:
    """Write references to a .bib file."""
    entries = []
    for ref in references:
        entry = reference_to_bibtex(ref)
        if entry:
            entries.append(entry)

    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(entries))
        f.write("\n")

    if not quiet:
        from rich.console import Console
        Console(stderr=True).print(
            f"Wrote {len(entries)} references to {output_path.resolve()}"
        )


def reference_to_bibtex(ref: Reference) -> str:
    """Convert a Reference to a BibTeX entry string."""
    # Generate a citation key
    key = _make_cite_key(ref)

    # Determine entry type
    entry_type = _guess_entry_type(ref)

    fields = []
    if ref.title:
        fields.append(f"  title = {{{ref.title}}}")
    if ref.authors:
        author_str = " and ".join(str(a) for a in ref.authors)
        fields.append(f"  author = {{{author_str}}}")
    if ref.year:
        fields.append(f"  year = {{{ref.year}}}")
    if ref.venue:
        if entry_type == "article":
            fields.append(f"  journal = {{{ref.venue}}}")
        else:
            fields.append(f"  booktitle = {{{ref.venue}}}")
    if ref.pages:
        fields.append(f"  pages = {{{ref.pages}}}")
    if ref.doi:
        fields.append(f"  doi = {{{ref.doi}}}")
    if ref.url:
        fields.append(f"  url = {{{ref.url}}}")

    if not fields:
        return ""

    fields_str = ",\n".join(fields)
    return f"@{entry_type}{{{key},\n{fields_str}\n}}"


def _make_cite_key(ref: Reference) -> str:
    """Generate a BibTeX citation key."""
    parts = []

    # First author last name
    if ref.authors:
        last = ref.authors[0].last or ref.authors[0].full.split()[-1] if ref.authors[0].full else "unknown"
        last = re.sub(r"[^a-zA-Z]", "", last).lower()
        parts.append(last)
    else:
        parts.append("unknown")

    # Year
    if ref.year:
        parts.append(str(ref.year))

    # First significant word of title
    if ref.title:
        words = re.findall(r"[a-zA-Z]+", ref.title.lower())
        stop_words = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "with", "is", "are"}
        for w in words:
            if w not in stop_words and len(w) > 2:
                parts.append(w)
                break

    return "-".join(parts) if parts else f"ref-{ref.source_index}"


def _guess_entry_type(ref: Reference) -> str:
    """Guess the BibTeX entry type from venue text."""
    venue_lower = ref.venue.lower() if ref.venue else ""

    if any(kw in venue_lower for kw in ("proceedings", "conference", "workshop", "symposium")):
        return "inproceedings"
    if any(kw in venue_lower for kw in ("journal", "transactions", "review", "letters")):
        return "article"
    if "arxiv" in venue_lower or "preprint" in venue_lower:
        return "misc"
    if any(kw in venue_lower for kw in ("book", "press", "publisher", "edition")):
        return "inbook"
    if "thesis" in venue_lower or "dissertation" in venue_lower:
        return "phdthesis"

    # Default to inproceedings for ACL papers
    return "inproceedings"
