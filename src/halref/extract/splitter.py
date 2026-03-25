"""Split reference section text into individual reference strings."""

from __future__ import annotations

import re


def split_references(text: str) -> list[str]:
    """Split reference section text into individual reference strings.

    Tries multiple strategies and picks the best result by quality.
    """
    if not text.strip():
        return []

    # Pre-process: dehyphenate words broken across lines
    text = dehyphenate(text)

    # Run all strategies and collect results
    strategies = {
        "blank_lines": _split_by_blank_lines(text),
        "author_year": _split_by_author_year_pattern(text),
        "numbers": _split_by_numbers(text),
    }

    # Pick the best strategy by quality score
    best_name = None
    best_refs = []
    best_score = -1

    for name, refs in strategies.items():
        if not refs:
            continue
        score = _quality_score(refs)
        if score > best_score:
            best_score = score
            best_refs = refs
            best_name = name

    if best_refs:
        return _merge_fragments(best_refs)

    # Fallback: return the whole text as one reference
    return [text.strip()] if text.strip() else []


def dehyphenate(text: str) -> str:
    """Remove hyphens from words broken across lines.

    Handles patterns like "represen-\\ntations" -> "representations"
    and "represen- tations" -> "representations"
    """
    # Pattern: word chars, hyphen, optional whitespace, newline, optional whitespace, word chars
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Also handle "word- word" on same line (from some extractors)
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)
    return text


def _quality_score(refs: list[str]) -> float:
    """Score a list of reference strings by quality.

    Higher is better. A good reference list has:
    - Most entries contain a publication year (19xx/20xx)
    - Most entries have author-like patterns (Name, Name and Name)
    - Reasonable length per entry (60-800 chars)
    - Not too many or too few entries
    """
    if not refs or len(refs) < 2:
        return 0.0

    score = 0.0
    year_count = 0
    author_count = 0

    for ref in refs:
        # Has a publication year?
        if re.search(r"\b(?:19|20)\d{2}\b", ref):
            year_count += 1

        # Has author-like patterns? (capitalized names with commas/and)
        if re.search(r"[A-Z][a-z]+.*(?:,|and)\s+[A-Z][a-z]+", ref):
            author_count += 1

        # Reasonable length?
        if 60 <= len(ref) <= 800:
            score += 1.0
        elif 40 <= len(ref) < 60:
            score += 0.3
        elif len(ref) > 800:
            score += 0.5
        else:
            score -= 1.0  # Very short = likely a fragment or table cell

        # Penalty: looks like table/list content, not a reference
        if re.match(r"^\d+\.\s+\w+:", ref):  # "4. Word: ..." pattern
            score -= 2.0
        if re.match(r"^[•\-\*]\s", ref):  # Bullet points
            score -= 2.0
        if "\t" in ref:  # Tab-separated (table data)
            score -= 1.0

    # What fraction have years? (most important signal)
    year_fraction = year_count / len(refs) if refs else 0
    score += year_fraction * len(refs) * 3

    # What fraction have author patterns?
    author_fraction = author_count / len(refs) if refs else 0
    score += author_fraction * len(refs) * 2

    # Penalize if very few have years (likely not references)
    if year_fraction < 0.3:
        score *= 0.2

    # Penalize too many refs (likely over-splitting)
    if len(refs) > 80:
        score *= 0.3

    return score


def _split_by_blank_lines(text: str) -> list[str]:
    """Split on double newlines (blank lines between references).

    This works best with pdfminer.six output which inserts blank lines.
    """
    chunks = re.split(r"\n\s*\n", text)
    refs = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 20:
            continue
        # Join wrapped lines within a reference
        chunk = re.sub(r"\n", " ", chunk)
        # Collapse multiple spaces
        chunk = re.sub(r"  +", " ", chunk)
        refs.append(chunk.strip())
    return refs


def _split_by_author_year_pattern(text: str) -> list[str]:
    """Split by detecting ACL natbib author-year patterns.

    ACL natbib references start with author names:
    - "Tom Brown, Benjamin Mann, ..."
    - "Jacob Devlin, Ming-Wei Chang, ..."
    - "Yinhan Liu, Myle Ott, ..."

    Each reference starts at a new line with a capitalized name and
    within that block contains a 4-digit year.
    """
    lines = text.split("\n")
    blocks = []
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Blank line — if we have content, finish this block
            if current_lines:
                blocks.append(" ".join(current_lines))
                current_lines = []
            continue

        # Detect if this line starts a new reference
        is_new_ref = _is_reference_start(stripped)

        if is_new_ref and current_lines:
            blocks.append(" ".join(current_lines))
            current_lines = [stripped]
        else:
            current_lines.append(stripped)

    if current_lines:
        blocks.append(" ".join(current_lines))

    # Filter: keep blocks that contain a year and are long enough
    refs = []
    for block in blocks:
        block = re.sub(r"  +", " ", block).strip()
        if len(block) < 30:
            continue
        if re.search(r"\b(?:19|20)\d{2}\b", block):
            refs.append(block)

    return refs


def _is_reference_start(line: str) -> bool:
    """Detect if a line starts a new ACL-format reference.

    A reference typically starts with:
    - A capitalized name (first or last) followed by more text
    - NOT a continuation word like "In", "Proceedings", "pages", etc.
    """
    # Must start with uppercase letter
    if not line or not line[0].isupper():
        return False

    # Exclude common continuation patterns
    continuation_starts = (
        "In ", "In\xa0", "Proceedings", "Journal", "Transactions",
        "Association", "Conference", "Annual", "Pages", "Volume",
        "Technical", "Chapter",
    )
    if line.startswith(continuation_starts):
        return False

    # Should look like a name: capitalized word followed by more content
    # and should contain a comma within first 60 chars (author list)
    # or look like "Firstname Lastname" pattern
    name_pattern = re.match(
        r"^[A-Z\u00C0-\u024F][a-z\u00C0-\u024F]+[\s,]",
        line,
    )
    if not name_pattern:
        return False

    # Should have a comma in the first 80 chars (typical of author lists)
    # or be "Firstname Lastname." pattern
    head = line[:80]
    if "," in head:
        return True
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+", line):
        return True

    return False


def _split_by_numbers(text: str) -> list[str]:
    """Split by numbered reference markers like [1], [2], etc."""
    # Find [N] patterns anywhere in text (not just line starts)
    pattern = re.compile(r"\[(\d+)\]\s*")
    matches = list(pattern.finditer(text))

    if len(matches) < 2:
        return []

    refs = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 20:
            chunk = re.sub(r"\n", " ", chunk)
            chunk = re.sub(r"  +", " ", chunk)
            refs.append(chunk.strip())

    return refs


def _merge_fragments(refs: list[str]) -> list[str]:
    """Merge obvious continuation fragments back into the previous reference.

    Only merges when strongly confident the chunk is a fragment, not a
    standalone reference. Conservative to avoid merging real references.

    Merges when the next chunk:
    - Starts with lowercase (continuation of a sentence)
    - Starts with venue continuation words like "volume", "pages", "pp."
    - Is purely a venue/location fragment (no year, no author pattern, starts mid-sentence)
    AND the current chunk shows signs of truncation (ends with comma or "and").
    """
    if len(refs) <= 1:
        return refs

    # Patterns at START of a chunk that indicate it's a fragment, not a new reference
    _FRAGMENT_START = re.compile(
        r"^(?:"
        r"[a-z]"                                # starts with lowercase = mid-sentence
        r"|volume\s+\d|pages?\s+\d|pp\.\s"     # pagination info
        r"|Findings\s+of\s|Proceedings\s+of\s"  # venue continuation
        r"|Conference\s+on\s|Association\s+for"  # venue continuation
        r"|Transactions\s+of\s"                 # journal continuation
        r"|Workshop\s+on\s"                     # workshop venue
        r")",
    )

    # Patterns at END of a chunk that indicate it's truncated
    _TRUNCATED_END = re.compile(
        r"(?:"
        r",\s*$"             # ends with comma
        r"|\band\s*$"        # ends with "and"
        r"|\bIn\s*$"         # ends with "In" (start of venue)
        r"|\bIn\s+the\s*$"  # ends with "In the"
        r")",
    )

    merged = []
    i = 0
    while i < len(refs):
        current = refs[i]

        # Look ahead: merge clear fragments
        while i + 1 < len(refs):
            next_ref = refs[i + 1]
            should_merge = False

            # Next chunk starts with a fragment pattern (lowercase, venue, pagination)
            if _FRAGMENT_START.match(next_ref):
                should_merge = True

            # Current ends truncated AND next has no year (venue/page continuation)
            elif _TRUNCATED_END.search(current) \
                    and not re.search(r"\b(?:19|20)\d{2}\b", next_ref):
                should_merge = True

            if should_merge:
                current = current.rstrip() + " " + next_ref.lstrip()
                i += 1
            else:
                break

        merged.append(current)
        i += 1

    return merged
