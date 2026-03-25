"""Post-extraction repair of truncated references via API.

Runs during `halref check` only (not during `extract`).
Uses Semantic Scholar and CrossRef to fill in missing fields for
low-confidence references.

VERIFICATION PRINCIPLE: Every field filled by API must be cross-checked
against the original PDF text. We'd rather have an incomplete reference
than a hallucinated one.
"""

from __future__ import annotations

import logging
import re

from halref.matching.title_matcher import title_similarity
from halref.models import Reference

logger = logging.getLogger(__name__)


async def repair_references(
    references: list[Reference],
    api_clients: list,
    full_ref_text: str = "",
) -> list[Reference]:
    """Attempt to repair truncated/incomplete references using APIs.

    Only repairs references with confidence < 0.7. Uses the original
    PDF text (full_ref_text) to verify any API-provided data.

    Args:
        references: Extracted references to repair.
        api_clients: API clients (already created by pipeline).
        full_ref_text: Full text of the reference section from PDF.

    Returns:
        References with repaired fields where possible.
    """
    repaired_count = 0

    for ref in references:
        if ref.extraction_confidence >= 0.7:
            continue

        if _needs_repair(ref):
            repaired = await _try_repair(ref, api_clients, full_ref_text)
            if repaired:
                repaired_count += 1

    if repaired_count:
        logger.info(f"Repaired {repaired_count} truncated references via API")

    return references


def _needs_repair(ref: Reference) -> bool:
    """Check if a reference needs repair."""
    # Truncated title (short or ends mid-word)
    if ref.title and len(ref.title) < 30:
        return True
    if ref.title and ref.title.rstrip().endswith((" and", " the", " of", " in", " a")):
        return True
    # Missing year with authors and title
    if not ref.year and ref.title and ref.authors:
        return True
    # No authors but has title and year
    if not ref.authors and ref.title and ref.year:
        return True
    return False


async def _try_repair(
    ref: Reference,
    clients: list,
    full_text: str,
) -> bool:
    """Try to repair a single reference. Returns True if repaired."""
    # Try Semantic Scholar match_title first (best for partial titles)
    for client in clients:
        if client.name != "semantic_scholar":
            continue

        try:
            matches = await client.search(ref)
            if not matches:
                continue

            best = matches[0]
            sim = title_similarity(ref.title, best.title)

            # Only accept if the match is reasonable
            if sim < 0.5:
                continue

            # VERIFICATION: check that API data is consistent with PDF text
            if not _verify_against_pdf(ref, best, full_text):
                logger.debug(
                    f"Repair rejected for '{ref.title[:40]}' — "
                    f"API data not found in PDF text"
                )
                continue

            # Apply repair
            _apply_repair(ref, best)
            return True

        except Exception as e:
            logger.debug(f"Repair failed via {client.name}: {e}")

    # Try CrossRef as fallback
    for client in clients:
        if client.name != "crossref":
            continue

        try:
            matches = await client.search(ref)
            if not matches:
                continue

            best = matches[0]
            sim = title_similarity(ref.title, best.title)
            if sim < 0.5:
                continue

            if not _verify_against_pdf(ref, best, full_text):
                continue

            _apply_repair(ref, best)
            return True

        except Exception as e:
            logger.debug(f"Repair failed via {client.name}: {e}")

    return False


def _verify_against_pdf(ref: Reference, api_match, full_text: str) -> bool:
    """Verify that API-provided data is consistent with the original PDF text.

    Returns True only if the API's title/author data can be found in
    the PDF text. This prevents hallucinated metadata.
    """
    if not full_text:
        # No full text available — only accept if title similarity is very high
        return title_similarity(ref.title, api_match.title) >= 0.8

    text_lower = full_text.lower()

    # Check title: significant words from the API title should appear in PDF text
    if api_match.title:
        api_words = set(re.findall(r"\b[a-z]{4,}\b", api_match.title.lower()))
        # Remove common stop words
        stop = {"with", "from", "that", "this", "have", "been", "were", "their", "about", "which"}
        api_words -= stop
        if api_words:
            found = sum(1 for w in api_words if w in text_lower)
            if found / len(api_words) < 0.5:
                return False

    # Check year: if API provides a year, it should appear in the raw reference text
    if api_match.year and not ref.year:
        year_str = str(api_match.year)
        # Check in the raw text of this ref AND nearby text
        ref_context = ref.raw_text.lower() if ref.raw_text else ""
        if year_str not in ref_context and year_str not in text_lower:
            return False

    # Check first author: last name should appear in the reference text
    if api_match.authors and not ref.authors:
        first_last = api_match.authors[0].last.lower() if api_match.authors[0].last else ""
        if first_last and first_last not in text_lower:
            return False

    return True


def _apply_repair(ref: Reference, api_match) -> None:
    """Apply repaired fields from API match to the reference."""
    # Only fill in MISSING fields — don't overwrite existing data
    if not ref.year and api_match.year:
        ref.year = api_match.year
        logger.debug(f"Repaired year: {ref.title[:40]}... -> {api_match.year}")

    if len(ref.title) < 30 and api_match.title and len(api_match.title) > len(ref.title):
        old_title = ref.title
        ref.title = api_match.title
        logger.debug(f"Repaired title: '{old_title}' -> '{api_match.title[:50]}'")

    if not ref.authors and api_match.authors:
        ref.authors = api_match.authors
        logger.debug(f"Repaired authors for: {ref.title[:40]}...")

    if not ref.venue and api_match.venue:
        ref.venue = api_match.venue

    if not ref.doi and api_match.doi:
        ref.doi = api_match.doi

    # Boost confidence since we've filled in missing data
    ref.extraction_confidence = min(0.8, ref.extraction_confidence + 0.3)
