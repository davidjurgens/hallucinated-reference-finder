"""Year, venue, and DOI matching."""

from __future__ import annotations

from rapidfuzz import fuzz


def year_matches(year_a: int | None, year_b: int | None, tolerance: int = 1) -> bool:
    """Check if two years match within tolerance."""
    if year_a is None or year_b is None:
        return True  # Can't compare, don't penalize
    return abs(year_a - year_b) <= tolerance


def year_difference(year_a: int | None, year_b: int | None) -> int | None:
    """Return absolute year difference, or None if not comparable."""
    if year_a is None or year_b is None:
        return None
    return abs(year_a - year_b)


def venue_similarity(venue_a: str, venue_b: str) -> float:
    """Compute similarity between two venue names."""
    if not venue_a or not venue_b:
        return 0.0

    a = venue_a.lower().strip()
    b = venue_b.lower().strip()

    if a == b:
        return 1.0

    return fuzz.token_sort_ratio(a, b) / 100.0


def doi_matches(doi_a: str, doi_b: str) -> bool | None:
    """Check if two DOIs match. Returns None if neither has a DOI."""
    if not doi_a or not doi_b:
        return None  # Can't compare
    return doi_a.lower().strip() == doi_b.lower().strip()
