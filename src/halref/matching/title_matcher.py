"""Fuzzy title comparison using rapidfuzz."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz


def normalize_title(title: str) -> str:
    """Normalize a title for comparison."""
    if not title:
        return ""
    # Unicode normalization
    title = unicodedata.normalize("NFKD", title)
    # Lowercase
    title = title.lower()
    # Remove punctuation except hyphens
    title = re.sub(r"[^\w\s-]", "", title)
    # Collapse whitespace
    title = re.sub(r"\s+", " ", title).strip()
    return title


def title_similarity(title_a: str, title_b: str) -> float:
    """Compute similarity between two titles.

    Uses multiple rapidfuzz metrics and returns the maximum score.

    Returns:
        Score in [0, 1] where 1.0 = identical.
    """
    a = normalize_title(title_a)
    b = normalize_title(title_b)

    if not a or not b:
        return 0.0

    # Exact match after normalization
    if a == b:
        return 1.0

    # Multiple similarity metrics — take the max
    scores = [
        fuzz.token_sort_ratio(a, b) / 100.0,
        fuzz.token_set_ratio(a, b) / 100.0,
        fuzz.ratio(a, b) / 100.0,
    ]

    return max(scores)
