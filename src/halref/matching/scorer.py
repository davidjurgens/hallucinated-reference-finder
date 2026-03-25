"""Composite hallucination scoring."""

from __future__ import annotations

from halref.config import MatchingWeights
from halref.matching.author_matcher import (
    author_set_overlap,
    check_author_order,
    check_first_author,
)
from halref.matching.metadata_matcher import doi_matches, year_difference, year_matches
from halref.matching.title_matcher import title_similarity
from halref.models import APIMatch, HallucinationSignal, MatchResult, Reference


def score_reference(
    reference: Reference,
    api_matches: list[APIMatch],
    weights: MatchingWeights | None = None,
) -> MatchResult:
    """Score a reference for hallucination likelihood.

    Returns:
        MatchResult with hallucination_score in [0, 1], signals, and match details.
    """
    if weights is None:
        weights = MatchingWeights()

    result = MatchResult(reference=reference, api_matches=api_matches)

    if not api_matches:
        result.hallucination_score = 0.95
        result.signals = [
            HallucinationSignal(
                name="no_match",
                value=1.0,
                weight=1.0,
                description="Not found in any database",
            )
        ]
        return result

    # Find best match by title similarity
    best_match, best_title_sim = _select_best_match(reference, api_matches)
    result.best_match = best_match
    result.title_similarity = best_title_sim

    # Compute individual signals
    signals: list[HallucinationSignal] = []

    # Signal 1: Title mismatch
    title_mismatch = max(0.0, 1.0 - best_title_sim)
    desc = f"Title similarity: {best_title_sim:.0%}"
    if best_title_sim < 0.5:
        desc = f"Title very different (similarity: {best_title_sim:.0%})"
    signals.append(HallucinationSignal(
        name="title_mismatch",
        value=title_mismatch,
        weight=weights.title,
        description=desc,
    ))

    # Signal 2: Author mismatch
    author_overlap_score = author_set_overlap(reference.authors, best_match.authors)
    result.author_overlap = author_overlap_score
    author_mismatch = max(0.0, 1.0 - author_overlap_score)
    desc = f"Author overlap: {author_overlap_score:.0%}"
    if author_overlap_score < 0.5 and reference.authors and best_match.authors:
        desc = f"Authors differ significantly (overlap: {author_overlap_score:.0%})"
    signals.append(HallucinationSignal(
        name="author_mismatch",
        value=author_mismatch,
        weight=weights.authors,
        description=desc,
    ))

    # Signal 3: Author order wrong
    order_ok = check_author_order(reference.authors, best_match.authors)
    first_ok = check_first_author(reference.authors, best_match.authors)
    result.author_order_correct = order_ok
    result.first_author_match = first_ok
    order_value = 0.0
    order_desc = "Author order matches"
    if not order_ok:
        order_value = 0.7
        order_desc = "Author order differs from matched paper"
    if not first_ok:
        order_value = 1.0
        order_desc = "Different first author"
    signals.append(HallucinationSignal(
        name="author_order_wrong",
        value=order_value,
        weight=weights.author_order,
        description=order_desc,
    ))

    # Signal 4: Year mismatch
    year_ok = year_matches(reference.year, best_match.year)
    result.year_match = year_ok
    year_diff = year_difference(reference.year, best_match.year)
    year_value = 0.0
    year_desc = "Year matches"
    if year_diff is not None and year_diff > 0:
        year_value = min(1.0, year_diff / 5.0)  # 5+ years off = max signal
        year_desc = f"Year off by {year_diff}"
    elif year_diff is None:
        year_desc = "Year not available for comparison"
    signals.append(HallucinationSignal(
        name="year_mismatch",
        value=year_value,
        weight=weights.year,
        description=year_desc,
    ))

    # Signal 5: API consensus (how many APIs found it)
    high_conf_matches = len([m for m in api_matches if _match_quality(reference, m) > 0.7])
    consensus_penalty = max(0.0, 1.0 - high_conf_matches / 3.0)
    desc = f"Found in {high_conf_matches} database(s)"
    signals.append(HallucinationSignal(
        name="low_api_consensus",
        value=consensus_penalty,
        weight=weights.consensus,
        description=desc,
    ))

    # DOI match is a strong override signal
    doi_result = doi_matches(reference.doi, best_match.doi)
    if doi_result is True:
        # DOI matches — very strong signal that it's real
        result.hallucination_score = 0.05
        signals.append(HallucinationSignal(
            name="doi_match",
            value=0.0,
            weight=0.0,
            description="DOI matches — confirmed real",
        ))
        result.signals = signals
        return result

    # Compute weighted score
    score = sum(s.value * s.weight for s in signals)
    result.hallucination_score = min(1.0, max(0.0, score))
    result.signals = signals

    return result


def _select_best_match(
    reference: Reference, matches: list[APIMatch]
) -> tuple[APIMatch, float]:
    """Select the best matching API result by title similarity."""
    best_match = matches[0]
    best_sim = 0.0

    for match in matches:
        sim = title_similarity(reference.title, match.title)
        if sim > best_sim:
            best_sim = sim
            best_match = match

    return best_match, best_sim


def _match_quality(reference: Reference, match: APIMatch) -> float:
    """Quick quality estimate for a match."""
    return title_similarity(reference.title, match.title)
