"""Rule-based agentic retry strategies for reference matching."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from halref.apis.base import BaseAPIClient
from halref.matching.title_matcher import title_similarity
from halref.models import APIMatch, Reference

logger = logging.getLogger(__name__)

# If any API returns a match with title similarity above this, stop querying more APIs
CONFIDENT_API_THRESHOLD = 0.90

# If any strategy produces a match above this, stop trying more strategies
CONFIDENT_STRATEGY_THRESHOLD = 0.95


class SearchStrategy(ABC):
    """A single strategy for finding a reference in APIs."""

    name: str = "base"

    @abstractmethod
    async def execute(
        self,
        reference: Reference,
        clients: list[BaseAPIClient],
    ) -> list[APIMatch]:
        ...


async def _waterfall_search(
    reference: Reference,
    clients: list[BaseAPIClient],
    ref_override: Reference | None = None,
    filter_fn=None,
) -> list[APIMatch]:
    """Query APIs in priority order, stop early when a confident match is found.

    Args:
        reference: The original reference (for similarity comparison).
        clients: API clients in priority order.
        ref_override: If set, use this for the API query instead of reference.
        filter_fn: Optional function(reference, match) -> bool to filter results.
    """
    query_ref = ref_override or reference
    all_matches: list[APIMatch] = []

    for client in clients:
        try:
            results = await client.search(query_ref)
            if filter_fn:
                results = [m for m in results if filter_fn(reference, m)]
            if results:
                all_matches.extend(results)
                # Check if we have a confident match — stop querying more APIs
                best_sim = max(
                    title_similarity(reference.title, m.title) for m in results
                )
                if best_sim >= CONFIDENT_API_THRESHOLD:
                    logger.debug(
                        f"Confident match from {client.name} (sim={best_sim:.2f}), "
                        f"skipping remaining APIs"
                    )
                    break
        except Exception as e:
            logger.debug(f"{client.name} failed: {e}")

    return all_matches


class DirectTitleSearch(SearchStrategy):
    """Search APIs with the full title as-is."""

    name = "direct_title"

    async def execute(self, reference: Reference, clients: list[BaseAPIClient]) -> list[APIMatch]:
        return await _waterfall_search(reference, clients)


class RemoveSubtitle(SearchStrategy):
    """Remove subtitle (text after colon or dash) and retry."""

    name = "remove_subtitle"

    async def execute(self, reference: Reference, clients: list[BaseAPIClient]) -> list[APIMatch]:
        title = reference.title
        short_title = re.split(r"[:\u2014\u2013–-]\s", title, maxsplit=1)[0].strip()
        if short_title == title or len(short_title) < 10:
            return []

        modified = reference.model_copy(update={"title": short_title})
        return await _waterfall_search(reference, clients, ref_override=modified)


class AuthorYearSearch(SearchStrategy):
    """Search by first author last name + year."""

    name = "author_year"

    async def execute(self, reference: Reference, clients: list[BaseAPIClient]) -> list[APIMatch]:
        if not reference.authors or not reference.year:
            return []

        first_last = reference.first_author_last()
        if not first_last:
            return []

        query_ref = reference.model_copy(
            update={"title": f"{first_last} {reference.year}"}
        )

        def title_filter(ref, match):
            return title_similarity(ref.title, match.title) > 0.5

        return await _waterfall_search(
            reference, clients, ref_override=query_ref, filter_fn=title_filter
        )


class PartialTitleSearch(SearchStrategy):
    """Search with the most distinctive part of the title."""

    name = "partial_title"

    STOP_WORDS = {
        "a", "an", "the", "of", "in", "on", "for", "to", "and", "with",
        "is", "are", "by", "from", "as", "at", "or", "not", "that", "this",
        "using", "via", "based", "towards", "through",
    }

    async def execute(self, reference: Reference, clients: list[BaseAPIClient]) -> list[APIMatch]:
        if not reference.title:
            return []

        words = re.findall(r"\b[a-zA-Z]{4,}\b", reference.title.lower())
        distinctive = [w for w in words if w not in self.STOP_WORDS]

        if len(distinctive) < 3:
            return []

        partial_title = " ".join(distinctive[:5])
        modified = reference.model_copy(update={"title": partial_title})

        def title_filter(ref, match):
            return title_similarity(ref.title, match.title) > 0.5

        return await _waterfall_search(
            reference, clients, ref_override=modified, filter_fn=title_filter
        )


class VenueConstrainedSearch(SearchStrategy):
    """Search with title + venue as filter."""

    name = "venue_constrained"

    async def execute(self, reference: Reference, clients: list[BaseAPIClient]) -> list[APIMatch]:
        if not reference.venue or not reference.title:
            return []

        combined = f"{reference.title} {reference.venue}"
        modified = reference.model_copy(update={"title": combined})
        return await _waterfall_search(reference, clients, ref_override=modified)


class VerificationAgent:
    """Runs search strategies as a waterfall — stops when confident."""

    def __init__(self, max_retries: int = 2):
        self.strategies: list[SearchStrategy] = [
            DirectTitleSearch(),
            RemoveSubtitle(),
            AuthorYearSearch(),
            PartialTitleSearch(),
            VenueConstrainedSearch(),
        ]
        self.max_retries = max_retries

    async def verify(
        self,
        reference: Reference,
        clients: list[BaseAPIClient],
    ) -> tuple[list[APIMatch], list[str]]:
        """Run strategies until a confident match is found.

        Two levels of waterfall:
        1. Within each strategy: APIs are queried in priority order, stopping
           when a confident match is found (CONFIDENT_API_THRESHOLD).
        2. Across strategies: stop trying more strategies when any produces
           a highly confident match (CONFIDENT_STRATEGY_THRESHOLD).

        Returns:
            Tuple of (all_matches, strategies_used).
        """
        all_matches: list[APIMatch] = []
        strategies_used: list[str] = []
        consecutive_failures = 0

        for strategy in self.strategies:
            if consecutive_failures >= self.max_retries and strategies_used:
                break

            try:
                matches = await strategy.execute(reference, clients)
                if matches:
                    all_matches.extend(matches)
                    strategies_used.append(strategy.name)
                    consecutive_failures = 0

                    # Check if we have a highly confident match across all results
                    best_sim = max(
                        title_similarity(reference.title, m.title) for m in all_matches
                    )
                    if best_sim >= CONFIDENT_STRATEGY_THRESHOLD:
                        logger.debug(
                            f"Confident strategy match (sim={best_sim:.2f}), "
                            f"stopping after {strategy.name}"
                        )
                        break
                else:
                    consecutive_failures += 1
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")
                consecutive_failures += 1

        return all_matches, strategies_used
