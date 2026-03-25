"""Semantic Scholar API client."""

from __future__ import annotations

import logging

import aiohttp

from halref.apis.base import BaseAPIClient
from halref.models import APIMatch, APISource, Author, Reference

logger = logging.getLogger(__name__)


class SemanticScholarClient(BaseAPIClient):
    name = "semantic_scholar"
    source = APISource.SEMANTIC_SCHOLAR

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str = ""):
        # S2 rate limits:
        #   With API key: 1 RPS (dedicated), can request higher
        #   Without: ~100 requests per 5 min shared pool (~0.33 RPS)
        rps = 1.0 if api_key else 0.3
        super().__init__(requests_per_second=rps, api_key=api_key)

    def _default_headers(self) -> dict[str, str]:
        headers = super()._default_headers()
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    async def search(self, reference: Reference) -> list[APIMatch]:
        """Search using match_title first, then fall back to keyword search."""
        if not reference.title:
            return []

        # match_title returns the single best match — fast and accurate
        # It returns 404 when no match is found (this is normal, not an error)
        match = await self._match_title(reference.title)
        if match:
            return [match]

        # Fall back to keyword search
        return await self._search_papers(reference.title)

    async def _match_title(self, title: str) -> APIMatch | None:
        """Use the /paper/search/match endpoint.

        This endpoint returns 404 when no match is found, which is
        expected behavior — we handle it gracefully.
        """
        session = await self._get_session()

        async with self._rate_limiter:
            try:
                async with session.get(
                    f"{self.BASE_URL}/paper/search/match",
                    params={
                        "query": title[:300],
                        "fields": "title,authors,year,venue,externalIds",
                    },
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("data", [])
                        if items:
                            return self._parse_paper(items[0])
                    elif resp.status == 404:
                        # No match found — this is normal
                        logger.debug(f"{self.name}: no match for '{title[:50]}'")
                    elif resp.status == 429:
                        logger.info(f"{self.name}: rate limited on match_title")
                    else:
                        logger.debug(f"{self.name}: match_title returned {resp.status}")
            except (aiohttp.ClientError, TimeoutError) as e:
                logger.debug(f"{self.name}: match_title failed ({e})")

        return None

    async def _search_papers(self, title: str) -> list[APIMatch]:
        """Use the /paper/search endpoint (keyword search)."""
        data = await self._request(
            f"{self.BASE_URL}/paper/search",
            params={
                "query": title[:300],
                "fields": "title,authors,year,venue,externalIds",
                "limit": 3,
            },
        )

        if not data or "data" not in data:
            return []

        return [self._parse_paper(p) for p in data["data"][:3]]

    def _parse_paper(self, paper: dict) -> APIMatch:
        """Parse a Semantic Scholar paper object into APIMatch."""
        authors = []
        for a in paper.get("authors", []):
            name = a.get("name", "")
            parts = name.rsplit(" ", 1)
            authors.append(Author(
                first=parts[0] if len(parts) > 1 else "",
                last=parts[-1],
                full=name,
            ))

        external_ids = paper.get("externalIds", {}) or {}
        doi = external_ids.get("DOI", "")

        return APIMatch(
            source=APISource.SEMANTIC_SCHOLAR,
            title=paper.get("title", ""),
            authors=authors,
            year=paper.get("year"),
            venue=paper.get("venue", ""),
            doi=doi,
            confidence=0.0,
            raw_response=paper,
        )
