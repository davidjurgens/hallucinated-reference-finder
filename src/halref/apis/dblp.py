"""DBLP API client."""

from __future__ import annotations

import re

from halref.apis.base import BaseAPIClient
from halref.models import APIMatch, APISource, Author, Reference


class DBLPClient(BaseAPIClient):
    name = "dblp"
    source = APISource.DBLP

    BASE_URL = "https://dblp.org/search/publ/api"

    def __init__(self):
        super().__init__(requests_per_second=2.0)

    async def search(self, reference: Reference) -> list[APIMatch]:
        """Search DBLP by title."""
        if not reference.title:
            return []

        data = await self._request(
            self.BASE_URL,
            params={
                "q": reference.title[:300],
                "format": "json",
                "h": 3,
            },
        )

        if not data:
            return []

        result = data.get("result", {})
        hits = result.get("hits", {}).get("hit", [])
        if not hits:
            return []

        return [self._parse_hit(h) for h in hits[:3]]

    def _parse_hit(self, hit: dict) -> APIMatch:
        """Parse a DBLP hit into APIMatch."""
        info = hit.get("info", {})

        # Parse authors
        authors = []
        authors_data = info.get("authors", {}).get("author", [])
        if isinstance(authors_data, dict):
            authors_data = [authors_data]
        for a in authors_data:
            name = a.get("text", "") if isinstance(a, dict) else str(a)
            parts = name.rsplit(" ", 1)
            authors.append(Author(
                first=parts[0] if len(parts) > 1 else "",
                last=parts[-1],
                full=name,
            ))

        title = info.get("title", "").rstrip(".")

        # Extract year
        year_str = info.get("year", "")
        year = int(year_str) if year_str and re.match(r"^\d{4}$", year_str) else None

        venue = info.get("venue", "")
        doi = info.get("doi", "")
        url = info.get("url", "")

        return APIMatch(
            source=APISource.DBLP,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            url=url,
            confidence=0.0,
            raw_response=info,
        )
