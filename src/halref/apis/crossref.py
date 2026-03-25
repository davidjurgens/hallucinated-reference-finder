"""CrossRef API client with native fuzzy matching."""

from __future__ import annotations

from halref.apis.base import BaseAPIClient
from halref.models import APIMatch, APISource, Author, Reference


class CrossRefClient(BaseAPIClient):
    name = "crossref"
    source = APISource.CROSSREF

    BASE_URL = "https://api.crossref.org"

    def __init__(self, mailto: str = ""):
        # CrossRef polite pool (with mailto): ~50 RPS
        # Without mailto: shared pool, be conservative
        rps = 5.0 if mailto else 1.0
        super().__init__(requests_per_second=rps)
        self._mailto = mailto

    def _default_headers(self) -> dict[str, str]:
        headers = super()._default_headers()
        if self._mailto:
            headers["User-Agent"] = f"halref/0.1.0 (mailto:{self._mailto})"
        return headers

    async def search(self, reference: Reference) -> list[APIMatch]:
        """Search using query.bibliographic for fuzzy matching."""
        if not reference.title:
            return []

        params: dict = {
            "query.bibliographic": reference.title[:500],
            "rows": 3,
        }
        if self._mailto:
            params["mailto"] = self._mailto

        # Add author filter if available
        if reference.authors:
            first_author = reference.first_author_last()
            if first_author:
                params["query.author"] = first_author

        data = await self._request(f"{self.BASE_URL}/works", params=params)

        if not data or "message" not in data:
            return []

        items = data["message"].get("items", [])
        return [self._parse_item(item) for item in items[:3]]

    def _parse_item(self, item: dict) -> APIMatch:
        """Parse a CrossRef work item into APIMatch."""
        authors = []
        for a in item.get("author", []):
            authors.append(Author(
                first=a.get("given", ""),
                last=a.get("family", ""),
                full=f"{a.get('given', '')} {a.get('family', '')}".strip(),
            ))

        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""

        venue_list = item.get("container-title", [])
        venue = venue_list[0] if venue_list else ""

        year = None
        for date_field in ("published", "issued", "created"):
            date_parts = item.get(date_field, {}).get("date-parts", [[]])
            if date_parts and date_parts[0] and date_parts[0][0]:
                year = date_parts[0][0]
                break

        return APIMatch(
            source=APISource.CROSSREF,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=item.get("DOI", ""),
            url=item.get("URL", ""),
            confidence=0.0,
            raw_response=item,
        )
