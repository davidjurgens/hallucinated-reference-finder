"""OpenAlex API client."""

from __future__ import annotations

from halref.apis.base import BaseAPIClient
from halref.models import APIMatch, APISource, Author, Reference


class OpenAlexClient(BaseAPIClient):
    name = "openalex"
    source = APISource.OPENALEX

    BASE_URL = "https://api.openalex.org"

    def __init__(self, api_key: str = ""):
        # OpenAlex: generous limits with key, conservative without
        rps = 5.0 if api_key else 1.0
        super().__init__(requests_per_second=rps, api_key=api_key)

    async def search(self, reference: Reference) -> list[APIMatch]:
        """Search OpenAlex by title."""
        if not reference.title:
            return []

        params: dict = {
            "search": reference.title[:300],
            "per_page": 3,
        }
        if self._api_key:
            params["api_key"] = self._api_key

        # Add year filter if available
        if reference.year:
            params["filter"] = f"publication_year:{reference.year}"

        data = await self._request(f"{self.BASE_URL}/works", params=params)

        if not data or "results" not in data:
            return []

        return [self._parse_work(w) for w in data["results"][:3]]

    def _parse_work(self, work: dict) -> APIMatch:
        """Parse an OpenAlex work into APIMatch."""
        authors = []
        for authorship in work.get("authorships", []):
            author_data = authorship.get("author", {})
            name = author_data.get("display_name", "")
            parts = name.rsplit(" ", 1)
            authors.append(Author(
                first=parts[0] if len(parts) > 1 else "",
                last=parts[-1],
                full=name,
            ))

        title = work.get("title", "") or ""

        # Get venue
        primary_location = work.get("primary_location", {}) or {}
        source = primary_location.get("source", {}) or {}
        venue = source.get("display_name", "")

        year = work.get("publication_year")
        doi = (work.get("doi") or "").replace("https://doi.org/", "")

        return APIMatch(
            source=APISource.OPENALEX,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            confidence=0.0,
            raw_response=work,
        )
