"""API-assisted field parser using CrossRef to validate/enrich parses."""

from __future__ import annotations

import asyncio

import aiohttp

from halref.extract.base import FieldParser
from halref.models import Author, Reference


class APIFieldParser(FieldParser):
    """Validate/enrich reference parsing by querying CrossRef.

    Sends the raw reference string to CrossRef's query.bibliographic endpoint.
    If a good match is found, uses the API's structured metadata.
    """

    name = "api"

    def __init__(self, mailto: str = ""):
        self.mailto = mailto

    def parse(self, raw_text: str) -> Reference:
        """Synchronous wrapper around async parse."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're already in an async context — can't use asyncio.run
            # Return minimal reference; the async version should be used instead
            return Reference(raw_text=raw_text, extraction_confidence=0.0)

        return asyncio.run(self.aparse(raw_text))

    async def aparse(self, raw_text: str) -> Reference:
        """Query CrossRef with the raw reference string."""
        params = {
            "query.bibliographic": raw_text[:500],  # CrossRef has query length limits
            "rows": 1,
        }
        if self.mailto:
            params["mailto"] = self.mailto

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.crossref.org/works",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        return Reference(raw_text=raw_text, extraction_confidence=0.0)
                    data = await resp.json()

            items = data.get("message", {}).get("items", [])
            if not items:
                return Reference(raw_text=raw_text, extraction_confidence=0.0)

            item = items[0]
            return self._build_reference(raw_text, item)

        except Exception:
            return Reference(raw_text=raw_text, extraction_confidence=0.0)

    def _build_reference(self, raw_text: str, item: dict) -> Reference:
        """Build Reference from CrossRef work item."""
        authors = []
        for a in item.get("author", []):
            authors.append(Author(
                first=a.get("given", ""),
                last=a.get("family", ""),
                full=f"{a.get('given', '')} {a.get('family', '')}".strip(),
            ))

        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""

        # Get venue from container-title
        venue_list = item.get("container-title", [])
        venue = venue_list[0] if venue_list else ""

        # Get year from published or issued
        year = None
        for date_field in ("published", "issued", "created"):
            date_parts = item.get(date_field, {}).get("date-parts", [[]])
            if date_parts and date_parts[0] and date_parts[0][0]:
                year = date_parts[0][0]
                break

        doi = item.get("DOI", "")
        page = item.get("page", "")

        ref = Reference(
            raw_text=raw_text,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            pages=page,
        )
        ref.extraction_confidence = self.parse_confidence(ref)
        return ref
