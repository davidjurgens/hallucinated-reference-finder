"""ACL Anthology client using local Python library."""

from __future__ import annotations

import logging

from halref.apis.base import BaseAPIClient
from halref.models import APIMatch, APISource, Author, Reference

logger = logging.getLogger(__name__)


class ACLAnthologyClient(BaseAPIClient):
    """Search ACL Anthology using the local Python library.

    Requires: pip install acl-anthology
    """

    name = "acl_anthology"
    source = APISource.ACL_ANTHOLOGY

    def __init__(self):
        super().__init__(requests_per_second=100.0)  # Local, no rate limit
        self._anthology = None
        self._available = None

    def _is_available(self) -> bool:
        if self._available is None:
            try:
                import acl_anthology  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.info("acl-anthology not installed, skipping. pip install acl-anthology")
        return self._available

    def _get_anthology(self):
        if self._anthology is None:
            from acl_anthology import Anthology
            self._anthology = Anthology.from_repo()
        return self._anthology

    async def search(self, reference: Reference) -> list[APIMatch]:
        """Search ACL Anthology by title."""
        if not reference.title or not self._is_available():
            return []

        try:
            from rapidfuzz import fuzz

            anthology = self._get_anthology()
            query_title = reference.title.lower().strip()

            best_matches = []

            # Search through anthology papers
            for paper in anthology.papers():
                if not paper.title:
                    continue
                paper_title = str(paper.title).lower().strip()
                score = fuzz.token_sort_ratio(query_title, paper_title)
                if score > 70:
                    best_matches.append((score, paper))

            # Sort by score and take top 3
            best_matches.sort(key=lambda x: x[0], reverse=True)

            return [self._parse_paper(paper, score) for score, paper in best_matches[:3]]

        except Exception as e:
            logger.warning(f"ACL Anthology search failed: {e}")
            return []

    def _parse_paper(self, paper, score: float) -> APIMatch:
        """Parse an ACL Anthology paper into APIMatch."""
        authors = []
        try:
            for person in paper.authors:
                name = str(person)
                parts = name.rsplit(" ", 1)
                authors.append(Author(
                    first=parts[0] if len(parts) > 1 else "",
                    last=parts[-1],
                    full=name,
                ))
        except Exception:
            pass

        year = None
        try:
            year = int(paper.year) if paper.year else None
        except (ValueError, TypeError):
            pass

        return APIMatch(
            source=APISource.ACL_ANTHOLOGY,
            title=str(paper.title) if paper.title else "",
            authors=authors,
            year=year,
            venue=str(paper.venue) if hasattr(paper, "venue") and paper.venue else "",
            doi=str(paper.doi) if hasattr(paper, "doi") and paper.doi else "",
            url=paper.url if hasattr(paper, "url") else "",
            confidence=score / 100.0,
            raw_response={},
        )
