"""Base async API client with rate limiting and retry."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import aiohttp
from aiolimiter import AsyncLimiter

from halref.models import APIMatch, Reference

logger = logging.getLogger(__name__)


class BaseAPIClient(ABC):
    """Base class for academic paper API clients."""

    name: str = "base"
    source: str = "base"

    def __init__(self, requests_per_second: float = 1.0, api_key: str = ""):
        # AsyncLimiter(max_rate, time_period)
        # For rates < 1 RPS, use longer time periods (e.g., 0.5 RPS = 1 req per 2 sec)
        if requests_per_second >= 1.0:
            self._rate_limiter = AsyncLimiter(requests_per_second, 1)
        else:
            self._rate_limiter = AsyncLimiter(1, 1.0 / requests_per_second)
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=self._default_headers(),
            )
        return self._session

    def _default_headers(self) -> dict[str, str]:
        return {
            "User-Agent": "halref/0.1.0 (hallucinated-reference-finder)",
            "Accept-Encoding": "gzip, deflate",
        }

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        url: str,
        params: dict | None = None,
        max_retries: int = 3,
    ) -> dict | list | None:
        """Make a rate-limited GET request with retry on 429 and transient errors."""
        session = await self._get_session()

        for attempt in range(max_retries):
            # Wait for rate limiter before each attempt
            async with self._rate_limiter:
                try:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 200:
                            return await resp.json()

                        if resp.status == 429:
                            retry_after = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                            logger.info(f"{self.name}: rate limited (429), waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue

                        if resp.status in (500, 502, 503, 504):
                            # Server error — retry with backoff
                            wait = 2 ** (attempt + 1)
                            logger.info(f"{self.name}: server error {resp.status}, retrying in {wait}s")
                            await asyncio.sleep(wait)
                            continue

                        # Client errors (400, 401, 403, 404) — don't retry
                        logger.debug(f"{self.name}: HTTP {resp.status} for {url}")
                        return None

                except asyncio.TimeoutError:
                    wait = 2 ** attempt
                    logger.info(f"{self.name}: timeout, retrying in {wait}s")
                    await asyncio.sleep(wait)
                except aiohttp.ClientError as e:
                    logger.warning(f"{self.name}: request failed ({e})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)

        logger.warning(f"{self.name}: all {max_retries} attempts failed for {url}")
        return None

    @abstractmethod
    async def search(self, reference: Reference) -> list[APIMatch]:
        """Search for a reference and return potential matches."""
        ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
