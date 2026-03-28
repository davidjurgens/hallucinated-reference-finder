"""LLM-based field parser using local models via OpenAI-compatible API."""

from __future__ import annotations

import json

from halref.extract.base import FieldParser
from halref.models import Author, Reference


SYSTEM_PROMPT = """You are a citation parser. Given a raw reference string from an academic paper, extract the structured fields. Return ONLY valid JSON with these fields:

{
  "title": "string",
  "authors": [{"first": "string", "last": "string"}],
  "year": integer or null,
  "venue": "string",
  "doi": "string",
  "pages": "string"
}

Rules:
- Parse author names into first and last name components
- For "et al.", include only the named authors
- Year should be a 4-digit integer or null
- Venue is the journal/conference name without "In" or "Proceedings of" prefix
- If a field is not present, use empty string or null for year
- Preserve Unicode characters exactly as written — do not transliterate diacritics (e.g., keep "Schütze" not "Schutze", "Søgaard" not "Sogaard")
- Return ONLY the JSON object, no additional text"""


class LLMFieldParser(FieldParser):
    """Parse references using a local LLM via OpenAI-compatible API."""

    name = "llm"

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def is_available(self) -> bool:
        try:
            import openai  # noqa: F401
            return bool(self.model)
        except ImportError:
            return False

    def parse(self, raw_text: str) -> Reference:
        import openai

        client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "not-needed",
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw_text},
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            content = response.choices[0].message.content or ""
            # Try to extract JSON from the response
            data = self._extract_json(content)
            if not data:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "LLMFieldParser: unparseable JSON response (model=%s): %s",
                    self.model, content[:200]
                )
            return self._build_reference(raw_text, data)

        except Exception:
            # If LLM fails, return a minimally populated reference
            return Reference(raw_text=raw_text, extraction_confidence=0.0)

    async def parse_async(self, raw_text: str) -> Reference:
        """Async version of parse() using AsyncOpenAI."""
        import openai
        client = openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "not-needed",
        )
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw_text},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            content = response.choices[0].message.content or ""
            data = self._extract_json(content)
            if not data:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "LLMFieldParser: unparseable JSON response (model=%s): %s",
                    self.model, content[:200]
                )
            return self._build_reference(raw_text, data)
        except Exception:
            return Reference(raw_text=raw_text, extraction_confidence=0.0)

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        return {}

    def _build_reference(self, raw_text: str, data: dict) -> Reference:
        """Build a Reference from parsed JSON data."""
        authors = []
        for a in data.get("authors", []):
            if isinstance(a, dict):
                first = a.get("first", "")
                last = a.get("last", "")
                authors.append(Author(
                    first=first,
                    last=last,
                    full=f"{first} {last}".strip(),
                ))

        ref = Reference(
            raw_text=raw_text,
            title=data.get("title", ""),
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue", ""),
            doi=data.get("doi", ""),
            pages=data.get("pages", ""),
        )
        ref.extraction_confidence = self.parse_confidence(ref)
        return ref
