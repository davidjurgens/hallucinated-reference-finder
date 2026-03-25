"""Optional LLM verification pass for ambiguous matches."""

from __future__ import annotations

import json
import logging

from halref.config import LLMConfig
from halref.models import APIMatch, MatchResult

logger = logging.getLogger(__name__)

VERIFY_PROMPT = """You are a citation verification expert. Compare these two references and determine if they refer to the same paper.

REFERENCE FROM PDF:
Title: {ref_title}
Authors: {ref_authors}
Year: {ref_year}

CANDIDATE MATCH FROM DATABASE:
Title: {match_title}
Authors: {match_authors}
Year: {match_year}
Venue: {match_venue}

Respond with ONLY a JSON object:
{{
  "same_paper": true/false,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation"
}}"""


async def llm_verify_match(
    result: MatchResult,
    llm_config: LLMConfig,
) -> MatchResult:
    """Use an LLM to verify ambiguous matches.

    Only called when hallucination_score is in the ambiguous range (0.3-0.7).
    """
    if not llm_config.enabled or not llm_config.model:
        return result

    if not result.best_match:
        return result

    try:
        import openai
    except ImportError:
        logger.warning("openai package not installed, skipping LLM verification")
        return result

    ref = result.reference
    match = result.best_match

    prompt = VERIFY_PROMPT.format(
        ref_title=ref.title,
        ref_authors=", ".join(str(a) for a in ref.authors),
        ref_year=ref.year or "unknown",
        match_title=match.title,
        match_authors=", ".join(str(a) for a in match.authors),
        match_year=match.year or "unknown",
        match_venue=match.venue,
    )

    try:
        client = openai.OpenAI(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key or "not-needed",
        )

        response = client.chat.completions.create(
            model=llm_config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )

        content = response.choices[0].message.content or ""
        data = _extract_json(content)

        if data.get("same_paper") is True and data.get("confidence", 0) > 0.7:
            # LLM confirms it's the same paper — reduce hallucination score
            result.hallucination_score = max(0.05, result.hallucination_score * 0.3)
            result.strategies_used.append("llm_verified_match")
        elif data.get("same_paper") is False and data.get("confidence", 0) > 0.7:
            # LLM says it's NOT the same — increase hallucination score
            result.hallucination_score = min(0.95, result.hallucination_score * 1.5 + 0.2)
            result.strategies_used.append("llm_rejected_match")

    except Exception as e:
        logger.warning(f"LLM verification failed: {e}")

    return result


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}
