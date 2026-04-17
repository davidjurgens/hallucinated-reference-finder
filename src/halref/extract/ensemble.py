"""Ensemble extraction: combine multiple text extractors and field parsers."""

from __future__ import annotations

import logging
from pathlib import Path

from halref.config import Config
from halref.extract.base import FieldParser, TextExtractor
from halref.extract.splitter import split_references
from halref.models import Reference

logger = logging.getLogger(__name__)


def get_text_extractors(config: Config) -> list[TextExtractor]:
    """Get available text extractors based on config."""
    extractors: list[TextExtractor] = []

    for name in config.extraction.text_extractors:
        if name == "pdfminer":
            from halref.extract.text_extractors.pdfminer_extractor import PdfminerExtractor
            extractors.append(PdfminerExtractor())
        elif name == "pypdf":
            from halref.extract.text_extractors.pypdf_extractor import PypdfExtractor
            extractors.append(PypdfExtractor())
        elif name == "pdfplumber":
            from halref.extract.text_extractors.pdfplumber_extractor import PdfplumberExtractor
            extractors.append(PdfplumberExtractor())
        elif name == "docling":
            from halref.extract.text_extractors.docling_extractor import DoclingExtractor
            ext = DoclingExtractor()
            if ext.is_available():
                extractors.append(ext)
            else:
                logger.warning("Docling not installed, skipping. pip install docling")
        elif name == "marker":
            from halref.extract.text_extractors.marker_extractor import MarkerExtractor
            ext = MarkerExtractor()
            if ext.is_available():
                extractors.append(ext)
            else:
                logger.warning("Marker not installed, skipping. pip install marker-pdf")

    if not extractors:
        # Default fallback chain
        from halref.extract.text_extractors.pdfminer_extractor import PdfminerExtractor
        extractors.append(PdfminerExtractor())

    return extractors


def get_field_parsers(config: Config) -> list[FieldParser]:
    """Get available field parsers based on config."""
    parsers: list[FieldParser] = []

    for name in config.extraction.field_parsers:
        if name == "regex":
            from halref.extract.field_parsers.regex_parser import RegexFieldParser
            parsers.append(RegexFieldParser())
        elif name == "heuristic":
            from halref.extract.field_parsers.heuristic_parser import HeuristicFieldParser
            parsers.append(HeuristicFieldParser())
        elif name == "llm":
            if config.llm.enabled and config.llm.model:
                from halref.extract.field_parsers.llm_parser import LLMFieldParser
                parser = LLMFieldParser(
                    base_url=config.llm.base_url,
                    model=config.llm.model,
                    api_key=config.llm.api_key,
                )
                if parser.is_available():
                    parsers.append(parser)
        elif name == "api":
            from halref.extract.field_parsers.api_parser import APIFieldParser
            mailto = config.apis.get("crossref", None)
            mailto_str = mailto.mailto if mailto else ""
            parsers.append(APIFieldParser(mailto=mailto_str))

    if not parsers:
        from halref.extract.field_parsers.regex_parser import RegexFieldParser
        from halref.extract.field_parsers.heuristic_parser import HeuristicFieldParser
        parsers = [RegexFieldParser(), HeuristicFieldParser()]

    return parsers


def extract_references(pdf_path: Path, config: Config) -> list[Reference]:
    """Extract references from a PDF using the multi-pronged ensemble.

    Returns:
        List of Reference objects with fields populated.
    """
    page_range = config.extraction.page_range()
    extractors = get_text_extractors(config)
    parsers = get_field_parsers(config)

    # Layer 1: Get reference text from each extractor
    all_ref_strings: dict[str, list[str]] = {}
    for extractor in extractors:
        try:
            text = extractor.extract_text(pdf_path, page_range=page_range)
            if text.strip():
                refs = split_references(text)
                all_ref_strings[extractor.name] = refs
                logger.info(f"{extractor.name}: found {len(refs)} references")
        except Exception as e:
            logger.warning(f"{extractor.name} failed: {e}")

    # If pypdf wasn't configured, run it as a comparison/fallback extractor.
    # This handles two cases in a single pass:
    #   1. All primary extractors failed (e.g. pdfminer returned garbled text) — pypdf
    #      is the last resort.
    #   2. Primary extractors succeeded but may have column-interleaving artifacts —
    #      the quality score (which penalizes interleaved refs) determines whether to
    #      switch to pypdf's cleaner output.
    if "pypdf" not in all_ref_strings and not any(e.name == "pypdf" for e in extractors):
        logger.debug("pypdf not configured — running as comparison/fallback extractor")
        try:
            from halref.extract.text_extractors.pypdf_extractor import PypdfExtractor
            pypdf_extractor = PypdfExtractor()
            pypdf_text = pypdf_extractor.extract_text(pdf_path, page_range=page_range)
            if pypdf_text.strip():
                pypdf_refs = split_references(pypdf_text)
                if pypdf_refs:
                    all_ref_strings["pypdf"] = pypdf_refs
                    logger.info(f"pypdf: found {len(pypdf_refs)} references")
        except Exception as e:
            logger.warning(f"pypdf comparison/fallback failed: {e}")

    if not all_ref_strings:
        logger.error("No text extractors produced results")
        return []

    # Log per-extractor stats before picking best
    for extractor_name, refs in all_ref_strings.items():
        quality = _ref_list_quality(refs)
        logger.debug(
            f"  {extractor_name}: {len(refs)} refs, quality_score={quality:.1f}"
        )

    # Pick the best extractor result by quality:
    # Prefer the one with the most references that have years (indicates real refs)
    for extractor_name, refs in all_ref_strings.items():
        quality = _ref_list_quality(refs)
        logger.debug(f"  {extractor_name}: {len(refs)} refs, quality_score={quality:.1f}")
    best_extractor = max(
        all_ref_strings.keys(),
        key=lambda k: _ref_list_quality(all_ref_strings[k]),
    )

    ref_strings = all_ref_strings[best_extractor]
    logger.info(f"Using {best_extractor} extraction ({len(ref_strings)} references)")

    # Layer 2: Parse each reference string
    references = []
    filtered_count = 0
    for i, raw_text in enumerate(ref_strings):
        ref = _parse_with_ensemble(raw_text, parsers)
        ref.source_index = i + 1

        # Filter out low-confidence non-references
        if ref.extraction_confidence < 0.3:
            filtered_count += 1
            logger.debug(f"Filtered low-confidence ref [{i+1}]: {raw_text[:80]}...")
            continue

        # Filter out false positives: venue/page fragments parsed as references
        if _is_false_positive(ref):
            filtered_count += 1
            logger.debug(f"Filtered false positive [{i+1}]: {raw_text[:80]}...")
            continue

        references.append(ref)

    if filtered_count:
        logger.info(f"Filtered {filtered_count} low-confidence/false-positive entries")

    return references


def _is_false_positive(ref) -> bool:
    """Detect references that are actually venue fragments, page info, or section headings."""
    import re

    title = (ref.title or "").strip()
    raw = (ref.raw_text or "").strip()

    if not title:
        return False

    title_lower = title.lower()

    # Title starts with venue keywords — this is a fragment, not a reference
    venue_starts = (
        "proceedings of", "in proceedings", "conference on", "conference of",
        "association for", "transactions of", "journal of", "advances in",
        "workshop on", "findings of", "annual meeting",
        "first ", "second ", "third ", "fourth ", "fifth ",  # "First BabyLM Workshop..."
    )
    if any(title_lower.startswith(v) for v in venue_starts):
        return True

    # Title CONTAINS venue keywords AND is short (fragment, not a real title)
    venue_keywords = ("association for computational linguistics", "workshop", "proceedings")
    if len(title) < 80 and any(kw in title_lower for kw in venue_keywords):
        # But not if it looks like a real paper title about these topics
        if not re.search(r"\b(?:19|20)\d{2}\b", raw[:60]):  # No year in first part
            return True

    # Title is mostly page/volume info
    if re.match(r"^[,)\s]*pages?\s+\d", title_lower):
        return True
    if re.match(r"^[,)\s]*volume\s+\d", title_lower):
        return True
    if re.match(r"^\d+[–\-]\d+[,\s]", title):  # "8192–8212, Suzhou, China"
        return True

    # Title is just a location (city, country)
    if re.match(r"^[A-Z][a-z]+,\s+[A-Z][a-z]+\.?\s*$", title):
        return True

    # Title contains only author names and no actual title content
    if not ref.year and len(title) < 60 and title.endswith(","):
        return True

    # Section heading parsed as reference (e.g., "A.1 Claim Dynamics and Structural Roles")
    if re.match(r"^[A-Z]\.\d+\s+", title):
        return True
    if re.match(r"^[A-Z]\s+[A-Z][a-z]+\s+[A-Z][a-z]+", title) and not ref.year:
        return True

    # Raw text is a journal issue reference (e.g., "Processing Systems, volume 31, pages 5598–5609")
    if re.match(r"^[A-Z].*,\s*(?:volume|pages?|pp\.)\s+\d", raw) and not ref.year:
        return True

    # Very short with no year — likely a fragment
    if len(raw) < 50 and not ref.year:
        return True

    return False


def _ref_list_quality(refs: list[str]) -> float:
    """Score a list of reference strings — higher is better."""
    import re
    if not refs:
        return 0.0
    year_count = sum(1 for r in refs if re.search(r"\b(?:19|20)\d{2}\b", r))
    good_length = sum(1 for r in refs if 40 <= len(r) <= 800)
    # Penalize column-interleaving artifacts: refs where a word runs directly
    # into "In Proceedings" with no space (e.g. "resoIn Proceedings of lution:")
    interleaved = sum(1 for r in refs if re.search(r"[a-z]In\s+(?:Proceedings|Pro\b)", r))
    return year_count * 2 + good_length - interleaved * 5


def _parse_with_ensemble(raw_text: str, parsers: list[FieldParser]) -> Reference:
    """Parse a single reference string using multiple parsers, pick best result."""
    candidates: list[Reference] = []

    for parser in parsers:
        try:
            ref = parser.parse(raw_text)
            ref.extraction_confidence = parser.parse_confidence(ref)
            candidates.append(ref)

            if ref.extraction_confidence >= 0.9:
                return ref
        except Exception as e:
            logger.debug(f"{parser.name} parser failed: {e}")

    if not candidates:
        return Reference(raw_text=raw_text, extraction_confidence=0.0)

    best = max(candidates, key=lambda r: r.extraction_confidence)

    if best.extraction_confidence < 0.7 and len(candidates) > 1:
        best = _merge_candidates(candidates, raw_text)

    return best


def _merge_candidates(candidates: list[Reference], raw_text: str) -> Reference:
    """Merge fields from multiple parse candidates."""
    merged = Reference(raw_text=raw_text)

    titles = [(c.title, c.extraction_confidence) for c in candidates if c.title]
    if titles:
        merged.title = max(titles, key=lambda t: (t[1], len(t[0])))[0]

    author_candidates = [(c.authors, c.extraction_confidence) for c in candidates if c.authors]
    if author_candidates:
        merged.authors = max(author_candidates, key=lambda a: (a[1], len(a[0])))[0]

    year_candidates = [(c.year, c.extraction_confidence) for c in candidates if c.year]
    if year_candidates:
        merged.year = max(year_candidates, key=lambda y: y[1])[0]

    venue_candidates = [(c.venue, c.extraction_confidence) for c in candidates if c.venue]
    if venue_candidates:
        merged.venue = max(venue_candidates, key=lambda v: (v[1], len(v[0])))[0]

    for c in candidates:
        if c.doi and not merged.doi:
            merged.doi = c.doi
        if c.url and not merged.url:
            merged.url = c.url

    merged.extraction_confidence = max(c.extraction_confidence for c in candidates)
    return merged
