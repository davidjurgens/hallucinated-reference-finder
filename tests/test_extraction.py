"""End-to-end test for reference extraction, batch dedup, and API matching.

Run: python tests/test_extraction.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

FIXTURES = Path(__file__).parent / "fixtures"

# Known references in test_refs.bib (title, year, first_author_last)
REAL_REFS = [
    ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2019, "Devlin"),
    ("Attention is All You Need", 2017, "Vaswani"),
    ("Language Models are Few-Shot Learners", 2020, "Brown"),
    ("RoBERTa: A Robustly Optimized BERT Pretraining Approach", 2019, "Liu"),
    ("Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", 2020, "Raffel"),
    ("Transformers: State-of-the-Art Natural Language Processing", 2020, "Wolf"),
    ("BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension", 2020, "Lewis"),
    ("Distributed Representations of Words and Phrases and their Compositionality", 2013, "Mikolov"),
    ("GloVe: Global Vectors for Word Representation", 2014, "Pennington"),
    ("Deep Contextualized Word Representations", 2018, "Peters"),
]

HALLUCINATED_REFS = [
    ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2019, "Chang"),  # Wrong author order
    ("Adaptive Cross-Lingual Transfer Learning for Low-Resource Sentiment Analysis", 2023, "Zhang"),  # Fictitious
    ("Attention is All You Need", 2019, "Vaswani"),  # Wrong year
    ("Neural Reference Verification with Contrastive Citation Embeddings", 2022, "Park"),  # Fictitious
    ("Recursive Hierarchical Attention Networks for Document-Level Machine Translation", 2020, "Devlin"),  # Fake title
]

passed = 0
failed = 0


def check(condition: bool, label: str, detail: str = "") -> bool:
    """Record a test result."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        detail_str = f" — {detail}" if detail else ""
        print(f"  [FAIL] {label}{detail_str}")
    return condition


# ---------------------------------------------------------------------------
# Single-file extraction tests
# ---------------------------------------------------------------------------

def test_extraction(pdf_name: str, expected_count: int, label: str) -> list:
    """Test extraction on a single PDF."""
    from halref.config import Config
    from halref.extract.ensemble import extract_references

    pdf_path = FIXTURES / pdf_name
    if not pdf_path.exists():
        print(f"  SKIP: {pdf_path} not found")
        return []

    config = Config.default()
    refs = extract_references(pdf_path, config)

    check(
        len(refs) >= expected_count * 0.7,
        f"{label}: extracted {len(refs)}/{expected_count} expected",
        f"got only {len(refs)}",
    )

    for r in refs:
        authors = ", ".join(a.last for a in r.authors[:2]) if r.authors else "?"
        title = r.title[:55] if r.title else "(no title)"
        year = r.year or "?"
        print(f"    [{r.source_index:2d}] {authors} ({year}). {title}")

    return refs


def test_field_accuracy(refs: list, known_refs: list, label: str) -> None:
    """Check extracted fields against known values."""
    from rapidfuzz import fuzz

    matched = 0
    for known_title, known_year, known_author in known_refs:
        best_score = 0
        best_ref = None
        for ref in refs:
            score = fuzz.token_sort_ratio(
                known_title.lower(), (ref.title or "").lower()
            )
            if score > best_score:
                best_score = score
                best_ref = ref

        if best_ref and best_score > 60:
            matched += 1
        else:
            print(f"    MISS: '{known_title[:50]}...' (best_score={best_score:.0f})")

    check(
        matched >= len(known_refs) * 0.7,
        f"{label}: matched {matched}/{len(known_refs)} known references",
    )


# ---------------------------------------------------------------------------
# Fixture-shape-specific extraction tests
# ---------------------------------------------------------------------------

# Known titles in the numbered-refs fixture (IEEE style)
NUMBERED_REF_TITLES = [
    "Attention is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    "Language Models are Few-Shot Learners",
    "Distributed Representations of Words and Phrases and their Compositionality",
    "GloVe: Global Vectors for Word Representation",
    "Deep Contextualized Word Representations",
    "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension",
    "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    "Transformers: State-of-the-Art Natural Language Processing",
]

# Known author last names with non-ASCII characters
UNICODE_AUTHOR_NAMES = [
    "Schütze",   # Hinrich Schütze — umlaut ü
    "Søgaard",   # Anders Søgaard — Danish ø
    "Hajič",     # Jan Hajič — Czech háček
    "Guzmán",    # Francisco Guzmán — Spanish tilde
]

# Known titles in the arXiv-style fixture
ARXIV_REF_TITLES = [
    "LLaMA 2: Open Foundation and Fine-Tuned Chat Models",
    "Mistral 7B",
    "Qwen Technical Report",
    "Training language models to follow instructions with human feedback",
    "Constitutional AI: Harmlessness from AI Feedback",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
]


def test_numbered_refs(pdf_name: str = "test_paper_numbered_refs.pdf") -> list:
    """Test extraction of numbered (IEEE-style) references.

    Numbered refs like '[1] A. Vaswani, ...' are a different code path from
    ACL author-year style. The splitter must recognise the '[N]' prefix as a
    boundary, not confuse it with an inline citation.
    """
    from halref.config import Config
    from halref.extract.ensemble import extract_references
    from rapidfuzz import fuzz

    pdf_path = FIXTURES / pdf_name
    if not pdf_path.exists():
        print(f"  SKIP: {pdf_path} not found (compile with pdflatex first)")
        return []

    config = Config.default()
    refs = extract_references(pdf_path, config)

    check(
        len(refs) >= 8,
        f"numbered refs: extracted {len(refs)} (expected ≥8)",
        f"got only {len(refs)}",
    )

    # Check that we're getting titles not just raw "[1] A. Vaswani..." blobs
    refs_with_titles = [r for r in refs if r.title and len(r.title) > 10]
    check(
        len(refs_with_titles) >= len(refs) * 0.7,
        f"numbered refs: {len(refs_with_titles)}/{len(refs)} have parsed titles",
        f"too many without titles",
    )

    # Spot-check a few known titles
    extracted_titles = [r.title or "" for r in refs]
    matched = sum(
        1 for t in NUMBERED_REF_TITLES
        if any(fuzz.token_sort_ratio(t.lower(), e.lower()) > 70 for e in extracted_titles)
    )
    check(
        matched >= len(NUMBERED_REF_TITLES) * 0.7,
        f"numbered refs: {matched}/{len(NUMBERED_REF_TITLES)} known titles matched",
    )

    # Years should be parseable (not embedded in '[1]' prefix)
    refs_with_years = [r for r in refs if r.year and 1990 <= r.year <= 2030]
    check(
        len(refs_with_years) >= len(refs) * 0.7,
        f"numbered refs: {len(refs_with_years)}/{len(refs)} have valid years",
    )

    return refs


def test_unicode_author_names(pdf_name: str = "test_paper_unicode_authors.pdf") -> list:
    """Test that non-ASCII author names are preserved through extraction.

    Diacritics (ü, ø, č, ã) are frequently stripped by PDF extractors or
    transliterated by parsers. This causes false-negative author matching
    when checking for hallucinated author order.
    """
    from halref.config import Config
    from halref.extract.ensemble import extract_references

    pdf_path = FIXTURES / pdf_name
    if not pdf_path.exists():
        print(f"  SKIP: {pdf_path} not found (compile with pdflatex first)")
        return []

    config = Config.default()
    refs = extract_references(pdf_path, config)

    check(
        len(refs) >= 5,
        f"unicode authors: extracted {len(refs)} refs (expected ≥5)",
    )

    # Collect all raw text and author strings
    all_text = " ".join(
        (r.raw_text or "") + " " + " ".join(a.full or a.last for a in r.authors)
        for r in refs
    )

    preserved = []
    transliterated = []
    for name in UNICODE_AUTHOR_NAMES:
        # Check canonical form preserved
        if name in all_text:
            preserved.append(name)
        else:
            # Check if it was silently ASCII-fied (e.g. "Schutze" not "Schütze")
            import unicodedata
            ascii_form = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode()
            if ascii_form.lower() in all_text.lower():
                transliterated.append(name)

    if preserved:
        print(f"    Unicode preserved: {preserved}")
    if transliterated:
        print(f"    Silently transliterated (diacritics stripped): {transliterated}")

    # At minimum the raw text should contain unicode — if it's all ASCII something
    # went wrong at the PDF extraction layer
    has_unicode_in_raw = any(
        ord(c) > 127 for r in refs for c in (r.raw_text or "")
    )
    check(
        has_unicode_in_raw or len(refs) == 0,
        "unicode authors: raw extracted text contains non-ASCII characters",
        "all text is ASCII — diacritics may have been stripped by extractor",
    )

    return refs


def test_arxiv_style_refs(pdf_name: str = "test_paper_arxiv_style.pdf") -> list:
    """Test extraction of arXiv-style references.

    arXiv refs lack formal venue names and page numbers, use 'et al.' author
    truncation, and often include DOI/URL fields. These patterns are common
    in fast-moving ML papers and stress-test the field parsers.
    """
    from halref.config import Config
    from halref.extract.ensemble import extract_references
    from rapidfuzz import fuzz

    pdf_path = FIXTURES / pdf_name
    if not pdf_path.exists():
        print(f"  SKIP: {pdf_path} not found (compile with pdflatex first)")
        return []

    config = Config.default()
    refs = extract_references(pdf_path, config)

    check(
        len(refs) >= 5,
        f"arxiv style: extracted {len(refs)} refs (expected ≥5)",
    )

    # Title matching — arXiv refs have real titles even without venue/pages
    extracted_titles = [r.title or "" for r in refs]
    matched = sum(
        1 for t in ARXIV_REF_TITLES
        if any(fuzz.token_sort_ratio(t.lower(), e.lower()) > 70 for e in extracted_titles)
    )
    check(
        matched >= len(ARXIV_REF_TITLES) * 0.6,
        f"arxiv style: {matched}/{len(ARXIV_REF_TITLES)} known titles matched",
    )

    # Year extraction — arXiv refs always have a year even without pages
    refs_with_years = [r for r in refs if r.year and 2020 <= r.year <= 2030]
    check(
        len(refs_with_years) >= len(refs) * 0.6,
        f"arxiv style: {len(refs_with_years)}/{len(refs)} have valid recent years",
    )

    # Author extraction with et al. — parsers should capture the named authors
    # and not include the literal string "et al" as an author name
    et_al_as_author = [
        r for r in refs
        if any("et al" in (a.last or "").lower() for a in r.authors)
    ]
    check(
        len(et_al_as_author) == 0,
        f"arxiv style: 'et al.' not parsed as an author name",
        f"found {len(et_al_as_author)} refs with 'et al' as author",
    )

    return refs


# ---------------------------------------------------------------------------
# Batch deduplication tests
# ---------------------------------------------------------------------------

def test_batch_dedup():
    """Test deduplication of references across multiple PDFs."""
    from halref.config import Config
    from halref.extract.ensemble import extract_references
    from halref.pipeline import _deduplicate_references

    config = Config.default()
    pdfs = [
        FIXTURES / "test_paper_real.pdf",
        FIXTURES / "test_paper_hallucinated.pdf",
        FIXTURES / "test_paper_long_refs.pdf",
    ]
    available = [p for p in pdfs if p.exists()]
    if len(available) < 2:
        print("  SKIP: need at least 2 test PDFs for batch dedup test")
        return

    per_file = {}
    for f in available:
        per_file[f] = extract_references(f, config)

    total_raw = sum(len(r) for r in per_file.values())
    deduped = _deduplicate_references(per_file)
    unique_count = len(deduped)
    shared_count = sum(1 for d in deduped if d.paper_count > 1)

    # Dedup should reduce total count
    check(
        unique_count < total_raw,
        f"dedup reduces count: {total_raw} raw -> {unique_count} unique",
        f"no reduction ({unique_count} >= {total_raw})",
    )

    # Should have some shared references (BERT, Attention, etc. are in all 3)
    check(
        shared_count >= 5,
        f"shared refs detected: {shared_count} shared across papers",
        f"only {shared_count} shared (expected >= 5)",
    )

    # Each shared ref should have paper_count > 1
    for d in deduped:
        if d.paper_count > 1:
            source_files = set(s[0].name for s in d.sources)
            check(
                len(source_files) > 1,
                f"shared ref '{d.canonical.title[:40]}...' in {len(source_files)} files",
            )
            break  # Just check one to keep output short

    # Canonical should be the highest-confidence extraction
    for d in deduped:
        if d.paper_count > 1:
            for pdf_path, ref_idx in d.sources:
                matching_refs = [r for r in per_file[pdf_path] if r.source_index == ref_idx]
                if matching_refs:
                    check(
                        d.canonical.extraction_confidence >= matching_refs[0].extraction_confidence,
                        f"canonical has best confidence ({d.canonical.extraction_confidence:.1f})",
                    )
                    break
            break


def test_batch_priority():
    """Test that references are prioritized correctly: fewer papers first, newest first."""
    from halref.config import Config
    from halref.extract.ensemble import extract_references
    from halref.pipeline import _deduplicate_references, _prioritize_references

    config = Config.default()
    pdfs = [
        FIXTURES / "test_paper_real.pdf",
        FIXTURES / "test_paper_hallucinated.pdf",
        FIXTURES / "test_paper_long_refs.pdf",
    ]
    available = [p for p in pdfs if p.exists()]
    if len(available) < 2:
        print("  SKIP: need at least 2 test PDFs")
        return

    per_file = {}
    for f in available:
        per_file[f] = extract_references(f, config)

    deduped = _deduplicate_references(per_file)
    prioritized = _prioritize_references(deduped)

    # Single-paper refs should come before multi-paper refs
    first_multi_idx = None
    last_single_idx = None
    for i, d in enumerate(prioritized):
        if d.paper_count == 1:
            last_single_idx = i
        elif first_multi_idx is None:
            first_multi_idx = i

    if first_multi_idx is not None and last_single_idx is not None:
        check(
            last_single_idx < first_multi_idx,
            "single-paper refs all before multi-paper refs",
            f"last single at {last_single_idx}, first multi at {first_multi_idx}",
        )

    # Within single-paper refs, newest should come first
    single_years = [
        d.canonical.year for d in prioritized
        if d.paper_count == 1 and d.canonical.year
    ]
    if len(single_years) >= 2:
        check(
            single_years == sorted(single_years, reverse=True),
            f"single-paper refs sorted newest first: {single_years[:5]}...",
            f"not sorted: {single_years[:5]}",
        )

    # The fictitious papers (only in hallucinated PDF) should be high priority
    top_5_titles = [d.canonical.title[:40].lower() for d in prioritized[:5]]
    fictitious_found = any("adaptive cross-lingual" in t for t in top_5_titles)
    check(
        fictitious_found,
        "fictitious 'Adaptive Cross-Lingual...' in top 5 priority",
        f"top 5: {top_5_titles}",
    )

    print(f"    Priority order (top 5):")
    for i, d in enumerate(prioritized[:5]):
        print(f"      {i+1}. [{d.paper_count}p] ({d.canonical.year or '?'}) {d.canonical.title[:50]}")


def test_batch_reassembly():
    """Test that deduplicated results map back correctly to per-file reports."""
    from halref.config import Config
    from halref.extract.ensemble import extract_references
    from halref.models import MatchResult, Reference
    from halref.pipeline import (
        DeduplicatedRef,
        _deduplicate_references,
        _reassemble_reports,
    )

    config = Config.default()
    pdfs = [
        FIXTURES / "test_paper_real.pdf",
        FIXTURES / "test_paper_hallucinated.pdf",
    ]
    available = [p for p in pdfs if p.exists()]
    if len(available) < 2:
        print("  SKIP: need 2 test PDFs")
        return

    per_file = {}
    for f in available:
        per_file[f] = extract_references(f, config)

    deduped = _deduplicate_references(per_file)

    # Simulate verification: give each deduped ref a fake result
    for d in deduped:
        d.result = MatchResult(
            reference=d.canonical,
            hallucination_score=0.1 if d.paper_count > 1 else 0.8,
        )

    reports = _reassemble_reports(per_file, deduped, config)

    # Should get one report per file
    check(
        len(reports) == len(available),
        f"one report per file: {len(reports)} reports for {len(available)} files",
    )

    # Each report should have same ref count as extracted
    for report in reports:
        pdf_path = Path(report.input_file)
        expected = len(per_file[pdf_path])
        check(
            report.total_references == expected,
            f"{pdf_path.name}: {report.total_references} refs in report (expected {expected})",
        )

    # Shared refs should have the same score across files
    # (since they were verified once via dedup)
    # Use full normalized title + year as key to avoid collisions between
    # e.g. real BERT (2019, Devlin) and hallucinated BERT (2019, Chang)
    if len(reports) >= 2:
        from halref.matching.title_matcher import normalize_title

        scores_by_key: dict[str, list[float]] = {}
        for report in reports:
            for result in report.results:
                ref = result.reference
                key = f"{normalize_title(ref.title)}|{ref.year}"
                if ref.title:
                    scores_by_key.setdefault(key, []).append(result.hallucination_score)

        consistent = 0
        total_shared = 0
        for key, scores in scores_by_key.items():
            if len(scores) > 1:
                total_shared += 1
                if len(set(scores)) == 1:
                    consistent += 1

        if total_shared > 0:
            check(
                consistent == total_shared,
                f"shared refs have consistent scores: {consistent}/{total_shared}",
                f"{total_shared - consistent} inconsistent",
            )


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

async def test_api_match() -> None:
    """Quick API test: search for a known paper on Semantic Scholar."""
    from halref.apis.semantic_scholar import SemanticScholarClient
    from halref.models import Reference

    async with SemanticScholarClient() as client:
        ref = Reference(title="Attention is All You Need")
        try:
            matches = await client.search(ref)
            if matches:
                m = matches[0]
                check(True, f"S2 found: '{m.title}' ({m.year})")
                check(
                    m.year == 2017,
                    f"S2 correct year: {m.year}",
                    f"expected 2017, got {m.year}",
                )
                first_author = m.authors[0].last if m.authors else ""
                check(
                    first_author.lower() == "vaswani",
                    f"S2 correct first author: {first_author}",
                    f"expected Vaswani, got {first_author}",
                )
            else:
                check(False, "S2 search returned results")
        except Exception as e:
            print(f"  [SKIP] API error: {e}")


async def test_api_fictitious() -> None:
    """Test that a fictitious paper is NOT found."""
    from halref.apis.semantic_scholar import SemanticScholarClient
    from halref.matching.title_matcher import title_similarity
    from halref.models import Reference

    async with SemanticScholarClient() as client:
        ref = Reference(title="Neural Reference Verification with Contrastive Citation Embeddings")
        try:
            matches = await client.search(ref)
            if matches:
                best_sim = max(title_similarity(ref.title, m.title) for m in matches)
                check(
                    best_sim < 0.8,
                    f"fictitious paper not closely matched (best sim={best_sim:.2f})",
                    f"unexpectedly high match: {best_sim:.2f}",
                )
            else:
                check(True, "fictitious paper returned no matches (expected)")
        except Exception as e:
            print(f"  [SKIP] API error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global passed, failed

    print("=" * 60)
    print("HALREF TEST SUITE")
    print("=" * 60)

    # --- Single-file extraction ---
    print("\n--- Test 1: Real references (test_paper_real.pdf) ---")
    refs_real = test_extraction("test_paper_real.pdf", 10, "extraction")
    if refs_real:
        test_field_accuracy(refs_real, REAL_REFS, "field accuracy")

    print("\n--- Test 2: Hallucinated refs (test_paper_hallucinated.pdf) ---")
    refs_hall = test_extraction("test_paper_hallucinated.pdf", 15, "extraction")
    if refs_hall:
        all_known = REAL_REFS[:5] + HALLUCINATED_REFS
        test_field_accuracy(refs_hall, all_known, "field accuracy")

    print("\n--- Test 3: Long references (test_paper_long_refs.pdf) ---")
    test_extraction("test_paper_long_refs.pdf", 20, "extraction")

    print("\n--- Test 4: Limitations before refs (test_paper_limitations.pdf) ---")
    refs_lim = test_extraction("test_paper_limitations.pdf", 10, "extraction")
    if refs_lim:
        test_field_accuracy(refs_lim, REAL_REFS, "field accuracy")

    # --- Citation style variants ---
    print("\n--- Test 5: Numbered refs / IEEE style (test_paper_numbered_refs.pdf) ---")
    print("    (Tests splitter with [N] prefix boundaries, year not embedded in prefix)")
    test_numbered_refs()

    print("\n--- Test 6: Unicode author names (test_paper_unicode_authors.pdf) ---")
    print("    (Tests that diacritics like ü, ø, č are preserved through extraction)")
    test_unicode_author_names()

    print("\n--- Test 7: arXiv-style refs (test_paper_arxiv_style.pdf) ---")
    print("    (Tests refs with no venue/pages, et al. truncation, DOI/URL fields)")
    test_arxiv_style_refs()

    # --- Batch dedup ---
    print("\n--- Test 8: Batch deduplication ---")
    test_batch_dedup()

    print("\n--- Test 9: Batch priority ordering ---")
    test_batch_priority()

    print("\n--- Test 10: Batch result reassembly ---")
    test_batch_reassembly()

    # --- API tests ---
    print("\n--- Test 11: API - known paper lookup ---")
    asyncio.run(test_api_match())

    print("\n--- Test 12: API - fictitious paper lookup ---")
    asyncio.run(test_api_fictitious())

    # --- Summary ---
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
