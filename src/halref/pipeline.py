"""Main orchestration pipeline."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from halref.config import Config
from halref.matching.title_matcher import normalize_title, title_similarity
from halref.models import BatchReport, MatchResult, Reference, VerificationReport

logger = logging.getLogger(__name__)

# Threshold for considering two references as duplicates across papers
DEDUP_SIMILARITY_THRESHOLD = 0.92

# References appearing in this many or more papers are considered "well-known"
# and can be searched with lower priority (or skipped if above skip threshold)
WELL_KNOWN_SEARCH_THRESHOLD = 3


@dataclass
class DeduplicatedRef:
    """A unique reference that may appear in multiple papers."""

    canonical: Reference  # Representative reference (best extraction)
    sources: list[tuple[Path, int]] = field(default_factory=list)  # (pdf_path, ref_index)
    paper_count: int = 0
    result: MatchResult | None = None  # Filled after verification


def extract_all(
    pdf_files: list[Path],
    config: Config,
) -> dict[Path, list[Reference]]:
    """Extract references from all PDFs (no verification).

    Call this first to get references, write .bib files, then pass
    the result to run_check_extracted() for API verification.
    """
    from halref.extract.ensemble import extract_references

    per_file_refs: dict[Path, list[Reference]] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=Console(stderr=True),
    ) as progress:
        task = progress.add_task("Extracting references...", total=len(pdf_files))
        for pdf_path in pdf_files:
            progress.update(task, description=f"Extracting {pdf_path.name}...")
            refs = extract_references(pdf_path, config)
            per_file_refs[pdf_path] = refs
            progress.advance(task)

    return per_file_refs


async def run_check(
    pdf_files: list[Path],
    config: Config,
    threshold: float = 0.5,
    per_file_refs: dict[Path, list[Reference]] | None = None,
) -> BatchReport:
    """Run the full verification pipeline on one or more PDFs.

    Args:
        pdf_files: PDF paths to check.
        config: Configuration.
        threshold: Hallucination score threshold.
        per_file_refs: Pre-extracted references (from extract_all). If None,
                       extraction is done here.

    For multiple PDFs:
    1. Extract references from all PDFs first (or use pre-extracted)
    2. Deduplicate references across papers
    3. Search unique references, prioritizing single-paper refs and newest first
    4. Map results back to per-PDF reports
    """
    console = Console(stderr=True)

    # Step 1: Extract (or use pre-extracted)
    if per_file_refs is None:
        per_file_refs = extract_all(pdf_files, config)

    if len(pdf_files) == 1:
        pdf_path = pdf_files[0]
        refs = per_file_refs.get(pdf_path, [])
        report = await _verify_single(pdf_path, refs, config)
        return BatchReport(
            reports=[report],
            total_files=1,
            total_references=report.total_references,
            total_flagged=len(report.flagged(threshold)),
        )

    # --- Batch mode: deduplicate, search, reassemble ---

    total_refs = sum(len(refs) for refs in per_file_refs.values())
    console.print(
        f"Extracted {total_refs} references from {len(pdf_files)} files",
        style="dim",
    )

    # Step 2: Deduplicate across papers
    deduped = _deduplicate_references(per_file_refs)
    unique_count = len(deduped)
    multi_count = sum(1 for d in deduped if d.paper_count > 1)
    console.print(
        f"Deduplicated to {unique_count} unique references "
        f"({multi_count} shared across papers)",
        style="dim",
    )

    # Step 3: Prioritize and search
    prioritized = _prioritize_references(deduped)
    clients = _create_api_clients(config)

    from halref.agent.strategies import VerificationAgent
    from halref.matching.scorer import score_reference

    agent = VerificationAgent(max_retries=config.agent.max_retries)
    sem = asyncio.Semaphore(5)

    async def verify_one(dref: DeduplicatedRef) -> None:
        async with sem:
            matches, strategies = await agent.verify(dref.canonical, clients)
            result = score_reference(dref.canonical, matches, config.matching.weights)
            result.strategies_used = strategies

            if config.llm.enabled and 0.3 <= result.hallucination_score <= 0.7:
                from halref.agent.llm_verify import llm_verify_match
                result = await llm_verify_match(result, config.llm)

            dref.result = result

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("~"),
        TimeRemainingColumn(),
        console=Console(stderr=True),
    ) as progress:
        task = progress.add_task("Verifying references...", total=len(prioritized))

        # Process in priority order but with concurrency
        batch_size = 5
        for i in range(0, len(prioritized), batch_size):
            batch = prioritized[i:i + batch_size]
            await asyncio.gather(*[verify_one(d) for d in batch])
            progress.advance(task, advance=len(batch))

    for client in clients:
        await client.close()

    # Step 4: Reassemble per-file reports
    reports = _reassemble_reports(per_file_refs, deduped, config)

    batch = BatchReport(
        reports=reports,
        total_files=len(reports),
        total_references=sum(r.total_references for r in reports),
        total_flagged=sum(len(r.flagged(threshold)) for r in reports),
    )
    return batch


def _deduplicate_references(
    per_file_refs: dict[Path, list[Reference]],
) -> list[DeduplicatedRef]:
    """Merge duplicate references across papers by title similarity.

    Returns a list of DeduplicatedRef, each representing a unique reference
    with pointers back to which files contained it.
    """
    deduped: list[DeduplicatedRef] = []

    for pdf_path, refs in per_file_refs.items():
        for ref in refs:
            if not ref.title:
                # Can't deduplicate without a title — treat as unique
                d = DeduplicatedRef(canonical=ref, paper_count=1)
                d.sources.append((pdf_path, ref.source_index))
                deduped.append(d)
                continue

            # Check if this reference already exists in deduped list
            norm_title = normalize_title(ref.title)
            matched = False

            for existing in deduped:
                if not existing.canonical.title:
                    continue
                sim = title_similarity(ref.title, existing.canonical.title)
                if sim >= DEDUP_SIMILARITY_THRESHOLD:
                    # Duplicate found — merge
                    existing.sources.append((pdf_path, ref.source_index))
                    existing.paper_count = len(set(s[0] for s in existing.sources))
                    # Keep the extraction with higher confidence as canonical
                    if ref.extraction_confidence > existing.canonical.extraction_confidence:
                        existing.canonical = ref
                    matched = True
                    break

            if not matched:
                d = DeduplicatedRef(canonical=ref, paper_count=1)
                d.sources.append((pdf_path, ref.source_index))
                deduped.append(d)

    return deduped


def _prioritize_references(
    deduped: list[DeduplicatedRef],
) -> list[DeduplicatedRef]:
    """Sort references for verification priority.

    Priority order:
    1. References in fewer papers first (single-paper refs are most suspicious)
    2. Within same paper_count, newest first (more recent = more likely LLM-generated)
    """
    def sort_key(d: DeduplicatedRef) -> tuple:
        paper_count = d.paper_count
        # Negative year so newest sorts first; None sorts last
        year = -(d.canonical.year or 0)
        return (paper_count, year)

    return sorted(deduped, key=sort_key)


def _reassemble_reports(
    per_file_refs: dict[Path, list[Reference]],
    deduped: list[DeduplicatedRef],
    config: Config,
) -> list[VerificationReport]:
    """Map deduplicated verification results back to per-file reports."""
    # Build lookup: (pdf_path, ref_index) -> MatchResult
    result_lookup: dict[tuple[Path, int], MatchResult] = {}
    for dref in deduped:
        if dref.result is None:
            continue
        for pdf_path, ref_index in dref.sources:
            # Clone the result but with the file-specific reference
            result_lookup[(pdf_path, ref_index)] = dref.result

    reports = []
    for pdf_path, refs in per_file_refs.items():
        results = []
        for ref in refs:
            key = (pdf_path, ref.source_index)
            if key in result_lookup:
                # Use the deduplicated result but attach this file's reference
                match_result = result_lookup[key].model_copy()
                match_result.reference = ref
                results.append(match_result)
            else:
                # No result (shouldn't happen, but handle gracefully)
                from halref.matching.scorer import score_reference
                results.append(score_reference(ref, [], config.matching.weights))

        report = VerificationReport(
            input_file=str(pdf_path),
            total_references=len(refs),
            flagged_count=len([r for r in results if r.hallucination_score >= 0.5]),
            results=results,
            extraction_methods=config.extraction.text_extractors,
            apis_used=[],
        )
        reports.append(report)

    return reports


async def _verify_single(
    pdf_path: Path,
    references: list[Reference],
    config: Config,
) -> VerificationReport:
    """Verify pre-extracted references from a single PDF."""
    if not references:
        logger.warning(f"No references extracted from {pdf_path}")
        return VerificationReport(
            input_file=str(pdf_path),
            extraction_methods=config.extraction.text_extractors,
        )

    clients = _create_api_clients(config)

    # Repair truncated references via API (verify against PDF text)
    from halref.extract.repair import repair_references
    from halref.extract.text_extractors.pdfminer_extractor import PdfminerExtractor
    try:
        full_ref_text = PdfminerExtractor().extract_text(pdf_path)
    except Exception:
        full_ref_text = ""
    references = await repair_references(references, clients, full_ref_text)

    from halref.agent.strategies import VerificationAgent
    from halref.matching.scorer import score_reference

    agent = VerificationAgent(max_retries=config.agent.max_retries)
    sem = asyncio.Semaphore(5)

    async def verify_one(ref: Reference) -> MatchResult:
        async with sem:
            matches, strategies = await agent.verify(ref, clients)
            result = score_reference(ref, matches, config.matching.weights)
            result.strategies_used = strategies

            if config.llm.enabled and 0.3 <= result.hallucination_score <= 0.7:
                from halref.agent.llm_verify import llm_verify_match
                result = await llm_verify_match(result, config.llm)

            return result

    # Prioritize: newest references first
    sorted_refs = sorted(references, key=lambda r: -(r.year or 0))
    results_map: dict[int, MatchResult] = {}

    results = await asyncio.gather(*[verify_one(ref) for ref in sorted_refs])
    for ref, result in zip(sorted_refs, results):
        results_map[ref.source_index] = result

    # Restore original order
    ordered_results = [results_map[ref.source_index] for ref in references]

    for client in clients:
        await client.close()

    report = VerificationReport(
        input_file=str(pdf_path),
        total_references=len(references),
        flagged_count=len([r for r in ordered_results if r.hallucination_score >= 0.5]),
        results=ordered_results,
        extraction_methods=config.extraction.text_extractors,
        apis_used=[c.name for c in clients],
    )
    return report


def run_extract(pdf_files: list[Path], config: Config) -> list[Reference]:
    """Extract references from PDFs (no API verification)."""
    from halref.extract.ensemble import extract_references

    all_refs = []
    for pdf_path in pdf_files:
        refs = extract_references(pdf_path, config)
        all_refs.extend(refs)
    return all_refs


def _create_api_clients(config: Config) -> list:
    """Create API clients based on config."""
    from halref.apis.crossref import CrossRefClient
    from halref.apis.dblp import DBLPClient
    from halref.apis.openalex import OpenAlexClient
    from halref.apis.semantic_scholar import SemanticScholarClient

    console = Console(stderr=True)
    clients = []
    missing_keys = []

    api_cfg = config.get_api_config("semantic_scholar")
    if api_cfg.enabled:
        clients.append(SemanticScholarClient(api_key=api_cfg.api_key))
        if not api_cfg.api_key:
            missing_keys.append(
                "[bold]Semantic Scholar[/bold]: No API key set (running at reduced rate).\n"
                "  Get a free key at: https://www.semanticscholar.org/product/api#api-key\n"
                "  Set via: SEMANTIC_SCHOLAR_API_KEY env var or apis.semantic_scholar.api_key in config"
            )

    api_cfg = config.get_api_config("crossref")
    if api_cfg.enabled:
        clients.append(CrossRefClient(mailto=api_cfg.mailto))
        if not api_cfg.mailto:
            missing_keys.append(
                "[bold]CrossRef[/bold]: No mailto set (using shared rate pool).\n"
                "  Set your email for the polite pool (faster rates, no signup needed).\n"
                "  Set via: CROSSREF_MAILTO env var or apis.crossref.mailto in config"
            )

    api_cfg = config.get_api_config("dblp")
    if api_cfg.enabled:
        clients.append(DBLPClient())

    api_cfg = config.get_api_config("openalex")
    if api_cfg.enabled:
        clients.append(OpenAlexClient(api_key=api_cfg.api_key))
        if not api_cfg.api_key:
            missing_keys.append(
                "[bold]OpenAlex[/bold]: No API key set (required since Feb 2026).\n"
                "  Get a free key at: https://openalex.org/users/me\n"
                "  Set via: OPENALEX_API_KEY env var or apis.openalex.api_key in config"
            )

    # ACL Anthology is last (local, slower to load)
    api_cfg = config.get_api_config("acl_anthology")
    if api_cfg.enabled:
        from halref.apis.acl_anthology import ACLAnthologyClient
        client = ACLAnthologyClient()
        if client._is_available():
            clients.append(client)

    if missing_keys:
        console.print("\n[yellow]API key recommendations:[/yellow]")
        for msg in missing_keys:
            console.print(f"  {msg}")
        console.print()

    return clients
