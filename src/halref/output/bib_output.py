"""Annotated BibTeX output with hallucination scores."""

from __future__ import annotations

from pathlib import Path

from halref.extract.bib_writer import reference_to_bibtex
from halref.models import BatchReport, MatchResult


def write_bib_report(batch: BatchReport, output_path: Path | None = None) -> str:
    """Write annotated BibTeX with hallucination scores as comments.

    Returns the BibTeX string. If output_path provided, also writes to file.
    """
    entries = []

    for report in batch.reports:
        if len(batch.reports) > 1:
            entries.append(f"% === {report.input_file} ===\n")

        for result in report.ranked():
            entry = _annotated_entry(result)
            if entry:
                entries.append(entry)

    bib_str = "\n\n".join(entries) + "\n"

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(bib_str)

    return bib_str


def _annotated_entry(result: MatchResult) -> str:
    """Create a BibTeX entry with hallucination annotation comments."""
    score = result.hallucination_score
    level = _severity_label(score)

    lines = []
    lines.append(f"% HALLUCINATION SCORE: {score:.2f} [{level}]")

    signals = result.signal_summary()
    if signals:
        lines.append(f"% SIGNALS: {'; '.join(signals)}")

    if result.best_match:
        lines.append(f"% BEST MATCH: {result.best_match.title} ({result.best_match.source.value})")
        if result.best_match.doi:
            lines.append(f"% MATCH DOI: {result.best_match.doi}")

    strategies = result.strategies_used
    if strategies:
        lines.append(f"% STRATEGIES: {', '.join(strategies)}")

    bib_entry = reference_to_bibtex(result.reference)
    if bib_entry:
        lines.append(bib_entry)

    return "\n".join(lines)


def _severity_label(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    elif score >= 0.3:
        return "LOW"
    else:
        return "OK"
