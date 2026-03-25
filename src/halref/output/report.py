"""Terminal report output using rich."""

from __future__ import annotations

import re

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from halref.models import Author, BatchReport, MatchResult


def print_terminal_report(
    batch: BatchReport,
    threshold: float = 0.5,
    show_ok: bool = False,
    console: Console | None = None,
) -> None:
    """Print a rich terminal report with diff highlighting.

    Args:
        batch: The verification results.
        threshold: Score threshold for flagging.
        show_ok: If False (default), skip references with score < threshold.
        console: Rich console to print to.
    """
    if console is None:
        console = Console()

    for report in batch.reports:
        console.print()
        apis = ", ".join(report.apis_used) if report.apis_used else "none"

        ranked = report.ranked()

        # Filter out garbage entries that shouldn't appear in reports at all
        # (these are also filtered from bib files by _is_valid_for_bib)
        displayable = [r for r in ranked if _is_displayable(r)]
        noise_count = len(ranked) - len(displayable)

        # Recount flagged based on displayable entries only
        valid_ref_count = len(displayable)
        flagged_count = len([r for r in displayable if r.hallucination_score >= threshold])

        console.print(Panel(
            f"[bold]{report.input_file}[/bold]\n"
            f"References: {valid_ref_count} | "
            f"Flagged (>={threshold}): {flagged_count} | "
            f"APIs: {apis}",
            title="Verification Report",
        ))

        # Separate into shown vs skipped by threshold
        shown = [r for r in displayable if show_ok or r.hallucination_score >= threshold]
        skipped = len(displayable) - len(shown)

        if not shown:
            console.print("  [green]All references verified OK[/green]")
            if skipped:
                console.print(f"  ({skipped} references with score < {threshold} not shown)", style="dim")
            continue

        table = Table(
            show_header=True,
            header_style="bold",
            show_lines=True,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("Score", width=5, justify="center")
        table.add_column("Extracted Reference", ratio=2)
        table.add_column("Best API Match", ratio=2)
        table.add_column("Differences", ratio=1)

        for result in shown:
            score_text = _score_text(result.hallucination_score)
            ref_cell = _format_reference(result)
            match_cell = _format_match(result)
            diff_cell = _format_diffs(result)

            table.add_row(
                str(result.reference.source_index),
                score_text,
                ref_cell,
                match_cell,
                diff_cell,
            )

        console.print(table)

        footer_parts = []
        if skipped:
            footer_parts.append(f"{skipped} OK refs not shown (use --show-ok)")
        if noise_count:
            footer_parts.append(f"{noise_count} incomplete extractions excluded")
        if footer_parts:
            console.print(f"  ({'; '.join(footer_parts)})", style="dim")

    # Batch summary
    if len(batch.reports) > 1:
        console.print()
        summary = Table(title="Batch Summary", show_header=True)
        summary.add_column("File")
        summary.add_column("Refs", justify="right")
        summary.add_column("Flagged", justify="right")
        summary.add_column("Max Score", justify="right")

        for row in batch.summary_rows(threshold):
            summary.add_row(
                row["file"],
                str(row["total_refs"]),
                str(row["flagged"]),
                f"{row['max_score']:.2f}",
            )

        console.print(summary)


def _format_reference(result: MatchResult) -> Text:
    """Format the extracted reference as a multi-line cell."""
    ref = result.reference
    text = Text()

    # Authors — full list
    authors = _full_author_str(ref.authors)
    text.append(authors, style="bold")

    # Year
    if ref.year:
        text.append(f" ({ref.year})", style="cyan")

    text.append("\n")

    # Title
    if ref.title:
        text.append(ref.title)

    # Venue
    if ref.venue:
        text.append(f"\n{ref.venue}", style="dim")

    return text


def _format_match(result: MatchResult) -> Text:
    """Format the best API match with field-level diff indicators."""
    text = Text()

    if not result.best_match:
        text.append("No match found", style="bold red")
        return text

    m = result.best_match
    ref = result.reference

    # Authors — full list, highlight differences
    match_authors = _full_author_str(m.authors)
    if ref.authors and m.authors:
        if not result.first_author_match:
            text.append(match_authors, style="bold red")
        elif not result.author_order_correct:
            text.append(match_authors, style="bold yellow")
        else:
            text.append(match_authors, style="bold")
    else:
        text.append(match_authors, style="bold")

    # Year — highlight if different
    if m.year:
        if ref.year and m.year != ref.year:
            text.append(f" ({m.year})", style="bold red")
        else:
            text.append(f" ({m.year})", style="cyan")

    text.append("\n")

    # Title
    if m.title:
        text.append(m.title)

    # Source + similarity
    parts = []
    if m.venue:
        parts.append(m.venue)
    parts.append(f"[{m.source.value}]")
    text.append(f"\n{' | '.join(parts)}", style="dim")

    if result.title_similarity > 0:
        sim = result.title_similarity
        style = "green" if sim >= 0.9 else "yellow" if sim >= 0.7 else "red"
        text.append(f" (sim: {sim:.0%})", style=style)

    return text


def _format_diffs(result: MatchResult) -> Text:
    """Format specific field differences between extracted and matched reference."""
    text = Text()

    if not result.best_match:
        text.append("Not found in\nany database", style="red")
        return text

    ref = result.reference
    m = result.best_match
    diffs = []

    # Year diff
    if ref.year and m.year and ref.year != m.year:
        diffs.append(("Year", f"{ref.year} vs {m.year}", "red"))

    # First author diff
    if not result.first_author_match and ref.authors and m.authors:
        ref_first = _clean_name(str(ref.authors[0]))
        match_first = _clean_name(str(m.authors[0]))
        diffs.append(("1st author", f"{ref_first}\nvs {match_first}", "red"))

    # Author order diff (but first author matches)
    elif not result.author_order_correct and ref.authors and m.authors:
        diffs.append(("Author order", "reordered", "yellow"))

    # Author count diff
    if ref.authors and m.authors:
        ref_count = len(ref.authors)
        match_count = len(m.authors)
        if abs(ref_count - match_count) > 2:
            diffs.append(("Author count", f"{ref_count} vs {match_count}", "yellow"))

    # Title similarity
    if result.title_similarity < 0.7:
        diffs.append(("Title", f"low match ({result.title_similarity:.0%})", "red"))
    elif result.title_similarity < 0.9:
        diffs.append(("Title", f"partial ({result.title_similarity:.0%})", "yellow"))

    if not diffs:
        text.append("OK", style="green")
        return text

    for i, (field, detail, style) in enumerate(diffs):
        if i > 0:
            text.append("\n")
        text.append(f"{field}: ", style="bold")
        text.append(detail, style=style)

    return text


def _score_text(score: float) -> Text:
    """Format score with color."""
    if score >= 0.7:
        return Text(f"{score:.2f}", style="bold red")
    elif score >= 0.5:
        return Text(f"{score:.2f}", style="yellow")
    elif score >= 0.3:
        return Text(f"{score:.2f}", style="dim yellow")
    else:
        return Text(f"{score:.2f}", style="green")


def _full_author_str(authors: list[Author]) -> str:
    """Format full author list (no truncation), filtering artifacts."""
    clean = [_clean_name(str(a)) for a in authors]
    clean = [n for n in clean if n and not re.match(r"^[\[\]\d\s,]+$", n) and len(n) > 1]

    if not clean:
        return "Unknown"
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def _clean_name(name: str) -> str:
    """Clean an author name string."""
    name = re.sub(r"^\[\d+\]\s*", "", name)
    return name.strip().strip(",").strip()


def _is_displayable(result: MatchResult) -> bool:
    """Check if a result should appear in the report at all.

    Excludes entries that are extraction noise — no useful data to show.
    These are the same entries filtered from bib files.
    """
    ref = result.reference
    title = (ref.title or "").strip()

    # Must have a title with at least 2 words
    if not title or len(title.split()) <= 1:
        return False

    # Title must be long enough to be meaningful
    if len(title) < 15:
        return False

    # Reject title that is clearly an author list (no year, commas, no punctuation)
    if not ref.year and "." not in title and ":" not in title:
        comma_count = title.count(",")
        word_count = len(title.split())
        if comma_count >= 2 and comma_count >= word_count * 0.3:
            return False

    # Must have at least a year OR authors
    if not ref.year and not ref.authors:
        return False

    return True
