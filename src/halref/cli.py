"""CLI entry point for halref."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="halref",
    help="Detect hallucinated references in academic PDFs.",
    no_args_is_help=True,
)
console = Console()


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


@app.command()
def check(
    paths: Annotated[list[Path], typer.Argument(help="PDF file(s) or directory to check")],
    outdir: Annotated[Path, typer.Option("--outdir", "-d", help="Output directory for all results")] = Path("halref_output"),
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config TOML file")] = None,
    ref_pages: Annotated[Optional[str], typer.Option("--ref-pages", help="Reference pages (1-indexed, e.g., '9-13')")] = None,
    format: Annotated[str, typer.Option("--format", "-f", help="Output format: terminal, json, bib, all")] = "all",
    apis: Annotated[Optional[str], typer.Option("--apis", help="Comma-separated API list (e.g., 's2,crossref,dblp')")] = None,
    llm: Annotated[bool, typer.Option("--llm", help="Enable LLM verification pass")] = False,
    llm_base_url: Annotated[Optional[str], typer.Option("--llm-base-url", help="LLM endpoint URL")] = None,
    llm_model: Annotated[Optional[str], typer.Option("--llm-model", help="LLM model name")] = None,
    threshold: Annotated[float, typer.Option("--threshold", "-t", help="Hallucination score threshold for flagging")] = 0.5,
    show_ok: Annotated[bool, typer.Option("--show-ok", help="Show all references including verified OK ones")] = False,
) -> None:
    """Check PDF(s) for hallucinated references.

    By default, only shows medium/high-risk references. Use --show-ok to see all.

    All output goes into --outdir (default: halref_output/):
      - Per-PDF .bib files with extracted references
      - report.json with full verification results
      - report_annotated.bib with hallucination scores
      - Terminal report printed to stdout
    """
    from halref.config import load_config

    t_start = time.time()

    cfg = load_config(str(config) if config else None)

    # CLI overrides
    if ref_pages:
        cfg.extraction.ref_pages = ref_pages
    if llm:
        cfg.llm.enabled = True
    if llm_base_url:
        cfg.llm.base_url = llm_base_url
    if llm_model:
        cfg.llm.model = llm_model

    # Resolve PDF paths
    pdf_files = _resolve_paths(paths)
    if not pdf_files:
        console.print("[red]No PDF files found.[/red]")
        raise typer.Exit(1)

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]halref check[/bold] — {len(pdf_files)} PDF(s)")
    console.print(f"Output: {outdir.resolve()}", style="dim")
    console.print()

    # Step 1: Extract references from all PDFs
    from halref.extract.bib_writer import write_bib
    from halref.pipeline import extract_all, run_check

    t_extract = time.time()
    per_file_refs = extract_all(pdf_files, cfg)
    total_refs = sum(len(r) for r in per_file_refs.values())
    console.print(
        f"[green]Extraction complete[/green] — "
        f"{total_refs} references from {len(pdf_files)} files "
        f"({_fmt_elapsed(time.time() - t_extract)})"
    )

    # Step 2: Write .bib files immediately (before verification)
    bib_dir = outdir / "bib"
    bib_dir.mkdir(exist_ok=True)
    for pdf_path, refs in per_file_refs.items():
        if refs:
            bib_path = bib_dir / f"{pdf_path.stem}.bib"
            write_bib(refs, bib_path)

    # Step 3: Verify against APIs (using pre-extracted refs)
    console.print()
    t_verify = time.time()
    report = asyncio.run(
        run_check(pdf_files, cfg, threshold=threshold, per_file_refs=per_file_refs)
    )
    console.print(
        f"[green]Verification complete[/green] — "
        f"{report.total_flagged} flagged of {report.total_references} "
        f"({_fmt_elapsed(time.time() - t_verify)})"
    )

    # Step 4: Output report(s)
    from halref.output.bib_output import write_bib_report
    from halref.output.json_output import write_json_report
    from halref.output.report import print_terminal_report

    write_json = format in ("json", "all")
    write_bib_annotated = format in ("bib", "all")
    write_terminal = format in ("terminal", "all")

    if write_json:
        json_path = outdir / "report.json"
        write_json_report(report, json_path)
        console.print(f"Wrote {json_path.resolve()}", style="dim")

    if write_bib_annotated:
        bib_report_path = outdir / "report_annotated.bib"
        write_bib_report(report, bib_report_path)
        console.print(f"Wrote {bib_report_path.resolve()}", style="dim")

    console.print()
    if write_terminal:
        print_terminal_report(report, threshold=threshold, show_ok=show_ok)

    console.print(
        f"\n[bold]Done[/bold] in {_fmt_elapsed(time.time() - t_start)}"
    )


@app.command()
def extract(
    paths: Annotated[list[Path], typer.Argument(help="PDF file(s) to extract references from")],
    outdir: Annotated[Path, typer.Option("--outdir", "-d", help="Output directory for .bib files")] = Path("halref_output"),
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config TOML file")] = None,
    ref_pages: Annotated[Optional[str], typer.Option("--ref-pages", help="Reference pages (1-indexed, e.g., '9-13')")] = None,
) -> None:
    """Extract references from PDF(s) and output as BibTeX.

    Saves one .bib file per PDF into --outdir (default: halref_output/).
    """
    from halref.config import load_config
    from halref.extract.bib_writer import write_bib
    from halref.extract.ensemble import extract_references

    t_start = time.time()

    cfg = load_config(str(config) if config else None)
    if ref_pages:
        cfg.extraction.ref_pages = ref_pages

    pdf_files = _resolve_paths(paths)
    if not pdf_files:
        console.print("[red]No PDF files found.[/red]")
        raise typer.Exit(1)

    outdir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]halref extract[/bold] — {len(pdf_files)} PDF(s)")
    console.print(f"Output: {outdir.resolve()}", style="dim")
    console.print()

    total = 0
    for pdf_path in pdf_files:
        t_file = time.time()
        refs = extract_references(pdf_path, cfg)
        if refs:
            bib_path = outdir / f"{pdf_path.stem}.bib"
            write_bib(refs, bib_path)
            total += len(refs)
        elapsed = _fmt_elapsed(time.time() - t_file)
        console.print(
            f"  {pdf_path.name}: {len(refs)} refs ({elapsed})",
            style="dim" if refs else "yellow",
        )

    console.print(
        f"\n[bold]Done[/bold] — {total} references in {_fmt_elapsed(time.time() - t_start)}"
    )


def _resolve_paths(paths: list[Path]) -> list[Path]:
    """Expand directories into PDF files."""
    pdf_files = []
    for p in paths:
        if p.is_dir():
            pdf_files.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf" and p.exists():
            pdf_files.append(p)
        else:
            console.print(f"[yellow]Skipping {p} (not a PDF or doesn't exist)[/yellow]")
    return pdf_files


if __name__ == "__main__":
    app()
