"""Download and compile 10 real ACL papers from arxiv for testing.

Downloads LaTeX source including .bib files, compiles PDFs locally.
Papers are stored in test_acl_source/{arxiv_id}/.

Run: python tests/fixtures/download_acl_papers.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SOURCE_DIR = ROOT / "test_acl_source"
ACL_STYLE_DIR = ROOT / "acl-style"

# Papers to download: (arxiv_id, bib_file_pattern, notes)
PAPERS = [
    ("2603.16073", "custom.bib", "ClaimFlow — 86 refs"),
    ("2601.18724", "acl_latex.bbl", "HalluCitation — 344 refs (bbl)"),
    ("2503.24047", "acl_latex.bib", "Large — 269 refs"),
    ("2502.18414", "custom.bib", "Medium — 53 refs"),
    ("2408.15496", "ACL.bib", "Small — 30 refs"),
    ("2308.10792", "custom.bib", "Large 2023 — 247 refs"),
    ("2307.10169", "main.bbl", "Very large — 688 refs (bbl)"),
    ("2205.12644", "main.bbl", "Small 2022 — 37 refs (bbl)"),
    ("2204.06745", "acl_citations.bib", "Medium 2022 — 171 refs"),
    ("2202.12837", "datasets.bib", "Medium 2022 — 99 refs"),
]


def download_source(arxiv_id: str) -> bool:
    """Download and extract arxiv source tarball."""
    dest = SOURCE_DIR / arxiv_id
    if dest.exists() and any(dest.glob("*.tex")):
        print(f"  Already exists, skipping download")
        return True

    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    print(f"  Downloading {url}...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "halref-test"})
        resp = urllib.request.urlopen(req, timeout=30)
        data = resp.read()
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

    try:
        tar = tarfile.open(fileobj=BytesIO(data), mode="r:gz")
        tar.extractall(path=str(dest))
        tar.close()
        print(f"  Extracted to {dest}")
        return True
    except Exception as e:
        # Might be a single file (not tarball)
        print(f"  Not a tarball, saving raw: {e}")
        (dest / "source.tex").write_bytes(data)
        return True


def compile_pdf(arxiv_id: str) -> Path | None:
    """Compile LaTeX to PDF. Returns path to PDF or None."""
    src = SOURCE_DIR / arxiv_id

    # Copy ACL style files if not present
    for style_file in ["acl.sty", "acl_natbib.bst"]:
        src_style = ACL_STYLE_DIR / style_file
        dst_style = src / style_file
        if src_style.exists() and not dst_style.exists():
            shutil.copy2(src_style, dst_style)

    # Find main tex file
    tex_files = list(src.glob("*.tex"))
    if not tex_files:
        print(f"  No .tex files found")
        return None

    # Prefer acl_latex.tex or main.tex
    main_tex = None
    for name in ["acl_latex.tex", "main.tex", "paper.tex", "ACL.tex"]:
        candidate = src / name
        if candidate.exists():
            main_tex = candidate
            break
    if not main_tex:
        main_tex = tex_files[0]

    tex_name = main_tex.stem
    print(f"  Compiling {main_tex.name}...")

    # Run pdflatex + bibtex
    env = os.environ.copy()
    for _ in range(3):  # pdflatex needs multiple passes
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", main_tex.name],
            cwd=str(src), capture_output=True, timeout=60, env=env,
        )

    # Run bibtex if .bib exists
    bib_files = list(src.glob("*.bib"))
    if bib_files:
        subprocess.run(
            ["bibtex", tex_name],
            cwd=str(src), capture_output=True, timeout=30,
        )
        # Two more pdflatex passes after bibtex
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", main_tex.name],
                cwd=str(src), capture_output=True, timeout=60, env=env,
            )

    pdf_path = src / f"{tex_name}.pdf"
    if pdf_path.exists():
        from pypdf import PdfReader
        pages = len(PdfReader(str(pdf_path)).pages)
        print(f"  Compiled: {pdf_path.name} ({pages} pages)")
        return pdf_path
    else:
        print(f"  Compilation failed")
        return None


def download_arxiv_pdf(arxiv_id: str) -> Path | None:
    """Download pre-compiled PDF from arxiv as fallback."""
    pdf_dir = ROOT / "test_arxiv"
    pdf_dir.mkdir(exist_ok=True)
    pdf_path = pdf_dir / f"{arxiv_id}.pdf"

    if pdf_path.exists():
        return pdf_path

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    print(f"  Downloading PDF fallback from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "halref-test"})
        resp = urllib.request.urlopen(req, timeout=30)
        pdf_path.write_bytes(resp.read())
        return pdf_path
    except Exception as e:
        print(f"  PDF download failed: {e}")
        return None


def main():
    SOURCE_DIR.mkdir(exist_ok=True)

    results = []
    for arxiv_id, bib_pattern, notes in PAPERS:
        print(f"\n{'='*50}")
        print(f"{arxiv_id} — {notes}")
        print(f"{'='*50}")

        # Download source
        if not download_source(arxiv_id):
            continue
        time.sleep(2)

        # Try to compile
        pdf = compile_pdf(arxiv_id)

        # Fallback to arxiv PDF
        if not pdf:
            pdf = download_arxiv_pdf(arxiv_id)
            time.sleep(2)

        # Check bib/bbl file exists
        src = SOURCE_DIR / arxiv_id
        bib_files = list(src.glob("*.bib")) + list(src.glob("*.bbl"))
        bib_names = [f.name for f in bib_files]

        status = "OK" if pdf and bib_files else "PARTIAL" if bib_files else "FAIL"
        results.append((arxiv_id, status, pdf, bib_names))
        print(f"  Status: {status} | PDF: {pdf is not None} | Bib: {bib_names}")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for arxiv_id, status, pdf, bibs in results:
        print(f"  [{status:7s}] {arxiv_id}: pdf={'yes' if pdf else 'no':3s} bib={bibs}")


if __name__ == "__main__":
    main()
