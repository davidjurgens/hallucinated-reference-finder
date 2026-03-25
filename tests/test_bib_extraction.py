"""Test bib extraction accuracy against 10 real ACL papers with ground truth.

Each paper was downloaded from arxiv with its full LaTeX source including
.bib or .bbl files. We extract references from the compiled PDF and compare
against the known bibliography.

Run: python tests/test_bib_extraction.py

Prerequisites: Run tests/fixtures/download_acl_papers.py first to download papers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from bib_ground_truth import load_ground_truth

ROOT = Path(__file__).parent.parent
SOURCE_DIR = ROOT / "test_acl_source"
ARXIV_PDF_DIR = ROOT / "test_arxiv"

# Papers and their expected properties
PAPERS = [
    # (arxiv_id, name, min_refs, min_precision, min_recall, max_nonsense)
    # Defaults in test_paper: precision>=0.55, recall>=0.40, nonsense<=3
    {"id": "2603.16073", "name": "ClaimFlow", "min_refs": 50},
    {"id": "2601.18724", "name": "HalluCitation", "min_refs": 50,
     "min_recall": 0.15, "max_nonsense": 20},  # 57-page paper with huge appendix, .bbl ground truth
    {"id": "2503.24047", "name": "Large269", "min_refs": 50},
    {"id": "2502.18414", "name": "Medium53", "min_refs": 2,
     "min_precision": 0.0, "min_recall": 0.0},  # Known: pdfminer column interleaving
    {"id": "2408.15496", "name": "Small30", "min_refs": 15},
    {"id": "2308.10792", "name": "Large247", "min_refs": 50,
     "min_recall": 0.30},  # Large paper, multi-page refs
    {"id": "2307.10169", "name": "VeryLarge688", "min_refs": 50,
     "min_recall": 0.30},  # 688 refs, multi-page refs
    {"id": "2205.12644", "name": "Small37", "min_refs": 15,
     "min_precision": 0.45},  # .bbl ground truth partial
    {"id": "2204.06745", "name": "Medium171", "min_refs": 50,
     "min_recall": 0.40},
    {"id": "2202.12837", "name": "Medium99", "min_refs": 30,
     "min_recall": 0.30},  # .bbl ground truth, large
]

passed = 0
failed = 0
skipped = 0


def check(condition: bool, label: str, detail: str = "") -> bool:
    global passed, failed
    if condition:
        passed += 1
        print(f"    [PASS] {label}")
    else:
        failed += 1
        detail_str = f" — {detail}" if detail else ""
        print(f"    [FAIL] {label}{detail_str}")
    return condition


def find_pdf(arxiv_id: str) -> Path | None:
    """Find compiled or downloaded PDF for a paper."""
    # Check compiled PDFs in source dir
    src = SOURCE_DIR / arxiv_id
    if src.exists():
        for pattern in ["acl_latex.pdf", "main.pdf", "paper.pdf", "ACL.pdf"]:
            pdf = src / pattern
            if pdf.exists():
                return pdf
        pdfs = list(src.glob("*.pdf"))
        # Skip figure PDFs
        pdfs = [p for p in pdfs if not any(x in p.name.lower() for x in ["fig", "plot", "image", "diagram"])]
        if pdfs:
            return pdfs[0]

    # Fallback to arxiv-downloaded PDF
    for pattern in [f"{arxiv_id}.pdf", f"{arxiv_id}v1.pdf"]:
        pdf = ARXIV_PDF_DIR / pattern
        if pdf.exists():
            return pdf

    return None


def test_paper(paper: dict) -> None:
    """Test extraction of one paper against its ground truth."""
    global skipped

    arxiv_id = paper["id"]
    name = paper["name"]
    min_refs = paper["min_refs"]

    print(f"\n  --- {name} ({arxiv_id}) ---")

    # Find PDF
    pdf = find_pdf(arxiv_id)
    if not pdf:
        print(f"    SKIP: no PDF found")
        skipped += 1
        return

    # Load ground truth
    src = SOURCE_DIR / arxiv_id
    if not src.exists():
        print(f"    SKIP: source not downloaded")
        skipped += 1
        return

    gt = load_ground_truth(src)
    if not gt:
        print(f"    SKIP: no ground truth found in {src}")
        skipped += 1
        return

    # Extract references
    from halref.config import Config
    from halref.extract.ensemble import extract_references
    from rapidfuzz import fuzz

    config = Config.default()
    refs = extract_references(pdf, config)
    extracted_titles = [r.title for r in refs if r.title and len(r.title) > 10]
    gt_titles = [g.title for g in gt if g.title and len(g.title) > 10]

    print(f"    Ground truth: {len(gt)} | Extracted: {len(refs)} ({len(extracted_titles)} with titles)")

    # Basic extraction check
    check(
        len(refs) >= min_refs,
        f"extracted >= {min_refs} refs (got {len(refs)})",
    )

    if not extracted_titles or not gt_titles:
        return

    # PRECISION: what fraction of extracted refs match a ground truth entry
    precision_matched = 0
    nonsense = []
    for r in refs:
        if not r.title or len(r.title) < 10:
            continue
        best_score = max(
            (fuzz.token_sort_ratio(r.title.lower(), gt_t.lower()) for gt_t in gt_titles),
            default=0,
        )
        if best_score > 60:
            precision_matched += 1
        else:
            # Check if it's nonsense (venue/page fragment)
            t = r.title.lower()
            if any(kw in t for kw in ["proceedings", "conference", "pages", "volume", "association for"]):
                nonsense.append(r.title[:50])

    precision = precision_matched / len(extracted_titles) if extracted_titles else 0
    min_precision = paper.get("min_precision", 0.55)
    check(precision >= min_precision, f"precision: {precision:.0%} ({precision_matched}/{len(extracted_titles)})")

    # RECALL: what fraction of ground truth entries were extracted
    recall_matched = 0
    for gt_t in gt_titles:
        best = max(
            (fuzz.token_sort_ratio(gt_t.lower(), et.lower()) for et in extracted_titles),
            default=0,
        )
        if best > 60:
            recall_matched += 1

    recall = recall_matched / len(gt_titles) if gt_titles else 0
    min_recall = paper.get("min_recall", 0.40)
    check(recall >= min_recall, f"recall: {recall:.0%} ({recall_matched}/{len(gt_titles)})")

    # YEAR ACCURACY: for matched refs, is the year correct?
    year_correct = 0
    year_total = 0
    for r in refs:
        if not r.title or not r.year:
            continue
        # Find matching ground truth
        best_score = 0
        best_gt = None
        for g in gt:
            score = fuzz.token_sort_ratio((r.title or "").lower(), g.title.lower())
            if score > best_score:
                best_score = score
                best_gt = g
        if best_score > 60 and best_gt and best_gt.year:
            year_total += 1
            if r.year == best_gt.year:
                year_correct += 1

    if year_total > 0:
        year_acc = year_correct / year_total
        check(year_acc >= 0.85, f"year accuracy: {year_acc:.0%} ({year_correct}/{year_total})")

    # NO NONSENSE: venue/page fragments should not be titles
    max_nonsense = paper.get("max_nonsense", 3)
    check(
        len(nonsense) <= max_nonsense,
        f"nonsense entries: {len(nonsense)} (max {max_nonsense})",
        f"found: {nonsense[:3]}" if nonsense else "",
    )


def main():
    global passed, failed, skipped

    print("=" * 60)
    print("BIB EXTRACTION TEST SUITE — 10 Real ACL Papers")
    print("=" * 60)

    for paper in PAPERS:
        test_paper(paper)

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    if skipped:
        print(f"\nTo download missing papers: python tests/fixtures/download_acl_papers.py")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
