"""Test extraction accuracy against papers with known ground truth .bib files.

These tests use real arxiv papers whose LaTeX source (including .bib files)
was downloaded, giving us the exact list of references the paper should contain.

Run: python tests/test_ground_truth.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SOURCE_DIR = Path(__file__).parent.parent / "test_acl_source"

passed = 0
failed = 0


def check(condition: bool, label: str, detail: str = "") -> bool:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        detail_str = f" — {detail}" if detail else ""
        print(f"  [FAIL] {label}{detail_str}")
    return condition


def parse_bib_entries(bib_path: Path) -> list[dict]:
    """Parse .bib file into list of {title, year, first_author}."""
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    entries = []
    for block in re.split(r"\n@", text):
        title_m = re.search(r"title\s*=\s*[{\"](.+?)[}\"]", block, re.DOTALL | re.IGNORECASE)
        year_m = re.search(r"year\s*=\s*[{\"]?(\d{4})[}\"]?", block, re.IGNORECASE)
        author_m = re.search(r"author\s*=\s*[{\"](.+?)[}\"]", block, re.DOTALL | re.IGNORECASE)

        if not title_m:
            continue
        title = re.sub(r"[{}]", "", title_m.group(1)).strip()
        title = re.sub(r"\s+", " ", title)
        if len(title) < 10:
            continue

        year = int(year_m.group(1)) if year_m else None
        first_author = ""
        if author_m:
            raw = re.sub(r"[{}]", "", author_m.group(1))
            parts = re.split(r"\s+and\s+", raw)[0].strip()
            if "," in parts:
                first_author = parts.split(",")[0].strip()
            else:
                words = parts.split()
                first_author = words[-1] if words else ""

        entries.append({"title": title, "year": year, "first_author": first_author})
    return entries


def test_paper(
    paper_id: str,
    pdf_path: Path,
    bib_path: Path,
    min_match_rate: float = 0.60,
    max_false_positive_rate: float = 0.20,
):
    """Test extraction of one paper against its ground truth .bib."""
    from halref.config import Config
    from halref.extract.ensemble import extract_references
    from rapidfuzz import fuzz

    if not pdf_path.exists():
        print(f"  SKIP: {pdf_path} not found")
        return
    if not bib_path.exists():
        print(f"  SKIP: {bib_path} not found")
        return

    gt_entries = parse_bib_entries(bib_path)
    config = Config.default()
    refs = extract_references(pdf_path, config)
    extracted_titles = [r.title for r in refs if r.title and len(r.title) > 10]

    print(f"  Ground truth: {len(gt_entries)} | Extracted: {len(refs)} ({len(extracted_titles)} with titles)")

    # Match rate: what % of extracted refs match a ground truth entry
    matched_extracted = 0
    false_positives = []
    for r in refs:
        if not r.title or len(r.title) < 10:
            continue
        best_score = 0
        for gt in gt_entries:
            score = fuzz.token_sort_ratio(r.title.lower(), gt["title"].lower())
            best_score = max(best_score, score)
        if best_score > 65:
            matched_extracted += 1
        else:
            false_positives.append((r.title[:60], best_score))

    match_rate = matched_extracted / len(extracted_titles) if extracted_titles else 0
    fp_rate = len(false_positives) / len(extracted_titles) if extracted_titles else 0

    check(
        match_rate >= min_match_rate,
        f"match rate: {match_rate:.0%} (extracted refs matching ground truth)",
        f"got {match_rate:.0%}, need {min_match_rate:.0%}",
    )

    check(
        fp_rate <= max_false_positive_rate,
        f"false positive rate: {fp_rate:.0%} ({len(false_positives)} unmatched)",
        f"got {fp_rate:.0%}, max {max_false_positive_rate:.0%}",
    )

    # Recall: what % of ground truth entries were extracted
    matched_gt = 0
    missed = []
    for gt in gt_entries:
        best = max(
            (fuzz.token_sort_ratio(gt["title"].lower(), t.lower()) for t in extracted_titles),
            default=0,
        )
        if best > 65:
            matched_gt += 1
        else:
            missed.append((gt["title"][:60], gt["year"], best))

    recall = matched_gt / len(gt_entries) if gt_entries else 0
    check(
        recall >= 0.50,
        f"recall: {recall:.0%} ({matched_gt}/{len(gt_entries)} ground truth found)",
        f"got {recall:.0%}",
    )

    # Show top issues
    if false_positives:
        print(f"    False positives (top 3):")
        for title, score in sorted(false_positives, key=lambda x: x[1])[:3]:
            print(f"      ({score:.0f}) {title}")

    if missed:
        print(f"    Missed from ground truth (top 3):")
        for title, year, score in sorted(missed, key=lambda x: x[2])[:3]:
            print(f"      ({year}) {title} (best={score:.0f})")


def main():
    global passed, failed

    print("=" * 60)
    print("GROUND TRUTH EXTRACTION TESTS")
    print("=" * 60)

    # ClaimFlow — has custom.bib with ~86 real entries
    print("\n--- ClaimFlow (2603.16073) ---")
    test_paper(
        "2603.16073",
        SOURCE_DIR / "2603.16073" / "acl_latex.pdf",
        SOURCE_DIR / "2603.16073" / "custom.bib",
        min_match_rate=0.55,
        max_false_positive_rate=0.35,
    )

    # ACL-rlg — has references.bib with 23 entries
    print("\n--- ACL-rlg (2502.15692) ---")
    # This paper uses COLING style; check if we compiled it
    pdf_candidates = [
        SOURCE_DIR / "2502.15692" / "main.pdf",
    ]
    pdf_path = None
    for p in pdf_candidates:
        if p.exists():
            pdf_path = p
            break

    if pdf_path:
        test_paper(
            "2502.15692",
            pdf_path,
            SOURCE_DIR / "2502.15692" / "references.bib",
            min_match_rate=0.55,
            max_false_positive_rate=0.45,  # Small bib, LaTeX markup causes mismatches
        )
    else:
        # Use the arxiv-downloaded PDF instead
        arxiv_pdf = Path("test_arxiv/ACL_rlg_Dataset.pdf")
        if arxiv_pdf.exists():
            test_paper(
                "2502.15692",
                arxiv_pdf,
                SOURCE_DIR / "2502.15692" / "references.bib",
                min_match_rate=0.55,
                max_false_positive_rate=0.45,
            )
        else:
            print("  SKIP: no PDF found for ACL-rlg")

    # Papers in papers/ directory (no .bib ground truth, but test extraction quality)
    print("\n--- Real papers (papers/) — extraction sanity checks ---")
    from halref.config import Config
    from halref.extract.ensemble import extract_references

    config = Config.default()
    papers_dir = Path("papers")
    if papers_dir.exists():
        for pdf in sorted(papers_dir.glob("*.pdf")):
            refs = extract_references(pdf, config)
            good = [r for r in refs if r.year and r.extraction_confidence >= 0.5]
            bad = [r for r in refs if not r.year or r.extraction_confidence < 0.5]

            check(
                len(refs) >= 5,
                f"{pdf.name}: extracted {len(refs)} refs",
                f"too few refs",
            )
            check(
                len(good) / len(refs) >= 0.80 if refs else True,
                f"{pdf.name}: {len(good)}/{len(refs)} high quality ({len(good)/len(refs)*100:.0f}%)",
                f"too many bad refs: {len(bad)}",
            )

    # Summary
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
