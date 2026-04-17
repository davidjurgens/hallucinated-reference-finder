# halref — Hallucinated Reference Finder

Detect hallucinated references in academic PDFs. Extracts citations from papers, verifies them against multiple academic databases, and produces a ranked report of references most likely to be fabricated.

**This tool is designed to work with papers that use `acl.sty`** (the standard ACL/EMNLP/NAACL LaTeX style). The reference extraction pipeline is tuned for the ACL natbib bibliography format and two-column layout. Papers using other styles may produce incomplete or incorrect results.

## What it does

1. **Extracts references** from PDF locally (no data leaves your machine)
2. **Converts to BibTeX** with parsed authors, titles, years, and venues
3. **Verifies each reference** against Semantic Scholar, CrossRef, DBLP, and OpenAlex
4. **Scores hallucination likelihood** based on title match, author accuracy, year correctness, and cross-database consensus
5. **Reports flagged references** with side-by-side comparison showing exactly what differs

## Quick start

```bash
# Install
pip install -e .

# Extract references from a PDF (no network needed)
halref extract paper.pdf -d output/

# Check a directory of PDFs for hallucinated references
halref check submissions/ -d results/

# Check with all references shown (not just flagged ones)
halref check paper.pdf -d results/ --show-ok

# Check a pre-existing .bib file (skip PDF extraction)
halref check-bib references.bib -d results/
```

## Installation

Requires Python 3.11+.

```bash
# Core install
pip install -e .

# With optional backends
pip install -e ".[docling,marker,llm,acl]"
```

### Dependencies

Core (always installed):
- `pdfminer.six` — primary PDF text extraction (best two-column handling)
- `pypdf` — secondary PDF extractor
- `pdfplumber` — fallback PDF extractor
- `rapidfuzz` — fuzzy string matching for title/author comparison
- `nameparser` — author name parsing
- `bibtexparser` — BibTeX reading/writing
- `aiohttp` + `aiolimiter` — async API clients with rate limiting
- `typer` + `rich` — CLI and terminal output
- `pydantic` — data models and configuration

Optional:
- `docling` — IBM's layout-aware document converter
- `marker-pdf` — ML-based PDF to markdown
- `openai` — for local LLM verification via vLLM/Ollama
- `acl-anthology` — ACL Anthology local search

## Usage

### Check references

```bash
# Single file
halref check paper.pdf

# Directory of PDFs
halref check submissions/

# Multiple files
halref check paper1.pdf paper2.pdf paper3.pdf

# Custom output directory and threshold
halref check submissions/ -d results/ --threshold 0.3

# Show all references, not just flagged
halref check paper.pdf --show-ok

# With local LLM for ambiguous cases
halref check paper.pdf --llm --llm-base-url http://localhost:11434/v1 --llm-model llama3
```

### Check a pre-existing .bib file

If you already have a `.bib` file (e.g., from a LaTeX project or another extraction tool), you can verify it directly without needing the PDF:

```bash
# Single .bib file
halref check-bib references.bib

# Directory of .bib files
halref check-bib bibs/ -d results/

# Show all references, not just flagged
halref check-bib references.bib --show-ok
```

### Extract references only (no verification)

```bash
halref extract paper.pdf -d output/
halref extract submissions/ -d bibs/
halref extract paper.pdf --ref-pages 9-13
```

### Output

All output goes to `--outdir` (default: `halref_output/`):

```
halref_output/
├── bib/
│   ├── paper1.bib          # Extracted references per PDF
│   └── paper2.bib
├── report.json              # Full verification results
└── report_annotated.bib     # BibTeX with hallucination scores as comments
```

The terminal report shows flagged references with:
- Side-by-side comparison of extracted vs. matched reference
- Color-coded differences (red = year/author mismatch, yellow = partial match)
- Full author lists for both versions
- Title similarity percentage

## Configuration

Create a `config.toml` or set environment variables:

```bash
# API keys (recommended for faster rate limits)
export SEMANTIC_SCHOLAR_API_KEY="your-key"   # https://www.semanticscholar.org/product/api#api-key
export OPENALEX_API_KEY="your-key"           # Required since Feb 2026 — https://openalex.org/settings/api-key
export CROSSREF_MAILTO="you@example.com"     # No signup, just your email
```

Or use a config file:

```toml
[extraction]
text_extractors = ["pdfminer"]         # pdfminer, pypdf, pdfplumber, docling, marker
field_parsers = ["regex", "heuristic"] # regex, heuristic, llm, api
ref_pages = ""                         # e.g., "9-13" to limit extraction

[apis.semantic_scholar]
api_key = ""
enabled = true

[apis.crossref]
mailto = ""
enabled = true

[apis.dblp]
enabled = true

[apis.openalex]
api_key = ""  # Required since Feb 2026 — get a free key at https://openalex.org/settings/api-key
enabled = true

[matching]
title_threshold = 0.85
author_threshold = 0.6

[matching.weights]
title = 0.30
authors = 0.25
author_order = 0.15
year = 0.15
consensus = 0.15

[llm]
enabled = false
base_url = "http://localhost:8000/v1"
model = ""
api_key = ""
```

A `.env` file in the project root is also loaded automatically.

## How it works

### Extraction pipeline

References are extracted locally — no PDF data is sent to any external service.

1. **Text extraction**: pdfminer.six handles two-column ACL layouts correctly. Scans backwards through the PDF to find the "References" heading, extracts through the end while stopping at Appendix/Limitations sections. Line numbers from review-mode PDFs are stripped automatically.

2. **Reference splitting**: Multiple strategies (blank-line separation, author-year pattern detection, numbered `[N]` markers) with quality-based selection. A post-processing pass merges fragments split across page boundaries.

3. **Field parsing**: Regex parser tuned for ACL natbib format (`Author, Author, and Author. Year. Title. In Venue.`) with a heuristic fallback. Handles hyphenated words, accented names, and `[N]` prefixes.

4. **Validation**: Entries with single-word titles, author-list-as-title patterns, or missing critical fields are filtered from both the BibTeX output and the report.

### Verification pipeline

5. **Batch deduplication**: When checking multiple PDFs, references are deduplicated across papers. Shared references (appearing in 3+ papers) are searched with lower priority since they're likely real.

6. **Waterfall API search**: APIs are queried in priority order (Semantic Scholar → CrossRef → DBLP → OpenAlex). Search stops early when a confident match is found (title similarity > 90%), avoiding unnecessary API calls.

7. **Agentic retry strategies**: If direct title search fails, the system tries: removing subtitles, searching by first author + year, partial title search with distinctive terms, and venue-constrained search.

8. **Repair**: Truncated references (from column-break extraction artifacts) are repaired by querying APIs for the complete metadata. All repaired data is verified against the original PDF text to prevent introducing hallucinated metadata.

9. **Scoring**: Each reference gets a hallucination score (0-1) based on weighted signals:
   - Title mismatch (30%)
   - Author mismatch (25%)
   - Author order wrong (15%)
   - Year mismatch (15%)
   - Low cross-API consensus (15%)

### Hallucination signals

The tool flags these specific patterns:

| Signal | Description |
|--------|-------------|
| **Not found** | No database contains a paper with a similar title |
| **Wrong first author** | The first author differs from the matched paper |
| **Author order swapped** | Same authors but in a different order |
| **Year mismatch** | Publication year differs (e.g., paper says 2019, actual is 2017) |
| **Different title** | Title is similar but not the same (possible fabrication) |
| **Low consensus** | Found in only 1 database instead of multiple |

## API rate limits

| API | Without key | With key | Key signup |
|-----|-------------|----------|------------|
| Semantic Scholar | ~0.3 RPS (shared pool) | 1 RPS (dedicated) | [Free](https://www.semanticscholar.org/product/api#api-key) |
| CrossRef | ~1 RPS | ~5 RPS (polite pool) | Just set your email |
| DBLP | ~2 RPS | N/A | No key needed |
| OpenAlex | ~1 RPS | ~5 RPS | [Free](https://openalex.org/settings/api-key) |

The tool respects rate limits automatically with per-API throttling and exponential backoff on 429 responses.

## Testing

```bash
# Generate test PDFs (requires pdflatex)
python tests/fixtures/generate_test_pdf.py

# Run extraction tests (synthetic papers with known references)
python tests/test_extraction.py

# Download 10 real ACL papers from arxiv for ground truth testing
python tests/fixtures/download_acl_papers.py

# Run ground truth extraction tests
python tests/test_bib_extraction.py
```

### Test results across 10 real ACL papers

| Metric | Value |
|--------|-------|
| Precision | 73-99% (median ~91%) |
| Recall | 37-90% (median ~73%) |
| Year accuracy | 96-100% |
| False positives | 0-3 per paper |

## Library usage

```python
import asyncio
from pathlib import Path
from halref.config import Config
from halref.extract.ensemble import extract_references
from halref.pipeline import run_check

# Extract references (offline)
config = Config.default()
refs = extract_references(Path("paper.pdf"), config)
for ref in refs:
    print(f"{ref.authors[0]} ({ref.year}). {ref.title}")

# Full verification
report = asyncio.run(run_check([Path("paper.pdf")], config))
for result in report.reports[0].ranked():
    if result.hallucination_score > 0.5:
        print(f"FLAGGED: {result.reference.title}")
        print(f"  Score: {result.hallucination_score:.2f}")
        print(f"  Signals: {result.signal_summary()}")
```

## Privacy

All PDF processing happens locally. No PDF content is ever sent to external services. Only reference metadata (titles, author names, years) is sent to academic search APIs during the `check` command. The `extract` command is fully offline.

## License

MIT
