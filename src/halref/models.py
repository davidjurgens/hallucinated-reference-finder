"""Data models for halref."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Author(BaseModel):
    """A parsed author name."""

    first: str = ""
    last: str = ""
    full: str = ""

    def normalized_last(self) -> str:
        return self.last.strip().lower()

    def __str__(self) -> str:
        return self.full or f"{self.first} {self.last}".strip()


class Reference(BaseModel):
    """A single parsed reference from a PDF."""

    raw_text: str = ""
    title: str = ""
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    pages: str = ""
    source_index: int = 0
    extraction_confidence: float = 0.0

    def first_author_last(self) -> str:
        if self.authors:
            return self.authors[0].normalized_last()
        return ""


class APISource(str, Enum):
    SEMANTIC_SCHOLAR = "semantic_scholar"
    CROSSREF = "crossref"
    DBLP = "dblp"
    OPENALEX = "openalex"
    ACL_ANTHOLOGY = "acl_anthology"


class APIMatch(BaseModel):
    """A single match returned by one API source."""

    source: APISource
    title: str = ""
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    confidence: float = 0.0
    raw_response: dict = Field(default_factory=dict)


class HallucinationSignal(BaseModel):
    """One signal contributing to the hallucination score."""

    name: str
    value: float  # 0.0 = no issue, 1.0 = strong signal
    weight: float
    description: str = ""


class MatchResult(BaseModel):
    """Aggregated verification result for one reference."""

    reference: Reference
    api_matches: list[APIMatch] = Field(default_factory=list)
    best_match: APIMatch | None = None
    title_similarity: float = 0.0
    author_overlap: float = 0.0
    author_order_correct: bool = True
    first_author_match: bool = True
    year_match: bool = True
    hallucination_score: float = 0.0
    signals: list[HallucinationSignal] = Field(default_factory=list)
    strategies_used: list[str] = Field(default_factory=list)

    def signal_summary(self) -> list[str]:
        return [s.description for s in self.signals if s.value > 0.3]


class VerificationReport(BaseModel):
    """Full output for one PDF."""

    input_file: str
    total_references: int = 0
    flagged_count: int = 0
    results: list[MatchResult] = Field(default_factory=list)
    extraction_methods: list[str] = Field(default_factory=list)
    apis_used: list[str] = Field(default_factory=list)

    def flagged(self, threshold: float = 0.5) -> list[MatchResult]:
        return [r for r in self.results if r.hallucination_score >= threshold]

    def ranked(self) -> list[MatchResult]:
        return sorted(self.results, key=lambda r: r.hallucination_score, reverse=True)


class BatchReport(BaseModel):
    """Summary across multiple PDFs."""

    reports: list[VerificationReport] = Field(default_factory=list)
    total_files: int = 0
    total_references: int = 0
    total_flagged: int = 0

    def summary_rows(self, threshold: float = 0.5) -> list[dict]:
        rows = []
        for report in self.reports:
            flagged = report.flagged(threshold)
            rows.append({
                "file": report.input_file,
                "total_refs": report.total_references,
                "flagged": len(flagged),
                "max_score": max((r.hallucination_score for r in report.results), default=0.0),
            })
        return rows
