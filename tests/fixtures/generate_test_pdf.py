"""Generate test PDFs with ACL-style references for testing extraction.

Creates two PDFs:
1. test_paper_real.pdf - All real references (should score low)
2. test_paper_hallucinated.pdf - Mix of real and hallucinated references (should flag hallucinated ones)

Usage:
    python tests/fixtures/generate_test_pdf.py
"""

import subprocess
import tempfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent

# Filler content to fill pages 1-8
LIPSUM = r"""
\section{Introduction}
Natural language processing has seen remarkable advances in recent years.
Large language models have demonstrated impressive capabilities across a wide range of tasks,
from text generation to question answering and beyond.
This paper presents a comprehensive survey of methods for detecting hallucinated
references in academic papers, a growing concern as AI-assisted writing becomes more prevalent.

The problem of hallucinated references is particularly insidious because the generated citations
often look plausible to human reviewers. They may contain real author names combined with
fictitious titles, or real titles with incorrect author orderings. Our work addresses this
challenge by combining multiple extraction backends with a multi-API verification pipeline.

\section{Background}
\subsection{Reference Extraction}
Extracting structured references from PDF documents has been studied extensively.
GROBID remains the gold standard for reference extraction from scientific documents,
achieving F1 scores of 0.87-0.90 on standard benchmarks. Other approaches include
rule-based systems, CRF-based sequence labeling, and more recently, transformer-based models.

The ACL Anthology provides a comprehensive collection of NLP research papers with structured
metadata, making it an invaluable resource for validation. The Semantic Scholar Academic Graph
indexes over 250 million papers with rich metadata including citation contexts.

\subsection{Citation Verification}
Prior work on citation verification has focused primarily on detecting broken URLs
and missing DOIs. More recent approaches leverage academic search APIs to verify
that cited papers actually exist. However, existing tools often rely on exact matching,
which fails to detect subtle hallucinations such as incorrect author orderings or
slightly modified titles.

\subsection{Hallucination Detection in LLM Outputs}
Large language models are known to produce hallucinated content, including fabricated
citations. Studies have shown that GPT-based models can generate plausible-looking
references that combine real author names with non-existent paper titles. This problem
is exacerbated in the academic domain where the consequences of citing non-existent
work can be severe.

\section{Methodology}
\subsection{Multi-Pronged Reference Extraction}
Our approach uses multiple extraction backends operating in parallel:
\begin{enumerate}
\item \textbf{pdfplumber}: Lightweight text extraction with targeted page support
\item \textbf{Docling}: Layout-aware extraction using deep learning models
\item \textbf{Marker}: ML-based PDF-to-markdown conversion
\end{enumerate}

Each backend produces reference section text, which is then split into individual
reference strings using hanging indent detection and blank line splitting.

\subsection{Field Parsing Pipeline}
Individual reference strings are parsed into structured fields using a cascade of parsers:
\begin{itemize}
\item Regex parser tuned for ACL format
\item Heuristic parser with name normalization
\item Optional LLM parser for difficult cases
\item API-assisted validation via CrossRef
\end{itemize}

\subsection{Multi-API Verification}
Extracted references are verified against five academic databases:
Semantic Scholar, CrossRef, DBLP, OpenAlex, and the ACL Anthology.
An agentic retry strategy reformulates queries when initial searches fail.

\subsection{Hallucination Scoring}
We compute a composite hallucination score based on weighted signals:
title similarity, author overlap, author order correctness, year matching,
and cross-API consensus. The score ranges from 0 (confirmed real) to 1
(almost certainly hallucinated).

\section{Experiments}
We evaluated our system on a dataset of 100 papers containing a mix of real
and hallucinated references. The hallucinated references were generated using
several perturbation strategies:
\begin{enumerate}
\item Swapping author order while keeping the title
\item Modifying the title slightly while keeping authors
\item Combining authors from one paper with the title of another
\item Generating completely fictitious references
\end{enumerate}

\subsection{Results}
Our system achieved 94\% precision and 89\% recall in detecting hallucinated
references. The most challenging cases were subtle author order swaps, where
the system achieved 82\% recall. Completely fictitious references were detected
with 99\% precision.

\subsection{Analysis}
The multi-API approach proved critical: references that appeared in multiple
databases received higher confidence scores, while those found in none were
almost always hallucinated. The agentic retry strategies improved recall by
7\% compared to direct title matching alone.

\subsection{Ablation Study}
We conducted an ablation study to evaluate the contribution of each component.
Removing the heuristic parser reduced extraction quality by 12\%.
Disabling the agentic retry reduced recall by 7\%.
Using only a single API reduced precision by 15\%.

\section{Discussion}
Our results demonstrate that multi-pronged verification is essential for
reliable hallucination detection. The combination of multiple extraction
backends with multiple verification APIs provides robustness against
individual component failures.

\subsection{Limitations}
Our approach has several limitations. First, it requires network access
for API verification, though the extraction phase is fully local.
Second, very recent papers may not yet be indexed in all databases.
Third, the scoring weights were tuned on a limited dataset and may
need adjustment for different domains.

\subsection{Future Work}
Future directions include integrating embedding-based matching for
semantic title comparison, supporting additional document formats
beyond PDF, and developing a browser extension for real-time
verification during paper review.

\section{Related Work}
Several tools have been developed for reference verification.
RefChecker uses LLM-powered extraction with multi-API verification.
CiteAudit provides the first open benchmark for hallucinated citation detection.
GPTZero's hallucination detector searches across open web and academic databases.
Our work differs in its focus on fully local extraction and agentic retry strategies.

\section{Conclusion}
We presented a comprehensive system for detecting hallucinated references
in academic papers. Our multi-pronged approach combines multiple extraction
backends, field parsers, and verification APIs with agentic retry strategies.
The system achieves high precision and recall while keeping all PDF processing
fully local for privacy. We release our tool as open-source software to help
the academic community maintain the integrity of scholarly references.

\section*{Acknowledgments}
We thank the reviewers for their helpful feedback.
"""

# Real references (verifiable in Semantic Scholar, DBLP, etc.)
REAL_REFERENCES = [
    r"Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. {BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding. In \textit{Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies}, pages 4171--4186.",
    r"Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, {\L}ukasz Kaiser, and Illia Polosukhin. 2017. Attention is All You Need. In \textit{Advances in Neural Information Processing Systems}, volume 30.",
    r"Brown, Tom, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and others. 2020. Language Models are Few-Shot Learners. In \textit{Advances in Neural Information Processing Systems}, volume 33, pages 1877--1901.",
    r"Liu, Yinhan, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. {R}o{BERT}a: A Robustly Optimized BERT Pretraining Approach. \textit{arXiv preprint arXiv:1907.11692}.",
    r"Raffel, Colin, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. \textit{Journal of Machine Learning Research}, 21(140):1--67.",
    r"Wolf, Thomas, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R{\'e}mi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-Art Natural Language Processing. In \textit{Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations}, pages 38--45.",
    r"Lewis, Mike, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. {BART}: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. In \textit{Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics}, pages 7871--7880.",
    r"Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. 2013. Distributed Representations of Words and Phrases and their Compositionality. In \textit{Advances in Neural Information Processing Systems}, volume 26.",
    r"Pennington, Jeffrey, Richard Socher, and Christopher Manning. 2014. {G}lo{V}e: Global Vectors for Word Representation. In \textit{Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing}, pages 1532--1543.",
    r"Peters, Matthew E., Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep Contextualized Word Representations. In \textit{Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies}, pages 2227--2237.",
]

# Hallucinated references (should be flagged)
HALLUCINATED_REFERENCES = [
    # Wrong author order (real paper, swapped authors)
    r"Chang, Ming-Wei, Jacob Devlin, Kristina Toutanova, and Kenton Lee. 2019. {BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding. In \textit{Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics}, pages 4171--4186.",
    # Completely fictitious
    r"Zhang, Wei, Sarah Johnson, and Michael Torres. 2023. Adaptive Cross-Lingual Transfer Learning for Low-Resource Sentiment Analysis. In \textit{Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics}, pages 892--905.",
    # Wrong year (real paper, wrong year)
    r"Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, {\L}ukasz Kaiser, and Illia Polosukhin. 2019. Attention is All You Need. In \textit{Advances in Neural Information Processing Systems}, volume 30.",
    # Fictitious with real-sounding authors
    r"Park, Jinhyuk, Elena Voronova, and David Martinez-Gonzalez. 2022. Neural Reference Verification with Contrastive Citation Embeddings. In \textit{Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing}, pages 1543--1557.",
    # Real authors, completely fake title
    r"Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2020. Recursive Hierarchical Attention Networks for Document-Level Machine Translation. In \textit{Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics}, pages 3421--3435.",
]


def generate_latex(references: list[str], filename: str) -> str:
    """Generate a complete LaTeX document with ACL-style formatting."""
    ref_items = "\n\n".join(
        rf"\bibitem{{{f'ref{i+1}'}}} {ref}" for i, ref in enumerate(references)
    )

    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{times}}
\usepackage{{geometry}}
\geometry{{a4paper, margin=2.5cm}}
\usepackage{{natbib}}
\usepackage{{lipsum}}

\title{{A Multi-Pronged Approach to Detecting Hallucinated References\\in Academic Papers}}
\author{{Test Author \\ Test University \\ test@example.com}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We present a comprehensive system for detecting hallucinated references
in academic papers. Our approach combines multiple PDF extraction backends
with a multi-API verification pipeline and agentic retry strategies.
We evaluate on a dataset of papers containing both real and fabricated
citations, achieving 94\% precision and 89\% recall. The system produces
a ranked list of references by hallucination likelihood, enabling
efficient review of potentially problematic citations.
\end{{abstract}}

{LIPSUM}

\begin{{thebibliography}}{{99}}

{ref_items}

\end{{thebibliography}}

\end{{document}}
"""


def compile_latex(tex_content: str, output_name: str) -> Path:
    """Compile LaTeX to PDF."""
    output_path = FIXTURES_DIR / f"{output_name}.pdf"

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "paper.tex"
        tex_path.write_text(tex_content, encoding="utf-8")

        # Run pdflatex twice for references
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

        pdf_path = Path(tmpdir) / "paper.pdf"
        if pdf_path.exists():
            import shutil
            shutil.copy2(pdf_path, output_path)
            print(f"Generated: {output_path}")
            return output_path
        else:
            print(f"Failed to compile {output_name}")
            print(result.stdout[-500:] if result.stdout else "")
            print(result.stderr[-500:] if result.stderr else "")
            return None


def main():
    # Paper 1: All real references
    tex1 = generate_latex(REAL_REFERENCES, "test_paper_real")
    compile_latex(tex1, "test_paper_real")

    # Paper 2: Mix of real and hallucinated references
    mixed = REAL_REFERENCES[:5] + HALLUCINATED_REFERENCES
    tex2 = generate_latex(mixed, "test_paper_hallucinated")
    compile_latex(tex2, "test_paper_hallucinated")

    print("\nDone! Test PDFs generated in tests/fixtures/")


if __name__ == "__main__":
    main()
