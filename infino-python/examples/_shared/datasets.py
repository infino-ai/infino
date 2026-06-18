"""Real public datasets for the RAG examples, pulled from the HuggingFace Hub.

Every loader streams a real, named dataset and takes a small deterministic
sample so the notebooks run in seconds with no large files committed. Bump the
`n` argument (or set it to `None` where supported) to scale up to the full
corpus — the example code does not change.
"""

import os

# Quiet the public-Hub access notices; these datasets need no auth token.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from datasets import load_dataset


def load_arxiv(n: int = 200) -> list[dict]:
    """Real arXiv ML papers (title + abstract).

    Source: `CShorten/ML-ArXiv-Papers` on the HuggingFace Hub. Streamed so we
    never download the whole set; the first `n` rows are deterministic.

    Returns a list of `{"title": str, "abstract": str}` with empty abstracts
    dropped.
    """
    stream = load_dataset("CShorten/ML-ArXiv-Papers", split="train", streaming=True)
    papers = []
    for row in stream:
        title = (row.get("title") or "").strip()
        abstract = (row.get("abstract") or "").strip()
        if not abstract:
            continue
        papers.append({"title": title, "abstract": abstract})
        if len(papers) >= n:
            break
    return papers
