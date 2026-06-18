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


def load_ms_marco(n_queries: int = 300) -> tuple[list[dict], list[dict]]:
    """Real MS MARCO passage-ranking data with ground-truth relevance labels.

    Source: `microsoft/ms_marco` (v1.1) on the HuggingFace Hub. Each query
    carries a handful of candidate passages, one or more flagged relevant
    (`is_selected == 1`). We flatten those into a passage corpus with stable
    `pid`s plus the set of relevant `pid`s per query — exactly what's needed to
    measure retrieval recall.

    Returns `(passages, queries)` where
      passages = [{"pid": int, "text": str}, ...]
      queries  = [{"query": str, "relevant_pids": list[int]}, ...]
    Queries with no relevant passage are dropped (nothing to score against).
    """
    stream = load_dataset("microsoft/ms_marco", "v1.1", split="validation", streaming=True)
    passages: list[dict] = []
    queries: list[dict] = []
    for row in stream:
        cand = row["passages"]
        relevant = []
        for text, selected in zip(cand["passage_text"], cand["is_selected"]):
            pid = len(passages)
            passages.append({"pid": pid, "text": text})
            if selected == 1:
                relevant.append(pid)
        if relevant:
            queries.append({"query": row["query"], "relevant_pids": relevant})
        if len(queries) >= n_queries:
            break
    return passages, queries
