"""Real public datasets for the examples, streamed from the HuggingFace Hub.

Each loader takes the first `n` rows; raise `n` to index more. The first call
downloads from the Hub; later calls are cached.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from datasets import load_dataset


def load_arxiv(n: int = 200) -> list[dict]:
    """arXiv ML papers (title + abstract) from `CShorten/ML-ArXiv-Papers`.

    Returns `[{"title": str, "abstract": str}]`, empty abstracts dropped.
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
    """MS MARCO passage ranking (v1.1) from `microsoft/ms_marco`, with labels.

    Flattens candidate passages into a corpus with stable `pid`s and records
    the relevant `pid`s per query (`is_selected == 1`). Returns
    `(passages, queries)`:
      passages = [{"pid": int, "text": str}, ...]
      queries  = [{"query": str, "relevant_pids": list[int]}, ...]
    Queries with no relevant passage are dropped.
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


def load_amazon(n: int = 1200) -> list[dict]:
    """Amazon product catalog from `smartcat/Amazon_Sample_Metadata_2023`.

    Keeps products with a usable price. Returns
    `[{"title", "text", "price", "rating", "category", "store"}]`, where `text`
    is title + description (indexed for search) and the rest are filterable metadata.
    """
    stream = load_dataset(
        "smartcat/Amazon_Sample_Metadata_2023", split="train", streaming=True
    )
    products: list[dict] = []
    for row in stream:
        raw_price = row.get("price")
        if raw_price in (None, "", "None"):
            continue
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            continue
        title = (row.get("title") or "").strip()
        if not title:
            continue

        description = row.get("description") or []
        if isinstance(description, list):
            description = " ".join(description)
        text = f"{title}. {str(description)[:400]}"

        products.append({
            "title": title,
            "text": text,
            "price": price,
            "rating": float(row.get("average_rating") or 0.0),
            "category": str(row.get("main_category") or "Unknown"),
            "store": str(row.get("store") or "Unknown"),
        })
        if len(products) >= n:
            break
    return products


def load_wikipedia(n: int = 100) -> list[dict]:
    """Wikipedia articles (Simple English) from `wikimedia/wikipedia`.

    Returns `[{"title": str, "text": str, "source": str}]`.
    """
    stream = load_dataset(
        "wikimedia/wikipedia", "20231101.simple", split="train", streaming=True
    )
    docs = []
    for row in stream:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        docs.append({
            "title": (row.get("title") or "").strip(),
            "text": text,
            "source": row.get("url") or "wikipedia",
        })
        if len(docs) >= n:
            break
    return docs
