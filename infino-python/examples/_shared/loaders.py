"""Real public datasets for the examples, streamed from the HuggingFace Hub.

Each loader takes the first `n` rows; raise `n` to index more. The first call
downloads from the Hub; later calls are cached.
"""

import datetime as dt
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


# Code bodies are stored truncated: enough for keyword/BM25 search and display
# without bloating the demo table. Raise it if you want to index whole functions.
CODE_MAX_CHARS = 2000

# The dataset streams in repo order, so the first rows cluster into a handful of
# projects. Cap functions per repo to spread the sample across many codebases.
MAX_FUNCS_PER_REPO = 40


def load_code_search(n: int = 800) -> list[dict]:
    """Python functions from `Nan-Do/code-search-net-python` (a Parquet mirror
    of CodeSearchNet).

    Keeps functions that have both a body and a docstring (the docstring is what
    we embed for natural-language search), capped per repo for variety. Returns
    `[{"func_name", "code", "docstring", "summary", "repo", "url", "language"}]`.
    """
    stream = load_dataset("Nan-Do/code-search-net-python", split="train", streaming=True)
    funcs: list[dict] = []
    per_repo: dict[str, int] = {}
    for row in stream:
        func_name = (row.get("func_name") or "").strip()
        code = (row.get("code") or row.get("original_string") or "").strip()
        docstring = (row.get("docstring") or "").strip()
        if not (func_name and code and docstring):
            continue
        repo = str(row.get("repo") or "unknown")
        if per_repo.get(repo, 0) >= MAX_FUNCS_PER_REPO:
            continue
        per_repo[repo] = per_repo.get(repo, 0) + 1
        funcs.append({
            "func_name": func_name,
            "code": code[:CODE_MAX_CHARS],
            "docstring": docstring,
            "summary": (row.get("summary") or docstring.splitlines()[0]).strip(),
            "repo": repo,
            "url": str(row.get("url") or ""),
            "language": str(row.get("language") or "python"),
        })
        if len(funcs) >= n:
            break
    return funcs


def _to_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_hackernews(n: int = 4000) -> list[dict]:
    """Hacker News stories from `julien040/hacker-news-posts` (posts only).

    The `time` field is a Unix epoch; we derive a `"YYYY-MM-DD HH:MM:SS"`
    string plus `year` / `month` for SQL bucketing. `points` is the upvote count
    (named to avoid clashing with the BM25 relevance `score`). Returns
    `[{"title", "by", "url", "points", "num_comments", "time", "month", "year"}]`.
    """
    stream = load_dataset("julien040/hacker-news-posts", split="train", streaming=True)
    stories: list[dict] = []
    for row in stream:
        title = (row.get("title") or "").strip()
        if not title:
            continue
        try:
            when = dt.datetime.fromtimestamp(int(row["time"]), dt.timezone.utc)
        except (TypeError, ValueError, KeyError):
            continue
        stories.append({
            "title": title,
            "by": str(row.get("author") or "unknown"),
            "url": str(row.get("url") or ""),
            "points": _to_int(row.get("score")),
            "num_comments": _to_int(row.get("comments")),
            "time": when.strftime("%Y-%m-%d %H:%M:%S"),
            "month": when.strftime("%Y-%m"),
            "year": when.year,
        })
        if len(stories) >= n:
            break
    return stories
