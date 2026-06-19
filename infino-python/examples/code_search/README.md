# Code search

Search real open-source code four ways over a **single** [Infino](https://pypi.org/project/infino/)
table — exact symbol lookup, natural-language (vector) search, keyword (BM25)
search inside function bodies, and a fused **hybrid** search — with no separate
symbol index, vector database, or text-search cluster to keep in sync.

The corpus is real Python functions from CodeSearchNet (name, source, docstring,
and the repo each came from), pulled from the HuggingFace Hub and indexed on
local disk. Embeddings are computed on-device — runs locally and key-free.

> Setup and the shared helpers are covered in the
> [examples README](../README.md). Run that setup once, then open the notebook.

## Example

| # | Example | What it shows | Dataset |
| - | ------- | ------------- | ------- |
| 1 | [`01_code_search.ipynb`](01_code_search.ipynb) | `exact_match` (symbol lookup), `vector_search` (NL → code), `bm25_search` (keyword in code), and `hybrid_search` (RRF fusion) over one table | CodeSearchNet (Python) |

## Why one engine

The same rows are exactly matchable, full-text searchable, vector searchable,
and SQL-queryable at once. A query like *"train a classifier on a folder of
images"* finds the right function by **meaning**; `exact_match("func_name", …)`
jumps to **every** definition of a name; BM25 finds functions whose **body**
mentions a specific API; and `hybrid_search` fuses keyword and meaning in one
SQL call — all against one copy of the data.
