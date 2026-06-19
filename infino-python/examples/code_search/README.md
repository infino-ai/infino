# Code search

Search a corpus of Python functions four ways over a **single**
[Infino](https://pypi.org/project/infino/) table:

- **`exact_match`** — jump to every definition of a function name.
- **`vector_search`** — find a function from a natural-language description.
- **`bm25_search`** — rank functions by a keyword in their body.
- **`hybrid_search`** — fuse keyword and meaning with RRF in one SQL call.

One table indexes the function name, source, docstring embedding, and repo
metadata together — no separate symbol index, vector database, or text-search
cluster. The dataset is CodeSearchNet (Python), pulled from the HuggingFace Hub
and indexed on local disk; embeddings run on-device, locally and key-free.

> Setup and the shared helpers are covered in the
> [examples README](../README.md). Run that setup once, then open the notebook.

## Example

| # | Example | What it shows | Dataset |
| - | ------- | ------------- | ------- |
| 1 | [`01_code_search.ipynb`](01_code_search.ipynb) | `exact_match` (symbol lookup), `vector_search` (NL → code), `bm25_search` (keyword in code), and `hybrid_search` (RRF fusion) over one table | CodeSearchNet (Python) |
