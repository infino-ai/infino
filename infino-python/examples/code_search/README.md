# Code search

Search a corpus of Python functions four ways over a **single**
[Infino](https://pypi.org/project/infino/) table:

- **`exact_match`** — jump to every definition of a function name, across every repo.
- **`vector_search`** — find a function from a natural-language description.
- **`bm25_search`** — rank functions by a keyword in their body.
- **`hybrid_search`** — fuse keyword and meaning with RRF in one SQL call.

One table indexes the function name, source, docstring embedding, and repo
metadata together — no separate symbol index, vector database, or text-search
cluster. The dataset is CodeSearchNet (Python), pulled from the HuggingFace Hub
and indexed on local disk; embeddings run on-device, locally and key-free.

> Setup and the shared helpers are covered in the
> [examples README](../README.md). Run that setup once, then open the notebook.

## Keyword vs meaning

The notebook's centerpiece: the same plain-English intent, run two ways over the
same table. Keyword search matches literal tokens; vector search matches what the
function *does*. For `"remove duplicate elements"` they return entirely different
functions:

```
keyword (BM25 over code)        meaning (vector over docstrings)
  all_files                       Index.get_duplicates      (pandas)
  symlink_remove                  Index._get_unique_index   (pandas)
  _remove_attributes              MultiCategoryProcessor.generate_classes
```

BM25 latches onto *remove* and *elements* and surfaces unrelated code; vector
search finds the de-duplication functions, which never use either word.
`hybrid_search` gives you both signals in a single query.

## Example

| # | Example | What it shows | Dataset |
| - | ------- | ------------- | ------- |
| 1 | [`01_code_search.ipynb`](01_code_search.ipynb) | `exact_match` (symbol lookup), `vector_search` (NL → code), `bm25_search` (keyword in code), and `hybrid_search` (RRF fusion) over one table | CodeSearchNet (Python) |
