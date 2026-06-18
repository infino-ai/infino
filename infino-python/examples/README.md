# Infino RAG examples

End-to-end Retrieval-Augmented Generation examples built on
[Infino](https://pypi.org/project/infino/) — one engine that runs **SQL,
full-text (BM25), and vector search** over a single copy of your data, stored as
Apache Parquet on local disk or object storage. No separate vector database to
run or keep in sync.

Each example uses a **real public dataset** (pulled from the HuggingFace Hub)
and runs **locally and key-free** — embeddings are computed on-device with
`sentence-transformers`; the index lives on local disk. An LLM answer step is
optional and only used if you set `OPENAI_API_KEY`.

## Setup

```sh
pip install -r requirements.txt
```

## Examples

| # | Example | What it shows | Dataset |
| - | ------- | ------------- | ------- |
| 1 | [`01_rag_pdf.ipynb`](01_rag_pdf.ipynb) | The canonical RAG pipeline — chunk, embed, vector-retrieve, ground an answer | arXiv papers |
| 2 | [`02_hybrid_rag.ipynb`](02_hybrid_rag.ipynb) | **Hybrid search**: BM25 + vector fused in one in-engine SQL query, with measured recall@10 | MS MARCO (labeled) |
| 3 | [`03_filtered_rag.ipynb`](03_filtered_rag.ipynb) | **Filtered & multi-tenant** retrieval: vector search + SQL `WHERE` over one table | Amazon product catalog |
| 4 | [`chat_app/`](chat_app/) | A **Streamlit chatbot** over your documents — hybrid retrieval, cited sources, durable local index | Wikipedia + your uploads |

The notebooks build on each other (1 → 2 → 3); the app composes all three.

## Why one engine

The same Infino table is simultaneously full-text searchable, vector
searchable, and queryable with SQL — over the same rows, one consistency model.
That is what lets hybrid search and metadata filtering happen **inside a single
query** instead of being stitched together across a database, a search cluster,
and a vector store.

## Shared helpers

`_shared/` holds the small pieces every example reuses:

- `embedding.py` — local `all-MiniLM-L6-v2` embeddings (384-dim, cosine)
- `chunking.py` — fixed-size, overlapping text chunker
- `datasets.py` — loaders for the real corpora above

To use a hosted embedder instead of the local model, swap the body of
`embed`/`embed_query` in `_shared/embedding.py` and set `DIM`/`METRIC` to match —
nothing else changes.
