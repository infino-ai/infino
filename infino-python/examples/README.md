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

### Optional: LLM answers

The retrieval examples run fully without an LLM (they print the grounded
context). To generate answers, configure either backend — `_shared/llm.py`
picks it up automatically, reading from a local `.azure.env` / `.env` if present:

- **Azure AI Foundry** (preferred): `AZURE_AI_ENDPOINT` (the OpenAI-compatible
  `https://<resource>.openai.azure.com/openai/v1` URL), `AZURE_AI_API_KEY`, and
  `DEFAULT_AZURE_MODEL`.
- **OpenAI**: `OPENAI_API_KEY` (optionally `OPENAI_MODEL`).

Keep credentials in an untracked env file — never commit keys.

## Examples

| # | Example | What it shows | Dataset |
| - | ------- | ------------- | ------- |
| 1 | [`01_rag_pdf.ipynb`](01_rag_pdf.ipynb) | The canonical RAG pipeline — chunk, embed, vector-retrieve, ground an answer | arXiv papers |
| 2 | [`02_hybrid_rag.ipynb`](02_hybrid_rag.ipynb) | **Hybrid search**: BM25 + vector fused in one in-engine SQL query, with measured recall@10 | MS MARCO (labeled) |
| 3 | [`03_filtered_rag.ipynb`](03_filtered_rag.ipynb) | **Filtered & multi-tenant** retrieval: vector search + SQL `WHERE` over one table | Amazon product catalog |
| 4 | [`04_chat_rag.ipynb`](04_chat_rag.ipynb) | **Conversational RAG** — multi-turn chat with memory, per-turn hybrid retrieval, cited sources, durable local index | Wikipedia |

The notebooks build on each other (1 → 2 → 3 → 4).

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
