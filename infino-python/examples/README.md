# Infino RAG examples

End-to-end Retrieval-Augmented Generation examples built on
[Infino](https://pypi.org/project/infino/) — one engine that runs **SQL,
full-text (BM25), and vector search** over a single copy of your data, stored as
Apache Parquet on local disk or object storage. No separate vector database to
run or keep in sync.

Each example uses a **real public dataset** (pulled from the HuggingFace Hub)
and runs **locally and key-free** — embeddings are computed on-device with
`sentence-transformers` and the index lives on local disk. Generating an answer
with an LLM is optional (see [below](#optional-llm-answers)); without one, the
examples print the retrieved context.

## Setup

```sh
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the embedding model (~90 MB) and the dataset sample, so
the first cell can take a minute; later runs use the cache.

### Optional: LLM answers

To generate answers (not just retrieve context), configure either backend —
`_shared/llm.py` picks it up automatically, reading from a local
`.azure.env` / `.env` file if present:

- **Azure AI Foundry** (preferred): `AZURE_AI_ENDPOINT` (the OpenAI-compatible
  `https://<resource>.openai.azure.com/openai/v1` URL), `AZURE_AI_API_KEY`, and
  `DEFAULT_AZURE_MODEL`.
- **OpenAI**: `OPENAI_API_KEY` (optionally `OPENAI_MODEL`).

Keep credentials in an untracked env file — never commit keys.

## Examples

Run them in order — each builds on the last.

| # | Example | What it shows | Dataset |
| - | ------- | ------------- | ------- |
| 1 | [`01_rag_pdf.ipynb`](01_rag_pdf.ipynb) | The canonical RAG pipeline — chunk, embed, vector-retrieve, ground an answer | arXiv papers |
| 2 | [`02_hybrid_rag.ipynb`](02_hybrid_rag.ipynb) | **Hybrid search** — BM25 + vector fused in one SQL call, scored against ground-truth labels | MS MARCO |
| 3 | [`03_filtered_rag.ipynb`](03_filtered_rag.ipynb) | **Filtered & multi-tenant** retrieval — vector search + `WHERE` filters and a keyword pushdown | Amazon product catalog |
| 4 | [`04_chat_rag.ipynb`](04_chat_rag.ipynb) | **Conversational RAG** — multi-turn chat with memory, per-turn hybrid retrieval, cited sources | Wikipedia |

## Why one engine

The same Infino table is simultaneously full-text searchable, vector searchable,
and SQL-queryable — over the same rows, one consistency model. So hybrid
retrieval and metadata filters run in **one SQL statement** against one copy of
the data, instead of being stitched together across a database, a search
cluster, and a vector store that you keep in sync.

## Scaling

The examples use small samples (100–1,200 rows) so they finish in seconds. To go
bigger, raise the `n` argument in the `load_*` calls. Embedding is the main cost
on a laptop (a few minutes per ~100k chunks on CPU) — batch it or switch to a
hosted embedder for large corpora. Infino itself indexes and queries millions of
rows; the data lives on disk (or object storage), not in RAM.

## Shared helpers

`_shared/` holds the small pieces every example reuses:

- `embedding.py` — local `all-MiniLM-L6-v2` embeddings (384-dim, cosine)
- `chunking.py` — fixed-size, overlapping text chunker
- `loaders.py` — loaders for the real corpora above (HuggingFace Hub)
- `sql.py` — tiny SQL helpers (literal quoting, empty-safe query)
- `llm.py` — optional answer generation (Azure AI Foundry or OpenAI)

To use a hosted embedder instead of the local model, update the `embed` /
`embed_query` bodies in `embedding.py` and set `DIM` / `METRIC` to match the new
model — then use those same values in each notebook's `IndexSpec(...)`.
