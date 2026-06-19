# Infino examples

Runnable examples built on [Infino](https://pypi.org/project/infino/) — one
engine that runs **SQL, full-text (BM25), and vector search** over a single copy
of your data, stored as Apache Parquet on local disk or object storage. No
separate vector database to run or keep in sync.

Each example uses a **real public dataset** (pulled from the HuggingFace Hub)
and runs **locally and key-free**.

## Categories

| Folder | What it covers |
| ------ | -------------- |
| [`rag/`](rag/) | Retrieval-Augmented Generation — chunk, embed, retrieve (vector / hybrid / filtered / conversational) and ground an answer |
| [`code_search/`](code_search/) | Code search — exact symbol lookup, natural-language (vector), keyword (BM25), and hybrid search over one table |
| [`analytics/`](analytics/) | SQL analytics + full-text — `GROUP BY` time-series, top-N, and leaderboards alongside BM25 search, no vector index |
| [`inventory/`](inventory/) | Mutable data — keep a live inventory current with `update` / `delete` / `optimize`, durable across reconnect |

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

## Shared helpers

`_shared/` holds the small pieces every example reuses (it lives at the
`examples/` root and is shared across all categories):

- `embedding.py` — local `all-MiniLM-L6-v2` embeddings (384-dim, cosine)
- `chunking.py` — fixed-size, overlapping text chunker
- `loaders.py` — loaders for the real corpora above (HuggingFace Hub)
- `sql.py` — tiny SQL helpers (literal quoting, empty-safe query)
- `llm.py` — optional answer generation (Azure AI Foundry or OpenAI)

Notebooks live one level down (e.g. `rag/`) and add the `examples/` root to
`sys.path` in their first code cell so `from _shared.… import …` resolves.

To use a hosted embedder instead of the local model, update the `embed` /
`embed_query` bodies in `embedding.py` and set `DIM` / `METRIC` to match the new
model — then use those same values in each notebook's `IndexSpec(...)`.
