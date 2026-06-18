# Agent memory

Give an AI agent durable long-term memory with infino — store memories and
recall them later by **meaning**, by **exact keyword**, or both, and query them
with **SQL** — all in one engine. No separate vector database, keyword index, or
metadata store to wire together.

What it shows:

- **Store** memories as rows: text + embedding + structured fields (`importance`,
  `created_at`) in a single table.
- **Semantic recall** — `vectorSearch` finds the right memory even when the query
  shares no words with it ("dangerous foods" → "allergic to peanuts").
- **Keyword recall** — `bm25Search` nails exact terms like names and IDs.
- **Hybrid recall** — BM25 + vector fused with Reciprocal Rank Fusion, so lexical
  anchors and paraphrased meaning are caught together.
- **SQL over memory** — `querySql` for structured/temporal recall (by importance,
  recency, …) — the view a plain vector store can't give you.

Embeddings come from a small local model (Hugging Face transformers.js,
`all-MiniLM-L6-v2`, 384-dim) — no API key. infino is bring-your-own-vectors, so
swap in any model or embeddings API; just keep the dimension in sync.

## Run

```sh
# 1. build the infino addon once (from the repo root — see the main README)
# 2. install this example's embedding model dependency:
npm install
# 3. run it:
node index.mjs
```

First run downloads the embedding model (~90 MB), cached afterward.
