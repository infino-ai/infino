# Hybrid search API with infino

A small **embedded search service**: infino runs inside your Node process, so you
get production-style hybrid (keyword + vector) search over your data with **no
separate search server to deploy**.

It indexes a **real product catalog** — a sample of Amazon product metadata pulled
key-free from the [HuggingFace Hub](https://huggingface.co/datasets/smartcat/Amazon_Sample_Metadata_2023)
(the same source the Python examples use) — and serves:

```
GET /search?q=<query>&k=<n>  →  { query, results: [{ title, category, price, rating, score }] }
```

Ranking is infino's native single-pass `hybrid_search` — BM25 and vector
similarity fused **in the engine**, in one query. Keyword-only search misses
intent-style queries ("a thoughtful birthday gift"); vector-only search misses
exact brand/keyword matches. Hybrid gets both.

## Run

Build the Node addon first (see the [`infino-node` README](../../README.md)),
then from this folder:

```sh
npm install
node index.mjs
```

`npm install` brings in `infino` and a small local embedder,
[`all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2) — no API key.
On first run it downloads the embedding model (~90 MB) and a catalog sample from
the Hub (set `CATALOG_N` to change the sample size, default 200), then caches the
model. The server listens on `:3000` (set `PORT` to change it):

```sh
curl 'http://localhost:3000/search?q=gift+for+someone+who+loves+cooking'
curl 'http://localhost:3000/search?q=something+to+keep+skin+moisturized&k=10'
```

(The catalog is whatever sample loaded, so exact results vary with the data.)

## As a test

On startup the server runs a self-check (queries itself and asserts the endpoint
returns well-formed hybrid results), so it doubles as an end-to-end smoke test.
Because results come from a live dataset sample, the check asserts **shape**, not
an exact hit. Run with `SMOKE=1` to do the check and exit instead of serving —
that's how CI gates it (`make node-example`):

```sh
SMOKE=1 node index.mjs
```
