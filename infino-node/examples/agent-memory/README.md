# Agent memory with infino

A small, runnable example of using infino as an AI agent's **long-term memory**.

An agent that talks with a user over weeks needs to remember the conversation and
recall the right bits later. A plain vector store gives you semantic search and
nothing else. infino stores each message once (its text plus an embedding) and
serves three kinds of recall over the *same* table:

1. **Hybrid search** — meaning *and* keywords, fused in a single query via the
   `hybrid_search` SQL table function.
2. **SQL over memory** — the structured questions an index alone can't answer:
   *"who said how much?"*, filter by session/date (`GROUP BY`, `WHERE`, `COUNT`).
3. **Mutation** — forget part of the history (`delete`).

That combination — one engine doing semantic recall *and* SQL over the same data
— is what makes infino a good fit for agent memory.

The example loads one real, months-long, multi-session conversation from
[LOCOMO](https://github.com/snap-research/locomo), a public long-term
conversational-memory dataset — a natural stand-in for an agent's chat history.
Each message becomes a memory, timestamped with its session.

## Run

Build the Node addon first (see the [`infino-node` README](../../README.md)),
then from this folder:

```sh
npm install
node index.mjs
```

`npm install` brings in `infino` and a small local embedder,
[`all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2) — no API key.
On first run it downloads the embedding model (~90 MB) and the conversation
(~3 MB), then caches the model. (Within this repo the `infino` dependency points
at the workspace build.)

The script ends with a few `assert` checks, so it doubles as an end-to-end smoke
test — a non-zero exit means a regression (it runs in CI via `make node-example`).
