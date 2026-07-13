# infino

[![npm](https://img.shields.io/npm/v/@infino-ai/infino.svg)](https://www.npmjs.com/package/@infino-ai/infino)
[![Node](https://img.shields.io/node/v/@infino-ai/infino.svg)](https://www.npmjs.com/package/@infino-ai/infino)
[![Downloads](https://img.shields.io/npm/dm/@infino-ai/infino.svg)](https://www.npmjs.com/package/@infino-ai/infino)
[![License](https://img.shields.io/npm/l/@infino-ai/infino.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**SQL, full-text, and vector search over your data on object storage — one engine, no server to run.**

Infino keeps your data in Apache Parquet on object storage (local disk, Amazon
S3, or any S3-compatible store) and runs SQL, full-text (BM25), and vector
search over it from a single system. Each file is a valid Parquet file with BM25
and vector indexes embedded directly inside it; a table composes many such files
with snapshot-isolated reads, append-only writes, and atomic commits. It runs in
your process — there is no daemon, no cluster, and no managed service to operate.

Use it for **RAG**, **agent memory**, **hybrid search**, and **semantic
search**: an embedded **vector database**, **full-text (BM25)** search engine,
and **SQL** query engine in one library.

## Install

```sh
npm install @infino-ai/infino
```

A prebuilt native binary is selected automatically at install time — no Rust
toolchain required. Supported platforms:

| Platform              | Architectures |
| --------------------- | ------------- |
| macOS                 | x64, arm64    |
| Linux (glibc)         | x64, arm64    |
| Linux (musl / Alpine) | x64, arm64    |

Requires Node.js >= 18. `apache-arrow` is installed as a dependency and used at
the boundary (passing in `Table`s, or `{ arrow: true }` results).

## Quickstart

```javascript
import { connect, IndexSpec } from "@infino-ai/infino";

// Connect to a catalog. Use a local path or an S3 URI for durable storage;
// "memory://" is ephemeral and handy for tests.
const db = connect("./data");

// Tiny stand-in for your embedding model so this runs as-is — a 16-dim
// one-hot by topic. Real embeddings are dense and higher-dimensional.
const embed = (topic) => { const v = Array(16).fill(0.0); v[topic] = 1.0; return v; };

// Declare a schema and which columns to index. An `_id` column is added
// automatically — you don't define it.
const docs = db.createTable(
  "docs",
  { source: "large_utf8", body: "large_utf8", embedding: { vector: 16 } },
  new IndexSpec().fts("body").vector("embedding", 16, 1, "cosine"),
);

// Append rows. One append is one atomic commit.
docs.append([
  { source: "help-center", body: "To cancel a subscription, open Settings then Billing.", embedding: embed(0) },
  { source: "help-center", body: "Refunds return to the original payment method.",         embedding: embed(0) },
  { source: "blog",        body: "Enable dark mode under Settings then Appearance.",        embedding: embed(1) },
]);

// Retrieve context to ground an agent's next answer — keyword, vector,
// hybrid (BM25 + vector fused in one pass), or SQL:
const keyword  = docs.bm25Search("body", "cancel subscription", 5);                          // BM25
const semantic = docs.vectorSearch("embedding", embed(0), 5);                                // vector kNN
const hybrid   = docs.hybridSearch("body", "cancel subscription", "embedding", embed(0), 5); // fused
const billing  = db.querySql("SELECT body FROM docs WHERE source = 'help-center'");          // SQL filter
```

CommonJS works too — `const { connect, IndexSpec } = require("@infino-ai/infino");`.

> The API is synchronous. In a long-running server, run calls in a
> [`worker_thread`](https://nodejs.org/api/worker_threads.html) so a query
> doesn't block the event loop.

## Documentation

Full docs, guides, and the API reference live at **[infino.ai/docs](https://infino.ai/docs)**:

- [Quickstart](https://infino.ai/docs/quickstart) — install to first query
- [Core concepts](https://infino.ai/docs/core-concepts) — superfiles, commits, and indexes
- Guides — [Tables & indexing](https://infino.ai/docs/guides/tables) ·
  [Search: BM25, vector, hybrid](https://infino.ai/docs/guides/search) ·
  [Embeddings](https://infino.ai/docs/guides/embeddings) ·
  [Storage & credentials](https://infino.ai/docs/guides/storage)
- [SQL reference](https://infino.ai/docs/sql-reference) — query tables and the search table-valued functions
- [API reference](https://infino.ai/docs/api-reference) — the full Node surface, generated from the package
- [Integrations](https://infino.ai/docs/integrations) — LangChain, CrewAI, Vercel AI SDK, MCP
- [Examples](examples) — runnable agent-memory and hybrid-search-service demos

## Building from source

The binding is built with [napi-rs](https://napi.rs/) and requires a Rust
toolchain.

```sh
cd infino-node
npm install && npm run build && npm test
```

## License

Apache-2.0.
