# infino

Fast search on object storage — SQL, full-text, and vector search — for Node.js.

Synchronous: pass arrays of objects (or apache-arrow `Table`s) in, get plain
records out; pass `{ arrow: true }` to a search or query for an apache-arrow
`Table` instead.

## Install

```sh
npm install infino --registry https://npm-proxy.fury.io/infino/
```

A prebuilt native binary is selected automatically at install time — no Rust
toolchain required. Supported platforms:

| Platform      | Architectures |
| ------------- | ------------- |
| macOS         | x64, arm64    |
| Linux (glibc) | x64, arm64    |

`apache-arrow` is installed as a dependency and used at the boundary (passing
in `Table`s, or `{ arrow: true }` results). Requires Node.js >= 18.

## Usage

```javascript
import { connect, IndexSpec } from "infino";

// A knowledge base your agent retrieves over. "memory://" is in-process;
// use "./data" or "s3://bucket/prefix" to persist.
const db = connect("memory://");

// Tiny stand-in for your embedding model so this runs as-is — a 16-dim
// one-hot by topic. Real embeddings are dense and higher-dimensional.
const embed = (topic) => { const v = Array(16).fill(0.0); v[topic] = 1.0; return v; };

const docs = db.createTable(
  "docs",
  { source: "large_utf8", body: "large_utf8", embedding: { vector: 16 } },
  new IndexSpec().fts("body").vector("embedding", 16, 1, "cosine"),
);

docs.append([
  { source: "help-center", body: "To cancel a subscription, open Settings then Billing.", embedding: embed(0) },
  { source: "help-center", body: "Refunds return to the original payment method.",         embedding: embed(0) },
  { source: "blog",        body: "Enable dark mode under Settings then Appearance.",        embedding: embed(1) },
]);

// Three ways to retrieve context to ground the agent's next answer:
const keyword  = docs.bm25Search("body", "cancel subscription", 5);            // BM25
const semantic = docs.vectorSearch("embedding", embed(0), 5);                  // vector kNN
const billing  = db.querySql("SELECT body FROM docs WHERE source = 'help-center'");  // SQL filter
```

CommonJS works too — `const { connect, IndexSpec } = require("infino");`.

## API

- `connect(uri, options?)` — backend from the URI scheme; S3-compatible
  static credentials via `options = { endpoint, region, accessKey, secretKey }`
  (endpoint requires the other three).
- `Connection`: `createTable(name, schema, IndexSpec)`, `openTable`,
  `dropTable(name, purge?)`, `listTables`, `querySql(sql, { arrow? })`.
- `Table`:
  - `append(data)` — an array of objects or an apache-arrow
    `Table`/`RecordBatch`. One `append` is one commit.
  - `bm25Search(col, q, k, { mode?, projection?, arrow? })` /
    `vectorSearch(col, query, k, { nprobe?, projection?, arrow? })` —
    ranked search; return matching rows as records (or an apache-arrow
    `Table` with `{ arrow: true }`). `query` is a `number[]` or
    `Float32Array`. `projection` (e.g. `["_id", "score"]`) selects the
    returned columns; omit for full rows.
  - `tokenMatch(col, q, { mode?, projection?, arrow? })` /
    `exactMatch(col, value, { projection?, arrow? })` — unranked matching
    rows (`score` is `0`).
  - `schema()` — the table's apache-arrow `Schema`.
- `IndexSpec().fts(col).vector(col, dim, nCent, metric)`.

Schema requirements: FTS columns must be Arrow `LargeUtf8`; vector columns
must be `FixedSizeList<Float32, dim>` with `dim` in `[16, 4096]`.

## Notes

- The API is **synchronous**. In a long-running server, run calls in a
  `worker_thread` so a query doesn't block the event loop.
- `_id` comes back as a JavaScript `bigint`.
```
