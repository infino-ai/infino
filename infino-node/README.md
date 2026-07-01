# infino

**SQL, full-text, and vector search over your data on object storage — one engine, no server to run.**

Infino keeps your data in Apache Parquet on object storage (local disk, Amazon
S3, or any S3-compatible store) and runs SQL, full-text (BM25), and vector
search over it from a single system. Each file is a valid Parquet file with BM25
and vector indexes embedded directly inside it; a table composes many such files
with snapshot-isolated reads, append-only writes, and atomic commits. It runs in
your process — there is no daemon, no cluster, and no managed service to operate.

Use it for **RAG**, **agent memory**, **hybrid search**, and **semantic search**:
it's an embedded **vector database**, **full-text (BM25)** search engine, and
**SQL** query engine in one library — no separate vector database or search
server to run.

Synchronous, Arrow at the boundary: pass arrays of objects (or apache-arrow
`Table`s) in, get plain records out; pass `{ arrow: true }` to a search or query
for an apache-arrow `Table` instead.

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

`apache-arrow` is installed as a dependency and used at the boundary (passing in
`Table`s, or `{ arrow: true }` results). Requires Node.js >= 18.

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

// Three ways to retrieve context to ground an agent's next answer:
const keyword  = docs.bm25Search("body", "cancel subscription", 5);            // BM25
const semantic = docs.vectorSearch("embedding", embed(0), 5);                  // vector kNN
const billing  = db.querySql("SELECT body FROM docs WHERE source = 'help-center'");  // SQL filter
```

CommonJS works too — `const { connect, IndexSpec } = require("@infino-ai/infino");`.

## Examples

Runnable, end-to-end examples in [`examples/`](examples) (each its own folder
with a README; build the addon first, then `npm install && node index.mjs`):

- [`agent-memory/`](examples/agent-memory) — infino as an AI agent's long-term
  memory: load a real multi-session conversation, then recall it with hybrid
  search, query it with SQL (`GROUP BY`, filters), and forget parts of it.
- [`hybrid-search-api/`](examples/hybrid-search-api) — an embedded HTTP search
  service over a real product catalog, ranked by native `hybrid_search` — with
  no separate search server to run.

## Core concepts

- **Connection** — a handle to a catalog (a set of tables under one URI). Open
  it with `connect(uri)`.
- **Table** — an append-only, snapshot-isolated collection of rows. Each table
  carries an auto-generated `_id` column.
- **IndexSpec** — declares which columns are full-text (BM25) and which are
  vector indexed. Columns without an index are still stored, filterable in SQL,
  and returnable via projection.
- **Commits** — every `append`, `update`, and `delete` is a single atomic
  commit. Readers see a consistent snapshot and are never torn by a concurrent
  write.
- **Arrow at the boundary** — searches return plain records (or an apache-arrow
  `Table` with `{ arrow: true }`); `append` and `update` accept an array of
  objects or an apache-arrow `Table` / `RecordBatch`.

## Full-text search

```javascript
const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
docs.append([{ title: "the quick brown fox" }, { title: "a lazy dog" }]);

// Ranked BM25 — higher score is a better match.
docs.bm25Search("title", "quick fox", 10);                  // OR by default
docs.bm25Search("title", "quick fox", 10, { mode: "and" }); // require all terms

// Unranked matching (score is 0): every row containing the term(s),
// or an exact whole-value match.
docs.tokenMatch("title", "fox");
docs.exactMatch("title", "the quick brown fox");

// Count matches without materializing any rows.
docs.count("title", "fox");                                 // OR by default
docs.count("title", "quick fox", { mode: "and" });         // require all terms
```

## Vector search

Vector columns are `FixedSizeList<Float32, dim>` with `dim` in `[16, 4096]`. The
distance metric is fixed when you declare the index (`"cosine"`, `"l2sq"`, or
`"negdot"`); for vector results a smaller score is nearer. The query vector is a
`number[]` or `Float32Array`.

```javascript
const spec = new IndexSpec().vector("emb", 384, 256, "cosine"); // (column, dim, nCent, metric)
const vecs = db.createTable("vecs", { emb: { vector: 384 } }, spec);

vecs.vectorSearch("emb", queryVector, 10);                    // top-10 nearest
vecs.vectorSearch("emb", queryVector, 10, { nprobe: 32 });    // probe more partitions (recall)
vecs.vectorSearch("emb", queryVector, 10, { rerankMult: 4 }); // wider exact-rerank pool (recall)
```

**Filtered vector search.** Restrict the kNN to rows matching a text predicate —
a pushdown *pre-filter*, so you get the nearest *matching* rows (not a
post-filter over the global top-k). The filter `column` must be FTS-indexed.

```javascript
vecs.vectorSearch("emb", queryVector, 10, {
  filter: { column: "title", query: "billing", mode: "or" },
});
```

## Hybrid search

Combine BM25 and vector search in **one call** — a single pass over both
indexes, fused inside the engine with reciprocal-rank fusion (no separate
reranker service, no two round-trips). Keyword-only search misses paraphrases;
vector-only search misses exact terms — hybrid gets both. Results come back
best-first with a fused `score` (higher is better).

```javascript
const spec = new IndexSpec().fts("body").vector("emb", 384, 256, "cosine");
const docs = db.createTable("docs", { body: "large_utf8", emb: { vector: 384 } }, spec);
docs.append([{ body: "To cancel a subscription, open Settings then Billing.", emb: embed(/* … */) }]);

// hybridSearch(textColumn, textQuery, vectorColumn, vectorQuery, k, opts?)
docs.hybridSearch("body", "cancel subscription", "emb", embed("how do I stop my plan?"), 10);
// opts: { mode } tunes the BM25 side, { nprobe } the vector side.

// The same fusion is also a SQL table function, so it composes in a query:
const qvec = embed("how do I stop my plan?").join(",");
db.querySql(
  `SELECT _id, score FROM hybrid_search('docs', 'body', 'cancel subscription', 'emb', '${qvec}', 10)`,
);
```

For a complete, runnable hybrid search service see the
[`hybrid-search-api` example](examples/hybrid-search-api).

## SQL

Run SQL across the catalog's tables for analytics and filtering; the search
functions are also available as SQL table functions. Results come back as plain
records (or an apache-arrow `Table` with `{ arrow: true }`).

```javascript
db.querySql("SELECT COUNT(*) AS n FROM docs");
db.querySql("SELECT title FROM docs WHERE title = 'a lazy dog'");

// The search methods are also SQL table functions — bm25_search, vector_search,
// and hybrid_search (see "Hybrid search" above) — so you can filter, join, and
// aggregate over search results.
db.querySql("SELECT _id, score FROM bm25_search('docs', 'title', 'fox', 10)");
```

## Projections

By default a search returns just `_id` and `score` — no row data is decoded.
Name the columns you want to materialize:

```javascript
docs.bm25Search("title", "fox", 10);                                   // _id + score only
docs.bm25Search("title", "fox", 10, { projection: ["_id", "title", "score"] });
```

## Updates and deletes

Mutations require durable storage (a local path or object store, not
`memory://`). The predicate is a SQL boolean expression — the same thing you'd
write after `WHERE` — evaluated against the table's columns.

```javascript
docs.append([{ title: "draft post" }, { title: "spam" }]);

// Delete every row matching the predicate.
docs.delete("title = 'spam'");

// Replace matched rows 1:1 with new rows (same input shapes as append).
const stats = docs.update("title = 'draft post'", [{ title: "published post" }]);
console.log(stats.matched, stats.nTombstoned, stats.nNotFound);
```

`update` is a one-to-one replacement: the number of matched rows must equal the
number you supply, otherwise it throws. Both methods return `{ matched,
nTombstoned, nNotFound }`.

## Optimize

Many small appends produce many small files. `optimize` compacts them —
merging small or underfilled files into larger ones — which keeps reads efficient.

```javascript
docs.optimize();                                                  // engine defaults
docs.optimize({ targetSuperfileSizeMb: 256, minFillPercent: 50 });
```

Compaction and interrupted writes can leave behind storage objects that are no
longer referenced. `gc` deletes them, reclaiming space. It only removes objects
older than a grace window (in seconds) so it never races a concurrent reader or
writer; requires durable storage.

```javascript
const report = docs.gc(3600);                                     // older than 1 hour
console.log(report.bytesFreed, report.objectsDeleted);
```

## Storage backends

`connect` selects the backend from the URI:

| URI                      | Backend                                  |
| ------------------------ | ---------------------------------------- |
| `./data`, `/abs/path`    | Local filesystem                         |
| `s3://bucket/prefix`     | Amazon S3 / S3-compatible object storage |
| `az://container/prefix`  | Azure Blob Storage                       |
| `memory://`              | In-process, ephemeral (testing)          |

Credentials go in `storageOptions`, keyed by the standard `object_store` config
strings (`aws_*` / `azure_*` — the same names the AWS and Azure SDKs use). Omit
them to use ambient cloud identity (IAM instance role / managed identity);
infino reads no credentials from the environment.

```javascript
// S3
const db = connect("s3://bucket/prefix", {
  storageOptions: {
    aws_access_key_id: "…",
    aws_secret_access_key: "…",
    aws_region: "us-east-1",
  },
});

// Azure
const db = connect("az://container/prefix", {
  storageOptions: {
    azure_storage_account_name: "…",
    azure_storage_account_key: "…",
  },
});
```

Common keys:

| Backend | Keys |
| ------- | ---- |
| S3      | `aws_access_key_id`, `aws_secret_access_key`, `aws_region`, `aws_session_token`, `aws_endpoint` |
| Azure   | `azure_storage_account_name`, `azure_storage_account_key`, `azure_storage_sas_key`, `azure_storage_client_id`, `azure_storage_client_secret`, `azure_storage_tenant_id` |

The full set is whatever `object_store` accepts for the backend; an unknown key
is rejected at `connect`. Pass `validate: true` to probe the backend at
`connect`, so wrong credentials or an unreachable bucket throw there instead of
on the first query. For an S3-compatible endpoint (MinIO / R2 / Ceph), set
`aws_endpoint` (with `aws_allow_http: "true"` for plain HTTP) alongside the
credentials.

### Local disk cache

For object-storage-backed catalogs, a local disk cache keeps hot data on fast
local storage. `coldFetchMode` controls how cache misses are served:
`"hybrid_with_prefetch"`, `"range_only"`, or
`"lazy_foreground_with_background_fill"`.

```javascript
const db = connect("s3://bucket/prefix", {
  cacheDir: "/mnt/nvme/infino-cache",
  cacheBudgetBytes: 64 * 1024 ** 3,
  coldFetchMode: "lazy_foreground_with_background_fill",
});
```

## Schema and type requirements

- Full-text columns must be Arrow `LargeUtf8` (`"large_utf8"` in a descriptor).
- Vector columns must be `FixedSizeList<Float32, dim>` (`{ vector: dim }`) with
  `dim` in `[16, 4096]`.
- The `_id` column is generated by the engine; do not declare it. It comes back
  as a JavaScript `bigint`.
- `createTable` accepts an apache-arrow `Schema` or a plain `{ column: type }`
  descriptor; `append` / `update` accept an array of objects or an apache-arrow
  `Table` / `RecordBatch`, coerced against the table's declared schema.

## API reference

- `connect(uri, options?)` — backend from the URI scheme. `options`:
  `storageOptions` (credentials, keyed by `object_store`'s `aws_*` / `azure_*`
  strings), `validate`, and a local disk cache (`cacheDir`, `cacheBudgetBytes`,
  `coldFetchMode`).
- `Connection`
  - `createTable(name, schema, IndexSpec)` / `openTable(name)` /
    `dropTable(name, purge?)` (`purge = true` also deletes the data) /
    `listTables()` / `querySql(sql, { arrow? })`.
- `Table`
  - `append(data)` — one `append` is one commit.
  - `bm25Search(col, q, k, { mode?, projection?, arrow? })` — ranked BM25.
  - `vectorSearch(col, query, k, { nprobe?, rerankMult?, filter?, projection?, arrow? })`
    — ranked kNN; `filter` (`{ column, query, mode? }`, `column` FTS-indexed) is
    a pushdown pre-filter.
  - `tokenMatch(col, q, { mode?, projection?, arrow? })` /
    `exactMatch(col, value, { projection?, arrow? })` — unranked (`score` is `0`).
  - `count(col, q, { mode? })` — match tally, no rows fetched.
  - `update(predicate, data)` / `delete(predicate)` — mutate rows matching a SQL
    predicate; return `{ matched, nTombstoned, nNotFound }`; require durable
    storage.
  - `optimize({ maxMemoryMb?, minFillPercent?, targetSuperfileSizeMb? })`.
  - `gc(graceSecs)` — delete orphaned storage objects older than the grace
    window; returns `{ bytesFreed, objectsDeleted, objectsSkippedLive,
    objectsSkippedTooNew, deleteErrors }`; requires durable storage.
  - `schema()` — the table's apache-arrow `Schema`.
- `IndexSpec().fts(col).vector(col, dim, nCent, metric)`.
- `BUILDER_ID` (named export) — the engine's build identifier string.

Search results default to `_id` + `score`; name columns in `projection` to
materialize row data.

## Building from source

The binding is built with [napi-rs](https://napi.rs/). Building requires a Rust
toolchain and access to crates.io.

```sh
cd infino-node
npm install && npm run build && npm test
```

## Notes

- The API is **synchronous**. In a long-running server, run calls in a
  `worker_thread` so a query doesn't block the event loop.

## FAQ

**Is infino a vector database?** It does vector search, but it's more than that —
an embedded engine that runs vector search *and* full-text (BM25) *and* SQL over
one copy of your data. Reach for it wherever you'd use a vector database, plus the
cases a vector store alone can't cover: keyword search, filtering, joins, and
aggregates.

**Does it need a server?** No. It runs in your Node.js process — no daemon, no
cluster, no managed service. Your data is Parquet on local disk or S3.

**Can it do hybrid (keyword + vector) search?** Yes, natively — BM25 and vector
fused in a single pass via `hybrid_search` (see [Hybrid search](#hybrid-search)),
not a client-side rerank.

**Where is my data stored?** As Apache Parquet files on local disk or any
S3-compatible object store; each file embeds its own BM25 and vector indexes.

**Does it work with TypeScript?** Yes — the package ships type definitions and the
API is identical from JavaScript and TypeScript. Both ESM `import` and CommonJS
`require` work.

**Do I need a Rust toolchain to install it?** No — a prebuilt native binary is
selected automatically at install (macOS and Linux, x64 and arm64).

**Is it a good fit for RAG or agent memory?** Yes, that's a primary use case:
store documents or conversation history once, retrieve with hybrid search, and
filter/aggregate with SQL. See the runnable [examples](examples).

## License

Apache-2.0.
