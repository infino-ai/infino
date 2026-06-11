# infino — Node.js bindings

napi-rs bindings over infino's catalog API. Sync and Node-idiomatic:
pass arrays of objects (or apache-arrow Tables) in, get plain records
out. A thin TypeScript wrapper (`infino/index.ts`, compiled by `tsc`)
hides the Arrow-IPC boundary over the napi addon (`infino/native.js`);
pass `{ arrow: true }` to a search/query to get an apache-arrow `Table`
instead.

```javascript
const { connect, IndexSpec } = require("infino");
const { Schema, Field, LargeUtf8 } = require("apache-arrow");

const db = connect("memory://"); // or "./data", "s3://bucket/prefix"

// Schema is an apache-arrow Schema. FTS columns must be LargeUtf8.
const schema = new Schema([new Field("title", new LargeUtf8(), false)]);
const docs = db.createTable("docs", schema, new IndexSpec().fts("title"));

// append plain objects — the wrapper builds Arrow under the hood.
docs.append([{ title: "the quick brown fox" }, { title: "a lazy dog" }]);

const rows = docs.bm25Search("title", "fox", 10);  // matching rows as records
const hits = docs.tokenMatch("title", "fox");      // unranked matching rows (score 0)
const out  = db.querySql("SELECT COUNT(*) AS n FROM docs"); // records (or { arrow: true })
```

## Build & test (requires online crates.io access + a Rust toolchain)

This crate is **excluded** from the infino cargo workspace so the core
Rust crate doesn't require a Node toolchain to build. Build it standalone
with the napi-rs CLI (not `cargo build -p`, which would need workspace
membership):

```sh
cd infino-node
npm install
npm run build       # napi build (native addon) + tsc (compiles infino/index.ts)
npm test            # node --test against the built wrapper
```

## Scope (v1 — mirrors the Python bindings)

Node-idiomatic: objects in, records out; Arrow is optional.

- `connect(uri, options?)` — backend from the URI scheme; S3-compatible
  static creds via `options = { endpoint, region, accessKey, secretKey }`
  (endpoint requires the other three).
- `Connection`: `createTable(name, arrowSchema, IndexSpec)`, `openTable`,
  `dropTable`, `listTables`, `querySql(sql, { arrow? })`.
- `Table`:
  - `append(data)` — accepts an array of objects, an apache-arrow
    `Table`/`RecordBatch`, or raw Arrow IPC bytes; one `append` is one
    commit.
  - `bm25Search(col, q, k, { mode?, materialize?, arrow? })` /
    `vectorSearch(col, query, k, { nprobe?, materialize?, arrow? })` —
    ranked search; return matching **rows** as records (or an apache-arrow
    `Table` with `{ arrow: true }`). `query` is a `number[]` or
    `Float32Array`. BM25 materializes by default; vector does not.
  - `tokenMatch(col, q, { mode?, projection?, arrow? })` /
    `exactMatch(col, value, { projection?, arrow? })` — unranked matching
    rows (`score` is `0`); same records/`{ arrow: true }` shape as the
    ranked searches.
  - `schema()` — the table's apache-arrow `Schema`.
- `IndexSpec().fts(col).vector(col, dim, nCent, metric)`.

### Schema requirements

- FTS columns must be Arrow `LargeUtf8` (not `Utf8`).
- Vector columns must be `FixedSizeList<Float32, dim>`, `dim` in `[16, 4096]`.

The schema passed to `createTable` and the data passed to `append` must
use these exact types (`append` re-wraps nullability under the declared
schema, but a genuine type mismatch errors).

### Decisions

- **Sync** for v1 (matches Rust + Python). A sync call blocks the event
  loop; in a long-running server run calls in a `worker_thread`. Async is
  an additive follow-up.
- **`SearchHit.id` is a `bigint`** — the core `_id` is 128-bit; JS
  `number` would lose precision past 2^53.
- **Objects in, records out** — the TypeScript wrapper (`infino/index.ts`)
  converts arrays of objects ↔ Arrow and decodes results to plain records,
  the same layered pattern LanceDB / nodejs-polars use (a hand-written
  TS layer over the napi addon, `infino/native.js`). JS↔Rust has no
  pyarrow-style zero-copy C-Data bridge, so bulk data crosses as Arrow
  IPC (one copy); the wrapper hides it. Query vectors cross as a
  `Float32Array` by reference.

Next: `update` / `delete`; richer connect options (disk cache); prebuilt
per-platform binaries + CI for `npm install infino` distribution.
