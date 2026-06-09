# infino — Node.js bindings

napi-rs bindings over infino's catalog API. Sync; Arrow is the
interchange (crossing the JS↔Rust boundary as Arrow IPC bytes).

```javascript
const { connect, IndexSpec } = require("infino");
const { Table, vectorFromArray, tableToIPC, LargeUtf8 } = require("apache-arrow");

const db = connect("memory://"); // or "./data", "s3://bucket/prefix"

// Arrow data crosses the boundary as IPC bytes. FTS columns must be
// LargeUtf8 (not Utf8). An empty table carries just the schema.
const ipc = (titles) =>
  Buffer.from(tableToIPC(new Table({ title: vectorFromArray(titles, new LargeUtf8()) }), "stream"));

const docs = db.createTable("docs", ipc([]), new IndexSpec().fts("title"));
docs.append(ipc(["the quick brown fox"]));

const hits = docs.bm25Search("title", "fox", 10); // [{ id: 1n, score: 1.23 }]
```

## Build & test (requires online crates.io access + a Rust toolchain)

This crate is **excluded** from the infino cargo workspace so the core
Rust crate doesn't require a Node toolchain to build. Build it standalone
with the napi-rs CLI (not `cargo build -p`, which would need workspace
membership):

```sh
cd infino-node
npm install
npm run build       # compiles the addon + generates index.js / index.d.ts
npm test            # node --test against the built addon
```

## Scope (v1 — mirrors the Python bindings)

- `connect(uri, options?)` — backend from the URI scheme; S3-compatible
  static creds via `options = { endpoint, region, accessKey, secretKey }`
  (endpoint requires the other three).
- `Connection`: `createTable(name, schemaIpc, IndexSpec)`, `openTable`,
  `dropTable`, `listTables`, `querySql` → Arrow IPC `Buffer`
  (`tableFromIPC`).
- `Table`: `append(ipcBuffer)`, `bm25Search`, `vectorSearch`, `schema`.
  `append` takes an Arrow IPC `Buffer` (`tableToIPC`) and re-wraps columns
  under the table's declared schema; one `append` is one commit.
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
- **Arrow IPC, not zero-copy C-Data** — JS↔Rust has no pyarrow-style
  C-Data bridge, so bulk data crosses as IPC bytes (one copy), the
  pragmatic Node approach. Query vectors are the exception: a
  `Float32Array` crosses by reference.

Next: `update` / `delete`; richer connect options (disk cache); prebuilt
per-platform binaries + CI for `npm install infino` distribution.
