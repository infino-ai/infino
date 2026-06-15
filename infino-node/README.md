# infino — Node.js bindings

Node.js bindings for infino. Synchronous; pass arrays of objects (or
apache-arrow Tables) in, get plain records out. Pass `{ arrow: true }` to
a search or query to get an apache-arrow `Table` instead.

## Install

```sh
npm install infino
```

A prebuilt native binary is selected automatically at install time — no Rust
toolchain needed. Supported platforms:

| Platform      | Architectures |
| ------------- | ------------- |
| macOS         | x64, arm64    |
| Linux (glibc) | x64, arm64    |

`apache-arrow` is installed as a dependency and is used at the boundary
(passing in Tables, or `{ arrow: true }` results). Node.js >= 18.

## Usage

```javascript
const { connect, IndexSpec } = require("infino");
const { Schema, Field, LargeUtf8 } = require("apache-arrow");

const db = connect("memory://"); // or "./data", "s3://bucket/prefix"

// FTS columns must be LargeUtf8.
const schema = new Schema([new Field("title", new LargeUtf8(), false)]);
const docs = db.createTable("docs", schema, new IndexSpec().fts("title"));

docs.append([{ title: "the quick brown fox" }, { title: "a lazy dog" }]);

const rows = docs.bm25Search("title", "fox", 10);  // matching rows as records
const hits = docs.tokenMatch("title", "fox");      // unranked matching rows (score 0)
const out  = db.querySql("SELECT COUNT(*) AS n FROM docs"); // records (or { arrow: true })
```

ES modules work the same way:

```javascript
import { connect, IndexSpec } from "infino";
```

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
