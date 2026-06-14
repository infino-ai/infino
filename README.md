# infino

Infino stores data in a search-optimized lakehouse format. **One file = a valid Apache Parquet file plus embedded BM25 + vector indexes** — readable as Parquet by
[DataFusion](https://datafusion.apache.org/) /
[DuckDB](https://duckdb.org/) /
[pyarrow](https://arrow.apache.org/docs/python/),
and as a search index by infino's reader.

## Links

- **[Superfile architecture →](docs/architecture/superfile.md)** —
  the single-file superfile format: a valid Parquet file with embedded
  full-text and vector indexes. Covers the layout, Parquet
  compatibility, and the full-text and vector index design.
- **[Supertable architecture →](docs/architecture/supertable.md)** —
  the table layer over superfile superfiles: manifest snapshots, the
  commit/publish path, pluggable storage, query fan-out with
  manifest-only skip pruning, and reader/writer concurrency.

## Quick example in Python

```python
import infino
import pyarrow as pa

db = infino.connect("memory://")              # or "./data", "s3://bucket/prefix"
schema = pa.schema([("title", pa.large_utf8())])
docs = db.create_table("docs", schema, infino.IndexSpec().fts("title"))

docs.append([{"title": "the quick brown fox"}])     # list[dict], pandas, or pyarrow

hits = docs.bm25_search("title", "fox", 10)         # pyarrow.Table (_id, title, score)
rows = db.query_sql("SELECT _id, title FROM docs")  # pyarrow.Table
```

The Python bindings (PyO3 + maturin) live in
[`infino-python/`](infino-python/) — see its README to build and test.

## Quick example in Node.js

```javascript
const { connect, IndexSpec } = require("infino");

const db = connect("memory://");                   // or "./data", "s3://bucket/prefix"

// A plain { column: type } schema — no apache-arrow needed.
const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));

// append plain objects; results come back as plain records.
docs.append([{ title: "the quick brown fox" }, { title: "a lazy dog" }]);

const rows = docs.bm25Search("title", "fox", 10);        // ranked rows as records
const hits = docs.tokenMatch("title", "fox");            // unranked matching rows (score 0)
const sql  = db.querySql("SELECT _id, title FROM docs"); // records (or { arrow: true })
```

The Node.js bindings live in
[`infino-node/`](infino-node/) — see its README to build and test. The
API is synchronous: objects in, plain records out; `_id` comes back as a
JavaScript `bigint`.

## Quick example in Rust

Open a connection, create a table with a full-text index, append rows,
then search — by keyword or with SQL across the catalog. The backend is
chosen by the URI scheme (`memory://`, a local path, `s3://…`, `az://…`).

```rust
use std::sync::Arc;

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{connect, BoolMode, IndexSpec};

let db = connect("memory://")?; // or "./data", "s3://bucket/prefix"

let schema = Arc::new(Schema::new(vec![Field::new("title", DataType::LargeUtf8, false)]));
let docs = db.create_table("docs", schema.clone(), IndexSpec::new().fts("title"))?;

// One `append` == one commit == one sealed, immutable superfile.
let batch = RecordBatch::try_new(
    schema,
    vec![Arc::new(LargeStringArray::from(vec!["the quick brown fox"]))],
)?;
docs.append(&batch)?;

// Keyword search (BM25): Arrow rows carrying the auto-injected `_id`,
// the projected columns, and a trailing `score`. Only the projected
// scalar columns are decoded; project just `_id` + `score` (or pass
// `None` for the whole row) to control the decode cost.
let batches = docs.bm25_search("title", "fox", 10, BoolMode::Or, Some(&["_id", "title", "score"]))?;
assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

// SQL across the catalog — every superfile is also a valid Parquet file.
let rows = db.query_sql("SELECT _id, title FROM docs")?;
assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Vector search is the same shape: declare a vector column with
`IndexSpec::new().vector("embedding", 384, 256, infino::Metric::Cosine)`
and call `vector_search`. SQL-native search — the `bm25_search` /
`vector_search` / `hybrid_search` table functions — composes with joins
and aggregations across catalog tables.

## SQL joins across tables

`query_sql` resolves every table the query names through the catalog and
registers them into one engine, so a join across two tables — or a join
of a search result against a table — is just SQL:

```rust
use std::sync::Arc;

use arrow_array::{Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{connect, IndexSpec};

let db = connect("memory://")?;

// Two tables sharing an `author_id`.
let authors_schema = Arc::new(Schema::new(vec![
    Field::new("author_id", DataType::Int64, false),
    Field::new("name", DataType::LargeUtf8, false),
]));
let authors = db.create_table("authors", authors_schema.clone(), IndexSpec::new())?;
authors.append(&RecordBatch::try_new(
    authors_schema,
    vec![
        Arc::new(Int64Array::from(vec![1])),
        Arc::new(LargeStringArray::from(vec!["alice"])),
    ],
)?)?;

let posts_schema = Arc::new(Schema::new(vec![
    Field::new("author_id", DataType::Int64, false),
    Field::new("body", DataType::LargeUtf8, false),
]));
let posts = db.create_table("posts", posts_schema.clone(), IndexSpec::new().fts("body"))?;
posts.append(&RecordBatch::try_new(
    posts_schema,
    vec![
        Arc::new(Int64Array::from(vec![1])),
        Arc::new(LargeStringArray::from(vec!["hello from alice"])),
    ],
)?)?;

// Join both tables in one query.
let rows = db.query_sql(
    "SELECT a.name, p.body \
     FROM posts p JOIN authors a ON p.author_id = a.author_id",
)?;
assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

A search TVF (`bm25_search('posts', 'body', 'alice', 10)`) can stand in
for either side of the join, so keyword/vector results compose with the
rest of the catalog the same way.

## Hybrid Search

Infino also wires indexes into SQL execution as **physical
access paths**:

```sql
-- The text predicate is answered from the FTS index — inverted index →
-- candidate rows → decode only those rows — never a full column scan.
SELECT category, AVG(rating)
FROM reviews
WHERE title = 'battery life'
GROUP BY category;
```

Equality, `IN`, and boolean combinations on an indexed text column
resolve through the index to an exact candidate row set before any
column data is read. Superfiles that can't match are never opened at all:
term blooms, value ranges, and vector centroids live side by side in the
manifest, so scalar, keyword, and vector signals prune through one
shared layer.

Retrieval composes the same way. The ranked `bm25_search` /
`vector_search` / `hybrid_search` and the unranked `token_match` /
`exact_match` are table functions so a candidate set is the 
*first stage of a plan* rather than its result:

```sql
-- Rank first; join and aggregate over just the candidates.
SELECT a.name, COUNT(*) AS hits
FROM bm25_search('posts', 'body', 'rust async', 100) p
JOIN authors a ON a.author_id = p.author_id
GROUP BY a.name
ORDER BY hits DESC;

-- Set algebra over index-bounded candidate sets: "rust but not compiler".
SELECT _id FROM token_match('posts', 'body', 'rust')
EXCEPT
SELECT _id FROM token_match('posts', 'body', 'compiler');
```

One snapshot, one copy of the data: sparse (BM25), dense (vector), and
structured (scalar) predicates compose inside the engine — no second
system to sync, no client-side result stitching.

## Stability

The public API is what's re-exported from the crate root — `connect` /
`connect_with`, `Connection`, `Supertable`, `IndexSpec`, `InfinoError`,
and the value types their signatures name. It is pinned by a
`cargo-public-api` snapshot (`public-api.txt`); any change to it is
reviewed as a contract change in the same pull request.

- **Versioning.** 0.x while the surface soaks; 1.0 once it has shipped
  without churn for a release or two. Pre-1.0 may break, but every break
  shows in the snapshot diff and is called out in the release notes.
- **`#[non_exhaustive]`** on growable public enums/structs (e.g.
  `InfinoError`, `MutationStats`), so adding a variant or field is not a
  breaking change.
- **Arrow / DataFusion are part of the contract.** The API is
  Arrow-native (`RecordBatch`, `SchemaRef`, `Expr`); a major bump of
  arrow / datafusion that changes an exposed type is a breaking change to
  infino. The supported version range is documented and CI-tested.
- **MSRV.** Raising the minimum Rust version is a minor bump, never a
  patch.
- **Deprecation.** Post-1.0, removals go through `#[deprecated]` for at
  least one minor release first.
- **Python.** The wheel tracks the crate version 1:1.
- **Node.** The npm package tracks the crate version 1:1.

## Development

```bash
git clone git@github.com:infino-ai/infino.git
cd infino
cargo build
cargo run --example demo   # end-to-end tour: build, BM25 + vector search, read back as Parquet
```

The toolchain is pinned by `rust-toolchain.toml`, so `rustup` installs
the right stable Rust on first build. Run `cargo test --features test-helpers`
for the suite (integration tests use `infino::test_helpers`) and `make ci`
before opening a pull request.

For an enhanced local development experience, install and configure
[pre-commit](https://pre-commit.com/#install) hooks with `pre-commit install`
to catch formatting and lint issues before committing.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

## Performance

Benchmarks live under [`benches/`](benches/) and use Infino's custom
benchmark harness so build, correctness, hot reads, cold object-store
reads, RSS, and markdown output all share one measured lifecycle. Run
`cargo bench` to reproduce them on your hardware.

## Tests

Run `cargo test --workspace` for the full suite. It covers the
end-to-end full-text, vector, and superfile pipelines, ingestion and
commit, and open-format compatibility — DataFusion reads superfiles as
plain Parquet, with column projection, GROUP BY, and predicate
pushdown all matching the columnar data.

**Memory safety.** The full-text surface runs clean under
[miri](https://github.com/rust-lang/miri) (Stacked Borrows + UB
detection) and
[AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html);
run `make miri` and `make asan`.
