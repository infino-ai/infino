# infino

Infino stores data in a search-optimized lakehouse format. **One file = a valid Apache Parquet file plus embedded BM25 + vector indexes** — readable as Parquet by
[DataFusion](https://datafusion.apache.org/) /
[DuckDB](https://duckdb.org/) /
[pyarrow](https://arrow.apache.org/docs/python/),
and as a search index by infino's reader.

## Links

- **[Superfile architecture →](docs/architecture/superfile.md)** —
  the single-file segment format: a valid Parquet file with embedded
  full-text and vector indexes. Covers the layout, Parquet
  compatibility, and the full-text and vector index design.
- **[Supertable architecture →](docs/architecture/supertable.md)** —
  the table layer over superfile segments: manifest snapshots, the
  commit/publish path, pluggable storage, query fan-out with
  manifest-only skip pruning, and reader/writer concurrency.

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

// One `append` == one commit == one sealed, immutable segment.
let batch = RecordBatch::try_new(
    schema,
    vec![Arc::new(LargeStringArray::from(vec!["the quick brown fox"]))],
)?;
docs.append(&batch)?;

// Keyword search (BM25): hits carry the auto-injected `_id` + score.
let hits = docs.bm25_search("title", "fox", 10, BoolMode::Or)?;
assert_eq!(hits.len(), 1);

// SQL across the catalog — every segment is also a valid Parquet file.
let rows = db.query_sql("SELECT _id, title FROM docs")?;
assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Vector search is the same shape: declare a vector column with
`IndexSpec::new().vector("embedding", 384, 256, infino::Metric::Cosine)`
and call `vector_search`. SQL-native search — the `bm25_search` /
`vector_search` / `hybrid_search` table functions — composes with joins
and aggregations across catalog tables.

## Quick example in Python

```python
import infino
import pyarrow as pa

db = infino.connect("memory://")              # or "./data", "s3://bucket/prefix"
schema = pa.schema([("title", pa.large_utf8())])
docs = db.create_table("docs", schema, infino.IndexSpec().fts("title"))

docs.append([{"title": "the quick brown fox"}])     # list[dict], pandas, or pyarrow

hits = docs.bm25_search("title", "fox", 10)         # [{"_id": ..., "score": ...}]
rows = db.query_sql("SELECT _id, title FROM docs")  # pyarrow.Table
```

The Python bindings (PyO3 + maturin) live in
[`infino-python/`](infino-python/) — see its README to build and test.

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

## Development

```bash
git clone git@github.com:infino-ai/infino.git
cd infino
cargo build
cargo run --example demo   # end-to-end tour: build, BM25 + vector search, read back as Parquet
```

The toolchain is pinned by `rust-toolchain.toml`, so `rustup` installs
the right stable Rust on first build. Run `cargo test --workspace` for
the suite and `make ci` before opening a pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

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
