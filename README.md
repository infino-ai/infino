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

## Quick example

A table has a full-text column (`title`) and a vector column
(`embedding`). You append Arrow record batches, commit to seal a
segment, then query it four ways — keyword, vector, SQL, or hybrid:

```rust
use infino::supertable::Supertable;
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::VectorSearchOptions;

let table = Supertable::create(options)?;     // schema + options: see examples/demo.rs

// Ingest: append record batches, commit to publish an immutable segment.
let mut writer = table.writer()?;
writer.append(&batch)?;                        // columns: title (text) + embedding (vector)
writer.commit()?;

// Keyword search (BM25):
let hits = table.bm25_search("title", "rust async", 10, BoolMode::Or)?;

// Vector search (k-NN):
let knn = table.vector_search("embedding", &query, 10, VectorSearchOptions::default())?;

// SQL (DataFusion under the hood; every segment is also valid Parquet):
let rows = table.query_sql("SELECT _id, title FROM bm25_search('title', 'rust async', 10)")?;

// Hybrid — keyword + vector fused in one query (reciprocal-rank fusion):
let fused = table.query_sql(
    "SELECT _id, title, score \
     FROM hybrid_search('title', 'rust async', 'embedding', '<query vector>', 10)",
)?;
```

A complete, runnable version (schema, options, building a vector
`RecordBatch`, reading segments back as plain Parquet) is in
[`examples/demo.rs`](examples/demo.rs) — run it with
`cargo run --example demo`.

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
