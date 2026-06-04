# infino

A search-optimized lakehouse format. **One file = a valid Apache Parquet
file plus embedded BM25 + vector indexes** — readable as Parquet by
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

```rust
use infino::superfile::{
    SuperfileReader, VectorSearchOptions, bm25_search, vector_search,
};
use infino::superfile::fts::reader::BoolMode;
use bytes::Bytes;

// Read a superfile (built via SuperfileBuilder).
let bytes: Bytes = std::fs::read("my.superfile")?.into();
let reader = SuperfileReader::open(bytes)?;

// BM25 search over the embedded FTS blob:
let hits = bm25_search(&reader, "title", "rust async", 10, BoolMode::Or)?;

// kNN search over the embedded vector blob:
let query = vec![/* dim=384 f32s */];
let hits = vector_search(&reader, "embedding", &query, 10,
                         VectorSearchOptions::default())?;

// And the same bytes are a valid Parquet file — register them
// with DataFusion / DuckDB / pyarrow and treat as a regular table.
```

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

Benchmarks live in the in-tree
[criterion](https://github.com/bheisler/criterion.rs) harness under
[`benches/`](benches/). Run `cargo bench` to reproduce them on your
hardware.

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
