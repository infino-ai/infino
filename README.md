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
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use infino::superfile::builder::{FtsConfig, VectorConfig};
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::vector::distance::Metric;
use infino::superfile::VectorSearchOptions;
use infino::supertable::{Supertable, SupertableOptions};

const DIM: usize = 384;

// A full-text `title` column + an `embedding` vector column. The `_id`
// column is injected by the supertable — don't declare it yourself.
let schema = Arc::new(Schema::new(vec![
    Field::new("title", DataType::LargeUtf8, false),
    Field::new(
        "embedding",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM as i32),
        false,
    ),
]));
let options = SupertableOptions::new(
    schema,
    vec![FtsConfig { column: "title".into() }],
    vec![VectorConfig::new("embedding".into(), DIM, 256, 0, Metric::Cosine)],
    None, // default tokenizer
)?;
let table = Supertable::open(options)?;

// BM25 over the FTS index — synchronous, fans out across segments for you:
let hits = table.bm25_search("title", "rust async", 10, BoolMode::Or)?;

// kNN over the vector index:
let query = vec![/* dim=384 f32s */];
let hits = table.vector_search("embedding", &query, 10, VectorSearchOptions::default())?;

// Or query it as SQL — DataFusion under the hood, and every segment is a
// valid Parquet file you can also hand to DuckDB / pyarrow directly:
let batches = table.query_sql(
    "SELECT _id, title FROM bm25_search('title', 'rust async', 10)",
)?;
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
