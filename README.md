# infino

A search-optimized lakehouse format. **One file = a valid Apache Parquet
file plus embedded BM25 + vector indexes** — readable as Parquet by
[DataFusion](https://datafusion.apache.org/) /
[DuckDB](https://duckdb.org/) /
[pyarrow](https://arrow.apache.org/docs/python/),
and as a search index by infino's reader.

## Links

- **[Superfile architecture →](docs/architecture/superfile.md)** —
  long-form reference: format spec, API surface, subsystem design
  (FTS / vector / format-surgery / allocator / commit), every
  major design decision with the alternatives that were rejected
  and why.
- **[Supertable architecture →](docs/architecture/supertable.md)** —
  the in-memory cross-segment query + manifest layer over
  superfile: ArcSwap-based reader-writer isolation, copy-on-
  write manifest, segment store, rayon-shard commit, query
  fan-out, manifest-only skip pruning, dual-pool concurrency,
  DataFusion SQL surface.

## Quick example

```rust
use infino::superfile::{
    SuperfileReader, VectorSearchOptions, bm25_search, vector_search,
};
use infino::superfile::fts::reader::BoolMode;
use bytes::Bytes;

// Read a superfile (built via SuperfileBuilder; see
// `docs/architecture/superfile.md` § Build path).
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

Absolute runtime numbers come from the in-tree criterion harness
under `benches/`. Run `cargo bench` after any change to the FTS or
vector pipeline. The architecture docs in `docs/architecture/`
describe where the wins come from — BM25 with the BMW / BMM walks
and the per-doc bail, IVF + 1-bit RaBitQ + Sq8/Bf16/Fp32 rerank,
the mimalloc-backed per-term `Vec<(u32, u32)>`, cluster-contiguous
vector storage, the ArcSwap reader-writer split.

## Tests

Suite breakdown:

- **End-to-end pipelines** — FTS, vector, superfile, ingestion
  threshold-flush + commit, crash-resistance (parent spawns
  aborting child; verifies committed superfiles survive SIGABRT).
- **Open-format compatibility** — DataFusion reads superfiles as
  plain Parquet; planted-row counts, GROUP BY, predicate pushdown
  all match.
- **Brute-force BM25 oracle** — top-k matches the textbook BM25
  formula on a 60-doc planted corpus + a Zipfian-shape stress.
  Catches scoring-math bugs that planted-ground-truth tests
  can't.
- **Brute-force vector oracle** — full-nprobe IVF recovers exact
  top-k for L2Sq / Cosine / NegDot.
- **CRC corruption** — every CRC-protected region rejects byte
  flips.
- **Recall measurement** — recall@10 ≥ 0.90 at default options on
  the standard test corpus.
- **Property tests** — PFOR encode/decode roundtrip.
- **In-module unit tests** — per-module test surfaces across the
  src tree.

Run `cargo test --workspace` for the full suite.

**Memory-safety lanes — both clean:**

| Lane | Result |
|---|---|
| `make miri` (Stacked Borrows + UB detection) | passing on the FTS surface, zero violations |
| `make asan` (LLVM AddressSanitizer) | passing on the FTS surface, zero memory errors |

The only `unsafe` block in `src/` is one
[`std::mem::transmute`](https://doc.rust-lang.org/std/mem/fn.transmute.html)
in
[`FtsBuilder::add_doc`](src/superfile/fts/builder.rs)
that extends a [`bumpalo`](https://github.com/fitzgen/bumpalo)-allocated
`&str`'s lifetime to `&'static str` so it can key the per-doc
HashMap — the lifetime is bounded by the per-doc Bump (which
outlives the HashMap by Rust's reverse-declaration drop order),
which both miri (Stacked Borrows) and asan (real allocator
instrumentation) sign off on.
