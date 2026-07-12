# infino

[![Crates.io](https://img.shields.io/crates/v/infino.svg)](https://crates.io/crates/infino)
[![docs.rs](https://img.shields.io/docsrs/infino)](https://docs.rs/infino)
[![CI](https://github.com/infino-ai/infino/actions/workflows/ci.yml/badge.svg)](https://github.com/infino-ai/infino/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/infino-ai/infino/blob/main/LICENSE)

**infino is a fast retrieval engine that runs SQL, full-text (BM25), and vector
search over a single copy of your data on object storage.** Data stays in Parquet
on S3 (or Azure, GCS, or local disk) and you query it at scale — embedded in your
process, with no separate search server or vector database to run.

- **Speed per dollar** — object-storage economics at search-engine speeds; on a
  1-million-document index, warm BM25 queries return in the microsecond range.
- **Multi-modal queries** — keyword (BM25), vector, and SQL over the same rows.
- **Object-storage-native** — snapshot-isolated reads and atomic commits over S3,
  Azure, GCS, or local disk.
- **Open format, no lock-in** — spec-compliant Parquet, so anything that reads
  Parquet can read your data.

## Install

```sh
cargo add infino
```

infino installs the [mimalloc](https://github.com/microsoft/mimalloc) global
allocator by default. If you embed infino in a process that already sets a global
allocator, turn it off to avoid a second one:
`infino = { version = "0.1", default-features = false }`.

## Quickstart

```rust
use std::sync::Arc;

use infino::arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use infino::arrow_schema::{DataType, Field, Schema};
use infino::{connect, BoolMode, IndexSpec, Metric, VectorFilter, VectorSearchOptions};

// Tiny stand-in for your embedding model so this runs as-is — a 16-dim
// one-hot by topic. Real embeddings are dense and higher-dimensional.
fn embed(topic: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; 16];
    v[topic] = 1.0;
    v
}

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// A knowledge base your agent retrieves over. "memory://" is in-process;
// use "./data" or "s3://bucket/prefix" to persist.
let db = connect("memory://")?;

let item = Arc::new(Field::new("item", DataType::Float32, true));
let schema = Arc::new(Schema::new(vec![
    Field::new("source", DataType::LargeUtf8, false),
    Field::new("body", DataType::LargeUtf8, false),
    Field::new("embedding", DataType::FixedSizeList(item.clone(), 16), false),
]));
let docs = db.create_table(
    "docs",
    schema.clone(),
    IndexSpec::new().fts("body").vector("embedding", 16, 1, Metric::Cosine),
)?;

let flat: Vec<f32> = [0usize, 0, 1].iter().flat_map(|&t| embed(t)).collect();
docs.append(&RecordBatch::try_new(
    schema,
    vec![
        Arc::new(LargeStringArray::from(vec!["help-center", "help-center", "blog"])),
        Arc::new(LargeStringArray::from(vec![
            "To cancel a subscription, open Settings then Billing.",
            "Refunds return to the original payment method.",
            "Enable dark mode under Settings then Appearance.",
        ])),
        Arc::new(FixedSizeListArray::new(item, 16, Arc::new(Float32Array::from(flat)), None)),
    ],
)?)?;

// Retrieve context to ground the agent's next answer:
let keyword = docs.bm25_search("body", "cancel subscription", 5, BoolMode::Or, None)?;
let semantic = docs.vector_search("embedding", &embed(0), 5, VectorSearchOptions::new(), None, None)?;
// hybrid: BM25 + vector, fused with reciprocal-rank fusion:
let hybrid = docs.hybrid_search(
    "body", "cancel subscription", BoolMode::Or,
    "embedding", &embed(0), VectorSearchOptions::new(), 5, None,
)?;
// vector kNN, restricted to rows whose body matches a keyword (pushdown filter):
let filtered = docs.vector_search(
    "embedding", &embed(0), 5, VectorSearchOptions::new(),
    Some(VectorFilter { column: "body", query: "billing", mode: BoolMode::Or }), None,
)?;
let billing = db.query_sql("SELECT body FROM docs WHERE source = 'help-center'")?;
assert_eq!(keyword.iter().map(|b| b.num_rows()).sum::<usize>(), 1);   // BM25
assert!(semantic.iter().map(|b| b.num_rows()).sum::<usize>() >= 1);   // vector kNN
assert!(hybrid.iter().map(|b| b.num_rows()).sum::<usize>() >= 1);     // hybrid (BM25 + vector)
assert_eq!(filtered.iter().map(|b| b.num_rows()).sum::<usize>(), 1);  // vector + keyword filter
assert_eq!(billing.iter().map(|b| b.num_rows()).sum::<usize>(), 2);   // SQL filter
# Ok(())
# }
```

## Operations

The public surface is a small connection-and-table API. Everything except the
two entry-point functions is a method on one of two handles, so the operations
live on the [`Connection`] and [`Supertable`] pages:

- [`connect`] / [`connect_with`] open a [`Connection`]. The backend follows the
  URI scheme (`s3://`, `az://`, `gs://`, `file://`, bare path, `memory://`);
  credentials are passed via [`ConnectOptions::with_storage_option`]
  (object_store's `aws_*` / `azure_*` / `google_*` keys), never read from the
  environment.
- [`Connection`] — the catalog: [`create_table`](Connection::create_table),
  [`open_table`](Connection::open_table), [`drop_table`](Connection::drop_table),
  [`list_tables`](Connection::list_tables), and
  [`query_sql`](Connection::query_sql).
- [`Supertable`] — a single table:
  - **Search** — [`bm25_search`](Supertable::bm25_search),
    [`vector_search`](Supertable::vector_search),
    [`hybrid_search`](Supertable::hybrid_search),
    [`token_match`](Supertable::token_match),
    [`exact_match`](Supertable::exact_match), and
    [`count`](Supertable::count). Each search returns Arrow rows as
    `Vec<RecordBatch>`.
  - **Write** — [`append`](Supertable::append),
    [`update`](Supertable::update), [`delete`](Supertable::delete).
  - **Maintain** — [`optimize`](Supertable::optimize),
    [`gc`](Supertable::gc), and [`schema`](Supertable::schema).

Supporting types: [`IndexSpec`], [`Metric`], [`BoolMode`],
[`VectorSearchOptions`], [`VectorFilter`], [`ConnectOptions`], [`MutationStats`],
[`GcReport`], and the [`InfinoError`], [`OptimizeError`], and [`GcError`] error
enums.

## Cargo features

- `default` — enables the bundled [mimalloc](https://github.com/microsoft/mimalloc)
  global allocator. Disable with `default-features = false` if your process
  already installs a global allocator.

## Other languages

infino also ships **Python** (`pip install infino`) and **Node.js**
(`npm install @infino-ai/infino`) bindings. For concepts, guides, and
multi-language examples, see the full documentation at
[infino.ai/docs](https://infino.ai/docs).
