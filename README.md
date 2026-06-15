# infino

[![CI](https://github.com/infino-ai/infino/actions/workflows/ci.yml/badge.svg)](https://github.com/infino-ai/infino/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Infino is a fast retrieval engine that runs SQL, full-text search, and vector search over a single copy of your data on object storage.** Point it at a bucket on S3 (or Azure, or local disk) and query the same rows three ways from one system — no separate search cluster, vector database, and warehouse to provision and keep in sync, and no daemon or managed service to operate.

The format is what makes this work: **every file is a valid Apache Parquet file with BM25 and vector indexes spliced in.** The same files read as plain Parquet through the Arrow ecosystem —
[DataFusion](https://datafusion.apache.org/),
[DuckDB](https://duckdb.org/),
[pyarrow](https://arrow.apache.org/docs/python/) —
and as a search index through infino's own reader, so your data stays open and portable while gaining low-latency search.

**Why infino**

- **Best performance per dollar** — engineered for the strongest speed-per-dollar trade-off, not just raw latency: object-storage economics plus a read path continuously benchmarked on speed *and* cost (bytes fetched, request count, memory footprint).
- **Three modalities, one engine** — keyword (BM25), vector, and SQL over the same rows; no copying data between systems to combine them.
- **Object-storage-native** — your data lives on S3, Azure, or local disk, with snapshot-isolated reads and append / update / delete through atomic commits. No cluster to stand up.
- **Open, no lock-in** — superfiles are spec-compliant Parquet, so anything that reads Parquet can read your data.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [SQL joins across tables](#sql-joins-across-tables)
- [Hybrid search](#hybrid-search)
- [Stability](#stability)
- [Development](#development)
- [Performance](#performance)
- [Tests](#tests)

## Install

**Python**

```sh
pip install infino
```

**Node.js**

```sh
npm install infino
```

**Rust**

```sh
cargo add infino
```

or in `Cargo.toml`:

```toml
[dependencies]
infino = "0.1"
```

## Quickstart

A memory store for your agent: persist what it learns on object storage, then recall the most relevant pieces to ground the next model call. One table holds the text and its embedding, indexed for both keyword and vector retrieval, with SQL over the same rows — no second system to sync. The backend is chosen by the connection URI (`memory://`, a local path, `s3://…`, `az://…`).

**Python**

```python
import infino
import pyarrow as pa

# Durable agent memory on object storage (or "./data", "memory://").
db = infino.connect("s3://my-agent/memory")

# A memory = text + its embedding. One table, keyword- and vector-indexed.
schema = pa.schema([
    pa.field("text", pa.large_utf8(), nullable=False),
    pa.field("embedding", pa.list_(pa.float32(), 1536), nullable=False),
])
mem = db.create_table(
    "memory", schema,
    infino.IndexSpec().fts("text").vector("embedding", 1536, 256, "cosine"),
)

# Remember. embed() is your embedding model (OpenAI, Cohere, a local model, …).
notes = ["the user prefers dark mode", "the cancel flow lives under Settings"]
mem.append([{"text": t, "embedding": embed(t)} for t in notes])

# Recall the most relevant memories for this turn → ground your LLM.
context = mem.vector_search("embedding", embed("how do I cancel?"), 5)

# Same rows, other lenses:
#   mem.bm25_search("text", "cancel", 5)                       # keyword
#   db.query_sql("SELECT text FROM memory")                    # SQL
```

**Node.js**

```javascript
import { connect, IndexSpec } from "infino";

// Durable agent memory on object storage (or "./data", "memory://").
const db = connect("s3://my-agent/memory");

// A memory = text + its embedding; one table, keyword- and vector-indexed.
const mem = db.createTable(
  "memory",
  { text: "large_utf8", embedding: { vector: 1536 } },
  new IndexSpec().fts("text").vector("embedding", 1536, 256, "cosine"),
);

// Remember. embed() is your embedding model (OpenAI, Cohere, a local model, …).
const notes = ["the user prefers dark mode", "the cancel flow lives under Settings"];
mem.append(notes.map((text) => ({ text, embedding: embed(text) })));

// Recall the most relevant memories for this turn → ground your LLM.
const context = mem.vectorSearch("embedding", embed("how do I cancel?"), 5);

// Same rows, other lenses: mem.bm25Search("text", "cancel", 5)
// or db.querySql("SELECT text FROM memory").
```

**Rust**

```rust,no_run
use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{connect, IndexSpec, Metric, VectorSearchOptions};

// embed(): your embedding model (OpenAI, Cohere, a local model, …).
fn embed(_text: &str) -> Vec<f32> { vec![0.0; 1536] }

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Durable agent memory on object storage (or "./data", "memory://").
let db = connect("s3://my-agent/memory")?;

// A memory = text + its embedding; one table, keyword- and vector-indexed.
let item = Arc::new(Field::new("item", DataType::Float32, true));
let schema = Arc::new(Schema::new(vec![
    Field::new("text", DataType::LargeUtf8, false),
    Field::new("embedding", DataType::FixedSizeList(item.clone(), 1536), false),
]));
let mem = db.create_table(
    "memory",
    schema.clone(),
    IndexSpec::new().fts("text").vector("embedding", 1536, 256, Metric::Cosine),
)?;

// Remember.
let notes = ["the user prefers dark mode", "the cancel flow lives under Settings"];
let flat: Vec<f32> = notes.iter().flat_map(|&t| embed(t)).collect();
let embeddings = FixedSizeListArray::new(item, 1536, Arc::new(Float32Array::from(flat)), None);
mem.append(&RecordBatch::try_new(
    schema,
    vec![Arc::new(LargeStringArray::from(notes.to_vec())), Arc::new(embeddings)],
)?)?;

// Recall the most relevant memories for this turn → ground your LLM.
let context =
    mem.vector_search("embedding", &embed("how do I cancel?"), 5, VectorSearchOptions::new(), None)?;
# let _ = context;
# Ok(())
# }
```

Bindings live in [`infino-python/`](infino-python/) (PyO3 + maturin) and
[`infino-node/`](infino-node/); see their READMEs to build from source.
The Node API is synchronous — objects in, plain records out, with `_id`
returned as a JavaScript `bigint`.

## Architecture

Three docs cover the design, from the high-level tour down to the
on-disk bytes:

- **[Overview →](docs/architecture/overview.md)** — the plain-language
  tour: what infino is, the mental model, and how it compares to other
  systems.
- **[Superfile format →](docs/architecture/superfile.md)** — the
  single-file superfile format: a valid Parquet file with embedded
  full-text and vector indexes. Covers the layout, Parquet
  compatibility, and the full-text and vector index design.
- **[Supertable layer →](docs/architecture/supertable.md)** — the table
  layer over many superfiles: manifest snapshots, the commit/publish
  path, pluggable storage, query fan-out with manifest-only skip
  pruning, and reader/writer concurrency.

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
