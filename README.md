# infino

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/infino-ai/infino)
[![Crates.io](https://img.shields.io/crates/v/infino.svg)](https://crates.io/crates/infino)
[![docs.rs](https://img.shields.io/docsrs/infino)](https://docs.rs/infino)
[![CI](https://github.com/infino-ai/infino/actions/workflows/ci.yml/badge.svg)](https://github.com/infino-ai/infino/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**infino is a fast retrieval engine that runs SQL, full-text search, and vector search over a single copy of your data on object storage.** Data stays in Parquet on S3 (or Azure, GCS, or local disk) and you can query it at scale.

**Why infino**

- **Speed per dollar** — infino optimizes for speed per dollar, making tradeoffs to achieve object-storage economics at search engine speeds. On a 1-million-document index, warm BM25 queries return in the microsecond range — see [benchmarks](benches/README.md).
- **Multi-modal queries** — keyword (BM25), vector, and SQL queries over the same rows, offering flexible query paths for agents.
- **Object-storage-native** — data lives on S3, Azure, GCS, or local disk, with snapshot-isolated reads and atomic commits.
- **Open format, no lock in** — text and numeric data is stored as spec-compliant Parquet, so anything that reads Parquet can read your data.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Cloud storage](#cloud-storage)
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

# Or with uv (https://docs.astral.sh/uv/):
uv pip install infino
```

**Node.js**

```sh
npm install @infino-ai/infino
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

The full Rust API reference is on [docs.rs/infino](https://docs.rs/infino).

infino installs the [mimalloc](https://github.com/microsoft/mimalloc)
global allocator by default. If you embed infino in a process that already
sets a global allocator, turn it off to avoid a second one:
`infino = { version = "0.1", default-features = false }`.

## Quickstart

**Python**

```python
import infino
import pyarrow as pa

# A knowledge base your agent retrieves over. "memory://" is in-process;
# use "./data" or "s3://bucket/prefix" to persist.
db = infino.connect("memory://")

# Tiny stand-in for your embedding model so this runs as-is — a 16-dim
# one-hot by topic. Real embeddings are dense and higher-dimensional.
def embed(topic):                       # 0 = billing, 1 = appearance
    v = [0.0] * 16
    v[topic] = 1.0
    return v

schema = pa.schema([
    pa.field("source", pa.large_utf8(), nullable=False),
    pa.field("body", pa.large_utf8(), nullable=False),
    pa.field("embedding", pa.list_(pa.float32(), 16), nullable=False),
])
docs = db.create_table(
    "docs", schema,
    infino.IndexSpec().fts("body").vector("embedding", 16, 1, "cosine"),
)

docs.append([
    {"source": "help-center", "body": "To cancel a subscription, open Settings then Billing.", "embedding": embed(0)},
    {"source": "help-center", "body": "Refunds return to the original payment method.",         "embedding": embed(0)},
    {"source": "blog",        "body": "Enable dark mode under Settings then Appearance.",        "embedding": embed(1)},
])

# Retrieve context to ground the agent's next answer:
keyword  = docs.bm25_search("body", "cancel subscription", 5)               # BM25
semantic = docs.vector_search("embedding", embed(0), 5)                     # vector kNN
# vector kNN, restricted to rows whose body matches a keyword (pushdown filter):
filtered = docs.vector_search("embedding", embed(0), 5, filter_column="body", filter_query="billing")
billing  = db.query_sql("SELECT body FROM docs WHERE source = 'help-center'")  # SQL filter
```

**Node.js**

```javascript
import { connect, IndexSpec } from "@infino-ai/infino";

// A knowledge base your agent retrieves over. "memory://" is in-process;
// use "./data" or "s3://bucket/prefix" to persist.
const db = connect("memory://");

// Tiny stand-in for your embedding model so this runs as-is — a 16-dim
// one-hot by topic. Real embeddings are dense and higher-dimensional.
const embed = (topic) => { const v = Array(16).fill(0.0); v[topic] = 1.0; return v; };

const docs = db.createTable(
  "docs",
  { source: "large_utf8", body: "large_utf8", embedding: { vector: 16 } },
  new IndexSpec().fts("body").vector("embedding", 16, 1, "cosine"),
);

docs.append([
  { source: "help-center", body: "To cancel a subscription, open Settings then Billing.", embedding: embed(0) },
  { source: "help-center", body: "Refunds return to the original payment method.",         embedding: embed(0) },
  { source: "blog",        body: "Enable dark mode under Settings then Appearance.",        embedding: embed(1) },
]);

// Retrieve context to ground the agent's next answer:
const keyword  = docs.bm25Search("body", "cancel subscription", 5);            // BM25
const semantic = docs.vectorSearch("embedding", embed(0), 5);                  // vector kNN
// vector kNN, restricted to rows whose body matches a keyword (pushdown filter):
const filtered = docs.vectorSearch("embedding", embed(0), 5, { filter: { column: "body", query: "billing" } });
const billing  = db.querySql("SELECT body FROM docs WHERE source = 'help-center'");  // SQL filter
```

**Rust**

```rust
use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
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
// vector kNN, restricted to rows whose body matches a keyword (pushdown filter):
let filtered = docs.vector_search(
    "embedding", &embed(0), 5, VectorSearchOptions::new(),
    Some(VectorFilter { column: "body", query: "billing", mode: BoolMode::Or }), None,
)?;
let billing = db.query_sql("SELECT body FROM docs WHERE source = 'help-center'")?;
assert_eq!(keyword.iter().map(|b| b.num_rows()).sum::<usize>(), 1);   // BM25
assert!(semantic.iter().map(|b| b.num_rows()).sum::<usize>() >= 1);   // vector kNN
assert_eq!(filtered.iter().map(|b| b.num_rows()).sum::<usize>(), 1);  // vector + keyword filter
assert_eq!(billing.iter().map(|b| b.num_rows()).sum::<usize>(), 2);   // SQL filter
# Ok(())
# }
```

Bindings live in [`infino-python/`](infino-python/) (PyO3 + maturin) and
[`infino-node/`](infino-node/); see their READMEs to build from source.
The Node API is synchronous — objects in, plain records out, with `_id`
returned as a JavaScript `bigint`.

### Open format: read it as Parquet

A superfile *is* a spec-compliant Parquet file. The embedded BM25 and
vector index regions are spliced in ahead of a standard Parquet footer
and pointed at by `inf.*` key/value metadata keys, which any conformant
Parquet reader ignores. So the columnar body opens in DuckDB, pandas,
pyarrow, or DataFusion with **no infino in the read path** and no export
step:

```python
import infino, pyarrow as pa, glob, duckdb

db = infino.connect("./data")          # persist to disk (not "memory://")
docs = db.create_table(
    "docs",
    pa.schema([
        pa.field("source", pa.large_utf8(), nullable=False),
        pa.field("body", pa.large_utf8(), nullable=False),
    ]),
    infino.IndexSpec().fts("body"),
)
docs.append([
    {"source": "help-center", "body": "To cancel a subscription, open Settings then Billing."},
    {"source": "help-center", "body": "Refunds return to the original payment method."},
    {"source": "blog",        "body": "Enable dark mode under Settings then Appearance."},
])

# The superfiles are ordinary files on disk (one write can shard into
# several, so read them as a set):
files = glob.glob("data/**/*.sf.parquet", recursive=True)
print(files[0])   # e.g. data/docs-18bc4051eb6a9468-0/data/seg-....sf.parquet

# Read them with a third-party engine, no infino in this line:
duckdb.sql("SELECT source, count(*) FROM read_parquet('data/**/*.sf.parquet') GROUP BY source").show()
# ┌─────────────┬──────────────┐
# │   source    │ count_star() │
# ├─────────────┼──────────────┤
# │ help-center │            2 │
# │ blog        │            1 │
# └─────────────┴──────────────┘
```

**Read-only openness.** Standard tools *read* a superfile's columns with
no export step. Rewriting it through a generic Parquet writer (e.g.
`pyarrow.parquet.write_table`) produces valid Parquet that has silently
dropped the embedded BM25/vector indexes, so it's no longer a superfile.
The compatibility is one-directional.

The shortest end-to-end demo (write a corpus, run BM25 + vector +
SQL/hybrid retrieval against it, then read the very same file back with
DuckDB *and* pyarrow) is
[`infino-python/examples/parquet_interop.py`](infino-python/examples/parquet_interop.py).

## Cloud storage

The backend is chosen by the URI scheme — `s3://bucket/prefix`,
`az://container/prefix`, `gs://bucket/prefix`, `file://path`, a bare path,
or `memory://`. Credentials go through `ConnectOptions`, keyed by
`object_store`'s config strings (`aws_*` / `azure_*` / `google_*` — the
names the AWS/Azure/GCS SDKs use). Infino reads no credentials from the
environment; omit them to use ambient cloud identity (IAM instance role /
managed identity / workload-identity ADC).

```rust
use infino::{connect_with, ConnectOptions};

// S3
let db = connect_with("s3://bucket/prefix", ConnectOptions::new()
    .with_storage_option("aws_access_key_id", "…")
    .with_storage_option("aws_secret_access_key", "…")
    .with_storage_option("aws_region", "us-east-1"))?;

// Azure
let db = connect_with("az://container/prefix", ConnectOptions::new()
    .with_storage_option("azure_storage_account_name", "…")
    .with_storage_option("azure_storage_account_key", "…"))?;

// GCS
let db = connect_with("gs://bucket/prefix", ConnectOptions::new()
    .with_storage_option("google_service_account_key", "…"))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Common keys:

| Backend | Keys |
| ------- | ---- |
| S3      | `aws_access_key_id`, `aws_secret_access_key`, `aws_region`, `aws_session_token`, `aws_endpoint` |
| Azure   | `azure_storage_account_name`, `azure_storage_account_key`, `azure_storage_sas_key`, `azure_storage_client_id`, `azure_storage_client_secret`, `azure_storage_tenant_id` |
| GCS     | `google_service_account` (path), `google_service_account_key` (inline JSON), `google_application_credentials`, `google_skip_signature` |

The full set is whatever `object_store` accepts for the backend; an unknown
or cross-backend key is rejected at connect. `with_validate(true)` opts into
a connect-time reachability probe, so bad credentials fail at `connect`
rather than on the first query. The same options exist in the config file
(`storage.storage_options`) and both bindings (`storage_options` +
`validate`).

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

For concepts, quickstart, guides, and examples (Python, Node.js, and Rust), see
the full documentation at **[infino.ai/docs](https://infino.ai/docs)**.

## SQL joins across tables

`query_sql` resolves every table the query names through the catalog into
one engine, and the `bm25_search` / `vector_search` / `hybrid_search`
table functions are relations too — so a single query can fuse keyword
and vector retrieval and join the result to an ordinary table. This is
the canonical agent retrieval, end to end: hybrid-search a knowledge
base, fuse the two rankings (reciprocal-rank fusion), and join provenance
— one snapshot, no client-side stitching.

```rust
use std::sync::Arc;

use arrow_array::{FixedSizeListArray, Float32Array, Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{connect, IndexSpec, Metric};

// Tiny stand-in for your embedding model so this runs as-is; real
// embeddings are dense and higher-dimensional (e.g. 1536).
fn embed(topic: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; 16];
    v[topic] = 1.0;
    v
}

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let db = connect("memory://")?;

// `docs`: text (BM25) + embedding (vector) + the source it came from.
let item = Arc::new(Field::new("item", DataType::Float32, true));
let docs_schema = Arc::new(Schema::new(vec![
    Field::new("source", DataType::LargeUtf8, false),
    Field::new("body", DataType::LargeUtf8, false),
    Field::new("embedding", DataType::FixedSizeList(item.clone(), 16), false),
]));
let docs = db.create_table(
    "docs",
    docs_schema.clone(),
    IndexSpec::new().fts("body").vector("embedding", 16, 1, Metric::Cosine),
)?;
let flat: Vec<f32> = [0usize, 0, 1].iter().flat_map(|&t| embed(t)).collect();
docs.append(&RecordBatch::try_new(
    docs_schema,
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

// `sources`: a plain table — where each source came from, and its trust.
let sources_schema = Arc::new(Schema::new(vec![
    Field::new("source", DataType::LargeUtf8, false),
    Field::new("url", DataType::LargeUtf8, false),
    Field::new("trust", DataType::Int64, false),
]));
let sources = db.create_table("sources", sources_schema.clone(), IndexSpec::new())?;
sources.append(&RecordBatch::try_new(
    sources_schema,
    vec![
        Arc::new(LargeStringArray::from(vec!["help-center", "blog"])),
        Arc::new(LargeStringArray::from(vec![
            "https://help.example.com",
            "https://blog.example.com",
        ])),
        Arc::new(Int64Array::from(vec![2, 1])),
    ],
)?)?;

// The agent's question, embedded like the corpus. The vector TVF takes
// the query vector as a comma-separated string, so build the SQL with it.
let qvec = embed(0).iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
let sql = format!(
    "WITH lexical AS (                       -- BM25 candidates, ranked
         SELECT _id, source, body, ROW_NUMBER() OVER (ORDER BY score DESC) AS rank
         FROM bm25_search('docs', 'body', 'how do I cancel my subscription?', 50)
     ),
     semantic AS (                           -- vector candidates (nearer = lower score)
         SELECT _id, source, body, ROW_NUMBER() OVER (ORDER BY score ASC) AS rank
         FROM vector_search('docs', 'embedding', '{qvec}', 50)
     )
     SELECT s.url,
            COALESCE(l.body, v.body) AS chunk,
            COALESCE(1.0/(60+l.rank), 0.0) + COALESCE(1.0/(60+v.rank), 0.0) AS relevance
     FROM lexical l
     FULL OUTER JOIN semantic v ON l._id = v._id      -- fuse lexical + semantic
     JOIN sources s ON s.source = COALESCE(l.source, v.source)   -- + provenance
     WHERE s.trust >= 1
     ORDER BY relevance DESC
     LIMIT 5"
);
let context = db.query_sql(&sql)?;
assert!(context.iter().map(|b| b.num_rows()).sum::<usize>() >= 1);
# Ok(())
# }
```

**Making it real.** `embed()` here is a 16-dim toy so the example runs as
written; swap in your embedding model and raise `dim` / `n_cent` to match
(e.g. 1536 / 256). The vector TVF takes the query vector as a
comma-separated string — that's the only reason the query is built with
`format!`. The SQL itself is identical from Python and Node; only table
creation and embedding differ.

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
- **MSRV.** The minimum supported Rust version is **1.95** (enforced by
  `rust-version` in `Cargo.toml`). Raising it is a minor bump, never a
  patch.
- **Deprecation.** Post-1.0, removals go through `#[deprecated]` for at
  least one minor release first.
- **Bindings version independently.** The Python (`pip install infino`)
  and Node (`npm install @infino-ai/infino`) packages are versioned on their own
  SemVer lines — each embeds its own copy of the engine, so a binding
  version need not match this crate's. See
  [`docs/versioning.md`](https://github.com/infino-ai/infino/blob/main/docs/versioning.md).

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
before opening a pull request. Browse the full API locally with `make doc`
(`cargo doc --no-deps --open` — the same docs [docs.rs](https://docs.rs/infino) renders).

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
