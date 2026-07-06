# Infino

[![PyPI](https://img.shields.io/pypi/v/infino.svg)](https://pypi.org/project/infino/)
[![Python](https://img.shields.io/pypi/pyversions/infino.svg)](https://pypi.org/project/infino/)
[![Downloads](https://img.shields.io/pypi/dm/infino.svg)](https://pypi.org/project/infino/)
[![License](https://img.shields.io/pypi/l/infino.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**SQL, full-text, and vector search over your data on object storage — one engine, no server to run.**

Infino keeps your data in Apache Parquet on object storage (local disk,
Amazon S3, or any S3-compatible store) and runs SQL, full-text (BM25),
and vector search over it from a single system. Each file is a valid
Parquet file with BM25 and vector indexes embedded directly inside it; a
table composes many such files with snapshot-isolated reads, append-only
writes, and atomic commits. It runs in your process — there is no daemon,
no cluster, and no managed service to operate.

Use it for **RAG**, **agent memory**, **hybrid search**, and **semantic
search**: an embedded **vector database**, **full-text (BM25)** search
engine, and **SQL** query engine in one library.

## Installation

```sh
pip install infino
```

Or with [uv](https://docs.astral.sh/uv/):

```sh
uv add infino            # add to a uv-managed project
uv pip install infino    # install into the active environment
```

Requires Python 3.9 or newer. `pyarrow` is installed as a dependency;
`pandas` is optional and used only if you pass DataFrames.

## Quickstart

```python
import infino
import pyarrow as pa

# Connect to a catalog. Use a local path or an S3 URI for durable storage;
# "memory://" is ephemeral and handy for tests.
db = infino.connect("./data")

# Tiny stand-in for your embedding model so this runs as-is — a 16-dim
# one-hot by topic. Real embeddings are dense and higher-dimensional.
def embed(topic):
    v = [0.0] * 16
    v[topic] = 1.0
    return v

# Declare a schema and which columns to index. An "_id" column is added
# automatically — you don't define it.
schema = pa.schema([
    pa.field("source", pa.large_utf8(), nullable=False),
    pa.field("body", pa.large_utf8(), nullable=False),
    pa.field("embedding", pa.list_(pa.float32(), 16), nullable=False),
])
docs = db.create_table(
    "docs", schema, infino.IndexSpec().fts("body").vector("embedding", 16, 1, "cosine")
)

# Append rows. One append is one atomic commit.
docs.append([
    {"source": "help-center", "body": "To cancel a subscription, open Settings then Billing.", "embedding": embed(0)},
    {"source": "help-center", "body": "Refunds return to the original payment method.",         "embedding": embed(0)},
    {"source": "blog",        "body": "Enable dark mode under Settings then Appearance.",        "embedding": embed(1)},
])

# Retrieve context to ground an agent's next answer — keyword, vector,
# hybrid (BM25 + vector fused in one pass), or SQL. Each returns a pyarrow.Table.
keyword  = docs.bm25_search("body", "cancel subscription", k=5)                           # BM25
semantic = docs.vector_search("embedding", embed(0), k=5)                                 # vector kNN
hybrid   = docs.hybrid_search("body", "cancel subscription", "embedding", embed(0), k=5)  # fused
billing  = db.query_sql("SELECT body FROM docs WHERE source = 'help-center'")             # SQL filter
```

## Documentation

Full docs, guides, and the API reference live at **[infino.ai/docs](https://infino.ai/docs)**:

- [Quickstart](https://infino.ai/docs/quickstart) — install to first query
- [Core concepts](https://infino.ai/docs/core-concepts) — superfiles, commits, and indexes
- Guides — [Tables & indexing](https://infino.ai/docs/guides/tables) ·
  [Search: BM25, vector, hybrid](https://infino.ai/docs/guides/search) ·
  [Embeddings](https://infino.ai/docs/guides/embeddings) ·
  [Storage & credentials](https://infino.ai/docs/guides/storage)
- [SQL reference](https://infino.ai/docs/sql-reference) — query tables and the search table-valued functions
- [API reference](https://infino.ai/docs/api-reference) — the full Python surface, generated from the package
- [Integrations](https://infino.ai/docs/integrations) — LangChain, CrewAI, Vercel AI SDK, MCP

## Building from source

The bindings are built with [maturin](https://www.maturin.rs/) and require a
Rust toolchain.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest pyarrow
maturin develop          # compile the extension and install it into the venv
pytest tests/
```

## License

Apache-2.0.
