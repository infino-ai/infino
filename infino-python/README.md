# Infino

[![PyPI](https://img.shields.io/pypi/v/infino.svg)](https://pypi.org/project/infino/)
[![Python](https://img.shields.io/pypi/pyversions/infino.svg)](https://pypi.org/project/infino/)
[![License](https://img.shields.io/pypi/l/infino.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**SQL, full-text, and vector search over your data on object storage — one engine, no server to run.**

Infino keeps your data in Apache Parquet on object storage (local disk,
Amazon S3, or any S3-compatible store) and runs SQL, full-text (BM25),
and vector search over it from a single system. Each file is a valid
Parquet file with BM25 and vector indexes embedded directly inside it; a
table composes many such files with snapshot-isolated reads, append-only
writes, and atomic commits. It runs in your process — there is no daemon,
no cluster, and no managed service to operate.

Apache Arrow is the interchange: schemas and batches cross the boundary
as `pyarrow` objects, and every search returns a `pyarrow.Table`.

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

# Declare a schema and which columns to index. An "_id" column is added
# automatically — you don't define it.
schema = pa.schema([pa.field("title", pa.large_utf8(), nullable=False)])
docs = db.create_table("docs", schema, infino.IndexSpec().fts("title"))

# Append rows. One append is one atomic commit.
docs.append([{"title": "the quick brown fox"}, {"title": "a lazy dog"}])

# Full-text search. Returns a pyarrow.Table of (_id, score).
hits = docs.bm25_search("title", "fox", k=10)
print(hits.column_names)        # ['_id', 'score']
```

## Core concepts

- **Connection** — a handle to a catalog (a set of tables under one URI).
  Open it with `infino.connect(uri)`.
- **Table** — an append-only, snapshot-isolated collection of rows. Each
  table carries an auto-generated `_id` column.
- **IndexSpec** — declares which columns are full-text (BM25) and which
  are vector indexed. Columns without an index are still stored,
  filterable in SQL, and returnable via projection.
- **Commits** — every `append`, `update`, and `delete` is a single atomic
  commit. Readers see a consistent snapshot and are never torn by a
  concurrent write. Batch rows into one `append` rather than calling it
  per row — each call writes a file and commits.
- **Arrow everywhere** — searches return `pyarrow.Table`; `append` and
  `update` accept Arrow, pandas, or `list[dict]`.

## Full-text search

```python
docs = db.create_table("docs", schema, infino.IndexSpec().fts("title"))
docs.append([{"title": "the quick brown fox"}, {"title": "a lazy dog"}])

# Ranked BM25 — higher score is a better match.
docs.bm25_search("title", "quick fox", k=10)               # OR by default
docs.bm25_search("title", "quick fox", k=10, mode="and")   # require all terms

# Unranked matching (score is 0.0): every row containing the term(s),
# or an exact whole-value match.
docs.token_match("title", "fox")
docs.exact_match("title", "the quick brown fox")
```

## Vector search

Vector columns are `fixed_size_list<float32, dim>` with `dim` in
`[16, 4096]`. The distance metric is fixed when you declare the index
(`"cosine"`, `"l2sq"`, or `"negdot"`); for vector results a smaller score
is nearer.

```python
dim = 384
schema = pa.schema([pa.field("emb", pa.list_(pa.float32(), dim), nullable=False)])
spec = infino.IndexSpec().vector("emb", dim, n_cent=256, metric="cosine")
vecs = db.create_table("vecs", schema, spec)

vecs.append(pa.record_batch([pa.array(embeddings, type=pa.list_(pa.float32(), dim))],
                            names=["emb"]))

vecs.vector_search("emb", query_vector, k=10)              # query_vector: list[float]
vecs.vector_search("emb", query_vector, k=10, nprobe=32)   # probe more partitions
```

To restrict the kNN to rows matching a text predicate, pass `filter_column`
and `filter_query` together (the column must be FTS-indexed) — a pushdown
pre-filter, so results are the nearest *matching* rows, not a post-filter over
the global top-k. `filter_mode` is `"or"` (default) or `"and"`:

```python
vecs.vector_search("emb", query_vector, k=10,
                   filter_column="body", filter_query="cancel subscription")
```

## SQL

Run SQL across the catalog's tables for analytics and filtering. Results
come back as a `pyarrow.Table`.

```python
db.query_sql("SELECT COUNT(*) AS n FROM docs")
db.query_sql("SELECT title FROM docs WHERE title = 'a lazy dog'")
```

### Search inside SQL

Search is also exposed as table-valued functions, so a ranked retrieval is
a relation you can join, filter, and aggregate over. Each takes the table
name as its first argument and yields `_id`, any scalar columns, and a
trailing `score`:

| Function                                                        | Returns                                  |
| --------------------------------------------------------------- | ---------------------------------------- |
| `bm25_search(table, column, query, k)`                          | Ranked BM25 (higher score is better)     |
| `bm25_search_prefix(table, column, prefix, k)`                  | BM25 with the last term prefix-expanded  |
| `vector_search(table, column, query_vector, k)`                 | kNN (smaller score is nearer)            |
| `hybrid_search(table, text_col, query, vec_col, query_vector, k)` | BM25 + vector fused with RRF (higher is better) |
| `token_match(table, column, query)`                             | Unranked term match (`score` is `0.0`)   |
| `exact_match(table, column, value)`                             | Unranked exact-value match               |

A query vector is written as a comma-separated string or a SQL array
literal; build it from a Python list with `",".join(map(str, vec))`.

```python
# Hybrid search: lexical + vector, fused by reciprocal-rank fusion.
qv = ",".join(map(str, query_vector))
db.query_sql(f"""
    SELECT _id, score
    FROM hybrid_search('docs', 'title', 'quick fox', 'emb', '{qv}', 10)
    ORDER BY score DESC
""")

# Compose retrieval with relational filtering and the catalog's other tables.
db.query_sql("""
    SELECT s._id, s.score
    FROM bm25_search('docs', 'title', 'fox', 50) AS s
    WHERE s._id IN (SELECT _id FROM docs WHERE title <> 'a lazy dog')
""")
```

`hybrid_search` and `bm25_search_prefix` are reachable only through SQL.
The SQL `vector_search` takes no `nprobe` or filter arguments — use the
`Table.vector_search` method when you need those.

## Projections

By default a search returns just `_id` and `score` — no row data is
decoded. Name the columns you want to materialize:

```python
docs.bm25_search("title", "fox", k=10)                          # _id + score only
docs.bm25_search("title", "fox", k=10, projection=["_id", "title", "score"])
vecs.vector_search("emb", query_vector, k=10, projection=["_id", "emb", "score"])
```

The SQL table-valued functions differ: they decode every scalar column by
default (`_id`, all scalars, trailing `score`). To return only `_id` and
`score`, select those columns explicitly in the query.

## Updates and deletes

Mutations require durable storage (a local path or object store, not
`memory://`). The predicate is a SQL boolean expression — the same thing
you would write after `WHERE` — evaluated against the table's columns.

```python
docs.append([{"title": "draft post"}, {"title": "spam"}])

# Delete every row matching the predicate.
docs.delete("title = 'spam'")

# Replace matched rows 1:1 with new rows (same input shapes as append).
stats = docs.update("title = 'draft post'", [{"title": "published post"}])
print(stats.matched, stats.n_tombstoned, stats.n_not_found)
```

`update` is a one-to-one replacement: the number of rows the predicate
matches must equal the number of rows you supply, otherwise it raises.
Both methods return a `MutationStats` with `matched`, `n_tombstoned`, and
`n_not_found`.

## Optimization

Many small appends produce many small files. `optimize` merges small or
underfilled files into larger ones, which keeps reads efficient.

```python
docs.optimize()                                             # engine defaults
docs.optimize(infino.OptimizeOptions(target_superfile_size_mb=256,
                                     min_fill_percent=50))
```

## Storage backends

`connect` selects the backend from the URI:

| URI                   | Backend                                  |
| --------------------- | ---------------------------------------- |
| `./data`, `/abs/path` | Local filesystem                         |
| `s3://bucket/prefix`  | Amazon S3 / S3-compatible object storage |
| `memory://`           | In-process, ephemeral (testing)          |

For S3-compatible stores that need an explicit endpoint and static
credentials, pass them as keyword arguments (omit them to use ambient AWS
credentials):

```python
db = infino.connect(
    "s3://bucket/prefix",
    endpoint="https://s3.example.com",
    region="us-east-1",
    access_key="…",
    secret_key="…",
)
```

### Local disk cache

For object-storage-backed catalogs, a local disk cache keeps hot data on
fast local storage. `cold_fetch_mode` controls how cache misses are
served: `"hybrid_with_prefetch"`, `"range_only"`, or
`"lazy_foreground_with_background_fill"`.

```python
db = infino.connect(
    "s3://bucket/prefix",
    cache_dir="/mnt/nvme/infino-cache",
    cache_budget_bytes=64 * 1024**3,
    cold_fetch_mode="lazy_foreground_with_background_fill",
)
```

## Schema and type requirements

- Full-text columns must be Arrow `large_utf8`.
- Vector columns must be `fixed_size_list<float32, dim>` with `dim` in
  `[16, 4096]`.
- The `_id` column is generated by the engine; do not declare it.
- `append` and `update` accept a `pyarrow.RecordBatch` or `Table`, a
  pandas `DataFrame`, or a `list[dict]`, coerced to Arrow against the
  table's declared schema.

## API reference

- `infino.connect(uri, *, endpoint=None, region=None, access_key=None, secret_key=None, cache_dir=None, cache_budget_bytes=None, cold_fetch_mode=None) -> Connection`
- `Connection`
  - `create_table(name, schema, index_spec) -> Table`
  - `open_table(name) -> Table`
  - `drop_table(name, purge=False)` — `purge=True` also deletes the data
  - `list_tables() -> list[str]`
  - `query_sql(sql) -> pyarrow.Table` — also exposes the search
    table-valued functions `bm25_search`, `bm25_search_prefix`,
    `vector_search`, `hybrid_search`, `token_match`, and `exact_match`
    (each takes the table name first; `hybrid_search` and
    `bm25_search_prefix` are SQL-only)
- `Table`
  - `append(data)`
  - `bm25_search(column, query, k, mode="or", projection=None) -> pyarrow.Table`
  - `vector_search(column, query, k, nprobe=None, filter_column=None, filter_query=None, filter_mode=None, projection=None) -> pyarrow.Table`
  - `token_match(column, query, mode="or", projection=None) -> pyarrow.Table`
  - `exact_match(column, value, projection=None) -> pyarrow.Table`
  - `delete(predicate) -> MutationStats`
  - `update(predicate, new_rows) -> MutationStats`
  - `optimize(settings=None)`
  - `schema() -> pyarrow.Schema`
- `IndexSpec().fts(column).vector(column, dim, n_cent, metric)`
- `OptimizeOptions(max_memory_mb=None, min_fill_percent=None, target_superfile_size_mb=None)`
- `MutationStats` — returned by `delete` / `update`; read-only attributes `matched`, `n_tombstoned`, `n_not_found`

## Building from source

The bindings are built with [maturin](https://www.maturin.rs/). Building
requires a Rust toolchain and access to crates.io.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest pyarrow
maturin develop          # compile the extension and install it into the venv
pytest tests/
```

Or with [uv](https://docs.astral.sh/uv/):

```sh
uv venv && source .venv/bin/activate
uv pip install maturin pytest pyarrow
maturin develop          # compile the extension and install it into the venv
uv run pytest tests/
```

## License

Apache-2.0.
