# infino — Python bindings

PyO3 + maturin bindings over infino's catalog API. Sync; Arrow is the
interchange.

```python
import infino
import pyarrow as pa

db = infino.connect("memory://")            # or "./data", "s3://bucket/prefix"
schema = pa.schema([pa.field("title", pa.large_utf8(), nullable=False)])
docs = db.create_table("docs", schema, infino.IndexSpec().fts("title"))

docs.append(pa.record_batch([pa.array(["the quick brown fox"])], names=["title"]))

hits = docs.bm25_search("title", "fox", 10)         # [{"_id": ..., "score": ...}]
table = db.query_sql("SELECT _id, score FROM bm25_search('docs', 'title', 'fox', 10)")
```

## Build & test (requires online crates.io access)

This crate is **excluded** from the infino cargo workspace so the core
Rust crate doesn't require Python to build. Build it standalone with
maturin (not `cargo build -p`, which would need workspace membership):

```sh
cd infino-python
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest pyarrow
maturin develop          # compile the extension + install into the venv
pytest tests/
```

## Scope

- `connect(uri, *, endpoint, region, access_key, secret_key)` — backend
  from the URI scheme; S3-compatible static creds via kwargs.
- `Connection`: `create_table(name, pyarrow.Schema, IndexSpec)`,
  `open_table`, `drop_table`, `list_tables`, `query_sql` → pyarrow Table.
- `Table`: `append(...)`, `bm25_search`, `vector_search`, `schema`.
  `append` accepts a pyarrow `RecordBatch` or `Table`, a pandas
  `DataFrame`, or a `list[dict]` — coerced to Arrow against the table's
  declared schema (Python sources are nullable; null-free columns are
  re-wrapped to match). One `append` is one commit.
- `IndexSpec().fts(col).vector(col, dim, n_cent, metric)`.

Next: `update` / `delete`; richer `ConnectOptions` (disk cache);
abi3 wheels + CI for distribution.
