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

rows = docs.bm25_search("title", "fox", 10)                                # pyarrow.Table (_id, title, score)
ids = docs.bm25_search("title", "fox", 10, projection=["_id", "score"])    # no scalar decode
table = db.query_sql("SELECT _id, score FROM bm25_search('docs', 'title', 'fox', 10)")
```

### Update & delete

Mutations need durable storage (a local path or object store, not
`memory://`). The predicate is a SQL boolean expression — the same thing
you'd put after `WHERE` — evaluated against the table's columns:

```python
db = infino.connect("./data")
docs = db.create_table("docs", schema, infino.IndexSpec().fts("title"))
docs.append([{"title": "draft post"}, {"title": "spam"}])

docs.delete("title = 'spam'")                          # drop matching rows
stats = docs.update("title = 'draft post'",            # replace matched rows 1:1
                    [{"title": "published post"}])
stats.matched, stats.n_tombstoned                      # -> (1, 1)

docs.compact()                                         # merge small superfiles
```

`update` is a 1:1 replacement: the number of rows the predicate matches
must equal the number of rows you supply, or it raises. `new_rows` takes
the same shapes as `append` (pyarrow, pandas, or `list[dict]`).

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
- `Table`: `append(...)`, `schema`, and the search surface —
  `bm25_search` / `vector_search` / `token_match` / `exact_match` all
  return a pyarrow `Table`. `projection` names the output columns
  (`_id`, any scalar column, or `score`); omitting it returns the
  engine-native `_id` + `score` pair with no scalar decode —
  materializing row data is an explicit opt-in by naming columns. The
  unranked `token_match` / `exact_match` rows carry `score == 0.0`.
  `append` accepts a pyarrow `RecordBatch` or `Table`, a pandas
  `DataFrame`, or a `list[dict]` — coerced to Arrow against the table's
  declared schema (Python sources are nullable; null-free columns are
  re-wrapped to match). One `append` is one commit.
- `Table` mutations — `delete(predicate)` and `update(predicate, new_rows)`
  take a SQL predicate string (e.g. `"status = 'spam'"`); `update` replaces
  the matched rows 1:1 with `new_rows` (same shapes as `append`). Both
  return `MutationStats` (`matched`, `n_tombstoned`, `n_not_found`) and need
  durable storage (not `memory://`).
- `Table.compact(CompactOptions(...))` — merge small / underfilled
  superfiles; omit the options for engine defaults.
- `IndexSpec().fts(col).vector(col, dim, n_cent, metric)`.

Next: richer `ConnectOptions` (disk cache); abi3 wheels + CI for
distribution.
