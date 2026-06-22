# FAQ

Operational and architecture answers for evaluating Infino. Each answer is
short; follow the link for the canonical detail.

## Does Infino need a server?

No. Infino is an **embedded engine, not a server** — it runs in-process, inside
your application. You add it as a library (`cargo add infino`,
`pip install infino`, or the npm package) and open a connection to a storage
root from your own code; the engine, including SQL (DataFusion under the hood),
executes in your process. There is no wire protocol yet, so external SQL clients
can't attach — SQL is reached through the connection's `query_sql`. See
[Opening Infino](architecture/overview.md#opening-infino).

## How do concurrent writes work?

**One writer is active per table at a time.** Appends, updates, and deletes are
staged on a writer and made durable by a single atomic commit — nothing is
persisted until the commit succeeds, and a reader sees either the pre-commit or
the post-commit state, never a partial one. Across processes the commit is
guarded: the pointer to the current manifest is swapped only if it hasn't moved,
so a writer working from a stale manifest can't overwrite another's commit — a
conflicting publish refreshes from the current manifest and retries with
backoff, up to a bounded number of attempts before surfacing a contention error.
See [Write](architecture/supertable.md#write),
[Commit pipeline](architecture/supertable.md#commit-pipeline), and
[Concurrency](architecture/supertable.md#concurrency).

## How fresh are reads? What is the consistency model?

**Snapshot isolation.** A reader pins the manifest that is current when it opens
and never observes a partially applied commit; publication is atomic, so a read
returns the table as of its pinned snapshot. Read freshness under concurrent
writers is governed by the table's configured consistency policy and applied by
the engine on every read — callers never refresh by hand. See
[Manifest](architecture/supertable.md#manifest) and
[Lifecycle](architecture/supertable.md#lifecycle).

## How do multiple processes or hosts share a table?

Through storage, not a coordinator. For a persistent table the backend holds the
superfiles, the manifest, and a **pointer to the current manifest**. A commit
writes the new superfiles and manifest first, then swings the pointer in one
atomic step, so any reader resolving the pointer lands on a complete manifest.
Concurrent writers from different processes are serialized by that guarded
pointer update (see concurrent writes above), so multiple processes or hosts can
read and write the same table on shared object storage with no separate service.
Verified by
[`tests/supertable_concurrent_processes.rs`](../tests/supertable_concurrent_processes.rs);
see [Storage](architecture/supertable.md#storage).

## Are writes durable and crash-safe?

Yes. Nothing is persisted until a commit, and a commit publishes atomically, so
a crash mid-write leaves the previously committed snapshot intact — there is no
half-applied state. Committed superfiles surviving an abort mid-flight is
verified by
[`tests/supertable_commit_crash_localfs.rs`](../tests/supertable_commit_crash_localfs.rs).

## Can DuckDB, pyarrow, or DataFusion read Infino's files?

Yes. Each superfile is a **valid Parquet file** — it begins with Parquet data
and ends with a standard Parquet footer — so DataFusion, DuckDB, and pyarrow can
open it as a normal table and project columns, filter rows, and run SQL over the
columnar data with no Infino-specific support. Compatibility is a property of
the bytes, not a conversion step. See
[Parquet compatibility](architecture/superfile.md#parquet-compatibility).

## Are the search indexes visible to ordinary Parquet readers?

The **scalar and text columns are** — they're ordinary Parquet columns any
reader sees. The **full-text and vector index regions are not**: they lie
outside the row-group ranges described by the footer and are namespaced in the
footer's key-value metadata, so a standard reader simply skips them. Infino uses
that same footer to locate the indexes when search is requested.

**Round-trip caveat:** because the indexes live outside the standard Parquet
structures, reading a superfile and rewriting it through a generic Parquet
writer preserves the columns but **drops the index regions** — the output is
still valid Parquet, but no longer searchable by Infino without re-indexing. See
[Parquet compatibility](architecture/superfile.md#parquet-compatibility).

## How does SQL run with no server?

DataFusion executes in-process; you reach it through the connection's
`query_sql`, which resolves every table the query names through the catalog into
one engine, so joins and aggregations span tables. The retrievers are also SQL
table-valued functions (`bm25_search`, `vector_search`, `token_match`,
`exact_match`, `hybrid_search`), so search composes into a SQL plan as a
relation.

```sql
-- Search is a table function, so it composes into a SQL plan as a relation:
-- rank `docs` by BM25, then group and count — one query, no separate service.
SELECT source, COUNT(*) AS hits
FROM bm25_search('docs', 'body', 'cancel subscription', 50)
GROUP BY source
```

See [Queries](architecture/supertable.md#queries).
