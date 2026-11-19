# How Infino compares

Infino is a search library built as an alternative to Lucene. It runs full-text (BM25), vector, hybrid, and SQL search over ordinary Parquet on local disk or object storage, embedded in your application with no server to operate. This page starts with Lucene and the engines built on it, then covers the other tools you might be weighing. Head-to-head numbers are in the [README](../README.md#infino-and-lucene) and on the public benchmark boards linked below; this page is about the architectural differences.

## vs Lucene

Lucene and Infino do the same job: an in-process library you index documents into and search. They differ in where the index lives and what can read it.

- **Format.** A Lucene index is a set of segment files that only Lucene can read, and it is its own copy of the data. An Infino table is Parquet files with the BM25 and vector indexes embedded in them, so DuckDB, pyarrow, and DataFusion read the same files as ordinary tables. Teams that want their search data in a lake too run no export pipeline.
- **Storage.** Lucene reads a directory on local disk. Infino reads local disk or object storage (S3, GCS, Azure) through a cache that holds only the byte ranges queries touch, so tables much larger than RAM or disk stay searchable.
- **Commits.** Lucene writes segments, refreshes readers, and merges in the background. Infino writes immutable files and swaps a manifest atomically; a reader keeps the version it opened, and `optimize()` compacts.
- **Full-text.** Both score with BM25. Lucene uses Block-Max WAND; Infino chooses per query among WAND, Block-Max WAND, MaxScore, stored counts, and bitsets. On [Search Benchmark, the Game](https://tantivy-search.github.io/bench/), Infino answers top-k and count queries in about half of Lucene's time.
- **Vectors.** Lucene keeps an HNSW graph per segment and is fastest when the graphs are held in memory. Infino's default is IVF with 1-bit codes paged from object storage, so pinned memory stays small at any scale; HNSW and flat modes are there when the data fits in RAM.
- **SQL.** Lucene has none. Infino runs full SQL on Apache DataFusion, with joins, aggregates, and window functions, and exposes BM25, vector, and hybrid search as table functions inside the statement.

Tantivy, a Rust library modeled on Lucene, shares its segment design; it is on the same Search Benchmark chart.

**Not in Infino yet:** fuzzy matching, highlighting, geo queries, and a wider set of language analyzers, all of which Lucene has. If you need one of them, [open a feature request](https://github.com/infino-ai/infino/issues/new).

## vs Elasticsearch and OpenSearch

Elasticsearch and OpenSearch run Lucene inside each shard and add the cluster around it: they place shards on nodes, copy each shard to replicas on other nodes' disks, and recover a lost node by copying shards between peers. Both can back colder tiers with object storage (searchable snapshots), but the hot tier is Lucene on local disks, multiplied by the replica count.

Infino's durable copy is the table in object storage, so compute is disposable and there is no shard rebalancing or peer recovery to run. The same table answers full SQL directly, where Elasticsearch and OpenSearch translate a SQL dialect into their query DSL.

On the [ClickBench board](https://benchmark.clickhouse.com/#system=-&type=+sac&machine=+c6a.4xlarge&cluster_size=-&opensource=-&hardware=+c&tuned=+n&metric=hot&queries=-), Infino uses more than ten times less CPU per query than Elasticsearch. Vector search against Elastic Cloud and OpenSearch is on the [VectorDBBench board](https://vdbbench-viewer-q6unoyyhua-uc.a.run.app/results).

**Where they fit better:** reach for Elasticsearch or OpenSearch when you need a managed multi-node service with their full API, dashboards, and plugin ecosystem, or the Lucene features listed above.

## vs a dedicated vector database

Dedicated vector databases store your vectors in their own service that you run and scale separately from your data. Infino keeps one copy of the data as Parquet in your own bucket and adds BM25 and SQL over the same rows, so vector search is one of four ways to query a single table rather than a separate system to operate and sync. Vector-search latency and recall against vector databases are on the [VectorDBBench board](https://vdbbench-viewer-q6unoyyhua-uc.a.run.app/results).

## vs an embedded vector library

Embedded ANN libraries give you fast vector search in-process, but they are vector-only: you still need a separate system for full-text and SQL, and they do not persist your data in an open, queryable format. Infino is also embedded, and the same file is searchable by full-text, vector, and SQL, and readable by any Parquet tool. Quantized vector-index quality against embedded libraries is on [RetrievalBench](https://github.com/infino-ai/retrievalbench).

## vs Postgres and query engines over Parquet

A Postgres vector extension adds vector columns to relational rows inside a running database server; reach for it when your vectors live next to transactional data you already serve from Postgres. Query engines such as DuckDB and DataFusion give you SQL over Parquet but no BM25 or vector index. A superfile stays readable by those engines, so you can keep using them for analytics on the same data. SQL against analytic engines is on the [ClickBench board](https://benchmark.clickhouse.com/#system=+ClickHouse%7CDuckDB%7CInfino%7CDataFusion%20%28Parquet%2C%20single%29&machine=+c6a.4xlarge&cluster_size=-&type=-&metric=hot).

## When Infino is not the right tool

Infino is append-first and commit-oriented, not a transactional row store, so if you need per-row updates with immediate read-your-write semantics, a database is the better fit. If you need a managed, always-on multi-node service that someone else operates for you, a hosted search or vector service fits that shape better. And warm-query latency assumes a filled cache: the first cold query pays file opens and cache fill. The current limitations are listed in the Limitations section of the [README](../README.md#limitations).

## Summary

| You want | Infino | Lucene, Elasticsearch, OpenSearch |
|---|---|---|
| One copy of the data, as open Parquet | yes | a separate index in Lucene's format |
| Indexes on object storage, compute disposable | yes | local disks per replica; object storage for cold tiers |
| Full-text, vector, hybrid, and SQL over one table | yes | full-text and vectors; SQL translated to the query DSL |
| Embedded in your process | yes | Lucene yes; Elasticsearch and OpenSearch run a cluster |
| Fuzzy matching, highlighting, geo, many analyzers | not yet | yes |
| Transactional per-row updates | no | no; a database fits better |

See the [FAQ](faq.md) for the shorter questions and [infino.ai/docs](https://infino.ai/docs) for the guides.
