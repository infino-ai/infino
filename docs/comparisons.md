# How Infino compares

Infino is a retrieval engine that runs full-text (BM25), vector, hybrid, and SQL search over ordinary Parquet on object storage, embedded in your application with no server to operate. This page explains how it relates to the categories of tools you might already use, and where a different tool is the better fit. Head-to-head numbers are on the public benchmark boards linked below and in the "Against other engines" section of the [README](README.md); this page is about the architectural differences.

## vs a dedicated vector database

Dedicated vector databases store your vectors in their own service that you run and scale separately from your data. Infino keeps one copy of the data as Parquet in your own bucket and adds BM25 and SQL over the same rows, so vector search is one of four ways to query a single table rather than a separate system to operate and sync. Vector-search latency and recall against vector databases are on the [VectorDBBench board](https://vdbbench-viewer-q6unoyyhua-uc.a.run.app/results).

## vs a Postgres extension for vectors

A Postgres vector extension adds vector columns to relational rows inside a running database server. Infino stores data as columnar Parquet on object storage and adds full-text and SQL analytics alongside vectors, with no database server to run and no separate storage tier to manage. Reach for Postgres when your vectors live next to transactional relational data you are already serving from Postgres; reach for Infino when the data is a large corpus on object storage that you want to search and analyze without standing up and scaling a database.

## vs a search engine cluster

Full-text search engines run a cluster (typically JVM) with their own storage format and node management. Infino embeds in your process and reads and writes Parquet in object storage, so there is no cluster to size, no separate index format, and no data movement out of your lake. Full-text latency is on [Search Benchmark, the Game](https://tantivy-search.github.io/bench/), and SQL against search engines is on the [ClickBench board](https://benchmark.clickhouse.com/#system=-&type=+sac&machine=+c6a.4xlarge&cluster_size=-&opensource=-&hardware=+c&tuned=+n&metric=hot&queries=-). Reach for a search engine cluster when you need its full operational feature set (for example rich aggregations tooling and a managed multi-node deployment); reach for Infino when you want embedded search over data that already lives as Parquet.

## vs an embedded vector library

Embedded ANN libraries give you fast vector search in-process, but they are vector-only: you still need a separate system for full-text and SQL, and they do not persist your data in an open, queryable format. Infino is also embedded and in-process, but a superfile is a valid Parquet file with BM25 and vector indexes inside it, so the same file is searchable by full-text, vector, and SQL, and readable by any Parquet tool. Quantized vector-index quality against embedded libraries is on [RetrievalBench](https://github.com/infino-ai/retrievalbench).

## vs a query engine over Parquet

Query engines such as DuckDB and DataFusion give you SQL over Parquet, but no built-in BM25 or vector index, so full-text and semantic search are not part of the file. Infino embeds those indexes into the Parquet file itself, so the same file supports SQL and full-text and vector search. In fact a superfile stays readable by those engines, so you can keep using them for analytics on the same data. SQL performance against analytic engines is on the [ClickBench board](https://benchmark.clickhouse.com/#system=+ClickHouse%7CDuckDB%7CInfino%7CDataFusion%20%28Parquet%2C%20single%29&machine=+c6a.4xlarge&cluster_size=-&type=-&metric=hot).

## When Infino is not the right tool

Infino is append-first and commit-oriented, not a transactional row store, so if you need per-row updates with immediate read-your-write transactional semantics, a database is the better fit. If you need a managed, always-on multi-node service that someone else operates for you, a hosted search or vector service fits that shape better. And warm-query latency assumes a filled cache: the first cold query pays file opens and cache fill. The current limitations are listed in the Limitations section of the [README](README.md).

## Summary

| You want | Infino | A dedicated system |
|---|---|---|
| One copy of the data, as open Parquet | yes | usually a separate store |
| Full-text, vector, hybrid, and SQL over one table | yes | typically one modality each |
| Embedded, no server or cluster to run | yes | usually a running service |
| Runs directly on object storage | yes | often its own storage tier |
| Transactional per-row updates | no | a database fits better |
| Fully managed multi-node service | no | a hosted service fits better |

See the [FAQ](faq.md) for the shorter questions and [infino.ai/docs](https://infino.ai/docs) for the guides.
