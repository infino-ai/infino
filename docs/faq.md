# FAQ

Common questions about what Infino is, how it works, and when to use it. For the full guides see [infino.ai/docs](https://infino.ai/docs); for the design references start with [docs/architecture/overview.md](architecture/overview.md).

## What is Infino?

Infino is a fast retrieval engine that stores your data as ordinary Apache Parquet on object storage (local disk, S3, GCS, or Azure) and runs full-text (BM25), vector, hybrid, and SQL search over it from one table. There is no server or managed service to operate: it embeds directly in your application as a Rust, Python, or Node library.

## Do I need a separate vector database?

No. Vector search, full-text search, and SQL all run over the same table, so you do not stand up a dedicated vector store next to your data. You keep one copy of the data, as Parquet, and query it four ways.

## What is a "superfile"?

A superfile is a single valid Parquet file with BM25 and vector indexes embedded inside it. Any Parquet reader can still open the file and read the columns; Infino uses the embedded indexes to search it. The `supertable` layer composes many superfiles into one queryable table with snapshot-isolated reads, append-only writes, and an atomic-commit manifest. See [docs/architecture/superfile.md](architecture/superfile.md) and [docs/architecture/supertable.md](architecture/supertable.md).

## Can I read the same file with other tools?

Yes. Because a superfile is a valid Parquet file, DuckDB, pyarrow, and DataFusion can read the same file directly. You are not locked into a proprietary format. See the [Parquet interop guide](https://infino.ai/docs/guides/parquet-interop).

## Which storage backends are supported?

Local disk, Amazon S3, Google Cloud Storage, and Azure Blob. Infino is object-storage-native: there is no separate stateful cluster holding your index, and the durable state is the Parquet in your bucket. See the [storage guide](https://infino.ai/docs/guides/storage).

## How does hybrid search work?

Hybrid search runs BM25 and vector retrieval and fuses them into one ranked result set in a single pass over the table, so you get lexical and semantic matches together without running two systems and stitching the results yourself. See the [hybrid search guide](https://infino.ai/docs/guides/hybrid-search-on-parquet).

## Can I bring my own embeddings?

Yes. You supply the vectors; Infino indexes and searches them. It does not tie you to a specific embedding model. See the [embeddings guide](https://infino.ai/docs/guides/embeddings).

## Is Infino a good fit for agent memory?

Yes. An agent's long-term memory is a retrieval problem over text and vectors on cheap, durable storage, which is exactly what Infino does: hybrid recall over Parquet in object storage, reachable directly or through MCP. See the [agent memory guide](https://infino.ai/docs/use-cases/agent-memory) and the [MCP integration](https://infino.ai/docs/integrations/mcp).

## Is it embedded, or a service?

Embedded. Infino runs in your process as a library. There is no daemon to run and no control plane between your app and object storage.

## What languages can I use it from?

Rust, Python, and Node.js. See the language and package table in the [README](README.md) and the [quickstart](https://infino.ai/docs/quickstart).

## How fast is it?

Infino's own recorded latency and throughput numbers, with the exact hardware and how to reproduce them, live in [benches/README.md](benches/README.md) and are summarized in the Performance section of the [README](README.md). Comparisons against other engines are shown in the README's "Against other engines" section, which links the public benchmark boards.

## What are the limitations?

Infino is append-first and commit-oriented rather than a transactional row store, and warm queries assume a filled cache. The current limitations are listed in the Limitations section of the [README](README.md). See also [comparisons.md](comparisons.md) for where a different tool is the better fit.

## What is the license?

Apache-2.0.

## How do I get started?

Install the binding for your language and follow the [quickstart](https://infino.ai/docs/quickstart): connect, create a table, append rows, and run your first search. The [README](README.md) has a copy-paste example.
