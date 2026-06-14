# Infino Architecture Overview

A plain-language tour of what Infino is, how it's built, and how it
differs from the database, search engine, and vector database alternatives. 
For more technical detail, see [superfile](./superfile.md) (the superfile format) and
[supertable](./supertable.md) (the table layer).

## What is Infino?

**Infino is a fast retrieval engine that keeps your data on cheap object storage (like Amazon S3) and runs SQL, full-text search, and vector search over it from a single system.**

For most retrieval and RAG workloads, hybrid queries are useful for improving the cost, latency, and accuracy of results. In Infino, SQL, keyword, and vector run over **one copy of the data** through **one query path**, making **hybrid search** a first-class, single-pass operation.

 The data lives in object storage at object-storage prices, and  
compute pulls in only what a query actually touches, caching the hot  
parts locally for speed. You do not need to maintain a separate cluster.

## The mental model

Think of three ideas:

1. **The data is just files.** Every chunk of the table is one
  self-contained file — a *superfile* — that holds the raw columns
   *and* the search indexes (keyword + vector) together. There are no
   sidecar index files to keep in sync, and the file is also a valid
   Apache Parquet file, so standard analytics tools (DuckDB,
   DataFusion, pandas/pyarrow) can read it directly.
2. **Files are never edited, only added.** A *supertable* is described
  by its *manifest* — the list of which files currently make up the
   table. Writes create new files and publish a new manifest; they never
   modify existing files. This makes consistency simple and snapshots
   essentially free.
3. **Storage and compute are separate.** The source of truth is object
  storage. Compute nodes are stateless readers that fetch byte ranges
   on demand and keep a local cache of hot data. You scale storage and
   compute independently and pay for each separately.

```text
        Application
            │  connect(uri) → Connection
            ▼
        Connection  ── catalog of tables (name → supertable) ──┐
            │  create / open / list / drop · cross-table SQL    │
            ▼                                                    │
        Supertable  ── manifest (the file list) ──┐             │
            │                                      │             │
   stateless compute                       immutable files      │
   + local cache (hot)                     (superfiles)          │
            │                                      │             │
            └──────────── byte-range reads ────────┴─────────────┘
                                 ▼
                      Object storage (S3) — cheap, durable,
                              the source of truth
```

## Opening Infino

Infino is an **embedded engine, not a server** — it runs inside your
process, the way SQLite or DuckDB do. You install it as a library (the
Rust crate via `cargo add infino`, or the Python wheel via
`pip install infino`) and open a **connection** to a root location on
storage from your own code; the engine, including its SQL (DataFusion
under the hood), executes in-process. There is no wire protocol yet, so
external SQL clients can't attach — SQL is reached through the
connection's `query_sql`. From that connection you work with tables:

- `connect("s3://bucket/prefix")` (or a local path, or `memory://`)
  returns a **`Connection`** — a *catalog of tables* persisted at that
  root. (Distinct from a supertable's *manifest*, which lists the files
  within one table.)
- From the connection you **create**, **open**, **list**, and **drop**
  tables, and run **SQL across them** (`query_sql`) — joins and
  aggregations span tables in the catalog in one engine.
- Each table is a **`Supertable`** handle: `append` rows, `update` /
  `delete`, and search it — `bm25_search` / `vector_search` (full-text
  and vector kNN, returning Arrow rows), the unranked `token_match` /
  `exact_match`, and `schema`. The same retrievers are also SQL
  table-valued functions, so search composes with the rest of a query.

The connection is the only entry point; everything below it (manifest,
storage, cache, query fan-out) is internal.

## How a query runs

To minimize latency and cost, infino tries to optimize data layout and data accesses so that a query rarely needs to download the whole dataset. It:

1. Starts from a **pinned snapshot** of the manifest (so concurrent
  writes never change the answer mid-query).
2. **Prunes** — uses small per-file summaries in the manifest
  (value ranges, a keyword "is this term present?" filter, vector
   centroids) to skip files that can't possibly match. This reads only
   the catalog, never file contents. The same summaries back every
   modality through one shared pruning layer, so a hybrid query prunes
   on scalar, keyword, *and* vector signals together before touching a
   single byte of a superfile.
3. **Fetches only what it needs** — for surviving files it pulls just
  the relevant byte ranges from object storage (a posting list, a
   handful of vector clusters), not the whole file. This holds for SQL
   too: a keyword filter on an indexed text column is answered from the
   index and decodes only the matching rows, rather than scanning the
   whole column. The same index answers unranked retrieval directly —
   boolean token matching and exact raw-value matching — so an
   equality or `IN` predicate on an indexed column resolves to a small
   candidate row set before any column data is read.
4. **Merges** the per-file results into one ranked answer.

The cost model that falls out of this is the headline: **you pay for
what you query, not for keeping everything in memory.**

## Caching: cold, warm, hot

Object storage is cheap and durable but relatively slow per request, so
infino layers a cache between compute and storage:

- **Cold** — data only in object storage. The first query is served
from ranged reads while the full file is fetched into the local cache
in the background.
- **Warm/Hot** — once a file is resident locally it's served from a
memory-mapped file, at local-disk/RAM speed.
- **Bounded & elastic** — the cache has a size budget and evicts cold
files when full. A separate memory budget can push pages out of RAM
while leaving the file on local disk, so it re-faults without a
re-download. Eviction is always safe: in-flight queries keep their
data alive.

The point: the **hot working set** lives close to compute for speed,
while the **long tail** lives in cheap storage — without you having to
decide up front which is which.

## How this differs from what's out there

Databases, search engines, and vector databases increasingly offer multiple
modalities — scalar, full-text, and vector — and several have moved at
least partway toward object storage. So the interesting question is
rarely "can system X do vectors / keyword / SQL?" (usually: yes, in some
form). It's about each system's **design center** — what it was built
around from day one — and the **tradeoffs in cost and complexity
for a given use case** that follow at scale. The categories below are
points on a spectrum, not walls.

### Traditional databases (Postgres, MySQL, …)

- Built for transactional workloads: row-oriented, tuned for
point reads/writes of individual records with strong consistency.
- The ecosystem has added real search capability — `pgvector` for
embeddings, built-in full-text — and for many teams that's the right
answer: one system you already run, no extra moving parts, perfectly
good at moderate scale.
- The tradeoff is architectural: storage and compute are **coupled**,
so as the dataset (and especially the vector count) grows, you scale
by buying a bigger box or sharding it yourself, and the cost curve
rises with total data rather than with what you query.
- **Sweet spot**: transactions and mixed small-record read/write, plus
search that piggybacks on data you already keep there. **Less ideal**:
very large search corpora where most data is cold.

### Search engines (Elasticsearch / OpenSearch / Lucene-based)

- The mature standard for full-text relevance, with a deep feature set
(analyzers, aggregations, highlighting) and increasingly capable
dense-vector support.
- They've also evolved toward tiered storage — **searchable snapshots /
frozen tiers** can back colder data with object storage — so the
"everything on local disk" picture is no longer the whole story.
- That said, the **design center is still the node-and-shard cluster**:
object-storage tiers are an addition to that model rather than the
default, and operational weight (sizing, shard rebalancing, JVM/heap
tuning, tier management) remains real. Keeping large indexes hot is
powerful but costly.
- **Sweet spot**: rich, mature full-text relevance and analytics on
data you actively query. **Cost/ops consideration**: scale and
always-hot footprint.

### Vector databases (Pinecone, Weaviate, Milvus, Qdrant, …)

- Purpose-built, often best-in-class approximate-nearest-neighbor over
embeddings, with strong recall/latency tuning.
- The category has matured well past "vectors only": most now offer
**metadata filtering and hybrid (keyword + vector) search**, and some
are themselves moving to object-storage-backed, separated-compute
architectures.
- The common tradeoff is **system count and role**: a vector DB is
frequently run *alongside* your system of record and your search
engine, so you keep multiple copies of data in sync; and full-text /
SQL maturity varies by vendor. Latency-first deployments that pin
vectors in RAM/SSD can get expensive as the corpus grows.
- **Sweet spot**: vector-first / AI-retrieval workloads. **Consideration**:
where it sits relative to your other systems, and breadth beyond
vectors.

### Object-storage-native search (Infino, TurboPuffer, LanceDB)

This camp's distinction isn't "the only ones who *can*" do these things
— it's that the architecture is **built around object storage and
multi-modal retrieval from the start**, rather than retrofitting either:

- **Object storage as the primary tier by default.** The full dataset
lives in S3 at S3 prices; compute is stateless and cached. Cold data
is cheap to *keep*, and you largely pay compute for what you *query*.
- **Separation of storage and compute** as the baseline — scale and
bill each independently; spin compute down without losing data.
- **Multi-modal as a first-class assumption**, not a later addition.

Where **Infino is distinctive even within this camp**:

- **The superfile is a valid Parquet file.** Data isn't trapped in a
proprietary index format — the *same bytes* are readable by the open
analytics ecosystem (DuckDB, DataFusion, pyarrow) with no export step,
while infino uses embedded index regions for search. Lower lock-in,
easy interop with existing data tooling.
- **Scalar + full-text + vector together in one immutable superfile** —
SQL, BM25, and IVF + RaBitQ vectors share one copy of the data and
one consistency model, instead of syncing a DB + a search engine + a
vector DB.
- **Hybrid search is a first-class citizen — and an access path, not just an API.** In most systems, hybrid search ends at top-k: two retrievals (BM25, ANN) plus rank fusion. Here every modality shares one copy and one query path, so you fuse keyword and vector relevance — with SQL filters — in a single query against a single snapshot, with no second system to keep in sync and no client-side result stitching. And because the retrievers are *relations* (table functions) and indexed text predicates resolve to candidate row sets inside the engine, search can be the first stage of a larger SQL plan — feeding joins, filters, and aggregates — rather than its result. The same index machinery prunes superfiles across SQL, full-text, and vector together, so the hybrid query is also the well-pruned, cheap one.

### At a glance

Read these as **design center and typical tradeoff**, not hard limits —
most systems are extending across the row over time.


|                          | Built around                             | Modalities (today)                                           | Cold-data cost curve                         | Format                                     |
| ------------------------ | ---------------------------------------- | ------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------ |
| **Traditional DB**       | Transactions, single node                | Scalar core; full-text + vectors added (e.g. pgvector)       | Rises with total data (coupled)              | Some prorietary, many support **Parquet**. |
| **Search engine**        | Node/shard cluster                       | Full-text core; vectors maturing; object-storage tiers added | Lower with frozen tiers, but cluster-centric | Proprietary                                |
| **Vector DB**            | ANN over embeddings                      | Vector core; hybrid + filtering increasingly common          | Varies; RAM/SSD-heavy if latency-pinned      | Proprietary                                |
| **Object Store Engines** | Object storage + multi-modal, by default | Scalar + full-text + vector + hybrid as a baseline           | Low; pay for what you query                  | Infino: **Parquet**                        |


## What Infino optimizes for

- **Cost at scale.** Object storage as the source of truth means
storing a lot of data is cheap; you pay compute only for the queries
you run and the hot set you cache.
- **One system, multiple query types — including hybrid.** SQL filters,
keyword relevance, and semantic similarity over the same data, fused
into a single hybrid query when you want both signals at once — no
multi-system sync, no duplicated copies, no client-side result merging.
- **Predictable performance tiers.** Bounded ranged reads on cold data,
local memory-mapped speed when hot, with the cache managing the
transition automatically.
- **Operational simplicity.** Immutable files + an atomic manifest swap
give clean snapshots, safe concurrent writers, and stateless,
disposable compute.
- **Openness / no lock-in.** Superfiles are Parquet; data stays usable by
the broader ecosystem.

## Where it fits best

Infino is best for:

- Large corpora where **most data is cold** but must stay searchable —
logs, documents, product catalogs, knowledge bases, chat/email
history — and where keeping it all hot is the cost pain.
- **RAG and AI retrieval** that needs **hybrid** relevance — semantic
(vector) fused with keyword and metadata/SQL filtering over the same
store — where consolidating onto one multi-modal system beats running
and syncing another.
- Teams feeling the **always-hot cost or operational weight** of their
current setup (e.g. a large search cluster, or a separate vector DB
kept in sync with a database) and open to a separated-compute,
object-storage-native model.
- Workloads with **bursty or elastic query volume**, where decoupled
compute can scale up and down against a stable storage tier.
- **Existing warehouse / lakehouse data** you want to search in place.
Because a superfile *is* a valid Parquet file, you add keyword, vector,
and hybrid retrieval over data already in your lakehouse's open format —
cutting the number of tools you operate and the cost of keeping
duplicate copies and separate systems in sync.

Equally fair to say where it's *not* the obvious choice: heavy transactional workloads belong in an OLTP database, and if you already
run one system that comfortably handles your scale and modalities, then where Infino helps is to consolidate to save cost/ops.

## A few terms you'll see through the repo

- **Connection** — the entry point: a catalog of tables rooted at a URI,
opened with `connect(uri)`. Create / open / list / drop tables and run
SQL across them. (The table catalog; not to be confused with a single
table's manifest.)
- **Superfile** — one immutable superfile file (columns + keyword index +
vector index), also a valid Parquet file.
- **Supertable** — the table: a manifest over many superfiles,
presenting them as one queryable table. Obtained from a `Connection`.
- **Manifest** — the immutable list of which files make up one table
right now; each commit publishes a new one atomically.
- **Snapshot read** — a query pinned to one manifest, so concurrent
writes never change its answer.
- **Pruning** — skipping files that can't match using small per-file
summaries, before touching any file contents.
- **BM25** — the standard keyword-relevance ranking for full-text
search.
- **Hybrid search** — a single query that fuses keyword (BM25) and
vector relevance (e.g. via reciprocal-rank fusion), optionally with
SQL filters, over one copy of the data and one snapshot.
- **IVF + RaBitQ** — the clustering + compact-binary-code technique
behind fast approximate vector search.
- **Object storage / S3** — cheap, durable, near-infinite remote
storage used as the source of truth.

