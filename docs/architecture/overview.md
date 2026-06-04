# infino Architecture Overview (GTM primer)

A plain-language tour of what infino is, how it's built, and how it
differs from the databases, search engines, and vector databases it
competes with. For the technical deep-dives, see
[superfile](./superfile.md) (the segment format) and
[supertable](./supertable.md) (the table layer). This document is the
"why it matters" layer on top of those.

## The one-liner

**infino is a search-and-retrieval database that keeps your data on
cheap object storage (like Amazon S3) and runs SQL, full-text (keyword)
search, and vector (semantic) search over it from a single system.**

You don't provision a big, always-on cluster sized for your *whole*
dataset. The data lives in object storage at object-storage prices, and
compute pulls in only what a query actually touches, caching the hot
parts locally for speed.

## The mental model

Think of three ideas:

1. **The data is just files.** Every chunk of the table is one
   self-contained file — a *superfile* — that holds the raw columns
   *and* the search indexes (keyword + vector) together. There are no
   sidecar index files to keep in sync, and the file is also a valid
   Apache Parquet file, so standard analytics tools (DuckDB,
   DataFusion, pandas/pyarrow) can read it directly.

2. **Files are never edited, only added.** A *supertable* is a catalog
   (the *manifest*) that lists which files currently make up the table.
   Writes create new files and publish a new manifest; they never
   modify existing files. This makes consistency simple and snapshots
   essentially free.

3. **Storage and compute are separate.** The source of truth is object
   storage. Compute nodes are stateless readers that fetch byte ranges
   on demand and keep a local cache of hot data. You scale storage and
   compute independently and pay for each separately.

```text
        Application
            │  SQL · keyword search · vector search
            ▼
        Supertable  ── manifest (the catalog of files) ──┐
            │                                             │
   stateless compute                              immutable files
   + local cache (hot)                            (superfiles)
            │                                             │
            └──────────── byte-range reads ───────────────┘
                                 ▼
                      Object storage (S3) — cheap, durable,
                              the source of truth
```

## How a query runs (and why it's cheap)

A query never downloads the whole dataset. It:

1. Starts from a **pinned snapshot** of the manifest (so concurrent
   writes never change the answer mid-query).
2. **Prunes** — uses small per-file summaries in the manifest
   (value ranges, a keyword "is this term present?" filter, vector
   centroids) to skip files that can't possibly match. This reads only
   the catalog, never file contents.
3. **Fetches only what it needs** — for surviving files it pulls just
   the relevant byte ranges from object storage (a posting list, a
   handful of vector clusters), not the whole file.
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

First, an honest caveat: this is a **converging** market. Databases,
search engines, and vector databases increasingly offer *all three*
modalities — scalar, full-text, and vector — and several have moved at
least partway toward object storage. So the interesting question is
rarely "can system X do vectors / keyword / SQL?" (usually: yes, in some
form). It's about each system's **design center** — what it was built
around from day one — and the **tradeoffs in maturity, cost, and
operational shape** that follow at scale. The categories below are
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

### Object-storage-native search (infino, and turbopuffer / "tpuf")

This camp's distinction isn't "the only ones who *can*" do these things
— it's that the architecture is **built around object storage and
multi-modal retrieval from the start**, rather than retrofitting either:

- **Object storage as the primary tier by default.** The full dataset
  lives in S3 at S3 prices; compute is stateless and cached. Cold data
  is cheap to *keep*, and you largely pay compute for what you *query*.
- **Separation of storage and compute** as the baseline — scale and
  bill each independently; spin compute down without losing data.
- **Multi-modal as a first-class assumption**, not a later addition.

Where **infino is distinctive even within this camp**:

- **The segment is a valid Parquet file.** Data isn't trapped in a
  proprietary index format — the *same bytes* are readable by the open
  analytics ecosystem (DuckDB, DataFusion, pyarrow) with no export step,
  while infino uses embedded index regions for search. Lower lock-in,
  easy interop with existing data tooling.
- **Scalar + full-text + vector together in one immutable segment** —
  SQL, BM25, and IVF + RaBitQ vectors share one copy of the data and
  one consistency model, instead of syncing a DB + a search engine + a
  vector DB.

These are differences of degree and default, not absolutes — useful
framing in a conversation with a buyer who already runs one of the
above.

### At a glance

Read these as **design center and typical tradeoff**, not hard limits —
most systems are extending across the row over time.

| | Built around | Modalities (today) | Cold-data cost curve | Format |
|---|---|---|---|---|
| **Traditional DB** | Transactions, single node | Scalar core; full-text + vectors added (e.g. pgvector) | Rises with total data (coupled) | Proprietary |
| **Search engine** | Node/shard cluster | Full-text core; vectors maturing; object-storage tiers added | Lower with frozen tiers, but cluster-centric | Proprietary |
| **Vector DB** | ANN over embeddings | Vector core; hybrid + filtering increasingly common | Varies; RAM/SSD-heavy if latency-pinned | Proprietary |
| **infino / tpuf** | Object storage + multi-modal, by default | Scalar + full-text + vector as a baseline | Low; pay for what you query | infino: **open Parquet** |

## What infino optimizes for

- **Cost at scale.** Object storage as the source of truth means
  storing a lot of data is cheap; you pay compute only for the queries
  you run and the hot set you cache.
- **One system, multiple query types.** SQL filters, keyword relevance,
  and semantic similarity over the same data — no multi-system sync,
  no duplicated copies.
- **Predictable performance tiers.** Bounded ranged reads on cold data,
  local memory-mapped speed when hot, with the cache managing the
  transition automatically.
- **Operational simplicity.** Immutable files + an atomic catalog swap
  give clean snapshots, safe concurrent writers, and stateless,
  disposable compute.
- **Openness / no lock-in.** Segments are Parquet; data stays usable by
  the broader ecosystem.

## Where it fits best (talking points for ICP)

Framed as *fit*, not as "everyone else is wrong" — the strongest pitch
meets a buyer where their current stack strains:

- Large corpora where **most data is cold** but must stay searchable —
  logs, documents, product catalogs, knowledge bases, chat/email
  history — and where keeping it all hot is the cost pain.
- **RAG and AI retrieval** that needs *both* semantic (vector) and
  keyword/metadata filtering over the same store, where consolidating
  onto one multi-modal system is more attractive than adding another.
- Teams feeling the **always-hot cost or operational weight** of their
  current setup (e.g. a large search cluster, or a separate vector DB
  kept in sync with a database) and open to a separated-compute,
  object-storage-native model.
- Workloads with **bursty or elastic query volume**, where decoupled
  compute can scale up and down against a stable storage tier.

Equally fair to say where it's *not* the obvious choice: heavy
transactional workloads belong in an OLTP database, and if you already
run one system that comfortably handles your scale and modalities,
"consolidate to save cost/ops" is the conversation — not "rip and
replace."

## A few terms you'll hear

- **Superfile** — one immutable segment file (columns + keyword index +
  vector index), also a valid Parquet file.
- **Supertable** — the table: a manifest (catalog) over many
  superfiles, presenting them as one queryable table.
- **Manifest** — the immutable list of which files make up the table
  right now; each commit publishes a new one atomically.
- **Snapshot read** — a query pinned to one manifest, so concurrent
  writes never change its answer.
- **Pruning** — skipping files that can't match using small per-file
  summaries, before touching any file contents.
- **BM25** — the standard keyword-relevance ranking for full-text
  search.
- **IVF + RaBitQ** — the clustering + compact-binary-code technique
  behind fast approximate vector search.
- **Object storage / S3** — cheap, durable, near-infinite remote
  storage used as the source of truth.

---

*For implementation detail, hand the candidate
[superfile.md](./superfile.md) and [supertable.md](./supertable.md);
this overview is the positioning layer above them.*
