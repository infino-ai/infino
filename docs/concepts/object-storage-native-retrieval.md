# Object-storage-native retrieval

**Object-storage-native retrieval is search that runs directly on data kept in
object storage — Amazon S3, Azure Blob, or local disk — instead of in a
database or search cluster that owns its own copy of the data.** The index and
the data live as ordinary files on the object store; queries read just the
bytes they need, on demand. There is no always-on storage tier to provision,
replicate, or keep in sync.

## Why it matters

Classic search and vector engines couple compute and storage: you load your
data into the engine's own nodes, the engine owns that copy, and you pay for
those nodes whether or not anyone is querying. Object-storage-native retrieval
breaks that coupling:

- **Storage is cheap and elastic.** Data sits in object storage at object-
  storage prices, with no replication factor to multiply your footprint.
- **Compute is stateless.** Any process can open the data and serve a query;
  there is no cluster to keep warm between queries.
- **One copy, open format.** Because the files are a standard format, the same
  bytes that serve search also serve analytics — no second system to sync.

This is decisive for **agent and RAG workloads**, where an agent issues
hundreds of retrievals per task. When each retrieval is cheap and the storage
bill is flat, latency and cost compound in your favour rather than against you.

## How retrieval works on object storage

The challenge object-storage-native retrieval solves is that object storage has
high *first-byte latency* — you cannot treat it like a local disk. Engines that
do this well share three moves:

1. **Self-describing files.** Each file carries its own index, so a reader can
   locate the relevant regions from the footer without a separate metadata
   service.
2. **Skip-pruning before fetch.** Lightweight summaries (value ranges, term
   filters, vector centroids) let a query rule out whole files *before*
   fetching any of their bytes, so a selective query touches a small fraction
   of the data.
3. **Range reads with caching.** Queries fetch only the byte ranges they need
   and cache them, so a cold first touch pays the object-store round trip once
   and warm queries run from local memory.

## How Infino implements it

Infino is an object-storage-native retrieval engine built around two layers:

- **The superfile** — a single file that is *also a valid Apache Parquet file*:
  Parquet data followed by a standard Parquet footer, with BM25 and vector
  index regions spliced in alongside. Anything that reads Parquet (DataFusion,
  DuckDB, pyarrow) can read a superfile's columns directly; Infino uses the
  same footer to find the indexes when a search runs.
- **The supertable** — many superfiles composed into one table with an
  atomic-commit manifest, snapshot-isolated reads, and append-only writes. The
  manifest holds the term filters, value ranges, and vector centroids that let
  a query prune superfiles it can't match before any bytes are fetched.

Search, vector, and SQL all run over that one copy of the data, in-process —
**no server, no daemon, no managed service.** On a 1-million-document index, a
warm single-term BM25 query returns in the **microsecond range**, and a 1M-doc
index builds in a couple of seconds with parallel writers (roughly **470K
documents per second**); see [`benches/README.md`](../../benches/README.md) for
the full, current figures and the hardware they were recorded on.

```python
import infino

# "memory://" is in-process; swap in "s3://bucket/prefix" to run the same
# queries directly against data on object storage — no server to stand up.
db = infino.connect("s3://my-bucket/corpus")
hits = db.query_sql("""
    SELECT body FROM bm25_search('docs', 'body', 'cancel subscription', 10)
""")
```

## When it fits — and when it doesn't

Object-storage-native retrieval is the right model when your data already lives
in (or can live in) object storage and you want search and retrieval over it
without standing up and paying for a separate storage tier — agent memory, RAG
over a document corpus, and search over data lakes are the sweet spot. It is
not built to be a transactional database for high-rate row-level updates. See
[Tradeoffs and limits](../tradeoffs.md) for the honest envelope.

## See also

- [Architecture overview](../architecture/overview.md) — the plain-language tour.
- [Superfile format](../architecture/superfile.md) — the single-file format and
  its Parquet compatibility.
- [Supertable layer](../architecture/supertable.md) — the table layer, manifest,
  and skip-pruning.
