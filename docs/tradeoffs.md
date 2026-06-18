# Tradeoffs and limits

**Infino optimizes for query speed-per-dollar on object storage. That choice
buys fast, cheap retrieval over data you already keep in object storage — and
it costs some things a tightly-coupled, in-cluster engine gives you.** This
page is the honest envelope: where Infino is the right tool, and where it
isn't.

## What Infino is good at

- **Fast warm queries.** On a 1-million-document index, a warm single-term BM25
  query returns in the **microsecond range**. Once a query's byte ranges are
  cached, search runs from local memory.
- **Multi-modal retrieval over one copy.** BM25 full-text, vector search, and
  SQL run over the same rows, and the search functions compose into SQL plans —
  no second system to sync, no client-side result stitching.
- **Flat storage economics.** Data lives in object storage at object-storage
  prices, as standard Parquet, with no replication factor multiplying the
  footprint and no always-on storage tier to pay for between queries.
- **No operational surface.** It runs in-process — no server, no daemon, no
  cluster to provision or keep warm.

## What Infino is not (yet) the right tool for

- **Cold first-touch latency.** Object storage has high first-byte latency, so
  the *first* query to touch a file pays an object-store round trip (hundreds
  of milliseconds) before its bytes are cached. Infino prunes aggressively to
  minimise what it fetches, but if your workload is dominated by one-shot
  queries against never-before-touched data, that round trip is real. Warm and
  repeated queries do not pay it.
- **Bulk-ingest throughput is solid, not the headline.** Building indexes and
  committing to object storage is parallel and respectable (a 1M-document index
  builds in a couple of seconds in memory, ~470K docs/s with parallel writers),
  but **ingest is not where Infino's biggest advantage is — query latency is.**
  If your workload is write-dominated rather than read-dominated, weigh that.
- **Not a transactional database.** Writes are append-only with an atomic
  commit as the durability boundary; updates and deletes go through tombstones,
  not in-place row mutation. Infino is built for retrieval over largely
  append-only corpora, not high-rate row-level OLTP.
- **No wire protocol yet.** Infino is embedded — you reach it as a library from
  your own process. There is no server endpoint for external SQL clients to
  attach to.

The current, full performance picture — warm vs. cold, ingest throughput, index
size, and the hardware it was recorded on — lives in
[`benches/README.md`](../benches/README.md).

## See also

- [Object-storage-native retrieval](concepts/object-storage-native-retrieval.md)
  — the model and why it makes these tradeoffs.
- [Architecture overview](architecture/overview.md) — where Infino fits best.
