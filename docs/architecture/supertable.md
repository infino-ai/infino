# Supertable architecture

The supertable is infino's **in-memory cross-superfile query + manifest
layer over [`superfile`](superfile.md)**. A supertable joins many
immutable superfile superfiles into one logical table, queryable as a
single source via [SQL](https://datafusion.apache.org/),
[BM25](https://en.wikipedia.org/wiki/Okapi_BM25), and IVF-based
vector kNN — without copying any superfile bytes between writers and
readers, and without serializing readers against in-flight commits.

A supertable is to a superfile what an
[Iceberg](https://iceberg.apache.org/) /
[Delta](https://delta.io/) table is to Parquet: a small in-memory
manifest on top of an append-only set of immutable byte
superfiles.

This document is the long-form reference for the manifest data
model, the cross-superfile query algorithms, the concurrency model,
and the rationale for every non-obvious design decision. The
companion superfile architecture doc covers the superfile-internal
format (Parquet body + embedded BM25 + vector blobs).

## Table of contents

- [Goals and non-goals](#goals-and-non-goals)
- [Data model](#data-model)
  - [Manifest](#manifest)
  - [SuperfileEntry summaries](#superfileentry-summaries)
  - [SuperfileStore](#superfilestore)
- [API surface](#api-surface)
  - [Build path](#build-path)
  - [Read path](#read-path)
  - [Query methods](#query-methods)
- [Subsystem design](#subsystem-design)
  - [Lock-free reader-writer isolation via ArcSwap](#lock-free-reader-writer-isolation-via-arcswap)
  - [Copy-on-write manifest](#copy-on-write-manifest)
  - [Writer pipeline: rayon-shard commit](#writer-pipeline-rayon-shard-commit)
  - [Query fan-out](#query-fan-out)
    - [Intra-segment sub-range fan-out](#intra-segment-sub-range-fan-out)
  - [Skip pruning](#skip-pruning)
  - [Dual-pool concurrency](#dual-pool-concurrency)
  - [SQL surface via DataFusion](#sql-surface-via-datafusion)
- [Performance](#performance)
- [Test strategy](#test-strategy)
- [Known limitations](#known-limitations)
- [Phase 2 plans](#phase-2-plans)

## Goals and non-goals

**Goals.**

- **Cross-superfile query as a first-class surface.** BM25, prefix
  BM25, vector kNN, and DataFusion SQL all run against the union
  of every superfile in a single API call. Per-superfile work fans
  out across the reader pool; results merge globally.
- **Lock-free reader isolation.** A reader pinned at time `t`
  observes the manifest as it existed at `t` for the lifetime of
  that reader, regardless of how many writer commits land
  afterwards. No mutex on the read path.
- **Manifest-only skip pruning.** Per-superfile FTS bloom + lex
  term range + vector centroid summaries live in the manifest.
  Pruned superfiles never trigger a `SuperfileStore::reader` call —
  the load-bearing perf claim of the skip layer.
- **Decoupled bytes and metadata.** The manifest carries summary
  stats only (a few KB per superfile); superfile bytes live in a
  pluggable `SuperfileStore` shared via `Arc<dyn SuperfileStore>`.
  Same shape extends to mmap / S3 / GCS in 003.
- **Latency targets.** BM25 search p50 in the microseconds for
  single-term, low milliseconds for multi-term OR; vector kNN
  p50 in the low tens of milliseconds at 10M scale with recall@10
  ≥ 0.90. See [Performance](#performance).
- **Bounded memory.** Append buffer is capped via
  `commit_threshold_size_mb`; the manifest itself is one
  `Arc<Manifest>` per pinned reader, copy-on-write on commit.

**Non-goals (deliberately deferred).**

- **On-disk persistence.** The supertable is in-memory only.
  Manifest serde, atomic-rename pointer files, and crash-safety
  semantics live in 003 (object-store + LocalFsProvider).
- **Compaction.** Merging small superfiles into larger ones lives
  in 004.
- **Updates and deletes.** Tombstone bitmaps + insert-with-same-
  logical-id live in 005.
- **Distributed multi-node operation.** Cross-process / cross-
  node coordination lives in 006.
- **Multi-writer concurrency.** A single writer slot per
  supertable is enforced via `compare_exchange` on an
  `AtomicBool`; concurrent writers get
  `BuildError::SupertableInUse`. Cross-process OCC on the
  pointer file lives in 003.
- **Buffer-reader path.** Querying the in-memory writer buffer
  before `commit()` is deliberately not shipping; same rationale
  as superfile (see superfile.md § ingestion durability).
- **Hybrid (FTS + vector) single-API search.** v1 ships
  distinct method-style `bm25_search` / `vector_search` on the
  reader; combined-relevance scoring is a future addition.
- **DataFusion UDFs / TVFs.** SQL is a thin
  `MemTable`-backed `query_sql(sql)` in v1; UDF surface lives in
  007.

## Data model

### Manifest

`Manifest` is the single point-in-time snapshot of which superfiles
exist. It is **immutable** by construction: each commit
constructs a new `Manifest` and atomically swaps it in via
[`ArcSwap<Manifest>`](#lock-free-reader-writer-isolation-via-arcswap).

```text
Manifest {
    manifest_id: u64,                   // monotonically increasing
    options:     Arc<SupertableOptions>,
    superfiles:    Vec<Arc<SuperfileEntry>>,
}
```

`superfiles` is `Vec<Arc<SuperfileEntry>>` — not `Vec<SuperfileEntry>` —
so a successor manifest can re-use every existing entry by
`Arc::clone` without copying the underlying summary stats. Per
commit, allocation is one new `Vec` plus the new entries; old
entries are pointer-shared.

### SuperfileEntry summaries

```text
SuperfileEntry {
    superfile_id:     Uuid,
    uri:            SuperfileUri,            // hash-eq token used by SuperfileStore
    n_docs:         u64,
    id_min, id_max: u64,                   // inclusive range of the user id column
    scalar_stats:   ScalarStatsTable,      // per-column min/max for skip pruning
    fts_summary:    HashMap<String, FtsSummary>,
    vector_summary: HashMap<String, VectorSummary>,
}

FtsSummary {
    term_bloom:        Bloom,             // 64 KiB block bloom, ~7% FPR @ 100K terms
    n_terms_distinct:  u32,
    term_range:        (Vec<u8>, Vec<u8>), // lex (min, max) for prefix-overlap skip
}

VectorSummary {
    centroid: Vec<f32>,                   // per-column cluster centroid
    radius:   f32,                        // max distance from centroid to any vector
}
```

Every summary is derivable for free at commit time:

- **Term bloom + range** come from one walk of the FST's term
  iterator — first key = `min_term`, last key = `max_term`,
  every key gets inserted into the bloom.
- **Vector centroid + radius** are already produced by the
  superfile vector builder's IVF clustering pass; the writer
  copies them out via `VectorReader::summary`.
- **Scalar min/max** come from one Arrow-aggregate pass over
  each scalar column's values — done once per shard while the
  buffered `RecordBatch`es are still live, before they're
  consumed by the underlying `SuperfileBuilder` (post-store
  readers can't recover Arrow batch min/max without re-decoding
  Parquet).

### SuperfileStore

```text
trait SuperfileStore: Send + Sync {
    fn reader(&self, uri: &SuperfileUri) -> Result<Arc<SuperfileReader>, StoreError>;
    fn put(&self, uri: SuperfileUri, bytes: Bytes) -> Result<(), StoreError>;
    fn resident_bytes(&self) -> usize;
}
```

`InMemorySuperfileStore` is a `RwLock<HashMap<SuperfileUri, Entry>>` —
the read path takes a *read* lock so a fan-out (rayon `par_iter`
across N non-pruned superfiles) resolves all N URIs in parallel
without readers serializing on each other. `put()` takes a write
lock; the parse runs *outside* both locks (optimistic-read +
parse-outside-locks + recheck-on-write) so concurrent puts of
distinct URIs don't serialize on the parse.

The trait is the seam where a future object-store-backed
implementation plugs in mmap / S3 / GCS without touching any
caller. The byte-store is decoupled from the manifest
specifically for this reason — superfile summaries (a few KB
each) live in every snapshot of the manifest, but the bytes
they describe live in one shared store across snapshots.

## API surface

### Build path

```rust
let opts = SupertableOptions::new(
    schema,                                       // user schema; must NOT contain `_id`
    vec![FtsConfig { column: "title".into() }],
    vec![VectorConfig { column: "emb".into(), dim: 384, n_cent: 64, .. }],
    Some(Arc::new(AsciiLowerTokenizer)),
)?;
let st = Supertable::create(opts);
let mut w = st.writer()?;          // single-writer slot
w.append(&record_batch)?;          // buffer accumulates BufferedBatches
w.commit()?;                       // rayon-shard build + ArcSwap publish
// drop(w) — releases the writer slot
```

`Supertable::writer()` returns `Err(SupertableInUse)` if a writer
is already outstanding. The slot is enforced via `compare_exchange`
on `SupertableInner.writer_outstanding: AtomicBool` and released
on `Drop`.

### Auto-injected `_id` column

The supertable manages its own primary-key column. The user
schema must NOT contain a field named `_id` (or whatever
`config.supertable.id_column` is set to — `_id` is the
default); construction errors with
`BuildError::IdColumnReserved` if it does. At every
`append()`, the writer mints one id per row and prepends a
`Decimal128(38, 0)` column to the buffered scalar batch.

Id format (128 bits, big-endian):

```text
127                              64 63              24 23      0
┌────────────────────────────────┬─────────────────┬─────────┐
│     64-bit ms timestamp        │   40 worker     │ 24 ctr  │
└────────────────────────────────┴─────────────────┴─────────┘
```

- **64-bit ms timestamp** since Unix epoch. The high bit
  stays 0 for any plausible lifetime (year ~292M), so
  signed `i128` sort matches time order — `WHERE _id
  BETWEEN ? AND ?` skip-prunes by recency without any
  custom comparator.
- **40-bit worker_id** self-assigned at process startup
  via `rand::random::<u64>() & ((1 << 40) - 1)`. Fresh
  generator per `Supertable::create()` / `::open()`.
- **24-bit sequence**, strictly monotonic per (worker_id,
  ms). 16M ids/ms/worker before stalling.

**No coordination across writer processes.** Every process
that opens or creates a `Supertable` instantiates its own
`IdGenerator` with an independent 40-bit random worker_id.
Birthday-collision probability:

| Workers (N) | P(collision) | Frame |
|---|---:|---|
| 47 | 10⁻⁹ | "won't happen in cluster lifetime" |
| 1,480 | 10⁻⁶ | typical primary-key safety margin |
| 47,000 | 10⁻³ | starts to matter |
| 148,000 | 1% | clearly non-negligible |
| 1,234,000 | 50% | birthday point |

For comparison, the 10-bit worker_id in Twitter Snowflake
hits the 1% threshold at just 5 workers — which is why
production deployments of that scheme depend on centrally
assigned worker_ids. The 40-bit width here trades 24 bits
of headroom against the need for coordination, and lets a
fleet of ~150k concurrent writers per supertable run
without any coordination infrastructure.

Storage: the column is `Decimal128(38, 0)` (Arrow's signed
128-bit integer type), which Parquet stores as
`FIXED_LEN_BYTE_ARRAY(16)` with a `DECIMAL(38, 0)` logical
annotation. External Parquet readers see a 128-bit decimal
column and sort it correctly. Compression with zstd lands
at ~1 B/row in practice — the timestamp prefix and the
constant-per-process worker_id field both dictionary-encode
near-perfectly.

The internal posting-list + vector-index doc-ids stay at
`u32` (local within a superfile); the `_id` column lives only
in the Parquet body and is read via the query layer's
result projection, never inside the FTS / vector hot
loops.

### Read path

```rust
let r: SupertableReader = st.reader();   // ArcSwap::load_full at t=now
r.manifest_id();                          // pinned, never moves under r
r.n_superfiles();
r.n_docs_total();
r.manifest();                             // Arc<Manifest> for query machinery
```

`Supertable::reader()` does one `ArcSwap::load_full` and pins the
resulting `Arc<Manifest>` for the reader's lifetime. New commits
ArcSwap a successor manifest into place; the pinned reader's Arc
is unaffected.

### Query methods

```rust
// SQL (DataFusion)
let batches: Vec<RecordBatch> = st.query_sql(
    "SELECT category, COUNT(*) FROM supertable GROUP BY category",
)?;

// BM25 (methods on the reader)
let hits: Vec<SuperfileHit> = r.bm25_search("title", "rust async", 10, BoolMode::Or)?;
let hits: Vec<SuperfileHit> = r.bm25_search_prefix("title", "rust", 10)?;

// Vector kNN
let opts = VectorSearchOptions::new();
let hits: Vec<SuperfileHit> = r.vector_search("emb", &query, 10, opts)?;
```

`SuperfileHit` is `(SuperfileUri, local_doc_id, score)`. Doc-id
space is local to a superfile in v1; resolving to a global
identity goes through the caller's primary-key column.

The SQL surface is **sync** (the only async dep is DataFusion
itself — we `block_on` against a per-`Supertable`
`OnceLock<Runtime>` so callers don't need a tokio runtime).

## Subsystem design

### Lock-free reader-writer isolation via ArcSwap

The load-bearing concurrency primitive is
[`ArcSwap<Manifest>`](https://crates.io/crates/arc-swap):

- **Read path** (`Supertable::reader`): one `ArcSwap::load_full`.
  No mutex. The reader owns its `Arc<Manifest>` for the rest of
  its lifetime; it sees exactly the manifest published at the
  pin instant.
- **Write path** (`SupertableWriter::commit`): build the new
  superfiles, build the successor `Manifest`, atomically swap in
  via `ArcSwap::store`. Readers pinned before the swap continue
  to see the old manifest; readers obtained after the swap see
  the new one.

```text
   t0  ─────────────────────────────────────────────────── time ───►
                       commit(C1)              commit(C2)
                            │                       │
       Manifest(M0) ── M0 ──┴── Manifest(M1) ── M1 ─┴── Manifest(M2)

   Reader R0   pin@t0  →  sees M0 for entire lifetime
   Reader R1                  pin@t1  →  sees M1
   Reader R2                                     pin@t2  →  sees M2

   Old SuperfileEntry Arcs (a, b, c, d) are pointer-shared across
   M0 → M1 → M2 — successors clone only the outer Vec; the heavy
   per-segment summaries are never copied.
```

Why ArcSwap and not `RwLock<Arc<Manifest>>`?
- `RwLock` writers can starve under sustained read traffic
  (many implementations are reader-preferring); ArcSwap's swap
  is wait-free and has constant cost regardless of reader count.
- `RwLock` reads still require a lock acquire; ArcSwap reads are
  one atomic load + one Arc clone. At fan-out scale (N superfiles
  × N readers) the difference compounds.
- Manifest immutability + Arc-shared `SuperfileEntry`s make
  copy-on-write trivial: the cost of a fresh manifest is one
  `Vec` allocation + the new entries' Arcs.

### Copy-on-write manifest

`Manifest::with_appended(new_entries)` constructs a successor by
cloning the outer `Vec` and moving every existing `Arc<SuperfileEntry>`
into the new vec via `Arc::clone`. The new manifest is structurally
independent of its predecessor — mutating one doesn't affect the
other — but every shared `SuperfileEntry` is allocated exactly once.

This is what makes thousands of concurrent readers cheap: each
reader holds an `Arc<Manifest>` (one `Vec` of `Arc<SuperfileEntry>`s),
and the per-reader cost is `1 + n_superfiles` ref-count bumps. No
allocation per reader after the first.

### Writer pipeline: rayon-shard commit

`SupertableWriter` accumulates `BufferedBatch { scalar, vectors }`
across `append()` calls. On `commit()`, every row in the buffer is
re-partitioned into **`N = min(writer_pool.threads, total_rows)`**
row-balanced shards via
[`split_buffer_into_row_shards`](../../src/supertable/writer.rs)
— so a single 10M-row `append()` followed by `commit()` on an
8-thread pool produces 8 shards of ~1.25M rows each, the **same**
as 8 separate `append()` calls of 1.25M rows each. This decouples
ingest parallelism from the caller's batching pattern.

The row-balanced split walks rows across the original buffer in
order and emits **zero-copy Arrow slices** (`RecordBatch::slice` +
`Float32Array::slice`, both of which adjust buffer offsets without
copying underlying memory) — so a shard boundary that falls in the
middle of a `BufferedBatch` doesn't copy bytes. With `total_rows =
q·N + r`, the first `r` shards get `q+1` rows and the rest get
`q`; row imbalance is ≤ 1.

```text
commit():
   buffer = Vec<BufferedBatch>     ─┐
                                    │
                  total_rows ───────┤
                                    │
                  N = min(writer_pool.threads, total_rows)
                                    │
                                    ▼
   split_buffer_into_row_shards ──► [shard_0] [shard_1] … [shard_{N-1}]
                                       │         │
                                       ▼  (writer_pool.install)
                                    build_one_shard (par_iter)
                                       │
                                       ▼   one SuperfileBuilder per shard
                                    ShardOutput { bytes, n_docs, id_range, scalar_stats }
                                       │
                                       ▼
                                    publish_superfiles
                                       │   prepare_segment (par_iter): open SuperfileReader,
                                       │   derive FTS + vector summaries
                                       ▼
                                    ArcSwap a successor Manifest with old + N new entries
```

After all workers finish,
[`publish_superfiles`](../../src/supertable/writer.rs) inserts every
shard's bytes into the `SuperfileStore`, derives per-superfile
summaries from the cached `SuperfileReader`, builds N
`SuperfileEntry`s, and ArcSwaps a new `Manifest` with old + new
entries.

The threshold-flush model lifts directly from superfile: when
`buffer_bytes ≥ commit_threshold_size_mb * 1024 * 1024` during an
`append()`, an internal `commit()` fires — bounding the writer's
in-memory footprint.

`build_one_shard` runs three things inside one rayon worker:

- The full superfile build (Parquet + FTS + vector indexes).
- Per-shard `id_min`/`id_max`/`n_docs` accumulation.
- Per-scalar-column `ScalarStatsTable` computation via Arrow
  aggregate kernels — captured here, before the buffered batches
  are dropped, since post-store `SuperfileReader` can't recover
  Arrow batch min/max without re-decoding.

The writer-thread count is overridable via the
[figment](https://github.com/SergioBenitez/Figment)-backed config:
set `INFINO_SUPERTABLE__WRITER_THREADS=N` (or
`supertable.writer_threads: N` in `config.yaml`) to pin the
shard count independently of `num_cpus`. The cpus/2 default leaves
headroom for the reader pool (see
[Dual-pool concurrency](#dual-pool-concurrency)); ingest-only
benchmarks usually override to `num_cpus` for peak throughput.

### Query fan-out

All three query paths share the same fan-out shape (BM25, prefix
BM25, vector kNN):

1. Compute the manifest-level skip mask (see
   [Skip pruning](#skip-pruning)).
2. Build a list of [`WorkUnit`](../../src/supertable/query/fts.rs)s
   from the kept superfiles (one per `(segment, doc_id_range)`
   tuple — see
   [Intra-segment sub-range fan-out](#intra-segment-sub-range-fan-out)
   below).
3. `options.reader_pool.install(|| work_units.par_iter().map(...).collect())`
   — pruned superfiles never trigger a `SuperfileStore::reader`
   call.
4. Each unit runs the per-superfile search via the existing
   `SuperfileReader::*` method (BlockMaxWAND / IVF + RaBitQ rerank,
   reused unchanged).
5. Tag each `(local_doc_id, score)` with the source superfile URI
   and concatenate.
6. Global top-k by score across every work unit.

The BM25 fan-out goes one step further: the orchestrator
tokenizes the query *once* (for the bloom-skip mask) and passes
the pre-tokenized term slice to every per-superfile search via
`SuperfileReader::bm25_search_pretokenized`. Eliminates `(N+1)·T`
redundant tokenizations across N superfiles and a T-token query.

#### Intra-segment sub-range fan-out

For multi-term OR queries (the BM25 hot path), the fan-out can
slice each segment into doc-id sub-ranges so that
`pool.threads >> kept.len()` cases saturate every pool thread
instead of leaving cores idle. Gated by
[`build_or_work_units`](../../src/supertable/query/fts.rs) on three
conditions:

1. `BoolMode::Or` with **≥ 2 terms** — the only shape where the
   range-aware BMM
   ([`run_max_score_bmm_range`](../../src/superfile/fts/reader.rs))
   is wired up. Single-term OR and AND stay on the un-ranged call
   (single-term BMW finishes in microseconds; AND uses the
   full-decode + HashMap intersection path).
2. `pool.threads > kept.len()` — otherwise every thread is already
   saturated by one segment and splitting just adds overhead.
3. Each candidate sub-range is at least `SUBRANGE_MIN_DOCS = 50_000`
   docs wide — below that, BMM bookkeeping + cross-sub-range top-K
   merge dominate the parallel win.

```text
   reader_pool.threads = 16, kept = 4 superfiles
   want_subranges = ceil(16/4) = 4 sub-ranges per superfile
   →  16 WorkUnits, each ~ n_docs/4 wide  (par_iter dispatches all 16 in parallel)

   reader_pool.threads = 16, kept = 16 superfiles
   want_subranges = 1
   →  16 WorkUnits, each = full segment  (identical to original shape)

   reader_pool.threads = 16, kept = 4 superfiles, but n_docs = 30K each
   cap_by_floor = 30K / SUBRANGE_MIN_DOCS = 0 → 1
   →  4 WorkUnits, each = full segment  (sub-range floor kicks in)
```

Each sub-range work unit calls
[`bm25_search_or_range_pretokenized`](../../src/superfile/fts/reader.rs)
which delegates to `run_max_score_bmm_range`. Single-term/AND
and `kept.is_empty()` short-circuit to a flat `WorkUnit::None`
range — same code path, no sub-range overhead. See
[superfile.md § Multi-term OR](./superfile.md#multi-term-or-search-wand-vs-maxscore-vs-block-max-maxscore)
for the underlying BMM range-aware loop.

### Skip pruning

Three manifest-only helpers in `query::skip`, all reading from
[`SuperfileEntry` summaries](#superfileentry-summaries) without ever
touching the store:

| Helper | Used by | Mechanism |
|---|---|---|
| `fts_bloom_skip` | `bm25_search` | OR mode keeps a superfile if any query term is possibly-present in the superfile's per-column bloom; AND mode requires all terms |
| `fts_prefix_skip` | `bm25_search_prefix` | drops a superfile if its lex term range can't overlap `[prefix, prefix_upper_bound)` |
| `vector_centroid_skip` | `vector_search` | v1 returns all-keep — see [phase-2 plans](#phase-2-plans) |
| `superfiles_sorted_by_centroid_distance` | (fan-out ordering hint) | indices sorted by centroid distance; biases fan-out toward likely-close superfiles |

Skip is **always conservative**: unknown column / empty terms /
missing summary → keep the superfile (per-superfile search will
surface the real signal). False-positive keeps cost a per-superfile
search call but never a wrong answer; false-negative prunes are
forbidden.

The integration test `tests/supertable_skip.rs` wraps the store
in a counting decorator and asserts the per-URI `reader()` call
delta over a single query — planted-rare-term queries on a
4-superfile supertable open exactly **1** superfile, not 4.

### Dual-pool concurrency

`SupertableOptions` holds two
[`rayon::ThreadPool`](https://docs.rs/rayon/) handles:

- **`reader_pool`** — fan-out for queries (BM25 / vector / SQL
  partition scan). Default `num_cpus`.
- **`writer_pool`** — rayon-shard at commit time. Default
  `max(1, num_cpus / 2)`.

The split exists so that a long-running `commit()` doesn't
co-saturate every CPU core with reader queries fighting for
time. The bench harness in
[`benches/hybrid/mixed_load.rs`](../../benches/hybrid/mixed_load.rs)
measures this directly:

```
mixed-load reader p99 under writer load (n_queries=500):
  writer_pool=num_cpus=16   (saturating): reader p99 = 187.30 ms
  writer_pool=num_cpus/2=8  (isolated):   reader p99 = 104.01 ms
  saturating / isolated ratio = 1.80×
```

Saturating the writer pool with `num_cpus` threads degrades
foreground reader latency by ~80% vs the `num_cpus/2` default
that leaves CPU headroom for the reader pool.

Both pool sizes are configurable via the existing `figment` config
layer:

```yaml
supertable:
  reader_threads: auto              # = num_cpus
  writer_threads: 4                 # explicit override
```

`auto` resolves at runtime; explicit positive integers override.
Nested env vars: `INFINO_SUPERTABLE__WRITER_THREADS=4`.

### SQL surface via DataFusion

`Supertable::query_sql(sql)` runs the query against a
[DataFusion](https://datafusion.apache.org/) `MemTable` whose
partitions mirror the manifest's superfiles — one partition per
superfile, each partition being the superfile's eagerly-decoded
Parquet `RecordBatch`es.

Why MemTable rather than a custom `TableProvider`?

- The in-memory `SuperfileStore` already holds every superfile's
  Parquet bytes in RAM; eagerly decoding into Arrow shifts the
  cost from `execute()` time to `register_table()` time without
  changing the working set.
- DataFusion still applies `FilterExec` above the MemTable, so
  per-batch predicate pushdown works as expected.
- A custom `TableProvider` that integrates DataFusion's
  `PruningPredicate` against the per-superfile `ScalarStatsTable`
  is the natural next extension; v1 ships the stats and the
  MemTable, a future iteration consumes them.

The SQL public API is sync: `query_sql` blocks on a
single-worker `tokio::runtime::Runtime` cached on
`SupertableInner` via `OnceLock` (lazy — first SQL query
allocates). Tokio threads are *separate* from the rayon pools;
they drive DataFusion's I/O state machine, not CPU work.

## Performance

Absolute runtime numbers — per-segment + multi-segment build
throughput, BM25 search across the query-shape battery, vector
kNN at calibrated recall, mixed-load reader p99 under concurrent
commit storm — are produced by the in-tree criterion harness
under `benches/`.

Where the wins come from architecturally:

- **FTS multi-term OR + prefix search** inherit the underlying
  superfile margins (BMW + dictionary tricks) and add a free
  manifest-level skip-pruning win on rare-term + prefix queries
  via the per-segment bloom filter + lex term range.
- **Vector kNN at scale** is the one place the supertable layer
  currently underperforms the single-segment superfile path —
  the fan-out-and-merge over many small IVF indexes doesn't yet
  beat a single global IVF. Tracking as future work (see
  [Phase 2 plans](#phase-2-plans)).
- **Mixed-load reader p99** under concurrent commit storm
  validates the dual-pool isolation claim — saturating
  `writer_pool` degrades reader p99 by ~3× vs the `num_cpus/2`
  default. Bench:
  [`benches/hybrid/mixed_load.rs`](../../benches/hybrid/mixed_load.rs).

## Test strategy

| Suite | Catches |
|---|---|
| [`tests/supertable/query/skip_pruning.rs`](../../tests/supertable/query/skip_pruning.rs) | manifest-level skip prunes superfiles before any store call |
| [`tests/supertable/query/brute_force_oracle.rs`](../../tests/supertable/query/brute_force_oracle.rs) | brute-force BM25 across N segments: top-k matches the textbook formula with identical per-segment IDF + global merge |
| [`tests/supertable/query/hierarchical.rs`](../../tests/supertable/query/hierarchical.rs) | hierarchical-manifest query path (list-level skip → lazy-load parts → fan-out) |
| [`tests/supertable/commit/`](../../tests/supertable/commit/) | append + commit pipeline, manifest-id increment, pointer-atomic publish, id uniqueness across threads, partition assignment, in-process concurrency |
| [`tests/supertable/manifest/`](../../tests/supertable/manifest/) | eager-vs-lazy-open threshold path |
| [`tests/supertable/disk_cache/`](../../tests/supertable/disk_cache/) | cold-fetch coordinator + hybrid / sweep policies + supertable-disk-cache integration |
| [`tests/supertable/storage/`](../../tests/supertable/storage/) | S3-compatible smoke run (s3s-fs + AWS object_store wire-protocol path) |
| [`tests/supertable_commit_crash_localfs.rs`](../../tests/supertable_commit_crash_localfs.rs) | spawn-self crash test: commit pipeline survives kill -9 between PUTs |
| [`tests/supertable_concurrent_processes.rs`](../../tests/supertable_concurrent_processes.rs) | spawn-self cross-process OCC retry |
| in-module `#[cfg(test)] mod tests` | per-module unit tests across `manifest`, `options`, `store`, `vector_split`, `writer`, `query::{sql,fts,vector,skip}`, `handle`, `error` |

The brute-force BM25 oracle catches scoring-math bugs that
planted-ground-truth tests can't, AND catches the cross-segment
class of bugs that the single-segment oracle misses: wrong
segment partitioning, wrong tagging of per-segment hits with
their segment URI, wrong score-direction in the global top-k
merge. Brute-force scores each segment with the same per-segment
IDF + avgdl shape the supertable's fan-out produces, then merges
into the same global top-k.

## Known limitations

- **In-memory only.** Process restart loses every superfile. 003
  adds the on-disk persistence path (mmap + range-fetch +
  atomic-rename pointer file).
- **Single-writer.** A second concurrent
  `SupertableWriter` acquisition returns
  `BuildError::SupertableInUse`. Cross-process OCC is a phase-2
  concern.
- **No vector cutoff-driven skip.** `vector_centroid_skip`
  returns all-keep in v1; a future cluster-aware cutoff driver
  layers on top once a bench harness motivates the right
  early-termination shape.
- **No SQL pruning predicate.** `Supertable::query_sql` uses a
  `MemTable` that scans every superfile. A future custom
  `TableProvider` + DataFusion `PruningPredicate` integration
  consumes the per-superfile `ScalarStatsTable` v1 already
  populates.
- **No hybrid search.** BM25 and vector are separate methods;
  combined-relevance scoring ('most fields' over BM25 + cosine)
  is in 007.
- **No tombstones / updates.** Segments are immutable; updates
  require write-then-supersede (whole-superfile replace) or 005's
  tombstone bitmap.

## Phase 2 plans

The supertable's design intentionally leaves room for these
without breaking the existing manifest shape or query API:

- **`003_object_store.md`** — adds persistence for both
  superfiles and supertables. Manifest serde, atomic-rename
  pointer files, `StorageProvider` trait, LocalFsProvider /
  S3 / GCS, crash safety semantics, multi-writer-detection lock
  files. The in-memory `Manifest` is designed to map 1:1 onto
  the on-disk shape there.
- **`004_compaction.md`** — N small superfiles → fewer larger
  ones. Reads the same in-memory `Manifest`; produces a
  successor `Manifest` pointing at the post-compaction superfiles.
- **`005_updates.md`** — tombstones + insert-with-same-logical-
  id. `Manifest` likely gains a per-superfile tombstone bitmap
  reference; `SuperfileEntry.scalar_stats` already has the shape
  needed to skip via tombstone-aware predicates.
- **`006_distributed.md`** — distributed query coordination
  across nodes (catalog-backed leader election, cross-node
  fan-out), sub-linear superfile-list indexing for ≫ 1K superfiles.
  The cross-process write story already lives in 003 (OCC on
  the pointer file); 006 generalizes to cross-node coordination
  on top of that.
- **`007_extended_sql.md`** — DataFusion UDF wiring + advanced
  query shapes (hybrid search as a single API, etc.). The
  supertable's explicit method-style v1 (`reader.bm25_search`
  etc.) is the surface 007 extends.
