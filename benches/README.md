# infino benches

Infino's in-tree benchmarks measure Infino itself. Cross-engine comparison
benches live in `retrievalbench`; these tables are the Infino reference numbers
those comparisons are checked against.

All benchmarks run on Infino's custom bench harness (one binary, no external
bench framework). The harness owns the measured lifecycle directly:

- generate the corpus once;
- build the artifact once;
- run correctness on that built artifact;
- run warm reads on that artifact;
- upload or commit that same artifact for object-store tiers;
- run cold reads against the uploaded/committed artifact with fresh cache state;
- sample RSS around the measured phase;
- render terminal and markdown reports through `report.rs`.

The invariant is simple: **the first measured build produces the artifact used by
correctness, warm reads, and cold upload/commit.** The benchmark must not rebuild a
second copy just to run correctness or object-store reads.

## Bench Shapes

- **Superfile** — single-artifact, in-memory read path. Default scale: `1M`
  docs, controlled by `INFINO_BENCH_SUPERFILE_DOCS`.
- **Supertable** — multi-artifact table committed to object storage and read
  through warm/cold table paths. Default scale: `10M` docs, controlled by
  `INFINO_BENCH_SUPERTABLE_DOCS`.
- Doc counts are plain integers — `100K`/`1M` suffixes do not parse.
- **Writer count** — build rows report `1 writer` and `N writers`. `N` defaults
  to the machine's logical core count and is controlled by
  `INFINO_BENCH_WRITERS`.

## Invocation

Selection is positional tokens after `--`: `[tier] [modality] [phase ...]`,
space-separated. Tier is `superfile` | `supertable`; modality is `fts` |
`vector` | `sql`; phase is `build` | `warm` | `cold` (`search` = warm+cold).
Omitted tokens mean "all".

```sh
# Run every tier × modality test, all phases.
cargo bench

# Run one cell, all phases.
cargo bench -- superfile fts
cargo bench -- supertable vector

# One tier, all three modalities.
cargo bench -- supertable

# Select phases.
cargo bench -- superfile sql cold
cargo bench -- supertable vector build warm

# Smaller local loop (plain integer; K/M suffixes do not parse).
INFINO_BENCH_SUPERFILE_DOCS=100000 cargo bench -- superfile fts warm

# Override the N-writers build row.
INFINO_BENCH_WRITERS=4 cargo bench -- superfile fts build

# Refresh the markdown sections in this file.
INFINO_BENCH_UPDATE_README=1 cargo bench -- superfile fts

# Diagnostics (standalone programs in the same binary; never implied by
# `all` or a bare `cargo bench`).
cargo bench -- diagnostic                  # all five
cargo bench -- diagnostic scale tombstone  # a subset, grouped
cargo bench -- tombstone                   # bare names also work
# Names: scale | tombstone | update | sql-diag | object-store
```

## Object-store backends

The supertable benches (and the superfile cold tier) run against an object
store, chosen **explicitly** by `INFINO_BENCH_STORE` — never inferred from
which credentials happen to be set:

| `INFINO_BENCH_STORE` | Backend | Extra env |
|---|---|---|
| _unset_ / `s3s_fs` | in-process s3s-fs emulator | — |
| `s3` | real AWS S3 | `INFINO_REAL_S3_BUCKET` + the standard `AWS_*` credentials |
| `azure` | real Azure Blob | `INFINO_REAL_AZURE_CONTAINER` + `AZURE_STORAGE_ACCOUNT_NAME` + `AZURE_STORAGE_ACCOUNT_KEY` |

```sh
# Superfile cold tiers: any backend (s3s-fs is the zero-setup default).
cargo bench -- superfile fts cold

# Supertable tests: real object store only (s3 or azure). s3s-fs lacks the
# multi-commit If-Match CAS the supertable commit needs, so it is rejected.
INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=my-bucket \
  cargo bench -- supertable fts
INFINO_BENCH_STORE=azure INFINO_REAL_AZURE_CONTAINER=my-container \
  AZURE_STORAGE_ACCOUNT_NAME=... AZURE_STORAGE_ACCOUNT_KEY=... \
  cargo bench -- supertable sql cold
```

A real-backend run writes under a unique prefix and deletes it on exit; set
`INFINO_BENCH_KEEP_TABLE=1` to keep it (the prefix is logged). The s3s-fs
emulator self-cleans and reproduces request/byte volume, not network latency.

## Prepared datasets

The supertable corpus is fully seeded, so an ingested table is reusable.
`dataset` verbs split the run: **prepare** once (ingest to a fixed prefix and
write a `dataset.json` sidecar), then **bench** the read phases against it as
many times as needed — no corpus generation, no ingest. Real object store
only.

```sh
# Prepare a dataset (one sub-prefix per modality: <prefix>/{fts,vector,sql}).
INFINO_BENCH_STORE=azure INFINO_REAL_AZURE_CONTAINER=my-container \
  cargo bench -- dataset prepare datasets/bench-10m

# Benchmark an existing dataset (fails fast if it is not there).
cargo bench -- dataset bench datasets/bench-10m vector warm

# End-to-end: prepare if absent, then bench.
cargo bench -- dataset run datasets/bench-10m fts
```

The sidecar records the corpus/index knobs the dataset was built with; the
bench refuses to open a dataset whose knobs don't match its own config
(re-prepare instead). `INFINO_BENCH_SUPERTABLE_DOCS` must therefore match the
prepare-time count. The `Dataset bench (Azure)` workflow drives the same
verbs from CI.

## Test Matrix

The matrix is tier × modality — six cells:

| Selector | Tier | Modality |
|---|---|---|
| `superfile fts` | superfile | FTS |
| `superfile vector` | superfile | vector |
| `superfile sql` | superfile | SQL |
| `supertable fts` | supertable | FTS |
| `supertable vector` | supertable | vector |
| `supertable sql` | supertable | SQL |

Each cell supports `build`, `warm`, and `cold`. If no cell is selected, all
six run. If no phase is supplied, all three phases run.

## Code Layout (`infino-bench-utils`)

```text
corpus.rs                   synthetic corpora + brute-force oracles
executors.rs                shared build/search/query executors + emitters
harness/                    engine interfaces and generic drivers
report.rs, markdown.rs      terminal + markdown rendering with deltas
rss.rs                      per-phase RSS sampling
tiers.rs                    object-store backend selection (s3s-fs / s3 / azure)
superfile.rs                superfile runners by modality (fts / vector / sql)
supertable.rs               supertable object-store runners by modality
ingest/, fixture/           supertable object-store helpers
scale.rs, sql_diag.rs       diagnostics (recall gates, SQL dispatch tax)
tombstone_overhead.rs       diagnostics (delete/tombstone query overhead)
supertable_update.rs        diagnostics (update/delete pipeline)
unified_object_store.rs     diagnostics (cold lazy-fetch request shape)
```

## Result Anchors

Each generated section is wrapped in
`<!-- BEGIN: bench/... --> <!-- END: bench/... -->` markers. When
`INFINO_BENCH_UPDATE_README=1` is set, the runners replace the matching
block in place. Cells render `value (delta)` against the previous run's
baseline (`target/infino-bench/<bench>.json`); `(new)` means no baseline
existed yet.

---

## Results

Current numbers: 1M docs per tier, real AWS S3 (us-east-1), recorded
2026-06-09. Supertable tables are 256 superfiles across 16 commits.

### FTS — superfile (single-superfile, 1M docs)

<!-- BEGIN: bench/fts/superfile/ingest -->
### Superfile FTS — ingest, single-superfile / in-memory (1M docs, Zipfian, 200 tokens/doc, 10K vocab)

_Host: unknown CPU · 10C/10T · macos/aarch64_

Build path: `SuperfileBuilder` → unified `.parquet` (same as production supertable commit), through the engine-generic `run_fts` driver the cross-engine comparison also uses. Rows are by writer count: `1 writer` is the single-threaded build (and the index queries run against); `N writers` is the sharded parallel build. Bandwidth is over the logical input text payload. Δ is vs the previous run.

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 23.11 s (new) | 43.3 K/s (new) | 87.0 MB/s (new) | 0 B (new) | 0 B (new) | 0 B (new) |
| 10 writers | 3.80 s (new) | 262.9 K/s (new) | 528.4 MB/s (new) | 0 B (new) | 0 B (new) | 0 B (new) |
<!-- END: bench/fts/superfile/ingest -->

<!-- BEGIN: bench/fts/superfile/search -->
### Superfile FTS — search, single-superfile / in-memory (1M docs)

_Host: unknown CPU · 10C/10T · macos/aarch64_

Warm = `SuperfileReader::open` in memory (per-query p50); cold = same `.parquet` on object storage via `DiskCacheStore::reader` -> `bm25_search` (production cold path). Δ is vs the previous run.

**OR queries**

| Query | warm | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- |
| single_rare | 1.25 µs (+7.2% worse) | 0 B (new) | 0 B (new) | 0 B (new) | 5.04 s (new) | 1.67 s (new) |
| single_df1 | 958 ns (+64.3% worse) | 0 B (new) | 0 B (new) | 0 B (new) | 5.04 s (new) | 7.88 µs (new) |
| single_common | 18.33 µs (-6.2% better) | 0 B (new) | 0 B (new) | 0 B (new) | 5.04 s (new) | 1.68 s (new) |
| two_term_or | 262.75 µs (+2.6% ~) | 0 B (new) | 0 B (new) | 0 B (new) | 5.03 s (new) | 1.69 s (new) |
| three_wide_or | 2.98 ms (-2.7% ~) | 0 B (new) | 0 B (new) | 0 B (new) | 5.04 s (new) | 1.62 s (new) |
| three_similar_or | 11.39 ms (-2.6% ~) | 0 B (new) | 0 B (new) | 0 B (new) | 5.04 s (new) | 1.62 s (new) |
| five_term_or | 20.29 ms (-0.7% ~) | 0 B (new) | 0 B (new) | 0 B (new) | 5.05 s (new) | 1.67 s (new) |
| ten_term_or | 65.84 ms (-2.5% ~) | 0 B (new) | 0 B (new) | 0 B (new) | 5.05 s (new) | 1.73 s (new) |

**AND queries**

| Query | warm | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- |
| two_term_and | 282.79 µs (+4.5% worse) | 0 B (new) | 0 B (new) | 0 B (new) | 5.05 s (new) | 1.67 s (new) |
| three_wide_and | 4.30 ms (-1.1% ~) | 0 B (new) | 0 B (new) | 0 B (new) | 5.05 s (new) | 1.62 s (new) |
| three_similar_and | 7.05 ms (-0.9% ~) | 0 B (new) | 0 B (new) | 0 B (new) | 5.05 s (new) | 1.63 s (new) |
| five_term_and | 8.54 ms (+1.5% ~) | 0 B (new) | 0 B (new) | 0 B (new) | 5.08 s (new) | 1.66 s (new) |
| ten_term_and | 10.31 ms (-0.1% ~) | 0 B (new) | 0 B (new) | 0 B (new) | 5.05 s (new) | 1.67 s (new) |

**Per-algorithm probes (WAND+BMW vs MaxScore+BMM)**

| Shape | WAND+BMW | MaxScore+BMM |
| --- | --- | --- |
| wide_3_or | 9.63 ms (+2.6% ~) | 3.04 ms (-0.0% ~) |
| similar_3_or | 18.14 ms (+0.7% ~) | 11.89 ms (+0.7% ~) |
| similar_5_or | 49.85 ms (+1.3% ~) | 20.29 ms (+0.7% ~) |
| similar_10_or | 425.63 ms (+1.1% ~) | 67.01 ms (-1.3% ~) |
<!-- END: bench/fts/superfile/search -->

<!-- BEGIN: bench/fts/superfile/negation -->
### Superfile FTS — negation (`-term`), warm (1M docs)

_Host: unknown CPU · 10C/10T · macos/aarch64_

Through the string `bm25_hits_async` path (parses the `-` sigil); a correctness gate (no hit contains a negated term) runs before timing. Δ is vs the previous run.

**Negation queries**

| Query | warm |
| --- | --- |
| mid_pos_common_neg | 1.63 ms (-0.4% ~) |
| mid_pos_rare_neg | 27.96 µs (+1.1% ~) |
| two_mid_or_common_neg | 4.55 ms (-0.8% ~) |
| two_mid_and_common_neg | 5.15 ms (+3.2% worse) |
<!-- END: bench/fts/superfile/negation -->

### FTS — supertable (multi-superfile, 1M docs, real S3)

<!-- BEGIN: bench/fts/supertable/ingest -->
### Supertable FTS — ingest, multi-superfile / object-store (1M docs, 16 commits)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| FTS-only | 53.83 s (new) | 18.6 K/s (new) | 256 | 9.61 GiB (new) | 8.62 GiB (new) | 8.80 GiB (new) |
<!-- END: bench/fts/supertable/ingest -->

<!-- BEGIN: bench/fts/supertable/search -->
### Supertable FTS — search, multi-superfile / object-store (1M docs)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

**OR queries**

| Query | warm | Peak RSS | Median RSS | P90 RSS | cold |
| --- | --- | --- | --- | --- | --- |
| single_rare | 25.94 ms (new) | 8.69 GiB (new) | 8.68 GiB (new) | 8.69 GiB (new) | 284.60 ms (new) |
| single_df1 | 2.40 ms (new) | 8.68 GiB (new) | 8.68 GiB (new) | 8.68 GiB (new) | 148.96 ms (new) |
| single_common | 22.03 ms (new) | 8.68 GiB (new) | 8.68 GiB (new) | 8.68 GiB (new) | 431.21 ms (new) |
| two_term_or | 36.57 ms (new) | 8.67 GiB (new) | 8.67 GiB (new) | 8.67 GiB (new) | 617.51 ms (new) |
| three_wide_or | 34.41 ms (new) | 8.67 GiB (new) | 8.67 GiB (new) | 8.67 GiB (new) | 522.29 ms (new) |
| three_similar_or | 29.76 ms (new) | 8.67 GiB (new) | 8.67 GiB (new) | 8.67 GiB (new) | 508.97 ms (new) |
| five_term_or | 37.35 ms (new) | 8.67 GiB (new) | 8.67 GiB (new) | 8.67 GiB (new) | 766.35 ms (new) |
| ten_term_or | 42.97 ms (new) | 8.68 GiB (new) | 8.68 GiB (new) | 8.68 GiB (new) | 667.52 ms (new) |

**AND queries**

| Query | warm | Peak RSS | Median RSS | P90 RSS | cold |
| --- | --- | --- | --- | --- | --- |
| two_term_and | 36.01 ms (new) | 8.67 GiB (new) | 8.67 GiB (new) | 8.67 GiB (new) | 494.09 ms (new) |
| three_wide_and | 34.52 ms (new) | 8.67 GiB (new) | 8.67 GiB (new) | 8.67 GiB (new) | 432.06 ms (new) |
| three_similar_and | 29.20 ms (new) | 8.67 GiB (new) | 8.67 GiB (new) | 8.67 GiB (new) | 494.39 ms (new) |
| five_term_and | 34.58 ms (new) | 8.67 GiB (new) | 8.67 GiB (new) | 8.67 GiB (new) | 418.77 ms (new) |
| ten_term_and | 34.90 ms (new) | 8.67 GiB (new) | 8.67 GiB (new) | 8.67 GiB (new) | 544.50 ms (new) |
<!-- END: bench/fts/supertable/search -->

### Vector — superfile (single-superfile, 1M × 384)

<!-- BEGIN: bench/vector/superfile/ingest -->
### Superfile vector — ingest, single-superfile / in-memory (1M docs × dim=384)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 20.77 s (new) | 48.1 K/s (new) | 73.9 MB/s (new) | 4.44 GiB (new) | 2.34 GiB (new) | 3.38 GiB (new) |
| 16 writers | 2.74 s (new) | 365.2 K/s (new) | 561.0 MB/s (new) | 7.85 GiB (new) | 6.74 GiB (new) | 7.65 GiB (new) |
<!-- END: bench/vector/superfile/ingest -->

<!-- BEGIN: bench/vector/superfile/search -->
### Superfile vector — search, single-superfile / in-memory (1M docs × dim=384)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

| Recall target | (p, r) | recall | warm | Peak RSS | Median RSS | P90 RSS | cold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | p=1, r=1024 | 0.962 | 1.06 ms (new) | 4.28 GiB (new) | 4.28 GiB (new) | 4.28 GiB (new) | 464.40 ms (new) |
| 0.95 | p=1, r=1024 | 0.962 | 1.05 ms (new) | 4.29 GiB (new) | 4.29 GiB (new) | 4.29 GiB (new) | 319.37 ms (new) |
| 0.99 | p=10, r=256 | 0.998 | 1.53 ms (new) | 4.30 GiB (new) | 4.29 GiB (new) | 4.30 GiB (new) | 443.60 ms (new) |
| default | p=8, r=20 | — | 853.92 µs (new) | 4.30 GiB (new) | 4.30 GiB (new) | 4.30 GiB (new) | 593.77 ms (new) |
<!-- END: bench/vector/superfile/search -->

### Vector — supertable (multi-superfile, 1M × 384, real S3)

<!-- BEGIN: bench/vector/supertable/ingest -->
### Supertable vector — ingest, multi-superfile / object-store (1M docs × dim=384, 16 commits)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| vector-only | 28.50 s (new) | 35.1 K/s (new) | 256 | 10.36 GiB (new) | 9.01 GiB (new) | 10.30 GiB (new) |
<!-- END: bench/vector/supertable/ingest -->

<!-- BEGIN: bench/vector/supertable/search -->
### Supertable vector — search, multi-superfile / object-store (1M docs × dim=384)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Correctness gate: recall@10 = 0.995 (nprobe=64, rerank=256, 20 queries).

| Recall target | (p, r) | recall | warm | Peak RSS | Median RSS | P90 RSS | cold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | p=5, r=1 | 0.988 | 5.63 ms (new) | 11.63 GiB (new) | 11.63 GiB (new) | 11.63 GiB (new) | 570.11 ms (new) |
| 0.95 | p=5, r=1 | 0.988 | 4.72 ms (new) | 12.27 GiB (new) | 12.26 GiB (new) | 12.27 GiB (new) | 608.43 ms (new) |
| 0.99 | p=10, r=1 | 0.996 | 5.04 ms (new) | 12.42 GiB (new) | 12.42 GiB (new) | 12.42 GiB (new) | 702.35 ms (new) |
| default | p=8, r=20 | — | 6.87 ms (new) | 12.35 GiB (new) | 12.19 GiB (new) | 12.35 GiB (new) | 671.72 ms (new) |
<!-- END: bench/vector/supertable/search -->

### Supertable — ingest summary (all shapes, real S3)

<!-- BEGIN: bench/supertable/ingest -->
### Supertable — ingest, multi-superfile / object-store (1M docs, 16 commits)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| FTS-only | 53.83 s (new) | 18.6 K/s (new) | 256 | 9.61 GiB (new) | 8.62 GiB (new) | 8.80 GiB (new) |
| vector-only | 28.50 s (new) | 35.1 K/s (new) | 256 | 10.36 GiB (new) | 9.01 GiB (new) | 10.30 GiB (new) |
| SQL | 75.02 s (new) | 13.3 K/s (new) | 256 | 11.42 GiB (new) | 9.44 GiB (new) | 11.02 GiB (new) |
<!-- END: bench/supertable/ingest -->

### SQL — superfile (single superfile, 1M rows)

<!-- BEGIN: bench/sql/build -->
### Superfile SQL — ingest, single superfile / in-memory (1M rows: title + category + score)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 10.47 s (new) | 95.5 K/s (new) | 192.0 MB/s (new) | 7.39 GiB (new) | 6.41 GiB (new) | 7.11 GiB (new) |
| 16 writers | 6.25 s (new) | 160.1 K/s (new) | 321.7 MB/s (new) | 15.21 GiB (new) | 12.83 GiB (new) | 14.91 GiB (new) |
<!-- END: bench/sql/build -->

<!-- BEGIN: bench/sql/query -->
### Superfile SQL — query, single superfile / in-memory (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

The headline comparison is the scan-vs-pushdown pairs: the *same* selective
equality (one matching row) run as a plain DataFusion scan vs through Infino's
token index (FTS-pushdown — the index selects the candidate rows, DataFusion
verifies). Same predicate, same result, so the gap is purely the index as an
access path.

**Aggregations & count-filters (read + compute, return few rows — not the index A/B)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| agg_max_title | 180.15 ms (new) | 1 | 10.63 GiB (new) | 7.75 GiB (new) | 10.30 GiB (new) |
| filter_category_count | 10.01 ms (new) | 1 | 7.57 GiB (new) | 7.57 GiB (new) | 7.57 GiB (new) |
| filter_rating_count | 7.63 ms (new) | 1 | 7.50 GiB (new) | 7.50 GiB (new) | 7.50 GiB (new) |
| count_star | 6.22 ms (new) | 1 | 7.49 GiB (new) | 7.49 GiB (new) | 7.49 GiB (new) |
| group_by_category | 7.20 ms (new) | 4 | 7.49 GiB (new) | 7.32 GiB (new) | 7.49 GiB (new) |

**Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 8.05 ms (new) | 1 | 7.26 GiB (new) | 7.26 GiB (new) | 7.26 GiB (new) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 9.99 ms (new) | 1 | 7.29 GiB (new) | 7.29 GiB (new) | 7.29 GiB (new) |

**FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 4.06 ms (new) | 1 | 7.29 GiB (new) | 7.29 GiB (new) | 7.29 GiB (new) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 1.70 ms (new) | 1 | 7.28 GiB (new) | 7.28 GiB (new) | 7.28 GiB (new) |

**Aggregate over FTS candidates — Full Scan (DataFusion only)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 10.23 ms (new) | 1 | 7.28 GiB (new) | 7.18 GiB (new) | 7.28 GiB (new) |
| SUM(rating)         key=? (1 row) | 10.16 ms (new) | 1 | 7.18 GiB (new) | 7.18 GiB (new) | 7.18 GiB (new) |
| MAX(rating)         key=? (1 row) | 10.77 ms (new) | 1 | 7.18 GiB (new) | 7.18 GiB (new) | 7.18 GiB (new) |
| AVG(rating)         key=? (1 row) | 10.40 ms (new) | 1 | 7.18 GiB (new) | 7.18 GiB (new) | 7.18 GiB (new) |
| SUM(rating) bucket IN all (1M rows) | 14.71 ms (new) | 1 | 7.16 GiB (new) | 7.16 GiB (new) | 7.16 GiB (new) |

**Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 1.86 ms (new) | 1 | 7.11 GiB (new) | 7.11 GiB (new) | 7.11 GiB (new) |
| SUM(rating)         key=? (1 row) | 1.92 ms (new) | 1 | 7.11 GiB (new) | 7.11 GiB (new) | 7.11 GiB (new) |
| MAX(rating)         key=? (1 row) | 2.07 ms (new) | 1 | 7.12 GiB (new) | 7.11 GiB (new) | 7.12 GiB (new) |
| AVG(rating)         key=? (1 row) | 1.75 ms (new) | 1 | 7.12 GiB (new) | 7.12 GiB (new) | 7.12 GiB (new) |
| SUM(rating) bucket IN all (1M rows) | 12.34 ms (new) | 1 | 7.12 GiB (new) | 7.11 GiB (new) | 7.12 GiB (new) |

**Search table functions (bm25 / vector / hybrid / token / exact)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| bm25_search | 839.02 µs (new) | 10 | 7.29 GiB (new) | 7.29 GiB (new) | 7.29 GiB (new) |
| vector_search | 1.49 ms (new) | 10 | 7.20 GiB (new) | 7.20 GiB (new) | 7.20 GiB (new) |
| hybrid_search | 1.64 ms (new) | 10 | 7.20 GiB (new) | 7.20 GiB (new) | 7.20 GiB (new) |
| token_match (all rows) | 96.21 ms (new) | 1000.0K | 7.28 GiB (new) | 7.26 GiB (new) | 7.28 GiB (new) |
| token_match (selective) | 488.25 µs (new) | 1 | 7.26 GiB (new) | 7.26 GiB (new) | 7.26 GiB (new) |
| exact_match | 2.97 ms (new) | 1 | 7.26 GiB (new) | 7.26 GiB (new) | 7.26 GiB (new) |
<!-- END: bench/sql/query -->

<!-- BEGIN: bench/sql/superfile/cold -->
### Superfile SQL — cold query, object-store (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

| Query | cold |
| --- | --- |
| agg_max_title | 1.75 s (new) |
| filter_category_count | 289.78 ms (new) |
| filter_rating_count | 369.47 ms (new) |
| count_star | 73.49 ms (new) |
| group_by_category | 203.36 ms (new) |
<!-- END: bench/sql/superfile/cold -->

### SQL — supertable (multi-superfile, 1M rows, real S3)

<!-- BEGIN: bench/sql/supertable/ingest -->
### Supertable SQL — ingest, multi-superfile / object-store (1M rows, 16 commits)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| SQL | 75.02 s (new) | 13.3 K/s (new) | 256 | 11.42 GiB (new) | 9.44 GiB (new) | 11.02 GiB (new) |
<!-- END: bench/sql/supertable/ingest -->

<!-- BEGIN: bench/sql/supertable/warm -->
### Supertable SQL — warm queries, warm cache / object-store (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

**Aggregations & count-filters (read + compute, return few rows — not the index A/B)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| agg_max_title | 182.73 ms (new) | 1 | 10.36 GiB (new) | 10.33 GiB (new) | 10.35 GiB (new) |
| filter_category_count | 23.30 ms (new) | 1 | 10.35 GiB (new) | 10.34 GiB (new) | 10.35 GiB (new) |
| filter_rating_count | 21.16 ms (new) | 1 | 10.15 GiB (new) | 10.15 GiB (new) | 10.15 GiB (new) |
| count_star | 21.84 ms (new) | 1 | 10.04 GiB (new) | 10.04 GiB (new) | 10.04 GiB (new) |
| group_by_category | 21.79 ms (new) | 4 | 10.04 GiB (new) | 10.04 GiB (new) | 10.04 GiB (new) |

**Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 7.78 ms (new) | 1 | 10.61 GiB (new) | 10.61 GiB (new) | 10.61 GiB (new) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 22.54 ms (new) | 1 | 10.62 GiB (new) | 10.61 GiB (new) | 10.62 GiB (new) |

**FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 3.96 ms (new) | 1 | 10.61 GiB (new) | 10.61 GiB (new) | 10.61 GiB (new) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 1.38 ms (new) | 1 | 10.61 GiB (new) | 10.61 GiB (new) | 10.61 GiB (new) |

**Aggregate over FTS candidates — Full Scan (DataFusion only)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 23.07 ms (new) | 1 | 10.49 GiB (new) | 10.49 GiB (new) | 10.49 GiB (new) |
| SUM(rating)         key=? (1 row) | 23.31 ms (new) | 1 | 10.47 GiB (new) | 10.46 GiB (new) | 10.47 GiB (new) |
| MAX(rating)         key=? (1 row) | 24.07 ms (new) | 1 | 10.46 GiB (new) | 10.44 GiB (new) | 10.46 GiB (new) |
| AVG(rating)         key=? (1 row) | 22.90 ms (new) | 1 | 10.44 GiB (new) | 10.44 GiB (new) | 10.44 GiB (new) |
| SUM(rating) bucket IN all (1M rows) | 30.30 ms (new) | 1 | 10.44 GiB (new) | 10.44 GiB (new) | 10.44 GiB (new) |

**Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 1.70 ms (new) | 1 | 10.44 GiB (new) | 10.44 GiB (new) | 10.44 GiB (new) |
| SUM(rating)         key=? (1 row) | 1.91 ms (new) | 1 | 10.44 GiB (new) | 10.44 GiB (new) | 10.44 GiB (new) |
| MAX(rating)         key=? (1 row) | 2.05 ms (new) | 1 | 10.44 GiB (new) | 10.44 GiB (new) | 10.44 GiB (new) |
| AVG(rating)         key=? (1 row) | 1.78 ms (new) | 1 | 10.44 GiB (new) | 10.44 GiB (new) | 10.44 GiB (new) |
| SUM(rating) bucket IN all (1M rows) | 67.47 ms (new) | 1 | 10.44 GiB (new) | 10.44 GiB (new) | 10.44 GiB (new) |

**Search table functions (bm25 / vector / hybrid / token / exact)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| bm25_search | 2.76 ms (new) | 10 | 10.07 GiB (new) | 10.04 GiB (new) | 10.07 GiB (new) |
| vector_search | 3.44 ms (new) | 10 | 10.09 GiB (new) | 10.07 GiB (new) | 10.09 GiB (new) |
| hybrid_search | 3.76 ms (new) | 10 | 10.09 GiB (new) | 10.09 GiB (new) | 10.09 GiB (new) |
| token_match (all rows) | 141.38 ms (new) | 1000.0K | 10.61 GiB (new) | 10.59 GiB (new) | 10.61 GiB (new) |
| token_match (selective) | 693.66 µs (new) | 1 | 10.61 GiB (new) | 10.61 GiB (new) | 10.61 GiB (new) |
| exact_match | 2.96 ms (new) | 1 | 10.61 GiB (new) | 10.61 GiB (new) | 10.61 GiB (new) |
<!-- END: bench/sql/supertable/warm -->

<!-- BEGIN: bench/sql/supertable/cold -->
### Supertable SQL — cold queries, fresh cache / object-store (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

| Query | cold |
| --- | --- |
| agg_max_title | 1.68 s (new) |
| filter_category_count | 1.22 s (new) |
| filter_rating_count | 1.11 s (new) |
| count_star | 145.31 ms (new) |
| group_by_category | 837.94 ms (new) |
<!-- END: bench/sql/supertable/cold -->
