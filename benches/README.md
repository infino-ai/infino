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

Multi-cell runs execute **each tier × modality cell in its own child process**
(a re-exec of the bench binary with that cell's selectors). RSS is per-process,
so a cell running after another would otherwise inherit its predecessors'
residue — measured at 1M docs, the supertable FTS cell reported ~9 GiB when run
in-process after the three superfile cells vs ~1.1 GiB isolated. A single
selected cell runs inline (its process is the isolation).

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

### FTS — superfile (single-segment, 1M docs)

<!-- BEGIN: bench/fts/superfile/ingest -->
### Superfile FTS — ingest, single-segment / in-memory (1M docs, Zipfian, 200 tokens/doc, 10K vocab)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Build path: `SuperfileBuilder` → unified `.parquet` (same as production supertable commit), through the engine-generic `run_fts` driver the cross-engine comparison also uses. Rows are by writer count: `1 writer` is the single-threaded build (and the index queries run against); `N writers` is the sharded parallel build. Bandwidth is over the logical input text payload. Δ is vs the previous run.

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 16.78 s (new) | 59.6 K/s (new) | 119.8 MB/s (new) | 5.78 GiB (new) | 3.75 GiB (new) | 4.87 GiB (new) |
| 16 writers | 2.11 s (new) | 473.5 K/s (new) | 951.8 MB/s (new) | 7.91 GiB (new) | 7.06 GiB (new) | 7.62 GiB (new) |
<!-- END: bench/fts/superfile/ingest -->

<!-- BEGIN: bench/fts/superfile/search -->
### Superfile FTS — search, single-segment / in-memory (1M docs)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Warm = `SuperfileReader::open` in memory (per-query p50); cold = same `.parquet` on object storage via `DiskCacheStore::reader` -> `bm25_search` (production cold path). Δ is vs the previous run.

**OR queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_rare | 9.69 µs (+1339.2% worse) | 10.78 ms (new) | 3.69 GiB (+1829.6% worse) | 3.69 GiB (+1829.9% worse) | 3.69 GiB (+1829.6% worse) | 184.21 ms (new) | 27.83 ms (new) |
| single_df1 | 559 ns (+68.4% worse) | 17.31 ms (new) | 3.69 GiB (+1829.2% worse) | 3.69 GiB (+1829.2% worse) | 3.69 GiB (+1829.2% worse) | 166.25 ms (new) | 11.26 µs (new) |
| single_common | 1.94 ms (+20514.4% worse) | 43.15 ms (new) | 3.69 GiB (+1829.6% worse) | 3.69 GiB (+1830.0% worse) | 3.69 GiB (+1829.6% worse) | 136.85 ms (new) | 44.49 ms (new) |
| two_term_or | 262.06 µs (+2547.3% worse) | 41.10 ms (new) | 3.69 GiB (+1830.7% worse) | 3.69 GiB (+1830.7% worse) | 3.69 GiB (+1830.7% worse) | 183.26 ms (new) | 56.60 ms (new) |
| three_wide_or | 2.53 ms (+11236.6% worse) | 49.63 ms (new) | 3.69 GiB (+1830.8% worse) | 3.69 GiB (+1830.8% worse) | 3.69 GiB (+1830.8% worse) | 255.93 ms (new) | 84.83 ms (new) |
| three_similar_or | 10.64 ms (+34021.4% worse) | 55.88 ms (new) | 3.69 GiB (+1830.8% worse) | 3.69 GiB (+1830.8% worse) | 3.69 GiB (+1830.8% worse) | 245.21 ms (new) | 92.85 ms (new) |
| five_term_or | 18.00 ms (+14930.0% worse) | 65.25 ms (new) | 3.69 GiB (+1830.8% worse) | 3.69 GiB (+1830.8% worse) | 3.69 GiB (+1830.8% worse) | 291.98 ms (new) | 101.58 ms (new) |
| ten_term_or | 52.83 ms (+15858.8% worse) | 99.47 ms (new) | 3.69 GiB (+1830.8% worse) | 3.69 GiB (+1830.8% worse) | 3.69 GiB (+1830.8% worse) | 192.65 ms (new) | 129.15 ms (new) |

**AND queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_term_and | 563.16 µs (+2843.4% worse) | 41.27 ms (new) | 3.70 GiB (+1831.9% worse) | 3.70 GiB (+1831.9% worse) | 3.70 GiB (+1831.9% worse) | 218.88 ms (new) | 63.92 ms (new) |
| three_wide_and | 4.36 ms (+41274.3% worse) | 51.11 ms (new) | 3.70 GiB (+1831.9% worse) | 3.70 GiB (+1831.9% worse) | 3.70 GiB (+1831.9% worse) | 166.18 ms (new) | 60.71 ms (new) |
| three_similar_and | 6.27 ms (+68734.9% worse) | 51.60 ms (new) | 3.70 GiB (+1831.9% worse) | 3.70 GiB (+1831.9% worse) | 3.70 GiB (+1831.9% worse) | 133.44 ms (new) | 44.19 ms (new) |
| five_term_and | 7.55 ms (+63093.1% worse) | 54.61 ms (new) | 3.70 GiB (+1831.9% worse) | 3.70 GiB (+1831.9% worse) | 3.70 GiB (+1831.9% worse) | 195.90 ms (new) | 71.95 ms (new) |
| ten_term_and | 8.68 ms (+47771.7% worse) | 54.06 ms (new) | 3.70 GiB (+1832.9% worse) | 3.70 GiB (+1832.9% worse) | 3.70 GiB (+1832.9% worse) | 186.91 ms (new) | 70.11 ms (new) |

**Per-algorithm probes (WAND+BMW vs MaxScore+BMM)**

| Shape | WAND+BMW | MaxScore+BMM |
| --- | --- | --- |
| wide_3_or | 9.36 ms (+23817.1% worse) | 2.64 ms (+12493.9% worse) |
| similar_3_or | 15.51 ms (+29395.5% worse) | 10.58 ms (+33858.9% worse) |
| similar_5_or | 44.33 ms (+20151.9% worse) | 17.96 ms (+15427.6% worse) |
| similar_10_or | 307.01 ms (+24679.0% worse) | 52.76 ms (+16138.8% worse) |
<!-- END: bench/fts/superfile/search -->

### FTS — supertable (multi-segment, 1M docs, real S3)

<!-- BEGIN: bench/fts/supertable/ingest -->
### Supertable FTS — ingest, multi-segment / object-store (1M docs, 16 commits)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Superfiles` is the committed segment count. Δ is vs the previous run.

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| FTS-only | 22.02 s (-89.1% better) | 45.4 K/s (-8.0% worse) | 256 | 9.14 GiB (+64.9% worse) | 8.99 GiB (+291.3% worse) | 9.06 GiB (+89.5% worse) |
<!-- END: bench/fts/supertable/ingest -->

<!-- BEGIN: bench/fts/supertable/search -->
### Supertable FTS — search, multi-segment / object-store (1M docs)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Warm = shared consumer + disk cache (untimed prewarm + wait_until_warm, then per-query p50 over repeated bm25_search). Cold = fresh disk cache + consumer per iteration, so each read pays the object-store cold open. Δ is vs the previous run.

**OR queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_rare | 1.17 ms (-89.7% better) | 11.97 ms (new) | 8.94 GiB (+537.7% worse) | 8.92 GiB (+545.5% worse) | 8.94 GiB (+537.7% worse) | 423.71 ms (-37.3% better) | 125.47 ms (-88.5% better) |
| single_df1 | 55.45 µs (-98.8% better) | 2.45 ms (new) | 8.88 GiB (+531.7% worse) | 8.87 GiB (+531.3% worse) | 8.88 GiB (+531.7% worse) | 424.92 ms (-37.4% better) | 15.26 ms (-95.0% better) |
| single_common | 1.51 ms (-80.2% better) | 10.91 ms (new) | 8.96 GiB (+523.3% worse) | 8.95 GiB (+523.0% worse) | 8.96 GiB (+523.3% worse) | 417.34 ms (-34.3% better) | 386.04 ms (-69.1% better) |
| two_term_or | 1.17 ms (-90.0% better) | 10.65 ms (new) | 8.96 GiB (+347.5% worse) | 8.95 GiB (+349.0% worse) | 8.96 GiB (+347.5% worse) | 424.43 ms (-36.9% better) | 238.64 ms (-78.6% better) |
| three_wide_or | 1.53 ms (-89.8% better) | 11.57 ms (new) | 8.95 GiB (+336.7% worse) | 8.95 GiB (+349.5% worse) | 8.95 GiB (+336.7% worse) | 458.43 ms (-39.9% better) | 265.22 ms (-75.9% better) |
| three_similar_or | 2.30 ms (-89.4% better) | 11.01 ms (new) | 8.95 GiB (+330.3% worse) | 8.95 GiB (+335.6% worse) | 8.95 GiB (+331.5% worse) | 535.91 ms (-6.9% better) | 331.15 ms (-71.9% better) |
| five_term_or | 3.06 ms (-91.4% better) | 12.44 ms (new) | 8.95 GiB (+325.7% worse) | 8.95 GiB (+336.8% worse) | 8.95 GiB (+325.7% worse) | 508.54 ms (-36.7% better) | 387.59 ms (-64.0% better) |
| ten_term_or | 8.24 ms (-90.1% better) | 17.01 ms (new) | 8.95 GiB (+326.5% worse) | 8.94 GiB (+331.6% worse) | 8.94 GiB (+326.4% worse) | 427.10 ms (-38.8% better) | 404.35 ms (-60.5% better) |

**AND queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_term_and | 1.29 ms (-89.5% better) | 10.55 ms (new) | 8.95 GiB (+322.4% worse) | 8.94 GiB (+323.6% worse) | 8.95 GiB (+322.4% worse) | 495.98 ms (-19.6% better) | 345.15 ms (-66.6% better) |
| three_wide_and | 1.51 ms (-90.7% better) | 11.86 ms (new) | 8.95 GiB (+319.1% worse) | 8.94 GiB (+319.1% worse) | 8.95 GiB (+319.1% worse) | 474.04 ms (-52.0% better) | 314.13 ms (-69.3% better) |
| three_similar_and | 1.96 ms (-88.5% better) | 11.03 ms (new) | 8.95 GiB (+319.6% worse) | 8.95 GiB (+328.0% worse) | 8.95 GiB (+319.6% worse) | 432.18 ms (-33.5% better) | 277.83 ms (-82.4% better) |
| five_term_and | 2.22 ms (-87.8% better) | 11.37 ms (new) | 8.95 GiB (+320.6% worse) | 8.95 GiB (+324.7% worse) | 8.95 GiB (+322.1% worse) | 435.40 ms (-36.5% better) | 327.89 ms (-79.7% better) |
| ten_term_and | 2.36 ms (-87.8% better) | 11.59 ms (new) | 8.95 GiB (+320.5% worse) | 8.95 GiB (+320.3% worse) | 8.95 GiB (+320.5% worse) | 384.99 ms (-42.5% better) | 291.47 ms (-75.0% better) |
<!-- END: bench/fts/supertable/search -->

### Vector — superfile (single-segment, 1M × 384)

<!-- BEGIN: bench/vector/superfile/ingest -->
### Superfile vector — ingest, single-segment / in-memory (1M docs × dim=384)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Build path: `SuperfileBuilder` → unified `.parquet`, through `VectorEngine`. Rows are by writer count; `1 writer` is the canonical artifact used by correctness/search/cold upload. Δ is vs the previous run.

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 20.67 s (new) | 48.4 K/s (new) | 74.3 MB/s (new) | 4.39 GiB (new) | 2.29 GiB (new) | 3.33 GiB (new) |
| 16 writers | 2.69 s (new) | 371.1 K/s (new) | 570.0 MB/s (new) | 7.85 GiB (new) | 6.87 GiB (new) | 7.85 GiB (new) |
<!-- END: bench/vector/superfile/ingest -->

<!-- BEGIN: bench/vector/superfile/search -->
### Superfile vector — search, single-segment / in-memory (1M docs × dim=384)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Correctness, warm search, and cold upload reuse the measured 1-writer artifact. Recall rows use the lowest-p50 calibrated point meeting each target; `default` is the user-facing option baseline. Δ is vs the previous run.

| Recall target | (p, r) | recall | warm | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | p=1, r=256 | 0.962 | 994.92 µs (new) | 4.27 GiB (new) | 4.27 GiB (new) | 4.27 GiB (new) | 77.74 µs (new) | 429.93 ms (new) |
| 0.95 | p=1, r=1024 | 0.962 | 1.01 ms (new) | 4.28 GiB (new) | 4.27 GiB (new) | 4.28 GiB (new) | 58.20 µs (new) | 343.42 ms (new) |
| 0.99 | p=10, r=256 | 0.998 | 1.40 ms (new) | 4.28 GiB (new) | 4.28 GiB (new) | 4.28 GiB (new) | 64.55 µs (new) | 493.51 ms (new) |
| default | p=8, r=20 | — | 989.61 µs (new) | 4.28 GiB (new) | 4.28 GiB (new) | 4.28 GiB (new) | 73.24 µs (new) | 480.92 ms (new) |
<!-- END: bench/vector/superfile/search -->

### Vector — supertable (multi-segment, 1M × 384, real S3)

<!-- BEGIN: bench/vector/supertable/ingest -->
### Supertable vector — ingest, multi-segment / object-store (1M docs × dim=384, 16 commits)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Superfiles` is the committed segment count. Δ is vs the previous run.

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| vector-only | 29.53 s (+9.5% worse) | 33.9 K/s (-8.6% worse) | 256 | 10.71 GiB (+283.1% worse) | 9.89 GiB (+496.8% worse) | 10.60 GiB (+322.9% worse) |
<!-- END: bench/vector/supertable/ingest -->

<!-- BEGIN: bench/vector/supertable/search -->
### Supertable vector — search, multi-segment / object-store (1M docs × dim=384)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Recall rows use the lowest-p50 calibrated (p, r) clearing each target (recall vs brute-force ground truth on the regenerated corpus); `default` is the user-facing config. Warm = hot disk cache sized to the index; cold = fresh disk cache + consumer per iteration. Δ is vs the previous run.

| Recall target | (p, r) | recall | warm | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | p=5, r=1 | 0.988 | 5.17 ms (-6.2% better) | 10.39 GiB (+288.9% worse) | 10.39 GiB (+289.1% worse) | 10.39 GiB (+288.9% worse) | 2.17 s (-38.5% better) | 597.87 ms (-10.0% better) |
| 0.95 | p=5, r=1 | 0.988 | 4.86 ms (-3.6% better) | 11.19 GiB (+327.8% worse) | 11.19 GiB (+328.3% worse) | 11.19 GiB (+327.8% worse) | 1.99 s (-35.0% better) | 522.71 ms (-31.5% better) |
| 0.99 | p=10, r=1 | 0.996 | 5.23 ms (-7.4% better) | 10.75 GiB (+314.7% worse) | 10.75 GiB (+314.8% worse) | 10.75 GiB (+314.7% worse) | 1.90 s (-49.3% better) | 523.56 ms (-34.6% better) |
| default | p=8, r=20 | — | 6.72 ms (-4.0% better) | 10.77 GiB (+315.2% worse) | 10.76 GiB (+315.2% worse) | 10.77 GiB (+315.2% worse) | 1.99 s (-36.8% better) | 630.17 ms (-24.0% better) |
<!-- END: bench/vector/supertable/search -->

### Supertable — ingest summary (all shapes, real S3)

<!-- BEGIN: bench/supertable/ingest -->
### Supertable — ingest, multi-segment / object-store (1M docs, 16 commits)

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

Build path: `SupertableWriter::append` + `commit` into an in-memory supertable, through the engine-generic `run_sql` driver the cross-engine comparison also uses. Rows are by writer count: `1 writer` is the canonical build queries run against; `N writers` is the sharded parallel build. Δ is vs the previous run.

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 10.10 s (new) | 99.0 K/s (new) | 198.9 MB/s (new) | 7.32 GiB (new) | 6.43 GiB (new) | 7.12 GiB (new) |
| 16 writers | 5.17 s (new) | 193.4 K/s (new) | 388.7 MB/s (new) | 16.45 GiB (new) | 13.42 GiB (new) | 15.95 GiB (new) |
<!-- END: bench/sql/build -->

<!-- BEGIN: bench/sql/query -->
### Superfile SQL — query, single superfile / in-memory (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Warm p50 over `query_sql` against the canonical 1-writer table. The headline comparison is Plain Scan vs FTS-pushdown (same selective equality, 1 row, sorted vs unsorted column). The first block is aggregations & count-filters. `Rows` is the result-set size. Δ is vs the previous run.

**Aggregations & count-filters (read + compute, return few rows — not the index A/B)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| agg_max_title | 179.31 ms (new) | 1 | 7.92 GiB (new) | 7.89 GiB (new) | 7.92 GiB (new) |
| filter_category_count | 10.18 ms (new) | 1 | 7.38 GiB (new) | 7.38 GiB (new) | 7.38 GiB (new) |
| filter_rating_count | 7.55 ms (new) | 1 | 7.26 GiB (new) | 7.26 GiB (new) | 7.26 GiB (new) |
| count_star | 6.09 ms (new) | 1 | 7.26 GiB (new) | 7.26 GiB (new) | 7.26 GiB (new) |
| group_by_category | 7.86 ms (new) | 4 | 7.26 GiB (new) | 7.26 GiB (new) | 7.26 GiB (new) |

**Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 8.20 ms (new) | 1 | 7.48 GiB (new) | 7.46 GiB (new) | 7.48 GiB (new) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 10.01 ms (new) | 1 | 7.50 GiB (new) | 7.50 GiB (new) | 7.50 GiB (new) |

**FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 3.68 ms (new) | 1 | 7.43 GiB (new) | 7.43 GiB (new) | 7.43 GiB (new) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 1.65 ms (new) | 1 | 7.43 GiB (new) | 7.43 GiB (new) | 7.43 GiB (new) |

**Aggregate over FTS candidates — Full Scan (DataFusion only)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 10.44 ms (new) | 1 | 7.44 GiB (new) | 7.43 GiB (new) | 7.44 GiB (new) |
| SUM(rating)         key=? (1 row) | 10.25 ms (new) | 1 | 7.45 GiB (new) | 7.45 GiB (new) | 7.45 GiB (new) |
| MAX(rating)         key=? (1 row) | 11.28 ms (new) | 1 | 7.45 GiB (new) | 7.45 GiB (new) | 7.45 GiB (new) |
| AVG(rating)         key=? (1 row) | 10.18 ms (new) | 1 | 7.45 GiB (new) | 7.45 GiB (new) | 7.45 GiB (new) |
| SUM(rating) bucket IN all (1M rows) | 14.88 ms (new) | 1 | 7.45 GiB (new) | 7.45 GiB (new) | 7.45 GiB (new) |

**Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 1.89 ms (new) | 1 | 7.43 GiB (new) | 7.43 GiB (new) | 7.43 GiB (new) |
| SUM(rating)         key=? (1 row) | 2.31 ms (new) | 1 | 7.43 GiB (new) | 7.43 GiB (new) | 7.43 GiB (new) |
| MAX(rating)         key=? (1 row) | 2.32 ms (new) | 1 | 7.43 GiB (new) | 7.43 GiB (new) | 7.43 GiB (new) |
| AVG(rating)         key=? (1 row) | 2.19 ms (new) | 1 | 7.43 GiB (new) | 7.43 GiB (new) | 7.43 GiB (new) |
| SUM(rating) bucket IN all (1M rows) | 12.25 ms (new) | 1 | 7.43 GiB (new) | 7.43 GiB (new) | 7.43 GiB (new) |

**Search table functions (bm25 / vector / hybrid / token / exact)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| bm25_search | 1.11 ms (new) | 10 | 7.26 GiB (new) | 7.26 GiB (new) | 7.26 GiB (new) |
| vector_search | 1.36 ms (new) | 10 | 7.26 GiB (new) | 7.26 GiB (new) | 7.26 GiB (new) |
| hybrid_search | 1.37 ms (new) | 10 | 7.26 GiB (new) | 7.26 GiB (new) | 7.26 GiB (new) |
| token_match (all rows) | 89.99 ms (new) | 1000.0K | 7.58 GiB (new) | 7.57 GiB (new) | 7.58 GiB (new) |
| token_match (selective) | 173.52 µs (new) | 1 | 7.46 GiB (new) | 7.46 GiB (new) | 7.46 GiB (new) |
| exact_match | 2.80 ms (new) | 1 | 7.47 GiB (new) | 7.46 GiB (new) | 7.47 GiB (new) |
<!-- END: bench/sql/query -->

<!-- BEGIN: bench/sql/superfile/cold -->
### Superfile SQL — cold query, object-store (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Cold p50 over `reader().query_sql` after reopening the same SQL table shape from object storage with a fresh disk cache per iteration. Δ is vs the previous run.

| Query | cold open | cold search |
| --- | --- | --- |
| agg_max_title | 369.56 ms (new) | 1.72 s (new) |
| filter_category_count | 249.79 ms (new) | 267.76 ms (new) |
| filter_rating_count | 275.04 ms (new) | 251.10 ms (new) |
| count_star | 366.15 ms (new) | 26.41 ms (new) |
| group_by_category | 383.92 ms (new) | 145.41 ms (new) |
<!-- END: bench/sql/superfile/cold -->

### SQL — supertable (multi-segment, 1M rows, real S3)

<!-- BEGIN: bench/sql/supertable/ingest -->
### Supertable SQL — ingest, multi-segment / object-store (1M rows, 16 commits)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Superfiles` is the committed segment count. Δ is vs the previous run.

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| SQL | 40.94 s (-90.0% better) | 24.4 K/s (-0.3% ~) | 256 | 10.23 GiB (+6.9% worse) | 9.57 GiB (+61.3% worse) | 9.90 GiB (+21.7% worse) |
<!-- END: bench/sql/supertable/ingest -->

<!-- BEGIN: bench/sql/supertable/warm -->
### Supertable SQL — warm queries, warm cache / object-store (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Warm = committed table reopened with a disk cache sized to the index; p50 over repeated `query_sql` calls. The headline comparison is Plain Scan vs FTS-pushdown (same selective equality). Δ is vs the previous run.

**Aggregations & count-filters (read + compute, return few rows — not the index A/B)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| agg_max_title | 179.53 ms (-89.1% better) | 1 | 11.48 GiB (+115.1% worse) | 11.46 GiB (+123.8% worse) | 11.48 GiB (+117.1% worse) |
| filter_category_count | 23.80 ms (-74.3% better) | 1 | 11.21 GiB (+135.7% worse) | 11.21 GiB (+136.7% worse) | 11.21 GiB (+135.7% worse) |
| filter_rating_count | 21.12 ms (-58.4% better) | 1 | 11.13 GiB (+135.3% worse) | 11.13 GiB (+136.7% worse) | 11.13 GiB (+135.3% worse) |
| count_star | 20.96 ms (-53.4% better) | 1 | 11.13 GiB (+136.8% worse) | 11.13 GiB (+136.7% worse) | 11.13 GiB (+136.8% worse) |
| group_by_category | 20.78 ms (-67.2% better) | 4 | 11.13 GiB (+141.2% worse) | 11.13 GiB (+141.2% worse) | 11.13 GiB (+141.2% worse) |

**Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 7.90 ms (-9.8% better) | 1 | 11.64 GiB (+91.8% worse) | 11.64 GiB (+91.7% worse) | 11.64 GiB (+91.8% worse) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 23.65 ms (-73.8% better) | 1 | 11.64 GiB (+77.4% worse) | 11.64 GiB (+86.0% worse) | 11.64 GiB (+77.4% worse) |

**FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 4.42 ms (+16.3% worse) | 1 | 11.59 GiB (+105.8% worse) | 11.59 GiB (+105.7% worse) | 11.59 GiB (+105.8% worse) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 1.60 ms (-29.2% better) | 1 | 11.59 GiB (+105.7% worse) | 11.59 GiB (+105.7% worse) | 11.59 GiB (+105.7% worse) |

**Aggregate over FTS candidates — Full Scan (DataFusion only)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 23.17 ms (-74.3% better) | 1 | 11.59 GiB (+105.7% worse) | 11.59 GiB (+106.0% worse) | 11.59 GiB (+105.7% worse) |
| SUM(rating)         key=? (1 row) | 23.49 ms (-73.9% better) | 1 | 11.59 GiB (+105.7% worse) | 11.58 GiB (+105.7% worse) | 11.59 GiB (+105.7% worse) |
| MAX(rating)         key=? (1 row) | 24.36 ms (-74.6% better) | 1 | 11.59 GiB (+105.0% worse) | 11.59 GiB (+105.1% worse) | 11.59 GiB (+105.0% worse) |
| AVG(rating)         key=? (1 row) | 23.45 ms (-73.9% better) | 1 | 11.58 GiB (+104.9% worse) | 11.58 GiB (+105.2% worse) | 11.58 GiB (+104.9% worse) |
| SUM(rating) bucket IN all (1M rows) | 30.69 ms (-69.0% better) | 1 | 11.58 GiB (+105.3% worse) | 11.58 GiB (+106.4% worse) | 11.58 GiB (+105.3% worse) |

**Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 1.68 ms (-35.5% better) | 1 | 11.58 GiB (+107.1% worse) | 11.58 GiB (+107.2% worse) | 11.58 GiB (+107.1% worse) |
| SUM(rating)         key=? (1 row) | 1.88 ms (-43.5% better) | 1 | 11.58 GiB (+107.1% worse) | 11.58 GiB (+107.1% worse) | 11.58 GiB (+107.1% worse) |
| MAX(rating)         key=? (1 row) | 2.08 ms (-38.2% better) | 1 | 11.58 GiB (+107.1% worse) | 11.58 GiB (+107.1% worse) | 11.58 GiB (+107.1% worse) |
| AVG(rating)         key=? (1 row) | 2.00 ms (-42.4% better) | 1 | 11.58 GiB (+107.1% worse) | 11.58 GiB (+107.1% worse) | 11.58 GiB (+107.1% worse) |
| SUM(rating) bucket IN all (1M rows) | 66.25 ms (-11.5% better) | 1 | 11.58 GiB (+107.0% worse) | 11.58 GiB (+107.1% worse) | 11.58 GiB (+107.0% worse) |

**Search table functions (bm25 / vector / hybrid / token / exact)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| bm25_search | 2.41 ms (-65.4% better) | 10 | 11.15 GiB (+138.1% worse) | 11.13 GiB (+141.2% worse) | 11.15 GiB (+138.1% worse) |
| vector_search | 3.56 ms (-65.3% better) | 10 | 11.17 GiB (+135.0% worse) | 11.15 GiB (+134.6% worse) | 11.17 GiB (+135.0% worse) |
| hybrid_search | 3.83 ms (-71.8% better) | 10 | 11.17 GiB (+135.1% worse) | 11.16 GiB (+135.0% worse) | 11.17 GiB (+135.1% worse) |
| token_match (all rows) | 124.17 ms (-88.6% better) | 1000.0K | 11.73 GiB (+88.5% worse) | 11.71 GiB (+89.1% worse) | 11.72 GiB (+88.9% worse) |
| token_match (selective) | 364.42 µs (-68.4% better) | 1 | 11.64 GiB (+92.1% worse) | 11.64 GiB (+92.1% worse) | 11.64 GiB (+92.1% worse) |
| exact_match | 3.03 ms (-17.8% better) | 1 | 11.64 GiB (+91.8% worse) | 11.64 GiB (+92.1% worse) | 11.64 GiB (+91.8% worse) |
<!-- END: bench/sql/supertable/warm -->

<!-- BEGIN: bench/sql/supertable/cold -->
### Supertable SQL — cold queries, fresh cache / object-store (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Cold = fresh disk cache + consumer per iteration, so each query pays the object-store cold open. Δ is vs the previous run.

| Query | cold open | cold search |
| --- | --- | --- |
| agg_max_title | 1.00 s (-53.6% better) | 1.30 s (-80.3% better) |
| filter_category_count | 977.62 ms (-57.8% better) | 1.19 s (-21.6% better) |
| filter_rating_count | 992.82 ms (-64.5% better) | 1.27 s (+2.6% ~) |
| count_star | 907.78 ms (-68.1% better) | 105.09 ms (-70.7% better) |
| group_by_category | 1.18 s (-53.6% better) | 973.06 ms (-11.9% better) |
<!-- END: bench/sql/supertable/cold -->
