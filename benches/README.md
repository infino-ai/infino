# infino benches

Infino's in-tree benchmarks measure Infino itself. Cross-engine comparison
benches live in a separate cross-engine comparison harness; these tables are the Infino reference numbers
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

## Vector search tuning

The vector benches calibrate each recall target by sweeping a probe/refine
grid, then report a user-facing `default` row. Three knobs control that row
and let you skip the sweep:

- `INFINO_BENCH_VECTOR_NPROBE` — probe count for the `default` row (default 8).
- `INFINO_BENCH_VECTOR_RERANK` — rerank multiplier for the `default` row
  (default 20).
- `INFINO_BENCH_SKIP_CALIBRATION=1` — measure **only** the fixed
  `(nprobe, rerank)` `default` row: skips the correctness gate, the
  recall-target calibration sweep, and brute-force ground-truth generation.
  This is the fast path for a fixed-config **cold-only** latency number on a
  many-segment supertable, where sweeping the full grid over a cold table is
  prohibitively slow.
- `INFINO_BENCH_PREFETCH_CONCURRENCY` — disk-cache prefetch fan-out for the
  cold-fill / promotion path on many-segment tables (default 8).

```sh
# Fast fixed-config cold vector latency (no calibration sweep):
INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=my-bucket INFINO_BENCH_SKIP_CALIBRATION=1 \
  INFINO_BENCH_VECTOR_NPROBE=8 INFINO_BENCH_VECTOR_RERANK=4 cargo bench -- supertable vector cold
```

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

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Build path: `SuperfileBuilder` → unified `.parquet` (same as production supertable commit), through the engine-generic `run_fts` driver the cross-engine comparison also uses. Rows are by writer count: `1 writer` is the single-threaded build (and the index queries run against); `N writers` is the sharded parallel build. Bandwidth is over the logical input text payload. Δ is vs the previous run.

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 17.04 s (+1.5% ~) | 58.7 K/s (-1.5% ~) | 118.0 MB/s (-1.5% ~) | 5.79 GiB (+0.2% ~) | 3.78 GiB (+0.7% ~) | 4.81 GiB (-1.3% ~) |
| 16 writers | 2.11 s (-0.1% ~) | 473.7 K/s (+0.1% ~) | 952.2 MB/s (+0.1% ~) | 8.01 GiB (+1.3% ~) | 7.21 GiB (+2.1% ~) | 7.64 GiB (+0.3% ~) |
<!-- END: bench/fts/superfile/ingest -->

<!-- BEGIN: bench/fts/superfile/search -->
### Superfile FTS — search, single-superfile / in-memory (1M docs)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Warm = `SuperfileReader::open` in memory (per-query p50); cold = same `.parquet` on object storage via `DiskCacheStore::reader` -> `bm25_search` (production cold path). Δ is vs the previous run.

**OR queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_rare | 6.28 µs (+571.9% worse) | 10.64 ms (+19.0% worse) | 3.67 GiB (+904.3% worse) | 3.67 GiB (+904.3% worse) | 3.67 GiB (+904.3% worse) | 164.60 ms (-10.6% better) | 28.27 ms (+1.6% ~) |
| single_df1 | 623 ns (+76.5% worse) | 17.26 ms (+4989028.0% worse) | 3.68 GiB (+909.7% worse) | 3.68 GiB (+909.7% worse) | 3.68 GiB (+909.7% worse) | 189.65 ms (+14.1% worse) | 11.13 µs (-1.1% ~) |
| single_common | 2.00 ms (+4951.9% worse) | 42.85 ms (+196.2% worse) | 3.68 GiB (+904.3% worse) | 3.68 GiB (+904.3% worse) | 3.68 GiB (+904.3% worse) | 171.75 ms (+25.5% worse) | 58.68 ms (+31.9% worse) |
| two_term_or | 226.83 µs (+920.1% worse) | 40.82 ms (+155.0% worse) | 3.68 GiB (+906.0% worse) | 3.68 GiB (+906.0% worse) | 3.68 GiB (+906.0% worse) | 218.66 ms (+19.3% worse) | 57.07 ms (+0.8% ~) |
| three_wide_or | 2.44 ms (+4285.9% worse) | 48.86 ms (+218.7% worse) | 3.68 GiB (+902.2% worse) | 3.68 GiB (+902.2% worse) | 3.68 GiB (+902.2% worse) | 190.61 ms (-25.5% better) | 58.79 ms (-30.7% better) |
| three_similar_or | 10.36 ms (+4975.0% worse) | 55.02 ms (+278.0% worse) | 3.68 GiB (+901.2% worse) | 3.68 GiB (+901.2% worse) | 3.68 GiB (+901.2% worse) | 218.08 ms (-11.1% better) | 52.64 ms (-43.3% better) |
| five_term_or | 17.77 ms (+3393.8% worse) | 64.40 ms (+305.3% worse) | 3.68 GiB (+901.1% worse) | 3.68 GiB (+901.1% worse) | 3.68 GiB (+901.1% worse) | 200.83 ms (-31.2% better) | 59.16 ms (-41.8% better) |
| ten_term_or | 52.39 ms (+3453.8% worse) | 98.39 ms (+517.7% worse) | 3.68 GiB (+901.2% worse) | 3.68 GiB (+901.2% worse) | 3.68 GiB (+901.2% worse) | 204.48 ms (+6.1% worse) | 112.53 ms (-12.9% better) |

**AND queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_term_and | 523.26 µs (+587.3% worse) | 41.28 ms (+155.9% worse) | 3.69 GiB (+902.7% worse) | 3.69 GiB (+902.7% worse) | 3.69 GiB (+902.7% worse) | 222.91 ms (+1.8% ~) | 56.13 ms (-12.2% better) |
| three_wide_and | 4.28 ms (+5460.9% worse) | 50.67 ms (+229.2% worse) | 3.69 GiB (+902.6% worse) | 3.69 GiB (+902.6% worse) | 3.69 GiB (+902.6% worse) | 191.48 ms (+15.2% worse) | 59.73 ms (-1.6% ~) |
| three_similar_and | 6.09 ms (+6556.7% worse) | 50.77 ms (+253.1% worse) | 3.68 GiB (+901.5% worse) | 3.68 GiB (+901.5% worse) | 3.68 GiB (+901.5% worse) | 171.49 ms (+28.5% worse) | 55.71 ms (+26.1% worse) |
| five_term_and | 7.49 ms (+6574.1% worse) | 54.07 ms (+252.7% worse) | 3.68 GiB (+901.5% worse) | 3.68 GiB (+901.5% worse) | 3.68 GiB (+901.5% worse) | 201.46 ms (+2.8% ~) | 71.01 ms (-1.3% ~) |
| ten_term_and | 8.65 ms (+6255.5% worse) | 53.39 ms (+40411.9% worse) | 3.68 GiB (+906.9% worse) | 3.68 GiB (+906.9% worse) | 3.68 GiB (+906.9% worse) | 256.80 ms (+37.4% worse) | 93.38 ms (+33.2% worse) |

**Per-algorithm probes (WAND+BMW vs MaxScore+BMM)**

| Shape | WAND+BMW | MaxScore+BMM |
| --- | --- | --- |
| wide_3_or | 9.23 ms (+4018.3% worse) | 2.47 ms (+4459.1% worse) |
| similar_3_or | 15.04 ms (+4955.3% worse) | 10.30 ms (+5284.1% worse) |
| similar_5_or | 44.07 ms (+3900.9% worse) | 17.80 ms (+3415.0% worse) |
| similar_10_or | 302.85 ms (+4488.4% worse) | 52.55 ms (+3420.0% worse) |
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

Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Superfiles` is the committed segment count. Δ is vs the previous run.

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| FTS-only | 25.87 s (+10.7% worse) | 38.7 K/s (-9.6% worse) | 256 | 1.31 GiB (+4.4% worse) | 1.10 GiB (+0.2% ~) | 1.23 GiB (+3.5% worse) |
<!-- END: bench/fts/supertable/ingest -->

<!-- BEGIN: bench/fts/supertable/search -->
### Supertable FTS — search, multi-superfile / object-store (1M docs)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Warm = shared consumer + disk cache (untimed prewarm + wait_until_warm, then per-query p50 over repeated bm25_search). Cold = fresh disk cache + consumer per iteration, so each read pays the object-store cold open. Δ is vs the previous run.

**OR queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_rare | 1.14 ms (+7.7% worse) | 8.93 ms (+0.2% ~) | 942.10 MiB (+0.5% ~) | 932.93 MiB (+0.2% ~) | 942.10 MiB (+0.5% ~) | 556.11 ms (+31.2% worse) | 272.69 ms (+117.3% worse) |
| single_df1 | 55.22 µs (-88.7% better) | 2.65 ms (-8.0% better) | 906.39 MiB (-0.2% ~) | 904.30 MiB (+0.5% ~) | 906.39 MiB (-0.2% ~) | 436.74 ms (+2.8% ~) | 16.64 ms (+9.0% worse) |
| single_common | 1.31 ms (+5.5% worse) | 10.55 ms (-0.6% ~) | 1.15 GiB (+5.6% worse) | 1.07 GiB (+8.4% worse) | 1.15 GiB (+5.6% worse) | 444.48 ms (+6.5% worse) | 360.95 ms (-6.5% better) |
| two_term_or | 1.16 ms (+11.3% worse) | 10.54 ms (+2.0% ~) | 1.22 GiB (+13.7% worse) | 1.11 GiB (+12.2% worse) | 1.22 GiB (+13.7% worse) | 440.47 ms (+3.8% worse) | 264.00 ms (+10.6% worse) |
| three_wide_or | 1.29 ms (+3.6% worse) | 11.90 ms (+6.6% worse) | 1.20 GiB (+7.7% worse) | 1.09 GiB (+3.3% worse) | 1.20 GiB (+7.7% worse) | 423.31 ms (-7.7% better) | 361.95 ms (+36.5% worse) |
| three_similar_or | 2.23 ms (+3.0% worse) | 10.79 ms (-1.4% ~) | 1.11 GiB (+1.8% ~) | 1013.32 MiB (-2.5% ~) | 1.11 GiB (+1.8% ~) | 505.38 ms (-5.7% better) | 397.19 ms (+19.9% worse) |
| five_term_or | 3.39 ms (+8.2% worse) | 12.49 ms (+4.2% worse) | 1.14 GiB (+4.7% worse) | 1.06 GiB (+2.7% ~) | 1.14 GiB (+4.7% worse) | 474.28 ms (-6.7% better) | 302.61 ms (-21.9% better) |
| ten_term_or | 7.93 ms (-2.5% ~) | 16.80 ms (+2.5% ~) | 1.18 GiB (+5.1% worse) | 1.07 GiB (+3.5% worse) | 1.18 GiB (+5.1% worse) | 417.63 ms (-2.2% ~) | 608.07 ms (+50.4% worse) |

**AND queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_term_and | 1.14 ms (-12.7% better) | 10.09 ms (-2.5% ~) | 1.10 GiB (+1.4% ~) | 1.05 GiB (+2.9% ~) | 1.10 GiB (+1.4% ~) | 488.90 ms (-1.4% ~) | 239.08 ms (-30.7% better) |
| three_wide_and | 1.40 ms (-11.5% better) | 11.72 ms (-1.3% ~) | 1.16 GiB (+8.2% worse) | 1.03 GiB (+0.4% ~) | 1.16 GiB (+8.2% worse) | 454.35 ms (-4.2% better) | 205.77 ms (-34.5% better) |
| three_similar_and | 1.88 ms (-8.0% better) | 10.90 ms (+0.6% ~) | 1.18 GiB (+11.5% worse) | 1.04 GiB (+3.9% worse) | 1.18 GiB (+11.5% worse) | 441.45 ms (+2.1% ~) | 203.19 ms (-26.9% better) |
| five_term_and | 2.13 ms (+1.6% ~) | 11.33 ms (+0.3% ~) | 1.17 GiB (+6.8% worse) | 1.07 GiB (+4.4% worse) | 1.17 GiB (+6.8% worse) | 453.28 ms (+4.1% worse) | 343.62 ms (+4.8% worse) |
| ten_term_and | 2.43 ms (-3.9% better) | 11.29 ms (-3.1% better) | 1.14 GiB (+1.4% ~) | 1.06 GiB (+6.1% worse) | 1.14 GiB (+1.4% ~) | 392.86 ms (+2.0% ~) | 304.12 ms (+4.3% worse) |
<!-- END: bench/fts/supertable/search -->

### Vector — superfile (single-superfile, 1M × 384)

<!-- BEGIN: bench/vector/superfile/ingest -->
### Superfile vector — ingest, single-superfile / in-memory (1M docs × dim=384)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Build path: `SuperfileBuilder` → unified `.parquet`, through `VectorEngine`. Rows are by writer count; `1 writer` is the canonical artifact used by correctness/search/cold upload. Δ is vs the previous run.

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 20.61 s (-0.3% ~) | 48.5 K/s (+0.3% ~) | 74.5 MB/s (+0.3% ~) | 3.87 GiB (-12.0% better) | 1.74 GiB (-24.0% better) | 2.79 GiB (-16.2% better) |
| 16 writers | 2.66 s (-1.5% ~) | 376.6 K/s (+1.5% ~) | 578.5 MB/s (+1.5% ~) | 6.93 GiB (-11.7% better) | 6.85 GiB (-0.2% ~) | 6.93 GiB (-11.7% better) |
<!-- END: bench/vector/superfile/ingest -->

<!-- BEGIN: bench/vector/superfile/search -->
### Superfile vector — search, single-superfile / in-memory (1M docs × dim=384)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Correctness, warm search, and cold upload reuse the measured 1-writer artifact. Recall rows use the lowest-p50 calibrated point meeting each target; `default` is the user-facing option baseline. Δ is vs the previous run.

| Recall target | (p, r) | recall | warm | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | p=1, r=256 | 0.962 | 1.04 ms (+1534.5% worse) | 4.11 GiB (+1779.5% worse) | 4.11 GiB (+1778.6% worse) | 4.11 GiB (+1779.5% worse) | 218.78 ms (+281331.1% worse) | 354.58 ms (-17.5% better) |
| 0.95 | p=1, r=256 | 0.962 | 1.01 ms (+494.1% worse) | 4.12 GiB (+1784.4% worse) | 4.12 GiB (+1783.5% worse) | 4.12 GiB (+1784.4% worse) | 179.32 ms (+308040.5% worse) | 215.49 ms (-37.3% better) |
| 0.99 | p=5, r=256 | 0.998 | 1.55 ms (+815.2% worse) | 4.13 GiB (+1785.4% worse) | 4.12 GiB (+1783.5% worse) | 4.13 GiB (+1785.4% worse) | 384.58 ms (+595665.8% worse) | 508.34 ms (+3.0% worse) |
| default | p=8, r=20 | — | 947.29 µs (+871.6% worse) | 4.13 GiB (+1786.9% worse) | 4.13 GiB (+1786.0% worse) | 4.13 GiB (+1786.9% worse) | 353.10 ms (+481996.7% worse) | 372.97 ms (-22.4% better) |
<!-- END: bench/vector/superfile/search -->

### Vector — supertable (multi-superfile, 1M × 384, real S3)

<!-- BEGIN: bench/vector/supertable/ingest -->
### Supertable vector — ingest, multi-superfile / object-store (1M docs × dim=384, 16 commits)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Superfiles` is the committed segment count. Δ is vs the previous run.

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| vector-only | 24.66 s (-16.5% better) | 40.6 K/s (+19.7% better) | 256 | 2.62 GiB (-75.6% better) | 1.91 GiB (-80.7% better) | 2.50 GiB (-76.4% better) |
<!-- END: bench/vector/supertable/ingest -->

<!-- BEGIN: bench/vector/supertable/search -->
### Supertable vector — search, multi-superfile / object-store (1M docs × dim=384)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Recall rows use the lowest-p50 calibrated (p, r) clearing each target (recall vs brute-force ground truth on the regenerated corpus); `default` is the user-facing config. Warm = hot disk cache sized to the index; cold = fresh disk cache + consumer per iteration. Δ is vs the previous run.

| Recall target | (p, r) | recall | warm | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | p=5, r=1 | 0.988 | 5.13 ms (-0.8% ~) | 2.64 GiB (-74.6% better) | 2.64 GiB (-74.6% better) | 2.64 GiB (-74.6% better) | 2.19 s (+1.2% ~) | 604.99 ms (+1.2% ~) |
| 0.95 | p=5, r=1 | 0.988 | 4.35 ms (-10.4% better) | 3.75 GiB (-66.4% better) | 3.75 GiB (-66.5% better) | 3.75 GiB (-66.4% better) | 2.00 s (+0.6% ~) | 429.61 ms (-17.8% better) |
| 0.99 | p=10, r=1 | 0.996 | 5.29 ms (+1.0% ~) | 3.17 GiB (-70.5% better) | 3.16 GiB (-70.6% better) | 3.17 GiB (-70.5% better) | 1.89 s (-0.5% ~) | 466.60 ms (-10.9% better) |
| default | p=8, r=20 | — | 6.82 ms (+1.5% ~) | 3.46 GiB (-67.9% better) | 3.45 GiB (-67.9% better) | 3.46 GiB (-67.9% better) | 1.96 s (-1.5% ~) | 585.76 ms (-7.0% better) |
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

Build path: `SupertableWriter::append` + `commit` into an in-memory supertable, through the engine-generic `run_sql` driver the cross-engine comparison also uses. Rows are by writer count: `1 writer` is the canonical build queries run against; `N writers` is the sharded parallel build. Δ is vs the previous run.

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 10.10 s (-0.0% ~) | 99.0 K/s (+0.0% ~) | 199.0 MB/s (+0.0% ~) | 4.88 GiB (-33.4% better) | 3.85 GiB (-40.1% better) | 4.61 GiB (-35.3% better) |
| 16 writers | 5.30 s (+2.5% ~) | 188.6 K/s (-2.5% ~) | 379.1 MB/s (-2.5% ~) | 14.08 GiB (-14.4% better) | 10.77 GiB (-19.7% better) | 13.42 GiB (-15.9% better) |
<!-- END: bench/sql/build -->

<!-- BEGIN: bench/sql/query -->
### Superfile SQL — query, single superfile / in-memory (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Warm p50 over `query_sql` against the canonical 1-writer table. The headline comparison is Plain Scan vs FTS-pushdown (same selective equality, 1 row, sorted vs unsorted column). The first block is aggregations & count-filters. `Rows` is the result-set size. Δ is vs the previous run.

**Aggregations & count-filters (read + compute, return few rows — not the index A/B)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| agg_max_title | 180.15 ms (+0.5% ~) | 1 | 5.63 GiB (-28.9% better) | 5.51 GiB (-30.1% better) | 5.57 GiB (-29.7% better) |
| filter_category_count | 10.08 ms (-1.0% ~) | 1 | 4.92 GiB (-33.4% better) | 4.92 GiB (-33.4% better) | 4.92 GiB (-33.4% better) |
| filter_rating_count | 7.50 ms (-0.6% ~) | 1 | 4.79 GiB (-34.1% better) | 4.79 GiB (-34.1% better) | 4.79 GiB (-34.1% better) |
| count_star | 6.39 ms (+4.9% worse) | 1 | 4.79 GiB (-34.1% better) | 4.78 GiB (-34.2% better) | 4.79 GiB (-34.1% better) |
| group_by_category | 7.62 ms (-3.0% ~) | 4 | 4.79 GiB (-34.1% better) | 4.78 GiB (-34.1% better) | 4.79 GiB (-34.1% better) |

**Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 7.76 ms (-5.3% better) | 1 | 5.02 GiB (-32.9% better) | 5.01 GiB (-32.9% better) | 5.02 GiB (-32.9% better) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 9.76 ms (-2.5% ~) | 1 | 5.05 GiB (-32.7% better) | 5.05 GiB (-32.7% better) | 5.05 GiB (-32.7% better) |

**FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 3.98 ms (+8.3% worse) | 1 | 4.97 GiB (-33.2% better) | 4.97 GiB (-33.2% better) | 4.97 GiB (-33.2% better) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 1.70 ms (+2.7% ~) | 1 | 4.97 GiB (-33.2% better) | 4.96 GiB (-33.2% better) | 4.97 GiB (-33.2% better) |

**Aggregate over FTS candidates — Full Scan (DataFusion only)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 9.89 ms (-5.3% better) | 1 | 4.97 GiB (-33.2% better) | 4.97 GiB (-33.2% better) | 4.97 GiB (-33.2% better) |
| SUM(rating)         key=? (1 row) | 10.63 ms (+3.7% worse) | 1 | 4.97 GiB (-33.3% better) | 4.97 GiB (-33.3% better) | 4.97 GiB (-33.3% better) |
| MAX(rating)         key=? (1 row) | 11.12 ms (-1.4% ~) | 1 | 4.97 GiB (-33.4% better) | 4.97 GiB (-33.3% better) | 4.97 GiB (-33.4% better) |
| AVG(rating)         key=? (1 row) | 10.05 ms (-1.2% ~) | 1 | 4.97 GiB (-33.3% better) | 4.97 GiB (-33.3% better) | 4.97 GiB (-33.3% better) |
| SUM(rating) bucket IN all (1M rows) | 14.58 ms (-2.0% ~) | 1 | 4.97 GiB (-33.3% better) | 4.97 GiB (-33.3% better) | 4.97 GiB (-33.3% better) |

**Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 1.95 ms (+3.4% worse) | 1 | 4.93 GiB (-33.6% better) | 4.93 GiB (-33.6% better) | 4.93 GiB (-33.6% better) |
| SUM(rating)         key=? (1 row) | 2.17 ms (-6.1% better) | 1 | 4.94 GiB (-33.6% better) | 4.93 GiB (-33.6% better) | 4.94 GiB (-33.6% better) |
| MAX(rating)         key=? (1 row) | 2.09 ms (-9.9% better) | 1 | 4.94 GiB (-33.6% better) | 4.94 GiB (-33.6% better) | 4.94 GiB (-33.6% better) |
| AVG(rating)         key=? (1 row) | 1.87 ms (-14.4% better) | 1 | 4.94 GiB (-33.6% better) | 4.94 GiB (-33.6% better) | 4.94 GiB (-33.6% better) |
| SUM(rating) bucket IN all (1M rows) | 12.06 ms (-1.5% ~) | 1 | 4.94 GiB (-33.5% better) | 4.94 GiB (-33.6% better) | 4.94 GiB (-33.5% better) |

**Search table functions (bm25 / vector / hybrid / token / exact)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| bm25_search | 913.73 µs (-17.7% better) | 10 | 4.79 GiB (-34.1% better) | 4.79 GiB (-34.1% better) | 4.79 GiB (-34.1% better) |
| vector_search | 1.31 ms (-3.9% better) | 10 | 4.79 GiB (-34.1% better) | 4.79 GiB (-34.1% better) | 4.79 GiB (-34.1% better) |
| hybrid_search | 1.26 ms (-7.4% better) | 10 | 4.79 GiB (-34.1% better) | 4.79 GiB (-34.1% better) | 4.79 GiB (-34.1% better) |
| token_match (all rows) | 67.83 ms (-24.6% better) | 1000.0K | 5.12 GiB (-32.4% better) | 5.09 GiB (-32.7% better) | 5.12 GiB (-32.4% better) |
| token_match (selective) | 256.05 µs (+47.6% worse) | 1 | 5.00 GiB (-33.0% better) | 5.00 GiB (-33.0% better) | 5.00 GiB (-33.0% better) |
| exact_match | 2.84 ms (+1.3% ~) | 1 | 5.01 GiB (-32.9% better) | 5.00 GiB (-32.9% better) | 5.01 GiB (-32.9% better) |
<!-- END: bench/sql/query -->

<!-- BEGIN: bench/sql/superfile/cold -->
### Superfile SQL — cold query, object-store (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Cold p50 over `reader().query_sql` after reopening the same SQL table shape from object storage with a fresh disk cache per iteration. Δ is vs the previous run.

| Query | cold open | cold search |
| --- | --- | --- |
| agg_max_title | 277.21 ms (-25.0% better) | 1.93 s (+12.3% worse) |
| filter_category_count | 273.12 ms (+9.3% worse) | 268.43 ms (+0.3% ~) |
| filter_rating_count | 251.27 ms (-8.6% better) | 394.90 ms (+57.3% worse) |
| count_star | 495.55 ms (+35.3% worse) | 69.88 ms (+164.6% worse) |
| group_by_category | 245.43 ms (-36.1% better) | 174.96 ms (+20.3% worse) |
<!-- END: bench/sql/superfile/cold -->

### SQL — supertable (multi-superfile, 1M rows, real S3)

<!-- BEGIN: bench/sql/supertable/ingest -->
### Supertable SQL — ingest, multi-superfile / object-store (1M rows, 16 commits)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Superfiles` is the committed segment count. Δ is vs the previous run.

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| SQL | 41.30 s (+0.9% ~) | 24.2 K/s (-0.9% ~) | 256 | 2.08 GiB (-79.7% better) | 1.57 GiB (-83.6% better) | 1.95 GiB (-80.4% better) |
<!-- END: bench/sql/supertable/ingest -->

<!-- BEGIN: bench/sql/supertable/warm -->
### Supertable SQL — warm queries, warm cache / object-store (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Warm = committed table reopened with a disk cache sized to the index; p50 over repeated `query_sql` calls. The headline comparison is Plain Scan vs FTS-pushdown (same selective equality). Δ is vs the previous run.

**Aggregations & count-filters (read + compute, return few rows — not the index A/B)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| agg_max_title | 176.28 ms (-1.8% ~) | 1 | 2.91 GiB (-74.6% better) | 2.83 GiB (-75.3% better) | 2.91 GiB (-74.7% better) |
| filter_category_count | 22.67 ms (-4.7% better) | 1 | 2.44 GiB (-78.3% better) | 2.44 GiB (-78.3% better) | 2.44 GiB (-78.3% better) |
| filter_rating_count | 20.08 ms (-4.9% better) | 1 | 2.30 GiB (-79.3% better) | 2.30 GiB (-79.4% better) | 2.30 GiB (-79.3% better) |
| count_star | 19.58 ms (-6.6% better) | 1 | 2.30 GiB (-79.3% better) | 2.30 GiB (-79.3% better) | 2.30 GiB (-79.3% better) |
| group_by_category | 21.23 ms (+2.2% ~) | 4 | 2.30 GiB (-79.3% better) | 2.30 GiB (-79.3% better) | 2.30 GiB (-79.3% better) |

**Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 7.52 ms (-4.7% better) | 1 | 2.84 GiB (-75.6% better) | 2.83 GiB (-75.7% better) | 2.84 GiB (-75.6% better) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 21.90 ms (-7.4% better) | 1 | 2.84 GiB (-75.6% better) | 2.84 GiB (-75.6% better) | 2.84 GiB (-75.6% better) |

**FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 4.23 ms (-4.4% better) | 1 | 2.77 GiB (-76.1% better) | 2.76 GiB (-76.2% better) | 2.77 GiB (-76.1% better) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 1.44 ms (-9.6% better) | 1 | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) |

**Aggregate over FTS candidates — Full Scan (DataFusion only)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 22.55 ms (-2.7% ~) | 1 | 2.77 GiB (-76.1% better) | 2.76 GiB (-76.2% better) | 2.77 GiB (-76.1% better) |
| SUM(rating)         key=? (1 row) | 22.95 ms (-2.3% ~) | 1 | 2.77 GiB (-76.1% better) | 2.76 GiB (-76.1% better) | 2.77 GiB (-76.1% better) |
| MAX(rating)         key=? (1 row) | 23.81 ms (-2.3% ~) | 1 | 2.77 GiB (-76.1% better) | 2.77 GiB (-76.1% better) | 2.77 GiB (-76.1% better) |
| AVG(rating)         key=? (1 row) | 22.56 ms (-3.8% better) | 1 | 2.77 GiB (-76.1% better) | 2.76 GiB (-76.1% better) | 2.77 GiB (-76.1% better) |
| SUM(rating) bucket IN all (1M rows) | 30.54 ms (-0.5% ~) | 1 | 2.77 GiB (-76.1% better) | 2.77 GiB (-76.1% better) | 2.77 GiB (-76.1% better) |

**Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 1.69 ms (+0.7% ~) | 1 | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) |
| SUM(rating)         key=? (1 row) | 1.94 ms (+3.3% worse) | 1 | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) |
| MAX(rating)         key=? (1 row) | 1.85 ms (-11.0% better) | 1 | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) |
| AVG(rating)         key=? (1 row) | 1.84 ms (-8.0% better) | 1 | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) |
| SUM(rating) bucket IN all (1M rows) | 66.05 ms (-0.3% ~) | 1 | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) | 2.76 GiB (-76.2% better) |

**Search table functions (bm25 / vector / hybrid / token / exact)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| bm25_search | 2.42 ms (+0.2% ~) | 10 | 2.34 GiB (-79.0% better) | 2.30 GiB (-79.4% better) | 2.34 GiB (-79.0% better) |
| vector_search | 3.75 ms (+5.4% worse) | 10 | 2.37 GiB (-78.8% better) | 2.34 GiB (-79.0% better) | 2.37 GiB (-78.8% better) |
| hybrid_search | 3.66 ms (-4.5% better) | 10 | 2.37 GiB (-78.8% better) | 2.36 GiB (-78.8% better) | 2.37 GiB (-78.8% better) |
| token_match (all rows) | 109.86 ms (-11.5% better) | 1000.0K | 2.93 GiB (-75.0% better) | 2.92 GiB (-75.1% better) | 2.93 GiB (-75.0% better) |
| token_match (selective) | 563.35 µs (+54.6% worse) | 1 | 2.83 GiB (-75.7% better) | 2.83 GiB (-75.7% better) | 2.83 GiB (-75.7% better) |
| exact_match | 2.87 ms (-5.3% better) | 1 | 2.84 GiB (-75.6% better) | 2.83 GiB (-75.7% better) | 2.84 GiB (-75.6% better) |
<!-- END: bench/sql/supertable/warm -->

<!-- BEGIN: bench/sql/supertable/cold -->
### Supertable SQL — cold queries, fresh cache / object-store (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Cold = fresh disk cache + consumer per iteration, so each query pays the object-store cold open. Δ is vs the previous run.

| Query | cold open | cold search |
| --- | --- | --- |
| agg_max_title | 1.77 s (+77.2% worse) | 1.63 s (+25.4% worse) |
| filter_category_count | 1.09 s (+11.3% worse) | 1.71 s (+44.1% worse) |
| filter_rating_count | 1.05 s (+5.4% worse) | 1.53 s (+20.1% worse) |
| count_star | 961.16 ms (+5.9% worse) | 122.19 ms (+16.3% worse) |
| group_by_category | 1.04 s (-11.5% better) | 876.90 ms (-9.9% better) |
<!-- END: bench/sql/supertable/cold -->
