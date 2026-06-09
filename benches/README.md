# infino benches

Infino's in-tree benchmarks measure Infino itself. Cross-engine comparison
benches live in `retrievalbench`; these tables are the Infino reference numbers
those comparisons are checked against.

The benchmark harness is moving to Infino's custom bench harness. The custom
harness owns the measured lifecycle directly:

- generate the corpus once;
- build the artifact once;
- run correctness on that built artifact;
- run hot reads on that artifact;
- upload or commit that same artifact for object-store tiers;
- run cold reads against the uploaded/committed artifact with fresh cache state;
- sample RSS around the measured phase;
- render terminal and markdown reports through `report.rs`.

The invariant is simple: **the first measured build produces the artifact used by
correctness, hot reads, and cold upload/commit.** The benchmark must not rebuild a
second copy just to run correctness or object-store reads.

## Bench Shapes

- **Superfile** — single-artifact, in-memory read path. Default scale: `1M`
  docs, controlled by `INFINO_BENCH_DOC_COUNT`.
- **Supertable** — multi-artifact table committed to object storage and read
  through hot/cold table paths. Default scale: `10M` docs, controlled by
  `INFINO_BENCH_DOC_COUNT`.
- **Writer count** — build rows report `1 writer` and `N writers`. `N` defaults
  to the machine's logical core count and is controlled by
  `INFINO_BENCH_WRITERS`.

## Invocation

```sh
cargo bench --bench superfile_fts
cargo bench --bench superfile_vector
cargo bench --bench supertable_all

# Smaller local loop. Doc counts are plain integers (a `100K`/`1M`-style
# suffix is not parsed and silently falls back to the default).
INFINO_BENCH_DOC_COUNT=100000 cargo bench --bench superfile_fts

# Override the N-writers build row.
INFINO_BENCH_WRITERS=4 cargo bench --bench superfile_fts

# Refresh the markdown sections in this file.
INFINO_BENCH_UPDATE_README=1 cargo bench --bench superfile_fts

# Diagnostics (not part of the default bench loop).
cargo bench --features bench-diagnostics --bench object-store
cargo bench --features bench-diagnostics --bench scale -- vector_recall
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
# Superfile benches: any backend (s3s-fs is the zero-setup default).
cargo bench --bench superfile_fts

# Supertable bench: real object store only (s3 or azure). s3s-fs lacks the
# multi-commit If-Match CAS the supertable commit needs, so it is rejected.
INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=my-bucket \
  cargo bench --bench supertable_all
INFINO_BENCH_STORE=azure INFINO_REAL_AZURE_CONTAINER=my-container \
  AZURE_STORAGE_ACCOUNT_NAME=... AZURE_STORAGE_ACCOUNT_KEY=... \
  cargo bench --bench supertable_all
```

A real-backend run writes under a unique prefix and deletes it on exit; set
`INFINO_BENCH_KEEP_TABLE=1` to keep it (the prefix is logged). The s3s-fs
emulator self-cleans and reproduces request/byte volume, not network latency.

## Migration Status

Only migrated sections should be treated as current. Sections that still show a
placeholder are waiting for their custom-harness migration.

- FTS superfile: custom harness, artifact reuse fixed.
- Vector superfile: pending `VectorEngine` migration.
- SQL: pending `SqlEngine` migration.
- Supertable object-store: pending custom harness migration.

See `bench-harness-migration-plan.md` in this worktree for the uncommitted
working plan.

## Code Layout (`infino-bench-utils`)

```text
corpus/                     synthetic corpora + brute-force oracles
harness/                    engine interfaces and generic drivers
report.rs                   terminal + markdown rendering with deltas
rss.rs                      per-phase RSS sampling
fts_superfile.rs            superfile FTS runner
vector_superfile.rs         superfile vector runner (migration pending)
ingest/, fixture/, bench/   supertable object-store helpers (migration pending)
```

## Result Anchors

Each generated section is wrapped in
`<!-- BEGIN: bench/... --> <!-- END: bench/... -->` markers. When
`INFINO_BENCH_UPDATE_README=1` is set, migrated runners replace the matching
block. The generated markdown is the human-facing artifact; migrated sections
are produced directly by the custom harness.

---

## Results

### FTS — superfile (single-segment, 1M docs)

<!-- BEGIN: bench/fts/superfile/ingest -->
_Run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench superfile_fts` to populate._
<!-- END: bench/fts/superfile/ingest -->

<!-- BEGIN: bench/fts/superfile/search -->
_Run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench superfile_fts` to populate._
<!-- END: bench/fts/superfile/search -->

### FTS — supertable (multi-segment, 10M docs)

<!-- BEGIN: bench/fts/supertable/search -->
_Pending custom-harness search migration._
<!-- END: bench/fts/supertable/search -->

### Vector — superfile (single-segment, 1M × 384)

<!-- BEGIN: bench/vector/superfile/ingest -->
_Pending `VectorEngine` migration._
<!-- END: bench/vector/superfile/ingest -->

<!-- BEGIN: bench/vector/superfile/search -->
_Pending `VectorEngine` migration._
<!-- END: bench/vector/superfile/search -->

### Supertable — ingest (multi-segment, object store)

<!-- BEGIN: bench/supertable/ingest -->
_Run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench supertable_all` to populate._
<!-- END: bench/supertable/ingest -->

### Vector — supertable (multi-segment, 10M × 384)

<!-- BEGIN: bench/vector/supertable/search -->
_Pending custom-harness search migration._
<!-- END: bench/vector/supertable/search -->

### SQL — in-memory supertable

<!-- BEGIN: bench/sql/build -->
### SQL — ingest, in-memory supertable (1M rows: title + category + score)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Build path: `SupertableWriter::append` + `commit` into an in-memory supertable, through the engine-generic `run_sql` driver the cross-engine comparison also uses. Rows are by writer count: `1 writer` is the canonical build queries run against; `N writers` is the sharded parallel build. Δ is vs the previous run.

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 12.79 s (new) | 78.2 K/s (new) | 157.1 MB/s (new) | 4.90 GiB (new) | 3.79 GiB (new) | 4.66 GiB (new) |
| 16 writers | 5.84 s (new) | 171.1 K/s (new) | 344.0 MB/s (new) | 12.16 GiB (new) | 10.48 GiB (new) | 12.08 GiB (new) |
<!-- END: bench/sql/build -->

<!-- BEGIN: bench/sql/query -->
### SQL — query, in-memory supertable (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Hot p50 over `Supertable::query_sql` against the canonical 1-writer table. The headline comparison is the last two blocks: the *same* selective equality (one matching row) run against a non-indexed column (Plain Scan — DataFusion decodes + filters) vs the byte-identical FTS-indexed `title` column (FTS-pushdown — infino's token index selects the candidate row, DataFusion verifies). Same predicate, same 1-row result, so the gap is purely the index. The first block is aggregations & count-filters (read + compute, return few rows) — general engine context, not a like-for-like index comparison; there is no bare `SELECT col` row because that only measures row materialization. `Rows` is the result-set size. Δ is vs the previous run.

**Aggregations & count-filters (read + compute, return few rows — not the index A/B)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| agg_max_title | 161.40 ms (new) | 1 | 12.00 GiB (new) | 5.94 GiB (new) | 10.68 GiB (new) |
| filter_category_count | 7.66 ms (new) | 1 | 5.74 GiB (new) | 5.58 GiB (new) | 5.74 GiB (new) |
| filter_rating_count | 5.03 ms (new) | 1 | 5.58 GiB (new) | 5.58 GiB (new) | 5.58 GiB (new) |
| count_star | 6.88 ms (new) | 1 | 5.46 GiB (new) | 5.46 GiB (new) | 5.46 GiB (new) |
| group_by_category | 5.57 ms (new) | 4 | 5.40 GiB (new) | 5.40 GiB (new) | 5.40 GiB (new) |

**Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 7.11 ms (new) | 1 | 5.30 GiB (new) | 5.30 GiB (new) | 5.30 GiB (new) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 6.69 ms (new) | 1 | 5.32 GiB (new) | 5.30 GiB (new) | 5.32 GiB (new) |

**FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 3.03 ms (new) | 1 | 5.32 GiB (new) | 5.32 GiB (new) | 5.32 GiB (new) |
| WHERE key   = ?  (unsorted col, min/max defeated) | 1.39 ms (new) | 1 | 5.32 GiB (new) | 5.32 GiB (new) | 5.32 GiB (new) |

**Aggregate over FTS candidates — Full Scan (DataFusion only)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 6.82 ms (new) | 1 | 5.32 GiB (new) | 5.28 GiB (new) | 5.32 GiB (new) |
| SUM(rating)         key=? (1 row) | 7.10 ms (new) | 1 | 5.25 GiB (new) | 5.25 GiB (new) | 5.25 GiB (new) |
| MAX(rating)         key=? (1 row) | 7.63 ms (new) | 1 | 5.25 GiB (new) | 5.22 GiB (new) | 5.25 GiB (new) |
| AVG(rating)         key=? (1 row) | 7.56 ms (new) | 1 | 5.22 GiB (new) | 5.22 GiB (new) | 5.22 GiB (new) |
| SUM(rating) bucket IN all (1M rows) | 11.10 ms (new) | 1 | 5.21 GiB (new) | 5.17 GiB (new) | 5.21 GiB (new) |

**Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 1.90 ms (new) | 1 | 5.17 GiB (new) | 5.17 GiB (new) | 5.17 GiB (new) |
| SUM(rating)         key=? (1 row) | 1.68 ms (new) | 1 | 5.17 GiB (new) | 5.17 GiB (new) | 5.17 GiB (new) |
| MAX(rating)         key=? (1 row) | 1.78 ms (new) | 1 | 5.16 GiB (new) | 5.16 GiB (new) | 5.16 GiB (new) |
| AVG(rating)         key=? (1 row) | 1.62 ms (new) | 1 | 5.15 GiB (new) | 5.15 GiB (new) | 5.15 GiB (new) |
| SUM(rating) bucket IN all (1M rows) | 8.91 ms (new) | 1 | 5.15 GiB (new) | 5.15 GiB (new) | 5.15 GiB (new) |

**Search table functions (bm25 / vector / hybrid / token / exact)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| bm25_search | 967.45 µs (new) | 10 | 5.41 GiB (new) | 5.41 GiB (new) | 5.41 GiB (new) |
| vector_search | 1.46 ms (new) | 10 | 5.37 GiB (new) | 5.37 GiB (new) | 5.37 GiB (new) |
| hybrid_search | 1.38 ms (new) | 10 | 5.37 GiB (new) | 5.37 GiB (new) | 5.37 GiB (new) |
| token_match (all rows) | 58.61 ms (new) | 1000.0K | 5.43 GiB (new) | 5.30 GiB (new) | 5.40 GiB (new) |
| token_match (selective) | 323.57 µs (new) | 1 | 5.29 GiB (new) | 5.29 GiB (new) | 5.29 GiB (new) |
| exact_match | 2.98 ms (new) | 1 | 5.30 GiB (new) | 5.29 GiB (new) | 5.30 GiB (new) |
<!-- END: bench/sql/query -->
