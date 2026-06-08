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
  docs, controlled by `INFINO_BENCH_SUPERFILE_DOCS`.
- **Supertable** — multi-artifact table committed to object storage and read
  through hot/cold table paths. Default scale: `10M` docs, controlled by
  `INFINO_BENCH_SUPERTABLE_DOCS`.
- **Writer count** — build rows report `1 writer` and `N writers`. `N` defaults
  to the machine's logical core count and is controlled by
  `INFINO_BENCH_WRITERS`.

## Invocation

```sh
cargo bench --bench superfile_fts
cargo bench --bench superfile_vector
cargo bench --bench supertable_all

# Smaller local loop.
INFINO_BENCH_SUPERFILE_DOCS=100K cargo bench --bench superfile_fts

# Override the N-writers build row.
INFINO_BENCH_WRITERS=4 cargo bench --bench superfile_fts

# Refresh the markdown sections in this file.
INFINO_BENCH_UPDATE_README=1 cargo bench --bench superfile_fts

# Diagnostics (not part of the default bench loop).
cargo bench --features bench-diagnostics --bench object-store
cargo bench --features bench-diagnostics --bench scale -- vector_recall
```

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
| 1 writer | 9.14 s (+0.5% ~) | 109.5 K/s (-0.5% ~) | 220.0 MB/s (-0.5% ~) | 5.02 GiB (-0.2% ~) | 3.86 GiB (-3.1% better) | 4.83 GiB (-0.6% ~) |
| 16 writers | 5.02 s (+10.6% worse) | 199.2 K/s (-9.6% worse) | 400.4 MB/s (-9.6% worse) | 13.13 GiB (-4.2% better) | 10.66 GiB (-2.4% ~) | 12.53 GiB (-5.9% better) |
<!-- END: bench/sql/build -->

<!-- BEGIN: bench/sql/query -->
### SQL — query, in-memory supertable (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Hot p50 over `Supertable::query_sql` against the canonical 1-writer table. The headline comparison is the last two blocks: the *same* selective equality (one matching row) run against a non-indexed column (Plain Scan — DataFusion decodes + filters) vs the byte-identical FTS-indexed `title` column (FTS-pushdown — infino's token index selects the candidate row, DataFusion verifies). Same predicate, same 1-row result, so the gap is purely the index. The first block is aggregations & count-filters (read + compute, return few rows) — general engine context, not a like-for-like index comparison; there is no bare `SELECT col` row because that only measures row materialization. `Rows` is the result-set size. Δ is vs the previous run.

**Aggregations & count-filters (read + compute, return few rows — not the index A/B)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| agg_max_title | 158.07 ms (+1.7% ~) | 1 | 11.38 GiB (-16.9% better) | 6.14 GiB (+1.3% ~) | 10.68 GiB (-12.9% better) |
| filter_category_count | 7.16 ms (+3.9% worse) | 1 | 5.59 GiB (+0.5% ~) | 5.54 GiB (-0.4% ~) | 5.59 GiB (+0.5% ~) |
| filter_rating_count | 4.74 ms (-6.4% better) | 1 | 5.54 GiB (-0.4% ~) | 5.54 GiB (-0.4% ~) | 5.54 GiB (-0.4% ~) |
| count_star | 7.10 ms (+7.5% worse) | 1 | 5.34 GiB (-0.5% ~) | 5.33 GiB (-0.5% ~) | 5.34 GiB (-0.5% ~) |
| group_by_category | 5.09 ms (+3.8% worse) | 4 | 5.34 GiB (-0.5% ~) | 5.29 GiB (+1.9% ~) | 5.34 GiB (-0.5% ~) |

**Plain Scan (DataFusion only) — selective equality, 1 matching row**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title_noidx = ?   (no index) | 7.01 ms (+0.7% ~) | 1 | 5.22 GiB (+0.1% ~) | 5.21 GiB (+0.3% ~) | 5.22 GiB (+0.1% ~) |

**FTS-pushdown (DataFusion + Infino) — SAME equality, 1 matching row**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?         (FTS index) | 17.84 ms (-2.9% ~) | 1 | 5.22 GiB (+1.5% ~) | 5.17 GiB (+1.1% ~) | 5.22 GiB (+1.5% ~) |

**Search table functions (bm25 / vector / hybrid / token / exact)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| bm25_search | 751.66 µs (new) | 10 | 5.09 GiB (new) | 5.08 GiB (new) | 5.09 GiB (new) |
| vector_search | 1.23 ms (new) | 10 | 5.09 GiB (new) | 5.09 GiB (new) | 5.09 GiB (new) |
| hybrid_search | 1.30 ms (new) | 10 | 5.09 GiB (new) | 5.09 GiB (new) | 5.09 GiB (new) |
| token_match | 57.82 ms (new) | 1000.0K | 5.23 GiB (new) | 5.22 GiB (new) | 5.23 GiB (new) |
| exact_match | 3.22 ms (new) | 1 | 5.21 GiB (new) | 5.21 GiB (new) | 5.21 GiB (new) |
<!-- END: bench/sql/query -->
