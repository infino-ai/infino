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
| 1 writer | 9.17 s | 109.1 K/s | 219.2 MB/s | 4.90 GiB | 3.92 GiB | 4.71 GiB |
| 16 writers | 4.63 s | 216.1 K/s | 434.3 MB/s | 13.75 GiB | 10.92 GiB | 13.39 GiB |
<!-- END: bench/sql/build -->

<!-- BEGIN: bench/sql/query -->
### SQL — query, in-memory supertable (1M rows)

_Host: Intel(R) Xeon(R) Platinum 8488C · 8C/16T · 31 GiB RAM · linux/x86_64_

Hot p50 over `Supertable::query_sql` against the canonical 1-writer table. The headline is the index A/B: the *same* selective equality (one matching row) on a non-indexed column (Plain Scan — DataFusion decodes + filters) vs an FTS-indexed column (FTS-pushdown — `token_match` selects the candidate row, `FilterExec` verifies). It is run on two column shapes: a **sorted** column (`title`, where DataFusion's min/max can prune on its own) and an **unsorted** high-cardinality column (`key`, where min/max spans every page and can't prune) — the unsorted column is the honest win-case. The aggregate-over-candidates block runs the aggregate shapes (`COUNT`/`SUM`/`MAX`/`AVG`) over the selective `key` plus one many-matches predicate (`bucket IN (b0..b9)` ≈ 1M rows) that trips the `df`-based selectivity gate and falls back to a plain scan. The first block is full-table aggregations/count-filters (no FTS predicate) for context. `Rows` is the result-set size.

**Aggregations & count-filters (full-table, no FTS predicate — context, not the index A/B)**

| Query | p50 | Rows |
| --- | --- | --- |
| agg_max_title | 157.72 ms | 1 |
| filter_category_count | 6.98 ms | 1 |
| filter_rating_count | 4.99 ms | 1 |
| count_star | 6.64 ms | 1 |
| group_by_category | 5.46 ms | 4 |

**Selective equality, 1 matching row — Plain Scan (DataFusion only) vs FTS-pushdown**

| Predicate | Plain Scan (DataFusion only) | FTS-pushdown (DataFusion + Infino) | Speedup |
| --- | --- | --- | --- |
| `WHERE title = ?` — sorted col, min/max prunes | 7.04 ms | 3.14 ms | 2.2× |
| `WHERE key = ?` — unsorted col, min/max defeated | 6.83 ms | 1.37 ms | 5.0× |

**Aggregate over FTS candidates — Full Scan (DataFusion only) vs FTS-pushdown**

| Query | Full Scan (DataFusion only) | FTS-pushdown | Speedup |
| --- | --- | --- | --- |
| `COUNT(*)  WHERE key = ?` (1 row) | 6.76 ms | 1.70 ms | 4.0× |
| `SUM(rating) WHERE key = ?` (1 row) | 6.96 ms | 1.64 ms | 4.2× |
| `MAX(rating) WHERE key = ?` (1 row) | 7.39 ms | 1.91 ms | 3.9× |
| `AVG(rating) WHERE key = ?` (1 row) | 6.90 ms | 1.79 ms | 3.9× |
| `SUM(rating) WHERE bucket IN (b0..b9)` (1M rows) | 11.14 ms | 8.99 ms | gated → scan |

The aggregate function is a passenger — the cost is the candidate access, not the reduction. The `bucket IN all` row matches ~every row, so the gate estimates the `IN`/`OR` union as `sum(df) ≈ 1M > 1%` and falls back to the scan (8.99 ms) instead of running the index path (44 ms ungated).

**Search table functions (bm25 / vector / hybrid / token / exact)**

| Query | p50 | Rows |
| --- | --- | --- |
| bm25_search | 1.11 ms | 10 |
| vector_search | 1.29 ms | 10 |
| hybrid_search | 1.55 ms | 10 |
| token_match (selective, 1 token) | 860.38 µs | 1 |
| token_match (all rows) | 58.45 ms | 1M |
| exact_match | 2.94 ms | 1 |
<!-- END: bench/sql/query -->
