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
_Run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench sql` to populate._
<!-- END: bench/sql/build -->

<!-- BEGIN: bench/sql/query -->
_Run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench sql` to populate._
<!-- END: bench/sql/query -->
