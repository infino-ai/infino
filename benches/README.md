# infino benches

Infino-only performance + correctness benches. Three criterion binaries:

- `superfile_fts` — FTS over one 1M-doc superfile
- `superfile_vector` — vector search over one 1M × 384 superfile
- `supertable_all` — one combined 10M-row supertable with both FTS and vector indexes

These benches measure infino in isolation — no third-party crates
enter this tree's dependency graph.

`cargo bench` runs only the regular local perf benches above. Diagnostic
benches are opt-in via `--features bench-diagnostics`:

- `object-store` — S3-compatible cold/warm lazy-fetch path over a unified 1M superfile.
- `scale` — release-profile recall gates such as `vector_recall`.

## Invocation

```sh
cargo bench --bench superfile_fts                  # 1M superfile FTS
cargo bench --bench superfile_vector               # 1M superfile vector
cargo bench --bench supertable_all                 # 10M supertable FTS + vector, one shared build

# Filter to one sub-group (criterion regex/prefix on the group name)
cargo bench --bench superfile_fts -- superfile_fts_build       # superfile FTS ingest
cargo bench --bench superfile_vector -- superfile_vec_build    # superfile vector ingest
cargo bench --bench supertable_all -- supertable_all_build     # shared FTS + vector supertable ingest
cargo bench --bench supertable_all -- supertable_fts_search    # supertable FTS search (needs ingest in same process)
cargo bench --bench supertable_all -- supertable_vec_search    # supertable vector search (needs ingest in same process)

# Search-only filter: include ingest in the same invocation (one process, shared fixture)
cargo bench --bench supertable_all -- supertable_all_build supertable_fts_search

# Knobs
INFINO_SUPERTABLE__WRITER_THREADS=32 cargo bench --bench supertable_all -- supertable_all_build
INFINO_SUPERTABLE_ALLOW_SEARCH_INGEST=1 cargo bench --bench supertable_all -- supertable_fts_search  # legacy: search triggers ingest
INFINO_BENCH_UPDATE_README=1 cargo bench --bench supertable_all

# Diagnostics (not run by plain `cargo bench`)
cargo bench --features bench-diagnostics --bench object-store
cargo bench --features bench-diagnostics --bench scale -- vector_recall
```

**Supertable search-only filters** (`supertable_fts_search`, `supertable_vec_search`)
require ingest in the **same** `cargo bench` process. Run
`supertable_all_build` first in that invocation, or use unfiltered
`cargo bench --bench supertable_all`. Set `INFINO_SUPERTABLE_ALLOW_SEARCH_INGEST=1`
to let a search-only run trigger ingest (old behaviour).

Superfile benches (1M) build their own fixture per binary; supertable
search groups run correctness (FTS oracle / vector recall floor) before timing
when ingest is already available.

## Code layout (`infino-bench-utils`)

```text
corpus/     synthetic rows + recall grading (streamed, small cache file)
ingest/     supertable append + commit → object storage
fixture/    one 10M ingest + search consumer per process
bench/      criterion groups (supertable ingest / FTS / vector search)
fts_superfile.rs, vector_superfile.rs   1M superfile bodies
```

## Result anchors

Each table below is wrapped in
`<!-- BEGIN: bench/... --> <!-- END: bench/... -->` markers; the bench's
markdown emitter rewrites the content between these markers when
`INFINO_BENCH_UPDATE_README=1` is set. Re-running a single bench with
a criterion filter refreshes only the matching section.

The markdown here is purely for human readers. Programmatic
consumers should read criterion's own
`target/criterion/<group>/<bench>/new/estimates.json` directly,
which is the structured source of truth the markdown is derived from.

---

## Results

### FTS — superfile (single-segment, 1M docs)

<!-- BEGIN: bench/fts/superfile/ingest -->
### Superfile FTS — ingest (1000000 docs, Zipfian, 200 tokens/doc, 10K vocab)

Build path: `SuperfileBuilder` → unified `.parquet` (same as production supertable commit).

| Engine                       | Time       | Throughput | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|------------------------------|------------|------------|-----------|------------|-----------|------------|
| infino_1thread               | 20.34 s    | 49.2 K/s   | 8.23 GiB  | 6.79 GiB   | 7.37 GiB  | —          |
| infino_rayon_default_threads | 2.09 s     | 479.4 K/s  | 9.78 GiB  | 8.32 GiB   | 9.15 GiB  | —          |

<!-- END: bench/fts/superfile/ingest -->

<!-- BEGIN: bench/fts/superfile/search -->
### Superfile FTS — search (1000000 docs)

Hot = `SuperfileReader::open` in memory; warm/cold = same `.parquet` on object storage via `DiskCacheStore::reader` → `bm25_search` (production cold/warm path).

| Query          | hot        | warm       | cold       | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|----------------|------------|------------|------------|-----------|------------|-----------|------------|
**OR queries:**

| single_rare    | 671 ns | — | — | 8.01 GiB  | 4.74 GiB   | 4.78 GiB  | —          |
| single_df1     | 279 ns | — | — | 8.01 GiB  | 4.74 GiB   | 4.78 GiB  | —          |
| single_common  | 26.67 µs | 27.12 µs | 306.60 ms | 8.01 GiB  | 4.74 GiB   | 4.78 GiB  | —          |
| two_term_or    | 183.65 µs | 184.09 µs | 346.61 ms | 8.01 GiB  | 4.74 GiB   | 4.78 GiB  | —          |
| three_wide_or  | 2.67 ms | 2.67 ms | 396.86 ms | 8.01 GiB  | 4.74 GiB   | 4.78 GiB  | —          |
| three_similar_or | 11.00 ms | — | — | 8.01 GiB  | 4.74 GiB   | 4.78 GiB  | —          |
| five_term_or   | 19.18 ms | — | — | 8.01 GiB  | 4.74 GiB   | 4.78 GiB  | —          |

**AND queries:**

| two_term_and   | 232.47 µs  | 8.01 GiB  | 4.74 GiB   | 4.78 GiB  | —          |
| three_wide_and | 4.04 ms    | 8.01 GiB  | 4.74 GiB   | 4.78 GiB  | —          |
| three_similar_and | 6.53 ms    | 8.01 GiB  | 4.74 GiB   | 4.78 GiB  | —          |
| five_term_and  | 8.01 ms    | 8.01 GiB  | 4.74 GiB   | 4.78 GiB  | —          |

**Per-algorithm probes** (WAND+BMW vs MaxScore+BMM):

| Shape         | WAND+BMW   | MaxScore+BMM |
|---------------|------------|--------------|
| wide_3_or     | 8.93 ms    | 2.67 ms      |
| similar_3_or  | 16.92 ms   | 11.02 ms     |
| similar_5_or  | 47.30 ms   | 19.21 ms     |

<!-- END: bench/fts/superfile/search -->

### FTS — supertable (multi-segment, 10M docs)

<!-- BEGIN: bench/supertable/ingest/supertable_fts_build -->
### Supertable FTS-only — ingest (10000000 docs × dim=384, 16 commits → 256 superfiles)

| Engine                  | Time       | Throughput | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|-------------------------|------------|------------|-----------|------------|-----------|------------|
| supertable | 344.78 s | 29.0 K/s | 6.93 GiB | 2.78 GiB | 5.70 GiB | — |

<!-- END: bench/supertable/ingest/supertable_fts_build -->

<!-- BEGIN: bench/fts/supertable/search -->
### Supertable FTS — search (10000000 docs, shared combined supertable)

Hot/warm/cold = object storage + disk cache (s3s-fs or `INFINO_REAL_S3_BUCKET`); warm waits for mmap promotion.

| Query          | hot        | warm       | cold       | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|----------------|------------|------------|------------|-----------|------------|-----------|------------|
| single_rare    | 3.29 ms | 3.02 ms | 3.28 s | 10.61 GiB | 4.80 GiB   | 8.15 GiB  | +466.2% regressed |
| single_common  | 3.66 ms | 3.01 ms | 3.35 s | 10.61 GiB | 4.80 GiB   | 8.15 GiB  | +466.2% regressed |
| two_term_or    | 4.06 ms | 3.46 ms | 2.96 s | 10.61 GiB | 4.80 GiB   | 8.15 GiB  | +466.2% regressed |
| three_wide_or  | 7.03 ms | 6.47 ms | 2.97 s | 10.61 GiB | 4.80 GiB   | 8.15 GiB  | +466.2% regressed |
| three_similar_or | 14.44 ms | 13.99 ms | 2.80 s | 10.61 GiB | 4.80 GiB   | 8.15 GiB  | +466.2% regressed |
| five_term_or   | 29.05 ms | 28.62 ms | 2.60 s | 10.61 GiB | 4.80 GiB   | 8.15 GiB  | +466.2% regressed |
| ten_term_or    | 77.34 ms | 77.30 ms | 2.49 s | 10.61 GiB | 4.80 GiB   | 8.15 GiB  | +466.2% regressed |
| prefix         | 61.42 ms | 63.83 ms | 2.75 s | 10.61 GiB | 4.80 GiB   | 8.15 GiB  | +466.2% regressed |

<!-- END: bench/fts/supertable/search -->

### Vector — superfile (single-segment, 1M × 384)

<!-- BEGIN: bench/vector/superfile/ingest -->
### Superfile vector — ingest (1000000 docs × dim=384, Gaussian planted clusters, cosine)

| Engine | Time | Throughput | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|--------|------|------------|----------|------------|---------|------------|
| infino | 18.89 s | 52.9 K/s | 4.15 GiB | 2.79 GiB | 3.67 GiB | — |

<!-- END: bench/vector/superfile/ingest -->

<!-- BEGIN: bench/vector/superfile/search -->
### Superfile vector — search (1000000 docs × dim=384, calibrated at recall targets)

Hot = `SuperfileReader::open` in memory; warm/cold = same `.parquet` on object storage via `DiskCacheStore::reader` → `vector_search` (production cold/warm path).

| Recall target | (p, r)     | hot        | warm       | cold       | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|---------------|------------|------------|------------|------------|----------|------------|---------|------------|
| 0.90          | (p=1, r=256) | 825.12 µs | 827.72 µs | 290.23 ms | 3.82 GiB | 3.80 GiB | 3.80 GiB | — |
| 0.95          | (p=5, r=256) | 970.47 µs | 966.88 µs | 306.32 ms | 3.82 GiB | 3.80 GiB | 3.80 GiB | — |
| 0.99          | — | — | — | — | — | — | — | — |

**infino default options** (`nprobe=8, rerank_mult=20` — user-facing latency baseline):

| Metric | Value |
|--------|-------|
| infino_default_options_top10 (hot) | 772.59 µs |
| infino_default_options_top10 (warm) | 772.95 µs |
| infino_default_options_top10 (cold) | 359.64 ms |
| infino_default_options_top10_peak_rss | 3.82 GiB |
| infino_default_options_top10_median_rss | 3.80 GiB |
| infino_default_options_top10_p90_rss | 3.80 GiB |

<!-- END: bench/vector/superfile/search -->

### Vector — supertable (multi-segment, 10M × 384)

<!-- BEGIN: bench/supertable/ingest/supertable_vec_build -->
### Supertable vector-only — ingest (10000000 docs × dim=384, 16 commits → 256 superfiles)

| Engine | Time | Throughput | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|--------|------|------------|----------|------------|---------|------------|
| supertable | 405.20 s | 24.7 K/s | 5.18 GiB | 2.94 GiB | 4.55 GiB | — |

<!-- END: bench/supertable/ingest/supertable_vec_build -->

<!-- BEGIN: bench/supertable/ingest/supertable_all_build -->
### Supertable combined FTS + vector — ingest (10000000 docs × dim=384, 16 commits → 256 superfiles)

| Engine | Time | Throughput | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|--------|------|------------|----------|------------|---------|------------|
| supertable | 494.81 s | 20.2 K/s | 7.47 GiB | 3.54 GiB | 6.36 GiB | +50951.9% regressed |

<!-- END: bench/supertable/ingest/supertable_all_build -->

<!-- BEGIN: bench/vector/supertable/search -->
### Supertable vector — search (10000000 docs × dim=384, calibrated at recall targets)

Hot/warm/cold = object storage + disk cache (s3s-fs or `INFINO_REAL_S3_BUCKET`); warm waits for mmap promotion.

| Recall target | (p/seg, r) | hot | warm | cold | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|---------------|------------|-----|------|------|----------|------------|---------|------------|
| 0.90 | (p=4, r=4) | 9.37 ms | 9.46 ms | 929.70 ms | 9.54 GiB | 9.51 GiB | 9.53 GiB | +376.3% regressed |
| 0.95 | (p=8, r=4) | 10.12 ms | 10.14 ms | 990.25 ms | 9.54 GiB | 9.51 GiB | 9.53 GiB | +376.3% regressed |
| 0.99 | (p=16, r=4) | 16.56 ms | 17.16 ms | 1.08 s | 9.54 GiB | 9.51 GiB | 9.53 GiB | +376.3% regressed |

<!-- END: bench/vector/supertable/search -->
