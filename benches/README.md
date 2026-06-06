# infino benches

Infino-only performance + correctness benches. Three criterion binaries:

- `superfile_fts` — FTS over one 1M-doc superfile
- `superfile_vector` — vector search over one 1M × 384 superfile
- `supertable_all` — one combined 10M-row supertable with both FTS and vector indexes

These benches measure infino in isolation — no third-party crates
enter this tree's dependency graph.

`cargo bench` runs only the regular local perf benches above. Diagnostic
benches are opt-in via `--features bench-diagnostics`:

- `object-store` — S3-compatible cold lazy-fetch path over a unified 1M superfile.
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
INFINO_BENCH_UPDATE_README=1 cargo bench --bench supertable_all

# Diagnostics (not run by plain `cargo bench`)
cargo bench --features bench-diagnostics --bench object-store
cargo bench --features bench-diagnostics --bench scale -- vector_recall
```

**Supertable search filters** (`supertable_fts_search`, `supertable_vec_search`)
build the shared combined fixture internally when needed. Build-only filters
skip search setup entirely.

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

Build path: `SuperfileBuilder` → unified `.parquet` (same as production supertable commit). Bandwidth (MB/s) is over the logical input text payload.

| Engine                       | Time       | Throughput | Bandwidth  | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|------------------------------|------------|------------|------------|-----------|------------|-----------|------------|
| infino_1thread               | 14.89 s    | 67.2 K/s   | 135.0 MB/s | 5.87 GiB  | 4.07 GiB   | 4.76 GiB  | +0.8% no change |
| infino_rayon_default_threads | 1.80 s     | 556.9 K/s  | 1.12 GB/s  | 7.57 GiB  | 6.31 GiB   | 7.04 GiB  | -6.3% improved |

<!-- END: bench/fts/superfile/ingest -->

<!-- BEGIN: bench/fts/superfile/search -->
### Superfile FTS — search (1000000 docs)

Hot = `SuperfileReader::open` in memory; cold = same `.parquet` on object storage via `DiskCacheStore::reader` → `bm25_search` (production cold path).

**OR queries:**

| Query          | hot        | cold       | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|----------------|------------|------------|-----------|------------|-----------|------------|
| single_rare    | 582 ns | 151.47 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |
| single_df1     | 243 ns | 115.97 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |
| single_common  | 16.43 µs | 159.62 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |
| two_term_or    | 158.41 µs | 140.11 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |
| three_wide_or  | 2.29 ms | 146.34 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |
| three_similar_or | 9.50 ms | 162.02 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |
| five_term_or   | 16.58 ms | 167.05 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |
| ten_term_or    | 49.42 ms | 215.66 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |

**AND queries:**

| Query          | hot        | cold       | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|----------------|------------|------------|-----------|------------|-----------|------------|
| two_term_and   | 199.03 µs | 139.63 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |
| three_wide_and | 3.39 ms | 163.54 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |
| three_similar_and | 5.64 ms | 158.71 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |
| five_term_and  | 6.94 ms | 159.05 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |
| ten_term_and   | 7.99 ms | 182.98 ms | 5.56 GiB  | 3.44 GiB   | 3.51 GiB  | -5.7% improved |

**Per-algorithm probes** (WAND+BMW vs MaxScore+BMM):

| Shape         | WAND+BMW   | MaxScore+BMM |
|---------------|------------|--------------|
| wide_3_or     | 7.64 ms    | 2.31 ms      |
| similar_3_or  | 14.14 ms   | 9.50 ms      |
| similar_5_or  | 40.34 ms   | 16.60 ms     |
| similar_10_or | 283.29 ms  | 49.49 ms     |

<!-- END: bench/fts/superfile/search -->

### FTS — supertable (multi-segment, 10M docs)

<!-- BEGIN: bench/supertable/ingest/supertable_fts_build -->
### Supertable FTS-only — ingest (10000000 docs × dim=384, 16 commits → 256 superfiles)

| Engine | Time | Throughput | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|--------|------|------------|----------|------------|---------|------------|
| supertable | 266.75 s | 37.5 K/s | 6.19 GiB | 2.11 GiB | 3.63 GiB | -0.6% no change |

<!-- END: bench/supertable/ingest/supertable_fts_build -->

<!-- BEGIN: bench/fts/supertable/search -->
### Supertable FTS — search (10000000 docs, shared combined supertable)

hot = warm steady state: every segment mmap-promoted via the public `Supertable::wait_until_warm` before timing, so reads hit resident pages (no object-store GETs). cold = fresh disk cache → object-store range GETs (s3s-fs or `INFINO_REAL_S3_BUCKET`), excluding the one-time manifest open.

| Query          | hot        | cold       | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|----------------|------------|------------|-----------|------------|-----------|------------|
| single_rare    | 2.78 ms | 320.45 ms | 10.88 GiB | 10.81 GiB  | 10.82 GiB | —          |
| single_common  | 2.79 ms | 316.83 ms | 10.88 GiB | 10.81 GiB  | 10.82 GiB | —          |
| two_term_or    | 3.20 ms | 434.43 ms | 10.88 GiB | 10.81 GiB  | 10.82 GiB | —          |
| three_wide_or  | 5.65 ms | 523.44 ms | 10.88 GiB | 10.81 GiB  | 10.82 GiB | —          |
| three_similar_or | 12.06 ms | 495.48 ms | 10.88 GiB | 10.81 GiB  | 10.82 GiB | —          |
| five_term_or   | 24.29 ms | 523.85 ms | 10.88 GiB | 10.81 GiB  | 10.82 GiB | —          |
| ten_term_or    | 65.30 ms | 441.77 ms | 10.88 GiB | 10.81 GiB  | 10.82 GiB | —          |
| prefix         | 51.61 ms | 484.51 ms | 10.88 GiB | 10.81 GiB  | 10.82 GiB | —          |

<!-- END: bench/fts/supertable/search -->

### Vector — superfile (single-segment, 1M × 384)

<!-- BEGIN: bench/vector/superfile/ingest -->
### Superfile vector — ingest (1000000 docs × dim=384, Gaussian planted clusters, cosine)

Build path: `SuperfileBuilder` → unified `.parquet` (same as production supertable commit). `infino_1thread` builds one superfile (the vector builder still uses intra-segment rayon for rotation/encode); `infino_rayon_default_threads` shards the corpus across the rayon pool into one superfile per shard (the multi-segment shape supertable commit produces). Bandwidth (MB/s) is over the raw f32 vector payload.

| Engine                       | Time       | Throughput | Bandwidth  | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|------------------------------|------------|------------|------------|-----------|------------|-----------|------------|
| infino_1thread               | 18.93 s    | 52.8 K/s   | 81.2 MB/s  | 4.36 GiB  | 1.88 GiB   | 2.88 GiB  | +0.3% no change |
| infino_rayon_default_threads | 2.18 s     | 458.1 K/s  | 703.7 MB/s | 6.75 GiB  | 5.28 GiB   | 6.58 GiB  | -3.4% no change |

<!-- END: bench/vector/superfile/ingest -->

<!-- BEGIN: bench/vector/superfile/search -->
### Superfile vector — search (1000000 docs × dim=384, calibrated at recall targets)

Hot = `SuperfileReader::open` in memory; cold = same `.parquet` on object storage via `DiskCacheStore::reader` → `vector_search` (production cold path).

| Recall target | (p, r)     | hot        | cold       | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|---------------|------------|------------|------------|----------|------------|---------|------------|
| 0.90          | (p=1, r=256) | 869.94 µs | 337.16 ms | 4.91 GiB | 4.13 GiB | 4.18 GiB | -0.2% no change |
| 0.95          | (p=1, r=256) | 867.16 µs | 432.55 ms | 4.91 GiB | 4.13 GiB | 4.18 GiB | -0.2% no change |
| 0.99          | (p=5, r=256) | 1.09 ms | 363.80 ms | 4.91 GiB | 4.13 GiB | 4.18 GiB | -0.2% no change |

**infino default options** (`nprobe=8, rerank_mult=20` — user-facing latency baseline):

| Metric | Value |
|--------|-------|
| infino_default_options_top10 (hot) | 617.41 µs |
| infino_default_options_top10 (cold) | 281.92 ms |
| infino_default_options_top10_peak_rss | 4.91 GiB |
| infino_default_options_top10_median_rss | 4.13 GiB |
| infino_default_options_top10_p90_rss | 4.18 GiB |

<!-- END: bench/vector/superfile/search -->

### Vector — supertable (multi-segment, 10M × 384)

<!-- BEGIN: bench/supertable/ingest/supertable_vec_build -->
### Supertable vector-only — ingest (10000000 docs × dim=384, 16 commits → 256 superfiles)

| Engine | Time | Throughput | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|--------|------|------------|----------|------------|---------|------------|
| supertable | 117.66 s | 85.0 K/s | 4.92 GiB | 3.05 GiB | 4.76 GiB | — |

<!-- END: bench/supertable/ingest/supertable_vec_build -->

<!-- BEGIN: bench/supertable/ingest/supertable_all_build -->
### Supertable combined FTS + vector — ingest (10000000 docs × dim=384, 16 commits → 256 superfiles)

| Engine | Time | Throughput | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|--------|------|------------|----------|------------|---------|------------|
| supertable | 379.09 s | 26.4 K/s | 9.21 GiB | 3.71 GiB | 6.37 GiB | — |

<!-- END: bench/supertable/ingest/supertable_all_build -->

<!-- BEGIN: bench/vector/supertable/search -->
### Supertable vector — search (10000000 docs × dim=384, calibrated at recall targets)

hot = warm steady state: every segment mmap-promoted via the public `Supertable::wait_until_warm` before timing, so reads hit resident pages (no object-store GETs). cold = fresh disk cache → object-store range GETs (s3s-fs or `INFINO_REAL_S3_BUCKET`), excluding the one-time manifest open.

| Recall target | (p/seg, r) | hot | cold | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|---------------|------------|-----|------|----------|------------|---------|------------|
| 0.90 | (p=4, r=4) | 10.02 ms | 1.03 s | 10.35 GiB | 10.33 GiB | 10.35 GiB | — |
| 0.95 | (p=8, r=4) | 10.73 ms | 1.07 s | 10.35 GiB | 10.33 GiB | 10.35 GiB | — |
| 0.99 | (p=32, r=4) | 23.03 ms | 1.42 s | 10.35 GiB | 10.33 GiB | 10.35 GiB | — |

<!-- END: bench/vector/supertable/search -->
