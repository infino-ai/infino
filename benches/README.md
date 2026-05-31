# infino benches

Infino-only performance + correctness benches. Two criterion binaries:

- `fts` — superfile (1M docs Zipfian) + supertable (10M docs)
- `vector` — superfile (1M × 384 cosine) + supertable (10M × 384, 4 superfiles)

These benches measure infino in isolation — no third-party crates
enter this tree's dependency graph.

## Invocation

```sh
cargo bench --bench fts                            # all FTS (1M + 10M)
cargo bench --bench vector                         # all vector (1M + 10M)

# Filter to one sub-group (criterion regex/prefix on the group name)
cargo bench --bench fts -- superfile_fts_build     # superfile FTS ingest
cargo bench --bench fts -- supertable_fts_search   # supertable FTS search
cargo bench --bench vector -- superfile_vec_build  # superfile vector ingest
cargo bench --bench vector -- supertable_vec_search # supertable vector search

# Knobs
INFINO_SUPERTABLE__WRITER_THREADS=32 cargo bench --bench fts -- supertable_fts_build
INFINO_BENCH_UPDATE_README=1 cargo bench --bench fts        # rewrite FTS result tables in place
INFINO_BENCH_UPDATE_README=1 cargo bench --bench vector     # rewrite vector result tables in place
```

Every invocation runs the correctness phase unconditionally
(criterion filters skip timing, not setup), so a filter to a search
group still validates the BMW oracle (FTS) and the recall-floor gate
(vector) before timing starts.

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

<!-- BEGIN: bench/fts/supertable/ingest -->
### Supertable FTS — ingest (10000000 docs, Zipfian, 200 tokens/doc, 10K vocab)

| Engine                  | Time       | Throughput | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|-------------------------|------------|------------|-----------|------------|-----------|------------|
| infino_auto_writer_pool | 41.97 s    | 238.3 K/s  | 0 B       | 0 B        | 0 B       | —          |

*Output cardinality: infino emits `min(writer_pool.threads, chunk_rows)` superfiles per commit across 16 bounded append chunks (writer auto = cpus/2). Override with `INFINO_SUPERTABLE__WRITER_THREADS=N` for a specific shard count.*

<!-- END: bench/fts/supertable/ingest -->

<!-- BEGIN: bench/fts/supertable/search -->
### Supertable FTS — search (10000000 docs)

| Query          | infino     | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|----------------|------------|-----------|------------|-----------|------------|
| single_rare    | 42.69 µs   | 0 B       | 0 B        | 0 B       | —          |
| single_common  | 57.57 µs   | 0 B       | 0 B        | 0 B       | —          |
| two_term_or    | 301.85 µs  | 0 B       | 0 B        | 0 B       | —          |
| three_wide_or  | 2.67 ms    | 0 B       | 0 B        | 0 B       | —          |
| three_similar_or | 8.32 ms    | 0 B       | 0 B        | 0 B       | —          |
| five_term_or   | 14.18 ms   | 0 B       | 0 B        | 0 B       | —          |
| prefix         | 33.19 ms   | 0 B       | 0 B        | 0 B       | —          |

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

<!-- BEGIN: bench/vector/supertable/ingest -->
### Supertable vector — ingest (10000000 docs × dim=384, sharded into 4 superfiles)

| Engine | Time | Throughput | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|--------|------|------------|----------|------------|---------|------------|
| supertable | 179.72 s | 55.6 K/s | 27.79 GiB | 23.66 GiB | 25.89 GiB | — |

<!-- END: bench/vector/supertable/ingest -->

<!-- BEGIN: bench/vector/supertable/search -->
### Supertable vector — search (10000000 docs × dim=384, calibrated at recall targets)

Hot = in-memory; warm/cold = object storage + disk cache (s3s-fs or `INFINO_REAL_S3_BUCKET`).

| Recall target | (p/seg, r) | hot | warm | cold | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|---------------|------------|-----|------|------|----------|------------|---------|------------|
| 0.90 | (p=4, r=4) | 33.21 ms | 33.54 ms | 904.77 ms | 28.59 GiB | 28.57 GiB | 28.57 GiB | — |
| 0.95 | — | — | — | — | — | — | — | — |
| 0.99 | — | — | — | — | — | — | — | — |

<!-- END: bench/vector/supertable/search -->
