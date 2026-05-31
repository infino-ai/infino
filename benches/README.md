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
| infino_1thread               | 20.18 s    | 49.6 K/s   | 8.20 GiB  | 6.78 GiB   | 7.37 GiB  | —          |
| infino_rayon_default_threads | 2.11 s     | 473.1 K/s  | 9.76 GiB  | 8.31 GiB   | 9.37 GiB  | —          |

<!-- END: bench/fts/superfile/ingest -->

<!-- BEGIN: bench/fts/superfile/search -->
### Superfile FTS — search (1000000 docs)

| Query          | infino     | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|----------------|------------|-----------|------------|-----------|------------|
**OR queries:**

| single_rare    | 258 ns     | 0 B       | 0 B        | 0 B       | —          |
| single_df1     | 117 ns     | 0 B       | 0 B        | 0 B       | —          |
| single_common  | 14.31 µs   | 0 B       | 0 B        | 0 B       | —          |
| two_term_or    | 103.78 µs  | 0 B       | 0 B        | 0 B       | —          |
| three_wide_or  | 1.27 ms    | 0 B       | 0 B        | 0 B       | —          |
| three_similar_or | 5.81 ms    | 0 B       | 0 B        | 0 B       | —          |
| five_term_or   | 9.90 ms    | 0 B       | 0 B        | 0 B       | —          |

**AND queries:**

| two_term_and   | 116.57 µs  | 0 B       | 0 B        | 0 B       | —          |
| three_wide_and | 2.03 ms    | 0 B       | 0 B        | 0 B       | —          |
| three_similar_and | 3.31 ms    | 0 B       | 0 B        | 0 B       | —          |
| five_term_and  | 3.97 ms    | 0 B       | 0 B        | 0 B       | —          |

**Per-algorithm probes** (WAND+BMW vs MaxScore+BMM):

| Shape         | WAND+BMW   | MaxScore+BMM |
|---------------|------------|--------------|
| wide_3_or     | 4.68 ms    | 1.24 ms      |
| similar_3_or  | 8.91 ms    | 5.77 ms      |
| similar_5_or  | 25.16 ms   | 9.88 ms      |

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
| infino | 18.87 s | 53.0 K/s | 4.16 GiB | 2.79 GiB | 3.66 GiB | — |

<!-- END: bench/vector/superfile/ingest -->

<!-- BEGIN: bench/vector/superfile/search -->
### Superfile vector — search (1000000 docs × dim=384, calibrated at recall targets)

Hot = `SuperfileReader::open` in memory; warm/cold = same `.parquet` on object storage via `DiskCacheStore::reader` → `vector_search` (production cold/warm path).

| Recall target | (p, r)     | hot        | warm       | cold       | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|---------------|------------|------------|------------|------------|----------|------------|---------|------------|
| 0.90          | (p=1, r=256) | 828.42 µs | 823.76 µs | 278.70 ms | 3.81 GiB | 3.78 GiB | 3.79 GiB | — |
| 0.95          | (p=5, r=256) | 964.98 µs | 967.19 µs | 273.96 ms | 3.81 GiB | 3.78 GiB | 3.79 GiB | — |
| 0.99          | — | — | — | — | — | — | — | — |

**infino default options** (`nprobe=8, rerank_mult=20` — user-facing latency baseline):

| Metric | Value |
|--------|-------|
| infino_default_options_top10 (hot) | 771.49 µs |
| infino_default_options_top10 (warm) | 780.77 µs |
| infino_default_options_top10 (cold) | 274.54 ms |
| infino_default_options_top10_peak_rss | 3.81 GiB |
| infino_default_options_top10_median_rss | 3.78 GiB |
| infino_default_options_top10_p90_rss | 3.79 GiB |

<!-- END: bench/vector/superfile/search -->

### Vector — supertable (multi-segment, 10M × 384)

<!-- BEGIN: bench/vector/supertable/ingest -->
### Supertable vector — ingest (10000000 docs × dim=384, sharded into 4 superfiles)

| Engine | Time | Throughput | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|--------|------|------------|----------|------------|---------|------------|
| supertable | 178.49 s | 56.0 K/s | 27.47 GiB | 23.72 GiB | 25.93 GiB | — |

<!-- END: bench/vector/supertable/ingest -->

<!-- BEGIN: bench/vector/supertable/search -->
### Supertable vector — search (10000000 docs × dim=384, calibrated at recall targets)

Hot = in-memory; warm/cold = object storage + disk cache (s3s-fs or `INFINO_REAL_S3_BUCKET`).

| Recall target | (p/seg, r) | hot | warm | cold | Peak RSS | Median RSS | P90 RSS | Peak RSS Δ |
|---------------|------------|-----|------|------|----------|------------|---------|------------|
| 0.90 | (p=2, r=4) | 29.01 ms | 262.26 ms | 1.13 s | 28.21 GiB | 28.20 GiB | 28.20 GiB | — |
| 0.95 | — | — | — | — | — | — | — | — |
| 0.99 | — | — | — | — | — | — | — | — |

<!-- END: bench/vector/supertable/search -->
