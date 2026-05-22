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

| Engine                       | Time       | Throughput | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|------------------------------|------------|------------|-----------|------------|-----------|------------|
| infino_1thread               | 9.30 s     | 107.5 K/s  | 0 B       | 0 B        | 0 B       | —          |
| infino_rayon_default_threads | 815.44 ms  | 1.23 M/s   | 0 B       | 0 B        | 0 B       | —          |

<!-- END: bench/fts/superfile/ingest -->

<!-- BEGIN: bench/fts/superfile/search -->
### Superfile FTS — search (1000000 docs)

| Query          | infino     | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|----------------|------------|-----------|------------|-----------|------------|
**OR queries:**

| single_rare    | 260 ns     | 0 B       | 0 B        | 0 B       | —          |
| single_df1     | 119 ns     | 0 B       | 0 B        | 0 B       | —          |
| single_common  | 14.83 µs   | 0 B       | 0 B        | 0 B       | —          |
| two_term_or    | 106.07 µs  | 0 B       | 0 B        | 0 B       | —          |
| three_wide_or  | 1.61 ms    | 0 B       | 0 B        | 0 B       | —          |
| three_similar_or | 9.07 ms    | 0 B       | 0 B        | 0 B       | —          |
| five_term_or   | 15.69 ms   | 0 B       | 0 B        | 0 B       | —          |

**AND queries:**

| two_term_and   | 4.44 ms    | 0 B       | 0 B        | 0 B       | —          |
| three_wide_and | 3.33 ms    | 0 B       | 0 B        | 0 B       | —          |
| three_similar_and | 4.58 ms    | 0 B       | 0 B        | 0 B       | —          |
| five_term_and  | 5.11 ms    | 0 B       | 0 B        | 0 B       | —          |

**Per-algorithm probes** (WAND+BMW vs MaxScore+BMM):

| Shape         | WAND+BMW   | MaxScore+BMM |
|---------------|------------|--------------|
| wide_3_or     | 4.80 ms    | 1.34 ms      |
| similar_3_or  | 9.17 ms    | 5.98 ms      |
| similar_5_or  | 26.35 ms   | 10.07 ms     |

<!-- END: bench/fts/superfile/search -->

### FTS — supertable (multi-segment, 10M docs)

<!-- BEGIN: bench/fts/supertable/ingest -->
### Supertable FTS — ingest (10000000 docs, Zipfian, 200 tokens/doc, 10K vocab)

| Engine                  | Time       | Throughput | Peak RSS  | Median RSS | P90 RSS   | Peak RSS Δ |
|-------------------------|------------|------------|-----------|------------|-----------|------------|
| infino_auto_writer_pool | 41.97 s    | 238.3 K/s  | 0 B       | 0 B        | 0 B       | —          |

*Output cardinality: infino emits `min(writer_pool.threads, total_rows)` superfiles per commit (auto = cpus/2). Override with `INFINO_SUPERTABLE__WRITER_THREADS=N` for a specific shard count.*

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
_run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench vector -- superfile_vec_build` to populate_
<!-- END: bench/vector/superfile/ingest -->

<!-- BEGIN: bench/vector/superfile/search -->
_run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench vector -- superfile_vec_search` to populate_
<!-- END: bench/vector/superfile/search -->

### Vector — supertable (multi-segment, 10M × 384)

<!-- BEGIN: bench/vector/supertable/ingest -->
_run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench vector -- supertable_vec_build` to populate_
<!-- END: bench/vector/supertable/ingest -->

<!-- BEGIN: bench/vector/supertable/search -->
_run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench vector -- supertable_vec_search` to populate_
<!-- END: bench/vector/supertable/search -->
