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
_run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench fts -- superfile_fts_build` to populate_
<!-- END: bench/fts/superfile/ingest -->

<!-- BEGIN: bench/fts/superfile/search -->
_run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench fts -- superfile_fts_search` to populate_
<!-- END: bench/fts/superfile/search -->

### FTS — supertable (multi-segment, 10M docs)

<!-- BEGIN: bench/fts/supertable/ingest -->
_run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench fts -- supertable_fts_build` to populate_
<!-- END: bench/fts/supertable/ingest -->

<!-- BEGIN: bench/fts/supertable/search -->
_run `INFINO_BENCH_UPDATE_README=1 cargo bench --bench fts -- supertable_fts_search` to populate_
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
