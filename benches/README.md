# Benchmarks

Infino benchmark measurements from the in-tree harness (`cargo bench`). One binary
builds the corpus, times ingest and queries, samples RSS, and prints the tables.

## Latest (CI)

Azure Blob, 1M-doc supertables, warm cache —
[`bd7caa3`](https://github.com/infino-ai/infino/commit/bd7caa3b42981a31657c88d1d984dda07a53da9f)
([run 32825375037](https://github.com/infino-ai/infino/actions/runs/32825375037)),
Intel Xeon 6973P-C (4C/7T pinned, 31 GiB).

| | Shape | Result |
|---|---|---:|
| FTS | single_rare top-10, warm+fetch | **166 µs** p50 |
| Vector | default top-10 cosine, post-drain, 1024-d | **663 µs** p50 · **0.992** recall@10 |
| Hybrid | hybrid_search via SQL, 10 rows | **3.47 ms** p50 |
| Ingest | FTS / vector / SQL | **62.1 / 7.8 / 17.0** K docs/s |

Raw logs: Actions artifacts on that run.

## Run

```sh
cargo bench                              # all cells
cargo bench -- supertable fts            # one cell
cargo bench -- superfile sql cold        # phases: build | warm | cold
INFINO_BENCH_SUPERFILE_DOCS=100000 \
  cargo bench -- superfile fts warm      # plain ints only (no 100K suffix)

cargo bench -- diagnostic                # optional diagnostics
cargo bench -- tombstone
```

Positional args after `--`: `[tier] [modality] [phase...] [corpus=…]`.
Tier: `superfile` | `supertable`. Modality: `fts` | `vector` | `sql`.
Omitted tokens mean "all".

## Knobs

| Env | Default | Meaning |
|---|---|---|
| `INFINO_BENCH_SUPERFILE_DOCS` | `1000000` | superfile corpus size |
| `INFINO_BENCH_SUPERTABLE_DOCS` | `10000000` | supertable corpus size |
| `INFINO_BENCH_STORE` | `rustfs` | `rustfs`, `s3`, `azure`, or `gcs` |
| `INFINO_BENCH_KEEP_TABLE` | unset | keep object-store prefix after the run |
| `INFINO_BENCH_UPDATE_README` | unset | rewrite the marked sections below |

Synthetic FTS corpus: seed `1`, 200 Zipfian tokens/doc, 10K vocab.
Synthetic vectors: cosine, **1024-d**, seed `1` (archived tables below used 384-d).

### Object store

```sh
# Local HTTPS S3 stand-in (default)
cargo bench -- superfile fts cold

INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=… \
  cargo bench -- supertable fts

INFINO_BENCH_STORE=azure INFINO_REAL_AZURE_CONTAINER=… \
  AZURE_STORAGE_ACCOUNT_NAME=… AZURE_STORAGE_ACCOUNT_KEY=… \
  cargo bench -- supertable sql cold
```

| Store | Extra env |
|---|---|
| `rustfs` | optional `INFINO_RUSTFS_*`, `RUSTFS_ACCESS_KEY`, `RUSTFS_SECRET_KEY` |
| `s3` | `INFINO_REAL_S3_BUCKET`, `AWS_*` |
| `azure` | `INFINO_REAL_AZURE_CONTAINER`, `AZURE_STORAGE_ACCOUNT_NAME`, `AZURE_STORAGE_ACCOUNT_KEY` |
| `gcs` | `INFINO_REAL_GCS_BUCKET`, `GOOGLE_APPLICATION_CREDENTIALS` |

### Prepared datasets

Ingest once, re-bench reads without regenerating:

```sh
INFINO_BENCH_STORE=azure INFINO_REAL_AZURE_CONTAINER=… \
  cargo bench -- dataset prepare datasets/bench-10m

cargo bench -- dataset bench datasets/bench-10m vector warm
cargo bench -- dataset run datasets/bench-10m fts
```

`INFINO_BENCH_SUPERTABLE_DOCS` must match the prepare-time count.

## Matrix

| Selector | Tier | Modality |
|---|---|---|
| `superfile fts` | single artifact, in-memory warm | FTS |
| `superfile vector` | | vector |
| `superfile sql` | | SQL |
| `supertable fts` | multi-artifact on object storage | FTS |
| `supertable vector` | | vector |
| `supertable sql` | | SQL |

Each cell: `build`, `warm`, `cold`. Multi-cell runs isolate each cell in its own process (RSS).

Vector cells report a `default` config row; probe/rerank are not env-tunable.
Recall is checked against brute-force ground truth.

## Layout

```text
corpus.rs, executors.rs, harness/, report.rs, markdown.rs, rss.rs
tiers.rs, rustfs_server.rs
superfile.rs, supertable.rs, ingest/, fixture/
scale.rs, sql_diag.rs, tombstone_overhead.rs, …
```

JSON metrics land in `target/infino-bench/<bench>.json` (local only).

### Superfile FTS

<!-- BEGIN: bench/fts/superfile/ingest -->
**Superfile FTS — ingest, single-superfile / in-memory (1M docs, Zipfian, 200 tokens/doc, 10K vocab)**

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 17.04 s | 58.7 K/s | 118.0 MB/s | 5.79 GiB | 3.78 GiB | 4.81 GiB |
| 16 writers | 2.11 s | 473.7 K/s | 952.2 MB/s | 8.01 GiB | 7.21 GiB | 7.64 GiB |
<!-- END: bench/fts/superfile/ingest -->

<!-- BEGIN: bench/fts/superfile/search -->
**Superfile FTS — search, single-superfile / in-memory (1M docs)**

**OR queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_rare | 6.28 µs | 10.64 ms | 3.67 GiB | 3.67 GiB | 3.67 GiB | 164.60 ms | 28.27 ms |
| single_df1 | 623 ns | 17.26 ms | 3.68 GiB | 3.68 GiB | 3.68 GiB | 189.65 ms | 11.13 µs |
| single_common | 2.00 ms | 42.85 ms | 3.68 GiB | 3.68 GiB | 3.68 GiB | 171.75 ms | 58.68 ms |
| two_term_or | 226.83 µs | 40.82 ms | 3.68 GiB | 3.68 GiB | 3.68 GiB | 218.66 ms | 57.07 ms |
| three_wide_or | 2.44 ms | 48.86 ms | 3.68 GiB | 3.68 GiB | 3.68 GiB | 190.61 ms | 58.79 ms |
| three_similar_or | 10.36 ms | 55.02 ms | 3.68 GiB | 3.68 GiB | 3.68 GiB | 218.08 ms | 52.64 ms |
| five_term_or | 17.77 ms | 64.40 ms | 3.68 GiB | 3.68 GiB | 3.68 GiB | 200.83 ms | 59.16 ms |
| ten_term_or | 52.39 ms | 98.39 ms | 3.68 GiB | 3.68 GiB | 3.68 GiB | 204.48 ms | 112.53 ms |

**AND queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_term_and | 523.26 µs | 41.28 ms | 3.69 GiB | 3.69 GiB | 3.69 GiB | 222.91 ms | 56.13 ms |
| three_wide_and | 4.28 ms | 50.67 ms | 3.69 GiB | 3.69 GiB | 3.69 GiB | 191.48 ms | 59.73 ms |
| three_similar_and | 6.09 ms | 50.77 ms | 3.68 GiB | 3.68 GiB | 3.68 GiB | 171.49 ms | 55.71 ms |
| five_term_and | 7.49 ms | 54.07 ms | 3.68 GiB | 3.68 GiB | 3.68 GiB | 201.46 ms | 71.01 ms |
| ten_term_and | 8.65 ms | 53.39 ms | 3.68 GiB | 3.68 GiB | 3.68 GiB | 256.80 ms | 93.38 ms |

**Per-algorithm probes (WAND+BMW vs MaxScore+BMM)**

| Shape | WAND+BMW | MaxScore+BMM |
| --- | --- | --- |
| wide_3_or | 9.23 ms | 2.47 ms |
| similar_3_or | 15.04 ms | 10.30 ms |
| similar_5_or | 44.07 ms | 17.80 ms |
| similar_10_or | 302.85 ms | 52.55 ms |
<!-- END: bench/fts/superfile/search -->

<!-- BEGIN: bench/fts/superfile/negation -->
**Superfile FTS — negation (`-term`), warm (1M docs)**

**Negation queries**

| Query | warm |
| --- | --- |
| mid_pos_common_neg | 1.63 ms |
| mid_pos_rare_neg | 27.96 µs |
| two_mid_or_common_neg | 4.55 ms |
| two_mid_and_common_neg | 5.15 ms |
<!-- END: bench/fts/superfile/negation -->

### Supertable FTS

<!-- BEGIN: bench/fts/supertable/ingest -->
**Supertable FTS — ingest, multi-superfile / object-store (1M docs, 16 commits)**

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| FTS-only | 25.87 s | 38.7 K/s | 256 | 1.31 GiB | 1.10 GiB | 1.23 GiB |
<!-- END: bench/fts/supertable/ingest -->

<!-- BEGIN: bench/fts/supertable/search -->
**Supertable FTS — search, multi-superfile / object-store (1M docs)**

**OR queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_rare | 1.14 ms | 8.93 ms | 942.10 MiB | 932.93 MiB | 942.10 MiB | 556.11 ms | 272.69 ms |
| single_df1 | 55.22 µs | 2.65 ms | 906.39 MiB | 904.30 MiB | 906.39 MiB | 436.74 ms | 16.64 ms |
| single_common | 1.31 ms | 10.55 ms | 1.15 GiB | 1.07 GiB | 1.15 GiB | 444.48 ms | 360.95 ms |
| two_term_or | 1.16 ms | 10.54 ms | 1.22 GiB | 1.11 GiB | 1.22 GiB | 440.47 ms | 264.00 ms |
| three_wide_or | 1.29 ms | 11.90 ms | 1.20 GiB | 1.09 GiB | 1.20 GiB | 423.31 ms | 361.95 ms |
| three_similar_or | 2.23 ms | 10.79 ms | 1.11 GiB | 1013.32 MiB | 1.11 GiB | 505.38 ms | 397.19 ms |
| five_term_or | 3.39 ms | 12.49 ms | 1.14 GiB | 1.06 GiB | 1.14 GiB | 474.28 ms | 302.61 ms |
| ten_term_or | 7.93 ms | 16.80 ms | 1.18 GiB | 1.07 GiB | 1.18 GiB | 417.63 ms | 608.07 ms |

**AND queries**

| Query | warm | warm +fetch | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_term_and | 1.14 ms | 10.09 ms | 1.10 GiB | 1.05 GiB | 1.10 GiB | 488.90 ms | 239.08 ms |
| three_wide_and | 1.40 ms | 11.72 ms | 1.16 GiB | 1.03 GiB | 1.16 GiB | 454.35 ms | 205.77 ms |
| three_similar_and | 1.88 ms | 10.90 ms | 1.18 GiB | 1.04 GiB | 1.18 GiB | 441.45 ms | 203.19 ms |
| five_term_and | 2.13 ms | 11.33 ms | 1.17 GiB | 1.07 GiB | 1.17 GiB | 453.28 ms | 343.62 ms |
| ten_term_and | 2.43 ms | 11.29 ms | 1.14 GiB | 1.06 GiB | 1.14 GiB | 392.86 ms | 304.12 ms |
<!-- END: bench/fts/supertable/search -->

### Superfile vector

<!-- BEGIN: bench/vector/superfile/ingest -->
**Superfile vector — ingest, single-superfile / in-memory (1M docs × dim=384)**

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 20.61 s | 48.5 K/s | 74.5 MB/s | 3.87 GiB | 1.74 GiB | 2.79 GiB |
| 16 writers | 2.66 s | 376.6 K/s | 578.5 MB/s | 6.93 GiB | 6.85 GiB | 6.93 GiB |
<!-- END: bench/vector/superfile/ingest -->

<!-- BEGIN: bench/vector/superfile/search -->
**Superfile vector — search, single-superfile / in-memory (1M docs × dim=384)**

| Recall target | (p, r) | recall | warm | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | p=1, r=256 | 0.962 | 1.04 ms | 4.11 GiB | 4.11 GiB | 4.11 GiB | 218.78 ms | 354.58 ms |
| 0.95 | p=1, r=256 | 0.962 | 1.01 ms | 4.12 GiB | 4.12 GiB | 4.12 GiB | 179.32 ms | 215.49 ms |
| 0.99 | p=5, r=256 | 0.998 | 1.55 ms | 4.13 GiB | 4.12 GiB | 4.13 GiB | 384.58 ms | 508.34 ms |
| default | p=8, r=20 | — | 947.29 µs | 4.13 GiB | 4.13 GiB | 4.13 GiB | 353.10 ms | 372.97 ms |
<!-- END: bench/vector/superfile/search -->

### Supertable vector

<!-- BEGIN: bench/vector/supertable/ingest -->
**Supertable vector — ingest, multi-superfile / object-store (1M docs × dim=384, 16 commits)**

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| vector-only | 24.66 s | 40.6 K/s | 256 | 2.62 GiB | 1.91 GiB | 2.50 GiB |
<!-- END: bench/vector/supertable/ingest -->

<!-- BEGIN: bench/vector/supertable/search -->
**Supertable vector — search, multi-superfile / object-store (1M docs × dim=384)**

| Recall target | (p, r) | recall | warm | Peak RSS | Median RSS | P90 RSS | cold open | cold search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90 | p=5, r=1 | 0.988 | 5.13 ms | 2.64 GiB | 2.64 GiB | 2.64 GiB | 2.19 s | 604.99 ms |
| 0.95 | p=5, r=1 | 0.988 | 4.35 ms | 3.75 GiB | 3.75 GiB | 3.75 GiB | 2.00 s | 429.61 ms |
| 0.99 | p=10, r=1 | 0.996 | 5.29 ms | 3.17 GiB | 3.16 GiB | 3.17 GiB | 1.89 s | 466.60 ms |
| default | p=8, r=20 | — | 6.82 ms | 3.46 GiB | 3.45 GiB | 3.46 GiB | 1.96 s | 585.76 ms |
<!-- END: bench/vector/supertable/search -->

### Supertable ingest summary

<!-- BEGIN: bench/supertable/ingest -->
**Supertable — ingest, multi-superfile / object-store (1M docs, 16 commits)**

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| FTS-only | 53.83 s | 18.6 K/s | 256 | 9.61 GiB | 8.62 GiB | 8.80 GiB |
| vector-only | 28.50 s | 35.1 K/s | 256 | 10.36 GiB | 9.01 GiB | 10.30 GiB |
| SQL | 75.02 s | 13.3 K/s | 256 | 11.42 GiB | 9.44 GiB | 11.02 GiB |
<!-- END: bench/supertable/ingest -->

### Superfile SQL

<!-- BEGIN: bench/sql/build -->
**Superfile SQL — ingest, single superfile / in-memory (1M rows: title + category + score)**

| Build | Time | Throughput | Bandwidth | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 writer | 10.10 s | 99.0 K/s | 199.0 MB/s | 4.88 GiB | 3.85 GiB | 4.61 GiB |
| 16 writers | 5.30 s | 188.6 K/s | 379.1 MB/s | 14.08 GiB | 10.77 GiB | 13.42 GiB |
<!-- END: bench/sql/build -->

<!-- BEGIN: bench/sql/query -->
**Superfile SQL — query, single superfile / in-memory (1M rows)**

**Aggregations & count-filters (read + compute, return few rows — not the index A/B)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| agg_max_title | 180.15 ms | 1 | 5.63 GiB | 5.51 GiB | 5.57 GiB |
| filter_category_count | 10.08 ms | 1 | 4.92 GiB | 4.92 GiB | 4.92 GiB |
| filter_rating_count | 7.50 ms | 1 | 4.79 GiB | 4.79 GiB | 4.79 GiB |
| count_star | 6.39 ms | 1 | 4.79 GiB | 4.78 GiB | 4.79 GiB |
| group_by_category | 7.62 ms | 4 | 4.79 GiB | 4.78 GiB | 4.79 GiB |

**Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 7.76 ms | 1 | 5.02 GiB | 5.01 GiB | 5.02 GiB |
| WHERE key   = ?  (unsorted col, min/max defeated) | 9.76 ms | 1 | 5.05 GiB | 5.05 GiB | 5.05 GiB |

**FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 3.98 ms | 1 | 4.97 GiB | 4.97 GiB | 4.97 GiB |
| WHERE key   = ?  (unsorted col, min/max defeated) | 1.70 ms | 1 | 4.97 GiB | 4.96 GiB | 4.97 GiB |

**Aggregate over FTS candidates — Full Scan (DataFusion only)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 9.89 ms | 1 | 4.97 GiB | 4.97 GiB | 4.97 GiB |
| SUM(rating)         key=? (1 row) | 10.63 ms | 1 | 4.97 GiB | 4.97 GiB | 4.97 GiB |
| MAX(rating)         key=? (1 row) | 11.12 ms | 1 | 4.97 GiB | 4.97 GiB | 4.97 GiB |
| AVG(rating)         key=? (1 row) | 10.05 ms | 1 | 4.97 GiB | 4.97 GiB | 4.97 GiB |
| SUM(rating) bucket IN all (1M rows) | 14.58 ms | 1 | 4.97 GiB | 4.97 GiB | 4.97 GiB |

**Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 1.95 ms | 1 | 4.93 GiB | 4.93 GiB | 4.93 GiB |
| SUM(rating)         key=? (1 row) | 2.17 ms | 1 | 4.94 GiB | 4.93 GiB | 4.94 GiB |
| MAX(rating)         key=? (1 row) | 2.09 ms | 1 | 4.94 GiB | 4.94 GiB | 4.94 GiB |
| AVG(rating)         key=? (1 row) | 1.87 ms | 1 | 4.94 GiB | 4.94 GiB | 4.94 GiB |
| SUM(rating) bucket IN all (1M rows) | 12.06 ms | 1 | 4.94 GiB | 4.94 GiB | 4.94 GiB |

**Search table functions (bm25 / vector / hybrid / token / exact)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| bm25_search | 913.73 µs | 10 | 4.79 GiB | 4.79 GiB | 4.79 GiB |
| vector_search | 1.31 ms | 10 | 4.79 GiB | 4.79 GiB | 4.79 GiB |
| hybrid_search | 1.26 ms | 10 | 4.79 GiB | 4.79 GiB | 4.79 GiB |
| token_match (all rows) | 67.83 ms | 1000.0K | 5.12 GiB | 5.09 GiB | 5.12 GiB |
| token_match (selective) | 256.05 µs | 1 | 5.00 GiB | 5.00 GiB | 5.00 GiB |
| exact_match | 2.84 ms | 1 | 5.01 GiB | 5.00 GiB | 5.01 GiB |
<!-- END: bench/sql/query -->

<!-- BEGIN: bench/sql/superfile/cold -->
**Superfile SQL — cold query, object-store (1M rows)**

| Query | cold open | cold search |
| --- | --- | --- |
| agg_max_title | 277.21 ms | 1.93 s |
| filter_category_count | 273.12 ms | 268.43 ms |
| filter_rating_count | 251.27 ms | 394.90 ms |
| count_star | 495.55 ms | 69.88 ms |
| group_by_category | 245.43 ms | 174.96 ms |
<!-- END: bench/sql/superfile/cold -->

### Supertable SQL

<!-- BEGIN: bench/sql/supertable/ingest -->
**Supertable SQL — ingest, multi-superfile / object-store (1M rows, 16 commits)**

| Shape | Time | Throughput | Superfiles | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- | --- |
| SQL | 41.30 s | 24.2 K/s | 256 | 2.08 GiB | 1.57 GiB | 1.95 GiB |
<!-- END: bench/sql/supertable/ingest -->

<!-- BEGIN: bench/sql/supertable/warm -->
**Supertable SQL — warm queries, warm cache / object-store (1M rows)**

**Aggregations & count-filters (read + compute, return few rows — not the index A/B)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| agg_max_title | 176.28 ms | 1 | 2.91 GiB | 2.83 GiB | 2.91 GiB |
| filter_category_count | 22.67 ms | 1 | 2.44 GiB | 2.44 GiB | 2.44 GiB |
| filter_rating_count | 20.08 ms | 1 | 2.30 GiB | 2.30 GiB | 2.30 GiB |
| count_star | 19.58 ms | 1 | 2.30 GiB | 2.30 GiB | 2.30 GiB |
| group_by_category | 21.23 ms | 4 | 2.30 GiB | 2.30 GiB | 2.30 GiB |

**Plain Scan (DataFusion only) — selective equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 7.52 ms | 1 | 2.84 GiB | 2.83 GiB | 2.84 GiB |
| WHERE key   = ?  (unsorted col, min/max defeated) | 21.90 ms | 1 | 2.84 GiB | 2.84 GiB | 2.84 GiB |

**FTS-pushdown (DataFusion + Infino) — SAME equality, 1 row (sorted vs unsorted col)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| WHERE title = ?  (sorted col, min/max prunes) | 4.23 ms | 1 | 2.77 GiB | 2.76 GiB | 2.77 GiB |
| WHERE key   = ?  (unsorted col, min/max defeated) | 1.44 ms | 1 | 2.76 GiB | 2.76 GiB | 2.76 GiB |

**Aggregate over FTS candidates — Full Scan (DataFusion only)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 22.55 ms | 1 | 2.77 GiB | 2.76 GiB | 2.77 GiB |
| SUM(rating)         key=? (1 row) | 22.95 ms | 1 | 2.77 GiB | 2.76 GiB | 2.77 GiB |
| MAX(rating)         key=? (1 row) | 23.81 ms | 1 | 2.77 GiB | 2.77 GiB | 2.77 GiB |
| AVG(rating)         key=? (1 row) | 22.56 ms | 1 | 2.77 GiB | 2.76 GiB | 2.77 GiB |
| SUM(rating) bucket IN all (1M rows) | 30.54 ms | 1 | 2.77 GiB | 2.77 GiB | 2.77 GiB |

**Aggregate over FTS candidates — FTS-pushdown (DataFusion + Infino token_match)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| COUNT(*)            key=? (1 row) | 1.69 ms | 1 | 2.76 GiB | 2.76 GiB | 2.76 GiB |
| SUM(rating)         key=? (1 row) | 1.94 ms | 1 | 2.76 GiB | 2.76 GiB | 2.76 GiB |
| MAX(rating)         key=? (1 row) | 1.85 ms | 1 | 2.76 GiB | 2.76 GiB | 2.76 GiB |
| AVG(rating)         key=? (1 row) | 1.84 ms | 1 | 2.76 GiB | 2.76 GiB | 2.76 GiB |
| SUM(rating) bucket IN all (1M rows) | 66.05 ms | 1 | 2.76 GiB | 2.76 GiB | 2.76 GiB |

**Search table functions (bm25 / vector / hybrid / token / exact)**

| Query | p50 | Rows | Peak RSS | Median RSS | P90 RSS |
| --- | --- | --- | --- | --- | --- |
| bm25_search | 2.42 ms | 10 | 2.34 GiB | 2.30 GiB | 2.34 GiB |
| vector_search | 3.75 ms | 10 | 2.37 GiB | 2.34 GiB | 2.37 GiB |
| hybrid_search | 3.66 ms | 10 | 2.37 GiB | 2.36 GiB | 2.37 GiB |
| token_match (all rows) | 109.86 ms | 1000.0K | 2.93 GiB | 2.92 GiB | 2.93 GiB |
| token_match (selective) | 563.35 µs | 1 | 2.83 GiB | 2.83 GiB | 2.83 GiB |
| exact_match | 2.87 ms | 1 | 2.84 GiB | 2.83 GiB | 2.84 GiB |
<!-- END: bench/sql/supertable/warm -->

<!-- BEGIN: bench/sql/supertable/cold -->
**Supertable SQL — cold queries, fresh cache / object-store (1M rows)**

| Query | cold open | cold search |
| --- | --- | --- |
| agg_max_title | 1.77 s | 1.63 s |
| filter_category_count | 1.09 s | 1.71 s |
| filter_rating_count | 1.05 s | 1.53 s |
| count_star | 961.16 ms | 122.19 ms |
| group_by_category | 1.04 s | 876.90 ms |
<!-- END: bench/sql/supertable/cold -->
