// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! SQL scan-path diagnostic — localizes where `Supertable::query_sql`
//! scalar-scan latency actually goes.
//!
//! The headline SQL bench (`cargo bench -- superfile sql`) reports scalar
//! scans (`scan_all`, `filter_category`, `filter_rating`) at ~300ms
//! while `count_star` / `group_by_category` land at single-digit ms.
//! This diagnostic decomposes that gap by timing infino's full
//! `query_sql` path against progressively-thinner DataFusion baselines
//! over the **same** data, so the cost can be attributed to a layer:
//!
//!   * `infino query_sql`        — the full path we want to speed up
//!     (superfile prune → in-memory object store → DataFusion
//!     `ParquetSource` → `FilterExec` → collect).
//!   * `  ├ parse+plan`          — `ctx.sql(...)` on the cached
//!     `SessionContext` (planning only, no execution).
//!   * `  └ execute`             — `DataFrame::collect()` (the scan).
//!   * `DataFusion / parquet`    — vanilla DataFusion reading the same
//!     superfile Parquet files from a temp dir via `register_parquet`.
//!     Isolates infino's provider/object-store wrapper: if this
//!     matches `query_sql`, the wrapper is free and the cost is
//!     DataFusion's Parquet scan itself.
//!   * `DataFusion / MemTable`   — DataFusion scan+filter+collect over
//!     the already-decoded Arrow batches (no Parquet at all). The
//!     floor for DataFusion's executor + output materialization.
//!   * `raw parquet-rs decode`   — `ParquetRecordBatchReaderBuilder`
//!     decoding only the projected column(s) straight from the
//!     superfile bytes, predicate applied by hand. The floor a custom
//!     `ExecutionPlan` that decodes our layout directly would
//!     approach.
//!
//! Reading the table: `query_sql − DataFusion/parquet` is infino
//! provider overhead; `DataFusion/parquet − DataFusion/MemTable` is
//! Parquet decode through DataFusion; `DataFusion/MemTable − raw
//! decode` is DataFusion executor + materialization overhead; `raw
//! decode` is the intrinsic cost of pulling the bytes off our format.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench -- sql-diag
//! INFINO_BENCH_SUPERTABLE_DOCS=1000000 cargo bench -- sql-diag
//! INFINO_DIAG_ITERS=20 cargo bench -- sql-diag
//! # delegate to the kernel-vs-query_sql TVF dispatch-tax diagnostic:
//! INFINO_SQL_DIAG=tvf cargo bench -- sql-diag
//! ```

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{Int64Array, LargeStringArray, RecordBatch};
use bytes::Bytes;
use datafusion::{
    datasource::MemTable,
    prelude::{ParquetReadOptions, SessionContext},
};
use parquet::arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder};
use rayon::prelude::*;
use tokio::runtime::Runtime;

use crate::{diag_common, markdown::fmt_count};

const TABLE: &str = "supertable";

/// One SQL shape exercised across every path. `raw_cols` are the
/// 0-based column indices in Infino's actual Parquet body (`_id`=0,
/// `title`=1, `category`=2, `rating`=3) the raw decoder must read; `keep`
/// decides which decoded rows survive for the by-hand floor.
struct Shape {
    name: &'static str,
    sql: &'static str,
    raw_cols: &'static [usize],
    keep: fn(category: &str, rating: i64) -> bool,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "scan_all",
        sql: "SELECT title FROM supertable",
        raw_cols: &[1],
        keep: |_, _| true,
    },
    Shape {
        name: "filter_category",
        sql: "SELECT title FROM supertable WHERE category = 'rust'",
        raw_cols: &[1, 2],
        keep: |c, _| c == "rust",
    },
    Shape {
        name: "filter_rating",
        sql: "SELECT title FROM supertable WHERE rating < 10",
        raw_cols: &[1, 3],
        keep: |_, r| r < 10,
    },
];

// Table build, corpus, chunk batching, and the stat/format helpers are
// shared with the FTS diagnostic — see `crate::diag_common`.

/// Raw parquet-rs decode of `cols` from every superfile, applying `keep`
/// to each row by hand and counting survivors — the direct-decode
/// floor a custom `ExecutionPlan` would approach. When only `title`
/// is projected (no filter columns) the per-row predicate is skipped
/// entirely, matching a filterless scan.
fn raw_decode(superfiles: &[Bytes], cols: &[usize], keep: fn(&str, i64) -> bool) -> usize {
    superfiles
        .par_iter()
        .map(|bytes| {
            let mut kept = 0usize;
            let builder =
                ParquetRecordBatchReaderBuilder::try_new(bytes.clone()).expect("parquet builder");
            let mask = ProjectionMask::roots(builder.parquet_schema(), cols.iter().copied());
            let reader = builder.with_projection(mask).build().expect("reader");
            for batch in reader {
                let batch = batch.expect("batch");
                let n = batch.num_rows();
                // Locate the filter columns within the *projected* batch by
                // name; absent ⇒ that predicate input wasn't projected, so
                // there is no filter on it (e.g. scan_all projects title only).
                let cat = batch.schema().index_of("category").ok().map(|i| {
                    batch
                        .column(i)
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .expect("category LargeUtf8")
                });
                let rating = batch.schema().index_of("rating").ok().map(|i| {
                    batch
                        .column(i)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .expect("rating Int64")
                });
                if cat.is_none() && rating.is_none() {
                    kept += n;
                    continue;
                }
                for row in 0..n {
                    let category = cat.map(|array| array.value(row)).unwrap_or("");
                    let rating = rating.map(|array| array.value(row)).unwrap_or(0);
                    if keep(category, rating) {
                        kept += 1;
                    }
                }
            }
            kept
        })
        .sum()
}

fn count_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

pub fn run() {
    if std::env::var("INFINO_SQL_DIAG").as_deref() == Ok("tvf") {
        // Delegate to the kernel-vs-query_sql dispatch-tax diagnostic
        // (TVF search path) that lives with the object-store bench.
        crate::unified_object_store::diag::diagnose_query_sql_overhead();
        return;
    }

    let cfg = diag_common::config();
    let n = cfg.n_docs;
    let iters = cfg.iters;
    eprintln!(
        "[sql-diag] scalar scan decomposition: n_docs={} iters={iters} \
         (knobs: INFINO_BENCH_SUPERTABLE_DOCS, INFINO_DIAG_ITERS)",
        fmt_count(n)
    );

    // ── Shared corpus + supertable build (see `diag_common`); the chunk
    //    batches come back for the DataFusion MemTable baseline below. ──
    eprintln!("[sql-diag] building infino supertable...");
    let build_t0 = Instant::now();
    let (table, batches) = diag_common::build_supertable(&cfg);
    eprintln!(
        "[sql-diag] supertable built in {:.1}s",
        build_t0.elapsed().as_secs_f64()
    );
    let reader = table.reader();
    let manifest = reader.manifest();
    let superfiles: Vec<Bytes> = manifest
        .superfiles
        .iter()
        .map(|entry| {
            manifest
                .options
                .store
                .reader(&entry.uri)
                .expect("in-memory superfile reader")
                .parquet_bytes()
                .cloned()
                .expect("in-memory reader carries complete bytes")
        })
        .collect();
    eprintln!(
        "[sql-diag] {} Infino superfile(s), {:.1} MiB total",
        superfiles.len(),
        superfiles.iter().map(Bytes::len).sum::<usize>() as f64 / (1024.0 * 1024.0)
    );

    // Spill superfiles to a temp dir for the vanilla-DataFusion baseline.
    let dir = tempfile::TempDir::new().expect("tempdir");
    for (i, bytes) in superfiles.iter().enumerate() {
        let path = dir.path().join(format!("seg_{i:05}.parquet"));
        std::fs::write(&path, bytes).expect("write superfile parquet");
    }

    let rt = Runtime::new().expect("tokio runtime");
    let cached_ctx = table.__debug_cached_session();

    eprintln!();
    eprintln!("[sql-diag] === per-query path decomposition (warm p50 / mean) ===");
    eprintln!(
        "[sql-diag] {:<16} {:>22} {:>22} {:>22} {:>22} {:>22}   rows",
        "query",
        "query_sql",
        "  ├ parse+plan",
        "  └ execute",
        "DataFusion/parquet",
        "DataFusion/MemTable",
    );

    for shape in SHAPES {
        // 1. infino query_sql (full path).
        let (full_p50, full_mean, full_rows) = diag_common::time_path(iters, || {
            table
                .reader()
                .query_sql(shape.sql)
                .map(|b| count_rows(&b))
                .expect("query_sql")
        });

        // 1a/1b. parse+plan vs execute on the cached SessionContext.
        let mut pp = Vec::with_capacity(iters);
        let mut ex = Vec::with_capacity(iters);
        let mut df_rows = 0usize;
        for _ in 0..iters {
            let t0 = Instant::now();
            let df = rt.block_on(cached_ctx.sql(shape.sql)).expect("ctx.sql");
            pp.push(t0.elapsed());
            let t1 = Instant::now();
            let out = rt.block_on(df.collect()).expect("collect");
            ex.push(t1.elapsed());
            df_rows = count_rows(&out);
        }
        let pp_p50 = diag_common::percentile(&mut pp, 50);
        let ex_p50 = diag_common::percentile(&mut ex, 50);

        // 2. vanilla DataFusion over the same parquet files (no infino).
        let (dfq_p50, dfq_mean, dfq_rows) = {
            let mut samples = Vec::with_capacity(iters);
            // warm
            let mut rows = rt.block_on(async {
                let ctx = SessionContext::new();
                ctx.register_parquet(
                    TABLE,
                    dir.path().to_str().expect("utf8 dir"),
                    ParquetReadOptions::default(),
                )
                .await
                .expect("register_parquet");
                let df = ctx.sql(shape.sql).await.expect("df sql");
                count_rows(&df.collect().await.expect("df collect"))
            });
            for _ in 0..iters {
                let t = Instant::now();
                let got = rt.block_on(async {
                    let ctx = SessionContext::new();
                    ctx.register_parquet(
                        TABLE,
                        dir.path().to_str().expect("utf8 dir"),
                        ParquetReadOptions::default(),
                    )
                    .await
                    .expect("register_parquet");
                    let df = ctx.sql(shape.sql).await.expect("df sql");
                    count_rows(&df.collect().await.expect("df collect"))
                });
                samples.push(t.elapsed());
                std::hint::black_box(got);
                rows = got;
            }
            let sum: u128 = samples.iter().map(|d| d.as_nanos()).sum();
            (
                diag_common::percentile(&mut samples, 50),
                Duration::from_nanos((sum / samples.len().max(1) as u128) as u64),
                rows,
            )
        };

        // 3. DataFusion MemTable over the already-decoded Arrow batches.
        let (mem_p50, mem_mean, mem_rows) = {
            let provider = MemTable::try_new(diag_common::diag_schema(), vec![batches.clone()])
                .expect("memtable");
            let provider = Arc::new(provider);
            let mut samples = Vec::with_capacity(iters);
            // warm
            let mut rows = rt.block_on(async {
                let ctx = SessionContext::new();
                ctx.register_table(TABLE, provider.clone())
                    .expect("register");
                let df = ctx.sql(shape.sql).await.expect("mem sql");
                count_rows(&df.collect().await.expect("mem collect"))
            });
            for _ in 0..iters {
                let t = Instant::now();
                let got = rt.block_on(async {
                    let ctx = SessionContext::new();
                    ctx.register_table(TABLE, provider.clone())
                        .expect("register");
                    let df = ctx.sql(shape.sql).await.expect("mem sql");
                    count_rows(&df.collect().await.expect("mem collect"))
                });
                samples.push(t.elapsed());
                std::hint::black_box(got);
                rows = got;
            }
            let sum: u128 = samples.iter().map(|d| d.as_nanos()).sum();
            (
                diag_common::percentile(&mut samples, 50),
                Duration::from_nanos((sum / samples.len().max(1) as u128) as u64),
                rows,
            )
        };

        // 4. raw parquet-rs decode floor.
        let (raw_p50, raw_mean, raw_rows) = diag_common::time_path(iters, || {
            raw_decode(&superfiles, shape.raw_cols, shape.keep)
        });

        // Sanity: every path must agree on the result-set size.
        assert_eq!(
            full_rows, df_rows,
            "{}: query_sql vs decomp rows",
            shape.name
        );
        assert_eq!(
            full_rows, dfq_rows,
            "{}: query_sql vs df-parquet rows",
            shape.name
        );
        assert_eq!(
            full_rows, mem_rows,
            "{}: query_sql vs memtable rows",
            shape.name
        );
        assert_eq!(
            full_rows, raw_rows,
            "{}: query_sql vs raw-decode rows",
            shape.name
        );

        eprintln!(
            "[sql-diag] {:<16} {} {} {} {} {}   {}",
            shape.name,
            diag_common::fmt(full_p50),
            diag_common::fmt(pp_p50),
            diag_common::fmt(ex_p50),
            diag_common::fmt(dfq_p50),
            diag_common::fmt(mem_p50),
            fmt_count(full_rows),
        );
        eprintln!(
            "[sql-diag] {:<16} {} {} {} {} {}   (mean; raw-decode floor below)",
            "",
            diag_common::fmt(full_mean),
            diag_common::fmt(Duration::ZERO),
            diag_common::fmt(Duration::ZERO),
            diag_common::fmt(dfq_mean),
            diag_common::fmt(mem_mean),
        );
        eprintln!(
            "[sql-diag] {:<16} raw parquet-rs decode floor: p50 {} / mean {}",
            "",
            diag_common::fmt(raw_p50),
            diag_common::fmt(raw_mean),
        );
        eprintln!();
    }

    eprintln!(
        "[sql-diag] read: (query_sql − DataFusion/parquet)=provider overhead; \
         (DataFusion/parquet − MemTable)=parquet decode; \
         (MemTable − raw)=executor+materialize; raw=intrinsic decode."
    );
}
