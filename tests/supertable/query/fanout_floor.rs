// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Fan-out floor microbench — decomposes the supertable-vs-superfile
//! warm-latency gap into its per-layer costs.
//!
//! A warm supertable query pays, on top of the per-segment kernel
//! work a superfile query would pay anyway:
//!
//!   1. the sync→async bridge + manifest pin,
//!   2. segment selection (bloom / term-range prune walk),
//!   3. the dispatch fan-out (one tokio task per kept segment:
//!      reader-cache lookup, kernel, tag, tombstone filter),
//!   4. the cross-segment top-k merge,
//!   5. (row-returning paths) hit→row resolution.
//!
//! The three query shapes here isolate those layers on a warm
//! in-memory table:
//!
//!   * `absent`  — term in no segment: bloom prunes everything, so the
//!     measurement is layers 1+2 alone (the pure orchestration floor).
//!   * `unique`  — term planted in exactly one segment: floor + one
//!     kernel + merge.
//!   * `common`  — term in every segment: floor + a full `SEGMENTS`-
//!     wide fan-out.
//!
//! Each shape is timed for `bm25_hits` (kernel surface only) and
//! `bm25_search` with an `["_id", "score"]` projection (adds the hit→
//! row resolution wave), so resolve cost falls out by subtraction.
//!
//! Gated `#[ignore]` — a timing probe, not a correctness gate. Run:
//!
//! ```text
//! cargo test --release --features test-helpers --test supertable \
//!   query::fanout_floor -- --ignored --nocapture
//! ```

#![deny(clippy::unwrap_used)]

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::reader::BoolMode,
    },
    supertable::{
        Supertable, SupertableOptions,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        storage::LocalFsStorageProvider,
    },
    test_helpers::{decimal128_id_field, decimal128_ids, default_tokenizer},
};
use tempfile::TempDir;

/// Commits — enough for the fan-out cost to dominate any single
/// kernel, while keeping the fixture build in the low seconds.
/// Override segment shape via `FLOOR_COMMITS` / `FLOOR_DOCS` to probe
/// fat-segment behavior (e.g. `FLOOR_COMMITS=2 FLOOR_DOCS=200000`
/// approximates production segment sizes, isolating kernel-init and
/// resolve costs that scale with segment size rather than count).
const SEGMENTS: usize = 64;
/// Docs per commit — small enough that per-segment scoring is cheap,
/// so the orchestration layers stand out in the deltas.
const DOCS_PER_SEGMENT: usize = 2048;

fn commits() -> usize {
    std::env::var("FLOOR_COMMITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(SEGMENTS)
}

fn docs_per_commit() -> usize {
    std::env::var("FLOOR_DOCS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DOCS_PER_SEGMENT)
}
/// Timed iterations per shape (p50 reported).
const ITERS: usize = 100;
/// Rayon pool width for the fixture's reader/writer pools.
const POOL_THREADS: usize = 8;
/// Top-k for every timed query.
const K: usize = 10;
/// Disk-cache budget for the mmap-mode consumer — far above any
/// fixture's index size so eviction never interferes with timing.
const MMAP_CACHE_BUDGET_BYTES: u64 = 8 << 30;
/// Upper bound on waiting for full mmap promotion of the mmap-mode
/// consumer; generous because promotion downloads every segment.
const WARM_PROMOTION_TIMEOUT: Duration = Duration::from_secs(600);
/// Position of `minflt` in `/proc/<pid>/task/<tid>/stat`, counting
/// fields after the `)` that closes `comm` (state ppid pgrp session
/// tty tpgid flags **minflt** ...).
const MINFLT_AFTER_PAREN: usize = 7;

fn options_title_only() -> SupertableOptions {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(POOL_THREADS)
            .build()
            .expect("pool"),
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![],
        Some(default_tokenizer()),
    )
    .expect("valid options")
    .with_writer_pool(Arc::clone(&pool))
    .with_reader_pool(pool)
}

/// Commit `seg` gets `docs_per_commit()` docs: every title contains
/// the all-commit term `common`; doc 0 of commit 0 additionally
/// carries the planted `uniqueterm`.
fn build_batch(seg: usize, schema: Arc<Schema>) -> RecordBatch {
    let n = docs_per_commit();
    let titles: Vec<String> = (0..n)
        .map(|i| {
            if seg == 0 && i == 0 {
                "common uniqueterm topic".to_string()
            } else {
                format!("common topic {} variant", seg * n + i)
            }
        })
        .collect();
    let arr = LargeStringArray::from(titles.iter().map(String::as_str).collect::<Vec<_>>());
    RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("batch")
}

/// `FLOOR_MMAP=1` switches the fixture from the in-memory reader
/// store to the production warm path: commit to localfs storage, drop
/// the producer, open a consumer through a fresh disk cache, prewarm
/// and wait for mmap promotion — the exact reader state the README's
/// supertable warm battery measures. Comparing the two modes on
/// identical data isolates the in-memory-vs-mmap reader axis.
fn mmap_mode() -> bool {
    std::env::var("FLOOR_MMAP").is_ok_and(|v| v == "1")
}

fn build_supertable() -> (Supertable, Vec<TempDir>) {
    if !mmap_mode() {
        let st = Supertable::create(options_title_only()).expect("create");
        let schema = st.options().schema.clone();
        let mut w = st.writer().expect("writer");
        for seg in 0..commits() {
            w.append(&build_batch(seg, schema.clone())).expect("append");
            w.commit().expect("commit");
        }
        drop(w);
        return (st, Vec::new());
    }

    // Producer: commit every segment to localfs object storage.
    let store_dir = TempDir::new().expect("storage tempdir");
    let storage: Arc<dyn infino::supertable::storage::StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("localfs"));
    let producer = Supertable::create(options_title_only().with_storage(Arc::clone(&storage)))
        .expect("create on storage");
    let schema = producer.options().schema.clone();
    let mut w = producer.writer().expect("writer");
    for seg in 0..commits() {
        w.append(&build_batch(seg, schema.clone())).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    drop(producer);

    // Consumer: fresh disk cache, same options (options-hash must
    // match), then force full mmap promotion — the README battery's
    // warm reader state.
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cfg = DiskCacheConfig {
        cache_root: cache_dir.path().to_path_buf(),
        disk_budget_bytes: MMAP_CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: false,
        ..Default::default()
    };
    let cache = DiskCacheStore::new_unpinned(Arc::clone(&storage), cfg).expect("disk cache");
    let consumer = Supertable::open(
        options_title_only()
            .with_storage(storage)
            .with_disk_cache(cache),
    )
    .expect("open consumer");
    let _ = consumer
        .reader()
        .bm25_hits("title", "common", K, BoolMode::Or)
        .expect("prewarm");
    consumer
        .wait_until_warm(WARM_PROMOTION_TIMEOUT)
        .expect("mmap promotion");
    (consumer, vec![store_dir, cache_dir])
}

/// The superfile-tier control: ONE segment holding the exact same
/// titles the supertable fixture splits across `commits() × shards`
/// segments. Timing the raw kernel against this quantifies the true
/// superfile→supertable ratio on identical data — no corpus, scale,
/// or measurement confounds.
fn build_one_superfile() -> SuperfileReader {
    let schema = Arc::new(Schema::new(vec![
        decimal128_id_field("doc_id"),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    let n = docs_per_commit();
    for seg in 0..commits() {
        let titles: Vec<String> = (0..n)
            .map(|i| {
                if seg == 0 && i == 0 {
                    "common uniqueterm topic".to_string()
                } else {
                    format!("common topic {} variant", seg * n + i)
                }
            })
            .collect();
        let ids = decimal128_ids((0..n as u64).map(|i| (seg * n) as u64 + i));
        let arr = LargeStringArray::from(titles.iter().map(String::as_str).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(arr)])
            .expect("superfile batch");
        b.add_batch(&batch, &[]).expect("add_batch");
    }
    let bytes = Bytes::from(b.finish().expect("finish superfile"));
    SuperfileReader::open(bytes).expect("open superfile")
}

fn p50(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[(samples.len() - 1) / 2]
}

/// Process-wide minor page faults, summed over every live thread's
/// `/proc/self/task/<tid>/stat`. `/proc/self/stat` alone counts only
/// the main thread, which parks in `block_on` while the query work —
/// and any faults — happen on tokio/rayon workers. Counted per timed
/// window: a recurring per-iteration fault count in mmap mode means
/// the kernel keeps un-wiring pages between queries; ~0 after the
/// warmup touch means the latency lives in CPU, not paging. (Threads
/// that exit take their counts with them — fine for a probe whose
/// pools live for the whole run.)
fn minflt() -> u64 {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0;
    };
    tasks
        .flatten()
        .filter_map(|t| std::fs::read_to_string(t.path().join("stat")).ok())
        .map(|stat| {
            // `comm` can contain spaces; field positions are only
            // stable after the closing ')'.
            let after = stat.rsplit_once(')').map(|(_, rest)| rest).unwrap_or("");
            after
                .split_whitespace()
                .nth(MINFLT_AFTER_PAREN)
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0)
        })
        .sum()
}

fn time_p50(mut f: impl FnMut()) -> (Duration, u64) {
    // One untimed warmup so lazy per-table state (runtime, caches)
    // isn't billed to the first sample.
    f();
    let faults_before = minflt();
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        samples.push(t.elapsed());
    }
    let faults_per_iter = minflt().saturating_sub(faults_before) / ITERS as u64;
    (p50(&mut samples), faults_per_iter)
}

#[test]
#[ignore = "perf microbench, not a correctness gate"]
fn fanout_floor_decomposition() {
    let (st, _guards) = build_supertable();
    let reader = st.reader();
    // The writer row-shards each commit (cpus/2 shards), so the real
    // segment count is a multiple of the commit count — report it.
    let n_segments = reader.n_superfiles();
    assert!(
        n_segments >= commits(),
        "expected at least one segment per commit, got {n_segments}"
    );

    // Superfile-tier control: ONE segment, same corpus, raw kernel.
    let superfile = build_one_superfile();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("control runtime");

    // (label, query term, expected to hit?)
    let shapes: &[(&str, &str, bool)] = &[
        ("absent (prune-all floor)", "zzzabsenttoken", false),
        ("unique (floor + 1 kernel)", "uniqueterm", true),
        ("common (floor + full fan-out)", "common", true),
    ];

    println!(
        "\n### Warm fan-out floor — {n_segments} segments ({} commits × {} docs), k={K}, p50 of {ITERS}, readers: {}\n",
        commits(),
        docs_per_commit(),
        if mmap_mode() {
            "mmap-promoted (disk cache)"
        } else {
            "in-memory"
        },
    );
    println!(
        "| shape | superfile (1 seg) | bm25_hits | search bare (id+score) | search +title | flt/it (sf/hits/bare/+title) |"
    );
    println!(
        "|-------|------------------:|----------:|-----------------------:|--------------:|------------------------------|"
    );

    // `FLOOR_SHAPE=<prefix>` runs only matching shapes — lets a
    // syscall/profiler census attribute counts to one query loop.
    let shape_filter = std::env::var("FLOOR_SHAPE").ok();

    for &(label, term, expect_hits) in shapes {
        if let Some(f) = &shape_filter
            && !label.starts_with(f.as_str())
        {
            continue;
        }
        let hits = reader
            .bm25_hits("title", term, K, BoolMode::Or)
            .expect("bm25_hits");
        assert_eq!(
            !hits.is_empty(),
            expect_hits,
            "{label}: unexpected hit set for {term:?}"
        );

        let (superfile_p50, superfile_flt) = time_p50(|| {
            let h = rt
                .block_on(superfile.bm25_hits_async("title", term, K, BoolMode::Or))
                .expect("superfile bm25");
            std::hint::black_box(h);
        });
        let (hits_p50, hits_flt) = time_p50(|| {
            let h = reader
                .bm25_hits("title", term, K, BoolMode::Or)
                .expect("bm25_hits");
            std::hint::black_box(h);
        });
        // Bare projection (`None`) = the public id+score contract —
        // arithmetic `_id` resolve, no Parquet involvement.
        let (ids_p50, ids_flt) = time_p50(|| {
            let b = reader
                .bm25_search("title", term, K, BoolMode::Or, None)
                .expect("bm25_search");
            std::hint::black_box(b);
        });
        // Naming a scalar column = the fetch phase; the delta vs the
        // bare column is the row-materialization cost, which scales
        // with segment/page size, not segment count.
        let (full_p50, full_flt) = time_p50(|| {
            let b = reader
                .bm25_search(
                    "title",
                    term,
                    K,
                    BoolMode::Or,
                    Some(&["_id", "title", "score"]),
                )
                .expect("bm25_search");
            std::hint::black_box(b);
        });
        println!(
            "| {label} | {:.1} µs | {:.1} µs | {:.1} µs | {:.1} µs | {superfile_flt}/{hits_flt}/{ids_flt}/{full_flt} |",
            superfile_p50.as_secs_f64() * 1e6,
            hits_p50.as_secs_f64() * 1e6,
            ids_p50.as_secs_f64() * 1e6,
            full_p50.as_secs_f64() * 1e6,
        );
    }
}
