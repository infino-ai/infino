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

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use infino::superfile::builder::FtsConfig;
use infino::superfile::fts::reader::BoolMode;
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::default_tokenizer;

/// Segment count — enough for the fan-out cost to dominate any single
/// kernel, while keeping the fixture build in the low seconds.
const SEGMENTS: usize = 64;
/// Docs per segment — small enough that per-segment scoring is cheap,
/// so the orchestration layers stand out in the deltas.
const DOCS_PER_SEGMENT: usize = 2048;
/// Timed iterations per shape (p50 reported).
const ITERS: usize = 100;
/// Rayon pool width for the fixture's reader/writer pools.
const POOL_THREADS: usize = 8;
/// Top-k for every timed query.
const K: usize = 10;

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
        }],
        vec![],
        Some(default_tokenizer()),
    )
    .expect("valid options")
    .with_writer_pool(Arc::clone(&pool))
    .with_reader_pool(pool)
}

/// Segment `seg` gets `DOCS_PER_SEGMENT` docs: every title contains
/// the all-segment term `common`; doc 0 of segment 0 additionally
/// carries the planted `uniqueterm`.
fn build_batch(seg: usize, schema: Arc<Schema>) -> RecordBatch {
    let titles: Vec<String> = (0..DOCS_PER_SEGMENT)
        .map(|i| {
            if seg == 0 && i == 0 {
                "common uniqueterm topic".to_string()
            } else {
                format!("common topic {} variant", seg * DOCS_PER_SEGMENT + i)
            }
        })
        .collect();
    let arr = LargeStringArray::from(titles.iter().map(String::as_str).collect::<Vec<_>>());
    RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("batch")
}

fn build_supertable() -> Supertable {
    let st = Supertable::create(options_title_only()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    for seg in 0..SEGMENTS {
        w.append(&build_batch(seg, schema.clone())).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

fn p50(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[(samples.len() - 1) / 2]
}

fn time_p50(mut f: impl FnMut()) -> Duration {
    // One untimed warmup so lazy per-table state (runtime, caches)
    // isn't billed to the first sample.
    f();
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        samples.push(t.elapsed());
    }
    p50(&mut samples)
}

#[test]
#[ignore = "perf microbench, not a correctness gate"]
fn fanout_floor_decomposition() {
    let st = build_supertable();
    let reader = st.reader();
    // The writer row-shards each commit (cpus/2 shards), so the real
    // segment count is a multiple of the commit count — report it.
    let n_segments = reader.n_superfiles();
    assert!(
        n_segments >= SEGMENTS,
        "expected at least one segment per commit, got {n_segments}"
    );

    // (label, query term, expected to hit?)
    let shapes: &[(&str, &str, bool)] = &[
        ("absent (prune-all floor)", "zzzabsenttoken", false),
        ("unique (floor + 1 kernel)", "uniqueterm", true),
        ("common (floor + full fan-out)", "common", true),
    ];

    println!(
        "\n### Warm fan-out floor — {n_segments} segments ({SEGMENTS} commits × {} docs), k={K}, p50 of {ITERS}\n",
        DOCS_PER_SEGMENT
    );
    println!("| shape | bm25_hits | bm25_search (_id, score) |");
    println!("|-------|----------:|-------------------------:|");

    for &(label, term, expect_hits) in shapes {
        let hits = reader
            .bm25_hits("title", term, K, BoolMode::Or)
            .expect("bm25_hits");
        assert_eq!(
            !hits.is_empty(),
            expect_hits,
            "{label}: unexpected hit set for {term:?}"
        );

        let hits_p50 = time_p50(|| {
            let h = reader
                .bm25_hits("title", term, K, BoolMode::Or)
                .expect("bm25_hits");
            std::hint::black_box(h);
        });
        let search_p50 = time_p50(|| {
            let b = reader
                .bm25_search("title", term, K, BoolMode::Or, Some(&["_id", "score"]))
                .expect("bm25_search");
            std::hint::black_box(b);
        });
        println!(
            "| {label} | {:.1} µs | {:.1} µs |",
            hits_p50.as_secs_f64() * 1e6,
            search_p50.as_secs_f64() * 1e6,
        );
    }
}
