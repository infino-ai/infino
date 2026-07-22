// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS query-path diagnostic — splits the scored top-k path into the FTS
//! kernel vs the supertable `_id`-resolution + result assembly.
//!
//! The serving path (`bm25_search(.., None)`) is *kernel + resolve*: the
//! kernel scores and returns superfile-local hits; resolution turns each
//! hit into its stable `_id` and builds the Arrow batch. The kernel alone
//! is `bm25_hits`. So:
//!
//!   resolve/assembly = full (`bm25_search`) − kernel (`bm25_hits`)
//!
//! Resolution scales with the number of hits returned (≤ k) and sits
//! *above* the FTS kernel, so kernel-side scoring changes can't move it.
//! At large k this diagnostic shows whether the top-k cost lives in the
//! kernel or in resolution — the split that decides where to optimize.
//!
//! Shares the build + config with the SQL diagnostic (see
//! [`crate::diag_common`]): one corpus, one scale knob
//! (`INFINO_BENCH_SUPERTABLE_DOCS`), one iters knob (`INFINO_DIAG_ITERS`).
//!
//! ```text
//! cargo bench -- fts-diag
//! INFINO_BENCH_SUPERTABLE_DOCS=1000000 cargo bench -- fts-diag
//! INFINO_DIAG_ITERS=30 cargo bench -- fts-diag
//! ```

use std::time::Instant;

use infino::superfile::fts::reader::BoolMode;

use crate::{diag_common, markdown::fmt_count};

/// Large-k retrieval — the regime where resolution cost, proportional to
/// hits returned, is most exposed.
const K: usize = 1000;

/// FTS column planted by [`diag_common::build_supertable`].
const COLUMN: &str = "title";

/// One query shape measured across the kernel and full paths.
struct FtsShape {
    name: &'static str,
    query: &'static str,
    mode: BoolMode,
}

/// Shapes chosen to span the two regimes the split matters for: a small
/// intersection (matches ≤ k ⇒ no pruning, every match scored *and*
/// resolved), a large intersection (heavy pruning), and a union.
const SHAPES: &[FtsShape] = &[
    FtsShape {
        name: "single_common",
        query: "term00001",
        mode: BoolMode::Or,
    },
    FtsShape {
        name: "small_and",
        query: "term00500 term01000",
        mode: BoolMode::And,
    },
    FtsShape {
        name: "large_and",
        query: "term00001 term00050",
        mode: BoolMode::And,
    },
    FtsShape {
        name: "union",
        query: "term00050 term00051 term00052",
        mode: BoolMode::Or,
    },
];

pub fn run() {
    let cfg = diag_common::config();
    eprintln!(
        "[fts-diag] kernel vs resolve/assembly split: n_docs={} iters={} k={K} \
         (knobs: INFINO_BENCH_SUPERTABLE_DOCS, INFINO_DIAG_ITERS)",
        fmt_count(cfg.n_docs),
        cfg.iters,
    );

    eprintln!("[fts-diag] building supertable...");
    let build_t0 = Instant::now();
    let (table, _batches) = diag_common::build_supertable(&cfg);
    let reader = table.reader();
    eprintln!(
        "[fts-diag] built in {:.1}s ({} superfile(s) after optimize)",
        build_t0.elapsed().as_secs_f64(),
        reader.manifest().superfiles.len(),
    );

    // Warm both paths for every shape (cache-hot before timing).
    for s in SHAPES {
        let _ = reader
            .bm25_search(COLUMN, s.query, K, s.mode, None)
            .expect("warm-up bm25_search");
    }

    // Warm the count path too.
    for s in SHAPES {
        let _ = reader
            .count(COLUMN, s.query, s.mode)
            .expect("warm-up count");
    }

    // Decompose the scored path into three additive layers:
    //   count  = posting traversal + block decode (no score, no heap)
    //   kernel = count + BM25 scoring + top-k heap   (= bm25_hits)
    //   full   = kernel + _id-resolution + result assembly (= bm25_search)
    // so score+heap = kernel − count, and resolve = full − kernel. (At
    // k=1000 over a large superfile the scored path prunes little, so
    // count is a fair traverse/decode floor for it.)
    eprintln!();
    eprintln!(
        "[fts-diag] {:<15}{:>8}{:>12}{:>12}{:>12}{:>13}{:>12}",
        "shape", "hits", "count", "kernel", "full", "score+heap", "resolve"
    );
    for s in SHAPES {
        let hits = reader
            .bm25_hits(COLUMN, s.query, K, s.mode)
            .expect("bm25_hits")
            .len();

        let mut count = Vec::with_capacity(cfg.iters);
        for _ in 0..cfg.iters {
            let t = Instant::now();
            let out = reader.count(COLUMN, s.query, s.mode).expect("count");
            count.push(t.elapsed());
            std::hint::black_box(out);
        }

        let mut kernel = Vec::with_capacity(cfg.iters);
        for _ in 0..cfg.iters {
            let t = Instant::now();
            let out = reader
                .bm25_hits(COLUMN, s.query, K, s.mode)
                .expect("kernel bm25_hits");
            kernel.push(t.elapsed());
            std::hint::black_box(out);
        }

        let mut full = Vec::with_capacity(cfg.iters);
        for _ in 0..cfg.iters {
            let t = Instant::now();
            let out = reader
                .bm25_search(COLUMN, s.query, K, s.mode, None)
                .expect("full bm25_search");
            full.push(t.elapsed());
            std::hint::black_box(out);
        }

        let cp = diag_common::percentile(&mut count, 50);
        let kp = diag_common::percentile(&mut kernel, 50);
        let fp = diag_common::percentile(&mut full, 50);
        let score_heap = kp.saturating_sub(cp);
        let resolve = fp.saturating_sub(kp);
        eprintln!(
            "[fts-diag] {:<15}{:>8}{:>12}{:>12}{:>12}{:>13}{:>12}",
            s.name,
            hits,
            diag_common::fmt(cp),
            diag_common::fmt(kp),
            diag_common::fmt(fp),
            diag_common::fmt(score_heap),
            diag_common::fmt(resolve),
        );
    }
}
