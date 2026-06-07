// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable object-store bench (infino-only entry point).
//!
//! Multi-segment ingest to object storage at the supertable scale
//! (`INFINO_BENCH_SUPERTABLE_DOCS`, default 10M), built through the
//! production `SupertableWriter::append` + `commit` path. Three index
//! shapes are measured for apples-to-apples comparison against
//! single-modality peers: FTS-only, vector-only, and combined FTS +
//! vector.
//!
//! **Real AWS S3 only.** The multi-commit build relies on conditional
//! `If-Match` PUTs that the `s3s-fs` emulator does not implement, and a
//! local filesystem backend would not measure object-store behavior, so
//! this bench requires `INFINO_REAL_S3_BUCKET` (+ AWS creds) and exits with
//! a message otherwise. Every object the run writes lands under one unique
//! prefix per shape, all of which are deleted before the runner returns.
//!
//! ## Invocation
//!
//! ```text
//! INFINO_REAL_S3_BUCKET=my-bench-bucket cargo bench --bench supertable_all
//! INFINO_REAL_S3_BUCKET=my-bench-bucket INFINO_BENCH_SUPERTABLE_DOCS=100000 cargo bench --bench supertable_all
//! INFINO_REAL_S3_BUCKET=my-bench-bucket INFINO_BENCH_UPDATE_README=1 cargo bench --bench supertable_all
//! ```

use std::time::{Duration, Instant};

use crate::ingest::supertable::{self, IngestResult, Modality};
use crate::markdown::{fmt_count, fmt_throughput, fmt_time};
use crate::report::{Better, Block, Cell, Report, Section, metric, text};
use crate::rss::{self, PeakSampler, RssStats};

struct ShapeResult {
    label: &'static str,
    built: IngestResult,
    wall: Duration,
    rss: RssStats,
}

/// Build one supertable shape on object storage, timed + RSS-sampled.
fn timed_build(label: &'static str, modality: Modality) -> ShapeResult {
    let sampler = PeakSampler::start_default();
    let t0 = Instant::now();
    let built = supertable::build_on_storage(modality);
    let wall = t0.elapsed();
    let rss = sampler.stop_stats();
    ShapeResult {
        label,
        built,
        wall,
        rss,
    }
}

fn ingest_row(n_docs: usize, shape: &ShapeResult) -> Vec<Cell> {
    let secs = shape.wall.as_secs_f64();
    let ns = secs * 1e9;
    let thr = n_docs as f64 / secs;
    vec![
        text(shape.label),
        metric(ns, fmt_time(ns), Better::Lower),
        metric(thr, fmt_throughput(thr), Better::Higher),
        text(fmt_count(shape.built.n_superfiles)),
        metric(
            shape.rss.peak_rss_bytes as f64,
            rss::fmt_bytes(shape.rss.peak_rss_bytes),
            Better::Lower,
        ),
        metric(
            shape.rss.median_rss_bytes as f64,
            rss::fmt_bytes(shape.rss.median_rss_bytes),
            Better::Lower,
        ),
        metric(
            shape.rss.p90_rss_bytes as f64,
            rss::fmt_bytes(shape.rss.p90_rss_bytes),
            Better::Lower,
        ),
    ]
}

pub fn run() {
    // Pre-flight: this bench only runs against real S3 (see module docs and
    // `tiers::supertable_storage_fixture`). Fail fast with a clear message
    // instead of a panic deep inside the first build.
    if crate::tiers::real_s3_bucket_env().is_none() {
        eprintln!("[supertable] skipped: {}", crate::tiers::SUPERTABLE_REQUIRES_REAL_S3);
        return;
    }

    let n_docs = supertable::n_docs();
    eprintln!(
        "[supertable] ingesting {} docs ({} commits) per shape to object storage...",
        fmt_count(n_docs),
        supertable::N_COMMIT_CHUNKS
    );

    let shapes = [
        timed_build("FTS-only", Modality::Fts),
        timed_build("vector-only", Modality::Vector),
        timed_build("combined FTS + vector", Modality::Combined),
    ];
    let storage_label = shapes[0].built.storage_label;

    let rows: Vec<Vec<Cell>> = shapes.iter().map(|s| ingest_row(n_docs, s)).collect();

    let mut report = Report::load("supertable");
    report.emit(&Section {
        anchor: "bench/supertable/ingest".into(),
        title: format!(
            "Supertable — ingest, multi-segment / object-store ({} docs × dim={}, {} commits, {storage_label})",
            fmt_count(n_docs),
            crate::corpus::DIM,
            supertable::N_COMMIT_CHUNKS
        ),
        note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). \
               Rows are the three index shapes built from the same seeded corpus, so each is directly \
               comparable to its single-modality peer. Throughput is rows/s; `Superfiles` is the \
               committed segment count. Δ is vs the previous run."
            .into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Shape".into(),
                "Time".into(),
                "Throughput".into(),
                "Superfiles".into(),
                "Peak RSS".into(),
                "Median RSS".into(),
                "P90 RSS".into(),
            ],
            rows,
        }],
    });
    report.save();

    // Delete every prefix this run wrote to real S3 so it accrues no
    // ongoing storage cost. Each shape built under its own unique prefix.
    for shape in &shapes {
        if let Some(cleanup) = &shape.built.cleanup {
            crate::tiers::cleanup_real_s3_prefix(cleanup);
        }
    }
}
