// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! See what the library's `tracing` output looks like.
//!
//! The library emits `tracing` events but never installs a subscriber —
//! that is the consumer's job. This example installs the standard
//! `fmt` subscriber, then drives a storage-backed table through the
//! lifecycle paths that carry logs (connect, create, append/commit,
//! search, SQL, reopen, drop). Level is `RUST_LOG`-driven, defaulting
//! to `info`:
//!
//! ```text
//! cargo run --example logging                 # info + warn + error
//! RUST_LOG=infino=debug cargo run --example logging   # + per-op debug
//! RUST_LOG=infino=trace cargo run --example logging   # + query fanout
//! ```
//!
//! The always-on events tell you *what* happened. To see *how long* each
//! append/commit, search, SQL, superfile-open, and object-store fetch
//! took, build with the `detailed-tracing` feature: it compiles in the
//! per-function timing spans, and this example then configures the
//! subscriber to print each span's busy/idle time as the span closes.
//!
//! ```text
//! cargo run --example logging --features detailed-tracing
//! RUST_LOG=infino=trace cargo run --example logging --features detailed-tracing
//! ```
//!
//! Without the feature the spans compile to nothing, so there is no
//! timing output (and no runtime cost) regardless of `RUST_LOG`.
//!
//! Backed by a temp directory (LocalFs) rather than `memory://`, so the
//! storage-branch logs (durable create/drop, commit publish, the
//! open-time recovery + GC sweeps) actually fire.

use std::sync::Arc;

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{BoolMode, IndexSpec, connect};
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // With `detailed-tracing` on, emit each span's busy/idle time when it
    // closes — the per-op timings. With the feature off the spans aren't
    // compiled at all, so there is nothing to report and we ask for none.
    #[cfg(feature = "detailed-tracing")]
    let span_events = FmtSpan::CLOSE;
    #[cfg(not(feature = "detailed-tracing"))]
    let span_events = FmtSpan::NONE;

    // `RUST_LOG` picks the level/target filter; fall back to `info` so a
    // bare `cargo run --example logging` shows the info/warn/error tier.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_span_events(span_events)
        .with_target(true)
        .init();

    // Durable backend so the storage-branch logs fire (memory:// would
    // skip create/drop-on-storage, commit publish, and the sweeps).
    let dir = tempfile::tempdir()?;
    let uri = dir.path().to_str().expect("utf8 tempdir path");

    let db = connect(uri)?;

    let schema = Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]));
    let docs = db.create_table("docs", schema.clone(), IndexSpec::new().fts("title"))?;

    let titles = ["the quick brown fox", "a lazy sleeping dog"];
    docs.append(&RecordBatch::try_new(
        schema,
        vec![Arc::new(LargeStringArray::from(titles.to_vec()))],
    )?)?;

    let _ = docs.bm25_search("title", "fox", 10, BoolMode::Or, Some(&["_id", "title"]))?;
    let _ = db.query_sql("SELECT _id, title FROM docs ORDER BY _id")?;

    // Reopen from storage (drives the open-time recovery + GC sweeps),
    // then drop with purge to exercise the destructive lifecycle log.
    let _reopened = db.open_table("docs")?;
    db.drop_table("docs", true)?;

    Ok(())
}
