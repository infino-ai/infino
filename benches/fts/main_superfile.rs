//! Superfile FTS bench (1M docs). Standalone binary.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench fts-superfile                            # all superfile FTS
//! cargo bench --bench fts-superfile -- superfile_fts_build     # ingest only
//! cargo bench --bench fts-superfile -- superfile_fts_search    # search only
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench fts-superfile
//! ```

// Cargo bench binaries don't share sub-modules through the file system the
// way `main.rs` does, so reach the shared `superfile.rs` body via `#[path]`.
// Helpers (corpus, markdown, rss) come from the `infino-bench-utils` crate
// — see `benches/utils/`.
#[path = "superfile.rs"]
mod superfile;

criterion::criterion_main!(superfile::benches);
