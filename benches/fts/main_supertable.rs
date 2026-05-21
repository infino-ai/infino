//! Supertable FTS bench (10M docs). Standalone binary.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench fts-supertable                            # all supertable FTS
//! cargo bench --bench fts-supertable -- supertable_fts_build    # ingest only
//! cargo bench --bench fts-supertable -- supertable_fts_search   # search only
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench fts-supertable
//! ```

// Cargo bench binaries don't share sub-modules through the file system the
// way `main.rs` does, so reach the shared `supertable.rs` body via `#[path]`.
// Helpers (corpus, markdown, rss) come from the `infino-bench-utils` crate
// — see `benches/utils/`.
#[path = "supertable.rs"]
mod supertable;

criterion::criterion_main!(supertable::benches);
