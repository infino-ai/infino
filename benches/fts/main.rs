//! FTS superfile bench bundle (infino-only). Supertable FTS benches
//! live in `benches/supertable/main.rs`, where they share one combined
//! 10M-row supertable with the vector supertable benches.
//!
//! Infino-only timing and correctness — no third-party crates in
//! the dependency graph of these benches.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench superfile_fts                         # 1M superfile FTS benches
//! cargo bench --bench superfile_fts -- superfile_fts_build  # only superfile ingest
//! cargo bench --bench superfile_fts -- superfile_fts_search # only superfile search
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench superfile_fts
//! ```

use infino_bench_utils::fts_superfile;

criterion::criterion_main!(fts_superfile::benches);
