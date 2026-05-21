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

#[path = "../utils/markdown.rs"]
mod markdown;

#[path = "superfile.rs"]
mod superfile;

criterion::criterion_main!(superfile::benches);
