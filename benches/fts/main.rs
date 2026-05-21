//! FTS bench bundle (infino-only). Wraps the superfile (1M docs) and
//! supertable (10M docs) FTS benches in a single criterion binary so
//! the topic has one `[[bench]]` stanza in `Cargo.toml`.
//!
//! Infino-only timing and correctness — no third-party crates in
//! the dependency graph of these benches.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench fts                                    # all FTS benches
//! cargo bench --bench fts -- superfile_fts_build             # only superfile ingest
//! cargo bench --bench fts -- superfile_fts_search            # only superfile search
//! cargo bench --bench fts -- supertable_fts_build            # only supertable ingest
//! cargo bench --bench fts -- supertable_fts_search           # only supertable search
//! cargo bench --bench fts -- _build                          # both ingest groups
//! cargo bench --bench fts -- _search                         # both search groups
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench fts       # rewrite README sections
//! ```

#[path = "../utils/markdown.rs"]
mod markdown;

#[path = "superfile.rs"]
mod superfile;
#[path = "supertable.rs"]
mod supertable;

criterion::criterion_main!(superfile::benches, supertable::benches);
