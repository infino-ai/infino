//! Vector bench bundle (infino-only). Wraps superfile (1M × 384) and
//! supertable (10M × 384) vector benches in a single criterion binary
//! so the topic has one `[[bench]]` stanza in `Cargo.toml`.
//!
//! Infino-only timing and correctness — no third-party crates in
//! the dependency graph of these benches.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench vector                                  # all vector benches
//! cargo bench --bench vector -- superfile_vec_build           # only superfile ingest
//! cargo bench --bench vector -- superfile_vec_search          # only superfile search
//! cargo bench --bench vector -- supertable_vec_build          # only supertable ingest
//! cargo bench --bench vector -- supertable_vec_search         # only supertable search
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench vector     # rewrite README sections
//! ```

#[path = "../utils/markdown.rs"]
mod markdown;

#[path = "superfile.rs"]
mod superfile;
#[path = "supertable.rs"]
mod supertable;

criterion::criterion_main!(superfile::benches, supertable::benches);
