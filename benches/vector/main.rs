//! Vector superfile bench bundle (infino-only). Supertable vector
//! benches live in `benches/supertable/main.rs`, where they share one
//! combined 10M-row supertable with the FTS supertable benches.
//!
//! Infino-only timing and correctness — no third-party crates in
//! the dependency graph of these benches.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench superfile_vector                         # 1M superfile vector benches
//! cargo bench --bench superfile_vector -- superfile_vec_build  # only superfile ingest
//! cargo bench --bench superfile_vector -- superfile_vec_search # only superfile search
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench superfile_vector
//! ```

use infino_bench_utils::vector_superfile;

criterion::criterion_main!(vector_superfile::benches);
