//! Vector benches over a 1M single superfile. The 10M supertable vector
//! benches live in `benches/supertable/main.rs`.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench superfile_vector                         # all
//! cargo bench --bench superfile_vector -- superfile_vec_build  # ingest only
//! cargo bench --bench superfile_vector -- superfile_vec_search # search only
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench superfile_vector
//! ```

use infino_bench_utils::tiers;
use infino_bench_utils::vector_superfile;

fn main() {
    vector_superfile::benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
    // Statics don't run `Drop` at exit; delete ephemeral remote prefixes here.
    tiers::cleanup_ephemeral();
}
