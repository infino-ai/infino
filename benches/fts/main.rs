//! FTS benches over a 1M single superfile. The 10M supertable FTS benches
//! live in `benches/supertable/main.rs`.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench superfile_fts                         # all
//! cargo bench --bench superfile_fts -- superfile_fts_build  # ingest only
//! cargo bench --bench superfile_fts -- superfile_fts_search # search only
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench superfile_fts
//! ```

use infino_bench_utils::fts_superfile;
use infino_bench_utils::tiers;

fn main() {
    fts_superfile::benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
    // Statics don't run `Drop` at exit; delete ephemeral remote prefixes here.
    tiers::cleanup_ephemeral();
}
