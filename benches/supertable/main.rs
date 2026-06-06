//! Supertable benches over a shared 10M combined supertable: ingest
//! timing + FTS search + vector search.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench supertable_all                         # all
//! cargo bench --bench supertable_all -- supertable_vec       # vector groups
//! cargo bench --bench supertable_all -- supertable_fts       # FTS groups
//! cargo bench --bench supertable_all -- _build               # ingest groups
//! cargo bench --bench supertable_all -- _search              # search groups
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench supertable_all
//! ```

use infino_bench_utils::bench::supertable;
use infino_bench_utils::tiers;

fn main() {
    supertable::benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
    // Statics don't run `Drop` at exit; delete ephemeral remote prefixes here.
    tiers::cleanup_ephemeral();
}
