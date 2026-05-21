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

use infino_bench_utils::fts_supertable;

criterion::criterion_main!(fts_supertable::benches);
