//! Supertable bench bundle (infino-only).
//!
//! Flow: `corpus` (synthetic stream) → `ingest` (object storage) →
//! `bench/supertable` (ingest timing + FTS search + vector search).
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench supertable_all                         # all supertable benches
//! cargo bench --bench supertable_all -- supertable_vec       # vector groups
//! cargo bench --bench supertable_all -- supertable_fts       # FTS groups
//! cargo bench --bench supertable_all -- _build               # shared ingest group
//! cargo bench --bench supertable_all -- _search              # search groups
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench supertable_all
//! ```

use infino_bench_utils::bench::supertable;

criterion::criterion_main!(supertable::benches);
