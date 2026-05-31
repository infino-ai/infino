//! Legacy standalone Supertable FTS bench entrypoint.
//!
//! The canonical 10M supertable bench is now `supertable_all`, which
//! builds one combined text+vector table and runs both FTS and vector
//! groups against it.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench supertable_all -- supertable_fts_search   # canonical FTS search
//! ```

use infino_bench_utils::fts_supertable;

criterion::criterion_main!(fts_supertable::benches);
