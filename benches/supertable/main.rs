//! Supertable object-store bench bundle (infino-only entry point). Uses
//! Infino's custom benchmark harness directly.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench supertable_all
//! INFINO_BENCH_SUPERTABLE_DOCS=100000 cargo bench --bench supertable_all
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench supertable_all
//! ```

fn main() {
    infino_bench_utils::supertable_bench::run();
}
