//! Unified object-store cold/warm bench (infino-only). Stands an
//! in-process `s3s-fs` server in for AWS S3 and measures the
//! lazy cold-open + first-search path over the network for a
//! single superfile that carries **both** a vector subsection and
//! an FTS subsection (the consolidated SQL/vector/FTS data layer),
//! plus the warm (mmap-promoted) searches. One `[[bench]]` stanza
//! in `Cargo.toml` so the topic stays self-contained.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --features bench-diagnostics --bench object-store
//! INFINO_REAL_S3_BUCKET=<bucket> cargo bench --features bench-diagnostics --bench object-store
//! INFINO_BENCH_UPDATE_README=1 cargo bench --features bench-diagnostics --bench object-store
//! ```

use infino_bench_utils::unified_object_store;

criterion::criterion_main!(unified_object_store::benches);
