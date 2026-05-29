//! Vector object-store cold/warm bench (infino-only). Stands an
//! in-process `s3s-fs` server in for AWS S3 and measures the Plan
//! 013 lazy cold-open + first-search path over the network, plus
//! the warm (mmap-promoted) search. One `[[bench]]` stanza in
//! `Cargo.toml` so the topic stays self-contained.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench vector-object-store
//! INFINO_BENCH_FULL=1 cargo bench --bench vector-object-store   # 1M-doc headline row
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench vector-object-store
//! ```

use infino_bench_utils::vector_object_store;

criterion::criterion_main!(vector_object_store::benches);
