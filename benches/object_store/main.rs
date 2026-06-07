// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Unified object-store cold bench (infino-only). Stands an
//! in-process `s3s-fs` server in for AWS S3 and measures the
//! lazy cold-open + first-search path over the network for a
//! single superfile that carries **both** a vector subsection and
//! an FTS subsection (the consolidated SQL/vector/FTS data layer).
//! The implementation — including the S3 latency + cost models and
//! the `INFINO_DIAG_*` deep-dive breakdowns — lives in
//! [`infino_bench_utils::unified_object_store`].
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --features bench-diagnostics --bench object-store
//! INFINO_REAL_S3_BUCKET=<bucket> cargo bench --features bench-diagnostics --bench object-store
//! INFINO_BENCH_UPDATE_README=1 cargo bench --features bench-diagnostics --bench object-store
//!
//! # Deep-dive diagnostics (request-shape + S3 latency-model breakdown):
//! INFINO_DIAG_COLD_PATH=1          cargo bench --features bench-diagnostics --bench object-store
//! INFINO_DIAG_REAL_S3=1            cargo bench --features bench-diagnostics --bench object-store
//! INFINO_DIAG_REAL_S3_SUPERTABLE=1 cargo bench --features bench-diagnostics --bench object-store
//! INFINO_DIAG_QUERY_SQL_OVERHEAD=1 cargo bench --features bench-diagnostics --bench object-store
//! ```

fn main() {
    infino_bench_utils::unified_object_store::run();
}
