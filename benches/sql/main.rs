// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! SQL bench bundle (infino-only entry point). Uses Infino's custom
//! benchmark harness directly.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench sql
//! INFINO_BENCH_DOC_COUNT=100000 cargo bench --bench sql
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench sql
//! ```

fn main() {
    infino_bench_utils::sql_bench::run();
}
