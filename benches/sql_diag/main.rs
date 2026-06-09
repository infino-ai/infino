// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! SQL scan-decomposition diagnostic (infino-only entry point). Uses
//! Infino's custom benchmark harness directly to localize where SQL
//! scan/filter latency goes.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench sql-diag
//! INFINO_SQL_DIAG=tvf cargo bench --bench sql-diag
//! ```

fn main() {
    infino_bench_utils::sql_diag::run();
}
