// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Vector superfile bench bundle (infino-only). Uses Infino's custom
//! benchmark harness directly.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench superfile_vector                         # 1M superfile vector benches
//! cargo bench --bench superfile_vector -- superfile_vec_build  # only superfile ingest
//! cargo bench --bench superfile_vector -- superfile_vec_search # only superfile search
//! INFINO_BENCH_UPDATE_README=1 cargo bench --bench superfile_vector
//! ```

use infino_bench_utils::vector_superfile;

fn main() {
    vector_superfile::run();
}
