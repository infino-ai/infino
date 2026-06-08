// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Scale bench bundle: at-scale pinned-recall assertion runners that
//! need release-profile compilation to finish in seconds rather than
//! minutes. Each `run()` prints single-line summaries per phase to
//! stdout.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --features bench-diagnostics --bench scale
//! cargo bench --features bench-diagnostics --bench scale -- vector_recall
//! ```

#[path = "vector_recall.rs"]
mod vector_recall;

fn main() {
    // `cargo bench --bench scale` forwards harness flags (e.g. `--bench`)
    // to this binary; treat anything starting with `-` as not-a-filter so
    // a bare invocation runs every sub-bench instead of silently matching
    // nothing. A real filter is the first non-flag arg (e.g. `vector_recall`).
    let filter = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .unwrap_or_default();
    let run_all = filter.is_empty();
    let want = |needle: &str| run_all || needle.contains(&filter);

    if want("vector_recall") {
        eprintln!("[scale] --- vector_recall ---");
        vector_recall::run();
    }
}
