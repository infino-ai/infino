// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! With tier selection switched off in the host configuration, a ranked
//! query reads no tier and resolves nothing from the index, and answers
//! as the table without tiers does. Its own binary, because the host
//! configuration is read once per process.
#![deny(clippy::unwrap_used)]

#[path = "supertable/query/tier_harness.rs"]
mod tier_harness;

use tempfile::TempDir;
use tier_harness::*;

#[test]
fn switching_tier_selection_off_reads_no_tier_and_answers_the_same() {
    let config = TempDir::new().expect("tempdir");
    std::fs::create_dir_all(config.path().join("infino")).expect("config dir");
    std::fs::write(
        config.path().join("infino/config.yaml"),
        "diagnostics:\n  disable_tier_selection: true\n",
    )
    .expect("config file");
    // The host may already point `XDG_CONFIG_HOME` somewhere (CI runners
    // do); the configuration is read once per process, below, so pointing
    // it at the test's own directory first is all the ownership needed.
    // SAFETY: this is the only test in the binary and nothing else has
    // started yet, so no other thread reads the environment.
    unsafe { std::env::set_var("XDG_CONFIG_HOME", config.path()) };
    assert!(infino::config::global().diagnostics.disable_tier_selection);
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table(&with_dir, true);
    let without = table(&without_dir, false);
    for (query, mode) in queries() {
        for k in [10usize, 100] {
            let (a, stats, _) = run(&with, &query, mode, k);
            let (b, _, _) = run(&without, &query, mode, k);
            assert_eq!(a, b, "{query:?} {mode:?} k={k}: same answer");
            assert_eq!(
                stats.tier_entries, 0,
                "{query:?} {mode:?} k={k}: no tier read"
            );
            assert_eq!(
                stats.superfiles_resolved, 0,
                "{query:?} {mode:?} k={k}: nothing resolved from the index"
            );
        }
    }
}
