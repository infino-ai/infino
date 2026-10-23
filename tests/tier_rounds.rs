// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! The tier merge stops at the configured round cap and still answers as
//! the table without tiers does: what it has read by then bounds every
//! superfile soundly, it just stops refining. Its own binary, because the
//! cap is read from the host configuration once per process.
#![deny(clippy::unwrap_used)]

#[path = "supertable/query/tier_harness.rs"]
mod tier_harness;

use tempfile::TempDir;
use tier_harness::*;

#[test]
fn a_round_cap_of_one_still_answers_the_same() {
    // The host configuration is read once per process: point it at a file
    // that caps the merge at one round before anything in the engine reads
    // it.
    let config = TempDir::new().expect("tempdir");
    std::fs::create_dir_all(config.path().join("infino")).expect("config dir");
    std::fs::write(
        config.path().join("infino/config.yaml"),
        "diagnostics:\n  tier_rounds: 1\n",
    )
    .expect("config file");
    // The host may already point `XDG_CONFIG_HOME` somewhere (CI runners
    // do); the configuration is read once per process, below, so pointing
    // it at the test's own directory first is all the ownership needed.
    // SAFETY: this is the only test in the binary and nothing else has
    // started yet, so no other thread reads the environment.
    unsafe { std::env::set_var("XDG_CONFIG_HOME", config.path()) };
    assert_eq!(
        infino::config::global().diagnostics.tier_rounds,
        1,
        "the cap is in force"
    );
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table(&with_dir, true);
    let without = table(&without_dir, false);
    for (query, mode) in queries() {
        for k in [10usize, 100] {
            let (a, stats, _) = run(&with, &query, mode, k);
            let (b, _, _) = run(&without, &query, mode, k);
            assert_eq!(
                a, b,
                "{query:?} {mode:?} k={k}: same answer under one round"
            );
            // One round reads at most one block of every term's head per
            // superfile (an inline head is a few entries).
            let terms = query.split_whitespace().count() as u64;
            assert!(
                stats.tier_entries <= SUPERFILES as u64 * terms * 64,
                "{query:?} {mode:?} k={k}: read {} tier entries in one round",
                stats.tier_entries
            );
        }
    }
}
