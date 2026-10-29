// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! A table that keeps no term index (`SupertableOptions::term_index` off)
//! writes no index object on any commit or maintenance pass, and a ranked
//! search answers as the table with the index does, reading nothing from
//! an index: it probes every superfile.
#![deny(clippy::unwrap_used)]

use std::path::Path;

use infino::{BoolMode, OptimizeOptions};
use tempfile::TempDir;

use super::tier_harness::*;

/// Where the engine keeps the term index's objects under the table's
/// storage root.
const TERM_INDEX_DIR: &str = "term-index";

/// The names of the term-index objects under `root`, none when the
/// directory was never created.
fn term_index_objects(root: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(root.join(TERM_INDEX_DIR)) else {
        return Vec::new();
    };
    entries
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect()
}

#[test]
fn a_table_without_a_term_index_writes_none_and_answers_the_same() {
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with = table(&with_dir, true);
    let without = table_with(
        &without_dir,
        Build {
            tiers: true,
            without_index: true,
            ..Build::default()
        },
    );
    assert!(
        !term_index_objects(with_dir.path()).is_empty(),
        "the table with the index wrote its objects"
    );
    assert_eq!(
        term_index_objects(without_dir.path()),
        Vec::<String>::new(),
        "twelve commits wrote no term-index object"
    );
    for (query, mode) in queries() {
        for k in KS {
            let (a, stats, _) = run(&with, &query, mode, k);
            let (b, without_stats, _) = run(&without, &query, mode, k);
            assert_eq!(a, b, "{query:?} {mode:?} k={k}: same answer");
            assert_eq!(
                without_stats.tier_entries, 0,
                "{query:?} {mode:?} k={k}: no tier read"
            );
            assert_eq!(
                without_stats.superfiles_resolved, 0,
                "{query:?} {mode:?} k={k}: nothing resolved from an index"
            );
            // Sanity on the comparison: the table with the index does use it.
            if k == 10 && mode == BoolMode::And && query == "the data" {
                assert!(
                    stats.tier_entries > 0,
                    "the table with the index reads its tiers"
                );
            }
        }
    }
    // Maintenance compacts the superfiles and would rebuild the index over
    // them; here it rebuilds nothing.
    without
        .optimize(&OptimizeOptions::default())
        .expect("optimize");
    assert_eq!(
        term_index_objects(without_dir.path()),
        Vec::<String>::new(),
        "optimize wrote no term-index object"
    );
    for (query, mode) in queries() {
        let (a, _, _) = run(&with, &query, mode, 10);
        let (b, stats, _) = run(&without, &query, mode, 10);
        let scores = |hits: &[(usize, u32, f32)]| hits.iter().map(|h| h.2).collect::<Vec<_>>();
        assert_eq!(
            scores(&a),
            scores(&b),
            "{query:?} {mode:?}: the same scores after compaction"
        );
        assert_eq!(stats.superfiles_resolved, 0, "{query:?} {mode:?}: no index");
    }
}
