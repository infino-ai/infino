// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A table whose superfiles each have a broken block in their id sidecar:
//! every query still answers, `_id` included, as it would on a good copy, and
//! a reindex repairs the sidecar.

use std::{fs, path::Path, time::Duration};

use arrow_array::{Decimal128Array, Int64Array, RecordBatch};
use infino::{
    Bm25SearchOptions, connect,
    superfile::ids::{check_id_sidecar, with_damaged_id_block},
};
use tempfile::TempDir;

use crate::reindex_fixture::{
    N_DOCS, Staleness, TABLE, edit_superfiles, repairing, superfile_paths, write_stale_table,
    write_table,
};

/// The sidecar block to break. Each fixture superfile fits in one block, so
/// this is its only block.
const DAMAGED_BLOCK: usize = 0;

/// Every `_id` a `common` search returns over the table at `root`, sorted.
fn search_ids(root: &Path) -> Vec<i128> {
    let db = connect(root.to_str().expect("utf-8 path")).expect("connect");
    let hits = db
        .open_table(TABLE)
        .expect("open the table")
        .bm25_search(
            "body",
            "common",
            N_DOCS as usize,
            Bm25SearchOptions::new(),
            Some(&["_id"]),
        )
        .expect("search returning _id");
    let mut ids: Vec<i128> = hits
        .iter()
        .flat_map(|batch: &RecordBatch| {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("decimal ids");
            (0..col.len()).map(|i| col.value(i)).collect::<Vec<_>>()
        })
        .collect();
    ids.sort_unstable();
    ids
}

#[test]
fn a_broken_id_sidecar_block_still_returns_every_id() {
    // The fixture table, searched once as written, then with every superfile's
    // id sidecar broken, read straight from storage on a fresh connection.
    //  - after the break, `COUNT(*)` and `SELECT body` return every row.
    //  - a search returning `_id` gives the same ids as before the break, now
    //    read from Parquet.
    // This test pins that a broken sidecar block never changes an answer.
    let root = TempDir::new().expect("tempdir");
    write_table(root.path());
    let want = search_ids(root.path());
    assert_eq!(want.len(), N_DOCS as usize, "every doc holds `common`");
    edit_superfiles(root.path(), |bytes| {
        with_damaged_id_block(bytes, DAMAGED_BLOCK)
    });

    let db = connect(root.path().to_str().expect("utf-8 path")).expect("connect");
    let count = db
        .query_sql(&format!("SELECT COUNT(*) AS n FROM {TABLE}"))
        .expect("count");
    let n = count[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("int64 count")
        .value(0);
    assert_eq!(n, i64::from(N_DOCS));
    let rows: usize = db
        .query_sql(&format!("SELECT body FROM {TABLE}"))
        .expect("column read")
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, N_DOCS as usize);

    assert_eq!(search_ids(root.path()), want);
}

#[test]
fn a_reindex_rebuilds_a_broken_id_sidecar_block() {
    // The fixture table, stale for a layout rewrite, searched once, then with
    // every superfile's id sidecar broken, and reindexed.
    //  - the reindex rewrites every superfile, instead of stopping at the first.
    //  - every rewritten superfile has a good sidecar again.
    //  - a search returning `_id` gives the same ids as before the break.
    // This test pins that a reindex repairs a broken sidecar block.
    let root = TempDir::new().expect("tempdir");
    let superfiles = write_stale_table(root.path(), Staleness::DuplicatedFooter);
    let want = search_ids(root.path());
    edit_superfiles(root.path(), |bytes| {
        with_damaged_id_block(bytes, DAMAGED_BLOCK)
    });

    let db = connect(root.path().to_str().expect("utf-8 path")).expect("connect");
    let table = db.open_table(TABLE).expect("open the table");
    let report = table
        .reindex(&repairing(Staleness::DuplicatedFooter))
        .expect("reindex the damaged table");
    assert_eq!(report.rewritten, superfiles, "{report:?}");
    table.gc(Duration::ZERO).expect("collect superseded bytes");

    for path in superfile_paths(root.path()) {
        let bytes = fs::read(&path).expect("read superfile");
        assert_eq!(check_id_sidecar(&bytes), Ok(()), "{}", path.display());
    }
    assert_eq!(search_ids(root.path()), want);
}
