// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! An embedder sizes the process's decoded-column cache in code, before any
//! table: the setter takes the place of a host config that keeps none, the
//! tables it then opens scan through the cache, and a size given once the
//! cache is built is refused and leaves it in place. Its own binary, because
//! the configuration is read, and the cache built, once per process.
#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use infino::{
    IndexSpec,
    arrow_array::{Array, Int64Array, RecordBatch, StringArray},
    arrow_schema::{DataType, Field, Schema},
    connect,
};
use tempfile::TempDir;

/// Bytes of decoded columns the process keeps.
const KEPT_BYTES: u64 = 64 * 1024 * 1024;

/// The plan `EXPLAIN` gives for `sql`, its lines joined.
fn plan(conn: &infino::Connection, sql: &str) -> String {
    conn.query_sql(&format!("EXPLAIN {sql}"))
        .unwrap_or_else(|e| panic!("EXPLAIN {sql}: {e}"))
        .iter()
        .flat_map(|b| {
            let plans = b
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("plan column")
                .clone();
            (0..plans.len())
                .filter(|i| plans.is_valid(*i))
                .map(|i| plans.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A table `name` of one `Int64` column `v`, holding a few rows.
fn table_with_rows(conn: &infino::Connection, name: &str) {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    conn.create_table(name, Arc::clone(&schema), IndexSpec::new())
        .unwrap_or_else(|e| panic!("create {name}: {e}"))
        .append(
            &RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![3, 1, 2]))])
                .expect("batch"),
        )
        .unwrap_or_else(|e| panic!("append {name}: {e}"));
}

/// Whether a statement the scan answers over `table` (not its statistics)
/// plans through the decoded-column cache.
fn scans_through_the_cache(conn: &infino::Connection, table: &str) -> bool {
    let explained = plan(conn, &format!("SELECT v FROM {table} WHERE v > 1"));
    explained.contains("CachedScanExec")
}

#[test]
fn a_size_set_before_any_table_sizes_the_cache_and_one_after_is_refused() {
    // A host config that keeps no cache, so only the setter can build one.
    let config = TempDir::new().expect("tempdir");
    std::fs::create_dir_all(config.path().join("infino")).expect("config dir");
    std::fs::write(
        config.path().join("infino/config.yaml"),
        "memory:\n  decoded_column_cache_bytes: 0\n",
    )
    .expect("config file");
    // SAFETY: this is the only test in the binary and nothing else has
    // started yet, so no other thread reads the environment.
    unsafe { std::env::set_var("XDG_CONFIG_HOME", config.path()) };

    // No table yet, so the cache is not built and the size takes effect.
    assert!(infino::set_decoded_column_cache_bytes(KEPT_BYTES));

    let root = TempDir::new().expect("tempdir");
    let conn = connect(format!("file://{}", root.path().display())).expect("connect");
    table_with_rows(&conn, "t");
    assert!(scans_through_the_cache(&conn, "t"));

    // Built: a later size, larger, smaller or none, changes nothing, and a
    // table made after it still scans through the cache.
    assert!(!infino::set_decoded_column_cache_bytes(2 * KEPT_BYTES));
    assert!(!infino::set_decoded_column_cache_bytes(0));
    table_with_rows(&conn, "u");
    assert!(scans_through_the_cache(&conn, "u"));
}
