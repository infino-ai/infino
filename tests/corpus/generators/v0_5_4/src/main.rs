// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Writes a corpus table with the 0.5.4 engine, the last release before the
//! position sub-index and bitset blocks — so its blob carries the 56-byte
//! header with an empty positions region.
//!
//! Usage: `cargo run -- <output-dir> <table-name>`

use std::{env, sync::Arc};

use infino::{
    IndexSpec,
    arrow_array::{LargeStringArray, RecordBatch},
    arrow_schema::{DataType, Field, Schema},
    connect,
};

include!("../../shared/corpus.rs");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let out_dir = args.next().ok_or("usage: <output-dir> <table-name>")?;
    let table = args.next().ok_or("usage: <output-dir> <table-name>")?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("notes", DataType::LargeUtf8, true),
    ]));

    let db = connect(&out_dir)?;
    let handle = db.create_table(
        &table,
        Arc::clone(&schema),
        IndexSpec::new().fts("body").fts("title").fts("notes"),
    )?;

    let bodies: Vec<String> = (0..N_DOCS).map(body).collect();
    let titles: Vec<String> = (0..N_DOCS).map(title).collect();
    let notes: Vec<Option<String>> = (0..N_DOCS).map(notes).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(bodies)),
            Arc::new(LargeStringArray::from(titles)),
            Arc::new(LargeStringArray::from(notes)),
        ],
    )?;
    handle.append(&batch)?;

    println!("wrote {N_DOCS} docs to {out_dir}/{table}");
    Ok(())
}
