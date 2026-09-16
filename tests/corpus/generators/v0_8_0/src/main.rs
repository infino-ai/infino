// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Writes a corpus table with the 0.8.0 engine, whose builder writes the
//! exact-f32 block-max and coarse-table layout. Positionless — this release
//! predates a public positions setter.
//!
//! Usage: `cargo run -- <output-dir> <table-name>`

use std::{env, sync::Arc};

use infino::{
    FtsField, IndexSpec,
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
        IndexSpec::new()
            .fts(FtsField::new("body"))
            .fts(FtsField::new("title"))
            .fts(FtsField::new("notes")),
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
