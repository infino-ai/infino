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

    write_text_corpus(
        &out_dir,
        &table,
        IndexSpec::new()
            .fts(FtsField::new("body"))
            .fts(FtsField::new("title"))
            .fts(FtsField::new("notes")),
    )
}
