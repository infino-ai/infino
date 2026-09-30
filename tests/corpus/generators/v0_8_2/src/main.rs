// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Writes a corpus table with the 0.8.2 engine, whose builder writes the
//! exact-f32 block-max and coarse-table layout, with `title` positional —
//! the last release writing this layout, and one of the three that expose
//! a public positions setter at all (0.8.1 was the first).
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
            .fts(FtsField::new("title").positions(true))
            .fts(FtsField::new("notes")),
    )
}
