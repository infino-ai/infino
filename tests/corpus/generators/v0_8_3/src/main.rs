// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Writes a corpus table with the 0.8.3 engine, the last release to write
//! the FST dictionary and long-form-only terms, with `title` positional.
//!
//! This is the shape a table created by the newest published release is
//! in, and the only one whose terms already carry the corrected
//! tokenization — it records no analysis revision all the same, because
//! the field postdates it.
//!
//! Usage: `cargo run -- <output-dir> <table-name> [profile]`
//!
//! `profile` is `hybrid` to add a vector column beside the text ones.
//! That shape exists to pin a limitation rather than a format: the rerank
//! codec is internal and never `Fp32` through the public API, so a
//! re-analysis of any table with a vector index is refused.

use std::{env, sync::Arc};

use infino::{
    FtsField, IndexSpec, Metric,
    arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch},
    arrow_schema::{DataType, Field, Schema},
    connect,
};

include!("../../shared/corpus.rs");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let out_dir = args.next().ok_or("usage: <output-dir> <table-name>")?;
    let table = args.next().ok_or("usage: <output-dir> <table-name>")?;
    let hybrid = args.next().as_deref() == Some("hybrid");

    let mut fields = vec![
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("notes", DataType::LargeUtf8, true),
    ];
    let mut spec = IndexSpec::new()
        .fts(FtsField::new("body"))
        .fts(FtsField::new("title").positions(true))
        .fts(FtsField::new("notes"));
    if hybrid {
        fields.push(Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDING_DIM as i32,
            ),
            false,
        ));
        // Cosine takes the engine's default codec. Nothing here selects
        // it — that is the point of the shape.
        spec = spec.vector("emb", EMBEDDING_DIM, Metric::Cosine);
    }
    let schema = Arc::new(Schema::new(fields));

    let db = connect(&out_dir)?;
    let handle = db.create_table(&table, Arc::clone(&schema), spec)?;

    let bodies: Vec<String> = (0..N_DOCS).map(body).collect();
    let titles: Vec<String> = (0..N_DOCS).map(title).collect();
    let notes: Vec<Option<String>> = (0..N_DOCS).map(notes).collect();

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(bodies)),
        Arc::new(LargeStringArray::from(titles)),
        Arc::new(LargeStringArray::from(notes)),
    ];
    if hybrid {
        let flat: Vec<f32> = (0..N_DOCS).flat_map(embedding).collect();
        columns.push(Arc::new(FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            EMBEDDING_DIM as i32,
            Arc::new(Float32Array::from(flat)) as ArrayRef,
            None,
        )?));
    }
    let batch = RecordBatch::try_new(schema, columns)?;
    handle.append(&batch)?;

    println!("wrote {N_DOCS} docs to {out_dir}/{table}");
    Ok(())
}
