// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

// The writing half of the shared corpus: the schema, batch and append
// every generator repeats. Includes the documents themselves.

include!("corpus_data.rs");

/// The three text columns every generated table carries.
///
/// Held here rather than restated per generator: the corpus is only a
/// controlled comparison if the tables differ in format shape and nothing
/// else, and a schema copied six times is six chances to break that.
#[allow(dead_code)]
pub fn text_fields() -> Vec<Field> {
    vec![
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("notes", DataType::LargeUtf8, true),
    ]
}

/// The text columns' data, in [`text_fields`] order.
#[allow(dead_code)]
pub fn text_columns() -> Vec<LargeStringArray> {
    vec![
        LargeStringArray::from((0..N_DOCS).map(body).collect::<Vec<_>>()),
        LargeStringArray::from((0..N_DOCS).map(title).collect::<Vec<_>>()),
        LargeStringArray::from((0..N_DOCS).map(notes).collect::<Vec<_>>()),
    ]
}

/// Write the shared corpus as a text-only table indexed by `spec`.
///
/// `spec` stays the caller's because it is the one thing that genuinely
/// differs: the `fts` setter takes a `&str` on the older engines and an
/// `FtsField` on the newer ones, and only some expose positions at all.
#[allow(dead_code)]
pub fn write_text_corpus(
    out_dir: &str,
    table: &str,
    spec: IndexSpec,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(text_fields()));
    let db = connect(out_dir)?;
    let handle = db.create_table(table, Arc::clone(&schema), spec)?;
    let columns: Vec<_> = text_columns()
        .into_iter()
        .map(|c| Arc::new(c) as _)
        .collect();
    handle.append(&RecordBatch::try_new(schema, columns)?)?;
    println!("wrote {N_DOCS} docs to {out_dir}/{table}");
    Ok(())
}
