// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared test fixtures for the `reader/` submodule tests: blob builders that
//! plant small, known corpora so each test asserts against a fixed layout.

use bytes::Bytes;

use crate::superfile::fts::builder::FtsBuilder;

/// A one-column (`body`) `standard`-analyzer blob holding `docs`, doc `i`
/// being `docs[i]`.
pub(super) fn build_standard_blob(docs: &[&str]) -> (Bytes, String) {
    build_standard_blob_with(docs, None)
}

/// [`build_standard_blob`], when `order` is given storing its documents in
/// that order: blob position `i` holds `docs[order[i]]`, the way a merge
/// that reorders writes a blob.
pub(super) fn build_standard_blob_with(docs: &[&str], order: Option<&[u32]>) -> (Bytes, String) {
    let mut b = FtsBuilder::new();
    b.register_column("body".into(), false)
        .expect("register column");
    let rows: Vec<u32> = match order {
        Some(order) => order.to_vec(),
        None => (0..docs.len() as u32).collect(),
    };
    for (position, &row) in rows.iter().enumerate() {
        b.add_doc(0, position as u32, docs[row as usize])
            .expect("add doc");
    }
    if order.is_some() {
        b.doc_map = Some(rows);
    }
    let bytes = b.finish().expect("finish");
    let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
    (Bytes::from(bytes), json.to_string())
}

/// A `standard`-analyzer corpus whose vocabulary carries the two letters
/// Unicode case folding widens past lowercasing: `ſ` (long s, kept by
/// `to_lowercase`) and the Kelvin sign U+212A (lowercased to `k` at index).
/// Terms: k, kelvin, riſe, rise, set, sun, sunset, ſun.
pub(super) fn build_standard_fold_blob() -> (Bytes, String) {
    build_standard_blob(&["ſun riſe", "SUN set", "sunset rise", "Kelvin \u{212A}"])
}

pub(super) fn build_blob() -> (Bytes, String) {
    // 3 docs, 1 column.
    let mut b = FtsBuilder::new();
    b.register_column("body".into(), false)
        .expect("register column");
    b.add_doc(0, 0, "rust async runtime").expect("add doc");
    b.add_doc(0, 1, "tokio is a rust runtime").expect("add doc");
    b.add_doc(0, 2, "java spring boot").expect("add doc");
    let bytes = b.finish().expect("finish");
    let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
    (Bytes::from(bytes), json.to_string())
}

/// Build a corpus that exercises both the df=1 inline-encoded
/// path and the df ≥ 2 PFOR path side-by-side.
pub(super) fn build_mixed_df_blob() -> (Bytes, String) {
    let mut b = FtsBuilder::new();
    b.register_column("body".into(), false)
        .expect("register column");
    // `common`     → df = 3 (PFOR form)
    // `rust`       → df = 2 (PFOR form)
    // `uniqzero`  → df = 1 (inline form)
    // `uniqtwo`  → df = 1 (inline form)
    b.add_doc(0, 0, "common rust uniqzero").expect("add doc");
    b.add_doc(0, 1, "common rust").expect("add doc");
    b.add_doc(0, 2, "common uniqtwo").expect("add doc");
    let bytes = b.finish().expect("finish");
    let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
    (Bytes::from(bytes), json.to_string())
}

// ---- phrase atoms ----

/// Corpus with controlled adjacency for "new york": docs 0, 2
/// match (doc 4 twice); docs 1, 3 contain both words but never
/// adjacent in order.
pub(super) fn build_phrase_blob() -> (Bytes, &'static str) {
    use crate::superfile::fts::builder::FtsBuilder;
    let mut b = FtsBuilder::new();
    b.register_column("title".into(), true).expect("register");
    let docs = [
        "new york city",
        "york new haven",
        "the new york times",
        "new haven york",
        "new york new york",
    ];
    for (i, d) in docs.iter().enumerate() {
        b.add_doc(0, i as u32, d).expect("add doc");
    }
    (
        Bytes::from(b.finish().expect("finish")),
        r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#,
    )
}
