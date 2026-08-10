// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared test fixtures for the `reader/` submodule tests: blob builders that
//! plant small, known corpora so each test asserts against a fixed layout.

use std::sync::Arc;

use bytes::Bytes;

use crate::superfile::fts::{builder::FtsBuilder, tokenize::AsciiLowerTokenizer};

pub(super) fn build_blob() -> (Bytes, String) {
    // 3 docs, 1 column.
    let tok = Arc::new(AsciiLowerTokenizer);
    let mut b = FtsBuilder::new(tok);
    b.register_column("body".into(), false)
        .expect("register column");
    b.add_doc(0, 0, "rust async runtime").expect("add doc");
    b.add_doc(0, 1, "tokio is a rust runtime").expect("add doc");
    b.add_doc(0, 2, "java spring boot").expect("add doc");
    let bytes = b.finish().expect("finish");
    let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
    (Bytes::from(bytes), json.to_string())
}

/// Build a corpus that exercises both the df=1 inline-encoded
/// path and the df ≥ 2 PFOR path side-by-side.
pub(super) fn build_mixed_df_blob() -> (Bytes, String) {
    let tok = Arc::new(AsciiLowerTokenizer);
    let mut b = FtsBuilder::new(tok);
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
    let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
    (Bytes::from(bytes), json.to_string())
}

// ---- phrase atoms ----

/// Corpus with controlled adjacency for "new york": docs 0, 2
/// match (doc 4 twice); docs 1, 3 contain both words but never
/// adjacent in order.
pub(super) fn build_phrase_blob() -> (Bytes, &'static str) {
    use crate::superfile::fts::builder::FtsBuilder;
    let mut b = FtsBuilder::new(crate::test_helpers::default_tokenizer());
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
        r#"[{"name":"title","tokenizer":"ascii_lower","positions":true}]"#,
    )
}
