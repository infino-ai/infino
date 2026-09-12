// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A V5 FTS blob written by the released 0.8.1 builder, checked in as a
//! fixture, must open and rank under the current reader exactly like a
//! fresh build of the same documents. The test-only V5 writer used by the
//! unit tests reproduces the format from its definition; this is the check
//! that the definition matches what actually shipped.

use std::sync::Arc;

use bytes::Bytes;
use infino::superfile::fts::{
    builder::FtsBuilder,
    reader::{BoolMode, FtsReader},
    tokenize::AsciiLowerTokenizer,
};

/// Written by `infino` v0.8.1 (`be681cad`) from [`fixture_doc`] over 600
/// documents into two columns, `body` (positionless) and `title`
/// (positional), both `ascii_lower`.
const FIXTURE: &[u8] = include_bytes!("../../fixtures/fts_blob_v0.8.1_ascii_lower.bin");
const COLUMNS: &str = r#"[{"name":"body","tokenizer":"ascii_lower"},{"name":"title","tokenizer":"ascii_lower","positions":true}]"#;
const N_DOCS: u32 = 600;

/// Every third row is null for both columns; the rest tie closely on
/// `common`, so block-max pruning is live at small `k`.
pub fn fixture_doc(i: u32) -> String {
    if i % 3 == 2 {
        return String::new();
    }
    match i % 4 {
        0 => format!("common shared d{i}"),
        1 => format!("common common beta d{i} pad pad"),
        2 => format!("common gamma d{i} pad"),
        _ => format!("common shared delta d{i} pad pad pad pad"),
    }
}

fn fresh_build() -> FtsReader {
    let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
    b.register_column("body".into(), false)
        .expect("register body");
    b.register_column("title".into(), true)
        .expect("register title");
    for i in 0..N_DOCS {
        let text = fixture_doc(i);
        b.add_doc(0, i, &text).expect("body");
        b.add_doc(1, i, &text).expect("title");
    }
    FtsReader::open(Bytes::from(b.finish().expect("finish")), COLUMNS).expect("open fresh")
}

#[tokio::test]
async fn a_blob_written_by_0_8_1_ranks_exactly_like_a_fresh_build() {
    let version = u32::from_le_bytes(FIXTURE[8..12].try_into().expect("version field"));
    assert_eq!(version, 5, "the fixture is a V5 blob");
    let old = FtsReader::open(Bytes::from_static(FIXTURE), COLUMNS).expect("open 0.8.1 blob");
    let fresh = fresh_build();

    // Corrected statistics: the average and collection size over the
    // documents that carry tokens, and the inflation an older file owes.
    let (tokens, docs) = (0..N_DOCS)
        .map(|i| fixture_doc(i).split_whitespace().count() as u64)
        .fold((0u64, 0u64), |(t, d), n| (t + n, d + u64::from(n > 0)));
    for (old_col, fresh_col) in old.fts_columns_config().zip(fresh.fts_columns_config()) {
        assert_eq!(old_col.name, fresh_col.name);
        assert_eq!(old_col.scored_doc_count(), docs);
        assert_eq!(old_col.length_stats.total_tokens, tokens);
        assert_eq!(
            old_col.avgdl(),
            fresh_col.avgdl(),
            "{}: corrected average",
            old_col.name
        );
        assert_eq!(fresh_col.bound_scale, 1.0);
        assert!(
            old_col.bound_scale > 1.0 / (old_col.params.k1 + 1.0),
            "{}: a row-average file owes inflation beyond the scale change, got {}",
            old_col.name,
            old_col.bound_scale
        );
    }

    for column in ["body", "title"] {
        for terms in [
            &["common"][..],
            &["common", "shared"][..],
            &["gamma", "beta"][..],
            &["d0"][..],
        ] {
            for k in [1usize, 3, 10, 100] {
                let a = old
                    .search(column, terms, k, BoolMode::Or)
                    .await
                    .expect("old search");
                let b = fresh
                    .search(column, terms, k, BoolMode::Or)
                    .await
                    .expect("fresh search");
                assert_eq!(a, b, "{column} {terms:?} k={k}");
            }
        }
    }
}
