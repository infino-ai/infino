// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Negation gates: [`ExcludeFilter`] (term negatives) and its
//! phrase-aware sibling [`AtomExcludeFilter`]. Both skip-probe their
//! negated cursors against a monotonically increasing candidate doc, so
//! a common negated list is never fully decoded. `pub(super)` within
//! `reader/` (ExcludeFilter stays pub(crate) — PreparedClauses carries it).

use super::{
    cursor::TermCursor,
    phrase::AnyCursor,
    work::{term_cursor_bytes, term_cursor_ranges},
};
use crate::superfile::error::FtsError;

/// Atom-walk exclusion gate: the heterogeneous sibling of
/// [`ExcludeFilter`], additionally able to exclude docs containing a
/// negated *phrase*. Same monotonic-doc contract.
pub(super) struct AtomExcludeFilter {
    pub(super) atoms: Vec<AnyCursor>,
    pub(super) last_doc: u32,
}

impl AtomExcludeFilter {
    pub(super) fn new(atoms: Vec<AnyCursor>) -> Self {
        Self { atoms, last_doc: 0 }
    }

    /// `false` iff `doc` matches any negated atom.
    pub(super) fn admits(&mut self, doc: u32) -> Result<bool, FtsError> {
        debug_assert!(
            doc >= self.last_doc,
            "AtomExcludeFilter fed non-monotonic doc: {doc} < {}",
            self.last_doc
        );
        self.last_doc = doc;
        for a in &mut self.atoms {
            a.skip_to(doc)?;
            if !a.is_exhausted() && a.current_doc_id() == doc {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Exclusion gate for negated (`-term`) clauses: holds one
/// [`TermCursor`] per negated term, streamed with `skip_to` (a common
/// negated list is never fully decoded). A doc is rejected if it appears
/// in any negated term's list.
///
/// Kernels take `Option<&mut ExcludeFilter>` (`None` = no negation)
/// rather than a generic filter parameter: monomorphizing the OR kernel
/// measured 25-30% slower even with a no-op filter, while the `None`
/// branch is constant per query, perfectly predicted, and free.
pub(crate) struct ExcludeFilter {
    pub(super) cursors: Vec<TermCursor>,
    /// Last doc-id passed to `admits`; guards the monotonic call order.
    pub(super) last_doc: u32,
}

impl ExcludeFilter {
    pub(super) fn new(cursors: Vec<TermCursor>) -> Self {
        Self {
            cursors,
            last_doc: 0,
        }
    }

    /// Posting-list bytes the negation cursors index into — see
    /// [`PreparedClauses::postings_bytes`].
    pub(super) fn postings_bytes(&self) -> u64 {
        term_cursor_bytes(&self.cursors)
    }

    /// Byte-source ranges the negation cursors' builds requested (one per
    /// PFOR term) — see [`PreparedClauses::planned_ranges`].
    pub(super) fn planned_ranges(&self) -> u64 {
        term_cursor_ranges(&self.cursors)
    }
}

impl ExcludeFilter {
    /// `false` iff `doc` is in any negated list.
    ///
    /// `doc` must be non-decreasing across a search: `skip_to` only
    /// moves forward. Every kernel walks candidates ascending, so this
    /// holds; the debug-assert guards a future caller that breaks it.
    #[inline]
    pub(super) fn admits(&mut self, doc: u32) -> bool {
        debug_assert!(
            doc >= self.last_doc,
            "ExcludeFilter fed non-monotonic doc: {doc} < {}",
            self.last_doc
        );
        self.last_doc = doc;
        for c in &mut self.cursors {
            c.skip_to(doc);
            if !c.is_exhausted() && c.current_doc_id() == doc {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;

    use super::{super::test_util::*, *};
    use crate::superfile::fts::{
        builder::FtsBuilder,
        posting::BLOCK_LEN,
        reader::FtsReader,
        tokenize::{AsciiLowerTokenizer, Phrase},
    };

    // ── ExcludeFilter (negation gate) ─────────────────────────────────
    // `build_blob` plants: "rust" in docs 0 and 1, "java" in doc 2.

    /// Build an `ExcludeFilter` over `terms` from the planted blob.
    async fn exclude_filter_for(reader: &FtsReader, terms: &[&str]) -> ExcludeFilter {
        let column_id = reader.resolve_column_id("body").expect("column exists");
        let cursors = reader
            .build_term_cursors(column_id, terms, None, false, None, None)
            .await
            .expect("build cursors");
        ExcludeFilter::new(cursors)
    }

    #[tokio::test]
    async fn exclude_filter_rejects_docs_in_negated_list() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let mut f = exclude_filter_for(&r, &["rust"]).await;
        // "rust" is in docs 0 and 1 → excluded; doc 2 survives.
        assert!(!f.admits(0));
        assert!(!f.admits(1));
        assert!(f.admits(2));
    }

    #[tokio::test]
    async fn exclude_filter_missing_term_excludes_nothing() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // A negated term absent from the dictionary yields no cursor, so
        // the filter admits every doc.
        let mut f = exclude_filter_for(&r, &["nonexistent"]).await;
        assert!(f.admits(0));
        assert!(f.admits(1));
        assert!(f.admits(2));
    }

    #[tokio::test]
    async fn exclude_filter_multiple_negated_terms() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // Negating "rust" (docs 0,1) and "java" (doc 2) excludes all
        // three — a doc is dropped if it matches ANY negated term.
        let mut f = exclude_filter_for(&r, &["rust", "java"]).await;
        assert!(!f.admits(0));
        assert!(!f.admits(1));
        assert!(!f.admits(2));
    }

    #[tokio::test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "non-monotonic")]
    async fn exclude_filter_panics_on_non_monotonic_feed() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let mut f = exclude_filter_for(&r, &["rust"]).await;
        // Feed a descending doc-id: `skip_to` can't seek backwards, so
        // the debug assertion catches the contract violation.
        let _ = f.admits(1);
        let _ = f.admits(0);
    }

    // ── Probes at block edges ─────────────────────────────────────────

    /// Rows in the block-edge corpora: enough for the negated list to
    /// span several full blocks and end in a partial one.
    const EDGE_DOCS: u32 = 1000;
    /// Every doc divisible by this carries both phrase words in the
    /// wrong order, so it is the survivor of a negated phrase.
    const REVERSED_EVERY: u32 = 7;

    /// A positionless corpus where `neg` sits in every even row and
    /// `pos` in every row: `neg`'s list is 500 postings, four blocks
    /// with a partial last one.
    fn edge_reader() -> FtsReader {
        let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
        b.register_column("body".into(), false).expect("register");
        for doc in 0..EDGE_DOCS {
            let text = if doc % 2 == 0 { "pos neg" } else { "pos" };
            b.add_doc(0, doc, text).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open")
    }

    /// A positional corpus where every row holds `a` and `b`, adjacent
    /// in order except every `REVERSED_EVERY`th row, which holds them
    /// reversed.
    fn phrase_edge_reader() -> FtsReader {
        let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
        b.register_column("body".into(), true).expect("register");
        for doc in 0..EDGE_DOCS {
            let text = if doc % REVERSED_EVERY == 0 {
                "b a"
            } else {
                "a b"
            };
            b.add_doc(0, doc, text).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"ascii_lower","positions":true}]"#;
        FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open")
    }

    /// The gate's answer at every block edge of the negated list: the
    /// block's last doc (present, rejected), the doc after it (absent,
    /// admitted), the next block's first doc (present, rejected), and
    /// finally docs past the list (admitted). Each probe is a `skip_to`
    /// that lands exactly on or just past a block boundary, where a
    /// seek that overshoots by one would admit a negated doc or reject
    /// a clean one.
    #[tokio::test]
    async fn exclude_filter_answers_correctly_at_every_block_edge() {
        let r = edge_reader();
        let column_id = r.resolve_column_id("body").expect("column exists");
        let cursors = r
            .build_term_cursors(column_id, &["neg"], None, false, None, None)
            .await
            .expect("build cursors");
        let edges: Vec<u32> = cursors[0].blocks.iter().map(|b| b.last_doc_id).collect();
        assert!(
            edges.len() >= 4,
            "premise: several blocks, got {}",
            edges.len()
        );
        assert_eq!(
            cursors[0].df as usize % BLOCK_LEN,
            (EDGE_DOCS as usize / 2) % BLOCK_LEN,
            "premise: the last block is partial"
        );
        let mut f = ExcludeFilter::new(cursors);
        for (i, &last) in edges.iter().enumerate() {
            assert_eq!(last % 2, 0, "block {i}: last doc is a planted even row");
            assert!(!f.admits(last), "block {i}: its last doc {last} is negated");
            assert!(f.admits(last + 1), "block {i}: doc {} is clean", last + 1);
            if i + 1 < edges.len() {
                let first_of_next = last + 2;
                assert!(
                    !f.admits(first_of_next),
                    "block {}: its first doc {first_of_next} is negated",
                    i + 1
                );
            }
        }
        let past = edges[edges.len() - 1] + 1;
        assert!(f.admits(past), "doc {past} after the list is clean");
        assert!(
            f.admits(EDGE_DOCS + 5000),
            "a doc far past the list is clean"
        );
    }

    /// The phrase-aware gate rejects exactly the docs holding the
    /// negated phrase in order, probed monotonically over every row,
    /// which walks both dense members across every block boundary and
    /// ends past the list. A term atom through the same gate agrees
    /// with `ExcludeFilter`.
    #[tokio::test]
    async fn atom_exclude_filter_rejects_the_negated_phrase_across_blocks() {
        let r = phrase_edge_reader();
        let column_id = r.resolve_column_id("body").expect("column exists");
        let phrases = vec![Phrase::adjacent(vec!["a".to_string(), "b".to_string()])];
        let (atoms, _) = r
            .build_atom_cursors(column_id, &[], &phrases, None, None)
            .await
            .expect("build atoms");
        let atoms: Vec<AnyCursor> = atoms.into_iter().flatten().collect();
        assert_eq!(atoms.len(), 1, "premise: one phrase atom");
        assert!(
            matches!(atoms[0], AnyCursor::Phrase(_)),
            "premise: a phrase atom"
        );
        let mut f = AtomExcludeFilter::new(atoms);
        for doc in 0..EDGE_DOCS {
            let admitted = f.admits(doc).expect("admits");
            assert_eq!(
                admitted,
                doc % REVERSED_EVERY == 0,
                "doc {doc}: only reversed rows survive the negated phrase"
            );
        }
        assert!(f.admits(EDGE_DOCS + 1).expect("admits"), "past the list");

        // A term atom is the same gate as `ExcludeFilter`.
        let r = edge_reader();
        let column_id = r.resolve_column_id("body").expect("column exists");
        let (atoms, _) = r
            .build_atom_cursors(column_id, &["neg"], &[], None, None)
            .await
            .expect("build atoms");
        let mut f = AtomExcludeFilter::new(atoms.into_iter().flatten().collect());
        for doc in (0..EDGE_DOCS).step_by(3) {
            assert_eq!(f.admits(doc).expect("admits"), doc % 2 == 1, "doc {doc}");
        }
        assert!(f.admits(EDGE_DOCS + 1).expect("admits"));
    }
}
