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
    use super::{super::test_util::*, *};
    use crate::superfile::fts::reader::FtsReader;

    // ── ExcludeFilter (negation gate) ─────────────────────────────────
    // `build_blob` plants: "rust" in docs 0 and 1, "java" in doc 2.

    /// Build an `ExcludeFilter` over `terms` from the planted blob.
    async fn exclude_filter_for(reader: &FtsReader, terms: &[&str]) -> ExcludeFilter {
        let column_id = reader.resolve_column_id("body").expect("column exists");
        let cursors = reader
            .build_term_cursors(column_id, terms, None, false)
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
}
