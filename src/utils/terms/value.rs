// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A term dictionary entry: where a term's postings are, or the whole
//! posting of a `df = 1` term small enough to inline.

/// Largest position an inline entry on a positional column carries in its
/// `tf` slot; a df=1 term whose one position is past it takes the
/// postings form.
pub(crate) const INLINE_TF_MAX: u32 = (1 << 30) - 1;

/// One term's dictionary entry.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum DictEntry {
    /// df ≥ 2 (or a df = 1 posting that could not inline) — fetch
    /// `postings_length` bytes from `metadata_offset`. `short` says
    /// which body those bytes hold: the long form (metadata header, skip
    /// table, PFOR blocks) or the short form (`fts::short`).
    Pfor {
        metadata_offset: u64,
        postings_length: u32,
        short: bool,
    },
    /// df = 1 — the entire posting is right here. No postings-region
    /// read required.
    Inline { doc_id: u32, tf: u32 },
}
