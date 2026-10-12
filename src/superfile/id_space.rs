// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The three identifier spaces a superfile addresses its documents in,
//! as distinct types rather than three shapes of bare integer.
//!
//! A superfile numbers the same document three ways, and the numbers are
//! not interchangeable:
//!
//! * [`RowId`] — the document's position among the superfile's Parquet
//!   rows. Tombstones, the `_id` pages, the vector blob, row groups and
//!   any row set a SQL predicate pushes down are all expressed here.
//! * [`FtsDocId`] — the document's position inside the superfile's FTS
//!   blob. Postings, skip tables, the term dictionary and the
//!   doc-length arrays are all expressed here.
//! * [`StableId`] — the `_id` the user sees, minted once at ingest and
//!   carried unchanged through every compaction. The only one of the
//!   three that means anything outside the superfile that holds it.
//!
//! The two positional spaces were the same number for the whole life of
//! the format up to and including version 7: the FTS blob stored its
//! documents in arrival order, so blob position and row position were
//! equal and the distinction cost nothing to ignore. Version 8 lets a
//! compaction store the blob's documents in an order of its own, which
//! makes them different numbers for the same document, and a value that
//! crosses from one space to the other untranslated is not a crash but a
//! wrong answer: a search names rows nobody asked for.
//!
//! [`DocMap`] is the one translation between them, and the types make
//! the places that need it impossible to miss.

use std::{
    fmt::{self, Display, Formatter},
    sync::Arc,
};

use roaring::RoaringBitmap;

/// A document's position among a superfile's Parquet rows.
///
/// This is the superfile's outward-facing numbering: what a tombstone
/// marks, what an `_id` page resolves, what a pushed-down row set names,
/// and what every search this crate exposes returns. Meaningful only
/// within the one superfile that holds the row.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct RowId(u32);

impl RowId {
    /// The row at position `n`.
    #[inline]
    pub const fn new(n: u32) -> Self {
        Self(n)
    }

    /// The position as a plain integer, for indexing and for the crate
    /// boundary where rows leave as numbers.
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl Display for RowId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

/// Comparison against a plain number, for the assertions and literals
/// that name a row directly. It relates a row to an untyped integer,
/// never to a [`FtsDocId`]: the operations that actually confuse the two
/// spaces, indexing an array or probing a row set, still have to name
/// the conversion.
impl PartialEq<u32> for RowId {
    fn eq(&self, other: &u32) -> bool {
        self.0 == *other
    }
}

impl From<RowId> for u64 {
    fn from(r: RowId) -> Self {
        Self::from(r.0)
    }
}

/// A document's position inside a superfile's FTS blob.
///
/// This is the FTS kernel's internal numbering: the id a posting list
/// stores, a skip table seeks on, and a doc-length array is indexed by.
/// Equal to the document's [`RowId`] up to format version 7 and in any
/// blob written in arrival order; different once a compaction reorders
/// the blob. It must be translated through [`DocMap`] before it is used
/// against anything keyed by rows.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct FtsDocId(u32);

impl FtsDocId {
    /// The blob document at position `n`.
    #[inline]
    pub const fn new(n: u32) -> Self {
        Self(n)
    }

    /// The position as a plain integer, for indexing the blob's own
    /// arrays and for the kernels that walk in this space throughout.
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl Display for FtsDocId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

/// Comparison against a plain number, as for [`RowId`].
impl PartialEq<u32> for FtsDocId {
    fn eq(&self, other: &u32) -> bool {
        self.0 == *other
    }
}

/// The `_id` a document carries for as long as it exists.
///
/// Minted once by the writer that ingested the row and never rewritten,
/// so it survives compaction, reordering and the move from one superfile
/// into another. Unlike [`RowId`] and [`FtsDocId`] it identifies a
/// document across the whole table rather than a position within one
/// file.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct StableId(i128);

impl StableId {
    /// The id with value `v`.
    #[inline]
    pub const fn new(v: i128) -> Self {
        Self(v)
    }

    /// The value as a plain integer, the form it is stored and compared
    /// in.
    #[inline]
    pub const fn get(self) -> i128 {
        self.0
    }
}

/// How a superfile's FTS blob numbers its documents relative to its
/// Parquet rows: the translation from [`FtsDocId`] to [`RowId`].
///
/// A blob written in arrival order needs no translation and carries
/// [`DocMap::Identity`], which costs nothing to consult. A blob a
/// compaction reordered carries [`DocMap::Permuted`], the row each blob
/// position stands for, read from the blob's doc-id map region.
///
/// Cloning is cheap either way — the permuted form is one `Arc` bump —
/// so every gate that needs to translate can hold its own handle.
#[derive(Clone, Debug, Default)]
pub enum DocMap {
    /// Blob position and row are the same number, which is every blob
    /// below format version 8 and every version 8 blob written in
    /// arrival order.
    #[default]
    Identity,
    /// The row each blob position stands for, indexed by [`FtsDocId`].
    Permuted(Arc<[RowId]>),
}

impl DocMap {
    /// The map a reordered blob's doc-id map region describes.
    #[inline]
    pub fn permuted(rows: Arc<[RowId]>) -> Self {
        Self::Permuted(rows)
    }

    /// The row `doc` stands for.
    ///
    /// A blob id past the end of a permuted map is a bug in the caller
    /// or a blob the CRC over the map region should already have
    /// rejected, so it trips an assertion where assertions run. In a
    /// release build it falls back to the identity instead of
    /// panicking, because a corrupt map should not take down a query.
    #[inline]
    pub fn row_of(&self, doc: FtsDocId) -> RowId {
        match self {
            Self::Identity => RowId(doc.get()),
            Self::Permuted(rows) => {
                debug_assert!(
                    (doc.get() as usize) < rows.len(),
                    "blob doc id {} is past the {} entries of this blob's map",
                    doc.get(),
                    rows.len()
                );
                rows.get(doc.get() as usize)
                    .copied()
                    .unwrap_or(RowId(doc.get()))
            }
        }
    }

    /// Whether the blob stores its documents in an order of its own, so
    /// that blob position and row are different numbers.
    #[inline]
    pub fn is_permuted(&self) -> bool {
        matches!(self, Self::Permuted(_))
    }
}

/// The rows a caller admits at all: a SQL `WHERE`'s candidate set,
/// resolved for one superfile.
///
/// A thin wrapper over the bitmap rather than the bitmap itself because
/// this set is the one thing that travels *into* an FTS kernel in row
/// space, while the kernel walks blob ids. Taking a [`RowId`] means a
/// blob id cannot reach [`Self::contains`] without going through
/// [`DocMap`] first — which is exactly the mistake that makes a scoped
/// search on a reordered blob match almost nothing.
///
/// Holding the `Arc` rather than the bitmap keeps the wrap free: the
/// set is resolved once per superfile and handed to a gate that outlives
/// the call, so wrapping must not copy it.
#[derive(Clone, Debug)]
pub struct RowSet(Arc<RoaringBitmap>);

impl RowSet {
    /// The set of the rows in `bitmap`, which is already in row space.
    #[inline]
    pub fn new(bitmap: Arc<RoaringBitmap>) -> Self {
        Self(bitmap)
    }

    /// Whether `row` is admitted.
    #[inline]
    pub fn contains(&self, row: RowId) -> bool {
        self.0.contains(row.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The identity map hands back the row a blob id already is.
    #[test]
    fn an_identity_map_translates_a_blob_id_to_the_same_row() {
        let map = DocMap::Identity;
        assert_eq!(map.row_of(FtsDocId::new(7)), RowId::new(7));
        assert!(!map.is_permuted());
    }

    /// A permuted map hands back the row its region recorded.
    #[test]
    fn a_permuted_map_translates_a_blob_id_to_its_recorded_row() {
        let rows: Arc<[RowId]> = vec![RowId::new(2), RowId::new(0), RowId::new(1)].into();
        let map = DocMap::permuted(rows);
        assert_eq!(map.row_of(FtsDocId::new(0)), RowId::new(2));
        assert_eq!(map.row_of(FtsDocId::new(1)), RowId::new(0));
        assert_eq!(map.row_of(FtsDocId::new(2)), RowId::new(1));
        assert!(map.is_permuted());
    }

    /// The row set answers in row space.
    #[test]
    fn a_row_set_admits_exactly_the_rows_it_was_built_from() {
        let set = RowSet::new(Arc::new([1u32, 4].into_iter().collect()));
        assert!(set.contains(RowId::new(1)));
        assert!(set.contains(RowId::new(4)));
        assert!(!set.contains(RowId::new(2)));
    }
}
