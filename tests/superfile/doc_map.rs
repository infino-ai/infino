// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The doc-id map read in place out of its region.
//!
//! A blob opened from a memory mapping keeps its map in the mapped bytes
//! rather than copying four bytes per document onto the heap, where they
//! would stay pinned for the life of the reader. Every position has to
//! resolve to the row the region holds, and a region that is not a whole
//! number of entries must still serve the entries it does hold.

#![deny(clippy::unwrap_used)]

use bytes::{BufMut, Bytes, BytesMut};
use infino::superfile::id_space::{DocMap, FtsDocId, RowId};

/// Documents in the map under test. Large enough that an index error shows
/// up as a wrong row rather than an accidental match.
const MAP_DOCS: u32 = 4096;

/// Bytes per map entry, matching the region's little-endian `u32` layout.
const ENTRY_BYTES: usize = 4;

/// A permutation with no fixed points worth relying on: every position maps
/// to a different row, so an off-by-one or a byte-order slip is visible.
fn rows_for(n: u32) -> Vec<RowId> {
    (0..n).map(|d| RowId::new((n - 1) - d)).collect()
}

fn region_for(rows: &[RowId]) -> Bytes {
    let mut buf = BytesMut::with_capacity(rows.len() * ENTRY_BYTES);
    for row in rows {
        buf.put_u32_le(row.get());
    }
    buf.freeze()
}

#[test]
fn every_position_resolves_to_the_row_its_region_holds() {
    let rows = rows_for(MAP_DOCS);
    let map = DocMap::permuted(region_for(&rows));

    assert!(map.is_permuted(), "a map with a region is a permutation");

    for d in 0..MAP_DOCS {
        let doc = FtsDocId::new(d);
        assert_eq!(
            map.row_of(doc),
            rows[d as usize],
            "position {d} did not resolve to the row the region holds"
        );
    }
}

#[test]
fn a_region_cut_mid_entry_still_serves_its_whole_entries() {
    let rows = rows_for(MAP_DOCS);
    let mut region = region_for(&rows);
    // Drop one byte, so the last entry is incomplete. Only whole entries are
    // addressable: reading past them trips the map's own assertion, so this
    // checks that truncation does not disturb the entries before the cut.
    region.truncate(region.len() - 1);
    let map = DocMap::permuted(region);

    for d in 0..(MAP_DOCS - 1) {
        let doc = FtsDocId::new(d);
        assert_eq!(
            map.row_of(doc),
            rows[d as usize],
            "position {d} stopped resolving once the region lost its last byte"
        );
    }
}
