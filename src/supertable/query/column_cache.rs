// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The SQL scan's decoded columns, kept across statements.
//!
//! A table scan decompresses and decodes each row group's column chunks on
//! every statement, and on a column of numbers that decode is most of what a
//! filter or an aggregate over it costs. Superfiles are immutable, so one row
//! group's column reads to the same array for as long as the file lives, under
//! one schema generation. Keeping that array lets the next scan of the column
//! read it instead of the pages ([`CachedScanExec`]).
//!
//! One cache serves the process, bounded by `memory.decoded_column_cache_bytes`;
//! the least recently used arrays leave first. With the setting at `0` there is
//! none, and every scan decodes as it always has.
//!
//! A scan that uses a few rows of a row group reads just those rows and keeps
//! nothing the first time, so one selective statement doesn't push out the
//! columns others read whole. The cache remembers that row group, and a
//! later scan of it decodes and keeps its columns.
//!
//! [`CachedScanExec`]: super::exec::cached_scan_exec::CachedScanExec

use std::sync::{Arc, OnceLock};

use arrow_array::ArrayRef;

use crate::{
    config,
    supertable::{manifest::SuperfileUri, query::scalar_cache::ByteLru},
};

/// One row group's column of one superfile, as a scan reads it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ColumnKey {
    pub(crate) uri: SuperfileUri,
    /// The schema generation the column was read under: a column's name,
    /// type and identity belong to it, so the array answers only for it.
    pub(crate) schema_id: u32,
    /// The table's name for the column.
    pub(crate) column: String,
    pub(crate) row_group: usize,
}

/// One row group of one superfile, under one schema generation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RowGroupKey {
    pub(crate) uri: SuperfileUri,
    pub(crate) schema_id: u32,
    pub(crate) row_group: usize,
}

/// How many row groups the cache remembers as read in part; the least
/// recently read leave first.
const READ_IN_PART_REMEMBERED: usize = 1 << 16;

/// Decoded row-group columns by [`ColumnKey`], bounded by their bytes, and
/// the row groups a scan read only some rows of without keeping them.
#[derive(Debug)]
pub(crate) struct DecodedColumnCache {
    columns: ByteLru<ColumnKey, ArrayRef>,
    /// Each counted as one byte against [`READ_IN_PART_REMEMBERED`].
    read_in_part: ByteLru<RowGroupKey, ()>,
}

impl DecodedColumnCache {
    /// A cache keeping up to `max_bytes` of decoded columns.
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            columns: ByteLru::new(max_bytes),
            read_in_part: ByteLru::new(READ_IN_PART_REMEMBERED),
        }
    }

    /// The most bytes the kept columns may hold together.
    pub(crate) fn max_bytes(&self) -> usize {
        self.columns.max_bytes()
    }

    /// The column kept under `key`, marking it used.
    pub(crate) fn get(&self, key: &ColumnKey) -> Option<ArrayRef> {
        self.columns.get(key)
    }

    /// Keep `column`, which holds `bytes`, under `key`.
    pub(crate) fn insert(&self, key: ColumnKey, column: ArrayRef, bytes: usize) {
        self.columns.insert(key, column, bytes);
    }

    /// Whether a scan read some rows of `row_group` before without keeping
    /// its columns.
    pub(crate) fn read_in_part_before(&self, row_group: &RowGroupKey) -> bool {
        self.read_in_part.get(row_group).is_some()
    }

    /// Remember that a scan read some rows of `row_group` without keeping
    /// its columns.
    pub(crate) fn note_read_in_part(&self, row_group: RowGroupKey) {
        self.read_in_part.insert(row_group, (), 1);
    }

    /// How many columns are kept.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.columns.len()
    }
}

/// The process's decoded-column cache, `None` when the config keeps none.
pub(crate) fn process_decoded_column_cache() -> Option<Arc<DecodedColumnCache>> {
    static CACHE: OnceLock<Option<Arc<DecodedColumnCache>>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let bytes = config::global().memory.decoded_column_cache_bytes;
            (bytes > 0).then(|| {
                Arc::new(DecodedColumnCache::new(
                    usize::try_from(bytes).unwrap_or(usize::MAX),
                ))
            })
        })
        .clone()
}
