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

/// Decoded row-group columns by [`ColumnKey`], bounded by their bytes.
pub(crate) type DecodedColumnCache = ByteLru<ColumnKey, ArrayRef>;

/// The process's decoded-column cache, `None` when the config keeps none.
pub(crate) fn process_decoded_column_cache() -> Option<Arc<DecodedColumnCache>> {
    static CACHE: OnceLock<Option<Arc<DecodedColumnCache>>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let bytes = config::global().memory.decoded_column_cache_bytes;
            (bytes > 0)
                .then(|| Arc::new(ByteLru::new(usize::try_from(bytes).unwrap_or(usize::MAX))))
        })
        .clone()
}
