// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable layer — the in-memory cross-superfile query + manifest
//! layer over [`SuperfileBuilder`] / [`SuperfileReader`].
//!
//! A supertable is to superfile what an Iceberg / Delta table is
//! to Parquet: a small in-memory manifest on top of an append-only
//! set of immutable superfile superfiles, queryable as one logical
//! table via SQL + FTS + vector kNN.
//!
//! ## Layout
//!
//! - [`options`] — `SupertableOptions` + `::new` validation.
//! - [`utils::vector_split`] — pulls `FixedSizeList<Float32>` columns
//!   out of an input `RecordBatch` so the scalar-only batch can be
//!   handed to the underlying [`SuperfileBuilder`].
//! - [`utils::idgen`] — 128-bit Snowflake-style generator for the
//!   auto-injected `_id` column.
//! - [`manifest`] — `Manifest`, `SuperfileEntry`, `ScalarStatsAgg`,
//!   `FtsSummaryAgg`, `VectorSummary`, plus the `Bloom` skip-summary
//!   container.
//! - [`handle`] — `Supertable` (clone-shared handle) and
//!   `SupertableReader` (snapshot-pinned reader).

pub(crate) mod build;
pub(crate) mod compaction;
pub mod error;
pub(crate) mod gc;
pub mod handle;
pub mod lazy_source;
pub mod manifest;
pub mod mutations;
pub(crate) mod optimize;
pub mod options;
pub mod query;
pub mod reader_cache;
pub mod stats;
pub mod tombstones;
pub mod utils;
pub mod wal;
pub mod writer;

/// Re-export of [`crate::storage`] under the
/// `supertable::storage::*` path. Storage moved out from
/// under `supertable` so the trait + impls can be consumed
/// by `superfile` (and any other crate-level module)
/// without inverting the layering — the alias preserves
/// existing call-site paths.
// `supertable::storage` alias. `pub` under `test-helpers` (tests reach
// `infino::supertable::storage::*`); `pub(crate)` otherwise so internal
// `supertable::storage::…` paths still resolve without re-exporting a
// crate-private module onto the public surface (which would be E0365).
#[cfg(feature = "test-helpers")]
pub use crate::storage;
#[cfg(not(feature = "test-helpers"))]
pub(crate) use crate::storage;

pub use crate::storage::{
    AzureStorageProvider, LocalFsStorageProvider, ObjectMeta, S3StorageProvider, StorageError,
    StorageProvider,
};
pub use error::{BuildError, CommitError, GcError, OpenError, OptimizeError, QueryError};
pub use gc::GcReport;
pub use handle::{Supertable, SupertableReader};
pub use lazy_source::StorageRangeSource;
pub use manifest::{
    FtsSummaryAgg, Manifest, ManifestLoadError, ManifestPartLoader, ScalarStatsAgg, SuperfileEntry,
    SuperfileList, SuperfileUri, VectorSummary,
};
pub use mutations::MutationStats;
pub use options::SupertableOptions;
pub use reader_cache::{InMemoryReaderCache, ReaderCacheError, SuperfileReaderCache};
pub use stats::SupertableStats;
pub use writer::SupertableWriter;
