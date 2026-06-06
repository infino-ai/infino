//! Supertable layer — the in-memory cross-segment query + manifest
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
//! - [`manifest`] — `Manifest`, `SuperfileEntry`, `ScalarStatsTable`,
//!   `FtsSummary`, `VectorSummary`, plus the `Bloom` skip-summary
//!   container.
//! - [`handle`] — `Supertable` (clone-shared handle) and
//!   `SupertableReader` (snapshot-pinned reader).

pub(crate) mod build;
pub mod error;
pub mod handle;
pub mod lazy_source;
pub mod manifest;
pub mod mutations;
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
pub use crate::storage;

pub use crate::storage::{
    AzureStorageProvider, LocalFsStorageProvider, ObjectMeta, S3StorageProvider, StorageError,
    StorageProvider,
};
pub use error::{BuildError, CommitError, OpenError, QueryError};
pub use handle::{Supertable, SupertableReader};
pub use lazy_source::StorageRangeSource;
pub use manifest::{
    FtsSummary, Manifest, ManifestLoadError, ManifestPartLoader, ScalarStatsTable, SuperfileEntry,
    SuperfileList, SuperfileUri, VectorSummary,
};
pub use options::SupertableOptions;
pub use reader_cache::{InMemoryReaderCache, ReaderCacheError, SuperfileReaderCache};
pub use stats::SupertableStats;
pub use writer::SupertableWriter;
