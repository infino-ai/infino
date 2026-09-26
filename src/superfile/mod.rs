// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Superfile module — the production-grade implementation of the embedded
//! BM25 + vector format.
//!
//! See `docs/architecture/superfile.md` for the live design reference.
//!
//! ## In-tree caller invariant
//!
//! The only in-tree caller of `SuperfileBuilder` /
//! `SuperfileReader` is the `supertable` layer. The
//! supertable owns the multi-superfile + manifest + storage
//! policy; each rayon shard worker uses `SuperfileBuilder`
//! one-shot (one `add_batch` loop → one `finish()`), and
//! each cached / opened superfile runs through
//! `SuperfileReader::open` once per cache hydration. The
//! builder is consume-on-`finish()`; a session that wants N
//! superfiles instantiates N builders.

pub(crate) mod bits;
pub mod builder;
pub mod error;
pub mod format;
pub mod fts;
// Visible to the layer-isolated integration tests and benches under
// `test-helpers`, `pub(crate)` otherwise — the same treatment the reader
// itself gets, so the id types stay off the shipped public contract while
// still typing the signatures those tests call.
#[cfg(feature = "test-helpers")]
pub mod id_space;
#[cfg(not(feature = "test-helpers"))]
pub(crate) mod id_space;
pub(crate) mod ids;
pub mod lazy_source;
pub mod reader;
pub mod stats;
pub mod vector;

pub use error::{BuildError, FtsError, ReadError, VectorError};
pub use lazy_source::{BytesLazyByteSource, LazyByteSource, LazyByteSourceError};
pub(crate) use lazy_source::{LazySubSource, PrefetchedSource};
pub use reader::{OpenOptions, SuperfileReader, VectorSearchOptions};
