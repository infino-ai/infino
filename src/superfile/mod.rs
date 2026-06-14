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

pub mod builder;
pub mod error;
pub mod format;
pub mod fts;
pub mod lazy_source;
pub mod reader;
pub mod stats;
pub mod vector;

pub use error::{BuildError, FtsError, ReadError, VectorError};
pub use lazy_source::{BytesLazyByteSource, LazyByteSource, LazyByteSourceError};
pub(crate) use lazy_source::{LazySubSource, PrefetchedSource};
pub use reader::{OpenOptions, SuperfileReader, VectorSearchOptions};
