// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Superfile-layer integration tests.
//!
//! One test binary (`cargo test --test superfile`) covers:
//!
//! - End-to-end FTS + vector + Parquet build → read
//!   (`pipeline`)
//! - FTS sub-component pipeline + brute-force BM25 oracle
//!   (`fts/*`)
//! - Vector sub-component pipeline + brute-force NN oracle
//!   (`vector/*`)
//! - Format-level coverage: CRC corruption rejection,
//!   DataFusion Parquet compat, lazy-source open path
//!   (`format/*`)
//!
//! Tests live in their natural subdirectory; invoke a
//! subset via cargo's module-path filter, e.g.:
//!
//!   cargo test --test superfile fts::
//!   cargo test --test superfile vector::brute_force_oracle::
//!   cargo test --test superfile format::crc_corruption::

mod format;
mod fts;
mod pipeline;
mod vector;
