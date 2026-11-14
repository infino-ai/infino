// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A table's schema after creation: what the schema write changes, what a
//! type change does to the files already written, what a document can grow,
//! and what every one of those looks like to a reader.

pub mod evolution;
pub mod property;
pub mod rows;
pub mod sparse;
pub mod types;
// Reads the process's memory through mimalloc.
#[cfg(feature = "mimalloc")]
pub mod wide_commit;
