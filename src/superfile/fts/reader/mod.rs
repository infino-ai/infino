// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS read path. The `FtsReader` type and its query kernels are split
//! across the submodules below; this file only wires them together and
//! re-exports the surface callers reach as `fts::reader::*`.

mod core;
mod cursor;
mod filter;
mod options;
mod phrase;
mod sink;
mod work;

pub use core::*;
pub use options::{Bm25SearchOptions, Bm25Stats, BoolMode};
pub use work::MatchWork;
