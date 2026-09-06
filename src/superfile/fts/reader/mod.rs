// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS read path. The `FtsReader` type and its query kernels are split
//! across the submodules below; this file only wires them together and
//! re-exports the surface callers reach as `fts::reader::*`.

mod core;
mod count;
mod cursor;
mod expand;
mod filter;
mod metadata;
mod options;
mod phrase;
mod scorers;
mod search;
mod sink;
#[cfg(test)]
mod test_util;
mod work;

pub use core::*;

pub(crate) use expand::{LONG_S_ASCII, TermPattern, has_fold_partner};
pub use metadata::{ColumnMeta, OpenOptions};
pub use options::{Bm25SearchOptions, Bm25Stats, BoolMode};
pub(crate) use search::FetchedTermMemo;
pub(crate) use sink::LiveFloor;
pub use work::MatchWork;
