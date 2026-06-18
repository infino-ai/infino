// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! How the embedded vector blob inside a superfile is organized.

/// Layout of the vector blob referenced by `inf.vec.offset` / `inf.vec.length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorLayout {
    /// Default IVF + RaBitQ multi-subsection blob (`VectorBuilder`).
    #[default]
    Ivf,
    /// Single contiguous cell posting blob (`cell_posting` module).
    /// One GET loads the whole posting list; search scans in memory.
    CellPosting,
}

impl VectorLayout {
    pub const KV_VALUE_IVF: &'static str = "ivf";
    pub const KV_VALUE_CELL_POSTING: &'static str = "cell_posting";

    pub fn as_kv_value(self) -> &'static str {
        match self {
            Self::Ivf => Self::KV_VALUE_IVF,
            Self::CellPosting => Self::KV_VALUE_CELL_POSTING,
        }
    }

    pub fn from_kv_value(s: &str) -> Option<Self> {
        match s {
            Self::KV_VALUE_IVF => Some(Self::Ivf),
            Self::KV_VALUE_CELL_POSTING => Some(Self::CellPosting),
            _ => None,
        }
    }
}
