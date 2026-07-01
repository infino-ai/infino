// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! How the embedded vector blob inside a superfile is organized.

/// Layout of the vector blob referenced by `inf.vec.offset` / `inf.vec.length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum VectorLayout {
    /// Default IVF + RaBitQ multi-subsection blob (`VectorBuilder`).
    #[default]
    Ivf,
    /// Single contiguous cell posting blob (`cell_posting` module).
    /// One GET loads the whole posting list; search scans in memory.
    CellPosting,
    /// SPFresh-style cell-local tree/run blob. The superfile envelope stays the
    /// same; only the vector subsection layout changes.
    Spfresh,
}

impl VectorLayout {
    pub(crate) const KV_VALUE_IVF: &'static str = "ivf";
    pub(crate) const KV_VALUE_CELL_POSTING: &'static str = "cell_posting";
    pub(crate) const KV_VALUE_SPFRESH: &'static str = "spfresh";

    pub(crate) fn as_kv_value(self) -> &'static str {
        match self {
            Self::Ivf => Self::KV_VALUE_IVF,
            Self::CellPosting => Self::KV_VALUE_CELL_POSTING,
            Self::Spfresh => Self::KV_VALUE_SPFRESH,
        }
    }

    pub(crate) fn from_kv_value(s: &str) -> Option<Self> {
        match s {
            Self::KV_VALUE_IVF => Some(Self::Ivf),
            Self::KV_VALUE_CELL_POSTING => Some(Self::CellPosting),
            Self::KV_VALUE_SPFRESH => Some(Self::Spfresh),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::VectorLayout;

    #[test]
    fn kv_value_round_trips_every_layout() {
        for layout in [
            VectorLayout::Ivf,
            VectorLayout::CellPosting,
            VectorLayout::Spfresh,
        ] {
            assert_eq!(
                VectorLayout::from_kv_value(layout.as_kv_value()),
                Some(layout)
            );
        }
        assert_eq!(VectorLayout::from_kv_value("unknown"), None);
    }
}
