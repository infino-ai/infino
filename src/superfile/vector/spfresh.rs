// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! SPFresh hidden-vector-index layout selection.
//!
//! This is only the P0 scaffold: it selects which vector subsection layout the
//! derived hidden vector-index table should use. The default is the current IVF
//! path so enabling the new layout is an explicit opt-in.

use std::env;
use std::sync::OnceLock;

use super::layout::VectorLayout;

/// Environment variable selecting the hidden vector-index layout.
pub(crate) const HIDDEN_INDEX_LAYOUT_ENV: &str = "INFINO_HIDDEN_INDEX";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HiddenIndexLayout {
    /// Current nested hidden-index layout: global `VectorCell` routing with IVF
    /// vector subsections inside hidden superfiles.
    Nested,
    /// New superfile SPFresh vector subsection layout, still under the existing
    /// global `VectorCell` outer routing.
    Spfresh,
}

impl HiddenIndexLayout {
    pub(crate) fn vector_layout(self) -> VectorLayout {
        match self {
            Self::Nested => VectorLayout::Ivf,
            Self::Spfresh => VectorLayout::Spfresh,
        }
    }
}

/// Selected hidden-index layout. Cached so a process does not switch formats
/// halfway through building/opening a hidden vector-index table.
pub(crate) fn hidden_index_layout() -> HiddenIndexLayout {
    static LAYOUT: OnceLock<HiddenIndexLayout> = OnceLock::new();
    *LAYOUT.get_or_init(|| {
        env::var(HIDDEN_INDEX_LAYOUT_ENV)
            .ok()
            .and_then(|value| parse_hidden_index_layout(value.trim()))
            .unwrap_or(HiddenIndexLayout::Nested)
    })
}

fn parse_hidden_index_layout(value: &str) -> Option<HiddenIndexLayout> {
    match value.to_ascii_lowercase().as_str() {
        "" | "nested" | "ivf" => Some(HiddenIndexLayout::Nested),
        "spfresh" => Some(HiddenIndexLayout::Spfresh),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{HiddenIndexLayout, parse_hidden_index_layout};
    use crate::superfile::vector::layout::VectorLayout;

    #[test]
    fn parses_layout_names() {
        assert_eq!(
            parse_hidden_index_layout("nested"),
            Some(HiddenIndexLayout::Nested)
        );
        assert_eq!(
            parse_hidden_index_layout("ivf"),
            Some(HiddenIndexLayout::Nested)
        );
        assert_eq!(
            parse_hidden_index_layout("spfresh"),
            Some(HiddenIndexLayout::Spfresh)
        );
        assert_eq!(parse_hidden_index_layout("opann"), None);
    }

    #[test]
    fn maps_to_vector_layout() {
        assert_eq!(HiddenIndexLayout::Nested.vector_layout(), VectorLayout::Ivf);
        assert_eq!(
            HiddenIndexLayout::Spfresh.vector_layout(),
            VectorLayout::Spfresh
        );
    }
}
