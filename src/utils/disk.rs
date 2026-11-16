// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Reads of a local filesystem's size.

use std::path::Path;

use fs4::{available_space, total_space};

/// The free and total bytes of the filesystem holding `path`, read at its
/// nearest existing ancestor. `None` when neither can be read.
pub(crate) fn disk_space(path: &Path) -> Option<(u64, u64)> {
    let existing = path.ancestors().find(|dir| dir.exists())?;
    Some((available_space(existing).ok()?, total_space(existing).ok()?))
}
