// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Telling whoever keeps a reader that a read loaded something into it.
//!
//! A reader loads parts of its superfile the first time a read needs them
//! (the page indexes, the FTS doc-id map, the norms, the vector columns'
//! norms) and keeps them for its life. A holder that weighs the reader by
//! what it holds ([`SuperfileReader::held_bytes`]) watches its loads, so it
//! weighs the reader again as each one lands rather than on its next use.
//!
//! [`SuperfileReader::held_bytes`]: super::SuperfileReader::held_bytes

use std::{
    fmt,
    sync::{Arc, OnceLock},
};

use crate::superfile::lazy_source::ARC_COUNTS_BYTES;

/// What a [`LoadWatch`] calls on each load.
pub(crate) type LoadListener = Arc<dyn Fn() + Send + Sync>;

/// One listener slot, shared by every clone, so a reader's parts and the
/// views derived from them report to the one listener.
#[derive(Clone, Default)]
pub(crate) struct LoadWatch(Arc<OnceLock<LoadListener>>);

impl LoadWatch {
    /// Call `listener` on every later load. The first listener set stays.
    pub(crate) fn watch(&self, listener: &LoadListener) {
        let _ = self.0.set(Arc::clone(listener));
    }

    /// Heap bytes of the listener slot: one allocation, shared by every clone. A vector column
    /// counts its own, one per cell of a multi-cell blob.
    pub(crate) const SLOT_BYTES: usize = ARC_COUNTS_BYTES + size_of::<OnceLock<LoadListener>>();

    /// Report a load. Called with none of the reader's own locks held,
    /// since the listener weighs the reader, which takes them.
    pub(crate) fn loaded(&self) {
        if let Some(listener) = self.0.get() {
            listener();
        }
    }
}

impl fmt::Debug for LoadWatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadWatch")
            .field("watched", &self.0.get().is_some())
            .finish()
    }
}
