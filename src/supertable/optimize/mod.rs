// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use crate::{
    Supertable,
    config::OptimizeOptions,
    supertable::{
        error::{GcError, OptimizeError},
        wal::gc::GcError as WalGcError,
    },
};

impl Supertable {
    /// Merge small or underfilled superfiles into larger ones, then run a
    /// best-effort gc sweep (orphaned superfiles/manifests + dead tombstone
    /// sidecars) and a best-effort WAL sweep (completed mutation state and
    /// arrow sidecars). Pass [`OptimizeOptions::default`] for engine
    /// defaults. Requires durable storage.
    #[doc(alias = "compact")]
    pub fn optimize(&self, opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        self.drain_hidden_vector_cells_sync()
            .map_err(|e| OptimizeError::Build(e.to_string()))?;
        self.compact(&opts.compaction)?;
        match self.gc(opts.gc.safety_gap) {
            Ok(_) | Err(GcError::NoStorage) => {}
            Err(e) => return Err(OptimizeError::Gc(e)),
        }
        match self.run_gc_sweep_once_blocking() {
            Ok(_) | Err(WalGcError::NoStorageAttached) => {}
            Err(e) => return Err(OptimizeError::WalGc(e)),
        }
        Ok(())
    }
}
