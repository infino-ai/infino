// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

#[cfg(feature = "detailed-tracing")]
use crate::utils::trace::OpOrigin;
use crate::{
    config::OptimizeOptions,
    supertable::{
        Supertable,
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
    // Shares every step below with the detached background sweeps, so the
    // span tags it `optimize`: same code, but a caller is blocked on it and
    // the latency is theirs.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(
            skip_all,
            fields(role = self.role().as_str(), origin = OpOrigin::Optimize.as_str())
        )
    )]
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
