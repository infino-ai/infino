// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Optimizing an index: the phases a caller asks for with `optimize()`.
//!
//! Compaction is the one phase this module owns outright; it lives in
//! [`compact`]. Every other phase (the hidden-cell drain, the
//! centroid-router refresh, the term-index rebuild, gc) is a maintenance
//! operation with callers of its own, sequenced here rather than
//! implemented here.

pub(crate) mod compact;

use std::time::Instant;

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
        // Optimize phase timers ([optphase]); gated, off by default. A measuring
        // stick for compaction scaling — see DiagnosticsSettings.
        let phase_timers = crate::config::global().diagnostics.optimize_phase_timers;
        let mut __t = Instant::now();
        // The edge columns are checked against the schema first: a wrong
        // spec is the caller's error, and it fails here with nothing
        // committed rather than after the compaction below.
        #[cfg(feature = "graph-index")]
        if let Some(spec) = &opts.adjacency {
            self.check_adjacency_spec(spec)
                .map_err(OptimizeError::from)?;
        }
        self.drain_hidden_vector_cells_sync()
            .map_err(OptimizeError::from)?;
        if phase_timers {
            tracing::info!(secs = __t.elapsed().as_secs_f64(), "[optphase] drain");
            __t = Instant::now();
        }
        self.compact_with(&opts.compaction, opts.recalibrate)?;
        if phase_timers {
            tracing::info!(
                secs = __t.elapsed().as_secs_f64(),
                "[optphase] compact_total"
            );
            __t = Instant::now();
        }

        if !opts.skip_router_cache_warmup {
            // Centroids have settled at the final generation (drain + compaction);
            // pre-build the centroid-router graph so the next centroid-graph query
            // loads it instead of building on the hot path. Best-effort.
            self.refresh_centroid_router_cache();
        }

        if phase_timers {
            tracing::info!(
                secs = __t.elapsed().as_secs_f64(),
                "[optphase] router_cache"
            );
        }
        // Rebuild the term index over the merged superfiles. Runs before gc
        // so the sweep keeps the fresh index and removes the one it replaced.
        self.refresh_term_index_sync()
            .map_err(OptimizeError::from)?;
        // The knowledge graph's adjacency over an edge table, rebuilt only
        // when the rows, the deletes or the edge columns changed; before gc
        // for the same reason. Best-effort like the hnsw build above: the
        // compaction is committed by now, and the walks keep serving the
        // prior generation until a later pass republishes, so a failed
        // publish is logged rather than failing an optimize that is done.
        #[cfg(feature = "graph-index")]
        if let Some(spec) = &opts.adjacency
            && let Err(e) = self.refresh_adjacency_sync(spec)
        {
            tracing::warn!(
                "adjacency refresh failed: {e}; the walks keep the prior generation until a \
                 later optimize republishes"
            );
        }
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
