// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Optimizing an index: the phases a caller asks for with `optimize()`.
//!
//! Compaction is the one phase this module owns outright; it lives in
//! [`compact`]. Every other phase (the hidden-cell drain, the
//! centroid-router refresh, the term-index upkeep, gc) is a maintenance
//! operation with callers of its own, sequenced here rather than
//! implemented here.

pub(crate) mod compact;

use std::time::Instant;

use compact::max_merge_input_bytes;
use fs4::available_space;

#[cfg(feature = "detailed-tracing")]
use crate::utils::trace::OpOrigin;
use crate::{
    config::{OptimizeOptions, scratch_root},
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
        if let Some(spec) = &opts.adjacency
            && opts.steps.check_adjacency_spec
        {
            self.check_adjacency_spec(spec)
                .map_err(OptimizeError::from)?;
        }

        if opts.steps.drain_hidden_cells {
            self.drain_hidden_vector_cells_sync()
                .map_err(OptimizeError::from)?;
            if phase_timers {
                tracing::info!(secs = __t.elapsed().as_secs_f64(), "[optphase] drain");
                __t = Instant::now();
            }
        }

        if opts.steps.compact {
            self.compact(&opts.compaction)?;
            if phase_timers {
                tracing::info!(
                    secs = __t.elapsed().as_secs_f64(),
                    "[optphase] compact_total"
                );
                __t = Instant::now();
            }
        }

        if opts.steps.router_cache_warmup {
            // Centroids have settled at the final generation (drain + compaction);
            // pre-build the centroid-router graph so the next centroid-graph query
            // loads it instead of building on the hot path. Best-effort.
            self.refresh_centroid_router_cache();

            if phase_timers {
                tracing::info!(
                    secs = __t.elapsed().as_secs_f64(),
                    "[optphase] router_cache"
                );
            }
        }
        // Index the superfiles the term index does not list, a batch at a
        // time, and fold its layers in runs no larger than a compaction
        // merge's input, or than half the free space where the fold spills
        // if that is more: a fold downloads its run there before merging,
        // so a host with room folds a root of fifty layers in one pass
        // where the merge cap alone took four, each reading its layers back
        // from storage and publishing a root. The merge cap stays the floor,
        // as it was before free space was read. Runs before gc so the sweep
        // keeps the roots it publishes.
        if opts.steps.refresh_term_index {
            let spill_root = self
                .inner()
                .options
                .term_index_build_policy()
                .scratch_root
                .unwrap_or_else(scratch_root);
            let scratch_bytes = available_space(spill_root).map_or(0, |bytes| bytes / 2);
            let fold_max_bytes = max_merge_input_bytes(&opts.compaction).max(scratch_bytes);
            self.refresh_term_index_sync(fold_max_bytes)
                .map_err(OptimizeError::from)?;
        }

        // The knowledge graph's adjacency over an edge table, rebuilt only
        // when the rows, the deletes or the edge columns changed; before gc
        // for the same reason. Best-effort like the hnsw build above: the
        // compaction is committed by now, and the walks keep serving the
        // prior generation until a later pass republishes, so a failed
        // publish is logged rather than failing an optimize that is done.
        #[cfg(feature = "graph-index")]
        if let Some(spec) = &opts.adjacency
            && opts.steps.refresh_adjacency_spec
            && let Err(e) = self.refresh_adjacency_sync(spec)
        {
            tracing::warn!(
                "adjacency refresh failed: {e}; the walks keep the prior generation until a \
                 later optimize republishes"
            );
        }

        if opts.steps.gc {
            match self.gc(opts.gc.safety_gap) {
                Ok(_) | Err(GcError::NoStorage) => {}
                Err(e) => return Err(OptimizeError::Gc(e)),
            }
        }
        if opts.steps.gc_sweep_once {
            match self.run_gc_sweep_once_blocking() {
                Ok(_) | Err(WalGcError::NoStorageAttached) => {}
                Err(e) => return Err(OptimizeError::WalGc(e)),
            }
        }
        Ok(())
    }
}
