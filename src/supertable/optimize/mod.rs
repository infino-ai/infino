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
        // Optimize phase timers ([optphase]); gated, off by default. A measuring
        // stick for compaction scaling — see DiagnosticsSettings.
        let phase_timers = crate::config::global().diagnostics.optimize_phase_timers;
        let mut __t = std::time::Instant::now();
        self.drain_hidden_vector_cells_sync()
            .map_err(|e| OptimizeError::Build(e.to_string()))?;
        if phase_timers {
            eprintln!("[optphase] drain {:.1}s", __t.elapsed().as_secs_f64());
            __t = std::time::Instant::now();
        }
        self.compact_with(&opts.compaction, opts.recalibrate)?;
        if phase_timers {
            eprintln!(
                "[optphase] compact_total {:.1}s",
                __t.elapsed().as_secs_f64()
            );
            __t = std::time::Instant::now();
        }
        // Centroids have settled at the final generation (drain + compaction);
        // pre-build the centroid-router graph so the next centroid-graph query
        // loads it instead of building on the hot path. Best-effort.
        self.refresh_centroid_router_cache();
        if phase_timers {
            eprintln!(
                "[optphase] router_cache {:.1}s",
                __t.elapsed().as_secs_f64()
            );
        }
        // Refresh the global term-stats sidecar over the post-merge
        // membership (compaction's removals dropped any prior reference —
        // see the manifest carry rule). Runs before gc so the sweep's live
        // set names the fresh artifact.
        self.refresh_term_stats_sync()
            .map_err(|e| OptimizeError::Build(e.to_string()))?;
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
