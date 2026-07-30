// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Vector kNN fan-out on [`Supertable`](super::super::Supertable).
//!
//! ## Public API
//!
//! The sync, user-facing entry points live on
//! [`Supertable`](super::super::Supertable):
//!
//! ```ignore
//! let opts = VectorSearchOptions::new();
//! // Bare call: `_id` + `score` only — no scalar decode.
//! let ids: Vec<RecordBatch> = table.vector_search("emb", &query_vec, 10, opts, None, None)?;
//! // Materialize row data by naming the columns to decode.
//! let rows: Vec<RecordBatch> = table.vector_search(
//!     "emb",
//!     &query_vec,
//!     10,
//!     opts,
//!     None,
//!     Some(&["_id", "title", "score"]),
//! )?;
//! ```
//!
//! Internally these drive the async kernel on the snapshot-pinned
//! [`SupertableReader`], whose `vector_search` (rows) / `vector_hits`
//! ([`SuperfileHit`], superfile-local) methods are the engine-facing
//! surface. Results are sorted by distance *ascending* — smaller is
//! closer (cosine: `1 - dot`, L2-sq: squared distance).
//!
//! ## Strategy
//!
//! Internally pins a snapshot reader and drives the async
//! kernel to completion via the sync→async bridge. The reader
//! holds a pinned `Arc<ManifestSnapshot>`; for each visible superfile we:
//!
//!   1. Fetch the superfile's `SuperfileReader` from the store.
//!   2. Delegate to `SuperfileReader::vector_search`
//!      (cluster-aware IVF + 1-bit RaBitQ shortlist + full-precision
//!      rerank, all inside one superfile).
//!   3. Tag each `(local_doc_id, distance)` with the superfile URI.
//!   4. Concatenate across superfiles and global-top-k by distance.
//!
//! Unlike BM25, vector distances are inherently comparable across
//! superfiles — both cosine and L2-sq are functions of the query
//! and the per-doc vector only, not of superfile-scoped statistics.
//! So the per-superfile top-k → concatenate → global top-k pattern
//! recovers exact recall (modulo each per-superfile IVF's nprobe-
//! driven recall tradeoff, which is identical to the single-
//! superfile case).
//!
//! Fan-out uses centroid pruning:
//!
//!   1. **Score & sort** — compute `distance(query, centroid)`
//!      for each superfile (SIMD-accelerated: AVX-512 / AVX2 /
//!      NEON) and sort ascending. This is free — centroids are
//!      manifest metadata, no S3 GETs.
//!   2. **Search closest** — search the top `k*2` (min 3)
//!      superfiles in parallel (`tokio::spawn` per superfile).
//!      Merge results via bounded heap.
//!
//! Every skipped superfile is a batch of GET requests the
//! object-store-native engine never issues. For cold queries
//! this is the difference between seconds and milliseconds.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet},
    future::Future,
    mem,
    sync::{Arc, Mutex, PoisonError},
    time::Instant,
};

use arrow::record_batch::RecordBatch;
use arrow_array::{Array, Decimal128Array};
use futures::future::try_join_all;
use roaring::RoaringBitmap;
use tokio::join;
use uuid::Uuid;

use super::{
    SuperfileHit,
    candidate::CandidatePlan,
    dispatch,
    exec::common::{SCORE_COLUMN, id_score_batch, resolve_hits_named, take_rows_byte_source},
    prune::{PruneLeaf, select_superfiles},
};
pub use crate::superfile::reader::VectorSearchOptions;
use crate::{
    config,
    storage::io_counters,
    superfile::{
        SuperfileReader,
        error::ReadError,
        fts::reader::BoolMode,
        vector::{
            distance::{Metric, distance, relative_score_window},
            layout::VectorLayout,
            reader::ScanCandidate,
        },
    },
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::{
            ManifestSnapshot, RABITQ_ADMIT_CELL_SHORTLIST_FRACTION,
            RABITQ_ADMIT_CELL_SHORTLIST_MIN, RabitqAdmitQuery, SuperfileEntry, SuperfileUri,
            VectorSummary,
            list::{CellRoutingParams, PartitionStrategy},
        },
        opann::REPLICA_CLOSURE_DISTANCE_RATIO,
        slow_vector_state::{CentroidSection, fetch_centroid_section},
        tombstones::SidecarCache,
    },
};

/// A calibrated law width at or below this resolves to `None`: one cell
/// is exactly the fine-first default's sweep, so the law only engages
/// when it demands a WIDER read than the default already performs.
const LAW_WIDTH_WITHIN_DEFAULT: usize = 1;

/// Oversample factor on the per-sweep rerank budget. Dividing
/// `k x rerank_mult` evenly across the sweep under-serves the nearest
/// cells (they hold most true neighbors, and the 1-bit estimate ranks
/// some of them deep within their cell): an even split measured
/// 0.9695 recall@100 on Cohere-1M vs 0.9937 for undivided caps, and 2x
/// headroom still only 0.9865. 4x holds the bar while keeping the total
/// survivor budget well below the undivided width sweep.
const WIDTH_BUDGET_OVERSAMPLE: usize = 4;

/// Candidate growth when a deleted row occupies a current top-k slot.
const DELETE_REFILL_GROWTH_FACTOR: usize = 2;

test_visible! {
/// Fallback fine-probe scale for untagged (pre-grid) user manifests, and
/// the widest explicit coarse sweep benches exercise. The routed user path
/// no longer defaults to this: unfiltered queries with no caller `nprobe`
/// use the same bounded cell routing as the hidden index (one grid-nearest
/// cell, slack-widened on near-ties) and span every commit fragment holding
/// the selected cell.
const USER_COARSE_CELLS: usize = 16;
}

test_visible! {
/// Fine IVF runs probed per (superfile, cell) fragment on the user path.
/// A fragment's fine runs are ranked by centroid distance and only the
/// closest runs are probed, so the 16-cell coarse sweep does not multiply
/// per-fragment read volume the way probing every run would.
const USER_FINE_RUNS_PER_FRAGMENT: usize = 8;
}

/// Filtered-default cell probe width: the same user-table search with the
/// allow-set pushed down, probing a fixed 4 grid cells instead of the
/// fine-first single cell. The nearest MATCHING rows sit deeper in the
/// unfiltered ranking (~rank k/selectivity) and spread across more cells,
/// so p=1 under-reaches (measured 0.489 @ 3.3 ms at 1M/256 with ~10%
/// selectivity) while wide sweeps pay latency far past parity (32 cells:
/// 0.827 @ 18 ms). Fine-run coverage inside a probed cell is already
/// complete at 4 runs (drain-diag: p4=1.000) and the default keeps 8.
/// Explicit caller `nprobe` overrides; the per-run width sweep keeps the
/// trade measured.
const FILTERED_USER_CELL_NPROBE: usize = 4;

// The admit window keeps the shared
// `manifest::RABITQ_ADMIT_CELL_SHORTLIST_FRACTION` (20%) slice of the
// ranked tagged cells for exact fp32 rescoring, floored by
// [`RABITQ_ADMIT_CELL_SHORTLIST_MIN`]. A cell's rank is its best fine
// centroid's 1-bit estimate, and every fine inside a kept cell is
// rescored exactly — so the window only has to land the exact-best cell
// (plus near-tie companions) *somewhere* inside it. 20% keeps the same
// coverage class as the recall-validated 48-of-256 window (post-drain
// recall matched the exact-everything scan at 0.995) while scaling with
// the ranked population — a fixed 48 under-covers larger grids (under
// 5% of 1024 cells). Applies identically to hidden cells and user
// commit fragments (one code path).

// The window floor is the shared
// `manifest::RABITQ_ADMIT_CELL_SHORTLIST_MIN` (48): below it the
// prefilter degenerates to scoring everything — identical to the exact
// path — and it is the validated absolute window at the 256-cell shapes,
// so small tables never see a narrower window than the measured one.

/// Minimum fine-ranked picks in the union cell selection used by the
/// non-default paths (filtered search, explicit caller nprobe). The fine
/// ranking's second pick closes the last coverage gap when the grid is
/// very coarse — measured at 10M/64c: fine p1 coverage 0.919 (union recall
/// landed exactly on it at 0.921) vs fine p2 coverage 0.997. An explicit
/// caller probe width larger than this takes precedence.
const UNION_FINE_PICKS_MIN: usize = 2;

/// Cell-probe floor for filtered (allow-set) queries over the hidden cell
/// index. The manifest's default routing (fine-first p=1) is calibrated
/// for unfiltered search, where fine p1 cell coverage measures 1.000; an
/// allow-set thins each cell's matching postings (~10% selectivity in the
/// bench), so the nearest *matching* neighbors spread past the top cell
/// and a narrow probe caps filtered recall well below the unfiltered
/// number. Consolidated cells make width nearly free under a filter
/// (allow-first shortlist + bounded rerank): width is nearly free
/// because the probe cost is carried by matching rows, not cells. The
/// 1M/256 sweep at 16-fine depth measured 6 cells → 0.873 @ 1.36 ms,
/// 128 → 0.940 @ 1.48 ms; the 10M/256 sweep measured 160 → 0.902 @
/// 4.16 ms, 224 → 0.933 @ 4.91 ms, 256 → 0.933 @ 5.16 ms. The full
/// grid buys the recall plateau for ~1 ms over the 128 default at 10M,
/// so filtered sweeps every cell — the residual loss is in-cell depth
/// ([`FILTERED_HIDDEN_FINE_NPROBE`]), not width. NOTE: absolute width
/// (= the whole pinned 256-cell grid); if the grid grows past it the
/// dial becomes a fraction.
const FILTERED_HIDDEN_CELL_NPROBE: usize = 256;

/// Fine-run probe depth inside each hidden cell for filtered queries.
/// The unfiltered default (8) is calibrated for top-10 neighbors, whose
/// fine-run coverage saturates at 4 (drain-diag p4 = 1.000); a filter's
/// nearest MATCHING rows sit at unfiltered rank ~k/selectivity and live
/// in deeper runs — the width sweep's 0.856 plateau across 6..16 cells
/// is in-cell loss, recovered by probing deeper, not wider.
const FILTERED_HIDDEN_FINE_NPROBE: usize = 16;

/// Build the fine-cluster probe set, then refill globally (best score first)
/// toward `gated_target` postings. Candidates without a cell go to `scored`
/// for the flat (non-cell) path.
///
/// The floor's grouping key depends on `generation_of`:
///
/// * `Some(birth_versions)` — the hidden drain path. A drain wave writes one
///   packed superfile per shard, each spanning several cells, all sharing the
///   wave's `birth_version`. Keep `max(keep_floor, floor(keep_pct × runs))`
///   runs **per drain wave, pooled across every cell and shard that wave
///   wrote** — so depth scales with the wave's fine-run count. A freshly
///   drained delta wave keeps its share of the shortlist beside the large base
///   wave, yet read volume tracks the number of drain waves — not the
///   probed-cell count, which the older per-`(cell, superfile)` key multiplied
///   against.
/// * `None` — the user/pre-drain path. Keep the same proportional amount per
///   `(cell, superfile)`: an undrained cell's rows scatter across every commit
///   fragment, so each fragment of a selected cell is probed (accepted read
///   amplification), and a small fragment is not crowded out by a larger
///   sibling in the same cell.
fn gate_fine_candidates_by_fragment(
    candidates: Vec<(usize, u32, f32, Option<u32>, u64)>,
    selected: &HashSet<u32>,
    selected_ordered: &[u32],
    keep_floor: usize,
    keep_pct: f64,
    gated_target: u64,
    candidate_counts: &HashMap<(usize, u32), u64>,
    scored: &mut Vec<(usize, u32, f32)>,
    generation_of: Option<&[u64]>,
) -> Vec<(usize, u32, f32)> {
    // Fine runs kept per group scale with the group's size: a fraction of its
    // fine-cluster count, never below the floor. So large cells (more fine
    // clusters) are probed proportionally deeper while small cells hold at the
    // floor.
    let fine_keep = |available: usize| -> usize {
        ((keep_pct * available as f64).floor() as usize)
            .max(keep_floor)
            .max(1)
            .min(available)
    };
    // Shared global refill: append best-scored leftovers until the shortlist
    // holds `gated_target` postings.
    let refill = |mut gated: Vec<(usize, u32, f32)>,
                  mut remaining: Vec<(usize, u32, f32, u64)>|
     -> Vec<(usize, u32, f32)> {
        let mut postings: u64 = gated
            .iter()
            .map(|(si, cluster, _)| candidate_counts.get(&(*si, *cluster)).copied().unwrap_or(0))
            .sum();
        if postings < gated_target {
            remaining.sort_unstable_by(|a, b| {
                a.2.partial_cmp(&b.2)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
            });
            for (si, cluster, score, count) in remaining {
                gated.push((si, cluster, score));
                postings += count;
                if postings >= gated_target {
                    break;
                }
            }
        }
        gated
    };
    if let Some(gen_of) = generation_of {
        // Pool a drain wave's fine runs across all its cells and shards, keyed
        // by `birth_version`, and keep the closest `fine_keep(runs)` per wave.
        let mut fine_by_generation: HashMap<u64, Vec<(usize, u32, f32, u64)>> = HashMap::new();
        for (si, cluster, score, cell, count) in candidates {
            match cell {
                Some(cell) if selected.contains(&cell) => {
                    let generation = gen_of.get(si).copied().unwrap_or(0);
                    fine_by_generation
                        .entry(generation)
                        .or_default()
                        .push((si, cluster, score, count));
                }
                Some(_) => {}
                None => scored.push((si, cluster, score)),
            }
        }
        let mut gated = Vec::new();
        let mut remaining = Vec::new();
        let mut generations: Vec<u64> = fine_by_generation.keys().copied().collect();
        generations.sort_unstable();
        for generation in generations {
            let Some(mut fine) = fine_by_generation.remove(&generation) else {
                continue;
            };
            fine.sort_unstable_by(|a, b| {
                a.2.partial_cmp(&b.2)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
            });
            let keep = fine_keep(fine.len());
            let tail = fine.split_off(keep);
            gated.extend(
                fine.into_iter()
                    .map(|(si, cluster, score, _)| (si, cluster, score)),
            );
            remaining.extend(tail);
        }
        return refill(gated, remaining);
    }
    let mut fine_by_fragment: HashMap<(u32, usize), Vec<(u32, f32, u64)>> = HashMap::new();
    for (si, cluster, score, cell, count) in candidates {
        match cell {
            Some(cell) if selected.contains(&cell) => fine_by_fragment
                .entry((cell, si))
                .or_default()
                .push((cluster, score, count)),
            Some(_) => {}
            None => scored.push((si, cluster, score)),
        }
    }
    let mut gated = Vec::new();
    let mut remaining = Vec::new();
    for &cell in selected_ordered {
        let mut fragment_ids: Vec<usize> = fine_by_fragment
            .keys()
            .filter_map(|(candidate_cell, si)| (*candidate_cell == cell).then_some(*si))
            .collect();
        fragment_ids.sort_unstable();
        for si in fragment_ids {
            let Some(mut fine) = fine_by_fragment.remove(&(cell, si)) else {
                continue;
            };
            fine.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            let keep = fine_keep(fine.len());
            let tail = fine.split_off(keep);
            gated.extend(
                fine.into_iter()
                    .map(|(cluster, score, _)| (si, cluster, score)),
            );
            remaining.extend(
                tail.into_iter()
                    .map(|(cluster, score, count)| (si, cluster, score, count)),
            );
        }
    }
    refill(gated, remaining)
}

/// Rank cells by their best (minimum) fine-run score among the query's
/// candidates — the fine-centroid cell ranking. Ascending score, ties broken
/// by lower cell id; cells with no candidate fine run are absent (they hold
/// no committed rows for this query's fan-out and cannot be probed anyway).
fn cells_ranked_by_fine_score(
    candidates: &[(usize, u32, f32, Option<u32>, u64)],
) -> Vec<(u32, f32)> {
    let mut best: HashMap<u32, f32> = HashMap::new();
    for &(_, _, score, cell, _) in candidates {
        if let Some(cell) = cell {
            best.entry(cell)
                .and_modify(|s| *s = s.min(score))
                .or_insert(score);
        }
    }
    let mut ranked: Vec<(u32, f32)> = best.into_iter().collect();
    ranked.sort_unstable_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked
}

/// Sum indexed row counts per cell from manifest vector summaries — no
/// distance work. Drives posting-widen and "does this cell exist?" checks
/// before fine centroid scoring.
fn postings_by_cell_from_summaries(
    superfiles: &[Arc<SuperfileEntry>],
    column: &str,
    allow: Option<&HashMap<SuperfileUri, Arc<RoaringBitmap>>>,
    superseded: &BTreeMap<Uuid, BTreeSet<u32>>,
) -> (HashMap<u32, u64>, bool) {
    let mut postings: HashMap<u32, u64> = HashMap::new();
    let mut any_tagged = false;
    for entry in superfiles {
        if allow.is_some_and(|m| !m.contains_key(&entry.uri)) {
            continue;
        }
        let Some(vs) = entry.vector_summary.get(column) else {
            continue;
        };
        for cell in &vs.cells {
            let Some(cell_id) = cell.cell_id else {
                continue;
            };
            // Cells superseded by an in-place split carry dead on-disk
            // blocks: exclude them so they are never routed to, scored,
            // or fetched — the split's successor cells hold the live rows.
            if superseded
                .get(&entry.superfile_id)
                .is_some_and(|s| s.contains(&cell_id))
            {
                continue;
            }
            any_tagged = true;
            let n: u64 = cell.clusters.counts.iter().map(|&c| u64::from(c)).sum();
            *postings.entry(cell_id).or_default() += n;
        }
    }
    (postings, any_tagged)
}

/// 1-bit admit window for `ranked_cells` distinct tagged cells: the
/// [`RABITQ_ADMIT_CELL_SHORTLIST_FRACTION`] slice of the ranked
/// population, floored by [`RABITQ_ADMIT_CELL_SHORTLIST_MIN`].
fn admit_shortlist_window(ranked_cells: usize) -> usize {
    let scaled = (ranked_cells as f64 * RABITQ_ADMIT_CELL_SHORTLIST_FRACTION).ceil() as usize;
    scaled.max(RABITQ_ADMIT_CELL_SHORTLIST_MIN)
}

/// One admit fine-centroid candidate:
/// `(superfile index, flat cluster id, score, cell id, indexed doc count)`.
type FineCandidate = (usize, u32, f32, Option<u32>, u64);

/// A summary cell selected for exact scoring whose fp32 centroids were
/// dropped at hydration (`summary_centroids_from_superfiles`): its exact
/// scores are read from the superfile's on-disk centroid region through
/// the reader cache instead.
struct DeferredCellRescore {
    si: usize,
    cell_id: Option<u32>,
    flat_base: u32,
}

// Stripped summary cells (fp32 dropped at hydration) always DEFER to an
// exact rescore — 1-bit estimates in routing measurably cost recall
// (filtered measured 0.722 against the 0.95 bar when the user path ranked
// on estimates). The rescore is cheap in both regimes: hidden manifests
// read the slow-CAS centroid-section spill; user manifests hydrate fp32
// once per generation from the FULL manifest parts (the user table's
// content-addressed fp32 store). See `rescore_deferred_cells`.

/// Validate one superfile's vector summary for `column` (present,
/// non-empty, dims matching the query) and hand it back. Shared by the
/// prefilter and exact passes of [`score_fine_candidates`].
fn eligible_summary<'e>(
    entry: &'e SuperfileEntry,
    column: &str,
    query_dim: usize,
) -> Result<&'e VectorSummary, QueryError> {
    match entry.vector_summary.get(column) {
        Some(vs) if !vs.cells.is_empty() => {
            for cell in &vs.cells {
                if cell.clusters.dim as usize != query_dim {
                    return Err(QueryError::Execute(format!(
                        "vector summary dimension {} for column `{column}` on superfile {} \
                         does not match query dimension {query_dim}",
                        cell.clusters.dim, entry.superfile_id,
                    )));
                }
            }
            Ok(vs)
        }
        Some(_) => Err(QueryError::Execute(format!(
            "superfile {} has no cluster centroids in its vector summary for \
             column `{column}` — malformed build; refusing to degrade to a \
             blind per-superfile probe",
            entry.superfile_id
        ))),
        None => Err(QueryError::Execute(format!(
            "superfile {} has no vector summary for column `{column}` — \
             malformed build; refusing to degrade to a blind per-superfile \
             probe",
            entry.superfile_id
        ))),
    }
}

/// Score fine IVF centroids in the eligible superfile summaries.
///
/// With `admit` set (`(prefilter query, must-include cells)`), a 1-bit
/// XOR+popcount pass first ranks tagged cells by their best estimated
/// fine-centroid score and keeps the [`admit_shortlist_window`] top
/// slice plus the caller's grid picks; the exact fp32 scan below then
/// skips instances outside that set. Every *emitted* score is exact
/// fp32, so routing and near-tie logic never see 1-bit noise — the
/// estimates only bound which cells get exact-scored. Untagged
/// (`cell_id: None`) summaries are always exact-scored.
fn score_fine_candidates(
    superfiles: &[Arc<SuperfileEntry>],
    column: &str,
    query: &[f32],
    metric: Metric,
    admit: Option<(&RabitqAdmitQuery, &[u32])>,
    allow: Option<&HashMap<SuperfileUri, Arc<RoaringBitmap>>>,
    superseded: &BTreeMap<Uuid, BTreeSet<u32>>,
) -> Result<(Vec<FineCandidate>, Vec<DeferredCellRescore>), QueryError> {
    let eligible = |entry: &Arc<SuperfileEntry>| allow.is_none_or(|m| m.contains_key(&entry.uri));
    // A cell superseded by an in-place split carries dead on-disk blocks. Skip
    // it in both the admit shortlist and the scoring/defer loop below, so its
    // blocks are never fine-scored, gated, or fetched — the same guard
    // `postings_by_cell_from_summaries` applies to routing. The split's
    // successor cells hold the live rows.
    let is_superseded = |entry: &Arc<SuperfileEntry>, cell_id: u32| {
        superseded
            .get(&entry.superfile_id)
            .is_some_and(|s| s.contains(&cell_id))
    };

    let shortlist: Option<HashSet<u32>> = if let Some((admit_q, must_include)) = admit {
        let mut cell_best: HashMap<u32, f32> = HashMap::new();
        for entry in superfiles.iter().filter(|e| eligible(e)) {
            let vs = eligible_summary(entry, column, query.len())?;
            for cell in &vs.cells {
                let Some(cell_id) = cell.cell_id else {
                    continue;
                };
                if is_superseded(entry, cell_id) {
                    continue;
                }
                let Some(est) = cell.clusters.estimate_min_admit_score(metric, admit_q) else {
                    continue;
                };
                cell_best
                    .entry(cell_id)
                    .and_modify(|best| {
                        if est < *best {
                            *best = est;
                        }
                    })
                    .or_insert(est);
            }
        }
        let mut ranked: Vec<(u32, f32)> = cell_best.into_iter().collect();
        ranked.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut keep: HashSet<u32> = ranked
            .iter()
            .take(admit_shortlist_window(ranked.len()))
            .map(|(cell, _)| *cell)
            .collect();
        keep.extend(must_include.iter().copied());
        Some(keep)
    } else {
        None
    };

    let mut candidates: Vec<FineCandidate> = Vec::new();
    let mut deferred: Vec<DeferredCellRescore> = Vec::new();
    for (si, entry) in superfiles.iter().enumerate() {
        if !eligible(entry) {
            continue;
        }
        let vs = eligible_summary(entry, column, query.len())?;
        let mut flat_base = 0u32;
        for cell in &vs.cells {
            // Flat cluster ids must stay identical whether or not a cell is
            // skipped, so flat_base always advances.
            let skipped = cell.cell_id.is_some_and(|cid| is_superseded(entry, cid))
                || shortlist
                    .as_ref()
                    .is_some_and(|keep| cell.cell_id.is_some_and(|cid| !keep.contains(&cid)));
            if !skipped {
                if cell.clusters.vectors_resident() {
                    cell.clusters
                        .score_clusters_into(metric, query, |local, score| {
                            let count = cell
                                .clusters
                                .counts
                                .get(local as usize)
                                .copied()
                                .unwrap_or(0) as u64;
                            candidates.push((si, flat_base + local, score, cell.cell_id, count));
                        });
                } else {
                    deferred.push(DeferredCellRescore {
                        si,
                        cell_id: cell.cell_id,
                        flat_base,
                    });
                }
            }
            flat_base = flat_base.saturating_add(cell.clusters.n_cent);
        }
    }
    Ok((candidates, deferred))
}

/// Union of the grid-ranked and fine-ranked cell selections, in probe
/// priority order: grid picks first, then fine picks not already selected.
///
/// The two rankings fail in opposite regimes, so probing their union holds
/// the coverage floor at every measured scale. Small cells make fine
/// centroids noisy — grid ranking wins. Large cells make the single grid
/// centroid a poor proxy — fine ranking wins. Used for filtered search and
/// explicit caller `nprobe`; default unfiltered stays fine-first.
fn union_cell_selection(grid: &[u32], fine: &[u32]) -> Vec<u32> {
    let mut selected: Vec<u32> = Vec::with_capacity(grid.len() + fine.len());
    for &cell in grid.iter().chain(fine) {
        if !selected.contains(&cell) {
            selected.push(cell);
        }
    }
    selected
}

/// Default-path cell selection, shared by the hidden (post-drain) and user
/// (pre-drain) branches: probe the fine-ranked top cell, adding the grid's
/// top cell only when its own fine score is a genuine near-tie of the fine
/// winner (same relative window replica closure uses at drain time, so
/// probing and replication agree on what counts as a boundary). At the
/// shipped grid shapes (256/1024 cells) fine p1 coverage measures 1.000
/// (drain-diag, 1M–100M), so a second unconditional pick only multiplies
/// the probed-cell fan without recall to show for it.
fn fine_first_cell_selection(fine_ranked: &[(u32, f32)], grid_top: Option<u32>) -> Vec<u32> {
    let Some(&(fine_top, fine_top_score)) = fine_ranked.first() else {
        return grid_top.into_iter().collect();
    };
    let mut cells = vec![fine_top];
    if let Some(grid_top) = grid_top
        && grid_top != fine_top
    {
        let tie_threshold =
            relative_score_window(fine_top_score, REPLICA_CLOSURE_DISTANCE_RATIO - 1.0);
        let grid_top_fine_score = fine_ranked
            .iter()
            .find(|(cell, _)| *cell == grid_top)
            .map(|(_, score)| *score);
        if grid_top_fine_score.is_some_and(|score| score <= tie_threshold) {
            cells.push(grid_top);
        }
    }
    cells
}

/// Map a per-superfile vector-search error to a query error. A budget refusal
/// keeps its own variant (found via `ReadError::over_budget`) so it surfaces as
/// the public `InfinoError::OverBudget`; anything else is a generic query error.
fn vector_read_query_error(e: ReadError) -> QueryError {
    if let Some(msg) = e.over_budget() {
        return QueryError::OverBudget(msg.to_string());
    }
    QueryError::Parquet(e.to_string())
}

/// An optional text-predicate filter for vector kNN search. When
/// supplied, kNN is ranked only among rows matching the predicate
/// (pushdown, not post-filter). Built from an FTS-indexed column, a
/// query string, and a [`BoolMode`].
pub struct VectorFilter<'a> {
    /// FTS-indexed column the predicate applies to.
    pub column: &'a str,
    /// Query string — tokenized with the index tokenizer.
    pub query: &'a str,
    /// Token matching mode (AND / OR).
    pub mode: BoolMode,
}

/// Prepared per-superfile allow-set for filtered vector kNN.
///
/// When `use_hidden_index` is true, `allow_by_uri` is keyed by hidden-index
/// superfile URIs (file-local ids). Otherwise it is keyed by user-table URIs.
#[derive(Clone)]
#[cfg(feature = "test-helpers")]
pub struct PreparedGlobalAllow {
    use_hidden_index: bool,
    allow_by_uri: HashMap<SuperfileUri, Arc<RoaringBitmap>>,
}

/// Prepared per-superfile allow-set for filtered vector kNN.
///
/// When `use_hidden_index` is true, `allow_by_uri` is keyed by hidden-index
/// superfile URIs (file-local ids). Otherwise it is keyed by user-table URIs.
#[derive(Clone)]
#[cfg(not(feature = "test-helpers"))]
pub(crate) struct PreparedGlobalAllow {
    use_hidden_index: bool,
    allow_by_uri: HashMap<SuperfileUri, Arc<RoaringBitmap>>,
}

/// Resolve stable user ids to their Parquet `(superfile, local_doc_id)`.
///
/// Contiguous id-ordered files use arithmetic. Cell-packed/gapped files are
/// read concurrently and each `_id` column is decoded at most once for the
/// entire top-k. The previous per-hit lookup could decode the same full column
/// twice per hit: once to identify its owner and again to locate its row.
async fn lookup_user_placements_by_id(
    manifest: &ManifestSnapshot,
    user_row_ids: &[i128],
) -> Result<Vec<(Arc<SuperfileEntry>, u32)>, QueryError> {
    if user_row_ids.is_empty() {
        return Ok(Vec::new());
    }
    let id_column = manifest.options.id_column.as_str();
    let entries = manifest
        .get_all_superfiles_loaded()
        .await
        .map_err(QueryError::ManifestLoad)?;
    let mut placements: Vec<Option<(Arc<SuperfileEntry>, u32)>> = vec![None; user_row_ids.len()];
    let mut gapped = Vec::new();

    for entry in entries {
        let matching: Vec<usize> = user_row_ids
            .iter()
            .enumerate()
            .filter_map(|(index, &id)| {
                (placements[index].is_none() && id >= entry.id_min && id <= entry.id_max)
                    .then_some(index)
            })
            .collect();
        if matching.is_empty() {
            continue;
        }
        if row_id_from_manifest_entry(&entry, 0).is_some() {
            for index in matching {
                let local = u32::try_from(user_row_ids[index] - entry.id_min).map_err(|_| {
                    QueryError::Execute(format!(
                        "local_doc_id out of range for id {}",
                        user_row_ids[index]
                    ))
                })?;
                placements[index] = Some((Arc::clone(&entry), local));
            }
        } else {
            gapped.push(entry);
        }
    }

    let decoded = try_join_all(gapped.into_iter().map(|entry| async move {
        let locals: Vec<u32> = (0..entry.n_docs as u32).collect();
        // Cell-packed user files carry stable ids inline in Parquet row order.
        // Read each compact cell region once instead of decoding the full
        // Parquet `_id` column to place a top-k scalar projection.
        let ids = read_ids_for_locals(manifest, &entry, &locals, id_column, true).await?;
        Ok::<_, QueryError>((entry, ids))
    }))
    .await?;
    let mut requested: HashMap<i128, Vec<usize>> = HashMap::new();
    for (index, &id) in user_row_ids.iter().enumerate() {
        if placements[index].is_none() {
            requested.entry(id).or_default().push(index);
        }
    }
    for (entry, ids) in decoded {
        for (local, id) in ids.into_iter().enumerate() {
            let Some(indexes) = requested.get(&id) else {
                continue;
            };
            let local = u32::try_from(local).map_err(|_| {
                QueryError::Execute(format!(
                    "local_doc_id out of range in user superfile {:?}",
                    entry.uri
                ))
            })?;
            for &index in indexes {
                if placements[index].is_none() {
                    placements[index] = Some((Arc::clone(&entry), local));
                }
            }
        }
    }

    placements
        .into_iter()
        .enumerate()
        .map(|(index, placement)| {
            placement.ok_or_else(|| {
                QueryError::Execute(format!("no user superfile owns id {}", user_row_ids[index]))
            })
        })
        .collect()
}

/// Extract the `_id` column (column 0, Decimal128) of `batch` as `Vec<i128>`.
fn id_values_from_batch(batch: &RecordBatch) -> Result<Vec<i128>, QueryError> {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .map(|a| a.values().to_vec())
        .ok_or_else(|| QueryError::Execute("_id column missing".into()))
}

/// Resolve a stable row id from manifest span arithmetic when the superfile
/// body stores rows in contiguous id order. `None` when the id span is gapped
/// (not a single contiguous append), so the caller must read the `_id` column.
pub(crate) fn row_id_from_manifest_entry(
    entry: &SuperfileEntry,
    local_doc_id: u32,
) -> Option<i128> {
    if entry.vector_layout == VectorLayout::MultiCellIvf {
        return None;
    }
    let n_docs = i128::from(entry.n_docs);
    let span = entry.id_max.checked_sub(entry.id_min)?.checked_add(1)?;
    if n_docs == 0 || span != n_docs {
        return None;
    }
    Some(entry.id_min + i128::from(local_doc_id))
}

/// Stable `_id` for every row in `entry` (`local` → `id_min + local` when the
/// manifest span is contiguous, else targeted column reads). Same tier order as
/// [`hidden_hits_user_ids`]: span arithmetic → resident `take_by_local_doc_ids`
/// → [`read_ids_for_locals`].
pub(crate) async fn stable_ids_by_local_for_routing(
    manifest: &ManifestSnapshot,
    entry: &SuperfileEntry,
    reader: &SuperfileReader,
) -> Result<Vec<i128>, QueryError> {
    if row_id_from_manifest_entry(entry, 0).is_some() {
        return Ok((0..entry.n_docs as u32)
            .map(|local| entry.id_min + i128::from(local))
            .collect());
    }
    let locals: Vec<u32> = (0..reader.n_docs() as u32).collect();
    // Hidden cell superfiles inline the stable `_id` in the IVF blob — resolve
    // straight from it (resident; no scalar `_id` column read) before falling
    // back to the column.
    if let Some(ids) = reader
        .vec()
        .and_then(|v| v.inline_stable_ids_for_locals(&locals))
    {
        return Ok(ids);
    }
    let id_column = reader.id_column();
    if reader.parquet_bytes().is_some() {
        let batch = reader
            .take_by_local_doc_ids(&locals, &[id_column])
            .map_err(|e| QueryError::Execute(e.to_string()))?;
        return id_values_from_batch(&batch);
    }
    read_ids_for_locals(manifest, entry, &locals, id_column, true).await
}

/// Read the `_id` column values at `local_ids` (in caller order) from one
/// superfile. Routed through the disk cache as a resident (mmap) read when a
/// cache is attached; falls back to object-store range GETs on lazy readers.
///
/// `allow_inline_region` selects the resolution source:
///
///   - `true` — prefer the IVF blob's inline `_id` region (hidden cells).
///   - `false` — never use the inline region; read the scalar `_id` column
///     (user superfiles after compaction — inline region is cluster-ordered).
async fn read_ids_for_locals(
    manifest: &ManifestSnapshot,
    entry: &SuperfileEntry,
    local_ids: &[u32],
    id_column: &str,
    allow_inline_region: bool,
) -> Result<Vec<i128>, QueryError> {
    // Storage is optional: store-only tables (no object-store backend) serve
    // the superfile bytes from the in-memory reader cache. Cell-ordered
    // MultiCell user commits resolve `_id` through here even without storage.
    let storage = manifest.options.storage.as_ref();
    let store = Arc::clone(&manifest.options.store);
    let disk_cache = manifest.options.disk_cache.as_ref();
    let reader = dispatch::open_reader(&store, disk_cache, storage, entry, false).await?;
    // The inline IVF region is usable as an `_id` source only when its rows
    // map 1:1 to Parquet rows. Boundary-replicated user commits break that:
    // the IVF carries stub rows beyond the Parquet count, so inline order
    // diverges from Parquet row order and the shortcut would pair ids with
    // the wrong locals. Hidden cells keep the shortcut — their replicas are
    // real Parquet rows, so the counts (and orders) match.
    let inline_is_parquet_ordered = reader.vec().is_none_or(|v| v.n_docs() == reader.n_docs());
    if allow_inline_region && inline_is_parquet_ordered {
        // Hidden cell superfiles inline the stable `_id` in the IVF blob — resolve
        // straight from it (resident; no scalar `_id` column read) when available.
        if let Some(ids) = reader
            .vec()
            .and_then(|v| v.inline_stable_ids_for_locals(local_ids))
        {
            return Ok(ids);
        }
        // Cold path: fetch the inline region async when present but not resident.
        if let Some(v) = reader.vec()
            && let Some(ids) = v
                .inline_stable_ids_for_locals_async(local_ids)
                .await
                .map_err(|e| QueryError::Execute(e.to_string()))?
        {
            return Ok(ids);
        }
    }
    if reader.parquet_bytes().is_some() {
        let batch = reader
            .take_by_local_doc_ids(local_ids, &[id_column])
            .map_err(|e| QueryError::Execute(e.to_string()))?;
        return id_values_from_batch(&batch);
    }
    let batch = take_rows_byte_source(&reader, local_ids, &[id_column])
        .await
        .map_err(|error| QueryError::Execute(error.to_string()))?;
    id_values_from_batch(&batch)
}

/// Remap step 1 (deduped): resolve the user `_id` that dual-write stamped into
/// each hidden-index hit, returned in `hidden_hits` order.
///
/// Arithmetic when a hidden superfile's id span is contiguous. The hidden index
/// is cell-partitioned, so a cell aggregates scattered user ids and its id
/// range is usually gapped — `id_min + local` rarely holds — so the common case
/// is a column read. Hits are grouped by hidden superfile so each gapped
/// superfile's `_id` column is read **once** (resident via the disk cache),
/// reading only the rows the hits touch — versus the previous per-hit
/// object-store read that dominated warm latency.
async fn hidden_hits_user_ids(
    hidden_manifest: &ManifestSnapshot,
    hidden_hits: &[SuperfileHit],
    id_column: &str,
) -> Result<Vec<i128>, QueryError> {
    let mut ids = vec![0i128; hidden_hits.len()];
    let mut by_superfile: HashMap<SuperfileUri, Vec<usize>> = HashMap::new();
    for (i, hit) in hidden_hits.iter().enumerate() {
        // Piggyback fast path: the search already resolved the user `_id`
        // from the inline region (prefetched in the fan-out wave) and stamped
        // it here — reuse it and skip this superfile's region/scalar read
        // entirely. Hits without it (incoming superfiles have no inline
        // region) fall through to the grouped read below.
        if let Some(id) = hit.stable_id {
            ids[i] = id;
            continue;
        }
        by_superfile.entry(hit.superfile).or_default().push(i);
    }
    for (uri, idxs) in by_superfile {
        let entry = hidden_manifest
            .lookup_superfile_entry(uri)
            .await
            .map_err(QueryError::ManifestLoad)?
            .ok_or_else(|| {
                QueryError::Execute(format!("hidden superfile {uri:?} missing from manifest"))
            })?;
        // Contiguous span → arithmetic, no read.
        if row_id_from_manifest_entry(&entry, 0).is_some() {
            for &i in &idxs {
                ids[i] = entry.id_min + i128::from(hidden_hits[i].local_doc_id);
            }
            continue;
        }
        // Gapped span → one resident read of just the rows these hits touch.
        let locals: Vec<u32> = idxs.iter().map(|&i| hidden_hits[i].local_doc_id).collect();
        let vals = read_ids_for_locals(hidden_manifest, &entry, &locals, id_column, true).await?;
        for (j, &i) in idxs.iter().enumerate() {
            ids[i] = vals[j];
        }
    }
    Ok(ids)
}

fn projection_is_id_score_only(projection: Option<&[&str]>, id_column: &str) -> bool {
    match projection {
        None => true,
        Some(names) => names == [id_column, SCORE_COLUMN] || names == [SCORE_COLUMN, id_column],
    }
}

fn is_hidden_vector_manifest(manifest: &ManifestSnapshot) -> bool {
    matches!(
        manifest.partition_strategy(),
        Some(PartitionStrategy::VectorCell { .. })
    )
}

/// Build `_id` + `score` directly from search-wave stable-ID stamps.
///
/// Identity resolution belongs before this boundary. Reaching output
/// materialization without a stamp is an upstream query bug; this function
/// never opens a manifest part or Parquet `_id` page.
pub(crate) fn hits_id_score_batch(
    user_reader: &SupertableReader,
    hits: &[SuperfileHit],
) -> Result<RecordBatch, QueryError> {
    let mut ids = Vec::with_capacity(hits.len());
    let mut scores = Vec::with_capacity(hits.len());
    for hit in hits {
        let id = hit.stable_id.ok_or_else(|| {
            QueryError::Execute(format!(
                "hit {:?}/{} missing stable _id before output materialization",
                hit.superfile, hit.local_doc_id
            ))
        })?;
        ids.push(id);
        scores.push(hit.score);
    }
    id_score_batch(user_reader, &ids, &scores).map_err(|e| QueryError::Execute(e.to_string()))
}

/// Locate each hit's user-table `(superfile, local_doc_id)` for scalar
/// column decode. Hidden-index hits already carry the user `_id` on
/// `stable_id`; user-table hits pass through unchanged.
pub(crate) async fn user_placement_for_scalar_resolve(
    user_reader: &SupertableReader,
    hits: &[SuperfileHit],
) -> Result<Vec<SuperfileHit>, QueryError> {
    if hits.is_empty() {
        return Ok(Vec::new());
    }
    let user_manifest = user_reader.manifest();
    let id_column = user_reader.options().id_column.as_str();
    let hidden_manifest = user_reader
        .vector_index_table()
        .map(|vit| Arc::clone(vit.pinned_reader().manifest()));
    let deleted = user_reader
        .vector_index_table()
        .and_then(|vit| vit.pinned_reader().hidden_deleted_ids().ok());
    let mut out: Vec<Option<SuperfileHit>> = vec![None; hits.len()];
    let mut placement_requests: Vec<(usize, i128)> = Vec::new();
    for (i, hit) in hits.iter().enumerate() {
        if let Some(user_entry) = user_manifest
            .lookup_superfile_entry(hit.superfile)
            .await
            .map_err(QueryError::ManifestLoad)?
            && !(user_entry.vector_layout == VectorLayout::MultiCellIvf && hit.stable_id.is_some())
        {
            out[i] = Some(*hit);
            continue;
        }
        let user_row_id = if let Some(id) = hit.stable_id {
            id
        } else if let Some(ref hm) = hidden_manifest {
            hidden_hits_user_ids(hm, std::slice::from_ref(hit), id_column).await?[0]
        } else {
            return Err(QueryError::Execute(format!(
                "hit superfile {:?} missing from manifests",
                hit.superfile
            )));
        };
        if deleted
            .as_ref()
            .is_some_and(|d| d.binary_search(&user_row_id).is_ok())
        {
            continue;
        }
        placement_requests.push((i, user_row_id));
    }
    let requested_ids: Vec<i128> = placement_requests.iter().map(|(_, id)| *id).collect();
    let placements = lookup_user_placements_by_id(user_manifest, &requested_ids).await?;
    for ((index, stable_id), (entry, local_doc_id)) in
        placement_requests.into_iter().zip(placements)
    {
        out[index] = Some(SuperfileHit {
            superfile: entry.uri,
            local_doc_id,
            score: hits[index].score,
            stable_id: Some(stable_id),
        });
    }
    Ok(out.into_iter().flatten().collect())
}

/// Score one deferred cell's fp32 centroids into `candidates` — shared by
/// the centroid-section (hidden) and full-part (user) rescore sources.
/// Returns false when the entry's summary doesn't validate against the
/// fp32 slice (caller keeps the cell deferred for the per-superfile
/// fallback wave).
fn score_cell_fp32(
    superfiles: &[Arc<SuperfileEntry>],
    column: &str,
    d: &DeferredCellRescore,
    fp32: &[f32],
    query: &[f32],
    metric: Metric,
    candidates: &mut Vec<FineCandidate>,
) -> bool {
    let entry = &superfiles[d.si];
    let Some(cell) = entry
        .vector_summary
        .get(column)
        .and_then(|vs| vs.cells.iter().find(|cell| cell.cell_id == d.cell_id))
    else {
        return false;
    };
    let dim = cell.clusters.dim as usize;
    if dim == 0 || fp32.len() != cell.clusters.n_cent as usize * dim {
        return false;
    }
    for (local, centroid) in fp32.chunks_exact(dim).enumerate() {
        let count = cell.clusters.counts.get(local).copied().unwrap_or(0) as u64;
        if count == 0 {
            continue;
        }
        let score = distance(metric, query, centroid);
        candidates.push((d.si, d.flat_base + local as u32, score, d.cell_id, count));
    }
    true
}

impl SupertableReader {
    /// Hydrate (or reuse) the slow-CAS centroid-section spill for this
    /// table: one streamed fetch of a single content-addressed object on
    /// the first cold rescore, then local `pread`s forever — instead of
    /// one block GET per shortlisted cell per query. `None` when the
    /// manifest carries no section ref (legacy) or the fetch failed
    /// (callers fall back to per-superfile centroid reads).
    async fn centroid_section(&self) -> Option<Arc<CentroidSection>> {
        let manifest = self.manifest();
        let reference = manifest.slow_vector_state_centroids_blob()?.clone();
        let storage = manifest.options.storage.as_ref()?;
        let slot = Arc::clone(&manifest.options.centroid_section_cache);
        // The lock is deliberately held ACROSS the fetch: it makes the
        // one-time hydration single-flight, so concurrent cold queries
        // wait for one section download instead of each pulling the whole
        // object. Steady state holds it only long enough to clone the Arc.
        let mut guard = slot.lock().await;
        if let Some(section) = guard.as_ref()
            && section.uri() == reference.uri
        {
            return Some(Arc::clone(section));
        }
        let entries = manifest.get_all_superfiles();
        match fetch_centroid_section(storage.as_ref(), &reference, entries).await {
            Ok(section) => {
                let section = Arc::new(section);
                *guard = Some(Arc::clone(&section));
                Some(section)
            }
            Err(error) => {
                eprintln!(
                    "[supertable] centroid section {} unavailable ({error}); deferred rescores \
                     will fail unless the parts cache covers their cells",
                    reference.uri
                );
                None
            }
        }
    }

    /// Exact admit scores for summary cells whose fp32 was dropped at
    /// hydration. Two sources, both manifest-published state, and they
    /// are exhaustive: hidden (VectorCell) manifests read the slow-CAS
    /// centroid-section spill (one object per generation — see
    /// [`Self::centroid_section`]); user manifests read the fp32
    /// hydrated once per generation from the FULL manifest parts. A cell
    /// neither can serve is corrupted routing state — the publish paths
    /// guarantee every stripped cell is covered (the section composer
    /// fails a republish rather than leave a hole) — so the query fails
    /// loudly instead of degrading onto some slower read path.
    async fn rescore_deferred_cells(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        column: &str,
        query: &[f32],
        metric: Metric,
        candidates: &mut Vec<FineCandidate>,
        deferred: Vec<DeferredCellRescore>,
    ) -> Result<(), QueryError> {
        let deferred = if deferred.is_empty() {
            deferred
        } else if let Some(section) = self.centroid_section().await {
            let mut leftovers = Vec::new();
            for d in deferred {
                let entry = &superfiles[d.si];
                let read = section
                    .read_cell(entry.superfile_id, column, d.cell_id)
                    .map_err(|e| {
                        QueryError::Execute(format!("centroid section spill read: {e}"))
                    })?;
                let Some(fp32) = read else {
                    leftovers.push(d);
                    continue;
                };
                if !score_cell_fp32(superfiles, column, &d, &fp32, query, metric, candidates) {
                    leftovers.push(d);
                }
            }
            leftovers
        } else {
            deferred
        };
        // User manifests carry no centroid section; their fp32 lives in
        // the FULL manifest parts (content-addressed), hydrated once per
        // generation and served from RAM after that.
        let deferred = if deferred.is_empty() {
            deferred
        } else if let Some(cache) = self.manifest().user_centroids_for_rescore().await {
            let mut leftovers = Vec::new();
            for d in deferred {
                let entry = &superfiles[d.si];
                let Some(fp32) = cache.cell(entry.superfile_id, column, d.cell_id) else {
                    leftovers.push(d);
                    continue;
                };
                if !score_cell_fp32(
                    superfiles,
                    column,
                    &d,
                    fp32.as_slice(),
                    query,
                    metric,
                    candidates,
                ) {
                    leftovers.push(d);
                }
            }
            leftovers
        } else {
            deferred
        };
        if let Some(d) = deferred.first() {
            let entry = &superfiles[d.si];
            return Err(QueryError::Execute(format!(
                "deferred admit rescore: no manifest-published fp32 covers superfile {} column \
                 {column} cell {:?} ({} cell(s) uncovered) — the centroid section / full parts \
                 must cover every stripped summary cell",
                entry.superfile_id,
                d.cell_id,
                deferred.len(),
            )));
        }
        Ok(())
    }

    /// Global cross-superfile cluster selection + waved fan-out. Shared
    /// by the user-table path and the hidden vector-index path.
    async fn fanout_vector_clusters(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }
        self.vector_fanout_over_superfiles(superfiles.to_vec(), column, query, k, options, None)
            .await
    }

    async fn vector_fanout_over_superfiles(
        &self,
        superfiles: Vec<Arc<SuperfileEntry>>,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        allow: Option<HashMap<SuperfileUri, Arc<RoaringBitmap>>>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let filtered = allow.is_some();
        let (resolved_nprobe, _) = options.resolve(filtered);
        let manifest = self.manifest();
        let hidden_vector_index = is_hidden_vector_manifest(manifest);
        // Borrow routing — do not clone the VectorCell centroid grid just to
        // read Copy `CellRoutingParams` (that clone used to drop the transposed
        // SIMD cache and force a per-query scalar transpose rebuild).
        let hidden_routing = manifest.vector_cell_routing();
        // The user-table path owns its coarse default (16 cells) for the
        // untagged fallback sweep. The filtered UNDRAINED-tail fan keeps
        // the default user-table search shape (fine-first p=1 + near-tie
        // slack) with the allow-set pushed down — latency parity with
        // unfiltered by design; drained rows route through the hidden
        // cell index (see `route_filtered_vector_hits_async`). Explicit
        // caller overrides keep the resolved value; hidden routing merges
        // its persisted CellRoutingParams with the filtered floor below.
        let nprobe = if !hidden_vector_index && !filtered && options.nprobe.is_none() {
            USER_COARSE_CELLS
        } else {
            resolved_nprobe
        };

        // ---- Global cross-superfile cluster selection.
        //
        // Each kept superfile's manifest summary carries its per-cluster
        // fp32 centroids. Rank every (superfile, cluster) with [`distance`]
        // on the resident centroid slices (zero-copy, no dequant), then
        // probe only the globally-closest clusters.
        // Undeclared column = caller error, rejected here — not a silent
        // L2Sq default that fails later with a per-superfile decode error.
        // `rot_seed` feeds the 1-bit admit prefilter (same rotation as the
        // column's row codes).
        let (metric, rot_seed) = manifest
            .options
            .vector_columns
            .iter()
            .find(|vc| vc.column == column)
            .map(|vc| (vc.metric, vc.rot_seed))
            .ok_or_else(|| QueryError::Execute(format!("unknown vector column `{column}`")))?;

        let grid = manifest
            .global_vector_index()
            .filter(|g| g.column == column)
            // Route on the same grid commit packing stamped cell tags from:
            // the finer user grid when trained, else the drain grid. (Hidden
            // manifests carry no `global_vector_index` and take the
            // `VectorCell` branch below.)
            .map(|g| g.user_grid())
            .filter(|grid| grid.n_cent > 0 && grid.dim as usize == query.len())
            .or_else(|| {
                manifest
                    .vector_cell_clusters(column)
                    .filter(|clusters| clusters.n_cent > 0 && clusters.dim as usize == query.len())
            });
        // Admit: rank the coarse grid, score every fine IVF centroid in
        // eligible summaries, then fine/grid cell selection + per-fragment
        // gate. Phase timers (INFINO_TRACE_VECTOR_WARM_PHASES): admit covers
        // that work; fanout_wall is probe+rerank+remap wall.
        let admit_t0 = io_counters::phase_start();
        let ranked_cells_scored: Option<Vec<(u32, f32)>> =
            grid.map(|grid| grid.rank_cells(metric, query));
        let ranked_cells: Option<Vec<u32>> = ranked_cells_scored
            .as_ref()
            .map(|cells| cells.iter().map(|(cell, _)| *cell).collect());

        // Cell cutoff shared by the hidden and user branches: probe the
        // `nprobe_min` nearest cells under GRID ranking, widening toward
        // `nprobe_max` while a cell's score stays within the slack threshold
        // of the nearest cell.
        let grid_cell_cutoff = |ranked: &[(u32, f32)], routing: &CellRoutingParams| -> usize {
            if ranked.is_empty() {
                return 0;
            }
            let mut cutoff = routing.nprobe_min.max(1).min(ranked.len());
            let max_cells = routing.nprobe_max.max(routing.nprobe_min).min(ranked.len());
            // Same window definition replica closure uses at drain time
            // (`relative_score_window`), so probing and replication agree
            // on what counts as a near-tie.
            let threshold = relative_score_window(ranked[0].1, routing.slack);
            while cutoff < max_cells && ranked[cutoff].1 <= threshold {
                cutoff += 1;
            }
            cutoff
        };
        let birth_versions: Vec<u64> = superfiles.iter().map(|e| e.birth_version).collect();
        let gated_target = (k as f64
            * f64::from(config::global().vector.drain_replica_target_factor.max(1.0)))
        .ceil() as u64;
        let allow_ref = allow.as_ref();
        // Cells retired by an in-place split are excluded from routing so
        // their dead on-disk blocks are never selected or fetched.
        let empty_superseded = BTreeMap::new();
        let superseded = manifest.get_superseded_cells().unwrap_or(&empty_superseded);
        let (postings_by_cell, any_tagged) =
            postings_by_cell_from_summaries(&superfiles, column, allow_ref, superseded);

        let mut gated = Vec::new();
        let mut scored = Vec::new();
        // Width the sweep was pinned to (caller nprobe or the width law);
        // `None` on the fine-first default and legacy paths.
        let mut sweep_width: Option<usize> = None;
        // Assigned in both admit arms; used below for the posting-aware
        // budget expand (keep scoring until we cover ≥ k postings).
        let candidate_counts: HashMap<(usize, u32), u64>;
        if let (Some(ranked_scored), true) = (&ranked_cells_scored, any_tagged) {
            // Base routing shape first (per branch), then one shared caller
            // override on top.
            let mut cell_routing = if hidden_vector_index {
                let base = hidden_routing.ok_or_else(|| {
                    QueryError::Execute("hidden manifest missing cell routing".into())
                })?;
                if filtered {
                    // Allow-set queries widen to the filtered floor and
                    // probe DEEPER fine runs per cell — the matching
                    // neighbors sit past the unfiltered top runs; the
                    // manifest's persisted routing still wins if broader.
                    CellRoutingParams {
                        nprobe_min: base.nprobe_min.max(FILTERED_HIDDEN_CELL_NPROBE),
                        nprobe_max: base.nprobe_max.max(FILTERED_HIDDEN_CELL_NPROBE),
                        fine_nprobe: base.fine_nprobe.max(FILTERED_HIDDEN_FINE_NPROBE),
                        ..base
                    }
                } else {
                    base
                }
            } else if filtered {
                // Filtered UNDRAINED-tail fan: the default user-table
                // search with a small fixed floor
                // ([`FILTERED_USER_CELL_NPROBE`]) — the nearest MATCHING
                // rows sit deeper than the fine-first single cell reaches.
                CellRoutingParams {
                    nprobe_min: FILTERED_USER_CELL_NPROBE,
                    nprobe_max: FILTERED_USER_CELL_NPROBE,
                    ..CellRoutingParams::default()
                }
            } else {
                CellRoutingParams::default()
            };
            // Per-table probe-width law: when the drain calibrated one and
            // the caller passed nothing, the law's width for this `k` acts
            // exactly like an explicit caller `nprobe` — same pin, same
            // per-fragment gating downstream. How far the true top-k
            // spreads over cells is a property of the corpus (synthetic
            // clustered data calibrates to 1 cell at k=10; Cohere-1M/768d
            // measured ~30 of 256 cells at k=100), so the default width
            // must be per-table and per-k, never a constant. A law width
            // of 1 resolves to `None` and keeps the fine-first p=1 path
            // byte-for-byte.
            let law_width: Option<usize> =
                if hidden_vector_index && !filtered && options.nprobe.is_none() {
                    hidden_routing
                        .and_then(|r| r.width_for_k_at(k))
                        .filter(|w| *w > LAW_WIDTH_WITHIN_DEFAULT)
                } else {
                    None
                };
            let populated_cells = postings_by_cell.len().max(1);
            sweep_width = apply_width_pin(
                &mut cell_routing,
                options.nprobe.map(|n| n.max(1)),
                law_width,
                filtered,
                populated_cells,
            );
            // Per-cell fine probe = max(floor, floor(pct × cell fine-cluster
            // count)), so depth scales with cell size. Filtered queries keep
            // their own fixed fine floor (pct = 0); the proportional depth
            // applies to the unfiltered default path. The floor comes from
            // routing (config-defaulted, or the hidden manifest's stamp); the
            // fraction is `vector.fine_nprobe_pct` (0.0 ⇒ off ⇒ fixed floor),
            // set via config.yaml / ./infino.yaml — no env var, no rebuild.
            let fine_nprobe_pct = if filtered {
                0.0
            } else {
                config::global().vector.fine_nprobe_pct
            };
            let ranked_for_beam: Vec<(u32, f32)> = ranked_scored
                .iter()
                .filter(|(cell, _)| postings_by_cell.contains_key(cell))
                .copied()
                .collect();
            if ranked_for_beam.is_empty() {
                return Err(QueryError::Execute(
                    "vector candidates name no cell present in the grid — \
                     malformed cell tags"
                        .into(),
                ));
            }
            let cutoff = grid_cell_cutoff(&ranked_for_beam, &cell_routing);
            // 1-bit prefilter for the exact fine scan: the grid's cutoff
            // picks are must-include so every cell the beam can select has
            // exact candidate scores (near-tie checks included). Filtered
            // queries use the same prefilter — same code, same budgets;
            // the allow-set only decides which rows may take shortlist
            // slots inside the probed runs.
            let admit_q = RabitqAdmitQuery::new(query.len(), rot_seed, query);
            let must_include: Vec<u32> = ranked_for_beam[..cutoff]
                .iter()
                .map(|(cell, _)| *cell)
                .collect();
            let (mut candidates, deferred) = score_fine_candidates(
                &superfiles,
                column,
                query,
                metric,
                Some((&admit_q, must_include.as_slice())),
                allow_ref,
                superseded,
            )?;
            if !deferred.is_empty() {
                self.rescore_deferred_cells(
                    &superfiles,
                    column,
                    query,
                    metric,
                    &mut candidates,
                    deferred,
                )
                .await?;
            }
            candidate_counts = candidates
                .iter()
                .map(|(si, cluster, _, _, count)| ((*si, *cluster), *count))
                .collect();
            let ranked = ranked_cells
                .as_ref()
                .expect("ranked cell ids exist with scored ranking");
            if hidden_vector_index {
                let fine_ranked = cells_ranked_by_fine_score(&candidates);
                // Default path: fine-first p=1, the same selection the user
                // (pre-drain) branch ships. Filtered search and explicit
                // caller nprobe keep the wider grid/fine union.
                let default_p1 = !filtered && options.nprobe.is_none() && cutoff == 1;
                let selected_cells_ordered: Vec<u32> = if default_p1 {
                    fine_first_cell_selection(
                        &fine_ranked,
                        ranked_for_beam.first().map(|(cell, _)| *cell),
                    )
                } else {
                    let grid_cells: Vec<u32> = ranked_for_beam[..cutoff]
                        .iter()
                        .map(|(cell, _)| *cell)
                        .collect();
                    let fine_cells: Vec<u32> = fine_ranked
                        .iter()
                        .take(cutoff.max(UNION_FINE_PICKS_MIN))
                        .map(|(cell, _)| *cell)
                        .collect();
                    union_cell_selection(&grid_cells, &fine_cells)
                };
                let selected_cells: HashSet<u32> = selected_cells_ordered.iter().copied().collect();
                // Wave-pooled depth is the p=1 read-volume model: keep runs
                // per drain wave so reads track wave count, not probed-cell
                // count — which also means widening the cell sweep alone
                // cannot deepen the read (measured flat 0.443 recall@100
                // from nprobe=4 through 256 on Cohere-1M). An explicit
                // caller `nprobe` — or the calibrated width law standing in
                // for one — is a request to actually read N cells, so gate
                // per (cell, fragment) exactly like the pre-drain user path:
                // depth follows width, read amplification is what was asked
                // for (explicitly by the caller, or measured as necessary
                // by the drain's calibration).
                let generation_of = if sweep_width.is_some() {
                    None
                } else {
                    Some(birth_versions.as_slice())
                };
                gated = gate_fine_candidates_by_fragment(
                    candidates,
                    &selected_cells,
                    &selected_cells_ordered,
                    cell_routing.fine_nprobe,
                    fine_nprobe_pct,
                    gated_target,
                    &candidate_counts,
                    &mut scored,
                    generation_of,
                );
            } else {
                // Fine-first p=1 over all scored fines. Explicit nprobe /
                // filtered search keep the grid/fine union.
                let fine_ranked = cells_ranked_by_fine_score(&candidates);
                let default_p1 = !filtered && options.nprobe.is_none() && cutoff == 1;
                let mut selected_cells: Vec<u32> = if default_p1 && !fine_ranked.is_empty() {
                    fine_first_cell_selection(&fine_ranked, ranked.first().copied())
                } else {
                    let grid_cells: Vec<u32> = ranked[..cutoff].to_vec();
                    let fine_cells: Vec<u32> = fine_ranked
                        .iter()
                        .take(cutoff)
                        .map(|(cell, _)| *cell)
                        .collect();
                    union_cell_selection(&grid_cells, &fine_cells)
                };
                let mut covered: u64 = selected_cells
                    .iter()
                    .map(|cell| postings_by_cell.get(cell).copied().unwrap_or(0))
                    .sum();
                for cell in ranked.iter().copied() {
                    if covered >= gated_target {
                        break;
                    }
                    if selected_cells.contains(&cell) {
                        continue;
                    }
                    covered += postings_by_cell.get(&cell).copied().unwrap_or(0);
                    selected_cells.push(cell);
                }
                let selected: HashSet<u32> = selected_cells.iter().copied().collect();
                gated = gate_fine_candidates_by_fragment(
                    candidates,
                    &selected,
                    &selected_cells,
                    USER_FINE_RUNS_PER_FRAGMENT,
                    0.0, // keep_pct: floor-only on the pre-drain user path
                    gated_target,
                    &candidate_counts,
                    &mut scored,
                    None,
                );
            }
        } else {
            // No grid, or untagged summaries: score every fine centroid
            // (legacy flat path, no prefilter). Stripped summaries defer to
            // the exact rescore — untagged legacy tables have no per-cell
            // gating to absorb estimate noise.
            let (mut candidates, deferred) = score_fine_candidates(
                &superfiles,
                column,
                query,
                metric,
                None,
                allow_ref,
                superseded,
            )?;
            if !deferred.is_empty() {
                self.rescore_deferred_cells(
                    &superfiles,
                    column,
                    query,
                    metric,
                    &mut candidates,
                    deferred,
                )
                .await?;
            }
            candidate_counts = candidates
                .iter()
                .map(|(si, cluster, _, _, count)| ((*si, *cluster), *count))
                .collect();
            scored = candidates
                .into_iter()
                .map(|(si, cluster, score, _, _)| (si, cluster, score))
                .collect();
        }

        // Every hidden-index search globally ranks fine centroids within the
        // selected cells. Filtering changes only which rows survive each
        // probe. User and undrained paths keep the closest
        // `USER_FINE_RUNS_PER_FRAGMENT` fine runs per immutable fragment
        // inside each selected coarse cell (posting-refilled toward the
        // gated target when the kept runs are too small to fill top-k).
        // Untagged legacy candidates still use the global fallback budget.
        let n_eligible = {
            let mut segs: Vec<usize> = scored
                .iter()
                .chain(gated.iter())
                .map(|&(si, _, _)| si)
                .collect();
            segs.sort_unstable();
            segs.dedup();
            segs.len()
        };
        // User/pre-drain keeps its existing nprobe × eligible-superfiles
        // budget. Hidden coverage and fine depth were already applied from the
        // persisted CellRoutingParams above.
        let scaled_budget = nprobe.saturating_mul(n_eligible.max(1)).max(nprobe);
        let default_budget = if hidden_vector_index {
            hidden_routing
                .expect("hidden manifest carries routing")
                .fine_nprobe
                .max(1)
        } else {
            scaled_budget
        };
        let budget = if hidden_vector_index {
            default_budget
        } else {
            config::global()
                .vector
                .inner_budget
                .map(|value| value.max(1))
                .unwrap_or(default_budget)
        };
        let cluster_count = |&(si, cluster, _): &(usize, u32, f32)| -> u64 {
            candidate_counts.get(&(si, cluster)).copied().unwrap_or(0)
        };
        let gated_postings: u64 = gated.iter().map(cluster_count).sum();
        if scored.len() > budget {
            // Break score ties by the centroid's `(superfile, cluster)` — a
            // unique total order — so the selected set is deterministic. With
            // an unstable partition on score alone, equidistant centroids
            // (common when vectors share a direction) land in the kept set
            // arbitrarily, so the fanned-out clusters, and thus the result
            // set, would vary run to run. Tie order among equal scores is
            // irrelevant to recall.
            scored.sort_unstable_by(|a, b| {
                a.2.partial_cmp(&b.2)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| (a.0, a.1).cmp(&(b.0, b.1)))
            });
            let mut kept = budget;
            let mut postings =
                gated_postings + scored[..kept].iter().map(cluster_count).sum::<u64>();
            while kept < scored.len() && postings < k as u64 {
                postings += cluster_count(&scored[kept]);
                kept += 1;
            }
            scored.truncate(kept);
        }
        let mut per_seg: HashMap<usize, Vec<u32>> = HashMap::new();
        for (si, c, _) in scored.into_iter().chain(gated) {
            per_seg.entry(si).or_default().push(c);
        }

        // Build fan-out units: selected superfiles probe their chosen
        // clusters; superfiles with centroids but no globally-selected
        // cluster are skipped (the cross-superfile win). For filtered
        // search each unit also carries its per-superfile allow-set (a
        // superfile reaching here is guaranteed present in `allow` —
        // empties were dropped above).
        //
        // Look the allow-set up only for a superfile that is actually
        // selected (scored a kept cluster) — a superfile that survived
        // vector pruning but whose predicate matched no row is absent from
        // `allow`, and must never be probed. Resolving the bitmap eagerly
        // for every entry would `expect`-panic on exactly those
        // filtered-out superfiles; gating it behind the selection guard
        // keeps the lookup on the path where presence is invariant.
        let mut units: Vec<(
            Arc<SuperfileEntry>,
            (usize, Vec<u32>, Option<Arc<RoaringBitmap>>),
        )> = Vec::new();
        for (si, entry) in superfiles.iter().enumerate() {
            let Some(ids) = per_seg.remove(&si) else {
                continue;
            };
            let bitmap = match allow.as_ref() {
                Some(m) => match m.get(&entry.uri) {
                    Some(bm) => Some(Arc::clone(bm)),
                    None => continue,
                },
                None => None,
            };
            units.push((Arc::clone(entry), (si, ids, bitmap)));
        }
        if units.is_empty() {
            if let Some(t0) = admit_t0 {
                io_counters::phase_record("vec.admit", t0.elapsed().as_micros() as u64);
            }
            return Ok(Vec::new());
        }
        if let Some(t0) = admit_t0 {
            io_counters::phase_record("vec.admit", t0.elapsed().as_micros() as u64);
        }

        // Fan out through the shared [`query::dispatch::fanout`] (also
        // used by FTS), but in waves capped by the configured reader
        // pool width. A cold vector kernel can hold large selected-cluster
        // `[codes][doc_ids]` prefix blocks while it builds its shortlist;
        // capping the number of concurrent superfiles keeps that transient
        // memory bounded by instance configuration instead of table size.
        // Skipped superfiles issue zero GETs.
        // Per-sweep rerank budget. The `k x rerank_mult` shortlist cap was
        // sized for the fine-first p=1 read: applied per probed cell it
        // exceeds any cell's row count, so at width every row of every
        // probed cell "survives" the 1-bit prune and the survivor-only
        // rerank fetch degenerates into fetching whole cells (measured:
        // ~6.5 MB x 54 cells per query at w=54). Divide the cap across the
        // sweep so the TOTAL survivor budget stays ~k x rerank_mult —
        // measured sufficient at that total: 0.9957 recall@100 on
        // Cohere-1M with ~25.6K survivors.
        // On the hidden width sweep the divide is superseded by GLOBAL
        // shortlist selection: warm cells scan at the undivided cap and the
        // supertable keeps the best `k x rerank_mult` estimates across the
        // whole sweep — exact, no even-split heuristic (measured 0.9937
        // undivided vs 0.9914 divided recall@100 on Cohere-1M). Cold cells
        // still rerank inside their own probe under the divided budget:
        // deferring them would hold their fetched blocks and their budget
        // reservation across the entire fan-out.
        let global_shortlist_width = if hidden_vector_index {
            sweep_width.filter(|w| *w > 1)
        } else {
            None
        };
        let mut cold_rerank_mult = 0;
        let options = match sweep_width {
            Some(w) if w > 1 => {
                let (_, rerank_mult) = options.resolve(filtered);
                let divided = rerank_mult
                    .saturating_mul(WIDTH_BUDGET_OVERSAMPLE)
                    .div_ceil(w)
                    .max(1);
                if global_shortlist_width.is_some() {
                    cold_rerank_mult = divided;
                    options
                } else {
                    options.with_rerank_mult(divided)
                }
            }
            _ => options,
        };
        let column_arc = Arc::new(column.to_owned());
        let query_arc = Arc::new(query.to_vec());
        let column_arc2 = Arc::clone(&column_arc);
        let query_arc2 = Arc::clone(&query_arc);
        let reader_pool = Arc::clone(&manifest.options.reader_pool);
        // Per-connection memory budget: gates each superfile's cold cluster-block fetch.
        let budget = Some(Arc::clone(&manifest.options.connection_memory_budget));
        let storage = manifest.options.storage.as_ref().map(Arc::clone);

        // `fanout_with`, not a plain post-rank filter: the body resolves each
        // superfile's tombstone bitmap *before* its kernel and pushes it down
        // as a deny set wherever IVF locals address Parquet rows (post-rank
        // filtering underflows the top-k). MultiCell user files are the
        // exception — their locals include boundary stubs, so deletes are
        // dropped after ranking by identity instead. The hidden path skips
        // sidecars entirely: its deletes ride inline in the hidden manifest
        // and are applied after remapping to user `_id`s.
        // Warm-cell estimate survivors from every scanned unit, pooled for
        // the global selection: (unit index, rot seed, candidates).
        let scan_pool: Arc<Mutex<Vec<(usize, u64, usize, Vec<ScanCandidate>)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let scan_pool_body = Arc::clone(&scan_pool);
        let body =
            move |reader: Arc<SuperfileReader>,
                  entry: Arc<SuperfileEntry>,
                  tombstone_cache: Option<Arc<SidecarCache>>,
                  now: Instant,
                  (si, ids, bitmap): (usize, Vec<u32>, Option<Arc<RoaringBitmap>>)| {
                let column = Arc::clone(&column_arc);
                let query = Arc::clone(&query_arc);
                let reader_pool = Arc::clone(&reader_pool);
                let budget = budget.clone();
                let storage = storage.clone();
                let scan_pool = Arc::clone(&scan_pool_body);
                async move {
                    // Unfiltered user path on row-addressable locals: resolve the
                    // bitmap once (warm after the orchestrator's prefetch) and
                    // push it down. Filtered search leaves it `None` — its
                    // allow-set already excludes tombstones.
                    let deny_pushdown = !hidden_vector_index
                        && bitmap.is_none()
                        && entry.vector_layout != VectorLayout::MultiCellIvf;
                    let deny = match tombstone_cache.as_ref() {
                        Some(cache) if deny_pushdown => {
                            dispatch::tombstone_deny_set(cache, entry.superfile_id, now)?
                        }
                        _ => None,
                    };
                    let pool = Some(Arc::clone(&reader_pool));
                    // Replicated hidden cells store boundary duplicates; fetch
                    // enough extra slots that the post-merge stable-id dedup
                    // still leaves k distinct rows.
                    let replica_overhead = reader
                        .vec()
                        .map(|v| (v.n_docs() as usize).saturating_sub(reader.n_docs() as usize))
                        .unwrap_or(0);
                    let k_fetch = k.saturating_add(replica_overhead);
                    let reader_for_ids = Arc::clone(&reader);
                    let hits = if global_shortlist_width.is_some() {
                        // Deferred-rerank scan: exact hits from cold cells now;
                        // warm-cell estimate survivors pooled for the global
                        // selection (phase C reranks the winners).
                        let scan = reader
                            .vector_scan_clusters_filtered(
                                &column,
                                &query,
                                k_fetch,
                                &ids,
                                options,
                                cold_rerank_mult,
                                bitmap,
                                deny,
                                pool,
                                budget,
                            )
                            .await
                            .map_err(vector_read_query_error)?;
                        if !scan.candidates.is_empty() {
                            scan_pool
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .push((si, scan.rot_seed, replica_overhead, scan.candidates));
                        }
                        scan.hits
                    } else {
                        reader
                            .vector_search_clusters_filtered(
                                &column, &query, k_fetch, &ids, options, bitmap, deny, pool, budget,
                            )
                            .await
                            .map_err(vector_read_query_error)?
                    };
                    let mut tagged = dispatch::tag_hits(&entry, hits);
                    // Prefer manifest span arithmetic; only touch `_id` pages /
                    // inline IVF regions when the layout is cell-packed or gapped.
                    io_counters::phase_timed_async("vec.stable_id", async {
                        dispatch::attach_stable_ids(&reader_for_ids, &entry, &mut tagged, false)
                            .await
                    })
                    .await?;
                    if !hidden_vector_index && !deny_pushdown {
                        // MultiCell user files (and any path that skipped the
                        // push-down): drop deleted rows by identity post-rank.
                        dispatch::apply_resolved_tombstone_filter(
                            &reader_for_ids,
                            storage.as_ref(),
                            tombstone_cache.as_ref(),
                            &entry,
                            &mut tagged,
                            now,
                        )
                        .await?;
                    }
                    Ok::<Vec<SuperfileHit>, QueryError>(tagged)
                }
            };
        // Filtered search holds a per-superfile RoaringBitmap while the
        // kernel builds its shortlist; wave-cap the fan-out by reader-pool
        // width so transient memory stays bounded. The unfiltered path
        // carries no bitmaps and fans out all units at once (matching
        // main's concurrency — every superfile GET overlaps on tokio).
        let fanout_t0 = io_counters::phase_start();
        let mut per_superfile = if allow.is_some() {
            let fanout_width = manifest.options.reader_pool.current_num_threads().max(1);
            let mut collected = Vec::new();
            while !units.is_empty() {
                let n = fanout_width.min(units.len());
                let wave: Vec<_> = units.drain(..n).collect();
                collected.extend(
                    dispatch::fanout_with(self, wave, !hidden_vector_index, false, body.clone())
                        .await?,
                );
            }
            collected
        } else {
            dispatch::fanout_with(self, units, !hidden_vector_index, false, body).await?
        };

        // Phase C of the deferred-rerank width sweep: select the best
        // `k x rerank_mult` estimates ACROSS every warm-scanned cell and
        // superfile, then rerank only those winners where they live. The
        // estimates are comparable across units — one rotation seed per
        // column, table-wide — asserted here at the only place different
        // units' estimates ever meet.
        if global_shortlist_width.is_some() {
            let pooled = {
                let mut guard = scan_pool.lock().unwrap_or_else(PoisonError::into_inner);
                mem::take(&mut *guard)
            };
            if !pooled.is_empty() {
                // Hard error, not debug_assert: this is the one site where
                // estimates from different superfiles are pooled and ranked
                // against each other. Backstopped by the open-time seed
                // check today, but if a future path ever admits a
                // differently-seeded unit, fail the query loudly instead of
                // silently ranking incomparable estimates.
                if pooled.windows(2).any(|w| w[0].1 != w[1].1) {
                    return Err(QueryError::Execute(
                        "pooled 1-bit estimates require one rotation seed per column".into(),
                    ));
                }
                let (_, rerank_mult) = options.resolve(filtered);
                // Mirror phase A/C's `k_fetch = k + replica_overhead` in the
                // global cut so boundary replicas (dormant today: overhead
                // is 0 with replication off) cannot take shortlist slots
                // from distinct rows before the stable-id dedup.
                let replica_overhead = pooled.iter().map(|(_, _, o, _)| *o).max().unwrap_or(0);
                let mut flat: Vec<(usize, ScanCandidate)> = pooled
                    .into_iter()
                    .flat_map(|(si, _, _, cands)| cands.into_iter().map(move |c| (si, c)))
                    .collect();
                flat = select_global_shortlist(
                    flat,
                    k.saturating_add(replica_overhead)
                        .saturating_mul(rerank_mult),
                );
                let mut winners_by_seg: HashMap<usize, Vec<ScanCandidate>> = HashMap::new();
                for (si, cand) in flat {
                    winners_by_seg.entry(si).or_default().push(cand);
                }
                let rerank_units: Vec<(Arc<SuperfileEntry>, Vec<ScanCandidate>)> = superfiles
                    .iter()
                    .enumerate()
                    .filter_map(|(si, entry)| {
                        winners_by_seg
                            .remove(&si)
                            .map(|sel| (Arc::clone(entry), sel))
                    })
                    .collect();
                let column = Arc::clone(&column_arc2);
                let query = Arc::clone(&query_arc2);
                let reader_pool = Arc::clone(&manifest.options.reader_pool);
                let body_c = move |reader: Arc<SuperfileReader>,
                                   entry: Arc<SuperfileEntry>,
                                   _tombstone_cache: Option<Arc<SidecarCache>>,
                                   _now: Instant,
                                   selected: Vec<ScanCandidate>| {
                    let column = Arc::clone(&column);
                    let query = Arc::clone(&query);
                    let reader_pool = Arc::clone(&reader_pool);
                    async move {
                        // Hidden-path invariants: no tombstone sidecars (the
                        // manifest's deletes apply after the stable-id
                        // remap upstream), replica slack mirrors phase A.
                        let replica_overhead = reader
                            .vec()
                            .map(|v| (v.n_docs() as usize).saturating_sub(reader.n_docs() as usize))
                            .unwrap_or(0);
                        let k_fetch = k.saturating_add(replica_overhead);
                        let reader_for_ids = Arc::clone(&reader);
                        let hits = reader
                            .vector_rerank_selected(
                                &column,
                                &query,
                                k_fetch,
                                selected,
                                Some(reader_pool),
                            )
                            .await
                            .map_err(vector_read_query_error)?;
                        let mut tagged = dispatch::tag_hits(&entry, hits);
                        io_counters::phase_timed_async("vec.stable_id", async {
                            dispatch::attach_stable_ids(&reader_for_ids, &entry, &mut tagged, false)
                                .await
                        })
                        .await?;
                        Ok::<Vec<SuperfileHit>, QueryError>(tagged)
                    }
                };
                per_superfile
                    .extend(dispatch::fanout_with(self, rerank_units, false, false, body_c).await?);
            }
        }
        if let Some(t0) = fanout_t0 {
            io_counters::phase_record("vec.fanout_wall", t0.elapsed().as_micros() as u64);
        }

        Ok(top_k_ascending(per_superfile, k))
    }

    /// Filtered single-column vector kNN: the k-nearest rows **among
    /// those matching a text predicate**, by pushdown.
    ///
    /// The predicate is resolved on the **user** table (FTS postings /
    /// blooms). When the hidden vector index is drained, kNN then ranks
    /// among matching rows on the **hidden** index — same post-drain path
    /// as unfiltered search and the bench. Pre-drain keeps the user-table
    /// fan-out. Superfiles whose predicate matches nothing are skipped.
    ///
    /// An empty `filter_query` (tokenizes to nothing) or a predicate
    /// that matches no row anywhere returns an empty `Vec`.
    ///
    /// `pub(crate)` async kernel — the public surface is the sync
    /// `vector_search` with a filter; this drives the cross-superfile fan-out.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, k = k, dim = query.len()))
    )]
    pub(crate) async fn vector_hits_filtered_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        filter: VectorFilter<'_>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        // Tokenize the predicate once with the index tokenizer (the same
        // tokenizer used at build time, so the terms match the postings AND
        // the manifest term blooms). No tokens (empty / punctuation-only) ⇒
        // nothing matches.
        let Some(tokenizer) = manifest.options.tokenizer.as_ref() else {
            return Ok(Vec::new());
        };
        let tokens: Vec<String> = tokenizer.tokenize(filter.query).collect();
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Manifest-only leaf survival: part-tier term bloom / range, then
        // per-superfile summaries — no superfile reads. Intersect with the
        // vector centroid prune so `token_match` opens only superfiles that
        // could match the predicate *and* might hold vector-near rows.
        let prune_leaves = [PruneLeaf::TermPresence {
            column: filter.column.to_owned(),
            terms: tokens.clone(),
            mode: filter.mode,
        }];
        let surviving: HashSet<u128> = select_superfiles(manifest, &prune_leaves)
            .await?
            .iter()
            .map(|e| e.superfile_id.as_u128())
            .collect();
        if surviving.is_empty() {
            return Ok(Vec::new());
        }
        let superfiles = self
            .vector_pruned_superfiles_intersect(manifest, &surviving)
            .await?;
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }

        // Resolve the exact per-superfile allow-set (`token_match` postings)
        // over the survivors; superfiles whose predicate matched no row are
        // dropped so they never fan out.
        let allow = self
            .candidate_bitmaps(&superfiles, filter.column, &tokens, filter.mode)
            .await?;
        if allow.is_empty() {
            return Ok(Vec::new());
        }

        self.route_filtered_vector_hits_async(superfiles, allow, column, query, k, options)
            .await
    }

    /// All loaded superfile entries intersected with a manifest-only
    /// survival set.
    async fn vector_pruned_superfiles_intersect(
        &self,
        manifest: &ManifestSnapshot,
        surviving: &HashSet<u128>,
    ) -> Result<Vec<Arc<SuperfileEntry>>, QueryError> {
        Ok(manifest
            .get_all_superfiles_loaded()
            .await
            .map_err(QueryError::ManifestLoad)?
            .into_iter()
            .filter(|e| surviving.contains(&e.superfile_id.as_u128()))
            .collect())
    }

    /// Resolve the text predicate (`filter_col` contains `tokens` under
    /// `mode`) to a per-superfile allow-set of matching `local_doc_id`s,
    /// over exactly the given vector-pruned `superfiles`.
    ///
    /// One `SuperfileReader::token_match` per superfile (postings-only,
    /// the leaf [`crate::supertable::query::candidate::CandidatePlan`]
    /// also uses), fanned out concurrently. Superfiles whose predicate
    /// matches no row are omitted from the returned map, so the caller
    /// skips them entirely.
    async fn candidate_bitmaps(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        filter_col: &str,
        tokens: &[String],
        mode: BoolMode,
    ) -> Result<HashMap<SuperfileUri, Arc<RoaringBitmap>>, QueryError> {
        let filter_col_arc = Arc::new(filter_col.to_owned());
        let tokens_arc: Arc<Vec<String>> = Arc::new(tokens.to_vec());
        self.fanout_candidate_bitmaps(superfiles, move |r, _entry| {
            let filter_col_arc = Arc::clone(&filter_col_arc);
            let tokens_arc = Arc::clone(&tokens_arc);
            async move {
                let refs: Vec<&str> = tokens_arc.iter().map(String::as_str).collect();
                r.token_match(&filter_col_arc, &refs, mode)
                    .await
                    .map_err(|e| QueryError::Parquet(e.to_string()))
                    .map(|docs| docs.into_iter().collect::<RoaringBitmap>())
            }
        })
        .await
    }

    /// Filtered vector kNN driven by a SQL `WHERE` [`CandidatePlan`] — the
    /// pushdown path for the `vector_search` table-valued function — rather
    /// than the single text-predicate shape of
    /// [`Self::vector_hits_filtered_async`].
    ///
    /// `plan` must be a **bounded** plan (not [`CandidatePlan::Unbounded`]):
    /// the caller routes `Unbounded` to the unfiltered
    /// [`Self::vector_search_async`], where DataFusion's `FilterExec`
    /// re-applies the predicate. For a bounded plan, the predicate is
    /// resolved on the user table and kNN runs on the hidden index when
    /// drained (same routing as [`Self::vector_hits_filtered_async`]).
    ///
    /// Manifest-only leaf survival runs before any superfile opens: bounded
    /// FTS leaves are lowered to term-bloom prunes and intersected with the
    /// vector centroid prune. The per-superfile allow-set (`plan.evaluate`)
    /// then runs only over that intersection.
    pub(crate) async fn vector_hits_filtered_by_plan(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        plan: &CandidatePlan,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        let superfiles = match plan.surviving_superfile_ids(manifest).await? {
            None => manifest
                .get_all_superfiles_loaded()
                .await
                .map_err(QueryError::ManifestLoad)?,
            Some(surviving) if surviving.is_empty() => return Ok(Vec::new()),
            Some(surviving) => {
                self.vector_pruned_superfiles_intersect(manifest, &surviving)
                    .await?
            }
        };
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }
        let allow = self.candidate_bitmaps_from_plan(&superfiles, plan).await?;
        if allow.is_empty() {
            return Ok(Vec::new());
        }
        self.route_filtered_vector_hits_async(superfiles, allow, column, query, k, options)
            .await
    }

    /// Convert user-table allow bitmaps (local doc ids) to stable `_id`s.
    async fn stable_ids_from_user_allow_async(
        &self,
        user_allow: &HashMap<SuperfileUri, Arc<RoaringBitmap>>,
    ) -> Result<Vec<i128>, QueryError> {
        let mut out: HashSet<i128> = HashSet::new();
        let manifest = self.manifest();
        let id_column = self.options().id_column.as_str();
        for (uri, bm) in user_allow {
            let entry = manifest
                .lookup_superfile_entry(*uri)
                .await
                .map_err(QueryError::ManifestLoad)?
                .ok_or_else(|| {
                    QueryError::Execute(format!("user superfile {uri:?} missing from manifest"))
                })?;
            if row_id_from_manifest_entry(&entry, 0).is_some() {
                for local in bm.iter() {
                    out.insert(entry.id_min + i128::from(local));
                }
                continue;
            }
            let locals: Vec<u32> = bm.iter().collect();
            let ids = read_ids_for_locals(manifest, &entry, &locals, id_column, false).await?;
            out.extend(ids);
        }
        Ok(out.into_iter().collect())
    }

    /// Filtered kNN: resolve the predicate on the user table, then search the
    /// hidden index when drained (same path as the bench). Pre-drain (no
    /// hidden superfiles) keeps the user-table fan-out.
    async fn route_filtered_vector_hits_async(
        &self,
        user_superfiles: Vec<Arc<SuperfileEntry>>,
        user_allow: HashMap<SuperfileUri, Arc<RoaringBitmap>>,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if user_allow.is_empty() {
            return Ok(Vec::new());
        }
        let drained = self
            .vector_index_table()
            .map(|hidden| hidden.pinned_reader().manifest().get_drained_ranges())
            .unwrap_or_default();
        let mut drained_allow = HashMap::new();
        let mut undrained_user = Vec::new();
        for entry in user_superfiles {
            if drained.contains(entry.birth_version) {
                if let Some(bitmap) = user_allow.get(&entry.uri) {
                    drained_allow.insert(entry.uri, Arc::clone(bitmap));
                }
            } else {
                undrained_user.push(entry);
            }
        }
        let user_hits = if undrained_user.is_empty() {
            Vec::new()
        } else {
            self.vector_fanout_over_superfiles(
                undrained_user,
                column,
                query,
                k,
                options,
                Some(user_allow.clone()),
            )
            .await?
        };
        let stable_ids = self
            .stable_ids_from_user_allow_async(&drained_allow)
            .await?;
        let hidden_hits = if stable_ids.is_empty() {
            Vec::new()
        } else {
            let prepared = self
                .prepare_vector_stable_allow_async(Arc::new(stable_ids))
                .await?;
            if !prepared.use_hidden_index {
                return Err(QueryError::Execute(
                    "drained filtered-vector ids resolved to a user allow-set instead of the \
                     hidden index"
                        .into(),
                ));
            }
            self.vector_hits_prepared_global_allow_async(column, query, k, options, &prepared)
                .await?
        };
        Ok(top_k_ascending(vec![hidden_hits, user_hits], k))
    }

    /// Test/bench-only bitmap-filtered vector kNN. `allow_global` uses the
    /// same global row numbering as the bench corpus and is translated to
    /// per-superfile `local_doc_id` bitmaps before entering the normal filtered
    /// fan-out. This lets the supertable bench mirror the superfile filtered
    /// recall probe without requiring an FTS predicate on the vector-only
    /// fixture.
    #[cfg(feature = "test-helpers")]
    pub async fn vector_hits_global_allow_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        allow_global: Arc<RoaringBitmap>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let prepared = self.prepare_vector_global_allow_async(allow_global).await?;
        self.vector_hits_prepared_global_allow_async(column, query, k, options, &prepared)
            .await
    }

    /// Build the per-superfile allow-set once from corpus-global ids.
    ///
    /// User-table path maps by contiguous ingest order. Post-drain path maps
    /// against hidden-cell stable ids and returns a hidden-index allow-set.
    #[cfg(feature = "test-helpers")]
    pub async fn prepare_vector_global_allow_async(
        &self,
        allow_global: Arc<RoaringBitmap>,
    ) -> Result<PreparedGlobalAllow, QueryError> {
        if allow_global.is_empty() {
            return Ok(PreparedGlobalAllow {
                use_hidden_index: self.vector_index_table().is_some(),
                allow_by_uri: HashMap::new(),
            });
        }
        if let Some(vit) = self.vector_index_table() {
            let hidden_reader = vit.pinned_reader();
            let hidden_manifest = Arc::clone(hidden_reader.manifest());
            let drained = hidden_manifest.get_drained_ranges();
            let superfiles = hidden_manifest
                .get_all_superfiles_loaded()
                .await
                .map_err(QueryError::ManifestLoad)?;
            if !superfiles.is_empty() {
                let allow_for_cell = Arc::clone(&allow_global);
                let manifest_for_ids = Arc::clone(&hidden_manifest);
                let allow_by_uri = hidden_reader
                    .fanout_candidate_bitmaps(&superfiles, move |r, entry| {
                        let allow_for_cell = Arc::clone(&allow_for_cell);
                        let manifest_for_ids = Arc::clone(&manifest_for_ids);
                        async move {
                            let stable_ids =
                                stable_ids_by_local_for_routing(&manifest_for_ids, &entry, &r)
                                    .await?;
                            let mut local = RoaringBitmap::new();
                            for (local_doc_id, stable_id) in stable_ids.into_iter().enumerate() {
                                if let Ok(global_id) = u32::try_from(stable_id)
                                    && allow_for_cell.contains(global_id)
                                {
                                    local.insert(local_doc_id as u32);
                                }
                            }
                            Ok(local)
                        }
                    })
                    .await?;
                if allow_by_uri.is_empty() {
                    return Err(QueryError::Execute(
                        "global allow ids for drained filtered-vector rows did not map to any \
                         hidden superfile"
                            .into(),
                    ));
                }
                return Ok(PreparedGlobalAllow {
                    use_hidden_index: true,
                    allow_by_uri,
                });
            }
            if !drained.is_empty() {
                return Err(QueryError::Execute(
                    "hidden vector manifest has drained ranges but no hidden superfiles".into(),
                ));
            }
        }

        let manifest = self.manifest();
        let superfiles = manifest
            .get_all_superfiles_loaded()
            .await
            .map_err(QueryError::ManifestLoad)?;
        let mut allow_by_uri: HashMap<SuperfileUri, RoaringBitmap> = HashMap::new();
        let mut allowed = allow_global.iter().peekable();
        let mut base = 0u64;
        for entry in &superfiles {
            let end = base.saturating_add(entry.n_docs);
            while allowed.peek().is_some_and(|&id| (id as u64) < base) {
                allowed.next();
            }
            let mut local = RoaringBitmap::new();
            while let Some(id) = allowed.peek().copied() {
                let id = id as u64;
                if id >= end {
                    break;
                }
                local.insert((id - base) as u32);
                allowed.next();
            }
            if !local.is_empty() {
                allow_by_uri.insert(entry.uri, local);
            }
            base = end;
        }
        Ok(PreparedGlobalAllow {
            use_hidden_index: false,
            allow_by_uri: allow_by_uri
                .into_iter()
                .map(|(uri, bm)| (uri, Arc::new(bm)))
                .collect(),
        })
    }

    /// Build a per-superfile allow-set from stable `_id` values.
    ///
    /// Post-drain: every supplied id is expected to describe a drained row and
    /// must map against hidden-cell stable ids; an empty mapping is an
    /// invariant error. Pre-drain (empty hidden membership and drained range)
    /// maps against the user table.
    #[cfg(feature = "test-helpers")]
    pub async fn prepare_vector_stable_allow_async(
        &self,
        allow_stable_ids: Arc<Vec<i128>>,
    ) -> Result<PreparedGlobalAllow, QueryError> {
        self.prepare_vector_stable_allow_inner(allow_stable_ids)
            .await
    }

    #[cfg(not(feature = "test-helpers"))]
    pub(crate) async fn prepare_vector_stable_allow_async(
        &self,
        allow_stable_ids: Arc<Vec<i128>>,
    ) -> Result<PreparedGlobalAllow, QueryError> {
        self.prepare_vector_stable_allow_inner(allow_stable_ids)
            .await
    }

    async fn prepare_vector_stable_allow_inner(
        &self,
        allow_stable_ids: Arc<Vec<i128>>,
    ) -> Result<PreparedGlobalAllow, QueryError> {
        if allow_stable_ids.is_empty() {
            return Ok(PreparedGlobalAllow {
                use_hidden_index: false,
                allow_by_uri: HashMap::new(),
            });
        }
        let allow_set: Arc<HashSet<i128>> =
            Arc::new(allow_stable_ids.iter().copied().collect::<HashSet<i128>>());
        if let Some(vit) = self.vector_index_table() {
            let hidden_reader = vit.pinned_reader();
            let hidden_manifest = Arc::clone(hidden_reader.manifest());
            let drained = hidden_manifest.get_drained_ranges();
            let superfiles = hidden_manifest
                .get_all_superfiles_loaded()
                .await
                .map_err(QueryError::ManifestLoad)?;
            if !superfiles.is_empty() {
                let allow_for_cell = Arc::clone(&allow_set);
                let manifest_for_ids = Arc::clone(&hidden_manifest);
                let allow_by_uri = hidden_reader
                    .fanout_candidate_bitmaps(&superfiles, move |r, entry| {
                        let allow_for_cell = Arc::clone(&allow_for_cell);
                        let manifest_for_ids = Arc::clone(&manifest_for_ids);
                        async move {
                            let stable_ids =
                                stable_ids_by_local_for_routing(&manifest_for_ids, &entry, &r)
                                    .await?;
                            let mut local = RoaringBitmap::new();
                            for (local_doc_id, stable_id) in stable_ids.into_iter().enumerate() {
                                if allow_for_cell.contains(&stable_id) {
                                    local.insert(local_doc_id as u32);
                                }
                            }
                            Ok(local)
                        }
                    })
                    .await?;
                if allow_by_uri.is_empty() {
                    return Err(QueryError::Execute(
                        "stable ids for drained filtered-vector rows did not map to any hidden \
                         superfile"
                            .into(),
                    ));
                }
                return Ok(PreparedGlobalAllow {
                    use_hidden_index: true,
                    allow_by_uri,
                });
            }
            if !drained.is_empty() {
                return Err(QueryError::Execute(
                    "hidden vector manifest has drained ranges but no hidden superfiles".into(),
                ));
            }
        }
        let manifest = self.manifest();
        let superfiles = manifest
            .get_all_superfiles_loaded()
            .await
            .map_err(QueryError::ManifestLoad)?;
        if superfiles.is_empty() {
            return Ok(PreparedGlobalAllow {
                use_hidden_index: false,
                allow_by_uri: HashMap::new(),
            });
        }
        let allow_for_user = Arc::clone(&allow_set);
        let manifest_for_ids = Arc::clone(manifest);
        let allow_by_uri = self
            .fanout_candidate_bitmaps(&superfiles, move |r, entry| {
                let allow_for_user = Arc::clone(&allow_for_user);
                let manifest_for_ids = Arc::clone(&manifest_for_ids);
                async move {
                    let stable_ids =
                        stable_ids_by_local_for_routing(&manifest_for_ids, &entry, &r).await?;
                    let mut local = RoaringBitmap::new();
                    for (local_doc_id, stable_id) in stable_ids.into_iter().enumerate() {
                        if allow_for_user.contains(&stable_id) {
                            local.insert(local_doc_id as u32);
                        }
                    }
                    Ok(local)
                }
            })
            .await?;
        Ok(PreparedGlobalAllow {
            use_hidden_index: false,
            allow_by_uri,
        })
    }

    /// Run filtered vector fan-out from a precomputed allow-set (user or
    /// hidden, as selected by [`PreparedGlobalAllow::use_hidden_index`]).
    #[cfg(feature = "test-helpers")]
    pub async fn vector_hits_prepared_global_allow_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        prepared: &PreparedGlobalAllow,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.vector_hits_prepared_global_allow_inner(column, query, k, options, prepared)
            .await
    }

    #[cfg(not(feature = "test-helpers"))]
    pub(crate) async fn vector_hits_prepared_global_allow_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        prepared: &PreparedGlobalAllow,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.vector_hits_prepared_global_allow_inner(column, query, k, options, prepared)
            .await
    }

    async fn vector_hits_prepared_global_allow_inner(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        prepared: &PreparedGlobalAllow,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 || prepared.allow_by_uri.is_empty() {
            return Ok(Vec::new());
        }
        if prepared.use_hidden_index {
            let vit = self.vector_index_table().ok_or_else(|| {
                QueryError::Execute("prepared hidden allow-set but no hidden index table".into())
            })?;
            let hidden_reader = vit.pinned_reader();
            let superfiles = hidden_reader
                .manifest()
                .get_all_superfiles_loaded()
                .await
                .map_err(QueryError::ManifestLoad)?;
            if superfiles.is_empty() {
                return Ok(Vec::new());
            }
            return hidden_reader
                .vector_fanout_over_superfiles(
                    superfiles,
                    column,
                    query,
                    k,
                    options,
                    Some(prepared.allow_by_uri.clone()),
                )
                .await;
        }
        let superfiles = self
            .manifest()
            .get_all_superfiles_loaded()
            .await
            .map_err(QueryError::ManifestLoad)?;
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }
        self.vector_fanout_over_superfiles(
            superfiles,
            column,
            query,
            k,
            options,
            Some(prepared.allow_by_uri.clone()),
        )
        .await
    }

    /// Resolve a [`CandidatePlan`] to a per-superfile allow-set of matching
    /// `local_doc_id`s over the given vector-pruned `superfiles` — the
    /// boolean-plan analog of [`Self::candidate_bitmaps`] (which evaluates a
    /// single term match). `token_match` leaves are combined by `AND`/`OR`;
    /// superfiles whose plan matches no row are omitted so the caller skips
    /// them. Tombstoned rows are dropped by the shared `fanout` (a deleted
    /// row must never be a kNN candidate).
    ///
    /// The caller passes only a bounded plan, so `evaluate` returns
    /// `Some(bitmap)` per superfile; a defensive `None` (unbounded) is
    /// treated as the empty set, skipping that superfile.
    async fn candidate_bitmaps_from_plan(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        plan: &CandidatePlan,
    ) -> Result<HashMap<SuperfileUri, Arc<RoaringBitmap>>, QueryError> {
        let plan_arc = Arc::new(plan.clone());
        self.fanout_candidate_bitmaps(superfiles, move |r, _entry| {
            let plan = Arc::clone(&plan_arc);
            async move {
                plan.evaluate(r.as_ref())
                    .await
                    .map_err(|e| QueryError::Parquet(e.to_string()))?
                    .ok_or_else(|| {
                        QueryError::Execute(
                            "bounded CandidatePlan evaluated to Unbounded — planner bug".into(),
                        )
                    })
            }
        })
        .await
    }

    /// Fan out over `superfiles`, resolve matching `local_doc_id`s per
    /// superfile via `doc_ids`, subtract tombstones, and drop empties.
    async fn fanout_candidate_bitmaps<F, Fut>(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        doc_ids: F,
    ) -> Result<HashMap<SuperfileUri, Arc<RoaringBitmap>>, QueryError>
    where
        F: Fn(Arc<SuperfileReader>, Arc<SuperfileEntry>) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = Result<RoaringBitmap, QueryError>> + Send,
    {
        let units: Vec<(Arc<SuperfileEntry>, ())> =
            superfiles.iter().map(|e| (Arc::clone(e), ())).collect();
        let body = move |r: Arc<SuperfileReader>,
                         entry: Arc<SuperfileEntry>,
                         tombstone_cache: Option<Arc<SidecarCache>>,
                         now: Instant,
                         _: ()| {
            let doc_ids = doc_ids.clone();
            async move {
                let mut bm = doc_ids(r, Arc::clone(&entry)).await?;
                subtract_tombstones(&mut bm, &entry, tombstone_cache.as_deref(), now)?;
                Ok((entry.uri, bm))
            }
        };
        let pairs: Vec<(SuperfileUri, RoaringBitmap)> =
            dispatch::fanout_with(self, units, true, false, body).await?;
        Ok(pairs
            .into_iter()
            .filter(|(_, bm)| !bm.is_empty())
            .map(|(uri, bm)| (uri, Arc::new(bm)))
            .collect())
    }
    pub(crate) async fn vector_search_user_table_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        let superfiles = manifest
            .get_all_superfiles_loaded()
            .await
            .map_err(QueryError::ManifestLoad)?;
        if superfiles.is_empty() {
            return Ok(Vec::new());
        }
        self.fanout_vector_clusters(&superfiles, column, query, k, options)
            .await
    }

    /// Global fine-centroid ranking over hidden coverage plus undrained user
    /// deltas, merged by stable identity.
    pub(crate) async fn vector_search_global_index_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let Some(vit) = self.vector_index_table() else {
            // A configured+materialized hidden index that failed to open is
            // present-but-broken: fail loud rather than silently brute-scanning
            // the user table (which would hide corruption and, if drained rows
            // were ever reclaimed, return incomplete results). A genuinely absent
            // index (never configured, or pre-first-drain) falls back.
            if let Some(reason) = self.hidden_index_open_error() {
                return Err(QueryError::Execute(format!(
                    "hidden vector index present but failed to open: {reason}"
                )));
            }
            return self
                .vector_search_user_table_async(column, query, k, options)
                .await;
        };

        // Wave 1: search the pinned hidden slow state while refreshing only the
        // fast delete state and loading any user parts known to be newer than
        // this exact hidden residency watermark.
        let hidden_reader = vit.pinned_reader();
        let hidden_manifest = Arc::clone(hidden_reader.manifest());
        let drained = hidden_manifest.get_drained_ranges();
        let hidden_entries = hidden_manifest
            .get_all_superfiles_loaded()
            .await
            .map_err(QueryError::ManifestLoad)?;
        let hidden_search = async {
            if hidden_entries.is_empty() {
                Ok(Vec::new())
            } else {
                hidden_reader
                    .fanout_vector_clusters(&hidden_entries, column, query, k, options)
                    .await
            }
        };
        let fast_state = async {
            vit.ensure_fresh_async().await;
            vit.pinned_reader()
                .hidden_deleted_ids()
                .map_err(|error| QueryError::Execute(error.to_string()))
        };
        let user_parts = self.manifest().get_undrained_superfiles_loaded(&drained);
        let (hidden_hits, deleted, user_entries) = join!(hidden_search, fast_state, user_parts);
        let mut hidden_hits = hidden_hits?;
        let deleted = deleted?;
        let user_entries = user_entries.map_err(QueryError::ManifestLoad)?;

        // Wave 2 only when the resident user list identifies files newer than
        // the hidden watermark.
        let mut user_hits = if user_entries.is_empty() {
            Vec::new()
        } else {
            self.fanout_vector_clusters(&user_entries, column, query, k, options)
                .await?
        };
        let refill_cap = k.saturating_add(deleted.len()).max(k);
        let mut requested = k;
        loop {
            let mut combined = top_k_ascending(vec![hidden_hits, user_hits], requested);
            if let Some(hit) = combined.iter().find(|hit| hit.stable_id.is_none()) {
                return Err(QueryError::Execute(format!(
                    "hit {:?}/{} missing stable _id before combined delete filtering",
                    hit.superfile, hit.local_doc_id
                )));
            }
            let live = combined
                .iter()
                .filter(|hit| {
                    hit.stable_id
                        .is_some_and(|id| deleted.binary_search(&id).is_err())
                })
                .count();
            // Deletes shrink the candidate pool in two places: a deleted id
            // can still occupy a combined slot (identity-filtered right
            // here), or the per-superfile tombstone filter inside the
            // fan-out already dropped it and the slot is simply missing.
            // Either way, while the live prefix is short and deletes exist,
            // grow the request toward the cap instead of returning an
            // underfull top-k while more live rows exist. When the table has
            // no deletes this stays the zero-extra-work fast path.
            let deleted_occupies_top_k = live < k && !deleted.is_empty();
            if !deleted_occupies_top_k || requested >= refill_cap {
                combined.retain(|hit| {
                    hit.stable_id
                        .is_some_and(|id| deleted.binary_search(&id).is_err())
                });
                combined.truncate(k);
                return Ok(combined);
            }

            let next = requested
                .saturating_mul(DELETE_REFILL_GROWTH_FACTOR)
                .min(refill_cap);
            if next == requested {
                combined.retain(|hit| {
                    hit.stable_id
                        .is_some_and(|id| deleted.binary_search(&id).is_err())
                });
                combined.truncate(k);
                return Ok(combined);
            }
            requested = next;
            let hidden_retry = async {
                if hidden_entries.is_empty() {
                    Ok(Vec::new())
                } else {
                    hidden_reader
                        .fanout_vector_clusters(&hidden_entries, column, query, requested, options)
                        .await
                }
            };
            let user_retry = async {
                if user_entries.is_empty() {
                    Ok(Vec::new())
                } else {
                    self.fanout_vector_clusters(&user_entries, column, query, requested, options)
                        .await
                }
            };
            let (next_hidden, next_user) = join!(hidden_retry, user_retry);
            hidden_hits = next_hidden?;
            user_hits = next_user?;
        }
    }

    /// Default async vector kernel — routes through the global hidden index
    /// when present (`vector_hits`, bare `vector_search` TVF).
    pub(crate) async fn vector_search_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.vector_search_global_index_async(column, query, k, options)
            .await
    }
}

impl SupertableReader {
    pub fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        filter: Option<VectorFilter<'_>>,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, QueryError> {
        // Mark a foreground query in flight so background cache-fills yield
        // S3 bandwidth to it; released when this query returns.
        let _fg = crate::supertable::reader_cache::disk::ForegroundQueryGuard::enter();
        self.block_on(async {
            let hits = match filter {
                None => {
                    self.vector_search_global_index_async(column, query, k, options)
                        .await?
                }
                Some(f) => {
                    self.vector_hits_filtered_async(column, query, k, options, f)
                        .await?
                }
            };
            let id_column = self.options().id_column.as_str();
            if projection_is_id_score_only(projection, id_column) {
                let batch = hits_id_score_batch(self, &hits)?;
                return Ok(vec![batch]);
            }
            let hits = user_placement_for_scalar_resolve(self, &hits).await?;
            let batch = resolve_hits_named(self, &hits, projection, "vector_search")
                .await
                .map_err(|e| QueryError::Execute(e.to_string()))?;
            Ok(vec![batch])
        })
    }

    pub fn vector_hits(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        filter: Option<VectorFilter<'_>>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        // Mark a foreground query in flight so background cache-fills yield
        // S3 bandwidth to it; released when this query returns.
        let _fg = crate::supertable::reader_cache::disk::ForegroundQueryGuard::enter();
        match filter {
            None => self.block_on(self.vector_search_global_index_async(column, query, k, options)),
            Some(f) => self.block_on(self.vector_hits_filtered_async(column, query, k, options, f)),
        }
    }
}

fn subtract_tombstones(
    bm: &mut RoaringBitmap,
    entry: &SuperfileEntry,
    tombstone_cache: Option<&SidecarCache>,
    now: Instant,
) -> Result<(), QueryError> {
    if let Some(cache) = tombstone_cache {
        let deleted = cache
            .bitmap_for(entry.superfile_id, now)
            .map_err(|e| QueryError::Store(format!("tombstone cache: {e}")))?;
        if !deleted.is_empty() {
            *bm -= &*deleted;
        }
    }
    Ok(())
}

/// Merge per-superfile hits and return the top-k by *ascending*
/// distance (smallest = closest). Uses a max-heap of size k so
/// we never sort more than k elements — O(S·k·log k) instead of
/// O(S·k·log(S·k)) for the full-sort approach.
/// One shared width override on top of the per-branch base routing:
/// explicit caller `nprobe` pins the cell sweep width on every branch —
/// an override is honored, never discarded — and with no override a
/// drain-calibrated law width pins the same way. Returns the engaged
/// `sweep_width` (drives the per-sweep rerank-budget divide and the
/// per-fragment fine gating downstream), `None` when nothing pinned.
///
/// Depth rides the width on the UNFILTERED path, for the pin exactly as
/// for the law: the law's coverage numbers assume a probed cell is read
/// in full (measured: half-depth caps recall at 0.964 where full depth
/// reaches 0.995), and a caller-widened sweep at the persisted
/// fine-first depth contributes only each cell's first runs — measured
/// ~0.83 recall@100 on Cohere-1M REGARDLESS of width, a dial that
/// widened without deepening. The per-fragment gate still clamps to
/// each fragment's real run count. FILTERED queries keep their own fine
/// floors (already applied to `routing` by the base branch) and never
/// engage `sweep_width`: a sparse allow-set's shortlist divided across
/// cells starves. `populated_cells` clamps the width the budget divide
/// splits by — never more than the cells that actually carry postings.
fn apply_width_pin(
    routing: &mut CellRoutingParams,
    caller_nprobe: Option<usize>,
    law_width: Option<usize>,
    filtered: bool,
    populated_cells: usize,
) -> Option<usize> {
    if let Some(nprobe) = caller_nprobe {
        routing.nprobe_min = nprobe;
        routing.nprobe_max = nprobe;
        if filtered {
            return None;
        }
        routing.fine_nprobe = usize::MAX;
        Some(nprobe.clamp(1, populated_cells))
    } else if let Some(width) = law_width {
        routing.nprobe_min = width;
        routing.nprobe_max = width;
        routing.fine_nprobe = usize::MAX;
        Some(width.min(populated_cells))
    } else {
        None
    }
}

/// Deterministic global shortlist selection for deferred-rerank width
/// sweeps: keep the `limit` best 1-bit estimates pooled across every
/// scanned unit. Higher estimate = better; ties break on
/// (unit, cell, pos, did), a total order, so the KEPT SET is
/// insertion-order independent across concurrent scans. O(n) partition,
/// not a sort — the winners need no internal order (phase C regroups by
/// unit and the rerank is exact).
fn select_global_shortlist(
    mut pooled: Vec<(usize, ScanCandidate)>,
    limit: usize,
) -> Vec<(usize, ScanCandidate)> {
    let cmp = |a: &(usize, ScanCandidate), b: &(usize, ScanCandidate)| {
        b.1.estimate.total_cmp(&a.1.estimate).then_with(|| {
            (a.0, a.1.cell_idx, a.1.pos, a.1.did).cmp(&(b.0, b.1.cell_idx, b.1.pos, b.1.did))
        })
    };
    if pooled.len() > limit {
        pooled.select_nth_unstable_by(limit, cmp);
        pooled.truncate(limit);
    }
    pooled
}

fn top_k_ascending(per_superfile: Vec<Vec<SuperfileHit>>, k: usize) -> Vec<SuperfileHit> {
    // Total order over hits: distance ascending, then the unique
    // `(superfile, local_doc_id)` key. The tie-break makes the kept set
    // deterministic when scores are equal (common when many rows share a
    // direction) — otherwise the k-boundary among ties would be resolved by
    // heap feed order (HashMap iteration + fan-out completion), which varies
    // run to run. Tie order never affects recall: equal-distance rows are
    // interchangeable.
    fn hit_order(a: &SuperfileHit, b: &SuperfileHit) -> Ordering {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.superfile.cmp(&b.superfile))
            .then_with(|| a.local_doc_id.cmp(&b.local_doc_id))
    }

    #[derive(PartialEq)]
    struct MaxByScore(SuperfileHit);
    impl Eq for MaxByScore {}
    impl PartialOrd for MaxByScore {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for MaxByScore {
        fn cmp(&self, other: &Self) -> Ordering {
            hit_order(&self.0, &other.0)
        }
    }

    // Boundary replicas are stored in different hidden cell superfiles but carry
    // the same user `_id` in `stable_id`. Collapse them before the top-k heap so
    // one logical row cannot occupy multiple result slots. On a score tie the
    // smaller `(superfile, local_doc_id)` wins, so the survivor is deterministic.
    // User-table hits without `stable_id` pass through unchanged.
    let mut best_by_id: HashMap<i128, SuperfileHit> = HashMap::new();
    let mut passthrough = Vec::new();
    for hit in per_superfile.into_iter().flatten() {
        if let Some(id) = hit.stable_id {
            best_by_id
                .entry(id)
                .and_modify(|existing| {
                    if hit_order(&hit, existing) == Ordering::Less {
                        *existing = hit;
                    }
                })
                .or_insert(hit);
        } else {
            passthrough.push(hit);
        }
    }

    // Max-heap keyed by `hit_order`: the peek is the current worst (largest
    // distance, largest tie-break key). Keep the k smallest under that total
    // order, evicting the worst when a strictly-better candidate arrives — so
    // the kept set is independent of insertion order.
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for hit in best_by_id.into_values().chain(passthrough) {
        if heap.len() < k {
            heap.push(MaxByScore(hit));
        } else if let Some(worst) = heap.peek()
            && hit_order(&hit, &worst.0) == Ordering::Less
        {
            heap.pop();
            heap.push(MaxByScore(hit));
        }
    }
    let mut result: Vec<SuperfileHit> = heap.into_iter().map(|m| m.0).collect();
    result.sort_unstable_by(hit_order);
    result
}

impl Supertable {
    /// Single-column vector kNN search over the current snapshot,
    /// returning Arrow rows nearest-first (distance score, smaller is
    /// nearer).
    ///
    /// `score` is a distance (`0.0` = perfect match) — the opposite
    /// direction from [`Supertable::bm25_search`]'s similarity. Fuse the
    /// two with [`Supertable::hybrid_search`], not by raw score.
    ///
    /// Pins a fresh reader (applying the read-consistency policy), runs
    /// the IVF fan-out, and resolves the top-`k` nearest hits to Arrow
    /// rows.
    ///
    /// `projection` selects output columns by name (any of `_id`, the
    /// visible scalar columns, or the trailing `score`); `None` returns
    /// the engine-native result — `_id` + `score` only. Only the
    /// projected scalar columns are decoded — kNN is usually a
    /// retrieval step, so materializing row data is an explicit opt-in
    /// by column name for the hits you keep.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_array::{FixedSizeListArray, Float32Array, RecordBatch};
    /// # use infino::arrow_array::types::Float32Type;
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, IndexSpec, Metric, VectorSearchOptions};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new(
    /// #     "emb",
    /// #     DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 16),
    /// #     false,
    /// # )]));
    /// # let vecs = db.create_table("vecs", schema.clone(), IndexSpec::new().vector("emb", 16, 1, Metric::Cosine))?;
    /// # let mut data = vec![0.0f32; 16]; data[0] = 1.0;
    /// # let col = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vec![Some(data.iter().copied().map(Some).collect::<Vec<_>>())], 16);
    /// # vecs.append(&RecordBatch::try_new(schema, vec![Arc::new(col)])?)?;
    /// # let mut query = vec![0.0f32; 16]; query[0] = 1.0;
    /// // Bare call → `_id` + `score`, no scalar decode:
    /// let hits = vecs.vector_search("emb", &query, 10, VectorSearchOptions::new(), None, None)?;
    /// assert_eq!(hits[0].num_columns(), 2);
    /// // Explicit projection names the same columns (scalar columns,
    /// // when present, materialize row data):
    /// let rows = vecs.vector_search(
    ///     "emb",
    ///     &query,
    ///     10,
    ///     VectorSearchOptions::new(),
    ///     None,
    ///     Some(&["_id", "score"]),
    /// )?;
    /// assert!(rows.iter().map(|b| b.num_rows()).sum::<usize>() >= 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, k = k, dim = query.len()))
    )]
    pub fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        options: VectorSearchOptions,
        filter: Option<VectorFilter<'_>>,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, crate::InfinoError> {
        self.reader()?
            .vector_search(column, query, k, options, filter, projection)
            .map_err(crate::InfinoError::from)
            .map_err(|e| e.with_context("vector_search", None))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet, HashMap, HashSet},
        sync::Arc,
    };

    use arrow::array::Array;
    use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use super::{
        RABITQ_ADMIT_CELL_SHORTLIST_MIN, SCORE_COLUMN, ScanCandidate, VectorFilter,
        VectorSearchOptions, admit_shortlist_window, apply_width_pin, cells_ranked_by_fine_score,
        gate_fine_candidates_by_fragment, hidden_hits_user_ids, is_hidden_vector_manifest,
        postings_by_cell_from_summaries, projection_is_id_score_only, score_fine_candidates,
        select_global_shortlist, union_cell_selection, vector_read_query_error,
    };
    use crate::{
        InfinoError,
        superfile::{
            SuperfileReader,
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig},
            error::{ReadError, VectorError},
            vector::{distance::Metric, rerank_codec::RerankCodec},
        },
        supertable::{
            Supertable, SupertableOptions,
            error::QueryError,
            manifest::{
                ClusterCentroids,
                list::{CellRoutingParams, PartitionStrategy},
            },
        },
        test_helpers::default_tokenizer as tok,
    };

    /// Drive an async future to completion on a throwaway current-thread
    /// runtime. Used only for the single-superfile `SuperfileReader`
    /// oracle, whose search surface is async-only; the supertable
    /// reader's own search methods are sync and need no runtime here.
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(fut)
    }

    /// Fine ranking takes each cell's best (minimum) candidate score,
    /// sorts ascending with lower-id tie-break, and ignores untagged
    /// (legacy, `None`-cell) candidates.
    /// Exactly the id + score columns (either order) or a bare `SELECT *`
    /// (None) takes the id/score fast path; anything else does not.
    #[test]
    fn projection_is_id_score_only_matches_id_score_combinations() {
        let id = "doc_id";
        assert!(projection_is_id_score_only(None, id));
        assert!(projection_is_id_score_only(Some(&[id, SCORE_COLUMN]), id));
        assert!(projection_is_id_score_only(Some(&[SCORE_COLUMN, id]), id));
        assert!(!projection_is_id_score_only(Some(&[id]), id));
        assert!(!projection_is_id_score_only(
            Some(&["other", SCORE_COLUMN]),
            id
        ));
    }

    /// The admit window scales with the ranked cell population (20%
    /// slice) and never narrows below the validated floor: small tables
    /// degenerate to exact-everything, the 256-cell shape widens just
    /// past its measured 48, and larger grids grow proportionally.
    #[test]
    fn admit_shortlist_window_scales_with_cell_population() {
        assert_eq!(admit_shortlist_window(0), RABITQ_ADMIT_CELL_SHORTLIST_MIN);
        assert_eq!(admit_shortlist_window(64), RABITQ_ADMIT_CELL_SHORTLIST_MIN);
        assert_eq!(admit_shortlist_window(240), RABITQ_ADMIT_CELL_SHORTLIST_MIN);
        assert_eq!(admit_shortlist_window(256), 52);
        assert_eq!(admit_shortlist_window(512), 103);
        assert_eq!(admit_shortlist_window(1024), 205);
        // Ceil, not floor: a fractional slice rounds up.
        assert_eq!(admit_shortlist_window(241), 49);
    }

    /// Union keeps grid picks first (probe priority), appends fine picks
    /// not already selected, and collapses to one cell when both rankings
    /// agree. Filtered search uses this after the exact cell scan.
    #[test]
    fn union_cell_selection_dedups_with_grid_priority() {
        assert_eq!(union_cell_selection(&[4], &[9]), vec![4, 9]);
        assert_eq!(union_cell_selection(&[4], &[4]), vec![4]);
        assert_eq!(union_cell_selection(&[4, 9], &[9, 1]), vec![4, 9, 1]);
        assert_eq!(union_cell_selection(&[], &[2]), vec![2]);
    }

    #[test]
    fn cells_ranked_by_fine_score_takes_min_per_cell_in_order() {
        let candidates: Vec<(usize, u32, f32, Option<u32>, u64)> = vec![
            (0, 0, 0.9, Some(7), 10),
            (0, 1, 0.2, Some(7), 10), // cell 7 best = 0.2
            (1, 2, 0.5, Some(3), 10), // cell 3 best = 0.5
            (1, 3, 0.5, Some(2), 10), // cell 2 ties cell 3 → lower id first
            (0, 4, 0.1, None, 10),    // untagged: ignored
        ];
        let ranked = cells_ranked_by_fine_score(&candidates);
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0], (7, 0.2));
        assert_eq!(ranked[1].0, 2, "score tie broken by lower cell id");
        assert_eq!(ranked[2].0, 3);
    }

    /// The inline stable-id fast path: hits carrying `stable_id` are resolved
    /// directly from the stamp, in hit order, without any manifest lookup or
    /// storage read (the superfile URIs below are random and absent from the
    /// manifest, so a fallback read would error).
    #[test]
    fn hidden_hits_user_ids_uses_inline_stable_id_fast_path() {
        let dim = 16;
        let table = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let reader = table.reader().expect("reader");
        let manifest = reader.manifest();

        let mk = |sid: i128| SuperfileHit {
            superfile: SuperfileUri(uuid::Uuid::new_v4()),
            local_doc_id: 0,
            score: 0.0,
            stable_id: Some(sid),
        };
        let hits = [mk(42)];
        let ids = block_on(hidden_hits_user_ids(manifest, &hits, "_id")).expect("resolve one id");
        assert_eq!(ids, vec![42], "single inline stable id returned verbatim");

        // Order is preserved across multiple stamped hits.
        let hits = [mk(42), mk(7)];
        let ids = block_on(hidden_hits_user_ids(manifest, &hits, "_id")).expect("resolve two ids");
        assert_eq!(ids, vec![42, 7], "inline stable ids returned in hit order");
    }

    #[test]
    fn hidden_classification_uses_manifest_strategy_when_options_are_unstamped() {
        let dim = 16;
        let table = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        assert!(table.options().partition_strategy.is_none());
        let mut centroid = vec![0.0; dim];
        centroid[0] = 1.0;
        let manifest = table
            .reader()
            .expect("reader")
            .manifest()
            .with_partition_strategy(PartitionStrategy::VectorCell {
                column: "emb".into(),
                clusters: ClusterCentroids::from_fp32(1, dim as u32, &centroid, vec![1]),
                routing: Default::default(),
            });
        assert!(is_hidden_vector_manifest(&manifest));
    }

    /// User path (`generation_of = None`): a small fragment sharing a cell with
    /// a much larger one must still be probed: its fine runs score worse and
    /// would lose every slot under a single global cap, so the per-`(cell,
    /// fragment)` keep floors it in.
    #[test]
    fn per_fragment_keep_probes_small_fragment_in_shared_cell() {
        // Cell 0, fragment 0 (large base): three near clusters.
        // Cell 0, fragment 1 (small delta): one farther cluster.
        let candidates = vec![
            (0usize, 10u32, 0.10f32, Some(0u32), 5u64),
            (0, 11, 0.11, Some(0), 5),
            (0, 12, 0.12, Some(0), 5),
            (1, 20, 0.30, Some(0), 5),
        ];
        let selected: HashSet<u32> = [0].into_iter().collect();
        let selected_ordered = [0u32];
        let candidate_counts: HashMap<(usize, u32), u64> = candidates
            .iter()
            .map(|(si, cluster, _, _, count)| ((*si, *cluster), *count))
            .collect();
        let mut scored = Vec::new();
        let gated = gate_fine_candidates_by_fragment(
            candidates,
            &selected,
            &selected_ordered,
            2,   // keep_floor
            0.0, // keep_pct: floor-only for this test
            1,   // gated_target: tiny so the global refill can't mask the floor
            &candidate_counts,
            &mut scored,
            None,
        );
        // The small fragment (si=1) is probed despite its worse score.
        assert!(
            gated.iter().any(|(si, _, _)| *si == 1),
            "small fragment starved from the probe set: {gated:?}"
        );
        // The large fragment keeps exactly keep_per_fragment=2 of its 3 runs.
        assert_eq!(gated.iter().filter(|(si, _, _)| *si == 0).count(), 2);
    }

    /// Hidden path (`generation_of = Some`): the fine-run keep is bounded per
    /// drain wave, pooled across every cell that wave wrote — so probing more
    /// cells does not multiply read volume. A single base wave packed across
    /// two probed cells keeps only `keep_per_fragment` runs total, not
    /// `keep_per_fragment` per cell.
    #[test]
    fn per_generation_keep_bounds_across_probed_cells() {
        // One drain wave (birth_version 100), superfile 0, packed across cells
        // 0 and 1 — three clusters in each.
        let candidates = vec![
            (0usize, 10u32, 0.10f32, Some(0u32), 5u64),
            (0, 11, 0.11, Some(0), 5),
            (0, 12, 0.12, Some(0), 5),
            (0, 20, 0.13, Some(1), 5),
            (0, 21, 0.14, Some(1), 5),
            (0, 22, 0.15, Some(1), 5),
        ];
        let selected: HashSet<u32> = [0, 1].into_iter().collect();
        let selected_ordered = [0u32, 1];
        let birth_versions = [100u64];
        let candidate_counts: HashMap<(usize, u32), u64> = candidates
            .iter()
            .map(|(si, cluster, _, _, count)| ((*si, *cluster), *count))
            .collect();
        let mut scored = Vec::new();
        let gated = gate_fine_candidates_by_fragment(
            candidates,
            &selected,
            &selected_ordered,
            2,   // keep_floor
            0.0, // keep_pct: floor-only for this test
            1,   // gated_target: tiny so the global refill can't mask the floor
            &candidate_counts,
            &mut scored,
            Some(&birth_versions),
        );
        // Bounded per wave across both cells: 2 total, not 2 per cell (=4).
        assert_eq!(
            gated.len(),
            2,
            "per-wave keep multiplied by cells: {gated:?}"
        );
        // The two globally-best runs win the slots, regardless of cell.
        let kept: HashSet<u32> = gated.iter().map(|(_, c, _)| *c).collect();
        assert_eq!(kept, [10u32, 11].into_iter().collect());
    }

    #[test]
    fn per_generation_keep_scales_with_fraction() {
        // One drain wave, six fine runs. A 50% fraction keeps three
        // (floor(0.5 × 6) = 3) — above the floor of one — so probe depth tracks the
        // wave's run count instead of a fixed absolute. This is what holds
        // recall as cells (and their fine-cluster counts) grow.
        let candidates = vec![
            (0usize, 10u32, 0.10f32, Some(0u32), 5u64),
            (0, 11, 0.11, Some(0), 5),
            (0, 12, 0.12, Some(0), 5),
            (0, 20, 0.13, Some(1), 5),
            (0, 21, 0.14, Some(1), 5),
            (0, 22, 0.15, Some(1), 5),
        ];
        let selected: HashSet<u32> = [0, 1].into_iter().collect();
        let selected_ordered = [0u32, 1];
        let birth_versions = [100u64];
        let candidate_counts: HashMap<(usize, u32), u64> = candidates
            .iter()
            .map(|(si, cluster, _, _, count)| ((*si, *cluster), *count))
            .collect();
        let mut scored = Vec::new();
        let gated = gate_fine_candidates_by_fragment(
            candidates,
            &selected,
            &selected_ordered,
            1,   // keep_floor: below the fraction, so the fraction wins
            0.5, // keep_pct: 50% of the wave's six runs -> three
            1,   // gated_target: tiny so the refill can't mask the fraction
            &candidate_counts,
            &mut scored,
            Some(&birth_versions),
        );
        assert_eq!(gated.len(), 3, "fraction keep miscounted: {gated:?}");
        // The three globally-best runs win the slots.
        let kept: HashSet<u32> = gated.iter().map(|(_, c, _)| *c).collect();
        assert_eq!(kept, [10u32, 11, 12].into_iter().collect());
    }

    /// Hidden path: a freshly drained delta wave sharing a cell with a large
    /// base wave still keeps its share — the per-wave floor protects it, the
    /// same invariant the user path relies on but keyed by `birth_version`.
    #[test]
    fn per_generation_keep_probes_small_delta_wave() {
        // Base wave (birth_version 100), superfile 0, cell 0: three near runs.
        // Delta wave (birth_version 200), superfile 1, cell 0: one farther run.
        let candidates = vec![
            (0usize, 10u32, 0.10f32, Some(0u32), 5u64),
            (0, 11, 0.11, Some(0), 5),
            (0, 12, 0.12, Some(0), 5),
            (1, 20, 0.30, Some(0), 5),
        ];
        let selected: HashSet<u32> = [0].into_iter().collect();
        let selected_ordered = [0u32];
        let birth_versions = [100u64, 200];
        let candidate_counts: HashMap<(usize, u32), u64> = candidates
            .iter()
            .map(|(si, cluster, _, _, count)| ((*si, *cluster), *count))
            .collect();
        let mut scored = Vec::new();
        let gated = gate_fine_candidates_by_fragment(
            candidates,
            &selected,
            &selected_ordered,
            2,   // keep_floor
            0.0, // keep_pct: floor-only for this test
            1,   // gated_target
            &candidate_counts,
            &mut scored,
            Some(&birth_versions),
        );
        // The delta wave (si=1) is probed despite its worse score.
        assert!(
            gated.iter().any(|(si, _, _)| *si == 1),
            "small delta wave starved from the probe set: {gated:?}"
        );
        // The base wave keeps exactly keep_per_fragment=2 of its 3 runs.
        assert_eq!(gated.iter().filter(|(si, _, _)| *si == 0).count(), 2);
    }

    #[test]
    fn over_budget_vector_error_surfaces_as_infino_over_budget() {
        // A cold vector search that crosses the budget returns
        // `VectorError::OverBudget` (see the vector reader tests). Confirm it
        // routes all the way to the public `InfinoError::OverBudget` and isn't
        // flattened to a generic query error.
        let read_err = ReadError::Vector(Box::new(VectorError::OverBudget("gate".into())));
        let q = vector_read_query_error(read_err);

        assert!(matches!(q, QueryError::OverBudget(_)), "got {q:?}");
        assert!(matches!(
            InfinoError::from(QueryError::OverBudget("x".into())),
            InfinoError::OverBudget(_)
        ));

        // A non-budget read error stays a generic query error.
        assert!(matches!(
            vector_read_query_error(ReadError::MissingKv("k")),
            QueryError::Parquet(_)
        ));
    }

    fn fixed_list_f32(dim: usize) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        )
    }

    /// Schema with id + title (FTS) + emb (vector). The supertable
    /// writer strips `emb` at commit time; vectors live in the
    /// embedded vector blob.
    fn schema_with_vector(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), false),
        ]))
    }

    fn options_one_superfile_per_commit(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(
            schema_with_vector(dim),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
                provided_centroids: None,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Construct a planted vector batch. Each doc gets a vector
    /// with one "active" component at dim `(global_id % dim)` set
    /// to 1.0 — keeps directions clearly separable so cosine
    /// distance from a query targeting a specific dim has only
    /// one cluster of close neighbors.
    fn build_vector_batch(start: u64, n: usize, dim: usize, schema: Arc<Schema>) -> RecordBatch {
        let titles = LargeStringArray::from((0..n).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let mut flat = Vec::<f32>::with_capacity(n * dim);
        for i in 0..n {
            let global = (start as usize) + i;
            for d in 0..dim {
                flat.push(if d == global % dim { 1.0 } else { 0.0 });
            }
        }
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(flat);
        let fsl = FixedSizeListArray::try_new(
            item_field,
            dim as i32,
            Arc::new(values) as Arc<dyn Array>,
            None,
        )
        .expect("FSL");
        RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(fsl)]).expect("batch")
    }

    /// Build a single-superfile oracle with the same `(id, title,
    /// emb)` rows. Note the separate `(scalar_batch, &[vector])`
    /// argument shape that `SuperfileBuilder::add_batch` takes —
    /// the supertable's writer wraps this for callers via
    /// `vector_split`, but for the oracle we plumb it manually.
    fn build_oracle_superfile(n_total: usize, dim: usize) -> Arc<SuperfileReader> {
        // Oracle path goes through SuperfileBuilder directly,
        // so we mimic the supertable's effective schema by hand:
        // `_id` is `Decimal128(38, 0)`, ids are 0..n.
        let scalar_schema = Arc::new(Schema::new(vec![
            Field::new(
                "_id",
                DataType::Decimal128(
                    crate::supertable::options::DECIMAL128_PRECISION,
                    crate::supertable::options::DECIMAL128_SCALE,
                ),
                false,
            ),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(
            scalar_schema.clone(),
            "_id",
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
                provided_centroids: None,
            }],
            Some(tok()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("builder");

        let ids = arrow_array::Decimal128Array::from((0..n_total as i128).collect::<Vec<_>>())
            .with_precision_and_scale(
                crate::supertable::options::DECIMAL128_PRECISION,
                crate::supertable::options::DECIMAL128_SCALE,
            )
            .expect("decimal128");
        let titles =
            LargeStringArray::from((0..n_total).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let scalar_batch =
            RecordBatch::try_new(scalar_schema, vec![Arc::new(ids), Arc::new(titles)])
                .expect("scalar batch");

        let mut flat = Vec::<f32>::with_capacity(n_total * dim);
        for i in 0..n_total {
            for d in 0..dim {
                flat.push(if d == i % dim { 1.0 } else { 0.0 });
            }
        }
        b.add_batch(&scalar_batch, &[flat.as_slice()])
            .expect("add_batch");
        let bytes = bytes::Bytes::from(b.finish().expect("finish"));
        Arc::new(SuperfileReader::open(bytes).expect("open"))
    }

    #[test]
    fn vector_search_empty_supertable_returns_empty() {
        let st = Supertable::create(options_one_superfile_per_commit(16)).expect("create");
        let r = st.reader().expect("reader");
        let q = vec![0.1f32; 16];
        let hits = r
            .vector_hits("emb", &q, 5, VectorSearchOptions::new(), None)
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn vector_search_k_zero_short_circuits() {
        let st = Supertable::create(options_one_superfile_per_commit(16)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_vector_batch(0, 8, 16, schema)).expect("a");
        w.commit().expect("c");
        let r = st.reader().expect("reader");
        let q = vec![0.1f32; 16];
        let hits = r
            .vector_hits("emb", &q, 0, VectorSearchOptions::new(), None)
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn vector_search_returns_ascending_distance_order() {
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_vector_batch(0, 8, dim, schema)).expect("a");
        w.commit().expect("c");
        let r = st.reader().expect("reader");
        // Query vector resembling row 0's pattern.
        let mut q = vec![0.0f32; dim];
        for (d, x) in q.iter_mut().enumerate() {
            *x = (d as f32) / 100.0 + 0.001;
        }
        let hits = r
            .vector_hits("emb", &q, 5, VectorSearchOptions::new(), None)
            .expect("query");
        assert!(!hits.is_empty());
        for w in hits.windows(2) {
            assert!(
                w[0].score <= w[1].score,
                "expected ascending: {:?} then {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn vector_search_top_k_caps_at_k() {
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        // Three commits → three superfiles × 8 docs = 24 docs.
        for chunk in 0..3u64 {
            w.append(&build_vector_batch(chunk * 8, 8, dim, schema.clone()))
                .expect("a");
            w.commit().expect("c");
        }
        let r = st.reader().expect("reader");
        let q = vec![0.1f32; dim];
        let hits = r
            .vector_hits("emb", &q, 7, VectorSearchOptions::new(), None)
            .expect("query");
        assert_eq!(hits.len(), 7);
    }

    #[test]
    fn vector_search_global_selection_recovers_neighbors_under_low_budget() {
        // 10 superfiles × 16 one-hot docs. Query e_0's true neighbors are
        // the 10 docs with id % dim == 0 (one per superfile) at cosine
        // distance 0; every other doc is orthogonal (distance 1). With
        // nprobe = 1 the global budget is only 10 clusters across all 10
        // superfiles — so this exercises real cross-superfile cluster
        // pruning (most of the 10 × n_cent clusters are skipped), and
        // recall@10 must still recover the concentrated neighbors.
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        let n_seg = 10u64;
        for chunk in 0..n_seg {
            w.append(&build_vector_batch(chunk * 16, 16, dim, schema.clone()))
                .expect("append");
            w.commit().expect("commit");
        }
        assert_eq!(st.reader().expect("reader").n_superfiles(), n_seg as usize);

        let mut q = vec![0f32; dim];
        q[0] = 1.0;
        let opts = VectorSearchOptions::new().with_nprobe(1);
        let hits = st
            .reader()
            .expect("reader")
            .vector_hits("emb", &q, 10, opts, None)
            .expect("query");

        let exact_neighbors = hits.iter().filter(|h| h.score < 1e-3).count();
        assert!(
            exact_neighbors >= 9,
            "recall@10 ≥ 0.90 under aggressive global cluster pruning; \
             recovered {exact_neighbors}/10 exact neighbors"
        );
    }

    #[test]
    fn vector_search_carries_superfile_uris_for_multi_superfile_results() {
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        for chunk in 0..3u64 {
            w.append(&build_vector_batch(chunk * 8, 8, dim, schema.clone()))
                .expect("a");
            w.commit().expect("c");
        }
        let r = st.reader().expect("reader");
        let q = vec![0.1f32; dim];
        let hits = r
            .vector_hits("emb", &q, 24, VectorSearchOptions::new(), None)
            .expect("query");
        let superfile_uris: HashSet<_> = hits.iter().map(|h| h.superfile).collect();
        // All three superfiles should contribute (high k pulls from
        // each).
        assert_eq!(superfile_uris.len(), 3);
    }

    #[test]
    fn vector_search_oracle_top_k_set_matches_single_superfile() {
        // Vector distances are superfile-independent — cosine /
        // L2-sq are functions of the query + per-doc vector only.
        // So the per-superfile-top-k → global-top-k pattern recovers
        // the same set as a single-superfile search, modulo each
        // IVF's nprobe-driven recall (we use a high-recall config).
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        // 24 docs across 3 superfiles.
        for chunk in 0..3u64 {
            w.append(&build_vector_batch(chunk * 8, 8, dim, schema.clone()))
                .expect("a");
            w.commit().expect("c");
        }
        let oracle = build_oracle_superfile(24, dim);

        // High-recall config: full nprobe + plenty of rerank.
        let opts = VectorSearchOptions::new().with_nprobe(4);

        // Query targets dim 0 — closest neighbors are docs whose
        // global id is 0 mod dim (i.e. 0 and 16 in 24 docs at
        // dim=16). Other docs have orthogonal vectors and contribute
        // cosine distance = 1.0.
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;

        // The oracle is a single-superfile `SuperfileReader` whose search
        // is async-only; drive it on a throwaway runtime. The supertable
        // reader below uses its sync public API.
        let oracle_hits =
            block_on(oracle.vector_hits_async("emb", &q, 2, opts)).expect("oracle query");
        let oracle_globals: HashSet<u32> = oracle_hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(oracle_globals, [0u32, 16].iter().copied().collect());

        let st_reader = st.reader().expect("reader");
        let st_hits = st_reader
            .vector_hits("emb", &q, 2, opts, None)
            .expect("supertable query");
        let manifest = st_reader.manifest();
        let st_globals: HashSet<u32> = st_hits
            .iter()
            .map(|h| {
                let seg_idx = manifest
                    .superfiles
                    .iter()
                    .position(|e| e.uri == h.superfile)
                    .expect("superfile in manifest");
                (seg_idx as u32) * 8 + h.local_doc_id
            })
            .collect();
        assert_eq!(st_hits.len(), oracle_hits.len());
        assert_eq!(st_globals, oracle_globals);
    }

    #[test]
    fn vector_search_unknown_column_errors() {
        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_vector_batch(0, 8, dim, schema)).expect("a");
        w.commit().expect("c");
        let r = st.reader().expect("reader");
        let q = vec![0.1f32; dim];
        let err = r
            .vector_hits("nope", &q, 5, VectorSearchOptions::new(), None)
            .expect_err("expected error");
        // Undeclared columns are rejected up front with a naming error —
        // not the old shape (silent metric default + blind per-superfile
        // probe, surfacing later as a kernel decode error).
        assert!(
            matches!(&err, QueryError::Execute(m) if m.contains("unknown vector column")),
            "got {err:?}"
        );
    }

    // ---- Tombstone filter helper: direct-call coverage --------------
    //
    // Exercises `apply_tombstone_filter` against a synthesized
    // bitmap + hit list without going through the full IVF +
    // lazy-source vector search path. The hook logic is identical
    // to the FTS path (both drop hits whose `local_doc_id` is in
    // the per-superfile bitmap); this direct test pins the
    // contract for the vector side.

    use tempfile::TempDir;
    use uuid::Uuid;

    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            manifest::{SuperfileEntry, SuperfileUri},
            query::SuperfileHit,
            tombstones::{SidecarCache, TombstoneSeqView, cache::DEFAULT_SEAL_TTL},
            wal::{WalStore, tombstones_codec::TombstonesSidecar},
        },
    };

    fn synthetic_entry(superfile_id: Uuid) -> SuperfileEntry {
        SuperfileEntry {
            birth_version: 0,
            superfile_id,
            uri: SuperfileUri(superfile_id),
            n_docs: 100,
            id_min: 0,
            id_max: 99,
            scalar_stats: std::collections::HashMap::new(),
            fts_summary: std::collections::HashMap::new(),
            vector_summary: std::collections::HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: crate::superfile::vector::layout::VectorLayout::Ivf,
            subsection_offsets: None,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_tombstone_filter_drops_set_bits() {
        // Build a SidecarCache backed by a real (LocalFs) storage so
        // the hook exercises the same cache machinery that the
        // production query path uses.
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let ws = WalStore::new(Arc::clone(&storage));
        let sf_id = Uuid::from_u128(0xFEEDFACE);
        let cache = Arc::new(SidecarCache::new(
            ws.clone(),
            DEFAULT_SEAL_TTL,
            Arc::new(TombstoneSeqView {
                manifest_id: 1,
                seqs: [(sf_id, 1u64)].into_iter().collect(),
            }),
        ));
        // Pre-populate a sidecar with doc-ids 1, 3, 5 set.
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(3);
        bitmap.insert(5);
        ws.put_tombstones(sf_id, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put sidecar");

        let entry = synthetic_entry(sf_id);
        let mut hits: Vec<SuperfileHit> = (0..8u32)
            .map(|d| SuperfileHit {
                superfile: entry.uri,
                local_doc_id: d,
                score: d as f32,
                stable_id: None,
            })
            .collect();

        crate::supertable::query::dispatch::apply_tombstone_filter(
            Some(&cache),
            &entry,
            &mut hits,
            std::time::Instant::now(),
        )
        .expect("filter");

        let remaining: Vec<u32> = hits.iter().map(|h| h.local_doc_id).collect();
        assert_eq!(remaining, vec![0u32, 2, 4, 6, 7]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_tombstone_filter_is_no_op_without_cache() {
        let entry = synthetic_entry(Uuid::from_u128(0xABCD));
        let mut hits: Vec<SuperfileHit> = (0..4u32)
            .map(|d| SuperfileHit {
                superfile: entry.uri,
                local_doc_id: d,
                score: 0.0,
                stable_id: None,
            })
            .collect();
        let original = hits.clone();
        crate::supertable::query::dispatch::apply_tombstone_filter(
            None,
            &entry,
            &mut hits,
            std::time::Instant::now(),
        )
        .expect("no-cache");
        assert_eq!(hits, original);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_tombstone_filter_short_circuits_on_empty_bitmap() {
        // Superfile absent from the seq map → the cache answers
        // "no tombstones" authoritatively (zero GETs) and
        // `bitmap.is_empty()` short-circuits the filter loop.
        // Hit list is unchanged.
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let ws = WalStore::new(Arc::clone(&storage));
        let cache = Arc::new(SidecarCache::new(
            ws,
            DEFAULT_SEAL_TTL,
            Arc::new(TombstoneSeqView::default()),
        ));

        let entry = synthetic_entry(Uuid::from_u128(0x1111));
        let mut hits: Vec<SuperfileHit> = (0..4u32)
            .map(|d| SuperfileHit {
                superfile: entry.uri,
                local_doc_id: d,
                score: 0.0,
                stable_id: None,
            })
            .collect();
        let original = hits.clone();
        crate::supertable::query::dispatch::apply_tombstone_filter(
            Some(&cache),
            &entry,
            &mut hits,
            std::time::Instant::now(),
        )
        .expect("filter");
        assert_eq!(hits, original);
    }

    /// A cell listed in a superfile's vector summary is routable and
    /// contributes a posting entry — unless the manifest marks it
    /// superseded (its on-disk blocks replaced by an in-place split), in
    /// which case it is dropped from the posting map so it is never
    /// selected, scored, or fetched. Supersession is keyed per superfile,
    /// so an entry for a different superfile leaves this one untouched.
    #[test]
    fn postings_by_cell_from_summaries_skips_superseded_cells() {
        use crate::supertable::manifest::{CellVectorSummary, ClusterCentroids, VectorSummary};

        const DIM: u32 = 4;
        let column = "emb";

        // One single-cluster cell carrying `count` indexed rows, so a
        // present cell contributes a nonzero posting sum.
        let cell = |cell_id: u32, count: u32| CellVectorSummary {
            cell_id: Some(cell_id),
            clusters: ClusterCentroids::from_fp32(
                1,
                DIM,
                &vec![cell_id as f32; DIM as usize],
                vec![count],
            ),
        };

        let sf_id = Uuid::from_u128(0xC0FFEE);
        let mut entry = synthetic_entry(sf_id);
        entry.vector_summary.insert(
            column.into(),
            VectorSummary {
                centroid: vec![0.0; DIM as usize],
                cells: vec![cell(1, 10), cell(2, 20), cell(3, 30)],
            },
        );
        let entries = vec![Arc::new(entry)];

        // No supersessions: every tagged cell is routable.
        let empty = BTreeMap::new();
        let (postings, any_tagged) =
            postings_by_cell_from_summaries(&entries, column, None, &empty);
        assert!(any_tagged);
        assert_eq!(postings.get(&1), Some(&10));
        assert_eq!(postings.get(&2), Some(&20));
        assert_eq!(postings.get(&3), Some(&30));

        // Cell 2 superseded for this superfile: dropped, the rest remain.
        let mut superseded = BTreeMap::new();
        superseded.insert(sf_id, BTreeSet::from([2u32]));
        let (postings, any_tagged) =
            postings_by_cell_from_summaries(&entries, column, None, &superseded);
        assert!(any_tagged, "surviving cells still tag");
        assert!(!postings.contains_key(&2), "superseded cell is skipped");
        assert_eq!(postings.get(&1), Some(&10));
        assert_eq!(postings.get(&3), Some(&30));

        // A supersession keyed to a different superfile does not affect
        // this one.
        let mut other = BTreeMap::new();
        other.insert(Uuid::from_u128(0xDEAD), BTreeSet::from([1u32]));
        let (postings, _) = postings_by_cell_from_summaries(&entries, column, None, &other);
        assert_eq!(postings.get(&1), Some(&10));
        assert_eq!(postings.get(&2), Some(&20));
        assert_eq!(postings.get(&3), Some(&30));
    }

    /// `score_fine_candidates` must skip a superseded cell exactly as the
    /// posting map does: a cell whose blocks were replaced by an in-place split
    /// is never fine-scored or deferred (hence never fetched), so the dead
    /// parent blocks cost nothing on queries that hit the split cell. Without
    /// the guard the parent's blocks are re-scored and re-fetched every query
    /// until a merge reclaims them (correct only via downstream stable-id dedup,
    /// but wasteful).
    #[test]
    fn score_fine_candidates_skips_superseded_cells() {
        use crate::supertable::manifest::{CellVectorSummary, ClusterCentroids, VectorSummary};

        const DIM: u32 = 4;
        let column = "emb";
        let cell = |cell_id: u32, count: u32| CellVectorSummary {
            cell_id: Some(cell_id),
            clusters: ClusterCentroids::from_fp32(
                1,
                DIM,
                &vec![cell_id as f32; DIM as usize],
                vec![count],
            ),
        };
        let sf_id = Uuid::from_u128(0xC0FFEE);
        let mut entry = synthetic_entry(sf_id);
        entry.vector_summary.insert(
            column.into(),
            VectorSummary {
                centroid: vec![0.0; DIM as usize],
                cells: vec![cell(1, 10), cell(2, 20), cell(3, 30)],
            },
        );
        let entries = vec![Arc::new(entry)];
        let query = vec![0.0f32; DIM as usize];

        // Every cell `score_fine_candidates` touches — scored (candidate
        // `cell_id`) or deferred (`DeferredCellRescore.cell_id`).
        let touched = |superseded: &BTreeMap<Uuid, BTreeSet<u32>>| -> HashSet<u32> {
            let (cands, deferred) = score_fine_candidates(
                &entries,
                column,
                &query,
                Metric::L2Sq,
                None,
                None,
                superseded,
            )
            .expect("score");
            cands
                .iter()
                .filter_map(|(_, _, _, cid, _)| *cid)
                .chain(deferred.iter().filter_map(|d| d.cell_id))
                .collect()
        };

        // No supersession: all three cells are touched.
        let empty = BTreeMap::new();
        assert_eq!(touched(&empty), HashSet::from([1, 2, 3]));

        // Cell 2 superseded for this superfile: never scored or deferred, so its
        // blocks are never fetched.
        let mut superseded = BTreeMap::new();
        superseded.insert(sf_id, BTreeSet::from([2u32]));
        assert_eq!(
            touched(&superseded),
            HashSet::from([1, 3]),
            "superseded cell is not fine-scored or fetched"
        );
    }

    #[test]
    fn hybrid_vector_leg_uses_user_superfiles_not_hidden() {
        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let opts = opts.with_storage(storage);
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 32, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");

        let reader = st.reader().expect("reader");
        let user_uris: HashSet<_> = reader.manifest().superfiles.iter().map(|e| e.uri).collect();
        assert!(
            reader.vector_index_table().is_some(),
            "hidden index must exist"
        );

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let hits = reader
            .hybrid_search(
                "title",
                "doc",
                crate::superfile::fts::reader::BoolMode::Or,
                "emb",
                &q,
                VectorSearchOptions::new(),
                5,
            )
            .expect("hybrid");
        assert!(!hits.is_empty());
        for hit in &hits {
            assert!(
                user_uris.contains(&hit.superfile),
                "hybrid vector leg must fan out on user superfiles, got {:?}",
                hit.superfile
            );
        }
    }

    #[test]
    fn vector_search_row_return_resolves_through_hidden_index() {
        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let opts = opts.with_storage(storage);
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 16, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let batches = st
            .reader()
            .expect("reader")
            .vector_search(
                "emb",
                &q,
                5,
                VectorSearchOptions::new(),
                None,
                Some(&["_id", "score"]),
            )
            .expect("vector_search rows");
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(
            rows >= 1,
            "row-returning vector_search must resolve user rows"
        );
    }

    /// Post-drain compaction workload: a larger corpus is drained then
    /// optimized (compaction merges/splits cells), searched, partly deleted,
    /// and optimized again — exercising the compaction path and confirming
    /// search survives it.
    #[test]
    fn compaction_after_drain_preserves_search() {
        use datafusion::prelude::{col, lit};

        use crate::config::OptimizeOptions;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        for c in 0..4u64 {
            let mut w = st.writer().expect("writer");
            w.append(&build_vector_batch(c * 32, 32, dim, schema.clone()))
                .expect("append");
            w.commit().expect("commit");
        }
        st.drain_vectors_to_cells_sync().expect("drain");

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let search = |st: &Supertable| {
            st.reader()
                .expect("reader")
                .vector_hits(
                    "emb",
                    &q,
                    20,
                    VectorSearchOptions::new().with_nprobe(4),
                    None,
                )
                .expect("search")
                .len()
        };
        assert!(
            search(&st) >= 8,
            "e_0's exact matches present pre-compaction"
        );

        st.optimize(&OptimizeOptions::default()).expect("optimize");
        assert!(
            search(&st) >= 8,
            "compaction must preserve the exact-match docs"
        );

        // Delete then compact again; search must still work.
        let stats = st.delete(col("title").eq(lit("doc 0"))).expect("delete");
        assert!(stats.n_tombstoned() >= 1, "delete tombstones matching docs");
        st.optimize(&OptimizeOptions::default())
            .expect("optimize after delete");
        assert!(
            !st.reader()
                .expect("reader")
                .vector_hits(
                    "emb",
                    &q,
                    20,
                    VectorSearchOptions::new().with_nprobe(4),
                    None
                )
                .expect("search after delete+compact")
                .is_empty(),
            "search still returns hits after delete + compaction"
        );
    }

    /// Warm-vs-cold parity for the width sweep — the review's "cold
    /// fixture". The warm arm defers survivors to the global shortlist;
    /// a cell with non-resident prefixes reranks in-probe under the
    /// divided cold budget and never enters global selection. At an
    /// untruncated budget both arms exact-rerank every candidate, so
    /// the same query on the same committed table must return identical
    /// hits regardless of cache state. (The replica variant stays
    /// blocked on the drain_replica_target_factor crash filed
    /// separately.)
    #[test]
    fn vector_cold_arm_matches_warm_top_k() {
        use crate::test_helpers::lazy_foreground_disk_cache;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;

        // Warm: default in-memory reader cache — every prefix resident,
        // the sweep defers to the global shortlist.
        let warm_hits = {
            let st = Supertable::create(
                options_one_superfile_per_commit(dim).with_storage(Arc::clone(&storage)),
            )
            .expect("create");
            for c in 0..4u64 {
                let mut w = st.writer().expect("writer");
                w.append(&build_vector_batch(c * 32, 32, dim, schema.clone()))
                    .expect("append");
                w.commit().expect("commit");
            }
            st.drain_vectors_to_cells_sync().expect("drain");
            st.reader()
                .expect("reader")
                .vector_hits(
                    "emb",
                    &q,
                    20,
                    VectorSearchOptions::new().with_nprobe(4),
                    None,
                )
                .expect("warm search")
        };

        // Cold: a fresh handle reading through a lazy disk-cache source —
        // nothing resident, every probed cell takes the cold arm.
        let cache_dir = tempfile::TempDir::new().expect("cache dir");
        let cache = lazy_foreground_disk_cache(Arc::clone(&storage), cache_dir.path());
        let st_cold = Supertable::open(
            options_one_superfile_per_commit(dim)
                .with_storage(Arc::clone(&storage))
                .with_disk_cache(Arc::clone(&cache)),
        )
        .expect("open cold");
        let cold_hits = st_cold
            .reader()
            .expect("reader")
            .vector_hits(
                "emb",
                &q,
                20,
                VectorSearchOptions::new().with_nprobe(4),
                None,
            )
            .expect("cold search");
        assert!(
            cache.stats().n_cold_fetches >= 1,
            "the cold handle must actually read through the lazy cache \
             (otherwise this fixture proves nothing)"
        );

        assert_eq!(
            warm_hits, cold_hits,
            "cache state must not change the top-k (warm deferred-global vs \
             cold divided in-probe, both exact at an untruncated budget)"
        );
    }

    /// A larger post-drain corpus (many docs across several commits, drained
    /// into multiple cells) searched with a wider `k` and `nprobe`, so the
    /// query reranks candidates spanning multiple clusters — the multi-cluster
    /// rerank / candidate-block path a single-cell search never reaches.
    #[test]
    fn vector_search_multi_cell_rerank_over_larger_corpus() {
        use crate::superfile::fts::reader::BoolMode;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        // Four commits × 32 docs → 128 docs over 16 one-hot directions, several
        // per cell, so a drained query reranks across multiple cells.
        for c in 0..4u64 {
            let mut w = st.writer().expect("writer");
            w.append(&build_vector_batch(c * 32, 32, dim, schema.clone()))
                .expect("append");
            w.commit().expect("commit");
        }
        st.drain_vectors_to_cells_sync().expect("drain");

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        // Wide k + nprobe: rerank spans several probed cells.
        let hits = st
            .reader()
            .expect("reader")
            .vector_hits(
                "emb",
                &q,
                20,
                VectorSearchOptions::new().with_nprobe(4),
                None,
            )
            .expect("wide search");
        let unfiltered_exact = near_count(&hits);
        assert!(
            unfiltered_exact >= 8,
            "e_0 has 8 exact matches across commits; wide search must find \
             them, got {unfiltered_exact}"
        );

        // Filtered variant over the same corpus. The title predicate
        // matches every row, so the allow-set machinery runs with full
        // coverage and must recover the SAME exact matches as the
        // unfiltered sweep — pinning that a filtered query with an explicit
        // `nprobe` keeps its full per-cell shortlist (the width-era
        // rerank-budget divide and per-fragment gating are unfiltered-only
        // semantics; a filtered sweep that picked them up would starve a
        // sparse allow-set).
        let filtered = st
            .reader()
            .expect("reader")
            .vector_hits(
                "emb",
                &q,
                20,
                VectorSearchOptions::new().with_nprobe(4),
                Some(VectorFilter {
                    column: "title",
                    query: "doc",
                    mode: BoolMode::Or,
                }),
            )
            .expect("filtered wide search");
        assert_eq!(
            near_count(&filtered),
            unfiltered_exact,
            "filtered+nprobe must find the same exact matches as the \
             unfiltered sweep"
        );
    }

    /// Score bound separating planted neighbors from orthogonal docs in the
    /// drained one-hot fixture: query-aligned directions score well below
    /// it, every other direction scores exactly 1.0 (cos = 0).
    const ORTHOGONAL_SCORE: f32 = 0.9;
    /// Fixture dimensionality — one one-hot direction per dim.
    const FIXTURE_DIM: usize = 16;
    /// Commits in the drained fixture (each its own user superfile).
    const FIXTURE_COMMITS: u64 = 4;
    /// Rows per fixture commit; `FIXTURE_COMMITS x this / FIXTURE_DIM`
    /// docs land on each one-hot direction.
    const FIXTURE_ROWS_PER_COMMIT: usize = 32;
    /// Docs planted per one-hot direction (128 docs mod 16 dims).
    const DOCS_PER_DIRECTION: usize =
        FIXTURE_COMMITS as usize * FIXTURE_ROWS_PER_COMMIT / FIXTURE_DIM;

    /// Drained planted fixture shared by the probe-width tests: 128 one-hot
    /// docs over 16 directions in 4 commits, drained into per-direction
    /// cells, plus a query leaning on three directions — its exact top-24
    /// spans three cells. Returns `(tempdir, table, query, k)`; the tempdir
    /// must outlive the table.
    fn drained_three_direction_fixture() -> (tempfile::TempDir, Supertable, Vec<f32>, usize) {
        let dim = FIXTURE_DIM;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        for c in 0..FIXTURE_COMMITS {
            let mut w = st.writer().expect("writer");
            w.append(&build_vector_batch(
                c * FIXTURE_ROWS_PER_COMMIT as u64,
                FIXTURE_ROWS_PER_COMMIT,
                dim,
                schema.clone(),
            ))
            .expect("append");
            w.commit().expect("commit");
        }
        st.drain_vectors_to_cells_sync().expect("drain");
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        q[1] = 0.9;
        q[2] = 0.8;
        (dir, st, q, 3 * DOCS_PER_DIRECTION)
    }

    /// Planted neighbors among `hits` (orthogonal docs score exactly 1.0).
    fn near_count(hits: &[SuperfileHit]) -> usize {
        hits.iter().filter(|h| h.score < ORTHOGONAL_SCORE).count()
    }

    /// Explicit caller `nprobe` widens the post-drain UNFILTERED cell sweep.
    ///
    /// Regression test for the discarded-override bug: unfiltered hidden
    /// routing used to keep the persisted p=1 params and ignore
    /// `with_nprobe`, so a query whose true top-k spans several cells could
    /// never recover the neighbors outside the single probed cell (measured
    /// on Cohere-1M/768d at k=100: recall capped at 0.61 with the probed
    /// cell scanned in full). The corpus below plants the top-k across
    /// THREE cells — two would pass even without the fix, because the
    /// no-override union selection already admits `UNION_FINE_PICKS_MIN = 2`
    /// fine-ranked cells.
    #[test]
    fn caller_nprobe_widens_unfiltered_post_drain_sweep() {
        let (_dir, st, q, k) = drained_three_direction_fixture();

        // Narrow override: the drain calibrated a wide width law for this
        // corpus, but `with_nprobe(1)` pulls the sweep back below it — the
        // override wins in both directions, it is never merely a floor.
        // (nprobe=1 still reads the grid∪fine union's UNION_FINE_PICKS_MIN
        // fine picks, so "narrow" means fewer cells than the law, not one.)
        let narrow_hits = st
            .reader()
            .expect("reader")
            .vector_hits(
                "emb",
                &q,
                k,
                VectorSearchOptions::new().with_nprobe(1),
                None,
            )
            .expect("narrow search");
        let narrow = near_count(&narrow_hits);
        assert!(
            narrow >= DOCS_PER_DIRECTION && narrow < k,
            "with_nprobe(1) must narrow the sweep below the law-widened \
             default (got {narrow} of {k})"
        );

        // Wide override: recovers the full exact top-k across three cells.
        let wide_hits = st
            .reader()
            .expect("reader")
            .vector_hits(
                "emb",
                &q,
                k,
                VectorSearchOptions::new().with_nprobe(DOCS_PER_DIRECTION),
                None,
            )
            .expect("wide search");
        assert_eq!(
            near_count(&wide_hits),
            k,
            "with_nprobe must widen the unfiltered post-drain sweep to all \
             {k} exact neighbors across three cells — caller nprobe is an \
             override, never discarded"
        );
    }

    /// [`apply_width_pin`] semantics, all four arms. Depth MUST ride the
    /// width on unfiltered pins — a widened sweep at the persisted
    /// fine-first depth reads only each cell's first runs and caps recall
    /// regardless of width (measured ~0.83 on Cohere-1M at any nprobe).
    #[test]
    fn width_pin_lifts_fine_depth_on_unfiltered_overrides() {
        let base = CellRoutingParams {
            nprobe_min: 1,
            nprobe_max: 1,
            fine_nprobe: 6,
            ..CellRoutingParams::default()
        };

        // Unfiltered caller nprobe: width pinned, depth lifted, sweep
        // engaged (clamped to populated cells).
        let mut r = base;
        let sweep = apply_width_pin(&mut r, Some(64), None, false, 40);
        assert_eq!((r.nprobe_min, r.nprobe_max), (64, 64));
        assert_eq!(r.fine_nprobe, usize::MAX, "depth rides the pin");
        assert_eq!(sweep, Some(40), "sweep width clamps to populated cells");

        // Filtered caller nprobe: width pinned, but the filtered fine
        // floor is preserved and the width machinery stays disengaged.
        let mut r = base;
        let sweep = apply_width_pin(&mut r, Some(64), None, true, 40);
        assert_eq!((r.nprobe_min, r.nprobe_max), (64, 64));
        assert_eq!(r.fine_nprobe, 6, "filtered floors untouched by the pin");
        assert_eq!(sweep, None, "filtered sweeps keep pre-width budget");

        // Law width (no caller override): same pin + depth as an explicit
        // unfiltered nprobe.
        let mut r = base;
        let sweep = apply_width_pin(&mut r, None, Some(45), false, 256);
        assert_eq!((r.nprobe_min, r.nprobe_max), (45, 45));
        assert_eq!(r.fine_nprobe, usize::MAX, "depth rides the law");
        assert_eq!(sweep, Some(45));

        // Nothing pinned: routing untouched.
        let mut r = base;
        let sweep = apply_width_pin(&mut r, None, None, false, 256);
        assert_eq!(r, base);
        assert_eq!(sweep, None);
    }

    /// The deferred-rerank width sweep under the tightest possible rerank
    /// budget: `rerank_mult = 1` leaves the global selection exactly `k`
    /// slots across ALL probed cells — the even-split divide would hand
    /// each cell a starved slice, but the global pool ranks every cell's
    /// estimates together, so the planted neighbors (whose one-hot
    /// estimates dominate the orthogonal rest) all survive selection and
    /// rerank to the exact top-k.
    #[test]
    fn global_shortlist_survives_minimal_rerank_budget() {
        let (_dir, st, q, k) = drained_three_direction_fixture();
        let hits = st
            .reader()
            .expect("reader")
            .vector_hits(
                "emb",
                &q,
                k,
                VectorSearchOptions::new()
                    .with_nprobe(DOCS_PER_DIRECTION)
                    .with_rerank_mult(1),
                None,
            )
            .expect("wide search, minimal budget");
        assert_eq!(
            near_count(&hits),
            k,
            "global shortlist selection at k x 1 must keep every planted \
             neighbor across three cells"
        );
    }

    /// [`select_global_shortlist`] is deterministic and order-independent:
    /// the same candidates in any arrival order produce the same cut, ties
    /// on estimate break on (unit, cell, pos, did), and the limit is a
    /// hard truncation.
    #[test]
    fn select_global_shortlist_is_deterministic() {
        let cand = |est: f32, cell: usize, pos: u32, did: u32| ScanCandidate {
            did,
            estimate: est,
            pos,
            cluster_id: 0,
            cell_idx: cell,
        };
        let a = vec![
            (0usize, cand(0.9, 0, 1, 1)),
            (1usize, cand(0.9, 0, 1, 1)),
            (0usize, cand(0.5, 1, 2, 2)),
            (1usize, cand(0.7, 0, 3, 3)),
        ];
        let mut b = a.clone();
        b.reverse();
        let pick = |v: Vec<(usize, ScanCandidate)>| {
            select_global_shortlist(v, 3)
                .into_iter()
                .map(|(si, c)| (si, c.cell_idx, c.pos, c.did))
                .collect::<Vec<_>>()
        };
        let mut from_a = pick(a);
        let mut from_b = pick(b);
        from_a.sort_unstable();
        from_b.sort_unstable();
        assert_eq!(
            from_a, from_b,
            "the kept SET must be arrival-order independent (its internal \
             order is unspecified — the partition does not sort)"
        );
        assert_eq!(from_a.len(), 3, "limit is a hard truncation");
        // Tie on estimate 0.9 breaks by unit, so both 0.9 copies stay and
        // the 0.7 candidate takes the last slot; 0.5 falls off the cut.
        assert_eq!(from_a, vec![(0, 0, 1, 1), (1, 0, 1, 1), (1, 0, 3, 3)]);
    }

    /// A clean drain calibrates the probe-width law from the table's own
    /// rows and stamps it into the manifest routing; a DEFAULT search (no
    /// nprobe, no config) then widens to the calibrated width. On this
    /// planted corpus the exact top-24 spans three cells, which the old
    /// fixed p=1 default could never recover (it returned one direction's
    /// 8 docs); the law measures the spread and buys the coverage without
    /// the caller knowing any knob exists.
    #[test]
    fn drain_calibrated_width_law_widens_default_search() {
        let (_dir, st, q, k) = drained_three_direction_fixture();
        let hits = st
            .reader()
            .expect("reader")
            .vector_hits("emb", &q, k, VectorSearchOptions::new(), None)
            .expect("default search");
        let near = near_count(&hits);
        assert_eq!(
            near, k,
            "drain-calibrated width law must widen the default sweep to all \
             {k} exact neighbors across three cells, got {near}"
        );
    }

    /// An incremental drain calibrates only the newly spilled tail; its
    /// measurement must never NARROW the stamped law (element-wise max
    /// merge), or a small tightly-clustered append would under-probe all
    /// older data. The delta here spans few directions, so its own law is
    /// far narrower than the fixture's — the default search must still
    /// recover the full three-cell top-k afterward.
    #[test]
    fn incremental_drain_never_narrows_the_width_law() {
        let (_dir, st, q, k) = drained_three_direction_fixture();
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(
            (FIXTURE_COMMITS * FIXTURE_ROWS_PER_COMMIT as u64) + 1,
            DOCS_PER_DIRECTION,
            FIXTURE_DIM,
            schema_with_vector(FIXTURE_DIM),
        ))
        .expect("append delta");
        w.commit().expect("commit delta");
        st.drain_vectors_to_cells_sync().expect("incremental drain");

        let hits = st
            .reader()
            .expect("reader")
            .vector_hits("emb", &q, k, VectorSearchOptions::new(), None)
            .expect("default search after incremental drain");
        let near = near_count(&hits);
        assert_eq!(
            near,
            k,
            "the delta-only calibration must not narrow the stamped law: \
             default search lost {} of {k} planted neighbors",
            k - near
        );
    }

    /// The `Supertable::vector_search` handle wrapper (tests normally call
    /// `reader().vector_search`) delegates to the reader and returns rows.
    #[test]
    fn supertable_vector_search_wrapper_returns_rows() {
        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 16, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        st.drain_vectors_to_cells_sync().expect("drain");

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let batches = st
            .vector_search(
                "emb",
                &q,
                5,
                VectorSearchOptions::new(),
                None,
                Some(&["_id"]),
            )
            .expect("handle-level vector_search");
        assert!(
            batches.iter().map(|b| b.num_rows()).sum::<usize>() >= 1,
            "handle wrapper must return rows"
        );
    }

    /// Bitmap-filtered vector search over corpus-global ids
    /// (`vector_hits_global_allow_async` → `prepare_vector_global_allow_async`,
    /// user-table path): only the allowed global rows (contiguous ingest order)
    /// are eligible, so every hit is within the allow-set.
    #[test]
    fn vector_hits_global_allow_restricts_to_allowed_ids() {
        use roaring::RoaringBitmap;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 16, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
        drop(w);

        // Pre-drain: allow only global rows 0,1,2 (ingest order → docs at dims
        // 0,1,2), mapped by the user-table path.
        let allow: Arc<RoaringBitmap> = Arc::new([0u32, 1, 2].into_iter().collect());
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let hits = block_on(st.reader().expect("reader").vector_hits_global_allow_async(
            "emb",
            &q,
            16,
            VectorSearchOptions::new().with_nprobe(32),
            allow,
        ))
        .expect("global-allow search");
        assert!(!hits.is_empty(), "the e_0 doc is allowed and must be found");
        assert!(
            hits.len() <= 3,
            "only the 3 allowed global rows may appear, got {}",
            hits.len()
        );
    }

    /// `prepare_vector_stable_allow_async` on a *valid* drained id resolves it
    /// to a hidden-cell allow-set (the success path; the existing test only
    /// covers the unknown-id error). Post-drain it must key the allow-set by
    /// hidden-index URIs and be non-empty.
    #[test]
    fn prepare_vector_stable_allow_maps_valid_drained_id() {
        use arrow_array::Decimal128Array;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 16, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        st.drain_vectors_to_cells_sync().expect("drain");

        // Pull one real stable id from a row-returning search.
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let batches = st
            .reader()
            .expect("reader")
            .vector_search(
                "emb",
                &q,
                1,
                VectorSearchOptions::new(),
                None,
                Some(&["_id"]),
            )
            .expect("row search");
        let id = batches
            .iter()
            .find_map(|b| {
                b.column_by_name("_id")
                    .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
                    .filter(|c| !c.is_empty())
                    .map(|c| c.value(0))
            })
            .expect("a resolved _id");

        let prepared = block_on(
            st.reader()
                .expect("reader")
                .prepare_vector_stable_allow_async(Arc::new(vec![id])),
        )
        .expect("valid drained id must map");
        assert!(
            prepared.use_hidden_index,
            "post-drain allow-set is keyed by the hidden index"
        );
        assert!(
            !prepared.allow_by_uri.is_empty(),
            "a valid id resolves to a non-empty hidden-cell allow-set"
        );
    }

    /// Row-returning vector search AFTER a drain resolves hidden-cell hits back
    /// to user `_id`s via the inline stable-id region — a path the pre-drain
    /// row-return test never reaches. The exact-match doc's id must come back.
    #[test]
    fn vector_search_rows_post_drain_resolve_hidden_ids() {
        use arrow_array::Decimal128Array;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 16, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        st.drain_vectors_to_cells_sync().expect("drain");

        // e_0 is the exact vector of doc 0 (id 0); it must resolve back.
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let batches = st
            .reader()
            .expect("reader")
            .vector_search(
                "emb",
                &q,
                5,
                VectorSearchOptions::new(),
                None,
                Some(&["_id", "score"]),
            )
            .expect("post-drain row search");
        let mut ids = Vec::new();
        for b in &batches {
            let col = b
                .column_by_name("_id")
                .expect("_id column")
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("_id is decimal128");
            for i in 0..col.len() {
                ids.push(col.value(i));
            }
        }
        assert_eq!(ids.len(), 5, "k=5 over 16 docs returns 5 rows");
        // Ids are assigned in append order (base + row index), so doc 0 — the
        // exact match for e_0 — carries the smallest id. It must rank first,
        // which proves the hidden-cell hit resolved back to the right user row.
        assert_eq!(
            ids[0],
            *ids.iter().min().expect("ids is non-empty"),
            "the exact-match doc must rank first, got {ids:?}"
        );
    }

    /// Pre-drain filtered, row-returning vector search across several user
    /// superfiles: exercises the token candidate-bitmap fan-out over the
    /// survivors and the stable-id resolution for row projection — the
    /// user-table filtered path (post-drain search fans out on the hidden
    /// index instead).
    #[test]
    fn filtered_vector_search_row_return_fans_out_over_user_superfiles() {
        use crate::superfile::fts::reader::BoolMode;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_superfile_per_commit(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        // Three commits → three user superfiles the filter must fan out across.
        for start in [0u64, 16, 32] {
            let mut w = st.writer().expect("writer");
            w.append(&build_vector_batch(start, 16, dim, schema.clone()))
                .expect("append");
            w.commit().expect("commit");
        }

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let batches = st
            .reader()
            .expect("reader")
            .vector_search(
                "emb",
                &q,
                10,
                VectorSearchOptions::new(),
                Some(VectorFilter {
                    column: "title",
                    query: "doc",
                    mode: BoolMode::Or,
                }),
                Some(&["_id", "score"]),
            )
            .expect("filtered row search");
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(
            rows >= 1,
            "filtered fan-out must resolve rows across user superfiles"
        );
    }

    /// Filtered vector search driven by a lowered [`CandidatePlan`] — the
    /// boolean-plan fan-out (`candidate_bitmaps_from_plan`). The plan resolves
    /// the title predicate to per-superfile candidate bitmaps, then vector
    /// ranking runs over the survivors.
    #[test]
    fn vector_hits_filtered_by_plan_returns_matching_docs() {
        use std::collections::HashSet;

        use datafusion::prelude::{col, lit};

        use crate::{
            superfile::vector::rerank_codec::RerankCodec,
            supertable::query::candidate::CandidatePlan,
        };

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let opts = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 32, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        st.drain_vectors_to_cells_sync().expect("drain");

        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let fts_cols: HashSet<&str> = HashSet::from(["title"]);
        let filters = [col("title").eq(lit("doc"))];
        let plan = CandidatePlan::from_filters(&filters, &fts_cols, &|col| {
            manifest.options.fts_tokenizer_for(col)
        });

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let hits = block_on(reader.vector_hits_filtered_by_plan(
            "emb",
            &q,
            10,
            VectorSearchOptions::new(),
            &plan,
        ))
        .expect("plan-filtered vector search");
        assert!(
            !hits.is_empty(),
            "the title-token plan must admit docs for vector ranking"
        );
    }

    /// Post-drain filtered search must fan out on the hidden index (same as
    /// the bench), not the user table. Predicate still resolves on user FTS.
    #[test]
    fn filtered_vector_search_post_drain_uses_hidden_index() {
        use crate::superfile::vector::rerank_codec::RerankCodec;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let opts = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let opts = opts.with_storage(storage);
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 32, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
        st.drain_vectors_to_cells_sync().expect("drain");

        let reader = st.reader().expect("reader");
        let user_uris: HashSet<_> = reader.manifest().superfiles.iter().map(|e| e.uri).collect();
        let hidden = reader
            .vector_index_table()
            .expect("hidden index must exist");
        let hidden_uris: HashSet<_> = hidden
            .reader()
            .expect("reader")
            .manifest()
            .superfiles
            .iter()
            .map(|e| e.uri)
            .collect();
        assert!(
            !hidden_uris.is_empty(),
            "drain must publish at least one hidden superfile"
        );

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let hits = reader
            .vector_hits(
                "emb",
                &q,
                5,
                VectorSearchOptions::new(),
                Some(VectorFilter {
                    column: "title",
                    query: "doc",
                    mode: crate::superfile::fts::reader::BoolMode::Or,
                }),
            )
            .expect("filtered vector_hits");
        assert!(!hits.is_empty(), "filtered search must return hits");
        for hit in &hits {
            assert!(
                hidden_uris.contains(&hit.superfile),
                "post-drain filtered hits must come from hidden superfiles, got {:?} \
                 (user={user_uris:?}, hidden={hidden_uris:?})",
                hit.superfile
            );
            assert!(
                !user_uris.contains(&hit.superfile),
                "post-drain filtered hits must not come from user superfiles"
            );
        }

        let mapping_error =
            block_on(reader.prepare_vector_stable_allow_async(Arc::new(vec![i128::MAX])))
                .err()
                .expect("unknown drained id must fail hidden mapping");
        assert!(
            mapping_error
                .to_string()
                .contains("did not map to any hidden superfile"),
            "unexpected mapping error: {mapping_error}"
        );
    }

    /// A post-drain vector search resolves hidden hits back to user `_id`s and
    /// subtracts tombstones: after deleting a doc that the query would return,
    /// it must drop out of the result set. Exercises the hidden-hit id
    /// resolution and tombstone-subtraction paths on the plain (unfiltered)
    /// query.
    #[test]
    fn vector_search_post_drain_excludes_deleted() {
        use datafusion::prelude::{col, lit};

        use crate::superfile::vector::rerank_codec::RerankCodec;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let opts = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, 32, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
        drop(w); // release the writer so the later delete can acquire one
        st.drain_vectors_to_cells_sync().expect("drain");

        // Query the one-hot e_0; doc 0 is an exact match, so it is returned.
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let hits_before = st
            .reader()
            .expect("reader")
            .vector_hits("emb", &q, 32, VectorSearchOptions::new(), None)
            .expect("pre-delete search");
        assert!(!hits_before.is_empty(), "docs retrievable pre-delete");

        // Delete that exact match; the query must subtract its tombstone.
        let stats = st.delete(col("title").eq(lit("doc 0"))).expect("delete");
        assert_eq!(stats.n_tombstoned(), 1, "exactly one row tombstoned");

        let hits_after = st
            .reader()
            .expect("reader")
            .vector_hits("emb", &q, 32, VectorSearchOptions::new(), None)
            .expect("post-delete search");
        assert_eq!(
            hits_after.len(),
            hits_before.len() - 1,
            "the deleted doc must drop out of the results"
        );
    }

    /// Commit writes user superfiles in the cell-packed (MultiCellIvf) layout,
    /// and boundary replicas are vector-only stubs: every ingested row is a
    /// Parquet primary exactly once, so the total Parquet row count across the
    /// user superfiles equals the number of ingested rows — no duplicate SQL
    /// rows even when boundary replication adds neighbor-cell postings.
    #[test]
    fn commit_user_superfiles_cell_packed_no_duplicate_parquet_rows() {
        use crate::superfile::vector::layout::VectorLayout;

        let dim = 16;
        let st = Supertable::create(options_one_superfile_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        let n = 200usize;
        w.append(&build_vector_batch(0, n, dim, schema))
            .expect("append");
        w.commit().expect("commit");

        let r = st.reader().expect("reader");
        let manifest = r.manifest();
        assert!(
            !manifest.superfiles.is_empty(),
            "commit must publish user superfiles"
        );
        let mut total_primary_rows = 0u64;
        for entry in manifest.superfiles.iter() {
            assert_eq!(
                entry.vector_layout,
                VectorLayout::MultiCellIvf,
                "commit must write cell-packed MultiCellIvf user superfiles, got {:?}",
                entry.vector_layout
            );
            total_primary_rows += entry.n_docs;
        }
        assert_eq!(
            total_primary_rows, n as u64,
            "each ingested row is a Parquet primary exactly once; boundary stubs \
             must not add Parquet rows (got {total_primary_rows}, expected {n})"
        );
    }

    /// A search over cell-packed user superfiles returns distinct rows and
    /// resolves their scalar columns, even though a row's vector can be found
    /// via both its primary cell and a boundary stub in a neighbor cell: the
    /// stub carries the primary's real `_id`, so dedup collapses the pair and
    /// scalar resolve maps back to the one row that owns the Parquet data.
    #[test]
    fn vector_search_dedups_and_resolves_with_stub_boundaries() {
        use arrow_array::Decimal128Array;

        use crate::superfile::vector::rerank_codec::RerankCodec;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let opts = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        let mut w = st.writer().expect("writer");
        let n = 200usize;
        w.append(&build_vector_batch(0, n, dim, schema))
            .expect("append");
        w.commit().expect("commit");

        let r = st.reader().expect("reader");
        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let k = 20usize;
        let batches = r
            .vector_search(
                "emb",
                &q,
                k,
                VectorSearchOptions::new().with_nprobe(4),
                None,
                Some(&["_id", "title"]),
            )
            .expect("vector_search");

        let mut seen: HashSet<i128> = HashSet::new();
        let mut total = 0usize;
        for b in &batches {
            let ids = b
                .column(0)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("_id column is Decimal128");
            let titles = b
                .column(1)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("title column is LargeString");
            assert_eq!(titles.len(), ids.len());
            for i in 0..ids.len() {
                total += 1;
                assert!(!titles.value(i).is_empty());
                assert!(
                    seen.insert(ids.value(i)),
                    "duplicate _id {} in results — a boundary stub was not deduped \
                     against its primary",
                    ids.value(i)
                );
            }
        }
        assert_eq!(total, k, "search must return k distinct rows, got {total}");
    }

    /// The inline stable-id region on cell-packed USER superfiles must answer
    /// parquet-local lookups with exactly the `_id` column's values — the
    /// contract `stable_ids_for_tagged_hits` (FTS/SQL post-top-k id stamping)
    /// relies on. Boundary stubs add neighbor-cell postings; if a shard's
    /// per-cell doc counts or region layout counted those stubs, the
    /// `file_local_to_cell` prefix sums would silently pair hits with the
    /// wrong rows' ids.
    #[test]
    fn user_multicell_inline_ids_match_parquet_id_column() {
        use arrow_array::Decimal128Array;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(
            options_one_superfile_per_commit(dim).with_storage(Arc::clone(&storage)),
        )
        .expect("create");
        let mut w = st.writer().expect("writer");
        let n = 200usize;
        w.append(&build_vector_batch(0, n, dim, schema))
            .expect("append");
        w.commit().expect("commit");

        let r = st.reader().expect("reader");
        let manifest = r.manifest();
        let mut checked_files = 0usize;
        for entry in manifest.superfiles.iter() {
            let reader = manifest
                .options
                .store
                .reader(&entry.uri)
                .expect("writer-published reader");
            let vec_reader = reader.vec().expect("vector reader");
            let locals: Vec<u32> = (0..entry.n_docs as u32).collect();
            // Ground truth: the parquet `_id` column at those rows.
            let batch = reader
                .take_by_local_doc_ids(&locals, &[reader.id_column()])
                .expect("take _id column");
            let truth = batch
                .column(0)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("_id is Decimal128");
            let Some(inline) = vec_reader.inline_stable_ids_for_locals(&locals) else {
                panic!(
                    "inline stable-id lookup unavailable on user superfile {:?} \
                     (layout {:?}): stable_ids_for_tagged_hits would silently fall \
                     back to the _id page read",
                    entry.uri, entry.vector_layout
                );
            };
            for (i, &local) in locals.iter().enumerate() {
                assert_eq!(
                    inline[i],
                    truth.value(i),
                    "inline stable-id for parquet-local {local} in {:?} diverges \
                     from the _id column",
                    entry.uri
                );
            }
            checked_files += 1;
        }
        assert!(checked_files > 0, "commit published no user superfiles");
    }

    #[test]
    fn global_union_includes_undrained_user_delta() {
        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(
            options_one_superfile_per_commit(dim).with_storage(Arc::clone(&storage)),
        )
        .expect("create");

        let mut writer = st.writer().expect("writer");
        writer
            .append(&build_vector_batch(0, 8, dim, Arc::clone(&schema)))
            .expect("append base");
        writer.commit().expect("commit base");
        drop(writer);
        st.drain_vectors_to_cells_sync().expect("drain base");

        let mut writer = st.writer().expect("writer delta");
        writer
            .append(&build_vector_batch(15, 1, dim, schema))
            .expect("append delta");
        writer.commit().expect("commit delta");
        drop(writer);

        let reader = st.reader().expect("reader");
        let hidden = reader.vector_index_table().expect("hidden index");
        let drained = hidden
            .reader()
            .expect("reader")
            .manifest()
            .get_drained_ranges();
        let undrained: Vec<_> = reader
            .manifest()
            .superfiles
            .iter()
            .filter(|entry| !drained.contains(entry.birth_version))
            .collect();
        assert_eq!(undrained.len(), 1);

        let mut query = vec![0.0f32; dim];
        query[15] = 1.0;
        let hits = reader
            .vector_hits("emb", &query, 1, VectorSearchOptions::new(), None)
            .expect("global union search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].superfile, undrained[0].uri);
    }
}
