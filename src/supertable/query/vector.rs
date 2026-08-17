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
    borrow::Cow,
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet},
    future::Future,
    mem,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{self, AtomicU64},
    },
    time::Instant,
};

use arrow::record_batch::RecordBatch;
use arrow_array::{Array, Decimal128Array};
use arrow_schema::Schema;
use futures::future::try_join_all;
use roaring::RoaringBitmap;
use tokio::{join, sync::OnceCell};
use uuid::Uuid;

use super::{
    SuperfileHit,
    candidate::CandidatePlan,
    dispatch,
    exec::common::{SCORE_COLUMN, id_score_batch, resolve_hits_named, take_rows_byte_source},
    prune::{PruneLeaf, select_superfiles},
};
pub use crate::superfile::reader::VectorSearchOptions;
#[cfg(feature = "test-helpers")]
use crate::test_helpers::{admit_trace, served_shortlist_probe};
use crate::{
    config,
    runtime_bridge::run_on_pool,
    runtime_metrics::op_stats::{self, OpStatsCollector},
    storage::io_counters,
    superfile::{
        SuperfileReader,
        error::ReadError,
        fts::reader::BoolMode,
        vector::{
            distance::{Metric, distance, normalize, relative_score_window},
            hnsw::{self, HnswParams, Sq16Scorer, encode_hnsw},
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
        options::{GappedPlacementCell, GappedPlacementIndex},
        slow_vector_state::{
            CentroidSection, ResidentGraphSections, fetch_centroid_section, fetch_graph_sections,
        },
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
    extension_depth: Option<(&HashSet<u32>, usize)>,
) -> Vec<(usize, u32, f32)> {
    // Fine runs kept per group scale with the group's size: a fraction of its
    // fine-cluster count, never below the floor. So large cells (more fine
    // clusters) are probed proportionally deeper while small cells hold at the
    // floor.
    //
    // `extension_depth`: #515 serve-window extension cells and their bounded
    // floor. On the law-pinned arm `keep_floor` is MAX (a probed law cell is
    // read in full — that coverage is what the law certified), but cells the
    // serve window added past the law's width are evidence picks, not law
    // picks: they keep the bounded pre-pin depth so extending the served set
    // never buys whole-cell reads. Only the per-`(cell, fragment)` arm
    // consults it — the wave-pooled arm's depth is already bounded per wave.
    let fine_keep = |available: usize, floor: usize| -> usize {
        ((keep_pct * available as f64).floor() as usize)
            .max(floor)
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
            let keep = fine_keep(fine.len(), keep_floor);
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
        let cell_floor = match extension_depth {
            Some((extension, depth)) if extension.contains(&cell) => depth,
            _ => keep_floor,
        };
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
            let keep = fine_keep(fine.len(), cell_floor);
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
/// write-side
/// [`crate::supertable::manifest::RABITQ_ADMIT_CELL_SHORTLIST_FRACTION`]
/// slice of the ranked population, floored by
/// [`RABITQ_ADMIT_CELL_SHORTLIST_MIN`] — the same window pre-#515
/// serving always used. This is only round 0 of admission: on the
/// law-served default path the #515 self-measured admit loop
/// ([`admit_extension_round`]) extends it from the query's own
/// evidence, with no query-side fraction of its own.
fn admit_shortlist_window(ranked_cells: usize) -> usize {
    let scaled = (ranked_cells as f64 * RABITQ_ADMIT_CELL_SHORTLIST_FRACTION).ceil() as usize;
    scaled.max(RABITQ_ADMIT_CELL_SHORTLIST_MIN)
}

/// One round of the #515 self-measured admit extension: which
/// not-yet-admitted cells could plausibly hold an exact fine score
/// inside the serve window, judged by their 1-bit estimate shifted by
/// the most favorable estimate-to-exact residual OBSERVED ON THIS
/// QUERY. The shift is an empirical extrapolation, not a proven
/// over-admit bound: the residual is SIGNED and measured only on the
/// already-scored (best-estimate) cells, so a distant truth cell whose
/// estimate is unusually pessimistic can sit outside the extension —
/// the measured under-admit tail (the #515 diag reads 99% full admit
/// coverage, not 100%). What does hold: the stamped width stays a
/// served floor, so admission is expected/measured ≥ the stamped-width
/// baseline, never below it. Decisive geometry: estimates cliff,
/// nothing qualifies, admission stays at the write window. Flat-scored
/// queries (question-vs-passage retrieval): the tie run qualifies and
/// admission follows the evidence. The caller iterates — newly
/// admitted cells' exact scores tighten/widen the measurement — until
/// a round admits nothing; each round admits ≥ 1 new cell, so
/// termination is bounded by the populated-cell count.
fn admit_extension_round(
    admit_ranking: &[(u32, f32)],
    admitted: &HashSet<u32>,
    exact_best_by_cell: &HashMap<u32, f32>,
    serve_threshold: f32,
) -> Vec<u32> {
    let mut residual_floor = f32::INFINITY;
    for (cell, estimate) in admit_ranking {
        if let Some(exact) = exact_best_by_cell.get(cell) {
            residual_floor = residual_floor.min(exact - estimate);
        }
    }
    if !residual_floor.is_finite() {
        return Vec::new();
    }
    admit_ranking
        .iter()
        .filter(|(cell, _)| !admitted.contains(cell))
        .filter(|(_, estimate)| estimate + residual_floor <= serve_threshold)
        .map(|(cell, _)| *cell)
        .collect()
}

/// Cell selection for the law-served DEFAULT arm (#515): the stamped
/// width is a SERVED FLOOR — the drain certified that many cells are
/// needed for its coverage target, and serving fewer would break the
/// law contract (planted-neighbor tests pin exactly the case where
/// truth spans cells whose fine scores are not tied). Past the floor,
/// selection follows the exact-fine ranking while the query's own
/// scores stay inside the serve window; those extension cells are read
/// at the bounded pre-pin depth (cells inside the floor keep
/// whole-cell depth — that is what the walk certified). Returns the
/// served cells in probe order (grid picks first, then fine) plus the
/// extension set.
fn law_floor_serve_selection(
    fine_ranked: &[(u32, f32)],
    grid_cells: &[u32],
    fine_base: usize,
    serve_threshold: f32,
) -> (Vec<u32>, HashSet<u32>) {
    let fine_cells: Vec<u32> = fine_ranked
        .iter()
        .enumerate()
        .take_while(|(rank, (_, score))| *rank < fine_base || *score <= serve_threshold)
        .map(|(_, (cell, _))| *cell)
        .collect();
    let extension: HashSet<u32> = fine_cells
        .iter()
        .skip(fine_base)
        .copied()
        .filter(|cell| !grid_cells.contains(cell))
        .collect();
    (union_cell_selection(grid_cells, &fine_cells), extension)
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

/// Rank every eligible tagged cell by its best ESTIMATED fine-centroid
/// score (1-bit XOR+popcount over the summary codes), ascending —
/// better first, ties on lower cell id. The estimates never leave the
/// admit stage: they bound which cells get exact-scored, and (#515)
/// they carry the self-measured admit extension — the caller compares
/// them against exact scores it already computed to measure this
/// table's estimate-to-exact residual per query.
fn estimate_admit_ranking(
    superfiles: &[Arc<SuperfileEntry>],
    column: &str,
    query_len: usize,
    metric: Metric,
    admit_q: &RabitqAdmitQuery,
    allow: Option<&HashMap<SuperfileUri, Arc<RoaringBitmap>>>,
    superseded: &BTreeMap<Uuid, BTreeSet<u32>>,
) -> Result<Vec<(u32, f32)>, QueryError> {
    let eligible = |entry: &Arc<SuperfileEntry>| allow.is_none_or(|m| m.contains_key(&entry.uri));
    let is_superseded = |entry: &Arc<SuperfileEntry>, cell_id: u32| {
        superseded
            .get(&entry.superfile_id)
            .is_some_and(|s| s.contains(&cell_id))
    };
    let mut cell_best: HashMap<u32, f32> = HashMap::new();
    for entry in superfiles.iter().filter(|e| eligible(e)) {
        let vs = eligible_summary(entry, column, query_len)?;
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
    Ok(ranked)
}

/// Score fine IVF centroids in the eligible superfile summaries.
///
/// With `admit` set (an explicit admitted-cell set), the exact fp32
/// scan skips instances outside that set; the caller owns admission
/// (write-window slice of [`estimate_admit_ranking`] plus grid picks,
/// extended by the #515 self-measured admit loop). Every *emitted*
/// score is exact fp32, so routing and near-tie logic never see 1-bit
/// noise. Untagged (`cell_id: None`) summaries are exact-scored when
/// `include_untagged` is set — round 0 only, so admit-extension rounds
/// never re-emit them.
fn score_fine_candidates(
    superfiles: &[Arc<SuperfileEntry>],
    column: &str,
    query: &[f32],
    metric: Metric,
    admit: Option<&HashSet<u32>>,
    include_untagged: bool,
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
    let shortlist: Option<&HashSet<u32>> = admit;

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
                || (cell.cell_id.is_none() && !include_untagged)
                || shortlist
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

/// Widen a grid cell cutoff so the probed cells' indexed row counts cover at
/// least `k`. `grid_cell_cutoff`'s slack window stops at the nearest near-tie
/// cluster — a single cell for a query blended between two clusters — so a
/// top-k request that spans cells could otherwise probe one undersized cell and
/// return fewer than `k` rows. Walks the cell ranking past `cutoff`, adding each
/// cell's indexed row count (from `postings_by_cell`) until coverage reaches `k`
/// or the ranking is exhausted. A no-op once the already-probed cells hold `k`
/// (the common single-cluster case, and every cell large relative to `k` at
/// scale), so a sufficient query is never widened.
fn cover_k_cell_cutoff(
    cutoff: usize,
    ranked: &[(u32, f32)],
    postings_by_cell: &HashMap<u32, u64>,
    k: usize,
) -> usize {
    let cell_rows = |cell: u32| postings_by_cell.get(&cell).copied().unwrap_or(0);
    let mut covered: u64 = ranked[..cutoff].iter().map(|(c, _)| cell_rows(*c)).sum();
    let mut cut = cutoff;
    while cut < ranked.len() && covered < k as u64 {
        covered += cell_rows(ranked[cut].0);
        cut += 1;
    }
    cut
}

// Serve-time near-tie window on the exact-fine cell ranking (#515),
// law-pinned default path only. Past the law's width, selection keeps
// following the ranking while each next cell's best exact fine score
// stays within this relative window of the winner's. On decisive
// geometry (synthetic, Cohere) fine scores cliff after the law's picks
// and the window never opens — serving is byte-identical to the pin.
// On flat-scored corpora (question-vs-passage retrieval: BioASQ
// measured its truth cells at exact-fine ranks ≤ 48, already inside
// the exactly-rescored admit window) the query's own evidence extends
// the served set. Bounds are the evidence itself and the admit window
// (only admitted cells are ranked) — deliberately no artificial cap: a
// constant ceiling would silently truncate exactly the tail queries
// this serves. Cost stays bounded per cell instead: extension cells
// are read at the pre-pin fine depth (the stamped law's runs-per-cell
// floor), never the pin's whole-cell depth. The window value comes
// from `config.yaml` (`vector.serve_near_tie_slack`, default = the
// measured real-query truth-cell slack p99, 0.287–0.294 across the
// #515 BioASQ diag configs; the dial grid read 0.9617 at 0.25 vs
// 0.9657 at 0.30 on the diag metric, Cohere controls unchanged at
// both) — a config default rather than a per-table drain stamp
// because the drain sample is corpus rows, which measure a different
// slack distribution than real queries (the same mismatch that
// under-stamped the width law on question-vs-passage corpora).

/// Default-path cell selection, shared by the hidden (post-drain) and user
/// (pre-drain) branches: probe the fine-ranked top cell, adding the grid's
/// top cell only when its own fine score is a genuine near-tie of the fine
/// winner (same relative window replica closure uses at drain time, so
/// probing and replication agree on what counts as a boundary). At the
/// shipped grid shapes (256/1024 cells) fine p1 coverage measures 1.000
/// (drain-diag, 1M–100M) on row-like queries. The #515 serve window
/// deliberately does NOT apply here: this arm gates wave-pooled, where
/// widening the cell sweep alone cannot deepen the read, so following
/// the ranking would add cells without adding recall.
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
    op_stats: &Option<Arc<OpStatsCollector>>,
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
    // Uris in the current fresh read list, to evict cache entries for superfiles
    // that have since been superseded/GC'd (keeps the cache bounded to the live
    // set, no process-lifetime growth).
    let mut live_uris: HashSet<SuperfileUri> = HashSet::new();

    for entry in entries {
        live_uris.insert(entry.uri);
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

    // Resolve ids still unplaced (they live in GAPPED superfiles) via a
    // read-time cache: one immutable sorted `stable_id -> local` index per
    // gapped superfile, keyed by (immutable) uri. Contiguous entries were
    // placed by arithmetic above and are never cached, so an all-contiguous
    // table caches nothing. A data change only builds the NEW superfiles;
    // unchanged ones stay cached and are never stale (#556).
    if !gapped.is_empty() {
        let slot = Arc::clone(&manifest.options.gapped_id_placement_cache);
        let version = manifest.get_manifest_id();
        // Under one short lock: prune MONOTONICALLY (only a generation at least
        // as new as the last prune may evict, so an older concurrent snapshot
        // never drops a newer snapshot's freshly-built maps — which would thrash
        // rebuilds under load), then take/create the per-uri single-flight cell
        // for each gapped superfile. Holding the `Arc<OnceCell>` keeps a
        // concurrent prune from dropping the cell out from under the build.
        let cells: Vec<(Arc<SuperfileEntry>, GappedPlacementCell)> = {
            let mut cache = slot.lock().await;
            // `>` not `>=`: a generation equal to the last prune has already
            // been pruned, and re-running `retain` for it walks the whole map
            // under this lock on EVERY projected query of a steady-state table
            // (the common case — the generation only moves on commit/compact).
            // The monotonic guarantee is unchanged: an older snapshot still
            // never evicts a newer one's freshly built maps.
            if version > cache.pruned_through {
                cache.entries.retain(|uri, _| live_uris.contains(uri));
                cache.pruned_through = version;
            }
            gapped
                .iter()
                .map(|e| {
                    let cell = Arc::clone(
                        cache
                            .entries
                            .entry(e.uri)
                            .or_insert_with(|| Arc::new(OnceCell::new())),
                    );
                    (Arc::clone(e), cell)
                })
                .collect()
        };
        // Build outside the lock. Each cell single-flights its superfile's index
        // (a second concurrent cold query for the same uri awaits this build,
        // not a duplicate one); the CPU inversion runs on the reader pool, not
        // the tokio blocking pool. `try_join_all` preserves `cells` order, so
        // the probe below stays deterministic in manifest order.
        let built: Vec<(Arc<SuperfileEntry>, Arc<GappedPlacementIndex>)> =
            try_join_all(cells.into_iter().map(|(entry, cell)| async move {
                let index = cell
                    .get_or_try_init(|| {
                        build_gapped_placement_index(manifest, &entry, id_column, op_stats)
                    })
                    .await?;
                Ok::<_, QueryError>((entry, Arc::clone(index)))
            }))
            .await?;
        // Probe in manifest (gapped) order: if an id is resident in two live
        // superfiles — an updated row's tombstoned-old copy plus its live copy,
        // both physical rows to `read_ids_for_locals` — first-wins resolves it to
        // the same superfile the pre-cache full scan did.
        for (entry, index) in &built {
            for (i, &id) in user_row_ids.iter().enumerate() {
                if placements[i].is_none()
                    && id >= entry.id_min
                    && id <= entry.id_max
                    && let Some(local) = index.local_for(id)
                {
                    placements[i] = Some((Arc::clone(entry), local));
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

/// Build one gapped superfile's sorted `stable_id -> local` index (#556): read
/// its `_id` column once, then sort + dedup on the reader pool. Memoized per
/// uri by the caller's single-flight cell.
///
/// The index's RESIDENT bytes are deliberately not charged to the connection
/// memory budget, only its transient build scratch is. The budget gates
/// MANDATORY work — ingest reserves through `reserve_build_scratch`
/// (`writer.rs`) and compaction through `merge_superfiles`
/// (`compaction/mod.rs`), and both HARD-FAIL when refused. A discretionary,
/// rebuildable read cache that pins budget bytes for as long as its superfile
/// stays live can push those two over the ceiling and keep them there: the only
/// thing that evicts an entry is its superfile being superseded, which is
/// exactly what compaction does, which is now the operation being denied. That
/// deadlock does not clear without a process restart, so the cache stays out of
/// the accounted set. Resident cost is bounded by the live gapped corpus
/// (~20 B/row) — the same proportional-to-corpus class as the transposed-code
/// cache — and every entry is reconstructible from immutable bytes.
async fn build_gapped_placement_index(
    manifest: &ManifestSnapshot,
    entry: &SuperfileEntry,
    id_column: &str,
    op_stats: &Option<Arc<OpStatsCollector>>,
) -> Result<Arc<GappedPlacementIndex>, QueryError> {
    let locals = Arc::new((0..entry.n_docs as u32).collect::<Vec<u32>>());
    let ids = read_ids_for_locals(manifest, entry, &locals, id_column, true, op_stats).await?;
    // Build scratch only, released the moment the sorted arrays are extracted.
    // The transient peak holds all three vectors at once — the decoded `ids`
    // and `locals` are still alive while the `(i128, u32)` pairs they zip into
    // are built — so account for the sum rather than the pair vector alone
    // (which undercounts the real high-water mark by ~2.6x). Refusal degrades
    // to an unaccounted build rather than failing the query — the vector scan
    // degrades the same way.
    let scratch_bytes = ids.len()
        * (std::mem::size_of::<i128>()
            + std::mem::size_of::<u32>()
            + std::mem::size_of::<(i128, u32)>());
    let scratch = manifest
        .options
        .connection_memory_budget
        .try_reserve(scratch_bytes)
        .ok();
    let pool = Arc::clone(&manifest.options.reader_pool);
    let index = run_on_pool(
        Some(&pool),
        "gapped id-map: reader pool dropped result",
        move || {
            let mut pairs: Vec<(i128, u32)> = ids.into_iter().zip(locals.iter().copied()).collect();
            // Sort by id, ties by local, so dedup keeps the SMALLEST local for a
            // repeated id — first-wins in Parquet row order, matching the old scan.
            pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            pairs.dedup_by_key(|p| p.0);
            let mut sorted_ids = Vec::with_capacity(pairs.len());
            let mut sorted_locals = Vec::with_capacity(pairs.len());
            for (id, local) in pairs {
                sorted_ids.push(id);
                sorted_locals.push(local);
            }
            // Scratch is dead here; release before the index outlives it.
            drop(scratch);
            GappedPlacementIndex::new(sorted_ids, sorted_locals)
        },
    )
    .await
    .map_err(|e| QueryError::Execute(e.to_string()))?;
    Ok(Arc::new(index))
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
    op_stats: &Option<Arc<OpStatsCollector>>,
) -> Result<Vec<i128>, QueryError> {
    if row_id_from_manifest_entry(entry, 0).is_some() {
        return Ok((0..entry.n_docs as u32)
            .map(|local| entry.id_min + i128::from(local))
            .collect());
    }
    let locals = Arc::new((0..reader.n_docs() as u32).collect::<Vec<u32>>());
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
    read_ids_for_locals(manifest, entry, &locals, id_column, true, op_stats).await
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
    local_ids: &Arc<Vec<u32>>,
    id_column: &str,
    allow_inline_region: bool,
    op_stats: &Option<Arc<OpStatsCollector>>,
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
        // The inline-region read is one planned range per file — the plan
        // requests it identically whether it resolves resident or cold.
        // Hidden cell superfiles inline the stable `_id` in the IVF blob — resolve
        // straight from it (resident; no scalar `_id` column read) when available.
        // Resident decode is CPU over the whole requested row set — the
        // placement-index build asks for EVERY row of the superfile — so it
        // rides the reader pool and this task awaits a oneshot, per the
        // rayon-for-CPU / tokio-for-I/O contract. Running it inline blocks a
        // tokio worker for the length of a full-column decode, stalling every
        // other query's I/O on that worker.
        let resident = {
            let reader = Arc::clone(&reader);
            // Arc clone, not a buffer copy: the placement-index build
            // passes EVERY row of the superfile here, so copying to
            // satisfy the 'static bound would double a 10M-row locals
            // list (~40 MB) per build.
            let locals = Arc::clone(local_ids);
            run_on_pool(
                Some(&manifest.options.reader_pool),
                "inline stable-id decode: reader pool dropped result",
                move || {
                    reader
                        .vec()
                        .and_then(|v| v.inline_stable_ids_for_locals(&locals))
                },
            )
            .await
            .map_err(|e| QueryError::Execute(e.to_string()))?
        };
        if let Some(ids) = resident {
            if let Some(stats) = op_stats {
                stats.add_planned_read_ranges(1);
            }
            return Ok(ids);
        }
        // Cold path: fetch the inline region async when present but not resident.
        if let Some(v) = reader.vec()
            && let Some(ids) = v
                .inline_stable_ids_for_locals_async(local_ids)
                .await
                .map_err(|e| QueryError::Execute(e.to_string()))?
        {
            if let Some(stats) = op_stats {
                stats.add_planned_read_ranges(1);
            }
            return Ok(ids);
        }
    }
    if reader.parquet_bytes().is_some() {
        // Same bridge as the inline path: a Parquet column take over the
        // requested rows is CPU, not I/O.
        let (batch, decode_ns) = {
            let reader = Arc::clone(&reader);
            let locals = Arc::clone(local_ids);
            let id_column = id_column.to_string();
            run_on_pool(
                Some(&manifest.options.reader_pool),
                "scalar id decode: reader pool dropped result",
                move || {
                    op_stats::timed_section(|| {
                        reader
                            .take_by_local_doc_ids(&locals, &[id_column.as_str()])
                            .map_err(|e| QueryError::Execute(e.to_string()))
                    })
                },
            )
            .await
            .map_err(|e| QueryError::Execute(e.to_string()))?
        };
        if let Some(stats) = op_stats {
            stats.add_kernel_cpu_ns(decode_ns);
        }
        return id_values_from_batch(&batch?);
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
    op_stats: &Option<Arc<OpStatsCollector>>,
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
        let locals = Arc::new(
            idxs.iter()
                .map(|&i| hidden_hits[i].local_doc_id)
                .collect::<Vec<u32>>(),
        );
        let vals = read_ids_for_locals(hidden_manifest, &entry, &locals, id_column, true, op_stats)
            .await?;
        for (j, &i) in idxs.iter().enumerate() {
            ids[i] = vals[j];
        }
    }
    Ok(ids)
}

/// Column positions in the batch [`hits_id_score_batch`] synthesizes.
const ID_SCORE_ID_COLUMN: usize = 0;
const ID_SCORE_SCORE_COLUMN: usize = 1;

/// Position of `name` in the batch [`hits_id_score_batch`] synthesizes,
/// or `None` when it is a user column that needs a data page.
///
/// THE rule for which columns come free with the search wave: the ids
/// are stamped on the hits and the scores are the kernel's output, so
/// any subset or ordering of the two serves without resolving
/// placements. Both entry points classify through this one function —
/// the public API by name directly, the SQL TVF by mapping its
/// DataFusion column indices back to names — so a third free column
/// cannot be added to one path and forgotten on the other.
pub(crate) fn free_column_slot(name: &str, id_column: &str) -> Option<usize> {
    // An id column literally named `score` makes the two free names one
    // name: the match below would answer `score` with the id slot. The
    // general path resolves it the same way (`index_of` takes the first
    // field, which is the id), so this changes no result today — it
    // stops the two paths from drifting if either resolution order
    // changes, and keeps the classifier from silently answering an
    // ambiguous request.
    if id_column == SCORE_COLUMN {
        return None;
    }
    match name {
        n if n == id_column => Some(ID_SCORE_ID_COLUMN),
        n if n == SCORE_COLUMN => Some(ID_SCORE_SCORE_COLUMN),
        _ => None,
    }
}

/// Whether `_id` and `score` unambiguously name the free columns for
/// this table — i.e. no user column shadows either.
///
/// Arrow permits duplicate field names and nothing rejects a user
/// column called `score` (or one matching the id column) at create
/// time. When one exists, `output_schema_with_score` carries the name
/// twice and the general path's `index_of` resolves it to the USER
/// column, decoding real data. The fast path would answer the same
/// projection with the synthesized value instead — a different result,
/// not just a different route — so it must decline. Checked against
/// the user-declared schema, which is a handful of fields and needs no
/// allocation (building the scalar schema per query would cost the
/// fast path the very work it exists to skip).
pub(crate) fn free_columns_unambiguous(user_schema: &Schema, id_column: &str) -> bool {
    id_column != SCORE_COLUMN
        && !user_schema
            .fields()
            .iter()
            .any(|f| f.name() == SCORE_COLUMN || f.name() == id_column)
}

/// Public-API projection policy: positions into the synthesized batch,
/// or `None` when the projection needs the general path.
///
/// `None` (engine-native) is the `_id` + `score` pair. An EMPTY
/// projection takes the general path: a zero-column batch carries its
/// row count differently, and reproducing that here would be a second
/// contract to maintain for a degenerate input. (The SQL TVF keeps its
/// own policy for empty — DataFusion emits one for `COUNT(*)`-shaped
/// plans, where a zero-column batch is exactly what it wants.)
pub(crate) fn id_score_projection_indices(
    projection: Option<&[&str]>,
    id_column: &str,
) -> Option<Vec<usize>> {
    let Some(names) = projection else {
        return Some(vec![ID_SCORE_ID_COLUMN, ID_SCORE_SCORE_COLUMN]);
    };
    if names.is_empty() {
        return None;
    }
    names
        .iter()
        .map(|name| free_column_slot(name, id_column))
        .collect()
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
    let hidden_manifest = user_reader.vector_index_table().map(|vit| {
        Arc::clone(
            vit.pinned_reader_with(user_reader.op_stats.clone())
                .manifest(),
        )
    });
    let deleted = user_reader.vector_index_table().and_then(|vit| {
        vit.pinned_reader_with(user_reader.op_stats.clone())
            .hidden_deleted_ids()
            .ok()
    });
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
            hidden_hits_user_ids(
                hm,
                std::slice::from_ref(hit),
                id_column,
                &user_reader.op_stats,
            )
            .await?[0]
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
    let placements =
        lookup_user_placements_by_id(user_manifest, &requested_ids, &user_reader.op_stats).await?;
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

/// Warn that `search_mode = hnsw_ivf` is set but this query fell through to the
/// ivf scan, with the diagnosis the situation actually warrants:
///   - `has_graph_ref = false`: the table carries no resident graph at all —
///     pre-drain, or the corpus exceeds `hnsw_max_docs`.
///   - `has_graph_ref = true`: a graph exists but not for THIS column (a
///     second same-dim vector column carries it, or a dim/emptiness mismatch).
///
/// The `_id` results are still correct; only the fast path is off. Deduped
/// per `(diagnosis, column)` so a miss on one column never suppresses the
/// warning another column would legitimately raise.
fn warn_hnsw_no_resident_graph(column: &str, has_graph_ref: bool) {
    static WARNED: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashSet<(bool, String)>>,
    > = std::sync::OnceLock::new();
    let warned = WARNED.get_or_init(Default::default);
    {
        let mut set = warned.lock().expect("hnsw warn dedup set");
        if !set.insert((has_graph_ref, column.to_string())) {
            return;
        }
    }
    if has_graph_ref {
        tracing::warn!(
            column,
            "search_mode=hnsw_ivf but no resident graph for this column (another column \
             carries the graph, or a dim/emptiness mismatch); serving via ivf scan"
        );
    } else {
        tracing::warn!(
            column,
            "search_mode=hnsw_ivf but no resident graph (pre-drain or corpus > hnsw_max_docs); \
             serving via ivf scan"
        );
    }
}

/// Assemble the persisted `hnsw` data bundle at drain time: walk
/// every superfile's materialized Sq16 rows for `column`, pool the
/// node-ordered code plane and the stable-doc-id map, build the HNSW over
/// the codes, and return the encoded bundle bytes. `Ok(None)` when the
/// column is absent, not Sq16, or empty; `Err` only on a genuine read
/// fault (the drain treats that as "skip the graph", never fatal).
/// Held-out query count for calibration recall measurement.
const HNSW_CALIB_QUERIES: usize = 200;
/// The `k` calibration and the incremental recall re-check measure at (the
/// engine's recall@10 acceptance anchor).
const HNSW_CALIB_RECALL_K: usize = 10;
/// Deterministic calibration seed (no wall-clock / system randomness).
const HNSW_CALIB_SEED: u64 = 0x_C0FF_EE00_CA11_B000;

/// Gather the Sq16 code plane for `column` across all superfiles, in manifest
/// order (so a full collection aligns positionally with the `doc_ids` gathered
/// in the same order). `stride_step = Some(k)` keeps only every k-th row — a
/// coarse, deterministic, evenly-spread sample that lets the probe bound its
/// memory to ~`n/k` rows instead of the whole plane; `None` collects every row.
async fn collect_hnsw_codes(
    manifest: &ManifestSnapshot,
    column: &str,
    stride: usize,
    stride_step: Option<usize>,
) -> Result<Vec<u8>, QueryError> {
    let store = Arc::clone(&manifest.options.store);
    let disk_cache = manifest.options.disk_cache.clone();
    let storage = manifest.options.storage.clone();
    let empty_superseded = BTreeMap::new();
    let superseded = manifest.get_superseded_cells().unwrap_or(&empty_superseded);
    let mut codes: Vec<u8> = Vec::new();
    let mut gi: usize = 0;
    for entry in manifest.get_all_superfiles() {
        let reader =
            dispatch::open_reader(&store, disk_cache.as_ref(), storage.as_ref(), entry, false)
                .await?;
        let Some(vr) = reader.vec() else { continue };
        let Some(rows) = vr
            .materialized_index_rows_excluding_async(column, superseded.get(&entry.superfile_id))
            .await
        else {
            continue;
        };
        for row in rows {
            if row.encoded.codes.len() != stride {
                return Err(QueryError::Execute(format!(
                    "hnsw: Sq16 row length {} != dim*2 ({stride}) on column `{column}`",
                    row.encoded.codes.len()
                )));
            }
            let keep = match stride_step {
                Some(k) => gi.is_multiple_of(k),
                None => true,
            };
            if keep {
                codes.extend_from_slice(&row.encoded.codes);
            }
            gi += 1;
        }
    }
    Ok(codes)
}

/// Cheap metadata-only row count for `column` across all superfiles, excluding
/// superseded cells — WITHOUT decoding any code plane. Returns the exact same
/// row total that the decode passes ([`collect_hnsw_codes`] /
/// `materialized_index_rows_excluding_async`) would produce, by mirroring their
/// column gate and superseded-cell exclusion against per-cell doc counts from
/// the blob directory. This lets [`assemble_hnsw_sections`] size the probe's
/// strided step up front, so it can emit the stable doc-ids and the strided
/// probe sample in a SINGLE decode of each superfile instead of two.
async fn count_hnsw_rows(manifest: &ManifestSnapshot, column: &str) -> Result<usize, QueryError> {
    let store = Arc::clone(&manifest.options.store);
    let disk_cache = manifest.options.disk_cache.clone();
    let storage = manifest.options.storage.clone();
    let empty_superseded = BTreeMap::new();
    let superseded = manifest.get_superseded_cells().unwrap_or(&empty_superseded);
    let mut n: usize = 0;
    for entry in manifest.get_all_superfiles() {
        let reader =
            dispatch::open_reader(&store, disk_cache.as_ref(), storage.as_ref(), entry, false)
                .await?;
        let Some(vr) = reader.vec() else { continue };
        if !vr.has_index_column(column) {
            continue;
        }
        let sup = superseded.get(&entry.superfile_id);
        if sup.is_none_or(|s| s.is_empty()) || !vr.is_multi_cell() {
            // No exclusions (or v1): the whole blob's rows are counted, exactly
            // as the decode path takes every materialized row.
            n += vr.n_docs() as usize;
        } else {
            // Multi-cell with superseded cells: count only the cells the decode
            // path keeps (it drops superseded cells' rows).
            for &cell_id in vr.packed_cell_ids() {
                if sup.is_some_and(|s| s.contains(&cell_id)) {
                    continue;
                }
                n += vr.packed_cell_n_docs(cell_id).unwrap_or(0) as usize;
            }
        }
    }
    Ok(n)
}

pub(crate) async fn assemble_hnsw_sections(
    manifest: &ManifestSnapshot,
    column: &str,
    op_stats: &Option<Arc<OpStatsCollector>>,
) -> Result<Option<Vec<u8>>, QueryError> {
    let Some(vc) = manifest
        .options
        .vector_columns
        .iter()
        .find(|vc| vc.column == column)
    else {
        return Ok(None);
    };
    if !vc.rerank_codec.is_sq16() {
        return Ok(None);
    }
    let dim = vc.dim;
    let stride = dim * 2;
    let store = Arc::clone(&manifest.options.store);
    let disk_cache = manifest.options.disk_cache.clone();
    let storage = manifest.options.storage.clone();

    // Gather the stable doc-ids (cheap: 16 B/row) and the row count first. The
    // Sq16 code plane itself (dim*2 B/row — GBs at scale) is gathered LATER: a
    // bounded strided sample for the probe, and the full plane only if the probe
    // passes. A graph-hostile corpus therefore never materializes the whole
    // plane just to decline (which, on a large drain, is the difference between
    // fitting in RAM and OOM).
    let empty_superseded = BTreeMap::new();
    let superseded = manifest.get_superseded_cells().unwrap_or(&empty_superseded);
    // Cheap metadata pre-count (NO code decode): the row total lets us size the
    // probe's strided step before touching any code plane, so the stable
    // doc-ids and the strided probe sample can both come out of ONE decode pass
    // below — previously two separate full decodes (a doc-ids-only decode, then
    // a strided-sample decode).
    let n = count_hnsw_rows(manifest, column).await?;
    if n == 0 {
        return Ok(None);
    }
    let vcfg = &config::global().vector;
    // (m0, ef) candidate grid for the calibrator. An explicit `hnsw_m0`
    // override collapses the m0 search to that value; the ef grid is capped by
    // `hnsw_ef_ceil`.
    let m0_cands: Vec<usize> = if vcfg.hnsw_m0 != 0 {
        vec![vcfg.hnsw_m0]
    } else {
        config::HNSW_M0_CANDIDATES.to_vec()
    };
    let ef_cands: Vec<usize> = config::HNSW_EF_CANDIDATES
        .iter()
        .copied()
        .filter(|&e| e <= vcfg.hnsw_ef_ceil)
        .collect();
    // Cheap probe gate. On a corpus larger than `hnsw_probe_max_docs`,
    // calibrate on a bounded subsample first. Subsample recall is OPTIMISTIC
    // (the m0 requirement grows with N), so a probe that cannot register is a
    // hard "graph-hostile distribution" signal — skip the expensive full build
    // and serve ivf. This gates on distribution, not size: a large but
    // graph-friendly corpus passes the probe and keeps its graph. A registrable
    // probe falls through to the authoritative full-corpus calibration below.
    let probe_cap = vcfg.hnsw_probe_max_docs as usize;
    // Probe path when the corpus exceeds the cap: keep every `step`-th row so
    // the sample is ~probe_cap rows. `None` = small corpus, no probe.
    let probe_step: Option<usize> = (n > probe_cap).then(|| (n / probe_cap).max(1));

    // Single decode pass over every superfile: read/decode each superfile's
    // rows ONCE and, in that same loop, (a) push the stable doc-id (same
    // superfile iteration order, same superseded-cell exclusion, so `doc_ids`
    // is identical to before) and (b) on the probe path, append the row's codes
    // to the strided sample when its global index is on the step (byte-identical
    // to the old `collect_hnsw_codes(.., Some(step))`). The full plane is still
    // read (below) only when the probe registers.
    let mut doc_ids: Vec<i128> = Vec::with_capacity(n);
    // On the probe path this carries only the strided probe sample; on the
    // small-corpus path (no probe) it carries the FULL decoded plane so the
    // full build below reuses it instead of re-reading + re-decoding every
    // superfile a second time. The two modes are mutually exclusive, so one
    // buffer serves both and peak memory is unchanged.
    let mut carried_codes: Vec<u8> = Vec::new();
    let mut gi: usize = 0;
    for entry in manifest.get_all_superfiles() {
        let reader =
            dispatch::open_reader(&store, disk_cache.as_ref(), storage.as_ref(), entry, false)
                .await?;
        let Some(vr) = reader.vec() else { continue };
        let Some(rows) = vr
            .materialized_index_rows_excluding_async(column, superseded.get(&entry.superfile_id))
            .await
        else {
            continue;
        };
        let ids =
            stable_ids_by_local_for_routing(manifest, entry, reader.as_ref(), op_stats).await?;
        for row in rows {
            if row.encoded.codes.len() != stride {
                return Err(QueryError::Execute(format!(
                    "hnsw: Sq16 row length {} != dim*2 ({stride}) on column `{column}`",
                    row.encoded.codes.len()
                )));
            }
            let local = row.local_doc_id as usize;
            let stable_id = *ids.get(local).ok_or_else(|| {
                QueryError::Execute(format!(
                    "hnsw: local_doc_id {local} out of range ({} ids) on `{column}`",
                    ids.len()
                ))
            })?;
            doc_ids.push(stable_id);
            match probe_step {
                // Probe path: keep only every `step`-th row for the sample.
                Some(step) if gi.is_multiple_of(step) => {
                    carried_codes.extend_from_slice(&row.encoded.codes);
                }
                Some(_) => {}
                // Small corpus (no probe): carry the whole plane for the build.
                None => carried_codes.extend_from_slice(&row.encoded.codes),
            }
            gi += 1;
        }
    }
    // The cheap pre-count must equal the decoded row count, or the strided step
    // was sized wrong and the probe sample would diverge from the old code.
    debug_assert_eq!(
        doc_ids.len(),
        n,
        "hnsw pre-count vs decoded row count mismatch"
    );
    if doc_ids.is_empty() {
        return Ok(None);
    }

    if probe_step.is_some() {
        // Reuse the strided sample gathered in the merged pass above — no second
        // decode. (Byte-identical to the prior `collect_hnsw_codes(Some(step))`.)
        let pcodes = std::mem::take(&mut carried_codes);
        let pn = pcodes.len() / stride;
        let pscorer = Sq16Scorer::from_codes(pcodes, dim, pn);
        // The calibrate build is pure CPU; run it on the reader pool (not the
        // global rayon pool, and not inline on the tokio worker) per the
        // concurrency contract. The candidate grids are tiny, so clone them
        // into the closure; the full-corpus pass below still owns the originals.
        let (target_recall, recall_slack, ef_construction) = (
            vcfg.target_recall,
            vcfg.hnsw_recall_slack,
            vcfg.hnsw_ef_construction,
        );
        let (pm0, pef) = (m0_cands.clone(), ef_cands.clone());
        let pchoice = run_on_pool(
            Some(&manifest.options.reader_pool),
            "hnsw probe calibrate: reader pool dropped result",
            move || {
                hnsw::calibrate_graph(
                    &pscorer,
                    &pm0,
                    &pef,
                    target_recall,
                    recall_slack,
                    ef_construction,
                    HNSW_CALIB_QUERIES,
                    HNSW_CALIB_RECALL_K,
                    HNSW_CALIB_SEED,
                )
                .0
            },
        )
        .await
        .map_err(|e| QueryError::Execute(e.to_string()))?;
        if !pchoice.registered {
            tracing::info!(
                column,
                dim,
                n,
                sampled = pn,
                recall = pchoice.recall,
                target = vcfg.target_recall,
                "hnsw probe: best recall below floor — graph-hostile, serving ivf \
                 (full build skipped)"
            );
            return Ok(None);
        }
        tracing::debug!(
            column,
            dim,
            n,
            sampled = pn,
            m0 = pchoice.m0,
            ef = pchoice.ef,
            recall = pchoice.recall,
            "hnsw probe: registrable — proceeding to full-corpus build"
        );
    }
    // Full code plane — reached only for a small corpus (n ≤ probe_cap) or a
    // passing probe. This is where the whole plane is finally materialized.
    // The scorer OWNS the plane; the encoder below borrows it back via
    // `scorer.codes()` rather than keeping a second owned copy alive through
    // the build + encode (the plane is multi-GB at the scale ceiling).
    // Small corpus reuses the full plane already decoded in the merged pass
    // above (no probe was taken from `carried_codes`); the probe path re-reads
    // the plane here, having kept it out of RAM through the probe to bound peak
    // memory at scale.
    let codes = if probe_step.is_none() {
        std::mem::take(&mut carried_codes)
    } else {
        collect_hnsw_codes(manifest, column, stride, None).await?
    };
    // Size the scorer from the DECODED plane, not the metadata pre-count `n`
    // (which only sizes the probe step). The decode can skip a superfile the
    // footer-only count still includes — `materialized_index_rows_*` returns
    // `None` on a transient range/parse fault, not just an absent column — so
    // trusting `n` here could slice the scorer past its buffer. If the decoded
    // plane and the doc-id pass disagree, a transient fault desynced the two
    // reads and a graph built now would mismap nodes to ids: skip it and serve
    // ivf; the next drain rebuilds.
    let decoded_rows = codes.len() / stride;
    if decoded_rows != doc_ids.len() {
        tracing::warn!(
            column,
            doc_ids = doc_ids.len(),
            decoded_rows,
            "hnsw: decoded plane row count != doc-id count (transient read fault?); \
             skipping graph build, serving ivf"
        );
        return Ok(None);
    }
    let scorer = Sq16Scorer::from_codes(codes, dim, decoded_rows);
    // Calibrate (m0, ef) to the table's recall bar on the FULL corpus (build
    // once at m0_max, prune down, sweep ef by re-search — free). The m0
    // requirement is scale-dependent, so this full-corpus pass is authoritative
    // (a subsample under-provisions it: 50K looks 0.99 while the full corpus
    // serves 0.86). The pruned max-graph IS what we persist: no second build.
    // If the graph can't clear the graceful floor, return None so queries serve
    // ivf (the self-driving decision — reuses the existing None→fallback).
    // Calibrate on the reader pool, not inline on the tokio worker or the
    // global rayon pool. The scorer owns the (multi-GB) plane, so move it into
    // the closure and hand it back for the encode below rather than cloning.
    let (target_recall, recall_slack, ef_construction) = (
        vcfg.target_recall,
        vcfg.hnsw_recall_slack,
        vcfg.hnsw_ef_construction,
    );
    let (scorer, choice, graph) = run_on_pool(
        Some(&manifest.options.reader_pool),
        "hnsw calibrate: reader pool dropped result",
        move || {
            let (choice, graph) = hnsw::calibrate_graph(
                &scorer,
                &m0_cands,
                &ef_cands,
                target_recall,
                recall_slack,
                ef_construction,
                HNSW_CALIB_QUERIES,
                HNSW_CALIB_RECALL_K,
                HNSW_CALIB_SEED,
            );
            (scorer, choice, graph)
        },
    )
    .await
    .map_err(|e| QueryError::Execute(e.to_string()))?;
    if !choice.registered {
        tracing::info!(
            column,
            dim,
            n,
            recall = choice.recall,
            target = vcfg.target_recall,
            "hnsw calibrate: best recall below floor — graph NOT registered, serving ivf"
        );
        return Ok(None);
    }
    tracing::info!(
        column,
        dim,
        n,
        m0 = choice.m0,
        ef = choice.ef,
        recall = choice.recall,
        target = vcfg.target_recall,
        at_target = choice.at_target,
        "hnsw calibrate: graph registered"
    );
    let graph = graph.expect("registered choice carries its pruned graph");
    Ok(Some(encode_hnsw(
        scorer.codes(),
        &doc_ids,
        &graph,
        dim,
        choice.ef,
        column,
    )))
}

/// Incrementally extend a prior persisted `hnsw` graph with a
/// freshly-drained append delta — no full rebuild. Reads only rows whose
/// `stable_id > prior_high_water` (the append boundary from the prior
/// bundle header; ids are assigned monotonically at drain), concatenates
/// their code plane + stable ids onto the prior graph's, and inserts only
/// the new nodes via [`Hnsw::extend`]. Returns
/// `(data_bundle_bytes, new_high_water, inserted_node_count)`.
///
/// `Ok(None)` means "cannot incrementally extend — do a full rebuild
/// instead": no new rows, a dim/codec mismatch, or a non-append change
/// (the prior count plus the delta does not equal the current row count,
/// so rows were removed or ids are not a clean monotonic extension). This
/// guard keeps the incremental path strictly for pure appends.
pub(crate) async fn assemble_hnsw_incremental(
    manifest: &ManifestSnapshot,
    column: &str,
    op_stats: &Option<Arc<OpStatsCollector>>,
    prior: crate::superfile::vector::hnsw::HnswIndex,
    prior_high_water: i128,
) -> Result<Option<(Vec<u8>, i128, usize)>, QueryError> {
    let Some(vc) = manifest
        .options
        .vector_columns
        .iter()
        .find(|vc| vc.column == column)
    else {
        return Ok(None);
    };
    let dim = prior.dim;
    if !vc.rerank_codec.is_sq16() || vc.dim != dim {
        return Ok(None);
    }
    let stride = dim * 2;
    let store = Arc::clone(&manifest.options.store);
    let disk_cache = manifest.options.disk_cache.clone();
    let storage = manifest.options.storage.clone();

    // Re-read ONLY the appended rows (stable_id past the prior high water).
    let empty_superseded = BTreeMap::new();
    let superseded = manifest.get_superseded_cells().unwrap_or(&empty_superseded);
    let mut new_codes: Vec<u8> = Vec::new();
    let mut new_doc_ids: Vec<i128> = Vec::new();
    for entry in manifest.get_all_superfiles() {
        let reader =
            dispatch::open_reader(&store, disk_cache.as_ref(), storage.as_ref(), entry, false)
                .await?;
        let Some(vr) = reader.vec() else { continue };
        let Some(rows) = vr
            .materialized_index_rows_excluding_async(column, superseded.get(&entry.superfile_id))
            .await
        else {
            continue;
        };
        let ids =
            stable_ids_by_local_for_routing(manifest, entry, reader.as_ref(), op_stats).await?;
        for row in rows {
            if row.encoded.codes.len() != stride {
                return Err(QueryError::Execute(format!(
                    "hnsw: Sq16 row length {} != dim*2 ({stride}) on column `{column}`",
                    row.encoded.codes.len()
                )));
            }
            let local = row.local_doc_id as usize;
            let stable_id = *ids.get(local).ok_or_else(|| {
                QueryError::Execute(format!(
                    "hnsw: local_doc_id {local} out of range ({} ids) on `{column}`",
                    ids.len()
                ))
            })?;
            if stable_id > prior_high_water {
                new_codes.extend_from_slice(&row.encoded.codes);
                new_doc_ids.push(stable_id);
            }
        }
    }
    if new_doc_ids.is_empty() {
        return Ok(None);
    }
    // Pure-append guard: prior population + delta must equal the current row
    // count. Otherwise rows were removed (or ids are not a clean monotonic
    // extension) and an incremental insert would be wrong — full rebuild.
    let current_total: usize = manifest
        .get_all_superfiles()
        .iter()
        .map(|e| e.n_docs as usize)
        .sum();
    if prior.doc_ids.len() + new_doc_ids.len() != current_total {
        return Ok(None);
    }

    let inserted = new_doc_ids.len();
    // Incremental drains INHERIT the prior graph's calibrated `(m0, ef)`: the
    // new nodes must match the existing base-layer degree (a mixed-degree
    // graph would be inconsistent), and the stamped query beam carries forward.
    let inherited_m0 = prior.graph.base_degree();
    let inherited_ef = prior.ef_search;
    let mut codes = prior.scorer.codes().to_vec();
    codes.extend_from_slice(&new_codes);
    let mut doc_ids = prior.doc_ids;
    doc_ids.extend_from_slice(&new_doc_ids);
    let total = doc_ids.len();
    // The scorer owns the extended plane; the encoder borrows it back via
    // `scorer.codes()` rather than holding a second owned copy.
    let scorer = Sq16Scorer::from_codes(codes, dim, total);
    let vcfg = &config::global().vector;
    let (target_recall, recall_slack, ef_construction, probe_cap) = (
        vcfg.target_recall,
        vcfg.hnsw_recall_slack,
        vcfg.hnsw_ef_construction,
        vcfg.hnsw_probe_max_docs as usize,
    );
    let floor = (target_recall - recall_slack).max(0.0);
    let prior_graph = prior.graph;
    // The extend fans the new-node inserts across rayon, and the recall
    // recheck is pure CPU; both belong on the reader pool, not inline on the
    // tokio worker or the global rayon pool — matching the full-build path.
    let (scorer, graph, recall) = run_on_pool(
        Some(&manifest.options.reader_pool),
        "hnsw incremental extend + recheck: reader pool dropped result",
        move || {
            let params = HnswParams {
                ef_construction,
                m0: inherited_m0,
                ..HnswParams::default()
            };
            // Insert ONLY the new node range into a copy of the prior graph.
            let graph = prior_graph.extend(&scorer, params);
            // Re-check recall on the grown graph. The base-layer degree
            // requirement rises with N, so inherited `(m0, ef)` calibrated at a
            // smaller population can drift below the bar as an append-only table
            // grows (a graph calibrated at 200K and served at 5M is exactly this
            // regime). If it no longer clears the register floor, the caller
            // does a full rebuild, which recalibrates `m0`/`ef` or de-registers.
            //
            // Above `hnsw_probe_max_docs` the exact recheck's ground truth is
            // O(corpus) per query, so measure a bounded strided subsample
            // instead — the same probe the full build gates on — keeping the
            // recheck ~O(probe_cap) regardless of how large the table has grown.
            let recall = if total > probe_cap {
                let step = total / probe_cap;
                let stride_bytes = dim * 2;
                let mut sample: Vec<u8> = Vec::with_capacity(probe_cap * stride_bytes);
                for (i, row) in scorer.codes().chunks_exact(stride_bytes).enumerate() {
                    if i.is_multiple_of(step) {
                        sample.extend_from_slice(row);
                    }
                }
                let pn = sample.len() / stride_bytes;
                let psc = Sq16Scorer::from_codes(sample, dim, pn);
                hnsw::calibrate_graph(
                    &psc,
                    &[inherited_m0],
                    &[inherited_ef],
                    target_recall,
                    recall_slack,
                    ef_construction,
                    HNSW_CALIB_QUERIES,
                    HNSW_CALIB_RECALL_K,
                    HNSW_CALIB_SEED,
                )
                .0
                .recall
            } else {
                hnsw::measure_recall(
                    &graph,
                    &scorer,
                    inherited_ef,
                    HNSW_CALIB_RECALL_K,
                    HNSW_CALIB_QUERIES,
                    HNSW_CALIB_SEED,
                )
            };
            (scorer, graph, recall)
        },
    )
    .await
    .map_err(|e| QueryError::Execute(e.to_string()))?;
    if recall < floor {
        return Ok(None);
    }
    let new_high_water = doc_ids.iter().copied().max().unwrap_or(prior_high_water);
    Ok(Some((
        encode_hnsw(scorer.codes(), &doc_ids, &graph, dim, inherited_ef, column),
        new_high_water,
        inserted,
    )))
}

impl SupertableReader {
    /// Serve top-k from the resident `hnsw` graph persisted at drain, at the
    /// `k`-scaled `ef` law — but ONLY when a valid persisted graph exists
    /// (post-drain, dim-matches, non-empty). Returns `Ok(None)` when there
    /// is no such graph (pre-drain, or corpus over `hnsw_max_docs` so the
    /// drain skipped the graph) so the caller falls through to the ivf scan;
    /// the graph is drain-persisted only, never built in-process at query
    /// time.
    ///
    /// Each hit's `superfile`/`local_doc_id` are left `nil`/`0`: the
    /// `_id`+score projection answers straight from `stable_id`, and any
    /// wider projection resolves the live `(superfile, local)` from that id
    /// through [`user_placement_for_scalar_resolve`] on the shared
    /// `vector_search` path — compaction-correct without baking physical
    /// rows into the graph. The graph walks on `−dot` (Sq16 grid, smaller is
    /// nearer), but the emitted `SuperfileHit.score` is shifted to `1 − dot` to
    /// match the cosine distance the ivf/scan arm emits — the two arms merge on
    /// raw score, so they MUST share one scale (an undrained user-arm hit and a
    /// drained graph hit are compared directly).
    async fn hnsw_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Option<Vec<SuperfileHit>>, QueryError> {
        if k == 0 {
            return Ok(Some(Vec::new()));
        }
        let Some(sections) = self.persisted_graph_sections().await else {
            return Ok(None);
        };
        let Some(data) = sections.data.as_ref() else {
            return Ok(None);
        };
        // The persisted graph is built for exactly one column. A table can
        // carry several same-dim vector columns, so a dim match alone is not
        // enough — a query on a DIFFERENT column must fall back to ivf rather
        // than be answered from this column's neighbors.
        if data.column != column || data.dim != query.len() || data.doc_ids.is_empty() {
            return Ok(None);
        }
        // Over-fetch to absorb boundary replicas. When
        // `drain_replica_target_factor > 1`, a user row is replicated across
        // hidden cells and appears as several graph nodes with the SAME stable
        // id; `top_k_ascending` collapses them by id, so a plain top-`k` walk
        // would return fewer than `k` distinct rows on a delete-free table. The
        // ivf arm over-fetches by the same factor (`k_fetch = k + overhead`).
        //
        // Caveat: this reads the CURRENT config, but replica duplication is a
        // build-time property baked into the persisted graph. Lowering
        // `drain_replica_target_factor` below the value the resident graph was
        // built at under-counts its replicas and can re-open an under-`k`
        // window until the next full rebuild restamps the graph.
        let replica_factor = config::global().vector.drain_replica_target_factor.max(1.0);
        let k_fetch = ((k as f32) * replica_factor).ceil() as usize;
        // The calibrated per-table `ef` stamped in the bundle drives the walk,
        // never below the (over-)fetch width. Every persisted bundle carries a
        // non-zero calibrated `ef`; a 0 (which cannot occur from the drain)
        // degrades to `k_fetch`, still a valid beam.
        let ef = data.ef_search.max(k_fetch);
        // The Sq16 walk is pure CPU (up to ef × m0 scores); run it on the
        // reader pool and await a oneshot, per the rayon-for-CPU / tokio-for-I/O
        // contract — inline it would block a tokio worker for the walk's whole
        // duration, stalling every other query's I/O on that worker.
        let manifest = self.manifest();
        let sections_for_walk = Arc::clone(&sections);
        let query_owned = query.to_vec();
        let hits: Vec<SuperfileHit> = run_on_pool(
            Some(&manifest.options.reader_pool),
            "hnsw serving walk: reader pool dropped result",
            move || {
                let data = sections_for_walk
                    .data
                    .as_ref()
                    .expect("data present: checked before dispatch");
                data.graph
                    .search(&data.scorer, &query_owned, k_fetch, ef)
                    .into_iter()
                    .filter_map(|(node, dist)| {
                        // Shift the graph's `−dot` onto the ivf/scan arm's
                        // `1 − dot` cosine scale so the cross-arm merge
                        // (top_k_ascending) compares like with like. Monotonic
                        // in `dist`, so intra-arm order is unchanged; the public
                        // `score` column stays non-negative.
                        Some(SuperfileHit {
                            superfile: SuperfileUri(Uuid::nil()),
                            local_doc_id: 0,
                            score: 1.0 + dist,
                            stable_id: Some(*data.doc_ids.get(node as usize)?),
                        })
                    })
                    .collect()
            },
        )
        .await
        .map_err(|e| QueryError::Execute(e.to_string()))?;
        Ok(Some(top_k_ascending(vec![hits], k)))
    }

    /// Hydrate (or reuse) the persisted `hnsw` graph sections for
    /// this table, mirroring [`Self::centroid_section`]: one fetch of the
    /// content-addressed graph blob on first use, cached resident on the
    /// handle and keyed by URI. `None` when the manifest carries no graph
    /// ref (older generation / above the scale ceiling) or the fetch failed
    /// — `hnsw_search` then returns `None` and the caller falls through to
    /// the ivf scan.
    async fn persisted_graph_sections(&self) -> Option<Arc<ResidentGraphSections>> {
        let manifest = self.manifest();
        let slot = Arc::clone(&manifest.options.graph_sections_cache);
        let Some(reference) = manifest.slow_vector_state_graphs_blob().cloned() else {
            // No graph ref for this generation (a drain declined the graph, or
            // the corpus crossed the scale ceiling). Drop any previously
            // hydrated sections so their multi-GiB plane is not pinned for the
            // process lifetime; the caller falls through to the ivf scan.
            let mut guard = slot.lock().await;
            *guard = None;
            return None;
        };
        let storage = manifest.options.storage.as_ref()?;
        // Fast path: reuse the resident sections when they already match. Only
        // the cache mutex is taken here, and never across a fetch, so a warm
        // query is never blocked behind a first-touch download.
        {
            let guard = slot.lock().await;
            if let Some(sections) = guard.as_ref()
                && sections.uri == reference.uri
            {
                return Some(Arc::clone(sections));
            }
        }
        // Single-flight hydration. `fetch_graph_sections` blake3-hashes and
        // copies a multi-GiB bundle; without a gate every racing first-touch
        // query would run its own download (N duplicate fetches and a transient
        // N×-plane RSS spike that can OOM). The first miss takes THIS gate and
        // hydrates; concurrent misses park on the gate and, once they hold it,
        // find the cache already published and return without a second fetch.
        // The gate — not the cache mutex — is what is held across the fetch, so
        // the warm fast path above is never serialized behind the download.
        let _hydrating = manifest.options.graph_hydration_lock.lock().await;
        {
            let guard = slot.lock().await;
            if let Some(sections) = guard.as_ref()
                && sections.uri == reference.uri
            {
                return Some(Arc::clone(sections));
            }
        }
        let sections = match fetch_graph_sections(storage.as_ref(), &reference).await {
            Ok(sections) => Arc::new(sections),
            Err(error) => {
                tracing::warn!(
                    "hnsw graph sections {} unavailable ({error}); falling back to \
                     the ivf scan",
                    reference.uri
                );
                return None;
            }
        };
        // Publish under the cache lock. The gate makes this the only in-flight
        // hydration, so it installs the uri it just fetched.
        {
            let mut guard = slot.lock().await;
            *guard = Some(Arc::clone(&sections));
        }
        Some(sections)
    }

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
                tracing::warn!(
                    uri = %reference.uri,
                    "centroid section unavailable ({error}); deferred rescores will fail \
                     unless the parts cache covers their cells"
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
            // Sync section: the preads block the thread (off-CPU), so one
            // bracket around the whole loop attributes only the fp32
            // scoring. Each successful cell read is one planned range —
            // a real per-query pread of the hydrated section, identical
            // at any cache temperature.
            let mut cells_read = 0u64;
            let (leftovers, rescore_ns) = op_stats::timed_section(|| {
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
                    cells_read += 1;
                    if !score_cell_fp32(superfiles, column, &d, &fp32, query, metric, candidates) {
                        leftovers.push(d);
                    }
                }
                Ok::<_, QueryError>(leftovers)
            });
            if let Some(stats) = &self.op_stats {
                stats.add_planned_read_ranges(cells_read);
                stats.add_kernel_cpu_ns(rescore_ns);
            }
            leftovers?
        } else {
            deferred
        };
        // User manifests carry no centroid section; their fp32 lives in
        // the FULL manifest parts (content-addressed), hydrated once per
        // generation and served from RAM after that.
        let deferred = if deferred.is_empty() {
            deferred
        } else if let Some(cache) = self.manifest().user_centroids_for_rescore().await {
            // RAM-hydrated per-generation cache: no read to count, only
            // the fp32 scoring CPU.
            let (leftovers, rescore_ns) = op_stats::timed_section(|| {
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
            });
            if let Some(stats) = &self.op_stats {
                stats.add_kernel_cpu_ns(rescore_ns);
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

    /// Global-fine fanout (`vector.search_mode = global_fine_centroid`, reading
    /// `fanout = vector.global_fine_fanout` clusters, clamped to the table's
    /// cluster total). Phase 1: score `query` against every cell's fp32 fine
    /// centroids from the resident centroid section (a RAM scan — no superfile
    /// opens for the scoring) and keep the global top-`fanout`
    /// `(superfile, flat cluster)` by ascending distance. Phase 2: scan only those clusters per superfile, pool the
    /// warm survivors across all cells, take one global shortlist cut, and
    /// exact-rerank where the winners live. The router overrides only cluster
    /// SELECTION; the byte fetch, 1-bit shortlist, rerank, and id remap are
    /// the stamped path's own code. A tree/HNSW centroid router replaces the
    /// brute-force scan later; see
    /// [`SuperfileReader::global_fine_cluster_scores`].
    async fn global_fine_fanout(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        column: &str,
        query: &[f32],
        k: usize,
        options: &VectorSearchOptions,
        fanout: usize,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let manifest = self.manifest();
        let section = self.centroid_section().await.ok_or_else(|| {
            QueryError::Execute("global-fine: centroid section unavailable".into())
        })?;
        let store = Arc::clone(&manifest.options.store);
        let disk_cache = manifest.options.disk_cache.clone();
        let storage = manifest.options.storage.clone();
        // Path-scoped rerank: a caller-set `rerank_mult` wins, else this
        // path's own configured default — never the shared 256 that serves
        // the stamped / filtered / user-table paths.
        let rerank_mult = options
            .rerank_mult()
            .unwrap_or(config::global().vector.global_fine_rerank_mult);

        // Phase 1: build the global candidate pool. Scores are comparable
        // across cells/superfiles (one metric + one query), so a single
        // top-`fc` cut over the pool is a valid global selection.
        let mut readers: Vec<Arc<SuperfileReader>> = Vec::with_capacity(superfiles.len());
        let mut cands: Vec<(usize, u32, f32)> = Vec::new();
        for entry in superfiles.iter() {
            let reader =
                dispatch::open_reader(&store, disk_cache.as_ref(), storage.as_ref(), entry, false)
                    .await?;
            let si = readers.len();
            if let Some(vr) = reader.vec() {
                let scores = vr
                    .global_fine_cluster_scores(column, query, section.as_ref(), entry.superfile_id)
                    .map_err(|e| QueryError::Execute(e.to_string()))?;
                for (flat, score) in scores {
                    cands.push((si, flat, score));
                }
            }
            readers.push(reader);
        }
        if cands.is_empty() {
            return Ok(Vec::new());
        }
        // `cands` now holds EVERY fine cluster across all cells/superfiles;
        // clamp the requested fanout to that total. Global top-`fc` by
        // ascending distance (smaller = nearer).
        let fc = fanout.clamp(1, cands.len());
        if cands.len() > fc {
            cands.select_nth_unstable_by(fc - 1, |a, b| a.2.total_cmp(&b.2));
            cands.truncate(fc);
        }
        let mut by_sf: HashMap<usize, Vec<u32>> = HashMap::new();
        for (si, flat, _) in cands {
            by_sf.entry(si).or_default().push(flat);
        }

        // Phase 2: DEFERRED per-cell scan on the selected clusters, pooling
        // warm survivors across every cell/superfile, then ONE global exact
        // rerank — the stamped whole-cell path's discipline
        // (`search_clusters_scan_async` -> `select_global_shortlist` ->
        // `vector_rerank_selected`). The immediate `search_clusters_async`
        // path reranks per cell and merges per-cell top-k, which mis-orders
        // ~4% of the true top-k at 10M (all neighbors are read — recall@100
        // is 1.0 — but a single cross-cell exact rerank is needed to seat
        // them in the top-k).
        let mut per_superfile: Vec<Vec<SuperfileHit>> = Vec::new();
        let mut pooled: Vec<(usize, ScanCandidate)> = Vec::new();
        for (si, flats) in by_sf {
            let entry = &superfiles[si];
            let reader = readers[si].as_ref();
            let Some(vr) = reader.vec() else { continue };
            // Per-cell coalesced read (`vector.global_fine_coalesce`): expand
            // each touched cell's selection to its [min..max] contiguous span
            // so the cell reads as one GET instead of scattered clusters.
            let flats = if config::global().vector.global_fine_coalesce {
                vr.coalesce_flats_to_cell_spans(&flats)
            } else {
                flats
            };
            let scan = vr
                .search_clusters_scan_async(
                    column,
                    query,
                    k,
                    &flats,
                    rerank_mult,
                    rerank_mult,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .map_err(|e| QueryError::Execute(e.to_string()))?;
            // Cold cells rerank in-scan; take their exact hits directly.
            if !scan.hits.is_empty() {
                let mut tagged = dispatch::tag_hits(entry, scan.hits);
                dispatch::attach_stable_ids(reader, entry, &mut tagged, false, &self.op_stats)
                    .await?;
                per_superfile.push(tagged);
            }
            for c in scan.candidates {
                pooled.push((si, c));
            }
        }
        // Phase C: single global exact rerank of the pooled warm survivors —
        // one cross-cell shortlist cut, reranked where the winners live.
        if !pooled.is_empty() {
            let shortlist_limit = k.saturating_mul(rerank_mult);
            let winners = select_global_shortlist(pooled, shortlist_limit, 0);
            let mut by_seg: HashMap<usize, Vec<ScanCandidate>> = HashMap::new();
            for (si, c) in winners {
                by_seg.entry(si).or_default().push(c);
            }
            for (si, selected) in by_seg {
                let entry = &superfiles[si];
                let reader = readers[si].as_ref();
                let (hits, _ns) = reader
                    .vector_rerank_selected(column, query, k, selected, None)
                    .await
                    .map_err(|e| QueryError::Execute(e.to_string()))?;
                let mut tagged = dispatch::tag_hits(entry, hits);
                dispatch::attach_stable_ids(reader, entry, &mut tagged, false, &self.op_stats)
                    .await?;
                per_superfile.push(tagged);
            }
        }
        Ok(top_k_ascending(per_superfile, k))
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
        // Global-fine routing (`vector.search_mode = global_fine_centroid`):
        // select the top-`global_fine_fanout` fine centroids GLOBALLY across
        // every cell/superfile from the resident centroid section, bypassing
        // the grid + stamped-law selection, and read only those clusters.
        // Unfiltered hidden path only; `stamped` (or a filtered/user-table
        // query) leaves the stamped-law path below untouched.
        let vcfg = &config::global().vector;
        // Validate the queried column exists BEFORE any serving branch. An
        // undeclared column is a caller error and must be rejected uniformly —
        // otherwise a drained table would answer an unknown-column query from
        // the graph (which carries its own column check but is reached first),
        // while an undrained table rejects it later at the grid lookup.
        if !manifest
            .options
            .vector_columns
            .iter()
            .any(|vc| vc.column == column)
        {
            return Err(QueryError::Execute(format!(
                "unknown vector column `{column}`"
            )));
        }
        // HNSW search mode (`vector.search_mode = hnsw_ivf`): walk the
        // resident graph built at drain over every row's Sq16 codes, bypassing
        // the grid, cell selection, and disk reads. Only the hidden (drained)
        // arm serves via the graph — the user/pre-drain arm always uses ivf,
        // exactly like the global-fine branch below (`hidden_vector_index`).
        // And even on the hidden arm, only when a VALID persisted graph
        // exists (dim-matches, right column, non-empty); if the drain skipped
        // it because the corpus exceeds `hnsw_max_docs`, or the query targets
        // a different column, fall through to the ivf scan (the `_ivf` in the
        // mode name).
        if !filtered && hidden_vector_index && vcfg.search_mode == config::VectorSearchMode::HnswIvf
        {
            if let Some(hits) = self.hnsw_search(column, query, k).await? {
                return Ok(hits);
            }
            warn_hnsw_no_resident_graph(column, manifest.slow_vector_state_graphs_blob().is_some());
            // fall through to the ivf scan below
        }
        if !filtered
            && hidden_vector_index
            && vcfg.search_mode == config::VectorSearchMode::GlobalFineCentroid
            && vcfg.global_fine_fanout > 0
        {
            let fanout = vcfg.global_fine_fanout;
            return self
                .global_fine_fanout(&superfiles, column, query, k, &options, fanout)
                .await;
        }
        // Borrow routing — do not clone the VectorCell centroid grid just to
        // read Copy `CellRoutingParams` (that clone used to drop the transposed
        // SIMD cache and force a per-query scalar transpose rebuild).
        let hidden_routing = manifest.vector_cell_routing();
        // Rerank law: the measured global survivor budget replaces the
        // `k x rerank_mult` DEFAULT on the unfiltered hidden path —
        // expressed as the equivalent multiplier so the divided cold
        // budget and the global shortlist cap both inherit it through
        // `resolve`. A caller-set `rerank_mult` wins over the law, exactly
        // as a caller-set `nprobe` wins over the width law. MUST live at
        // fn scope: an earlier form rebound `options` inside the admit
        // arm, where the shadow died at the arm's brace — the phase-C
        // budget then resolved the ORIGINAL options and the served
        // default silently reverted to the constant (measured as
        // default == rm=256 on a law-stamped table).
        let law_rerank = rerank_mult_from_law(
            hidden_vector_index,
            filtered,
            options.rerank_mult(),
            hidden_routing.as_ref(),
            k,
        );
        // Whether the rerank budget below is the LAW's (scalable with the
        // served width, #515) or an explicit caller value (exact request,
        // never scaled).
        let law_rerank_served = law_rerank.is_some();
        let options = match law_rerank {
            Some(mult) => options.with_rerank_mult(mult),
            None => options,
        };
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
        // The grid/centroid ranking is the admit stage's kernel section —
        // pure CPU over manifest summaries on this thread.
        let ranked_cells_scored: Option<Vec<(u32, f32)>> =
            op_stats::timed_kernel(&self.op_stats, || {
                grid.map(|grid| grid.rank_cells(metric, query))
            });
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
        // (#515) Served cells over stamped width, set by the law-served
        // union arm when the serve window extends past the stamp; scales
        // the pooled rerank budget below so the pool grows with the cells
        // the evidence actually serves. (1, 1) everywhere else — decisive
        // geometry, explicit caller values, filtered search.
        let mut served_cells_over_width: (usize, usize) = (1, 1);
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
            // Fine-depth law: a measured floor on runs-per-probed-cell for
            // the unfiltered hidden path. Applied to the BASE routing before
            // any pin — pin arms then lift depth to MAX on top, so the law
            // only matters where no pin engages (the width<=1 default path,
            // exactly where a flat config floor was measured scale-fragile:
            // 10M post-drain 0.982 at floor 4 vs 0.996 at 8, identical
            // latency).
            if hidden_vector_index
                && !filtered
                && let Some(fine) = hidden_routing.and_then(|r| r.fine_for_k_at(k))
            {
                cell_routing.fine_nprobe = cell_routing.fine_nprobe.max(fine);
            }
            let law_width: Option<usize> =
                if hidden_vector_index && !filtered && options.nprobe.is_none() {
                    hidden_routing
                        .and_then(|r| r.width_for_k_at(k))
                        .filter(|w| *w > LAW_WIDTH_WITHIN_DEFAULT)
                } else {
                    None
                };
            // Depth the DEFAULT (unpinned) path would read per cell — the
            // stamped fine-depth law over the routing base. Captured before
            // the pin lifts `fine_nprobe` to MAX: #515 serve-window
            // extension cells are gated at this bounded depth, not the
            // pin's whole-cell depth.
            let prepin_fine_depth = cell_routing.fine_nprobe;
            let populated_cells = postings_by_cell.len().max(1);
            sweep_width = apply_width_pin(
                &mut cell_routing,
                options.nprobe.map(|n| n.max(1)),
                law_width,
                filtered,
                populated_cells,
            );
            tracing::debug!(
                k,
                ?law_width,
                nprobe_min = cell_routing.nprobe_min,
                nprobe_max = cell_routing.nprobe_max,
                fine_nprobe = cell_routing.fine_nprobe,
                prepin_fine_depth,
                "vector width pin resolved"
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
            // #515 near-tie serve window, config-defaulted like the
            // fraction above (config.yaml / ./infino.yaml — no env var,
            // no rebuild); see the serve-window doc at the top of the
            // file for why this is a config default and not a drain
            // stamp.
            let serve_near_tie_slack = config::global().vector.serve_near_tie_slack;
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
            // Default routing only: a top-k request with no caller `nprobe`
            // must probe enough cells to actually hold `k` rows. The slack
            // window above stops at the nearest near-tie cluster, so a query
            // blended between two clusters selects one undersized cell and
            // returns fewer than `k`; widen along the grid ranking until the
            // probed cells cover `k`. No-op once the nearest cell(s) already
            // hold `k`. An explicit caller `nprobe` is a deliberate width
            // constraint and is left untouched even when it under-fills `k`.
            let cutoff = if options.nprobe.is_none() {
                cover_k_cell_cutoff(cutoff, &ranked_for_beam, &postings_by_cell, k)
            } else {
                cutoff
            };
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
            // Round 0: the pre-#515 write-window slice plus the grid's
            // must-include picks, exactly scored.
            let admit_ranking = estimate_admit_ranking(
                &superfiles,
                column,
                query.len(),
                metric,
                &admit_q,
                allow_ref,
                superseded,
            )?;
            let mut admitted: HashSet<u32> = admit_ranking
                .iter()
                .take(admit_shortlist_window(admit_ranking.len()))
                .map(|(cell, _)| *cell)
                .collect();
            admitted.extend(must_include.iter().copied());
            let (mut candidates, deferred) = score_fine_candidates(
                &superfiles,
                column,
                query,
                metric,
                Some(&admitted),
                true,
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
            // #515 self-measured admit loop, law-served default path only
            // (explicit caller `nprobe` is an exact request; filtered
            // search keeps its own floors; width-1 stamps take fine-first
            // where estimates cliff and the loop would admit nothing).
            // Each round measures the estimate-to-exact residual from the
            // cells already exactly scored and admits the cells whose
            // estimates could plausibly land inside the serve window;
            // admission is evidence-bounded, no query-side fraction.
            let law_default = !filtered && options.nprobe.is_none() && law_width.is_some();
            if law_default {
                loop {
                    let fine_ranked_now = cells_ranked_by_fine_score(&candidates);
                    let Some(&(_, best_exact)) = fine_ranked_now.first() else {
                        break;
                    };
                    let serve_threshold = relative_score_window(best_exact, serve_near_tie_slack);
                    let exact_best_by_cell: HashMap<u32, f32> =
                        fine_ranked_now.into_iter().collect();
                    let round = admit_extension_round(
                        &admit_ranking,
                        &admitted,
                        &exact_best_by_cell,
                        serve_threshold,
                    );
                    if round.is_empty() {
                        break;
                    }
                    let delta: HashSet<u32> = round.into_iter().collect();
                    admitted.extend(delta.iter().copied());
                    let (mut delta_candidates, delta_deferred) = score_fine_candidates(
                        &superfiles,
                        column,
                        query,
                        metric,
                        Some(&delta),
                        false,
                        allow_ref,
                        superseded,
                    )?;
                    if !delta_deferred.is_empty() {
                        self.rescore_deferred_cells(
                            &superfiles,
                            column,
                            query,
                            metric,
                            &mut delta_candidates,
                            delta_deferred,
                        )
                        .await?;
                    }
                    candidates.extend(delta_candidates);
                }
            }
            #[cfg(feature = "test-helpers")]
            admit_trace::record_admit(admitted.iter().copied().collect());
            candidate_counts = candidates
                .iter()
                .map(|(si, cluster, _, _, count)| ((*si, *cluster), *count))
                .collect();
            let ranked = ranked_cells
                .as_ref()
                .expect("ranked cell ids exist with scored ranking");
            if hidden_vector_index {
                let fine_ranked = cells_ranked_by_fine_score(&candidates);
                #[cfg(feature = "test-helpers")]
                admit_trace::record_fine(fine_ranked.clone());
                // Default path: fine-first p=1, the same selection the user
                // (pre-drain) branch ships. Filtered search and explicit
                // caller nprobe keep the wider grid/fine union.
                let default_p1 = !filtered && options.nprobe.is_none() && cutoff == 1;
                let (selected_cells_ordered, extension_cells): (Vec<u32>, HashSet<u32>) =
                    if default_p1 {
                        (
                            fine_first_cell_selection(
                                &fine_ranked,
                                ranked_for_beam.first().map(|(cell, _)| *cell),
                            ),
                            HashSet::new(),
                        )
                    } else if law_default {
                        // #515 law arm: the stamped width is a SERVED floor
                        // (the law contract); the serve window follows the
                        // exact-fine ranking beyond it while the query's own
                        // scores stay near-tied — flat-scored queries widen
                        // themselves, cliff-scored queries serve exactly the
                        // law width, byte-identical to before. Extension
                        // cells read at the bounded pre-pin fine depth.
                        let grid_cells: Vec<u32> = ranked_for_beam[..cutoff]
                            .iter()
                            .map(|(cell, _)| *cell)
                            .collect();
                        match fine_ranked
                            .first()
                            .map(|(_, score)| relative_score_window(*score, serve_near_tie_slack))
                        {
                            Some(threshold) => law_floor_serve_selection(
                                &fine_ranked,
                                &grid_cells,
                                cutoff.max(UNION_FINE_PICKS_MIN),
                                threshold,
                            ),
                            None => (grid_cells, HashSet::new()),
                        }
                    } else {
                        // Explicit caller `nprobe` / filtered search: the
                        // exact-request grid/fine union, pre-#515 shape —
                        // no serve window, no extension.
                        let grid_cells: Vec<u32> = ranked_for_beam[..cutoff]
                            .iter()
                            .map(|(cell, _)| *cell)
                            .collect();
                        let fine_cells: Vec<u32> = fine_ranked
                            .iter()
                            .take(cutoff.max(UNION_FINE_PICKS_MIN))
                            .map(|(cell, _)| *cell)
                            .collect();
                        (
                            union_cell_selection(&grid_cells, &fine_cells),
                            HashSet::new(),
                        )
                    };
                if law_default {
                    served_cells_over_width = (
                        selected_cells_ordered.len().max(1),
                        sweep_width.unwrap_or(1).max(1),
                    );
                }
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
                    (!extension_cells.is_empty()).then_some((&extension_cells, prepin_fine_depth)),
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
                true,
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
        // The PLAN's rerank multiplier — post-law, pre-width-divide. The
        // canonical rerank-row pricing and phase C's selection cap both
        // read this value; the divided cold budget below is an execution
        // detail that must never leak into the priced count.
        let (_, plan_rerank_mult) = options.resolve(filtered);
        let mut cold_rerank_mult = 0;
        let options = match sweep_width {
            // (#537) An explicit caller nprobe keeps the FULL per-cell
            // budget — the divide below is for the law arm only. Divided,
            // per-cell retention shrinks as the caller widens, and the
            // 1-bit in-cell ranking is too weak to hold the true
            // neighbors in a thin cut: on the 10M dense grid (~2.8K-row
            // cells) measured recall tracks the divided depth down its
            // ladder — whole-cell at w<=4 serves 0.993, w=16 (~6% of
            // each cell) serves 0.85, all-cells (20 rows/cell) serves
            // 0.51 — monotone in DEPTH, inverted in width. Undivided,
            // widening adds cells at constant depth, so the sweep is
            // monotone and read cost is linear in the width the caller
            // asked for: explicit nprobe is a diagnostic surface, and
            // "probe everything" honestly costs a scan.
            Some(w) if w > 1 && options.nprobe.is_some() => {
                if global_shortlist_width.is_some() {
                    let (_, rerank_mult) = options.resolve(filtered);
                    cold_rerank_mult = rerank_mult;
                }
                options
            }
            // Law arm: the stamped budget is calibrated as a TOTAL at the
            // stamped width, so it divides across the sweep.
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
        let op_stats_scan = self.op_stats.clone();
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
        // Widest replica overhead across every scanned unit — phase C's
        // selection cap reads it so pooled-set membership (which shifts
        // with cache temperature) can never shrink the cap.
        let max_replica_overhead = Arc::new(AtomicU64::new(0));
        let max_replica_overhead_body = Arc::clone(&max_replica_overhead);
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
                let max_replica_overhead = Arc::clone(&max_replica_overhead_body);
                let op_stats = op_stats_scan.clone();
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
                        if let Some(stats) = &op_stats {
                            stats.add_vector_scan(scan.cells_scanned, scan.candidates_scanned);
                            // Request-shaped ranges only (cluster index +
                            // prefixes/blocks + Sq8 meta). Rerank rows are
                            // diagnostics; their cost rides the priced
                            // CPU watermark.
                            stats.add_planned_read_ranges(scan.ranges_requested);
                            stats.add_vector_rows_reranked(scan.rows_reranked);
                            stats.add_kernel_cpu_ns(scan.kernel_cpu_ns);
                        }
                        max_replica_overhead
                            .fetch_max(replica_overhead as u64, atomic::Ordering::Relaxed);
                        if !scan.candidates.is_empty() {
                            scan_pool
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .push((si, scan.rot_seed, replica_overhead, scan.candidates));
                        }
                        scan.hits
                    } else {
                        let (hits, tally) = reader
                            .vector_search_clusters_filtered(
                                &column, &query, k_fetch, &ids, options, bitmap, deny, pool, budget,
                            )
                            .await
                            .map_err(vector_read_query_error)?;
                        if let Some(stats) = &op_stats {
                            stats.add_vector_scan(tally.cells_scanned, tally.candidates_scanned);
                            // Request-shaped ranges only — rerank rows are
                            // diagnostics; their cost rides the priced CPU
                            // watermark (survivor fetches coalesce into a
                            // handful of real requests, so counting one
                            // range per row would not be request-shaped).
                            stats.add_planned_read_ranges(tally.ranges_requested);
                            stats.add_vector_rows_reranked(tally.rows_reranked);
                            stats.add_kernel_cpu_ns(tally.kernel_cpu_ns);
                        }
                        hits
                    };
                    let mut tagged = dispatch::tag_hits(&entry, hits);
                    // Prefer manifest span arithmetic; only touch `_id` pages /
                    // inline IVF regions when the layout is cell-packed or gapped.
                    io_counters::phase_timed_async("vec.stable_id", async {
                        dispatch::attach_stable_ids(
                            &reader_for_ids,
                            &entry,
                            &mut tagged,
                            false,
                            &op_stats,
                        )
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
                            &op_stats,
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
            // Rerank rows are deliberately NOT in the priced range
            // counter: `planned_read_ranges` is request-shaped (numbers
            // commensurate with real object-store requests, which
            // coalesce survivor rows into a handful of GETs), and the
            // platform prices it at a per-request rate. The rerank leg's
            // cost is CPU-dominated and carried by the priced CPU
            // watermark; the row counts stay visible in the
            // rows-reranked / candidates diagnostics.
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
                // Mirror phase A/C's `k_fetch = k + replica_overhead` in the
                // global cut so boundary replicas (dormant today: overhead
                // is 0 with replication off) cannot take shortlist slots
                // from distinct rows before the stable-id dedup. Taken
                // over ALL scanned units — not just the pooled (warm)
                // ones — so under replication the selection cap is
                // temperature-invariant and always agrees with the
                // canonical priced budget above, which uses the same max.
                // (Pooled membership shifts with cache temperature: a
                // fully cold unit reranks in-scan and pools nothing, so
                // a pooled-only max could shrink the cap on cold runs.)
                let replica_overhead =
                    usize::try_from(max_replica_overhead.load(atomic::Ordering::Relaxed))
                        .unwrap_or(0);
                let mut flat: Vec<(usize, ScanCandidate)> = pooled
                    .into_iter()
                    .flat_map(|(si, _, _, cands)| cands.into_iter().map(move |c| (si, c)))
                    .collect();
                // Per-cell floor ONLY under an explicit caller nprobe. The
                // floor exists for the #494 inversion — far cells' 1-bit
                // noise evicting near cells' true neighbors from the fixed
                // budget as a CALLER widens the sweep — so it guards
                // exactly the widths a caller pins. Law-served defaults
                // are calibrated against realized recall at their own
                // stamped width and run floor-free: at law widths the
                // floor measured +0.0001 recall for ~3.4 ms of extra
                // rerank at k=100 (Cohere-1M), and at width 1-2 it is a
                // near-no-op by construction. Re-measured at the WIDENED
                // widths the #515 admit extension serves (Cohere defaults
                // with the loop active, diffuse stamps both k): still
                // floor-free, 0.9955 @ k=100 and 0.9940 @ k=10 — above
                // the stamped-width baseline, no inversion dip. If a
                // future gate dips, the targeted fix is arming floor = k
                // on the extension subset only, not on the law's picks.
                // Floor = k when it applies:
                // even if the entire true top-k concentrates in one probed
                // cell, that cell's floor carries it into the exact rerank
                // (replicas never collide inside one cell, so the floor
                // needs no replica overhead).
                //
                // (#537) The floor's DEPTH must not shrink as the caller
                // widens — and the depth that holds recall is the full
                // per-cell budget, not a share of it. The measured 10M
                // ladder (see the width-divide comment above the fan-out)
                // tracks per-cell retention depth almost mechanically:
                // whole-cell retention serves 0.993, ~6% of a cell serves
                // 0.85, 20 rows serves 0.51 — in-cell 1-bit ranking is
                // too weak to concentrate the true neighbors into a thin
                // cut, so no fixed pool or stamped share survives a wide
                // sweep. Under an explicit caller nprobe every scanned
                // cell therefore keeps the same k x rerank_mult depth the
                // narrow probe would give it: widening adds cells at
                // constant depth, the exact rerank adjudicates, and cost
                // is linear in the width the caller asked for — the pin
                // arm's stated semantics.
                let cell_floor = if options.nprobe.is_some() {
                    k.saturating_mul(plan_rerank_mult)
                } else {
                    0
                };
                // (#515) The LAW's rerank budget is calibrated on drain
                // rows at the stamped width; when the serve window extends
                // serving past that width, the same budget starves the
                // widened candidate set — measured at true defaults on
                // BioASQ-1M: stamped budget serves 0.9510 where the knee
                // sits at ~3-6x (rm=32 → 0.9820, rm=64 → 0.9880, flat by
                // 128). Scale the pooled budget by served-cells over
                // stamped-width, so the pool grows exactly with the cells
                // the evidence serves: BioASQ lands at the measured knee;
                // decisive geometry serves width == stamp and is
                // unchanged. Explicit caller rerank_mult stays an exact,
                // unscaled request.
                let shortlist_limit = deferred_shortlist_limit(
                    k,
                    replica_overhead,
                    plan_rerank_mult,
                    law_rerank_served,
                    options.nprobe.is_some(),
                    served_cells_over_width,
                );
                // Regression probe for the serve-the-law scope bug: recall
                // floors can't see a re-shadowed `options` (the constant
                // budget only ADDS survivors); the served limit can.
                #[cfg(feature = "test-helpers")]
                served_shortlist_probe::record(shortlist_limit, cell_floor);
                flat = select_global_shortlist(flat, shortlist_limit, cell_floor);
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
                if let Some(stats) = &self.op_stats {
                    // Actual winner rows; their planned ranges were priced
                    // canonically above, from the budget these winners were
                    // selected under.
                    let rows: u64 = rerank_units.iter().map(|(_, sel)| sel.len() as u64).sum();
                    stats.add_vector_rows_reranked(rows);
                }
                let column = Arc::clone(&column_arc2);
                let query = Arc::clone(&query_arc2);
                let reader_pool = Arc::clone(&manifest.options.reader_pool);
                let op_stats_c = self.op_stats.clone();
                let body_c = move |reader: Arc<SuperfileReader>,
                                   entry: Arc<SuperfileEntry>,
                                   _tombstone_cache: Option<Arc<SidecarCache>>,
                                   _now: Instant,
                                   selected: Vec<ScanCandidate>| {
                    let column = Arc::clone(&column);
                    let query = Arc::clone(&query);
                    let reader_pool = Arc::clone(&reader_pool);
                    let op_stats = op_stats_c.clone();
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
                        let (hits, rerank_kernel_ns) = reader
                            .vector_rerank_selected(
                                &column,
                                &query,
                                k_fetch,
                                selected,
                                Some(reader_pool),
                            )
                            .await
                            .map_err(vector_read_query_error)?;
                        if let Some(stats) = &op_stats {
                            stats.add_kernel_cpu_ns(rerank_kernel_ns);
                        }
                        let mut tagged = dispatch::tag_hits(&entry, hits);
                        io_counters::phase_timed_async("vec.stable_id", async {
                            dispatch::attach_stable_ids(
                                &reader_for_ids,
                                &entry,
                                &mut tagged,
                                false,
                                &op_stats,
                            )
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
        let op_stats = self.op_stats.clone();
        self.fanout_candidate_bitmaps(superfiles, move |r, _entry| {
            let filter_col_arc = Arc::clone(&filter_col_arc);
            let tokens_arc = Arc::clone(&tokens_arc);
            let op_stats = op_stats.clone();
            async move {
                let refs: Vec<&str> = tokens_arc.iter().map(String::as_str).collect();
                let (docs, work) = r
                    .token_match(&filter_col_arc, &refs, mode)
                    .await
                    .map_err(|e| QueryError::Parquet(e.to_string()))?;
                // The predicate-resolution leg of filtered vector search
                // is FTS work like any other; flush it per superfile.
                if let Some(stats) = &op_stats {
                    stats.add_fts_postings_bytes(work.postings_bytes);
                    stats.add_planned_read_ranges(work.planned_ranges);
                    stats.add_kernel_cpu_ns(work.kernel_cpu_ns);
                }
                Ok(docs.into_iter().collect::<RoaringBitmap>())
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
        // SQL exec's filtered arm enters here without the sync wrappers —
        // apply the cosine query calibration (#512) at this seam.
        let query = calibrated_query(self, column, query);
        let query: &[f32] = &query;
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
            let locals = Arc::new(bm.iter().collect::<Vec<u32>>());
            let ids =
                read_ids_for_locals(manifest, &entry, &locals, id_column, false, &self.op_stats)
                    .await?;
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
            .map(|hidden| {
                hidden
                    .pinned_reader_with(self.op_stats.clone())
                    .manifest()
                    .get_drained_ranges()
            })
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
            let hidden_reader = vit.pinned_reader_with(self.op_stats.clone());
            let hidden_manifest = Arc::clone(hidden_reader.manifest());
            let drained = hidden_manifest.get_drained_ranges();
            let superfiles = hidden_manifest
                .get_all_superfiles_loaded()
                .await
                .map_err(QueryError::ManifestLoad)?;
            if !superfiles.is_empty() {
                let allow_for_cell = Arc::clone(&allow_global);
                let manifest_for_ids = Arc::clone(&hidden_manifest);
                let routing_stats = hidden_reader.op_stats.clone();
                let allow_by_uri = hidden_reader
                    .fanout_candidate_bitmaps(&superfiles, move |r, entry| {
                        let allow_for_cell = Arc::clone(&allow_for_cell);
                        let manifest_for_ids = Arc::clone(&manifest_for_ids);
                        let routing_stats = routing_stats.clone();
                        async move {
                            let stable_ids = stable_ids_by_local_for_routing(
                                &manifest_for_ids,
                                &entry,
                                &r,
                                &routing_stats,
                            )
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

    /// Test-only diagnostic (#515): map every drained row's stable id to
    /// its hidden cell. Packed hidden superfiles are cell-contiguous in
    /// local-doc order, so walking each superfile's local-order stable
    /// ids against its summary's per-cell counts recovers the assignment
    /// without touching posting payloads beyond the routing-id reads the
    /// filtered path already performs.
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn diag_hidden_stable_cell_map(
        &self,
        column: &str,
    ) -> Result<HashMap<i128, u32>, QueryError> {
        let Some(vit) = self.vector_index_table() else {
            return Ok(HashMap::new());
        };
        let hidden_reader = vit.pinned_reader_with(self.op_stats.clone());
        let hidden_manifest = Arc::clone(hidden_reader.manifest());
        let superfiles = hidden_manifest
            .get_all_superfiles_loaded()
            .await
            .map_err(QueryError::ManifestLoad)?;
        let map = Arc::new(Mutex::new(HashMap::new()));
        let column_owned = column.to_string();
        let map_for_fanout = Arc::clone(&map);
        let manifest_for_ids = Arc::clone(&hidden_manifest);
        let routing_stats = hidden_reader.op_stats.clone();
        let _ = hidden_reader
            .fanout_candidate_bitmaps(&superfiles, move |r, entry| {
                let map = Arc::clone(&map_for_fanout);
                let manifest_for_ids = Arc::clone(&manifest_for_ids);
                let column = column_owned.clone();
                let routing_stats = routing_stats.clone();
                async move {
                    let stable_ids = stable_ids_by_local_for_routing(
                        &manifest_for_ids,
                        &entry,
                        &r,
                        &routing_stats,
                    )
                    .await?;
                    if let Some(vs) = entry.vector_summary.get(&column) {
                        let mut idx = 0usize;
                        let mut guard = map.lock().expect("diag cell-map lock");
                        for cell in &vs.cells {
                            let n: u64 = cell.clusters.counts.iter().map(|&c| u64::from(c)).sum();
                            for _ in 0..n {
                                if idx >= stable_ids.len() {
                                    break;
                                }
                                if let Some(cid) = cell.cell_id {
                                    guard.insert(stable_ids[idx], cid);
                                }
                                idx += 1;
                            }
                        }
                    }
                    Ok(RoaringBitmap::new())
                }
            })
            .await?;
        let map = Arc::try_unwrap(map)
            .map(|m| m.into_inner().expect("diag cell-map lock"))
            .unwrap_or_default();
        Ok(map)
    }

    /// Test-only diagnostic (#515): the hidden table's stamped probe
    /// laws — `(width_for_k, fine_for_k, rerank_for_k)` — read from the
    /// hidden manifest's partition strategy, exactly as the bench's
    /// hidden-stats line reports them. `None` when there is no hidden
    /// index or no cell strategy.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn diag_hidden_probe_laws(&self) -> Option<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        let vit = self.vector_index_table()?;
        match vit
            .pinned_reader_with(self.op_stats.clone())
            .manifest()
            .get_partition_strategy()
        {
            PartitionStrategy::VectorCell { routing, .. } => Some((
                routing.width_for_k.to_vec(),
                routing.fine_for_k.to_vec(),
                routing.rerank_for_k.to_vec(),
            )),
            _ => None,
        }
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
            let hidden_reader = vit.pinned_reader_with(self.op_stats.clone());
            let hidden_manifest = Arc::clone(hidden_reader.manifest());
            let drained = hidden_manifest.get_drained_ranges();
            let superfiles = hidden_manifest
                .get_all_superfiles_loaded()
                .await
                .map_err(QueryError::ManifestLoad)?;
            if !superfiles.is_empty() {
                let allow_for_cell = Arc::clone(&allow_set);
                let manifest_for_ids = Arc::clone(&hidden_manifest);
                let routing_stats = hidden_reader.op_stats.clone();
                let allow_by_uri = hidden_reader
                    .fanout_candidate_bitmaps(&superfiles, move |r, entry| {
                        let allow_for_cell = Arc::clone(&allow_for_cell);
                        let manifest_for_ids = Arc::clone(&manifest_for_ids);
                        let routing_stats = routing_stats.clone();
                        async move {
                            let stable_ids = stable_ids_by_local_for_routing(
                                &manifest_for_ids,
                                &entry,
                                &r,
                                &routing_stats,
                            )
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
        let routing_stats = self.op_stats.clone();
        let allow_by_uri = self
            .fanout_candidate_bitmaps(&superfiles, move |r, entry| {
                let allow_for_user = Arc::clone(&allow_for_user);
                let manifest_for_ids = Arc::clone(&manifest_for_ids);
                let routing_stats = routing_stats.clone();
                async move {
                    let stable_ids = stable_ids_by_local_for_routing(
                        &manifest_for_ids,
                        &entry,
                        &r,
                        &routing_stats,
                    )
                    .await?;
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
        // Bench/test entry that bypasses the sync wrappers — apply the
        // cosine query calibration (#512) so measured scores match the
        // public path. (The pub(crate) twin is only reached from flows
        // that already calibrated at their own entry.)
        let query = calibrated_query(self, column, query);
        self.vector_hits_prepared_global_allow_inner(column, &query, k, options, prepared)
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
            let hidden_reader = vit.pinned_reader_with(self.op_stats.clone());
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
        let op_stats = self.op_stats.clone();
        self.fanout_candidate_bitmaps(superfiles, move |r, _entry| {
            let plan = Arc::clone(&plan_arc);
            let op_stats = op_stats.clone();
            async move {
                let (bitmap, work) = plan
                    .evaluate(r.as_ref())
                    .await
                    .map_err(|e| QueryError::Parquet(e.to_string()))?;
                // The SQL predicate's posting walks, summed across the
                // plan tree — the pushdown leg of the vector TVF.
                if let Some(stats) = &op_stats {
                    stats.add_fts_postings_bytes(work.postings_bytes);
                    stats.add_planned_read_ranges(work.planned_ranges);
                    stats.add_kernel_cpu_ns(work.kernel_cpu_ns);
                }
                bitmap.ok_or_else(|| {
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
        let hidden_reader = vit.pinned_reader_with(self.op_stats.clone());
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
            vit.pinned_reader_with(self.op_stats.clone())
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
        // SQL exec enters here without passing the sync wrappers — apply
        // the same cosine query calibration (#512) at this seam.
        let query = calibrated_query(self, column, query);
        self.vector_search_global_index_async(column, &query, k, options)
            .await
    }
}

/// Cosine queries normalize once at each entry seam (#512): ranking is
/// query-norm-invariant (the rerank kernel divides by the per-DOC norm),
/// but a non-unit query scales every returned score by its own norm —
/// normalizing keeps scores calibrated cosine. Other metrics pass
/// through untouched; unit queries are a fp-noise no-op. The seams are
/// chosen so every caller path crosses exactly one: the sync wrappers
/// (`vector_search`/`vector_hits`), the SQL exec arms
/// (`vector_search_async`, `vector_hits_filtered_by_plan`), the hybrid
/// path (`hybrid_search_async`), and the test-helpers allow-set entry.
pub(crate) fn calibrated_query<'q>(
    reader: &SupertableReader,
    column: &str,
    query: &'q [f32],
) -> Cow<'q, [f32]> {
    let cosine = reader
        .options()
        .vector_columns
        .iter()
        .any(|c| c.column == column && c.metric == Metric::Cosine);
    calibrated_query_for(cosine, query)
}

/// The metric-independent half of [`calibrated_query`]: normalize iff the
/// column's metric is cosine. Split out so the contract — cosine
/// normalizes, every other metric passes through by reference — is
/// directly unit-testable without a reader fixture.
fn calibrated_query_for(cosine: bool, query: &[f32]) -> Cow<'_, [f32]> {
    if cosine {
        let mut q = query.to_vec();
        normalize(&mut q);
        Cow::Owned(q)
    } else {
        Cow::Borrowed(query)
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
        let query = calibrated_query(self, column, query);
        let query: &[f32] = &query;
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
            // GUARDED on every hit carrying a stamp, exactly as the hybrid
            // path guards its own fast path (`hybrid_exec.rs`): a superfile
            // entry with no row-id base stamps `stable_id: None`
            // (`dispatch.rs`), and `hits_id_score_batch` treats that as an
            // upstream bug. Falling through resolves the id by placement
            // instead of failing the query.
            if free_columns_unambiguous(&self.options().schema, id_column)
                && let Some(indices) = id_score_projection_indices(projection, id_column)
                && hits.iter().all(|hit| hit.stable_id.is_some())
            {
                let batch = hits_id_score_batch(self, &hits)?
                    .project(&indices)
                    .map_err(|e| QueryError::Execute(e.to_string()))?;
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
        let query = calibrated_query(self, column, query);
        let query: &[f32] = &query;
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
/// The rerank-law multiplier for this query, or `None` to keep the
/// caller's options untouched. `Some` only when ALL of: the query runs
/// on the hidden vector-index table, it is unfiltered (filtered queries
/// keep their own budget model), the caller set no `rerank_mult` (caller
/// intent always wins, exactly as a caller `nprobe` wins over the width
/// law), and the manifest carries a calibrated rerank point at this `k`.
/// The measured global survivor budget is expressed as the equivalent
/// multiplier so the divided cold budget and the global shortlist cap
/// both inherit it through `resolve`.
fn rerank_mult_from_law(
    hidden_vector_index: bool,
    filtered: bool,
    caller_rerank_mult: Option<usize>,
    hidden_routing: Option<&CellRoutingParams>,
    k: usize,
) -> Option<usize> {
    if !hidden_vector_index || filtered || caller_rerank_mult.is_some() {
        return None;
    }
    hidden_routing
        .and_then(|r| r.rerank_for_k_at(k))
        .map(|n| n.div_ceil(k.max(1)).max(1))
}

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

/// The deferred-rerank plan's global shortlist budget. Phase C's selection
/// cap and the canonical priced rerank-row count both derive from this one
/// formula so the two can never drift: `(k + replica_overhead) x
/// rerank_mult`, scaled by served-cells-over-stamped-width when the LAW's
/// budget (not an explicit caller request) is serving a widened sweep.
fn deferred_shortlist_limit(
    k: usize,
    replica_overhead: usize,
    rerank_mult: usize,
    law_rerank_served: bool,
    caller_nprobe: bool,
    served_cells_over_width: (usize, usize),
) -> usize {
    let base = k
        .saturating_add(replica_overhead)
        .saturating_mul(rerank_mult);
    if law_rerank_served && !caller_nprobe {
        let (served, stamped_width) = served_cells_over_width;
        // Total over a degenerate zero stamp: the law path never
        // produces one today ((1, 1) default, `.max(1)` at assignment),
        // but a plain helper must not be able to panic.
        base.saturating_mul(served).div_ceil(stamped_width.max(1))
    } else {
        base
    }
}

/// Deterministic global shortlist selection for deferred-rerank width
/// sweeps: keep the `limit` best 1-bit estimates pooled across every
/// scanned unit, plus each scanned (unit, cell)'s `cell_floor` best.
/// Higher estimate = better; ties break on (unit, cell, pos, did), a
/// total order, so the KEPT SET is insertion-order independent across
/// concurrent scans. O(n) partition, not a sort — the winners need no
/// internal order (phase C regroups by unit and the rerank is exact).
fn select_global_shortlist(
    mut pooled: Vec<(usize, ScanCandidate)>,
    limit: usize,
    cell_floor: usize,
) -> Vec<(usize, ScanCandidate)> {
    let cmp = |a: &(usize, ScanCandidate), b: &(usize, ScanCandidate)| {
        b.1.estimate.total_cmp(&a.1.estimate).then_with(|| {
            (a.0, a.1.cell_idx, a.1.pos, a.1.did).cmp(&(b.0, b.1.cell_idx, b.1.pos, b.1.did))
        })
    };
    if pooled.len() <= limit {
        return pooled;
    }
    // Per-cell floor: every scanned (unit, cell) keeps its `cell_floor`
    // best candidates REGARDLESS of the global competition. The floored
    // core is monotone by construction — a candidate admitted by its own
    // cell's floor cannot be evicted by far cells' 1-bit false positives —
    // which FLOORS the worst case rather than proving end-to-end
    // monotonicity: a candidate ranked below its cell's floor but inside
    // the global pool (a band that grows with `rerank_mult`) keeps the
    // pooled selection's width-blind behavior. Without the floor that
    // band is the WHOLE selection, and recall inverts as nprobe grows
    // (measured at 10M on the post-split grid: 0.994 at nprobe=2 falling
    // to 0.388 at all cells). The floor is `k` at the call site: even if
    // the entire true top-k lives in one cell, that cell's floor carries
    // it. Cost is bounded and linear: at most `cells x cell_floor` extra
    // survivors for the exact rerank.
    //
    // The floor selects in place on the pool's per-cell grouping, BEFORE
    // the global cut permutes it: each unit's scan emits its cells as
    // contiguous blocks (cells complete independently and append whole;
    // completion order varies, contiguity does not), so one boundary
    // walk partitions each group to its `cell_floor` best — no keys, no
    // hashing, no extra streaming passes. An earlier form grouped the
    // post-cut tail with HashMap<(unit, cell), Vec<idx>>; SipHash over
    // the few-hundred-thousand-candidate tail cost ~7 ms of a 19.8 ms
    // Cohere-1M k=100 query, several times the floored rows' whole
    // rerank. If grouping is ever violated (one cell split across R
    // runs), each run keeps its own floor-best — a SUPERSET of the
    // guarantee, never fewer survivors: recall-safe, cost bounded by
    // R x cell_floor.
    let mut floor_keep: Vec<(usize, ScanCandidate)> = Vec::new();
    if cell_floor > 0 {
        let mut start = 0;
        while start < pooled.len() {
            let (si, cell) = (pooled[start].0, pooled[start].1.cell_idx);
            let mut end = start + 1;
            while end < pooled.len() && pooled[end].0 == si && pooled[end].1.cell_idx == cell {
                end += 1;
            }
            let group = &mut pooled[start..end];
            if group.len() > cell_floor {
                group.select_nth_unstable_by(cell_floor, cmp);
                floor_keep.extend_from_slice(&group[..cell_floor]);
            } else {
                floor_keep.extend_from_slice(group);
            }
            start = end;
        }
    }
    // Global cut: O(n) partition puts the pooled top-`limit` in the
    // prefix (unordered — phase C regroups by unit and the rerank is
    // exact, so the winners need no internal order).
    pooled.select_nth_unstable_by(limit, cmp);
    pooled.truncate(limit);
    if floor_keep.is_empty() {
        return pooled;
    }
    // Kept set = global prefix ∪ per-cell floor picks. The two overlap
    // heavily (a cell's best are usually global winners), so dedup by
    // the total-order identity key — both sets are shortlist-scale
    // (~limit + cells x cell_floor keys), not pool-scale, so this is the
    // one place hashing stays cheap.
    let kept: HashSet<(usize, usize, u32, u32)> = pooled
        .iter()
        .map(|(si, c)| (*si, c.cell_idx, c.pos, c.did))
        .collect();
    for (si, cand) in floor_keep {
        if !kept.contains(&(si, cand.cell_idx, cand.pos, cand.did)) {
            pooled.push((si, cand));
        }
    }
    pooled
}

fn top_k_ascending(per_superfile: Vec<Vec<SuperfileHit>>, k: usize) -> Vec<SuperfileHit> {
    // Total order over hits: distance ascending, then the unique
    // `(superfile, local_doc_id)` key, then `stable_id`. The tie-break makes
    // the kept set deterministic when scores are equal (common when many rows
    // share a direction) — otherwise the k-boundary among ties would be
    // resolved by heap feed order (HashMap iteration + fan-out completion),
    // which varies run to run. Tie order never affects recall: equal-distance
    // rows are interchangeable. The `stable_id` leg is load-bearing for graph
    // (hnsw) hits: they carry no `(superfile, local_doc_id)` (both nil/0), so
    // without it every equal-distance graph hit compares Equal and the
    // k-boundary among them would again be nondeterministic.
    fn hit_order(a: &SuperfileHit, b: &SuperfileHit) -> Ordering {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.superfile.cmp(&b.superfile))
            .then_with(|| a.local_doc_id.cmp(&b.local_doc_id))
            .then_with(|| a.stable_id.cmp(&b.stable_id))
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
    /// # use infino::{connect, IndexSpec, Metric};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new(
    /// #     "emb",
    /// #     DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 16),
    /// #     false,
    /// # )]));
    /// # let vecs = db.create_table("vecs", schema.clone(), IndexSpec::new().vector("emb", 16, Metric::Cosine))?;
    /// # let mut data = vec![0.0f32; 16]; data[0] = 1.0;
    /// # let col = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vec![Some(data.iter().copied().map(Some).collect::<Vec<_>>())], 16);
    /// # vecs.append(&RecordBatch::try_new(schema, vec![Arc::new(col)])?)?;
    /// # let mut query = vec![0.0f32; 16]; query[0] = 1.0;
    /// // Bare call → `_id` + `score`, no scalar decode:
    /// let hits = vecs.vector_search("emb", &query, 10, None, None)?;
    /// assert_eq!(hits[0].num_columns(), 2);
    /// // Explicit projection names the same columns (scalar columns,
    /// // when present, materialize row data):
    /// let rows = vecs.vector_search(
    ///     "emb",
    ///     &query,
    ///     10,
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
        borrow::Cow,
        collections::{BTreeMap, BTreeSet, HashMap, HashSet},
        sync::Arc,
    };

    use arrow::array::Array;

    /// Cosine columns normalize the query; every other metric passes the
    /// caller's slice through untouched, by reference (no copy, no scale).
    #[test]
    fn calibrated_query_normalizes_cosine_only() {
        let q = [3.0f32, 4.0];
        let cos = calibrated_query_for(true, &q);
        assert!((cos[0] - 0.6).abs() < 1e-6);
        assert!((cos[1] - 0.8).abs() < 1e-6);
        let non_cos = calibrated_query_for(false, &q);
        assert!(matches!(non_cos, Cow::Borrowed(_)));
        assert_eq!(&*non_cos, &q[..]);
    }
    use arrow_array::{
        Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
    };
    use arrow_schema::{DataType, Field, Schema};

    use super::{
        RABITQ_ADMIT_CELL_SHORTLIST_MIN, SCORE_COLUMN, ScanCandidate, VectorFilter,
        VectorSearchOptions, admit_extension_round, admit_shortlist_window, apply_width_pin,
        calibrated_query_for, cells_ranked_by_fine_score, free_column_slot,
        free_columns_unambiguous, gate_fine_candidates_by_fragment, hidden_hits_user_ids,
        id_score_projection_indices, is_hidden_vector_manifest, law_floor_serve_selection,
        postings_by_cell_from_summaries, rerank_mult_from_law, score_fine_candidates,
        select_global_shortlist, union_cell_selection, vector_read_query_error,
    };
    use crate::{
        BoolMode, InfinoError,
        superfile::{
            SuperfileReader,
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig},
            error::{ReadError, VectorError},
            fts::reader::Bm25Stats,
            vector::{distance::Metric, rerank_codec::RerankCodec},
        },
        supertable::{
            Supertable, SupertableOptions,
            error::QueryError,
            manifest::{
                ClusterCentroids, ManifestSnapshot,
                list::{CellRoutingParams, PartitionStrategy},
            },
            writer::{recalibrate_probe_laws, split_overflow_cell},
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
    /// A user column that shadows a free name must disable the fast
    /// path entirely. Nothing rejects a column called `score` (or one
    /// matching the id column) at create time, and when one exists the
    /// general path's `index_of` resolves the name to the USER column
    /// and decodes real data — so answering the same projection with
    /// the synthesized value would be a different RESULT, not just a
    /// different route.
    #[test]
    fn a_user_column_shadowing_a_free_name_declines_the_fast_path() {
        let id = "doc_id";
        let clean = Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("body", DataType::LargeUtf8, false),
        ]);
        assert!(free_columns_unambiguous(&clean, id));

        let shadows_score = Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(SCORE_COLUMN, DataType::Float32, false),
        ]);
        assert!(
            !free_columns_unambiguous(&shadows_score, id),
            "a visible `score` column must decline the fast path"
        );

        let shadows_id = Schema::new(vec![Field::new(id, DataType::Int64, false)]);
        assert!(
            !free_columns_unambiguous(&shadows_id, id),
            "a user column matching the id column must decline the fast path"
        );

        // An id column named `score` collapses the two free names into
        // one. The general path answers `score` with the id (index_of
        // takes the first field, which is the id), so declining here
        // changes no result — it keeps the classifier from answering an
        // ambiguous request and stops the paths drifting.
        assert_eq!(free_column_slot(SCORE_COLUMN, SCORE_COLUMN), None);
        assert!(!free_columns_unambiguous(&clean, SCORE_COLUMN));
    }

    /// The SQL TVF classifies by DataFusion column INDEX and the public
    /// API by NAME, so they cannot share a signature — but they must
    /// share the RULE. Both resolve through [`free_column_slot`]; this
    /// pins that the index path (index -> field name -> slot) lands on
    /// exactly what the name path returns, for every projection shape.
    /// A third free column added to `free_column_slot` is then picked
    /// up by both without touching either call site.
    #[test]
    fn tvf_index_classification_matches_the_name_path() {
        let id = "doc_id";
        // The TVF's output schema: scalar columns, `score` appended.
        let names = [id, "title", "body", SCORE_COLUMN];
        // Index path: what exec::vector_exec derives per requested index.
        let by_index = |requested: &[usize]| -> Option<Vec<usize>> {
            requested
                .iter()
                .map(|&i| names.get(i).and_then(|n| free_column_slot(n, id)))
                .collect()
        };
        for requested in [
            vec![0, 3],    // _id + score
            vec![3, 0],    // reversed
            vec![0],       // _id alone
            vec![3],       // score alone
            vec![0, 1, 3], // includes a user column
            vec![1],       // user column alone
        ] {
            let as_names: Vec<&str> = requested.iter().map(|&i| names[i]).collect();
            assert_eq!(
                by_index(&requested),
                id_score_projection_indices(Some(&as_names), id),
                "index and name classification disagree for {requested:?}"
            );
        }
    }

    /// ANY subset/order of `_id` + `score` takes the fast path, and the
    /// returned indices reproduce the requested order — `_id` alone is
    /// the regression: it used to miss the exact-pair match and pay a
    /// placement resolve for ids already stamped on the hits. A name
    /// needing a user data page (or an empty projection) declines.
    #[test]
    fn id_score_projection_admits_any_subset_of_id_and_score() {
        let id = "doc_id";
        assert_eq!(id_score_projection_indices(None, id), Some(vec![0, 1]));
        assert_eq!(
            id_score_projection_indices(Some(&[id, SCORE_COLUMN]), id),
            Some(vec![0, 1])
        );
        assert_eq!(
            id_score_projection_indices(Some(&[SCORE_COLUMN, id]), id),
            Some(vec![1, 0])
        );
        assert_eq!(id_score_projection_indices(Some(&[id]), id), Some(vec![0]));
        assert_eq!(
            id_score_projection_indices(Some(&[SCORE_COLUMN]), id),
            Some(vec![1])
        );
        assert_eq!(
            id_score_projection_indices(Some(&[id, SCORE_COLUMN, id]), id),
            Some(vec![0, 1, 0])
        );
        assert_eq!(
            id_score_projection_indices(Some(&["other", SCORE_COLUMN]), id),
            None
        );
        assert_eq!(id_score_projection_indices(Some(&[]), id), None);
    }

    /// The admit window scales with the ranked cell population (the
    /// write-side 20% slice — round 0 of admission) and never narrows
    /// below the validated floor: small tables degenerate to
    /// exact-everything, the 256-cell shape widens just past its
    /// measured 48, and larger grids grow proportionally.
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

    /// The self-measured admit extension (#515) follows the query's own
    /// evidence: with a flat estimate spectrum and an observed
    /// estimate-to-exact residual, the near-tie run past the write
    /// window qualifies; a cliff-scored spectrum admits nothing. The
    /// residual floor is measured from already-scored cells, so with no
    /// exact scores in hand the round admits nothing (no guess).
    #[test]
    fn admit_extension_round_follows_evidence() {
        // Cells 1..=6 ranked by estimate; 1-3 admitted and exactly
        // scored. Exact = estimate + 0.05 everywhere (residual floor
        // 0.05). Winner exact 0.15 → serve threshold 0.30 (window 100%
        // for arithmetic clarity).
        let ranking = vec![
            (1u32, 0.10f32),
            (2, 0.12),
            (3, 0.14),
            (4, 0.20),
            (5, 0.24),
            (6, 0.80),
        ];
        let admitted: HashSet<u32> = [1, 2, 3].into_iter().collect();
        let exacts: HashMap<u32, f32> = [(1u32, 0.15f32), (2, 0.17), (3, 0.19)].into();
        // Flat spectrum: cells 4 and 5 could land inside the window
        // (estimate + 0.05 ≤ 0.30); the far cell 6 cannot.
        assert_eq!(
            admit_extension_round(&ranking, &admitted, &exacts, 0.30),
            vec![4, 5]
        );
        // Cliff: a tight threshold admits nothing.
        assert!(admit_extension_round(&ranking, &admitted, &exacts, 0.16).is_empty());
        // No exact scores observed → no residual measurement → nothing.
        assert!(admit_extension_round(&ranking, &admitted, &HashMap::new(), 0.30).is_empty());
    }

    /// The law arm (#515) serves the stamped width as a FLOOR — cliff
    /// scores never shrink it (the law contract) — and follows the
    /// near-tie run beyond it: extension cells (past the floor, not
    /// grid picks) read at bounded pre-pin depth, floor cells at
    /// whole-cell depth as certified.
    #[test]
    fn law_floor_serve_selection_extends_past_the_served_floor() {
        // Cliff after the floor: exactly the floor is served (grid
        // first in probe order), no extension.
        let cliff = vec![(7u32, 0.10f32), (8, 0.90), (9, 0.95)];
        let (cells, ext) = law_floor_serve_selection(&cliff, &[7, 8], 2, 0.20);
        assert_eq!(cells, vec![7, 8]);
        assert!(ext.is_empty());
        // Flat run: the near-tie run past the floor is served and lands
        // in the extension set; floor cells do not.
        let flat = vec![(7u32, 0.10f32), (8, 0.11), (9, 0.12)];
        let (cells, ext) = law_floor_serve_selection(&flat, &[7, 8], 2, 0.20);
        assert_eq!(cells, vec![7, 8, 9]);
        assert_eq!(ext, [9u32].into_iter().collect());
        // A grid pick past the floor's fine ranks is served via the
        // union but never depth-bounded (it is a law pick, not
        // evidence): extension excludes grid cells.
        let (cells, ext) = law_floor_serve_selection(&flat, &[9], 2, 0.20);
        assert_eq!(cells, vec![9, 7, 8]);
        assert!(ext.is_empty());
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
        let ids =
            block_on(hidden_hits_user_ids(manifest, &hits, "_id", &None)).expect("resolve one id");
        assert_eq!(ids, vec![42], "single inline stable id returned verbatim");

        // Order is preserved across multiple stamped hits.
        let hits = [mk(42), mk(7)];
        let ids =
            block_on(hidden_hits_user_ids(manifest, &hits, "_id", &None)).expect("resolve two ids");
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

    /// #515 serve-window extension cells never inherit the pin's whole-cell
    /// depth: under a MAX `keep_floor` (the law-pinned arm), a cell named in
    /// `extension_depth` keeps only its bounded floor while pinned cells are
    /// still read in full. This is the regression guard for the defect where
    /// evidence extension silently multiplied whole-cell reads.
    #[test]
    fn extension_cells_keep_bounded_depth_under_pin() {
        // Cell 0 (law-pinned): three runs. Cell 1 (serve-window extension):
        // three runs, slightly worse scores.
        let candidates = vec![
            (0usize, 10u32, 0.10f32, Some(0u32), 5u64),
            (0, 11, 0.11, Some(0), 5),
            (0, 12, 0.12, Some(0), 5),
            (0, 20, 0.20, Some(1), 5),
            (0, 21, 0.21, Some(1), 5),
            (0, 22, 0.22, Some(1), 5),
        ];
        let selected: HashSet<u32> = [0, 1].into_iter().collect();
        let selected_ordered = [0u32, 1];
        let extension: HashSet<u32> = [1].into_iter().collect();
        let candidate_counts: HashMap<(usize, u32), u64> = candidates
            .iter()
            .map(|(si, cluster, _, _, count)| ((*si, *cluster), *count))
            .collect();
        let mut scored = Vec::new();
        let gated = gate_fine_candidates_by_fragment(
            candidates,
            &selected,
            &selected_ordered,
            usize::MAX, // keep_floor: the pin's whole-cell depth
            0.0,        // keep_pct: floor-only for this test
            1,          // gated_target: tiny so the refill can't mask the bound
            &candidate_counts,
            &mut scored,
            None,
            Some((&extension, 1)), // extension cell bounded to one run
        );
        // The pinned cell is read in full (all three runs).
        let pinned: HashSet<u32> = gated
            .iter()
            .filter(|(_, c, _)| *c < 20)
            .map(|(_, c, _)| *c)
            .collect();
        assert_eq!(pinned, [10u32, 11, 12].into_iter().collect());
        // The extension cell keeps exactly its bounded floor: its best run.
        let extended: Vec<u32> = gated
            .iter()
            .filter(|(_, c, _)| *c >= 20)
            .map(|(_, c, _)| *c)
            .collect();
        assert_eq!(
            extended,
            vec![20],
            "extension cell read past its bounded depth: {gated:?}"
        );
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
            None,
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
            None,
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
            None,
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

    // ---- Gapped id->local placement cache: direct-call coverage (#556) ----
    //
    // `lookup_user_placements_by_id` resolves each vector hit's stable `_id`
    // to its owning (superfile, local-row). Contiguous superfiles resolve by
    // span arithmetic; a GAPPED superfile (drained/cell-packed, or any span
    // != n_docs) has no stable_id->local index, so the reverse map is built
    // by reading its `_id` column once and memoized per (immutable) superfile
    // uri. These tests pin the placement result across both kinds and the
    // cache contract: built once then reused, contiguous entries never cached,
    // and eviction to the live read set (no unbounded growth).

    /// One plain (non-cell) superfile whose `_id` column holds exactly `ids`,
    /// serialized. Mirrors `build_oracle_superfile` but with an explicit
    /// (possibly gapped) id set.
    fn superfile_bytes_with_ids(ids: &[i128], dim: usize) -> bytes::Bytes {
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
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
                provided_centroids: None,
            }],
            Some(tok()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let id_arr = Decimal128Array::from(ids.to_vec())
            .with_precision_and_scale(
                crate::supertable::options::DECIMAL128_PRECISION,
                crate::supertable::options::DECIMAL128_SCALE,
            )
            .expect("decimal128");
        let titles = LargeStringArray::from(
            (0..ids.len())
                .map(|i| format!("doc {i}"))
                .collect::<Vec<_>>(),
        );
        let scalar_batch =
            RecordBatch::try_new(scalar_schema, vec![Arc::new(id_arr), Arc::new(titles)])
                .expect("scalar batch");
        let mut flat = Vec::<f32>::with_capacity(ids.len() * dim);
        for i in 0..ids.len() {
            for d in 0..dim {
                flat.push(if d == i % dim { 1.0 } else { 0.0 });
            }
        }
        b.add_batch(&scalar_batch, &[flat.as_slice()])
            .expect("add_batch");
        bytes::Bytes::from(b.finish().expect("finish"))
    }

    /// A gapped `SuperfileEntry` (span != n_docs, so
    /// `row_id_from_manifest_entry` returns `None`) carrying `ids`, with its
    /// bytes registered in `store` under a `seed`-derived uri.
    fn insert_gapped_entry(
        ids: &[i128],
        dim: usize,
        store: &Arc<dyn crate::supertable::reader_cache::SuperfileReaderCache>,
        seed: u128,
    ) -> Arc<SuperfileEntry> {
        let id = Uuid::from_u128(seed);
        let uri = SuperfileUri(id);
        store
            .insert(uri, superfile_bytes_with_ids(ids, dim))
            .expect("insert superfile bytes");
        Arc::new(SuperfileEntry {
            birth_version: 0,
            superfile_id: id,
            uri,
            n_docs: ids.len() as u64,
            id_min: *ids.iter().min().expect("nonempty ids"),
            id_max: *ids.iter().max().expect("nonempty ids"),
            scalar_stats: std::collections::HashMap::new(),
            fts_summary: std::collections::HashMap::new(),
            vector_summary: std::collections::HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: crate::superfile::vector::layout::VectorLayout::Ivf,
            subsection_offsets: None,
        })
    }

    /// A contiguous `SuperfileEntry` (span == n_docs) spanning
    /// `id_min..id_min+n_docs`. Resolved by arithmetic, never read, never
    /// cached, so it needs no bytes in the store.
    fn contiguous_entry(id_min: i128, n_docs: u64, seed: u128) -> Arc<SuperfileEntry> {
        let id = Uuid::from_u128(seed);
        Arc::new(SuperfileEntry {
            birth_version: 0,
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs,
            id_min,
            id_max: id_min + n_docs as i128 - 1,
            scalar_stats: std::collections::HashMap::new(),
            fts_summary: std::collections::HashMap::new(),
            vector_summary: std::collections::HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: crate::superfile::vector::layout::VectorLayout::Ivf,
            subsection_offsets: None,
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn gapped_id_placement_cache_places_reuses_and_prunes_monotonically() {
        let dim = 16;
        let opts = options_one_superfile_per_commit(dim);
        let store = Arc::clone(&opts.store);
        let opts = Arc::new(opts);

        // Gapped: ids {0,5,10} -> span 11 != n_docs 3.
        let gapped = insert_gapped_entry(&[0, 5, 10], dim, &store, 0xA1);
        // Contiguous: ids {100,101,102} -> span 3 == n_docs 3 (arithmetic).
        let contig = contiguous_entry(100, 3, 0xC0);

        let m = ManifestSnapshot::new(
            1,
            Arc::clone(&opts),
            vec![Arc::clone(&gapped), Arc::clone(&contig)],
            None,
            None,
        );

        // Placement is correct for both kinds (same result the old full-scan
        // produced): gapped ids resolve to their Parquet row, contiguous by
        // arithmetic.
        let got = super::lookup_user_placements_by_id(&m, &[0, 5, 10, 100], &None)
            .await
            .expect("placements");
        assert_eq!((got[0].0.uri, got[0].1), (gapped.uri, 0));
        assert_eq!((got[1].0.uri, got[1].1), (gapped.uri, 1));
        assert_eq!((got[2].0.uri, got[2].1), (gapped.uri, 2));
        assert_eq!((got[3].0.uri, got[3].1), (contig.uri, 0));

        // Only the gapped superfile is cached; contiguous never is.
        let first_index = {
            let cache = opts.gapped_id_placement_cache.lock().await;
            assert_eq!(
                cache.entries.len(),
                1,
                "only the gapped superfile is cached"
            );
            assert!(cache.entries.contains_key(&gapped.uri));
            assert!(
                !cache.entries.contains_key(&contig.uri),
                "contiguous entries resolve by arithmetic and are never cached"
            );
            Arc::clone(
                cache
                    .entries
                    .get(&gapped.uri)
                    .expect("cell")
                    .get()
                    .expect("index built"),
            )
        };

        // A second query reuses the same index (built once, not per query).
        super::lookup_user_placements_by_id(&m, &[5], &None)
            .await
            .expect("2nd query");
        {
            let cache = opts.gapped_id_placement_cache.lock().await;
            let second_index = Arc::clone(
                cache
                    .entries
                    .get(&gapped.uri)
                    .expect("cell")
                    .get()
                    .expect("index built"),
            );
            assert!(
                Arc::ptr_eq(&first_index, &second_index),
                "cached index reused, not rebuilt"
            );
        }

        // A NEWER generation whose live set drops the old gapped superfile
        // evicts its map (bounded to the live read set, no lifetime growth).
        let gapped2 = insert_gapped_entry(&[200, 205, 210], dim, &store, 0xB2);
        let m_next =
            ManifestSnapshot::new(2, Arc::clone(&opts), vec![Arc::clone(&gapped2)], None, None);
        super::lookup_user_placements_by_id(&m_next, &[200, 210], &None)
            .await
            .expect("next-gen query");
        {
            let cache = opts.gapped_id_placement_cache.lock().await;
            assert!(
                !cache.entries.contains_key(&gapped.uri),
                "superseded gapped superfile is evicted by the newer generation"
            );
            assert!(
                cache.entries.contains_key(&gapped2.uri),
                "the live gapped superfile is cached"
            );
            assert_eq!(cache.entries.len(), 1);
            assert_eq!(cache.pruned_through, 2);
        }

        // An OLDER generation (version 1 < the watermark 2) must NOT evict the
        // newer generation's map, even though gapped2 isn't in its live set --
        // otherwise concurrent MVCC snapshots ping-pong rebuilds. It only
        // rebuilds its own gapped map.
        super::lookup_user_placements_by_id(&m, &[0], &None)
            .await
            .expect("older-gen query");
        {
            let cache = opts.gapped_id_placement_cache.lock().await;
            assert!(
                cache.entries.contains_key(&gapped2.uri),
                "older generation did not evict the newer generation's map"
            );
            assert!(
                cache.entries.contains_key(&gapped.uri),
                "older generation rebuilt its own map"
            );
            assert_eq!(cache.entries.len(), 2);
            assert_eq!(
                cache.pruned_through, 2,
                "prune watermark stays at the newest generation seen"
            );
        }
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
                true,
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
            select_global_shortlist(v, 3, 0)
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

    /// The rerank-law application gates: the measured budget converts to a
    /// multiplier ONLY on the unfiltered hidden path with no caller
    /// override — a caller `rerank_mult` always wins, filtered queries
    /// keep their own budget model, non-hidden tables never consult the
    /// law, and an uncalibrated `k` (cleared high-k point, or `k` past the
    /// knot table) falls back to the configured default rather than a
    /// clamped-down budget. Guards the silent failure modes: dropping the
    /// caller-override gate would override caller intent, dropping the
    /// `!filtered` gate would mis-budget filtered queries — both with zero
    /// other test failures.
    #[test]
    fn rerank_law_yields_to_caller_filter_and_uncalibrated_k() {
        let routing = CellRoutingParams {
            rerank_for_k: [40, 320, 2400, 0],
            ..CellRoutingParams::default()
        };
        let r = Some(&routing);
        // Engages: unfiltered hidden path, no caller override, calibrated k.
        // k=10 budget 320 -> equivalent multiplier ceil(320/10) = 32.
        assert_eq!(rerank_mult_from_law(true, false, None, r, 10), Some(32));
        // Caller override wins.
        assert_eq!(rerank_mult_from_law(true, false, Some(8), r, 10), None);
        // Filtered queries keep their own budget model.
        assert_eq!(rerank_mult_from_law(true, true, None, r, 10), None);
        // Non-hidden tables never consult the law.
        assert_eq!(rerank_mult_from_law(false, false, None, r, 10), None);
        // Uncalibrated k (the k=1000 point cleared): default fallback, not
        // a clamp down to the k=100 budget.
        assert_eq!(rerank_mult_from_law(true, false, None, r, 1000), None);
        // No routing at all (user table, undrained): default fallback.
        assert_eq!(rerank_mult_from_law(true, false, None, None, 10), None);
    }

    /// The per-cell floor rescues each scanned cell's best candidates from
    /// global eviction: a cell whose candidates all rank below the global
    /// cut still lands its `cell_floor` best in the kept set, counting any
    /// of its candidates already kept globally against the floor.
    #[test]
    fn select_global_shortlist_cell_floor_rescues_swamped_cells() {
        let cand = |est: f32, cell: usize, pos: u32, did: u32| ScanCandidate {
            did,
            estimate: est,
            pos,
            cluster_id: 0,
            cell_idx: cell,
        };
        // Cell 0 floods the pool with high estimates; cell 1 holds the
        // (lower-estimate) true neighbors the fixed cut would evict.
        let mut pooled: Vec<(usize, ScanCandidate)> =
            (0..10).map(|i| (0usize, cand(0.9, 0, i, i))).collect();
        pooled.push((0, cand(0.30, 1, 100, 100)));
        pooled.push((0, cand(0.20, 1, 101, 101)));
        pooled.push((0, cand(0.10, 1, 102, 102)));

        let kept = select_global_shortlist(pooled.clone(), 4, 0);
        assert!(
            kept.iter().all(|(_, c)| c.cell_idx == 0),
            "without a floor the flooded cell evicts cell 1 entirely"
        );

        let kept = select_global_shortlist(pooled, 4, 2);
        let cell1: Vec<u32> = kept
            .iter()
            .filter(|(_, c)| c.cell_idx == 1)
            .map(|(_, c)| c.did)
            .collect();
        assert_eq!(
            cell1,
            vec![100, 101],
            "the floor keeps cell 1's two best despite global eviction"
        );
        assert_eq!(
            kept.iter().filter(|(_, c)| c.cell_idx == 0).count(),
            4,
            "the global prefix is untouched by the rescue"
        );
    }

    /// Monotonicity by construction: widening the sweep (adding a new
    /// cell's candidates to the pool) never evicts another cell's floored
    /// survivors — the exact property whose absence inverts recall as
    /// nprobe grows.
    #[test]
    fn select_global_shortlist_widening_never_evicts_floored() {
        let cand = |est: f32, cell: usize, pos: u32, did: u32| ScanCandidate {
            did,
            estimate: est,
            pos,
            cluster_id: 0,
            cell_idx: cell,
        };
        const FLOOR: usize = 2;
        const LIMIT: usize = 4;
        // Narrow sweep: cells 0 and 1.
        let narrow: Vec<(usize, ScanCandidate)> = vec![
            (0, cand(0.9, 0, 0, 0)),
            (0, cand(0.8, 0, 1, 1)),
            (0, cand(0.4, 1, 2, 2)),
            (0, cand(0.3, 1, 3, 3)),
        ];
        // Wide sweep: cell 2 floods with better estimates than cell 1's.
        let mut wide = narrow.clone();
        for i in 0..8u32 {
            wide.push((0, cand(0.7, 2, 10 + i, 10 + i)));
        }
        let keep_ids = |v: Vec<(usize, ScanCandidate)>| {
            let mut ids: Vec<u32> = select_global_shortlist(v, LIMIT, FLOOR)
                .into_iter()
                .map(|(_, c)| c.did)
                .collect();
            ids.sort_unstable();
            ids
        };
        let narrow_kept = keep_ids(narrow);
        let wide_kept = keep_ids(wide);
        for id in &narrow_kept {
            assert!(
                wide_kept.contains(id),
                "widening dropped candidate {id}: floored survivors must \
                 be immune to added cells (kept narrow {narrow_kept:?} vs \
                 wide {wide_kept:?})"
            );
        }
    }

    /// Brute-force oracle for the grouped-walk floor: on a seeded
    /// pseudo-random pool spanning several units and cells, grouped
    /// per (unit, cell) as the scan wave emits it, the kept set must
    /// equal the definition computed naively — the global top-`limit`
    /// by the total order, unioned with every (unit, cell)'s
    /// `cell_floor` best. On an UNGROUPED pool (the documented
    /// degradation) the kept set must be a superset of that definition:
    /// split groups keep more, never fewer.
    #[test]
    fn select_global_shortlist_matches_naive_floor_definition() {
        // Deterministic LCG (numerical-recipes constants) — no RNG dep,
        // stable across runs.
        let mut state: u64 = 0x5eed_1234_abcd_9876;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        const POOL: usize = 4_000;
        const UNITS: usize = 3;
        const CELLS: usize = 17;
        const LIMIT: usize = 300;
        const FLOOR: usize = 9;
        let mut pooled: Vec<(usize, ScanCandidate)> = Vec::with_capacity(POOL);
        for i in 0..POOL {
            let r = next();
            // Coarse estimate buckets force plenty of exact ties so the
            // deterministic tie-break is exercised, not just f32 order.
            let est = ((r >> 32) % 64) as f32 / 64.0;
            pooled.push((
                (r % UNITS as u64) as usize,
                ScanCandidate {
                    did: i as u32,
                    estimate: est,
                    pos: i as u32,
                    cluster_id: 0,
                    cell_idx: ((r >> 8) % CELLS as u64) as usize,
                },
            ));
        }
        let cmp = |a: &(usize, ScanCandidate), b: &(usize, ScanCandidate)| {
            b.1.estimate.total_cmp(&a.1.estimate).then_with(|| {
                (a.0, a.1.cell_idx, a.1.pos, a.1.did).cmp(&(b.0, b.1.cell_idx, b.1.pos, b.1.did))
            })
        };
        // Naive reference: full sort, take the global prefix, then each
        // cell's floor-best from the full sorted order.
        let mut sorted = pooled.clone();
        sorted.sort_by(cmp);
        let mut expect: HashSet<u32> = sorted[..LIMIT].iter().map(|(_, c)| c.did).collect();
        let mut per_cell: HashMap<(usize, usize), usize> = HashMap::new();
        for (si, cand) in &sorted {
            let taken = per_cell.entry((*si, cand.cell_idx)).or_default();
            if *taken < FLOOR {
                expect.insert(cand.did);
                *taken += 1;
            }
        }
        // Grouped input (the production shape): exact equality.
        let mut grouped = pooled.clone();
        grouped.sort_by_key(|(si, c)| (*si, c.cell_idx));
        let kept: HashSet<u32> = select_global_shortlist(grouped, LIMIT, FLOOR)
            .into_iter()
            .map(|(_, c)| c.did)
            .collect();
        assert_eq!(kept, expect, "kept set must match the naive definition");
        // Ungrouped input (the documented degradation): a superset —
        // every guaranteed survivor still kept, extras allowed.
        let kept_ungrouped: HashSet<u32> = select_global_shortlist(pooled, LIMIT, FLOOR)
            .into_iter()
            .map(|(_, c)| c.did)
            .collect();
        assert!(
            kept_ungrouped.is_superset(&expect),
            "ungrouped pools must never lose a guaranteed survivor"
        );
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

    /// The public unranked `count` surface over a multi-superfile table —
    /// the count fan sums per-superfile match counts without scoring or
    /// row materialization. Fixture titles repeat "doc {0..31}" per
    /// commit, so token "5" counts one row per superfile and "doc"
    /// counts every row.
    #[test]
    fn count_sums_matches_across_superfiles() {
        let (_dir, st, _q, _k) = drained_three_direction_fixture();
        let reader = st.reader().expect("reader");
        assert_eq!(
            reader
                .count("title", "5", BoolMode::And)
                .expect("sparse count"),
            FIXTURE_COMMITS,
            "one match per superfile"
        );
        assert_eq!(
            reader
                .count("title", "doc", BoolMode::And)
                .expect("dense count"),
            (FIXTURE_COMMITS as usize * FIXTURE_ROWS_PER_COMMIT) as u64,
            "every row matches"
        );
    }

    /// BM25 with GLOBAL statistics over a fragmented table: the
    /// corpus-wide document count and per-term document frequencies are
    /// gathered across every superfile before scoring, so a term's idf —
    /// and a doc's score — does not depend on which superfile the doc
    /// landed in. The default per-superfile mode never runs that gather;
    /// this is the global mode's only end-to-end exercise.
    #[test]
    fn bm25_global_stats_scores_across_superfiles() {
        let (_dir, st, _q, _k) = drained_three_direction_fixture();
        let reader = st.reader().expect("reader");
        let batches = reader
            .bm25_search("title", "5", 8, BoolMode::And, Bm25Stats::Global, None)
            .expect("global-stats bm25");
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            rows, FIXTURE_COMMITS as usize,
            "token \"5\" lives in one row per superfile; global-idf \
             scoring must find all of them"
        );
    }

    /// Issue #512 regression: a cosine corpus ingested RAW (non-unit)
    /// must rank identically to the same corpus ingested unit-normalized,
    /// and a scaled query must return the same hits with the same
    /// calibrated scores. Before the ingest-normalize seam, the portable
    /// fixed Sq8 grid silently clamped out-of-range components at encode
    /// (measured −9.6 pts recall@10 on raw Cohere) and this asserted
    /// parity did not hold.
    #[test]
    fn raw_cosine_corpus_ranks_like_the_normalized_twin() {
        /// Scale applied to the raw twin — pushes many components far
        /// outside the fixed grid so pre-fix clamping is severe.
        const RAW_SCALE: f32 = 3.7;
        let dim = FIXTURE_DIM;
        let build = |scale: f32| {
            let schema = schema_with_vector(dim);
            let opts = options_one_superfile_per_commit(dim);
            let dir = tempfile::TempDir::new().expect("tempdir");
            let storage: Arc<dyn StorageProvider> =
                Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
            let st = Supertable::create(opts.with_storage(storage)).expect("create");
            let mut w = st.writer().expect("writer");
            // The one-hot fixture batch, scaled: every component of every
            // doc multiplies by `scale`, so the raw twin is far off the
            // unit sphere while directions are identical.
            let base = build_vector_batch(0, FIXTURE_ROWS_PER_COMMIT, dim, schema.clone());
            let emb = base
                .column(1)
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .expect("fsl")
                .clone();
            let scaled: Vec<f32> = emb
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("f32")
                .values()
                .iter()
                .map(|v| v * scale)
                .collect();
            let fsl = FixedSizeListArray::try_new(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    true,
                )),
                dim as i32,
                Arc::new(Float32Array::from(scaled)) as Arc<dyn Array>,
                None,
            )
            .expect("fsl scaled");
            let batch =
                RecordBatch::try_new(schema.clone(), vec![base.column(0).clone(), Arc::new(fsl)])
                    .expect("batch");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
            drop(w);
            st.drain_vectors_to_cells_sync().expect("drain");
            (dir, st)
        };
        let (_d_unit, unit) = build(1.0);
        let (_d_raw, raw) = build(RAW_SCALE);

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        q[1] = 0.7;
        let scaled_q: Vec<f32> = q.iter().map(|v| v * RAW_SCALE).collect();
        let unit_hits = unit
            .reader()
            .expect("reader")
            .vector_hits("emb", &q, 8, VectorSearchOptions::new(), None)
            .expect("unit search");
        let raw_hits = raw
            .reader()
            .expect("reader")
            .vector_hits("emb", &scaled_q, 8, VectorSearchOptions::new(), None)
            .expect("raw search");
        // Stable ids are per-table snowflakes; compare table-relative
        // positions (ids are minted sequentially over the same ingest
        // order, so offsets from the smallest returned id identify docs).
        let positions = |hits: &[SuperfileHit]| -> Vec<i128> {
            let ids: Vec<i128> = hits.iter().map(|h| h.stable_id.expect("id")).collect();
            let base = *ids.iter().min().expect("hits");
            ids.iter().map(|id| id - base).collect()
        };
        assert_eq!(
            positions(&unit_hits),
            positions(&raw_hits),
            "raw corpus + scaled query must rank exactly like the unit twin"
        );
        for (u, r) in unit_hits.iter().zip(raw_hits.iter()) {
            assert!(
                (u.score - r.score).abs() < 1e-3,
                "calibrated scores must match: {} vs {}",
                u.score,
                r.score
            );
        }
    }

    /// The PUBLIC predicate-filtered path end-to-end: a `VectorFilter`
    /// resolves its allow-set through the engine's own per-superfile
    /// `token_match` fan and the filtered kNN returns ONLY matching rows.
    /// First unit coverage of the public filter entry — the bench battery
    /// exercises it, but benches don't gate. Fixture titles repeat "doc
    /// {0..31}" per commit, so the token "5" matches exactly one row in
    /// each of the four commits, and a matching-all token ("doc") must
    /// reproduce a full top-k.
    ///
    /// Hit identity is checked against an `exact_match("doc 5")` oracle,
    /// never via arithmetic on the generated `_id`s: the fixture has no
    /// explicit id column, and a minted snowflake id's low sequence bits
    /// track row position only while each 32-row batch mints with the
    /// generator's per-millisecond sequence on a multiple of 32 and no
    /// millisecond tick lands mid-batch. The open-time handle-id mint
    /// sharing a millisecond with the first commit's batch is enough to
    /// shift every sequence by one, and parallel test load makes exactly
    /// that timing likely.
    #[test]
    fn vector_filter_restricts_hits_to_predicate_matches() {
        let (_dir, st, q, _k) = drained_three_direction_fixture();
        let reader = st.reader().expect("reader");
        let matched = reader
            .vector_hits(
                "emb",
                &q,
                10,
                VectorSearchOptions::new(),
                Some(VectorFilter {
                    column: "title",
                    query: "5",
                    mode: BoolMode::And,
                }),
            )
            .expect("sparse filtered search");
        assert_eq!(
            matched.len(),
            FIXTURE_COMMITS as usize,
            "the predicate matches one row per commit — nothing more"
        );
        // Identity oracle: resolve the predicate rows' stable ids through
        // the stored-text-verified exact-match surface — a different path
        // from the token_match fan the filter itself resolves through.
        let predicate_rows: HashSet<i128> = st
            .exact_match("title", "doc 5", Some(&["_id"]))
            .expect("exact-match oracle")
            .iter()
            .filter_map(|b| {
                b.column_by_name("_id")
                    .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
            })
            .flat_map(|c| c.values().iter().copied())
            .collect();
        assert_eq!(
            predicate_rows.len(),
            FIXTURE_COMMITS as usize,
            "oracle sanity: exactly one \"doc 5\" row per commit"
        );
        assert!(
            matched
                .iter()
                .all(|h| h.stable_id.is_some_and(|id| predicate_rows.contains(&id))),
            "every hit is a predicate row, not a nearest neighbor: {matched:?}"
        );

        let all = reader
            .vector_hits(
                "emb",
                &q,
                10,
                VectorSearchOptions::new(),
                Some(VectorFilter {
                    column: "title",
                    query: "doc",
                    mode: BoolMode::And,
                }),
            )
            .expect("match-all filtered search");
        assert_eq!(
            all.len(),
            10,
            "a predicate matching every row fills the full top-k"
        );
    }

    /// The recall target guarded end-to-end THROUGH the stamped laws:
    /// after a compaction-style reshape (splitting the cell holding the
    /// query's strongest direction) and `recalibrate_probe_laws`, a
    /// DEFAULT search — no caller knobs, every probe decision resolved
    /// from the restamped width/fine/rerank laws — must still recover
    /// the full planted top-k against the exact oracle. This is the
    /// assertion the law-shape checks cannot provide: a recalibration
    /// that under-stamps any law (a shallowing merge rule, a biased
    /// query sample, a clamped-down rerank budget) passes every
    /// nonzero/monotone check and fails only here, as lost recall.
    #[test]
    fn recalibrated_laws_serve_full_recall_at_default_search() {
        let (_dir, st, q, k) = drained_three_direction_fixture();
        let hidden = st
            .reader()
            .expect("reader")
            .vector_index_table()
            .expect("hidden index")
            .clone();
        // Split the cell carrying the query's strongest direction: its
        // planted neighbors now span TWO cells, so a stale pre-split law
        // (or an under-restamped one) leaves part of the top-k unprobed.
        let strategy = hidden
            .reader()
            .expect("hidden reader")
            .manifest()
            .get_partition_strategy();
        let PartitionStrategy::VectorCell { clusters, .. } = strategy else {
            panic!("hidden index must be VectorCell");
        };
        let mut direction = vec![0.0f32; FIXTURE_DIM];
        direction[0] = 1.0;
        let target_cell = clusters.nearest_cell(Metric::Cosine, &direction);
        hidden
            .block_on_query(split_overflow_cell(
                hidden.inner().clone(),
                target_cell,
                0.0,
            ))
            .expect("split")
            .expect("populated cell must split");
        let stamped = hidden
            .block_on_query(recalibrate_probe_laws(hidden.inner()))
            .expect("recalibrate");
        assert!(stamped, "the reshaped grid must restamp the laws");

        let hits = st
            .reader()
            .expect("reader")
            .vector_hits("emb", &q, k, VectorSearchOptions::new(), None)
            .expect("default search after recalibration");
        let near = near_count(&hits);
        assert_eq!(
            near, k,
            "recalibrated laws must serve the full {k} planted neighbors \
             at default settings, got {near}"
        );
    }

    /// A cell split retires the parent cell's blocks in place; its rows
    /// survive under the split's successor cells. The graph assemblers must
    /// skip the superseded parent, exactly as the ivf routing path does — or
    /// the same `stable_id` enters the graph twice (parent copy + child copy).
    /// At serve time `top_k_ascending` collapses the duplicates by id, so a
    /// plain top-`k` walk then returns fewer than `k` distinct rows, silently.
    /// Build a Sq16 table, split a populated cell so a superseded parent
    /// exists, then reassemble the graph exactly as the drain does and assert
    /// the split conserves the live population — one node per id, no
    /// duplicate stable ids.
    #[test]
    fn hnsw_assembly_skips_superseded_split_parent_cells() {
        use std::collections::BTreeSet;

        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_col_sq16(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        const COMMITS: u64 = 4;
        const ROWS_PER_COMMIT: usize = 64;
        const TOTAL: usize = COMMITS as usize * ROWS_PER_COMMIT;
        for c in 0..COMMITS {
            let mut w = st.writer().expect("writer");
            w.append(&build_vector_batch(
                c * ROWS_PER_COMMIT as u64,
                ROWS_PER_COMMIT,
                dim,
                schema.clone(),
            ))
            .expect("append");
            w.commit().expect("commit");
        }
        st.drain_vectors_to_cells_sync().expect("drain");

        let hidden = st
            .reader()
            .expect("reader")
            .vector_index_table()
            .expect("hidden index")
            .clone();

        // Reassemble the graph the way the drain does (over the CURRENT hidden
        // manifest) and return its per-node stable ids.
        let assemble_ids = |hidden: &Arc<Supertable>| -> Vec<i128> {
            let reader = hidden.reader().expect("hidden reader");
            let manifest = reader.manifest();
            let bundle = block_on(super::assemble_hnsw_sections(manifest, "emb", &None))
                .expect("assemble ok")
                .expect("sq16 rows must assemble into a graph");
            let decoded =
                crate::superfile::vector::hnsw::decode_hnsw(&bundle).expect("decode data bundle");
            assert_eq!(
                decoded.graph.len(),
                decoded.doc_ids.len(),
                "the graph has one node per stable id"
            );
            decoded.doc_ids
        };

        // Control: the freshly drained graph covers each live id exactly once.
        // The replica factor defaults to 1.0, so there are no intentional
        // duplicate nodes — any duplicate below is the superseded-parent bug.
        let before = assemble_ids(&hidden);
        let distinct_before: BTreeSet<i128> = before.iter().copied().collect();
        assert_eq!(
            before.len(),
            TOTAL,
            "graph covers every live id before split"
        );
        assert_eq!(
            distinct_before.len(),
            TOTAL,
            "no duplicate ids before split (replica factor 1.0)"
        );

        // Split the busiest populated cell so a superseded parent exists.
        let strategy = hidden
            .reader()
            .expect("hidden reader")
            .manifest()
            .get_partition_strategy();
        let PartitionStrategy::VectorCell { clusters, .. } = strategy else {
            panic!("hidden index must be VectorCell after drain");
        };
        let busiest = (0..clusters.n_cent)
            .filter(|&c| clusters.counts[c as usize] > 0)
            .max_by_key(|&c| clusters.counts[c as usize])
            .expect("a populated cell to split");
        hidden
            .block_on_query(split_overflow_cell(hidden.inner().clone(), busiest, 0.0))
            .expect("split call")
            .expect("a populated cell must split");
        assert!(
            hidden
                .reader()
                .expect("hidden reader")
                .manifest()
                .get_superseded_cells()
                .is_some_and(|m| m.values().any(|cells| cells.contains(&busiest))),
            "the split must mark the parent cell superseded"
        );

        // After the split the parent cell is superseded and its rows live under
        // the successor cells. A superseded-aware reassembly still covers each
        // live id exactly once; the buggy path would re-ingest the parent's
        // rows and inflate the node count past TOTAL with duplicate ids.
        let after = assemble_ids(&hidden);
        let distinct_after: BTreeSet<i128> = after.iter().copied().collect();
        assert_eq!(
            after.len(),
            TOTAL,
            "the split conserves the live population: still {TOTAL} nodes, not \
             parent+child duplicates ({} nodes)",
            after.len()
        );
        assert_eq!(
            distinct_after.len(),
            TOTAL,
            "no stable id may appear twice after the split"
        );
        assert_eq!(
            distinct_after, distinct_before,
            "the split repacks the same live ids, adds/removes none"
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

    /// Single vector column (`emb`), Sq16 rerank codec — so the drain builds
    /// and persists the resident per-row graph the hnsw serving path walks.
    fn options_one_col_sq16(dim: usize) -> SupertableOptions {
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
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq16,
                provided_centroids: None,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// The resident-graph (hnsw) arm emits `score` on the SAME cosine-distance
    /// scale as the ivf arm — `1 - dot`, non-negative, ~0 for a perfect match —
    /// not the graph's internal `-dot`. The two arms merge on raw score, so a
    /// mismatched scale ranks drained non-matches above an exact match and
    /// surfaces a negative distance in the public `score` column. Sq16 codec +
    /// a well-separated corpus so the drained graph registers and serves.
    #[test]
    fn hnsw_graph_arm_emits_cosine_scale_scores() {
        let dim = 32usize;
        let n = 256usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_col_sq16(dim);

        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, n, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        st.drain_vectors_to_cells_sync().expect("drain");

        // Exact match for the docs at direction 5 (id % dim == 5).
        let mut q = vec![0.0f32; dim];
        q[5] = 1.0;
        let hits = st
            .reader()
            .expect("reader")
            .vector_hits("emb", &q, 10, VectorSearchOptions::new(), None)
            .expect("post-drain graph search");
        assert!(!hits.is_empty(), "graph must return hits");
        for h in &hits {
            assert!(
                h.score >= 0.0,
                "cosine distance is non-negative; a negative score means the graph \
                 arm leaked its internal -dot: {}",
                h.score
            );
        }
        assert!(
            hits[0].score < 0.05,
            "an exact match is distance ~0 on the cosine scale, got {}",
            hits[0].score
        );
    }

    /// An undeclared vector column is a caller error and must be rejected on a
    /// DRAINED table too. Before the column validation was hoisted above the
    /// graph branch, a drained table answered an unknown-column query from the
    /// resident graph (which matched on dimension alone) instead of erroring —
    /// the same silent mis-answer a wrong same-dim column would get. The
    /// bundle now also stamps its column so the serving walk can reject a
    /// mismatch (see `hnsw_bundle_roundtrip`).
    #[test]
    fn hnsw_unknown_column_errors_on_drained_table() {
        let dim = 32usize;
        let n = 128usize;
        let schema = schema_with_vector(dim);
        let opts = options_one_col_sq16(dim);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(opts.with_storage(storage)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, n, dim, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        st.drain_vectors_to_cells_sync().expect("drain");

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let err = st
            .reader()
            .expect("reader")
            .vector_hits("does_not_exist", &q, 5, VectorSearchOptions::new(), None)
            .expect_err("an unknown column must error on a drained table, not serve the graph");
        assert!(
            format!("{err}").contains("unknown vector column"),
            "expected an unknown-column error, got {err}"
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

    /// Regression for the pre-drain hnsw collapse (recall 0.240): with
    /// `search_mode` defaulting to hnsw, a PRE-DRAIN table has no hidden index
    /// and no persisted graph, so the query must serve via ivf and return the
    /// exact match — not take the graph path and collapse onto `_id = 0`.
    /// Guards both the `hidden_vector_index` gate and the removed lazy build.
    #[test]
    fn pre_drain_hnsw_default_serves_correct_rows_via_ivf() {
        let dim = 16usize;
        let schema = schema_with_vector(dim);
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let st = Supertable::create(options_one_superfile_per_commit(dim).with_storage(storage))
            .expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_vector_batch(0, dim, dim, schema))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        // No drain: pre-drain query for the one-hot row at dim 3.
        let mut q = vec![0.0f32; dim];
        q[3] = 1.0;
        let hits = st
            .reader()
            .expect("reader")
            .vector_hits("emb", &q, 1, VectorSearchOptions::new(), None)
            .expect("pre-drain search");
        assert_eq!(
            hits.len(),
            1,
            "pre-drain query must serve via ivf (no graph pre-drain)"
        );
        assert!(
            hits[0].score < 1e-3,
            "top hit must be the exact one-hot match (distance ~0), not a collapse: {}",
            hits[0].score
        );
    }
}
