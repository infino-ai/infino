// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! SPFresh hidden-vector-index layout selection and vector blob codec.
//!
//! The environment selector chooses whether hidden vector-index superfiles use
//! the current nested IVF subsection or the SPFresh run subsection. The blob
//! codec below is the P1 storage primitive: a small directory of cell-local runs
//! followed by contiguous Sq8+epsilon row payloads.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    env,
    io::Error as IoError,
    ops::Range,
    sync::{Arc, OnceLock},
};

use bytes::Bytes;
use roaring::RoaringBitmap;

use crate::superfile::{
    BuildError, ReadError,
    error::VectorError,
    format::vec::{METRIC_ID_COSINE, METRIC_ID_L2SQ, METRIC_ID_NEGDOT},
    lazy_source::{LazyByteSource, LazyByteSourceError, Source},
    vector::{
        builder::{VectorConfig, derive_sq8_quantizer_from_min_max},
        cell_posting::{
            EncodedCellRow, MaterializedIvfRow, materialize_sq8_residual_row_into_cluster_quant,
        },
        distance::{
            Metric, SQ8_RESIDUAL_DIVISOR, Sq8ResidualEpsilonKernel, dequantize_sq8_residual_into,
            distance,
        },
        kmeans::kmeans_with_assignments,
        layout::VectorLayout,
    },
};

/// Environment variable selecting the hidden vector-index layout.
pub(crate) const HIDDEN_INDEX_LAYOUT_ENV: &str = "INFINO_HIDDEN_INDEX";

const MAGIC: &[u8] = b"infino.spfresh.v1\n";
const U32_BYTES: usize = 4;
const U64_BYTES: usize = 8;
const I128_BYTES: usize = 16;
const F32_BYTES: usize = 4;
const METRIC_BYTES: usize = 1;
const HEADER_RESERVED_BYTES: usize = 3;
const ROW_BYTES_PER_DIM: usize = 2;
const RUN_DIR_ENTRY_BYTES: usize = 24;
const HEADER_BYTES: usize =
    MAGIC.len() + U32_BYTES + METRIC_BYTES + HEADER_RESERVED_BYTES + U32_BYTES + U32_BYTES;
// A run's identity is its fine centroid (`cluster_id`); the coarse VectorCell is
// a superfile-level property (partition_hint / manifest cell tree), not stamped
// per run.
const RUN_CLUSTER_ID_OFF: usize = 0;
const RUN_ROW_COUNT_OFF: usize = RUN_CLUSTER_ID_OFF + U32_BYTES;
const RUN_BODY_OFFSET_OFF: usize = RUN_ROW_COUNT_OFF + U32_BYTES;
const RUN_BODY_LENGTH_OFF: usize = RUN_BODY_OFFSET_OFF + U64_BYTES;
const SQ8_CODE_MAX: f32 = 255.0;
const SQ8_RESIDUAL_I8_CLAMP: f32 = 127.0;

/// Environment override for the SPANN (1+eps) replication closure radius. `0.0`
/// (the default) means hard assignment — each vector is written to its single
/// nearest centroid, preserving pre-replication behavior. Positive values widen
/// the closure so boundary vectors replicate into several nearby centroids.
pub(crate) const REPLICATION_EPS_ENV: &str = "INFINO_REPLICATION_EPS";

/// Upper bound on replicas per vector, mirroring SPANN's closure cap. RNG
/// pruning normally keeps the count well below this; the cap only guards
/// pathological dense regions where many centroids fall inside the closure.
const REPLICA_CAP: usize = 8;

/// Target byte size of one cluster run — the load-bearing ~2 MB invariant that
/// keeps a single probe one cheap range GET regardless of corpus size. The fine
/// centroid count is derived from this target, never fixed.
const TARGET_RUN_BYTES: usize = 2 * 1024 * 1024;

/// Lloyd iterations when training fine centroids for one coarse cell at drain.
const FINE_KMEANS_ITERS: usize = 8;

/// Deterministic seed for fine-centroid training so a re-drain is reproducible.
const FINE_KMEANS_SEED: u64 = 0x5F1E_A17E;

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

#[derive(Debug, Clone)]
struct ColumnState {
    config: VectorConfig,
    ids: Vec<u32>,
    vectors: Vec<f32>,
    materialized_rows: Option<Vec<MaterializedIvfRow>>,
    next_local_id: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct SpfreshBlobBuilder {
    columns: Vec<ColumnState>,
}

#[derive(Debug, Clone)]
pub(crate) struct RunInput {
    pub(crate) cluster_id: u32,
    pub(crate) rows: Vec<MaterializedIvfRow>,
}

#[derive(Debug, Clone)]
pub(crate) struct SpfreshRun {
    pub(crate) cluster_id: u32,
    pub(crate) row_count: u32,
    body_range: Range<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct SpfreshRunProbe {
    pub(crate) run_id: usize,
    pub(crate) body_range: Option<Range<usize>>,
    pub(crate) row_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SpfreshSearchHit {
    pub(crate) local_doc_id: u32,
    pub(crate) stable_id: i128,
    pub(crate) score: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct SpfreshBlobReader {
    source: Source,
    dim: usize,
    metric: Metric,
    runs: Vec<SpfreshRun>,
    n_rows: u32,
}

#[derive(Debug, Clone, Copy)]
struct SpfreshHeader {
    dim: usize,
    metric: Metric,
    run_count: usize,
    n_rows: u32,
    directory_end: usize,
}

#[derive(Debug, Clone)]
struct EncodedRun {
    cluster_id: u32,
    row_count: u32,
    body: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Fp32RunInput {
    cluster_id: u32,
    local_ids: Vec<u32>,
    stable_ids: Vec<i128>,
    vectors: Vec<f32>,
}

impl Default for SpfreshBlobBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SpfreshBlobBuilder {
    pub(crate) fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    pub(crate) fn register_column(&mut self, config: VectorConfig) -> Result<(), BuildError> {
        if self
            .columns
            .iter()
            .any(|column| column.config.column == config.column)
        {
            return Err(BuildError::DuplicateLogicalName(config.column));
        }
        self.columns.push(ColumnState {
            config,
            ids: Vec::new(),
            vectors: Vec::new(),
            materialized_rows: None,
            next_local_id: 0,
        });
        Ok(())
    }

    pub(crate) fn add(&mut self, col_id: u32, vector: &[f32]) -> Result<(), BuildError> {
        let col = self
            .columns
            .get_mut(col_id as usize)
            .ok_or_else(|| BuildError::VectorSchemaMismatch(format!("column id {col_id}")))?;
        if vector.len() != col.config.dim {
            return Err(BuildError::VectorDimMismatch {
                column: col.config.column.clone(),
                expected: col.config.dim,
                actual: vector.len(),
            });
        }
        col.ids.push(col.next_local_id);
        col.next_local_id += 1;
        col.vectors.extend_from_slice(vector);
        Ok(())
    }

    pub(crate) fn load_materialized_rows(
        &mut self,
        col_id: u32,
        rows: Vec<MaterializedIvfRow>,
    ) -> Result<(), BuildError> {
        let col = self
            .columns
            .get_mut(col_id as usize)
            .ok_or_else(|| BuildError::VectorSchemaMismatch(format!("column id {col_id}")))?;
        col.materialized_rows = Some(rows);
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<Vec<u8>, BuildError> {
        if self.columns.is_empty() {
            return Ok(Vec::new());
        }
        if self.columns.len() != 1 {
            return Err(BuildError::VectorSchemaMismatch(
                "SPFresh vector blob supports exactly one vector column".into(),
            ));
        }
        let col = self.columns.into_iter().next().expect("checked one column");
        if let Some(rows) = col.materialized_rows {
            let runs = materialized_rows_to_runs(rows);
            return encode_materialized_runs(col.config.metric, col.config.dim, &runs);
        }
        let runs = fp32_rows_to_runs(&col);
        encode_fp32_runs(col.config.metric, col.config.dim, &runs)
    }
}

impl SpfreshBlobReader {
    pub(crate) fn open(bytes: Bytes) -> Result<Self, VectorError> {
        Self::open_with_source(Source::InMemory(bytes))
    }

    pub(crate) async fn open_lazy(source: Arc<dyn LazyByteSource>) -> Result<Self, VectorError> {
        let source = Source::Lazy(source);
        let header_bytes = source
            .range_async(0..HEADER_BYTES)
            .await
            .map_err(lazy_source_error)?;
        let header = parse_header(&header_bytes)?;
        let directory_bytes = source
            .range_async(HEADER_BYTES..header.directory_end)
            .await
            .map_err(lazy_source_error)?;
        let runs = parse_runs(&directory_bytes, header, source.len())?;
        Ok(Self {
            source,
            dim: header.dim,
            metric: header.metric,
            runs,
            n_rows: header.n_rows,
        })
    }

    fn open_with_source(source: Source) -> Result<Self, VectorError> {
        let header_bytes = fetch_sync(&source, 0..HEADER_BYTES, "header")?;
        let header = parse_header(&header_bytes)?;
        let directory_bytes = fetch_sync(&source, HEADER_BYTES..header.directory_end, "directory")?;
        let runs = parse_runs(&directory_bytes, header, source.len())?;
        Ok(Self {
            source,
            dim: header.dim,
            metric: header.metric,
            runs,
            n_rows: header.n_rows,
        })
    }

    #[cfg(test)]
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    #[cfg(test)]
    pub(crate) fn metric(&self) -> Metric {
        self.metric
    }

    #[cfg(test)]
    pub(crate) fn n_rows(&self) -> u32 {
        self.n_rows
    }

    pub(crate) fn runs(&self) -> &[SpfreshRun] {
        &self.runs
    }

    pub(crate) fn run_range(&self, run_idx: usize) -> Option<Range<usize>> {
        self.runs.get(run_idx).map(|run| run.body_range.clone())
    }

    #[cfg(test)]
    pub(crate) fn run_bytes(&self, run_idx: usize) -> Option<Bytes> {
        self.run_range(run_idx)
            .and_then(|range| self.source.get_range(range).ok())
    }

    #[cfg(test)]
    pub(crate) fn runs_for_cluster(&self, cluster_id: u32) -> Vec<usize> {
        self.runs
            .iter()
            .enumerate()
            .filter_map(|(idx, run)| (run.cluster_id == cluster_id).then_some(idx))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn search_runs(
        &self,
        run_ids: &[usize],
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(u32, f32)>, VectorError> {
        self.search_runs_filtered(run_ids, query, k, None, None)
    }

    #[cfg(test)]
    pub(crate) fn search_runs_filtered(
        &self,
        run_ids: &[usize],
        query: &[f32],
        k: usize,
        allow: Option<&RoaringBitmap>,
        deny: Option<&RoaringBitmap>,
    ) -> Result<Vec<(u32, f32)>, VectorError> {
        Ok(self
            .search_runs_filtered_with_stable_ids(run_ids, query, k, allow, deny)?
            .into_iter()
            .map(|hit| (hit.local_doc_id, hit.score))
            .collect())
    }

    #[cfg(test)]
    fn search_runs_filtered_with_stable_ids(
        &self,
        run_ids: &[usize],
        query: &[f32],
        k: usize,
        allow: Option<&RoaringBitmap>,
        deny: Option<&RoaringBitmap>,
    ) -> Result<Vec<SpfreshSearchHit>, VectorError> {
        let probes: Vec<SpfreshRunProbe> = run_ids
            .iter()
            .map(|&run_id| {
                let row_count = self.runs.get(run_id).map_or(0, |run| run.row_count);
                SpfreshRunProbe {
                    run_id,
                    body_range: None,
                    row_count,
                }
            })
            .collect();
        self.search_run_probes_filtered_with_stable_ids(&probes, query, k, allow, deny)
    }

    pub(crate) async fn search_run_probes_filtered_with_stable_ids_async(
        &self,
        probes: &[SpfreshRunProbe],
        query: &[f32],
        k: usize,
        allow: Option<&RoaringBitmap>,
        deny: Option<&RoaringBitmap>,
    ) -> Result<Vec<SpfreshSearchHit>, VectorError> {
        if query.len() != self.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let selected = self.selected_run_bodies(probes)?;
        let ranges: Vec<Range<usize>> = selected
            .iter()
            .map(|(range, _row_count)| range.clone())
            .collect();
        let bodies = self
            .source
            .get_ranges_parallel_async(&ranges)
            .await
            .map_err(lazy_source_error)?;
        let mut heap = BinaryHeap::<WorstHit>::new();
        for (body, (_range, row_count)) in bodies.into_iter().zip(selected) {
            score_run_body(
                &body,
                self.dim,
                self.metric,
                row_count,
                query,
                k,
                allow,
                deny,
                &mut heap,
            )?;
        }
        let mut out: Vec<SpfreshSearchHit> = heap.into_iter().map(|hit| hit.0).collect();
        out.sort_by(|a, b| cmp_f32(a.score, b.score));
        Ok(out)
    }

    #[cfg(test)]
    fn search_run_probes_filtered_with_stable_ids(
        &self,
        probes: &[SpfreshRunProbe],
        query: &[f32],
        k: usize,
        allow: Option<&RoaringBitmap>,
        deny: Option<&RoaringBitmap>,
    ) -> Result<Vec<SpfreshSearchHit>, VectorError> {
        if query.len() != self.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let selected = self.selected_run_bodies(probes)?;
        let ranges: Vec<Range<usize>> = selected
            .iter()
            .map(|(range, _row_count)| range.clone())
            .collect();
        let bodies = self
            .source
            .get_ranges_parallel(&ranges)
            .map_err(lazy_source_error)?;
        let mut heap = BinaryHeap::<WorstHit>::new();
        for (body, (_range, row_count)) in bodies.into_iter().zip(selected) {
            score_run_body(
                &body,
                self.dim,
                self.metric,
                row_count,
                query,
                k,
                allow,
                deny,
                &mut heap,
            )?;
        }
        let mut out: Vec<SpfreshSearchHit> = heap.into_iter().map(|hit| hit.0).collect();
        out.sort_by(|a, b| cmp_f32(a.score, b.score));
        Ok(out)
    }

    fn selected_run_bodies(
        &self,
        probes: &[SpfreshRunProbe],
    ) -> Result<Vec<(Range<usize>, usize)>, VectorError> {
        let mut selected = Vec::with_capacity(probes.len());
        for probe in probes {
            let run = self
                .runs
                .get(probe.run_id)
                .ok_or_else(|| malformed(format!("SPFresh run {} out of range", probe.run_id)))?;
            let row_count = if probe.body_range.is_some() {
                probe.row_count
            } else {
                run.row_count
            } as usize;
            let range = probe
                .body_range
                .clone()
                .unwrap_or_else(|| run.body_range.clone());
            let expected = run_body_len(self.dim, self.metric, row_count);
            if range.len() != expected {
                return Err(malformed(format!(
                    "SPFresh run {} range has {} bytes, expected {expected}",
                    probe.run_id,
                    range.len()
                )));
            }
            if range.end > self.source.len() {
                return Err(malformed(format!(
                    "SPFresh run {} range out of bounds",
                    probe.run_id
                )));
            }
            selected.push((range, row_count));
        }
        Ok(selected)
    }

    pub(crate) fn stable_ids_for_locals(&self, locals: &[u32]) -> Result<Vec<i128>, VectorError> {
        if locals.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(locals.len());
        let rows = self.materialized_rows()?;
        for local in locals {
            let stable_id = rows
                .iter()
                .find(|row| row.local_doc_id == *local)
                .map(|row| row.stable_id)
                .ok_or_else(|| malformed(format!("SPFresh local id {local} missing")))?;
            out.push(stable_id);
        }
        Ok(out)
    }

    pub(crate) fn materialized_rows(&self) -> Result<Vec<MaterializedIvfRow>, VectorError> {
        let mut out = Vec::with_capacity(self.n_rows as usize);
        for run in &self.runs {
            let body = self
                .source
                .get_range(run.body_range.clone())
                .map_err(lazy_source_error)?;
            decode_run_body_materialized(
                &body,
                self.dim,
                self.metric,
                run.cluster_id,
                run.row_count as usize,
                &mut out,
            )?;
        }
        Ok(out)
    }
}

pub(crate) fn encode_materialized_runs(
    metric: Metric,
    dim: usize,
    runs: &[RunInput],
) -> Result<Vec<u8>, BuildError> {
    let mut encoded = Vec::with_capacity(runs.len());
    for run in runs {
        encoded.push(encode_materialized_run(metric, dim, run)?);
    }
    encode_blob(metric, dim, &encoded)
}

fn encode_fp32_runs(
    metric: Metric,
    dim: usize,
    runs: &[Fp32RunInput],
) -> Result<Vec<u8>, BuildError> {
    let mut encoded = Vec::with_capacity(runs.len());
    for run in runs {
        encoded.push(encode_fp32_run(metric, dim, run)?);
    }
    encode_blob(metric, dim, &encoded)
}

fn encode_blob(metric: Metric, dim: usize, runs: &[EncodedRun]) -> Result<Vec<u8>, BuildError> {
    let directory_bytes = runs
        .len()
        .checked_mul(RUN_DIR_ENTRY_BYTES)
        .ok_or_else(|| BuildError::VectorSchemaMismatch("SPFresh directory overflow".into()))?;
    let mut body_offset = HEADER_BYTES + directory_bytes;
    let row_count = runs.iter().map(|run| run.row_count).sum::<u32>();
    let mut out =
        Vec::with_capacity(body_offset + runs.iter().map(|run| run.body.len()).sum::<usize>());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    out.push(metric_id(metric));
    out.extend_from_slice(&[0; HEADER_RESERVED_BYTES]);
    out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
    out.extend_from_slice(&row_count.to_le_bytes());
    for run in runs {
        out.extend_from_slice(&run.cluster_id.to_le_bytes());
        out.extend_from_slice(&run.row_count.to_le_bytes());
        out.extend_from_slice(&(body_offset as u64).to_le_bytes());
        out.extend_from_slice(&(run.body.len() as u64).to_le_bytes());
        body_offset += run.body.len();
    }
    for run in runs {
        out.extend_from_slice(&run.body);
    }
    Ok(out)
}

fn materialized_rows_to_runs(rows: Vec<MaterializedIvfRow>) -> Vec<RunInput> {
    let mut grouped: HashMap<u32, Vec<MaterializedIvfRow>> = HashMap::new();
    for row in rows {
        grouped.entry(row.cluster).or_default().push(row);
    }
    let mut groups: Vec<(u32, Vec<MaterializedIvfRow>)> = grouped.into_iter().collect();
    groups.sort_by_key(|(cluster, _)| *cluster);
    groups
        .into_iter()
        .map(|(cluster, rows)| RunInput {
            cluster_id: cluster,
            rows,
        })
        .collect()
}

fn fp32_rows_to_runs(col: &ColumnState) -> Vec<Fp32RunInput> {
    fp32_rows_to_runs_for_target(col, TARGET_RUN_BYTES)
}

fn fp32_rows_to_runs_for_target(col: &ColumnState, target_bytes: usize) -> Vec<Fp32RunInput> {
    let dim = col.config.dim;
    let n_rows = col.ids.len();
    if n_rows == 0 {
        return Vec::new();
    }
    let eps = replication_eps();
    let k_fine = fine_centroid_count_for_target(n_rows, dim, target_bytes);
    let (centroids, _) = kmeans_with_assignments(
        &col.vectors,
        dim,
        k_fine,
        FINE_KMEANS_ITERS,
        FINE_KMEANS_SEED,
    );
    let mut grouped: HashMap<u32, Fp32RunInput> = HashMap::new();
    for row_idx in 0..n_rows {
        let vector = &col.vectors[row_idx * dim..(row_idx + 1) * dim];
        // User superfiles train their own fine-centroid runs at the same ~2 MiB
        // target as hidden maintenance. The coarse global cell grid is only the
        // outer router; it must not become the run key.
        let cells = assign_replicas(col.config.metric, vector, dim, &centroids, eps);
        for cell in cells {
            let entry = grouped.entry(cell).or_insert_with(|| Fp32RunInput {
                cluster_id: cell,
                local_ids: Vec::new(),
                stable_ids: Vec::new(),
                vectors: Vec::new(),
            });
            entry.local_ids.push(col.ids[row_idx]);
            entry.stable_ids.push(i128::from(col.ids[row_idx]));
            entry.vectors.extend_from_slice(vector);
        }
    }
    let mut runs: Vec<Fp32RunInput> = grouped.into_values().collect();
    runs.sort_by_key(|run| run.cluster_id);
    runs
}

/// Replication closure radius, read once from [`REPLICATION_EPS_ENV`]. Default
/// `0.0` (hard assignment) until the replication rollout raises it.
pub(crate) fn replication_eps() -> f32 {
    static EPS: OnceLock<f32> = OnceLock::new();
    *EPS.get_or_init(|| {
        env::var(REPLICATION_EPS_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(0.0)
    })
}

/// Replica set for one vector: SPANN-style (1+eps) closure with RNG pruning over
/// the global fine-centroid set. Returns the fine-centroid ids the vector must
/// be written to, nearest first. Interior points return exactly one id; boundary
/// points return several (bounded by RNG pruning and [`REPLICA_CAP`]).
///
/// This is the single assignment authority shared by every path that writes
/// vector rows (commit, drain, compaction) so a row's replica set is identical
/// wherever it is written. Computing the closure over the *global* centroid set
/// (not per outer cell) is deliberate: a boundary vector replicates into
/// centroids owned by different outer `VectorCell`s, so the coarse router stays a
/// cost-only pre-filter and recall lives entirely in this closure.
pub(crate) fn assign_replicas(
    metric: Metric,
    vector: &[f32],
    dim: usize,
    centroids: &[f32],
    eps: f32,
) -> Vec<u32> {
    let n_cent = centroids.len() / dim;
    if n_cent == 0 {
        return Vec::new();
    }
    let centroid = |c: usize| &centroids[c * dim..(c + 1) * dim];

    let mut dists: Vec<(u32, f32)> = Vec::with_capacity(n_cent);
    let mut nearest_id = 0u32;
    let mut nearest_d = f32::INFINITY;
    for c in 0..n_cent {
        let d = distance(metric, vector, centroid(c));
        if d < nearest_d {
            nearest_d = d;
            nearest_id = c as u32;
        }
        dists.push((c as u32, d));
    }

    // eps <= 0: hard assignment — identical to a plain nearest-centroid argmin.
    if eps <= 0.0 {
        return vec![nearest_id];
    }

    // (1+eps) closure. `nearest_d + eps*|nearest_d|` equals `(1+eps)*nearest_d`
    // for the non-negative distances (L2Sq, Cosine) and widens in the correct
    // direction for signed NegDot scores, where a bare `(1+eps)*` multiply would
    // move the threshold the wrong way.
    let threshold = nearest_d + eps * nearest_d.abs();
    let mut candidates: Vec<(u32, f32)> =
        dists.into_iter().filter(|(_, d)| *d <= threshold).collect();
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    // RNG prune: keep a candidate only if no already-kept centroid is closer to
    // it than the vector is. A kept centroid nearer than the vector already
    // covers that region, so the replica would add no coverage.
    let mut kept: Vec<u32> = Vec::new();
    for (c, d_c) in candidates {
        if kept.len() >= REPLICA_CAP {
            break;
        }
        let dominated = kept
            .iter()
            .any(|&k| distance(metric, centroid(k as usize), centroid(c as usize)) < d_c);
        if !dominated {
            kept.push(c);
        }
    }
    kept
}

/// Per-row byte cost inside a run body: Sq8 codes + residuals, the local id, the
/// stable id, and a norm word (present for L2/Cosine; counted always so runs
/// stay at or under the target).
fn run_row_stride(dim: usize) -> usize {
    dim * ROW_BYTES_PER_DIM + U32_BYTES + I128_BYTES + F32_BYTES
}

fn parse_header(header: &[u8]) -> Result<SpfreshHeader, VectorError> {
    if header.len() < HEADER_BYTES {
        return Err(malformed("SPFresh blob header truncated"));
    }
    let actual = &header[..MAGIC.len()];
    if actual != MAGIC {
        return Err(VectorError::Read(ReadError::BadMagic {
            section: "vector/spfresh",
            expected: MAGIC,
            actual: actual.to_vec(),
        }));
    }
    let mut offset = MAGIC.len();
    let dim = read_u32_at(header, offset, "dim")? as usize;
    offset += U32_BYTES;
    let metric = metric_from_id(header[offset])?;
    offset += METRIC_BYTES + HEADER_RESERVED_BYTES;
    let run_count = read_u32_at(header, offset, "run_count")? as usize;
    offset += U32_BYTES;
    let n_rows = read_u32_at(header, offset, "row_count")?;
    let directory_bytes = run_count
        .checked_mul(RUN_DIR_ENTRY_BYTES)
        .ok_or_else(|| malformed("SPFresh run directory overflow"))?;
    let directory_end = HEADER_BYTES
        .checked_add(directory_bytes)
        .ok_or_else(|| malformed("SPFresh run directory overflow"))?;
    Ok(SpfreshHeader {
        dim,
        metric,
        run_count,
        n_rows,
        directory_end,
    })
}

fn parse_runs(
    directory: &[u8],
    header: SpfreshHeader,
    source_len: usize,
) -> Result<Vec<SpfreshRun>, VectorError> {
    let expected_directory_len = header
        .directory_end
        .checked_sub(HEADER_BYTES)
        .ok_or_else(|| malformed("SPFresh run directory underflow"))?;
    if directory.len() < expected_directory_len {
        return Err(malformed("SPFresh run directory truncated"));
    }
    let mut runs = Vec::with_capacity(header.run_count);
    for run_idx in 0..header.run_count {
        let entry = run_idx * RUN_DIR_ENTRY_BYTES;
        let cluster_id = read_u32_at(directory, entry + RUN_CLUSTER_ID_OFF, "cluster_id")?;
        let row_count = read_u32_at(directory, entry + RUN_ROW_COUNT_OFF, "row_count")?;
        let body_offset =
            read_u64_at(directory, entry + RUN_BODY_OFFSET_OFF, "body_offset")? as usize;
        let body_length =
            read_u64_at(directory, entry + RUN_BODY_LENGTH_OFF, "body_length")? as usize;
        let body_end = body_offset
            .checked_add(body_length)
            .ok_or_else(|| malformed("SPFresh run body overflow"))?;
        if body_offset < header.directory_end || body_end > source_len {
            return Err(malformed("SPFresh run body out of bounds"));
        }
        let expected = run_body_len(header.dim, header.metric, row_count as usize);
        if body_length != expected {
            return Err(malformed(format!(
                "SPFresh run body has {body_length} bytes, expected {expected}"
            )));
        }
        runs.push(SpfreshRun {
            cluster_id,
            row_count,
            body_range: body_offset..body_end,
        });
    }
    Ok(runs)
}

/// Number of fine centroids (~2 MB runs) for `n_rows`, derived from the run-size
/// invariant rather than a fixed constant. Always `>= 1` and never exceeds
/// `n_rows`.
fn fine_centroid_count_for_target(n_rows: usize, dim: usize, target_bytes: usize) -> usize {
    if n_rows <= 1 {
        return 1;
    }
    let per_run = (target_bytes / run_row_stride(dim)).max(1);
    n_rows.div_ceil(per_run).clamp(1, n_rows)
}

/// Train fine centroids over one coarse cell's `rows`, set each row's `cluster`
/// to its (within-cell) fine-centroid id (`0..k_fine`), and return the trained
/// fine centroids (`k_fine * dim` fp32). This is the eps=0 base assignment; the
/// drain layers boundary replication on top via [`assign_replicas`] over the
/// union of all cells' returned centroids. Fine centroids are trained per coarse
/// cell (ids cell-local); the manifest keys runs by the owning coarse cell.
pub(crate) fn assign_fine_clusters(rows: &mut [MaterializedIvfRow], dim: usize) -> Vec<f32> {
    assign_fine_clusters_for_target(rows, dim, TARGET_RUN_BYTES)
}

fn assign_fine_clusters_for_target(
    rows: &mut [MaterializedIvfRow],
    dim: usize,
    target_bytes: usize,
) -> Vec<f32> {
    if rows.is_empty() {
        return Vec::new();
    }
    let k_fine = fine_centroid_count_for_target(rows.len(), dim, target_bytes);
    let mut fp32 = vec![0f32; rows.len() * dim];
    for (i, row) in rows.iter().enumerate() {
        dequantize_row_into(row, dim, &mut fp32[i * dim..(i + 1) * dim]);
    }
    // k_fine == 1 yields the cell mean and all-zero assignments — the single-run
    // case, identical to the previous hard "everything to cluster 0" behavior.
    let (centroids, assignments) =
        kmeans_with_assignments(&fp32, dim, k_fine, FINE_KMEANS_ITERS, FINE_KMEANS_SEED);
    for (row, cluster) in rows.iter_mut().zip(assignments) {
        row.cluster = cluster;
    }
    centroids
}

/// Dequantize one materialized row's Sq8+eps payload to fp32 `out` (`dim`).
fn dequantize_row_into(row: &MaterializedIvfRow, _dim: usize, out: &mut [f32]) {
    dequantize_sq8_residual_into(
        &row.encoded.scale,
        &row.encoded.offset,
        &row.encoded.codes,
        &row.encoded.residuals,
        out,
    );
}

/// Replica fine-centroid ids for one row over a flat global fine-centroid set
/// (`global_centroids` = `n * dim` fp32), via [`assign_replicas`] on the
/// dequantized row. Returns indices into that global set. The drain maps each
/// index back to an owning coarse cell + cell-local fine id.
pub(crate) fn fine_replicas_for_row(
    metric: Metric,
    dim: usize,
    global_centroids: &[f32],
    row: &MaterializedIvfRow,
    eps: f32,
) -> Vec<u32> {
    let mut fp32 = vec![0f32; dim];
    dequantize_row_into(row, dim, &mut fp32);
    assign_replicas(metric, &fp32, dim, global_centroids, eps)
}

fn encode_materialized_run(
    metric: Metric,
    dim: usize,
    run: &RunInput,
) -> Result<EncodedRun, BuildError> {
    let row_count = run.rows.len();
    let (scale, offset) = derive_quantizer_from_materialized(dim, &run.rows)?;
    let mut rows = vec![0u8; row_count * dim * ROW_BYTES_PER_DIM];
    let mut local_ids = Vec::with_capacity(row_count);
    let mut stable_ids = Vec::with_capacity(row_count);
    let store_norm = stores_norms(metric);
    let mut norms = store_norm.then(|| Vec::with_capacity(row_count));
    for (idx, row) in run.rows.iter().enumerate() {
        local_ids.push(row.local_doc_id);
        stable_ids.push(row.stable_id);
        let start = idx * dim * ROW_BYTES_PER_DIM;
        let norm = materialize_sq8_residual_row_into_cluster_quant(
            &row.encoded,
            &scale,
            &offset,
            dim,
            &mut rows[start..start + dim * ROW_BYTES_PER_DIM],
            store_norm,
        );
        if let (Some(norms), Some(norm)) = (norms.as_mut(), norm) {
            norms.push(norm);
        }
    }
    Ok(EncodedRun {
        cluster_id: run.cluster_id,
        row_count: row_count as u32,
        body: encode_run_body(
            dim,
            metric,
            &scale,
            &offset,
            &rows,
            &local_ids,
            &stable_ids,
            norms.as_deref(),
        )?,
    })
}

fn encode_fp32_run(
    metric: Metric,
    dim: usize,
    run: &Fp32RunInput,
) -> Result<EncodedRun, BuildError> {
    let row_count = run.local_ids.len();
    let (scale, offset) = derive_quantizer_from_fp32(dim, &run.vectors);
    let mut rows = Vec::with_capacity(row_count * dim * ROW_BYTES_PER_DIM);
    let mut norms = stores_norms(metric).then(|| Vec::with_capacity(row_count));
    for row_idx in 0..row_count {
        let src = &run.vectors[row_idx * dim..(row_idx + 1) * dim];
        let mut acc = 0.0f64;
        for d in 0..dim {
            let q = if scale[d] > 0.0 {
                ((src[d] - offset[d]) / scale[d])
                    .round()
                    .clamp(0.0, SQ8_CODE_MAX) as u8
            } else {
                0
            };
            rows.push(q);
        }
        for d in 0..dim {
            let q = rows[row_idx * dim * ROW_BYTES_PER_DIM + d];
            let base = offset[d] + q as f32 * scale[d];
            let step = scale[d] / SQ8_RESIDUAL_DIVISOR;
            let eps = if step > 0.0 {
                ((src[d] - base) / step)
                    .round()
                    .clamp(-SQ8_RESIDUAL_I8_CLAMP, SQ8_RESIDUAL_I8_CLAMP) as i8
            } else {
                0
            };
            rows.push(eps.to_le_bytes()[0]);
            if stores_norms(metric) {
                let corrected = base + (eps as f32) * step;
                acc += (corrected as f64) * (corrected as f64);
            }
        }
        if let Some(norms) = norms.as_mut() {
            norms.push(acc as f32);
        }
    }
    Ok(EncodedRun {
        cluster_id: run.cluster_id,
        row_count: row_count as u32,
        body: encode_run_body(
            dim,
            metric,
            &scale,
            &offset,
            &rows,
            &run.local_ids,
            &run.stable_ids,
            norms.as_deref(),
        )?,
    })
}

fn derive_quantizer_from_materialized(
    dim: usize,
    rows: &[MaterializedIvfRow],
) -> Result<(Vec<f32>, Vec<f32>), BuildError> {
    if rows.is_empty() {
        return Ok((vec![1.0; dim], vec![0.0; dim]));
    }
    let mut min = vec![f32::INFINITY; dim];
    let mut max = vec![f32::NEG_INFINITY; dim];
    let mut row_fp = vec![0.0f32; dim];
    for row in rows {
        dequantize_row(&row.encoded, dim, &mut row_fp)?;
        for d in 0..dim {
            min[d] = min[d].min(row_fp[d]);
            max[d] = max[d].max(row_fp[d]);
        }
    }
    Ok(derive_sq8_quantizer_from_min_max(&min, &max))
}

fn derive_quantizer_from_fp32(dim: usize, vectors: &[f32]) -> (Vec<f32>, Vec<f32>) {
    if vectors.is_empty() {
        return (vec![1.0; dim], vec![0.0; dim]);
    }
    let mut min = vec![f32::INFINITY; dim];
    let mut max = vec![f32::NEG_INFINITY; dim];
    for row in vectors.chunks_exact(dim) {
        for d in 0..dim {
            min[d] = min[d].min(row[d]);
            max[d] = max[d].max(row[d]);
        }
    }
    derive_sq8_quantizer_from_min_max(&min, &max)
}

fn dequantize_row(row: &EncodedCellRow, dim: usize, out: &mut [f32]) -> Result<(), BuildError> {
    if row.codes.len() != dim || row.residuals.len() != dim {
        return Err(BuildError::VectorSchemaMismatch(
            "SPFresh materialized row dim mismatch".into(),
        ));
    }
    dequantize_sq8_residual_into(&row.scale, &row.offset, &row.codes, &row.residuals, out);
    Ok(())
}

fn encode_run_body(
    dim: usize,
    metric: Metric,
    scale: &[f32],
    offset: &[f32],
    rows: &[u8],
    local_ids: &[u32],
    stable_ids: &[i128],
    norms: Option<&[f32]>,
) -> Result<Vec<u8>, BuildError> {
    let row_count = local_ids.len();
    if stable_ids.len() != row_count || rows.len() != row_count * dim * ROW_BYTES_PER_DIM {
        return Err(BuildError::VectorSchemaMismatch(
            "SPFresh run body length mismatch".into(),
        ));
    }
    let mut out = Vec::with_capacity(run_body_len(dim, metric, row_count));
    for v in scale {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in offset {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(rows);
    for id in local_ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    for id in stable_ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    if stores_norms(metric) {
        let norms = norms.ok_or_else(|| {
            BuildError::VectorSchemaMismatch("SPFresh L2/Cosine run missing norms".into())
        })?;
        if norms.len() != row_count {
            return Err(BuildError::VectorSchemaMismatch(
                "SPFresh norm count mismatch".into(),
            ));
        }
        for norm in norms {
            out.extend_from_slice(&norm.to_le_bytes());
        }
    }
    Ok(out)
}

/// Byte offsets of the sub-regions inside one run body. A run body is laid out
/// as `scale | offset | rows | local_ids | stable_ids | norms?`; every reader
/// (`score_run_body`, `decode_run_body_materialized`) and the size computation
/// (`run_body_len`) derives its offsets here so the layout is written once.
struct RunBodyLayout {
    offset_start: usize,
    rows_start: usize,
    ids_start: usize,
    stable_ids_start: usize,
    norms_start: usize,
    total: usize,
}

impl RunBodyLayout {
    fn new(dim: usize, metric: Metric, row_count: usize) -> Self {
        let offset_start = dim * F32_BYTES;
        let rows_start = offset_start + dim * F32_BYTES;
        let ids_start = rows_start + row_count * dim * ROW_BYTES_PER_DIM;
        let stable_ids_start = ids_start + row_count * U32_BYTES;
        let norms_start = stable_ids_start + row_count * I128_BYTES;
        let total = norms_start
            + if stores_norms(metric) {
                row_count * F32_BYTES
            } else {
                0
            };
        Self {
            offset_start,
            rows_start,
            ids_start,
            stable_ids_start,
            norms_start,
            total,
        }
    }
}

fn score_run_body(
    body: &[u8],
    dim: usize,
    metric: Metric,
    row_count: usize,
    query: &[f32],
    k: usize,
    allow: Option<&RoaringBitmap>,
    deny: Option<&RoaringBitmap>,
    heap: &mut BinaryHeap<WorstHit>,
) -> Result<(), VectorError> {
    let layout = RunBodyLayout::new(dim, metric, row_count);
    let scale = read_f32_vec(body, 0, dim)?;
    let offset = read_f32_vec(body, layout.offset_start, dim)?;
    let norms = if stores_norms(metric) {
        Some(read_f32_vec(body, layout.norms_start, row_count)?)
    } else {
        None
    };
    let kernel = Sq8ResidualEpsilonKernel::new(
        metric,
        query,
        &scale,
        &offset,
        SQ8_RESIDUAL_DIVISOR,
        norms.as_deref(),
    );
    for row in 0..row_count {
        let row_base = layout.rows_start + row * dim * ROW_BYTES_PER_DIM;
        let id_base = layout.ids_start + row * U32_BYTES;
        let stable_id_base = layout.stable_ids_start + row * I128_BYTES;
        let local_id = read_u32_at(body, id_base, "local_id")?;
        if allow.is_some_and(|bitmap| !bitmap.contains(local_id))
            || deny.is_some_and(|bitmap| bitmap.contains(local_id))
        {
            continue;
        }
        let stable_id = read_i128_at(body, stable_id_base, "stable_id")?;
        let codes = &body[row_base..row_base + dim];
        let residuals = &body[row_base + dim..row_base + dim + dim];
        let norm = norms.as_ref().map(|values| values[row]);
        let dist = kernel.distance_with_norm(codes, residuals, norm);
        let hit = WorstHit(SpfreshSearchHit {
            local_doc_id: local_id,
            stable_id,
            score: dist,
        });
        if heap.len() < k {
            heap.push(hit);
        } else if let Some(worst) = heap.peek()
            && cmp_f32(hit.0.score, worst.0.score).is_lt()
        {
            heap.pop();
            heap.push(hit);
        }
    }
    Ok(())
}

fn decode_run_body_materialized(
    body: &[u8],
    dim: usize,
    metric: Metric,
    cluster_id: u32,
    row_count: usize,
    out: &mut Vec<MaterializedIvfRow>,
) -> Result<(), VectorError> {
    let layout = RunBodyLayout::new(dim, metric, row_count);
    let scale: Arc<[f32]> = read_f32_vec(body, 0, dim)?.into();
    let offset: Arc<[f32]> = read_f32_vec(body, layout.offset_start, dim)?.into();
    let norms = if stores_norms(metric) {
        Some(read_f32_vec(body, layout.norms_start, row_count)?)
    } else {
        None
    };
    for row in 0..row_count {
        let row_base = layout.rows_start + row * dim * ROW_BYTES_PER_DIM;
        let id_base = layout.ids_start + row * U32_BYTES;
        let stable_id_base = layout.stable_ids_start + row * I128_BYTES;
        let local_doc_id = read_u32_at(body, id_base, "local_id")?;
        let stable_id = read_i128_at(body, stable_id_base, "stable_id")?;
        let codes = body[row_base..row_base + dim].to_vec();
        let residuals = body[row_base + dim..row_base + dim + dim].to_vec();
        out.push(MaterializedIvfRow {
            local_doc_id,
            stable_id,
            cluster: cluster_id,
            rabitq_code: Vec::new(),
            encoded: EncodedCellRow {
                stable_id,
                scale: Arc::clone(&scale),
                offset: Arc::clone(&offset),
                codes,
                residuals,
                norm_sq: norms.as_ref().map(|values| values[row]),
            },
        });
    }
    Ok(())
}

fn run_body_len(dim: usize, metric: Metric, row_count: usize) -> usize {
    RunBodyLayout::new(dim, metric, row_count).total
}

fn read_f32_vec(body: &[u8], offset: usize, count: usize) -> Result<Vec<f32>, VectorError> {
    let end = offset
        .checked_add(count * F32_BYTES)
        .ok_or_else(|| malformed("SPFresh f32 range overflow"))?;
    if end > body.len() {
        return Err(malformed("SPFresh f32 range truncated"));
    }
    let mut out = Vec::with_capacity(count);
    for chunk in body[offset..end].chunks_exact(F32_BYTES) {
        out.push(f32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| malformed("SPFresh f32 slice"))?,
        ));
    }
    Ok(out)
}

fn read_u32_at(body: &[u8], offset: usize, field: &str) -> Result<u32, VectorError> {
    let end = offset
        .checked_add(U32_BYTES)
        .ok_or_else(|| malformed(format!("SPFresh {field} offset overflow")))?;
    if end > body.len() {
        return Err(malformed(format!("SPFresh {field} truncated")));
    }
    let arr: [u8; U32_BYTES] = body[offset..end]
        .try_into()
        .map_err(|_| malformed(format!("SPFresh {field} slice")))?;
    Ok(u32::from_le_bytes(arr))
}

fn read_u64_at(body: &[u8], offset: usize, field: &str) -> Result<u64, VectorError> {
    let end = offset
        .checked_add(U64_BYTES)
        .ok_or_else(|| malformed(format!("SPFresh {field} offset overflow")))?;
    if end > body.len() {
        return Err(malformed(format!("SPFresh {field} truncated")));
    }
    let arr: [u8; U64_BYTES] = body[offset..end]
        .try_into()
        .map_err(|_| malformed(format!("SPFresh {field} slice")))?;
    Ok(u64::from_le_bytes(arr))
}

fn read_i128_at(body: &[u8], offset: usize, field: &str) -> Result<i128, VectorError> {
    let end = offset
        .checked_add(I128_BYTES)
        .ok_or_else(|| malformed(format!("SPFresh {field} offset overflow")))?;
    if end > body.len() {
        return Err(malformed(format!("SPFresh {field} truncated")));
    }
    let arr: [u8; I128_BYTES] = body[offset..end]
        .try_into()
        .map_err(|_| malformed(format!("SPFresh {field} slice")))?;
    Ok(i128::from_le_bytes(arr))
}

fn fetch_sync(source: &Source, range: Range<usize>, what: &str) -> Result<Bytes, VectorError> {
    let start = range.start;
    let end = range.end;
    source
        .try_get_range_sync(range)
        .ok_or_else(|| malformed(format!("SPFresh {what} range {start}..{end} unavailable")))
}

fn lazy_source_error(error: LazyByteSourceError) -> VectorError {
    VectorError::Read(ReadError::Io(IoError::other(error.to_string())))
}

fn metric_id(metric: Metric) -> u8 {
    match metric {
        Metric::L2Sq => METRIC_ID_L2SQ as u8,
        Metric::Cosine => METRIC_ID_COSINE as u8,
        Metric::NegDot => METRIC_ID_NEGDOT as u8,
    }
}

fn metric_from_id(id: u8) -> Result<Metric, VectorError> {
    match id as u32 {
        METRIC_ID_L2SQ => Ok(Metric::L2Sq),
        METRIC_ID_COSINE => Ok(Metric::Cosine),
        METRIC_ID_NEGDOT => Ok(Metric::NegDot),
        _ => Err(malformed(format!("SPFresh unknown metric id {id}"))),
    }
}

fn stores_norms(metric: Metric) -> bool {
    matches!(metric, Metric::L2Sq | Metric::Cosine)
}

fn malformed(message: impl Into<String>) -> VectorError {
    VectorError::Read(ReadError::MalformedVersion(message.into()))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct WorstHit(SpfreshSearchHit);

impl Eq for WorstHit {}

impl Ord for WorstHit {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_f32(self.0.score, other.0.score)
    }
}

impl PartialOrd for WorstHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn cmp_f32(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use std::{
        ops::Range,
        sync::{Arc, Mutex},
    };

    use async_trait::async_trait;
    use bytes::Bytes;

    use super::{
        ColumnState, HiddenIndexLayout, RunInput, SpfreshBlobReader, SpfreshRunProbe,
        assign_fine_clusters_for_target, assign_replicas, encode_materialized_runs,
        fine_centroid_count_for_target, fp32_rows_to_runs_for_target, parse_hidden_index_layout,
        run_row_stride,
    };
    use crate::superfile::{
        lazy_source::{LazyByteSource, LazyByteSourceError},
        vector::{
            builder::VectorConfig,
            cell_posting::{EncodedCellRow, MaterializedIvfRow},
            distance::Metric,
            layout::VectorLayout,
            rerank_codec::RerankCodec,
        },
    };

    /// Three well-separated centroids on a line, `dim = 2`.
    const LINE_CENTROIDS_3: [f32; 6] = [0.0, 0.0, 10.0, 0.0, 20.0, 0.0];
    /// Two well-separated centroids on a line, `dim = 2`.
    const LINE_CENTROIDS_2: [f32; 4] = [0.0, 0.0, 10.0, 0.0];

    #[derive(Debug)]
    struct RecordingSource {
        bytes: Bytes,
        requests: Mutex<Vec<Range<u64>>>,
    }

    impl RecordingSource {
        fn new(bytes: Bytes) -> Self {
            Self {
                bytes,
                requests: Mutex::new(Vec::new()),
            }
        }

        fn clear_requests(&self) {
            self.requests.lock().expect("requests lock").clear();
        }

        fn requests(&self) -> Vec<Range<u64>> {
            self.requests.lock().expect("requests lock").clone()
        }
    }

    #[async_trait]
    impl LazyByteSource for RecordingSource {
        fn size(&self) -> u64 {
            self.bytes.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.requests
                .lock()
                .expect("requests lock")
                .push(start..start + len);
            if start.saturating_add(len) > self.size() {
                return Err(LazyByteSourceError::OutOfBounds {
                    start,
                    len,
                    size: self.size(),
                });
            }
            let start = start as usize;
            let end = start + len as usize;
            Ok(self.bytes.slice(start..end))
        }
    }

    #[test]
    fn assign_replicas_hard_assignment_at_eps_zero() {
        // Default eps=0 must reproduce plain nearest-centroid (single home).
        let v = [1.0, 0.0];
        let out = assign_replicas(Metric::L2Sq, &v, 2, &LINE_CENTROIDS_3, 0.0);
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn assign_replicas_interior_point_returns_single_centroid() {
        // v is clearly closest to c0; a modest closure still selects only c0.
        let v = [4.0, 0.0];
        let out = assign_replicas(Metric::L2Sq, &v, 2, &LINE_CENTROIDS_3, 0.1);
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn assign_replicas_boundary_point_replicates_across_far_centroids() {
        // Near-equidistant between two far-apart centroids: neither covers the
        // other, so RNG keeps both replicas.
        let v = [5.0, 0.1];
        let out = assign_replicas(Metric::L2Sq, &v, 2, &LINE_CENTROIDS_2, 0.1);
        assert_eq!(out, vec![0, 1]);
    }

    #[test]
    fn assign_replicas_eps_widens_candidate_set() {
        // Same point, two eps values: small eps keeps only the nearest, large
        // eps admits the far (uncovered) centroid as a boundary replica.
        let v = [4.0, 0.0];
        assert_eq!(
            assign_replicas(Metric::L2Sq, &v, 2, &LINE_CENTROIDS_2, 0.1),
            vec![0]
        );
        assert_eq!(
            assign_replicas(Metric::L2Sq, &v, 2, &LINE_CENTROIDS_2, 1.5),
            vec![0, 1]
        );
    }

    #[test]
    fn assign_replicas_rng_prunes_covered_centroid() {
        // Collinear centroids, point above the middle one. The nearer middle
        // centroid covers both outer centroids, so RNG keeps only the middle.
        let centroids = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let v = [1.0, 3.0];
        let out = assign_replicas(Metric::L2Sq, &v, 2, &centroids, 0.3);
        assert_eq!(out, vec![1]);
    }

    #[test]
    fn assign_replicas_empty_centroids_is_empty() {
        let out = assign_replicas(Metric::L2Sq, &[1.0, 2.0], 2, &[], 0.5);
        assert!(out.is_empty());
    }

    #[test]
    fn fine_centroid_count_scales_with_rows() {
        // dim=16 -> run_row_stride = 16*2 + 4 + 16 + 4 = 56 bytes/row; a 5600
        // byte target holds 100 rows per run, so K grows with the row count.
        assert_eq!(fine_centroid_count_for_target(1000, 16, 5600), 10);
        assert_eq!(fine_centroid_count_for_target(50, 16, 5600), 1);
        assert_eq!(fine_centroid_count_for_target(0, 16, 5600), 1);
        assert_eq!(fine_centroid_count_for_target(1, 16, 5600), 1);
        // Tiny target: per-run clamps to >=1 row and K never exceeds n_rows.
        assert_eq!(fine_centroid_count_for_target(5, 16, 56), 5);
    }

    #[test]
    fn fp32_commit_rows_train_fine_runs_from_target_size() {
        let config = VectorConfig {
            column: "emb".into(),
            dim: 2,
            n_cent: 2,
            rot_seed: 0,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
            provided_centroids: None,
        };
        let col = ColumnState {
            config,
            ids: vec![0, 1, 2, 3],
            vectors: vec![
                0.0, 0.0, //
                0.1, 0.0, //
                100.0, 0.0, //
                100.1, 0.0,
            ],
            materialized_rows: None,
            next_local_id: 4,
        };
        let runs = fp32_rows_to_runs_for_target(&col, run_row_stride(2) * 2);
        assert_eq!(runs.len(), 2);
        let mut row_counts: Vec<usize> = runs.iter().map(|run| run.local_ids.len()).collect();
        row_counts.sort_unstable();
        assert_eq!(row_counts, vec![2, 2]);
    }

    #[test]
    fn assign_fine_clusters_single_run_for_small_input() {
        let mut rows = materialized_rows(16, 0, 4, 0);
        // Large target -> one run; every row lands in fine cluster 0.
        assign_fine_clusters_for_target(&mut rows, 16, 1 << 20);
        assert!(rows.iter().all(|r| r.cluster == 0));
    }

    #[test]
    fn assign_fine_clusters_splits_separated_rows() {
        // Two tight groups on dim 0 (values ~0 and ~100). A small target forces
        // k_fine=2; k-means separates them into distinct fine clusters.
        let mut rows = materialized_rows(16, 0, 2, 0);
        rows.extend(materialized_rows(16, 100, 2, 100));
        // 4 rows, 56 B/row, target 112 -> 2 rows/run -> k_fine=2.
        assign_fine_clusters_for_target(&mut rows, 16, 112);
        let low = rows[0].cluster;
        let high = rows[2].cluster;
        assert_ne!(
            low, high,
            "separated groups must get distinct fine clusters"
        );
        assert_eq!(rows[1].cluster, low);
        assert_eq!(rows[3].cluster, high);
    }

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

    #[test]
    fn vector_blob_round_trips_runs() {
        let dim = 16usize;
        let rows = materialized_rows(dim, 0, 4, 100);
        let blob = encode_materialized_runs(
            Metric::L2Sq,
            dim,
            &[RunInput {
                cluster_id: 11,
                rows,
            }],
        )
        .expect("encode");
        let reader = SpfreshBlobReader::open(Bytes::from(blob)).expect("open");
        assert_eq!(reader.dim(), dim);
        assert_eq!(reader.metric(), Metric::L2Sq);
        assert_eq!(reader.n_rows(), 4);
        assert_eq!(reader.runs().len(), 1);
        assert_eq!(reader.runs()[0].cluster_id, 11);
    }

    #[test]
    fn materialized_rows_round_trip_from_blob() {
        let dim = 16usize;
        let rows = materialized_rows(dim, 3, 4, 200);
        let blob = encode_materialized_runs(
            Metric::L2Sq,
            dim,
            &[RunInput {
                cluster_id: 5,
                rows: rows.clone(),
            }],
        )
        .expect("encode");
        let reader = SpfreshBlobReader::open(Bytes::from(blob)).expect("open");
        let decoded = reader.materialized_rows().expect("materialize");
        assert_eq!(decoded.len(), rows.len());
        for (actual, expected) in decoded.iter().zip(rows.iter()) {
            assert_eq!(actual.local_doc_id, expected.local_doc_id);
            assert_eq!(actual.stable_id, expected.stable_id);
            assert_eq!(actual.cluster, 5);
            assert_eq!(actual.encoded.stable_id, expected.stable_id);
            assert_eq!(actual.encoded.codes.len(), dim);
            assert_eq!(actual.encoded.residuals.len(), dim);
            assert_eq!(actual.encoded.scale.len(), dim);
            assert_eq!(actual.encoded.offset.len(), dim);
            assert!(actual.encoded.norm_sq.is_some());
        }
    }

    #[test]
    fn selected_run_fetch_uses_contiguous_body_range() {
        let dim = 16usize;
        let run_a = RunInput {
            cluster_id: 1,
            rows: materialized_rows(dim, 0, 3, 0),
        };
        let run_b = RunInput {
            cluster_id: 2,
            rows: materialized_rows(dim, 10, 2, 10),
        };
        let blob = encode_materialized_runs(Metric::L2Sq, dim, &[run_a, run_b]).expect("encode");
        let reader = SpfreshBlobReader::open(Bytes::from(blob)).expect("open");
        let first = reader.run_range(0).expect("first range");
        let second = reader.run_range(1).expect("second range");
        assert!(first.end <= second.start);
        assert_eq!(reader.run_bytes(1).expect("run bytes").len(), second.len());
        assert_eq!(reader.runs_for_cluster(2), vec![1]);
    }

    #[test]
    fn selected_run_search_matches_bruteforce_nearest() {
        let dim = 16usize;
        let rows = materialized_rows(dim, 0, 8, 0);
        let blob = encode_materialized_runs(
            Metric::L2Sq,
            dim,
            &[RunInput {
                cluster_id: 0,
                rows,
            }],
        )
        .expect("encode");
        let reader = SpfreshBlobReader::open(Bytes::from(blob)).expect("open");
        let mut query = vec![0.0f32; dim];
        query[0] = 5.1;
        let hits = reader
            .search_runs(&[0], &query, 3)
            .expect("search selected run");
        assert_eq!(hits[0].0, 5);
    }

    #[tokio::test]
    async fn lazy_selected_run_search_fetches_only_selected_body_and_stable_id() {
        let dim = 16usize;
        let run_a = RunInput {
            cluster_id: 1,
            rows: materialized_rows(dim, 0, 3, 0),
        };
        let run_b = RunInput {
            cluster_id: 2,
            rows: materialized_rows(dim, 10, 2, 10),
        };
        let blob = Bytes::from(
            encode_materialized_runs(Metric::L2Sq, dim, &[run_a, run_b]).expect("encode"),
        );
        let source = Arc::new(RecordingSource::new(blob));
        let lazy_source: Arc<dyn LazyByteSource> = source.clone();
        let reader = SpfreshBlobReader::open_lazy(lazy_source)
            .await
            .expect("lazy open");
        let selected_range = reader.run_range(1).expect("selected range");
        source.clear_requests();

        let mut query = vec![0.0f32; dim];
        query[0] = 10.1;
        let hits = reader
            .search_run_probes_filtered_with_stable_ids_async(
                &[SpfreshRunProbe {
                    run_id: 1,
                    body_range: Some(selected_range.clone()),
                    row_count: reader.runs()[1].row_count,
                }],
                &query,
                1,
                None,
                None,
            )
            .await
            .expect("selected search");

        assert_eq!(
            source.requests(),
            vec![selected_range.start as u64..selected_range.end as u64]
        );
        assert_eq!(hits[0].local_doc_id, 10);
        assert_eq!(hits[0].stable_id, 10);
    }

    fn materialized_rows(
        dim: usize,
        first_local_id: u32,
        count: u32,
        first_stable_id: i128,
    ) -> Vec<MaterializedIvfRow> {
        let scale: Arc<[f32]> = Arc::from(vec![1.0; dim]);
        let offset: Arc<[f32]> = Arc::from(vec![0.0; dim]);
        (0..count)
            .map(|idx| {
                let value = first_local_id + idx;
                let mut codes = vec![0u8; dim];
                codes[0] = value as u8;
                MaterializedIvfRow {
                    local_doc_id: value,
                    stable_id: first_stable_id + i128::from(idx),
                    cluster: 0,
                    rabitq_code: Vec::new(),
                    encoded: EncodedCellRow {
                        stable_id: first_stable_id + i128::from(idx),
                        scale: scale.clone(),
                        offset: offset.clone(),
                        codes,
                        residuals: vec![0u8; dim],
                        norm_sq: Some((value * value) as f32),
                    },
                }
            })
            .collect()
    }
}
