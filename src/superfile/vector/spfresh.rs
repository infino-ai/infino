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
    ops::Range,
    sync::{Arc, OnceLock},
};

use bytes::Bytes;
use roaring::RoaringBitmap;

use crate::superfile::{
    BuildError, ReadError,
    error::VectorError,
    format::vec::{METRIC_ID_COSINE, METRIC_ID_L2SQ, METRIC_ID_NEGDOT},
    vector::{
        builder::{VectorConfig, derive_sq8_quantizer_from_min_max},
        cell_posting::{
            EncodedCellRow, MaterializedIvfRow, materialize_sq8_residual_row_into_cluster_quant,
        },
        distance::{
            Metric, SQ8_RESIDUAL_DIVISOR, Sq8ResidualEpsilonKernel, dequantize_sq8_residual_into,
            distance,
        },
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
const RUN_DIR_ENTRY_BYTES: usize = 28;
const HEADER_BYTES: usize =
    MAGIC.len() + U32_BYTES + METRIC_BYTES + HEADER_RESERVED_BYTES + U32_BYTES + U32_BYTES;
const RUN_CELL_ID_OFF: usize = 0;
const RUN_CLUSTER_ID_OFF: usize = RUN_CELL_ID_OFF + U32_BYTES;
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
    pub(crate) cell_id: u32,
    pub(crate) cluster_id: u32,
    pub(crate) rows: Vec<MaterializedIvfRow>,
}

#[derive(Debug, Clone)]
pub(crate) struct SpfreshRun {
    pub(crate) cell_id: u32,
    pub(crate) cluster_id: u32,
    pub(crate) row_count: u32,
    body_range: Range<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct SpfreshBlobReader {
    bytes: Bytes,
    dim: usize,
    metric: Metric,
    runs: Vec<SpfreshRun>,
    n_rows: u32,
}

#[derive(Debug, Clone)]
struct EncodedRun {
    cell_id: u32,
    cluster_id: u32,
    row_count: u32,
    body: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Fp32RunInput {
    cell_id: u32,
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
        if bytes.len() < HEADER_BYTES {
            return Err(malformed("SPFresh blob header truncated"));
        }
        let actual = &bytes[..MAGIC.len()];
        if actual != MAGIC {
            return Err(VectorError::Read(ReadError::BadMagic {
                section: "vector/spfresh",
                expected: MAGIC,
                actual: actual.to_vec(),
            }));
        }
        let mut offset = MAGIC.len();
        let dim = read_u32_at(&bytes, offset, "dim")? as usize;
        offset += U32_BYTES;
        let metric = metric_from_id(bytes[offset])?;
        offset += METRIC_BYTES + HEADER_RESERVED_BYTES;
        let run_count = read_u32_at(&bytes, offset, "run_count")? as usize;
        offset += U32_BYTES;
        let n_rows = read_u32_at(&bytes, offset, "row_count")?;
        let directory_bytes = run_count
            .checked_mul(RUN_DIR_ENTRY_BYTES)
            .ok_or_else(|| malformed("SPFresh run directory overflow"))?;
        let directory_end = HEADER_BYTES
            .checked_add(directory_bytes)
            .ok_or_else(|| malformed("SPFresh run directory overflow"))?;
        if bytes.len() < directory_end {
            return Err(malformed("SPFresh run directory truncated"));
        }
        let mut runs = Vec::with_capacity(run_count);
        for run_idx in 0..run_count {
            let entry = HEADER_BYTES + run_idx * RUN_DIR_ENTRY_BYTES;
            let cell_id = read_u32_at(&bytes, entry + RUN_CELL_ID_OFF, "cell_id")?;
            let cluster_id = read_u32_at(&bytes, entry + RUN_CLUSTER_ID_OFF, "cluster_id")?;
            let row_count = read_u32_at(&bytes, entry + RUN_ROW_COUNT_OFF, "row_count")?;
            let body_offset =
                read_u64_at(&bytes, entry + RUN_BODY_OFFSET_OFF, "body_offset")? as usize;
            let body_length =
                read_u64_at(&bytes, entry + RUN_BODY_LENGTH_OFF, "body_length")? as usize;
            let body_end = body_offset
                .checked_add(body_length)
                .ok_or_else(|| malformed("SPFresh run body overflow"))?;
            if body_offset < directory_end || body_end > bytes.len() {
                return Err(malformed("SPFresh run body out of bounds"));
            }
            let expected = run_body_len(dim, metric, row_count as usize);
            if body_length != expected {
                return Err(malformed(format!(
                    "SPFresh run body has {body_length} bytes, expected {expected}"
                )));
            }
            runs.push(SpfreshRun {
                cell_id,
                cluster_id,
                row_count,
                body_range: body_offset..body_end,
            });
        }
        Ok(Self {
            bytes,
            dim,
            metric,
            runs,
            n_rows,
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
        self.run_range(run_idx).map(|range| self.bytes.slice(range))
    }

    #[cfg(test)]
    pub(crate) fn runs_for_cell(&self, cell_id: u32) -> Vec<usize> {
        self.runs
            .iter()
            .enumerate()
            .filter_map(|(idx, run)| (run.cell_id == cell_id).then_some(idx))
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

    pub(crate) fn search_runs_filtered(
        &self,
        run_ids: &[usize],
        query: &[f32],
        k: usize,
        allow: Option<&RoaringBitmap>,
        deny: Option<&RoaringBitmap>,
    ) -> Result<Vec<(u32, f32)>, VectorError> {
        if query.len() != self.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut heap = BinaryHeap::<WorstHit>::new();
        for &run_id in run_ids {
            let run = self
                .runs
                .get(run_id)
                .ok_or_else(|| malformed(format!("SPFresh run {run_id} out of range")))?;
            let body = self.bytes.slice(run.body_range.clone());
            score_run_body(
                &body,
                self.dim,
                self.metric,
                run.row_count as usize,
                query,
                k,
                allow,
                deny,
                &mut heap,
            )?;
        }
        let mut out: Vec<(u32, f32)> = heap.into_iter().map(|hit| hit.0).collect();
        out.sort_by(|a, b| cmp_f32(a.1, b.1));
        Ok(out)
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
            let body = self.bytes.slice(run.body_range.clone());
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
        out.extend_from_slice(&run.cell_id.to_le_bytes());
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
            cell_id: cluster,
            cluster_id: cluster,
            rows,
        })
        .collect()
}

fn fp32_rows_to_runs(col: &ColumnState) -> Vec<Fp32RunInput> {
    let dim = col.config.dim;
    let n_rows = col.ids.len();
    let eps = replication_eps();
    let mut grouped: HashMap<u32, Fp32RunInput> = HashMap::new();
    for row_idx in 0..n_rows {
        let vector = &col.vectors[row_idx * dim..(row_idx + 1) * dim];
        // Shared replica assignment: `assign_replicas` returns the single
        // nearest centroid at the default eps=0 (hard assignment) and a boundary
        // replica set once eps>0. With no provided centroids there is one run.
        let cells = match col.config.provided_centroids.as_ref() {
            Some(centroids) => assign_replicas(col.config.metric, vector, dim, centroids, eps),
            None => vec![0],
        };
        for cell in cells {
            let entry = grouped.entry(cell).or_insert_with(|| Fp32RunInput {
                cell_id: cell,
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
    runs.sort_by_key(|run| (run.cell_id, run.cluster_id));
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
        cell_id: run.cell_id,
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
        cell_id: run.cell_id,
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
    let scale = read_f32_vec(body, 0, dim)?;
    let offset = read_f32_vec(body, dim * F32_BYTES, dim)?;
    let rows_start = dim * F32_BYTES * 2;
    let rows_len = row_count * dim * ROW_BYTES_PER_DIM;
    let ids_start = rows_start + rows_len;
    let stable_ids_start = ids_start + row_count * U32_BYTES;
    let norms_start = stable_ids_start + row_count * I128_BYTES;
    let norms = if stores_norms(metric) {
        Some(read_f32_vec(body, norms_start, row_count)?)
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
        let row_base = rows_start + row * dim * ROW_BYTES_PER_DIM;
        let id_base = ids_start + row * U32_BYTES;
        let local_id = read_u32_at(body, id_base, "local_id")?;
        if allow.is_some_and(|bitmap| !bitmap.contains(local_id))
            || deny.is_some_and(|bitmap| bitmap.contains(local_id))
        {
            continue;
        }
        let codes = &body[row_base..row_base + dim];
        let residuals = &body[row_base + dim..row_base + dim + dim];
        let norm = norms.as_ref().map(|values| values[row]);
        let dist = kernel.distance_with_norm(codes, residuals, norm);
        let hit = WorstHit((local_id, dist));
        if heap.len() < k {
            heap.push(hit);
        } else if let Some(worst) = heap.peek()
            && cmp_f32(hit.0.1, worst.0.1).is_lt()
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
    let scale: Arc<[f32]> = read_f32_vec(body, 0, dim)?.into();
    let offset: Arc<[f32]> = read_f32_vec(body, dim * F32_BYTES, dim)?.into();
    let rows_start = dim * F32_BYTES * 2;
    let rows_len = row_count * dim * ROW_BYTES_PER_DIM;
    let ids_start = rows_start + rows_len;
    let stable_ids_start = ids_start + row_count * U32_BYTES;
    let norms_start = stable_ids_start + row_count * I128_BYTES;
    let norms = if stores_norms(metric) {
        Some(read_f32_vec(body, norms_start, row_count)?)
    } else {
        None
    };
    for row in 0..row_count {
        let row_base = rows_start + row * dim * ROW_BYTES_PER_DIM;
        let id_base = ids_start + row * U32_BYTES;
        let stable_id_base = stable_ids_start + row * I128_BYTES;
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
    let quantizer_bytes = dim * F32_BYTES * 2;
    let row_bytes = row_count * dim * ROW_BYTES_PER_DIM;
    let local_id_bytes = row_count * U32_BYTES;
    let stable_id_bytes = row_count * I128_BYTES;
    let norm_bytes = if stores_norms(metric) {
        row_count * F32_BYTES
    } else {
        0
    };
    quantizer_bytes + row_bytes + local_id_bytes + stable_id_bytes + norm_bytes
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
struct WorstHit((u32, f32));

impl Eq for WorstHit {}

impl Ord for WorstHit {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_f32(self.0.1, other.0.1)
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
    use std::sync::Arc;

    use bytes::Bytes;

    use super::{
        HiddenIndexLayout, RunInput, SpfreshBlobReader, assign_replicas, encode_materialized_runs,
        parse_hidden_index_layout,
    };
    use crate::superfile::vector::{
        cell_posting::{EncodedCellRow, MaterializedIvfRow},
        distance::Metric,
        layout::VectorLayout,
    };

    /// Three well-separated centroids on a line, `dim = 2`.
    const LINE_CENTROIDS_3: [f32; 6] = [0.0, 0.0, 10.0, 0.0, 20.0, 0.0];
    /// Two well-separated centroids on a line, `dim = 2`.
    const LINE_CENTROIDS_2: [f32; 4] = [0.0, 0.0, 10.0, 0.0];

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
                cell_id: 7,
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
        assert_eq!(reader.runs()[0].cell_id, 7);
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
                cell_id: 5,
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
            cell_id: 1,
            cluster_id: 1,
            rows: materialized_rows(dim, 0, 3, 0),
        };
        let run_b = RunInput {
            cell_id: 2,
            cluster_id: 2,
            rows: materialized_rows(dim, 10, 2, 10),
        };
        let blob = encode_materialized_runs(Metric::L2Sq, dim, &[run_a, run_b]).expect("encode");
        let reader = SpfreshBlobReader::open(Bytes::from(blob)).expect("open");
        let first = reader.run_range(0).expect("first range");
        let second = reader.run_range(1).expect("second range");
        assert!(first.end <= second.start);
        assert_eq!(reader.run_bytes(1).expect("run bytes").len(), second.len());
        assert_eq!(reader.runs_for_cell(2), vec![1]);
    }

    #[test]
    fn selected_run_search_matches_bruteforce_nearest() {
        let dim = 16usize;
        let rows = materialized_rows(dim, 0, 8, 0);
        let blob = encode_materialized_runs(
            Metric::L2Sq,
            dim,
            &[RunInput {
                cell_id: 0,
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
