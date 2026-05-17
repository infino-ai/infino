//! Vector blob builder. Multi-column unified blob with per-column
//! self-contained subsections.
//!
//! Each column's subsection is a self-contained IVF + RaBitQ index:
//! summary centroid + radius, IVF centroids (from k-means), cluster
//! index, 1-bit codes, full-precision vectors, doc_ids — all in
//! cluster-contiguous order so the rerank loop stays in cache.
//!
//! See `docs/architecture/superfile.md` for the full byte-level spec.

use crate::superfile::BuildError;
use crate::superfile::format::checksum::crc32c;
use crate::superfile::format::{self, FST_SEPARATOR, RESERVED_PREFIX};
use crate::superfile::vector::distance::{Metric, l2_sq};
use crate::superfile::vector::kmeans::{assign_to_centroids, kmeans};
use crate::superfile::vector::quant::BitQuantizer;
use crate::superfile::vector::reservoir::{Reservoir, default_kmeans_sample_size};
use crate::superfile::vector::rotation::RandomRotation;
use rayon::prelude::*;

/// Outer-header size (magic + version + n_columns + n_docs + dir_offset).
const OUTER_HEADER_SIZE: usize = 32;

/// Subsection-directory entry size in bytes.
const DIR_ENTRY_SIZE: usize = 64;

/// Per-column sub-header size (inside each subsection).
const SUB_HEADER_SIZE: usize = 56;

/// Metric ID encoding for the directory entry. Spec: 0 = L2Sq, 1 = Cosine,
/// 2 = NegDot.
fn metric_id(m: Metric) -> u32 {
    match m {
        Metric::L2Sq => 0,
        Metric::Cosine => 1,
        Metric::NegDot => 2,
    }
}

/// Per-column user-supplied build configuration.
#[derive(Debug, Clone)]
pub struct VectorConfig {
    pub name: String,
    pub dim: usize,
    pub n_cent: usize,
    pub rot_seed: u64,
    pub metric: Metric,
}

/// Per-column build-time state.
struct ColumnState {
    config: VectorConfig,
    /// Contiguous: `dim * n_docs` f32s, push-order matches doc_id order.
    /// Held for assignment + RaBitQ encode + cluster-contiguous layout
    /// during `finish()`. 010 M2 leaves this in place; 010 M3 replaces
    /// it with a (`pre_spill_buffer`, optional disk spill) pair so
    /// peak resident memory becomes a function of
    /// `(reservoir, spill_threshold)` instead of `(n_docs, dim)`.
    vectors: Vec<f32>,
    n_docs: u32,
    /// Reservoir-sampled k-means training set. Bounded at
    /// `default_kmeans_sample_size(n_cent)` rows regardless of
    /// `n_docs`; once `n_docs > sample_size` further `add()` calls
    /// only touch the reservoir with probability `sample_size /
    /// n_docs` (Vitter Algorithm L). Trained against by k-means in
    /// `finish()`; the resulting centroids are then used to assign
    /// the full corpus.
    reservoir: Reservoir,
}

/// Multi-column vector blob builder.
pub struct VectorBuilder {
    columns: Vec<ColumnState>,
}

impl Default for VectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorBuilder {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    /// Register a vector column up-front. Returns the assigned
    /// `column_id` (declaration order).
    pub fn register_column(&mut self, config: VectorConfig) -> Result<u32, BuildError> {
        if config.name.as_bytes().contains(&FST_SEPARATOR) {
            return Err(BuildError::ReservedSeparatorInColumnName(config.name));
        }
        if config.name.starts_with(RESERVED_PREFIX) {
            return Err(BuildError::ReservedPrefixInColumnName(config.name));
        }
        if !(16..=4096).contains(&config.dim) {
            return Err(BuildError::VectorDimOutOfRange {
                column: config.name.clone(),
                dim: config.dim,
            });
        }
        if self.columns.iter().any(|c| c.config.name == config.name) {
            return Err(BuildError::DuplicateColumnName(config.name));
        }
        let column_id = self.columns.len() as u32;
        let sample_size = default_kmeans_sample_size(config.n_cent);
        // Seed the reservoir RNG from `rot_seed ^ 0x5a5a` so it
        // stays deterministic with the column config but uses a
        // distinct stream from `RandomRotation` (which seeds from
        // `rot_seed` directly) and `kmeans` (which seeds from
        // `rot_seed + 7`). Three disjoint streams, three
        // deterministic seeds, one knob on the user's end.
        let reservoir_seed = config.rot_seed ^ 0x5a5a_5a5a_5a5a_5a5a;
        let reservoir = Reservoir::new(sample_size, config.dim, reservoir_seed);
        self.columns.push(ColumnState {
            config,
            vectors: Vec::new(),
            n_docs: 0,
            reservoir,
        });
        Ok(column_id)
    }

    /// Override the k-means training sample size for a column. Must
    /// be called before the first `add()` for the column — calling it
    /// later silently discards already-observed reservoir state and
    /// only future `add()` calls feed into the new reservoir.
    ///
    /// The default sample size is `default_kmeans_sample_size(n_cent)`
    /// (`100K-500K` depending on `n_cent`). This override exists for
    /// (a) the M2 sample-size sweep on synthetic recall corpora and
    /// (b) future advanced callers that want to dial sample size to
    /// match a recall vs. memory trade-off they've profiled.
    ///
    /// Returns an error if `column_id` is out of range.
    pub fn set_kmeans_sample_size(
        &mut self,
        column_id: u32,
        sample_size: usize,
    ) -> Result<(), BuildError> {
        let idx = column_id as usize;
        if idx >= self.columns.len() {
            return Err(BuildError::FtsColumnTypeInvalid {
                column: format!("(unregistered vector column_id {column_id})"),
                actual: "n/a".to_string(),
            });
        }
        let col = &mut self.columns[idx];
        let reservoir_seed = col.config.rot_seed ^ 0x5a5a_5a5a_5a5a_5a5a;
        col.reservoir = Reservoir::new(sample_size, col.config.dim, reservoir_seed);
        Ok(())
    }

    /// Append one vector to the named column. Caller must invoke once
    /// per (column, doc) pair, with doc-id order matching insertion
    /// order. The vector slice must have length equal to the column's
    /// `dim`.
    pub fn add(&mut self, column_id: u32, vec: &[f32]) -> Result<(), BuildError> {
        let idx = column_id as usize;
        if idx >= self.columns.len() {
            return Err(BuildError::FtsColumnTypeInvalid {
                column: format!("(unregistered vector column_id {column_id})"),
                actual: "n/a".to_string(),
            });
        }
        let col = &mut self.columns[idx];
        if vec.len() != col.config.dim {
            return Err(BuildError::FtsColumnTypeInvalid {
                column: col.config.name.clone(),
                actual: format!("vec.len()={} != dim={}", vec.len(), col.config.dim),
            });
        }
        col.reservoir.update(vec);
        col.vectors.extend_from_slice(vec);
        col.n_docs += 1;
        Ok(())
    }

    /// Finalise and emit the unified vector blob. Consumes the builder.
    pub fn finish(self) -> Vec<u8> {
        let n_columns = self.columns.len() as u32;
        // n_docs in the outer header is the max across columns (the
        // per-segment count). Spec: same across all columns.
        let n_docs: u64 = self
            .columns
            .iter()
            .map(|c| c.n_docs as u64)
            .max()
            .unwrap_or(0);

        // 1. Build each per-column subsection independently. Each
        //    subsection is self-contained — sub-header + summary +
        //    centroids + cluster index + codes + full + doc_ids + CRC.
        let mut subsections: Vec<SubsectionBytes> = Vec::with_capacity(self.columns.len());
        for col in &self.columns {
            subsections.push(build_subsection(
                &col.config,
                col.reservoir.sample(),
                &col.vectors,
                col.n_docs,
            ));
        }

        // 2. Layout: outer_header(32) + directory(n_columns * 64) +
        //    dir_crc(4) + subsections concatenated + outer_crc(4).
        let directory_offset = OUTER_HEADER_SIZE as u64;
        let directory_size = (n_columns as usize) * DIR_ENTRY_SIZE;
        let mut subsection_start_off =
            directory_offset + directory_size as u64 + 4 /* dir CRC */;

        // 3. Assemble directory entries with absolute offsets.
        let mut directory: Vec<u8> = Vec::with_capacity(directory_size);
        for (i, sub) in subsections.iter().enumerate() {
            let col = &self.columns[i];
            let summary_offset_abs = subsection_start_off + sub.summary_offset_in_sub as u64;
            // entry: 64 bytes; see DIR_ENTRY_SIZE
            directory.extend_from_slice(&(i as u32).to_le_bytes()); // column_id
            directory.extend_from_slice(&(col.config.dim as u32).to_le_bytes()); // dim
            directory.extend_from_slice(&(col.config.n_cent as u32).to_le_bytes()); // n_cent
            directory.extend_from_slice(&metric_id(col.config.metric).to_le_bytes()); // metric_id
            directory.extend_from_slice(&col.config.rot_seed.to_le_bytes()); // rot_seed (8)
            directory.extend_from_slice(&subsection_start_off.to_le_bytes()); // subsection_offset (8)
            directory.extend_from_slice(&(sub.bytes.len() as u64).to_le_bytes()); // subsection_length (8)
            directory.extend_from_slice(&summary_offset_abs.to_le_bytes()); // summary_offset (8)
            directory.extend_from_slice(&((col.config.dim * 4) as u32).to_le_bytes()); // summary_length (4)
            directory.extend_from_slice(&0u32.to_le_bytes()); // reserved (4)
            directory.extend_from_slice(&0u64.to_le_bytes()); // future_reserved (8)
            debug_assert_eq!(directory.len() % DIR_ENTRY_SIZE, 0);

            subsection_start_off += sub.bytes.len() as u64;
        }
        let dir_crc = crc32c(&directory);

        // 4. Concatenate the final blob.
        let mut blob: Vec<u8> = Vec::with_capacity(
            OUTER_HEADER_SIZE
                + directory_size
                + 4
                + subsections.iter().map(|s| s.bytes.len()).sum::<usize>()
                + 4,
        );

        // Outer header (32 bytes).
        blob.extend_from_slice(format::vec::OUTER_MAGIC); // 8
        blob.extend_from_slice(&format::vec::VERSION.to_le_bytes()); // 4
        blob.extend_from_slice(&n_columns.to_le_bytes()); // 4
        blob.extend_from_slice(&n_docs.to_le_bytes()); // 8
        blob.extend_from_slice(&directory_offset.to_le_bytes()); // 8
        debug_assert_eq!(blob.len(), OUTER_HEADER_SIZE);

        // Directory + CRC.
        blob.extend_from_slice(&directory);
        blob.extend_from_slice(&dir_crc.to_le_bytes());

        // Subsections.
        for sub in &subsections {
            blob.extend_from_slice(&sub.bytes);
        }

        // Trailing whole-blob CRC32C (covers everything from byte 0 to
        // here).
        let outer_crc = crc32c(&blob);
        blob.extend_from_slice(&outer_crc.to_le_bytes());

        blob
    }
}

/// Builder output for one column's subsection.
struct SubsectionBytes {
    bytes: Vec<u8>,
    /// Byte offset of the summary centroid relative to the subsection
    /// start (matches the directory entry's `summary_offset` after
    /// translation to absolute).
    summary_offset_in_sub: usize,
}

/// Build one column's subsection. Pure function — no shared state with
/// other columns. Layout:
///
/// ```text
///   [Sub-header — 56 bytes]
///   [Summary centroid + radius]   — dim f32s
///   [IVF centroids]               — n_cent × dim × f32
///   [Cluster index]               — n_cent × (u32 doc_off, u32 doc_count)
///   [1-bit codes]                 — n_docs × ceil(dim/8) (cluster-contiguous)
///   [Full-precision vectors]      — n_docs × dim × f32 (cluster-contiguous)
///   [Doc IDs]                     — n_docs × u32 (local_doc_id in cluster order)
///   [Trailing CRC32C]             — u32 over all bytes above
/// ```
fn build_subsection(
    cfg: &VectorConfig,
    sample: &[f32],
    vectors: &[f32],
    n_docs_input: u32,
) -> SubsectionBytes {
    let dim = cfg.dim;
    let n_docs = n_docs_input as usize;
    let sample_rows = sample.len() / dim;
    // n_cent must be in [1, min(n_docs, sample_rows)]. Both bounds
    // are required: n_cent > n_docs makes the IVF degenerate; n_cent
    // > sample_rows would crash k-means since `k > n` in the trainer
    // is rejected at the assertion in `kmeans()`. In the steady-state
    // case (n_docs > sample_size), the sample_rows bound is the
    // active one and equals `default_kmeans_sample_size(cfg.n_cent)`,
    // which is `≥ 100K ≥ any sane n_cent`.
    let n_cent = cfg
        .n_cent
        .max(1)
        .min(n_docs.max(1))
        .min(sample_rows.max(1));

    // 1. K-means on the reservoir sample (010 M2). The sample is a
    //    uniform random draw from the full corpus bounded at
    //    `default_kmeans_sample_size(n_cent)` rows, so memory scales
    //    with `(n_cent, dim)` instead of `(n_docs, dim)`. Recall
    //    against an oracle is the gate; 010 M2 has tests at multiple
    //    sample_size points to confirm there's no degradation past
    //    the default.
    let centroids = if sample_rows == 0 || n_docs == 0 {
        vec![0.0f32; n_cent * dim]
    } else {
        kmeans(sample, dim, n_cent, 5, cfg.rot_seed)
    };

    // 2. Assign the full corpus against the reservoir-trained
    //    centroids. The pre-M2 path piggy-backed assignments off
    //    k-means' final iteration to save a pass, but that pass
    //    was over the same corpus k-means trained on. Now training
    //    is on a strict subset; the corpus needs its own assign.
    //    Roughly 2.4 s at 1M × 384 on CPU.
    let kmeans_assignments: Vec<u32> = if n_docs == 0 {
        Vec::new()
    } else {
        let mut a = vec![0u32; n_docs];
        assign_to_centroids(vectors, &centroids, dim, n_cent, &mut a);
        a
    };

    // 2. Summary centroid (mean of centroids) + summary radius (max
    //    L2 distance from summary centroid to any input vector).
    let mut summary_centroid = vec![0.0f32; dim];
    if !centroids.is_empty() {
        let mut acc = vec![0.0f64; dim];
        for c in 0..n_cent {
            let cv = &centroids[c * dim..(c + 1) * dim];
            for (a, &x) in acc.iter_mut().zip(cv) {
                *a += x as f64;
            }
        }
        let inv = 1.0 / (n_cent as f64);
        for (s, a) in summary_centroid.iter_mut().zip(&acc) {
            *s = (*a * inv) as f32;
        }
    }
    let summary_radius: f32 = (0..n_docs)
        .map(|d| {
            let v = &vectors[d * dim..(d + 1) * dim];
            l2_sq(v, &summary_centroid).sqrt()
        })
        .fold(0.0_f32, f32::max);
    let summary_radius_x100 = (summary_radius * 100.0).max(0.0).min(u32::MAX as f32) as u32;

    // 3. Use the assignments from the last k-means iteration directly.
    //    They reflect the same centroids we just emitted (k-means
    //    didn't run an update step *after* its final assignment), so
    //    re-running an assignment pass here would produce identical
    //    output — pure redundant work.
    let assignments: Vec<u32> = if n_docs == 0 {
        Vec::new()
    } else {
        kmeans_assignments
    };

    // 4. Encode all docs (rotate then sign-quantize) — parallel.
    let rot = RandomRotation::new(dim, cfg.rot_seed);
    let quant = BitQuantizer::new(dim);
    let code_bytes = quant.code_bytes();
    let mut all_codes = vec![0u8; n_docs * code_bytes];
    all_codes
        .par_chunks_mut(code_bytes)
        .enumerate()
        .for_each(|(d, code_out)| {
            let v = &vectors[d * dim..(d + 1) * dim];
            let mut rot_buf = vec![0f32; dim];
            rot.apply(v, &mut rot_buf);
            quant.encode_rotated_into(&rot_buf, code_out);
        });

    // 5. Lay out codes + full + doc_ids in cluster-contiguous order.
    let mut cluster_docs: Vec<Vec<u32>> = vec![Vec::new(); n_cent];
    for (d, &c) in assignments.iter().enumerate() {
        cluster_docs[c as usize].push(d as u32);
    }
    let mut codes_layout = vec![0u8; n_docs * code_bytes];
    let mut full_layout = vec![0f32; n_docs * dim];
    let mut doc_ids_layout = vec![0u32; n_docs];
    let mut cluster_index: Vec<(u32, u32)> = Vec::with_capacity(n_cent);
    let mut write_cursor: usize = 0;
    for cdocs in cluster_docs.iter().take(n_cent) {
        let cluster_off = write_cursor as u32;
        let cluster_count = cdocs.len() as u32;
        cluster_index.push((cluster_off, cluster_count));
        for &d in cdocs {
            let d_us = d as usize;
            codes_layout[write_cursor * code_bytes..(write_cursor + 1) * code_bytes]
                .copy_from_slice(&all_codes[d_us * code_bytes..(d_us + 1) * code_bytes]);
            full_layout[write_cursor * dim..(write_cursor + 1) * dim]
                .copy_from_slice(&vectors[d_us * dim..(d_us + 1) * dim]);
            doc_ids_layout[write_cursor] = d;
            write_cursor += 1;
        }
    }
    debug_assert_eq!(write_cursor, n_docs);

    // 6. Build the subsection bytes.
    let summary_size = dim * 4;
    let centroids_size = n_cent * dim * 4;
    let cluster_idx_size = n_cent * 8;
    let codes_size = n_docs * code_bytes;
    let full_size = n_docs * dim * 4;
    let doc_ids_size = n_docs * 4;

    // Offsets relative to subsection start.
    let summary_off = SUB_HEADER_SIZE;
    let centroids_off = summary_off + summary_size;
    let cluster_idx_off = centroids_off + centroids_size;
    let codes_off = cluster_idx_off + cluster_idx_size;
    let full_off = codes_off + codes_size;
    // doc_ids start at full_off + full_size; not stored explicitly in the sub-header.

    let total_size_before_crc = SUB_HEADER_SIZE
        + summary_size
        + centroids_size
        + cluster_idx_size
        + codes_size
        + full_size
        + doc_ids_size;

    let mut bytes: Vec<u8> = Vec::with_capacity(total_size_before_crc + 4);

    // Sub-header (56 bytes).
    bytes.extend_from_slice(format::vec::SUB_MAGIC); // 8
    bytes.extend_from_slice(&format::vec::VERSION.to_le_bytes()); // 4
    bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved (4)
    bytes.extend_from_slice(&(summary_off as u64).to_le_bytes()); // summary_centroid_offset (8)
    bytes.extend_from_slice(&summary_radius_x100.to_le_bytes()); // 4
    bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved (4)
    bytes.extend_from_slice(&(centroids_off as u64).to_le_bytes()); // 8
    bytes.extend_from_slice(&(cluster_idx_off as u64).to_le_bytes()); // 8
    bytes.extend_from_slice(&(codes_off as u32).to_le_bytes()); // 4
    bytes.extend_from_slice(&(full_off as u32).to_le_bytes()); // 4
    debug_assert_eq!(bytes.len(), SUB_HEADER_SIZE);

    // Summary centroid (dim f32s).
    bytes.extend_from_slice(bytemuck::cast_slice(&summary_centroid));
    // Centroids.
    bytes.extend_from_slice(bytemuck::cast_slice(&centroids));
    // Cluster index.
    for (off, count) in &cluster_index {
        bytes.extend_from_slice(&off.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    // Codes.
    bytes.extend_from_slice(&codes_layout);
    // Full.
    bytes.extend_from_slice(bytemuck::cast_slice(&full_layout));
    // Doc IDs.
    bytes.extend_from_slice(bytemuck::cast_slice(&doc_ids_layout));
    debug_assert_eq!(bytes.len(), total_size_before_crc);

    // Trailing CRC over the subsection body.
    let crc = crc32c(&bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());

    SubsectionBytes {
        bytes,
        summary_offset_in_sub: summary_off,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(name: &str, dim: usize) -> VectorConfig {
        VectorConfig {
            name: name.to_string(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::L2Sq,
        }
    }

    #[test]
    fn register_column_returns_sequential_ids() {
        let mut b = VectorBuilder::new();
        assert_eq!(b.register_column(cfg("a", 16)).expect("register column"), 0);
        assert_eq!(b.register_column(cfg("b", 32)).expect("register column"), 1);
    }

    #[test]
    fn register_column_rejects_separator_in_name() {
        let mut b = VectorBuilder::new();
        let bad = cfg("a\x1Fb", 16);
        let err = b.register_column(bad).expect_err("expected error");
        assert!(matches!(err, BuildError::ReservedSeparatorInColumnName(_)));
    }

    #[test]
    fn register_column_rejects_inf_prefix() {
        let mut b = VectorBuilder::new();
        let bad = cfg("inf.embedding", 16);
        let err = b.register_column(bad).expect_err("expected error");
        assert!(matches!(err, BuildError::ReservedPrefixInColumnName(_)));
    }

    #[test]
    fn register_column_rejects_dim_too_small() {
        let mut b = VectorBuilder::new();
        let err = b.register_column(cfg("a", 8)).expect_err("expected error");
        assert!(matches!(err, BuildError::VectorDimOutOfRange { .. }));
    }

    #[test]
    fn register_column_rejects_dim_too_large() {
        let mut b = VectorBuilder::new();
        let err = b
            .register_column(cfg("a", 5000))
            .expect_err("expected error");
        assert!(matches!(err, BuildError::VectorDimOutOfRange { .. }));
    }

    #[test]
    fn register_column_rejects_duplicate() {
        let mut b = VectorBuilder::new();
        b.register_column(cfg("a", 16)).expect("register column");
        let err = b.register_column(cfg("a", 32)).expect_err("expected error");
        assert!(matches!(err, BuildError::DuplicateColumnName(_)));
    }

    #[test]
    fn add_rejects_unknown_column_id() {
        let mut b = VectorBuilder::new();
        b.register_column(cfg("a", 16)).expect("register column");
        let err = b.add(99, &[0.0; 16]).expect_err("expected error");
        assert!(matches!(err, BuildError::FtsColumnTypeInvalid { .. }));
    }

    #[test]
    fn add_rejects_wrong_dim() {
        let mut b = VectorBuilder::new();
        b.register_column(cfg("a", 16)).expect("register column");
        let err = b.add(0, &[0.0; 8]).expect_err("expected error");
        assert!(matches!(err, BuildError::FtsColumnTypeInvalid { .. }));
    }

    #[test]
    fn finish_emits_valid_outer_header() {
        let mut b = VectorBuilder::new();
        b.register_column(cfg("a", 16)).expect("register column");
        for i in 0..32 {
            let v: Vec<f32> = (0..16).map(|j| (i + j) as f32).collect();
            b.add(0, &v).expect("add to vector builder");
        }
        let blob = b.finish();
        assert_eq!(&blob[0..8], format::vec::OUTER_MAGIC);
        let version = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
        assert_eq!(version, format::vec::VERSION);
        let n_cols = u32::from_le_bytes([blob[12], blob[13], blob[14], blob[15]]);
        assert_eq!(n_cols, 1);
    }

    #[test]
    fn finish_with_no_docs_produces_valid_blob() {
        let mut b = VectorBuilder::new();
        b.register_column(cfg("a", 16)).expect("register column");
        let blob = b.finish();
        assert_eq!(&blob[0..8], format::vec::OUTER_MAGIC);
        // n_docs == 0
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&blob[16..24]);
        assert_eq!(u64::from_le_bytes(buf), 0);
    }

    #[test]
    fn finish_two_columns_at_different_dims() {
        let mut b = VectorBuilder::new();
        b.register_column(cfg("a", 16)).expect("register column");
        b.register_column(cfg("b", 32)).expect("register column");
        for _ in 0..16 {
            b.add(0, &[1.0; 16]).expect("add to vector builder");
            b.add(1, &[1.0; 32]).expect("add to vector builder");
        }
        let blob = b.finish();
        let n_cols = u32::from_le_bytes([blob[12], blob[13], blob[14], blob[15]]);
        assert_eq!(n_cols, 2);
        // Different dims means different subsection sizes.
        // The directory should reflect it: parse first two entries.
        let dir_off = OUTER_HEADER_SIZE;
        let entry_a_dim = u32::from_le_bytes([
            blob[dir_off + 4],
            blob[dir_off + 5],
            blob[dir_off + 6],
            blob[dir_off + 7],
        ]);
        let entry_b_dim = u32::from_le_bytes([
            blob[dir_off + DIR_ENTRY_SIZE + 4],
            blob[dir_off + DIR_ENTRY_SIZE + 5],
            blob[dir_off + DIR_ENTRY_SIZE + 6],
            blob[dir_off + DIR_ENTRY_SIZE + 7],
        ]);
        assert_eq!(entry_a_dim, 16);
        assert_eq!(entry_b_dim, 32);
    }
}
