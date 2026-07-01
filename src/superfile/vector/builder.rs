// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Vector blob builder. Multi-column unified blob with per-column
//! self-contained subsections.
//!
//! Each column's subsection is a self-contained IVF + RaBitQ index:
//! summary centroid + radius, IVF centroids (from k-means), cluster
//! index, 1-bit codes, full-precision vectors, doc_ids — all in
//! cluster-contiguous order so the rerank loop stays in cache.
//!
//! See `docs/architecture/superfile.md` for the full byte-level spec.

use std::{
    cmp::Ordering,
    fs::{File, metadata},
    io::{BufReader, BufWriter, Error as IoError, ErrorKind, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use rayon::prelude::*;

use crate::superfile::{
    BuildError,
    format::{
        self, FST_SEPARATOR, RESERVED_PREFIX,
        checksum::{crc32c, crc32c_append},
        vec::{
            CLUSTER_IDX_COUNT_OFFSET, CLUSTER_IDX_ENTRY_BYTES, MAGIC_BYTES, U32_BYTES, U64_BYTES,
            sub_hdr,
        },
    },
    vector::{
        cell_posting::{
            MaterializedIvfRow, encoded_component_at,
            materialize_sq8_residual_row_into_cluster_quant,
        },
        distance::{Metric, SQ8_RESIDUAL_DIVISOR, l2_sq, metric_distance_by},
        ivf_merge::MergedIvfSubsection,
        kmeans::{assign_to_centroids, kmeans},
        quant::BitQuantizer,
        rerank_codec::RerankCodec,
        reservoir::{Reservoir, default_kmeans_sample_size, partition_kmeans_sample_size},
        rotation::RandomRotation,
        spill::{ChunkedVectorSource, InMemoryVectorSource, MmapVectorSource, SpillWriter},
        sq8_simd::{Sq8EncodeConsts, sq8_encode_row, update_min_max},
    },
};

/// Outer-header size (magic + version + n_columns + n_docs + dir_offset).
const OUTER_HEADER_SIZE: usize = format::vec::OUTER_HEADER_SIZE;

/// Subsection-directory entry size in bytes.
const DIR_ENTRY_SIZE: usize = format::vec::DIR_ENTRY_SIZE;

/// Per-column sub-header size (inside each subsection).
const SUB_HEADER_SIZE: usize = format::vec::SUB_HEADER_SIZE;

/// Smallest accepted vector dimension. Below this the IVF + 1-bit
/// quantizer carries too little signal to be worth indexing.
const VECTOR_DIM_MIN: usize = 16;

/// Largest accepted vector dimension. Caps per-vector build memory
/// and keeps the rotation matrix (`dim × dim`) tractable.
const VECTOR_DIM_MAX: usize = 4096;

/// XOR mask applied to a column's `rot_seed` to seed the training
/// reservoir RNG. Keeps the reservoir's PRNG stream deterministic
/// with the column config but distinct from the rotation and
/// k-means streams.
const RESERVOIR_SEED_XOR_MASK: u64 = 0x5a5a_5a5a_5a5a_5a5a;

/// Lloyd k-means iteration count for pass-1 centroid training. Five
/// is the standard turn-key default; returns diminish past it on
/// typical embedding distributions.
const KMEANS_ITERS: usize = 5;

/// Target memory budget (~128 MiB) for one pass-2 rotated chunk
/// (`chunk_rows × dim × 4` bytes); the chunk row count is derived
/// from this so resident memory stays bounded independent of `dim`.
const PASS2_CHUNK_MEM_BUDGET_BYTES: usize = 128 << 20;

/// Floor on pass-2 chunk rows, keeping chunks wide enough to stay
/// SIMD-friendly even at extreme dimensions.
const PASS2_CHUNK_ROWS_MIN: usize = 1024;

/// Ceiling on pass-2 chunk rows, capping per-chunk RAM at small dims.
const PASS2_CHUNK_ROWS_MAX: usize = 65_536;

/// Superfile-local document thresholds for capping the physical IVF centroid
/// count. Caller-supplied `n_cent` remains a tuning knob, but the builder will
/// not train more centroids than this row-count policy permits for the physical
/// superfile being written.
const N_CENT_LARGE_DOC_THRESHOLD: usize = 5_000_000;
/// Maximum IVF centroids for large physical vector indexes.
const N_CENT_LARGE: usize = 4096;
/// Medium-index document threshold for the IVF centroid cap.
const N_CENT_MEDIUM_DOC_THRESHOLD: usize = 100_000;
/// Maximum IVF centroids for medium physical vector indexes.
const N_CENT_MEDIUM: usize = 1024;
/// Maximum IVF centroids for small physical vector indexes.
const N_CENT_SMALL: usize = 64;

/// Fixed-point scale for the per-subsection summary radius. The
/// radius is stored as `round(radius × 100)` in a `u32` and decoded
/// back by the reader as `value / 100.0`.
const SUMMARY_RADIUS_SCALE: f32 = 100.0;

/// Maximum Sq8 code value: each component quantizes to one unsigned
/// byte, so the per-dim scale maps a cluster's value span onto
/// `[0, SQ8_CODE_MAX]`.
const SQ8_CODE_MAX: f32 = 255.0;

/// Symmetric clamp bound for the Sq8ResidualEpsilon i8 leg. The residual is
/// stored as a signed byte but clamped to ±127 (not i8::MIN) so the
/// quantized magnitude stays symmetric about zero.
const SQ8_RESIDUAL_I8_CLAMP: f32 = 127.0;

fn n_cent_row_count_cap(n_docs: usize) -> usize {
    if n_docs >= N_CENT_LARGE_DOC_THRESHOLD {
        N_CENT_LARGE
    } else if n_docs >= N_CENT_MEDIUM_DOC_THRESHOLD {
        N_CENT_MEDIUM
    } else {
        N_CENT_SMALL
    }
}

/// Metric ID encoding for the directory entry. Spec: 0 = L2Sq, 1 = Cosine,
/// 2 = NegDot.
fn metric_id(m: Metric) -> u32 {
    match m {
        Metric::L2Sq => format::vec::METRIC_ID_L2SQ,
        Metric::Cosine => format::vec::METRIC_ID_COSINE,
        Metric::NegDot => format::vec::METRIC_ID_NEGDOT,
    }
}

/// Per-vector-index build configuration.
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// Logical column name. Must not collide with any other
    /// logical index in the same superfile (FTS or vector). Named
    /// `column` for API compatibility with `FtsConfig::column`; semantically
    /// this is the logical vector index name; this is also the on-disk
    /// JSON key in `inf.vec.columns`.
    pub column: String,
    pub dim: usize,
    pub n_cent: usize,
    pub rot_seed: u64,
    pub metric: Metric,
    /// On-disk rerank codec for this column. See [`RerankCodec`]
    /// for the supported codecs and their size/recall trade-offs.
    pub rerank_codec: RerankCodec,
    /// When `Some`, the IVF build skips local k-means and partitions
    /// against these caller-supplied centroids (cluster-major fp32,
    /// `n_cent * dim`). Used by the hidden-index incoming build so every
    /// shard shares the global cell ordinals and the drain can splice
    /// cluster `c` → cell `c` without re-clustering. `n_cent` is taken
    /// from the supplied centroids (`len / dim`), overriding the field.
    pub provided_centroids: Option<std::sync::Arc<[f32]>>,
}

impl VectorConfig {
    /// Construct a config with the default rerank codec
    /// ([`RerankCodec::default()`]). Use [`Self::with_rerank_codec`]
    /// to override.
    pub fn new(column: String, dim: usize, n_cent: usize, rot_seed: u64, metric: Metric) -> Self {
        Self {
            column,
            dim,
            n_cent,
            rot_seed,
            metric,
            rerank_codec: RerankCodec::default(),
            provided_centroids: None,
        }
    }

    /// Override the rerank codec.
    #[must_use]
    pub fn with_rerank_codec(mut self, codec: RerankCodec) -> Self {
        self.rerank_codec = codec;
        self
    }

    /// Partition against caller-supplied global centroids instead of
    /// local k-means. See [`Self::provided_centroids`].
    #[must_use]
    pub fn with_provided_centroids(mut self, centroids: Option<std::sync::Arc<[f32]>>) -> Self {
        self.provided_centroids = centroids;
        self
    }
}

/// Default spill threshold: total bytes the in-memory pre-spill
/// buffer is allowed to grow to before the column transitions to
/// the on-disk path. 256 MiB is a constant — independent of
/// reservoir size or `n_cent` — so the worst-case pre-flush
/// resident moment (`reservoir + spill_threshold`) stays linear
/// in reservoir only and never compounds. design § "spill_threshold_bytes default".
const DEFAULT_SPILL_THRESHOLD_BYTES: usize = 256 * 1024 * 1024;

/// Per-column build-time state. With the streaming build path
/// the column holds at most three independent buffers:
///
/// - [`Reservoir`]: bounded k-means training sample. Dropped at
///   the pass 1 → pass 2 boundary inside `build_subsection_streaming`.
/// - `pre_spill_buffer`: lossless input backing while
///   `n_docs * dim * 4 ≤ spill_threshold_bytes`. Drained to
///   capacity 0 once the threshold is crossed.
/// - `spill`: an `Option<SpillWriter>` that owns an
///   append-only temp file containing the full input corpus in
///   raw little-endian f32 once the threshold is crossed.
///
/// At any given moment one of `pre_spill_buffer` or `spill` is
/// the canonical input store; the reservoir is always live (and
/// orthogonal). Once `finish()` runs, the active store is wrapped
/// in a [`ChunkedVectorSource`] for pass 2.
struct ColumnState {
    config: VectorConfig,
    n_docs: u32,
    reservoir: Reservoir,
    /// Lossless input backing while below the spill threshold.
    /// Holds vectors in insertion order, never overwrites. Drained
    /// to `Vec::new()` (releasing capacity) the moment the build
    /// transitions to the spill path.
    pre_spill_buffer: Vec<f32>,
    /// Once `pre_spill_buffer.len() * 4 + vec.len() * 4 >
    /// spill_threshold_bytes` on an `add()`, this becomes `Some`,
    /// the pre-spill buffer is flushed into it, and from then on
    /// every `add()` writes the new vector straight to disk.
    spill: Option<SpillWriter>,
    spill_threshold_bytes: usize,
    /// Sq8-native maintenance rows: when set, finish uses the materialized IVF
    /// rebuild path instead of the fp32 ingest pipeline.
    materialized_rows: Option<Vec<MaterializedIvfRow>>,
    /// Pre-built subsection bytes from byte-splice merge (compaction path).
    prebuilt_subsection: Option<SubsectionBytes>,
}

/// Lazily-created scratch directory for vector spill and bucket files.
///
/// `VectorBuilder::new()` should be cheap for tiny builders. We only
/// allocate the backing tempdir when the build actually needs scratch:
/// either input spills during `add()` or finish-time bucket files are
/// produced.
#[derive(Default)]
struct ScratchDir {
    parent: Option<PathBuf>,
    tempdir: Option<tempfile::TempDir>,
}

impl ScratchDir {
    fn in_parent(parent: PathBuf) -> Result<Self, BuildError> {
        let meta = metadata(&parent)?;
        if !meta.is_dir() {
            return Err(BuildError::Io(IoError::new(
                ErrorKind::InvalidInput,
                format!("VectorBuilder scratch path is not a directory: {parent:?}"),
            )));
        }
        Ok(Self {
            parent: Some(parent),
            tempdir: None,
        })
    }

    fn path(&mut self) -> Result<&Path, BuildError> {
        if self.tempdir.is_none() {
            let tmp = if let Some(parent) = &self.parent {
                tempfile::TempDir::new_in(parent)?
            } else {
                tempfile::tempdir()?
            };
            self.tempdir = Some(tmp);
        }
        Ok(self
            .tempdir
            .as_ref()
            .expect("scratch tempdir initialized")
            .path())
    }
}

/// Multi-index vector blob builder. The streaming build path changes
/// the builder from "accumulate full corpus in RAM" to
/// "reservoir-sample + spill to disk past a threshold"; peak
/// resident memory is now a function of `(reservoir, n_cent,
/// dim, chunk_size, bucket_buf_size)` rather than `(n_docs,
/// dim)`.
pub struct VectorBuilder {
    columns: Vec<ColumnState>,
    /// Per-builder scratch directory holder. The actual tempdir is
    /// created lazily, so callers whose builders are dropped before
    /// spill/finish do not pay filesystem setup cost.
    scratch_dir: ScratchDir,
    spill_threshold_bytes: usize,
}

impl Default for VectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorBuilder {
    /// Construct a builder with the default scratch directory
    /// (under `$TMPDIR` via `tempfile::tempdir()`) and the
    /// default 256 MiB spill threshold.
    ///
    /// The scratch tempdir is created lazily when the build first
    /// needs scratch space. Operators running large builds should
    /// prefer [`Self::with_scratch`] pointing at an instance-store
    /// NVMe partition.
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            scratch_dir: ScratchDir::default(),
            spill_threshold_bytes: DEFAULT_SPILL_THRESHOLD_BYTES,
        }
    }

    /// Construct a builder with `scratch` as the scratch root.
    /// The directory must already exist and be writable. Used
    /// for benchmarks + production deployments that want to pin
    /// scratch to instance-store NVMe (`/mnt/nvme0/infino-build`,
    /// etc.) instead of the default `$TMPDIR` (which on EC2 is
    /// typically EBS-backed `/tmp`).
    pub fn with_scratch(scratch: PathBuf) -> Result<Self, BuildError> {
        Ok(Self {
            columns: Vec::new(),
            scratch_dir: ScratchDir::in_parent(scratch)?,
            spill_threshold_bytes: DEFAULT_SPILL_THRESHOLD_BYTES,
        })
    }

    /// Override the spill threshold (bytes the pre-spill buffer
    /// can grow to before flushing to disk). Must be called
    /// before any `add()` for the override to apply — column
    /// states copy this on construction, so changes after a
    /// column is registered don't retroactively apply.
    ///
    /// 256 MiB is the default; useful overrides include 0 (force
    /// every column straight to spill, for testing the spill
    /// path) and very large values (force pure in-RAM builds for
    /// tiny corpora where the spill path isn't worth the
    /// overhead).
    pub fn set_spill_threshold_bytes(&mut self, threshold: usize) {
        self.spill_threshold_bytes = threshold;
    }

    /// Register a logical vector index up-front. Returns the assigned
    /// `column_id` (declaration order).
    pub fn register_column(&mut self, config: VectorConfig) -> Result<u32, BuildError> {
        if config.column.as_bytes().contains(&FST_SEPARATOR) {
            return Err(BuildError::ReservedSeparatorInColumnName(config.column));
        }
        if config.column.starts_with(RESERVED_PREFIX) {
            return Err(BuildError::ReservedPrefixInColumnName(config.column));
        }
        if !(VECTOR_DIM_MIN..=VECTOR_DIM_MAX).contains(&config.dim) {
            return Err(BuildError::VectorDimOutOfRange {
                column: config.column.clone(),
                dim: config.dim,
            });
        }
        if self
            .columns
            .iter()
            .any(|c| c.config.column == config.column)
        {
            return Err(BuildError::DuplicateColumnName(config.column));
        }
        if !config.rerank_codec.is_implemented() {
            return Err(BuildError::VectorRerankCodecUnimplemented {
                column: config.column.clone(),
                codec: config.rerank_codec.name(),
            });
        }
        let column_id = self.columns.len() as u32;
        let sample_size = default_kmeans_sample_size(config.n_cent);
        // Seed the reservoir RNG from `rot_seed ^ 0x5a5a` so it
        // stays deterministic with the column config but uses a
        // distinct stream from `RandomRotation` (which seeds from
        // `rot_seed` directly) and `kmeans` (which seeds from
        // `rot_seed + 7`). Three disjoint streams, three
        // deterministic seeds, one knob on the user's end.
        let reservoir_seed = config.rot_seed ^ RESERVOIR_SEED_XOR_MASK;
        let reservoir = Reservoir::new(sample_size, config.dim, reservoir_seed);
        let spill_threshold_bytes = self.spill_threshold_bytes;
        self.columns.push(ColumnState {
            config,
            n_docs: 0,
            reservoir,
            pre_spill_buffer: Vec::new(),
            spill: None,
            spill_threshold_bytes,
            materialized_rows: None,
            prebuilt_subsection: None,
        });
        Ok(column_id)
    }

    /// Load Sq8+ε maintenance rows for one column. Reuses the normal IVF
    /// subsection writer on finish — no fp32 corpus decode.
    pub(crate) fn load_materialized_rows(
        &mut self,
        column_id: u32,
        rows: Vec<MaterializedIvfRow>,
    ) -> Result<(), BuildError> {
        let idx = column_id as usize;
        let col = self
            .columns
            .get_mut(idx)
            .ok_or_else(|| BuildError::FtsColumnTypeInvalid {
                column: format!("(unregistered vector column_id {column_id})"),
                actual: "n/a".to_string(),
            })?;
        if !matches!(col.config.rerank_codec, RerankCodec::Sq8ResidualEpsilon) {
            return Err(BuildError::VectorRerankCodecUnimplemented {
                column: col.config.column.clone(),
                codec: col.config.rerank_codec.name(),
            });
        }
        col.n_docs = rows.len() as u32;
        col.materialized_rows = Some(rows);
        Ok(())
    }

    /// Inject a pre-built IVF subsection (byte-splice merge). Skips the
    /// materialized rebuild and fp32 ingest paths on finish.
    pub(crate) fn set_prebuilt_subsection(
        &mut self,
        column_id: u32,
        subsection: MergedIvfSubsection,
    ) -> Result<(), BuildError> {
        let idx = column_id as usize;
        let col = self
            .columns
            .get_mut(idx)
            .ok_or_else(|| BuildError::FtsColumnTypeInvalid {
                column: format!("(unregistered vector column_id {column_id})"),
                actual: "n/a".to_string(),
            })?;
        col.n_docs = subsection.n_docs;
        col.materialized_rows = None;
        col.prebuilt_subsection = Some(SubsectionBytes {
            bytes: subsection.bytes,
            n_cent: subsection.n_cent,
            summary_offset_in_sub: subsection.summary_offset_in_sub,
            codec_meta_offset_in_sub: subsection.codec_meta_offset_in_sub,
            codec_meta_size: subsection.codec_meta_size,
        });
        Ok(())
    }

    /// Override the k-means training sample size for a column. Must
    /// be called before the first `add()` for the column — calling it
    /// later silently discards already-observed reservoir state and
    /// only future `add()` calls feed into the new reservoir.
    ///
    /// The default sample size is `default_kmeans_sample_size(n_cent)`
    /// (`100K-500K` depending on `n_cent`). This override exists for
    /// (a) sample-size sweeps on synthetic recall corpora and
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
        let reservoir_seed = col.config.rot_seed ^ RESERVOIR_SEED_XOR_MASK;
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
        {
            let col = &mut self.columns[idx];
            if vec.len() != col.config.dim {
                return Err(BuildError::FtsColumnTypeInvalid {
                    column: col.config.column.clone(),
                    actual: format!("vec.len()={} != dim={}", vec.len(), col.config.dim),
                });
            }
            col.reservoir.update(vec);

            // Append to the lossless input backing. Three cases,
            // in order of likelihood once a build is established:
            //
            //   1. Spill is already active (column has already
            //      crossed the threshold): write the vector
            //      directly to disk via SpillWriter. The buffer is
            //      empty in this state.
            //   2. This add() crosses the threshold: create the
            //      SpillWriter, drain pre_spill_buffer in one
            //      batched write, append the new vector, then
            //      release pre_spill_buffer capacity.
            //   3. Pre-spill mode: extend the in-RAM buffer.
            //
            // The post-spill steady state hits case 1, which is the
            // hot path. The branch order is chosen to put case 1
            // first so the predictor learns the steady state.
            let vec_bytes = vec.len() * 4;
            let buf_bytes = col.pre_spill_buffer.len() * 4;
            if let Some(spill) = col.spill.as_mut() {
                spill.write_vec(vec)?;
                col.n_docs += 1;
                return Ok(());
            }
            if buf_bytes + vec_bytes <= col.spill_threshold_bytes {
                col.pre_spill_buffer.extend_from_slice(vec);
                col.n_docs += 1;
                return Ok(());
            }
        }

        let path = self
            .scratch_dir
            .path()?
            .join(format!("infino_input_spill_col{column_id}.bin"));
        let col = &mut self.columns[idx];
        let mut spill = SpillWriter::create(path)?;
        spill.write_all(bytemuck::cast_slice(&col.pre_spill_buffer))?;
        spill.write_vec(vec)?;
        col.pre_spill_buffer = Vec::new();
        col.spill = Some(spill);
        col.n_docs += 1;
        Ok(())
    }

    /// Finalise and emit the unified vector blob. Consumes the
    /// builder.
    ///
    /// Returns a `BuildError::Io` for the spill / scratch I/O
    /// errors of the streaming build. Callers that previously
    /// expected `-> Vec<u8>` need to `?` the result; the
    /// `SuperfileBuilder` shim does so already.
    pub fn finish(self) -> Result<Vec<u8>, BuildError> {
        // Capacity hint: the largest known-cheap pre-allocation is
        // `OUTER_HEADER_SIZE + (n_columns × DIR_ENTRY_SIZE) + 8`
        // (header + directory + dir_crc + outer_crc). Subsection
        // bytes are unknown until built; the inner `Write` impl on
        // `Vec` will grow as needed.
        let header_dir_hint = OUTER_HEADER_SIZE + (self.columns.len() * DIR_ENTRY_SIZE) + 8;
        let mut buf: Vec<u8> = Vec::with_capacity(header_dir_hint);
        self.finish_to(&mut buf)?;
        Ok(buf)
    }

    /// Streaming variant: write the final blob progressively to
    /// `w` without materialising it as a contiguous `Vec<u8>`.
    ///
    /// The output bytes (outer header, directory + dir CRC, each
    /// subsection, trailing outer CRC) are identical to those
    /// produced by [`Self::finish`] — `finish` is now a thin
    /// wrapper that calls `finish_to(&mut Vec<u8>)`.
    ///
    /// The trailing outer CRC32C is computed incrementally via
    /// `crc32c_append` so we never need to retain the full blob
    /// in memory to checksum it.
    ///
    /// Subsections are still built one-at-a-time into a
    /// `Vec<u8>` (their internal CRC is computed at the end of
    /// each subsection's body); each subsection is dropped as
    /// soon as it has been written to `w`, so peak heap drops
    /// from `sum_of_subsection_sizes + final_blob_size` to
    /// `max_subsection_size`. Per-subsection streaming would
    /// push the floor lower still.
    ///
    /// Object-storage callers can pass a multipart upload
    /// writer here so superfile build never owns the full blob in
    /// RAM.
    pub fn finish_to<W: Write>(self, mut w: W) -> Result<(), BuildError> {
        let VectorBuilder {
            columns,
            mut scratch_dir,
            spill_threshold_bytes: _,
        } = self;

        let n_columns = columns.len() as u32;
        // n_docs in the outer header is the max across columns
        // (per-superfile doc count; spec: same across all columns).
        let n_docs: u64 = columns.iter().map(|c| c.n_docs as u64).max().unwrap_or(0);

        // Snapshot config + n_docs first so the directory loop
        // can read them after we've consumed each ColumnState.
        let column_configs: Vec<(VectorConfig, u32)> = columns
            .iter()
            .map(|c| (c.config.clone(), c.n_docs))
            .collect();

        // 1. Build each per-column subsection independently. Each
        //    subsection is self-contained — sub-header + summary +
        //    centroids + cluster index + codes + full + doc_ids + CRC.
        //    Consumes each ColumnState so the reservoir,
        //    pre_spill_buffer, and (if any) spill file can be
        //    released as soon as the subsection bytes for that
        //    column are produced.
        let mut subsections: Vec<SubsectionBytes> = Vec::with_capacity(columns.len());
        if !columns.is_empty() {
            let scratch_path = scratch_dir.path()?.to_path_buf();
            for (col_idx, col) in columns.into_iter().enumerate() {
                if let Some(prebuilt) = col.prebuilt_subsection {
                    subsections.push(prebuilt);
                    continue;
                }
                subsections.push(build_subsection_streaming(
                    col_idx as u32,
                    col,
                    &scratch_path,
                )?);
            }
        }

        // 2. Layout: outer_header(32) + directory(n_columns * 64) +
        //    dir_crc(4) + subsections concatenated + outer_crc(4).
        let directory_offset = OUTER_HEADER_SIZE as u64;
        let directory_size = (n_columns as usize) * DIR_ENTRY_SIZE;
        let mut subsection_start_off =
            directory_offset + directory_size as u64 + format::CRC_BYTES as u64;

        // 3. Assemble directory entries with absolute offsets.
        //    Byte 52 carries the rerank-codec discriminator.
        //    Bytes 56..64 carry codec_meta offset/length within the
        //    subsection so lazy open can fetch subsection headers and
        //    Sq8 metadata in the same network batch.
        let mut directory: Vec<u8> = Vec::with_capacity(directory_size);
        for (i, sub) in subsections.iter().enumerate() {
            let (cfg, _) = &column_configs[i];
            let summary_offset_abs = subsection_start_off + sub.summary_offset_in_sub as u64;
            directory.extend_from_slice(&(i as u32).to_le_bytes()); // column_id
            directory.extend_from_slice(&(cfg.dim as u32).to_le_bytes()); // dim
            directory.extend_from_slice(&(sub.n_cent as u32).to_le_bytes()); // physical n_cent
            directory.extend_from_slice(&metric_id(cfg.metric).to_le_bytes()); // metric_id
            directory.extend_from_slice(&cfg.rot_seed.to_le_bytes()); // rot_seed (8)
            directory.extend_from_slice(&subsection_start_off.to_le_bytes()); // subsection_offset (8)
            directory.extend_from_slice(&(sub.bytes.len() as u64).to_le_bytes()); // subsection_length (8)
            directory.extend_from_slice(&summary_offset_abs.to_le_bytes()); // summary_offset (8)
            directory.extend_from_slice(&((cfg.dim * 4) as u32).to_le_bytes()); // summary_length (4)
            // bytes 52..56 — codec_id (1) + reserved (3)
            directory.push(cfg.rerank_codec.codec_id()); // codec_id (1)
            directory.extend_from_slice(&[0u8; 3]); // reserved (3)
            directory.extend_from_slice(&(sub.codec_meta_offset_in_sub as u32).to_le_bytes());
            directory.extend_from_slice(&(sub.codec_meta_size as u32).to_le_bytes());
            debug_assert_eq!(directory.len() % DIR_ENTRY_SIZE, 0);

            subsection_start_off += sub.bytes.len() as u64;
        }
        let dir_crc = crc32c(&directory);

        // 4. Stream out: outer_header → directory → dir_crc →
        //    subsections (drained, one at a time) → outer_crc.
        //    A running CRC32C accumulator covers every byte
        //    written before the outer CRC itself, so we never
        //    need the full blob in memory to checksum it.

        // Outer header (32 bytes).
        let mut outer_header: [u8; OUTER_HEADER_SIZE] = [0; OUTER_HEADER_SIZE];
        {
            let mut cursor = &mut outer_header[..];
            cursor
                .write_all(format::vec::OUTER_MAGIC) // 8
                .map_err(BuildError::Io)?;
            cursor
                .write_all(&format::vec::VERSION.to_le_bytes()) // 4
                .map_err(BuildError::Io)?;
            cursor
                .write_all(&n_columns.to_le_bytes()) // 4
                .map_err(BuildError::Io)?;
            cursor
                .write_all(&n_docs.to_le_bytes()) // 8
                .map_err(BuildError::Io)?;
            cursor
                .write_all(&directory_offset.to_le_bytes()) // 8
                .map_err(BuildError::Io)?;
            debug_assert!(cursor.is_empty());
        }

        let mut outer_crc_acc: u32 = 0;
        w.write_all(&outer_header).map_err(BuildError::Io)?;
        outer_crc_acc = crc32c_append(outer_crc_acc, &outer_header);

        // Directory + dir CRC.
        w.write_all(&directory).map_err(BuildError::Io)?;
        outer_crc_acc = crc32c_append(outer_crc_acc, &directory);
        let dir_crc_le = dir_crc.to_le_bytes();
        w.write_all(&dir_crc_le).map_err(BuildError::Io)?;
        outer_crc_acc = crc32c_append(outer_crc_acc, &dir_crc_le);
        drop(directory);

        // Subsections — drain so each subsection Vec drops the
        // instant we've finished writing + CRC-ing it. At 10M ×
        // 384 a subsection is ~15 GiB, so retaining all of them
        // until the last byte is written would double the peak.
        for sub in subsections.drain(..) {
            w.write_all(&sub.bytes).map_err(BuildError::Io)?;
            outer_crc_acc = crc32c_append(outer_crc_acc, &sub.bytes);
        }

        // Trailing whole-blob CRC32C.
        let outer_crc_le = outer_crc_acc.to_le_bytes();
        w.write_all(&outer_crc_le).map_err(BuildError::Io)?;

        // scratch_dir is dropped at end of scope, removing spill +
        // bucket files. Keeping it alive until here ensures the
        // mmap-backed pass-2 source in build_subsection_streaming
        // had a live file path for the duration of its scan.
        drop(scratch_dir);

        Ok(())
    }
}

/// Builder output for one column's subsection.
struct SubsectionBytes {
    bytes: Vec<u8>,
    /// Physical IVF centroid count written into this subsection.
    /// May be lower than the configured `n_cent` for tiny shards
    /// where `n_docs < n_cent`.
    n_cent: usize,
    /// Byte offset of the summary centroid relative to the subsection
    /// start (matches the directory entry's `summary_offset` after
    /// translation to absolute).
    summary_offset_in_sub: usize,
    /// Byte offset / length of codec_meta relative to the subsection
    /// start. Both are zero when the subsection has no codec_meta.
    codec_meta_offset_in_sub: usize,
    codec_meta_size: usize,
}

/// Per-bucket BufWriter capacity. 64 KiB amortises one syscall
/// per ~1300 dim=384 bucket rows (each row = 4 + code_bytes +
/// dim*4 = ~1588 B). At very high n_cent (≥ 8192) the n_cent ×
/// 64 KiB total dominates the resident set; this is worth
/// revisiting if profiling shows it.
const BUCKET_BUF_SIZE: usize = 64 * 1024;

/// Adaptive chunk size for pass 2: keeps `chunk_rotated`
/// (`chunk_rows × dim × 4` bytes) below ~128 MiB while
/// preserving SIMD-friendly width at extreme dims.
///
/// At `dim = 16`: `(128 << 20) / 64 = 2 097 152` → clamped to
/// 65 536 (16 MiB chunk). At `dim = 384`: 87 381 → clamped to
/// 65 536 (95 MiB). At `dim = 1024`: 32 768 (128 MiB). At
/// `dim = 4096`: 8 192 (128 MiB). The 1024 floor keeps the
/// chunk wide enough to stay SIMD-friendly even at extreme
/// dimensions.
fn chunk_rows_for_dim(dim: usize) -> usize {
    let cap_by_mem = PASS2_CHUNK_MEM_BUDGET_BYTES / (dim.max(1) * 4);
    cap_by_mem.clamp(PASS2_CHUNK_ROWS_MIN, PASS2_CHUNK_ROWS_MAX)
}

/// Build one column's subsection via the streaming path.
/// Consumes the entire `ColumnState` so the reservoir +
/// pre-spill buffer + spill file are released as soon as their
/// contribution to the subsection is complete.
///
/// Layout produced (identical to the legacy `build_subsection`
/// shape — only the build process changed):
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
///
/// Algorithm (three passes — pass 1 is in-memory, passes 2 and
/// 3 are streaming over the corpus):
///
/// 1. **Pass 1 (small):** k-means on the reservoir sample,
///    yielding `n_cent × dim` centroids. Drops the reservoir
///    before pass 2.
/// 2. **Pass 2 (streaming):** for each chunk of `chunk_rows`
///    vectors from the input source: assign on unrotated rows,
///    rotate, encode to 1-bit codes, append the
///    `(local_doc_id, code, full_vec)` tuple to the assigned
///    centroid's bucket file, and fold the row into the
///    summary radius. Per-centroid bucket files preserve
///    cluster-contiguity for pass 3 without a third corpus
///    pass.
/// 3. **Pass 3 (sequential):** read each bucket file in
///    centroid order, materialising the cluster-contiguous
///    `codes[]`, `full[]`, and `doc_ids[]` regions and the
///    cluster-index entries.
/// Build one column's subsection from Sq8+ε maintenance rows. Reuses the same
/// on-disk IVF layout and pass-3 assembly as [`build_subsection_streaming`].
fn build_subsection_from_materialized(
    cfg: VectorConfig,
    mut rows: Vec<MaterializedIvfRow>,
) -> Result<SubsectionBytes, BuildError> {
    rows.sort_by_key(|r| r.local_doc_id);
    let n_docs = rows.len();
    if n_docs == 0 {
        return Err(BuildError::VectorSchemaMismatch(
            "materialized IVF rebuild requires at least one row".into(),
        ));
    }
    let dim = cfg.dim;
    let (n_cent, centroids) = if let Some(global) = cfg.provided_centroids.as_ref() {
        // Partition against the global cell centroids instead of training
        // local ones. Every incoming shard then shares cell ordinals, so the
        // drain splices cluster `c` → cell `c` with no re-clustering. Keep all
        // `n_cent` cells even when this shard has fewer rows (empty clusters are
        // count-0) so ordinal `c` always means cell `c`.
        debug_assert!(dim > 0 && global.len() % dim == 0);
        let nc = global.len() / dim.max(1);
        (nc, global.to_vec())
    } else {
        let n_cent = cfg
            .n_cent
            .max(1)
            .min(n_cent_row_count_cap(n_docs))
            .min(n_docs.max(1));
        // Train centroids the same way the user-superfile build does: on a bounded
        // sample (NOT every row), via the shared `kmeans` (random init + parallel).
        // The previous bespoke `encoded_ivf_kmeans` trained over every row with
        // O(k²·n) farthest-point seeding on the single-thread maint pool, which hung
        // on large cells. Only the sampled rows are decoded to fp32, so there is no
        // full-corpus fp32 buffer.
        let sample_size = partition_kmeans_sample_size(n_cent).min(n_docs);
        let mut sample = vec![0f32; sample_size * dim];
        for s in 0..sample_size {
            let idx = if sample_size == n_docs {
                s
            } else {
                s * n_docs / sample_size
            };
            let enc = &rows[idx].encoded;
            for (d, slot) in sample[s * dim..(s + 1) * dim].iter_mut().enumerate() {
                *slot = encoded_component_at(enc, d);
            }
        }
        let centroids = kmeans(&sample, dim, n_cent, KMEANS_ITERS, cfg.rot_seed);
        (n_cent, centroids)
    };

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

    let quant = BitQuantizer::new(dim);
    let code_bytes = quant.code_bytes();
    let codec = cfg.rerank_codec;
    debug_assert!(matches!(codec, RerankCodec::Sq8ResidualEpsilon));

    let mut buckets: Vec<Vec<&MaterializedIvfRow>> = vec![Vec::new(); n_cent];
    for row in &rows {
        let mut best_c = 0usize;
        let mut best_score = f32::INFINITY;
        for c in 0..n_cent {
            let cv = &centroids[c * dim..(c + 1) * dim];
            let score = metric_distance_by(
                cfg.metric,
                dim,
                |d| encoded_component_at(&row.encoded, d),
                |d| cv[d],
            );
            if score < best_score {
                best_score = score;
                best_c = c;
            }
        }
        buckets[best_c].push(row);
    }

    let mut bucket_counts = vec![0u32; n_cent];
    let mut summary_radius_sq_max = 0.0f32;
    for (c, bucket) in buckets.iter().enumerate() {
        bucket_counts[c] = bucket.len() as u32;
        for row in bucket {
            let r_sq = (0..dim)
                .map(|d| {
                    let v = encoded_component_at(&row.encoded, d);
                    let diff = v - summary_centroid[d];
                    diff * diff
                })
                .sum::<f32>();
            if r_sq > summary_radius_sq_max {
                summary_radius_sq_max = r_sq;
            }
        }
    }

    // Per-cluster Sq8 quantizer: calibrate from the cluster's per-dim min/max,
    // exactly as the user/streaming build does (`derive_sq8_quantizer_from_min_max`
    // over `update_min_max`). This is O(rows · dim); the previous medoid-based
    // derivation was O(rows² · dim) (summed pairwise distances per cluster) and
    // single-threaded — it spun for minutes on a large post-routing cluster.
    let sq8_quantizers: Vec<(Vec<f32>, Vec<f32>)> = buckets
        .iter()
        .map(|bucket| {
            if bucket.is_empty() {
                return (vec![1.0f32; dim], vec![0.0f32; dim]);
            }
            let mut min = vec![f32::INFINITY; dim];
            let mut max = vec![f32::NEG_INFINITY; dim];
            let mut row_fp = vec![0.0f32; dim];
            for row in bucket.iter() {
                for (d, slot) in row_fp.iter_mut().enumerate() {
                    *slot = encoded_component_at(&row.encoded, d);
                }
                update_min_max(&row_fp, &mut min, &mut max);
            }
            derive_sq8_quantizer_from_min_max(&min, &max)
        })
        .collect();

    let summary_radius_x100 = (summary_radius_sq_max.sqrt() * SUMMARY_RADIUS_SCALE)
        .max(0.0)
        .min(u32::MAX as f32) as u32;

    let cluster_order = centroid_storage_order(&centroids, n_cent, dim);
    let codec_meta_size = codec.codec_meta_bytes(dim, n_docs, n_cent, cfg.metric);
    let per_vec_bytes = codec.per_vector_bytes(dim);
    let cluster_stride = code_bytes + format::vec::DOC_ID_BYTES + per_vec_bytes;
    // The materialized build is the hidden-cell path: inline the stable `_id`
    // as a trailing i128-per-doc region so id+score queries and the drain read
    // it straight from the blob (no scalar `_id` column resolution).
    let stable_ids_region_bytes = n_docs * format::vec::STABLE_ID_BYTES;
    let layout = IvfSubsectionLayout::compute(
        dim,
        n_cent,
        n_docs,
        cluster_stride,
        codec_meta_size,
        stable_ids_region_bytes,
    );
    let total_size_before_crc = layout.total_size_before_crc;

    let mut bytes = alloc_ivf_subsection_with_header(
        &layout,
        codec_meta_size,
        summary_radius_x100,
        &summary_centroid,
        &centroids,
    );

    let sq8_scale_block_off = layout.codec_meta_off;
    let sq8_offset_block_off = sq8_scale_block_off + n_cent * dim * 4;
    let sq8_norms_block_off = match cfg.metric {
        Metric::L2Sq | Metric::Cosine => Some(sq8_offset_block_off + n_cent * dim * 4),
        Metric::NegDot => None,
    };

    for (cid, (scale_c, offset_c)) in sq8_quantizers.iter().enumerate().take(n_cent) {
        let sc_off = sq8_scale_block_off + cid * dim * 4;
        bytes[sc_off..sc_off + dim * 4].copy_from_slice(bytemuck::cast_slice(scale_c));
        let oc_off = sq8_offset_block_off + cid * dim * 4;
        bytes[oc_off..oc_off + dim * 4].copy_from_slice(bytemuck::cast_slice(offset_c));
    }

    let store_norm = sq8_norms_block_off.is_some();
    write_ivf_cluster_blocks(
        &mut bytes,
        &layout,
        &cluster_order,
        &bucket_counts,
        code_bytes,
        per_vec_bytes,
        |bytes, centroid_id, blk| {
            let bucket = &buckets[centroid_id];
            let (scale_c, offset_c) = &sq8_quantizers[centroid_id];
            let mut row_buf = vec![0u8; dim * 2];
            for (i, row) in bucket.iter().enumerate() {
                let code_off = blk.codes_base + i * code_bytes;
                bytes[code_off..code_off + code_bytes].copy_from_slice(&row.rabitq_code);
                let id_off = blk.ids_base + i * format::vec::DOC_ID_BYTES;
                bytes[id_off..id_off + format::vec::DOC_ID_BYTES]
                    .copy_from_slice(&row.local_doc_id.to_le_bytes());

                let norm_sq = materialize_sq8_residual_row_into_cluster_quant(
                    &row.encoded,
                    scale_c,
                    offset_c,
                    dim,
                    &mut row_buf,
                    store_norm,
                );
                let full_off = blk.rerank_base + i * per_vec_bytes;
                bytes[full_off..full_off + dim * 2].copy_from_slice(&row_buf);
                if let (Some(norms_off), Some(n_sq)) = (sq8_norms_block_off, norm_sq) {
                    let n_off = norms_off + (blk.first_row + i) * 4;
                    bytes[n_off..n_off + 4].copy_from_slice(&n_sq.to_le_bytes());
                }
            }
            Ok(())
        },
    )?;

    // Inline stable-`_id` region: one i128 per doc, indexed by `local_doc_id`
    // (the same index a hidden hit's positional doc-id resolves through), so an
    // id+score query and the drain read the stable `_id` straight from the blob
    // instead of resolving it through a scalar `_id` column. It sits between
    // codec_meta and the per-cluster blocks (see `IvfSubsectionLayout`); the
    // whole subsection is CRC'd below, so the region is covered.
    if let Some(stable_ids_off) = layout.stable_ids_off {
        for row in &rows {
            let off = stable_ids_off + (row.local_doc_id as usize) * format::vec::STABLE_ID_BYTES;
            bytes[off..off + format::vec::STABLE_ID_BYTES]
                .copy_from_slice(&row.stable_id.to_le_bytes());
        }
    }
    debug_assert_eq!(bytes.len(), total_size_before_crc);

    let crc = crc32c(&bytes);
    let mut out = bytes;
    out.extend_from_slice(&crc.to_le_bytes());

    Ok(SubsectionBytes {
        bytes: out,
        n_cent,
        summary_offset_in_sub: layout.summary_off,
        codec_meta_offset_in_sub: if codec_meta_size == 0 {
            0
        } else {
            layout.codec_meta_off
        },
        codec_meta_size,
    })
}

fn build_subsection_streaming(
    column_id: u32,
    col: ColumnState,
    scratch: &Path,
) -> Result<SubsectionBytes, BuildError> {
    let ColumnState {
        config: cfg,
        n_docs: n_docs_u32,
        reservoir,
        pre_spill_buffer,
        spill,
        spill_threshold_bytes: _,
        materialized_rows,
        prebuilt_subsection: _,
    } = col;

    if let Some(rows) = materialized_rows {
        drop(reservoir);
        return build_subsection_from_materialized(cfg, rows);
    }

    let dim = cfg.dim;
    let n_docs = n_docs_u32 as usize;
    let sample_rows = reservoir.n_rows();
    // ---- Pass 1: centroids ----
    // If a GLOBAL grid is provided, partition against it (cluster c == global
    // cell c) instead of training local k-means — the precondition for the
    // splice drain to route c → cell c doc-correctly. Keep all `n_cent` cells
    // even when this shard has fewer rows (empty clusters are count-0), so
    // ordinal c always means cell c. Otherwise train local centroids on the
    // reservoir sample. (Mirrors `build_subsection_from_materialized`.)
    let (n_cent, centroids) = if let Some(global) = cfg.provided_centroids.as_ref() {
        debug_assert!(dim > 0 && global.len() % dim == 0);
        let nc = (global.len() / dim.max(1)).max(1);
        drop(reservoir);
        (nc, global.to_vec())
    } else {
        // n_cent must be in `[1, min(n_docs, sample_rows)]`. Both bounds
        // are required: `n_cent > n_docs` makes the IVF degenerate;
        // `n_cent > sample_rows` would crash k-means (`k > n` is asserted
        // by the trainer). At steady-state shapes (`n_docs > sample_size`,
        // `sample_size ≥ 100_000`) the sample_rows bound is the active
        // one and is comfortably above any sane n_cent.
        let n_cent = cfg
            .n_cent
            .max(1)
            .min(n_cent_row_count_cap(n_docs))
            .min(n_docs.max(1))
            .min(sample_rows.max(1));
        let centroids = if sample_rows == 0 || n_docs == 0 {
            vec![0.0f32; n_cent * dim]
        } else {
            kmeans(reservoir.sample(), dim, n_cent, KMEANS_ITERS, cfg.rot_seed)
        };
        // Drop the reservoir immediately — k-means has converged
        // and the sample bytes aren't needed for pass 2.
        drop(reservoir);
        (n_cent, centroids)
    };

    // Summary centroid: mean of trained centroids. Cheap and only
    // depends on centroids, so compute now before pass 2 so we can
    // fold each row's distance into `summary_radius_sq_max` inline.
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

    let rotation = RandomRotation::new(dim, cfg.rot_seed);
    let quant = BitQuantizer::new(dim);
    let code_bytes = quant.code_bytes();

    // Pre-create all bucket file writers up-front so pass 2's hot
    // loop doesn't pay a `File::create` per row when a new cluster
    // is first hit. At `n_cent = 1024, BUCKET_BUF_SIZE = 64 KiB`
    // the writer-buffer total is 64 MiB; at `n_cent = 4096` it's
    // 256 MiB. Both match the design budget.
    let mut bucket_writers: Vec<BufWriter<File>> = Vec::with_capacity(n_cent);
    for c in 0..n_cent {
        let path = scratch.join(format!("infino_bucket_col{column_id}_c{c}.bin"));
        let file = File::create(&path)?;
        bucket_writers.push(BufWriter::with_capacity(BUCKET_BUF_SIZE, file));
    }
    let mut bucket_counts = vec![0u32; n_cent];

    // Initialise the source. Two cases:
    //
    //   - Column never crossed the spill threshold: build an
    //     InMemoryVectorSource wrapping the pre_spill_buffer
    //     (moved into Arc) — pass 2 iterates over RAM, zero I/O.
    //   - Column crossed the threshold: finish the SpillWriter to
    //     flush + fsync, then mmap the resulting file via
    //     MmapVectorSource. Pass 2 iterates over the mmap, with
    //     the kernel page cache handling streaming reads.
    let chunk_rows = chunk_rows_for_dim(dim);
    let mut summary_radius_sq_max: f32 = 0.0;
    let codec = cfg.rerank_codec;
    // `Sq8ResidualEpsilon` uses per-cluster scale/offset codec_meta plus
    // an i8 residual sidecar in `full[]`.
    let sq8_family = matches!(codec, RerankCodec::Sq8ResidualEpsilon);
    let (mut sq8_min_arr, mut sq8_max_arr): (Vec<f32>, Vec<f32>) = if sq8_family {
        (
            vec![f32::INFINITY; n_cent * dim],
            vec![f32::NEG_INFINITY; n_cent * dim],
        )
    } else {
        (Vec::new(), Vec::new())
    };
    if n_docs > 0 {
        let mut source: Box<dyn ChunkedVectorSource> = if let Some(spill) = spill {
            // Crossed the threshold during add(): close the
            // writer and open the spill file mmap-style. The
            // pre_spill_buffer is empty in this state (drained
            // when the threshold was crossed).
            debug_assert!(
                pre_spill_buffer.is_empty(),
                "spill active but pre_spill_buffer still has {} f32s",
                pre_spill_buffer.len()
            );
            let path = spill.finish()?;
            Box::new(MmapVectorSource::open(&path, dim, chunk_rows)?)
        } else {
            // Stayed in RAM: own the f32 buffer in an Arc so the
            // InMemoryVectorSource lives independent of the
            // builder's stack frame.
            Box::new(InMemoryVectorSource::new(
                Arc::new(pre_spill_buffer),
                dim,
                chunk_rows,
            ))
        };

        let sq8_acc: Option<(&mut [f32], &mut [f32])> = if sq8_family {
            Some((&mut sq8_min_arr, &mut sq8_max_arr))
        } else {
            None
        };
        run_pass2(
            source.as_mut(),
            dim,
            n_cent,
            code_bytes,
            &centroids,
            &rotation,
            &quant,
            &summary_centroid,
            &mut bucket_writers,
            &mut bucket_counts,
            &mut summary_radius_sq_max,
            codec,
            sq8_acc,
        )?;
    }

    let sq8_quantizers: Vec<(Vec<f32>, Vec<f32>)> = if sq8_family {
        (0..n_cent)
            .map(|c| {
                let off = c * dim;
                derive_sq8_quantizer_from_min_max(
                    &sq8_min_arr[off..off + dim],
                    &sq8_max_arr[off..off + dim],
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    drop(sq8_min_arr);
    drop(sq8_max_arr);

    // Flush + close every bucket writer before pass 3 reads the
    // files. The Drop of `bucket_writers` would do this, but
    // BufWriter's Drop swallows flush errors — explicit flush()
    // surfaces them as BuildError::Io.
    let mut bucket_files: Vec<File> = Vec::with_capacity(n_cent);
    for w in bucket_writers {
        let mut inner = w.into_inner().map_err(|e| BuildError::Io(e.into_error()))?;
        inner.flush()?;
        bucket_files.push(inner);
    }
    drop(bucket_files);

    let summary_radius_x100 = (summary_radius_sq_max.sqrt() * SUMMARY_RADIUS_SCALE)
        .max(0.0)
        .min(u32::MAX as f32) as u32;

    // ---- Pass 3: stream buckets into the final subsection bytes ----
    //
    // layout with main's streaming assembly: allocate the
    // subsection up front, write the open-time region, then per-
    // cluster bulk-read each bucket into `[codes_chunk | doc_ids_chunk
    // | full_chunk]` without a `full_layout` staging buffer.
    // Centroid storage order only affects pass-3 block packing
    // (sequential on-disk layout). `cluster_idx` and the centroid
    // table stay indexed by centroid id so the reader can address
    // slot `c` at `cluster_idx[c*8]` / `centroids[c*dim*4]`.
    let cluster_order = centroid_storage_order(&centroids, n_cent, dim);

    // 6. Build the subsection bytes.
    //    subsection layout
    //    (see `format::vec::SUBSECTION_VERSION` for the spec):
    //
    //      [sub_header]
    //      [summary_centroid][centroids][cluster_idx][codec_meta]   ← open-time region
    //      [per-cluster blocks: each = codes_chunk + doc_ids_chunk + full_chunk]
    //      [crc]
    //
    //    Two wins fold into this single layout:
    //      (a) open-time region contiguous at the subsection head
    //          so one range fetch covers everything search needs
    //          before picking a cluster (~1.5 MB at 1M × 384 sq8,
    //          16 MB at 10M × 1024 sq8).
    //      (b) per-cluster `codes + doc_ids + full` interleave so
    //          each probed cluster GET pulls all search-time bytes
    //          in one range. `codes_chunk` is the 1-bit RaBitQ
    //          estimate-code bytes; `full_chunk` is the optional
    //          Fp32/Sq8 rerank payload for the same docs.
    //
    //    Only this layout version is accepted on read; any other
    //    value at the version slot is rejected as malformed.
    //
    //    Codec-specific shape:
    //      Fp32: empty codec_meta; full_chunk stores the fp32
    //            vectors byte-for-byte inside each cluster block.
    //      Sq8:  codec_meta = `scale[n_cent × dim] +
    //            offset[n_cent × dim] + (per-doc norms[n_docs]
    //            for L2Sq)`. full_chunk stores dim u8 codes per
    //            doc, encoded against that doc's cluster quantizer.
    //            ~4× smaller than Fp32; recall stays > 0.99 at
    //            default rerank_mult.
    //      None: empty codec_meta; empty full_chunk. Subsection
    //            collapses to summary + centroids + cluster_idx
    //            + per-cluster blocks — the 1-bit shortlist's
    //            top-K is the final answer.
    let codec_meta_size = codec.codec_meta_bytes(dim, n_docs, n_cent, cfg.metric);
    let per_vec_bytes = codec.per_vector_bytes(dim);
    // v2 layout: each per-cluster block carries `codes_chunk +
    // doc_ids_chunk + full_chunk` for that cluster's docs, so one
    // range GET per probed cluster pulls the 1-bit estimate codes,
    // the doc-ids, AND the full-precision rerank vectors together.
    // There is no separate trailing `full[]` region — the rerank
    // bytes a query needs ride along with the cluster block it
    // already fetches, dropping cold first-search from
    // `nprobe + 1 fat-range` GETs (which over-fetched the whole
    // rerank region) to `nprobe` GETs of ~cluster-sized blocks.
    let cluster_stride = code_bytes + format::vec::DOC_ID_BYTES + per_vec_bytes;
    // Streaming (user-superfile) build: no inline stable-`_id` region.
    let layout =
        IvfSubsectionLayout::compute(dim, n_cent, n_docs, cluster_stride, codec_meta_size, 0);
    let total_size_before_crc = layout.total_size_before_crc;

    let mut bytes = alloc_ivf_subsection_with_header(
        &layout,
        codec_meta_size,
        summary_radius_x100,
        &summary_centroid,
        &centroids,
    );

    let sq8_scale_block_off = layout.codec_meta_off;
    let sq8_offset_block_off = sq8_scale_block_off + n_cent * dim * 4;
    let sq8_norms_block_off = if sq8_family && matches!(cfg.metric, Metric::L2Sq | Metric::Cosine) {
        Some(sq8_offset_block_off + n_cent * dim * 4)
    } else {
        None
    };

    if sq8_family {
        for (cid, (scale_c, offset_c)) in sq8_quantizers.iter().enumerate().take(n_cent) {
            let sc_off = sq8_scale_block_off + cid * dim * 4;
            bytes[sc_off..sc_off + dim * 4].copy_from_slice(bytemuck::cast_slice(scale_c));
            let oc_off = sq8_offset_block_off + cid * dim * 4;
            bytes[oc_off..oc_off + dim * 4].copy_from_slice(bytemuck::cast_slice(offset_c));
        }
    }

    let full_row_bytes_in_bucket = if codec.writes_full() { dim * 4 } else { 0 };
    // Buffers reused across clusters (cleared/resized per cluster inside the
    // writer) so the per-cluster file reads don't reallocate each iteration.
    let mut id_block: Vec<u8> = Vec::new();
    let mut code_block: Vec<u8> = Vec::new();
    let mut full_block: Vec<u8> = Vec::new();

    write_ivf_cluster_blocks(
        &mut bytes,
        &layout,
        &cluster_order,
        &bucket_counts,
        code_bytes,
        per_vec_bytes,
        |bytes, centroid_id, blk| {
            let path = scratch.join(format!("infino_bucket_col{column_id}_c{centroid_id}.bin"));
            let mut reader = BufReader::with_capacity(BUCKET_BUF_SIZE, File::open(&path)?);

            id_block.resize(blk.count * format::vec::DOC_ID_BYTES, 0);
            code_block.resize(blk.count * code_bytes, 0);
            if full_row_bytes_in_bucket > 0 {
                full_block.resize(blk.count * full_row_bytes_in_bucket, 0);
            }
            for i in 0..blk.count {
                reader.read_exact(&mut id_block[i * 4..(i + 1) * 4])?;
                reader.read_exact(&mut code_block[i * code_bytes..(i + 1) * code_bytes])?;
                if full_row_bytes_in_bucket > 0 {
                    let off = i * full_row_bytes_in_bucket;
                    reader.read_exact(&mut full_block[off..off + full_row_bytes_in_bucket])?;
                }
            }

            bytes[blk.codes_base..blk.codes_base + blk.count * code_bytes]
                .copy_from_slice(&code_block);
            bytes[blk.ids_base..blk.ids_base + blk.count * format::vec::DOC_ID_BYTES]
                .copy_from_slice(&id_block);

            match codec {
                RerankCodec::RabitqOnly => {}
                RerankCodec::Fp32 => {
                    bytes[blk.rerank_base..blk.rerank_base + blk.count * dim * 4]
                        .copy_from_slice(&full_block);
                }
                RerankCodec::Sq8ResidualEpsilon => {
                    let cluster_rows: &[f32] = bytemuck::cast_slice(&full_block);
                    let (scale_c, offset_c) = &sq8_quantizers[centroid_id];
                    let ec = Sq8EncodeConsts::from_scale_offset(scale_c, offset_c);
                    encode_sq8_residual_cluster_simd(
                        cluster_rows,
                        dim,
                        blk.count,
                        blk.first_row,
                        blk.rerank_base,
                        sq8_norms_block_off,
                        &ec.inv_scale,
                        &ec.c2,
                        scale_c,
                        offset_c,
                        bytes,
                    );
                }
            }
            Ok(())
        },
    )?;
    debug_assert_eq!(bytes.len(), total_size_before_crc);

    let crc = crc32c(&bytes);
    let mut out = bytes;
    out.extend_from_slice(&crc.to_le_bytes());

    Ok(SubsectionBytes {
        bytes: out,
        n_cent,
        summary_offset_in_sub: layout.summary_off,
        codec_meta_offset_in_sub: if codec_meta_size == 0 {
            0
        } else {
            layout.codec_meta_off
        },
        codec_meta_size,
    })
}

/// `Sq8ResidualEpsilon` per-cluster encode. Writes a row-interleaved
/// `[code dim u8 ‖ residual dim i8]` body (`2 × dim` bytes per row)
/// at `full_chunk_base + i × 2·dim`. The Sq8 code is the same
/// `sq8_encode_row` quantization; the residual code captures the
/// quantization error at `scale_c[d] / SQ8_RESIDUAL_DIVISOR`-sized
/// signed steps. Per-doc norms are computed against the fully
/// residual-corrected vector so the search-side kernel's
/// Cosine/L2Sq normalization matches the bytes on disk.
#[allow(clippy::too_many_arguments)]
fn encode_sq8_residual_cluster_simd(
    cluster_rows: &[f32],
    dim: usize,
    cluster_count: usize,
    cluster_doc_off: usize,
    full_chunk_base: usize,
    sq8_norms_block_off: Option<usize>,
    inv_scale_c: &[f32],
    c2_c: &[f32],
    scale_c: &[f32],
    offset_c: &[f32],
    bytes: &mut [u8],
) {
    debug_assert_eq!(cluster_rows.len(), cluster_count * dim);
    let residual_divisor = SQ8_RESIDUAL_DIVISOR;
    let row_bytes = dim * 2;

    for i in 0..cluster_count {
        let src = &cluster_rows[i * dim..(i + 1) * dim];
        let pos = cluster_doc_off + i;
        let row_off = full_chunk_base + i * row_bytes;
        let code_off = row_off;
        let res_off = row_off + dim;
        // Sq8 code leg (identical quantization to plain Sq8).
        sq8_encode_row(src, inv_scale_c, c2_c, &mut bytes[code_off..code_off + dim]);
        // Residual leg + (optional) residual-corrected norm.
        let mut acc = 0.0f64;
        for d in 0..dim {
            let qc = bytes[code_off + d];
            let base = (qc as f32) * scale_c[d] + offset_c[d];
            let step = scale_c[d] / residual_divisor;
            let rq = if step > 0.0 {
                ((src[d] - base) / step)
                    .round()
                    .clamp(-SQ8_RESIDUAL_I8_CLAMP, SQ8_RESIDUAL_I8_CLAMP) as i8
            } else {
                0
            };
            bytes[res_off + d] = rq.to_le_bytes()[0];
            if sq8_norms_block_off.is_some() {
                let x = base + (rq as f32) * step;
                acc += (x as f64) * (x as f64);
            }
        }
        if let Some(norms_off) = sq8_norms_block_off {
            let n_off = norms_off + pos * 4;
            bytes[n_off..n_off + 4].copy_from_slice(&(acc as f32).to_le_bytes());
        }
    }
}

/// Sq8 per-cluster (min, max) → (scale, offset) derivation. Shared with the
/// cell-posting encode path so both derive the quantizer identically.
#[inline]
pub(crate) fn derive_sq8_quantizer_from_min_max(min: &[f32], max: &[f32]) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(min.len(), max.len());
    let dim = min.len();
    let mut scale = vec![0.0f32; dim];
    let mut offset = vec![0.0f32; dim];
    for d in 0..dim {
        let span = max[d] - min[d];
        if span > 0.0 && span.is_finite() {
            offset[d] = min[d];
            scale[d] = span / SQ8_CODE_MAX;
        } else {
            offset[d] = if min[d].is_finite() { min[d] } else { 0.0 };
            scale[d] = 1.0;
        }
    }
    (scale, offset)
}

/// Physical storage order for centroids: a recursive widest-span median split
/// that clusters spatially-near centroids together (better page locality for
/// nprobe scans). The canonical ordering — shared with the byte-splice merge
/// path (`ivf_merge`) so a merged subsection lays clusters out the same way a
/// freshly-built one does.
pub(crate) fn centroid_storage_order(centroids: &[f32], n_cent: usize, dim: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n_cent).collect();
    order_centroids_recursive(&mut order, centroids, dim);
    order
}

fn order_centroids_recursive(order: &mut [usize], centroids: &[f32], dim: usize) {
    if order.len() <= 1 || dim == 0 {
        return;
    }

    let mut best_dim = 0usize;
    let mut best_span = 0.0f32;
    for d in 0..dim {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &c in order.iter() {
            let v = centroids[c * dim + d];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let span = hi - lo;
        if span > best_span {
            best_span = span;
            best_dim = d;
        }
    }

    order.sort_unstable_by(|&a, &b| {
        centroids[a * dim + best_dim]
            .partial_cmp(&centroids[b * dim + best_dim])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mid = order.len() / 2;
    let (left, right) = order.split_at_mut(mid);
    order_centroids_recursive(left, centroids, dim);
    order_centroids_recursive(right, centroids, dim);
}

/// Byte offsets (relative to subsection start) of an IVF subsection's regions.
/// Shared by every IVF subsection writer — the streaming fp32 build, the Sq8
/// materialized rebuild, and the byte-splice merge — so the layout math lives
/// in exactly one place.
pub(crate) struct IvfSubsectionLayout {
    pub summary_off: usize,
    pub centroids_off: usize,
    pub cluster_idx_off: usize,
    /// Start of the codec-meta region; for the Sq8 family this is also the
    /// per-cluster Sq8 `scale` block offset.
    pub codec_meta_off: usize,
    pub per_cluster_blocks_off: usize,
    /// Offset of the inline stable-`_id` region (one i128 per doc, indexed by
    /// `local_doc_id`), present only for the materialized (hidden-cell) build.
    /// `None` when no region was requested. The region sits *between* the
    /// codec-meta region and the per-cluster blocks, so the per-cluster blocks
    /// stay the last data region before the CRC — the reader still derives
    /// `n_docs` from that trailing region's length, and infers this region's
    /// presence/size from the offset gap (no header flag).
    pub stable_ids_off: Option<usize>,
    pub total_size_before_crc: usize,
}

impl IvfSubsectionLayout {
    /// Compute the region offsets. `per_cluster_stride` is
    /// `code_bytes + DOC_ID_BYTES + per_vec_bytes`; `codec_meta_size` is the
    /// codec's metadata region size (0 when it has none). `stable_ids_region_bytes`
    /// is `n_docs * STABLE_ID_BYTES` for the materialized (hidden-cell) build that
    /// inlines the stable `_id`, and 0 for the streaming/merge builds.
    pub(crate) fn compute(
        dim: usize,
        n_cent: usize,
        n_docs: usize,
        per_cluster_stride: usize,
        codec_meta_size: usize,
        stable_ids_region_bytes: usize,
    ) -> Self {
        let summary_off = SUB_HEADER_SIZE;
        let centroids_off = summary_off + dim * 4;
        let cluster_idx_off = centroids_off + n_cent * dim * 4;
        let codec_meta_off = cluster_idx_off + n_cent * CLUSTER_IDX_ENTRY_BYTES;
        // The stable-`_id` region (if any) goes between codec_meta and the
        // per-cluster blocks, so the blocks remain the trailing data region.
        let codec_meta_end = codec_meta_off + codec_meta_size;
        let stable_ids_off = (stable_ids_region_bytes > 0).then_some(codec_meta_end);
        let per_cluster_blocks_off = codec_meta_end + stable_ids_region_bytes;
        let total_size_before_crc = per_cluster_blocks_off + n_docs * per_cluster_stride;
        Self {
            summary_off,
            centroids_off,
            cluster_idx_off,
            codec_meta_off,
            per_cluster_blocks_off,
            stable_ids_off,
            total_size_before_crc,
        }
    }
}

/// Allocate the subsection buffer (sized to `total_size_before_crc`, CRC not yet
/// appended) and write the fixed prefix every IVF subsection shares: the 56-byte
/// sub-header, the summary centroid, and the per-cluster centroids. The caller
/// fills the cluster index, codec-meta/per-cluster blocks, then appends the CRC.
pub(crate) fn alloc_ivf_subsection_with_header(
    layout: &IvfSubsectionLayout,
    codec_meta_size: usize,
    summary_radius_x100: u32,
    summary_centroid: &[f32],
    centroids: &[f32],
) -> Vec<u8> {
    let mut bytes = vec![0u8; layout.total_size_before_crc];
    bytes[0..MAGIC_BYTES].copy_from_slice(format::vec::SUB_MAGIC);
    bytes[sub_hdr::VERSION_OFF..sub_hdr::VERSION_OFF + U32_BYTES]
        .copy_from_slice(&format::vec::SUBSECTION_VERSION.to_le_bytes());
    bytes[sub_hdr::CODEC_META_SIZE_OFF..sub_hdr::CODEC_META_SIZE_OFF + U32_BYTES]
        .copy_from_slice(&(codec_meta_size as u32).to_le_bytes());
    bytes[sub_hdr::SUMMARY_OFF_OFF..sub_hdr::SUMMARY_OFF_OFF + U64_BYTES]
        .copy_from_slice(&(layout.summary_off as u64).to_le_bytes());
    bytes[sub_hdr::SUMMARY_RADIUS_X100_OFF..sub_hdr::SUMMARY_RADIUS_X100_OFF + U32_BYTES]
        .copy_from_slice(&summary_radius_x100.to_le_bytes());
    bytes[sub_hdr::CENTROIDS_OFF_OFF..sub_hdr::CENTROIDS_OFF_OFF + U64_BYTES]
        .copy_from_slice(&(layout.centroids_off as u64).to_le_bytes());
    bytes[sub_hdr::CLUSTER_IDX_OFF_OFF..sub_hdr::CLUSTER_IDX_OFF_OFF + U64_BYTES]
        .copy_from_slice(&(layout.cluster_idx_off as u64).to_le_bytes());
    bytes[sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF..sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF + U64_BYTES]
        .copy_from_slice(&(layout.per_cluster_blocks_off as u64).to_le_bytes());
    bytes[layout.summary_off..layout.summary_off + summary_centroid.len() * 4]
        .copy_from_slice(bytemuck::cast_slice(summary_centroid));
    bytes[layout.centroids_off..layout.centroids_off + centroids.len() * 4]
        .copy_from_slice(bytemuck::cast_slice(centroids));
    bytes
}

/// One cluster's `code‖doc_id‖rerank` sub-region offsets, handed to the
/// per-cluster row writer by [`write_ivf_cluster_blocks`].
pub(crate) struct ClusterBlock {
    /// Byte offset of this cluster's 1-bit code sub-region.
    pub codes_base: usize,
    /// Byte offset of this cluster's doc-id sub-region.
    pub ids_base: usize,
    /// Byte offset of this cluster's rerank sub-region.
    pub rerank_base: usize,
    /// Global row index of this cluster's first row — equals the total row
    /// count of all earlier clusters, i.e. the cluster's `doc_off`. Used to
    /// index the trailing per-doc norms sidecar.
    pub first_row: usize,
    /// Row count in this cluster.
    pub count: usize,
}

/// Write the cluster index and drive per-cluster block production for an IVF
/// subsection — the codec-agnostic loop every IVF writer shares. Walks
/// `cluster_order`, writes each centroid's `(doc_off, count)` index slot,
/// computes the per-cluster `code‖doc_id‖rerank` sub-region offsets, and calls
/// `write_cluster` to fill that cluster's rows. The cluster-index entries,
/// block cursor, and stride math live here; only the row source (fp32 bucket
/// file / encoded rows / source IVF bytes) and rerank transcode are
/// caller-specific.
pub(crate) fn write_ivf_cluster_blocks<F>(
    bytes: &mut [u8],
    layout: &IvfSubsectionLayout,
    cluster_order: &[usize],
    cluster_counts: &[u32],
    code_bytes: usize,
    per_vec_bytes: usize,
    mut write_cluster: F,
) -> Result<(), BuildError>
where
    F: FnMut(&mut [u8], usize, &ClusterBlock) -> Result<(), BuildError>,
{
    let cluster_stride = code_bytes + format::vec::DOC_ID_BYTES + per_vec_bytes;
    let mut block_cursor = 0usize;
    let mut acc_off = 0usize;
    for &centroid_id in cluster_order {
        let cnt = cluster_counts[centroid_id] as usize;
        let idx_base = layout.cluster_idx_off + centroid_id * CLUSTER_IDX_ENTRY_BYTES;
        bytes[idx_base..idx_base + CLUSTER_IDX_COUNT_OFFSET]
            .copy_from_slice(&(acc_off as u32).to_le_bytes());
        bytes[idx_base + CLUSTER_IDX_COUNT_OFFSET..idx_base + CLUSTER_IDX_ENTRY_BYTES]
            .copy_from_slice(&(cnt as u32).to_le_bytes());
        if cnt > 0 {
            let block_base = layout.per_cluster_blocks_off + block_cursor;
            let codes_len = cnt * code_bytes;
            let ids_len = cnt * format::vec::DOC_ID_BYTES;
            let block = ClusterBlock {
                codes_base: block_base,
                ids_base: block_base + codes_len,
                rerank_base: block_base + codes_len + ids_len,
                first_row: acc_off,
                count: cnt,
            };
            write_cluster(bytes, centroid_id, &block)?;
            block_cursor += cnt * cluster_stride;
        }
        acc_off += cnt;
    }
    Ok(())
}

/// Pass 2 of `build_subsection_streaming`: walk the input
/// corpus chunk-by-chunk, assign each row to its centroid,
/// rotate + 1-bit encode it, fold its un-rotated distance into
/// the summary radius, and append the `(local_doc_id, code,
/// full_vec)` tuple to the assigned centroid's bucket writer.
///
/// Extracted as a helper so the (long) match between
/// `InMemoryVectorSource` and `MmapVectorSource` doesn't drag
/// the body of `build_subsection_streaming` along the type
/// erasure path twice.
#[allow(clippy::too_many_arguments)]
fn run_pass2(
    source: &mut dyn ChunkedVectorSource,
    dim: usize,
    n_cent: usize,
    code_bytes: usize,
    centroids: &[f32],
    rotation: &RandomRotation,
    quant: &BitQuantizer,
    summary_centroid: &[f32],
    bucket_writers: &mut [BufWriter<File>],
    bucket_counts: &mut [u32],
    summary_radius_sq_max: &mut f32,
    codec: RerankCodec,
    mut sq8_min_max: Option<(&mut [f32], &mut [f32])>,
) -> Result<(), BuildError> {
    let chunk_rows_cap = source.chunk_rows();
    // Pre-allocate per-chunk scratch reused across iterations to
    // keep pass-2 allocations off the hot path.
    let mut chunk_rotated = vec![0f32; chunk_rows_cap * dim];
    let mut chunk_assignments = vec![0u32; chunk_rows_cap];
    let mut chunk_codes = vec![0u8; chunk_rows_cap * code_bytes];
    let mut global_doc_id: u32 = 0;

    while let Some(chunk) = source.next_chunk() {
        let actual_rows = chunk.len() / dim;
        debug_assert!(actual_rows <= chunk_rows_cap);

        // Assignment runs on unrotated input rows against the
        // unrotated centroids — same convention as the legacy
        // build_subsection. RaBitQ's random rotation is only
        // applied for encoding, not for clustering.
        let asgn = &mut chunk_assignments[..actual_rows];
        assign_to_centroids(&chunk[..actual_rows * dim], centroids, dim, n_cent, asgn);

        // Rotate in parallel — each row's rotation is independent
        // and rayon's per-row chunk size is dim*4 bytes, well
        // above the per-task overhead break-even.
        chunk_rotated[..actual_rows * dim]
            .par_chunks_mut(dim)
            .zip(chunk[..actual_rows * dim].par_chunks(dim))
            .for_each(|(dst, src)| rotation.apply(src, dst));

        // Encode each rotated row to its 1-bit code, also in
        // parallel — encode is byte-wise and SIMD-friendly so
        // the per-row work is cheap, but at 1M+ rows even
        // saving 50 ns per row from rayon adds up.
        chunk_codes[..actual_rows * code_bytes]
            .par_chunks_mut(code_bytes)
            .enumerate()
            .for_each(|(r, code_out)| {
                let rot_row = &chunk_rotated[r * dim..(r + 1) * dim];
                quant.encode_rotated_into(rot_row, code_out);
            });

        // Summary radius: max over rows of L2² distance to
        // summary_centroid (un-rotated input space). Parallel
        // reduce — final sqrt is applied once in the caller.
        let chunk_max = (0..actual_rows)
            .into_par_iter()
            .map(|r| {
                let v = &chunk[r * dim..(r + 1) * dim];
                l2_sq(v, summary_centroid)
            })
            .reduce(|| 0.0f32, f32::max);
        if chunk_max > *summary_radius_sq_max {
            *summary_radius_sq_max = chunk_max;
        }

        // Route rows to bucket writers. Sequential per-bucket
        // — BufWriter is !Sync and a per-bucket Mutex would
        // serialize anyway. The sequential write is dominated
        // by the kernel-buffered write path (BufWriter
        // amortises to ~one syscall per 64 KiB / 1 588 B ≈ 41
        // rows at dim=384), not by the in-process loop body.
        //
        // for `RerankCodec::RabitqOnly` we skip the per-row
        // fp32 vector write entirely — pass 3 doesn't materialise
        // `full_layout` for that codec, and the on-disk superfile
        // has no `full[]` region, so spilling the vectors to a
        // bucket file would be pure wasted I/O. At dim=384 this
        // drops the per-row bucket write from 1 588 B to 52 B
        // (4 doc_id + 48 code), a ~30× pass-2 I/O reduction.
        let write_full = codec.writes_full();
        let mut sq8_acc = sq8_min_max.as_mut();
        for r in 0..actual_rows {
            let cid = asgn[r] as usize;
            let local_doc_id = global_doc_id + r as u32;
            let writer = &mut bucket_writers[cid];
            writer.write_all(&local_doc_id.to_le_bytes())?;
            writer.write_all(&chunk_codes[r * code_bytes..(r + 1) * code_bytes])?;
            if write_full {
                writer.write_all(bytemuck::cast_slice(&chunk[r * dim..(r + 1) * dim]))?;
            }
            if let Some((mn, mx)) = sq8_acc.as_deref_mut() {
                let row = &chunk[r * dim..(r + 1) * dim];
                let off = cid * dim;
                update_min_max(row, &mut mn[off..off + dim], &mut mx[off..off + dim]);
            }
            bucket_counts[cid] += 1;
        }
        global_doc_id += actual_rows as u32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs::{read, write};

    use super::*;

    /// Drive an async reader call to completion. The materialized read-back is
    /// async (the drain fetches-on-miss); these tests use in-memory readers, so
    /// every fetch resolves without yielding and a current-thread runtime is
    /// enough.
    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build current-thread runtime")
            .block_on(f)
    }

    fn cfg(name: &str, dim: usize) -> VectorConfig {
        VectorConfig {
            column: name.to_string(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
            provided_centroids: None,
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
        let blob = b.finish().expect("finish");
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
        let blob = b.finish().expect("finish");
        assert_eq!(&blob[0..8], format::vec::OUTER_MAGIC);
        // n_docs == 0
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&blob[16..24]);
        assert_eq!(u64::from_le_bytes(buf), 0);
    }

    /// The materialized (hidden-cell) build inlines the stable `_id` as a
    /// trailing i128-per-doc region; a streaming build emits none. Round-trip
    /// both through `materialized_index_rows`: the materialized rebuild must
    /// carry each row's stable `_id` straight from the blob (no scalar column),
    /// while the streaming blob reports `0` (region absent).
    #[test]
    fn materialized_build_round_trips_inline_stable_ids() {
        use bytes::Bytes;

        use crate::superfile::vector::reader::VectorReader;

        let dim = 16;
        let n = 24usize;
        let json = format!(
            r#"[{{"column":"v","dim":{dim},"n_cent":4,"rot_seed":7,"metric":"cosine"}}]"#
        );
        let cfg = || VectorConfig {
            column: "v".into(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8ResidualEpsilon,
            provided_centroids: None,
        };

        // Streaming build: distinct vectors at local_doc_ids 0..n.
        let mut b = VectorBuilder::new();
        b.register_column(cfg()).expect("register");
        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0 + (i as f32);
            v[(i * 7) % dim] += 0.5;
            // arg 0 is the column id; local_doc_ids auto-assign as 0..n.
            b.add(0, &v).expect("add");
        }
        let stream_blob = b.finish().expect("finish streaming");
        let stream_reader =
            VectorReader::open(Bytes::from(stream_blob), &json).expect("open streaming");
        let mut stream_rows = block_on(stream_reader.materialized_index_rows_async("v"))
            .expect("streaming rows");
        // Streaming subsection has no inline region.
        assert!(
            stream_rows.iter().all(|r| r.stable_id == 0),
            "streaming build must not carry inline stable_ids"
        );

        // Assign a distinct, nonzero stable `_id` per row (keyed by local id),
        // then rebuild through the materialized path so the region is written.
        let want = |local: u32| -> i128 { 1_700_000_000_000i128 + local as i128 };
        for r in &mut stream_rows {
            r.stable_id = want(r.local_doc_id);
            r.encoded.stable_id = r.stable_id;
        }

        let mut mb = VectorBuilder::new();
        mb.register_column(cfg()).expect("register mat");
        mb.load_materialized_rows(0, stream_rows)
            .expect("load materialized");
        let mat_blob = mb.finish().expect("finish materialized");
        let mat_reader =
            VectorReader::open(Bytes::from(mat_blob), &json).expect("open materialized");
        assert_eq!(mat_reader.n_docs(), n as u64);

        let mat_rows = block_on(mat_reader.materialized_index_rows_async("v"))
            .expect("materialized rows");
        assert_eq!(mat_rows.len(), n);
        for r in &mat_rows {
            assert_eq!(
                r.stable_id,
                want(r.local_doc_id),
                "inline stable_id must round-trip for local {}",
                r.local_doc_id
            );
            assert_eq!(
                r.encoded.stable_id, r.stable_id,
                "EncodedCellRow.stable_id must match"
            );
        }
    }

    /// A byte-splice compaction merge of two materialized (hidden-cell)
    /// subsections must carry the inline stable-`_id` region forward, rewritten
    /// in merged local-id order (each input's ids shifted by its `doc_id_offset`).
    /// Without carry-through a compacted cell would silently lose its inline ids.
    #[test]
    fn sq8_merge_carries_inline_stable_ids_through_compaction() {
        use bytes::Bytes;

        use crate::superfile::vector::{
            ivf_merge::merge_sq8_ivf_subsections, reader::VectorReader,
        };

        let dim = 16;
        let json = format!(
            r#"[{{"column":"v","dim":{dim},"n_cent":4,"rot_seed":7,"metric":"cosine"}}]"#
        );
        let cfg = || VectorConfig {
            column: "v".into(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8ResidualEpsilon,
            provided_centroids: None,
        };

        // Build one materialized cell blob of `n` rows whose stable `_id`s are
        // `id_base + local`, by streaming then rebuilding through the
        // materialized path (which writes the inline region).
        let build_cell = |n: usize, id_base: i128| -> Bytes {
            let mut b = VectorBuilder::new();
            b.register_column(cfg()).expect("register");
            for i in 0..n {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0 + (i as f32);
                v[(i * 5) % dim] += 0.25;
                b.add(0, &v).expect("add");
            }
            let stream = b.finish().expect("finish streaming");
            let r = VectorReader::open(Bytes::from(stream), &json).expect("open streaming");
            let mut rows = block_on(r.materialized_index_rows_async("v")).expect("rows");
            for row in &mut rows {
                row.stable_id = id_base + row.local_doc_id as i128;
                row.encoded.stable_id = row.stable_id;
            }
            let mut mb = VectorBuilder::new();
            mb.register_column(cfg()).expect("register mat");
            mb.load_materialized_rows(0, rows).expect("load materialized");
            Bytes::from(mb.finish().expect("finish materialized"))
        };

        let (na, nb) = (10usize, 8usize);
        let blob_a = build_cell(na, 5_000);
        let blob_b = build_cell(nb, 9_000);
        let reader_a = VectorReader::open(blob_a, &json).expect("open A");
        let reader_b = VectorReader::open(blob_b, &json).expect("open B");

        // Merge: B's local ids shift by `na` (its doc_id_offset).
        let merged = merge_sq8_ivf_subsections(&[
            (&reader_a, "v", 0),
            (&reader_b, "v", na as u32),
        ])
        .expect("merge");
        assert_eq!(merged.n_docs as usize, na + nb);

        let mut wb = VectorBuilder::new();
        wb.register_column(cfg()).expect("register merged");
        wb.set_prebuilt_subsection(0, merged)
            .expect("set prebuilt");
        let merged_blob = wb.finish().expect("finish merged");
        let reader_m = VectorReader::open(Bytes::from(merged_blob), &json).expect("open merged");

        let rows = block_on(reader_m.materialized_index_rows_async("v")).expect("merged rows");
        assert_eq!(rows.len(), na + nb);
        for r in &rows {
            // A occupies merged locals 0..na (id 5000+local); B occupies
            // na..na+nb (id 9000+(local-na)).
            let want = if (r.local_doc_id as usize) < na {
                5_000 + r.local_doc_id as i128
            } else {
                9_000 + (r.local_doc_id as i128 - na as i128)
            };
            assert_eq!(
                r.stable_id, want,
                "merged inline stable_id wrong for local {}",
                r.local_doc_id
            );
        }
    }

    #[test]
    fn sq8_tiny_shard_writes_physical_n_cent_to_directory() {
        use bytes::Bytes;

        use crate::superfile::vector::reader::VectorReader;

        let dim = 16;
        let configured_n_cent = 4;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent: configured_n_cent,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8ResidualEpsilon,
            provided_centroids: None,
        })
        .expect("register sq8 column");
        b.add(0, &[1.0; 16]).expect("add single row");

        let blob = b.finish().expect("finish tiny sq8 shard");
        let dir_off = OUTER_HEADER_SIZE;
        let physical_n_cent = u32::from_le_bytes(
            blob[dir_off + 8..dir_off + 12]
                .try_into()
                .expect("n_cent bytes"),
        );
        assert_eq!(
            physical_n_cent, 1,
            "directory must describe physical IVF layout, not configured n_cent"
        );

        let json = format!(
            r#"[{{"column":"v","dim":{dim},"n_cent":{configured_n_cent},"rot_seed":7,"metric":"cosine"}}]"#
        );
        let reader = VectorReader::open(Bytes::from(blob), &json).expect("open tiny sq8 shard");
        assert_eq!(reader.n_docs(), 1);
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
        let blob = b.finish().expect("finish");
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

    /// Force the spill path with `set_spill_threshold_bytes(0)`
    /// so every column transitions to the on-disk SpillWriter on
    /// the first `add()`. Then build, open, and assert the
    /// resulting blob round-trips correctly. This is the only
    /// unit-test-level coverage of the
    /// SpillWriter → MmapVectorSource pass-2 path; default-
    /// threshold builds at unit-test corpora (≤ 100 docs) never
    /// trigger the spill branch.
    #[test]
    fn build_via_forced_spill_path_round_trips() {
        let dim = 16;
        let n_docs = 64usize;
        let n_cent = 4usize;
        let mut b = VectorBuilder::new();
        b.set_spill_threshold_bytes(0);
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
            provided_centroids: None,
        })
        .expect("register column");
        // Generate a small but distinguishable corpus where each
        // doc has a unique signature in its first element.
        let mut corpus = Vec::with_capacity(n_docs * dim);
        for d in 0..n_docs {
            let mut row = vec![0.0f32; dim];
            row[0] = d as f32;
            row[1] = (d as f32) * 0.5;
            row[2] = -(d as f32);
            corpus.extend_from_slice(&row);
            b.add(0, &row).expect("add via forced-spill path");
        }
        let blob = b.finish().expect("finish via forced-spill path");
        // Header magic must still be intact.
        assert_eq!(&blob[0..8], format::vec::OUTER_MAGIC);
        let n_cols = u32::from_le_bytes([blob[12], blob[13], blob[14], blob[15]]);
        assert_eq!(n_cols, 1);
        let n_docs_hdr = u64::from_le_bytes(blob[16..24].try_into().expect("8 bytes"));
        assert_eq!(n_docs_hdr, n_docs as u64);
    }

    /// Same shape as the test above but contrasts the two paths
    /// directly: with the default threshold the build runs
    /// entirely in RAM; with threshold=0 it goes through the
    /// spill file. Both must produce blobs that decode to a
    /// reader returning the same self-NN top-1 result for every
    /// query (the recall-floor invariant — bit-for-bit equality
    /// isn't required because bucket-flush ordering is
    /// implementation-defined, but the retrieval contract holds).
    #[tokio::test]
    async fn forced_spill_path_matches_in_ram_path_on_self_nn() {
        use bytes::Bytes;

        use crate::superfile::vector::reader::VectorReader;
        let dim = 16;
        let n_docs = 50;
        let n_cent = 4;
        let mut corpus = Vec::with_capacity(n_docs * dim);
        for d in 0..n_docs {
            let mut row = vec![0.0f32; dim];
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = ((d as f32) * 0.07 + (j as f32) * 0.13).sin();
            }
            corpus.extend_from_slice(&row);
        }
        let build = |force_spill: bool| -> Vec<u8> {
            let mut b = VectorBuilder::new();
            if force_spill {
                b.set_spill_threshold_bytes(0);
            }
            b.register_column(VectorConfig {
                column: "v".into(),
                dim,
                n_cent,
                rot_seed: 7,
                metric: Metric::L2Sq,
                rerank_codec: RerankCodec::Fp32,
                provided_centroids: None,
            })
            .expect("register column");
            for d in 0..n_docs {
                b.add(0, &corpus[d * dim..(d + 1) * dim])
                    .expect("add to vector builder");
            }
            b.finish().expect("finish")
        };

        let blob_ram = build(false);
        let blob_spill = build(true);
        let json = format!(
            r#"[{{"column":"v","dim":{dim},"n_cent":{n_cent},"rot_seed":7,"metric":"l2sq"}}]"#
        );
        let r_ram = VectorReader::open(Bytes::from(blob_ram), &json).expect("open ram");
        let r_spill = VectorReader::open(Bytes::from(blob_spill), &json).expect("open spill");

        // Maximal-coverage retrieval: full IVF sweep and a rerank
        // pool wide enough to cover every doc. With these knobs
        // the rerank dominates and self (with L2Sq distance 0)
        // must be top-1 — independent of the 1-bit code's
        // ranking noise.
        let nprobe = n_cent;
        let rerank_mult = n_docs + 1;
        for q in 0..n_docs {
            let query = &corpus[q * dim..(q + 1) * dim];
            let top_ram = r_ram
                .search("v", query, 1, nprobe, rerank_mult)
                .await
                .expect("search ram");
            let top_spill = r_spill
                .search("v", query, 1, nprobe, rerank_mult)
                .await
                .expect("search spill");
            // Both paths must return self as top-1 — that's the
            // strict recall invariant, independent of the
            // implementation-defined bucket-flush ordering.
            assert_eq!(
                top_ram[0].0 as usize, q,
                "in-RAM path missed self-NN at q={q}"
            );
            assert_eq!(
                top_spill[0].0 as usize, q,
                "spill path missed self-NN at q={q}"
            );
        }
    }

    /// `finish_to(Vec<u8>)` must produce byte-for-byte identical
    /// output to `finish()` for the same logical builder state.
    /// The build path is deterministic in everything that matters
    /// (rot_seed, reservoir seed, bucket flush ordering), so any
    /// drift here would indicate a regression in either the
    /// streaming wrap or the underlying determinism contract.
    #[test]
    fn finish_to_matches_finish_byte_for_byte() {
        let build = || -> VectorBuilder {
            let mut b = VectorBuilder::new();
            b.register_column(cfg("v", 16)).expect("register column");
            for i in 0..32 {
                let v: Vec<f32> = (0..16).map(|j| ((i + j) as f32) * 0.1).collect();
                b.add(0, &v).expect("add to vector builder");
            }
            b
        };

        let blob_finish = build().finish().expect("finish");
        let mut blob_finish_to: Vec<u8> = Vec::new();
        build()
            .finish_to(&mut blob_finish_to)
            .expect("finish_to Vec<u8>");
        assert_eq!(
            blob_finish, blob_finish_to,
            "finish_to must produce identical bytes to finish"
        );
    }

    /// Streaming output to a `Cursor<Vec<u8>>` (the canonical
    /// in-tree writer for testing streaming behaviour): the
    /// resulting bytes
    /// carry a valid outer magic + a valid trailing whole-blob
    /// CRC32C that round-trips when recomputed over the body.
    #[test]
    fn finish_to_cursor_round_trips_outer_crc() {
        use std::io::Cursor;
        let mut b = VectorBuilder::new();
        b.register_column(cfg("v", 16)).expect("register column");
        for i in 0..32 {
            let v: Vec<f32> = (0..16).map(|j| ((i + j) as f32) * 0.1).collect();
            b.add(0, &v).expect("add to vector builder");
        }
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            b.finish_to(cursor).expect("finish_to Cursor");
        }
        assert_eq!(
            &buf[0..8],
            format::vec::OUTER_MAGIC,
            "outer magic preserved"
        );
        assert!(
            buf.len() >= OUTER_HEADER_SIZE + DIR_ENTRY_SIZE + 4 + 4,
            "blob too short: {} bytes",
            buf.len()
        );
        let body_len = buf.len() - 4;
        let trailing_crc = u32::from_le_bytes([
            buf[body_len],
            buf[body_len + 1],
            buf[body_len + 2],
            buf[body_len + 3],
        ]);
        let recomputed = crc32c(&buf[..body_len]);
        assert_eq!(
            trailing_crc, recomputed,
            "trailing outer CRC32C must match recomputed body CRC"
        );
    }

    /// Round-trip integrity through an actual `Write` impl that
    /// isn't `Vec<u8>`: write to a temp file, mmap-read it back,
    /// open it with `VectorReader`, and confirm a search returns
    /// a sane result. This catches any case where the running
    /// CRC32C accumulator drifts between the streaming write
    /// path and a one-shot `crc32c(&blob)` over the same bytes.
    #[tokio::test]
    async fn finish_to_temp_file_round_trips_through_reader() {
        use std::io::BufWriter;

        use bytes::Bytes;

        use crate::superfile::vector::reader::VectorReader;
        let dim = 16usize;
        let n_docs = 32usize;
        let n_cent = 4usize;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
            provided_centroids: None,
        })
        .expect("register column");
        for d in 0..n_docs {
            let row: Vec<f32> = (0..dim)
                .map(|j| ((d as f32) * 0.07 + (j as f32) * 0.13).sin())
                .collect();
            b.add(0, &row).expect("add to vector builder");
        }
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("vector_blob.bin");
        {
            let file = File::create(&path).expect("create blob file");
            let writer = BufWriter::new(file);
            b.finish_to(writer).expect("finish_to BufWriter<File>");
        }
        let blob = read(&path).expect("read blob file");
        let json = format!(
            r#"[{{"column":"v","dim":{dim},"n_cent":{n_cent},"rot_seed":7,"metric":"l2sq"}}]"#
        );
        let reader = VectorReader::open(Bytes::from(blob), &json)
            .expect("open VectorReader from streamed blob");
        let query: Vec<f32> = (0..dim).map(|j| ((j as f32) * 0.13).sin()).collect();
        let hits = reader
            .search("v", &query, 5, n_cent, n_docs + 1)
            .await
            .expect("kNN search");
        assert!(!hits.is_empty(), "search returned no hits");
    }

    /// `VectorConfig::new` fills the default rerank codec, and
    /// `with_rerank_codec` overrides it without touching the other
    /// fields.
    #[test]
    fn vector_config_new_and_with_rerank_codec() {
        let dim = 16usize;
        let n_cent = 4usize;
        let rot_seed = 7u64;
        let base = VectorConfig::new("v".into(), dim, n_cent, rot_seed, Metric::Cosine);
        assert_eq!(base.column, "v");
        assert_eq!(base.dim, dim);
        assert_eq!(base.n_cent, n_cent);
        assert_eq!(base.rot_seed, rot_seed);
        assert_eq!(base.metric, Metric::Cosine);
        assert_eq!(base.rerank_codec, RerankCodec::default());

        let overridden = base.with_rerank_codec(RerankCodec::Fp32);
        assert_eq!(overridden.rerank_codec, RerankCodec::Fp32);
        assert_eq!(overridden.column, "v");
    }

    /// `VectorBuilder::default` delegates to `new`, producing an
    /// empty builder ready to register columns.
    #[test]
    fn vector_builder_default_matches_new() {
        let mut b = VectorBuilder::default();
        assert_eq!(b.register_column(cfg("a", 16)).expect("register column"), 0);
    }

    /// `set_kmeans_sample_size` succeeds for a registered column and
    /// returns the unregistered-column error otherwise.
    #[test]
    fn set_kmeans_sample_size_ok_and_unregistered() {
        const SAMPLE_SIZE: usize = 1024;
        let mut b = VectorBuilder::new();
        b.register_column(cfg("a", 16)).expect("register column");
        b.set_kmeans_sample_size(0, SAMPLE_SIZE)
            .expect("resize sample for registered column");
        let err = b
            .set_kmeans_sample_size(9, SAMPLE_SIZE)
            .expect_err("unregistered column id");
        assert!(matches!(err, BuildError::FtsColumnTypeInvalid { .. }));
    }

    /// `with_scratch` accepts an existing directory (driving
    /// `ScratchDir::in_parent`) and rejects a path that is not a
    /// directory.
    #[test]
    fn with_scratch_accepts_dir_and_rejects_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut b = VectorBuilder::with_scratch(dir.path().to_path_buf())
            .expect("scratch under existing dir");
        assert_eq!(b.register_column(cfg("a", 16)).expect("register column"), 0);

        let file_path = dir.path().join("not-a-dir");
        write(&file_path, b"x").expect("write file");
        // `VectorBuilder` is not `Debug`, so match the result rather
        // than calling `expect_err` (which would require `T: Debug`).
        match VectorBuilder::with_scratch(file_path) {
            Ok(_) => panic!("scratch path is a file, expected rejection"),
            Err(err) => assert!(matches!(err, BuildError::Io(_))),
        }
    }
}
