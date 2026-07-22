// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Vector blob builder. Multi-column unified blob with per-column
//! self-contained subsections.
//!
//! Each column's subsection is a self-contained IVF + RaBitQ index:
//! summary centroid, IVF centroids (from k-means), cluster
//! index, 1-bit codes, full-precision vectors, doc_ids — all in
//! cluster-contiguous order so the rerank loop stays in cache.
//!
//! See `docs/architecture/superfile.md` for the full byte-level spec.

use std::{
    cmp::Ordering,
    fs::{self, File, OpenOptions, metadata},
    io::{self, BufReader, BufWriter, Error as IoError, ErrorKind, Read, Seek, SeekFrom, Write},
    mem::size_of,
    path::{Path, PathBuf},
    sync::Arc,
};

use rayon::prelude::*;
use tempfile::{tempdir, tempdir_in};

use crate::{
    config,
    superfile::{
        BuildError,
        format::{
            self, FST_SEPARATOR, RESERVED_PREFIX,
            checksum::{crc32c, crc32c_append},
            vec::{
                CELL_DIR_ENTRY_SIZE, CLUSTER_IDX_COUNT_OFFSET, CLUSTER_IDX_ENTRY_BYTES,
                MAGIC_BYTES, U32_BYTES, U64_BYTES, cell_dir_entry, sub_hdr,
            },
        },
        vector::{
            cell_posting::{MaterializedIvfRow, sq8_residual_norm_sq},
            distance::{Metric, dequantize_sq8_residual_into, distance, mean_f32_cluster_major},
            ivf_merge::MergedIvfSubsection,
            kmeans::{assign_to_centroids, kmeans, kmeans_with_assignments},
            quant::BitQuantizer,
            rerank_codec::{RerankCodec, SQ8_FIXED_OFFSET, SQ8_FIXED_SCALE},
            reservoir::{Reservoir, default_kmeans_sample_size, partition_kmeans_sample_size},
            rotation::RandomRotation,
            spill::{
                ChunkedVectorSource, InMemoryVectorSource, MmapVectorSource, SpillWriter,
                SpilledCellRows,
            },
            sq8_simd::{Sq8EncodeConsts, encode_sq8_residual_row, update_min_max},
        },
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

/// Upper bound on one trained fine run's sample share, as a multiple of the
/// requested per-run target (`sample rows / requested n_cent`). The run-count
/// formula (`n_cent = ceil(rows / rows_per_run)`) only fixes the *mean* run
/// at the fine-run byte target; the actual sizes fall out of the k-means
/// assignment, and Lloyd over a consolidated drain cell (dozens of centroids
/// over a few dense blobs) can park starved seeds and leave one run holding
/// a huge share of the cell — measured 48× the mean at 100M-doc drain cells.
/// An oversized run's summary centroid sits at the blob's center of mass,
/// far from its boundary rows, so cell routing misranks and recall caps out.
/// Runs whose sample count exceeds this bound are re-split until every run
/// fits, making the byte target an invariant rather than an expectation.
const FINE_RUN_SPLIT_BOUND_FACTOR: usize = 2;

/// Maximum split rounds before accepting residual imbalance. Each round
/// re-trains only rows inside still-oversized runs, so work decays
/// geometrically; the cap bounds the pathological worst case.
const FINE_RUN_SPLIT_MAX_ROUNDS: usize = 4;

/// Seed offset for the split re-k-means, keeping its PRNG stream distinct
/// from the base k-means stream (which offsets `rot_seed` internally).
const FINE_RUN_SPLIT_SEED_OFFSET: u64 = 101;

/// Cell row count marking the consolidated-cell regime where the drain's
/// per-cell k-means degenerates and the cell pack drops the legacy
/// row-count `n_cent` cap (byte-target count goes uncapped). Oversized
/// fine-run splitting always runs — under fine-first `cells 1..1`, a
/// 2×-skewed run parks its summary centroid at the blob center and
/// caps post-drain recall (measured 0.985 vs 0.995 at 1M / 256 cells
/// with identical cell membership). The value is bracketed by
/// measurement: 10M's largest cell (65,918 rows) must keep the capped
/// `n_cent` — the armed 2-GET cold-probe gate was measured on it, and
/// uncapping alone moved the probe to 4 GETs — while 100M's median
/// cell (95,801 rows) needs the uncapped byte-target count. 80,000
/// sits strictly between the two design points. Distinct from the
/// drain-sample boost threshold (65,536 in `reservoir.rs`), which is
/// part of the measured 10M baseline and must keep covering its
/// biggest cells.
const CONSOLIDATED_CELL_ROWS_THRESHOLD: usize = 80_000;

/// Target memory budget (~128 MiB) for one pass-2 rotated chunk
/// (`chunk_rows × dim × 4` bytes); the chunk row count is derived
/// from this so resident memory stays bounded independent of `dim`.
const PASS2_CHUNK_MEM_BUDGET_BYTES: usize = 128 << 20;

/// Floor on pass-2 chunk rows, keeping chunks wide enough to stay
/// SIMD-friendly even at extreme dimensions.
const PASS2_CHUNK_ROWS_MIN: usize = 1024;

/// Ceiling on pass-2 chunk rows, capping per-chunk RAM at small dims.
const PASS2_CHUNK_ROWS_MAX: usize = 65_536;
/// Target bytes read from one fine-cluster bucket at a time while assembling
/// a streamed materialized subsection.
const MATERIALIZED_BUCKET_CHUNK_BYTES: usize = 16 << 20;
/// Approximate live bytes per dimension while assigning one materialized row:
/// two encoded bytes plus one decoded f32.
const MATERIALIZED_ASSIGN_BYTES_PER_DIM: usize = 2 + size_of::<f32>();

/// Superfile-local document thresholds for capping the physical IVF centroid
/// count. On the streaming global build (the user-table superfile path,
/// where `cfg.n_cent` is a corpus-scale config knob — e.g. 1024/4096 —
/// hitting one commit-sized superfile) the cap always applies. Cell packs
/// apply it only below `CONSOLIDATED_CELL_ROWS_THRESHOLD` — there it is
/// part of the measured 1M/10M layouts — and drop it above, where its
/// 100K-row step sits mid drain-cell range at 100M and clamping the
/// byte-target `n_cent` (92 → 64 at a 97K-row cell) fattens runs past
/// the target.
const N_CENT_LARGE_DOC_THRESHOLD: usize = 5_000_000;
/// Maximum IVF centroids for large physical vector indexes.
const N_CENT_LARGE: usize = 4096;
/// Medium-index document threshold for the IVF centroid cap.
const N_CENT_MEDIUM_DOC_THRESHOLD: usize = 100_000;
/// Maximum IVF centroids for medium physical vector indexes.
const N_CENT_MEDIUM: usize = 1024;
/// Maximum IVF centroids for small physical vector indexes.
const N_CENT_SMALL: usize = 64;

/// Maximum Sq8 code value: each component quantizes to one unsigned
/// byte, so the per-dim scale maps a cluster's value span onto
/// `[0, SQ8_CODE_MAX]`.
const SQ8_CODE_MAX: f32 = 255.0;

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
    /// Construct a config with the configured cosine default codec
    /// ([`crate::config::VectorSettings::rerank_codec`], usually
    /// [`RerankCodec::Sq8FixedResidual`]) and locally fitted residual
    /// encoding for metrics whose values are not bounded to [-1, 1].
    pub fn new(column: String, dim: usize, n_cent: usize, rot_seed: u64, metric: Metric) -> Self {
        Self {
            column,
            dim,
            n_cent,
            rot_seed,
            metric,
            rerank_codec: if metric == Metric::Cosine {
                config::global().vector.rerank_codec
            } else {
                RerankCodec::Sq8Residual
            },
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
    /// Optional stable `_id`s for the fp32 streaming cell-pack path. Normal
    /// ingest leaves this absent; commit-as-drain uses it so MultiCell user
    /// postings can resolve primaries and boundary stubs by stable id.
    inline_stable_ids: Option<Vec<i128>>,
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
        if !config.rerank_codec.supports_metric(config.metric) {
            return Err(BuildError::VectorSchemaMismatch(format!(
                "vector index {:?}: codec {} supports cosine metric only",
                config.column,
                config.rerank_codec.name()
            )));
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
            inline_stable_ids: None,
        });
        Ok(column_id)
    }

    /// Load Sq8+ε maintenance rows for one column. Reuses the normal IVF
    /// subsection writer on finish — no fp32 corpus decode. Currently a
    /// test-only helper (the live maintenance paths splice prebuilt subsections);
    /// it backs the materialized-row / inline-stable-id round-trip tests.
    #[allow(dead_code)]
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
        if !col.config.rerank_codec.is_sq8_residual_family() {
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
        if subsection.rerank_codec != col.config.rerank_codec {
            return Err(BuildError::VectorSchemaMismatch(format!(
                "prebuilt subsection codec {} does not match destination codec {}",
                subsection.rerank_codec.name(),
                col.config.rerank_codec.name()
            )));
        }
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

/// Byte source for one complete cell-IVF subsection in a v2 multi-cell blob.
///
/// Commit uses the in-memory implementation below. Drain implements this for
/// its disk-spilled subsections, so both paths share the exact same directory,
/// CRC, and byte-assembly implementation.
pub(crate) trait MultiCellSubsectionSource {
    fn cell_id(&self) -> u32;
    fn n_docs(&self) -> u32;
    fn len(&self) -> u64;
    fn rerank_codec(&self) -> RerankCodec;
    fn write_to(&self, output: &mut dyn Write) -> Result<(), BuildError>;
}

struct BorrowedMultiCellSubsection<'a> {
    cell_id: u32,
    subsection: &'a MergedIvfSubsection,
}

impl MultiCellSubsectionSource for BorrowedMultiCellSubsection<'_> {
    fn cell_id(&self) -> u32 {
        self.cell_id
    }

    fn n_docs(&self) -> u32 {
        self.subsection.n_docs
    }

    fn len(&self) -> u64 {
        self.subsection.bytes.len() as u64
    }

    fn rerank_codec(&self) -> RerankCodec {
        self.subsection.rerank_codec
    }

    fn write_to(&self, output: &mut dyn Write) -> Result<(), BuildError> {
        output
            .write_all(&self.subsection.bytes)
            .map_err(BuildError::Io)
    }
}

struct CrcWriter<'a, W> {
    output: &'a mut W,
    crc: u32,
}

impl<W: Write> Write for CrcWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.output.write(buf)?;
        self.crc = crc32c_append(self.crc, &buf[..written]);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}

/// Stream a v2 multi-cell vector blob to `output`: one logical column packing
/// many complete cell-IVF subsections behind a cell directory of
/// `(global_cell_id, subsection_off, subsection_len)`.
///
/// `cells` must be non-empty and sorted by ascending `global_cell_id` with
/// unique ids. Each subsection is a standard `INFVECC1` IVF (unchanged).
pub(crate) fn finish_multi_cell_blob_to<W, S>(cells: &[S], mut output: W) -> Result<(), BuildError>
where
    W: Write,
    S: MultiCellSubsectionSource,
{
    if cells.is_empty() {
        return Err(BuildError::VectorSchemaMismatch(
            "multi-cell vector blob requires at least one cell IVF".into(),
        ));
    }
    let packed_codec = cells[0].rerank_codec();
    if cells.iter().any(|cell| cell.rerank_codec() != packed_codec) {
        return Err(BuildError::VectorSchemaMismatch(
            "multi-cell blob cannot mix rerank codecs".into(),
        ));
    }
    for pair in cells.windows(2) {
        if pair[0].cell_id() >= pair[1].cell_id() {
            return Err(BuildError::VectorSchemaMismatch(
                "multi-cell cells must be sorted by unique ascending cell_id".into(),
            ));
        }
    }

    let n_cells = cells.len() as u32;
    let n_docs: u64 = cells.iter().map(|cell| u64::from(cell.n_docs())).sum();
    let directory_offset = OUTER_HEADER_SIZE as u64;
    let directory_size = cells.len() * CELL_DIR_ENTRY_SIZE;
    let mut subsection_start = directory_offset + directory_size as u64 + format::CRC_BYTES as u64;

    let mut directory = Vec::with_capacity(directory_size);
    for cell in cells {
        directory.extend_from_slice(&cell.cell_id().to_le_bytes());
        directory.extend_from_slice(&subsection_start.to_le_bytes());
        directory.extend_from_slice(&cell.len().to_le_bytes());
        directory.extend_from_slice(&u32::from(cell.rerank_codec().codec_id()).to_le_bytes());
        debug_assert_eq!(directory.len() % CELL_DIR_ENTRY_SIZE, 0);
        let _ = cell_dir_entry::CELL_ID_OFF;
        subsection_start += cell.len();
    }
    let dir_crc = crc32c(&directory);

    let mut outer_header = [0u8; OUTER_HEADER_SIZE];
    {
        let mut cursor = &mut outer_header[..];
        cursor
            .write_all(format::vec::OUTER_MAGIC)
            .map_err(BuildError::Io)?;
        cursor
            .write_all(&format::vec::VERSION_MULTI_CELL.to_le_bytes())
            .map_err(BuildError::Io)?;
        cursor
            .write_all(&n_cells.to_le_bytes())
            .map_err(BuildError::Io)?;
        cursor
            .write_all(&n_docs.to_le_bytes())
            .map_err(BuildError::Io)?;
        cursor
            .write_all(&directory_offset.to_le_bytes())
            .map_err(BuildError::Io)?;
        debug_assert!(cursor.is_empty());
    }

    let outer_crc = {
        let mut crc_output = CrcWriter {
            output: &mut output,
            crc: 0,
        };
        crc_output
            .write_all(&outer_header)
            .map_err(BuildError::Io)?;
        crc_output.write_all(&directory).map_err(BuildError::Io)?;
        crc_output
            .write_all(&dir_crc.to_le_bytes())
            .map_err(BuildError::Io)?;
        for cell in cells {
            cell.write_to(&mut crc_output)?;
        }
        crc_output.flush().map_err(BuildError::Io)?;
        crc_output.crc
    };
    output
        .write_all(&outer_crc.to_le_bytes())
        .map_err(BuildError::Io)?;
    output.flush().map_err(BuildError::Io)?;
    Ok(())
}

/// In-memory convenience wrapper used by normal commit builds.
pub(crate) fn finish_multi_cell_blob(
    cells: &[(u32, MergedIvfSubsection)],
) -> Result<Vec<u8>, BuildError> {
    let sources: Vec<BorrowedMultiCellSubsection<'_>> = cells
        .iter()
        .map(|(cell_id, subsection)| BorrowedMultiCellSubsection {
            cell_id: *cell_id,
            subsection,
        })
        .collect();
    let capacity = OUTER_HEADER_SIZE
        + cells.len() * CELL_DIR_ENTRY_SIZE
        + 2 * format::CRC_BYTES
        + cells
            .iter()
            .map(|(_, subsection)| subsection.bytes.len())
            .sum::<usize>();
    let mut output = Vec::with_capacity(capacity);
    finish_multi_cell_blob_to(&sources, &mut output)?;
    Ok(output)
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

/// Metadata for one materialized IVF subsection streamed directly to a file.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StreamedIvfSubsection {
    pub(crate) n_docs: u32,
    pub(crate) rerank_codec: RerankCodec,
    pub(crate) subsection_len: u64,
    /// Physical fine-centroid count written into the subsection.
    pub(crate) n_cent: usize,
    /// Byte offset of the summary centroid relative to the subsection start.
    pub(crate) summary_offset_in_sub: usize,
    /// Byte offset / length of codec_meta relative to the subsection start.
    /// Both zero when the subsection has no codec_meta.
    pub(crate) codec_meta_offset_in_sub: usize,
    pub(crate) codec_meta_size: usize,
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
///   [Summary centroid]            — dim f32s
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
///    centroid's bucket file. Per-centroid bucket files preserve
///    cluster-contiguity for pass 3 without a third corpus
///    pass.
/// 3. **Pass 3 (sequential):** read each bucket file in
///    centroid order, materialising the cluster-contiguous
///    `codes[]`, `full[]`, and `doc_ids[]` regions and the
///    cluster-index entries.
/// Build one column's subsection from Sq8+ε maintenance rows. Reuses the same
/// on-disk IVF layout and pass-3 assembly as [`build_subsection_streaming`].
/// Opt-in phase timers for the materialized (drain) IVF build, enabled by
/// `diagnostics.drain_build_timers`. The per-cell build runs in parallel (rayon),
/// so each phase accumulates into a shared atomic — the total is summed CPU
/// across cells, not wall-clock, which is what tells us whether the build is
/// compute-bound and where (train vs assign vs calibrate). The drain resets
/// before its build loop and logs the totals after. Zero overhead when off (a
/// cached env check gates the clock).
pub(crate) mod build_phase_timers {
    use std::{
        sync::{
            OnceLock,
            atomic::{AtomicU64, Ordering},
        },
        time::Instant,
    };

    pub static TRAIN_US: AtomicU64 = AtomicU64::new(0);
    pub static ASSIGN_US: AtomicU64 = AtomicU64::new(0);
    pub static CALIB_US: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| crate::config::global().diagnostics.drain_build_timers)
    }

    /// Run `f`, adding its elapsed micros to `counter` when timing is enabled.
    pub fn timed<T>(counter: &AtomicU64, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let t = Instant::now();
        let out = f();
        counter.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        out
    }

    pub fn reset() {
        TRAIN_US.store(0, Ordering::Relaxed);
        ASSIGN_US.store(0, Ordering::Relaxed);
        CALIB_US.store(0, Ordering::Relaxed);
    }

    /// (train_ms, assign_ms, calibrate_ms), summed CPU across cells.
    pub fn snapshot_ms() -> (f64, f64, f64) {
        let ms = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1000.0;
        (ms(&TRAIN_US), ms(&ASSIGN_US), ms(&CALIB_US))
    }
}

fn materialized_centroids(cfg: &VectorConfig, n_docs: usize, sample: &[f32]) -> (usize, Vec<f32>) {
    let dim = cfg.dim;
    if let Some(global) = cfg.provided_centroids.as_ref() {
        // Global-aligned build: cluster index == cell id is a routing
        // contract (the drain's assign-skip and the grid summaries key on
        // it), so provided centroids keep their order verbatim — never
        // reorder them.
        debug_assert!(dim > 0 && global.len() % dim == 0);
        let n_cent = global.len() / dim.max(1);
        return (n_cent, global.to_vec());
    }
    // `n_cent` cap switches at the consolidated-cell boundary (see
    // `CONSOLIDATED_CELL_ROWS_THRESHOLD`): consolidated cells take the
    // row-derived byte-target uncapped; everything below keeps the
    // legacy row-count cap so the 1M/10M cold-GET layout stays on the
    // measured shape. Oversized fine-run splitting always runs — the
    // cap alone does not stop Lloyd from parking 2×-skewed runs that
    // flatten fine-first p=1 recall below 0.99.
    let consolidated = n_docs > CONSOLIDATED_CELL_ROWS_THRESHOLD;
    let requested = if consolidated {
        cfg.n_cent.max(1).min(n_docs.max(1))
    } else {
        cfg.n_cent
            .max(1)
            .min(n_cent_row_count_cap(n_docs))
            .min(n_docs.max(1))
    };
    // Keep the final Lloyd assignments — `split_oversized_fine_runs`
    // consumes them for the first bound check so a balanced pack (the
    // common commit-cell case) does not pay a redundant full sample
    // assign on top of the train. `kmeans` alone would drop them and
    // force that duplicate pass (see `kmeans_with_assignments`).
    let (mut centroids, assignments) =
        kmeans_with_assignments(sample, dim, requested, KMEANS_ITERS, cfg.rot_seed);
    let n_cent = split_oversized_fine_runs(
        &mut centroids,
        sample,
        dim,
        requested,
        cfg.rot_seed,
        Some(assignments),
    );
    order_centroids_geometrically(&mut centroids, dim, n_cent);
    (n_cent, centroids)
}

/// Enforce the fine-run size bound on trained centroids: while any run's
/// sample population exceeds [`FINE_RUN_SPLIT_BOUND_FACTOR`] × the requested
/// per-run target, re-run k-means locally on that run's sample rows and
/// replace its one centroid with enough sub-centroids to land each subrun
/// back on the target. Returns the final centroid count (`centroids` grows
/// in place). Single-run packs (the commit-time cell delta shape,
/// `requested == 1`) can never exceed the bound and pass through untouched.
///
/// When `initial_assignments` matches the current `centroids` (the final
/// Lloyd labeling from [`kmeans_with_assignments`]), the first round
/// skips `assign_to_centroids` and only scans counts — balanced packs
/// return without another distance pass. Pass `None` when the caller
/// has no labeling (tests / synthetic centroid fixtures); the first
/// round then assigns normally.
///
/// Deterministic: assignment and sub-k-means are seeded from `seed` plus a
/// fixed offset mixed with the round and run index.
fn split_oversized_fine_runs(
    centroids: &mut Vec<f32>,
    sample: &[f32],
    dim: usize,
    requested: usize,
    seed: u64,
    initial_assignments: Option<Vec<u32>>,
) -> usize {
    let mut n_cent = centroids.len() / dim.max(1);
    let sample_n = sample.len() / dim.max(1);
    if n_cent <= 1 || sample_n == 0 {
        return n_cent;
    }
    let target = sample_n.div_ceil(requested.max(1)).max(1);
    let bound = target.saturating_mul(FINE_RUN_SPLIT_BOUND_FACTOR);
    // Reuse the caller's Lloyd labeling on round 0 when it covers every
    // sample row; any later round (or a missing/mismatched labeling)
    // re-assigns against the current centroids.
    let (mut assignments, mut need_assign) = match initial_assignments {
        Some(a) if a.len() == sample_n && a.iter().all(|&idx| (idx as usize) < n_cent) => {
            (a, false)
        }
        _ => (vec![0u32; sample_n], true),
    };
    for round in 0..FINE_RUN_SPLIT_MAX_ROUNDS {
        if need_assign {
            assign_to_centroids(sample, centroids, dim, n_cent, &mut assignments);
        }
        need_assign = true;
        let mut counts = vec![0usize; n_cent];
        for &a in &assignments {
            counts[a as usize] += 1;
        }
        let oversized: Vec<usize> = (0..n_cent).filter(|&c| counts[c] > bound).collect();
        if oversized.is_empty() {
            break;
        }
        for &c in &oversized {
            let members: Vec<usize> = (0..sample_n)
                .filter(|&r| assignments[r] as usize == c)
                .collect();
            // An oversized run has > bound ≥ 2 members, so k lands in
            // [2, members.len()] and the sub-k-means input is never empty.
            let k = members.len().div_ceil(target).max(2).min(members.len());
            let mut rows = Vec::with_capacity(members.len() * dim);
            for &r in &members {
                rows.extend_from_slice(&sample[r * dim..(r + 1) * dim]);
            }
            let sub_seed = seed
                .wrapping_add(FINE_RUN_SPLIT_SEED_OFFSET)
                .wrapping_add(((round as u64) << u32::BITS) | c as u64);
            let sub = kmeans(&rows, dim, k, KMEANS_ITERS, sub_seed);
            centroids[c * dim..(c + 1) * dim].copy_from_slice(&sub[..dim]);
            centroids.extend_from_slice(&sub[dim..]);
        }
        n_cent = centroids.len() / dim;
    }
    n_cent
}

/// Reorder trained fine centroids into a deterministic greedy
/// nearest-neighbor chain: start at the centroid nearest the centroid
/// mean, then repeatedly append the unvisited centroid nearest the
/// chain's tail (ties broken by lower index).
///
/// Per-cluster blocks are written in centroid order, so geometric
/// neighbors become **file neighbors**: a query's top fine runs are
/// mutual neighbors around the query point, so they land adjacent and
/// the cold fetch coalesces them into one range — independent of k-means
/// init order, row arrival order, or the upstream grid shape. Without
/// this the layout is init-order random: measured at 10M, bit-identical
/// hidden cell topology packed from 256- vs 512-cell user grids flipped
/// the post-drain first cold query between 2 and 4 GETs on layout luck
/// alone, and compaction's repack reshuffled it again (2 → 3 GETs on an
/// unchanged table).
///
/// Ordering always chains on L2² over the stored fp32 — well-defined for
/// every metric (cosine corpora arrive normalized, where L2 order is
/// angular order) — and permuting whole clusters is semantically
/// neutral: assignments, summaries, and the cluster index all derive
/// from the same order downstream.
fn order_centroids_geometrically(centroids: &mut [f32], dim: usize, n_cent: usize) {
    if n_cent <= 2 || centroids.len() != n_cent * dim {
        return;
    }
    let mean = mean_f32_cluster_major(centroids, dim, n_cent);
    let dist = |a: &[f32], c: usize| distance(Metric::L2Sq, a, &centroids[c * dim..(c + 1) * dim]);
    let mut visited = vec![false; n_cent];
    let mut order = Vec::with_capacity(n_cent);
    let mut current = (0..n_cent)
        .min_by(|&a, &b| dist(&mean, a).total_cmp(&dist(&mean, b)))
        .unwrap_or(0);
    visited[current] = true;
    order.push(current);
    while order.len() < n_cent {
        let tail = centroids[current * dim..(current + 1) * dim].to_vec();
        let next = (0..n_cent)
            .filter(|&c| !visited[c])
            .min_by(|&a, &b| dist(&tail, a).total_cmp(&dist(&tail, b)))
            .expect("unvisited centroid remains");
        visited[next] = true;
        order.push(next);
        current = next;
    }
    let mut reordered = vec![0.0f32; centroids.len()];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        reordered[new_idx * dim..(new_idx + 1) * dim]
            .copy_from_slice(&centroids[old_idx * dim..(old_idx + 1) * dim]);
    }
    centroids.copy_from_slice(&reordered);
}

/// In-RAM encoded rebuild (maintenance / compaction): thin adapter over the
/// shared cell-pack core. Rows are packed in `local_doc_id` order; callers
/// pass dense `0..n` ids (the streamed core renumbers rows positionally, so
/// sorted-dense input keeps ids stable).
fn build_subsection_from_materialized(
    cfg: VectorConfig,
    mut rows: Vec<MaterializedIvfRow>,
) -> Result<SubsectionBytes, BuildError> {
    rows.sort_by_key(|r| r.local_doc_id);
    let merged = build_cell_subsection_in_memory(cfg, CellPackSource::Rows(rows))?;
    Ok(SubsectionBytes {
        bytes: merged.bytes,
        n_cent: merged.n_cent,
        summary_offset_in_sub: merged.summary_offset_in_sub,
        codec_meta_offset_in_sub: merged.codec_meta_offset_in_sub,
        codec_meta_size: merged.codec_meta_size,
    })
}

/// Build one complete cell-IVF subsection from materialized Sq8 rows,
/// returned as a [`MergedIvfSubsection`] ready for multi-cell packing.
pub(crate) fn build_merged_subsection_from_materialized(
    cfg: VectorConfig,
    rows: Vec<MaterializedIvfRow>,
) -> Result<MergedIvfSubsection, BuildError> {
    let n_docs = rows.len() as u32;
    let rerank_codec = cfg.rerank_codec;
    let sub = build_subsection_from_materialized(cfg, rows)?;
    Ok(MergedIvfSubsection {
        bytes: sub.bytes,
        n_cent: sub.n_cent,
        n_docs,
        rerank_codec,
        summary_offset_in_sub: sub.summary_offset_in_sub,
        codec_meta_offset_in_sub: sub.codec_meta_offset_in_sub,
        codec_meta_size: sub.codec_meta_size,
    })
}

fn materialized_chunk_rows_for_dim(dim: usize) -> usize {
    let row_bytes = dim.max(1).saturating_mul(MATERIALIZED_ASSIGN_BYTES_PER_DIM);
    (PASS2_CHUNK_MEM_BUDGET_BYTES / row_bytes).clamp(PASS2_CHUNK_ROWS_MIN, PASS2_CHUNK_ROWS_MAX)
}

/// SplitMix64 finalizer — the per-bucket jitter hash for the training
/// sampler. Deterministic, seedable, and cheap; statistically well-mixed
/// output for sequential inputs.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Seed decorrelator for the training sampler's jitter stream, keeping it
/// distinct from the k-means and reservoir streams derived from the same
/// column `rot_seed`.
const SAMPLE_JITTER_SEED_XOR: u64 = 0xa5a5_5a5a_c3c3_3c3c;

/// Training-sample index: sample `s` of `sample_size` maps to a row in its
/// stride bucket `[s·n/sample, (s+1)·n/sample)` (identity when the sample
/// covers the corpus). Buckets are disjoint, so indices stay strictly
/// increasing — the streaming spill reader consumes them in one pass.
///
/// Consolidated cells (past [`CONSOLIDATED_CELL_ROWS_THRESHOLD`], where
/// the row-fraction sample makes the stride an exact integer) pick a
/// seeded pseudo-random row inside the bucket instead of its first row.
/// That jitter is load-bearing: a plain integer stride aliases with
/// periodic arrival order. Measured at 100M (n/4 sample = exact stride 4,
/// corpus cycling planted clusters round-robin): entire blobs were never
/// sampled, k-means trained no centroid near them, and each invisible
/// blob collapsed onto its nearest trained centroid at scatter — 92K-row
/// mega-runs the oversized-run split cannot see (the sink centroid looks
/// normal-sized in the sample), a flat 0.13 fine-coverage curve, and
/// post-drain recall 0.136.
///
/// At or below the threshold the plain stride ships unchanged: those
/// samples are points-per-centroid sized (non-integer stride, no aliasing
/// measured) and the 1M/10M cold-probe gates were measured on exactly
/// that layout — jittering them re-rolled chain adjacency and moved the
/// 10M post-drain probe from 2 GETs / 7.5 MiB to 4 / 17.5 (gate-caught).
/// Known sliver: cells between the sample-boost threshold (65,536) and
/// this one keep the baseline's near-integer stride, tolerated because
/// the measured-good 10M layout includes its 65,918-row cell. One
/// definition shared by every cell-pack training sampler, so the spilled
/// / in-RAM / fp32 feeders train fine centroids on the same rows.
#[inline]
fn sampled_index(s: usize, sample_size: usize, n_docs: usize, seed: u64) -> usize {
    if sample_size >= n_docs {
        return s;
    }
    let base = s * n_docs / sample_size;
    if n_docs <= CONSOLIDATED_CELL_ROWS_THRESHOLD {
        return base;
    }
    let next = ((s + 1) * n_docs / sample_size).min(n_docs);
    let width = next.saturating_sub(base).max(1) as u64;
    let jitter = splitmix64(seed ^ SAMPLE_JITTER_SEED_XOR ^ (s as u64)) % width;
    base + jitter as usize
}

/// Payload half of one bucket record: pinned Sq8+ε bytes copied verbatim
/// (fixed-quantizer codecs) or the fp32 row (fitted codecs, which also fold
/// the per-cluster min/max for the later quantizer fit).
enum BucketRecordPayload<'a> {
    FixedSq8 {
        codes: &'a [u8],
        residuals: &'a [u8],
    },
    Fp32(&'a [f32]),
}

/// Append one bucket record (`local id ‖ RaBitQ code ‖ payload`) plus its
/// stable-id entry, and bump the bucket count. The single owner of the
/// bucket record layout — every feeder (spilled, in-RAM encoded, fp32) goes
/// through here, so the streamed pack readers see one format.
#[allow(clippy::too_many_arguments)]
fn write_bucket_record(
    cluster: usize,
    local_doc_id: u32,
    stable_id: i128,
    rabitq_code: &[u8],
    payload: BucketRecordPayload<'_>,
    dim: usize,
    bucket_writers: &mut [BufWriter<File>],
    bucket_counts: &mut [u32],
    stable_ids: &mut BufWriter<File>,
    min_max: &mut Option<(&mut [f32], &mut [f32])>,
) -> Result<(), BuildError> {
    stable_ids.write_all(&stable_id.to_le_bytes())?;
    let writer = &mut bucket_writers[cluster];
    writer.write_all(&local_doc_id.to_le_bytes())?;
    writer.write_all(rabitq_code)?;
    match payload {
        BucketRecordPayload::FixedSq8 { codes, residuals } => {
            writer.write_all(codes)?;
            writer.write_all(residuals)?;
        }
        BucketRecordPayload::Fp32(fp) => {
            writer.write_all(bytemuck::cast_slice(fp))?;
            if let Some((min, max)) = min_max.as_mut() {
                let offset = cluster * dim;
                update_min_max(
                    fp,
                    &mut min[offset..offset + dim],
                    &mut max[offset..offset + dim],
                );
            }
        }
    }
    bucket_counts[cluster] = bucket_counts[cluster].saturating_add(1);
    Ok(())
}

fn sample_spilled_materialized_rows(
    spill: &SpilledCellRows,
    sample_size: usize,
    chunk_rows: usize,
    seed: u64,
) -> Result<Vec<f32>, BuildError> {
    if sample_size == 0 {
        return Ok(Vec::new());
    }
    let n_docs = spill.n_rows();
    let dim = spill.dim();
    let targets: Vec<usize> = (0..sample_size)
        .map(|sample| sampled_index(sample, sample_size, n_docs, seed))
        .collect();
    let mut sample = vec![0.0f32; sample_size * dim];
    let mut reader = spill.reader()?;
    let mut row_base = 0usize;
    let mut target_idx = 0usize;
    while row_base < n_docs {
        let rows = reader.next_chunk(chunk_rows)?;
        if rows.is_empty() {
            break;
        }
        let row_end = row_base + rows.len();
        while target_idx < targets.len() && targets[target_idx] < row_end {
            let row = &rows[targets[target_idx] - row_base];
            if row.encoded.rerank_codec != spill.rerank_codec() {
                return Err(BuildError::VectorSchemaMismatch(
                    "materialized spill mixes rerank codecs".into(),
                ));
            }
            dequantize_sq8_residual_into(
                &row.encoded.scale,
                &row.encoded.offset,
                &row.encoded.codes,
                &row.encoded.residuals,
                row.encoded
                    .rerank_codec
                    .residual_divisor()
                    .expect("materialized spill uses residual-family codec"),
                &mut sample[target_idx * dim..(target_idx + 1) * dim],
            );
            target_idx += 1;
        }
        row_base = row_end;
    }
    if target_idx != sample_size {
        return Err(BuildError::VectorSchemaMismatch(format!(
            "materialized spill yielded {target_idx} of {sample_size} training rows"
        )));
    }
    Ok(sample)
}

/// Bucket-scatter one chunk of encoded rows: decode to fp32, assign against
/// the fine centroids, and append each row's bucket record (`local id ‖
/// RaBitQ code ‖ payload`). Fixed codec copies the stored Sq8+ε bytes
/// verbatim; fitted codecs write the decoded fp32 payload and fold per-dim
/// min/max for the per-cluster quantizer fit. Shared by the spilled and
/// in-RAM encoded feeders.
#[allow(clippy::too_many_arguments)]
fn bucket_encoded_rows_chunk(
    rows: &[MaterializedIvfRow],
    base_local: u32,
    cfg: &VectorConfig,
    centroids: &[f32],
    n_cent: usize,
    bucket_writers: &mut [BufWriter<File>],
    bucket_counts: &mut [u32],
    stable_ids: &mut BufWriter<File>,
    min_max: &mut Option<(&mut [f32], &mut [f32])>,
) -> Result<(), BuildError> {
    let dim = cfg.dim;
    let code_bytes = dim.div_ceil(u8::BITS as usize);
    let fixed = cfg.rerank_codec.uses_fixed_quantizer();
    for row in rows {
        if row.encoded.rerank_codec != cfg.rerank_codec || row.rabitq_code.len() != code_bytes {
            return Err(BuildError::VectorSchemaMismatch(
                "materialized row does not match destination vector config".into(),
            ));
        }
    }
    let mut decoded = vec![0.0f32; rows.len() * dim];
    decoded
        .par_chunks_mut(dim)
        .zip(rows.par_iter())
        .for_each(|(out, row)| {
            dequantize_sq8_residual_into(
                &row.encoded.scale,
                &row.encoded.offset,
                &row.encoded.codes,
                &row.encoded.residuals,
                row.encoded
                    .rerank_codec
                    .residual_divisor()
                    .expect("materialized row uses residual-family codec"),
                out,
            );
        });
    let mut assignments = vec![0u32; rows.len()];
    assign_to_centroids(&decoded, centroids, dim, n_cent, &mut assignments);
    for (row_idx, (row, &cluster)) in rows.iter().zip(&assignments).enumerate() {
        let payload = if fixed {
            BucketRecordPayload::FixedSq8 {
                codes: &row.encoded.codes,
                residuals: &row.encoded.residuals,
            }
        } else {
            BucketRecordPayload::Fp32(&decoded[row_idx * dim..(row_idx + 1) * dim])
        };
        write_bucket_record(
            cluster as usize,
            base_local + row_idx as u32,
            row.stable_id,
            &row.rabitq_code,
            payload,
            dim,
            bucket_writers,
            bucket_counts,
            stable_ids,
            min_max,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn stream_materialized_rows_to_buckets(
    spill: &SpilledCellRows,
    cfg: &VectorConfig,
    centroids: &[f32],
    n_cent: usize,
    bucket_writers: &mut [BufWriter<File>],
    bucket_counts: &mut [u32],
    stable_ids: &mut BufWriter<File>,
    sq8_min_max: Option<(&mut [f32], &mut [f32])>,
) -> Result<(), BuildError> {
    let chunk_rows = materialized_chunk_rows_for_dim(cfg.dim);
    let mut min_max = sq8_min_max;
    let mut reader = spill.reader()?;
    let mut next_local = 0u32;
    while next_local < spill.n_rows() as u32 {
        let rows = reader.next_chunk(chunk_rows)?;
        if rows.is_empty() {
            break;
        }
        bucket_encoded_rows_chunk(
            &rows,
            next_local,
            cfg,
            centroids,
            n_cent,
            bucket_writers,
            bucket_counts,
            stable_ids,
            &mut min_max,
        )?;
        next_local += rows.len() as u32;
    }
    if next_local as usize != spill.n_rows() {
        return Err(BuildError::VectorSchemaMismatch(format!(
            "materialized spill streamed {next_local} of {} rows",
            spill.n_rows()
        )));
    }
    Ok(())
}

/// In-RAM encoded feeder: identical bucket records to the spilled feeder,
/// chunked over the row slice.
#[allow(clippy::too_many_arguments)]
fn stream_ram_rows_to_buckets(
    rows: &[MaterializedIvfRow],
    cfg: &VectorConfig,
    centroids: &[f32],
    n_cent: usize,
    bucket_writers: &mut [BufWriter<File>],
    bucket_counts: &mut [u32],
    stable_ids: &mut BufWriter<File>,
    sq8_min_max: Option<(&mut [f32], &mut [f32])>,
) -> Result<(), BuildError> {
    let chunk_rows = materialized_chunk_rows_for_dim(cfg.dim);
    let mut min_max = sq8_min_max;
    for (chunk_idx, chunk) in rows.chunks(chunk_rows).enumerate() {
        bucket_encoded_rows_chunk(
            chunk,
            (chunk_idx * chunk_rows) as u32,
            cfg,
            centroids,
            n_cent,
            bucket_writers,
            bucket_counts,
            stable_ids,
            &mut min_max,
        )?;
    }
    Ok(())
}

/// Commit-time fp32 feeder: assign on the raw fp32 rows, rotate +
/// RaBitQ-encode per chunk, and write the same bucket records as the encoded
/// feeders. Fixed codec encodes the Sq8+ε payload once on the pinned grid
/// (byte-identical to what a drain re-pack reads back); fitted codecs write
/// the fp32 payload directly — no quantization round-trip.
#[allow(clippy::too_many_arguments)]
fn stream_fp32_rows_to_buckets(
    vectors: &[f32],
    stable_ids_in: &[i128],
    cfg: &VectorConfig,
    centroids: &[f32],
    n_cent: usize,
    bucket_writers: &mut [BufWriter<File>],
    bucket_counts: &mut [u32],
    stable_ids: &mut BufWriter<File>,
    sq8_min_max: Option<(&mut [f32], &mut [f32])>,
) -> Result<(), BuildError> {
    let dim = cfg.dim;
    let n_docs = vectors.len() / dim;
    let fixed = cfg.rerank_codec.uses_fixed_quantizer();
    let mut min_max = sq8_min_max;
    let rotation = RandomRotation::new(dim, cfg.rot_seed);
    let quant = BitQuantizer::new(dim);
    let code_bytes = quant.code_bytes();
    let divisor = cfg
        .rerank_codec
        .residual_divisor()
        .expect("residual-family codec has divisor");
    let (fixed_scale, fixed_offset) = fixed_sq8_quantizer(dim);
    let encode_consts = Sq8EncodeConsts::from_scale_offset(&fixed_scale, &fixed_offset);
    let chunk_rows = materialized_chunk_rows_for_dim(dim);
    let mut chunk_rotated = vec![0.0f32; chunk_rows * dim];
    let mut chunk_codes = vec![0u8; chunk_rows * code_bytes];
    let mut chunk_payload = if fixed {
        vec![0u8; chunk_rows * dim * 2]
    } else {
        Vec::new()
    };
    let mut row_base = 0usize;
    while row_base < n_docs {
        let take = (n_docs - row_base).min(chunk_rows);
        let chunk = &vectors[row_base * dim..(row_base + take) * dim];
        let mut assignments = vec![0u32; take];
        assign_to_centroids(chunk, centroids, dim, n_cent, &mut assignments);
        chunk_rotated[..take * dim]
            .par_chunks_mut(dim)
            .zip(chunk.par_chunks(dim))
            .for_each(|(dst, src)| rotation.apply(src, dst));
        chunk_codes[..take * code_bytes]
            .par_chunks_mut(code_bytes)
            .zip(chunk_rotated[..take * dim].par_chunks(dim))
            .for_each(|(code, rot)| quant.encode_rotated_into(rot, code));
        if fixed {
            chunk_payload[..take * dim * 2]
                .par_chunks_mut(dim * 2)
                .zip(chunk.par_chunks(dim))
                .for_each_init(
                    || vec![0.0f32; dim],
                    |recon, (payload, row)| {
                        let (code_out, residual_out) = payload.split_at_mut(dim);
                        encode_sq8_residual_row(
                            row,
                            &encode_consts,
                            &fixed_scale,
                            &fixed_offset,
                            code_out,
                            residual_out,
                            recon,
                            false,
                            divisor,
                        );
                    },
                );
        }
        for i in 0..take {
            let payload = if fixed {
                let (codes, residuals) =
                    chunk_payload[i * dim * 2..(i + 1) * dim * 2].split_at(dim);
                BucketRecordPayload::FixedSq8 { codes, residuals }
            } else {
                BucketRecordPayload::Fp32(&chunk[i * dim..(i + 1) * dim])
            };
            write_bucket_record(
                assignments[i] as usize,
                (row_base + i) as u32,
                stable_ids_in[row_base + i],
                &chunk_codes[i * code_bytes..(i + 1) * code_bytes],
                payload,
                dim,
                bucket_writers,
                bucket_counts,
                stable_ids,
                &mut min_max,
            )?;
        }
        row_base += take;
    }
    Ok(())
}

fn write_at(file: &mut File, offset: usize, bytes: &[u8]) -> Result<(), BuildError> {
    file.seek(SeekFrom::Start(offset as u64))?;
    file.write_all(bytes)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn stream_bucket_into_subsection(
    output: &mut File,
    bucket_path: &Path,
    block: &ClusterBlock,
    code_bytes: usize,
    dim: usize,
    codec: RerankCodec,
    scale: &[f32],
    offset: &[f32],
    norms_offset: Option<usize>,
) -> Result<(), BuildError> {
    let fixed = codec.uses_fixed_quantizer();
    let payload_bytes = if fixed {
        dim * 2
    } else {
        dim * size_of::<f32>()
    };
    let record_bytes = format::vec::DOC_ID_BYTES + code_bytes + payload_bytes;
    let chunk_rows = (MATERIALIZED_BUCKET_CHUNK_BYTES / record_bytes.max(1)).max(1);
    let mut reader = BufReader::new(File::open(bucket_path)?);
    let mut rows_done = 0usize;
    let encode_consts = (!fixed).then(|| Sq8EncodeConsts::from_scale_offset(scale, offset));
    let mut recon = vec![0.0f32; dim];
    let mut fp_row = vec![0.0f32; dim];
    while rows_done < block.count {
        let take = (block.count - rows_done).min(chunk_rows);
        let mut records = vec![0u8; take * record_bytes];
        reader.read_exact(&mut records)?;
        let mut ids = vec![0u8; take * format::vec::DOC_ID_BYTES];
        let mut codes = vec![0u8; take * code_bytes];
        let mut rerank = vec![0u8; take * dim * 2];
        let mut norms = norms_offset.map(|_| vec![0u8; take * size_of::<f32>()]);
        for row_idx in 0..take {
            let record = &records[row_idx * record_bytes..(row_idx + 1) * record_bytes];
            let id_end = format::vec::DOC_ID_BYTES;
            let code_end = id_end + code_bytes;
            ids[row_idx * id_end..(row_idx + 1) * id_end].copy_from_slice(&record[..id_end]);
            codes[row_idx * code_bytes..(row_idx + 1) * code_bytes]
                .copy_from_slice(&record[id_end..code_end]);
            let rerank_row = &mut rerank[row_idx * dim * 2..(row_idx + 1) * dim * 2];
            let norm = if fixed {
                rerank_row.copy_from_slice(&record[code_end..code_end + dim * 2]);
                norms_offset.map(|_| {
                    sq8_residual_norm_sq(
                        scale,
                        offset,
                        &rerank_row[..dim],
                        &rerank_row[dim..],
                        codec
                            .residual_divisor()
                            .expect("fixed residual codec has divisor"),
                    )
                })
            } else {
                for (value, bytes) in fp_row
                    .iter_mut()
                    .zip(record[code_end..].chunks_exact(size_of::<f32>()))
                {
                    *value = f32::from_le_bytes(bytes.try_into().expect("4-byte f32 bucket value"));
                }
                let (code_out, residual_out) = rerank_row.split_at_mut(dim);
                encode_sq8_residual_row(
                    &fp_row,
                    encode_consts
                        .as_ref()
                        .expect("non-fixed materialized bucket has encode constants"),
                    scale,
                    offset,
                    code_out,
                    residual_out,
                    &mut recon,
                    norms_offset.is_some(),
                    codec
                        .residual_divisor()
                        .expect("residual-family codec has divisor"),
                )
            };
            if let (Some(norm), Some(norm_bytes)) = (norm, norms.as_mut()) {
                let start = row_idx * size_of::<f32>();
                norm_bytes[start..start + size_of::<f32>()].copy_from_slice(&norm.to_le_bytes());
            }
        }
        write_at(output, block.codes_base + rows_done * code_bytes, &codes)?;
        write_at(
            output,
            block.ids_base + rows_done * format::vec::DOC_ID_BYTES,
            &ids,
        )?;
        write_at(output, block.rerank_base + rows_done * dim * 2, &rerank)?;
        if let (Some(norms_base), Some(norm_bytes)) = (norms_offset, norms) {
            write_at(
                output,
                norms_base + (block.first_row + rows_done) * size_of::<f32>(),
                &norm_bytes,
            )?;
        }
        rows_done += take;
    }
    Ok(())
}

/// Row feed for the shared cell-pack core — one builder, three feeders.
/// The core trains fine centroids, assigns, buckets, and assembles the same
/// subsection bytes regardless of where the rows come from; the feeder only
/// decides how rows are read and what the bucket payload is.
pub(crate) enum CellPackSource<'a> {
    /// Cross-commit drain scratch: encoded rows spilled on disk.
    Spilled(&'a SpilledCellRows),
    /// In-RAM encoded rows (compaction / maintenance rebuilds). Rows must
    /// carry dense `0..n` local doc ids; they are packed in that order.
    Rows(Vec<MaterializedIvfRow>),
    /// Commit-time cell pack: the in-RAM fp32 buffer plus inline stable ids.
    /// fp32 is authoritative — training, assignment, and fitted-codec
    /// payloads use it directly (no quantization round-trip); fixed-codec
    /// payloads are encoded once on the pinned grid.
    Fp32 {
        vectors: &'a [f32],
        stable_ids: &'a [i128],
    },
}

impl CellPackSource<'_> {
    fn n_docs(&self, dim: usize) -> usize {
        match self {
            Self::Spilled(spill) => spill.n_rows(),
            Self::Rows(rows) => rows.len(),
            Self::Fp32 { vectors, .. } => vectors.len() / dim.max(1),
        }
    }
}

/// Jitter-sampled fp32 training rows decoded from in-RAM encoded rows.
/// Same index selection as the spilled sampler.
fn sample_ram_materialized_rows(
    rows: &[MaterializedIvfRow],
    sample_size: usize,
    dim: usize,
    seed: u64,
) -> Vec<f32> {
    let n_docs = rows.len();
    let mut sample = vec![0.0f32; sample_size * dim];
    for s in 0..sample_size {
        let idx = sampled_index(s, sample_size, n_docs, seed);
        let enc = &rows[idx].encoded;
        dequantize_sq8_residual_into(
            &enc.scale,
            &enc.offset,
            &enc.codes,
            &enc.residuals,
            enc.rerank_codec
                .residual_divisor()
                .expect("residual-family source has divisor"),
            &mut sample[s * dim..(s + 1) * dim],
        );
    }
    sample
}

/// Jitter-sampled fp32 training rows copied straight from an fp32 corpus.
fn sample_fp32_rows(vectors: &[f32], sample_size: usize, dim: usize, seed: u64) -> Vec<f32> {
    let n_docs = vectors.len() / dim.max(1);
    let mut sample = vec![0.0f32; sample_size * dim];
    for s in 0..sample_size {
        let idx = sampled_index(s, sample_size, n_docs, seed);
        sample[s * dim..(s + 1) * dim].copy_from_slice(&vectors[idx * dim..(idx + 1) * dim]);
    }
    sample
}

/// Shared cell-pack core: build one complete cell-IVF subsection from any
/// [`CellPackSource`], writing the subsection and stable-id stream directly
/// to disk. The single builder behind the drain (spilled rows), the
/// commit-time cell pack (fp32 buffer), and maintenance rebuilds (in-RAM
/// encoded rows).
pub(crate) fn build_cell_subsection_from_source(
    cfg: VectorConfig,
    source: CellPackSource<'_>,
    subsection_path: &Path,
    stable_ids_path: &Path,
    scratch: &Path,
) -> Result<StreamedIvfSubsection, BuildError> {
    let dim = cfg.dim;
    if dim == 0 {
        return Err(BuildError::VectorSchemaMismatch(
            "cell IVF build requires dim > 0".into(),
        ));
    }
    let n_docs = source.n_docs(dim);
    if n_docs == 0 {
        return Err(BuildError::VectorSchemaMismatch(
            "cell IVF build requires at least one row".into(),
        ));
    }
    if !cfg.rerank_codec.is_sq8_residual_family() || !cfg.rerank_codec.supports_metric(cfg.metric) {
        return Err(BuildError::VectorSchemaMismatch(format!(
            "cell IVF build does not support codec {} with metric {:?}",
            cfg.rerank_codec.name(),
            cfg.metric
        )));
    }
    match &source {
        CellPackSource::Spilled(spill) => {
            if spill.dim() != dim || cfg.rerank_codec != spill.rerank_codec() {
                return Err(BuildError::VectorSchemaMismatch(
                    "streamed materialized IVF codec or shape mismatch".into(),
                ));
            }
        }
        CellPackSource::Rows(rows) => {
            if rows
                .iter()
                .any(|row| row.encoded.rerank_codec != cfg.rerank_codec)
            {
                return Err(BuildError::VectorSchemaMismatch(
                    "materialized IVF rebuild requires one matching residual-family codec".into(),
                ));
            }
        }
        CellPackSource::Fp32 {
            vectors,
            stable_ids,
        } => {
            if !vectors.len().is_multiple_of(dim) {
                return Err(BuildError::VectorSchemaMismatch(format!(
                    "fp32 corpus length {} is not a multiple of dim {dim}",
                    vectors.len()
                )));
            }
            if stable_ids.len() != n_docs {
                return Err(BuildError::VectorSchemaMismatch(format!(
                    "fp32 cell IVF stable_ids len {} != n_docs {n_docs}",
                    stable_ids.len()
                )));
            }
        }
    }
    // Mirrors the `materialized_centroids` `n_cent` cap switch so the
    // sample is sized for the runs actually trained: consolidated cells
    // uncapped, sub-threshold cells under the legacy row-count cap.
    let requested_n_cent = if n_docs > CONSOLIDATED_CELL_ROWS_THRESHOLD {
        cfg.n_cent.max(1).min(n_docs)
    } else {
        cfg.n_cent
            .max(1)
            .min(n_cent_row_count_cap(n_docs))
            .min(n_docs)
    };
    let sample_size = if cfg.provided_centroids.is_some() {
        0
    } else {
        partition_kmeans_sample_size(requested_n_cent, n_docs).min(n_docs)
    };
    let chunk_rows = materialized_chunk_rows_for_dim(dim);
    let sample = match &source {
        CellPackSource::Spilled(spill) => {
            sample_spilled_materialized_rows(spill, sample_size, chunk_rows, cfg.rot_seed)?
        }
        CellPackSource::Rows(rows) => {
            sample_ram_materialized_rows(rows, sample_size, dim, cfg.rot_seed)
        }
        CellPackSource::Fp32 { vectors, .. } => {
            sample_fp32_rows(vectors, sample_size, dim, cfg.rot_seed)
        }
    };
    let (n_cent, centroids) = build_phase_timers::timed(&build_phase_timers::TRAIN_US, || {
        materialized_centroids(&cfg, n_docs, &sample)
    });
    let summary_centroid = mean_f32_cluster_major(&centroids, dim, n_cent);
    let code_bytes = dim.div_ceil(u8::BITS as usize);
    let bucket_dir = tempdir_in(scratch)?;
    let mut bucket_writers = Vec::with_capacity(n_cent);
    for centroid in 0..n_cent {
        let path = bucket_dir.path().join(format!("cluster-{centroid}.bin"));
        bucket_writers.push(BufWriter::with_capacity(
            BUCKET_BUF_SIZE,
            File::create(path)?,
        ));
    }
    let mut bucket_counts = vec![0u32; n_cent];
    let fit_quantizer = matches!(cfg.rerank_codec, RerankCodec::Sq8Residual);
    let (mut sq8_min, mut sq8_max) = if fit_quantizer {
        (
            vec![f32::INFINITY; n_cent * dim],
            vec![f32::NEG_INFINITY; n_cent * dim],
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let mut stable_ids = BufWriter::new(File::create(stable_ids_path)?);
    let min_max = fit_quantizer.then_some((sq8_min.as_mut_slice(), sq8_max.as_mut_slice()));
    build_phase_timers::timed(&build_phase_timers::ASSIGN_US, || match &source {
        CellPackSource::Spilled(spill) => stream_materialized_rows_to_buckets(
            spill,
            &cfg,
            &centroids,
            n_cent,
            &mut bucket_writers,
            &mut bucket_counts,
            &mut stable_ids,
            min_max,
        ),
        CellPackSource::Rows(rows) => stream_ram_rows_to_buckets(
            rows,
            &cfg,
            &centroids,
            n_cent,
            &mut bucket_writers,
            &mut bucket_counts,
            &mut stable_ids,
            min_max,
        ),
        CellPackSource::Fp32 {
            vectors,
            stable_ids: inline_ids,
        } => stream_fp32_rows_to_buckets(
            vectors,
            inline_ids,
            &cfg,
            &centroids,
            n_cent,
            &mut bucket_writers,
            &mut bucket_counts,
            &mut stable_ids,
            min_max,
        ),
    })?;
    stable_ids.flush()?;
    stable_ids.get_ref().sync_all()?;
    for writer in bucket_writers {
        writer
            .into_inner()
            .map_err(|error| BuildError::Io(error.into_error()))?
            .sync_all()?;
    }
    let quantizers: Vec<(Vec<f32>, Vec<f32>)> = if fit_quantizer {
        (0..n_cent)
            .map(|centroid| {
                let start = centroid * dim;
                derive_sq8_quantizer_from_min_max(
                    &sq8_min[start..start + dim],
                    &sq8_max[start..start + dim],
                )
            })
            .collect()
    } else {
        (0..n_cent).map(|_| fixed_sq8_quantizer(dim)).collect()
    };
    let codec = cfg.rerank_codec;
    let codec_meta_size = codec.codec_meta_bytes(dim, n_docs, n_cent, cfg.metric);
    let per_vec_bytes = codec.per_vector_bytes(dim);
    let cluster_stride = code_bytes + format::vec::DOC_ID_BYTES + per_vec_bytes;
    let stable_ids_region_bytes = n_docs * format::vec::STABLE_ID_BYTES;
    let layout = IvfSubsectionLayout::compute(
        dim,
        n_cent,
        n_docs,
        cluster_stride,
        codec_meta_size,
        stable_ids_region_bytes,
    );
    let cluster_order = centroid_storage_order(&centroids, n_cent, dim);
    let planned = plan_ivf_cluster_blocks(
        &layout,
        &cluster_order,
        &bucket_counts,
        code_bytes,
        per_vec_bytes,
    );
    let open_region_len = layout
        .stable_ids_off
        .unwrap_or(layout.per_cluster_blocks_off);
    let mut open_region = vec![0u8; open_region_len];
    write_ivf_subsection_header(
        &mut open_region,
        &layout,
        codec_meta_size,
        &summary_centroid,
        &centroids,
    );
    for planned_block in &planned {
        let idx = planned_block.cluster_idx_offset;
        open_region[idx..idx + CLUSTER_IDX_COUNT_OFFSET]
            .copy_from_slice(&(planned_block.doc_offset as u32).to_le_bytes());
        open_region[idx + CLUSTER_IDX_COUNT_OFFSET..idx + CLUSTER_IDX_ENTRY_BYTES]
            .copy_from_slice(&(planned_block.count as u32).to_le_bytes());
    }
    let scale_offset = layout.codec_meta_off;
    let offset_offset = scale_offset + n_cent * dim * size_of::<f32>();
    for (centroid, (scale, offset)) in quantizers.iter().enumerate() {
        let start = centroid * dim * size_of::<f32>();
        open_region[scale_offset + start..scale_offset + start + dim * size_of::<f32>()]
            .copy_from_slice(bytemuck::cast_slice(scale));
        open_region[offset_offset + start..offset_offset + start + dim * size_of::<f32>()]
            .copy_from_slice(bytemuck::cast_slice(offset));
    }
    let norms_offset = matches!(cfg.metric, Metric::L2Sq | Metric::Cosine)
        .then_some(offset_offset + n_cent * dim * size_of::<f32>());
    let mut output = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(subsection_path)?;
    output.set_len((layout.total_size_before_crc + format::CRC_BYTES) as u64)?;
    write_at(&mut output, 0, &open_region)?;
    let stable_ids_offset = layout
        .stable_ids_off
        .expect("streamed materialized IVF always has stable ids");
    output.seek(SeekFrom::Start(stable_ids_offset as u64))?;
    let copied = io::copy(
        &mut BufReader::new(File::open(stable_ids_path)?),
        &mut output,
    )?;
    if copied != stable_ids_region_bytes as u64 {
        return Err(BuildError::VectorSchemaMismatch(format!(
            "streamed stable-id bytes {copied} != expected {stable_ids_region_bytes}"
        )));
    }
    for planned_block in &planned {
        let Some(block) = planned_block.block.as_ref() else {
            continue;
        };
        let path = bucket_dir
            .path()
            .join(format!("cluster-{}.bin", planned_block.centroid_id));
        let (scale, offset) = &quantizers[planned_block.centroid_id];
        stream_bucket_into_subsection(
            &mut output,
            &path,
            block,
            code_bytes,
            dim,
            codec,
            scale,
            offset,
            norms_offset,
        )?;
    }
    output.flush()?;
    output.seek(SeekFrom::Start(0))?;
    let mut remaining = layout.total_size_before_crc;
    let mut crc = 0u32;
    let mut crc_buffer = vec![0u8; MATERIALIZED_BUCKET_CHUNK_BYTES];
    while remaining > 0 {
        let take = remaining.min(crc_buffer.len());
        output.read_exact(&mut crc_buffer[..take])?;
        crc = crc32c_append(crc, &crc_buffer[..take]);
        remaining -= take;
    }
    output.seek(SeekFrom::Start(layout.total_size_before_crc as u64))?;
    output.write_all(&crc.to_le_bytes())?;
    output.flush()?;
    output.sync_all()?;
    Ok(StreamedIvfSubsection {
        n_docs: n_docs as u32,
        rerank_codec: codec,
        subsection_len: (layout.total_size_before_crc + format::CRC_BYTES) as u64,
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

/// Build one complete cell IVF from a cross-batch materialized-row spill,
/// writing the subsection and stable-id stream directly to disk. Thin
/// wrapper over the shared cell-pack core.
pub(crate) fn build_merged_subsection_from_spilled_materialized(
    cfg: VectorConfig,
    spill: &SpilledCellRows,
    subsection_path: &Path,
    stable_ids_path: &Path,
    scratch: &Path,
) -> Result<StreamedIvfSubsection, BuildError> {
    build_cell_subsection_from_source(
        cfg,
        CellPackSource::Spilled(spill),
        subsection_path,
        stable_ids_path,
        scratch,
    )
}

/// Run the shared cell-pack core into a scratch tempdir and read the
/// finished subsection back as in-memory bytes — the shape the commit-time
/// superfile assembly and the maintenance merge paths consume.
fn build_cell_subsection_in_memory(
    cfg: VectorConfig,
    source: CellPackSource<'_>,
) -> Result<MergedIvfSubsection, BuildError> {
    let scratch = tempdir()?;
    let subsection_path = scratch.path().join("cell.ivf");
    let stable_ids_path = scratch.path().join("cell.ids");
    let built = build_cell_subsection_from_source(
        cfg,
        source,
        &subsection_path,
        &stable_ids_path,
        scratch.path(),
    )?;
    let bytes = fs::read(&subsection_path)?;
    if bytes.len() as u64 != built.subsection_len {
        return Err(BuildError::VectorSchemaMismatch(format!(
            "cell subsection read-back {} bytes != built {}",
            bytes.len(),
            built.subsection_len
        )));
    }
    Ok(MergedIvfSubsection {
        bytes,
        n_cent: built.n_cent,
        n_docs: built.n_docs,
        rerank_codec: built.rerank_codec,
        summary_offset_in_sub: built.summary_offset_in_sub,
        codec_meta_offset_in_sub: built.codec_meta_offset_in_sub,
        codec_meta_size: built.codec_meta_size,
    })
}

/// Build one complete cell-IVF subsection from an in-memory fp32 corpus —
/// the commit-time cell pack. Thin adapter over the shared cell-pack core
/// (the same trainer, assigner, bucket scatter, and assembly the drain's
/// cross-commit spill feeder runs), so commit and drain pack cells through
/// literally one builder. `stable_ids` adds the inline identity region
/// needed by packed cells and boundary stubs.
///
/// `vectors` is row-major (`n_docs × dim`); length must be a multiple of `cfg.dim`.
pub(crate) fn build_merged_subsection_from_fp32(
    cfg: VectorConfig,
    vectors: Arc<Vec<f32>>,
    stable_ids: &[i128],
) -> Result<MergedIvfSubsection, BuildError> {
    build_cell_subsection_in_memory(
        cfg,
        CellPackSource::Fp32 {
            vectors: &vectors,
            stable_ids,
        },
    )
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
        inline_stable_ids,
    } = col;

    if let Some(rows) = materialized_rows {
        drop(reservoir);
        drop(inline_stable_ids);
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

    // Summary centroid: mean of trained centroids.
    let summary_centroid = mean_f32_cluster_major(&centroids, dim, n_cent);

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
    let codec = cfg.rerank_codec;
    if !codec.supports_metric(cfg.metric) {
        return Err(BuildError::VectorSchemaMismatch(format!(
            "vector index {:?}: codec {} supports cosine metric only",
            cfg.column,
            codec.name()
        )));
    }
    // Residual-family codecs use per-cluster scale/offset codec_meta plus
    // an i8 residual sidecar in `full[]`.
    let sq8_family = codec.is_sq8_residual_family();
    let fit_sq8_quantizer = matches!(codec, RerankCodec::Sq8Residual);
    let (mut sq8_min_arr, mut sq8_max_arr): (Vec<f32>, Vec<f32>) = if fit_sq8_quantizer {
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

        let sq8_acc: Option<(&mut [f32], &mut [f32])> = if fit_sq8_quantizer {
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
            &mut bucket_writers,
            &mut bucket_counts,
            codec,
            sq8_acc,
        )?;
    }

    let sq8_quantizers: Vec<(Vec<f32>, Vec<f32>)> = if fit_sq8_quantizer {
        (0..n_cent)
            .map(|c| {
                let off = c * dim;
                derive_sq8_quantizer_from_min_max(
                    &sq8_min_arr[off..off + dim],
                    &sq8_max_arr[off..off + dim],
                )
            })
            .collect()
    } else if codec.uses_fixed_quantizer() {
        (0..n_cent).map(|_| fixed_sq8_quantizer(dim)).collect()
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
    // Ordinary streaming ingest: no inline stable-`_id` region. The fp32
    // cell-pack entry point passes ids and gets the same inline region as the
    // materialized drain path.
    let stable_ids_region_bytes = match &inline_stable_ids {
        Some(ids) if ids.len() == n_docs => n_docs * format::vec::STABLE_ID_BYTES,
        Some(ids) => {
            return Err(BuildError::VectorSchemaMismatch(format!(
                "streaming inline_stable_ids len {} != n_docs {n_docs}",
                ids.len()
            )));
        }
        None => 0,
    };
    let layout = IvfSubsectionLayout::compute(
        dim,
        n_cent,
        n_docs,
        cluster_stride,
        codec_meta_size,
        stable_ids_region_bytes,
    );
    let total_size_before_crc = layout.total_size_before_crc;

    let mut bytes =
        alloc_ivf_subsection_with_header(&layout, codec_meta_size, &summary_centroid, &centroids);

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
                RerankCodec::Sq8Residual | RerankCodec::Sq8FixedResidual => {
                    let cluster_rows: &[f32] = bytemuck::cast_slice(&full_block);
                    let (scale_c, offset_c) = &sq8_quantizers[centroid_id];
                    encode_sq8_residual_cluster_simd(
                        cluster_rows,
                        dim,
                        blk.count,
                        blk.first_row,
                        blk.rerank_base,
                        sq8_norms_block_off,
                        scale_c,
                        offset_c,
                        bytes,
                        codec
                            .residual_divisor()
                            .expect("residual-family codec has divisor"),
                    );
                }
            }
            Ok(())
        },
    )?;
    if let (Some(stable_ids_off), Some(ids)) = (layout.stable_ids_off, inline_stable_ids.as_ref()) {
        for (local, &stable_id) in ids.iter().enumerate() {
            let off = stable_ids_off + local * format::vec::STABLE_ID_BYTES;
            bytes[off..off + format::vec::STABLE_ID_BYTES]
                .copy_from_slice(&stable_id.to_le_bytes());
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

/// Residual-family per-cluster encode. Writes a row-interleaved
/// `[code dim u8 ‖ residual dim i8]` body (`2 × dim` bytes per row)
/// at `full_chunk_base + i × 2·dim`. The Sq8 code is the same
/// `sq8_encode_row` quantization; the residual code captures the
/// quantization error at `scale_c[d] / residual_divisor`-sized
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
    scale_c: &[f32],
    offset_c: &[f32],
    bytes: &mut [u8],
    residual_divisor: f32,
) {
    debug_assert_eq!(cluster_rows.len(), cluster_count * dim);
    let row_bytes = dim * 2;
    // Code-leg quantizer constants are constant across the cluster — build
    // once here, not once per row.
    let consts = Sq8EncodeConsts::from_scale_offset(scale_c, offset_c);
    let store_norm = sq8_norms_block_off.is_some();
    let mut recon = vec![0f32; dim];

    for i in 0..cluster_count {
        let src = &cluster_rows[i * dim..(i + 1) * dim];
        let pos = cluster_doc_off + i;
        let row_off = full_chunk_base + i * row_bytes;
        let row_bytes_mut = &mut bytes[row_off..row_off + row_bytes];
        let (code_slice, res_slice) = row_bytes_mut.split_at_mut(dim);
        let norm = encode_sq8_residual_row(
            src,
            &consts,
            scale_c,
            offset_c,
            code_slice,
            res_slice,
            &mut recon,
            store_norm,
            residual_divisor,
        );
        if let (Some(norms_off), Some(n_sq)) = (sq8_norms_block_off, norm) {
            let n_off = norms_off + pos * 4;
            bytes[n_off..n_off + 4].copy_from_slice(&n_sq.to_le_bytes());
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

/// Fixed absolute quantizer shared by every Sq8FixedResidual cluster.
pub(crate) fn fixed_sq8_quantizer(dim: usize) -> (Vec<f32>, Vec<f32>) {
    (vec![SQ8_FIXED_SCALE; dim], vec![SQ8_FIXED_OFFSET; dim])
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
    summary_centroid: &[f32],
    centroids: &[f32],
) -> Vec<u8> {
    let mut bytes = vec![0u8; layout.total_size_before_crc];
    write_ivf_subsection_header(
        &mut bytes,
        layout,
        codec_meta_size,
        summary_centroid,
        centroids,
    );
    bytes
}

/// Write the fixed subsection header and centroid regions into an already
/// allocated destination. Shared by the in-memory and direct-to-file builders.
fn write_ivf_subsection_header(
    bytes: &mut [u8],
    layout: &IvfSubsectionLayout,
    codec_meta_size: usize,
    summary_centroid: &[f32],
    centroids: &[f32],
) {
    debug_assert!(bytes.len() >= layout.codec_meta_off);
    bytes[0..MAGIC_BYTES].copy_from_slice(format::vec::SUB_MAGIC);
    bytes[sub_hdr::VERSION_OFF..sub_hdr::VERSION_OFF + U32_BYTES]
        .copy_from_slice(&format::vec::SUBSECTION_VERSION.to_le_bytes());
    bytes[sub_hdr::CODEC_META_SIZE_OFF..sub_hdr::CODEC_META_SIZE_OFF + U32_BYTES]
        .copy_from_slice(&(codec_meta_size as u32).to_le_bytes());
    bytes[sub_hdr::SUMMARY_OFF_OFF..sub_hdr::SUMMARY_OFF_OFF + U64_BYTES]
        .copy_from_slice(&(layout.summary_off as u64).to_le_bytes());
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
}

/// One cluster's `code‖doc_id‖rerank` sub-region offsets, handed to the
/// per-cluster row writer by [`write_ivf_cluster_blocks`].
#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy)]
struct PlannedClusterBlock {
    centroid_id: usize,
    cluster_idx_offset: usize,
    doc_offset: usize,
    count: usize,
    block: Option<ClusterBlock>,
}

fn plan_ivf_cluster_blocks(
    layout: &IvfSubsectionLayout,
    cluster_order: &[usize],
    cluster_counts: &[u32],
    code_bytes: usize,
    per_vec_bytes: usize,
) -> Vec<PlannedClusterBlock> {
    let cluster_stride = code_bytes + format::vec::DOC_ID_BYTES + per_vec_bytes;
    let mut block_cursor = 0usize;
    let mut doc_offset = 0usize;
    let mut planned = Vec::with_capacity(cluster_order.len());
    for &centroid_id in cluster_order {
        let count = cluster_counts[centroid_id] as usize;
        let cluster_idx_offset = layout.cluster_idx_off + centroid_id * CLUSTER_IDX_ENTRY_BYTES;
        let block = (count > 0).then(|| {
            let block_base = layout.per_cluster_blocks_off + block_cursor;
            let codes_len = count * code_bytes;
            let ids_len = count * format::vec::DOC_ID_BYTES;
            ClusterBlock {
                codes_base: block_base,
                ids_base: block_base + codes_len,
                rerank_base: block_base + codes_len + ids_len,
                first_row: doc_offset,
                count,
            }
        });
        planned.push(PlannedClusterBlock {
            centroid_id,
            cluster_idx_offset,
            doc_offset,
            count,
            block,
        });
        block_cursor += count * cluster_stride;
        doc_offset += count;
    }
    planned
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
    for planned in plan_ivf_cluster_blocks(
        layout,
        cluster_order,
        cluster_counts,
        code_bytes,
        per_vec_bytes,
    ) {
        let idx_base = planned.cluster_idx_offset;
        bytes[idx_base..idx_base + CLUSTER_IDX_COUNT_OFFSET]
            .copy_from_slice(&(planned.doc_offset as u32).to_le_bytes());
        bytes[idx_base + CLUSTER_IDX_COUNT_OFFSET..idx_base + CLUSTER_IDX_ENTRY_BYTES]
            .copy_from_slice(&(planned.count as u32).to_le_bytes());
        if let Some(block) = planned.block {
            write_cluster(bytes, planned.centroid_id, &block)?;
        }
    }
    Ok(())
}

/// Pass 2 of `build_subsection_streaming`: walk the input
/// corpus chunk-by-chunk, assign each row to its centroid,
/// rotate + 1-bit encode it, fold its un-rotated distance into
/// and append the `(local_doc_id, code,
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
    bucket_writers: &mut [BufWriter<File>],
    bucket_counts: &mut [u32],
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

    use bytes::Bytes;
    use tempfile::tempdir;

    use super::*;
    use crate::superfile::vector::{
        cell_posting::EncodedCellRow, reader::VectorReader, spill::MaterializedRowSpillWriter,
    };

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

    /// The centroid-count cap steps up at the two document-count thresholds.
    #[test]
    fn n_cent_row_count_cap_steps_at_thresholds() {
        assert_eq!(n_cent_row_count_cap(0), N_CENT_SMALL);
        assert_eq!(
            n_cent_row_count_cap(N_CENT_MEDIUM_DOC_THRESHOLD - 1),
            N_CENT_SMALL
        );
        assert_eq!(
            n_cent_row_count_cap(N_CENT_MEDIUM_DOC_THRESHOLD),
            N_CENT_MEDIUM
        );
        assert_eq!(
            n_cent_row_count_cap(N_CENT_LARGE_DOC_THRESHOLD - 1),
            N_CENT_MEDIUM
        );
        assert_eq!(
            n_cent_row_count_cap(N_CENT_LARGE_DOC_THRESHOLD),
            N_CENT_LARGE
        );
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
        let json =
            format!(r#"[{{"column":"v","dim":{dim},"n_cent":4,"rot_seed":7,"metric":"cosine"}}]"#);
        let cfg = || VectorConfig {
            column: "v".into(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8Residual,
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
        let mut stream_rows =
            block_on(stream_reader.materialized_index_rows_async("v")).expect("streaming rows");
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

        let mat_rows =
            block_on(mat_reader.materialized_index_rows_async("v")).expect("materialized rows");
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
        let json =
            format!(r#"[{{"column":"v","dim":{dim},"n_cent":4,"rot_seed":7,"metric":"cosine"}}]"#);
        let cfg = || VectorConfig {
            column: "v".into(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8Residual,
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
            mb.load_materialized_rows(0, rows)
                .expect("load materialized");
            Bytes::from(mb.finish().expect("finish materialized"))
        };

        let (na, nb) = (10usize, 8usize);
        let blob_a = build_cell(na, 5_000);
        let blob_b = build_cell(nb, 9_000);
        let reader_a = VectorReader::open(blob_a, &json).expect("open A");
        let reader_b = VectorReader::open(blob_b, &json).expect("open B");

        // Merge: B's local ids shift by `na` (its doc_id_offset).
        let merged = merge_sq8_ivf_subsections(&[(&reader_a, "v", 0), (&reader_b, "v", na as u32)])
            .expect("merge");
        assert_eq!(merged.n_docs as usize, na + nb);

        let mut wb = VectorBuilder::new();
        wb.register_column(cfg()).expect("register merged");
        wb.set_prebuilt_subsection(0, merged).expect("set prebuilt");
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
            rerank_codec: RerankCodec::Sq8Residual,
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

    /// Greedy nearest-neighbor chain: planted 1-D centroids in shuffled
    /// order come out in walk order, starting nearest the mean —
    /// deterministic and input-order independent.
    #[test]
    fn order_centroids_geometrically_chains_neighbors() {
        const DIM: usize = 4;
        // Positions 0, 10, 21, 33 on every axis, supplied shuffled. Mean
        // is 16 → start at 21 (distance 5 beats 10's 6), then chain
        // 21 → 10 → 0 → 33.
        let positions = [33.0f32, 10.0, 0.0, 21.0];
        let mut centroids = Vec::with_capacity(positions.len() * DIM);
        for p in positions {
            centroids.extend(std::iter::repeat_n(p, DIM));
        }
        order_centroids_geometrically(&mut centroids, DIM, positions.len());
        let ordered: Vec<f32> = (0..positions.len()).map(|c| centroids[c * DIM]).collect();
        assert_eq!(ordered, vec![21.0, 10.0, 0.0, 33.0]);

        // A different input permutation converges to the same chain.
        let positions_b = [0.0f32, 33.0, 21.0, 10.0];
        let mut centroids_b = Vec::with_capacity(positions_b.len() * DIM);
        for p in positions_b {
            centroids_b.extend(std::iter::repeat_n(p, DIM));
        }
        order_centroids_geometrically(&mut centroids_b, DIM, positions_b.len());
        let ordered_b: Vec<f32> = (0..positions_b.len())
            .map(|c| centroids_b[c * DIM])
            .collect();
        assert_eq!(ordered_b, ordered, "chain order is input-order invariant");
    }

    // ---- fine-run size bound (split_oversized_fine_runs) ---------------

    /// Sample dim for the split tests — small but above trivial.
    const SPLIT_DIM: usize = 8;
    /// Requested run count for the split tests: 600 sample rows / 10 runs
    /// gives target 60, bound 120.
    const SPLIT_REQUESTED: usize = 10;

    /// 600 rows in two tight blobs (500 near the origin, 100 near 10.0),
    /// jittered deterministically by an LCG so k-means has structure to
    /// subdivide.
    fn two_blob_sample() -> Vec<f32> {
        let mut rows = Vec::with_capacity(600 * SPLIT_DIM);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut jitter = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) % 1000) as f32 / 1000.0 - 0.5
        };
        for r in 0..600 {
            let base = if r < 500 { 0.0f32 } else { 10.0 };
            for _ in 0..SPLIT_DIM {
                rows.push(base + jitter());
            }
        }
        rows
    }

    /// Counts per centroid after a fresh nearest-centroid assignment.
    fn run_counts(sample: &[f32], centroids: &[f32], n_cent: usize) -> Vec<usize> {
        let mut assignments = vec![0u32; sample.len() / SPLIT_DIM];
        assign_to_centroids(sample, centroids, SPLIT_DIM, n_cent, &mut assignments);
        let mut counts = vec![0usize; n_cent];
        for &a in &assignments {
            counts[a as usize] += 1;
        }
        counts
    }

    /// A centroid set that dumps every row into one run (one centroid on
    /// the data, the rest far away) must come back with every run at or
    /// under the bound, growing `n_cent` as needed.
    #[test]
    fn split_oversized_fine_runs_bounds_every_run() {
        let sample = two_blob_sample();
        let sample_n = sample.len() / SPLIT_DIM;
        let target = sample_n.div_ceil(SPLIT_REQUESTED);
        let bound = target * FINE_RUN_SPLIT_BOUND_FACTOR;

        // Centroid 0 sits on the data; 1..5 are parked far away (the
        // starved-seed shape Lloyd produces on blob-heavy cells).
        let mut centroids = vec![0.0f32; 5 * SPLIT_DIM];
        for c in 1..5 {
            for d in 0..SPLIT_DIM {
                centroids[c * SPLIT_DIM + d] = 1_000.0 + c as f32;
            }
        }
        let before = run_counts(&sample, &centroids, 5);
        assert!(
            before.iter().any(|&n| n > bound),
            "fixture must start oversized (max run {} ≤ bound {bound})",
            before.iter().max().expect("nonempty")
        );

        let n_cent =
            split_oversized_fine_runs(&mut centroids, &sample, SPLIT_DIM, SPLIT_REQUESTED, 7, None);
        assert_eq!(centroids.len(), n_cent * SPLIT_DIM);
        assert!(n_cent > 5, "split must add sub-centroids");
        let after = run_counts(&sample, &centroids, n_cent);
        assert!(
            after.iter().all(|&n| n <= bound),
            "every run must fit the bound {bound}: {after:?}"
        );
    }

    /// The consolidated-cell training sampler must not phase-lock to
    /// periodic arrival order. With a corpus whose rows cycle 4 classes
    /// round-robin and a sample of exactly n/4 (the 100M row-fraction
    /// shape, integer stride 4), a plain stride samples one class only and
    /// the other three train no centroid — measured at 100M as 92K-row
    /// mega-runs and post-drain recall 0.136. The jittered indices must
    /// cover every class, stay strictly increasing (the spill reader
    /// streams them in one pass), stay in bounds, and be deterministic per
    /// seed. At or below the consolidated threshold the plain stride ships
    /// unchanged — the 1M/10M cold-probe gates were measured on it.
    #[test]
    fn sampled_index_breaks_periodic_aliasing_on_consolidated_cells() {
        let n_docs = CONSOLIDATED_CELL_ROWS_THRESHOLD * 2;
        let sample_size = n_docs / 4;
        let seed = 7u64;
        let indices: Vec<usize> = (0..sample_size)
            .map(|s| sampled_index(s, sample_size, n_docs, seed))
            .collect();
        let mut class_seen = [false; 4];
        for (s, &idx) in indices.iter().enumerate() {
            assert!(idx < n_docs, "index {idx} out of bounds");
            if s > 0 {
                assert!(
                    idx > indices[s - 1],
                    "indices must be strictly increasing: {} then {idx}",
                    indices[s - 1]
                );
            }
            class_seen[idx % 4] = true;
        }
        assert_eq!(
            class_seen, [true; 4],
            "every periodic class must be sampled (plain stride 4 sees one)"
        );
        let again: Vec<usize> = (0..sample_size)
            .map(|s| sampled_index(s, sample_size, n_docs, seed))
            .collect();
        assert_eq!(indices, again, "same seed must select the same rows");

        // Sub-threshold cells keep the measured plain stride, and a
        // full-coverage sample stays the identity.
        let small = CONSOLIDATED_CELL_ROWS_THRESHOLD;
        for s in [0usize, 7, 1000] {
            assert_eq!(
                sampled_index(s, small / 4, small, seed),
                s * 4,
                "sub-threshold sampling must stay the plain stride"
            );
        }
        assert_eq!(sampled_index(3, 8, 8, seed), 3);
    }

    /// Single-run packs (the commit-time cell delta shape) pass through
    /// untouched — the commit path stays byte-identical.
    #[test]
    fn split_oversized_fine_runs_leaves_single_run_untouched() {
        let sample = two_blob_sample();
        let mut centroids = vec![0.5f32; SPLIT_DIM];
        let original = centroids.clone();
        let n_cent = split_oversized_fine_runs(&mut centroids, &sample, SPLIT_DIM, 1, 7, None);
        assert_eq!(n_cent, 1);
        assert_eq!(centroids, original);
    }

    /// The cell-pack `n_cent` policy switches at the consolidated-cell
    /// boundary: at or below it the legacy row-count cap binds the
    /// *requested* count (the measured 1M/10M layout); above it the
    /// byte-target count is taken uncapped. Run-split may grow past the
    /// request on either side when Lloyd leaves an oversized run.
    #[test]
    fn materialized_centroids_caps_only_below_consolidated_threshold() {
        let dim = 8;
        // Byte-target n_cent above the legacy sub-100K cap of 64.
        let requested = 92;
        let sample: Vec<f32> = (0..256 * dim).map(|i| (i % 251) as f32 * 0.01).collect();
        let mk = |n_docs: usize| {
            let cfg = VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: requested,
                rot_seed: 7,
                metric: Metric::L2Sq,
                rerank_codec: RerankCodec::Sq8Residual,
                provided_centroids: None,
            };
            materialized_centroids(&cfg, n_docs, &sample).0
        };
        let below = mk(CONSOLIDATED_CELL_ROWS_THRESHOLD);
        let above = mk(CONSOLIDATED_CELL_ROWS_THRESHOLD + 1);
        // Sub-threshold trains from the legacy cap; run-split may grow a
        // few sub-centroids on top, but must not jump to the uncapped
        // byte-target request.
        assert!(
            below >= N_CENT_SMALL && below < requested,
            "sub-threshold cell stays near the legacy cap (got {below}, cap {N_CENT_SMALL}, byte-target {requested})"
        );
        assert!(
            above >= requested,
            "consolidated cell takes the byte-target count uncapped"
        );
    }

    /// Sub-threshold cells still split oversized fine runs. Without this,
    /// fine-first p=1 post-drain recall flaps 0.985↔0.995 at 1M/256c when
    /// Lloyd parks a 2×-skewed run (same cell membership both ways).
    #[test]
    fn materialized_centroids_splits_oversized_runs_below_threshold() {
        let sample = two_blob_sample();
        let sample_n = sample.len() / SPLIT_DIM;
        let target = sample_n.div_ceil(SPLIT_REQUESTED);
        let bound = target * FINE_RUN_SPLIT_BOUND_FACTOR;
        let cfg = VectorConfig {
            column: "emb".into(),
            dim: SPLIT_DIM,
            n_cent: SPLIT_REQUESTED,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        };
        let (n_cent, centroids) =
            materialized_centroids(&cfg, CONSOLIDATED_CELL_ROWS_THRESHOLD, &sample);
        assert!(
            n_cent >= SPLIT_REQUESTED,
            "split may grow past the request, never shrink it"
        );
        let after = run_counts(&sample, &centroids, n_cent);
        assert!(
            after.iter().all(|&n| n <= bound),
            "sub-threshold cell must still bound every fine run at {bound}: {after:?}"
        );
    }

    /// Same inputs and seed produce identical output centroids.
    #[test]
    fn split_oversized_fine_runs_is_deterministic() {
        let sample = two_blob_sample();
        let make = || {
            let mut centroids = vec![0.0f32; 5 * SPLIT_DIM];
            for c in 1..5 {
                for d in 0..SPLIT_DIM {
                    centroids[c * SPLIT_DIM + d] = 1_000.0 + c as f32;
                }
            }
            let n = split_oversized_fine_runs(
                &mut centroids,
                &sample,
                SPLIT_DIM,
                SPLIT_REQUESTED,
                7,
                None,
            );
            (n, centroids)
        };
        assert_eq!(make(), make());
    }

    #[test]
    fn build_merged_subsection_from_fp32_stays_in_memory() {
        use bytes::Bytes;

        use crate::superfile::vector::reader::VectorReader;

        let dim = 16;
        let n = 5;
        let mut corpus = Vec::with_capacity(n * dim);
        for r in 0..n {
            for c in 0..dim {
                corpus.push((r as f32) * 0.01 + (c as f32) * 0.001);
            }
        }
        let cfg = VectorConfig {
            column: "v".into(),
            dim,
            n_cent: 64,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        };
        let ids: Vec<i128> = (0..n as i128).map(|i| 9_000 + i).collect();
        let sub = build_merged_subsection_from_fp32(cfg.clone(), Arc::new(corpus), &ids)
            .expect("fp32 build");
        assert_eq!(sub.n_docs, n as u32);
        assert!(sub.n_cent >= 1);
        assert!(!sub.bytes.is_empty());
        let blob = finish_multi_cell_blob(&[(0, sub)]).expect("multi-cell blob");
        let json =
            format!(r#"[{{"column":"v","dim":{dim},"n_cent":64,"rot_seed":7,"metric":"l2sq"}}]"#);
        let reader = VectorReader::open(Bytes::from(blob), &json).expect("open fp32 cell pack");
        let resolved = reader
            .inline_stable_ids_for_locals(&[0, 1, 2])
            .expect("inline stable ids");
        assert_eq!(resolved, ids[0..3]);
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

    /// Two complete cell-IVFs packed into one v2 multi-cell blob round-trip
    /// through open + flat cluster_centroids + packed_cell_ids.
    #[test]
    fn multi_cell_blob_round_trips_cell_directory_and_centroids() {
        use std::sync::Arc;

        use bytes::Bytes;

        use crate::superfile::vector::{
            builder::{build_merged_subsection_from_materialized, finish_multi_cell_blob},
            cell_posting::EncodedCellRow,
            reader::VectorReader,
        };

        let dim = 16;
        let make_rows = |cell: u32, n: usize| -> Vec<MaterializedIvfRow> {
            let scale: Arc<[f32]> = Arc::from(vec![1.0f32; dim]);
            let offset: Arc<[f32]> = Arc::from(vec![0.0f32; dim]);
            (0..n)
                .map(|i| {
                    let local = i as u32;
                    let stable_id = (cell as i128) * 1_000 + local as i128;
                    let mut codes = vec![0u8; dim];
                    codes[0] = (cell as u8).wrapping_add(i as u8);
                    let encoded = EncodedCellRow {
                        stable_id,
                        rerank_codec: RerankCodec::Sq8Residual,
                        scale: Arc::clone(&scale),
                        offset: Arc::clone(&offset),
                        codes,
                        residuals: vec![0u8; dim],
                        norm_sq: Some(1.0),
                    };
                    MaterializedIvfRow {
                        local_doc_id: local,
                        stable_id,
                        cluster: 0,
                        rabitq_code: vec![0u8; dim.div_ceil(8)],
                        encoded,
                    }
                })
                .collect()
        };
        let cfg = |n_cent: usize| VectorConfig {
            column: "emb".into(),
            dim,
            n_cent,
            rot_seed: 1,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        };
        let sub0 = build_merged_subsection_from_materialized(cfg(2), make_rows(0, 4))
            .expect("cell 0 subsection");
        let sub1 = build_merged_subsection_from_materialized(cfg(2), make_rows(1, 3))
            .expect("cell 1 subsection");
        let cells = vec![(0, sub0), (1, sub1)];
        let blob = finish_multi_cell_blob(&cells).expect("pack");
        let streamed_sources: Vec<BorrowedMultiCellSubsection<'_>> = cells
            .iter()
            .map(|(cell_id, subsection)| BorrowedMultiCellSubsection {
                cell_id: *cell_id,
                subsection,
            })
            .collect();
        let mut streamed = Vec::new();
        finish_multi_cell_blob_to(&streamed_sources, &mut streamed).expect("stream pack");
        assert_eq!(
            streamed, blob,
            "streamed and in-memory multi-cell assembly must be byte-identical"
        );
        let json =
            format!(r#"[{{"column":"emb","dim":{dim},"n_cent":2,"rot_seed":1,"metric":"l2sq"}}]"#);
        let reader = VectorReader::open(Bytes::from(blob.clone()), &json).expect("open multi-cell");
        assert!(reader.is_multi_cell());
        assert_eq!(reader.packed_cell_ids(), &[0, 1]);
        assert_eq!(reader.n_docs(), 7);
        let (n_cent, got_dim, _centroids, counts) = reader
            .cluster_centroids("emb")
            .expect("concatenated centroids");
        assert_eq!(got_dim, dim as u32);
        // Two cells × n_cent=2 each → flat directory must expose all four
        // fine centroids (not only the first cell).
        assert_eq!(n_cent, 4, "flat n_cent must sum packed cells, got {n_cent}");
        assert_eq!(counts.len(), n_cent as usize);
        assert!(counts.iter().any(|&c| c > 0));
        assert_eq!(reader.resolve_flat_cluster(0), Some((0, 0)));
        assert_eq!(reader.resolve_flat_cluster(2), Some((1, 0)));
        assert_eq!(reader.resolve_flat_cluster(3), Some((1, 1)));

        let mut zero_codec = blob;
        let directory_start = OUTER_HEADER_SIZE;
        let directory_size = 2 * CELL_DIR_ENTRY_SIZE;
        for entry in 0..2 {
            let codec_off =
                directory_start + entry * CELL_DIR_ENTRY_SIZE + cell_dir_entry::CODEC_ID_OFF;
            zero_codec[codec_off..codec_off + U32_BYTES].copy_from_slice(&0u32.to_le_bytes());
        }
        let directory_crc = crc32c(&zero_codec[directory_start..directory_start + directory_size]);
        let crc_off = directory_start + directory_size;
        zero_codec[crc_off..crc_off + format::CRC_BYTES]
            .copy_from_slice(&directory_crc.to_le_bytes());
        assert!(
            VectorReader::open(Bytes::from(zero_codec), &json).is_err(),
            "zero codec id has no v2 compatibility fallback"
        );
    }

    #[test]
    fn fixed_residual_multi_cell_rebuild_preserves_payload_bytes() {
        let dim = 16;
        let make_rows = |cell: u32| -> Vec<MaterializedIvfRow> {
            let scale: Arc<[f32]> = Arc::from(vec![SQ8_FIXED_SCALE; dim]);
            let offset: Arc<[f32]> = Arc::from(vec![SQ8_FIXED_OFFSET; dim]);
            (0..4)
                .map(|i| {
                    let stable_id = i128::from(cell) * 100 + i;
                    MaterializedIvfRow {
                        local_doc_id: i as u32,
                        stable_id,
                        cluster: 0,
                        rabitq_code: vec![0; dim.div_ceil(8)],
                        encoded: EncodedCellRow {
                            stable_id,
                            rerank_codec: RerankCodec::Sq8FixedResidual,
                            scale: Arc::clone(&scale),
                            offset: Arc::clone(&offset),
                            codes: vec![64 + i as u8; dim],
                            residuals: vec![i as i8 as u8; dim],
                            norm_sq: None,
                        },
                    }
                })
                .collect()
        };
        let config = VectorConfig {
            column: "emb".into(),
            dim,
            n_cent: 2,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8FixedResidual,
            provided_centroids: None,
        };
        let source_rows: Vec<MaterializedIvfRow> = [make_rows(0), make_rows(1)].concat();
        let sub0 = build_merged_subsection_from_materialized(config.clone(), make_rows(0))
            .expect("fixed cell 0");
        let sub1 =
            build_merged_subsection_from_materialized(config, make_rows(1)).expect("fixed cell 1");
        let blob = finish_multi_cell_blob(&[(0, sub0), (1, sub1)]).expect("pack fixed cells");
        let json = r#"[{"column":"emb","dim":16,"n_cent":2,"rot_seed":7,"metric":"cosine"}]"#;
        let reader = VectorReader::open(Bytes::from(blob), json).expect("open fixed multi-cell");
        assert!(
            reader
                .vector_columns_config()
                .all(|column| column.rerank_codec == RerankCodec::Sq8FixedResidual)
        );
        let mut rebuilt =
            block_on(reader.materialized_index_rows_async("emb")).expect("materialize fixed rows");
        rebuilt.sort_by_key(|row| row.stable_id);
        let mut expected = source_rows;
        expected.sort_by_key(|row| row.stable_id);
        for (before, after) in expected.iter().zip(&rebuilt) {
            assert_eq!(before.stable_id, after.stable_id);
            assert_eq!(before.encoded.codes, after.encoded.codes);
            assert_eq!(before.encoded.residuals, after.encoded.residuals);
            assert_eq!(after.encoded.rerank_codec, RerankCodec::Sq8FixedResidual);
        }
    }

    #[test]
    fn streamed_materialized_cell_matches_in_memory_fixed_residual() {
        let dim = 16;
        let scale: Arc<[f32]> = Arc::from(vec![SQ8_FIXED_SCALE; dim]);
        let offset: Arc<[f32]> = Arc::from(vec![SQ8_FIXED_OFFSET; dim]);
        let rows: Vec<MaterializedIvfRow> = (0..32)
            .map(|row| {
                let stable_id = i128::from(row) * 17 + 3;
                MaterializedIvfRow {
                    local_doc_id: row,
                    stable_id,
                    cluster: 0,
                    rabitq_code: vec![row as u8; dim.div_ceil(8)],
                    encoded: EncodedCellRow {
                        stable_id,
                        rerank_codec: RerankCodec::Sq8FixedResidual,
                        scale: Arc::clone(&scale),
                        offset: Arc::clone(&offset),
                        codes: vec![32 + row as u8; dim],
                        residuals: vec![(row as i8 - 16) as u8; dim],
                        norm_sq: None,
                    },
                }
            })
            .collect();
        let config = VectorConfig {
            column: "emb".into(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8FixedResidual,
            provided_centroids: None,
        };
        let expected = build_merged_subsection_from_materialized(config.clone(), rows.clone())
            .expect("in-memory materialized build");
        let directory = tempdir().expect("tempdir");
        let mut spill_writer =
            MaterializedRowSpillWriter::create(directory.path(), 11, dim, dim.div_ceil(8))
                .expect("spill writer");
        for row in &rows {
            spill_writer.append(row).expect("spill row");
        }
        let spill = spill_writer.finish().expect("finish spill");
        let subsection_path = directory.path().join("streamed.ivf");
        let stable_ids_path = directory.path().join("streamed.ids");
        let built = build_merged_subsection_from_spilled_materialized(
            config,
            &spill,
            &subsection_path,
            &stable_ids_path,
            directory.path(),
        )
        .expect("streamed materialized build");
        assert_eq!(built.n_docs, rows.len() as u32);
        assert_eq!(
            read(&subsection_path).expect("read streamed subsection"),
            expected.bytes,
            "streamed and in-memory materialized builders must be byte-identical"
        );
        let expected_ids: Vec<u8> = rows
            .iter()
            .flat_map(|row| row.stable_id.to_le_bytes())
            .collect();
        assert_eq!(
            read(&stable_ids_path).expect("read stable ids"),
            expected_ids
        );
    }

    /// Multi-cell cluster search must return nearest-first. A descending
    /// merge truncates to the farthest hits and collapses packed-shard recall.
    #[tokio::test]
    async fn multi_cell_search_returns_ascending_distance() {
        use std::sync::Arc;

        use bytes::Bytes;

        use crate::superfile::vector::{
            builder::{build_merged_subsection_from_materialized, finish_multi_cell_blob},
            cell_posting::EncodedCellRow,
            reader::VectorReader,
        };

        let dim = 16;
        let make_rows = |cell: u32, n: usize| -> Vec<MaterializedIvfRow> {
            let scale: Arc<[f32]> = Arc::from(vec![1.0f32; dim]);
            let offset: Arc<[f32]> = Arc::from(vec![0.0f32; dim]);
            (0..n)
                .map(|i| {
                    let local = i as u32;
                    let stable_id = (cell as i128) * 1_000 + local as i128;
                    let mut codes = vec![0u8; dim];
                    codes[0] = (cell as u8).wrapping_add(i as u8);
                    MaterializedIvfRow {
                        local_doc_id: local,
                        stable_id,
                        cluster: 0,
                        rabitq_code: vec![0u8; dim.div_ceil(8)],
                        encoded: EncodedCellRow {
                            stable_id,
                            rerank_codec: RerankCodec::Sq8Residual,
                            scale: Arc::clone(&scale),
                            offset: Arc::clone(&offset),
                            codes,
                            residuals: vec![0u8; dim],
                            norm_sq: Some(1.0),
                        },
                    }
                })
                .collect()
        };
        let cfg = |n_cent: usize| VectorConfig {
            column: "emb".into(),
            dim,
            n_cent,
            rot_seed: 1,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        };
        let sub0 =
            build_merged_subsection_from_materialized(cfg(2), make_rows(0, 4)).expect("cell 0");
        let sub1 =
            build_merged_subsection_from_materialized(cfg(2), make_rows(1, 3)).expect("cell 1");
        let blob = finish_multi_cell_blob(&[(0, sub0), (1, sub1)]).expect("pack");
        let json =
            format!(r#"[{{"column":"emb","dim":{dim},"n_cent":2,"rot_seed":1,"metric":"l2sq"}}]"#);
        let reader = VectorReader::open(Bytes::from(blob), &json).expect("open");
        let q = vec![0.0f32; dim];
        let hits = reader
            .search_clusters_async("emb", &q, 3, &[0, 1, 2, 3], 8, None, None, None, None)
            .await
            .expect("search");
        assert_eq!(hits.len(), 3);
        assert!(
            hits[0].1 <= hits[1].1 && hits[1].1 <= hits[2].1,
            "multi-cell hits must be ascending distance, got {hits:?}"
        );
    }

    /// File-local allow bitmaps must be remapped to cell-local ids before
    /// probing each packed cell. Passing the file-local set through unchanged
    /// drops every match in cells after the first (cell-local ids restart at 0).
    #[tokio::test]
    async fn multi_cell_search_remaps_file_local_allow() {
        use std::sync::Arc;

        use bytes::Bytes;
        use roaring::RoaringBitmap;

        use crate::superfile::vector::{
            builder::{build_merged_subsection_from_materialized, finish_multi_cell_blob},
            cell_posting::EncodedCellRow,
            reader::VectorReader,
        };

        let dim = 16;
        let make_rows = |cell: u32, n: usize| -> Vec<MaterializedIvfRow> {
            let scale: Arc<[f32]> = Arc::from(vec![1.0f32; dim]);
            let offset: Arc<[f32]> = Arc::from(vec![0.0f32; dim]);
            (0..n)
                .map(|i| {
                    let local = i as u32;
                    let stable_id = (cell as i128) * 1_000 + local as i128;
                    let mut codes = vec![0u8; dim];
                    codes[0] = (cell as u8).wrapping_add(i as u8);
                    MaterializedIvfRow {
                        local_doc_id: local,
                        stable_id,
                        cluster: 0,
                        rabitq_code: vec![0u8; dim.div_ceil(8)],
                        encoded: EncodedCellRow {
                            stable_id,
                            rerank_codec: RerankCodec::Sq8Residual,
                            scale: Arc::clone(&scale),
                            offset: Arc::clone(&offset),
                            codes,
                            residuals: vec![0u8; dim],
                            norm_sq: Some(1.0),
                        },
                    }
                })
                .collect()
        };
        let cfg = |n_cent: usize| VectorConfig {
            column: "emb".into(),
            dim,
            n_cent,
            rot_seed: 1,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        };
        // cell0 → file-local 0..3; cell1 → file-local 4..6.
        let sub0 =
            build_merged_subsection_from_materialized(cfg(2), make_rows(0, 4)).expect("cell 0");
        let sub1 =
            build_merged_subsection_from_materialized(cfg(2), make_rows(1, 3)).expect("cell 1");
        let blob = finish_multi_cell_blob(&[(0, sub0), (1, sub1)]).expect("pack");
        let json =
            format!(r#"[{{"column":"emb","dim":{dim},"n_cent":2,"rot_seed":1,"metric":"l2sq"}}]"#);
        let reader = VectorReader::open(Bytes::from(blob), &json).expect("open");
        let mut allow = RoaringBitmap::new();
        allow.insert(4);
        allow.insert(5);
        allow.insert(6);
        let q = vec![0.0f32; dim];
        let hits = reader
            .search_clusters_async(
                "emb",
                &q,
                3,
                &[0, 1, 2, 3],
                8,
                Some(Arc::new(allow)),
                None,
                None,
                None,
            )
            .await
            .expect("filtered multi-cell search");
        assert_eq!(
            hits.len(),
            3,
            "expected all three allowed cell1 rows, got {hits:?}"
        );
        for (file_local, _) in &hits {
            assert!(
                *file_local >= 4,
                "allow was cell1-only (file-local 4..6); got hit {file_local}"
            );
        }
    }

    /// Materializing a multi-cell blob by cell id returns only that cell's
    /// rows; full materialize concatenates all cells with file-local ids.
    #[tokio::test]
    async fn multi_cell_materialize_filters_by_cell_directory() {
        use std::sync::Arc;

        use bytes::Bytes;

        use crate::superfile::vector::{
            builder::{build_merged_subsection_from_materialized, finish_multi_cell_blob},
            cell_posting::EncodedCellRow,
            reader::VectorReader,
        };

        let dim = 16;
        let make_rows = |cell: u32, n: usize| -> Vec<MaterializedIvfRow> {
            let scale: Arc<[f32]> = Arc::from(vec![1.0f32; dim]);
            let offset: Arc<[f32]> = Arc::from(vec![0.0f32; dim]);
            (0..n)
                .map(|i| {
                    let local = i as u32;
                    let stable_id = (cell as i128) * 1_000 + local as i128;
                    MaterializedIvfRow {
                        local_doc_id: local,
                        stable_id,
                        cluster: 0,
                        rabitq_code: vec![0u8; dim.div_ceil(8)],
                        encoded: EncodedCellRow {
                            stable_id,
                            rerank_codec: RerankCodec::Sq8Residual,
                            scale: Arc::clone(&scale),
                            offset: Arc::clone(&offset),
                            codes: vec![cell as u8; dim],
                            residuals: vec![0u8; dim],
                            norm_sq: Some(1.0),
                        },
                    }
                })
                .collect()
        };
        let cfg = |n_cent: usize| VectorConfig {
            column: "emb".into(),
            dim,
            n_cent,
            rot_seed: 1,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
            provided_centroids: None,
        };
        let sub0 =
            build_merged_subsection_from_materialized(cfg(2), make_rows(7, 3)).expect("cell 7");
        let sub1 =
            build_merged_subsection_from_materialized(cfg(2), make_rows(15, 2)).expect("cell 15");
        let blob = finish_multi_cell_blob(&[(7, sub0), (15, sub1)]).expect("pack");
        let json =
            format!(r#"[{{"column":"emb","dim":{dim},"n_cent":2,"rot_seed":1,"metric":"l2sq"}}]"#);
        let reader = VectorReader::open(Bytes::from(blob), &json).expect("open");

        let only15 = reader
            .materialized_cells_rows_async(Some(&[15]))
            .await
            .expect("materialize cell 15");
        assert_eq!(only15.len(), 1);
        assert_eq!(only15[0].0, 15);
        assert_eq!(only15[0].1.len(), 2);
        assert!(only15[0].1.iter().all(|r| r.stable_id / 1000 == 15));

        let all = reader
            .materialized_index_rows_async("emb")
            .await
            .expect("all cells");
        assert_eq!(all.len(), 5);
        // File-local ids: cell 7 → 0..2, cell 15 → 3..4.
        let locals: Vec<u32> = all.iter().map(|r| r.local_doc_id).collect();
        assert_eq!(locals, vec![0, 1, 2, 3, 4]);
        assert_eq!(reader.packed_cell_n_docs(7), Some(3));
        assert_eq!(reader.packed_cell_n_docs(15), Some(2));
        assert_eq!(reader.packed_cell_n_docs(99), None);

        // Overflow discovery uses per-cell counts from the directory — the
        // largest cell (7 with 3 docs) is preferred over cell 15 (2 docs).
        let counts: Vec<(u32, u32)> = reader
            .packed_cell_ids()
            .iter()
            .filter_map(|&c| reader.packed_cell_n_docs(c).map(|n| (c, n)))
            .collect();
        let overflow = counts
            .iter()
            .copied()
            .max_by_key(|(_, n)| *n)
            .expect("counts");
        assert_eq!(overflow, (7, 3));

        // Packed summary is doc-weighted across cells (3 docs in cell 7 +
        // 2 in cell 15), not just the first cell's IVF summary.
        let packed_summary = reader.summary("emb").expect("packed summary");
        assert_eq!(packed_summary.len(), dim);
        let cell7_only = {
            let sub0 = build_merged_subsection_from_materialized(cfg(2), make_rows(7, 3))
                .expect("cell 7 alone");
            let blob0 = finish_multi_cell_blob(&[(7, sub0)]).expect("pack one");
            VectorReader::open(Bytes::from(blob0), &json)
                .expect("open")
                .summary("emb")
                .expect("cell7 summary")
        };
        let cell15_only = {
            let sub1 = build_merged_subsection_from_materialized(cfg(2), make_rows(15, 2))
                .expect("cell 15 alone");
            let blob1 = finish_multi_cell_blob(&[(15, sub1)]).expect("pack one");
            VectorReader::open(Bytes::from(blob1), &json)
                .expect("open")
                .summary("emb")
                .expect("cell15 summary")
        };
        for d in 0..dim {
            let expected = (cell7_only[d] * 3.0 + cell15_only[d] * 2.0) / 5.0;
            assert!(
                (packed_summary[d] - expected).abs() < 1e-5,
                "dim {d}: packed={} expected={expected}",
                packed_summary[d]
            );
        }

        // File-local ids (search / parquet order) must resolve across cells —
        // not only against the first cell's stable_id region.
        let file_locals: Vec<u32> = (0..5).collect();
        let resolved = reader
            .inline_stable_ids_for_locals(&file_locals)
            .expect("multi-cell file-local stable ids");
        assert_eq!(
            resolved,
            vec![7000, 7001, 7002, 15000, 15001],
            "file-local → stable_id must span both packed cells"
        );
    }
}
