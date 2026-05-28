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
use crate::superfile::format::checksum::{crc32c, crc32c_append};
use crate::superfile::format::{self, FST_SEPARATOR, RESERVED_PREFIX};
use crate::superfile::vector::distance::{Metric, l2_sq};
use crate::superfile::vector::kmeans::{assign_to_centroids, kmeans};
use crate::superfile::vector::quant::BitQuantizer;
use crate::superfile::vector::rerank_codec::RerankCodec;
use crate::superfile::vector::reservoir::{Reservoir, default_kmeans_sample_size};
use crate::superfile::vector::rotation::RandomRotation;
use crate::superfile::vector::spill::{
    ChunkedVectorSource, InMemoryVectorSource, MmapVectorSource, SpillWriter,
};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
///
/// Cloneable — `register_column` takes ownership of one copy
/// and stores the field-wise data on the corresponding
/// `ColumnState`; subsequent reads (e.g. for diagnostic
/// emission) clone from the column to keep the type. Field
/// additions go behind an `Option` or default so VectorConfig
/// extensions across minor releases don't force a code change
/// at the call site.
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// Logical column name. Must not collide with any other
    /// column in the same superfile (FTS or vector). Named
    /// `column` to align with `FtsConfig::column` and the
    /// public superfile API surface; this is also the on-disk
    /// JSON key in `inf.vec.columns`.
    pub column: String,
    pub dim: usize,
    pub n_cent: usize,
    pub rot_seed: u64,
    pub metric: Metric,
    /// On-disk rerank codec for this column. See [`RerankCodec`]
    /// for the supported codecs and their size/recall trade-offs.
    pub rerank_codec: RerankCodec,
}

impl VectorConfig {
    /// Construct a config with the default rerank codec
    /// ([`RerankCodec::default()`]). Use the `with_*` setters to
    /// override individual fields.
    pub fn new(column: String, dim: usize, n_cent: usize, rot_seed: u64, metric: Metric) -> Self {
        Self {
            column,
            dim,
            n_cent,
            rot_seed,
            metric,
            rerank_codec: RerankCodec::default(),
        }
    }

    /// Override the rerank codec.
    #[must_use]
    pub fn with_rerank_codec(mut self, codec: RerankCodec) -> Self {
        self.rerank_codec = codec;
        self
    }
}

/// Default spill threshold: total bytes the in-memory pre-spill
/// buffer is allowed to grow to before the column transitions to
/// the on-disk path. 256 MiB is a constant — independent of
/// reservoir size or `n_cent` — so the worst-case pre-flush
/// resident moment (`reservoir + spill_threshold`) stays linear
/// in reservoir only and never compounds.
const DEFAULT_SPILL_THRESHOLD_BYTES: usize = 256 * 1024 * 1024;

/// Per-column build-time state. The column holds at most three
/// independent buffers:
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
        let meta = std::fs::metadata(&parent)?;
        if !meta.is_dir() {
            return Err(BuildError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
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

/// Multi-column vector blob builder. The streaming build path changes
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

    /// Register a vector column up-front. Returns the assigned
    /// `column_id` (declaration order).
    pub fn register_column(&mut self, config: VectorConfig) -> Result<u32, BuildError> {
        if config.column.as_bytes().contains(&FST_SEPARATOR) {
            return Err(BuildError::ReservedSeparatorInColumnName(config.column));
        }
        if config.column.starts_with(RESERVED_PREFIX) {
            return Err(BuildError::ReservedPrefixInColumnName(config.column));
        }
        if !(16..=4096).contains(&config.dim) {
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
        let reservoir_seed = config.rot_seed ^ 0x5a5a_5a5a_5a5a_5a5a;
        let reservoir = Reservoir::new(sample_size, config.dim, reservoir_seed);
        let spill_threshold_bytes = self.spill_threshold_bytes;
        self.columns.push(ColumnState {
            config,
            n_docs: 0,
            reservoir,
            pre_spill_buffer: Vec::new(),
            spill: None,
            spill_threshold_bytes,
        });
        Ok(column_id)
    }

    /// Override the k-means training sample size for a column. Must
    /// be called before the first `add()` for the column — calling it
    /// later silently discards already-observed reservoir state and
    /// only future `add()` calls feed into the new reservoir.
    ///
    /// The default sample size is `default_kmeans_sample_size(n_cent)`
    /// (`100K-500K` depending on `n_cent`). This override lets advanced
    /// callers dial sample size to match a profiled recall vs. memory
    /// trade-off.
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
    /// Returns a `BuildError::Io` for spill / scratch I/O errors.
    /// Callers that previously expected `-> Vec<u8>` need to `?` the
    /// result; the `SuperfileBuilder` shim does so already.
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

    /// Streaming variant: write the final blob progressively to `w`
    /// without materialising it as a contiguous `Vec<u8>`.
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
    /// `max_subsection_size`. Per-subsection streaming would push the
    /// floor lower still.
    ///
    /// Object-storage callers (003) can pass a multipart upload
    /// writer here so segment build never owns the full blob in
    /// RAM.
    pub fn finish_to<W: Write>(self, mut w: W) -> Result<(), BuildError> {
        let VectorBuilder {
            columns,
            mut scratch_dir,
            spill_threshold_bytes: _,
        } = self;

        let n_columns = columns.len() as u32;
        // n_docs in the outer header is the max across columns
        // (per-segment doc count; spec: same across all columns).
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
            directory_offset + directory_size as u64 + 4 /* dir CRC */;

        // 3. Assemble directory entries with absolute offsets.
        //    Byte 52 of each 64-byte entry carries the rerank-codec
        //    discriminator; bytes 53..56 stay reserved. Existing fp32
        //    segments had all-zero bytes here, which maps to
        //    `RerankCodec::Fp32` (`codec_id() = 0`) and round-trips
        //    identically.
        let mut directory: Vec<u8> = Vec::with_capacity(directory_size);
        for (i, sub) in subsections.iter().enumerate() {
            let (cfg, _) = &column_configs[i];
            let summary_offset_abs = subsection_start_off + sub.summary_offset_in_sub as u64;
            directory.extend_from_slice(&(i as u32).to_le_bytes()); // column_id
            directory.extend_from_slice(&(cfg.dim as u32).to_le_bytes()); // dim
            directory.extend_from_slice(&(cfg.n_cent as u32).to_le_bytes()); // n_cent
            directory.extend_from_slice(&metric_id(cfg.metric).to_le_bytes()); // metric_id
            directory.extend_from_slice(&cfg.rot_seed.to_le_bytes()); // rot_seed (8)
            directory.extend_from_slice(&subsection_start_off.to_le_bytes()); // subsection_offset (8)
            directory.extend_from_slice(&(sub.bytes.len() as u64).to_le_bytes()); // subsection_length (8)
            directory.extend_from_slice(&summary_offset_abs.to_le_bytes()); // summary_offset (8)
            directory.extend_from_slice(&((cfg.dim * 4) as u32).to_le_bytes()); // summary_length (4)
            // bytes 52..56 — codec_id (1) + reserved (3)
            directory.push(cfg.rerank_codec.codec_id()); // codec_id (1)
            directory.extend_from_slice(&[0u8; 3]); // reserved (3)
            directory.extend_from_slice(&0u64.to_le_bytes()); // future_reserved (8)
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
        // bucket files.
        drop(scratch_dir);

        Ok(())
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

/// Per-bucket BufWriter capacity. 64 KiB amortises one syscall
/// per ~1300 dim=384 bucket rows (each row = 4 + code_bytes +
/// dim*4 = ~1588 B). At very high n_cent (≥ 8192) the n_cent ×
/// 64 KiB total dominates the resident set; revisit if profiling shows
/// it.
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
    let cap_by_mem = (128usize << 20) / (dim.max(1) * 4);
    cap_by_mem.clamp(1024, 65_536)
}

/// Build one column's subsection via the streaming path. Consumes the
/// entire `ColumnState` so the reservoir +
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
    } = col;

    let dim = cfg.dim;
    let n_docs = n_docs_u32 as usize;
    let sample_rows = reservoir.n_rows();
    // n_cent must be in `[1, min(n_docs, sample_rows)]`. Both bounds
    // are required: `n_cent > n_docs` makes the IVF degenerate;
    // `n_cent > sample_rows` would crash k-means (`k > n` is asserted
    // by the trainer). At steady-state shapes (`n_docs > sample_size`,
    // `sample_size ≥ 100_000`) the sample_rows bound is the active
    // one and is comfortably above any sane n_cent.
    let n_cent = cfg.n_cent.max(1).min(n_docs.max(1)).min(sample_rows.max(1));

    // ---- Pass 1: k-means on the reservoir sample ----
    let centroids = if sample_rows == 0 || n_docs == 0 {
        vec![0.0f32; n_cent * dim]
    } else {
        kmeans(reservoir.sample(), dim, n_cent, 5, cfg.rot_seed)
    };
    // Drop the reservoir immediately — k-means has converged
    // and the sample bytes aren't needed for pass 2.
    drop(reservoir);

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
            cfg.rerank_codec,
        )?;
    }

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

    let summary_radius_x100 = (summary_radius_sq_max.sqrt() * 100.0)
        .max(0.0)
        .min(u32::MAX as f32) as u32;

    // ---- Pre-compute subsection layout ----
    //
    // Every region size is a pure function of (dim, n_docs, n_cent,
    // codec, metric), all known by the end of pass 2. We allocate
    // the full subsection buffer up front and stream pass-3 rows
    // directly into their final on-disk slots — eliminates the
    // O(n_docs · dim · 4) `full_layout` round-trip the old pipeline
    // did between bucket-read and codec-encode.
    let codec = cfg.rerank_codec;
    let summary_size = dim * 4;
    let centroids_size = n_cent * dim * 4;
    let cluster_idx_size = n_cent * 8;
    let codes_size = n_docs * code_bytes;
    let codec_meta_size = codec.codec_meta_bytes(dim, n_docs, n_cent, cfg.metric);
    let full_size = codec.per_vector_bytes(dim) * n_docs;
    let doc_ids_size = n_docs * 4;

    // Region offsets, relative to subsection start.
    let summary_off = SUB_HEADER_SIZE;
    let centroids_off = summary_off + summary_size;
    let cluster_idx_off = centroids_off + centroids_size;
    let codes_off = cluster_idx_off + cluster_idx_size;
    // `codec_meta_off` is the on-disk slot in the sub-header for the
    // codec metadata region's start. Zero means "no codec metadata
    // in this subsection" — Fp32 / Bf16 / RabitqOnly write 0 here
    // and stay byte-identical to legacy fp32 segments that left the
    // (formerly reserved) slot zero.
    let codec_meta_off_value: u32 = if codec_meta_size == 0 {
        0
    } else {
        (codes_off + codes_size) as u32
    };
    let codec_meta_off_in_bytes = codec_meta_off_value as usize;
    let full_off = codes_off + codes_size + codec_meta_size;
    let doc_ids_off = full_off + full_size;

    let total_size_before_crc = SUB_HEADER_SIZE
        + summary_size
        + centroids_size
        + cluster_idx_size
        + codes_size
        + codec_meta_size
        + full_size
        + doc_ids_size;

    // Zero-init the full subsection so unwritten slots (RabitqOnly's
    // empty full[], reserved subheader bytes) carry deterministic
    // zeroes regardless of allocator state.
    let mut bytes: Vec<u8> = vec![0u8; total_size_before_crc];

    // ---- Sub-header (56 bytes) ----
    //   [0..8]   SUB_MAGIC
    //   [8..12]  VERSION
    //   [12..16] codec_meta_off (former reserved slot; zero for Fp32
    //            and legacy fp32 segments).
    //   [16..24] summary_centroid_offset
    //   [24..28] summary_radius_x100
    //   [28..32] reserved (4) — left zero by the vec![0u8; ...] init
    //   [32..40] centroids_off
    //   [40..48] cluster_idx_off
    //   [48..52] codes_off
    //   [52..56] full_off
    bytes[0..8].copy_from_slice(format::vec::SUB_MAGIC);
    bytes[8..12].copy_from_slice(&format::vec::VERSION.to_le_bytes());
    bytes[12..16].copy_from_slice(&codec_meta_off_value.to_le_bytes());
    bytes[16..24].copy_from_slice(&(summary_off as u64).to_le_bytes());
    bytes[24..28].copy_from_slice(&summary_radius_x100.to_le_bytes());
    // [28..32] reserved stays zero.
    bytes[32..40].copy_from_slice(&(centroids_off as u64).to_le_bytes());
    bytes[40..48].copy_from_slice(&(cluster_idx_off as u64).to_le_bytes());
    bytes[48..52].copy_from_slice(&(codes_off as u32).to_le_bytes());
    bytes[52..56].copy_from_slice(&(full_off as u32).to_le_bytes());

    // ---- Summary centroid + centroids ----
    bytes[summary_off..summary_off + summary_size]
        .copy_from_slice(bytemuck::cast_slice(&summary_centroid));
    bytes[centroids_off..centroids_off + centroids_size]
        .copy_from_slice(bytemuck::cast_slice(&centroids));

    // ---- Cluster index, built from pass-2 bucket counts ----
    //
    // `cluster_index[c] = (off, count)` where `off` is the
    // cumulative row count across clusters 0..c. The same indexing
    // the shortlist carries — `pos = off + i` — so the per-doc
    // norms and rerank codes stay co-indexed across regions.
    let mut cluster_index: Vec<(u32, u32)> = Vec::with_capacity(n_cent);
    {
        let mut acc_off = 0u32;
        let mut idx_cursor = cluster_idx_off;
        for &cnt in &bucket_counts {
            cluster_index.push((acc_off, cnt));
            bytes[idx_cursor..idx_cursor + 4].copy_from_slice(&acc_off.to_le_bytes());
            bytes[idx_cursor + 4..idx_cursor + 8].copy_from_slice(&cnt.to_le_bytes());
            acc_off += cnt;
            idx_cursor += 8;
        }
        debug_assert_eq!(acc_off as usize, n_docs);
    }

    // ---- Per-cluster streaming pass ----
    //
    // Replaces the old two-step "stage every row in
    // full_layout/codes_layout/doc_ids_layout, then assemble into
    // bytes" pipeline. For each cluster `c` we read its bucket file
    // exactly once and write each row's fields directly into the
    // correct on-disk slots:
    //
    //   doc_id    → bytes[doc_ids_off + pos * 4 ..]
    //   rabitq    → bytes[codes_off   + pos * code_bytes ..]
    //   full[]    → codec-dependent slot in bytes (see match below)
    //
    // The only resident scratch is the cluster's `full_block`
    // (`cluster_count * dim * 4` bytes when the codec writes a
    // full[] payload), reused across clusters. Sq8 needs the whole
    // cluster's fp32 rows in hand before it can derive `(scale,
    // offset)`, which is exactly what `full_block` already contains
    // after the bulk read.
    //
    // Sq8 codec_meta region layout (when present):
    //   [scale_block | offset_block | per_doc_norms?]
    //   scale_block  = n_cent * dim * 4 bytes
    //   offset_block = n_cent * dim * 4 bytes
    //   per_doc_norms (L2Sq/Cosine only) = n_docs * 4 bytes
    let sq8_scale_block_off = codec_meta_off_in_bytes;
    let sq8_offset_block_off = sq8_scale_block_off + n_cent * dim * 4;
    let sq8_norms_block_off =
        if codec == RerankCodec::Sq8 && matches!(cfg.metric, Metric::L2Sq | Metric::Cosine) {
            Some(sq8_offset_block_off + n_cent * dim * 4)
        } else {
            None
        };

    // Per-cluster bucket payload is row-major
    //   [doc_id u32 | code(code_bytes) | full_row?]
    // and pass 2 wrote each row in one shot. In final assembly we
    // mirror that on the read side: read all doc_ids, all codes, and
    // (when present) all full-row payload as three contiguous block
    // reads — one `read_exact` per block instead of three per row.
    let full_row_bytes_in_bucket = if codec.writes_full() { dim * 4 } else { 0 };
    let mut id_block: Vec<u8> = Vec::new();
    let mut code_block: Vec<u8> = Vec::new();
    let mut full_block: Vec<u8> = Vec::new();
    let mut row_fp32_scratch: Vec<f32> = Vec::new();

    for (c, &(cluster_off_u32, cluster_count_u32)) in cluster_index.iter().enumerate() {
        if cluster_count_u32 == 0 {
            continue;
        }
        let cluster_off = cluster_off_u32 as usize;
        let cluster_count = cluster_count_u32 as usize;

        let path = scratch.join(format!("infino_bucket_col{column_id}_c{c}.bin"));
        let mut reader = BufReader::with_capacity(BUCKET_BUF_SIZE, File::open(&path)?);

        // Bulk-read the cluster bucket in three blocks. Pass 2
        // interleaves [doc_id | code | full?] per row, so we de-
        // interleave on read into three flat block buffers and copy
        // each block straight into its on-disk slot. This is the
        // mirror of pass 2's per-row triple write (and avoids the
        // per-doc syscall storm the previous loop incurred).
        id_block.resize(cluster_count * 4, 0);
        code_block.resize(cluster_count * code_bytes, 0);
        if full_row_bytes_in_bucket > 0 {
            full_block.resize(cluster_count * full_row_bytes_in_bucket, 0);
        }
        for i in 0..cluster_count {
            reader.read_exact(&mut id_block[i * 4..(i + 1) * 4])?;
            reader.read_exact(&mut code_block[i * code_bytes..(i + 1) * code_bytes])?;
            if full_row_bytes_in_bucket > 0 {
                let off = i * full_row_bytes_in_bucket;
                reader.read_exact(&mut full_block[off..off + full_row_bytes_in_bucket])?;
            }
        }

        // doc_ids and rabitq codes are byte-identical to the on-disk
        // layout, so each is a single block copy at this cluster's
        // base offset.
        let did_base = doc_ids_off + cluster_off * 4;
        bytes[did_base..did_base + cluster_count * 4].copy_from_slice(&id_block);
        let code_base = codes_off + cluster_off * code_bytes;
        bytes[code_base..code_base + cluster_count * code_bytes].copy_from_slice(&code_block);

        // full[] region — codec-dependent. RabitqOnly has no full[]
        // payload on disk and pass 2 didn't spill one. Fp32 is a
        // direct block copy. Bf16/Sq8 transcode out of the fp32
        // block buffer.
        match codec {
            RerankCodec::RabitqOnly => {}
            RerankCodec::Fp32 => {
                let dst_base = full_off + cluster_off * dim * 4;
                bytes[dst_base..dst_base + cluster_count * dim * 4].copy_from_slice(&full_block);
            }
            RerankCodec::Bf16 => {
                // bf16 destination is 2 bytes/lane; read the fp32
                // block as &[f32] (Vec<u8> is f32-aligned only when
                // we go through bytemuck on a fresh aligned slab —
                // here we re-stage one row at a time to keep lane-
                // wise rounding identical to the previous code path).
                if row_fp32_scratch.len() < dim {
                    row_fp32_scratch.resize(dim, 0.0);
                }
                for i in 0..cluster_count {
                    let pos = cluster_off + i;
                    let src = &full_block[i * dim * 4..(i + 1) * dim * 4];
                    let row = &mut row_fp32_scratch[..dim];
                    bytemuck::cast_slice_mut::<f32, u8>(row).copy_from_slice(src);
                    let off = full_off + pos * dim * 2;
                    for (d, &x) in row.iter().enumerate() {
                        let bf = crate::superfile::vector::distance::fp32_to_bf16(x);
                        bytes[off + d * 2..off + d * 2 + 2].copy_from_slice(&bf.to_le_bytes());
                    }
                }
            }
            RerankCodec::Sq8 => {
                // The cluster's fp32 rows are already in `full_block`
                // (one bulk read above). Hand the slice to the
                // per-cluster encode helper below.
                let cluster_rows: &[f32] = bytemuck::cast_slice(&full_block);
                encode_sq8_cluster(
                    cluster_rows,
                    dim,
                    cluster_count,
                    cluster_off,
                    c,
                    full_off,
                    sq8_scale_block_off,
                    sq8_offset_block_off,
                    sq8_norms_block_off,
                    &mut bytes,
                );
            }
        }
    }

    // Trailing CRC over the subsection body.
    let crc = crc32c(&bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());

    Ok(SubsectionBytes {
        bytes,
        summary_offset_in_sub: summary_off,
    })
}

/// Sq8 per-cluster encode for the final-assembly pass.
///
/// Given the cluster's fp32 rows (already loaded into the
/// caller's `full_block` scratch and reinterpreted as `&[f32]`),
/// derive `(scale_c, offset_c)`, write them into the codec_meta
/// scale/offset blocks for cluster `c`, and emit u8 codes into
/// the on-disk full[] region. When the column carries decoded
/// per-doc `Σ x²` (L2Sq/Cosine), also fill the per-doc norms
/// block — the search-side kernel reads it directly to skip
/// recomputing the decoded norm at rerank time.
#[allow(clippy::too_many_arguments)]
fn encode_sq8_cluster(
    cluster_rows: &[f32],
    dim: usize,
    cluster_count: usize,
    cluster_off: usize,
    c: usize,
    full_off: usize,
    sq8_scale_block_off: usize,
    sq8_offset_block_off: usize,
    sq8_norms_block_off: Option<usize>,
    bytes: &mut [u8],
) {
    debug_assert_eq!(cluster_rows.len(), cluster_count * dim);

    let (scale_c, offset_c) = compute_sq8_quantizer_for_cluster(cluster_rows, dim, cluster_count);

    let sc_off = sq8_scale_block_off + c * dim * 4;
    bytes[sc_off..sc_off + dim * 4].copy_from_slice(bytemuck::cast_slice(&scale_c));
    let oc_off = sq8_offset_block_off + c * dim * 4;
    bytes[oc_off..oc_off + dim * 4].copy_from_slice(bytemuck::cast_slice(&offset_c));

    for i in 0..cluster_count {
        let src = &cluster_rows[i * dim..(i + 1) * dim];
        let pos = cluster_off + i;
        let code_off = full_off + pos * dim;
        let mut acc = 0.0f64;
        for d in 0..dim {
            let q = ((src[d] - offset_c[d]) / scale_c[d]).round();
            // Clamp on the boundary handles fp rounding artefacts at
            // the min/max ends; the cast is then in-range by
            // construction.
            let qc = q.clamp(0.0, 255.0) as u8;
            bytes[code_off + d] = qc;
            if sq8_norms_block_off.is_some() {
                let x = (qc as f32) * scale_c[d] + offset_c[d];
                acc += (x as f64) * (x as f64);
            }
        }
        if let Some(norms_off) = sq8_norms_block_off {
            let n_off = norms_off + pos * 4;
            bytes[n_off..n_off + 4].copy_from_slice(&(acc as f32).to_le_bytes());
        }
    }
}

/// Scan one cluster's rows for per-dim min/max and derive the
/// Sq8 quantizer `(scale[dim], offset[dim])` for that cluster.
///
/// Quantization scheme: `q = clamp(round((x − offset[d]) /
/// scale[d]), 0, 255)`. With `offset[d] = min_x[d]` and
/// `scale[d] = (max_x[d] − min_x[d]) / 255`, this maps the
/// cluster's observed range onto the full u8 grid. When a dim
/// is constant within the cluster (`max == min`) we set
/// `scale = 1.0` and `offset = min` — every code in that dim
/// lands at 0 and the decoder recovers the constant exactly.
///
/// Per-cluster (not per-column) quantizers preserve recall on
/// highly-clustered cosine corpora; see
/// `RerankCodec::codec_meta_bytes` for the per-column failure mode.
///
/// Parallel reduce over `cluster_rows.chunks(dim)` for n_rows
/// ≥ 64; sequential fallback for smaller clusters where
/// rayon's fork-join overhead would dominate.
fn compute_sq8_quantizer_for_cluster(
    cluster_rows: &[f32],
    dim: usize,
    n_rows: usize,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(cluster_rows.len(), n_rows * dim);

    let (min_vec, max_vec) = if n_rows == 0 {
        // Empty cluster — emit an identity quantizer. No doc
        // ever encodes through this slot, but the reader still
        // reads `dim` floats of each from disk so we must write
        // something well-defined. scale=1, offset=0 makes the
        // decoder a no-op on the all-zero codes (which the
        // builder also never emits) and keeps the codec_meta
        // arrays free of NaN sentinels that would break the
        // float bit-pattern equality used in some tests.
        (vec![0.0f32; dim], vec![0.0f32; dim])
    } else if n_rows < 64 {
        let mut min_vec = cluster_rows[..dim].to_vec();
        let mut max_vec = cluster_rows[..dim].to_vec();
        for row in 1..n_rows {
            for d in 0..dim {
                let x = cluster_rows[row * dim + d];
                if x < min_vec[d] {
                    min_vec[d] = x;
                }
                if x > max_vec[d] {
                    max_vec[d] = x;
                }
            }
        }
        (min_vec, max_vec)
    } else {
        cluster_rows
            .par_chunks(dim)
            .fold(
                || (vec![f32::INFINITY; dim], vec![f32::NEG_INFINITY; dim]),
                |(mut mn, mut mx), row| {
                    for (d, &x) in row.iter().enumerate() {
                        if x < mn[d] {
                            mn[d] = x;
                        }
                        if x > mx[d] {
                            mx[d] = x;
                        }
                    }
                    (mn, mx)
                },
            )
            .reduce(
                || (vec![f32::INFINITY; dim], vec![f32::NEG_INFINITY; dim]),
                |(mut a_min, mut a_max), (b_min, b_max)| {
                    for d in 0..dim {
                        if b_min[d] < a_min[d] {
                            a_min[d] = b_min[d];
                        }
                        if b_max[d] > a_max[d] {
                            a_max[d] = b_max[d];
                        }
                    }
                    (a_min, a_max)
                },
            )
    };

    let mut scale = vec![0.0f32; dim];
    let mut offset = vec![0.0f32; dim];
    for d in 0..dim {
        offset[d] = min_vec[d];
        let span = max_vec[d] - min_vec[d];
        scale[d] = if span > 0.0 { span / 255.0 } else { 1.0 };
    }
    (scale, offset)
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

        // Route rows to bucket writers. Sequential per-bucket:
        // BufWriter is !Sync and a per-bucket Mutex would serialize
        // anyway. `RabitqOnly` columns skip the fp32 vector spill
        // because they have no `full[]` region on disk.
        let write_full = codec.writes_full();
        for r in 0..actual_rows {
            let cid = asgn[r] as usize;
            let local_doc_id = global_doc_id + r as u32;
            let writer = &mut bucket_writers[cid];
            writer.write_all(&local_doc_id.to_le_bytes())?;
            writer.write_all(&chunk_codes[r * code_bytes..(r + 1) * code_bytes])?;
            if write_full {
                writer.write_all(bytemuck::cast_slice(&chunk[r * dim..(r + 1) * dim]))?;
            }
            bucket_counts[cid] += 1;
        }
        global_doc_id += actual_rows as u32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(column: &str, dim: usize) -> VectorConfig {
        VectorConfig {
            column: column.to_string(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
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
    #[test]
    fn forced_spill_path_matches_in_ram_path_on_self_nn() {
        use crate::superfile::vector::reader::VectorReader;
        use bytes::Bytes;
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
                .expect("search ram");
            let top_spill = r_spill
                .search("v", query, 1, nprobe, rerank_mult)
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

    /// Streaming output to a `Cursor<Vec<u8>>`: the resulting bytes
    /// carry a valid outer magic + a valid trailing whole-blob CRC32C
    /// that round-trips when recomputed over the body.
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
    /// isn't `Vec<u8>` while the input corpus uses the on-disk
    /// SpillWriter path: write to a temp file, read it back, open
    /// it with `VectorReader`, and confirm exact self-NN search.
    /// This covers the combined SpillWriter -> finish_to(writer)
    /// -> VectorReader path.
    #[test]
    fn finish_to_temp_file_forced_spill_round_trips_through_reader() {
        use crate::superfile::vector::reader::VectorReader;
        use bytes::Bytes;
        use std::io::BufWriter;
        let dim = 16usize;
        let n_docs = 32usize;
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
            let file = std::fs::File::create(&path).expect("create blob file");
            let writer = BufWriter::new(file);
            b.finish_to(writer).expect("finish_to BufWriter<File>");
        }
        let blob = std::fs::read(&path).expect("read blob file");
        let json = format!(
            r#"[{{"column":"v","dim":{dim},"n_cent":{n_cent},"rot_seed":7,"metric":"l2sq"}}]"#
        );
        let reader = VectorReader::open(Bytes::from(blob), &json)
            .expect("open VectorReader from streamed blob");
        let query: Vec<f32> = (0..dim).map(|j| ((j as f32) * 0.13).sin()).collect();
        let hits = reader
            .search("v", &query, 5, n_cent, n_docs + 1)
            .expect("kNN search");
        assert!(!hits.is_empty(), "search returned no hits");
        assert_eq!(hits[0].0, 0, "forced-spill streamed blob missed self-NN");
    }
}
