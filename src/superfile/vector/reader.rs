// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Vector blob reader. Multi-column kNN search via IVF + 1-bit RaBitQ
//! shortlist + full-precision rerank.
//!
//! Opens the unified-blob byte layout produced by
//! [`super::builder::VectorBuilder::finish`] and exposes per-column
//! kNN search.
//!
//! Self-contained: owns its `Bytes`. Per-column metadata is parsed
//! eagerly at `open()`; per-query work happens on demand.

use crate::superfile::format::checksum::crc32c;
use crate::superfile::format::{self};
pub(crate) use crate::superfile::lazy_source::Source;
use crate::superfile::lazy_source::{LazyByteSource, PrefetchedSource};
use crate::superfile::vector::distance::{
    Metric, SQ8_RESIDUAL_DIVISOR, Sq8Kernel, Sq8ResidualKernel, distance_bytes,
    distance_bytes_codec,
};
use crate::superfile::vector::quant::BitQuantizer;
use crate::superfile::vector::rerank_codec::RerankCodec;
use crate::superfile::vector::rotation::RandomRotation;
use crate::superfile::{ReadError, error::VectorError};
use bytes::Bytes;
use rayon::prelude::*;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use crate::superfile::format::vec::{
    CLUSTER_IDX_COUNT_OFFSET, CLUSTER_IDX_ENTRY_BYTES, MAGIC_BYTES, U32_BYTES, U64_BYTES,
    dir_entry, outer_hdr, sub_hdr,
};

const OUTER_HEADER_SIZE: usize = format::vec::OUTER_HEADER_SIZE;
const DIR_ENTRY_SIZE: usize = format::vec::DIR_ENTRY_SIZE;
const SUB_HEADER_SIZE: usize = format::vec::SUB_HEADER_SIZE;

/// Fixed-point scale for the per-subsection summary radius. The
/// builder stores `round(radius × 100)` in a `u32`; the reader
/// recovers the radius by dividing by this. Must match
/// `builder::SUMMARY_RADIUS_SCALE`.
const SUMMARY_RADIUS_SCALE: f32 = 100.0;

/// Shortlist multiplier for the Sq8Residual refine pass. After the
/// first-pass Sq8 scan, only the top `SQ8_RESIDUAL_REFINE_MULT × k`
/// survivors are re-scored with the more expensive residual leg.
const SQ8_RESIDUAL_REFINE_MULT: usize = 2;

/// JSON-deserialized form of one entry in `inf.vec.columns`. The KV
/// value is a JSON array of these in declaration order.
#[derive(Debug, Clone, Deserialize)]
pub struct VectorColumnConfig {
    pub column: String,
    pub dim: usize,
    pub n_cent: usize,
    pub rot_seed: u64,
    /// `"l2sq"`, `"cosine"`, or `"negdot"`.
    pub metric: String,
}

#[derive(Debug, Clone)]
pub(super) enum Sq8ColumnMeta {
    Eager {
        scale: Vec<f32>,
        offset: Vec<f32>,
        per_doc_norms: Option<Arc<[f32]>>,
    },
    Lazy {
        scale_abs_off: usize,
        offset_abs_off: usize,
        norms_abs_off: Option<usize>,
    },
}

#[derive(Debug)]
struct Sq8ParsedMeta {
    scale: Vec<f32>,
    offset: Vec<f32>,
    per_doc_norms: Option<Arc<[f32]>>,
}

/// Per-column reader state; cached at open time.
#[derive(Debug)]
pub struct ColumnReader {
    pub name: String,
    pub dim: usize,
    pub n_cent: u32,
    pub n_docs: u32,
    pub metric: Metric,
    pub rot_seed: u64,
    /// — on-disk rerank codec for this column. Today
    /// admits Fp32, Sq8, and RabitqOnly; the parser rejects
    /// every other codec at open time with a `MalformedVersion`
    /// until support for it is added (the `None` codec is not yet
    /// implemented).
    pub rerank_codec: RerankCodec,
    /// `Sq8`-only quantizer metadata, materialised
    /// at open time from the `codec_meta` region. `None` for
    /// every other codec (Fp32 / RabitqOnly). At dim=384 the
    /// scale + offset arrays are 3 KB total; for L2Sq columns
    /// the per-doc norms add `n_docs × 4` bytes (4 MB at 1M
    /// docs / column). Materialising here amortizes the parse
    /// across every search call.
    pub(super) sq8_meta: Option<Sq8ColumnMeta>,
    lazy_sq8_parsed: OnceLock<Arc<Sq8ParsedMeta>>,
    /// Byte range of this column's subsection within the outer blob.
    subsection_range: Range<usize>,
    /// Offsets relative to the subsection start.
    summary_off: usize,
    summary_radius: f32,
    centroids_off: usize,
    cluster_idx_off: usize,
    /// relative offset of the per-column
    /// `codec_meta` region inside the subsection. `0` means
    /// "no codec_meta" (Fp32 / RabitqOnly); non-zero is only
    /// produced by codecs whose `codec_meta_bytes(...) > 0`
    /// (`Sq8` is the only one today). In the current layout
    /// `codec_meta` sits between `cluster_idx` and the
    /// per-cluster blocks (inside the open-time region).
    #[allow(dead_code)]
    codec_meta_off: usize,
    /// Relative offset of the per-cluster blocks region. Each
    /// cluster `c` lives at
    /// `per_cluster_blocks_off + doc_off[c] * stride` for
    /// `count[c] * stride` bytes, where `stride = code_bytes + 4
    /// + per_vec_bytes`, formatted as `[codes_chunk:
    /// count*code_bytes][doc_ids_chunk: count*4][full_chunk:
    /// count*per_vec_bytes]`. The full-precision rerank vectors
    /// are interleaved into each block (no separate `full[]`
    /// region) so one range GET per probed cluster covers the
    /// estimate codes, doc-ids, and rerank vectors together.
    per_cluster_blocks_off: usize,
    quant: BitQuantizer,
    /// Cached random rotation built once at open from `(dim, rot_seed)`.
    /// Construction is `O(dim³)` for Gram-Schmidt — at dim=384 that's
    /// ~7.9 ms, dominant over every other per-query stage if rebuilt
    /// per `search()`. Build once, reuse forever.
    rot: RandomRotation,
}

impl ColumnReader {
    /// byte range covering one cluster's
    /// `[codes_chunk + doc_ids_chunk]` block as a single
    /// contiguous span. Pulled in **one** range fetch per
    /// probed cluster; the cold-first-search budget collapses
    /// to `nprobe + 1` range GETs (nprobe cluster blocks + 1
    /// rerank run) on a freshly-opened lazy reader, down from
    /// `2 × nprobe + 1` on the older split-range path.
    ///
    /// Block layout: each cluster's block is
    /// `count * (code_bytes + 4)` bytes formatted as
    /// `[codes: count*code_bytes][doc_ids: count*4]`. The
    /// per-cluster `(doc_off, count)` entry recorded in
    /// `cluster_idx` addresses both halves with no extra
    /// lookup: byte offset = `per_cluster_blocks_off +
    /// doc_off * (code_bytes + 4)`.
    /// Full per-cluster block range `[codes][doc_ids][full]`. The
    /// production search now fetches only the codes+doc_ids prefix
    /// (`cluster_codes_doc_ids_range`) plus survivor `full[]` rows
    /// (`cluster_rerank_row_range`), so this whole-block range is
    /// retained for the layout-invariant test that pins the on-disk
    /// shape.
    pub(super) fn cluster_block_range(
        &self,
        cluster_doc_off: u32,
        cluster_count: u32,
    ) -> Range<usize> {
        let sub_start = self.subsection_range.start;
        let stride = self.per_cluster_doc_stride();
        let start = sub_start + self.per_cluster_blocks_off + (cluster_doc_off as usize) * stride;
        let len = (cluster_count as usize) * stride;
        start..start + len
    }

    pub(super) fn cluster_codes_doc_ids_range(
        &self,
        cluster_doc_off: u32,
        cluster_count: u32,
    ) -> Range<usize> {
        let sub_start = self.subsection_range.start;
        let start = sub_start
            + self.per_cluster_blocks_off
            + (cluster_doc_off as usize) * self.per_cluster_doc_stride();
        let len = (cluster_count as usize) * (self.quant.code_bytes() + format::vec::DOC_ID_BYTES);
        start..start + len
    }

    pub(super) fn cluster_rerank_row_range(
        &self,
        cluster_doc_off: u32,
        cluster_count: u32,
        local_idx: usize,
    ) -> Range<usize> {
        let sub_start = self.subsection_range.start;
        let block_start = sub_start
            + self.per_cluster_blocks_off
            + (cluster_doc_off as usize) * self.per_cluster_doc_stride();
        let prefix_len =
            (cluster_count as usize) * (self.quant.code_bytes() + format::vec::DOC_ID_BYTES);
        let start =
            block_start + prefix_len + local_idx * self.rerank_codec.per_vector_bytes(self.dim);
        start..start + self.rerank_codec.per_vector_bytes(self.dim)
    }

    /// Per-doc byte stride inside a cluster block:
    /// `code_bytes + 4 (doc_id) + per_vec_bytes (full rerank)`.
    /// A cluster's block packs `cnt` docs at this stride as
    /// `[codes_chunk][doc_ids_chunk][full_chunk]`.
    pub(super) fn per_cluster_doc_stride(&self) -> usize {
        self.quant.code_bytes()
            + format::vec::DOC_ID_BYTES
            + self.rerank_codec.per_vector_bytes(self.dim)
    }
}

/// Per-open knobs for [`VectorReader::open_with`] and
/// [`VectorReader::open_lazy`]. `Default` is the safe choice
/// (CRC verification on). The argumentless [`VectorReader::open`]
/// takes the default; the lazy path uses
/// [`Self::for_object_store`] which turns CRC off (a full-blob
/// scan would defeat the cold-open byte budget).
///
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Verify the per-subsection CRC during open. Each subsection is
    /// scanned in full (≈1.5 GiB at 1M × 384, single column), so this
    /// dominates cold-open time when on. Defaults to `true`; the
    /// argumentless [`VectorReader::open`] uses this default.
    /// Flip to `false` when storage is already trusted (content-
    /// addressed object store, checksummed filesystem) to skip
    /// the scan.
    pub verify_crc: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { verify_crc: true }
    }
}

impl OpenOptions {
    /// defaults tuned for an object-store-backed
    /// `Source::Lazy` open: `verify_crc = false` (a full-blob
    /// scan would defeat every cold-open byte-budget number in
    /// the plan; deployments that need CRC verification opt
    /// back in and accept the cost).
    pub fn for_object_store() -> Self {
        Self { verify_crc: false }
    }
}

/// Multi-column vector blob reader. `Send + Sync`; concurrent
/// searches share the underlying [`Source`] (refcount-shared via
/// `Bytes` / `Arc<dyn LazyByteSource>`).
#[derive(Debug)]
pub struct VectorReader {
    source: Source,
    n_docs: u64,
    columns: Vec<ColumnReader>,
    column_id_by_name: HashMap<String, u32>,
}

impl VectorReader {
    /// Open the reader. `columns_json` is the value of the
    /// `inf.vec.columns` Parquet KV key (a JSON array of
    /// [`VectorColumnConfig`]).
    /// Open the reader with default options (CRC verification on).
    pub fn open(blob: Bytes, columns_json: &str) -> Result<Self, VectorError> {
        Self::open_with(blob, columns_json, OpenOptions::default())
    }

    /// Open with explicit options. The fast path is
    /// `OpenOptions { verify_crc: false }` which skips both the
    /// outer (whole-blob) CRC and the per-subsection CRC scans —
    /// at 1M × 384 cold open drops from ~132 ms to ~2 ms. Use this
    /// when the underlying storage is trusted (e.g. local disk after
    /// a successful file integrity check) or when CRC verification
    /// is performed elsewhere in the stack.
    pub fn open_with(
        blob: Bytes,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, VectorError> {
        // every byte fetch routes through `Source::try_get_range_sync`
        // so a future lazy variant can intercept the same call sites
        // without a second rewrite. `InMemory` returns zero-copy
        // `Bytes::slice` views; refcount bumps only.
        Self::open_with_source(Source::InMemory(blob), columns_json, opts)
    }

    /// Async open against a [`LazyByteSource`].
    ///
    /// The lazy open path fetches exactly the bytes the structural
    /// decode reads:
    ///   - outer header (`32 B`);
    ///   - directory + directory CRC;
    ///   - each subsection header (`56 B`);
    ///   - Sq8 `codec_meta` only (scale/offset/norm tables).
    ///
    /// Centroids, cluster indexes, and per-cluster blocks are search-time
    /// data, not open-time data. They stay lazy so cold open is governed
    /// by metadata bytes and serial dependency depth instead of a large
    /// speculative slab.
    ///
    /// `opts.verify_crc = true` is honored, but it forces every
    /// subsection to be fetched in full and defeats the cold-open
    /// cold-open byte budget — only set it when the
    /// underlying storage is untrusted and CRC verification is
    /// load-bearing. The convenience constructor
    /// [`OpenOptions::for_object_store`] sets it to `false`
    /// (the load-bearing default discussed in the plan's
    /// "verify_crc trade-off" section).
    pub async fn open_lazy(
        source: Arc<dyn LazyByteSource>,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, VectorError> {
        let blob_size = source.size() as usize;
        if blob_size < OUTER_HEADER_SIZE + 4 {
            return Err(VectorError::Read(ReadError::MissingKv(
                "vector blob header",
            )));
        }

        let header_bytes = source
            .range(0, OUTER_HEADER_SIZE as u64)
            .await
            .map_err(|e| {
                VectorError::Read(ReadError::MalformedVersion(format!(
                    "lazy open: outer header fetch: {e}"
                )))
            })?;
        if &header_bytes[0..MAGIC_BYTES] != format::vec::OUTER_MAGIC {
            return Err(VectorError::Read(ReadError::BadMagic {
                section: "vector",
                expected: format::vec::OUTER_MAGIC,
                actual: header_bytes[0..MAGIC_BYTES].to_vec(),
            }));
        }
        let version =
            read_u32_le(&header_bytes[outer_hdr::VERSION_OFF..outer_hdr::VERSION_OFF + U32_BYTES]);
        if version != format::vec::VERSION {
            return Err(VectorError::Read(ReadError::UnsupportedVersion(format!(
                "vector blob version {version}"
            ))));
        }
        let n_columns = read_u32_le(
            &header_bytes[outer_hdr::N_COLUMNS_OFF..outer_hdr::N_COLUMNS_OFF + U32_BYTES],
        ) as usize;
        let dir_offset = read_u64_le(
            &header_bytes[outer_hdr::DIR_OFFSET_OFF..outer_hdr::DIR_OFFSET_OFF + U64_BYTES],
        ) as usize;
        let dir_size = n_columns * DIR_ENTRY_SIZE;
        let dir_end = dir_offset + dir_size + format::CRC_BYTES;
        if dir_end > blob_size {
            return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                "lazy open: directory end {dir_end} exceeds blob size {blob_size}",
            ))));
        }

        let dir_prefetch = source
            .range(dir_offset as u64, (dir_end - dir_offset) as u64)
            .await
            .map_err(|e| {
                VectorError::Read(ReadError::MalformedVersion(format!(
                    "lazy open: directory fetch: {e}"
                )))
            })?;

        // Validate directory CRC against the prefetched bytes
        // before walking subsection metadata. A directory-CRC
        // mismatch on the lazy path is the closest thing we
        // have to an end-to-end integrity check when
        // `verify_crc = false`.
        let dir_bytes_slice = &dir_prefetch[0..dir_size];
        let dir_crc_expected = read_u32_le(&dir_prefetch[dir_size..dir_size + format::CRC_BYTES]);
        let dir_crc_actual = crc32c(dir_bytes_slice);
        if dir_crc_expected != dir_crc_actual {
            return Err(VectorError::Read(ReadError::ChecksumMismatch {
                section: "vector/directory",
                column: String::new(),
            }));
        }

        // Stage the overlay with the exact header and directory bytes.
        let mut overlay = PrefetchedSource::new(Arc::clone(&source));
        overlay.install(0, header_bytes.clone());
        overlay.install(dir_offset as u64, dir_prefetch.clone());

        let mut subsection_meta = Vec::with_capacity(n_columns);
        let mut subheader_ranges = Vec::with_capacity(n_columns);
        for i in 0..n_columns {
            let entry_off = i * DIR_ENTRY_SIZE;
            let subsection_off = read_u64_le(
                &dir_bytes_slice[entry_off + dir_entry::SUBSECTION_OFF_OFF
                    ..entry_off + dir_entry::SUBSECTION_OFF_OFF + U64_BYTES],
            ) as usize;
            let subsection_len = read_u64_le(
                &dir_bytes_slice[entry_off + dir_entry::SUBSECTION_LEN_OFF
                    ..entry_off + dir_entry::SUBSECTION_LEN_OFF + U64_BYTES],
            ) as usize;
            let dir_codec_meta_off = read_u32_le(
                &dir_bytes_slice[entry_off + dir_entry::CODEC_META_OFF_OFF
                    ..entry_off + dir_entry::CODEC_META_OFF_OFF + U32_BYTES],
            ) as usize;
            let dir_codec_meta_size = read_u32_le(
                &dir_bytes_slice[entry_off + dir_entry::CODEC_META_SIZE_OFF
                    ..entry_off + dir_entry::CODEC_META_SIZE_OFF + U32_BYTES],
            ) as usize;
            if subsection_len < SUB_HEADER_SIZE + format::CRC_BYTES {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} too short ({subsection_len} bytes)"
                ))));
            }
            let sub_end = subsection_off + subsection_len;
            if sub_end > blob_size {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} runs past blob",
                ))));
            }
            if dir_codec_meta_size > 0 {
                let meta_end = dir_codec_meta_off + dir_codec_meta_size;
                if dir_codec_meta_off < SUB_HEADER_SIZE || meta_end > subsection_len - 4 {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "subsection {i} directory codec_meta range [{dir_codec_meta_off}..\
                         {meta_end}) outside subsection body length {}",
                        subsection_len - 4
                    ))));
                }
            }
            subsection_meta.push((
                i,
                entry_off,
                subsection_off,
                subsection_len,
                sub_end,
                dir_codec_meta_off,
                dir_codec_meta_size,
            ));
            subheader_ranges.push((i, subsection_off));
        }

        let subheaders_fut =
            futures::future::try_join_all(subheader_ranges.iter().map(|&(i, subsection_off)| {
                let source = Arc::clone(&source);
                async move {
                    let bytes = source
                        .range(subsection_off as u64, SUB_HEADER_SIZE as u64)
                        .await
                        .map_err(|e| {
                            VectorError::Read(ReadError::MalformedVersion(format!(
                                "lazy open: subsection {i} sub-header fetch: {e}"
                            )))
                        })?;
                    Ok::<_, VectorError>((i, subsection_off, bytes))
                }
            }));
        let subheaders = subheaders_fut.await?;

        for (i, subsection_off, sub_header) in subheaders {
            if &sub_header[0..MAGIC_BYTES] != format::vec::SUB_MAGIC {
                return Err(VectorError::Read(ReadError::BadMagic {
                    section: "vector/subsection",
                    expected: format::vec::SUB_MAGIC,
                    actual: sub_header[0..MAGIC_BYTES].to_vec(),
                }));
            }
            overlay.install(subsection_off as u64, sub_header.clone());
            let (_, entry_off, _, subsection_len, sub_end, dir_codec_meta_off, dir_codec_meta_size) =
                subsection_meta[i];
            let per_cluster_blocks_off = read_u64_le(
                &sub_header[sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF
                    ..sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF + U64_BYTES],
            ) as usize;
            let open_time_abs_end = subsection_off + per_cluster_blocks_off;
            if open_time_abs_end > sub_end {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} per_cluster_blocks_off {per_cluster_blocks_off} \
                     exceeds subsection length {subsection_len}",
                ))));
            }
            let codec_meta_size = read_u32_le(
                &sub_header[sub_hdr::CODEC_META_SIZE_OFF..sub_hdr::CODEC_META_SIZE_OFF + U32_BYTES],
            ) as usize;

            // Codec_meta lives at `[cluster_idx_off + n_cent*8 ..
            // per_cluster_blocks_off]`. We only need it for Sq8
            // columns (non-Sq8 declares codec_meta_size = 0).
            //
            // Exact-open path: fetch only the codec_meta bytes,
            // not the centroids / cluster_idx prefix that precedes
            // them in the subsection.
            if codec_meta_size > 0 {
                let cluster_idx_off = read_u64_le(
                    &sub_header
                        [sub_hdr::CLUSTER_IDX_OFF_OFF..sub_hdr::CLUSTER_IDX_OFF_OFF + U64_BYTES],
                ) as usize;
                let n_cent = read_u32_le(
                    &dir_bytes_slice[entry_off + dir_entry::N_CENT_OFF
                        ..entry_off + dir_entry::N_CENT_OFF + U32_BYTES],
                ) as usize;
                let codec_meta_off = cluster_idx_off + n_cent * CLUSTER_IDX_ENTRY_BYTES;
                let codec_meta_abs_off = subsection_off + codec_meta_off;
                if codec_meta_abs_off + codec_meta_size != open_time_abs_end {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "subsection {i} codec_meta_size {codec_meta_size} does not end at \
                         per_cluster_blocks_off {per_cluster_blocks_off}"
                    ))));
                }
                if dir_codec_meta_off != codec_meta_off || dir_codec_meta_size != codec_meta_size {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "subsection {i} directory codec_meta range \
                         off={dir_codec_meta_off} len={dir_codec_meta_size} does not match \
                         subheader-derived off={codec_meta_off} len={codec_meta_size}"
                    ))));
                }
                let _ = subsection_len;
            } else if dir_codec_meta_size != 0 || dir_codec_meta_off != 0 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} has zero codec_meta_size but directory declares \
                     off={dir_codec_meta_off} len={dir_codec_meta_size}"
                ))));
            }
        }

        Self::open_with_source(Source::Lazy(Arc::new(overlay)), columns_json, opts)
    }

    /// open over an arbitrary [`Source`].
    ///
    /// The structural decode path is the same as
    /// [`Self::open_with`]; this entry just accepts a pre-built
    /// `Source`. Used by:
    /// - Test helpers that need to wire a counting / mock
    ///   [`LazyByteSource`] under a `Source::Lazy` (e.g. the
    ///   range-counting integration test).
    /// - [`Self::open_lazy`], which pre-fetches the
    ///   open-time region into a [`PrefetchedSource`] overlay
    ///   and hands the overlay through as `Source::Lazy`.
    ///
    /// Contract on `Source::Lazy`: the lazy source's
    /// `try_get_range_sync` must resolve every range request
    /// the structural decode path issues — sub-header (56 B per
    /// column) and codec_meta tail (Sq8 columns only). The
    /// `open_lazy` path guarantees this via the overlay; callers
    /// constructing a `Source::Lazy` directly (tests, mmap-
    /// backed sources) must arrange equivalent residency.
    pub(crate) fn open_with_source(
        source: Source,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, VectorError> {
        if source.len() < OUTER_HEADER_SIZE + format::CRC_BYTES {
            return Err(VectorError::Read(ReadError::MissingKv(
                "vector blob header",
            )));
        }

        // Pull the fixed-size outer header in one fetch — five small
        // reads collapse into one `Bytes::slice`.
        let header = fetch_sync(&source, 0..OUTER_HEADER_SIZE, "outer header")?;
        if &header[0..MAGIC_BYTES] != format::vec::OUTER_MAGIC {
            return Err(VectorError::Read(ReadError::BadMagic {
                section: "vector",
                expected: format::vec::OUTER_MAGIC,
                actual: header[0..MAGIC_BYTES].to_vec(),
            }));
        }

        let version =
            read_u32_le(&header[outer_hdr::VERSION_OFF..outer_hdr::VERSION_OFF + U32_BYTES]);
        if version != format::vec::VERSION {
            return Err(VectorError::Read(ReadError::UnsupportedVersion(format!(
                "vector blob version {version}"
            ))));
        }

        let n_columns =
            read_u32_le(&header[outer_hdr::N_COLUMNS_OFF..outer_hdr::N_COLUMNS_OFF + U32_BYTES])
                as usize;
        let n_docs = read_u64_le(&header[outer_hdr::N_DOCS_OFF..outer_hdr::N_DOCS_OFF + U64_BYTES]);
        let dir_offset =
            read_u64_le(&header[outer_hdr::DIR_OFFSET_OFF..outer_hdr::DIR_OFFSET_OFF + U64_BYTES])
                as usize;

        // Verify directory CRC (cheap, needed before we can parallelize
        // subsection CRCs since we walk dir entries to find them).
        let dir_size = n_columns * DIR_ENTRY_SIZE;
        if dir_offset + dir_size + 4 > source.len() {
            return Err(VectorError::Read(ReadError::MalformedVersion(
                "vector directory runs past blob".into(),
            )));
        }
        let dir_bytes = fetch_sync(&source, dir_offset..dir_offset + dir_size, "directory")?;
        let dir_crc_bytes = fetch_sync(
            &source,
            dir_offset + dir_size..dir_offset + dir_size + 4,
            "directory crc",
        )?;
        let dir_crc_expected = read_u32_le(&dir_crc_bytes);
        let dir_crc_actual = crc32c(&dir_bytes);
        if dir_crc_expected != dir_crc_actual {
            return Err(VectorError::Read(ReadError::ChecksumMismatch {
                section: "vector/directory",
                column: String::new(),
            }));
        }

        // CRC verification: the outer-blob scan and per-subsection scans
        // each touch ~1.5 GiB at 1M × 384; together they're the bulk of
        // cold-open cost when `verify_crc=true`. Two stacked optimizations:
        //
        // 1. CLMUL-vectorized CRC32C via `crc-fast` in the checksum
        //    module — folds 8 lanes in vector regs instead of one
        //    serial dependency chain on `_mm_crc32_u64`, ~10× single-
        //    thread throughput on the boxes we measure.
        // 2. Run outer + per-subsection scans in parallel via rayon —
        //    each subsection's CRC is independent. The outer scan
        //    overlaps with the largest subsection on its own thread.
        //
        // Skip the whole pass via `OpenOptions { verify_crc: false }`
        // if upstream storage is trusted (content-addressed object
        // store, etc.); that path is ~12 ms regardless.
        if opts.verify_crc {
            // `Bytes` instead of `&'a [u8]` so the par_iter doesn't need
            // a lifetime parameter — each job owns a refcount-shared view
            // into the source, which is itself shared via the outer
            // `Source` for the duration of `open_with`.
            struct CrcJob {
                idx: i32,
                bytes: Bytes,
                expected: u32,
            }

            let mut jobs: Vec<CrcJob> = Vec::with_capacity(n_columns + 1);

            let outer_total = source.len();
            let outer_crc_pos = outer_total - format::CRC_BYTES;
            let outer_body = fetch_sync(&source, 0..outer_crc_pos, "outer body")?;
            let outer_crc_bytes = fetch_sync(&source, outer_crc_pos..outer_total, "outer crc")?;
            jobs.push(CrcJob {
                idx: -1,
                bytes: outer_body,
                expected: read_u32_le(&outer_crc_bytes),
            });

            for i in 0..n_columns {
                let entry_off = i * DIR_ENTRY_SIZE;
                let subsection_off = read_u64_le(
                    &dir_bytes[entry_off + dir_entry::SUBSECTION_OFF_OFF
                        ..entry_off + dir_entry::SUBSECTION_OFF_OFF + U64_BYTES],
                ) as usize;
                let subsection_len = read_u64_le(
                    &dir_bytes[entry_off + dir_entry::SUBSECTION_LEN_OFF
                        ..entry_off + dir_entry::SUBSECTION_LEN_OFF + U64_BYTES],
                ) as usize;
                let sub_end = subsection_off + subsection_len;
                if sub_end > source.len() {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "subsection {i} runs past blob"
                    ))));
                }
                let sub = fetch_sync(&source, subsection_off..sub_end, "subsection")?;
                if sub.len() < SUB_HEADER_SIZE + format::CRC_BYTES {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "subsection {i} too short"
                    ))));
                }
                let sub_crc_pos = sub.len() - format::CRC_BYTES;
                // `Bytes::slice` is zero-copy — no second `try_get_range_sync`
                // round-trip needed since we already hold the subsection.
                let sub_body = sub.slice(0..sub_crc_pos);
                let sub_crc_bytes = sub.slice(sub_crc_pos..sub.len());
                jobs.push(CrcJob {
                    idx: i as i32,
                    bytes: sub_body,
                    expected: read_u32_le(&sub_crc_bytes),
                });
            }

            // Serial CRC verify over the (handful of) subsections — a
            // one-time open cost, not query-hot, so it stays serial,
            // off the rayon scan path.
            let mismatch = jobs.iter().find_map(|job| {
                if crc32c(&job.bytes) != job.expected {
                    Some(job.idx)
                } else {
                    None
                }
            });
            if let Some(idx) = mismatch {
                if idx < 0 {
                    return Err(VectorError::Read(ReadError::ChecksumMismatch {
                        section: "vector",
                        column: String::new(),
                    }));
                } else {
                    let i = idx as usize;
                    return Err(VectorError::Read(ReadError::ChecksumMismatch {
                        section: "vector/subsection",
                        column: format!(" (column index {i})"),
                    }));
                }
            }
        }

        // Parse JSON.
        let cols_json: Vec<VectorColumnConfig> =
            serde_json::from_str(columns_json).map_err(|e| {
                VectorError::Read(ReadError::MalformedVersion(format!(
                    "inf.vec.columns JSON: {e}"
                )))
            })?;
        if cols_json.len() != n_columns {
            return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                "inf.vec.columns has {} entries, header says {n_columns}",
                cols_json.len()
            ))));
        }

        // Parse each directory entry, build ColumnReader.
        let mut columns = Vec::with_capacity(n_columns);
        let mut column_id_by_name = HashMap::with_capacity(n_columns);
        for (i, cfg) in cols_json.iter().enumerate() {
            let entry_off = i * DIR_ENTRY_SIZE;
            let column_id = read_u32_le(&dir_bytes[entry_off..entry_off + U32_BYTES]);
            if column_id != i as u32 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "vector dir entry {i} has column_id {column_id}"
                ))));
            }
            let dim = read_u32_le(
                &dir_bytes
                    [entry_off + dir_entry::DIM_OFF..entry_off + dir_entry::DIM_OFF + U32_BYTES],
            ) as usize;
            let n_cent = read_u32_le(
                &dir_bytes[entry_off + dir_entry::N_CENT_OFF
                    ..entry_off + dir_entry::N_CENT_OFF + U32_BYTES],
            );
            let metric_id = read_u32_le(
                &dir_bytes[entry_off + dir_entry::METRIC_ID_OFF
                    ..entry_off + dir_entry::METRIC_ID_OFF + U32_BYTES],
            );
            let rot_seed = read_u64_le(
                &dir_bytes[entry_off + dir_entry::ROT_SEED_OFF
                    ..entry_off + dir_entry::ROT_SEED_OFF + U64_BYTES],
            );
            let subsection_off = read_u64_le(
                &dir_bytes[entry_off + dir_entry::SUBSECTION_OFF_OFF
                    ..entry_off + dir_entry::SUBSECTION_OFF_OFF + U64_BYTES],
            ) as usize;
            let subsection_len = read_u64_le(
                &dir_bytes[entry_off + dir_entry::SUBSECTION_LEN_OFF
                    ..entry_off + dir_entry::SUBSECTION_LEN_OFF + U64_BYTES],
            ) as usize;
            // bytes [40..48] = summary_offset (absolute), [48..52] = summary_length,
            // [52..56] = codec_id (1) + reserved (3)
            let _summary_off_abs = read_u64_le(
                &dir_bytes[entry_off + dir_entry::SUMMARY_ABS_OFF
                    ..entry_off + dir_entry::SUMMARY_ABS_OFF + U64_BYTES],
            );
            let codec_id = dir_bytes[entry_off + dir_entry::CODEC_ID_OFF];
            let rerank_codec = RerankCodec::from_codec_id(codec_id).ok_or_else(|| {
                VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' has unknown rerank-codec id {codec_id} \
                     (known ids: 0=fp32, 1=sq8, 2=rabitq_only)",
                    cfg.column
                )))
            })?;
            if !rerank_codec.is_implemented() {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' uses rerank codec {} which is not implemented yet \
                     (`fp32`, `sq8`, `rabitq_only` are the supported codecs)",
                    cfg.column,
                    rerank_codec.name()
                ))));
            }

            // Validate against JSON.
            if dim != cfg.dim {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' dim mismatch: dir={dim} json={}",
                    cfg.column, cfg.dim
                ))));
            }
            if rot_seed != cfg.rot_seed {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' rot_seed mismatch",
                    cfg.column
                ))));
            }
            let metric = match metric_id {
                format::vec::METRIC_ID_L2SQ => Metric::L2Sq,
                format::vec::METRIC_ID_COSINE => Metric::Cosine,
                format::vec::METRIC_ID_NEGDOT => Metric::NegDot,
                _ => {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "unknown metric_id {metric_id} for column '{}'",
                        cfg.column
                    ))));
                }
            };

            // Validate subsection bounds + magic.
            //
            // Open-time region fetch. The reader's
            // open path only reads the sub-header + (when present)
            // codec_meta from the subsection. Per-cluster blocks,
            // full[], and the trailing CRC are search-time concerns.
            //
            // To minimize cold-open byte volume against an object-
            // store-backed `Source::Lazy`, fetch in two phases:
            //   Phase A — sub-header (56 B) at `[subsection_off..
            //     subsection_off + SUB_HEADER_SIZE]`. Carries
            //     codec_meta_size and per_cluster_blocks_off, which
            //     together fix the open-time region's end offset.
            //   Phase B — codec_meta tail at `[subsection_off +
            //     cluster_idx_off + n_cent*8 .. subsection_off +
            //     per_cluster_blocks_off]` (Sq8 columns only;
            //     skipped when codec_meta_size == 0).
            //
            // On `Source::InMemory` both fetches are zero-copy
            // `Bytes::slice` views — identical cost to the previous
            // single full-subsection slice. On `Source::Lazy` they
            // resolve through the `PrefetchedSource` overlay
            // installed by `open_lazy` (zero underlying GETs) when
            // the caller pre-fetched the open-time region; on a
            // bare `Source::Lazy` they fall through to the
            // underlying async `range` and round-trip per fetch.
            let sub_end = subsection_off + subsection_len;
            if sub_end > source.len() {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} runs past blob"
                ))));
            }
            if subsection_len < SUB_HEADER_SIZE + format::CRC_BYTES {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "subsection {i} too short"
                ))));
            }
            let sub_header = fetch_sync(
                &source,
                subsection_off..subsection_off + SUB_HEADER_SIZE,
                "sub_header",
            )?;
            if &sub_header[0..MAGIC_BYTES] != format::vec::SUB_MAGIC {
                return Err(VectorError::Read(ReadError::BadMagic {
                    section: "vector/subsection",
                    expected: format::vec::SUB_MAGIC,
                    actual: sub_header[0..MAGIC_BYTES].to_vec(),
                }));
            }
            // CRC was either already verified above in the parallel
            // pre-pass (when `opts.verify_crc` is true) or skipped on
            // the lazy fast path. Either way `sub_crc_pos` is derived
            // from `subsection_len` (directory entry), not from a
            // resident buffer.
            let sub_crc_pos = subsection_len - format::CRC_BYTES;

            // Sub-header parse. Only one layout version is
            // accepted; any other value is rejected as malformed.
            // See `format::vec::SUBSECTION_VERSION` for the
            // byte-level spec.
            //   [ 8..12] SUBSECTION_VERSION
            //   [12..16] codec_meta_size (u32 LE)
            //   [16..24] summary_centroid_offset (u64 LE)
            //   [24..28] summary_radius_x100 (u32 LE)
            //   [28..32] reserved (u32)
            //   [32..40] centroids_off (u64 LE)
            //   [40..48] cluster_idx_off (u64 LE)
            //   [48..56] per_cluster_blocks_off (u64 LE)
            let subsection_version =
                read_u32_le(&sub_header[sub_hdr::VERSION_OFF..sub_hdr::VERSION_OFF + U32_BYTES]);
            if subsection_version != format::vec::SUBSECTION_VERSION {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' has unsupported subsection layout version \
                     {subsection_version}; this build supports only {}",
                    cfg.column,
                    format::vec::SUBSECTION_VERSION
                ))));
            }

            let quant = BitQuantizer::new(dim);
            let code_bytes = quant.code_bytes();
            if code_bytes == 0 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' dim={dim} yields code_bytes=0",
                    cfg.column
                ))));
            }
            let per_vec_bytes = rerank_codec.per_vector_bytes(dim);
            let codec_meta_required_zero =
                matches!(rerank_codec, RerankCodec::Fp32 | RerankCodec::RabitqOnly);

            let codec_meta_size = read_u32_le(
                &sub_header[sub_hdr::CODEC_META_SIZE_OFF..sub_hdr::CODEC_META_SIZE_OFF + U32_BYTES],
            ) as usize;
            let summary_off = read_u64_le(
                &sub_header[sub_hdr::SUMMARY_OFF_OFF..sub_hdr::SUMMARY_OFF_OFF + U64_BYTES],
            ) as usize;
            let summary_radius_x100 = read_u32_le(
                &sub_header[sub_hdr::SUMMARY_RADIUS_X100_OFF
                    ..sub_hdr::SUMMARY_RADIUS_X100_OFF + U32_BYTES],
            );
            let centroids_off = read_u64_le(
                &sub_header[sub_hdr::CENTROIDS_OFF_OFF..sub_hdr::CENTROIDS_OFF_OFF + U64_BYTES],
            ) as usize;
            let cluster_idx_off = read_u64_le(
                &sub_header[sub_hdr::CLUSTER_IDX_OFF_OFF..sub_hdr::CLUSTER_IDX_OFF_OFF + U64_BYTES],
            ) as usize;
            let per_cluster_blocks_off = read_u64_le(
                &sub_header[sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF
                    ..sub_hdr::PER_CLUSTER_BLOCKS_OFF_OFF + U64_BYTES],
            ) as usize;

            if codec_meta_required_zero && codec_meta_size != 0 {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' has codec_meta_size={codec_meta_size} for codec {}; \
                     fp32/rabitq_only must write codec_meta_size=0",
                    cfg.column,
                    rerank_codec.name()
                ))));
            }

            // codec_meta sits immediately after cluster_idx (when
            // non-empty); 0 means "no codec_meta" and skips the
            // sq8_meta parse below.
            let cluster_idx_size = (n_cent as usize) * CLUSTER_IDX_ENTRY_BYTES;
            let codec_meta_off = if codec_meta_size == 0 {
                0
            } else {
                let off = cluster_idx_off + cluster_idx_size;
                // codec_meta must immediately precede the
                // per-cluster blocks region by exactly its
                // declared size. Any gap is a malformed superfile.
                if off + codec_meta_size != per_cluster_blocks_off {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "column '{}' codec_meta region [{off}..{}) does not abut \
                         per_cluster_blocks_off={per_cluster_blocks_off}",
                        cfg.column,
                        off + codec_meta_size
                    ))));
                }
                off
            };

            // Per-cluster blocks fill [per_cluster_blocks_off..
            // sub_crc_pos). Each doc contributes
            // `code_bytes + 4 (doc_id) + per_vec_bytes (full)` —
            // codes, doc-id, and rerank vector interleaved per
            // cluster. Solve for n_docs from the region size.
            let blocks_region_size = sub_crc_pos - per_cluster_blocks_off;
            let per_doc_stride = code_bytes + format::vec::DOC_ID_BYTES + per_vec_bytes;
            if per_doc_stride == 0 || !blocks_region_size.is_multiple_of(per_doc_stride) {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' per_cluster_blocks region {blocks_region_size} bytes \
                     not divisible by per-doc stride {per_doc_stride}",
                    cfg.column
                ))));
            }
            let col_n_docs = (blocks_region_size / per_doc_stride) as u32;
            let actual_codec_meta_size = codec_meta_size;

            // Sq8 + L2Sq adds the per-doc norms tail to codec_meta
            // (`n_docs * 4` bytes); now that `col_n_docs` is known
            // we can validate the declared size against the codec's
            // exact expectation.
            let expected_codec_meta_size =
                rerank_codec.codec_meta_bytes(dim, col_n_docs as usize, n_cent as usize, metric);
            if actual_codec_meta_size != expected_codec_meta_size {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' codec_meta_size={actual_codec_meta_size} on disk but \
                     codec {} / metric {metric:?} expects {expected_codec_meta_size} bytes",
                    cfg.column,
                    rerank_codec.name()
                ))));
            }

            let summary_radius = (summary_radius_x100 as f32) / SUMMARY_RADIUS_SCALE;

            let sq8_meta = if matches!(rerank_codec, RerankCodec::Sq8Residual) {
                let meta_abs_start = subsection_off + codec_meta_off;
                let meta_abs_end = meta_abs_start + actual_codec_meta_size;
                let so_block_bytes = (n_cent as usize) * dim * 4;
                let scale_end = so_block_bytes;
                let offset_end = scale_end + so_block_bytes;
                if let Some(meta_bytes) = source.try_get_range_sync(meta_abs_start..meta_abs_end) {
                    let scale = parse_f32_le_vec(&meta_bytes[0..scale_end]);
                    let offset = parse_f32_le_vec(&meta_bytes[scale_end..offset_end]);
                    let per_doc_norms: Option<Arc<[f32]>> =
                        if matches!(metric, Metric::L2Sq | Metric::Cosine) {
                            let norms_end = offset_end + (col_n_docs as usize) * 4;
                            debug_assert_eq!(norms_end, actual_codec_meta_size);
                            Some(Arc::from(parse_f32_le_vec(
                                &meta_bytes[offset_end..norms_end],
                            )))
                        } else {
                            None
                        };
                    Some(Sq8ColumnMeta::Eager {
                        scale,
                        offset,
                        per_doc_norms,
                    })
                } else {
                    Some(Sq8ColumnMeta::Lazy {
                        scale_abs_off: meta_abs_start,
                        offset_abs_off: meta_abs_start + scale_end,
                        norms_abs_off: matches!(metric, Metric::L2Sq | Metric::Cosine)
                            .then_some(meta_abs_start + offset_end),
                    })
                }
            } else {
                None
            };

            // Structural bounds. cluster_idx fits before the
            // per-cluster blocks region. The
            // `blocks_region_size.is_multiple_of(...)` check
            // above already pinned n_docs and that the per-cluster
            // blocks region tiles exactly to the CRC; this check
            // guards an off-by-one in the cluster_idx slot.
            let cluster_idx_end = cluster_idx_off + cluster_idx_size;
            if cluster_idx_end > sub_crc_pos {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' cluster index runs past subsection",
                    cfg.column
                ))));
            }

            // Soft cross-check: cfg.metric matches blob's metric.
            let cfg_metric = match cfg.metric.as_str() {
                "l2sq" => Some(Metric::L2Sq),
                "cosine" => Some(Metric::Cosine),
                "negdot" => Some(Metric::NegDot),
                _ => None,
            };
            if let Some(m) = cfg_metric
                && m != metric
            {
                return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                    "column '{}' metric mismatch: dir={metric:?} json={}",
                    cfg.column, cfg.metric
                ))));
            }

            columns.push(ColumnReader {
                name: cfg.column.clone(),
                dim,
                n_cent,
                n_docs: col_n_docs,
                metric,
                rot_seed,
                rerank_codec,
                sq8_meta,
                lazy_sq8_parsed: OnceLock::new(),
                subsection_range: subsection_off..sub_end,
                summary_off,
                summary_radius,
                centroids_off,
                cluster_idx_off,
                codec_meta_off,
                per_cluster_blocks_off,
                quant,
                rot: RandomRotation::new(dim, rot_seed),
            });
            column_id_by_name.insert(cfg.column.clone(), i as u32);
        }

        Ok(VectorReader {
            source,
            n_docs,
            columns,
            column_id_by_name,
        })
    }

    pub fn n_docs(&self) -> u64 {
        self.n_docs
    }

    pub fn vector_columns(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|c| c.name.as_str())
    }
    pub fn vector_columns_config(&self) -> impl Iterator<Item = &ColumnReader> {
        self.columns.iter()
    }

    pub(crate) fn public_rerank_mult(&self, _column: &str, base: usize) -> usize {
        base
    }

    /// Per-column summary centroid + radius, used by the storage plan
    /// for cross-superfile skip pruning.
    pub fn summary(&self, column: &str) -> Option<(Vec<f32>, f32)> {
        let cid = *self.column_id_by_name.get(column)?;
        let col = &self.columns[cid as usize];
        // byte access routed through `Source::try_get_range_sync`
        // — zero-copy on `InMemory`, lazy on `Source::Lazy`.
        let sub = self
            .source
            .try_get_range_sync(col.subsection_range.clone())?;
        let off = col.summary_off;
        let dim = col.dim;
        let centroid: Vec<f32> = (0..dim)
            .map(|i| {
                let s = off + i * 4;
                f32::from_le_bytes([sub[s], sub[s + 1], sub[s + 2], sub[s + 3]])
            })
            .collect();
        Some((centroid, col.summary_radius))
    }

    /// The column's per-cluster IVF centroids (fp32, cluster-major,
    /// `n_cent * dim`) plus each cluster's indexed doc count. Returns
    /// `(n_cent, dim, centroids, counts)`. Used by the writer to stage
    /// quantized cluster centroids into the manifest for cross-superfile
    /// global cluster selection. `None` if the column is unknown or the
    /// centroid/cluster_idx bytes aren't resident.
    pub fn cluster_centroids(&self, column: &str) -> Option<(u32, u32, Vec<f32>, Vec<u32>)> {
        let cid = *self.column_id_by_name.get(column)?;
        let col = &self.columns[cid as usize];
        let sub = self
            .source
            .try_get_range_sync(col.subsection_range.clone())?;
        let n_cent = col.n_cent as usize;
        let dim = col.dim;
        let stride = dim * 4;

        // Centroids: fp32, cluster-major, at `centroids_off`.
        let mut centroids = Vec::with_capacity(n_cent * dim);
        for c in 0..n_cent {
            let base = col.centroids_off + c * stride;
            for d in 0..dim {
                let s = base + d * 4;
                centroids.push(f32::from_le_bytes([
                    sub[s],
                    sub[s + 1],
                    sub[s + 2],
                    sub[s + 3],
                ]));
            }
        }

        // cluster_idx: `n_cent` × `(doc_off: u32, count: u32)`; we want
        // the count (second u32 of each 8-byte entry).
        let mut counts = Vec::with_capacity(n_cent);
        for c in 0..n_cent {
            let b = col.cluster_idx_off + c * CLUSTER_IDX_ENTRY_BYTES + CLUSTER_IDX_COUNT_OFFSET;
            counts.push(u32::from_le_bytes([
                sub[b],
                sub[b + 1],
                sub[b + 2],
                sub[b + 3],
            ]));
        }

        Some((col.n_cent, dim as u32, centroids, counts))
    }

    /// Single-column kNN search. Returns `(local_doc_id,
    /// distance)` sorted ascending by distance (smaller = closer
    /// for every metric).
    ///
    /// Sync — every public surface in `src/` is sync. Routes
    /// per-region byte
    /// access through [`Source::get_range`], which is itself
    /// sync and bridges to the underlying async
    /// `LazyByteSource::range` only on a cold `Source::Lazy`
    /// miss (via `block_in_place + Handle::block_on`, same
    /// pattern as `supertable::query::superfile_reader`). On
    /// `Source::InMemory` and on `Source::Lazy` warm caches
    /// (`BytesLazyByteSource`, mmap-backed) every fetch resolves
    /// zero-copy on the sync fast path.
    ///
    /// Range count per cold first search at `nprobe = 8` on the
    /// v0 layout:
    ///
    /// - 1 range for centroids (`n_cent × dim × 4` bytes)
    /// - 1 range for the cluster_idx header (`n_cent × 8` bytes)
    /// - `nprobe` ranges for per-cluster codes
    /// - `nprobe` ranges for per-cluster doc_ids
    /// - 1 fat range covering the rerank batch in `full[]` from
    ///   `min(pos)` to `max(pos) + 1`
    ///
    /// At `nprobe = 8`: 2 + 16 + 1 = **19 ranges**. Rerank `pos`
    /// is captured inline in the shortlist tuple at code-scoring
    /// time (each candidate's position is `off + i` where
    /// `(off, cnt)` is the cluster's entry and `i` is the
    /// in-cluster index), so there is no `doc_to_pos` lookup
    /// table at all — that 4 MB / 1M-doc allocation was deleted
    /// once an audit confirmed zero external readers.
    pub async fn search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_mult: usize,
    ) -> Result<Vec<(u32, f32)>, VectorError> {
        let (col, validated) = self.resolve_column(column, query, k)?;
        if !validated {
            return Ok(Vec::new());
        }
        // Centroids are always fp32 (4 bytes/dim) regardless of codec.
        let centroid_stride = col.dim * 4;
        let sub_start = col.subsection_range.start;

        // 1. Centroids + cluster_idx region. These are contiguous
        //    in the subsection, and search needs both before it can
        //    issue per-cluster range requests. Fetching them as one
        //    span saves one request and one foreground RTT batch on
        //    cold object-store search.
        let centroids_start = sub_start + col.centroids_off;
        let centroids_end = centroids_start + (col.n_cent as usize) * centroid_stride;
        let idx_start = sub_start + col.cluster_idx_off;
        let idx_end = idx_start + (col.n_cent as usize) * CLUSTER_IDX_ENTRY_BYTES;
        let centroid_idx_region = self
            .source
            .get_range(centroids_start..idx_end)
            .map_err(|e| VectorError::LazySource(e.to_string()))?;
        let centroids = centroid_idx_region.slice(0..centroids_end - centroids_start);
        let cluster_idx =
            centroid_idx_region.slice(idx_start - centroids_start..idx_end - centroids_start);

        let nprobe_eff = nprobe.min(col.n_cent as usize).max(1);
        // 2. Score centroids → top `nprobe` clusters. Only the
        // retained probe set is fully sorted; the tail centroids are
        // partitioned away with `select_nth_unstable_by`.
        let centroid_scores = score_centroids(&centroids, col, query, nprobe_eff);

        // 3. Rotate query once for the 1-bit code estimator.
        let mut q_rot = vec![0f32; col.dim];
        col.rot.apply(query, &mut q_rot);

        // 4. Per-cluster fetches and shortlist build. Shortlist
        //    tuple is (doc_id, estimate, pos, cluster_id);
        //    pos = off + i and cluster_id are captured inline at
        //    no extra fetch cost. cluster_id is consumed by the
        //    Sq8PerCluster rerank dispatch to pick each
        //    candidate's quantizer; Fp32/RabitqOnly rerank paths
        //    ignore it.
        //
        //    codes and doc_ids per cluster live in
        //    one contiguous block on disk (`per-cluster blocks`
        //    region under the v1 layout), so each cluster pulls
        //    in **one** `get_range` call. those
        //    `nprobe` per-cluster GETs fire **concurrently**
        //    via [`Source::get_ranges_parallel`] instead of
        //    serially via per-call [`Source::get_range`]. On a
        //    `Source::Lazy` backed by object storage the cold
        //    first-search wall-clock collapses from
        //    `sum_c RTT(c)` to `max_c RTT(c)` (one HTTP/2
        //    multiplexed batch). On warm/in-memory paths the
        //    requests resolve through the sync zero-copy
        //    fast path with no extra cost.
        let _ = sub_start; // retained for downstream offset math below
        let cb = col.quant.code_bytes();
        let mut cluster_meta: Vec<(usize, u32, u32)> = Vec::with_capacity(nprobe_eff);
        let mut cluster_prefix_ranges: Vec<Range<usize>> = Vec::with_capacity(nprobe_eff);
        for &(c, _) in &centroid_scores {
            let (off, cnt) = read_cluster_entry(&cluster_idx, c);
            if cnt == 0 {
                continue;
            }
            cluster_prefix_ranges.push(col.cluster_codes_doc_ids_range(off, cnt));
            cluster_meta.push((c, off, cnt));
        }
        let lazy_sq8_meta_range = lazy_sq8_meta_range(col);
        let prefix_blocks_sync: Option<Vec<Bytes>> = cluster_prefix_ranges
            .iter()
            .map(|range| self.source.try_get_range_sync(range.clone()))
            .collect();
        // Survivor-only rerank fetch on BOTH the warm and cold paths.
        // Coarse-score off the cheap `[codes][doc_ids]` prefix, then
        // pull the full rerank vectors ONLY for the survivors:
        //   * warm — the prefix is already resident (the sync probe
        //     above hits), and survivor rows are sliced from the
        //     resident superfile; zero GETs either wave.
        //   * cold — fetch the prefixes over the wire in one coalesced
        //     RTT batch, score, then fetch the survivor rows in a
        //     second small batch. The dominant per-candidate `full[]`
        //     bytes (~3.4 MiB/superfile — the volume that saturates S3
        //     read throughput on a 256-way cold fan-out) are never
        //     moved for non-survivors.
        // The scoring math is identical to the old full-block path —
        // same codes, same coarse shortlist, same fp32/Sq8 rerank — so
        // recall is unchanged; only *which* bytes are fetched differs.
        let survivor_only_rerank_fetch = true;
        let (cluster_blocks, lazy_sq8_meta_bytes) = if let Some(prefix_blocks) = prefix_blocks_sync
        {
            let meta_bytes = if let Some(range) = lazy_sq8_meta_range {
                let mut fetched = self
                    .source
                    .get_ranges_parallel(&[range])
                    .map_err(|e| VectorError::LazySource(e.to_string()))?;
                fetched.pop()
            } else {
                None
            };
            (prefix_blocks, meta_bytes)
        } else {
            // Cold: fetch only the codes+doc_ids prefixes (coalesced)
            // plus the Sq8 meta in one batch. Full vectors are fetched
            // later, for survivors only.
            get_cluster_ranges_coalesced_with_extra(
                &self.source,
                &cluster_prefix_ranges,
                lazy_sq8_meta_range,
            )
            .map_err(|e| VectorError::LazySource(e.to_string()))?
        };
        debug_assert_eq!(cluster_blocks.len(), cluster_meta.len());

        // Score the 1-bit shortlist and build rerank references — the
        // pure-CPU stage shared with `search_async` (see
        // [`build_shortlist`]). Each cluster block is
        // `[codes][doc_ids][full?]`; scoring reads the prefix, and the
        // survivor `full[]` rows are fetched below — the only step
        // that differs from the async path.
        let (candidates, survivor_full_ranges) = match build_shortlist(
            col,
            &q_rot,
            cb,
            &cluster_meta,
            &cluster_blocks,
            survivor_only_rerank_fetch,
            k,
            rerank_mult,
        )
        .await
        {
            ShortlistOutcome::Done(out) => return Ok(out),
            ShortlistOutcome::Rerank {
                candidates,
                survivor_full_ranges,
            } => (candidates, survivor_full_ranges),
        };
        // Coalesce the survivor rows (scattered single-vector ranges
        // inside each cluster's `full[]` region) into a small second
        // wave; warm ranges resolve sync/zero-copy, so this is a cheap
        // sort.
        let survivor_full_rows = match survivor_full_ranges {
            Some(ranges) => Some(
                get_cluster_ranges_coalesced(&self.source, &ranges)
                    .map_err(|e| VectorError::LazySource(e.to_string()))?,
            ),
            None => None,
        };

        // 8. CPU-only rerank using the true metric. Sq8 columns
        //    pre-build a per-query kernel that folds the per-dim
        //    scale/offset into the query (one `dim/8` SIMD pass);
        //    the per-doc inner step is then a plain u8→f32 widen
        //    + SIMD dot. Fp32 takes the flat dispatch.
        rerank_candidates_from_blocks(
            &self.source,
            lazy_sq8_meta_bytes.as_ref(),
            &cluster_blocks,
            survivor_full_rows.as_deref(),
            &candidates,
            col,
            query,
            k,
        )
        .await
        .map_err(|e| VectorError::LazySource(e.to_string()))
    }

    /// Async sibling of [`Self::search`]. Byte-for-byte the same IVF
    /// kernel — identical centroid scoring, coarse 1-bit shortlist,
    /// survivor-only rerank, and the same coalesced range plans, so
    /// recall is identical — but the three fetch waves (centroid+idx
    /// region, per-cluster code prefixes + Sq8 meta, survivor rerank
    /// rows) are `await`ed on the caller's runtime instead of bridged
    /// through a per-call throwaway runtime. This is what lets the
    /// supertable vector fan-out drive every superfile concurrently on
    /// the shared query runtime — mirroring the FTS
    /// `bm25_search_pretokenized` path — rather than serializing cold
    /// object-store GETs. The CPU steps (centroid/code scoring,
    /// rerank) call the same helpers as the sync path and parallelize
    /// on the global rayon pool; warm/in-memory ranges still resolve
    /// sync/zero-copy via `try_get_range_sync` with no `await`.
    pub async fn search_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_mult: usize,
    ) -> Result<Vec<(u32, f32)>, VectorError> {
        let (col, validated) = self.resolve_column(column, query, k)?;
        if !validated {
            return Ok(Vec::new());
        }
        let centroid_stride = col.dim * 4;
        let sub_start = col.subsection_range.start;

        // 1. Centroids + cluster_idx region (one contiguous span).
        let centroids_start = sub_start + col.centroids_off;
        let centroids_end = centroids_start + (col.n_cent as usize) * centroid_stride;
        let idx_start = sub_start + col.cluster_idx_off;
        let idx_end = idx_start + (col.n_cent as usize) * CLUSTER_IDX_ENTRY_BYTES;
        let centroid_idx_region = self
            .source
            .range_async(centroids_start..idx_end)
            .await
            .map_err(|e| VectorError::LazySource(e.to_string()))?;
        let centroids = centroid_idx_region.slice(0..centroids_end - centroids_start);
        let cluster_idx =
            centroid_idx_region.slice(idx_start - centroids_start..idx_end - centroids_start);

        let nprobe_eff = nprobe.min(col.n_cent as usize).max(1);
        // 2. Score centroids → top `nprobe` clusters.
        let centroid_scores = score_centroids(&centroids, col, query, nprobe_eff);

        // 3. Rotate query once for the 1-bit code estimator.
        let mut q_rot = vec![0f32; col.dim];
        col.rot.apply(query, &mut q_rot);

        // 4. Probe the centroid-scored clusters through the shared tail
        //    (also used by the externally-selected
        //    `search_clusters_async` path).
        let _ = sub_start;
        let chosen: Vec<usize> = centroid_scores.iter().map(|&(c, _)| c).collect();
        self.probe_clusters_async(col, query, &q_rot, &cluster_idx, &chosen, k, rerank_mult)
            .await
    }

    /// Async IVF probe over an **externally chosen** set of cluster ids.
    /// The cross-superfile global selector picks these from the manifest's
    /// per-cluster centroids, so this skips the superfile's own centroid
    /// scoring entirely — it fetches just the cluster index, then probes
    /// exactly `clusters` (ids ≥ `n_cent` and empty clusters are
    /// ignored). The shortlist + rerank are byte-for-byte the same as
    /// [`Self::search_async`].
    pub async fn search_clusters_async(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        clusters: &[u32],
        rerank_mult: usize,
    ) -> Result<Vec<(u32, f32)>, VectorError> {
        let (col, validated) = self.resolve_column(column, query, k)?;
        if !validated {
            return Ok(Vec::new());
        }
        let sub_start = col.subsection_range.start;
        let idx_start = sub_start + col.cluster_idx_off;
        let idx_end = idx_start + (col.n_cent as usize) * CLUSTER_IDX_ENTRY_BYTES;
        let cluster_idx = self
            .source
            .range_async(idx_start..idx_end)
            .await
            .map_err(|e| VectorError::LazySource(e.to_string()))?;
        let mut q_rot = vec![0f32; col.dim];
        col.rot.apply(query, &mut q_rot);
        let chosen: Vec<usize> = clusters.iter().map(|&c| c as usize).collect();
        self.probe_clusters_async(col, query, &q_rot, &cluster_idx, &chosen, k, rerank_mult)
            .await
    }

    /// Shared async tail of the IVF probe: given a chosen set of cluster
    /// ids plus the already-fetched cluster index, fetch each non-empty
    /// cluster's block, build the 1-bit shortlist, and rerank to top-k.
    /// Used by [`Self::search_async`] (clusters from this superfile's
    /// centroid scoring) and [`Self::search_clusters_async`] (clusters
    /// from the global cross-superfile selector).
    async fn probe_clusters_async(
        &self,
        col: &ColumnReader,
        query: &[f32],
        q_rot: &[f32],
        cluster_idx: &[u8],
        chosen: &[usize],
        k: usize,
        rerank_mult: usize,
    ) -> Result<Vec<(u32, f32)>, VectorError> {
        let cb = col.quant.code_bytes();
        let mut cluster_meta: Vec<(usize, u32, u32)> = Vec::with_capacity(chosen.len());
        let mut cluster_prefix_ranges: Vec<Range<usize>> = Vec::with_capacity(chosen.len());
        for &c in chosen {
            if c >= col.n_cent as usize {
                continue;
            }
            let (off, cnt) = read_cluster_entry(cluster_idx, c);
            if cnt == 0 {
                continue;
            }
            cluster_prefix_ranges.push(col.cluster_codes_doc_ids_range(off, cnt));
            cluster_meta.push((c, off, cnt));
        }
        if cluster_meta.is_empty() {
            return Ok(Vec::new());
        }
        let lazy_sq8_meta_range = lazy_sq8_meta_range(col);
        // Warm fast path: every prefix already resident → sync zero-copy.
        let prefix_blocks_sync: Option<Vec<Bytes>> = cluster_prefix_ranges
            .iter()
            .map(|range| self.source.try_get_range_sync(range.clone()))
            .collect();
        let survivor_only_rerank_fetch = true;
        let (cluster_blocks, lazy_sq8_meta_bytes) = if let Some(prefix_blocks) = prefix_blocks_sync
        {
            let meta_bytes = if let Some(range) = lazy_sq8_meta_range {
                let mut fetched = self
                    .source
                    .get_ranges_parallel_async(&[range])
                    .await
                    .map_err(|e| VectorError::LazySource(e.to_string()))?;
                fetched.pop()
            } else {
                None
            };
            (prefix_blocks, meta_bytes)
        } else {
            // Cold: codes+doc_ids prefixes (coalesced) + Sq8 meta in one
            // concurrent batch on the caller's runtime.
            get_cluster_ranges_coalesced_with_extra_async(
                &self.source,
                &cluster_prefix_ranges,
                lazy_sq8_meta_range,
            )
            .await
            .map_err(|e| VectorError::LazySource(e.to_string()))?
        };
        debug_assert_eq!(cluster_blocks.len(), cluster_meta.len());

        // Shared pure-CPU shortlist + candidate-build stage (see
        // [`build_shortlist`]); only the survivor-row fetch below
        // diverges from the sync path.
        let (candidates, survivor_full_ranges) = match build_shortlist(
            col,
            q_rot,
            cb,
            &cluster_meta,
            &cluster_blocks,
            survivor_only_rerank_fetch,
            k,
            rerank_mult,
        )
        .await
        {
            ShortlistOutcome::Done(out) => return Ok(out),
            ShortlistOutcome::Rerank {
                candidates,
                survivor_full_ranges,
            } => (candidates, survivor_full_ranges),
        };
        // Survivor rerank rows in one concurrent batch on the caller's
        // runtime; warm ranges resolve sync/zero-copy with no await.
        let survivor_full_rows = match survivor_full_ranges {
            Some(ranges) => Some(
                get_cluster_ranges_coalesced_async(&self.source, &ranges)
                    .await
                    .map_err(|e| VectorError::LazySource(e.to_string()))?,
            ),
            None => None,
        };

        rerank_candidates_from_blocks(
            &self.source,
            lazy_sq8_meta_bytes.as_ref(),
            &cluster_blocks,
            survivor_full_rows.as_deref(),
            &candidates,
            col,
            query,
            k,
        )
        .await
        .map_err(|e| VectorError::LazySource(e.to_string()))
    }

    /// Look up the column by name and validate `query.len() == col.dim`
    /// + the "empty work" short-circuit (`k == 0` or `n_docs == 0`).
    /// `Ok((col, true))` = real search to follow; `Ok((col, false))`
    /// = empty-result short circuit, caller returns `Ok(Vec::new())`.
    #[inline]
    /// Retrieve original vectors in their insertion order for fp32-encoded columns.
    /// Returns an error if the column uses a different encoding (Sq8Residual or RabitqOnly).
    pub fn get_vectors_fp32(&self, column: &str) -> Result<Vec<Vec<f32>>, VectorError> {
        let cid = *self
            .column_id_by_name
            .get(column)
            .ok_or_else(|| VectorError::UnknownColumn(column.to_string()))?;
        let col = &self.columns[cid as usize];

        if col.rerank_codec != RerankCodec::Fp32 {
            return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                "column '{}' uses rerank codec {} instead of Fp32",
                col.name,
                col.rerank_codec.name()
            ))));
        }

        if col.n_docs == 0 {
            return Ok(Vec::new());
        }

        let sub_start = col.subsection_range.start;
        let idx_start = sub_start + col.cluster_idx_off;
        let idx_end = idx_start + (col.n_cent as usize) * 8;
        let cluster_idx = self
            .source
            .get_range(idx_start..idx_end)
            .map_err(|e| VectorError::LazySource(e.to_string()))?;

        let cb = col.quant.code_bytes();
        let per_vec_bytes = col.rerank_codec.per_vector_bytes(col.dim);

        // Collect all cluster ranges needed for fetching
        let mut cluster_ranges: Vec<Range<usize>> = Vec::new();
        let mut cluster_meta: Vec<(usize, u32, u32)> = Vec::new();

        for c in 0..col.n_cent as usize {
            let (off, cnt) = read_cluster_entry(&cluster_idx, c);
            if cnt == 0 {
                continue;
            }
            cluster_ranges.push(col.cluster_block_range(off, cnt));
            cluster_meta.push((c, off, cnt));
        }

        if cluster_ranges.is_empty() {
            return Ok(Vec::new());
        }

        // Fetch all cluster blocks
        let cluster_blocks = self
            .source
            .get_ranges_parallel(&cluster_ranges)
            .map_err(|e| VectorError::LazySource(e.to_string()))?;

        // Allocate output vector with doc_id -> vector mapping
        let mut result: Vec<Option<Vec<f32>>> = vec![None; col.n_docs as usize];

        // Process each cluster block
        for (bi, block) in cluster_blocks.iter().enumerate() {
            let (_, _off, cnt) = cluster_meta[bi];
            let cnt_usize = cnt as usize;

            // Layout within the block: [codes_chunk][doc_ids_chunk][full_chunk]
            let codes_len = cnt_usize * cb;
            let doc_ids_len = cnt_usize * 4;
            let full_start = codes_len + doc_ids_len;

            // Extract doc_ids from the block
            let doc_ids_slice = block.slice(codes_len..codes_len + doc_ids_len);

            // Extract and reconstruct vectors
            for i in 0..cnt_usize {
                let doc_id = u32::from_le_bytes([
                    doc_ids_slice[i * 4],
                    doc_ids_slice[i * 4 + 1],
                    doc_ids_slice[i * 4 + 2],
                    doc_ids_slice[i * 4 + 3],
                ]) as usize;

                let vec_start = full_start + i * per_vec_bytes;
                let vec_end = vec_start + per_vec_bytes;
                let vec_bytes = block.slice(vec_start..vec_end);

                // Convert bytes to f32 vector
                // For Fp32 codec, per_vec_bytes = dim * 4, so we expect dim f32s
                let vec_f32: Vec<f32> = vec_bytes
                    .as_ref()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                if vec_f32.len() != col.dim {
                    return Err(VectorError::Read(ReadError::MalformedVersion(format!(
                        "vector size mismatch: got {}, expected {}",
                        vec_f32.len(),
                        col.dim
                    ))));
                }

                if doc_id < col.n_docs as usize {
                    result[doc_id] = Some(vec_f32);
                }
            }
        }

        // Convert to final result, checking all vectors were found
        result
            .into_iter()
            .enumerate()
            .map(|(idx, vec_opt)| {
                vec_opt.ok_or_else(|| {
                    VectorError::Read(ReadError::MalformedVersion(format!(
                        "missing vector for doc_id {}",
                        idx
                    )))
                })
            })
            .collect()
    }

    fn resolve_column(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
    ) -> Result<(&ColumnReader, bool), VectorError> {
        let cid = *self
            .column_id_by_name
            .get(column)
            .ok_or_else(|| VectorError::UnknownColumn(column.to_string()))?;
        let col = &self.columns[cid as usize];
        if query.len() != col.dim {
            return Err(VectorError::DimensionMismatch {
                expected: col.dim,
                got: query.len(),
            });
        }
        if k == 0 || col.n_docs == 0 {
            return Ok((col, false));
        }
        Ok((col, true))
    }
}

/// Outcome of the 1-bit shortlist + candidate-build stage shared by
/// [`VectorReader::search`] and [`VectorReader::search_async`].
enum ShortlistOutcome {
    /// Final result — no rerank fetch needed: empty shortlist,
    /// `coarse_limit == 0`, or a `RabitqOnly` column whose 1-bit
    /// shortlist *is* the ranking.
    Done(Vec<(u32, f32)>),
    /// Survivors to rerank against the true metric.
    /// `survivor_full_ranges` (when `Some`) are the per-survivor
    /// `full[]` rows the caller fetches — sync or async, the only
    /// step that differs between the two search paths.
    Rerank {
        candidates: Vec<RerankCandidate>,
        survivor_full_ranges: Option<Vec<Range<usize>>>,
    },
}

/// Pure-CPU stage shared by the sync and async vector search paths.
///
/// Scores the probed clusters' 1-bit codes into a bounded shortlist,
/// short-circuits `RabitqOnly` columns (whose shortlist is the final
/// ranking), and otherwise builds the rerank references plus the
/// survivor `full[]` ranges to fetch. Holds no I/O: the caller does
/// the survivor-row fetch (sync vs async — the sole divergence) and
/// then runs [`rerank_candidates_from_blocks`]. Factoring this out
/// keeps `search` / `search_async` down to their fetch waves around a
/// single shared kernel, so the two can't drift in scoring/recall.
#[allow(clippy::too_many_arguments)]
async fn build_shortlist(
    col: &ColumnReader,
    q_rot: &[f32],
    cb: usize,
    cluster_meta: &[(usize, u32, u32)],
    cluster_blocks: &[Bytes],
    survivor_only_rerank_fetch: bool,
    k: usize,
    rerank_mult: usize,
) -> ShortlistOutcome {
    let full_vec_bytes = col.rerank_codec.per_vector_bytes(col.dim);
    // Score each probed cluster's 1-bit codes into the shortlist.
    // The per-cluster slices are zero-copy `Bytes` views; the actual
    // estimate scan is the hot CPU work, parallelized across clusters
    // once the candidate pool is large enough to amortize the rayon
    // hand-off. Cluster scoring is order-independent: every survivor
    // is re-sorted by estimate below, so parallel and serial
    // shortlists rank identically.
    let total_candidates: usize = cluster_meta.iter().map(|&(_, _, cnt)| cnt as usize).sum();
    let coarse_limit = if matches!(col.rerank_codec, RerankCodec::RabitqOnly) {
        k
    } else {
        k.saturating_mul(rerank_mult)
    };
    if coarse_limit == 0 {
        return ShortlistOutcome::Done(Vec::new());
    }
    let score_block =
        |heap: &mut BoundedCoarseHeap, (&(c, off, cnt), block): (&(usize, u32, u32), &Bytes)| {
            let codes_len = (cnt as usize) * cb;
            let doc_ids_len = (cnt as usize) * 4;
            debug_assert_eq!(
                block.len(),
                if survivor_only_rerank_fetch {
                    codes_len + doc_ids_len
                } else {
                    codes_len + doc_ids_len + (cnt as usize) * full_vec_bytes
                }
            );
            let codes = block.slice(0..codes_len);
            let doc_ids = block.slice(codes_len..codes_len + doc_ids_len);
            score_cluster_codes_into_heap(
                &codes, &doc_ids, cnt, off, c as u32, &col.quant, q_rot, heap,
            );
        };
    let shortlist_heap = if total_candidates >= PARALLEL_SCAN_MIN && cluster_meta.len() > 1 {
        // Parallelize the coarse 1-bit scan across the global rayon pool,
        // bridged back via a oneshot so no tokio worker blocks under the
        // compute. Cluster scoring is order-independent — every survivor
        // is re-sorted below — so chunked-parallel and serial shortlists
        // rank identically. Partial heaps merge after.
        let n_tasks = parallel_chunks(cluster_meta.len());
        let chunk = cluster_meta.len().div_ceil(n_tasks).max(1);
        let quant = col.quant.clone();
        let q_rot_v: Vec<f32> = q_rot.to_vec();
        let meta_owned: Vec<(usize, u32, u32)> = cluster_meta.to_vec();
        let blocks_owned: Vec<Bytes> = cluster_blocks.to_vec();
        let (tx, rx) = tokio::sync::oneshot::channel();
        rayon::spawn(move || {
            let acc = meta_owned
                .par_chunks(chunk)
                .zip(blocks_owned.par_chunks(chunk))
                .map(|(meta_chunk, block_chunk)| {
                    let mut heap = BoundedCoarseHeap::new(coarse_limit);
                    for (&(c, off, cnt), block) in meta_chunk.iter().zip(block_chunk.iter()) {
                        let codes_len = (cnt as usize) * cb;
                        let doc_ids_len = (cnt as usize) * 4;
                        let codes = block.slice(0..codes_len);
                        let doc_ids = block.slice(codes_len..codes_len + doc_ids_len);
                        score_cluster_codes_into_heap(
                            &codes, &doc_ids, cnt, off, c as u32, &quant, &q_rot_v, &mut heap,
                        );
                    }
                    heap
                })
                .reduce(
                    || BoundedCoarseHeap::new(coarse_limit),
                    |mut a, b| {
                        a.merge(b);
                        a
                    },
                );
            let _ = tx.send(acc);
        });
        rx.await
            .expect("vector shortlist rayon task dropped result")
    } else {
        let mut heap = BoundedCoarseHeap::new(coarse_limit);
        for item in cluster_meta.iter().zip(cluster_blocks.iter()) {
            score_block(&mut heap, item);
        }
        heap
    };
    let mut shortlist = shortlist_heap.into_vec();

    if shortlist.is_empty() {
        return ShortlistOutcome::Done(Vec::new());
    }

    // `RabitqOnly` short-circuit: the 1-bit shortlist *is* the final
    // ranking — no `full[]` region on disk, no rerank step. Partial-
    // sort to the top-k by descending estimate, then flip the sign so
    // the returned `(doc_id, distance)` pairs follow the standard
    // "smaller = closer" convention. The value is a 1-bit-derived
    // score, not a true metric distance; for these columns recall is
    // the contract, not numerical agreement with fp32. `rerank_mult`
    // is intentionally ignored — there's nothing to refine.
    if matches!(col.rerank_codec, RerankCodec::RabitqOnly) {
        let _ = rerank_mult;
        shortlist.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        return ShortlistOutcome::Done(
            shortlist
                .into_iter()
                .map(|(did, est, _pos, _c)| (did, -est))
                .collect(),
        );
    }

    // Build lightweight rerank references into the cluster blocks
    // already in hand — no second fetch and no survivor byte packing.
    // Each block's `full_chunk` follows its `[codes][doc_ids]` prefix;
    // the candidate at cluster-order position `pos` lives at in-block
    // offset `cnt*cb + cnt*4 + local*stride`.
    let mut block_by_cid: HashMap<u32, usize> = HashMap::with_capacity(cluster_meta.len());
    for (bi, &(c, _, _)) in cluster_meta.iter().enumerate() {
        block_by_cid.insert(c as u32, bi);
    }
    let stride = full_vec_bytes;
    let mut candidates = Vec::with_capacity(shortlist.len());
    let mut survivor_full_ranges = if survivor_only_rerank_fetch {
        Some(Vec::with_capacity(shortlist.len()))
    } else {
        None
    };
    for &(did, _, pos, cluster_id) in &shortlist {
        let bi = block_by_cid[&cluster_id];
        let (_, off, cnt) = cluster_meta[bi];
        let full_start = (cnt as usize) * cb + (cnt as usize) * 4;
        let local = (pos - off) as usize;
        let full_idx = if let Some(ranges) = survivor_full_ranges.as_mut() {
            let idx = ranges.len();
            ranges.push(col.cluster_rerank_row_range(off, cnt, local));
            Some(idx)
        } else {
            None
        };
        candidates.push(RerankCandidate {
            did,
            pos,
            cluster_id,
            block_idx: bi,
            full_off: full_start + local * stride,
            full_idx,
        });
    }
    ShortlistOutcome::Rerank {
        candidates,
        survivor_full_ranges,
    }
}

/// Score `query` against every centroid in `centroids_bytes` and
/// return the top `nprobe` `(cluster_id, distance)` pairs sorted by
/// ascending distance (closest first).
///
/// Takes a `&[u8]` view so the caller can hand in either an
/// in-memory subsection slice or the just-fetched centroids
/// region bytes from [`Source::get_range`] — both reach this
/// helper through the same shape.
#[inline]
fn score_centroids(
    centroids_bytes: &[u8],
    col: &ColumnReader,
    query: &[f32],
    nprobe: usize,
) -> Vec<(usize, f32)> {
    // Centroids are stored as fp32 regardless of the column's rerank
    // codec — only the per-doc `full[]` region compresses. `distance_bytes`
    // assumes fp32, which is correct here.
    let centroid_stride = col.dim * 4;
    let mut scores: Vec<(usize, f32)> = (0..col.n_cent as usize)
        .map(|c| {
            let bytes = &centroids_bytes[c * centroid_stride..(c + 1) * centroid_stride];
            (c, distance_bytes(col.metric, query, bytes))
        })
        .collect();
    if nprobe < scores.len() {
        scores.select_nth_unstable_by(nprobe, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        });
        scores.truncate(nprobe);
    }
    scores.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    scores
}

/// Minimum candidate-pool size before per-query scans (coarse 1-bit
/// scoring and rerank) switch from a serial loop to a rayon parallel
/// scan. Below this the fixed rayon dispatch cost outweighs the
/// multicore speedup, so small queries — notably the 1M single-
/// superfile nprobe=1 hot path — stay serial, while the 10M
/// supertable's `nprobe × superfiles` fan-out goes parallel.
const PARALLEL_SCAN_MIN: usize = 2048;

/// Number of chunks to split a parallel rayon scan into — the machine's
/// logical parallelism, capped by the item count so we never make more
/// chunks than there is work.
fn parallel_chunks(n_items: usize) -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .min(n_items)
        .max(1)
}

/// Map `f` over `items` on the global rayon pool, preserving input
/// order. The order-independent vector scans (rerank) use this; the
/// compute runs on rayon (`par_iter().map().collect()`) bridged back to
/// the async caller via a oneshot, so no tokio worker blocks under it.
/// `f` and the items must be `'static` so the work can move onto rayon.
async fn par_map<T, R, F>(items: Vec<T>, f: F) -> Vec<R>
where
    T: Send + Sync + 'static,
    R: Send + 'static,
    F: Fn(&T) -> R + Send + Sync + 'static,
{
    if parallel_chunks(items.len()) <= 1 {
        return items.iter().map(&f).collect();
    }
    let (tx, rx) = tokio::sync::oneshot::channel();
    rayon::spawn(move || {
        let out: Vec<R> = items.par_iter().map(f).collect();
        let _ = tx.send(out);
    });
    rx.await.expect("rerank rayon task dropped result")
}

#[inline]
fn score_cluster_codes_into_heap(
    cluster_codes: &[u8],
    cluster_doc_ids: &[u8],
    cnt: u32,
    off: u32,
    cluster_id: u32,
    quant: &BitQuantizer,
    q_rot: &[f32],
    out: &mut BoundedCoarseHeap,
) {
    let cb = quant.code_bytes();
    let q_total: f32 = q_rot.iter().sum();
    for i in 0..cnt as usize {
        let code = &cluster_codes[i * cb..(i + 1) * cb];
        let est = quant.estimate_dot_rotated_with_total(q_rot, code, q_total);
        let did = u32::from_le_bytes([
            cluster_doc_ids[i * 4],
            cluster_doc_ids[i * 4 + 1],
            cluster_doc_ids[i * 4 + 2],
            cluster_doc_ids[i * 4 + 3],
        ]);
        out.push(CoarseCandidate {
            did,
            estimate: est,
            pos: off + i as u32,
            cluster_id,
        });
    }
}

#[derive(Clone, Copy, Debug)]
struct CoarseCandidate {
    did: u32,
    estimate: f32,
    pos: u32,
    cluster_id: u32,
}

impl PartialEq for CoarseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.estimate == other.estimate
            && self.did == other.did
            && self.pos == other.pos
            && self.cluster_id == other.cluster_id
    }
}

impl Eq for CoarseCandidate {}

impl PartialOrd for CoarseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CoarseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap. Reverse estimate ordering so `peek()`
        // is the worst retained candidate; higher estimates are better.
        other
            .estimate
            .partial_cmp(&self.estimate)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.did.cmp(&self.did))
            .then_with(|| other.pos.cmp(&self.pos))
            .then_with(|| other.cluster_id.cmp(&self.cluster_id))
    }
}

struct BoundedCoarseHeap {
    limit: usize,
    heap: BinaryHeap<CoarseCandidate>,
}

impl BoundedCoarseHeap {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit.max(1)),
        }
    }

    #[inline]
    fn push(&mut self, candidate: CoarseCandidate) {
        if self.limit == 0 {
            return;
        }
        if self.heap.len() < self.limit {
            self.heap.push(candidate);
            return;
        }
        if self
            .heap
            .peek()
            .is_some_and(|worst| candidate.estimate > worst.estimate)
        {
            let mut worst = self
                .heap
                .peek_mut()
                .expect("heap is non-empty because len == limit");
            *worst = candidate;
        }
    }

    fn merge(&mut self, other: BoundedCoarseHeap) {
        for candidate in other.heap {
            self.push(candidate);
        }
    }

    fn into_vec(self) -> Vec<(u32, f32, u32, u32)> {
        self.heap
            .into_iter()
            .map(|candidate| {
                (
                    candidate.did,
                    candidate.estimate,
                    candidate.pos,
                    candidate.cluster_id,
                )
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
struct RerankCandidate {
    did: u32,
    pos: u32,
    cluster_id: u32,
    block_idx: usize,
    full_off: usize,
    full_idx: Option<usize>,
}

#[inline]
fn candidate_full_bytes<'a>(
    blocks: &'a [Bytes],
    survivor_full_rows: Option<&'a [Bytes]>,
    cand: &RerankCandidate,
    stride: usize,
) -> &'a [u8] {
    if let (Some(rows), Some(idx)) = (survivor_full_rows, cand.full_idx) {
        return &rows[idx];
    }
    &blocks[cand.block_idx][cand.full_off..cand.full_off + stride]
}

/// Decode one cluster's `(off, cnt)` entry from
/// `cluster_idx_slice` (the `n_cent × 8` bytes of the column's
/// cluster index header). `c` is the cluster id.
#[inline]
fn read_cluster_entry(cluster_idx_slice: &[u8], c: usize) -> (u32, u32) {
    let base = c * 8;
    let off = u32::from_le_bytes([
        cluster_idx_slice[base],
        cluster_idx_slice[base + 1],
        cluster_idx_slice[base + 2],
        cluster_idx_slice[base + 3],
    ]);
    let cnt = u32::from_le_bytes([
        cluster_idx_slice[base + 4],
        cluster_idx_slice[base + 5],
        cluster_idx_slice[base + 6],
        cluster_idx_slice[base + 7],
    ]);
    (off, cnt)
}

/// Full-precision rerank over `shortlist`, returning the top-`k`
/// `(doc_id, distance)` pairs sorted by ascending distance.
///
/// `candidates` points into the already-fetched per-cluster blocks:
/// each entry carries `(block_idx, full_off)` for its `full[]` row.
/// That avoids allocating and copying a packed survivor buffer on
/// every query while still keeping rerank byte lookup O(1).
///
/// Dispatches on `col.rerank_codec`:
/// - **Fp32**: flat dispatch via [`distance_bytes_codec`]
///   (fp32 zero-copy SIMD).
/// - **Sq8**: builds a per-query [`Sq8Kernel`] from the column's
///   `codec_meta` once (folds scale/offset into the query so the
///   per-doc inner step is a plain u8→f32 widen + SIMD dot;
///   per-doc decoded-norm cached at encode time short-circuits
///   `Σx²` for L2Sq).
async fn rerank_candidates_from_blocks(
    source: &Source,
    lazy_sq8_meta_bytes: Option<&Bytes>,
    cluster_blocks: &[Bytes],
    survivor_full_rows: Option<&[Bytes]>,
    candidates: &[RerankCandidate],
    col: &ColumnReader,
    query: &[f32],
    k: usize,
) -> Result<Vec<(u32, f32)>, crate::superfile::lazy_source::LazyByteSourceError> {
    let stride = col.rerank_codec.per_vector_bytes(col.dim);
    let mut reranked: Vec<(u32, f32)> = match col.rerank_codec {
        RerankCodec::Fp32 => {
            // Exact fp32 rerank — every survivor is independent, so the
            // gather + SIMD distance runs in parallel across the rayon
            // pool once the shortlist is large enough to amortize the
            // hand-off. The output is sorted by distance below, so
            // parallel and serial rank identically.
            if candidates.len() >= PARALLEL_SCAN_MIN {
                let metric = col.metric;
                let codec = col.rerank_codec;
                let blocks: Arc<Vec<Bytes>> = Arc::new(cluster_blocks.to_vec());
                let survivors: Option<Arc<Vec<Bytes>>> =
                    survivor_full_rows.map(|s| Arc::new(s.to_vec()));
                let query: Arc<Vec<f32>> = Arc::new(query.to_vec());
                par_map(candidates.to_vec(), move |cand: &RerankCandidate| {
                    let bytes = candidate_full_bytes(
                        &blocks,
                        survivors.as_deref().map(|s| s.as_slice()),
                        cand,
                        stride,
                    );
                    (cand.did, distance_bytes_codec(metric, codec, &query, bytes))
                })
                .await
            } else {
                candidates
                    .iter()
                    .map(|cand| {
                        let bytes =
                            candidate_full_bytes(cluster_blocks, survivor_full_rows, cand, stride);
                        (
                            cand.did,
                            distance_bytes_codec(col.metric, col.rerank_codec, query, bytes),
                        )
                    })
                    .collect()
            }
        }
        RerankCodec::Sq8Residual => {
            let meta = col
                .sq8_meta
                .as_ref()
                .expect("Sq8Residual column must carry sq8_meta (built in open_with)");
            let dim = col.dim;
            // `Sq8Residual` stores `[code dim u8 ‖ residual dim i8]`
            // per vector (`stride == 2·dim`); the first `dim` bytes
            // are the Sq8 code leg the shortlist scoring reads.
            match meta {
                Sq8ColumnMeta::Eager {
                    scale,
                    offset,
                    per_doc_norms,
                } => {
                    sq8_score_and_refine(
                        candidates,
                        cluster_blocks,
                        survivor_full_rows,
                        col,
                        query,
                        scale,
                        offset,
                        per_doc_norms.clone(),
                        k,
                        stride,
                    )
                    .await
                }
                Sq8ColumnMeta::Lazy {
                    scale_abs_off,
                    offset_abs_off,
                    norms_abs_off,
                } => {
                    if let Some(meta_bytes) = lazy_sq8_meta_bytes {
                        let parsed = Arc::clone(col.lazy_sq8_parsed.get_or_init(|| {
                            Arc::new(parse_sq8_meta_bytes(
                                meta_bytes,
                                col.n_cent as usize,
                                dim,
                                col.n_docs as usize,
                                norms_abs_off.is_some(),
                            ))
                        }));
                        return Ok(sq8_score_and_refine(
                            candidates,
                            cluster_blocks,
                            survivor_full_rows,
                            col,
                            query,
                            parsed.scale.as_slice(),
                            parsed.offset.as_slice(),
                            parsed.per_doc_norms.clone(),
                            k,
                            stride,
                        )
                        .await);
                    }
                    let mut clusters: Vec<u32> = candidates.iter().map(|c| c.cluster_id).collect();
                    clusters.sort_unstable();
                    clusters.dedup();

                    let cluster_meta_len = dim * 4;
                    let mut ranges = Vec::with_capacity(clusters.len() * 2);
                    for &cluster_id in &clusters {
                        let c = cluster_id as usize;
                        let scale_start = *scale_abs_off + c * cluster_meta_len;
                        let offset_start = *offset_abs_off + c * cluster_meta_len;
                        ranges.push(scale_start..scale_start + cluster_meta_len);
                        ranges.push(offset_start..offset_start + cluster_meta_len);
                    }
                    let bytes = source.get_ranges_parallel(&ranges)?;
                    let mut scale_offset_by_cluster: HashMap<u32, (Vec<f32>, Vec<f32>)> =
                        HashMap::with_capacity(clusters.len());
                    for (idx, &cluster_id) in clusters.iter().enumerate() {
                        let scale = parse_f32_le_vec(&bytes[idx * 2]);
                        let offset = parse_f32_le_vec(&bytes[idx * 2 + 1]);
                        scale_offset_by_cluster.insert(cluster_id, (scale, offset));
                    }

                    let norm_by_pos = if let Some(norms_abs_off) = norms_abs_off {
                        let mut spans: HashMap<u32, (u32, u32)> = HashMap::new();
                        for cand in candidates {
                            spans
                                .entry(cand.cluster_id)
                                .and_modify(|(lo, hi)| {
                                    *lo = (*lo).min(cand.pos);
                                    *hi = (*hi).max(cand.pos);
                                })
                                .or_insert((cand.pos, cand.pos));
                        }
                        let mut span_items: Vec<(u32, u32, u32)> = spans
                            .into_iter()
                            .map(|(cluster_id, (lo, hi))| (cluster_id, lo, hi))
                            .collect();
                        span_items.sort_unstable_by_key(|&(cluster_id, _, _)| cluster_id);
                        let norm_ranges: Vec<Range<usize>> = span_items
                            .iter()
                            .map(|&(_, lo, hi)| {
                                let start = *norms_abs_off + lo as usize * 4;
                                start..start + (hi - lo + 1) as usize * 4
                            })
                            .collect();
                        let norm_bytes = source.get_ranges_parallel(&norm_ranges)?;
                        let mut out = HashMap::new();
                        for ((_, lo, hi), bytes) in span_items.into_iter().zip(norm_bytes) {
                            let vals = parse_f32_le_vec(&bytes);
                            for (i, pos) in (lo..=hi).enumerate() {
                                out.insert(pos, vals[i]);
                            }
                        }
                        Some(out)
                    } else {
                        None
                    };

                    let scored: Vec<(u32, f32, usize, u32, u32)> = candidates
                        .iter()
                        .enumerate()
                        .map(|(i, cand)| {
                            let row = candidate_full_bytes(
                                cluster_blocks,
                                survivor_full_rows,
                                cand,
                                stride,
                            );
                            let code = &row[..dim];
                            let (scale, offset) = scale_offset_by_cluster
                                .get(&cand.cluster_id)
                                .expect("cluster metadata fetched");
                            let kernel = Sq8Kernel::new(
                                col.metric,
                                query,
                                scale.as_slice(),
                                offset.as_slice(),
                                None,
                            );
                            let norm = norm_by_pos.as_ref().and_then(|m| m.get(&cand.pos).copied());
                            (
                                cand.did,
                                kernel.distance_with_norm(code, norm),
                                i,
                                cand.pos,
                                cand.cluster_id,
                            )
                        })
                        .collect();
                    // Refine the top final-set with the residual leg.
                    // The residual kernel takes its per-doc norm
                    // explicitly because the lazy norms live in a
                    // sparse `pos → norm` map, not a contiguous slice.
                    let mut scored = scored;
                    scored
                        .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    let final_refine = k
                        .saturating_mul(SQ8_RESIDUAL_REFINE_MULT)
                        .max(k)
                        .min(scored.len());
                    scored.truncate(final_refine);
                    let mut rk: HashMap<u32, Sq8ResidualKernel> = HashMap::new();
                    scored
                        .into_iter()
                        .map(|(did, _, i, pos, cluster_id)| {
                            let row = candidate_full_bytes(
                                cluster_blocks,
                                survivor_full_rows,
                                &candidates[i],
                                stride,
                            );
                            let code = &row[..dim];
                            let residual = &row[dim..dim * 2];
                            let kernel = rk.entry(cluster_id).or_insert_with(|| {
                                let (scale, offset) = scale_offset_by_cluster
                                    .get(&cluster_id)
                                    .expect("cluster metadata fetched");
                                Sq8ResidualKernel::new(
                                    col.metric,
                                    query,
                                    scale.as_slice(),
                                    offset.as_slice(),
                                    SQ8_RESIDUAL_DIVISOR,
                                    None,
                                )
                            });
                            let norm = norm_by_pos.as_ref().and_then(|m| m.get(&pos).copied());
                            (did, kernel.distance_with_norm(code, residual, norm))
                        })
                        .collect()
                }
            }
        }
        RerankCodec::RabitqOnly => unreachable!(
            "rerank_candidates_in_run reached with None codec — None columns \
             have no full[] region and should short-circuit before the rerank step"
        ),
    };
    reranked.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    reranked.truncate(k);
    Ok(reranked)
}

/// Shared Sq8 first-pass scorer used by both the eager and
/// lazy-with-parsed-cache arms of `rerank_candidates_from_blocks`.
/// Builds one [`Sq8Kernel`] per distinct probed cluster from the
/// provided `scale`/`offset` slices, scores every candidate (parallel
/// when the shortlist exceeds [`PARALLEL_SCAN_MIN`]), then applies the
/// residual refinement via [`residual_refine_from_blocks`].
///
/// Both code paths keep their own data-access strategy (eager mmap vs
/// lazy range GETs); only the scoring math is shared here.
async fn sq8_score_and_refine(
    candidates: &[RerankCandidate],
    cluster_blocks: &[Bytes],
    survivor_full_rows: Option<&[Bytes]>,
    col: &ColumnReader,
    query: &[f32],
    scale: &[f32],
    offset: &[f32],
    per_doc_norms: Option<Arc<[f32]>>,
    k: usize,
    stride: usize,
) -> Vec<(u32, f32)> {
    let dim = col.dim;
    let mut cids: Vec<u32> = candidates.iter().map(|c| c.cluster_id).collect();
    cids.sort_unstable();
    cids.dedup();
    let kernels: HashMap<u32, Sq8Kernel> = cids
        .into_iter()
        .map(|cid| {
            let c = cid as usize;
            let scale_c = &scale[c * dim..(c + 1) * dim];
            let offset_c = &offset[c * dim..(c + 1) * dim];
            (
                cid,
                Sq8Kernel::new(col.metric, query, scale_c, offset_c, per_doc_norms.clone()),
            )
        })
        .collect();
    let score_one = |(i, cand): (usize, &RerankCandidate)| {
        let row = candidate_full_bytes(cluster_blocks, survivor_full_rows, cand, stride);
        let code = &row[..dim];
        let kernel = kernels
            .get(&cand.cluster_id)
            .expect("kernel prebuilt for every probed cluster");
        (
            cand.did,
            kernel.distance_at(cand.pos, code),
            i,
            cand.pos,
            cand.cluster_id,
        )
    };
    let scored: Vec<(u32, f32, usize, u32, u32)> = if candidates.len() >= PARALLEL_SCAN_MIN {
        // Order-independent first-pass Sq8 scoring across the rayon
        // pool. Kernels are `'static` (norms shared by `Arc`), so each
        // chunk runs on a rayon worker with no copy.
        let kernels = Arc::new(kernels);
        let blocks: Arc<Vec<Bytes>> = Arc::new(cluster_blocks.to_vec());
        let survivors: Option<Arc<Vec<Bytes>>> = survivor_full_rows.map(|s| Arc::new(s.to_vec()));
        let items: Vec<(usize, RerankCandidate)> = candidates.iter().cloned().enumerate().collect();
        par_map(items, move |item: &(usize, RerankCandidate)| {
            let (i, cand) = (item.0, &item.1);
            let row = candidate_full_bytes(
                &blocks,
                survivors.as_deref().map(|s| s.as_slice()),
                cand,
                stride,
            );
            let code = &row[..dim];
            let kernel = kernels
                .get(&cand.cluster_id)
                .expect("kernel prebuilt for every probed cluster");
            (
                cand.did,
                kernel.distance_at(cand.pos, code),
                i,
                cand.pos,
                cand.cluster_id,
            )
        })
        .await
    } else {
        candidates.iter().enumerate().map(score_one).collect()
    };
    residual_refine_from_blocks(
        scored,
        cluster_blocks,
        survivor_full_rows,
        candidates,
        stride,
        dim,
        k,
        |cluster_id| {
            let c = cluster_id as usize;
            Sq8ResidualKernel::new(
                col.metric,
                query,
                &scale[c * dim..(c + 1) * dim],
                &offset[c * dim..(c + 1) * dim],
                SQ8_RESIDUAL_DIVISOR,
                per_doc_norms.as_deref(),
            )
        },
    )
}

/// `Sq8Residual` final-refine pass. Takes the Sq8-scored shortlist
/// (`(did, sq8_dist, candidate_idx, pos, cluster_id)`), keeps the lowest
/// `2·k` by Sq8 distance, then re-scores just that set with the
/// residual-corrected [`Sq8ResidualKernel`] (built per cluster via
/// `make_kernel`). The candidate index points into `candidates`,
/// whose row bytes are read directly from `cluster_blocks`.
fn residual_refine_from_blocks<'a>(
    mut scored: Vec<(u32, f32, usize, u32, u32)>,
    cluster_blocks: &[Bytes],
    survivor_full_rows: Option<&[Bytes]>,
    candidates: &[RerankCandidate],
    stride: usize,
    dim: usize,
    k: usize,
    make_kernel: impl Fn(u32) -> Sq8ResidualKernel<'a>,
) -> Vec<(u32, f32)> {
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let final_refine = k
        .saturating_mul(SQ8_RESIDUAL_REFINE_MULT)
        .max(k)
        .min(scored.len());
    scored.truncate(final_refine);
    let mut rk: HashMap<u32, Sq8ResidualKernel> = HashMap::new();
    scored
        .into_iter()
        .map(|(did, _, i, pos, cluster_id)| {
            let row =
                candidate_full_bytes(cluster_blocks, survivor_full_rows, &candidates[i], stride);
            let code = &row[..dim];
            let residual = &row[dim..dim * 2];
            let kernel = rk
                .entry(cluster_id)
                .or_insert_with(|| make_kernel(cluster_id));
            (did, kernel.distance_at(pos, code, residual))
        })
        .collect()
}

fn parse_sq8_meta_bytes(
    bytes: &[u8],
    n_cent: usize,
    dim: usize,
    n_docs: usize,
    has_norms: bool,
) -> Sq8ParsedMeta {
    let so_block_bytes = n_cent * dim * 4;
    let scale_end = so_block_bytes;
    let offset_end = scale_end + so_block_bytes;
    let scale = parse_f32_le_vec(&bytes[0..scale_end]);
    let offset = parse_f32_le_vec(&bytes[scale_end..offset_end]);
    let per_doc_norms = has_norms.then(|| {
        let norms_end = offset_end + n_docs * 4;
        Arc::from(parse_f32_le_vec(&bytes[offset_end..norms_end]))
    });
    Sq8ParsedMeta {
        scale,
        offset,
        per_doc_norms,
    }
}

#[inline]
fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Decode an aligned-or-not `&[u8]` of length `4·N` as a
/// `Vec<f32>` of length `N`. Used for Sq8's `codec_meta` arrays
/// (scale, offset, per-doc norms) where the byte slice can land
/// at any alignment relative to the `Bytes` backing — see the
/// reader-side note where this is called for the alignment
/// argument. Slow path (4 byte reads per f32) but only runs at
/// open time over at-most-`8·dim + 4·n_docs` bytes per Sq8
/// column; the per-query inner loop never goes through here.
#[inline]
fn parse_f32_le_vec(bytes: &[u8]) -> Vec<f32> {
    debug_assert!(bytes.len().is_multiple_of(4));
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

#[inline]
fn read_u64_le(b: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[0..8]);
    u64::from_le_bytes(buf)
}

const CLUSTER_RANGE_COALESCE_MAX_GAP: usize = 64 * 1024;
const CLUSTER_RANGE_COALESCE_MAX_OVERFETCH: usize = 512 * 1024;

fn lazy_sq8_meta_range(col: &ColumnReader) -> Option<Range<usize>> {
    let Sq8ColumnMeta::Lazy { scale_abs_off, .. } = col.sq8_meta.as_ref()? else {
        return None;
    };
    let scale_offset_bytes = 2 * (col.n_cent as usize) * col.dim * 4;
    let norm_bytes = if matches!(col.metric, Metric::L2Sq | Metric::Cosine) {
        (col.n_docs as usize) * 4
    } else {
        0
    };
    Some(*scale_abs_off..*scale_abs_off + scale_offset_bytes + norm_bytes)
}

/// Coalescing plan for a set of cluster-block ranges. Computed once
/// and applied to either a sync ([`Source::get_ranges_parallel`]) or
/// async ([`Source::get_ranges_parallel_async`]) batch fetch — the
/// byte selection is identical regardless of which path executes it,
/// so the sync and async vector kernels return bit-identical results.
struct CoalescePlan {
    /// Merged ranges to actually GET (adjacent / near-adjacent
    /// cluster blocks fused into one request to cut the GET count).
    fetch_ranges: Vec<Range<usize>>,
    /// Per original input range, in input order: `(fetch_idx,
    /// offset_in_fetch, len)` — how to slice the requested range
    /// back out of its merged fetch block.
    scatter: Vec<(usize, usize, usize)>,
}

/// Group `ranges` into coalesced fetch spans (same gap/overfetch rule
/// the per-cluster cold fan-out has always used) and record how to
/// slice each original range back out. Pure; no I/O.
fn plan_cluster_coalesce(ranges: &[Range<usize>]) -> CoalescePlan {
    let mut sorted: Vec<(usize, Range<usize>)> = ranges.iter().cloned().enumerate().collect();
    sorted.sort_unstable_by_key(|(_, r)| (r.start, r.end));

    // groups: (merged_range, payload_len, members[(orig_idx, range)])
    let mut groups: Vec<(Range<usize>, usize, Vec<(usize, Range<usize>)>)> = Vec::new();
    for (idx, range) in sorted {
        if let Some((merged, payload_len, members)) = groups.last_mut() {
            let gap = range.start.saturating_sub(merged.end);
            let merged_end = merged.end.max(range.end);
            let new_payload_len = *payload_len + range.len();
            let new_overfetch = (merged_end - merged.start).saturating_sub(new_payload_len);
            if range.start <= merged.end
                || (gap <= CLUSTER_RANGE_COALESCE_MAX_GAP
                    && new_overfetch <= CLUSTER_RANGE_COALESCE_MAX_OVERFETCH)
            {
                merged.end = merged_end;
                *payload_len = new_payload_len;
                members.push((idx, range));
                continue;
            }
        }
        groups.push((range.clone(), range.len(), vec![(idx, range)]));
    }

    let fetch_ranges: Vec<Range<usize>> = groups.iter().map(|(r, _, _)| r.clone()).collect();
    let mut scatter: Vec<(usize, usize, usize)> = vec![(0, 0, 0); ranges.len()];
    for (gi, (merged_range, _, members)) in groups.iter().enumerate() {
        for (idx, range) in members {
            scatter[*idx] = (gi, range.start - merged_range.start, range.len());
        }
    }
    CoalescePlan {
        fetch_ranges,
        scatter,
    }
}

/// Slice the requested ranges back out of the fetched merged blocks.
fn apply_coalesce(plan: &CoalescePlan, fetched: &[Bytes]) -> Vec<Bytes> {
    plan.scatter
        .iter()
        .map(|&(gi, off, len)| fetched[gi].slice(off..off + len))
        .collect()
}

fn get_cluster_ranges_coalesced_with_extra(
    source: &Source,
    ranges: &[Range<usize>],
    extra: Option<Range<usize>>,
) -> Result<(Vec<Bytes>, Option<Bytes>), crate::superfile::lazy_source::LazyByteSourceError> {
    let Some(extra) = extra else {
        return Ok((get_cluster_ranges_coalesced(source, ranges)?, None));
    };
    let plan = plan_cluster_coalesce(ranges);
    let mut fetch = plan.fetch_ranges.clone();
    fetch.push(extra);
    let mut fetched = source.get_ranges_parallel(&fetch)?;
    let extra_bytes = fetched.pop();
    Ok((apply_coalesce(&plan, &fetched), extra_bytes))
}

/// Async sibling of [`get_cluster_ranges_coalesced_with_extra`]. Same
/// coalescing plan, dispatched as one `try_join_all` batch on the
/// caller's runtime so connections pool and the fan-out is concurrent.
async fn get_cluster_ranges_coalesced_with_extra_async(
    source: &Source,
    ranges: &[Range<usize>],
    extra: Option<Range<usize>>,
) -> Result<(Vec<Bytes>, Option<Bytes>), crate::superfile::lazy_source::LazyByteSourceError> {
    let Some(extra) = extra else {
        return Ok((
            get_cluster_ranges_coalesced_async(source, ranges).await?,
            None,
        ));
    };
    let plan = plan_cluster_coalesce(ranges);
    let mut fetch = plan.fetch_ranges.clone();
    fetch.push(extra);
    let mut fetched = source.get_ranges_parallel_async(&fetch).await?;
    let extra_bytes = fetched.pop();
    Ok((apply_coalesce(&plan, &fetched), extra_bytes))
}

fn get_cluster_ranges_coalesced(
    source: &Source,
    ranges: &[Range<usize>],
) -> Result<Vec<Bytes>, crate::superfile::lazy_source::LazyByteSourceError> {
    if ranges.is_empty() {
        return Ok(Vec::new());
    }
    if ranges.len() == 1 {
        return source.get_ranges_parallel(ranges);
    }
    let plan = plan_cluster_coalesce(ranges);
    let fetched = source.get_ranges_parallel(&plan.fetch_ranges)?;
    Ok(apply_coalesce(&plan, &fetched))
}

/// Async sibling of [`get_cluster_ranges_coalesced`].
async fn get_cluster_ranges_coalesced_async(
    source: &Source,
    ranges: &[Range<usize>],
) -> Result<Vec<Bytes>, crate::superfile::lazy_source::LazyByteSourceError> {
    if ranges.is_empty() {
        return Ok(Vec::new());
    }
    if ranges.len() == 1 {
        return source.get_ranges_parallel_async(ranges).await;
    }
    let plan = plan_cluster_coalesce(ranges);
    let fetched = source.get_ranges_parallel_async(&plan.fetch_ranges).await?;
    Ok(apply_coalesce(&plan, &fetched))
}

/// Best-effort sync byte fetch with a typed error. Used throughout
/// `open_with` so every byte access goes through the `Source`
/// abstraction — the lazy variant plumbs the eager-prefetch
/// path through the same call sites without a second rewrite.
///
/// Failure mode here means the range is out-of-bounds or not
/// present in the sync cache. On `Source::InMemory`,
/// any in-bounds range succeeds zero-copy; this only fires on a
/// malformed blob today.
#[inline]
fn fetch_sync(source: &Source, range: Range<usize>, what: &str) -> Result<Bytes, VectorError> {
    let start = range.start;
    let end = range.end;
    source.try_get_range_sync(range).ok_or_else(|| {
        VectorError::Read(ReadError::MalformedVersion(format!(
            "vector {what} range {start}..{end} past blob"
        )))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::vector::builder::{VectorBuilder, VectorConfig};

    fn build_blob(n_docs: u32, dim: usize, n_cent: usize, metric: Metric) -> (Bytes, String) {
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "embedding".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric,
            rerank_codec: RerankCodec::Fp32,
        })
        .expect("register column");
        for i in 0..n_docs {
            // Deterministic "random" vector.
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i.wrapping_mul(31) + j as u32) % 100) as f32 * 0.01)
                .collect();
            b.add(0, &v).expect("add to vector builder");
        }
        let bytes = b.finish().expect("finish vector builder");
        let metric_s = match metric {
            Metric::L2Sq => "l2sq",
            Metric::Cosine => "cosine",
            Metric::NegDot => "negdot",
        };
        let json = format!(
            r#"[{{"column":"embedding","dim":{dim},"n_cent":{n_cent},"rot_seed":7,"metric":"{metric_s}"}}]"#
        );
        (Bytes::from(bytes), json)
    }

    #[test]
    fn open_accepts_valid_blob() {
        let (blob, json) = build_blob(64, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open should succeed");
        assert_eq!(r.n_docs(), 64);
        let cols: Vec<&str> = r.vector_columns().collect();
        assert_eq!(cols, vec!["embedding"]);
    }

    #[test]
    fn open_rejects_bad_magic() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let mut bytes = blob.to_vec();
        bytes[0] = b'X';
        let err = VectorReader::open(Bytes::from(bytes), &json).expect_err("expected error");
        assert!(matches!(err, VectorError::Read(ReadError::BadMagic { .. })));
    }

    #[test]
    fn open_rejects_short_blob() {
        let err = VectorReader::open(Bytes::from(vec![0u8; 8]), "[]").expect_err("expected error");
        assert!(matches!(err, VectorError::Read(_)));
    }

    #[test]
    fn open_detects_corruption_via_outer_crc() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let mut bytes = blob.to_vec();
        // Flip a byte in the middle of the directory area.
        let pos = OUTER_HEADER_SIZE + 10;
        bytes[pos] ^= 0xFF;
        let err = VectorReader::open(Bytes::from(bytes), &json).expect_err("expected error");
        assert!(matches!(
            err,
            VectorError::Read(ReadError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn open_with_skip_crc_accepts_corrupted_directory_bytes() {
        // The fast-open path explicitly skips CRC verification — so
        // a flipped byte in the directory area opens cleanly. The
        // caller is responsible for upstream integrity.
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let mut bytes = blob.to_vec();
        let pos = OUTER_HEADER_SIZE + 10;
        bytes[pos] ^= 0xFF;
        let r =
            VectorReader::open_with(Bytes::from(bytes), &json, OpenOptions { verify_crc: false });
        // Open succeeds; the corruption may surface later as a
        // bad-magic / bounds error or be silently absorbed depending
        // on which byte got flipped. The contract is "we don't
        // verify"; not "we'll always read sensibly."
        let _ = r;
    }

    #[test]
    fn open_with_default_options_matches_open() {
        // Sanity: default opts produce identical results to `open`.
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r1 = VectorReader::open(blob.clone(), &json).expect("open VectorReader");
        let r2 = VectorReader::open_with(blob, &json, OpenOptions::default())
            .expect("open VectorReader");
        assert_eq!(r1.n_docs(), r2.n_docs());
    }

    #[test]
    fn public_rerank_mult_honors_requested_value() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let fp32 = VectorReader::open(blob, &json).expect("open fp32 VectorReader");
        assert_eq!(fp32.public_rerank_mult("embedding", 4), 4);

        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "embedding".into(),
            dim: 16,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
        })
        .expect("register Sq8 column");
        for i in 0..32u32 {
            let v: Vec<f32> = (0..16)
                .map(|j| ((i.wrapping_mul(31) + j as u32) % 100) as f32 * 0.01)
                .collect();
            b.add(0, &v).expect("add to vector builder");
        }
        let sq8 = VectorReader::open(
            Bytes::from(b.finish().expect("finish Sq8 vector builder")),
            r#"[{"column":"embedding","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#,
        )
        .expect("open Sq8 VectorReader");
        assert_eq!(sq8.public_rerank_mult("embedding", 4), 4);
    }

    #[tokio::test]
    async fn search_self_query_returns_self_as_top1() {
        let dim = 16;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "embedding".into(),
            dim,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
        })
        .expect("register column");
        let mut all_vecs = Vec::new();
        for i in 0..64u32 {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i.wrapping_mul(13) + j as u32 * 5) % 100) as f32)
                .collect();
            b.add(0, &v).expect("add to vector builder");
            all_vecs.push(v);
        }
        let bytes = b.finish().expect("finish vector builder");
        let json = r#"[{"column":"embedding","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#;
        let r = VectorReader::open(Bytes::from(bytes), json).expect("open VectorReader");

        // Pick a doc, query with its own vector → top-1 is self with distance 0.
        let target = 17;
        let hits = r
            .search("embedding", &all_vecs[target], 5, 4, 5)
            .await
            .expect("FTS search");
        assert!(!hits.is_empty(), "search should return hits");
        assert_eq!(hits[0].0, target as u32, "self should be nearest");
        assert!(
            hits[0].1 < 1e-3,
            "self distance should be ~0, got {}",
            hits[0].1
        );
    }

    #[tokio::test]
    async fn search_unknown_column_errors() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let err = r
            .search("nonexistent", &[0.0; 16], 5, 4, 5)
            .await
            .expect_err("expected error");
        assert!(matches!(err, VectorError::UnknownColumn(_)));
    }

    #[tokio::test]
    async fn search_dim_mismatch_errors() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let err = r
            .search("embedding", &[0.0; 8], 5, 4, 5)
            .await
            .expect_err("expected error");
        assert!(matches!(err, VectorError::DimensionMismatch { .. }));
    }

    #[tokio::test]
    async fn search_zero_k_returns_empty() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let hits = r
            .search("embedding", &[0.0; 16], 0, 4, 5)
            .await
            .expect("FTS search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_results_sorted_ascending_by_distance() {
        let (blob, json) = build_blob(64, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let q = vec![0.5; 16];
        let hits = r
            .search("embedding", &q, 10, 4, 5)
            .await
            .expect("FTS search");
        for w in hits.windows(2) {
            assert!(w[0].1 <= w[1].1, "distances should be ascending");
        }
    }

    #[test]
    fn summary_returns_dim_centroid_and_radius() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open VectorReader");
        let (centroid, radius) = r.summary("embedding").expect("vector summary");
        assert_eq!(centroid.len(), 16);
        assert!(radius >= 0.0);
        assert!(r.summary("nonexistent").is_none());
    }

    #[tokio::test]
    async fn search_clusters_async_probing_all_matches_full_nprobe() {
        // The externally-selected path probing *every* cluster must
        // recover the same top-k set as a full-nprobe `search_async` —
        // same shortlist, same rerank. (Compared as a set: equal
        // distances could tie-break differently across cluster-visit
        // orders.)
        use std::collections::HashSet;
        let (blob, json, all) =
            build_small_superfile(32, 4, 64, RerankCodec::Sq8Residual, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open");
        let q = &all[0];
        let (k, rerank, n_cent) = (5usize, 5usize, 4u32);

        let full = r
            .search_async("v", q, k, n_cent as usize, rerank)
            .await
            .expect("search_async");
        let probed = r
            .search_clusters_async("v", q, k, &(0..n_cent).collect::<Vec<_>>(), rerank)
            .await
            .expect("search_clusters_async");

        assert!(!full.is_empty(), "self-query returns hits");
        assert_eq!(full.len(), probed.len(), "same number of hits");
        let full_ids: HashSet<u32> = full.iter().map(|(id, _)| *id).collect();
        let probed_ids: HashSet<u32> = probed.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            full_ids, probed_ids,
            "probing all clusters must match a full-nprobe search"
        );

        // Probing no clusters returns nothing.
        let none = r
            .search_clusters_async("v", q, k, &[], rerank)
            .await
            .expect("search_clusters_async empty");
        assert!(none.is_empty(), "probing no clusters returns no hits");
    }

    // -----------------------------------------------------------------
    // Source enum sanity tests
    // -----------------------------------------------------------------
    //
    // The `Source` enum reroutes runtime byte access through
    // it; the eager open path takes a `Bytes`, the lazy path adds
    // `open_lazy`. These tests directly exercise the `Source`
    // contract so any future refactor that breaks the InMemory
    // zero-copy invariant or mis-implements the Lazy wrapper fails
    // here rather than at the wider recall oracle gate.

    #[test]
    fn source_in_memory_try_get_range_sync_zero_copy() {
        let payload = Bytes::from(vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let src = Source::InMemory(payload.clone());
        let slice = src
            .try_get_range_sync(3..7)
            .expect("in-bounds InMemory sync must succeed");
        assert_eq!(slice.as_ref(), &payload[3..7]);
        // Zero-copy invariant: returned Bytes points into the
        // same allocation as the source.
        let expected_ptr = unsafe { payload.as_ptr().add(3) };
        assert_eq!(slice.as_ptr(), expected_ptr);
    }

    #[test]
    fn source_in_memory_try_get_range_sync_out_of_bounds_returns_none() {
        let src = Source::InMemory(Bytes::from(vec![0u8; 4]));
        assert!(src.try_get_range_sync(0..100).is_none());
        assert!(src.try_get_range_sync(8..10).is_none());
    }

    #[test]
    fn source_in_memory_get_range_returns_zero_copy_slice() {
        let payload = Bytes::from(vec![100u8, 101, 102, 103, 104, 105]);
        let src = Source::InMemory(payload.clone());
        let got = src
            .get_range(1..5)
            .expect("InMemory sync get_range always succeeds for in-bounds ranges");
        assert_eq!(got.as_ref(), &payload[1..5]);
    }

    #[test]
    fn source_lazy_try_get_range_sync_falls_through_to_trait_default_or_impl() {
        // Wrap an in-memory blob in the trait-shaped
        // `BytesLazyByteSource`, then in `Source::Lazy`. The lazy
        // adapter's `try_get_range_sync` impl returns `Some` for
        // in-bounds ranges; we exercise the full enum dispatch
        // path here so the Lazy arm of `Source::try_get_range_sync`
        // doesn't drift apart from the InMemory arm.
        use crate::superfile::lazy_source::BytesLazyByteSource;
        let payload = Bytes::from(vec![7u8, 8, 9, 10, 11, 12, 13, 14]);
        let lazy: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(payload.clone()));
        let src = Source::Lazy(lazy);
        let slice = src
            .try_get_range_sync(2..6)
            .expect("BytesLazyByteSource always serves sync");
        assert_eq!(slice.as_ref(), &payload[2..6]);
    }

    #[test]
    fn source_lazy_get_range_serves_warm_cache_via_try_get_range_sync() {
        use crate::superfile::lazy_source::BytesLazyByteSource;
        let payload = Bytes::from(vec![21u8, 22, 23, 24, 25, 26, 27]);
        let lazy: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(payload.clone()));
        let src = Source::Lazy(lazy);
        // BytesLazyByteSource overrides try_get_range_sync to
        // return Some for every in-bounds range, so get_range
        // takes the sync fast path — no block_on bridge fires.
        let got = src.get_range(1..5).expect("warm cache sync hit");
        assert_eq!(got.as_ref(), &payload[1..5]);
    }

    /// `Source::Clone` lets readers share the underlying
    /// state cheaply (refcount bump). Clones must observe
    /// identical bytes — no fork between paths.
    #[test]
    fn source_clone_observes_identical_bytes() {
        let payload = Bytes::from(vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let a = Source::InMemory(payload.clone());
        let b = a.clone();
        let sa = a.try_get_range_sync(2..6).expect("sa");
        let sb = b.try_get_range_sync(2..6).expect("sb");
        assert_eq!(sa.as_ref(), sb.as_ref());
        assert_eq!(sa.as_ptr(), sb.as_ptr());
    }

    #[test]
    fn open_rejects_columns_json_mismatch() {
        let (blob, _) = build_blob(32, 16, 4, Metric::L2Sq);
        // header says 1 column; pass 2-column JSON.
        let bad_json = r#"[{"column":"a","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"},{"column":"b","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#;
        let err = VectorReader::open(blob, bad_json).expect_err("expected error");
        assert!(matches!(
            err,
            VectorError::Read(ReadError::MalformedVersion(_))
        ));
    }

    // -----------------------------------------------------------------
    // rerank-codec discriminator round-trip
    // -----------------------------------------------------------------
    //
    // The codec discriminator rides as byte 52 of the per-column
    // directory entry; the codec_meta region offset rides as bytes
    // 12..16 of the sub-header. Both are zero on older fp32
    // superfiles. `Fp32` / `Sq8` / `RabitqOnly` are wired end-to-end;
    // must still round-trip as a typed `MalformedVersion` at open
    // time so a future superfile built by a newer binary fails loud
    // against an older binary rather than mis-decoding.

    use crate::superfile::format::checksum::crc32c;

    /// a fresh `Fp32` build round-trips through the
    /// reader with `ColumnReader.rerank_codec == Fp32` — the
    /// directory-entry codec byte makes it back out of the on-disk
    /// representation unchanged. The structural assertion pins the
    /// on-disk discriminator contract.
    #[test]
    fn open_round_trips_fp32_codec_discriminator() {
        let (blob, json) = build_blob(64, 16, 4, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open");
        assert_eq!(r.columns.len(), 1);
        assert_eq!(
            r.columns[0].rerank_codec,
            RerankCodec::Fp32,
            "Fp32 build must surface as RerankCodec::Fp32 on the reader"
        );
        assert_eq!(
            r.columns[0].codec_meta_off, 0,
            "Fp32 superfiles must write codec_meta_off = 0 (zero-size region)"
        );
    }

    /// every codec the enum exposes is now wired end-
    /// to-end (`Fp32`, `Sq8`, `RabitqOnly`), so
    /// `register_column` must accept all of them. The check exists
    /// so adding a *new* unimplemented variant in the future
    /// surfaces here loud and fast.
    #[test]
    fn register_column_accepts_every_codec() {
        for codec in [
            RerankCodec::Fp32,
            RerankCodec::Sq8Residual,
            RerankCodec::Sq8Residual,
            RerankCodec::RabitqOnly,
        ] {
            let mut b = VectorBuilder::new();
            b.register_column(VectorConfig {
                column: "v".into(),
                dim: 16,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::L2Sq,
                rerank_codec: codec,
            })
            .unwrap_or_else(|e| panic!("codec {codec:?} must register, got {e:?}"));
        }
    }

    /// building a column with `RerankCodec::Sq8Residual`
    /// round-trips through the reader. The codec discriminator
    /// surfaces on `ColumnReader.rerank_codec`; the codec_meta
    /// region carries `scale[dim] + offset[dim]` (always) plus
    /// per-doc norms (L2Sq only). The on-disk `full[]` region is
    /// `n_docs × 2·dim` bytes for `Sq8Residual`: one u8 code plus
    /// one i8 residual per dimension.
    #[test]
    fn open_round_trips_sq8_codec_discriminator_l2sq() {
        let dim = 32usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
        })
        .expect("register column");
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim).map(|j| (i + j as u32) as f32 * 0.1).collect();
            b.add(0, &v).expect("add");
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":32,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        assert_eq!(r.columns.len(), 1);
        let col = &r.columns[0];
        assert_eq!(col.rerank_codec, RerankCodec::Sq8Residual);

        // codec_meta_off must be non-zero for Sq8 — codec_meta
        // sits inside the open-time region between cluster_idx
        // and the per-cluster blocks.
        assert_ne!(col.codec_meta_off, 0, "Sq8 must declare codec_meta_off > 0");
        // full[] is n_docs × 2·dim (code + residual sidecar),
        // interleaved into the per-cluster blocks region. The
        // full portion is `region_size - n_docs × (code_bytes + 4)`.
        let cb = col.quant.code_bytes();
        let region_size = (col.subsection_range.len() - 4) - col.per_cluster_blocks_off;
        let actual_full_size = region_size - (col.n_docs as usize) * (cb + 4);
        assert_eq!(actual_full_size, (col.n_docs as usize) * dim * 2);

        // sq8_meta materialised at open: per-cluster scale +
        // offset (Sq8PerCluster layout — n_cent × dim floats
        // each), per-doc norms present for L2Sq.
        let meta = col
            .sq8_meta
            .as_ref()
            .expect("Sq8 column must materialise sq8_meta at open");
        let Sq8ColumnMeta::Eager {
            scale,
            offset,
            per_doc_norms,
        } = meta
        else {
            panic!("eager open must materialise Sq8 metadata");
        };
        assert_eq!(scale.len(), (col.n_cent as usize) * dim);
        assert_eq!(offset.len(), (col.n_cent as usize) * dim);
        let norms = per_doc_norms
            .as_ref()
            .expect("L2Sq Sq8 column must carry per-doc norms");
        assert_eq!(norms.len(), col.n_docs as usize);
    }

    /// `Sq8Residual` (the default codec) round-trips through the
    /// reader. The on-disk `full[]` body is `n_docs × 2·dim` bytes
    /// (`[code dim u8 ‖ residual dim i8]`); codec_meta matches Sq8
    /// (per-cluster scale/offset + per-doc norms). The residual leg
    /// rides in `full[]`, not codec_meta.
    #[test]
    fn open_round_trips_sq8_residual_codec_default() {
        let dim = 32usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        // Register via the struct default for rerank_codec to pin
        // that the build default is Sq8Residual.
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::default(),
        })
        .expect("register column");
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim).map(|j| (i + j as u32) as f32 * 0.1).collect();
            b.add(0, &v).expect("add");
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":32,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        let col = &r.columns[0];
        assert_eq!(col.rerank_codec, RerankCodec::Sq8Residual);
        assert_ne!(
            col.codec_meta_off, 0,
            "Sq8Residual must declare codec_meta_off > 0"
        );

        // full[] is n_docs × 2·dim (code + residual sidecar).
        let cb = col.quant.code_bytes();
        let region_size = (col.subsection_range.len() - 4) - col.per_cluster_blocks_off;
        let actual_full_size = region_size - (col.n_docs as usize) * (cb + 4);
        assert_eq!(actual_full_size, (col.n_docs as usize) * dim * 2);
        assert!(col.sq8_meta.is_some());
    }

    /// End-to-end: a `Sq8Residual` cosine self-query returns the
    /// planted doc as top-1. Exercises the residual refine pass in
    /// the eager rerank path.
    #[tokio::test]
    async fn sq8_residual_self_query_round_trips_top1_cosine() {
        let dim = 32usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 29,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8Residual,
        })
        .expect("register column");
        let make = |i: u32| -> Vec<f32> {
            let raw: Vec<f32> = (0..dim)
                .map(|j| {
                    let h = (i.wrapping_mul(0x9E37_79B9)) ^ ((j as u32).wrapping_mul(0x85EB_CA77));
                    let h = h.wrapping_mul(0xC2B2_AE35);
                    ((h & 0xFFFF) as f32) / 65535.0
                })
                .collect();
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            raw.into_iter().map(|x| x / norm).collect()
        };
        let mut all = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v = make(i);
            b.add(0, &v).expect("add");
            all.push(v);
        }
        let blob = b.finish().expect("finish");
        let json =
            r#"[{"column":"v","dim":32,"n_cent":4,"rot_seed":29,"metric":"cosine"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        let col = &r.columns[0];
        assert_eq!(col.rerank_codec, RerankCodec::Sq8Residual);
        let hits = r
            .search("v", &all[42], 5, n_cent, 20)
            .await
            .expect("search must succeed on Sq8Residual cosine column");
        assert_eq!(
            hits[0].0, 42,
            "Sq8Residual cosine self-query must recover self"
        );
    }

    /// + Sq8PerCluster: cosine Sq8 columns carry the
    /// per-doc decoded-norm cache — the rerank kernel normalizes
    /// the decoded vector with it (`1 − dot / |x_decoded|`). Only
    /// negdot drops the norms (its `Σx²` term cancels out),
    /// shrinking codec_meta from `8·n_cent·dim + 4·n_docs` to
    /// `8·n_cent·dim`.
    #[test]
    fn open_sq8_cosine_carries_per_doc_norms() {
        let dim = 16usize;
        let n_cent = 4usize;
        let n_docs = 32u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 11,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8Residual,
        })
        .expect("register column");
        for i in 0..n_docs {
            // Pre-normalised vectors so cosine has a meaningful
            // reference; the test only checks the codec_meta shape,
            // not the recall.
            let mut v: Vec<f32> = (0..dim)
                .map(|j| (i + j as u32) as f32 * 0.1 + 0.5)
                .collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            b.add(0, &v).expect("add");
        }
        let blob = b.finish().expect("finish");
        let json =
            r#"[{"column":"v","dim":16,"n_cent":4,"rot_seed":11,"metric":"cosine"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        let col = &r.columns[0];
        let meta = col.sq8_meta.as_ref().expect("Sq8 must carry sq8_meta");
        let Sq8ColumnMeta::Eager {
            scale,
            offset,
            per_doc_norms,
        } = meta
        else {
            panic!("eager open must materialise Sq8 metadata");
        };
        let norms = per_doc_norms.as_ref().expect(
            "Cosine Sq8 must carry per-doc norms to normalize the decoded vector at rerank",
        );
        assert_eq!(norms.len(), n_docs as usize);
        assert_eq!(scale.len(), n_cent * dim);
        assert_eq!(offset.len(), n_cent * dim);
    }

    /// pins the per-doc-norms indexing contract —
    /// the on-disk norms array is indexed by **position in
    /// `full[]`** (matching the rerank shortlist's `pos`),
    /// not by `doc_id`. The two diverge whenever the writer
    /// pool's cluster-contiguous order differs from insertion
    /// order, which it does in practice (rows get scattered
    /// across clusters by the k-means assignment, so pos ≠ id
    /// for almost every doc).
    ///
    /// Pin: insert N docs whose decoded norms strictly increase
    /// with insertion order, build, open, and assert that the
    /// open-time norms array — read in pos order — does **not**
    /// equal the insertion-order norms. If it does, we're
    /// silently indexing the wrong way; an L2Sq distance lookup
    /// would then return some other doc's norm and corrupt the
    /// rerank ordering.
    ///
    /// We also recompute each `norms[pos]` from the planted
    /// vectors via the per-pos `doc_id` and confirm it matches
    /// — proving the pos-indexed lookup actually resolves to
    /// "this doc's decoded L2 norm".
    #[tokio::test]
    async fn sq8_per_doc_norms_indexed_by_pos_not_doc_id() {
        let dim = 16usize;
        let n_cent = 4usize;
        let n_docs = 32u32;
        // Vectors whose L2 norm grows monotonically with doc_id,
        // while their direction cycles by doc_id. That decouples
        // insertion order from cluster order: k-means groups mostly
        // by direction, not by the monotonic norm ramp, so pos order
        // is observably different from doc_id order.
        let make = |i: u32| -> Vec<f32> {
            let s = 1.0 + (i as f32) * 0.5;
            let phase = (i % n_cent as u32) as f32;
            (0..dim)
                .map(|j| {
                    let sign = if (j + phase as usize) % n_cent < n_cent / 2 {
                        1.0
                    } else {
                        -1.0
                    };
                    sign * (s + (j as f32) * 0.1)
                })
                .collect()
        };
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 23,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
        })
        .expect("register column");
        let mut planted = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v = make(i);
            b.add(0, &v).expect("add");
            planted.push(v);
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":16,"n_cent":4,"rot_seed":23,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        let col = &r.columns[0];
        let meta = col.sq8_meta.as_ref().expect("Sq8 meta present");
        let Sq8ColumnMeta::Eager { per_doc_norms, .. } = meta else {
            panic!("eager open must materialise Sq8 metadata");
        };
        let norms_by_pos = per_doc_norms
            .as_ref()
            .expect("L2Sq Sq8 carries per-doc norms");

        // Insertion-order norms (computed against the fp32
        // originals; these monotonically increase by design).
        let insertion_norms: Vec<f32> = planted
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f32>())
            .collect();

        // If norms were indexed by doc_id, `norms_by_pos[i]`
        // would equal `insertion_norms[i]` up to quantization
        // (a few percent). Cluster-scattered builds reorder
        // docs across positions, so the two sequences should
        // disagree on most slots — this asserts the reorder
        // actually happened (the pin would be vacuous if every
        // doc landed at `pos = doc_id`).
        let n_matching = insertion_norms
            .iter()
            .zip(norms_by_pos.iter())
            .filter(|(ins, pos_n)| (**ins - **pos_n).abs() < 0.5)
            .count();
        assert!(
            n_matching < n_docs as usize,
            "expected k-means + rotation to scatter docs across positions, \
             but norms_by_pos matches insertion_norms in {n_matching}/{n_docs} \
             slots — test corpus may have landed all docs in pos == doc_id order, \
             defeating the indexing pin"
        );

        // For every pos, confirm `norms_by_pos[pos]` equals the
        // decoded L2 norm of the doc at that pos. We don't know
        // the pos↔doc_id mapping from outside the reader, but a
        // self-query against `planted[i]` should return doc_id=i
        // at top-1; the returned distance should be ~0 (matches
        // the quantized doc to itself). That same distance,
        // recomputed via the kernel using doc_i's planted
        // values, requires `norms_by_pos[pos_of(i)]` to be doc_i's
        // decoded norm — exactly the contract we're pinning.
        // A mis-index would surface as a non-zero self-distance
        // larger than the quantization error tolerance.
        for i in [0u32, 7, 15, 23, 31] {
            // rerank_mult=64 → refine=64 ≥ n_docs=32 → every
            // candidate is reranked. Removes the 1-bit shortlist
            // as a confounding variable: any miss here is a real
            // norms-indexing bug, not a Hamming-recall artifact.
            let hits = r
                .search("v", &planted[i as usize], 1, 4, 64)
                .await
                .expect("self-query");
            assert_eq!(hits[0].0, i, "self-query top-1 doc_id for doc {i}");
            // Quantization noise bound: per-dim error ≤ scale/2
            // ≈ span/510. For our corpus, dim spans are ~ 16, so
            // |q-x|² ≤ dim · (span/510)² ≈ 16 · 0.001 ≈ 0.016.
            // A norms-table mis-index would push this to the
            // order of the other docs' norms (≥ 1 unit).
            assert!(
                hits[0].1 <= 0.5,
                "doc {i}: self-query distance {} too large — likely norms \
                 mis-indexed (pos vs doc_id swap)",
                hits[0].1
            );
        }
    }

    /// an Sq8 build + open + self-query recovers the
    /// planted self-vector at top-1. End-to-end through the
    /// codec-aware rerank dispatch + Sq8Kernel — any layout drift
    /// (codec_meta order, code stride, per-doc-norm indexing)
    /// would surface as wrong-doc or out-of-bounds.
    #[tokio::test]
    async fn sq8_self_query_round_trips_top1_l2sq() {
        let dim = 32usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 13,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Sq8Residual,
        })
        .expect("register column");
        let make = |i: u32| -> Vec<f32> {
            (0..dim)
                .map(|j| ((i.wrapping_mul(17) + j as u32 * 3) % 64) as f32 * 0.5)
                .collect()
        };
        let mut all = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v = make(i);
            b.add(0, &v).expect("add");
            all.push(v);
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":32,"n_cent":4,"rot_seed":13,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        // Exhaustive rerank (rerank_mult=20 → refine=100 ≥ n_docs=64)
        // so the test pins Sq8 codec correctness independently of
        // the 1-bit shortlist's recall ceiling.
        let hits = r
            .search("v", &all[17], 5, 4, 20)
            .await
            .expect("search must succeed on Sq8 column");
        assert_eq!(hits[0].0, 17, "Sq8 self-query must recover self at top-1");
        // Sq8 round-trip error: per-dim quantization step is
        // `scale ≈ (max-min)/255`. For this corpus, dim values
        // sit in [0, 31.5] so per-dim error ≲ 0.06, |q-x|² over
        // 32 dims ≲ 32 × 0.06² ≈ 0.12. Pinning a generous bound
        // to keep the test robust to RNG quirks.
        assert!(
            hits[0].1 <= 1.0,
            "Sq8 self-query distance {} should be small (≤ 1.0)",
            hits[0].1
        );
    }

    /// Sq8 self-query top-1 round-trips under Cosine
    /// too. Exercises the Cosine branch of `Sq8Kernel::distance_at`
    /// (no per-doc-norm lookup, `dist = 1 − dot`).
    ///
    /// Corpus design (matters!): unit-norm vectors drawn from
    /// hashed-uniform values per (doc, dim) so neighbor pairs land
    /// at `dot ≈ 1/√dim ≈ 0.18` — gap to self of ~0.82, well above
    /// the Sq8 quantization noise floor (~0.005 for this corpus).
    /// An earlier draft used `((i·23 + j·5) % 50 + 1)` which made
    /// adjacent docs near-parallel (dot ≈ 0.99) and triggered a
    /// quantization-driven swap of doc 5 ↔ doc 42 on self-query —
    /// real Sq8+Cosine behaviour on pathological inputs, not a
    /// kernel bug, but not a useful pin for codec correctness.
    /// Real cosine workloads (semantic embeddings) look like the
    /// current corpus, not the pathological one.
    #[tokio::test]
    async fn sq8_self_query_round_trips_top1_cosine() {
        let dim = 32usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 19,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8Residual,
        })
        .expect("register column");
        let make = |i: u32| -> Vec<f32> {
            let raw: Vec<f32> = (0..dim)
                .map(|j| {
                    // Per-(doc, dim) hash → uniform u16 → fp32 in
                    // [0, 1). Two docs from this generator have
                    // expected dot product ≈ 1/√dim ≈ 0.18 after
                    // L2-normalization.
                    let h = (i.wrapping_mul(0x9E37_79B9)) ^ ((j as u32).wrapping_mul(0x85EB_CA77));
                    let h = h.wrapping_mul(0xC2B2_AE35);
                    ((h & 0xFFFF) as f32) / 65535.0
                })
                .collect();
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            raw.into_iter().map(|x| x / norm).collect()
        };
        let mut all = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v = make(i);
            b.add(0, &v).expect("add");
            all.push(v);
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":32,"n_cent":4,"rot_seed":19,"metric":"cosine"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        // Exhaustive rerank (rerank_mult=20 → refine=100 ≥ n_docs=64)
        // so any failure here pins Sq8 codec correctness rather than
        // 1-bit shortlist recall.
        let hits = r
            .search("v", &all[42], 5, 4, 20)
            .await
            .expect("search must succeed on Sq8 cosine column");
        assert_eq!(hits[0].0, 42, "Sq8 cosine self-query must recover self");
    }

    // -----------------------------------------------------------------
    // `None` codec (no rerank column)
    // -----------------------------------------------------------------
    //
    // The `None` codec drops the `full[]` region entirely. The
    // 1-bit shortlist *is* the final ranking; the on-disk
    // superfile shrinks by ~30× at 1M × 384. Distance values
    // returned from `search()` are `-estimate` (1-bit dot
    // estimate, sign-flipped so smaller = closer holds) — not a
    // true metric distance.

    /// building with `RerankCodec::RabitqOnly` succeeds
    /// and the on-disk superfile carries a zero-length `full[]`
    /// region. Also pins the directory-entry discriminator
    /// (`codec_id = 3`) and the zero-byte codec_meta invariant
    /// (`codec_meta_off = 0`).
    #[test]
    fn open_round_trips_none_codec_discriminator() {
        let dim = 16usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::RabitqOnly,
        })
        .expect("register None column");
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim).map(|j| (i + j as u32) as f32 * 0.1).collect();
            b.add(0, &v).expect("add");
        }
        let blob = b.finish().expect("finish");

        let json =
            r#"[{"column":"v","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");
        assert_eq!(r.columns.len(), 1);
        let col = &r.columns[0];
        assert_eq!(
            col.rerank_codec,
            RerankCodec::RabitqOnly,
            "None build must surface as RerankCodec::RabitqOnly on the reader"
        );
        assert_eq!(
            col.codec_meta_off, 0,
            "None superfiles must write codec_meta_off = 0 (zero-byte meta region)"
        );
        // `None` superfiles have zero-length full[] (per_vec_bytes
        // = 0), so each per-cluster block is just
        // `[codes][doc_ids]` — the blocks region is exactly
        // `n_docs × (code_bytes + 4)` with no full bytes.
        let cb = col.quant.code_bytes();
        let region_size = (col.subsection_range.len() - 4) - col.per_cluster_blocks_off;
        assert_eq!(
            region_size,
            (n_docs as usize) * (cb + 4),
            "None superfiles interleave no full[] bytes — blocks region is \
             exactly n_docs × (code_bytes + 4)"
        );
        assert_eq!(col.n_docs, n_docs);
    }

    /// a `None`-codec column's self-query returns
    /// the planted vector inside the top-K of the 1-bit
    /// shortlist. At dim=128 / n_docs=64 with a well-separated
    /// corpus the 1-bit estimator's top-K reliably contains the
    /// self-vector even without rerank — exactly the contract
    /// `None` opts into. Returned distances are `-estimate`
    /// (sign-flipped so smaller = closer holds).
    #[tokio::test]
    async fn none_self_query_in_top_k_via_shortlist_only() {
        let dim = 128usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 11,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::RabitqOnly,
        })
        .expect("register None column");
        // Angularly diverse corpus — hashed-uniform vectors,
        // L2-normalized. Two docs from this generator have
        // expected dot ≈ 1/√dim ≈ 0.09, so 1-bit estimates
        // separate cleanly and the self-vector dominates the
        // shortlist for its own query.
        let make = |i: u32| -> Vec<f32> {
            let raw: Vec<f32> = (0..dim)
                .map(|j| {
                    let h = (i.wrapping_mul(0x9E37_79B9)) ^ ((j as u32).wrapping_mul(0x85EB_CA77));
                    let h = h.wrapping_mul(0xC2B2_AE35);
                    ((h & 0xFFFF) as f32) / 65535.0 - 0.5
                })
                .collect();
            let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            raw.into_iter().map(|x| x / norm).collect()
        };
        let mut all = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v = make(i);
            b.add(0, &v).expect("add");
            all.push(v);
        }
        let blob = b.finish().expect("finish");
        let json =
            r#"[{"column":"v","dim":128,"n_cent":4,"rot_seed":11,"metric":"l2sq"}]"#.to_string();
        let r = VectorReader::open(Bytes::from(blob), &json).expect("open");

        // nprobe = n_cent so every cluster contributes to the
        // shortlist — the test asserts the 1-bit shortlist's
        // top-K contract, not the cluster-probing one. rerank_mult
        // is intentionally ignored by the None path (asserted
        // here by passing a value that would otherwise oversample).
        let hits = r
            .search("v", &all[17], 5, n_cent, 5)
            .await
            .expect("None-codec search must succeed");
        assert!(
            !hits.is_empty(),
            "None-codec search must return at least one hit"
        );
        assert!(
            hits.iter().any(|(did, _)| *did == 17),
            "self-query must surface the planted vector in top-K, got {hits:?}"
        );
        // Distances are `-estimate` — non-positive for a
        // self-query (the 1-bit dot estimate of a vector with
        // itself is strictly positive after the random rotation).
        assert!(
            hits.iter().all(|(_, d)| d.is_finite()),
            "all None-codec distances must be finite, got {hits:?}"
        );
        // Top-1's distance must be ≤ any other hit's (ascending
        // sort contract).
        for w in hits.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "None-codec hits must be sorted ascending by distance, got {hits:?}"
            );
        }
    }

    /// a `None`-codec search over a counting
    /// lazy source must not perform any range fetch past the
    /// `doc_ids` region — proven indirectly via the total
    /// range count: 2 centroids-region + 2 cluster-idx-region
    /// + `2 × nprobe` (codes + doc_ids per probed cluster). A
    /// regression that left the fat `full[]` `get_range` in
    /// for None columns would surface as one extra range
    /// request, which this asserts away. The structural
    /// invariant (full[] is zero-length on disk) is pinned by
    /// `open_round_trips_none_codec_discriminator`; this test
    /// pins the read-path side.
    #[tokio::test]
    async fn none_search_issues_no_full_region_fetch() {
        let dim = 32usize;
        let n_cent = 4usize;
        let n_docs = 32u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 13,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::RabitqOnly,
        })
        .expect("register None column");
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim).map(|j| (i + j as u32) as f32 * 0.1).collect();
            b.add(0, &v).expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json =
            r#"[{"column":"v","dim":32,"n_cent":4,"rot_seed":13,"metric":"l2sq"}]"#.to_string();

        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_calls = counting.async_counter();
        let sync_calls = counting.sync_counter();
        let r = VectorReader::open_with_source(
            Source::Lazy(StdArc::clone(&counting) as StdArc<dyn LazyByteSource>),
            &json,
            OpenOptions::default(),
        )
        .expect("open lazy");

        // Reset counters: open() touches the directory + every
        // sub-header. We only want to count search-time fetches.
        async_calls.store(0, AtomicOrdering::Relaxed);
        sync_calls.store(0, AtomicOrdering::Relaxed);
        let query: Vec<f32> = (0..dim).map(|j| j as f32 * 0.1).collect();
        let _ = r.search("v", &query, 5, n_cent, 5).await.expect("search");

        // Upper-bound sync fetches for None / nprobe = n_cent:
        //   centroids (1) + cluster_idx (1)
        // + per-cluster codes (≤ n_cent)
        // + per-cluster doc_ids (≤ n_cent)
        // = at most 2 + 2·n_cent = 10
        //
        // The Fp32/Sq8 paths would add one more fat
        // `full[]` get_range on top — that's the leak this
        // test guards against. Empty clusters reduce the
        // upper bound (per-cluster fetches skip on cnt == 0)
        // but never raise it. Async should stay at 0 —
        // warm-cache lazy never falls through to the async
        // bridge for in-memory blobs.
        let sync_count = sync_calls.load(AtomicOrdering::Relaxed) as usize;
        let async_count = async_calls.load(AtomicOrdering::Relaxed);
        assert_eq!(
            async_count, 0,
            "None-codec search on warm lazy must not bridge to async"
        );
        let max_expected = 2 + 2 * n_cent;
        assert!(
            sync_count <= max_expected,
            "None-codec search must issue at most 2 + 2·nprobe = {max_expected} \
             sync fetches (centroids + cluster_idx + per-cluster codes + \
             per-cluster doc_ids); got {sync_count} — any extra is a leaked \
             full[] fetch"
        );
        // A search that probed at least one non-empty cluster
        // must issue ≥ 2 fetches after spatial cluster ordering
        // and bounded range coalescing: centroids+idx plus at
        // least one merged cluster block.
        assert!(
            sync_count >= 2,
            "test corpus produced only empty clusters? got sync_count={sync_count}"
        );
    }

    /// a directory entry carrying an unknown codec id
    /// (anything outside `0..=3` — e.g. `255` from a corrupted /
    /// future-format superfile) errors as `MalformedVersion`. The
    /// safety net catches both forward-compat reads (future codec
    /// ids land in the gap) and on-disk corruption.
    #[test]
    fn open_rejects_superfile_with_unknown_codec_id() {
        let (blob, json) = build_blob(64, 16, 4, Metric::L2Sq);
        let mut bytes = blob.to_vec();

        const OUTER_HDR: usize = 32;
        const DIR_ENTRY: usize = 64;
        let dir_off = OUTER_HDR;
        let codec_byte_off = dir_off + 52;
        bytes[codec_byte_off] = 200u8; // unassigned

        let dir_bytes = &bytes[dir_off..dir_off + DIR_ENTRY];
        let new_crc = crc32c(dir_bytes);
        let crc_off = dir_off + DIR_ENTRY;
        bytes[crc_off..crc_off + 4].copy_from_slice(&new_crc.to_le_bytes());

        let err =
            VectorReader::open_with(Bytes::from(bytes), &json, OpenOptions { verify_crc: false })
                .expect_err("unknown codec id must error at open");
        assert!(
            matches!(err, VectorError::Read(ReadError::MalformedVersion(_))),
            "expected MalformedVersion for unknown codec id, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("unknown") || msg.contains("200"),
            "error must call out the unknown id, got: {msg}"
        );
    }

    // -----------------------------------------------------------------
    // lazy open + inline-`pos` search
    // -----------------------------------------------------------------
    //
    // Open touches only the structural-decode regions (directory,
    // sub-header, summary, centroids, cluster_idx). Search carries
    // `pos = off + i` inline in the shortlist tuple — there is no
    // `doc_to_pos` lookup table to populate (deleted after
    // an audit confirmed zero external readers). The structural
    // memory-ceiling tests below ride on these invariants.

    // -----------------------------------------------------------------
    // diagnostic — Sq8 vs Fp32 recall on planted-cluster
    // cosine corpus
    // -----------------------------------------------------------------
    //
    // The 1M × 384 bench measured Sq8 recall@10 = 0.860 vs Fp32 = 0.964
    // — well outside the "< 0.005 drop on normalized embeddings"
    // envelope. The hypothesis is that the **per-column** Sq8 quantizer
    // wastes most of its 256 buckets on cross-cluster spread: per-dim
    // global range across 1M docs ≈ 0.4, intra-cluster spread ≈ 0.015,
    // so within any one cluster only ~10 buckets are used. The intra-
    // cluster cosine differences between top-K candidates are smaller
    // than the per-bucket quantization noise → reranks flip.
    //
    // This `#[ignore]`-gated diagnostic reproduces the recall drop at
    // 16k × 384 (1/64 scale) and prints corpus geometry stats. Run
    // with `cargo test --lib -- sq8_recall_diagnostic --ignored
    // --nocapture` to inspect. Per-column-quantizer fix (or fallback
    // to Sq8 default) is decided based on what this prints.
    #[tokio::test]
    #[ignore = "recall diagnostic; ~10s; --ignored --nocapture"]
    async fn sq8_recall_diagnostic_planted_cluster_cosine() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, StandardNormal};

        let n_docs = 16_000u32;
        let dim = 384usize;
        let n_cent_planted = 64usize;
        let n_cent_ivf = 256usize;
        let seed: u64 = 1;

        // 1. Build the corpus — same shape as benches/utils/corpus.rs:
        //    planted centers from 3·N(0,1) per dim, per-doc =
        //    center + 0.3·N(0,1), L2-normalized.
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = StandardNormal;
        let centers: Vec<Vec<f32>> = (0..n_cent_planted)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        let s: f64 = dist.sample(&mut rng);
                        (s as f32) * 3.0
                    })
                    .collect()
            })
            .collect();
        let mut all: Vec<Vec<f32>> = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs as usize {
            let center = &centers[i % n_cent_planted];
            let mut v: Vec<f32> = center
                .iter()
                .map(|&c| {
                    let s: f64 = dist.sample(&mut rng);
                    c + (s as f32) * 0.3
                })
                .collect();
            let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= nrm;
            }
            all.push(v);
        }

        // 2. Corpus geometry: per-dim global range vs intra-cluster spread.
        let mut g_min = vec![f32::INFINITY; dim];
        let mut g_max = vec![f32::NEG_INFINITY; dim];
        for v in &all {
            for d in 0..dim {
                if v[d] < g_min[d] {
                    g_min[d] = v[d];
                }
                if v[d] > g_max[d] {
                    g_max[d] = v[d];
                }
            }
        }
        let g_ranges: Vec<f32> = (0..dim).map(|d| g_max[d] - g_min[d]).collect();
        let mean_g_range: f32 = g_ranges.iter().sum::<f32>() / dim as f32;
        let max_g_range: f32 = g_ranges.iter().cloned().fold(0.0f32, f32::max);

        let mut c0_min = vec![f32::INFINITY; dim];
        let mut c0_max = vec![f32::NEG_INFINITY; dim];
        let mut c0_count = 0u32;
        for (i, v) in all.iter().enumerate() {
            if i % n_cent_planted == 0 {
                c0_count += 1;
                for d in 0..dim {
                    if v[d] < c0_min[d] {
                        c0_min[d] = v[d];
                    }
                    if v[d] > c0_max[d] {
                        c0_max[d] = v[d];
                    }
                }
            }
        }
        let intra_ranges: Vec<f32> = (0..dim).map(|d| c0_max[d] - c0_min[d]).collect();
        let mean_intra: f32 = intra_ranges.iter().sum::<f32>() / dim as f32;

        eprintln!("--- corpus geometry (16k × 384, 64 planted centers, cosine, L2-normalized) ---");
        eprintln!(
            "per-dim global range: mean={mean_g_range:.4}  max={max_g_range:.4}  \
             bucket_width@255={:.6}",
            mean_g_range / 255.0
        );
        eprintln!("per-dim intra-cluster-0 range ({c0_count} docs): mean={mean_intra:.4}");
        eprintln!(
            "bucket-waste factor (global / intra): {:.1}x — Sq8 uses ~{} of 256 buckets per cluster",
            mean_g_range / mean_intra.max(1e-9),
            (255.0 * mean_intra / mean_g_range).round() as i32
        );

        // 3. Build Fp32 + Sq8 superfiles from the same corpus.
        let build = |codec: RerankCodec| -> Bytes {
            let mut b = VectorBuilder::new();
            b.register_column(VectorConfig {
                column: "v".into(),
                dim,
                n_cent: n_cent_ivf,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: codec,
            })
            .expect("register");
            for v in &all {
                b.add(0, v).expect("add");
            }
            Bytes::from(b.finish().expect("finish"))
        };
        let fp32_blob = build(RerankCodec::Fp32);
        let sq8_blob = build(RerankCodec::Sq8Residual);
        eprintln!(
            "--- superfile sizes ---\n\
             fp32: {:.2} MiB (1.00x)\n\
             sq8:  {:.2} MiB ({:.2}x)",
            fp32_blob.len() as f64 / 1024.0 / 1024.0,
            sq8_blob.len() as f64 / 1024.0 / 1024.0,
            sq8_blob.len() as f64 / fp32_blob.len() as f64
        );

        let json = format!(
            r#"[{{"column":"v","dim":{dim},"n_cent":{n_cent_ivf},"rot_seed":7,"metric":"cosine"}}]"#
        );
        let r_fp32 = VectorReader::open(fp32_blob, &json).expect("open fp32");
        let r_sq8 = VectorReader::open(sq8_blob, &json).expect("open sq8");

        // 4. Brute-force ground truth (cosine sim descending = neg-dot
        //    ascending — both engines return smaller-is-closer).
        let n_queries = 100usize;
        let k = 10usize;
        let nprobe = n_cent_ivf / 4;
        let rerank_mult = 50usize; // plan-doc Sq8 floor at dim ≤ 384
        let ground_truth: Vec<std::collections::HashSet<u32>> = (0..n_queries)
            .map(|qi| {
                let q = &all[qi];
                let mut sims: Vec<(u32, f32)> = (0..all.len())
                    .map(|j| {
                        let d: f32 = (0..dim).map(|i| q[i] * all[j][i]).sum();
                        (j as u32, d)
                    })
                    .collect();
                sims.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                sims.into_iter().take(k).map(|(id, _)| id).collect()
            })
            .collect();

        eprintln!(
            "--- recall@{k} on {n_queries} self-queries (nprobe={nprobe}, rerank_mult={rerank_mult}) ---"
        );
        let mut recalls = Vec::new();
        for (reader, label) in [(&r_fp32, "fp32"), (&r_sq8, "sq8 ")] {
            let mut total_match = 0usize;
            for qi in 0..n_queries {
                let hits = reader
                    .search("v", &all[qi], k, nprobe, rerank_mult)
                    .await
                    .expect("search");
                let hit_ids: std::collections::HashSet<u32> =
                    hits.into_iter().map(|(id, _)| id).collect();
                let gt = &ground_truth[qi];
                total_match += gt.iter().filter(|id| hit_ids.contains(id)).count();
            }
            let recall = total_match as f32 / (n_queries * k) as f32;
            eprintln!("recall@{k} ({label}): {recall:.4}");
            recalls.push(recall);
        }
        let r_fp = recalls[0];
        let r_sq = recalls[1];
        eprintln!("drop (fp32 - sq8 ): {:.4}", r_fp - r_sq);
        eprintln!(
            "(plan acceptance #2: drop must be \u{2264} 0.01; bench measured 0.10 drop at 1M scale)"
        );

        // -- Probe: vary rerank_mult to isolate shortlist depth vs rerank noise --
        eprintln!("\n--- rerank_mult sweep (Sq8, same corpus/queries) ---");
        for &rm in &[20usize, 50, 100, 200, 400] {
            let mut tm = 0usize;
            for qi in 0..n_queries {
                let hits = r_sq8
                    .search("v", &all[qi], k, nprobe, rm)
                    .await
                    .expect("search");
                let hit_ids: std::collections::HashSet<u32> =
                    hits.into_iter().map(|(id, _)| id).collect();
                tm += ground_truth[qi]
                    .iter()
                    .filter(|id| hit_ids.contains(id))
                    .count();
            }
            eprintln!(
                "  rerank_mult={rm:>4}: sq8 recall@{k} = {:.4}",
                tm as f32 / (n_queries * k) as f32
            );
        }

        // -- Probe: typical top-10 cosine spread (signal that
        //    Sq8 noise must beat).
        let mut spreads = Vec::with_capacity(n_queries);
        for qi in 0..n_queries.min(20) {
            let q = &all[qi];
            let mut sims: Vec<f32> = (0..all.len())
                .map(|j| (0..dim).map(|i| q[i] * all[j][i]).sum::<f32>())
                .collect();
            sims.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let top11: Vec<f32> = sims.iter().take(11).cloned().collect();
            // Spread between top-1 (self, sim=1) and top-10
            let span = top11[0] - top11[10];
            // Median consecutive gap among top-10
            let mut gaps: Vec<f32> = (1..11).map(|i| top11[i - 1] - top11[i]).collect();
            gaps.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med_gap = gaps[gaps.len() / 2];
            spreads.push((span, med_gap));
        }
        let mean_span: f32 = spreads.iter().map(|(s, _)| s).sum::<f32>() / spreads.len() as f32;
        let mean_gap: f32 = spreads.iter().map(|(_, g)| g).sum::<f32>() / spreads.len() as f32;
        eprintln!("\n--- top-10 cosine geometry (the signal Sq8 noise must beat) ---");
        eprintln!(
            "  mean top1-to-top10 span:      {mean_span:.4}\n  \
             mean consecutive median gap:  {mean_gap:.5}\n  \
             Sq8 noise est. (3e-5) vs gap: ratio = {:.2}%",
            3e-5_f32 / mean_gap.max(1e-9) * 100.0
        );
    }

    /// Search-shape corpus used by the inline-pos tests and the
    /// sync-search / counting-source tests. Picks a non-trivial
    /// `n_docs ≥ n_cent` so each cluster has multiple candidates.
    fn build_search_corpus() -> (Bytes, String, Vec<Vec<f32>>) {
        let dim = 16usize;
        let n_cent = 4usize;
        let n_docs = 64u32;
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "embedding".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
        })
        .expect("register column");
        let mut all = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i.wrapping_mul(13) + j as u32 * 5) % 100) as f32)
                .collect();
            b.add(0, &v).expect("add to vector builder");
            all.push(v);
        }
        let bytes = b.finish().expect("finish vector builder");
        let json = r#"[{"column":"embedding","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#
            .to_string();
        (Bytes::from(bytes), json, all)
    }

    /// Self-query smoke: lazy default open must
    /// recover the planted self-vector at top-1, confirming the
    /// inline-`pos` rerank path returns the correct results on
    /// the search-shape corpus the search tests use.
    #[tokio::test]
    async fn lazy_default_search_recovers_self_query() {
        let (blob, json, all) = build_search_corpus();
        let r = VectorReader::open(blob, &json).expect("open");
        let hits = r
            .search("embedding", &all[17], 5, 4, 5)
            .await
            .expect("search must succeed on lazy InMemory");
        assert_eq!(hits[0].0, 17, "self-query must recover self");
    }

    // -----------------------------------------------------------------
    // sync `search()` on `Source::Lazy`
    // -----------------------------------------------------------------
    //
    // These tests pin the sync-only contract: the *only* public
    // entry point is sync
    // `search()`. It works on every `Source` variant — `InMemory`
    // and warm-cache `Source::Lazy` resolve every range through
    // `try_get_range_sync` (zero-copy); cold-miss `Source::Lazy`
    // bridges to the source's async `range()` via
    // `block_in_place + Handle::block_on` / one-shot
    // `current_thread` `Runtime`, the same pattern
    // `supertable::query::superfile_reader` uses for the disk-cache
    // fetch path. No `search_async` is exposed at the public
    // surface; the cold-path async bridging is hidden inside
    // `Source::get_range`.
    //
    // A `CountingLazyByteSource` test helper wraps a `Bytes`
    // payload and counts every `range` / `try_get_range_sync`
    // call against an `AtomicU64`. The `disable_sync` switch
    // lets a test force the cold-miss path (sync access
    // disabled) — exposes any silent fallthrough that would
    // bypass the block_on bridge.

    use crate::superfile::lazy_source::{BytesLazyByteSource, LazyByteSource, LazyByteSourceError};
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};

    /// Test-only [`LazyByteSource`] that wraps a `Bytes` payload and
    /// records every async / sync range request as a counter. The
    /// two `*_returns_none` switches let a test force either the
    /// "async only" path (sync access disabled) or "sync only" path
    /// (async access disabled — exposes any silent fallthrough to
    /// `range` on the supposedly-sync code path).
    #[derive(Debug)]
    struct CountingLazyByteSource {
        bytes: Bytes,
        /// Counts of every `range()` invocation.
        async_calls: StdArc<AtomicU64>,
        /// Counts of every `try_get_range_sync()` invocation.
        sync_calls: StdArc<AtomicU64>,
        /// If `true`, `try_get_range_sync` returns `None` for every
        /// in-bounds range — forces the caller to the async path.
        sync_disabled: AtomicBool,
        /// Current in-flight `range()` futures (entry-bumped,
        /// drop-decremented). pairs with
        /// `max_in_flight` to pin that
        /// [`Source::get_ranges_parallel`] dispatches its cold
        /// fetches concurrently rather than serially.
        in_flight: StdArc<AtomicU64>,
        max_in_flight: StdArc<AtomicU64>,
        /// Per-`range()` artificial latency. Defaults to zero
        /// (legacy callers); the parallel-dispatch test sets it
        /// to a small delay so concurrent futures actually
        /// overlap in wall-clock instead of completing in the
        /// trivial sync slice path inside `range`.
        async_latency_us: AtomicU64,
    }

    impl CountingLazyByteSource {
        fn new(bytes: Bytes) -> Self {
            Self {
                bytes,
                async_calls: StdArc::new(AtomicU64::new(0)),
                sync_calls: StdArc::new(AtomicU64::new(0)),
                sync_disabled: AtomicBool::new(false),
                in_flight: StdArc::new(AtomicU64::new(0)),
                max_in_flight: StdArc::new(AtomicU64::new(0)),
                async_latency_us: AtomicU64::new(0),
            }
        }

        fn async_counter(&self) -> StdArc<AtomicU64> {
            StdArc::clone(&self.async_calls)
        }

        fn sync_counter(&self) -> StdArc<AtomicU64> {
            StdArc::clone(&self.sync_calls)
        }

        fn disable_sync(&self) {
            self.sync_disabled.store(true, AtomicOrdering::Relaxed);
        }

        /// Max-concurrent observer — sampled at every `range()`
        /// entry. Concurrent fetches will produce a value `> 1`;
        /// serial fetches stay at `1`.
        fn max_in_flight_counter(&self) -> StdArc<AtomicU64> {
            StdArc::clone(&self.max_in_flight)
        }

        /// Set per-`range()` artificial latency. Used by the
        /// parallel-dispatch test to ensure concurrent futures
        /// overlap in wall-clock (without latency, the trivial
        /// `bytes.slice(...)` body of `range()` resolves
        /// instantaneously and in-flight peaks at 1 even when
        /// many futures were spawned together).
        fn set_async_latency(&self, latency: std::time::Duration) {
            self.async_latency_us
                .store(latency.as_micros() as u64, AtomicOrdering::Relaxed);
        }
    }

    /// RAII guard: bumps `in_flight` on construct, decrements
    /// on drop, and bumps `max_in_flight` if the new in-flight
    /// count exceeds the previous max. Pairs with
    /// [`CountingLazyByteSource::max_in_flight_counter`] to give
    /// the parallel-dispatch test a single observable for
    /// "fetches actually overlapped."
    struct InFlightGuard {
        in_flight: StdArc<AtomicU64>,
        max_in_flight: StdArc<AtomicU64>,
    }

    impl InFlightGuard {
        fn enter(in_flight: StdArc<AtomicU64>, max_in_flight: StdArc<AtomicU64>) -> Self {
            let now = in_flight.fetch_add(1, AtomicOrdering::AcqRel) + 1;
            // Bump max_in_flight monotonically.
            max_in_flight.fetch_max(now, AtomicOrdering::AcqRel);
            Self {
                in_flight,
                max_in_flight,
            }
        }
    }

    impl Drop for InFlightGuard {
        fn drop(&mut self) {
            self.in_flight.fetch_sub(1, AtomicOrdering::AcqRel);
            // max_in_flight is monotonic by design; nothing to
            // unwind on drop.
            let _ = &self.max_in_flight;
        }
    }

    #[async_trait::async_trait]
    impl LazyByteSource for CountingLazyByteSource {
        fn size(&self) -> u64 {
            self.bytes.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.async_calls.fetch_add(1, AtomicOrdering::Relaxed);
            let _guard = InFlightGuard::enter(
                StdArc::clone(&self.in_flight),
                StdArc::clone(&self.max_in_flight),
            );
            let latency_us = self.async_latency_us.load(AtomicOrdering::Relaxed);
            if latency_us > 0 {
                tokio::time::sleep(std::time::Duration::from_micros(latency_us)).await;
            }
            let total = self.bytes.len() as u64;
            if start.saturating_add(len) > total {
                return Err(LazyByteSourceError::OutOfBounds {
                    start,
                    len,
                    size: total,
                });
            }
            let s = start as usize;
            Ok(self.bytes.slice(s..s + len as usize))
        }

        fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
            self.sync_calls.fetch_add(1, AtomicOrdering::Relaxed);
            if self.sync_disabled.load(AtomicOrdering::Relaxed) {
                return None;
            }
            let total = self.bytes.len() as u64;
            if start.saturating_add(len) > total {
                return None;
            }
            let s = start as usize;
            Some(self.bytes.slice(s..s + len as usize))
        }
    }

    /// Sync `search()` on a `Source::Lazy` whose `try_get_range_sync`
    /// always succeeds (warm cache) behaves identically to the
    /// `Source::InMemory` path. This is the steady-state shape the
    /// supertable reader sees today (the reader_cache is in-process,
    /// so every range is resident).
    #[tokio::test]
    async fn search_on_lazy_source_with_warm_sync_cache_matches_in_memory() {
        let (blob, json, all) = build_search_corpus();
        let r_mem = VectorReader::open(blob.clone(), &json).expect("InMemory open");
        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let r_lazy =
            VectorReader::open_with_source(Source::Lazy(counting), &json, OpenOptions::default())
                .expect("lazy open with warm sync cache");

        for &q_idx in &[0usize, 17, 31, 63] {
            let hits_mem = r_mem
                .search("embedding", &all[q_idx], 5, 4, 5)
                .await
                .expect("InMemory search");
            let hits_lazy = r_lazy
                .search("embedding", &all[q_idx], 5, 4, 5)
                .await
                .expect("Lazy(warm) search");
            assert_eq!(
                hits_mem, hits_lazy,
                "lazy warm-sync results must match InMemory for query {q_idx}"
            );
        }
    }

    /// Sync `search()` on a `Source::Lazy` whose
    /// `try_get_range_sync` returns `None` for every range still
    /// succeeds — `Source::get_range` bridges to the source's
    /// async `range()` via the one-shot `current_thread`
    /// `Runtime` fallback (no ambient tokio runtime in
    /// `#[test]`). Results must equal the `Source::InMemory`
    /// baseline.
    ///
    /// This is the cold-path proof — the public sync surface
    /// works against an arbitrary async-only `LazyByteSource`
    /// impl. Production callers always have an ambient runtime
    /// (the supertable owns one), so the `block_in_place +
    /// Handle::block_on` branch is what fires there; this test
    /// exercises the no-ambient-runtime fallback branch to
    /// keep that path live.
    #[tokio::test]
    async fn search_on_lazy_source_with_no_sync_fallback_bridges_to_async() {
        let (blob, json, all) = build_search_corpus();
        let r_mem = VectorReader::open(blob.clone(), &json).expect("InMemory baseline");
        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_counter = counting.async_counter();
        let r_lazy = VectorReader::open_with_source(
            Source::Lazy(StdArc::clone(&counting) as StdArc<dyn LazyByteSource>),
            &json,
            OpenOptions::default(),
        )
        .expect("lazy open");
        counting.disable_sync();

        for &q_idx in &[0usize, 17, 31, 63] {
            let hits_mem = r_mem
                .search("embedding", &all[q_idx], 5, 4, 5)
                .await
                .expect("InMemory search");
            let hits_lazy = r_lazy
                .search("embedding", &all[q_idx], 5, 4, 5)
                .await
                .expect("sync search must succeed via block_on bridge");
            assert_eq!(
                hits_mem, hits_lazy,
                "sync search with block_on bridge must match InMemory for query {q_idx}"
            );
        }

        assert!(
            async_counter.load(AtomicOrdering::Relaxed) > 0,
            "with sync access disabled, every fetch must route through \
             the source's async range() via the block_on bridge"
        );
    }

    /// Range-counting test. Sync `search()` issues per-region /
    /// per-cluster `Source::get_range` calls:
    ///
    /// - 1 range for centroids
    /// - 1 range for cluster_idx
    /// - 1 range per probed cluster (codes + doc_ids are
    ///   interleaved in one block, so one range per cluster)
    /// - 1 fat range for the rerank batch in `full[]`
    ///
    /// At `nprobe = N` with all probed clusters non-empty that is
    /// `2 + N + 1` ranges before coalescing. The corpus here has
    /// `n_cent = 4` and the test uses `nprobe = 4`; spatial
    /// cluster ordering can merge adjacent cluster blocks into
    /// fewer physical GETs, so the observed budget is `2..=5`.
    ///
    /// Forcing `try_get_range_sync` off makes every range route
    /// through the source's async `range()` via the block_on
    /// bridge, so the `async_calls` counter is the right
    /// instrumentation for "how many distinct ranges did
    /// `search()` request".
    ///
    /// A regression that smuggles in extra range fetches — e.g.
    /// reintroducing the whole-subsection fallback, or pulling the
    /// full `doc_ids` region over the wire at open — surfaces here
    /// rather than at the production object-store harness.
    #[tokio::test]
    async fn search_cold_first_search_range_count_per_cluster() {
        let (blob, json, all) = build_search_corpus();
        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_counter = counting.async_counter();
        let sync_counter = counting.sync_counter();
        let r = VectorReader::open_with_source(
            Source::Lazy(StdArc::clone(&counting) as StdArc<dyn LazyByteSource>),
            &json,
            OpenOptions::default(),
        )
        .expect("lazy open");

        let async_after_open = async_counter.load(AtomicOrdering::Relaxed);
        let sync_after_open = sync_counter.load(AtomicOrdering::Relaxed);
        assert_eq!(
            async_after_open, 0,
            "open path uses try_get_range_sync only — no async fetches expected"
        );
        assert!(
            sync_after_open > 0,
            "open path should have issued sync range fetches"
        );

        counting.disable_sync();
        let hits = r
            .search("embedding", &all[7], 5, 4, 5)
            .await
            .expect("sync search via block_on bridge");
        assert!(!hits.is_empty(), "search should return hits");

        let async_calls_for_first_search =
            async_counter.load(AtomicOrdering::Relaxed) - async_after_open;
        // At nprobe=4 with this corpus, all probed clusters are
        // non-empty. Spatial cluster ordering can merge the
        // cluster blocks into fewer physical GETs.
        assert!(
            (2..=5).contains(&(async_calls_for_first_search as usize)),
            "per-cluster path: cold first search expected to issue \
             2..=5 ranges (centroids+cluster_idx + coalesced/interleaved \
             cluster blocks). Got {async_calls_for_first_search}."
        );
    }

    /// `BytesLazyByteSource` (the production-ready in-memory
    /// `LazyByteSource` impl) yields the same sync `search()`
    /// results as `Source::InMemory`. Locks in the contract that
    /// the trait-based path doesn't accidentally diverge from the
    /// enum-based fast path.
    #[tokio::test]
    async fn search_matches_in_memory_through_bytes_lazy_source() {
        let (blob, json, all) = build_search_corpus();
        let r_mem = VectorReader::open(blob.clone(), &json).expect("InMemory baseline");
        let lazy_src: StdArc<dyn LazyByteSource> = StdArc::new(BytesLazyByteSource::new(blob));
        let r_lazy =
            VectorReader::open_with_source(Source::Lazy(lazy_src), &json, OpenOptions::default())
                .expect("lazy open via BytesLazyByteSource");

        for &q_idx in &[3usize, 19, 47] {
            let hits_mem = r_mem
                .search("embedding", &all[q_idx], 5, 4, 5)
                .await
                .expect("InMemory search");
            let hits_lazy = r_lazy
                .search("embedding", &all[q_idx], 5, 4, 5)
                .await
                .expect("BytesLazyByteSource sync search");
            assert_eq!(
                hits_mem, hits_lazy,
                "BytesLazyByteSource results must match InMemory for query {q_idx}"
            );
        }
    }

    // -----------------------------------------------------------------
    // § Acceptance #2 — memory-ceiling unit test
    // -----------------------------------------------------------------
    //
    // The headline guarantee is "resident set per open
    // vector superfile is bounded by O(n_cent × dim × 4 + small)",
    // independent of `n_docs`. Acceptance criterion #2 spells it
    // out: opening a `Source::Lazy` over a mmap-backed
    // `BytesLazyByteSource` at 1M × 384 with
    // `OpenOptions { verify_crc: false }` must leave the process
    // RSS delta ≤ 10 MB per opened column.
    //
    // Why mmap specifically: this is exactly how the disk cache
    // feeds bytes into `SuperfileReader` —
    // `Bytes::from_owner(Arc<Mmap>)`. The kernel never faults the
    // bulk codes/full/doc_ids pages on the default path because
    // nothing in `open_with_source` accesses them: the CRC scan
    // is gated on `verify_crc`, search uses inline `pos`
    // so no `doc_ids` walk happens, and the structural-decode
    // bytes (outer header + dir + sub_header) are a handful of
    // pages. The resident allocation is dominated by the rotation
    // matrix (≈ 590 KB at dim=384) and small column metadata —
    // well inside the 10 MB ceiling at any practical
    // `n_docs`.
    //
    // Companion smoke test below (`mem_ceiling_lazy_open_smoke`)
    // runs in default `cargo test --lib` at a smaller scale so
    // every PR gets continuous feedback on this guarantee
    // without paying for a 1M-doc build. The 1M × 384 plan-spec
    // version is `#[ignore]`'d because
    // `VectorBuilder.finish_to(...)` at that scale takes ~35 s in
    // release / several minutes in debug. Run explicitly:
    //
    // ```bash
    // cargo test --release -p infino --lib \
    //     mem_ceiling_lazy_open_under_10mib -- --ignored --nocapture
    // ```

    /// `Bytes::from_owner` adapter for `Arc<memmap2::Mmap>` —
    /// mirrors `supertable::reader_cache::disk::ArcMmapOwner`
    /// (which is private to that module). Sharing the mapping
    /// via `Arc<Mmap>` keeps it alive for the reader's lifetime
    /// while also letting the test anchor the mmap explicitly.
    struct MmapOwner(StdArc<memmap2::Mmap>);

    impl AsRef<[u8]> for MmapOwner {
        fn as_ref(&self) -> &[u8] {
            self.0.as_ref()
        }
    }

    /// Build an `(n_docs × dim)` corpus, register a single
    /// vector column with the requested IVF shape, and stream
    /// the resulting unified-blob bytes to `tmp` via
    /// `VectorBuilder::finish_to`. The streaming
    /// write avoids materializing a 1.5 GiB `Vec<u8>` in the
    /// test's address space at 1M × 384 — the build's transient
    /// peak doesn't survive the `before` RSS snapshot.
    ///
    /// Deterministic per-row vector: `seed = i × 0x9E3779B1`
    /// folded through a linear congruential step per dim slot.
    /// Same shape the bench corpus generators use, inlined so
    /// the unit test doesn't reach into the bench harness.
    fn build_corpus_to_file(
        path: &std::path::Path,
        n_docs: u32,
        dim: usize,
        n_cent: usize,
    ) -> String {
        use std::io::BufWriter;

        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "embedding".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
        })
        .expect("register column");
        let mut v = vec![0f32; dim];
        for i in 0..n_docs {
            let mut seed = i.wrapping_mul(0x9E37_79B1);
            for slot in v.iter_mut() {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                *slot = ((seed >> 16) as f32) / 65_535.0;
            }
            b.add(0, &v).expect("add to vector builder");
        }
        let file = std::fs::File::create(path).expect("create tempfile");
        let writer = BufWriter::new(file);
        b.finish_to(writer).expect("finish_to BufWriter<File>");

        format!(
            r#"[{{"column":"embedding","dim":{dim},"n_cent":{n_cent},"rot_seed":7,"metric":"l2sq"}}]"#
        )
    }

    /// Open a `Source::Lazy` over a mmap'd corpus file and
    /// return the process RSS delta (bytes) attributable to
    /// the open. Anchors `(reader, mmap_arc)` past the
    /// after-RSS read so neither is dropped before
    /// measurement.
    ///
    /// `memory_stats::memory_stats()` reads `/proc/self/statm`
    /// on Linux — cheap syscall, no allocations of its own.
    /// `physical_mem` is the kernel's RSS counter (anon +
    /// file-mapped). Faulted mmap pages count; unfaulted
    /// pages don't. The whole point of the test is that the
    /// open path only touches a handful of pages (outer
    /// header, directory, per-subsection header) and leaves
    /// the rest of the file unmapped.
    fn measure_lazy_open_rss_delta(corpus_path: &std::path::Path, json: &str) -> (usize, usize) {
        let file = std::fs::File::open(corpus_path).expect("reopen corpus readonly");
        let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap corpus");
        let mmap_arc = StdArc::new(mmap);
        let bytes = Bytes::from_owner(MmapOwner(StdArc::clone(&mmap_arc)));
        let lazy: StdArc<dyn LazyByteSource> = StdArc::new(BytesLazyByteSource::new(bytes));

        let before = memory_stats::memory_stats()
            .expect("memory_stats supported")
            .physical_mem;

        let reader = VectorReader::open_with_source(
            Source::Lazy(lazy),
            json,
            OpenOptions { verify_crc: false },
        )
        .expect("lazy open");

        let after = memory_stats::memory_stats()
            .expect("memory_stats supported")
            .physical_mem;

        let n_cols = reader.columns.len();
        let delta = after.saturating_sub(before);

        // Keep both alive past the RSS reads — dropping
        // `reader` before reading `after` would silently
        // make the delta look smaller than reality.
        std::hint::black_box((&reader, &mmap_arc));
        drop(reader);
        drop(mmap_arc);

        (delta, n_cols)
    }

    /// **acceptance criterion #2 (plan-spec scale).**
    ///
    /// 1 M × 384, `n_cent = 1024`. `#[ignore]`-gated because
    /// the `VectorBuilder.finish_to(...)` call takes ~35 s in
    /// release. Run explicitly:
    ///
    /// ```bash
    /// cargo test --release -p infino --lib \
    ///     mem_ceiling_lazy_open_under_10mib -- --ignored --nocapture
    /// ```
    ///
    /// A regression that re-introduces eager subsection
    /// materialization (the older behaviour) or that scans
    /// `doc_ids` at open will push per-column RSS past the
    /// 10 MB ceiling and fail here rather than at the 100 M
    /// production OOM.
    #[test]
    #[ignore]
    fn mem_ceiling_lazy_open_under_10mib() {
        const N_DOCS: u32 = 1_000_000;
        const DIM: usize = 384;
        const N_CENT: usize = 1024;

        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let json = build_corpus_to_file(tmp.path(), N_DOCS, DIM, N_CENT);

        let (delta_bytes, n_cols) = measure_lazy_open_rss_delta(tmp.path(), &json);
        let delta_mib = delta_bytes as f64 / (1024.0 * 1024.0);
        let per_col_mib = delta_mib / (n_cols.max(1) as f64);

        eprintln!(
            "mem_ceiling_lazy_open_under_10mib (1M × {DIM}, n_cent={N_CENT}): \
             RSS delta = {delta_mib:.3} MiB over {n_cols} column(s) \
             = {per_col_mib:.3} MiB/col"
        );

        assert!(
            per_col_mib <= 10.0,
            "acceptance #2: lazy open RSS delta \
             {per_col_mib:.3} MiB/col exceeds 10 MiB ceiling \
             at 1M × {DIM}, n_cent={N_CENT} (total delta \
             {delta_mib:.3} MiB over {n_cols} column(s))."
        );
    }

    /// **acceptance criterion #2 (smoke scale).**
    ///
    /// 50 k × 64, `n_cent = 64`. Runs in default
    /// `cargo test --lib` (~1–2 s build) so every PR gets
    /// continuous feedback on the structural property: lazy
    /// open touches only the structural-decode pages, never
    /// the bulk codes/full/doc_ids regions. The 10 MiB ceiling
    /// at the plan's headline 1M × 384 scale is asserted at
    /// the same value here because the resident allocation
    /// (mostly the rotation matrix at `dim²·4` = 16 KB for
    /// dim=64) is *smaller* at smoke scale, not larger — if
    /// this fires, the bigger test will too.
    ///
    /// `dim = 64` keeps the corpus tiny (~13 MB on disk) and
    /// the rotation matrix Gram-Schmidt fast.
    #[test]
    fn mem_ceiling_lazy_open_smoke() {
        const N_DOCS: u32 = 50_000;
        const DIM: usize = 64;
        const N_CENT: usize = 64;

        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let json = build_corpus_to_file(tmp.path(), N_DOCS, DIM, N_CENT);

        let (delta_bytes, n_cols) = measure_lazy_open_rss_delta(tmp.path(), &json);
        let delta_mib = delta_bytes as f64 / (1024.0 * 1024.0);
        let per_col_mib = delta_mib / (n_cols.max(1) as f64);

        eprintln!(
            "mem_ceiling_lazy_open_smoke ({N_DOCS} × {DIM}, n_cent={N_CENT}): \
             RSS delta = {delta_mib:.3} MiB over {n_cols} column(s) \
             = {per_col_mib:.3} MiB/col"
        );

        assert!(
            per_col_mib <= 10.0,
            "lazy open smoke RSS delta {per_col_mib:.3} MiB/col \
             exceeds 10 MiB ceiling at {N_DOCS} × {DIM} \
             (total delta {delta_mib:.3} MiB over {n_cols} column(s))."
        );
    }

    // -----------------------------------------------------------------
    // — supertable-scale memory ceiling
    // -----------------------------------------------------------------
    //
    // The single-superfile `mem_ceiling_lazy_open_*` tests above pin the
    // per-reader bound. These multi-superfile variants pin the
    // *supertable-shaped* bound: open N superfiles concurrently — same
    // shape `Supertable::commit` produces (N = N_SUPERFILES_BENCH × num_cpus
    // because `split_buffer_into_row_shards` shards each commit's
    // buffer into one superfile per writer-pool thread) — and assert the
    // total anon RSS delta scales as `N × O(centroids + rotation +
    // small)`, not as `N × subsection_size`.
    //
    // What this proves (and what it doesn't):
    //
    // - PROVES: a supertable opened with the production disk-cache
    //   path (`Source::InMemory(Bytes::from_owner(mmap))` per superfile —
    //   see `supertable::reader_cache::disk::insert`) keeps anon
    //   RSS bounded across an arbitrary number of superfiles, with no
    //   per-doc anon term. Equivalent here because
    //   `Bytes::from_owner` is zero-copy over the mmap, and the
    //   lazy-open path doesn't touch `doc_ids[]` / `full[]` at
    //   open time (the inline `pos` removes the only reason
    //   open ever touched `doc_ids[]`).
    //
    // - DOES NOT PROVE: the in-process `InMemoryReaderCache` path
    //   (`Bytes::from(Vec)` per superfile — see
    //   `supertable::reader_cache::in_memory::insert`) has the same
    //   bound. That path holds each superfile's bytes in anon by
    //   construction (no mmap involved). The in-memory cache is the
    //   test/bench path; production attaches a `StorageProvider` and
    //   routes through the disk cache. A separate test for the
    //   in-memory cache path is out of scope here — that path's
    //   anon cost is its declared contract.
    //
    // The bench's 10M × 4-commit × num_cpus-thread shape produces
    // exactly the topology these tests exercise. The smoke variant
    // mirrors the bench's *layout* at a tiny corpus size (4 superfiles
    // × 50 k docs × 64 dim) so every PR catches regressions
    // (~5 s build). The `#[ignore]`'d plan-spec variant uses the
    // bench's actual per-superfile shape (16 superfiles × 625 k docs ×
    // 384 dim × n_cent_per_superfile matching the bench's
    // `n_cent_total / 4`) and runs only when called out.

    /// Open `N` superfile files (built by `build_corpus_to_file`) via
    /// `Source::Lazy(BytesLazyByteSource over Arc<Mmap>)` and return
    /// the total RSS delta attributable to those opens. Anchors
    /// `(readers, mmaps)` past the after-RSS read.
    fn measure_lazy_multi_superfile_open_rss_delta(
        corpus_paths: &[std::path::PathBuf],
        jsons: &[String],
    ) -> (usize, usize, usize) {
        assert_eq!(corpus_paths.len(), jsons.len(), "paths/jsons must align");
        let n_superfiles = corpus_paths.len();

        // Pre-build (mmap, lazy source) pairs *before* the `before`
        // snapshot so the syscalls don't contaminate the delta — we
        // only want the open path's allocations in the measurement.
        let mut lazies: Vec<(StdArc<memmap2::Mmap>, StdArc<dyn LazyByteSource>)> =
            Vec::with_capacity(n_superfiles);
        for path in corpus_paths {
            let file = std::fs::File::open(path).expect("reopen corpus readonly");
            let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("mmap corpus");
            let mmap_arc = StdArc::new(mmap);
            let bytes = Bytes::from_owner(MmapOwner(StdArc::clone(&mmap_arc)));
            let lazy: StdArc<dyn LazyByteSource> = StdArc::new(BytesLazyByteSource::new(bytes));
            lazies.push((mmap_arc, lazy));
        }

        let before = memory_stats::memory_stats()
            .expect("memory_stats supported")
            .physical_mem;

        let mut readers: Vec<VectorReader> = Vec::with_capacity(n_superfiles);
        let mut n_cols_total = 0usize;
        for ((_, lazy), json) in lazies.iter().zip(jsons.iter()) {
            let reader = VectorReader::open_with_source(
                Source::Lazy(StdArc::clone(lazy)),
                json,
                OpenOptions { verify_crc: false },
            )
            .expect("lazy open");
            n_cols_total += reader.columns.len();
            readers.push(reader);
        }

        let after = memory_stats::memory_stats()
            .expect("memory_stats supported")
            .physical_mem;

        let delta = after.saturating_sub(before);

        // Keep both alive past the RSS reads — dropping any reader
        // (or mmap) before reading `after` would silently shrink the
        // measured delta.
        std::hint::black_box((&readers, &lazies));
        drop(readers);
        drop(lazies);

        (delta, n_cols_total, n_superfiles)
    }

    /// **supertable-scale memory ceiling (smoke).**
    ///
    /// Mirrors the bench's 4-commit × num_cpus-thread shape at a
    /// tiny corpus size. Builds 4 superfile files (each 50 k × 64
    /// dim × n_cent=64 — same shape as
    /// `mem_ceiling_lazy_open_smoke`), opens all 4 lazy, and
    /// asserts the total anon RSS delta is ≤ 10 MiB. With
    /// per-superfile ceiling of 10 MiB / column from the single-
    /// superfile smoke and a 4× multiplier in the worst case
    /// (centroids + rotation matrix per superfile), 10 MiB total
    /// gives plenty of headroom while still failing loud if a
    /// regression makes per-superfile opens allocate per-doc.
    ///
    /// Runs in the default `cargo test --lib` suite (~3–5 s
    /// total) so every PR validates the supertable-shape bound.
    #[test]
    fn mem_ceiling_lazy_multi_superfile_open_smoke() {
        const N_SUPERFILES: usize = 4;
        const N_DOCS_PER_SEG: u32 = 50_000;
        const DIM: usize = 64;
        const N_CENT: usize = 64;

        let mut tmps: Vec<tempfile::NamedTempFile> = Vec::with_capacity(N_SUPERFILES);
        let mut paths: Vec<std::path::PathBuf> = Vec::with_capacity(N_SUPERFILES);
        let mut jsons: Vec<String> = Vec::with_capacity(N_SUPERFILES);
        for _ in 0..N_SUPERFILES {
            let tmp = tempfile::NamedTempFile::new().expect("tempfile");
            let json = build_corpus_to_file(tmp.path(), N_DOCS_PER_SEG, DIM, N_CENT);
            paths.push(tmp.path().to_path_buf());
            jsons.push(json);
            tmps.push(tmp); // keep the tempfile alive until end
        }

        let (delta_bytes, n_cols_total, n_superfiles) =
            measure_lazy_multi_superfile_open_rss_delta(&paths, &jsons);
        let delta_mib = delta_bytes as f64 / (1024.0 * 1024.0);
        let per_seg_mib = delta_mib / n_superfiles as f64;

        eprintln!(
            "mem_ceiling_lazy_multi_superfile_open_smoke ({N_SUPERFILES} × {N_DOCS_PER_SEG} × \
             {DIM}, n_cent={N_CENT}): RSS delta = {delta_mib:.3} MiB over {n_superfiles} \
             superfile(s) ({n_cols_total} column(s) total) = {per_seg_mib:.3} MiB/superfile"
        );

        assert!(
            delta_mib <= 10.0,
            "supertable-shape lazy open RSS delta {delta_mib:.3} MiB exceeds 10 MiB ceiling \
             at {N_SUPERFILES} × {N_DOCS_PER_SEG} × {DIM} — regression hint: each superfile may \
             be touching its doc_ids/full[]/codes region at open"
        );

        drop(tmps);
    }

    /// **supertable-scale memory ceiling (plan-spec).**
    ///
    /// Mirrors the bench's actual 10M × 4-commit ×
    /// 4-thread-writer-pool topology: 16 superfiles × 625 k docs ×
    /// 384 dim × `n_cent_per_superfile = n_cent(10M) / 4` (the
    /// bench's `corpus::n_cent(10M)` returns 1024, so this is
    /// 256). Each superfile file is ~960 MiB on disk; the test
    /// writes ~15 GiB total to the tempdir. Build time is
    /// dominated by the 16 sequential streaming builds at
    /// ~10 s each in release ≈ 3 min total.
    ///
    /// `#[ignore]`-gated. Run explicitly:
    ///
    /// ```bash
    /// cargo test --release -p infino --lib \
    ///     mem_ceiling_lazy_supertable_scale_under_50mib -- --ignored --nocapture
    /// ```
    ///
    /// Bound: 50 MiB total anon over the 16 superfiles. The
    /// per-superfile open materialises:
    /// - rotation matrix: `dim² × 4 = 576 KiB` at dim=384
    /// - centroids buffer (in lazy source page cache, not anon):
    ///   `n_cent × dim × 4 = 384 KiB` at the smoke shape
    /// - per-column header / cluster_idx slices (KiB-range)
    ///
    /// Add a 2× safety margin for allocator overhead +
    /// reader-struct fields, multiply by 16 superfiles → ~20 MiB
    /// theoretical, 50 MiB ceiling for headroom. A regression
    /// that re-introduces eager subsection materialisation
    /// would blow this to ~15 GiB (the full corpus) and fail
    /// loud here rather than at the production 100 M OOM.
    #[test]
    #[ignore]
    fn mem_ceiling_lazy_supertable_scale_under_50mib() {
        const N_SUPERFILES: usize = 16;
        const N_DOCS_PER_SEG: u32 = 625_000;
        const DIM: usize = 384;
        const N_CENT_PER_SEG: usize = 256;

        let mut tmps: Vec<tempfile::NamedTempFile> = Vec::with_capacity(N_SUPERFILES);
        let mut paths: Vec<std::path::PathBuf> = Vec::with_capacity(N_SUPERFILES);
        let mut jsons: Vec<String> = Vec::with_capacity(N_SUPERFILES);
        for i in 0..N_SUPERFILES {
            let tmp = tempfile::NamedTempFile::new().expect("tempfile");
            eprintln!(
                "  building superfile {i:2}/{N_SUPERFILES} \
                 ({N_DOCS_PER_SEG} × {DIM}, n_cent={N_CENT_PER_SEG})…"
            );
            let json = build_corpus_to_file(tmp.path(), N_DOCS_PER_SEG, DIM, N_CENT_PER_SEG);
            paths.push(tmp.path().to_path_buf());
            jsons.push(json);
            tmps.push(tmp);
        }

        let (delta_bytes, n_cols_total, n_superfiles) =
            measure_lazy_multi_superfile_open_rss_delta(&paths, &jsons);
        let delta_mib = delta_bytes as f64 / (1024.0 * 1024.0);
        let per_seg_mib = delta_mib / n_superfiles as f64;

        eprintln!(
            "mem_ceiling_lazy_supertable_scale_under_50mib ({N_SUPERFILES} × {N_DOCS_PER_SEG} × \
             {DIM}, n_cent={N_CENT_PER_SEG}): RSS delta = {delta_mib:.3} MiB over \
             {n_superfiles} superfile(s) ({n_cols_total} column(s) total) = \
             {per_seg_mib:.3} MiB/superfile"
        );

        assert!(
            delta_mib <= 50.0,
            "supertable-scale (10M-bench shape) lazy open RSS delta {delta_mib:.3} MiB \
             exceeds 50 MiB ceiling at {N_SUPERFILES} × {N_DOCS_PER_SEG} × {DIM}. \
             Eager re-introduction would push this past 15 GiB."
        );

        drop(tmps);
    }

    /// **many-superfiles stress test (100M
    /// aspiration shape).**
    ///
    /// The honest scale test for "100M docs across a supertable"
    /// can't materialise 100M production-shape superfiles on a
    /// developer box (the per-superfile 625k × 384 shape used in
    /// the bench produces ~960 MiB on disk × 160 superfiles = 150
    /// GiB of corpus). Instead, this test pins the *structural*
    /// memory bound by varying the high-cardinality axis (superfile
    /// count) at a thin per-superfile shape: **100 superfiles × 50 k
    /// docs × 128 dim × 128 n_cent**.
    ///
    /// What this proves:
    ///
    /// - Per-superfile open allocation is `O(n_cent × dim × 4 +
    ///   rotation + small)` — no `n_docs` term. At this shape:
    ///   centroids 64 KiB + rotation matrix 64 KiB + column
    ///   metadata ≪ 1 MiB per superfile. Total expected RSS delta
    ///   ≪ 200 MiB across 100 superfiles; 400 MiB ceiling for
    ///   allocator overhead + reader-struct fields.
    ///
    /// - The deletion of `doc_to_pos` made superfile-count
    ///   the only scaling dimension. A regression that reintroduced
    ///   any per-doc resident state — e.g. a returning lookup
    ///   table at `n_docs × 4` bytes per column — would here
    ///   allocate 100 × 50 k × 4 = 20 MiB anon just for tables
    ///   (small but growing); at the bench's 100 superfiles × 625 k
    ///   the same regression is 250 MiB.
    ///
    /// Each superfile file is ~25 MiB on disk; the test writes
    /// ~2.5 GiB total to the tempdir. Build time is dominated by
    /// the 100 sequential streaming builds (~1.5 s each in
    /// release ≈ 2.5 min total).
    ///
    /// `#[ignore]`-gated. Run explicitly:
    ///
    /// ```bash
    /// cargo test --release -p infino --lib \
    ///     mem_ceiling_lazy_many_superfiles_under_400mib -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn mem_ceiling_lazy_many_superfiles_under_400mib() {
        const N_SUPERFILES: usize = 100;
        const N_DOCS_PER_SEG: u32 = 50_000;
        const DIM: usize = 128;
        const N_CENT_PER_SEG: usize = 128;

        let mut tmps: Vec<tempfile::NamedTempFile> = Vec::with_capacity(N_SUPERFILES);
        let mut paths: Vec<std::path::PathBuf> = Vec::with_capacity(N_SUPERFILES);
        let mut jsons: Vec<String> = Vec::with_capacity(N_SUPERFILES);
        for i in 0..N_SUPERFILES {
            let tmp = tempfile::NamedTempFile::new().expect("tempfile");
            if i % 10 == 0 {
                eprintln!(
                    "  building superfile {i:3}/{N_SUPERFILES} \
                     ({N_DOCS_PER_SEG} × {DIM}, n_cent={N_CENT_PER_SEG})…"
                );
            }
            let json = build_corpus_to_file(tmp.path(), N_DOCS_PER_SEG, DIM, N_CENT_PER_SEG);
            paths.push(tmp.path().to_path_buf());
            jsons.push(json);
            tmps.push(tmp);
        }

        let (delta_bytes, n_cols_total, n_superfiles) =
            measure_lazy_multi_superfile_open_rss_delta(&paths, &jsons);
        let delta_mib = delta_bytes as f64 / (1024.0 * 1024.0);
        let per_seg_mib = delta_mib / n_superfiles as f64;

        eprintln!(
            "mem_ceiling_lazy_many_superfiles_under_400mib ({N_SUPERFILES} × {N_DOCS_PER_SEG} × \
             {DIM}, n_cent={N_CENT_PER_SEG}): RSS delta = {delta_mib:.3} MiB over \
             {n_superfiles} superfile(s) ({n_cols_total} column(s) total) = \
             {per_seg_mib:.3} MiB/superfile"
        );

        assert!(
            delta_mib <= 400.0,
            "many-superfiles lazy open RSS delta {delta_mib:.3} MiB exceeds 400 MiB ceiling \
             at {N_SUPERFILES} × {N_DOCS_PER_SEG} × {DIM}. A regression that reintroduced \
             any per-doc resident state would push this much higher; the deletion of \
             doc_to_pos is what keeps the bound structural."
        );

        drop(tmps);
    }

    // -----------------------------------------------------------------
    // VectorReader::open_lazy cold-open range budget + round-trip
    // parity. The lazy open path fetches exact metadata ranges:
    // outer header, directory + CRC, subsection headers, and Sq8
    // codec_meta. It does not prefetch centroids, cluster_idx, or
    // per-cluster blocks; those are search-time data.
    // -----------------------------------------------------------------

    fn build_small_superfile(
        dim: usize,
        n_cent: usize,
        n_docs: u32,
        codec: RerankCodec,
        metric: Metric,
    ) -> (Bytes, String, Vec<Vec<f32>>) {
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: 41,
            metric,
            rerank_codec: codec,
        })
        .expect("register column");
        let mut all = Vec::with_capacity(n_docs as usize);
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim).map(|j| (i + j as u32) as f32 * 0.1).collect();
            b.add(0, &v).expect("add");
            all.push(v);
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let metric_str = match metric {
            Metric::L2Sq => "l2sq",
            Metric::Cosine => "cosine",
            Metric::NegDot => "negdot",
        };
        let json = format!(
            r#"[{{"column":"v","dim":{dim},"n_cent":{n_cent},"rot_seed":41,"metric":"{metric_str}"}}]"#,
        );
        (blob, json, all)
    }

    #[tokio::test]
    async fn open_lazy_small_sq8_superfile_fetches_exact_metadata_ranges() {
        let (blob, json, _) =
            build_small_superfile(32, 4, 64, RerankCodec::Sq8Residual, Metric::L2Sq);
        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_counter = counting.async_counter();

        let _reader = VectorReader::open_lazy(
            StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
            &json,
            OpenOptions::for_object_store(),
        )
        .await
        .expect("open_lazy small Sq8");

        let n_calls = async_counter.load(AtomicOrdering::Relaxed);
        assert_eq!(
            n_calls, 3,
            "small Sq8 open_lazy must issue exactly 3 async range calls \
             (outer header, directory+crc, subsection header); \
             observed {n_calls}",
        );
    }

    #[tokio::test]
    async fn open_lazy_small_superfile_fetches_no_codec_meta_for_non_sq8() {
        for codec in [RerankCodec::Fp32, RerankCodec::RabitqOnly] {
            let (blob, json, _) = build_small_superfile(32, 4, 64, codec, Metric::L2Sq);
            let counting = StdArc::new(CountingLazyByteSource::new(blob));
            let async_counter = counting.async_counter();

            let _reader = VectorReader::open_lazy(
                StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
                &json,
                OpenOptions::for_object_store(),
            )
            .await
            .unwrap_or_else(|e| panic!("open_lazy {codec:?}: {e:?}"));

            let n_calls = async_counter.load(AtomicOrdering::Relaxed);
            assert_eq!(
                n_calls, 3,
                "open_lazy ({codec:?}) must issue exactly 3 async range calls \
                 (outer header, directory+crc, subsection header); observed {n_calls}",
            );
        }
    }

    /// round-trip parity. A search against an
    /// `open_lazy` reader returns the same `(doc_id, distance)`
    /// hits as the eager `open()` path. Confirms the open-path
    /// refactor (Phase A sub-header + Phase B codec_meta) and
    /// the overlay round-trip preserve every search-critical
    /// metadata field.
    #[tokio::test]
    async fn open_lazy_search_matches_eager_open_per_codec() {
        for codec in [
            RerankCodec::Fp32,
            RerankCodec::Sq8Residual,
            RerankCodec::RabitqOnly,
        ] {
            let (blob, json, all) = build_small_superfile(32, 4, 64, codec, Metric::L2Sq);
            let r_eager = VectorReader::open(blob.clone(), &json)
                .unwrap_or_else(|e| panic!("eager open {codec:?}: {e:?}"));
            let counting = StdArc::new(CountingLazyByteSource::new(blob));
            let r_lazy = VectorReader::open_lazy(
                StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
                &json,
                OpenOptions::for_object_store(),
            )
            .await
            .unwrap_or_else(|e| panic!("open_lazy {codec:?}: {e:?}"));

            for &q_idx in &[0usize, 7, 17, 31] {
                let hits_eager = r_eager
                    .search("v", &all[q_idx], 5, 4, 5)
                    .await
                    .unwrap_or_else(|e| panic!("eager search {codec:?}: {e:?}"));
                let hits_lazy = r_lazy
                    .search("v", &all[q_idx], 5, 4, 5)
                    .await
                    .unwrap_or_else(|e| panic!("lazy search {codec:?}: {e:?}"));
                assert_eq!(
                    hits_eager, hits_lazy,
                    "search results must match between eager and lazy open \
                     (codec {codec:?}, query {q_idx})",
                );
            }
        }
    }

    /// Cold first search after `open_lazy` issues at most
    /// `nprobe + 2` underlying async range GETs against the
    /// LazyByteSource: centroids, cluster_idx, and one interleaved
    /// cluster block per probed non-empty cluster. Rerank adds no
    /// extra GET because full vectors ride inside the cluster blocks.
    ///
    /// Headline budget for the cold first-search phase
    /// (≤ 12 ranges, ≤ 5 MB at 1M × 384 sq8, nprobe = 8). The
    /// small-superfile test here pins the structural shape; the
    /// s3s-fs bench measures the real wall-clock against AWS-
    /// shape RTTs.
    ///
    /// "At most" because some probed clusters can be empty
    /// (zero-count entries skip the block fetch entirely); for a
    /// well-distributed corpus the budget is hit exactly.
    #[tokio::test]
    async fn cold_first_search_after_open_lazy_within_nprobe_plus_one_ranges() {
        let (blob, json, all) =
            build_small_superfile(32, 8, 128, RerankCodec::Sq8Residual, Metric::L2Sq);

        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_counter = counting.async_counter();
        // Disable BytesLazyByteSource's zero-copy sync path so
        // every non-overlay read is forced through the async
        // `range` bridge — that's what an object-store-backed
        // source actually pays per region.
        counting.disable_sync();

        let r_lazy = VectorReader::open_lazy(
            StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
            &json,
            OpenOptions::for_object_store(),
        )
        .await
        .expect("open_lazy");

        let after_open = async_counter.load(AtomicOrdering::Relaxed);
        assert_eq!(
            after_open, 3,
            "Sq8 open_lazy must issue exactly the open-time metadata ranges \
             (header, directory, subheader); codec_meta is deferred to the \
             first search on the object-store path; observed {after_open}",
        );

        let nprobe = 4usize;
        let _hits = r_lazy
            .search("v", &all[0], 5, nprobe, 5)
            .await
            .expect("cold first search");

        let after_search = async_counter.load(AtomicOrdering::Relaxed);
        let search_calls = after_search - after_open;
        let max_expected = (nprobe + 1) as u64;
        assert!(
            search_calls <= max_expected,
            "cold first search at nprobe={nprobe} must issue ≤ {max_expected} async \
             range GETs (centroids+cluster_idx + one interleaved block per probed \
             cluster); observed {search_calls}",
        );
        assert!(
            search_calls >= 2,
            "cold first search must issue at least 2 async range GETs (centroids+ \
             cluster_idx + ≥1 cluster block); observed {search_calls} suggests \
             search accidentally short-circuited the cold fetch paths",
        );
    }

    /// cold first search must dispatch its
    /// per-cluster block fetches **concurrently**, not
    /// serially. The total range-GET count was already
    /// pinned by the range-budget test above; this test pins
    /// the round-trip count.
    ///
    /// Each `range()` call holds an in-flight slot (RAII
    /// guard); peak in-flight ≥ 2 proves the cluster fetches
    /// overlapped. We pad `range()` with a small artificial
    /// latency so a serial implementation completes each
    /// future before the next is awaited — without the
    /// latency, the trivial `bytes.slice(...)` body
    /// resolves instantly and even a serial caller looks
    /// concurrent (in-flight peaks at 1 indistinguishably).
    ///
    /// Runs on the multi-thread runtime for the same
    /// `block_in_place` reason as the range-budget test above.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cold_first_search_dispatches_cluster_gets_concurrently() {
        let (blob, json, all) =
            build_small_superfile(32, 8, 256, RerankCodec::Sq8Residual, Metric::L2Sq);

        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        let async_counter = counting.async_counter();
        let max_in_flight = counting.max_in_flight_counter();
        counting.disable_sync();
        counting.set_async_latency(std::time::Duration::from_millis(5));

        let r_lazy = VectorReader::open_lazy(
            StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
            &json,
            OpenOptions::for_object_store(),
        )
        .await
        .expect("open_lazy");

        // Reset max_in_flight after open (we only want to
        // pin the search-side dispatch shape; open is its
        // own budget exercise).
        max_in_flight.store(0, AtomicOrdering::Release);
        let async_after_open = async_counter.load(AtomicOrdering::Relaxed);

        let nprobe = 8usize;
        let q = all[0].clone();
        let hits = r_lazy
            .search("v", &q, 5, nprobe, 5)
            .await
            .expect("cold first search");
        assert!(!hits.is_empty(), "self-query should return ≥ 1 hit");

        let peak = max_in_flight.load(AtomicOrdering::Acquire);
        let search_calls = async_counter.load(AtomicOrdering::Relaxed) - async_after_open;
        if search_calls >= 3 {
            // When coalescing still leaves multiple search-side
            // ranges, they must overlap. A serial dispatch
            // peaks at exactly 1.
            assert!(
                peak >= 2,
                "cold first search per-cluster fetches must overlap when multiple \
                 search-side ranges remain (peak in-flight ≥ 2); observed {peak} \
                 across {search_calls} calls",
            );
        } else {
            assert!(
                peak >= 1,
                "coalesced cold first search should still issue at least one \
                 search-side async range; observed peak={peak}, calls={search_calls}",
            );
        }
    }

    /// round-trip parity for the unified
    /// codes+doc_ids per-cluster fetch path. The combined block
    /// gets sliced into a `codes` prefix and `doc_ids` suffix
    /// inside the search hot loop; this test pins that the
    /// slice boundaries land at exactly `count * code_bytes`
    /// (i.e. the bit-identical results survive the refactor
    /// from two separate ranges to one combined block).
    #[tokio::test]
    async fn m3_combined_cluster_fetch_matches_eager_open_per_codec() {
        for codec in [
            RerankCodec::Fp32,
            RerankCodec::Sq8Residual,
            RerankCodec::RabitqOnly,
        ] {
            let (blob, json, all) = build_small_superfile(32, 4, 64, codec, Metric::L2Sq);
            let r_eager = VectorReader::open(blob.clone(), &json)
                .unwrap_or_else(|e| panic!("eager open {codec:?}: {e:?}"));
            let counting = StdArc::new(CountingLazyByteSource::new(blob));
            let r_lazy = VectorReader::open_lazy(
                StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
                &json,
                OpenOptions::for_object_store(),
            )
            .await
            .unwrap_or_else(|e| panic!("open_lazy {codec:?}: {e:?}"));

            for &q_idx in &[0usize, 7, 17, 31] {
                let hits_eager = r_eager
                    .search("v", &all[q_idx], 5, 4, 5)
                    .await
                    .unwrap_or_else(|e| panic!("eager search {codec:?}: {e:?}"));
                let hits_lazy = r_lazy
                    .search("v", &all[q_idx], 5, 4, 5)
                    .await
                    .unwrap_or_else(|e| panic!("lazy search {codec:?}: {e:?}"));
                assert_eq!(
                    hits_eager, hits_lazy,
                    "combined cluster fetch must produce bit-identical search \
                     results vs eager (codec {codec:?}, query {q_idx})",
                );
            }
        }
    }

    /// pins the `cluster_block_range` address math
    /// against the per-cluster block spec
    /// (`[codes: cnt*cb][doc_ids: cnt*4]`). Walks every non-
    /// empty cluster and checks the block range size matches
    /// `cnt × (cb + 4)` exactly, the start aligns with
    /// `per_cluster_blocks_off + doc_off × (cb + 4)`, and the
    /// codes/doc_ids halves slice in at the expected boundary
    /// inside the fetched block.
    #[test]
    fn cluster_block_range_matches_v1_layout_invariant() {
        let (blob, json, _) =
            build_small_superfile(32, 4, 64, RerankCodec::Sq8Residual, Metric::L2Sq);
        let r = VectorReader::open(blob, &json).expect("open");
        let col = &r.columns[0];
        let cb = col.quant.code_bytes();
        let pvb = col.rerank_codec.per_vector_bytes(col.dim);
        // Interleaved layout: each per-cluster block is
        // `[codes][doc_ids][full]` at stride `cb + 4 + pvb`.
        let stride = cb + 4 + pvb;

        let cluster_idx_bytes = r
            .source
            .try_get_range_sync(
                col.subsection_range.start + col.cluster_idx_off
                    ..col.subsection_range.start
                        + col.cluster_idx_off
                        + (col.n_cent as usize) * CLUSTER_IDX_ENTRY_BYTES,
            )
            .expect("cluster_idx must be resident in InMemory source");

        let mut n_non_empty = 0usize;
        for c in 0..col.n_cent {
            let (off, cnt) = read_cluster_entry(&cluster_idx_bytes, c as usize);
            if cnt == 0 {
                continue;
            }
            n_non_empty += 1;
            let block = col.cluster_block_range(off, cnt);
            let expected_start =
                col.subsection_range.start + col.per_cluster_blocks_off + (off as usize) * stride;
            let expected_len = (cnt as usize) * stride;
            assert_eq!(
                block.start, expected_start,
                "cluster {c} block start must equal \
                 per_cluster_blocks_off + doc_off × stride",
            );
            assert_eq!(
                block.len(),
                expected_len,
                "cluster {c} block size must equal cnt × (cb + 4 + per_vec_bytes)",
            );
            // Inside the fetched block, `[0..cnt*cb)` is codes,
            // `[cnt*cb..cnt*(cb+4))` is doc_ids, and the remaining
            // `cnt*pvb` bytes are the interleaved full[] vectors —
            // the exact boundaries the search() hot path slices at.
            let codes_end = (cnt as usize) * cb;
            let doc_ids_end = codes_end + (cnt as usize) * 4;
            assert!(
                doc_ids_end <= block.len(),
                "codes + doc_ids prefix must fit inside the block"
            );
            assert_eq!(
                block.len() - doc_ids_end,
                (cnt as usize) * pvb,
                "full suffix must be cnt × per_vec_bytes bytes",
            );
        }
        assert!(
            n_non_empty > 0,
            "test corpus must populate at least one cluster"
        );
    }

    /// verify the `Source::Lazy` reader constructed
    /// by `open_lazy` exposes the same column metadata as the
    /// eager reader (dim, n_cent, n_docs, codec, sq8_meta shape).
    /// The structural decode that produces `ColumnReader` runs
    /// against the overlay; this test pins that every parsed
    /// field surfaces unchanged.
    #[tokio::test]
    async fn open_lazy_column_metadata_matches_eager_open() {
        let (blob, json, _) =
            build_small_superfile(32, 4, 64, RerankCodec::Sq8Residual, Metric::L2Sq);
        let r_eager = VectorReader::open(blob.clone(), &json).expect("eager open");
        let counting = StdArc::new(CountingLazyByteSource::new(blob));
        // Simulate the object-store path: with no zero-copy sync read
        // available, open defers Sq8 codec_meta to the first search,
        // so the lazy column resolves to `Sq8ColumnMeta::Lazy`.
        counting.disable_sync();
        let r_lazy = VectorReader::open_lazy(
            StdArc::clone(&counting) as StdArc<dyn LazyByteSource>,
            &json,
            OpenOptions::for_object_store(),
        )
        .await
        .expect("open_lazy");

        assert_eq!(r_eager.columns.len(), r_lazy.columns.len());
        let col_eager = &r_eager.columns[0];
        let col_lazy = &r_lazy.columns[0];
        assert_eq!(col_eager.name, col_lazy.name);
        assert_eq!(col_eager.dim, col_lazy.dim);
        assert_eq!(col_eager.n_cent, col_lazy.n_cent);
        assert_eq!(col_eager.n_docs, col_lazy.n_docs);
        assert_eq!(col_eager.rerank_codec, col_lazy.rerank_codec);
        assert_eq!(col_eager.metric, col_lazy.metric);

        let meta_eager = col_eager.sq8_meta.as_ref().expect("eager Sq8 meta");
        let meta_lazy = col_lazy.sq8_meta.as_ref().expect("lazy Sq8 meta");
        assert!(
            matches!(meta_eager, Sq8ColumnMeta::Eager { .. }),
            "eager open should materialise Sq8 metadata"
        );
        assert!(
            matches!(meta_lazy, Sq8ColumnMeta::Lazy { .. }),
            "lazy open should defer Sq8 metadata to search"
        );
    }

    #[test]
    fn get_vectors_fp32_returns_vectors_in_original_order() {
        let n_docs = 64u32;
        let dim = 16;
        let n_cent = 4;

        // Build a blob with Fp32 encoding
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "embedding".into(),
            dim,
            n_cent,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
        })
        .expect("register column");

        // Create deterministic vectors
        let mut input_vectors = Vec::new();
        for i in 0..n_docs {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i.wrapping_mul(31) + j as u32) % 100) as f32 * 0.01)
                .collect();
            input_vectors.push(v.clone());
            b.add(0, &v).expect("add to vector builder");
        }

        let bytes = b.finish().expect("finish vector builder");
        let json = format!(
            r#"[{{"column":"embedding","dim":{dim},"n_cent":{n_cent},"rot_seed":7,"metric":"l2sq"}}]"#
        );
        let reader = VectorReader::open(Bytes::from(bytes), &json).expect("open should succeed");

        // Retrieve vectors via the new function
        let retrieved = reader
            .get_vectors_fp32("embedding")
            .expect("get_vectors_fp32 should succeed");

        // Verify all vectors are returned
        assert_eq!(retrieved.len(), n_docs as usize);

        // Verify vectors match original vectors (within floating point precision)
        for (i, retrieved_vec) in retrieved.iter().enumerate() {
            assert_eq!(retrieved_vec.len(), dim);
            for (j, &val) in retrieved_vec.iter().enumerate() {
                let expected = input_vectors[i][j];
                assert!(
                    (val - expected).abs() < 1e-6,
                    "vector {} dimension {} mismatch: got {}, expected {}",
                    i,
                    j,
                    val,
                    expected
                );
            }
        }
    }

    #[test]
    fn get_vectors_fp32_rejects_non_fp32_codec() {
        // blob was built with Sq8Residual by default, not Fp32
        let mut builder = VectorBuilder::new();
        builder
            .register_column(VectorConfig {
                column: "embedding".into(),
                dim: 16,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::L2Sq,
                rerank_codec: RerankCodec::Sq8Residual,
            })
            .expect("register column");
        for i in 0u32..32 {
            let v: Vec<f32> = (0..16)
                .map(|j| ((i.wrapping_mul(31) + j as u32) % 100) as f32 * 0.01)
                .collect();
            builder.add(0, &v).expect("add");
        }
        let sq8_bytes = builder.finish().expect("finish");
        let sq8_json =
            r#"[{"column":"embedding","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#
                .to_string();
        let reader = VectorReader::open(Bytes::from(sq8_bytes), &sq8_json).expect("open");

        // Should error because codec is Sq8Residual, not Fp32
        let result = reader.get_vectors_fp32("embedding");
        assert!(result.is_err());
        if let Err(VectorError::Read(ReadError::MalformedVersion(msg))) = result {
            assert!(msg.contains("Fp32"));
        } else {
            panic!("expected MalformedVersion error, got {:?}", result);
        }
    }

    #[test]
    fn get_vectors_fp32_rejects_unknown_column() {
        let (blob, json) = build_blob(32, 16, 4, Metric::L2Sq);
        let reader = VectorReader::open(blob, &json).expect("open should succeed");

        let result = reader.get_vectors_fp32("nonexistent");
        assert!(matches!(result, Err(VectorError::UnknownColumn(_))));
    }

    #[test]
    fn get_vectors_fp32_returns_empty_for_no_docs() {
        let mut b = VectorBuilder::new();
        b.register_column(VectorConfig {
            column: "embedding".into(),
            dim: 16,
            n_cent: 4,
            rot_seed: 7,
            metric: Metric::L2Sq,
            rerank_codec: RerankCodec::Fp32,
        })
        .expect("register column");
        let bytes = b.finish().expect("finish vector builder");
        let json = r#"[{"column":"embedding","dim":16,"n_cent":4,"rot_seed":7,"metric":"l2sq"}]"#
            .to_string();
        let reader = VectorReader::open(Bytes::from(bytes), &json).expect("open should succeed");

        let retrieved = reader
            .get_vectors_fp32("embedding")
            .expect("get_vectors_fp32 should succeed");
        assert!(retrieved.is_empty());
    }
}
