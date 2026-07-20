// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Slow-CAS vector-state blob: a table's superfile entries, verbatim, in one
//! content-addressed object.
//!
//! The manifest's fast CAS (pointer + list) churns on every commit of any
//! sort, and every rewritten part gets a fresh identity — so routing state
//! carried only by the parts is re-fetched whenever anything changes. This
//! blob gives the drain-owned slow state (per-superfile fp32 centroids +
//! offsets inside each [`SuperfileEntry`]) its own identity: drain / hidden
//! compaction publish it after membership settles, `ManifestSnapshot::update` clears
//! the reference on any membership change, and every other manifest
//! transition (deleted-id stamps, user commits) preserves it — so a loaded
//! consumer keeps its decoded entries in memory until the drainer actually
//! replaces them.
//!
//! Format: the blob IS a [`ManifestPart`] encoding (`part::encode` /
//! `part::decode`) with the nil part id — zero new entry serialization.
//! Same logical entries produce byte-identical blobs and therefore the same
//! content-addressed URI, so republishing unchanged state is a no-op PUT.
//! This module owns only the storage format and fetch/verify discipline; the
//! decoded entries live in the hydrated `ManifestSnapshot` (there is deliberately no
//! separate cache).

use std::{collections::HashMap, io, mem::size_of, ops::Range, os::unix::fs::FileExt, sync::Arc};

use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, stream};
use tempfile::NamedTempFile;
use tokio::task::spawn_blocking;
use uuid::Uuid;

use crate::{
    storage::{StorageError, StorageProvider},
    supertable::manifest::{
        SuperfileEntry, VectorSummary,
        encoding::SummaryWireMode,
        list::RoutingRef,
        part::{self, ContentHash, ManifestPart, PartId},
    },
};

/// Versioned envelope used only while a drain checkpoint is active. Final
/// settled state keeps the legacy manifest-part-only encoding.
const CHECKPOINT_MAGIC: &[u8; 8] = b"INFSVS02";
const CHECKPOINT_HEADER_BYTES: usize = CHECKPOINT_MAGIC.len() + 3 * size_of::<u64>();
const CHECKPOINT_VISIBLE_LEN_OFF: usize = CHECKPOINT_MAGIC.len();
const CHECKPOINT_METADATA_LEN_OFF: usize = CHECKPOINT_VISIBLE_LEN_OFF + size_of::<u64>();
const CHECKPOINT_PENDING_LEN_OFF: usize = CHECKPOINT_METADATA_LEN_OFF + size_of::<u64>();

/// Object-storage prefix for content-addressed slow vector-state blobs,
/// relative to the owning table's storage provider (the hidden table's
/// provider is prefixed, so blobs land under the hidden subtree and request
/// metering attributes them to the hidden index automatically).
pub(crate) const STORAGE_PREFIX: &str = "slow-vector-state/";

/// Bytes per striped range-GET when fetching a large slow-state blob. One
/// HTTP stream tops out well under NIC line rate (~200–400 MB/s measured on
/// Azure), and at 100M docs the blob is multi-GiB — a single `get` put ~10 s
/// of pure transfer into every cold open. 64 MiB per range keeps the
/// request count negligible against GET pricing while letting the streams
/// aggregate toward line rate.
const STRIPED_FETCH_CHUNK_BYTES: u64 = 64 << 20;

/// Concurrent range-GETs per striped blob fetch. Matches the connection
/// count one host can productively drive against Azure/S3 before the
/// per-stream gain flattens.
const STRIPED_FETCH_MAX_IN_FLIGHT: usize = 16;

/// Object-storage path for a content-addressed slow vector-state blob.
pub(crate) fn storage_path(hash: &ContentHash) -> String {
    format!("{STORAGE_PREFIX}state-{}.bin", hash.to_hex())
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SlowVectorStateError {
    #[error("storage: {0}")]
    Storage(String),
    #[error("content hash mismatch")]
    HashMismatch,
    #[error("state parse: {0}")]
    Parse(String),
}

#[derive(Debug, Clone)]
pub(crate) struct PendingDrainState {
    /// Opaque drain-epoch metadata owned by `writer.rs`.
    pub metadata: Vec<u8>,
    /// Uploaded, not-yet-visible worker shard entries.
    pub entries: Vec<Arc<SuperfileEntry>>,
}

#[derive(Debug, Clone)]
pub(crate) struct SlowVectorState {
    /// Current manifest-visible hidden membership.
    pub entries: Vec<Arc<SuperfileEntry>>,
    /// In-progress drain completion state, absent after final publication.
    pub pending_drain: Option<PendingDrainState>,
}

/// Serialize `entries` through the manifest-part codec in ROUTING wire
/// form — cluster blocks as counts + 1-bit admit slab, no fp32. This is
/// the ONLY entry encoding the state blob uses: fp32 lives in exactly one
/// CAS object per generation (the centroid section), and every reader —
/// consumer or writer — hydrates entries from this routing-shaped blob
/// and reaches fp32 through the section. The part id is the nil UUID:
/// the blob is not a real part, and a constant id keeps the encoding
/// deterministic (same entries ⇒ same bytes ⇒ same [`ContentHash`]).
pub(crate) fn encode_entries(entries: &[Arc<SuperfileEntry>]) -> Vec<u8> {
    encode_entries_with_mode(entries, SummaryWireMode::RoutingOnly)
}

/// Full-wire encoding (fp32 inline) — ONLY for the drain checkpoint's
/// PENDING segment: pending entries are writer-only crash-resume state,
/// not consumer-visible membership, and the resume path needs their fp32
/// before any section covering them exists.
fn encode_entries_full(entries: &[Arc<SuperfileEntry>]) -> Vec<u8> {
    encode_entries_with_mode(entries, SummaryWireMode::Full)
}

fn encode_entries_with_mode(entries: &[Arc<SuperfileEntry>], mode: SummaryWireMode) -> Vec<u8> {
    let synthetic = ManifestPart {
        format_version: part::FORMAT_VERSION.into(),
        part_id: PartId(Uuid::nil()),
        superfiles: entries.to_vec(),
    };
    part::encode_with_mode(&synthetic, mode)
}

/// Decode a blob written by [`encode_entries`].
pub(crate) fn decode_entries(
    bytes: &[u8],
) -> Result<Vec<Arc<SuperfileEntry>>, SlowVectorStateError> {
    let decoded = part::decode(bytes).map_err(|e| SlowVectorStateError::Parse(e.to_string()))?;
    Ok(decoded.superfiles)
}

fn encode_checkpoint_state(
    entries: &[Arc<SuperfileEntry>],
    pending: &PendingDrainState,
) -> Vec<u8> {
    let visible = encode_entries(entries);
    // Pending entries keep fp32 inline: they are the crash-resume state a
    // restarted drain continues from, and the published section covers
    // only the VISIBLE membership.
    let pending_entries = encode_entries_full(&pending.entries);
    let mut bytes = Vec::with_capacity(
        CHECKPOINT_HEADER_BYTES + visible.len() + pending.metadata.len() + pending_entries.len(),
    );
    bytes.extend_from_slice(CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&(visible.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(pending.metadata.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(pending_entries.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&visible);
    bytes.extend_from_slice(&pending.metadata);
    bytes.extend_from_slice(&pending_entries);
    bytes
}

pub(crate) fn decode_state(bytes: &[u8]) -> Result<SlowVectorState, SlowVectorStateError> {
    if !bytes.starts_with(CHECKPOINT_MAGIC) {
        return Ok(SlowVectorState {
            entries: decode_entries(bytes)?,
            pending_drain: None,
        });
    }
    if bytes.len() < CHECKPOINT_HEADER_BYTES {
        return Err(SlowVectorStateError::Parse(
            "checkpoint header truncated".into(),
        ));
    }
    let visible_len = u64::from_le_bytes(
        bytes[CHECKPOINT_VISIBLE_LEN_OFF..CHECKPOINT_METADATA_LEN_OFF]
            .try_into()
            .expect("checkpoint visible length slice"),
    ) as usize;
    let metadata_len = u64::from_le_bytes(
        bytes[CHECKPOINT_METADATA_LEN_OFF..CHECKPOINT_PENDING_LEN_OFF]
            .try_into()
            .expect("checkpoint metadata length slice"),
    ) as usize;
    let pending_len = u64::from_le_bytes(
        bytes[CHECKPOINT_PENDING_LEN_OFF..CHECKPOINT_HEADER_BYTES]
            .try_into()
            .expect("checkpoint entries length slice"),
    ) as usize;
    let visible_end = CHECKPOINT_HEADER_BYTES
        .checked_add(visible_len)
        .ok_or_else(|| SlowVectorStateError::Parse("checkpoint length overflow".into()))?;
    let metadata_end = visible_end
        .checked_add(metadata_len)
        .ok_or_else(|| SlowVectorStateError::Parse("checkpoint length overflow".into()))?;
    let pending_end = metadata_end
        .checked_add(pending_len)
        .ok_or_else(|| SlowVectorStateError::Parse("checkpoint length overflow".into()))?;
    if pending_end != bytes.len() {
        return Err(SlowVectorStateError::Parse(
            "checkpoint envelope length mismatch".into(),
        ));
    }
    Ok(SlowVectorState {
        entries: decode_entries(&bytes[CHECKPOINT_HEADER_BYTES..visible_end])?,
        pending_drain: Some(PendingDrainState {
            metadata: bytes[visible_end..metadata_end].to_vec(),
            entries: decode_entries(&bytes[metadata_end..pending_end])?,
        }),
    })
}

async fn write_bytes(
    storage: &dyn StorageProvider,
    bytes: Vec<u8>,
) -> Result<(String, ContentHash), SlowVectorStateError> {
    let content_hash = ContentHash::of(&bytes);
    let uri = storage_path(&content_hash);
    match storage.put_atomic(&uri, Bytes::from(bytes)).await {
        Ok(_) | Err(StorageError::PreconditionFailed { .. }) => {}
        Err(error) => return Err(SlowVectorStateError::Storage(error.to_string())),
    }
    Ok((uri, content_hash))
}

/// References to one published slow-state generation — exactly TWO
/// content-addressed objects, each byte stored once in the layout its
/// reader needs:
///
/// * the state blob (`uri`/`content_hash`) — routing-shaped entries
///   (counts + 1-bit admit slabs, no fp32) plus any drain checkpoint
///   state. Everyone hydrates from it: consumer opens and writer opens
///   alike.
/// * the centroid section (`centroids`) — every summary cell's fp32
///   fine centroids, raw and contiguous. Anyone needing exact centroid
///   scores (the admit rescore, grid bootstrap, maintenance) fetches it
///   once per generation and reads cells locally.
#[derive(Debug, Clone)]
pub(crate) struct PublishedState {
    pub uri: String,
    pub content_hash: ContentHash,
    pub centroids: RoutingRef,
}

/// Contiguous fp32 fine-centroid section for the stripped-summary admit
/// rescore: every visible entry's summary cells' centroids concatenated
/// in `(entry order, column name order, cell order)` — cluster-major
/// f32 LE per cell. No embedded index: the state blob carries each
/// cell's `n_cent`/`dim`, so a consumer walking the same order computes
/// identical offsets ([`section_len`] is the cross-check).
///
/// fp32 is stored ONCE per generation (here — entries hydrate stripped),
/// so a republish composes each cell from the first available source:
/// resident fp32 (entries built by this maintenance pass) or the
/// PREVIOUS generation's section (carried-forward entries — superfiles
/// are immutable, so their centroids never change between generations).
/// A cell reachable from neither is a caller bug: membership only ever
/// adds freshly-built entries or carries forward sectioned ones.
pub(crate) fn compose_centroid_section(
    entries: &[Arc<SuperfileEntry>],
    previous: Option<&CentroidSection>,
) -> Result<Vec<u8>, SlowVectorStateError> {
    let mut out = Vec::with_capacity(section_len(entries));
    for entry in entries {
        for (column, summary) in sorted_summaries(entry) {
            for cell in &summary.cells {
                let expected = cell.clusters.n_cent as usize * cell.clusters.dim as usize;
                if expected == 0 {
                    continue;
                }
                if cell.clusters.vectors_resident() {
                    for &v in &cell.clusters.centroids {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    continue;
                }
                let carried = previous
                    .map(|section| section.read_cell(entry.superfile_id, column, cell.cell_id))
                    .transpose()
                    .map_err(|e| {
                        SlowVectorStateError::Storage(format!("previous section read: {e}"))
                    })?
                    .flatten()
                    .filter(|fp32| fp32.len() == expected);
                match carried {
                    Some(fp32) => {
                        for &v in &fp32 {
                            out.extend_from_slice(&v.to_le_bytes());
                        }
                    }
                    None => {
                        return Err(SlowVectorStateError::Parse(format!(
                            "centroid section compose: no fp32 source for superfile {} column \
                             {column} cell {:?} (stripped entry and no covering previous \
                             section)",
                            entry.superfile_id, cell.cell_id
                        )));
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Byte length [`encode_centroid_section`] produces for `entries`,
/// computable from stripped summaries (`n_cent`/`dim` survive the strip)
/// — consumers verify their walk against the published blob size.
pub(crate) fn section_len(entries: &[Arc<SuperfileEntry>]) -> usize {
    let mut total = 0usize;
    for entry in entries {
        for (_, summary) in sorted_summaries(entry) {
            for cell in &summary.cells {
                total += cell.clusters.n_cent as usize * cell.clusters.dim as usize * 4;
            }
        }
    }
    total
}

/// One entry's vector summaries in column-name order — the deterministic
/// iteration both the section encoder and the consumer offset walk share.
pub(crate) fn sorted_summaries(
    entry: &SuperfileEntry,
) -> impl Iterator<Item = (&String, &VectorSummary)> {
    let mut summaries: Vec<_> = entry.vector_summary.iter().collect();
    summaries.sort_by(|a, b| a.0.cmp(b.0));
    summaries.into_iter()
}

/// One cell's slice inside a spilled centroid section.
struct SectionCell {
    cell_id: Option<u32>,
    offset: u64,
    n_cent: u32,
    dim: u32,
}

/// One hydrated centroid-section generation: the section bytes spilled to
/// a local temp file (served by `pread` — page-cache-backed, evictable,
/// no `unsafe`) plus the per-`(superfile, column)` cell offsets from the
/// shared deterministic walk. Content-addressed by `uri`; a new drain
/// generation publishes a new URI and replaces the cached instance.
pub(crate) struct CentroidSection {
    uri: String,
    spill: NamedTempFile,
    cells: HashMap<(Uuid, String), Vec<SectionCell>>,
}

impl CentroidSection {
    /// Content-addressed URI this section was fetched from.
    pub(crate) fn uri(&self) -> &str {
        &self.uri
    }

    /// Read one cell's fp32 fine centroids (cluster-major, `n_cent × dim`).
    /// `Ok(None)` when the `(superfile, column, cell)` triple is not in the
    /// section; a spill-read fault is an error, never "absent" — mapping it
    /// to `None` would silently reroute the caller to a fallback source.
    pub(crate) fn read_cell(
        &self,
        superfile_id: Uuid,
        column: &str,
        cell_id: Option<u32>,
    ) -> io::Result<Option<Vec<f32>>> {
        let Some(cells) = self.cells.get(&(superfile_id, column.to_owned())) else {
            return Ok(None);
        };
        let Some(cell) = cells.iter().find(|c| c.cell_id == cell_id) else {
            return Ok(None);
        };
        let len = cell.n_cent as usize * cell.dim as usize * 4;
        let mut buf = vec![0u8; len];
        self.spill.as_file().read_exact_at(&mut buf, cell.offset)?;
        Ok(Some(
            buf.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("chunks_exact(4)")))
                .collect(),
        ))
    }
}

/// Fetch the centroid-section sibling, verify its hash, spill it to a
/// local temp file, and index it with the shared offset walk over
/// `entries` (which must be the same visible membership, in the same
/// order, that [`write_state`] published — the size cross-check fails
/// loudly on any drift and callers fall back to per-superfile reads).
pub(crate) async fn fetch_centroid_section(
    storage: &dyn StorageProvider,
    reference: &RoutingRef,
    entries: &[Arc<SuperfileEntry>],
) -> Result<CentroidSection, SlowVectorStateError> {
    let store_err = |e: StorageError| SlowVectorStateError::Storage(e.to_string());
    let io_err = |e: io::Error| SlowVectorStateError::Storage(format!("section spill: {e}"));
    let expected_len = section_len(entries) as u64;
    let meta = storage.head(&reference.uri).await.map_err(store_err)?;
    if meta.size != expected_len {
        return Err(SlowVectorStateError::Parse(format!(
            "centroid section is {} bytes, membership walk expects {expected_len}",
            meta.size
        )));
    }
    let spill = NamedTempFile::new().map_err(io_err)?;
    let mut hasher = blake3::Hasher::new();
    // Same striping policy as [`fetch_blob_striped`]: parallel range-GETs,
    // bounded in flight. `buffered` yields chunks in range order, so each
    // is hashed and written at its offset as it lands, then dropped —
    // peak memory stays at in-flight × chunk, never the whole section.
    let ranges = striped_ranges(meta.size, STRIPED_FETCH_CHUNK_BYTES);
    let mut chunks = stream::iter(ranges.into_iter().map(|range| {
        let uri = &reference.uri;
        async move {
            let offset = range.start;
            storage.get_range(uri, range).await.map(|b| (offset, b))
        }
    }))
    .buffered(STRIPED_FETCH_MAX_IN_FLIGHT);
    while let Some(chunk) = chunks.next().await {
        let (offset, bytes) = chunk.map_err(store_err)?;
        hasher.update(&bytes);
        spill
            .as_file()
            .write_all_at(&bytes, offset)
            .map_err(io_err)?;
    }
    if ContentHash(*hasher.finalize().as_bytes()) != reference.content_hash {
        return Err(SlowVectorStateError::HashMismatch);
    }

    let mut cells: HashMap<(Uuid, String), Vec<SectionCell>> = HashMap::new();
    let mut cursor = 0u64;
    for entry in entries {
        for (column, summary) in sorted_summaries(entry) {
            let mut list = Vec::with_capacity(summary.cells.len());
            for cell in &summary.cells {
                list.push(SectionCell {
                    cell_id: cell.cell_id,
                    offset: cursor,
                    n_cent: cell.clusters.n_cent,
                    dim: cell.clusters.dim,
                });
                cursor += cell.clusters.n_cent as u64 * cell.clusters.dim as u64 * 4;
            }
            cells.insert((entry.superfile_id, column.clone()), list);
        }
    }
    Ok(CentroidSection {
        uri: reference.uri.clone(),
        spill,
        cells,
    })
}

/// Content-address and PUT the blob for `entries` plus its routing
/// sibling. Idempotent: URIs are hash-derived, so a raced identical PUT
/// surfacing [`StorageError::PreconditionFailed`] means the bytes are
/// already durable. Visibility is decided by the manifest-list ref
/// stamp, not by these PUTs.
pub(crate) async fn write_state(
    storage: &dyn StorageProvider,
    entries: &[Arc<SuperfileEntry>],
    previous_section: Option<&CentroidSection>,
) -> Result<PublishedState, SlowVectorStateError> {
    write_blob_and_section(storage, encode_entries(entries), entries, previous_section).await
}

/// Publish current visible membership plus an in-progress drain checkpoint in
/// the same content-addressed slow-CAS state referenced by the hidden manifest.
/// Checkpoint (pending) state rides the routing-shaped blob; consumers ignore
/// it, the drain-resume path reads it.
pub(crate) async fn write_state_with_pending_drain(
    storage: &dyn StorageProvider,
    entries: &[Arc<SuperfileEntry>],
    pending: &PendingDrainState,
    previous_section: Option<&CentroidSection>,
) -> Result<PublishedState, SlowVectorStateError> {
    write_blob_and_section(
        storage,
        encode_checkpoint_state(entries, pending),
        entries,
        previous_section,
    )
    .await
}

/// PUT the routing-shaped state blob and the fp32 centroid section
/// concurrently — the two-object generation described on
/// [`PublishedState`].
async fn write_blob_and_section(
    storage: &dyn StorageProvider,
    blob_bytes: Vec<u8>,
    entries: &[Arc<SuperfileEntry>],
    previous_section: Option<&CentroidSection>,
) -> Result<PublishedState, SlowVectorStateError> {
    let centroid_bytes = compose_centroid_section(entries, previous_section)?;
    let (blob, centroids) = tokio::join!(
        write_bytes(storage, blob_bytes),
        write_bytes(storage, centroid_bytes)
    );
    let (uri, content_hash) = blob?;
    let (centroids_uri, centroids_hash) = centroids?;
    Ok(PublishedState {
        uri,
        content_hash,
        centroids: RoutingRef {
            uri: centroids_uri,
            content_hash: centroids_hash,
        },
    })
}

/// Fetch the blob at `uri`, verify its bytes hash to `expected`, and decode.
/// Callers fall back to manifest-part loading on any error — a bad blob must
/// never fail a table open or a query.
pub(crate) async fn load_state(
    storage: &dyn StorageProvider,
    uri: &str,
    expected: &ContentHash,
) -> Result<Vec<Arc<SuperfileEntry>>, SlowVectorStateError> {
    Ok(load_full_state(storage, uri, expected).await?.entries)
}

pub(crate) async fn load_full_state(
    storage: &dyn StorageProvider,
    uri: &str,
    expected: &ContentHash,
) -> Result<SlowVectorState, SlowVectorStateError> {
    let bytes = fetch_blob_striped(storage, uri, STRIPED_FETCH_CHUNK_BYTES).await?;
    let expected = *expected;
    // blake3 over the whole blob plus the Avro parse is a CPU wave
    // (multi-GiB at 100M docs); run it on the blocking pool so the
    // runtime keeps driving I/O instead of stalling behind the decode.
    match spawn_blocking(move || {
        if ContentHash::of(bytes.as_ref()) != expected {
            return Err(SlowVectorStateError::HashMismatch);
        }
        decode_state(bytes.as_ref())
    })
    .await
    {
        Ok(result) => result,
        Err(join_error) => Err(SlowVectorStateError::Parse(format!(
            "slow-state decode task failed: {join_error}"
        ))),
    }
}

/// Fetch one object as bytes, striping objects larger than `chunk_bytes`
/// across parallel range-GETs ([`STRIPED_FETCH_MAX_IN_FLIGHT`] in flight).
/// Objects at or under one chunk keep the single-request `get` — requests
/// are the priced dimension, so small blobs must not fan out.
async fn fetch_blob_striped(
    storage: &dyn StorageProvider,
    uri: &str,
    chunk_bytes: u64,
) -> Result<Bytes, SlowVectorStateError> {
    let store_err = |e: StorageError| SlowVectorStateError::Storage(e.to_string());
    let meta = storage.head(uri).await.map_err(store_err)?;
    if meta.size <= chunk_bytes {
        let (bytes, _) = storage.get(uri).await.map_err(store_err)?;
        return Ok(bytes);
    }
    let ranges = striped_ranges(meta.size, chunk_bytes);
    // `buffered` preserves range order, so the concatenation below
    // reassembles the object byte-exactly; the content hash check in
    // the caller is the end-to-end integrity gate.
    let chunks: Vec<Bytes> = stream::iter(
        ranges
            .into_iter()
            .map(|range| async move { storage.get_range(uri, range).await }),
    )
    .buffered(STRIPED_FETCH_MAX_IN_FLIGHT)
    .try_collect()
    .await
    .map_err(store_err)?;
    let mut out = Vec::with_capacity(meta.size as usize);
    for chunk in &chunks {
        out.extend_from_slice(chunk);
    }
    Ok(Bytes::from(out))
}

/// Consecutive `chunk_bytes`-sized ranges covering `0..size` — the shared
/// stripe plan for [`fetch_blob_striped`] and [`fetch_centroid_section`].
fn striped_ranges(size: u64, chunk_bytes: u64) -> Vec<Range<u64>> {
    (0..size)
        .step_by(chunk_bytes.max(1) as usize)
        .map(|start| start..(start + chunk_bytes).min(size))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tempfile::tempdir;

    use super::*;
    use crate::{
        storage::LocalFsStorageProvider,
        superfile::vector::{layout::VectorLayout, quant::BitQuantizer, rotation::RandomRotation},
        supertable::manifest::{CellVectorSummary, ClusterCentroids, SuperfileUri, VectorSummary},
    };

    /// Doc count for the first fixture entry; arbitrary but distinct from
    /// `SECOND_N_DOCS` so field mix-ups fail loudly.
    const FIRST_N_DOCS: u64 = 42;
    /// Doc count for the second fixture entry.
    const SECOND_N_DOCS: u64 = 7;

    fn entry(n_docs: u64, cell: u32) -> Arc<SuperfileEntry> {
        let id = Uuid::new_v4();
        Arc::new(SuperfileEntry {
            birth_version: 3,
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs,
            id_min: 10,
            id_max: 10 + n_docs.saturating_sub(1) as i128,
            scalar_stats: HashMap::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: cell.to_le_bytes().to_vec(),
            partition_hint: Some(cell),
            vector_layout: VectorLayout::Ivf,
            subsection_offsets: None,
        })
    }

    /// Dim for the routing-sibling fixture's vector summary.
    const ROUTING_FIXTURE_DIM: usize = 16;
    /// Rot seed for the routing-sibling fixture's admit slab.
    const ROUTING_FIXTURE_ROT_SEED: u64 = 9;

    /// [`entry`] plus a one-cell vector summary whose admit slab is built —
    /// the shape a real write path produces.
    fn entry_with_summary(n_docs: u64, cell: u32) -> Arc<SuperfileEntry> {
        let mut e = Arc::try_unwrap(entry(n_docs, cell)).expect("fresh arc");
        let mut flat = vec![0.0f32; 2 * ROUTING_FIXTURE_DIM];
        flat[0] = 1.0;
        flat[ROUTING_FIXTURE_DIM + 3] = -1.0;
        let clusters =
            ClusterCentroids::from_fp32(2, ROUTING_FIXTURE_DIM as u32, &flat, vec![3, 4]);
        clusters.prewarm_admit_codes(
            &RandomRotation::new(ROUTING_FIXTURE_DIM, ROUTING_FIXTURE_ROT_SEED),
            &BitQuantizer::new(ROUTING_FIXTURE_DIM),
            ROUTING_FIXTURE_ROT_SEED,
        );
        e.vector_summary.insert(
            "emb".into(),
            VectorSummary {
                centroid: vec![0.5; ROUTING_FIXTURE_DIM],
                cells: vec![CellVectorSummary {
                    cell_id: Some(cell),
                    clusters,
                }],
            },
        );
        Arc::new(e)
    }

    fn assert_entries_match(a: &SuperfileEntry, b: &SuperfileEntry) {
        assert_eq!(a.superfile_id, b.superfile_id);
        assert_eq!(a.uri, b.uri);
        assert_eq!(a.n_docs, b.n_docs);
        assert_eq!(a.id_min, b.id_min);
        assert_eq!(a.id_max, b.id_max);
        assert_eq!(a.partition_key, b.partition_key);
        assert_eq!(a.partition_hint, b.partition_hint);
        assert_eq!(a.birth_version, b.birth_version);
    }

    #[test]
    fn entries_roundtrip_and_deterministic() {
        let entries = vec![entry(FIRST_N_DOCS, 0), entry(SECOND_N_DOCS, 5)];
        let bytes = encode_entries(&entries);
        let decoded = decode_entries(&bytes).expect("decode");
        assert_eq!(decoded.len(), entries.len());
        for (d, e) in decoded.iter().zip(entries.iter()) {
            assert_entries_match(d, e);
        }
        // Same logical entries ⇒ same bytes ⇒ same content hash ⇒ same URI —
        // the property the content-addressed republish-is-a-no-op rides on.
        let again = encode_entries(&entries);
        assert_eq!(bytes, again);
        assert_eq!(
            storage_path(&ContentHash::of(&bytes)),
            storage_path(&ContentHash::of(&again))
        );
    }

    #[test]
    fn decode_garbage_is_parse_error() {
        let err = decode_entries(&[0u8; 16]).expect_err("garbage");
        assert!(matches!(err, SlowVectorStateError::Parse(_)), "{err:?}");
    }

    /// The centroid section round-trips through publish + fetch: the walk
    /// length matches the encoded bytes, `write_state` publishes the
    /// sibling, and `fetch_centroid_section` serves each cell's fp32
    /// exactly as the summary carried it.
    #[tokio::test]
    async fn centroid_section_roundtrip_serves_cell_fp32() {
        let entries = vec![
            entry_with_summary(FIRST_N_DOCS, 1),
            entry_with_summary(SECOND_N_DOCS, 6),
        ];
        let section = compose_centroid_section(&entries, None).expect("compose from resident");
        assert_eq!(section.len(), section_len(&entries));
        assert!(!section.is_empty(), "fixture summaries carry fp32");

        let dir = tempdir().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("storage");
        let published = write_state(&storage, &entries, None)
            .await
            .expect("publish");
        assert_eq!(published.centroids.content_hash, ContentHash::of(&section));

        let fetched = fetch_centroid_section(&storage, &published.centroids, &entries)
            .await
            .expect("fetch section");
        for entry in &entries {
            let cell = &entry.vector_summary["emb"].cells[0];
            let got = fetched
                .read_cell(entry.superfile_id, "emb", cell.cell_id)
                .expect("spill read")
                .expect("cell served");
            assert_eq!(got, cell.clusters.centroids, "fp32 must round-trip");
        }
        // Unknown cells miss cleanly (caller falls back).
        assert!(
            fetched
                .read_cell(entries[0].superfile_id, "emb", Some(999))
                .expect("spill read")
                .is_none()
        );
    }

    #[test]
    fn pending_drain_envelope_round_trips_visible_and_pending_entries() {
        let visible = vec![entry(FIRST_N_DOCS, 1)];
        let pending_entries = vec![entry(SECOND_N_DOCS, 2)];
        let pending = PendingDrainState {
            metadata: b"checkpoint metadata".to_vec(),
            entries: pending_entries.clone(),
        };
        let bytes = encode_checkpoint_state(&visible, &pending);
        let decoded = decode_state(&bytes).expect("decode checkpoint envelope");
        assert_eq!(decoded.entries.len(), 1);
        assert_entries_match(&decoded.entries[0], &visible[0]);
        let decoded_pending = decoded.pending_drain.expect("pending drain");
        assert_eq!(decoded_pending.metadata, pending.metadata);
        assert_eq!(decoded_pending.entries.len(), 1);
        assert_entries_match(&decoded_pending.entries[0], &pending_entries[0]);
    }

    /// Chunk size that forces the striped path on a tiny fixture blob.
    const TINY_STRIPE_CHUNK_BYTES: u64 = 64;

    #[tokio::test]
    async fn striped_fetch_reassembles_byte_exact() {
        let dir = tempdir().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("localfs");
        let entries = vec![entry(FIRST_N_DOCS, 0), entry(SECOND_N_DOCS, 1)];
        let uri = write_state(&storage, &entries, None)
            .await
            .expect("write")
            .uri;
        let whole = storage.get(&uri).await.expect("whole get").0;
        assert!(
            whole.len() as u64 > TINY_STRIPE_CHUNK_BYTES,
            "fixture must exceed one stripe chunk"
        );
        let striped = fetch_blob_striped(&storage, &uri, TINY_STRIPE_CHUNK_BYTES)
            .await
            .expect("striped fetch");
        assert_eq!(striped, whole, "striped reassembly must be byte-exact");
    }

    #[tokio::test]
    async fn write_load_verifies_hash_and_is_idempotent() {
        let dir = tempdir().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");
        let entries = vec![entry(FIRST_N_DOCS, 1)];

        let published = write_state(&storage, &entries, None).await.expect("write");
        let (uri, hash) = (published.uri, published.content_hash);
        // Re-publishing identical content must succeed (PreconditionFailed
        // from the hash-derived URI is benign by construction).
        let republished = write_state(&storage, &entries, None)
            .await
            .expect("rewrite");
        assert_eq!(uri, republished.uri);
        assert_eq!(hash, republished.content_hash);
        assert_eq!(published.centroids, republished.centroids);

        let loaded = load_state(&storage, &uri, &hash).await.expect("load");
        assert_eq!(loaded.len(), 1);
        assert_entries_match(&loaded[0], &entries[0]);

        let wrong = ContentHash::of(b"wrong");
        let err = load_state(&storage, &uri, &wrong)
            .await
            .expect_err("hash mismatch");
        assert!(matches!(err, SlowVectorStateError::HashMismatch), "{err:?}");

        let missing = load_state(&storage, "slow-vector-state/absent.bin", &hash)
            .await
            .expect_err("missing object");
        assert!(
            matches!(missing, SlowVectorStateError::Storage(_)),
            "{missing:?}"
        );
    }

    /// The state blob is routing-shaped: entries decode straight into the
    /// stripped shape with the write-time admit slab intact, and fp32
    /// lives only in the section object next to it.
    #[tokio::test]
    async fn state_blob_decodes_stripped_entries_with_slab() {
        let dir = tempdir().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");
        let entries = vec![entry_with_summary(FIRST_N_DOCS, 1)];
        let expected_slab = entries[0].vector_summary["emb"].cells[0]
            .clusters
            .admit_codes_built()
            .expect("write-time slab")
            .clone();

        let published = write_state(&storage, &entries, None).await.expect("write");
        assert_ne!(
            published.uri, published.centroids.uri,
            "state blob and centroid section are distinct objects"
        );
        let blob_len = storage.get(&published.uri).await.expect("blob get").0.len();
        let section_len_bytes = storage
            .get(&published.centroids.uri)
            .await
            .expect("section get")
            .0
            .len();
        assert_eq!(
            section_len_bytes,
            section_len(&entries),
            "section carries exactly the summaries' fp32 bytes"
        );
        assert!(blob_len > 0, "state blob must be non-empty");

        let loaded = load_state(&storage, &published.uri, &published.content_hash)
            .await
            .expect("load blob");
        assert_eq!(loaded.len(), 1);
        assert_entries_match(&loaded[0], &entries[0]);
        let clusters = &loaded[0].vector_summary["emb"].cells[0].clusters;
        assert!(
            !clusters.vectors_resident(),
            "state-blob entries land in the stripped shape"
        );
        assert_eq!(
            *clusters.admit_codes_built().expect("slab seeded"),
            expected_slab,
            "slab survives the routing wire form"
        );

        // Checkpoint publications keep the same shape — consumers opening
        // mid-drain fetch only the routing layer; PENDING entries keep
        // fp32 inline (writer-only crash-resume state).
        let pending = PendingDrainState {
            metadata: b"epoch".to_vec(),
            entries: vec![entry_with_summary(SECOND_N_DOCS, 2)],
        };
        let checkpoint = write_state_with_pending_drain(&storage, &entries, &pending, None)
            .await
            .expect("checkpoint write");
        let state = load_full_state(&storage, &checkpoint.uri, &checkpoint.content_hash)
            .await
            .expect("load checkpoint");
        assert_eq!(
            state.entries.len(),
            entries.len(),
            "checkpoint blob carries the visible entries"
        );
        assert!(
            !state.entries[0].vector_summary["emb"].cells[0]
                .clusters
                .vectors_resident(),
            "visible checkpoint entries are stripped"
        );
        let pending_loaded = state.pending_drain.expect("pending state rides the blob");
        assert!(
            pending_loaded.entries[0].vector_summary["emb"].cells[0]
                .clusters
                .vectors_resident(),
            "pending entries keep fp32 inline for drain resume"
        );
    }

    /// A republish whose carried-forward entries are STRIPPED (the shape
    /// hydration produces) composes the new section from the previous
    /// generation's section — byte-identical to composing from resident
    /// fp32 — and fails loudly when no source covers a cell.
    #[tokio::test]
    async fn compose_section_from_previous_generation() {
        let dir = tempdir().expect("tempdir");
        let storage = LocalFsStorageProvider::new(dir.path()).expect("provider");
        let resident = vec![
            entry_with_summary(FIRST_N_DOCS, 1),
            entry_with_summary(SECOND_N_DOCS, 6),
        ];
        let published = write_state(&storage, &resident, None)
            .await
            .expect("first publish");
        let expected_section = compose_centroid_section(&resident, None).expect("resident bytes");

        // Round-trip the entries through the routing wire — the stripped
        // shape a writer's hydrated manifest carries at republish time.
        let stripped = load_state(&storage, &published.uri, &published.content_hash)
            .await
            .expect("hydrate stripped");
        assert!(
            stripped
                .iter()
                .all(|e| !e.vector_summary["emb"].cells[0].clusters.vectors_resident()),
            "fixture must exercise the stripped path"
        );
        assert!(
            compose_centroid_section(&stripped, None).is_err(),
            "no fp32 source must fail loudly, never publish a hole"
        );

        let previous = fetch_centroid_section(&storage, &published.centroids, &resident)
            .await
            .expect("fetch previous section");
        let composed =
            compose_centroid_section(&stripped, Some(&previous)).expect("compose from previous");
        assert_eq!(
            composed, expected_section,
            "carried-forward cells must compose byte-identical fp32"
        );

        let republished = write_state(&storage, &stripped, Some(&previous))
            .await
            .expect("republish from stripped");
        assert_eq!(
            republished.centroids.content_hash,
            ContentHash::of(&expected_section),
            "republished section must address the same fp32 bytes"
        );
    }
}
