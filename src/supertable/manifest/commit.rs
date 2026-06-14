// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Atomic-rename pointer commit.
//!
//! The persistence primitives the writer sits on:
//!
//! - Directory layout under `<supertable_root>/`:
//!   - `_supertable/current` — the pointer file. The only
//!     file ever atomically renamed; visibility barrier for a
//!     commit.
//!   - `manifest-lists/list-NNNNNN.json` — immutable per
//!     manifest version. Conditional-create on PUT (S3
//!     `If-None-Match: *` / `O_EXCL` on LocalFS).
//!   - `manifests/part-<content-hash>.avro.zst` — immutable,
//!     content-addressed. Two writers that produce identical
//!     bytes target the same URI; the second's `put_atomic`
//!     surfaces `PreconditionFailed`, which is benign and
//!     swallowed by [`write_manifest_part`].
//!
//! - [`PointerFile`] in-memory shape + text wire format.
//!
//! - [`commit_manifest`] orchestrates the commit:
//!   1. Encode the new manifest list (JSON).
//!   2. Encode each new manifest part (Avro+zstd) →
//!      content-addressed URI.
//!   3. **In parallel** (`futures::future::join_all`): write
//!      the list, write each new part. None depend on each
//!      other — the list references parts by URI = blake3
//!      hash of bytes, computable before any I/O.
//!   4. Await all of the above (visibility barrier #1).
//!   5. Write the pointer file conditionally:
//!      `put_atomic` on first commit (no prev pointer);
//!      `put_if_match` against the prior pointer's etag on
//!      subsequent commits. This is the **single visibility
//!      barrier** that publishes the new manifest version.
//!
//! Why the parallel-issue shape: hierarchical manifest adds
//! files but should not add RTTs. List and parts are
//! independent of each other (content-addressing makes the
//! URI predictable before any PUT); a serial implementation
//! is correctness-equivalent but pessimistic on object stores.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use futures::future;

use crate::storage::{StorageError, StorageProvider};
use crate::supertable::error::CommitError;
use crate::supertable::manifest::list::{
    self as list_mod, ManifestList, ManifestListEntry, PartitionStrategy,
};
use crate::supertable::manifest::part::{
    self as part_mod, BLAKE3_DIGEST_BYTES, BLAKE3_HEX_LEN, ContentHash, ManifestPart, PartId,
};
use crate::supertable::manifest::partition::{assign_partition, encode_partition_key};
use crate::supertable::{Manifest, SuperfileEntry, SupertableOptions};

/// Pointer-file location under the supertable root. The only
/// path that ever gets atomically renamed; everything else is
/// content-addressed and immutable, so a torn write on those
/// paths is invisible (no committed pointer references it).
pub const POINTER_PATH: &str = "_supertable/current";

/// Subdirectory for manifest list files.
pub const MANIFEST_LISTS_DIR: &str = "manifest-lists";

/// Subdirectory for manifest part files.
pub const MANIFEST_PARTS_DIR: &str = "manifests";

/// Zstd compression level for manifest parts and the manifest list.
/// Level 3 is zstd's own default — a balanced ratio/speed point that
/// keeps commit latency low while compressing the Avro-encoded
/// manifest well. (Valid range is 1..=22.)
pub const MANIFEST_ZSTD_LEVEL: i32 = 3;

/// Build the URI for a manifest list at a given manifest_id.
/// 6-digit zero-pad gives stable lexicographic ordering for
/// `aws s3 ls`-style listings up through 999,999 versions.
pub fn list_uri(manifest_id: u64) -> String {
    format!("{MANIFEST_LISTS_DIR}/list-{manifest_id:06}.json")
}

/// Build the URI for a manifest part at a given content hash.
/// Content-addressed URI so two writers producing identical
/// bytes resolve to the same URI — the load-bearing property
/// for cross-version part reuse.
pub fn part_uri(content_hash: &ContentHash) -> String {
    format!(
        "{MANIFEST_PARTS_DIR}/part-{}.avro.zst",
        content_hash.to_hex()
    )
}

/// In-memory pointer file. Lives at [`POINTER_PATH`]; its
/// atomic rename is the visibility barrier for a commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointerFile {
    pub manifest_id: u64,
    pub manifest_list_uri: String,
    pub content_hash: ContentHash,
}

impl PointerFile {
    /// Serialize to the on-disk text format.
    ///
    /// ```text
    /// manifest_id=42
    /// manifest_list_uri=manifest-lists/list-000042.json
    /// content_hash=blake3:def...
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        format!(
            "manifest_id={}\nmanifest_list_uri={}\ncontent_hash=blake3:{}\n",
            self.manifest_id,
            self.manifest_list_uri,
            self.content_hash.to_hex(),
        )
        .into_bytes()
    }

    /// Parse the on-disk text format.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CommitError> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| CommitError::PointerParse(format!("not utf-8: {e}")))?;

        let mut manifest_id: Option<u64> = None;
        let mut manifest_list_uri: Option<String> = None;
        let mut content_hash: Option<ContentHash> = None;

        for line in s.lines() {
            if line.is_empty() {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| CommitError::PointerParse(format!("no '=' in line: {line:?}")))?;
            match key {
                "manifest_id" => {
                    manifest_id = Some(
                        value
                            .parse::<u64>()
                            .map_err(|e| CommitError::PointerParse(format!("manifest_id: {e}")))?,
                    );
                }
                "manifest_list_uri" => {
                    manifest_list_uri = Some(value.to_string());
                }
                "content_hash" => {
                    let hex = value.strip_prefix("blake3:").ok_or_else(|| {
                        CommitError::PointerParse(format!(
                            "content_hash missing 'blake3:' prefix: {value}"
                        ))
                    })?;
                    if hex.len() != BLAKE3_HEX_LEN {
                        return Err(CommitError::PointerParse(format!(
                            "content_hash hex must be {BLAKE3_HEX_LEN} chars; got {}",
                            hex.len()
                        )));
                    }
                    let mut bytes = [0u8; BLAKE3_DIGEST_BYTES];
                    for i in 0..BLAKE3_DIGEST_BYTES {
                        bytes[i] =
                            u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| {
                                CommitError::PointerParse(format!("content_hash hex: {hex}"))
                            })?;
                    }
                    content_hash = Some(ContentHash(bytes));
                }
                _ => {
                    // Unknown key — tolerate for forward compat (a
                    // future plan can add fields; old readers ignore).
                }
            }
        }

        Ok(Self {
            manifest_id: manifest_id
                .ok_or_else(|| CommitError::PointerParse("missing manifest_id".into()))?,
            manifest_list_uri: manifest_list_uri
                .ok_or_else(|| CommitError::PointerParse("missing manifest_list_uri".into()))?,
            content_hash: content_hash
                .ok_or_else(|| CommitError::PointerParse("missing content_hash".into()))?,
        })
    }
}

/// Read the pointer file from storage.
///
/// Returns `Ok(None)` if the pointer doesn't exist (fresh
/// supertable). Returns `Err` on any other failure.
pub async fn read_pointer(
    storage: &dyn StorageProvider,
) -> Result<Option<PointerFile>, CommitError> {
    match storage.get(POINTER_PATH).await {
        Ok((bytes, _)) => Ok(Some(PointerFile::from_bytes(&bytes)?)),
        Err(StorageError::NotFound { .. }) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Outcome of writing a manifest part — returned by
/// [`write_manifest_part`] so the caller can build the list
/// entry without re-computing.
#[derive(Debug, Clone)]
pub struct PartWriteResult {
    pub part_id: PartId,
    pub uri: String,
    pub content_hash: ContentHash,
    pub size_bytes_compressed: u64,
    pub size_bytes_uncompressed: u64,
}

/// Outcome of writing a manifest list.
#[derive(Debug, Clone)]
pub struct ListWriteResult {
    pub uri: String,
    pub content_hash: ContentHash,
    pub size_bytes: u64,
}

/// Encode + write one manifest part. Content-addressed:
/// `put_atomic` lands the bytes if the target doesn't exist;
/// if it already exists (another writer raced to the same
/// content), [`StorageError::PreconditionFailed`] is **swallowed**
/// — the bytes are bit-identical to what's already there, so
/// the commit can proceed.
pub async fn write_manifest_part(
    storage: &dyn StorageProvider,
    part: &ManifestPart,
    zstd_level: i32,
) -> Result<PartWriteResult, CommitError> {
    let compressed = part_mod::encode(part, zstd_level);
    let content_hash = ContentHash::of(&compressed);
    let uri = part_uri(&content_hash);
    let size_compressed = compressed.len() as u64;

    // Uncompressed size for the manifest list's size_bytes_uncompressed
    // field. Cheapest correct path is to decompress and measure;
    // the zstd frame header encodes the content length but extracting
    // it is more code than a full decompress is worth at part scale.
    let size_uncompressed = zstd::stream::decode_all(compressed.as_slice())
        .map_err(|e| CommitError::Encode(format!("zstd self-decode: {e}")))?
        .len() as u64;

    match storage
        .put_atomic(&uri, bytes::Bytes::from(compressed))
        .await
    {
        Ok(_) => {}
        // Content-addressed: same hash → same bytes. Already
        // there is benign — another writer wrote the same
        // content. Treat as success.
        Err(StorageError::PreconditionFailed { .. }) => {}
        Err(e) => return Err(e.into()),
    }

    Ok(PartWriteResult {
        part_id: part.part_id,
        uri,
        content_hash,
        size_bytes_compressed: size_compressed,
        size_bytes_uncompressed: size_uncompressed,
    })
}

/// Encode + write a manifest list. Conditional-create
/// (`put_atomic`) — exactly one writer succeeds in publishing
/// a given `manifest_id`'s list; concurrent attempts surface
/// `PreconditionFailed` and the caller's commit fails (the
/// writer's OCC retry loop catches this).
pub async fn write_manifest_list(
    storage: &dyn StorageProvider,
    list: &ManifestList,
) -> Result<ListWriteResult, CommitError> {
    let json = list_mod::encode(list).map_err(|e| CommitError::Encode(e.to_string()))?;
    let content_hash = ContentHash::of(&json);
    let uri = list_uri(list.manifest_id);
    let size = json.len() as u64;
    storage.put_atomic(&uri, bytes::Bytes::from(json)).await?;
    Ok(ListWriteResult {
        uri,
        content_hash,
        size_bytes: size,
    })
}

/// Write the pointer file.
///
/// - `expected_prev_etag = None` ⇒ create-only (initial commit
///   on a fresh supertable). Uses `put_atomic`.
/// - `expected_prev_etag = Some(...)` ⇒ CAS-fenced update.
///   Uses `put_if_match`.
///
/// On `PreconditionFailed`, surfaces
/// `CommitError::WriteContentionExhausted` so callers can map
/// it to the OCC retry loop or to a "first commit lost a
/// race" message.
pub async fn write_pointer(
    storage: &dyn StorageProvider,
    pointer: &PointerFile,
    expected_prev_etag: Option<&str>,
) -> Result<(), CommitError> {
    let bytes = bytes::Bytes::from(pointer.to_bytes());
    let result = match expected_prev_etag {
        None => storage.put_atomic(POINTER_PATH, bytes).await,
        Some(_) => {
            storage
                .put_if_match(POINTER_PATH, bytes, expected_prev_etag)
                .await
        }
    };
    match result {
        Ok(_) => Ok(()),
        Err(StorageError::PreconditionFailed { .. }) => Err(CommitError::WriteContentionExhausted),
        Err(e) => Err(e.into()),
    }
}

/// Commit a new manifest version.
///
/// Orchestrates the four-step sequence:
///
/// 1. **In parallel** — write each new manifest part + write
///    the new manifest list. Independent of each other; the
///    list references parts by URI (= blake3 of bytes,
///    computed before any I/O). Issued via
///    [`futures::future::join_all`].
/// 2. Await all of the above (visibility barrier #1: parts
///    and list must be durable before the pointer publishes).
/// 3. Build the new pointer file (manifest_id, list_uri,
///    list_content_hash).
/// 4. Conditional pointer-PUT (visibility barrier #2: the
///    rename is the only thing readers observe).
///
/// `parts_to_write` should contain **only the parts that need
/// to be persisted** (i.e., new + changed). Reused parts from
/// the previous manifest version are not in this list — their
/// URIs are already in `new_list.parts[i].uri`. This is the
/// "part reuse" optimization: commits that touch zero
/// partitions write zero new part files.
pub async fn commit_manifest(
    storage: &dyn StorageProvider,
    expected_prev_etag: Option<&str>,
    new_list: &ManifestList,
    parts_to_write: &[&ManifestPart],
    zstd_level: i32,
) -> Result<PointerFile, CommitError> {
    // Step 1+2: parallel write of (list, parts).
    //
    // Both futures are independent — the list's references to
    // each part's URI are content-addressable from the
    // in-memory bytes before any I/O, so there's no
    // happens-before edge between them.
    let list_fut = write_manifest_list(storage, new_list);
    let part_futs = parts_to_write
        .iter()
        .map(|p| write_manifest_part(storage, p, zstd_level));
    let part_join = future::join_all(part_futs);

    let (list_res, part_results) = tokio::join!(list_fut, part_join);
    // Translate `Storage(PreconditionFailed)` from sub-writes
    // into `WriteContentionExhausted` so callers (and the
    // writer's OCC retry loop) can match on one variant
    // regardless of which CAS lost the race — list or pointer.
    let list_res = list_res.map_err(translate_contention)?;
    for part_result in part_results {
        let _ = part_result.map_err(translate_contention)?;
    }

    // Step 3: build pointer.
    let pointer = PointerFile {
        manifest_id: new_list.manifest_id,
        manifest_list_uri: list_res.uri,
        content_hash: list_res.content_hash,
    };

    // Step 4: conditional pointer write — the visibility
    // barrier. Until this succeeds, no reader sees the new
    // manifest version.
    write_pointer(storage, &pointer, expected_prev_etag).await?;
    Ok(pointer)
}

/// Test-helper alias so test code can construct a
/// `Arc<dyn StorageProvider>` and pass it through this
/// module's `&dyn StorageProvider`-typed APIs in one cast.
#[doc(hidden)]
pub fn as_dyn(p: &Arc<dyn StorageProvider>) -> &dyn StorageProvider {
    p.as_ref()
}

/// `PreconditionFailed` from a sub-write (manifest list or
/// manifest part) is semantically the same as the pointer-CAS
/// losing the race — both mean another writer beat us to the
/// same manifest_id. Caller maps to OCC retry or to a
/// terminal "write contention" error to the user. Other
/// errors pass through unchanged.
fn translate_contention(e: CommitError) -> CommitError {
    match e {
        CommitError::Storage(StorageError::PreconditionFailed { .. }) => {
            CommitError::WriteContentionExhausted
        }
        other => other,
    }
}

/// Verifies the current in-memory manifest is the latest one and returns
/// the current manifest etag for the given manifest, or `None` if the
/// manifest is not yet committed.
pub async fn get_current_manifest_etag(
    storage: &Arc<dyn StorageProvider>,
    current: Arc<Manifest>,
) -> Result<Option<String>, CommitError> {
    let (bytes, meta) = match storage.get(POINTER_PATH).await {
        Ok((bytes, meta)) => (bytes, meta),
        Err(StorageError::NotFound { .. }) => return Ok(None),
        Err(e) => return Err(crate::supertable::CommitError::from(e)),
    };

    let pointer_file = PointerFile::from_bytes(&bytes)?;
    let Some(meta_list) = current.list.as_ref() else {
        // no manifest list for in-memory supertables
        return Ok(None);
    };
    if pointer_file.manifest_id == meta_list.manifest_id {
        return Ok(meta.etag);
    }
    Err(CommitError::WriteContentionExhausted)
}

/// build one `ManifestPart` from `superfiles` + the
/// matching `ManifestListEntry`. Encodes the part once,
/// content-hashes it, and computes the list-level aggregate
/// skip summaries that `list_prune` reads at query time.
fn build_part_and_entry(
    opts: &SupertableOptions,
    superfiles: Vec<Arc<SuperfileEntry>>,
    partition_key: Vec<u8>,
) -> Result<
    (
        crate::supertable::manifest::list::ManifestListEntry,
        crate::supertable::manifest::part::ManifestPart,
    ),
    crate::supertable::CommitError,
> {
    let _ = opts; // reserved for future per-options encoding tweaks (zstd level, etc.)

    let part = ManifestPart {
        format_version: part_mod::FORMAT_VERSION.into(),
        part_id: PartId::new_v4(),
        superfiles,
    };
    let compressed = part_mod::encode(&part, MANIFEST_ZSTD_LEVEL);
    let size_compressed = compressed.len() as u64;
    let content_hash = ContentHash::of(&compressed);
    let size_uncompressed = zstd::stream::decode_all(compressed.as_slice())
        .map(|v| v.len() as u64)
        .unwrap_or(size_compressed);
    let aggregates = crate::supertable::manifest::aggregates::compute(&part.superfiles);
    let entry = ManifestListEntry {
        part_id: part.part_id,
        uri: part_uri(&content_hash),
        n_superfiles: part.superfiles.len() as u64,
        size_bytes_compressed: size_compressed,
        size_bytes_uncompressed: size_uncompressed,
        content_hash,
        partition_key,
        id_range: aggregates.id_range,
        scalar_stats_agg: aggregates.scalar_stats_agg,
        fts_summary_agg: aggregates.fts_summary_agg,
        vector_summary_agg: aggregates.vector_summary_agg,
    };
    Ok((entry, part))
}

/// Returns the new ManifestListEntries when `new_entries` are added to `old` manifest. This
/// operation may create new ManifestParts. The function also returns the new ManifestParts that
/// the caller can decide to write to storage.
pub async fn rebalance_for_commit(
    opts: &Arc<SupertableOptions>,
    old: &Arc<crate::supertable::Manifest>,
    new_entries: &[Arc<SuperfileEntry>],
    entries_to_remove: &[Arc<SuperfileEntry>],
) -> Result<(Vec<ManifestListEntry>, Vec<ManifestPart>), CommitError> {
    // 1. Resolve the effective partition strategy. Locked at
    //    first commit: read from the existing manifest list
    //    if present, else use the options default.
    let strategy: PartitionStrategy = old
        .list
        .as_ref()
        .map(|l| l.partition_strategy.clone())
        .unwrap_or_else(|| opts.effective_partition_strategy());

    // 2. Group new entries by partition_key (the on-disk
    //    encoding the list + parts carry).
    let mut new_by_partition: BTreeMap<Vec<u8>, Vec<Arc<SuperfileEntry>>> = BTreeMap::new();
    for entry in new_entries {
        let pk = assign_partition(entry, &strategy)?;
        new_by_partition
            .entry(encode_partition_key(&pk))
            .or_default()
            .push(Arc::clone(entry));
    }

    let mut removals_by_partition: BTreeMap<Vec<u8>, Vec<Arc<SuperfileEntry>>> = BTreeMap::new();
    for entry in entries_to_remove {
        let pk = assign_partition(entry, &strategy)?;
        removals_by_partition
            .entry(encode_partition_key(&pk))
            .or_default()
            .push(Arc::clone(entry));
    }

    // 3. Walk the existing list entries, classify each by
    //    whether it's the *latest* entry for its partition.
    //    The "rewrite latest part" policy: only the
    //    most recent entry per partition is a candidate for
    //    rewrite; older entries for the same partition (from
    //    a prior part-split) carry over unchanged.
    let mut latest_index_for_partition: HashMap<Vec<u8>, usize> = HashMap::new();
    if let Some(old_list) = old.list.as_ref() {
        for (i, entry) in old_list.parts.iter().enumerate() {
            latest_index_for_partition.insert(entry.partition_key.clone(), i);
        }
    }
    // The output list entries — built incrementally as we
    // walk existing entries + emit new ones for cold
    // partitions. Order: existing entries (touched ones
    // replaced in place; untouched preserved) followed by
    // entries for cold partitions.
    let mut out_list_entries: Vec<ManifestListEntry> = Vec::new();
    let mut parts_to_write: Vec<ManifestPart> = Vec::new();
    let mut handled_partitions: HashSet<Vec<u8>> = HashSet::new();

    if let Some(old_list) = old.list.as_ref() {
        for (i, entry) in old_list.parts.iter().enumerate() {
            let is_latest_for_partition = latest_index_for_partition
                .get(&entry.partition_key)
                .copied()
                == Some(i);
            let touched = new_by_partition.contains_key(&entry.partition_key);

            if is_latest_for_partition && touched {
                // Rewrite path: rebuild this part with its
                // existing superfiles + the new ones for this
                // partition.
                let new_for_pk = new_by_partition
                    .remove(&entry.partition_key)
                    .expect("touched implies present");
                let existing_part = old.part(entry.part_id).await.map_err(|e| {
                    crate::supertable::CommitError::PointerParse(format!(
                        "loading existing part {} for partition rewrite: {e}",
                        entry.part_id.0
                    ))
                })?;
                let combined_n = existing_part.superfiles.len() + new_for_pk.len();
                let combined_superfiles: Vec<Arc<SuperfileEntry>> = existing_part
                    .superfiles
                    .iter()
                    .cloned()
                    .chain(new_for_pk)
                    .collect();

                if combined_n as u64 > opts.target_superfiles_per_partition {
                    // Split: keep the existing entry as-is
                    // (older split entry from now on) and
                    // emit a fresh part with just the new
                    // superfiles for this partition.
                    out_list_entries.push(entry.clone());
                    let new_superfiles: Vec<Arc<SuperfileEntry>> =
                        combined_superfiles[existing_part.superfiles.len()..].to_vec();
                    let (fresh_entry, fresh_part) =
                        build_part_and_entry(opts, new_superfiles, entry.partition_key.clone())?;
                    out_list_entries.push(fresh_entry);
                    parts_to_write.push(fresh_part);
                } else {
                    // Rewrite: replace this entry with the
                    // combined-superfiles part.
                    let (rebuilt_entry, rebuilt_part) = build_part_and_entry(
                        opts,
                        combined_superfiles,
                        entry.partition_key.clone(),
                    )?;
                    out_list_entries.push(rebuilt_entry);
                    parts_to_write.push(rebuilt_part);
                }
                handled_partitions.insert(entry.partition_key.clone());
            } else {
                // Carry over: either an older entry for a
                // touched partition (handled when we hit the
                // latest), or an entry for an untouched
                // partition. Either way, content-hash + URI
                // unchanged — no re-encode, no PUT.
                out_list_entries.push(entry.clone());
            }
        }
    }

    // Cold partitions (touched but no prior entry): emit a
    // fresh part with just the new superfiles.
    for (pk, new_for_pk) in new_by_partition {
        if handled_partitions.contains(&pk) {
            continue;
        }
        let (fresh_entry, fresh_part) = build_part_and_entry(opts, new_for_pk, pk)?;
        out_list_entries.push(fresh_entry);
        parts_to_write.push(fresh_part);
    }

    // At this point, out_list_entries contains all new ManifestListEntries that will be written.
    // If these out_list_entries i.e Vec<ManifestListEntry> cause new ManifestParts to be created, those
    // are stored in parts_to_write.

    let mut out_list_entries_after_removal = Vec::new();
    for entry in out_list_entries {
        let Some(removals) = removals_by_partition.get(&entry.partition_key) else {
            // If this entry belongs to a partition which has no removals, we can keep it as-is.
            // This will also not need any change to parts_to_write.
            out_list_entries_after_removal.push(entry);
            continue;
        };

        let removal_ids = removals.iter().map(|r| r.superfile_id).collect::<Vec<_>>();
        // TODO: Handle merging 2 parts into one if their sum is within threshold

        // First we fetch the latest superfile entries - either from parts_to_write or the old manifest.
        let (final_superfile_entries, existing_part_to_update) = if let Some(existing_part) =
            parts_to_write
                .iter_mut()
                .find(|p| p.part_id == entry.part_id)
        {
            (
                existing_part
                    .superfiles
                    .iter()
                    .filter(|s| !removal_ids.contains(&s.superfile_id))
                    .cloned()
                    .collect::<Vec<_>>(),
                Some(existing_part),
            )
        } else if let Ok(existing_part) = old.part(entry.part_id).await {
            (
                existing_part
                    .superfiles
                    .iter()
                    .filter(|s| !removal_ids.contains(&s.superfile_id))
                    .cloned()
                    .collect::<Vec<_>>(),
                None,
            )
        } else {
            return Err(CommitError::PointerParse(
                "Failed to find existing part for removal".to_string(),
            ));
        };

        // Now we build the fresh part and entry based on the final superfile entries.
        let (fresh_entry, fresh_part) =
            build_part_and_entry(opts, final_superfile_entries, entry.partition_key)?;

        if let Some(existing_part) = existing_part_to_update {
            *existing_part = fresh_part;
        } else {
            parts_to_write.push(fresh_part);
        }

        out_list_entries_after_removal.push(fresh_entry);
    }

    Ok((out_list_entries_after_removal, parts_to_write))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::storage::LocalFsStorageProvider;
    use crate::supertable::{Manifest, SuperfileEntry, SuperfileUri};
    use arrow_schema::{DataType, Field};
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::OnceCell;

    // ---- URI helpers ---------------------------------------------------

    #[test]
    fn list_uri_zero_pads_to_six_digits() {
        // 6-digit zero-pad gives stable lexicographic ordering
        // for `aws s3 ls`-style listings up to 999,999 versions.
        assert_eq!(list_uri(0), "manifest-lists/list-000000.json");
        assert_eq!(list_uri(42), "manifest-lists/list-000042.json");
        assert_eq!(list_uri(123_456), "manifest-lists/list-123456.json");
    }

    #[test]
    fn list_uri_overflows_padding_for_large_ids_intentionally() {
        // Past 6 digits the format widens — no truncation, just
        // breaks lex ordering. Spec'd behaviour; locked in to
        // catch accidental width changes.
        assert_eq!(list_uri(1_000_000), "manifest-lists/list-1000000.json");
    }

    #[test]
    fn part_uri_uses_content_hash_hex() {
        // Content-addressed: two writers producing identical
        // bytes resolve to the same URI. Verified by computing
        // the same hash twice and confirming URI equality.
        let h = ContentHash::of(b"hello manifest part");
        let uri_a = part_uri(&h);
        let uri_b = part_uri(&ContentHash::of(b"hello manifest part"));
        assert_eq!(uri_a, uri_b);
        assert!(uri_a.starts_with("manifests/part-"));
        assert!(uri_a.ends_with(".avro.zst"));
        assert_eq!(uri_a, format!("manifests/part-{}.avro.zst", h.to_hex()));
    }

    // ---- PointerFile round-trip ----------------------------------------

    fn sample_pointer() -> PointerFile {
        let mut bytes = [0u8; 32];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        PointerFile {
            manifest_id: 7,
            manifest_list_uri: "manifest-lists/list-000007.json".into(),
            content_hash: ContentHash(bytes),
        }
    }

    #[test]
    fn pointer_file_text_roundtrip() {
        // to_bytes ↔ from_bytes is the on-disk wire format —
        // any change to either side that drops a field or
        // changes line-ordering rules surfaces here.
        let p = sample_pointer();
        let bytes = p.to_bytes();
        let s = std::str::from_utf8(&bytes).expect("utf-8");
        assert!(s.contains("manifest_id=7"));
        assert!(s.contains("manifest_list_uri=manifest-lists/list-000007.json"));
        assert!(s.contains("content_hash=blake3:"));
        let parsed = PointerFile::from_bytes(&bytes).expect("parse");
        assert_eq!(parsed, p);
    }

    #[test]
    fn pointer_file_from_bytes_skips_blank_lines() {
        let bytes = b"\nmanifest_id=1\n\nmanifest_list_uri=foo.json\ncontent_hash=blake3:0000000000000000000000000000000000000000000000000000000000000000\n";
        let parsed = PointerFile::from_bytes(bytes).expect("parse");
        assert_eq!(parsed.manifest_id, 1);
        assert_eq!(parsed.manifest_list_uri, "foo.json");
        assert_eq!(parsed.content_hash.0, [0u8; 32]);
    }

    #[test]
    fn pointer_file_from_bytes_ignores_unknown_keys() {
        // Forward-compat: unknown keys must not error so that
        // an older reader can open a pointer that a future
        // writer extended.
        let bytes = b"manifest_id=2\nmanifest_list_uri=x.json\ncontent_hash=blake3:1111111111111111111111111111111111111111111111111111111111111111\nfuture_field=ignored\n";
        let parsed = PointerFile::from_bytes(bytes).expect("parse");
        assert_eq!(parsed.manifest_id, 2);
    }

    // ---- PointerFile parse errors --------------------------------------

    fn assert_parse_err(bytes: &[u8], needle: &str) {
        let err = PointerFile::from_bytes(bytes).expect_err("must error");
        match err {
            CommitError::PointerParse(msg) => assert!(
                msg.contains(needle),
                "expected `{needle}` in error; got: {msg}"
            ),
            other => panic!("expected PointerParse; got {other:?}"),
        }
    }

    #[test]
    fn pointer_file_from_bytes_rejects_invalid_utf8() {
        // 0xff is invalid UTF-8 as a standalone byte. Catches
        // garbage in the pointer file (e.g. partial write) at
        // parse time instead of letting it propagate.
        let bytes = [0xff, 0xfe, 0xfd];
        assert_parse_err(&bytes, "not utf-8");
    }

    #[test]
    fn pointer_file_from_bytes_rejects_missing_equals() {
        // A non-blank line without `=` can't be a key/value
        // pair, so we surface the bad line text in the error.
        assert_parse_err(b"manifest_id 1\n", "no '='");
    }

    #[test]
    fn pointer_file_from_bytes_rejects_bad_manifest_id() {
        // `manifest_id` is a u64; non-numeric values must fail
        // with a clear parse error rather than silently rolling
        // forward.
        assert_parse_err(
            b"manifest_id=abc\nmanifest_list_uri=x\ncontent_hash=blake3:0000000000000000000000000000000000000000000000000000000000000000\n",
            "manifest_id",
        );
    }

    #[test]
    fn pointer_file_from_bytes_rejects_content_hash_without_prefix() {
        assert_parse_err(
            b"manifest_id=1\nmanifest_list_uri=x\ncontent_hash=cafebabe\n",
            "blake3:",
        );
    }

    #[test]
    fn pointer_file_from_bytes_rejects_short_hex() {
        // blake3 is always 32 bytes → 64 hex chars; anything
        // else is malformed.
        assert_parse_err(
            b"manifest_id=1\nmanifest_list_uri=x\ncontent_hash=blake3:dead\n",
            "64 chars",
        );
    }

    #[test]
    fn pointer_file_from_bytes_rejects_bad_hex_chars() {
        // 64 chars but containing a non-hex char → parse error
        // from u8::from_str_radix.
        let mut hex = String::from("blake3:");
        hex.push_str(&"z".repeat(64));
        let payload = format!("manifest_id=1\nmanifest_list_uri=x\ncontent_hash={hex}\n");
        assert_parse_err(payload.as_bytes(), "content_hash hex");
    }

    #[test]
    fn pointer_file_from_bytes_rejects_missing_manifest_id() {
        assert_parse_err(
            b"manifest_list_uri=x\ncontent_hash=blake3:0000000000000000000000000000000000000000000000000000000000000000\n",
            "missing manifest_id",
        );
    }

    #[test]
    fn pointer_file_from_bytes_rejects_missing_list_uri() {
        assert_parse_err(
            b"manifest_id=1\ncontent_hash=blake3:0000000000000000000000000000000000000000000000000000000000000000\n",
            "missing manifest_list_uri",
        );
    }

    #[test]
    fn pointer_file_from_bytes_rejects_missing_content_hash() {
        assert_parse_err(
            b"manifest_id=1\nmanifest_list_uri=x\n",
            "missing content_hash",
        );
    }

    // ---- translate_contention ------------------------------------------

    #[test]
    fn translate_contention_maps_precondition_failed() {
        let in_err = CommitError::Storage(StorageError::PreconditionFailed { uri: "x".into() });
        match translate_contention(in_err) {
            CommitError::WriteContentionExhausted => {}
            other => panic!("expected WriteContentionExhausted; got {other:?}"),
        }
    }

    #[test]
    fn translate_contention_passes_through_other_storage_errors() {
        // Anything other than PreconditionFailed must pass
        // through unchanged — those are real errors the caller
        // mustn't mask as "lost a race".
        let in_err = CommitError::Encode("downstream zstd".into());
        match translate_contention(in_err) {
            CommitError::Encode(_) => {}
            other => panic!("expected Encode passthrough; got {other:?}"),
        }
    }

    // ---- read_pointer / write_pointer / write_manifest_list -------------
    //
    // Drive the storage-touching helpers through LocalFs so the
    // success + storage-not-found + CAS-failure branches all
    // get coverage without spinning up the s3s test harness.

    fn local_storage() -> (TempDir, Arc<dyn StorageProvider>) {
        let dir = TempDir::new().expect("tempdir");
        let store: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
        (dir, store)
    }

    #[tokio::test]
    async fn read_pointer_returns_none_when_absent() {
        // Fresh supertable: no pointer file. read_pointer must
        // surface this as Ok(None), not Err.
        let (_dir, storage) = local_storage();
        let p = read_pointer(storage.as_ref()).await.expect("read");
        assert!(p.is_none());
    }

    #[tokio::test]
    async fn write_pointer_create_then_read_roundtrip() {
        // Initial commit shape: no expected_prev_etag, so
        // write_pointer routes through put_atomic and lands
        // the new pointer file.
        let (_dir, storage) = local_storage();
        let p = sample_pointer();
        write_pointer(storage.as_ref(), &p, None)
            .await
            .expect("write");
        let read = read_pointer(storage.as_ref())
            .await
            .expect("read")
            .expect("present");
        assert_eq!(read, p);
    }

    #[tokio::test]
    async fn write_pointer_second_create_surfaces_contention() {
        // put_atomic against an existing path is the on-disk
        // contention case for the first-commit branch: the
        // CAS fence already lost a race. The function must
        // translate the storage's PreconditionFailed into
        // WriteContentionExhausted so the OCC retry loop can
        // recognise it.
        let (_dir, storage) = local_storage();
        let p = sample_pointer();
        write_pointer(storage.as_ref(), &p, None)
            .await
            .expect("first");
        let err = write_pointer(storage.as_ref(), &p, None)
            .await
            .expect_err("second must lose");
        assert!(
            matches!(err, CommitError::WriteContentionExhausted),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn write_manifest_list_succeeds_and_addresses_uri() {
        // write_manifest_list encodes JSON, computes a hash,
        // and PUTs at list_uri(manifest_id). Verify the
        // returned URI matches the deterministic naming rule
        // and the bytes are reachable through `get`.
        use crate::supertable::manifest::list::{
            FORMAT_VERSION as LIST_FORMAT_VERSION, ManifestList, PartitionStrategy,
        };
        let (_dir, storage) = local_storage();
        // Smallest valid ManifestList shape — no parts, no
        // columns, an empty schema. Encoding only requires the
        // format header + the empty collections.
        let list = ManifestList {
            format_version: LIST_FORMAT_VERSION.into(),
            manifest_id: 1,
            options_hash: ContentHash([0u8; 32]),
            schema: Vec::new(),
            id_column: "_id".into(),
            fts_columns: Vec::new(),
            vector_columns: Vec::new(),
            partition_strategy: PartitionStrategy::TimeRange {
                column: "_id".into(),
                granularity_secs: 86_400,
            },
            parts: Vec::new(),
        };
        let res = write_manifest_list(storage.as_ref(), &list)
            .await
            .expect("write");
        assert_eq!(res.uri, list_uri(1));
        assert!(res.size_bytes > 0);
        // Read back to confirm bytes land at the URI.
        let _ = storage.get(&res.uri).await.expect("get list back");
    }

    #[test]
    fn point_constants_match_layout_doc() {
        // Smoke that the directory-layout constants haven't
        // drifted from the module docs. A rename of any of
        // these is observable through the on-disk shape and
        // would silently invalidate existing supertables on
        // upgrade — surfaces it as a test failure first.
        assert_eq!(POINTER_PATH, "_supertable/current");
        assert_eq!(MANIFEST_LISTS_DIR, "manifest-lists");
        assert_eq!(MANIFEST_PARTS_DIR, "manifests");
    }

    // ---- rebalance_for_commit ------------------------------------------

    fn make_superfile_entry(docs: u64, pk: Vec<u8>) -> Arc<SuperfileEntry> {
        Arc::new(SuperfileEntry {
            superfile_id: uuid::Uuid::new_v4(),
            uri: SuperfileUri::new_v4(),
            n_docs: docs,
            id_min: 0,
            id_max: docs as i128 - 1,
            scalar_stats: Default::default(),
            fts_summary: Default::default(),
            vector_summary: Default::default(),
            partition_key: pk,
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    fn hash_bucket_0_pk() -> Vec<u8> {
        // Hash partition with n_buckets=1 encodes to [0, 0, 0, 0] in little-endian
        vec![0, 0, 0, 0]
    }

    fn simple_schema() -> std::sync::Arc<arrow_schema::Schema> {
        std::sync::Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "text",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn make_opts() -> std::sync::Arc<SupertableOptions> {
        SupertableOptions::new(simple_schema(), vec![], vec![], None)
            .map(Arc::new)
            .expect("valid options")
    }

    fn empty_manifest(opts: &Arc<SupertableOptions>) -> Arc<Manifest> {
        Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList::empty(opts.clone()),
            list: Some(ManifestList {
                format_version: list_mod::FORMAT_VERSION.into(),
                manifest_id: 0,
                options_hash: ContentHash([0u8; 32]),
                schema: vec![],
                id_column: "_id".into(),
                fts_columns: vec![],
                vector_columns: vec![],
                partition_strategy: PartitionStrategy::Hash {
                    column: "_id".into(),
                    n_buckets: 1,
                },
                parts: vec![],
            }),
            parts: dashmap::DashMap::new(),
            loader: None,
        })
    }

    #[tokio::test]
    async fn rebalance_fresh_start_cold_partition_should_create_entry() {
        let opts = make_opts();
        let old_manifest = empty_manifest(&opts);
        let pk = hash_bucket_0_pk();

        let new_entry = make_superfile_entry(100, pk.clone());
        let new_entries = vec![new_entry];

        let (list_entries, parts) = rebalance_for_commit(&opts, &old_manifest, &new_entries, &[])
            .await
            .expect("rebalance");

        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].partition_key, pk);
        assert_eq!(list_entries[0].n_superfiles, 1);
        assert_eq!(parts[0].superfiles.len(), 1);
        assert_eq!(parts[0].superfiles[0].n_docs, 100);
    }

    #[tokio::test]
    async fn rebalance_fresh_start_multiple_cold_partitions_should_create_entries() {
        // With Hash strategy (n_buckets=1), all entries map to the same partition.
        let opts = make_opts();
        let old_manifest = empty_manifest(&opts);
        let pk = hash_bucket_0_pk();

        let entry1 = make_superfile_entry(100, pk.clone());
        let entry2 = make_superfile_entry(200, pk.clone());
        let new_entries = vec![entry1, entry2];

        let (list_entries, parts) = rebalance_for_commit(&opts, &old_manifest, &new_entries, &[])
            .await
            .expect("rebalance");

        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].partition_key, pk);
        assert_eq!(list_entries[0].n_superfiles, 2);
        assert_eq!(parts[0].superfiles.len(), 2);
        let total_docs: u64 = parts[0].superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 300);
    }

    #[tokio::test]
    async fn rebalance_add_to_existing_partition_rewrites_part() {
        // Adding a new entry to an existing single-part partition rewrites that part.
        let opts = make_opts();
        let pk_untouched = hash_bucket_0_pk();

        let (_dir, storage) = local_storage();

        let old_superfile = make_superfile_entry(100, pk_untouched.clone());
        let existing_part = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![old_superfile.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestListEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 1,
                partition_key: pk_untouched.clone(),
                id_range: (0, 99),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);

        let parts = dashmap::DashMap::new();
        parts.insert(
            pw.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![old_superfile],
            },
            list: Some(list),
            parts,
            loader: Some(Arc::new(loader)),
        });

        // Add new entry to the SAME partition (not a new/cold partition)
        let new_entry = make_superfile_entry(50, pk_untouched.clone());
        let new_entries = vec![new_entry];

        let (list_entries, parts) = rebalance_for_commit(&opts, &old_manifest, &new_entries, &[])
            .await
            .expect("rebalance");

        // Should have 1 list entry (rewritten old one)
        assert_eq!(list_entries.len(), 1);
        // Should have 1 new part (the rewritten one)
        assert_eq!(parts.len(), 1);

        // Entry should be for the same partition
        assert_eq!(list_entries[0].partition_key, pk_untouched);
        assert_eq!(list_entries[0].n_superfiles, 2);

        // Part should have combined superfiles
        assert_eq!(parts[0].superfiles.len(), 2);
        let total_docs: u64 = parts[0].superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 150);
    }

    #[tokio::test]
    async fn rebalance_rewrite_partition_within_target() {
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_partition = 3;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf1 = make_superfile_entry(100, pk.clone());
        let sf2 = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf1.clone(), sf2.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestListEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                partition_key: pk.clone(),
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);

        let parts = dashmap::DashMap::new();
        parts.insert(
            pw.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf1, sf2],
            },
            list: Some(list),
            parts,
            loader: Some(Arc::new(loader)),
        });

        // Add 1 new superfile to same partition (2 + 1 = 3, within target)
        let new_entry = make_superfile_entry(75, pk.clone());
        let new_entries = vec![new_entry];

        let (list_entries, parts) = rebalance_for_commit(&opts, &old_manifest, &new_entries, &[])
            .await
            .expect("rebalance");

        // Rewrite case: 1 list entry (old entry replaced), 1 new part
        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);

        // Entry should be for same partition
        assert_eq!(list_entries[0].partition_key, pk);
        assert_eq!(list_entries[0].n_superfiles, 3);

        // Part should have all 3 superfiles combined
        let part = &parts[0];
        assert_eq!(part.superfiles.len(), 3);
        // Verify combined doc count
        let total_docs: u64 = part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 325); // 100 + 150 + 75
    }

    #[tokio::test]
    async fn rebalance_split_partition_exceeds_target() {
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_partition = 2;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf1 = make_superfile_entry(100, pk.clone());
        let sf2 = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf1.clone(), sf2.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestListEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                partition_key: pk.clone(),
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);

        let parts = dashmap::DashMap::new();
        parts.insert(
            pw.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf1, sf2],
            },
            list: Some(list),
            parts,
            loader: Some(Arc::new(loader)),
        });

        // Add 2 new superfiles to same partition (2 + 2 = 4, exceeds target of 2)
        let new_entry1 = make_superfile_entry(75, pk.clone());
        let new_entry2 = make_superfile_entry(80, pk.clone());
        let new_entries = vec![new_entry1, new_entry2];

        let (list_entries, parts) = rebalance_for_commit(&opts, &old_manifest, &new_entries, &[])
            .await
            .expect("rebalance");

        // Split case: 2 list entries (old + fresh for split), 1 new part (fresh)
        assert_eq!(list_entries.len(), 2);
        assert_eq!(parts.len(), 1);

        // Both entries should be for same partition
        assert_eq!(list_entries[0].partition_key, pk);
        assert_eq!(list_entries[1].partition_key, pk);

        // First entry (old) should still have original superfiles
        assert_eq!(list_entries[0].n_superfiles, 2);

        // Second entry (fresh) should have the new superfiles
        assert_eq!(list_entries[1].n_superfiles, 2);

        // The one new part should have exactly the 2 new superfiles
        let part = &parts[0];
        assert_eq!(part.superfiles.len(), 2);
        let total_docs: u64 = part.superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 155); // 75 + 80
    }

    fn make_superfile_entry_hinted(docs: u64, pk: Vec<u8>, hint: u32) -> Arc<SuperfileEntry> {
        Arc::new(SuperfileEntry {
            superfile_id: uuid::Uuid::new_v4(),
            uri: SuperfileUri::new_v4(),
            n_docs: docs,
            id_min: 0,
            id_max: docs as i128 - 1,
            scalar_stats: Default::default(),
            fts_summary: Default::default(),
            vector_summary: Default::default(),
            partition_key: pk,
            partition_hint: Some(hint),
            subsection_offsets: None,
        })
    }

    fn hash2_pk(bucket: u32) -> Vec<u8> {
        bucket.to_le_bytes().to_vec()
    }

    #[tokio::test]
    async fn rebalance_older_entry_preserved_when_latest_rewritten() {
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_partition = 2;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf_old = make_superfile_entry(100, pk.clone());
        let sf_latest = make_superfile_entry(150, pk.clone());

        let part_old = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_old.clone()],
        };
        let pw_old = write_manifest_part(storage.as_ref(), &part_old, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_old");

        let part_latest = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_latest.clone()],
        };
        let pw_latest = write_manifest_part(storage.as_ref(), &part_latest, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_latest");

        // Old manifest with TWO entries for same partition (result of prior split)
        // Second one is the "latest" for that partition
        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![
                ManifestListEntry {
                    part_id: pw_old.part_id,
                    uri: pw_old.uri.clone(),
                    content_hash: pw_old.content_hash,
                    size_bytes_compressed: pw_old.size_bytes_compressed,
                    size_bytes_uncompressed: pw_old.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk.clone(),
                    id_range: (0, 99),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestListEntry {
                    part_id: pw_latest.part_id,
                    uri: pw_latest.uri,
                    content_hash: pw_latest.content_hash,
                    size_bytes_compressed: pw_latest.size_bytes_compressed,
                    size_bytes_uncompressed: pw_latest.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk.clone(),
                    id_range: (0, 149),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);

        let parts = dashmap::DashMap::new();
        parts.insert(
            part_latest.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_latest)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_old, sf_latest],
            },
            list: Some(list),
            parts,
            loader: Some(Arc::new(loader)),
        });

        // Add one new entry for the partition
        let new_entries = vec![make_superfile_entry(75, pk.clone())];

        let (list_entries, parts) = rebalance_for_commit(&opts, &old_manifest, &new_entries, &[])
            .await
            .expect("rebalance");

        // Expect: old entry (preserved) + latest entry (rewritten) = 2 list entries
        // Expect: 1 new part (latest rewrite)
        assert_eq!(list_entries.len(), 2);
        assert_eq!(parts.len(), 1);

        // Both should be for same partition
        assert_eq!(list_entries[0].partition_key, pk);
        assert_eq!(list_entries[1].partition_key, pk);

        // First entry should carry over the old one unchanged
        assert_eq!(list_entries[0].n_superfiles, 1);
        // URI should be exactly the same as the original written part
        assert_eq!(list_entries[0].uri, pw_old.uri);

        // Second entry should be the rewritten latest (1 + 1 = 2 superfiles)
        assert_eq!(list_entries[1].n_superfiles, 2);

        // New part should have the combined latest + new
        assert_eq!(parts[0].superfiles.len(), 2);
        let total_docs: u64 = parts[0].superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 225); // 150 + 75
    }

    // ---- cross-partition tests --------------------------------------------

    #[tokio::test]
    async fn rebalance_two_partitions_both_touched() {
        // Two distinct partitions each have one existing superfile; a new
        // entry is added to both. Both should be rewritten independently.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_partition = 3;
        let opts = Arc::new(base_opts);

        let pk_a = hash2_pk(0);
        let pk_b = hash2_pk(1);
        let (_dir, storage) = local_storage();

        let sf_a = make_superfile_entry_hinted(100, pk_a.clone(), 0);
        let part_a = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a.clone()],
        };
        let pw_a = write_manifest_part(storage.as_ref(), &part_a, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a");

        let sf_b = make_superfile_entry_hinted(200, pk_b.clone(), 1);
        let part_b = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_b.clone()],
        };
        let pw_b = write_manifest_part(storage.as_ref(), &part_b, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_b");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 2,
            },
            parts: vec![
                ManifestListEntry {
                    part_id: pw_a.part_id,
                    uri: pw_a.uri,
                    content_hash: pw_a.content_hash,
                    size_bytes_compressed: pw_a.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk_a.clone(),
                    id_range: (0, 99),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestListEntry {
                    part_id: pw_b.part_id,
                    uri: pw_b.uri,
                    content_hash: pw_b.content_hash,
                    size_bytes_compressed: pw_b.size_bytes_compressed,
                    size_bytes_uncompressed: pw_b.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk_b.clone(),
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);
        let parts_map = dashmap::DashMap::new();
        parts_map.insert(
            part_a.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a)))),
        );
        parts_map.insert(
            part_b.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_b)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_a, sf_b],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let new_entries = vec![
            make_superfile_entry_hinted(50, pk_a.clone(), 0),
            make_superfile_entry_hinted(80, pk_b.clone(), 1),
        ];

        let (list_entries, parts) = rebalance_for_commit(&opts, &old_manifest, &new_entries, &[])
            .await
            .expect("rebalance");

        // Both partitions are rewritten: 2 list entries, 2 new parts
        assert_eq!(list_entries.len(), 2);
        assert_eq!(parts.len(), 2);

        // Order preserved: partition A first, then B
        assert_eq!(list_entries[0].partition_key, pk_a);
        assert_eq!(list_entries[1].partition_key, pk_b);

        // Partition A: 1 existing + 1 new = 2 superfiles, 150 docs
        assert_eq!(list_entries[0].n_superfiles, 2);
        assert_eq!(parts[0].superfiles.len(), 2);
        let docs_a: u64 = parts[0].superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(docs_a, 150);

        // Partition B: 1 existing + 1 new = 2 superfiles, 280 docs
        assert_eq!(list_entries[1].n_superfiles, 2);
        assert_eq!(parts[1].superfiles.len(), 2);
        let docs_b: u64 = parts[1].superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(docs_b, 280);
    }

    #[tokio::test]
    async fn rebalance_two_partitions_one_touched_exact_carry_over() {
        // Partition A is touched (gets a new entry); partition B is not.
        // Verifies that B's list entry carries over with the exact URI and
        // content_hash that were written — no re-encode, no PUT.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_partition = 3;
        let opts = Arc::new(base_opts);

        let pk_a = hash2_pk(0);
        let pk_b = hash2_pk(1);
        let (_dir, storage) = local_storage();

        let sf_a = make_superfile_entry_hinted(100, pk_a.clone(), 0);
        let part_a = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a.clone()],
        };
        let pw_a = write_manifest_part(storage.as_ref(), &part_a, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a");

        let sf_b = make_superfile_entry_hinted(200, pk_b.clone(), 1);
        let part_b = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_b.clone()],
        };
        let pw_b = write_manifest_part(storage.as_ref(), &part_b, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_b");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 2,
            },
            parts: vec![
                ManifestListEntry {
                    part_id: pw_a.part_id,
                    uri: pw_a.uri,
                    content_hash: pw_a.content_hash,
                    size_bytes_compressed: pw_a.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk_a.clone(),
                    id_range: (0, 99),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestListEntry {
                    part_id: pw_b.part_id,
                    uri: pw_b.uri.clone(),
                    content_hash: pw_b.content_hash,
                    size_bytes_compressed: pw_b.size_bytes_compressed,
                    size_bytes_uncompressed: pw_b.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk_b.clone(),
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);
        let parts_map = dashmap::DashMap::new();
        parts_map.insert(
            part_a.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a)))),
        );
        parts_map.insert(
            part_b.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_b)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_a, sf_b],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        // Only touch partition A
        let new_entries = vec![make_superfile_entry_hinted(50, pk_a.clone(), 0)];

        let (list_entries, parts) = rebalance_for_commit(&opts, &old_manifest, &new_entries, &[])
            .await
            .expect("rebalance");

        // 2 list entries (A rewritten, B carried over), 1 new part (A only)
        assert_eq!(list_entries.len(), 2);
        assert_eq!(parts.len(), 1);

        // Partition A: rewritten with 2 superfiles, 150 docs
        assert_eq!(list_entries[0].partition_key, pk_a);
        assert_eq!(list_entries[0].n_superfiles, 2);
        assert_eq!(parts[0].superfiles.len(), 2);
        let docs_a: u64 = parts[0].superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(docs_a, 150);

        // Partition B: exact carry-over — URI and content_hash unchanged
        assert_eq!(list_entries[1].partition_key, pk_b);
        assert_eq!(list_entries[1].n_superfiles, 1);
        assert_eq!(list_entries[1].uri, pw_b.uri);
        assert_eq!(list_entries[1].content_hash, pw_b.content_hash);
    }

    #[tokio::test]
    async fn rebalance_two_partitions_each_with_prior_split() {
        // Each partition already has two parts from a prior split: an older
        // frozen part and a latest mutable part. Adding one new entry to each
        // partition should rewrite only the latest part for each, carrying
        // the older parts over unchanged.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_partition = 2;
        let opts = Arc::new(base_opts);

        let pk_a = hash2_pk(0);
        let pk_b = hash2_pk(1);
        let (_dir, storage) = local_storage();

        // Partition A: two parts
        let sf_a_old = make_superfile_entry_hinted(100, pk_a.clone(), 0);
        let part_a_old = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_old.clone()],
        };
        let pw_a_old = write_manifest_part(storage.as_ref(), &part_a_old, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a_old");

        let sf_a_latest = make_superfile_entry_hinted(150, pk_a.clone(), 0);
        let part_a_latest = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_latest.clone()],
        };
        let pw_a_latest =
            write_manifest_part(storage.as_ref(), &part_a_latest, MANIFEST_ZSTD_LEVEL)
                .await
                .expect("write part_a_latest");

        // Partition B: two parts
        let sf_b_old = make_superfile_entry_hinted(200, pk_b.clone(), 1);
        let part_b_old = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_b_old.clone()],
        };
        let pw_b_old = write_manifest_part(storage.as_ref(), &part_b_old, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_b_old");

        let sf_b_latest = make_superfile_entry_hinted(250, pk_b.clone(), 1);
        let part_b_latest = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_b_latest.clone()],
        };
        let pw_b_latest =
            write_manifest_part(storage.as_ref(), &part_b_latest, MANIFEST_ZSTD_LEVEL)
                .await
                .expect("write part_b_latest");

        // List order: [a_old, a_latest, b_old, b_latest]
        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 2,
            },
            parts: vec![
                ManifestListEntry {
                    part_id: pw_a_old.part_id,
                    uri: pw_a_old.uri.clone(),
                    content_hash: pw_a_old.content_hash,
                    size_bytes_compressed: pw_a_old.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_old.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk_a.clone(),
                    id_range: (0, 99),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestListEntry {
                    part_id: pw_a_latest.part_id,
                    uri: pw_a_latest.uri,
                    content_hash: pw_a_latest.content_hash,
                    size_bytes_compressed: pw_a_latest.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_latest.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk_a.clone(),
                    id_range: (0, 149),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestListEntry {
                    part_id: pw_b_old.part_id,
                    uri: pw_b_old.uri.clone(),
                    content_hash: pw_b_old.content_hash,
                    size_bytes_compressed: pw_b_old.size_bytes_compressed,
                    size_bytes_uncompressed: pw_b_old.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk_b.clone(),
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestListEntry {
                    part_id: pw_b_latest.part_id,
                    uri: pw_b_latest.uri,
                    content_hash: pw_b_latest.content_hash,
                    size_bytes_compressed: pw_b_latest.size_bytes_compressed,
                    size_bytes_uncompressed: pw_b_latest.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk_b.clone(),
                    id_range: (0, 249),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);
        let parts_map = dashmap::DashMap::new();
        parts_map.insert(
            part_a_latest.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a_latest)))),
        );
        parts_map.insert(
            part_b_latest.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_b_latest)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_a_old, sf_a_latest, sf_b_old, sf_b_latest],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let new_entries = vec![
            make_superfile_entry_hinted(75, pk_a.clone(), 0),
            make_superfile_entry_hinted(90, pk_b.clone(), 1),
        ];

        let (list_entries, parts) = rebalance_for_commit(&opts, &old_manifest, &new_entries, &[])
            .await
            .expect("rebalance");

        // 4 list entries: [a_old, a_rewritten, b_old, b_rewritten]
        assert_eq!(list_entries.len(), 4);
        // 2 new parts: one rewrite per partition
        assert_eq!(parts.len(), 2);

        // [0] Partition A old: carried over exactly — URI and content_hash unchanged
        assert_eq!(list_entries[0].partition_key, pk_a);
        assert_eq!(list_entries[0].n_superfiles, 1);
        assert_eq!(list_entries[0].uri, pw_a_old.uri);
        assert_eq!(list_entries[0].content_hash, pw_a_old.content_hash);

        // [1] Partition A latest: rewritten with 1 existing + 1 new = 2 superfiles, 225 docs
        assert_eq!(list_entries[1].partition_key, pk_a);
        assert_eq!(list_entries[1].n_superfiles, 2);
        assert_eq!(parts[0].superfiles.len(), 2);
        let docs_a: u64 = parts[0].superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(docs_a, 225); // 150 + 75

        // [2] Partition B old: carried over exactly — URI and content_hash unchanged
        assert_eq!(list_entries[2].partition_key, pk_b);
        assert_eq!(list_entries[2].n_superfiles, 1);
        assert_eq!(list_entries[2].uri, pw_b_old.uri);
        assert_eq!(list_entries[2].content_hash, pw_b_old.content_hash);

        // [3] Partition B latest: rewritten with 1 existing + 1 new = 2 superfiles, 340 docs
        assert_eq!(list_entries[3].partition_key, pk_b);
        assert_eq!(list_entries[3].n_superfiles, 2);
        assert_eq!(parts[1].superfiles.len(), 2);
        let docs_b: u64 = parts[1].superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(docs_b, 340); // 250 + 90
    }

    // ---- removal tests ---------------------------------------------------

    #[tokio::test]
    async fn rebalance_remove_one_superfile_from_partition() {
        // Partition has 2 superfiles; remove one. Verifies the part is
        // rewritten containing only the superfile that was not removed.
        let opts = make_opts();
        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf_keep = make_superfile_entry(100, pk.clone());
        let sf_remove = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_keep.clone(), sf_remove.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestListEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                partition_key: pk.clone(),
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);
        let parts_map = dashmap::DashMap::new();
        parts_map.insert(
            existing_part.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_keep.clone(), sf_remove.clone()],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let (list_entries, parts) =
            rebalance_for_commit(&opts, &old_manifest, &[], std::slice::from_ref(&sf_remove))
                .await
                .expect("rebalance");

        // Part rewritten with 1 superfile; no cold entries
        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].partition_key, pk);
        assert_eq!(list_entries[0].n_superfiles, 1);
        assert_eq!(parts[0].superfiles.len(), 1);
        assert_eq!(parts[0].superfiles[0].superfile_id, sf_keep.superfile_id);
        let total_docs: u64 = parts[0].superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 100);
    }

    #[tokio::test]
    async fn rebalance_add_and_remove_in_same_partition() {
        // One new superfile is added while one existing superfile is removed
        // in the same partition. The resulting part should contain the
        // surviving existing superfile plus the new one — not the removed one.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_partition = 3;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf_keep = make_superfile_entry(100, pk.clone());
        let sf_remove = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_keep.clone(), sf_remove.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestListEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                partition_key: pk.clone(),
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);
        let parts_map = dashmap::DashMap::new();
        parts_map.insert(
            existing_part.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_keep.clone(), sf_remove.clone()],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let sf_new = make_superfile_entry(75, pk.clone());
        let new_entries = vec![sf_new.clone()];

        let (list_entries, parts) = rebalance_for_commit(
            &opts,
            &old_manifest,
            &new_entries,
            std::slice::from_ref(&sf_remove),
        )
        .await
        .expect("rebalance");

        // Net result: 1 list entry, 1 part — sf_keep + sf_new, sf_remove absent
        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].partition_key, pk);
        assert_eq!(list_entries[0].n_superfiles, 2);
        assert_eq!(parts[0].superfiles.len(), 2);

        let ids: Vec<_> = parts[0].superfiles.iter().map(|s| s.superfile_id).collect();
        assert!(ids.contains(&sf_keep.superfile_id));
        assert!(ids.contains(&sf_new.superfile_id));
        assert!(!ids.contains(&sf_remove.superfile_id));

        let total_docs: u64 = parts[0].superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_docs, 175); // 100 + 75
    }

    #[tokio::test]
    async fn rebalance_remove_from_one_partition_other_carried_over_exactly() {
        // Two partitions: remove a superfile from partition A, leave partition B alone.
        // Verifies partition B's list entry is carried over with the exact URI and
        // content_hash — no re-encode, no PUT — while partition A is rewritten.
        let opts = make_opts();
        let pk_a = hash2_pk(0);
        let pk_b = hash2_pk(1);
        let (_dir, storage) = local_storage();

        let sf_a_keep = make_superfile_entry_hinted(100, pk_a.clone(), 0);
        let sf_a_remove = make_superfile_entry_hinted(150, pk_a.clone(), 0);
        let part_a = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_keep.clone(), sf_a_remove.clone()],
        };
        let pw_a = write_manifest_part(storage.as_ref(), &part_a, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a");

        let sf_b = make_superfile_entry_hinted(200, pk_b.clone(), 1);
        let part_b = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_b.clone()],
        };
        let pw_b = write_manifest_part(storage.as_ref(), &part_b, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_b");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 2,
            },
            parts: vec![
                ManifestListEntry {
                    part_id: pw_a.part_id,
                    uri: pw_a.uri,
                    content_hash: pw_a.content_hash,
                    size_bytes_compressed: pw_a.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a.size_bytes_uncompressed,
                    n_superfiles: 2,
                    partition_key: pk_a.clone(),
                    id_range: (0, 149),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestListEntry {
                    part_id: pw_b.part_id,
                    uri: pw_b.uri.clone(),
                    content_hash: pw_b.content_hash,
                    size_bytes_compressed: pw_b.size_bytes_compressed,
                    size_bytes_uncompressed: pw_b.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk_b.clone(),
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);
        let parts_map = dashmap::DashMap::new();
        parts_map.insert(
            part_a.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a)))),
        );
        parts_map.insert(
            part_b.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_b)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf_a_keep.clone(), sf_a_remove.clone(), sf_b.clone()],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let (list_entries, parts) = rebalance_for_commit(
            &opts,
            &old_manifest,
            &[],
            std::slice::from_ref(&sf_a_remove),
        )
        .await
        .expect("rebalance");

        // 2 list entries, 1 new part (only partition A was rewritten)
        assert_eq!(list_entries.len(), 2);
        assert_eq!(parts.len(), 1);

        // Partition A: rewritten with 1 surviving superfile
        assert_eq!(list_entries[0].partition_key, pk_a);
        assert_eq!(list_entries[0].n_superfiles, 1);
        assert_eq!(parts[0].superfiles.len(), 1);
        assert_eq!(parts[0].superfiles[0].superfile_id, sf_a_keep.superfile_id);
        let docs_a: u64 = parts[0].superfiles.iter().map(|s| s.n_docs).sum();
        assert_eq!(docs_a, 100);

        // Partition B: exact carry-over — URI and content_hash unchanged
        assert_eq!(list_entries[1].partition_key, pk_b);
        assert_eq!(list_entries[1].n_superfiles, 1);
        assert_eq!(list_entries[1].uri, pw_b.uri);
        assert_eq!(list_entries[1].content_hash, pw_b.content_hash);
    }

    #[tokio::test]
    async fn rebalance_remove_from_latest_part_in_split_partition() {
        // Partition A has two parts from a prior split: part_a_old (frozen, 1 sf)
        // and part_a_latest (mutable, 2 sfs). We remove sf_a_latest_remove,
        // which lives in the SECOND (latest) part.
        //
        // Bug: the removal loop calls removals_by_partition.remove(&partition_key)
        // for each entry in out_list_entries. When part_a_old is processed first,
        // the key [0,0,0,0] is consumed from the map. When part_a_latest is
        // processed second, remove() returns None and the entry carries over
        // unchanged — sf_a_latest_remove is never removed. As a side effect,
        // part_a_old is unnecessarily rewritten (its URI changes even though its
        // contents did not).
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_partition = 2;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        // part_a_old: frozen entry from a prior split
        let sf_a_old = make_superfile_entry(100, pk.clone());
        let part_a_old = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_old.clone()],
        };
        let pw_a_old = write_manifest_part(storage.as_ref(), &part_a_old, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a_old");

        // part_a_latest: current mutable entry; contains the sf to remove
        let sf_a_latest_keep = make_superfile_entry(150, pk.clone());
        let sf_a_latest_remove = make_superfile_entry(200, pk.clone());
        let part_a_latest = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_latest_keep.clone(), sf_a_latest_remove.clone()],
        };
        let pw_a_latest =
            write_manifest_part(storage.as_ref(), &part_a_latest, MANIFEST_ZSTD_LEVEL)
                .await
                .expect("write part_a_latest");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![
                ManifestListEntry {
                    part_id: pw_a_old.part_id,
                    uri: pw_a_old.uri.clone(),
                    content_hash: pw_a_old.content_hash,
                    size_bytes_compressed: pw_a_old.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_old.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk.clone(),
                    id_range: (0, 99),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestListEntry {
                    part_id: pw_a_latest.part_id,
                    uri: pw_a_latest.uri.clone(),
                    content_hash: pw_a_latest.content_hash,
                    size_bytes_compressed: pw_a_latest.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_latest.size_bytes_uncompressed,
                    n_superfiles: 2,
                    partition_key: pk.clone(),
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);
        let parts_map = dashmap::DashMap::new();
        parts_map.insert(
            part_a_old.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a_old)))),
        );
        parts_map.insert(
            part_a_latest.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a_latest)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![
                    sf_a_old.clone(),
                    sf_a_latest_keep.clone(),
                    sf_a_latest_remove.clone(),
                ],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let (list_entries, parts) = rebalance_for_commit(
            &opts,
            &old_manifest,
            &[],
            std::slice::from_ref(&sf_a_latest_remove),
        )
        .await
        .expect("rebalance");

        assert_eq!(list_entries.len(), 2);
        // Both parts in the split are rewritten: any part in a partition with a
        // pending removal is rewritten regardless of whether the removal matched
        // anything in it.
        assert_eq!(parts.len(), 2);

        // Both list entries are for the same partition
        assert_eq!(list_entries[0].partition_key, pk);
        assert_eq!(list_entries[1].partition_key, pk);

        // sf_a_old survives (in one of the output parts)
        // sf_a_latest_keep survives (in one of the output parts)
        // sf_a_latest_remove is absent from every output part
        let all_ids: Vec<_> = parts
            .iter()
            .flat_map(|p| p.superfiles.iter())
            .map(|s| s.superfile_id)
            .collect();
        assert!(
            all_ids.contains(&sf_a_old.superfile_id),
            "sf_a_old must survive"
        );
        assert!(
            all_ids.contains(&sf_a_latest_keep.superfile_id),
            "sf_a_latest_keep must survive"
        );
        assert!(
            !all_ids.contains(&sf_a_latest_remove.superfile_id),
            "sf_a_latest_remove must be absent"
        );

        // Each rewritten part has exactly 1 superfile
        assert_eq!(list_entries[0].n_superfiles, 1);
        assert_eq!(list_entries[1].n_superfiles, 1);
    }

    #[tokio::test]
    async fn rebalance_remove_all_superfiles_empties_partition() {
        // All superfiles in a partition are removed. Documents the current
        // behavior: the list entry survives with n_superfiles=0 and the
        // part has no superfiles (empty partition).
        let opts = make_opts();
        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf1 = make_superfile_entry(100, pk.clone());
        let sf2 = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf1.clone(), sf2.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestListEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                partition_key: pk.clone(),
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);
        let parts_map = dashmap::DashMap::new();
        parts_map.insert(
            existing_part.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf1.clone(), sf2.clone()],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let (list_entries, parts) =
            rebalance_for_commit(&opts, &old_manifest, &[], &[sf1.clone(), sf2.clone()])
                .await
                .expect("rebalance");

        // Both superfiles removed: list entry remains with n_superfiles=0
        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].partition_key, pk);
        assert_eq!(list_entries[0].n_superfiles, 0);
        assert_eq!(parts[0].superfiles.len(), 0);
    }

    #[tokio::test]
    async fn rebalance_remove_nonexistent_superfile_id_is_noop() {
        // entries_to_remove contains a superfile_id that is not present in any
        // part. The filter matches nothing and both original superfiles survive.
        // The part is still rewritten (the removal loop doesn't skip parts where
        // no removal matched), so n_superfiles stays at 2.
        let opts = make_opts();
        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        let sf1 = make_superfile_entry(100, pk.clone());
        let sf2 = make_superfile_entry(150, pk.clone());

        let existing_part = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf1.clone(), sf2.clone()],
        };
        let pw = write_manifest_part(storage.as_ref(), &existing_part, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![ManifestListEntry {
                part_id: pw.part_id,
                uri: pw.uri,
                content_hash: pw.content_hash,
                size_bytes_compressed: pw.size_bytes_compressed,
                size_bytes_uncompressed: pw.size_bytes_uncompressed,
                n_superfiles: 2,
                partition_key: pk.clone(),
                id_range: (0, 149),
                scalar_stats_agg: Default::default(),
                fts_summary_agg: Default::default(),
                vector_summary_agg: Default::default(),
            }],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);
        let parts_map = dashmap::DashMap::new();
        parts_map.insert(
            existing_part.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(existing_part)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![sf1.clone(), sf2.clone()],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        // sf_ghost was never added to any part; its superfile_id won't match anything
        let sf_ghost = make_superfile_entry(50, pk.clone());

        let (list_entries, parts) =
            rebalance_for_commit(&opts, &old_manifest, &[], std::slice::from_ref(&sf_ghost))
                .await
                .expect("rebalance");

        assert_eq!(list_entries.len(), 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(list_entries[0].partition_key, pk);
        assert_eq!(list_entries[0].n_superfiles, 2);
        assert_eq!(parts[0].superfiles.len(), 2);

        let ids: Vec<_> = parts[0].superfiles.iter().map(|s| s.superfile_id).collect();
        assert!(ids.contains(&sf1.superfile_id), "sf1 must survive");
        assert!(ids.contains(&sf2.superfile_id), "sf2 must survive");
        assert!(
            !ids.contains(&sf_ghost.superfile_id),
            "ghost id must not appear"
        );
    }

    #[tokio::test]
    async fn rebalance_remove_from_older_frozen_part_in_split_partition() {
        // Partition A has two parts from a prior split: part_a_old (frozen, 2
        // sfs: sf_a_old_keep + sf_a_old_remove) and part_a_latest (mutable, 1
        // sf). We remove sf_a_old_remove, which lives in the FIRST (older,
        // frozen) part.
        //
        // Because the fix applies the removal set to every part in the partition,
        // both parts are rewritten. sf_a_old_remove is absent from the output;
        // sf_a_old_keep and sf_a_latest survive.
        let mut base_opts =
            SupertableOptions::new(simple_schema(), vec![], vec![], None).expect("valid options");
        base_opts.target_superfiles_per_partition = 2;
        let opts = Arc::new(base_opts);

        let pk = hash_bucket_0_pk();
        let (_dir, storage) = local_storage();

        // part_a_old: frozen entry — contains the sf to remove
        let sf_a_old_keep = make_superfile_entry(100, pk.clone());
        let sf_a_old_remove = make_superfile_entry(150, pk.clone());
        let part_a_old = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_old_keep.clone(), sf_a_old_remove.clone()],
        };
        let pw_a_old = write_manifest_part(storage.as_ref(), &part_a_old, MANIFEST_ZSTD_LEVEL)
            .await
            .expect("write part_a_old");

        // part_a_latest: mutable entry — does not contain the sf to remove
        let sf_a_latest = make_superfile_entry(200, pk.clone());
        let part_a_latest = ManifestPart {
            format_version: part_mod::FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles: vec![sf_a_latest.clone()],
        };
        let pw_a_latest =
            write_manifest_part(storage.as_ref(), &part_a_latest, MANIFEST_ZSTD_LEVEL)
                .await
                .expect("write part_a_latest");

        let list = ManifestList {
            format_version: list_mod::FORMAT_VERSION.into(),
            manifest_id: 0,
            options_hash: ContentHash([0u8; 32]),
            schema: vec![],
            id_column: "_id".into(),
            fts_columns: vec![],
            vector_columns: vec![],
            partition_strategy: PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 1,
            },
            parts: vec![
                ManifestListEntry {
                    part_id: pw_a_old.part_id,
                    uri: pw_a_old.uri,
                    content_hash: pw_a_old.content_hash,
                    size_bytes_compressed: pw_a_old.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_old.size_bytes_uncompressed,
                    n_superfiles: 2,
                    partition_key: pk.clone(),
                    id_range: (0, 149),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
                ManifestListEntry {
                    part_id: pw_a_latest.part_id,
                    uri: pw_a_latest.uri,
                    content_hash: pw_a_latest.content_hash,
                    size_bytes_compressed: pw_a_latest.size_bytes_compressed,
                    size_bytes_uncompressed: pw_a_latest.size_bytes_uncompressed,
                    n_superfiles: 1,
                    partition_key: pk.clone(),
                    id_range: (0, 199),
                    scalar_stats_agg: Default::default(),
                    fts_summary_agg: Default::default(),
                    vector_summary_agg: Default::default(),
                },
            ],
        };
        let loader = crate::supertable::manifest::ManifestPartLoader::new(storage, &list);
        let parts_map = dashmap::DashMap::new();
        parts_map.insert(
            part_a_old.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a_old)))),
        );
        parts_map.insert(
            part_a_latest.part_id,
            Arc::new(OnceCell::new_with(Some(Arc::new(part_a_latest)))),
        );
        let old_manifest = Arc::new(Manifest {
            superfile_list: crate::supertable::manifest::SuperfileList {
                manifest_id: 0,
                options: opts.clone(),
                superfiles: vec![
                    sf_a_old_keep.clone(),
                    sf_a_old_remove.clone(),
                    sf_a_latest.clone(),
                ],
            },
            list: Some(list),
            parts: parts_map,
            loader: Some(Arc::new(loader)),
        });

        let (list_entries, parts) = rebalance_for_commit(
            &opts,
            &old_manifest,
            &[],
            std::slice::from_ref(&sf_a_old_remove),
        )
        .await
        .expect("rebalance");

        assert_eq!(list_entries.len(), 2);
        // Both parts rewritten: the fix applies the removal set to every part in
        // the partition, so the latest is also rewritten (no match, same content)
        assert_eq!(parts.len(), 2);

        assert_eq!(list_entries[0].partition_key, pk);
        assert_eq!(list_entries[1].partition_key, pk);

        // sf_a_old_keep and sf_a_latest survive; sf_a_old_remove is absent
        let all_ids: Vec<_> = parts
            .iter()
            .flat_map(|p| p.superfiles.iter())
            .map(|s| s.superfile_id)
            .collect();
        assert!(
            all_ids.contains(&sf_a_old_keep.superfile_id),
            "sf_a_old_keep must survive"
        );
        assert!(
            all_ids.contains(&sf_a_latest.superfile_id),
            "sf_a_latest must survive"
        );
        assert!(
            !all_ids.contains(&sf_a_old_remove.superfile_id),
            "sf_a_old_remove must be absent"
        );

        // Old part now has 1 sf (sf_a_old_remove was removed)
        assert_eq!(list_entries[0].n_superfiles, 1);
        // Latest part still has 1 sf (removal did not touch it)
        assert_eq!(list_entries[1].n_superfiles, 1);
    }
}
