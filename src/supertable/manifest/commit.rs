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

use std::sync::Arc;

use futures::future;

use crate::storage::{StorageError, StorageProvider};
use crate::supertable::error::CommitError;
use crate::supertable::manifest::list::{self as list_mod, ManifestList};
use crate::supertable::manifest::part::{self as part_mod, ContentHash, ManifestPart, PartId};

/// Pointer-file location under the supertable root. The only
/// path that ever gets atomically renamed; everything else is
/// content-addressed and immutable, so a torn write on those
/// paths is invisible (no committed pointer references it).
pub const POINTER_PATH: &str = "_supertable/current";

/// Subdirectory for manifest list files.
pub const MANIFEST_LISTS_DIR: &str = "manifest-lists";

/// Subdirectory for manifest part files.
pub const MANIFEST_PARTS_DIR: &str = "manifests";

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
                    if hex.len() != 64 {
                        return Err(CommitError::PointerParse(format!(
                            "content_hash hex must be 64 chars; got {}",
                            hex.len()
                        )));
                    }
                    let mut bytes = [0u8; 32];
                    for i in 0..32 {
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
        Ok(bytes) => Ok(Some(PointerFile::from_bytes(&bytes)?)),
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
        Ok(()) => {}
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
        Ok(()) => Ok(()),
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
