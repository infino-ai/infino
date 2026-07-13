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
//!   - `manifest/manifest-NNNNNN.json` — immutable per
//!     manifest version. Conditional-create on PUT (S3
//!     `If-None-Match: *` / `O_EXCL` on LocalFS).
//!   - `manifest-parts/part-<content-hash>.avro.zst` — immutable,
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

use std::{str::from_utf8, sync::Arc};

use bytes::Bytes;
use zstd::zstd_safe::get_frame_content_size;

use crate::{
    storage::{ObjectMeta, StorageError, StorageProvider},
    supertable::{
        ManifestLoadError, ManifestSnapshot,
        error::CommitError,
        manifest::{
            list::{self as list_mod, Manifest as PersistedManifest},
            part::{
                self as part_mod, BLAKE3_DIGEST_BYTES, BLAKE3_HEX_LEN, ContentHash, ManifestPart,
                PartId,
            },
        },
    },
};

/// Pointer-file location under the supertable root. The only
/// path that ever gets atomically renamed; everything else is
/// content-addressed and immutable, so a torn write on those
/// paths is invisible (no committed pointer references it).
pub const POINTER_PATH: &str = "_supertable/current";

/// Subdirectory for manifest files.
pub const MANIFEST_DIR: &str = "manifest";

/// Subdirectory for manifest part files.
pub const MANIFEST_PARTS_DIR: &str = "manifest-parts";

/// Build the URI for a manifest at a given manifest_id.
/// 6-digit zero-pad gives stable lexicographic ordering for
/// `aws s3 ls`-style listings up through 999,999 versions.
pub fn manifest_uri(manifest_id: u64) -> String {
    format!("{MANIFEST_DIR}/manifest-{manifest_id:06}.json")
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
    pub manifest_uri: String,
    pub content_hash: ContentHash,
}

impl PointerFile {
    /// Serialize to the on-disk text format.
    ///
    /// ```text
    /// manifest_id=42
    /// manifest_uri=manifest/manifest-000042.json
    /// content_hash=blake3:def...
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        format!(
            "manifest_id={}\nmanifest_uri={}\ncontent_hash=blake3:{}\n",
            self.manifest_id,
            self.manifest_uri,
            self.content_hash.to_hex(),
        )
        .into_bytes()
    }

    /// Parse the on-disk text format.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ManifestLoadError> {
        let s = from_utf8(bytes)
            .map_err(|e| ManifestLoadError::PointerParse(format!("not utf-8: {e}")))?;

        let mut manifest_id: Option<u64> = None;
        let mut manifest_uri: Option<String> = None;
        let mut content_hash: Option<ContentHash> = None;

        for line in s.lines() {
            if line.is_empty() {
                continue;
            }
            let (key, value) = line.split_once('=').ok_or_else(|| {
                ManifestLoadError::PointerParse(format!("no '=' in line: {line:?}"))
            })?;
            match key {
                "manifest_id" => {
                    manifest_id = Some(value.parse::<u64>().map_err(|e| {
                        ManifestLoadError::PointerParse(format!("manifest_id: {e}"))
                    })?);
                }
                "manifest_uri" => {
                    manifest_uri = Some(value.to_string());
                }
                "content_hash" => {
                    let hex = value.strip_prefix("blake3:").ok_or_else(|| {
                        ManifestLoadError::PointerParse(format!(
                            "content_hash missing 'blake3:' prefix: {value}"
                        ))
                    })?;
                    if hex.len() != BLAKE3_HEX_LEN {
                        return Err(ManifestLoadError::PointerParse(format!(
                            "content_hash hex must be {BLAKE3_HEX_LEN} chars; got {}",
                            hex.len()
                        )));
                    }
                    let mut bytes = [0u8; BLAKE3_DIGEST_BYTES];
                    for i in 0..BLAKE3_DIGEST_BYTES {
                        bytes[i] =
                            u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| {
                                ManifestLoadError::PointerParse(format!("content_hash hex: {hex}"))
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
                .ok_or_else(|| ManifestLoadError::PointerParse("missing manifest_id".into()))?,
            manifest_uri: manifest_uri
                .ok_or_else(|| ManifestLoadError::PointerParse("missing manifest_uri".into()))?,
            content_hash: content_hash
                .ok_or_else(|| ManifestLoadError::PointerParse("missing content_hash".into()))?,
        })
    }

    pub fn get_manifest_id(&self) -> u64 {
        self.manifest_id
    }
}

/// Read the pointer file from storage.
///
/// Returns `Ok(None)` if the pointer doesn't exist (fresh
/// supertable). Returns `Err` on any other failure.
pub async fn read_pointer(
    storage: &dyn StorageProvider,
) -> Result<Option<(PointerFile, ObjectMeta)>, ManifestLoadError> {
    match storage.get(POINTER_PATH).await {
        Ok((bytes, meta)) => Ok(Some((PointerFile::from_bytes(&bytes)?, meta))),
        Err(StorageError::NotFound { .. }) => Ok(None),
        Err(e) => Err(ManifestLoadError::Storage(e)),
    }
}

/// Outcome of a conditional pointer probe ([`probe_pointer`]).
pub enum PointerProbe {
    /// No pointer object exists — never-created table, or a lost
    /// pointer. Mirrors [`read_pointer`]'s `Ok(None)`.
    Absent,
    /// The pointer's etag matches the caller's last-seen etag: no
    /// commit has been published since, so the caller's manifest
    /// view is already current. No body was transferred.
    NotModified,
    /// The pointer was read (unconditionally, or because it changed
    /// since the caller's etag). `meta.etag` is the token to carry
    /// into the next probe.
    Read(PointerFile, ObjectMeta),
}

/// Read the pointer file, skipping the body when it hasn't changed.
///
/// With `if_none_match: Some(etag)` this is the read-path freshness
/// probe: on S3/Azure an unchanged pointer answers as a bodyless
/// HTTP 304 ([`PointerProbe::NotModified`]), so the per-query
/// consistency check under `Consistency::Strong` (and the per-window
/// check under `BoundedStaleness`) costs a roundtrip but no
/// transfer or parse. `None` degrades to a plain [`read_pointer`].
pub async fn probe_pointer(
    storage: &dyn StorageProvider,
    if_none_match: Option<&str>,
) -> Result<PointerProbe, ManifestLoadError> {
    let Some(etag) = if_none_match else {
        return Ok(match read_pointer(storage).await? {
            Some((pointer, meta)) => PointerProbe::Read(pointer, meta),
            None => PointerProbe::Absent,
        });
    };
    match storage.get_if_none_match(POINTER_PATH, etag).await {
        Ok(None) => Ok(PointerProbe::NotModified),
        Ok(Some((bytes, meta))) => Ok(PointerProbe::Read(PointerFile::from_bytes(&bytes)?, meta)),
        Err(StorageError::NotFound { .. }) => Ok(PointerProbe::Absent),
        Err(e) => Err(ManifestLoadError::Storage(e)),
    }
}

pub struct EncodedPart {
    pub part: ManifestPart,
    pub encoded: Vec<u8>,
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

/// Outcome of writing a manifest.
#[derive(Debug, Clone)]
pub struct ManifestWriteResult {
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
    let size_uncompressed = frame_content_size(&compressed, size_compressed);

    match storage.put_atomic(&uri, Bytes::from(compressed)).await {
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

/// Read the decompressed size from the zstd frame header in O(1), no
/// actual decompression.
pub(crate) fn frame_content_size(compressed: &[u8], fallback: u64) -> u64 {
    get_frame_content_size(compressed)
        .ok()
        .flatten()
        .unwrap_or(fallback)
}

/// PUT pre-encoded part bytes to storage.
pub(crate) async fn write_part_bytes(
    storage: &dyn StorageProvider,
    encoded: &[u8],
) -> Result<(), CommitError> {
    let uri = part_uri(&ContentHash::of(encoded));
    match storage
        .put_atomic(&uri, Bytes::copy_from_slice(encoded))
        .await
    {
        Ok(_) | Err(StorageError::PreconditionFailed { .. }) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Encode + write a manifest list. Conditional-create
/// (`put_atomic`) — exactly one writer succeeds in publishing
/// a given `manifest_id`'s manifest; concurrent attempts surface
/// `PreconditionFailed` and the caller's commit fails (the
/// writer's OCC retry loop catches this).
pub async fn write_manifest(
    storage: &dyn StorageProvider,
    list: &PersistedManifest,
) -> Result<ManifestWriteResult, CommitError> {
    let json = list_mod::encode(list).map_err(|e| CommitError::Encode(e.to_string()))?;
    let content_hash = ContentHash::of(&json);
    let uri = manifest_uri(list.manifest_id);
    let size = json.len() as u64;
    storage.put_atomic(&uri, Bytes::from(json)).await?;
    Ok(ManifestWriteResult {
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
    let bytes = Bytes::from(pointer.to_bytes());
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
pub(crate) fn translate_contention(e: CommitError) -> CommitError {
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
    current: Arc<ManifestSnapshot>,
) -> Result<Option<String>, CommitError> {
    let Some((pointer_file, meta)) = read_pointer(storage.as_ref())
        .await
        .map_err(|e| CommitError::PointerParse(e.to_string()))?
    else {
        return Ok(None);
    };
    let Some(meta_list) = current.list.as_ref() else {
        // no manifest list for in-memory supertables
        return Ok(None);
    };
    if pointer_file.get_manifest_id() == meta_list.manifest_id {
        return Ok(meta.etag);
    }
    Err(CommitError::WriteContentionExhausted)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::LocalFsStorageProvider,
        supertable::manifest::list::{
            FORMAT_VERSION as LIST_FORMAT_VERSION, Manifest as PersistedManifest, PartitionStrategy,
        },
    };

    // ---- URI helpers ---------------------------------------------------

    #[test]
    fn manifest_uri_zero_pads_to_six_digits() {
        // 6-digit zero-pad gives stable lexicographic ordering
        // for `aws s3 ls`-style listings up to 999,999 versions.
        assert_eq!(manifest_uri(0), "manifest/manifest-000000.json");
        assert_eq!(manifest_uri(42), "manifest/manifest-000042.json");
        assert_eq!(manifest_uri(123_456), "manifest/manifest-123456.json");
    }

    #[test]
    fn manifest_uri_overflows_padding_for_large_ids_intentionally() {
        // Past 6 digits the format widens — no truncation, just
        // breaks lex ordering. Spec'd behaviour; locked in to
        // catch accidental width changes.
        assert_eq!(manifest_uri(1_000_000), "manifest/manifest-1000000.json");
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
        assert!(uri_a.starts_with("manifest-parts/part-"));
        assert!(uri_a.ends_with(".avro.zst"));
        assert_eq!(
            uri_a,
            format!("manifest-parts/part-{}.avro.zst", h.to_hex())
        );
    }

    // ---- PointerFile round-trip ----------------------------------------

    fn sample_pointer() -> PointerFile {
        let mut bytes = [0u8; 32];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        PointerFile {
            manifest_id: 7,
            manifest_uri: "manifest/manifest-000007.json".into(),
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
        let s = from_utf8(&bytes).expect("utf-8");
        assert!(s.contains("manifest_id=7"));
        assert!(s.contains("manifest_uri=manifest/manifest-000007.json"));
        assert!(s.contains("content_hash=blake3:"));
        let parsed = PointerFile::from_bytes(&bytes).expect("parse");
        assert_eq!(parsed, p);
    }

    #[test]
    fn pointer_file_from_bytes_skips_blank_lines() {
        let bytes = b"\nmanifest_id=1\n\nmanifest_uri=foo.json\ncontent_hash=blake3:0000000000000000000000000000000000000000000000000000000000000000\n";
        let parsed = PointerFile::from_bytes(bytes).expect("parse");
        assert_eq!(parsed.manifest_id, 1);
        assert_eq!(parsed.manifest_uri, "foo.json");
        assert_eq!(parsed.content_hash.0, [0u8; 32]);
    }

    #[test]
    fn pointer_file_from_bytes_ignores_unknown_keys() {
        // Forward-compat: unknown keys must not error so that
        // an older reader can open a pointer that a future
        // writer extended.
        let bytes = b"manifest_id=2\nmanifest_uri=x.json\ncontent_hash=blake3:1111111111111111111111111111111111111111111111111111111111111111\nfuture_field=ignored\n";
        let parsed = PointerFile::from_bytes(bytes).expect("parse");
        assert_eq!(parsed.manifest_id, 2);
    }

    // ---- PointerFile parse errors --------------------------------------

    fn assert_parse_err(bytes: &[u8], needle: &str) {
        let err = PointerFile::from_bytes(bytes).expect_err("must error");
        match err {
            ManifestLoadError::PointerParse(msg) => assert!(
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
            b"manifest_id=abc\nmanifest_uri=x\ncontent_hash=blake3:0000000000000000000000000000000000000000000000000000000000000000\n",
            "manifest_id",
        );
    }

    #[test]
    fn pointer_file_from_bytes_rejects_content_hash_without_prefix() {
        assert_parse_err(
            b"manifest_id=1\nmanifest_uri=x\ncontent_hash=cafebabe\n",
            "blake3:",
        );
    }

    #[test]
    fn pointer_file_from_bytes_rejects_short_hex() {
        // blake3 is always 32 bytes → 64 hex chars; anything
        // else is malformed.
        assert_parse_err(
            b"manifest_id=1\nmanifest_uri=x\ncontent_hash=blake3:dead\n",
            "64 chars",
        );
    }

    #[test]
    fn pointer_file_from_bytes_rejects_bad_hex_chars() {
        // 64 chars but containing a non-hex char → parse error
        // from u8::from_str_radix.
        let mut hex = String::from("blake3:");
        hex.push_str(&"z".repeat(64));
        let payload = format!("manifest_id=1\nmanifest_uri=x\ncontent_hash={hex}\n");
        assert_parse_err(payload.as_bytes(), "content_hash hex");
    }

    #[test]
    fn pointer_file_from_bytes_rejects_missing_manifest_id() {
        assert_parse_err(
            b"manifest_uri=x\ncontent_hash=blake3:0000000000000000000000000000000000000000000000000000000000000000\n",
            "missing manifest_id",
        );
    }

    #[test]
    fn pointer_file_from_bytes_rejects_missing_manifest_uri() {
        assert_parse_err(
            b"manifest_id=1\ncontent_hash=blake3:0000000000000000000000000000000000000000000000000000000000000000\n",
            "missing manifest_uri",
        );
    }

    #[test]
    fn pointer_file_from_bytes_rejects_missing_content_hash() {
        assert_parse_err(b"manifest_id=1\nmanifest_uri=x\n", "missing content_hash");
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
    async fn probe_pointer_covers_absent_read_and_not_modified() {
        let (_dir, storage) = local_storage();

        // Absent pointer, with and without a stale etag in hand.
        assert!(matches!(
            probe_pointer(storage.as_ref(), None).await.expect("probe"),
            PointerProbe::Absent
        ));
        assert!(matches!(
            probe_pointer(storage.as_ref(), Some("\"gone\""))
                .await
                .expect("probe"),
            PointerProbe::Absent
        ));

        // First read (no etag) returns the pointer + its etag.
        let p = sample_pointer();
        write_pointer(storage.as_ref(), &p, None)
            .await
            .expect("write");
        let PointerProbe::Read(read, meta) =
            probe_pointer(storage.as_ref(), None).await.expect("probe")
        else {
            panic!("expected Read");
        };
        assert_eq!(read, p);
        let etag = meta.etag.expect("localfs reports etags");

        // Same etag → NotModified, no pointer parse.
        assert!(matches!(
            probe_pointer(storage.as_ref(), Some(&etag))
                .await
                .expect("probe"),
            PointerProbe::NotModified
        ));

        // Pointer rewritten (next manifest version) → the stale etag
        // reads through to the new pointer. The longer uri changes
        // the byte length, so the mtime+size etag can't collide even
        // within one filesystem clock tick.
        let mut p2 = sample_pointer();
        p2.manifest_id += 1;
        p2.manifest_uri = "manifest/manifest-000008-successor.json".into();
        write_pointer(storage.as_ref(), &p2, Some(&etag))
            .await
            .expect("cas rewrite");
        let PointerProbe::Read(read2, meta2) = probe_pointer(storage.as_ref(), Some(&etag))
            .await
            .expect("probe")
        else {
            panic!("expected Read after rewrite");
        };
        assert_eq!(read2, p2);
        assert_ne!(meta2.etag.expect("etag"), etag);
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
        let (read, _) = read_pointer(storage.as_ref())
            .await
            .expect("read")
            .expect("some");
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
        // and PUTs at manifest_uri(manifest_id). Verify the
        // returned URI matches the deterministic naming rule
        // and the bytes are reachable through `get`.
        let (_dir, storage) = local_storage();
        // Smallest valid Manifest shape — no parts, no
        // columns, an empty schema. Encoding only requires the
        // format header + the empty collections.
        let list = PersistedManifest {
            tombstone_seqs: Default::default(),
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
        let res = write_manifest(storage.as_ref(), &list)
            .await
            .expect("write");
        assert_eq!(res.uri, manifest_uri(1));
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
        assert_eq!(MANIFEST_DIR, "manifest");
        assert_eq!(MANIFEST_PARTS_DIR, "manifest-parts");
    }
}
