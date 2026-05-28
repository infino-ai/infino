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

#[cfg(test)]
mod tests {
    use super::*;

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

    use crate::storage::LocalFsStorageProvider;
    use std::sync::Arc;
    use tempfile::TempDir;

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
}
