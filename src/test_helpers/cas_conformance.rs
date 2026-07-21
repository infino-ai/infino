// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Backend-agnostic conditional-write (CAS) conformance check.
//!
//! Exercises the CAS-token invariant documented on
//! [`StorageProvider`](crate::storage::StorageProvider): the token from
//! `head`/`get`, the token returned by `put_*`, and the token accepted by
//! `put_if_match` are all the same kind. The chained-token step is the one
//! that catches a backend returning the wrong token kind (e.g. GCS handing
//! back an HTTP ETag instead of the object generation) — that is the exact
//! bug the WAL persistence layer would hit, since it chains the returned
//! token into the next CAS without re-reading.

use bytes::Bytes;

use crate::storage::{StorageError, StorageProvider};

/// Run the full CAS contract against `p` at a fresh `key`. Backend-agnostic:
/// passes for any provider whose read/return/accept tokens are the same kind,
/// and fails (at the chained-update step) for one that isn't.
///
/// `expect_stale_rejected` gates the final stale-token assertion: real S3 /
/// Azure / GCS and LocalFs enforce it, but the `s3s-fs` emulator does not
/// honor a stale conditional update (its 412 path is covered by the real-S3
/// integration smoke instead), so its caller passes `false`.
pub async fn cas_conformance(p: &dyn StorageProvider, key: &str, expect_stale_rejected: bool) {
    // Distinct payload lengths: LocalFS etags often incorporate size + mtime, so
    // equal-length bodies written in the same mtime tick can share an etag and
    // make a "stale" conditional update spuriously succeed.
    const V1: &[u8] = b"v1";
    const V2: &[u8] = b"v2-xx";
    const V3: &[u8] = b"v3-xxxx";
    const V4: &[u8] = b"v4-xxxxxx";

    // 1. Create establishes the object and yields a token.
    let tok_create = p
        .put_atomic(key, Bytes::from_static(V1))
        .await
        .expect("put_atomic create");

    // 2. The read token round-trips a conditional update.
    let (_, meta) = p.get(key).await.expect("get after create");
    let read_tok = meta.etag.clone();
    let tok_after_v2 = p
        .put_if_match(key, Bytes::from_static(V2), read_tok.as_deref())
        .await
        .expect("conditional update with the read token");

    // 3. THE KEY ASSERTION: the token *returned* by the previous put must
    //    itself be a valid precondition for the next update — this is the
    //    chained path WAL persistence relies on (no re-read between steps).
    //    A backend that returns the wrong token kind fails right here.
    if tok_after_v2.is_some() {
        p.put_if_match(key, Bytes::from_static(V3), tok_after_v2.as_deref())
            .await
            .expect("chained update with the token RETURNED by put_if_match");
    }

    // 4. A stale token (from step 1, now two generations behind) is rejected.
    //    Some emulators don't enforce this; guard on the token being present
    //    and distinct so the assertion only fires where the backend can honor
    //    it (real S3/Azure/GCS do).
    if expect_stale_rejected && tok_create.is_some() && tok_create != tok_after_v2 {
        let stale = p
            .put_if_match(key, Bytes::from_static(V4), tok_create.as_deref())
            .await;
        assert!(
            matches!(stale, Err(StorageError::PreconditionFailed { .. })),
            "stale token must be PreconditionFailed; got {stale:?}"
        );
    }

    p.delete(key).await.expect("cleanup");
}
