// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Fault-injecting [`StorageProvider`] wrapper for error-path tests.
//!
//! Wraps a real provider and fails selected operations with
//! [`StorageError::TransientExhausted`] — the "provider retries already
//! exhausted, caller must surface" shape — so tests can assert that a
//! storage failure mid-commit / mid-load / mid-query surfaces as a clean
//! error and never as a panic, a wrong result, or a mislabeled
//! write-contention retry loop.
//!
//! Rules are armed per (operation, URI fragment, count) and burn down as
//! they fire, so a test can fail exactly the first N matching calls and
//! then watch the same code path recover. Operations without an armed
//! rule pass through untouched, including listing and the
//! `object_store` handle, so the wrapped provider behaves identically to
//! the inner one everywhere a test isn't deliberately failing it.

use std::{
    io::Error as IoError,
    ops::Range,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use bytes::Bytes;

use crate::{
    runtime_metrics::io::UsageMeter,
    storage::{ObjectMeta, StorageError, StorageProvider},
};

/// Storage operation a fault rule can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultOp {
    Head,
    Get,
    GetRange,
    PutAtomic,
    PutIfMatch,
    Delete,
}

/// The failure an armed rule injects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    /// [`StorageError::TransientExhausted`] — the provider's own retries
    /// are spent, so the caller must surface it. The shape [`FaultStorage::fail`]
    /// arms, and the one that must never be mistaken for contention.
    Transient,
    /// [`StorageError::PreconditionFailed`] — a conditional write that lost
    /// its race. The one storage error the commit / mutation paths are
    /// allowed to treat as contention, so tests use it to drive a CAS-loss
    /// to its retry budget and check what the caller finally sees.
    Precondition,
    /// [`StorageError::PermissionDenied`] — the backend refused the
    /// credentials in use. Neither a retry nor a reissue helps, so tests use
    /// it to check that the condition survives every wrapper it passes
    /// through instead of arriving as a generic fault.
    PermissionDenied,
}

/// One armed fault: the next `remaining` calls of `op` whose URI
/// contains `uri_fragment` fail with `kind`.
#[derive(Debug)]
struct FaultRule {
    op: FaultOp,
    kind: FaultKind,
    uri_fragment: String,
    remaining: usize,
}

/// See the module docs. Construct with [`FaultStorage::wrap`], arm with
/// [`FaultStorage::fail`], and pass the `Arc` anywhere a
/// `Arc<dyn StorageProvider>` goes.
#[derive(Debug)]
pub struct FaultStorage {
    inner: Arc<dyn StorageProvider>,
    rules: Mutex<Vec<FaultRule>>,
    fired: AtomicUsize,
}

impl FaultStorage {
    pub fn wrap(inner: Arc<dyn StorageProvider>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            rules: Mutex::new(Vec::new()),
            fired: AtomicUsize::new(0),
        })
    }

    /// Arm a rule: the next `times` calls of `op` whose URI **contains**
    /// `uri_fragment` fail with [`StorageError::TransientExhausted`].
    ///
    /// Substring matching is deliberate — it lets a rule arm a whole
    /// namespace (`"data/"`, a hidden-index prefix) as easily as one
    /// object. Rules meant for a single object should pass its full path;
    /// the repo's numbered object names are fixed-width (zero-padded), so
    /// a full path never substring-matches a sibling.
    pub fn fail(&self, op: FaultOp, uri_fragment: &str, times: usize) {
        self.fail_with(FaultKind::Transient, op, uri_fragment, times);
    }

    /// Arm a rule that fails with `kind`. Same matching rules as
    /// [`Self::fail`]; use [`FaultKind::Precondition`] to model a peer
    /// writer winning the CAS instead of a broken store.
    pub fn fail_with(&self, kind: FaultKind, op: FaultOp, uri_fragment: &str, times: usize) {
        self.rules_guard().push(FaultRule {
            op,
            kind,
            uri_fragment: uri_fragment.to_string(),
            remaining: times,
        });
    }

    /// Disarm every remaining rule.
    pub fn clear(&self) {
        self.rules_guard().clear();
    }

    /// Total faults fired since construction — lets a test assert the
    /// failure it observed actually came from the injection.
    pub fn fired(&self) -> usize {
        self.fired.load(Ordering::SeqCst)
    }

    /// The rules table, recovering from mutex poisoning: the helper's
    /// contract is failures-as-clean-errors, so a panic elsewhere must not
    /// turn every later storage call into a second panic. The state is
    /// safe to reuse — mutations under the lock are single-field writes.
    fn rules_guard(&self) -> MutexGuard<'_, Vec<FaultRule>> {
        match self.rules.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn check(&self, op: FaultOp, uri: &str) -> Result<(), StorageError> {
        let mut rules = self.rules_guard();
        for rule in rules.iter_mut() {
            if rule.op == op && rule.remaining > 0 && uri.contains(&rule.uri_fragment) {
                rule.remaining -= 1;
                self.fired.fetch_add(1, Ordering::SeqCst);
                return Err(match rule.kind {
                    FaultKind::Transient => StorageError::TransientExhausted {
                        uri: uri.to_string(),
                        source: Box::new(IoError::other("injected fault")),
                    },
                    FaultKind::Precondition => StorageError::PreconditionFailed {
                        uri: uri.to_string(),
                    },
                    FaultKind::PermissionDenied => StorageError::PermissionDenied {
                        uri: uri.to_string(),
                    },
                });
            }
        }
        Ok(())
    }
}

#[async_trait]
impl StorageProvider for FaultStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.check(FaultOp::Head, uri)?;
        self.inner.head(uri).await
    }
    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.check(FaultOp::Get, uri)?;
        self.inner.get(uri).await
    }
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.check(FaultOp::GetRange, uri)?;
        self.inner.get_range(uri, range).await
    }
    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.check(FaultOp::PutAtomic, uri)?;
        self.inner.put_atomic(uri, bytes).await
    }
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        self.check(FaultOp::PutIfMatch, uri)?;
        self.inner.put_if_match(uri, bytes, expected_etag).await
    }
    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.inner.put_multipart(uri).await
    }
    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.check(FaultOp::Delete, uri)?;
        self.inner.delete(uri).await
    }
    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        self.inner.list_with_prefix_metadata(prefix).await
    }
    fn object_store_handle(
        &self,
        _uri: &str,
    ) -> Option<(Arc<dyn object_store::ObjectStore>, object_store::path::Path)> {
        // Deliberately withheld: the raw handle would let parquet/SQL
        // readers issue range GETs that bypass the fault rules entirely.
        // `None` routes those callers onto the whole-object fallback,
        // which flows through the checked methods above.
        None
    }
    fn usage_meter(&self) -> Arc<UsageMeter> {
        self.inner.usage_meter()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::storage::LocalFsStorageProvider;

    /// Unmatched operations pass through untouched; matched rules burn
    /// down; `clear` disarms; the passthrough surface (multipart handle,
    /// withheld object-store handle, delegated meter) behaves as
    /// documented.
    #[tokio::test]
    async fn rules_burn_down_and_passthroughs_stay_transparent() {
        let dir = TempDir::new().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local"));
        let faults = FaultStorage::wrap(local);

        faults
            .put_atomic("data/a.bin", Bytes::from_static(b"abc"))
            .await
            .expect("no rules armed");

        faults.fail(FaultOp::Get, "data/", 1);
        assert!(faults.get("data/a.bin").await.is_err(), "armed rule fires");
        assert_eq!(faults.fired(), 1);
        let (bytes, _) = faults.get("data/a.bin").await.expect("rule burned down");
        assert_eq!(bytes.as_ref(), b"abc");

        // The kind is per-rule: a precondition rule models a lost CAS, not
        // a broken store, and callers distinguish the two.
        faults.fail_with(FaultKind::Precondition, FaultOp::PutIfMatch, "data/", 1);
        let err = faults
            .put_if_match("data/a.bin", Bytes::from_static(b"xyz"), None)
            .await
            .expect_err("armed precondition rule fires");
        assert!(
            matches!(err, StorageError::PreconditionFailed { .. }),
            "expected PreconditionFailed, got {err:?}"
        );

        faults.fail(FaultOp::Get, "data/", 1);
        faults.clear();
        faults
            .get("data/a.bin")
            .await
            .expect("cleared rules never fire");
        assert_eq!(
            faults.fired(),
            2,
            "the get and the precondition rule fired, nothing since"
        );

        // Passthroughs: multipart reaches the inner provider, the raw
        // object-store handle is withheld, the meter is the inner one.
        let mut upload = faults.put_multipart("data/m.bin").await.expect("multipart");
        upload.abort().await.expect("abort");
        assert!(faults.object_store_handle("data/a.bin").is_none());
        let _ = faults.usage_meter();
        let listed = faults
            .list_with_prefix_metadata("data/")
            .await
            .expect("listing delegates");
        assert_eq!(listed.len(), 1, "only the seeded object");
    }
}
