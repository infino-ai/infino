// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! WALs a handle left short of `Complete` after their rows were retired.
//!
//! A mutation whose rows are retired reports its outcome even when a step
//! that finishes it fails (see `TombstonePhaseOutcome::CompletionDeferred`),
//! and leaves its WAL for recovery. Until that WAL is finished a reader may
//! still see the retired rows, and the full recovery sweep runs only when a
//! handle opens — which a long-lived handle never does again. So the handle
//! that left the WAL finishes it itself: at the start of its next commit,
//! and from a bounded background retry started when the WAL is left.

use std::{
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use tokio::time::sleep;
use tracing::{debug, warn};

use crate::supertable::{
    handle::{Supertable, SupertableInner},
    wal::{
        lease::DEFAULT_LEASE_DURATION, persistence::WalStore, recovery::finish_one,
        state_doc::WalId,
    },
};

/// Wait before the background retry's first attempt. Long enough for a
/// lost race on a busy manifest to have passed; short next to how long a
/// reader would otherwise see the retired rows.
const FIRST_RETRY_DELAY: Duration = Duration::from_secs(5);

/// Attempts the background retry makes, each waiting twice as long as the
/// one before it: about five minutes in all from [`FIRST_RETRY_DELAY`].
/// What is still unfinished after them waits for the handle's next commit,
/// or for the sweep the next open runs.
const RETRY_ROUNDS: u32 = 6;

/// The WALs one handle left unfinished, and whether a background retry is
/// working through them.
#[derive(Debug, Default)]
pub(crate) struct UnfinishedWals {
    wal_ids: Mutex<Vec<WalId>>,
    retrying: AtomicBool,
}

impl UnfinishedWals {
    /// Remember `wal_id` as left unfinished by this handle.
    pub(crate) fn note(&self, wal_id: WalId) {
        let mut ids = self.lock();
        if !ids.contains(&wal_id) {
            ids.push(wal_id);
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// The WALs still unfinished, for tests.
    #[cfg(test)]
    pub(crate) fn pending(&self) -> Vec<WalId> {
        self.lock().clone()
    }

    /// Keep the background retry from starting, so a test can watch a WAL
    /// stay unfinished and finish it by another route.
    #[cfg(test)]
    pub(crate) fn hold_background_retry(&self) {
        self.retrying.store(true, Ordering::SeqCst);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<WalId>> {
        self.wal_ids.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Drive each WAL this handle left unfinished to `Complete` through the
/// recovery step for one WAL, keeping the ones that are not done yet.
/// Returns how many remain. Taken off the list while they are driven, so
/// two callers never drive the same WAL at once.
pub(in crate::supertable) async fn finish_unfinished(supertable: &Supertable) -> usize {
    let inner = supertable.inner();
    let Some(storage) = inner.options.storage.clone() else {
        return 0;
    };
    let taken = std::mem::take(&mut *inner.unfinished_wals.lock());
    if taken.is_empty() {
        return 0;
    }
    let wal_store = WalStore::new(storage);
    let mut left = Vec::new();
    for wal_id in taken {
        let done = finish_one(
            supertable,
            &wal_store,
            wal_id,
            inner.handle_id,
            DEFAULT_LEASE_DURATION,
        )
        .await;
        if !done {
            left.push(wal_id);
        }
    }
    let remaining = left.len();
    for wal_id in left {
        inner.unfinished_wals.note(wal_id);
    }
    remaining
}

/// Start the background retry over this handle's unfinished WALs, unless
/// there are none or one is already running. It holds the handle only
/// weakly between attempts, so it never keeps a dropped handle alive.
pub(in crate::supertable) fn retry_in_background(inner: &Arc<SupertableInner>) {
    let unfinished = &inner.unfinished_wals;
    if unfinished.is_empty() || unfinished.retrying.swap(true, Ordering::SeqCst) {
        return;
    }
    let handle = Arc::downgrade(inner);
    inner.query_runtime().spawn(async move {
        let mut delay = FIRST_RETRY_DELAY;
        for _ in 0..RETRY_ROUNDS {
            sleep(delay).await;
            delay = delay.saturating_mul(2);
            let Some(inner) = handle.upgrade() else {
                return;
            };
            let remaining = finish_unfinished(&Supertable::from_inner(inner)).await;
            if remaining == 0 {
                break;
            }
            debug!(remaining, "unfinished mutation WALs remain; retrying later");
        }
        if let Some(inner) = handle.upgrade() {
            if !inner.unfinished_wals.is_empty() {
                warn!(
                    "mutation WALs this handle left unfinished outlasted its background retry; \
                     its next commit, or the next open's recovery sweep, finishes them"
                );
            }
            inner
                .unfinished_wals
                .retrying
                .store(false, Ordering::SeqCst);
        }
    });
}
