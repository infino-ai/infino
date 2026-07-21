// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Bench facade over the engine [`UsageMeter`].
//!
//! There is **no** second counter implementation here. [`wrap`] returns a
//! handle to the provider's existing meter so phase windows can
//! `snapshot` / `since` without wrapping I/O a second time.

use std::sync::Arc;

pub use infino::storage::{ClassIo, N_URI_CLASSES, TraceEntry, UriClass};
use infino::storage::{StorageProvider, UsageMeter, UsageSnapshot};

use crate::rss;

/// Alias kept so cost / report call sites need not rename every use.
pub type ObjectStoreMeter = UsageSnapshot;

/// One cold consumer's metered windows, split at the phase boundaries the
/// cost model prices separately.
#[derive(Debug, Clone, Copy)]
pub struct ColdStoreSplit {
    pub open: ObjectStoreMeter,
    pub first_query: ObjectStoreMeter,
    pub second_query: ObjectStoreMeter,
    pub repeat_query: ObjectStoreMeter,
}

/// Handle to a storage provider and its engine usage meter.
pub struct MeteredStorage {
    provider: Arc<dyn StorageProvider>,
    meter: Arc<UsageMeter>,
}

/// Bind to the provider's engine meter (no wrapper, no second ledger).
pub fn wrap(storage: Arc<dyn StorageProvider>) -> MeteredStorage {
    let meter = storage.usage_meter();
    MeteredStorage {
        provider: storage,
        meter,
    }
}

impl MeteredStorage {
    pub fn provider(&self) -> Arc<dyn StorageProvider> {
        Arc::clone(&self.provider)
    }

    pub fn meter(&self) -> Arc<UsageMeter> {
        Arc::clone(&self.meter)
    }

    pub fn snapshot(&self) -> ObjectStoreMeter {
        self.meter.snapshot()
    }

    pub fn start_trace(&self) {
        self.meter.start_trace();
    }

    pub fn take_trace(&self) -> Vec<TraceEntry> {
        self.meter.take_trace()
    }
}

/// Background-fill GETs reshaped as a foreground meter for formatters.
pub fn background_fill_meter(snap: &ObjectStoreMeter) -> ObjectStoreMeter {
    ObjectStoreMeter {
        get_count: snap.bg_get_count,
        get_bytes: snap.bg_get_bytes,
        ..Default::default()
    }
}

/// Merge background-fill GETs from two windows.
pub fn merge_background_fill(a: &ObjectStoreMeter, b: &ObjectStoreMeter) -> ObjectStoreMeter {
    ObjectStoreMeter {
        get_count: a.bg_get_count.saturating_add(b.bg_get_count),
        get_bytes: a.bg_get_bytes.saturating_add(b.bg_get_bytes),
        ..Default::default()
    }
}

/// Human breakdown of non-zero URI classes (bench report formatting).
pub fn fmt_get_class_breakdown(snap: &ObjectStoreMeter) -> String {
    let parts: Vec<String> = snap
        .get_by_class
        .iter()
        .enumerate()
        .filter(|(_, c)| c.get_count > 0)
        .map(|(i, c)| {
            format!(
                "{} {} GET ({})",
                UriClass::from_index(i).label(),
                c.get_count,
                rss::fmt_bytes(c.get_bytes),
            )
        })
        .collect();
    if parts.is_empty() {
        "0 GET".into()
    } else {
        parts.join(" · ")
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use infino::storage::LocalFsStorageProvider;
    use object_store::ObjectStoreExt;

    use super::*;

    #[tokio::test]
    async fn wrap_reads_engine_meter_including_object_store_handle() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        // Fresh meter — do not share the process-default ledger with
        // unrelated concurrent tests.
        let ledger = Arc::new(UsageMeter::new());
        let provider: Arc<dyn StorageProvider> = Arc::new(
            LocalFsStorageProvider::new_with_meter(dir.path(), Arc::clone(&ledger))
                .expect("localfs"),
        );
        provider
            .put_atomic("seg/x.bin", Bytes::from_static(b"0123456789"))
            .await
            .expect("put");

        let meter = wrap(Arc::clone(&provider));
        let before = meter.snapshot();
        let (store, path) = meter
            .provider()
            .object_store_handle("seg/x.bin")
            .expect("handle");
        let _ = store.get(&path).await.expect("get");
        let delta = meter.snapshot().since(&before);
        assert_eq!(delta.get_count, 1);
        assert_eq!(delta.get_bytes, 10);
    }
}
