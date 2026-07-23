// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Connection-scoped object-store usage meter — the **sole** I/O ledger.
//!
//! Providers and [`super::counting`] wrappers record here. Benches and the
//! usage flush read [`UsageMeter::snapshot`]; they must not keep a parallel
//! counter implementation.

use std::{
    array, fmt,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use serde::{Deserialize, Serialize};

use super::io_counters::io_is_background;

/// Path token of the hidden vector-index sibling's storage prefix
/// (`_infino_<uuid>_vector_index/...` under the table root).
const HIDDEN_INDEX_PATH_TOKEN: &str = "_vector_index";
/// Manifest-namespace path tokens on either table.
const MANIFEST_PATH_TOKENS: [&str; 4] = [
    "_supertable/",
    "manifest/",
    "manifest-parts/",
    "slow-vector-state/",
];

/// Number of [`UriClass`] variants (array-indexed counters).
pub const N_URI_CLASSES: usize = 4;

/// Which table + namespace a request URI belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UriClass {
    UserData,
    UserManifest,
    HiddenData,
    HiddenManifest,
}

impl UriClass {
    pub fn of(uri: &str) -> Self {
        let hidden = uri.contains(HIDDEN_INDEX_PATH_TOKEN);
        let manifest = MANIFEST_PATH_TOKENS.iter().any(|t| uri.contains(t));
        match (hidden, manifest) {
            (true, true) => Self::HiddenManifest,
            (true, false) => Self::HiddenData,
            (false, true) => Self::UserManifest,
            (false, false) => Self::UserData,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::UserData => "user data",
            Self::UserManifest => "user manifest",
            Self::HiddenData => "hidden data",
            Self::HiddenManifest => "hidden manifest",
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::UserData => 0,
            Self::UserManifest => 1,
            Self::HiddenData => 2,
            Self::HiddenManifest => 3,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::UserData,
            1 => Self::UserManifest,
            2 => Self::HiddenData,
            _ => Self::HiddenManifest,
        }
    }

    pub fn is_hidden(self) -> bool {
        matches!(self, Self::HiddenData | Self::HiddenManifest)
    }
}

/// Per-[`UriClass`] GET counters inside one metering window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassIo {
    pub get_count: u64,
    pub get_bytes: u64,
}

/// One traced read request while a trace window is active.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub uri: String,
    pub range: Option<(u64, u64)>,
    pub bytes: u64,
}

/// Request + byte counts observed in one metering window (or cumulative).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageSnapshot {
    pub head_count: u64,
    pub get_count: u64,
    pub get_bytes: u64,
    pub bg_get_count: u64,
    pub bg_get_bytes: u64,
    pub put_count: u64,
    pub put_bytes: u64,
    pub list_count: u64,
    pub delete_count: u64,
    pub get_by_class: [ClassIo; N_URI_CLASSES],
}

impl UsageSnapshot {
    /// Counts accrued since `earlier` (saturating).
    pub fn since(&self, earlier: &UsageSnapshot) -> UsageSnapshot {
        let mut get_by_class = [ClassIo::default(); N_URI_CLASSES];
        for (i, slot) in get_by_class.iter_mut().enumerate() {
            slot.get_count = self.get_by_class[i]
                .get_count
                .saturating_sub(earlier.get_by_class[i].get_count);
            slot.get_bytes = self.get_by_class[i]
                .get_bytes
                .saturating_sub(earlier.get_by_class[i].get_bytes);
        }
        UsageSnapshot {
            head_count: self.head_count.saturating_sub(earlier.head_count),
            get_count: self.get_count.saturating_sub(earlier.get_count),
            get_bytes: self.get_bytes.saturating_sub(earlier.get_bytes),
            bg_get_count: self.bg_get_count.saturating_sub(earlier.bg_get_count),
            bg_get_bytes: self.bg_get_bytes.saturating_sub(earlier.bg_get_bytes),
            put_count: self.put_count.saturating_sub(earlier.put_count),
            put_bytes: self.put_bytes.saturating_sub(earlier.put_bytes),
            list_count: self.list_count.saturating_sub(earlier.list_count),
            delete_count: self.delete_count.saturating_sub(earlier.delete_count),
            get_by_class,
        }
    }

    /// Read-class requests (HEAD + foreground GET).
    pub fn read_requests(&self) -> u64 {
        self.head_count + self.get_count
    }

    pub fn class_io(&self, class: UriClass) -> ClassIo {
        self.get_by_class[class.index()]
    }

    /// Hidden-table GET count (data + manifest classes).
    #[cfg(test)]
    pub(crate) fn hidden_get_count(&self) -> u64 {
        self.class_io(UriClass::HiddenData).get_count
            + self.class_io(UriClass::HiddenManifest).get_count
    }

    /// Hidden-table GET bytes (data + manifest classes).
    #[cfg(test)]
    pub(crate) fn hidden_get_bytes(&self) -> u64 {
        self.class_io(UriClass::HiddenData).get_bytes
            + self.class_io(UriClass::HiddenManifest).get_bytes
    }

    /// True when every counter is zero.
    pub fn is_zero(&self) -> bool {
        *self == UsageSnapshot::default()
    }
}

/// Process-wide default meter for providers constructed outside a
/// [`crate::catalog::Connection`] (unit tests, ad-hoc LocalFs). Connection
/// paths inject their own [`Arc<UsageMeter>`] instead.
fn process_default_meter() -> Arc<UsageMeter> {
    static METER: OnceLock<Arc<UsageMeter>> = OnceLock::new();
    Arc::clone(METER.get_or_init(UsageMeter::new))
}

/// Connection-scoped (or process-default) usage ledger.
pub struct UsageMeter {
    head_count: AtomicU64,
    get_count: AtomicU64,
    get_bytes: AtomicU64,
    bg_get_count: AtomicU64,
    bg_get_bytes: AtomicU64,
    put_count: AtomicU64,
    put_bytes: AtomicU64,
    list_count: AtomicU64,
    delete_count: AtomicU64,
    class_get_count: [AtomicU64; N_URI_CLASSES],
    class_get_bytes: [AtomicU64; N_URI_CLASSES],
    trace: Mutex<Option<Vec<TraceEntry>>>,
}

impl Default for UsageMeter {
    fn default() -> Self {
        Self {
            head_count: AtomicU64::new(0),
            get_count: AtomicU64::new(0),
            get_bytes: AtomicU64::new(0),
            bg_get_count: AtomicU64::new(0),
            bg_get_bytes: AtomicU64::new(0),
            put_count: AtomicU64::new(0),
            put_bytes: AtomicU64::new(0),
            list_count: AtomicU64::new(0),
            delete_count: AtomicU64::new(0),
            class_get_count: array::from_fn(|_| AtomicU64::new(0)),
            class_get_bytes: array::from_fn(|_| AtomicU64::new(0)),
            trace: Mutex::new(None),
        }
    }
}

impl fmt::Debug for UsageMeter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UsageMeter")
            .field("snapshot", &self.snapshot())
            .finish()
    }
}

impl UsageMeter {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Meter used when a provider is built without an injected connection meter.
    pub fn process_default() -> Arc<Self> {
        process_default_meter()
    }

    pub fn snapshot(&self) -> UsageSnapshot {
        let mut get_by_class = [ClassIo::default(); N_URI_CLASSES];
        for (i, slot) in get_by_class.iter_mut().enumerate() {
            slot.get_count = self.class_get_count[i].load(Ordering::Relaxed);
            slot.get_bytes = self.class_get_bytes[i].load(Ordering::Relaxed);
        }
        UsageSnapshot {
            head_count: self.head_count.load(Ordering::Relaxed),
            get_count: self.get_count.load(Ordering::Relaxed),
            get_bytes: self.get_bytes.load(Ordering::Relaxed),
            bg_get_count: self.bg_get_count.load(Ordering::Relaxed),
            bg_get_bytes: self.bg_get_bytes.load(Ordering::Relaxed),
            put_count: self.put_count.load(Ordering::Relaxed),
            put_bytes: self.put_bytes.load(Ordering::Relaxed),
            list_count: self.list_count.load(Ordering::Relaxed),
            delete_count: self.delete_count.load(Ordering::Relaxed),
            get_by_class,
        }
    }

    pub fn record_head(&self) {
        self.head_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful GET / range / tail. `uri` drives [`UriClass`];
    /// background tasks (`io_counters::scope_background`) land in `bg_get_*`.
    pub fn record_get(&self, uri: &str, range: Option<(u64, u64)>, bytes: u64) {
        if io_is_background() {
            self.bg_get_count.fetch_add(1, Ordering::Relaxed);
            self.bg_get_bytes.fetch_add(bytes, Ordering::Relaxed);
            return;
        }
        self.get_count.fetch_add(1, Ordering::Relaxed);
        self.get_bytes.fetch_add(bytes, Ordering::Relaxed);
        let class = UriClass::of(uri).index();
        self.class_get_count[class].fetch_add(1, Ordering::Relaxed);
        self.class_get_bytes[class].fetch_add(bytes, Ordering::Relaxed);
        if let Ok(mut trace) = self.trace.lock()
            && let Some(entries) = trace.as_mut()
        {
            entries.push(TraceEntry {
                uri: uri.to_string(),
                range,
                bytes,
            });
        }
    }

    pub fn record_put(&self, bytes: u64) {
        self.put_count.fetch_add(1, Ordering::Relaxed);
        self.put_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_list(&self) {
        self.list_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_delete(&self) {
        self.delete_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Swap-out get-family counters for drain diagnostics (legacy `take` shape).
    /// Returns `(get_count, get_bytes, hidden_get_count, hidden_get_bytes)`.
    pub fn take_gets(&self) -> (u64, u64, u64, u64) {
        let get_count = self.get_count.swap(0, Ordering::Relaxed);
        let get_bytes = self.get_bytes.swap(0, Ordering::Relaxed);
        let mut hidden_count = 0u64;
        let mut hidden_bytes = 0u64;
        for i in [
            UriClass::HiddenData.index(),
            UriClass::HiddenManifest.index(),
        ] {
            hidden_count += self.class_get_count[i].swap(0, Ordering::Relaxed);
            hidden_bytes += self.class_get_bytes[i].swap(0, Ordering::Relaxed);
        }
        // Also clear user-class slots so class totals stay consistent with get_*.
        for i in [UriClass::UserData.index(), UriClass::UserManifest.index()] {
            let _ = self.class_get_count[i].swap(0, Ordering::Relaxed);
            let _ = self.class_get_bytes[i].swap(0, Ordering::Relaxed);
        }
        (get_count, get_bytes, hidden_count, hidden_bytes)
    }

    pub fn start_trace(&self) {
        if let Ok(mut t) = self.trace.lock() {
            *t = Some(Vec::new());
        }
    }

    pub fn take_trace(&self) -> Vec<TraceEntry> {
        self.trace
            .lock()
            .ok()
            .and_then(|mut t| t.take())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::io_counters::scope_background;

    #[test]
    fn uri_class_covers_user_and_hidden() {
        assert_eq!(UriClass::of("superfiles/ab12.parquet"), UriClass::UserData);
        assert_eq!(UriClass::of("_supertable/current"), UriClass::UserManifest);
        let hidden = "_infino_0000-uuid_vector_index/";
        assert_eq!(
            UriClass::of(&format!("{hidden}superfiles/cd34.parquet")),
            UriClass::HiddenData
        );
        assert_eq!(
            UriClass::of(&format!("{hidden}_supertable/current")),
            UriClass::HiddenManifest
        );
    }

    #[test]
    fn since_subtracts_fieldwise() {
        let mut earlier = UsageSnapshot {
            get_count: 10,
            get_bytes: 100,
            ..Default::default()
        };
        earlier.get_by_class[UriClass::HiddenData.index()] = ClassIo {
            get_count: 4,
            get_bytes: 40,
        };
        let mut later = UsageSnapshot {
            get_count: 25,
            get_bytes: 400,
            ..Default::default()
        };
        later.get_by_class[UriClass::HiddenData.index()] = ClassIo {
            get_count: 9,
            get_bytes: 140,
        };
        let delta = later.since(&earlier);
        assert_eq!(delta.get_count, 15);
        assert_eq!(delta.get_bytes, 300);
        assert_eq!(
            delta.get_by_class[UriClass::HiddenData.index()],
            ClassIo {
                get_count: 5,
                get_bytes: 100,
            }
        );
    }

    #[tokio::test]
    async fn background_gets_are_split() {
        let m = UsageMeter::new();
        m.record_get("superfiles/a.parquet", None, 10);
        scope_background(async {
            m.record_get("superfiles/b.parquet", Some((0, 5)), 5);
        })
        .await;
        let s = m.snapshot();
        assert_eq!(s.get_count, 1);
        assert_eq!(s.get_bytes, 10);
        assert_eq!(s.bg_get_count, 1);
        assert_eq!(s.bg_get_bytes, 5);
    }
}
