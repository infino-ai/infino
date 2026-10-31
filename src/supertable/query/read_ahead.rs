// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Read-ahead for SQL scans of superfiles that are not local yet.
//!
//! The Parquet reader asks for one row group's columns, decodes them, and only
//! then asks for the next row group. On a cold superfile every ask is a GET to
//! object storage, so the scan waits on one GET at a time while the CPU sits
//! idle. Read-ahead downloads the next row groups while the scan decodes:
//!
//! ```text
//!   without:  fetch 0 | decode 0 | fetch 1 | decode 1 | fetch 2 | decode 2 | ...
//!   with:     fetch 0 | decode 0 | fetch 1 | decode 1 | decode 2 | decode 3 | ...
//!                                           └fetch 2,3┘ └────fetch 4,5────┘
//! ```
//!
//! **It downloads only what the scan would download itself, just earlier.**
//!
//! # Which row groups
//!
//! DataFusion prunes a file's row groups (by the partition's byte range, the
//! row-group statistics and the page index) only after it creates the reader,
//! so the reader repeats that pruning first. A row group the scan reads only
//! in part (deleted rows, pages the page index rules out) is left to the scan.
//!
//! # When it starts and stops
//!
//! - It starts once the scan reads a second planned row group in order, and
//!   that read took most of each column (see [`MIN_CHUNK_READ_PERCENT`]).
//! - It stops for good when the scan reads out of order, keeping only the
//!   batch the scan is in. At most two batches are wasted.
//! - It never starts under a `LIMIT` or a top-k (see [`ReadAheadOff`]), or
//!   while every read finds its bytes already local.
//!
//! # How it downloads
//!
//! Two row groups at a time (a batch), and the next batch starts once the
//! scan reads from the last one. A batch merges byte ranges the way the scan
//! would (see [`fetch_spans`]), and keeps clear of bytes another read may be
//! downloading at the same moment, so no block is downloaded twice (see
//! [`leave_shared_ends`]).
//!
//! # Memory
//!
//! Downloaded bytes are held here until the scan reads them, reserved from the
//! connection's memory budget: at most half of it ([`BUDGET_SHARE_PERCENT`]),
//! 16 MiB per reader (one file in one partition) and 256 MiB per scan. A batch
//! is charged what the cache downloads for it, which can be more than the
//! scan's bytes. It also reserves room in the disk cache's write queue first,
//! so the cache keeps what it downloads.
//!
//! When anything else on the connection needs memory the budget can't give, a
//! join's build side say, read-ahead gives back everything it holds and stays
//! off for the rest of the scan (see [`Reclaim`]). The scan then reads those
//! row groups itself, usually from the disk cache.
//!
//! # Counting
//!
//! Read-ahead's downloads are not counted as page reads, and the scan's reads
//! are counted as usual, so `sql_page_bytes` is the same with or without it.
//!
//! [`ReadAheadOff`]: crate::supertable::query::read_ahead_rule::ReadAheadOff

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, mem,
    ops::Range,
    panic::resume_unwind,
    sync::{
        Arc, Mutex, MutexGuard, Once, OnceLock, PoisonError, Weak,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use arrow_schema::SchemaRef;
use bytes::Bytes;
use datafusion::{
    datasource::{
        listing::PartitionedFile,
        physical_plan::parquet::{
            PagePruningAccessPlanFilter, ParquetAccessPlan, ParquetFileMetrics, RowGroupAccess,
            RowGroupAccessPlanFilter, apply_file_schema_type_coercions,
        },
    },
    object_store::path::Path as ObjPath,
    physical_expr::{PhysicalExpr, PhysicalExprSimplifier},
    physical_expr_adapter::{DefaultPhysicalExprAdapterFactory, PhysicalExprAdapterFactory},
    physical_optimizer::pruning::PruningPredicate,
    physical_plan::metrics::ExecutionPlanMetricsSet,
};
use futures::{FutureExt, future::BoxFuture};
use object_store::{ObjectStore, Result as OsResult};
use parquet::{
    arrow::{
        arrow_reader::ArrowReaderOptions,
        async_reader::{AsyncFileReader, ParquetObjectReader},
        parquet_to_arrow_schema,
    },
    errors::{ParquetError, Result as ParquetResult},
    file::metadata::{ParquetMetaData, RowGroupMetaData},
};
use tokio::{
    spawn,
    task::{AbortHandle, JoinHandle},
};
use tracing::Instrument;

use crate::{
    memory::{ConnectionMemoryBudget, Reclaim, Reservation},
    superfile::CacheBlocks,
    supertable::query::df_object_store::{SuperfileObjectStore, merged_ranges},
};

/// Fewest row groups a batch takes, when they fit the reader's cap.
const READ_AHEAD_MIN_ROW_GROUPS: usize = 2;

/// Batches a reader keeps in flight: one the scan decodes from and one
/// downloading, so each batch is sized to half the reader's cap.
const READ_AHEAD_BATCHES_IN_FLIGHT: usize = 2;

/// Most bytes one reader (one file in one partition) may hold. A batch
/// takes as many row groups as fit half of it, which on a file of small row
/// groups is what turns a round trip per row group into one per batch; a
/// row group bigger than half of it is read ahead on its own, as long as it
/// fits the whole. A row group bigger than this is left to the scan.
const READ_AHEAD_MAX_BYTES: usize = 16 << 20;

/// Most bytes all readers of one scan may hold, so a scan stays bounded even
/// when the memory budget has no limit.
const READ_AHEAD_SCAN_MAX_BYTES: usize = 256 << 20;

/// How much of each column, in percent, a row group's read must take before
/// read-ahead downloads the next row groups' columns whole. A scan the page
/// index narrows to a few pages doesn't read ahead.
const MIN_CHUNK_READ_PERCENT: u64 = 75;

/// Read-ahead uses at most this share of the connection's memory budget, and
/// leaves the rest to joins, sorts and appends, which can't do without it.
const BUDGET_SHARE_PERCENT: usize = 50;

/// Widest run of skipped pages inside one column chunk that the scan reads
/// through to fetch the cold kept pages on either side in one GET. The store
/// already reads through gaps up to 1 MiB (see [`merged_ranges`]); inside one
/// chunk this goes to 2 MiB, about what one GET's first byte wait
/// (about 25 ms) could move on one stream (about 60 MB/s). It saves GETs, which
/// are billed while in-region bytes are not, not time: the two GETs would have
/// run in parallel, and the one GET now streams the gap too.
pub(super) const PAGE_READ_THROUGH_BYTES: u64 = 2 << 20;

/// Shared by every reader of one scan: the store, the memory budget and the
/// scan's filter.
pub(crate) struct ScanReadAhead {
    store: Arc<SuperfileObjectStore>,
    budget: Arc<ConnectionMemoryBudget>,
    /// The scan's filter (including a join's filter that fills in while the
    /// join runs), and the adapter that maps it onto each file's schema. Set
    /// when the scan starts, see [`Self::set_predicate`]; unset without one.
    predicate: OnceLock<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExprAdapterFactory>)>,
    /// The table's schema, which the filter is written against.
    schema: SchemaRef,
    /// Turned off by [`ReadAheadOff`] when a `LIMIT` or top-k can stop the
    /// scan early.
    ///
    /// [`ReadAheadOff`]: crate::supertable::query::read_ahead_rule::ReadAheadOff
    enabled: AtomicBool,
    /// Bytes all readers of this scan hold, capped at
    /// [`READ_AHEAD_SCAN_MAX_BYTES`].
    held: AtomicUsize,
    /// Every reader's batches, so [`Reclaim::reclaim`] can give them back.
    batches: Mutex<Vec<Weak<Mutex<Held>>>>,
    /// Registers the scan with the budget on its first batch.
    registered: Once,
    /// Less than [`READ_AHEAD_MAX_BYTES`] for a test over a small file.
    #[cfg(test)]
    reader_cap_bytes: usize,
}

impl fmt::Debug for ScanReadAhead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanReadAhead")
            .field("enabled", &self.enabled.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl ScanReadAhead {
    /// Read-ahead for a scan over `schema`, on until [`ReadAheadOff`] turns it
    /// off.
    ///
    /// [`ReadAheadOff`]: crate::supertable::query::read_ahead_rule::ReadAheadOff
    pub(crate) fn new(
        store: Arc<SuperfileObjectStore>,
        budget: Arc<ConnectionMemoryBudget>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            store,
            budget,
            predicate: OnceLock::new(),
            schema,
            enabled: AtomicBool::new(true),
            held: AtomicUsize::new(0),
            batches: Mutex::new(Vec::new()),
            registered: Once::new(),
            #[cfg(test)]
            reader_cap_bytes: READ_AHEAD_MAX_BYTES,
        }
    }

    /// Most bytes one reader may hold: [`READ_AHEAD_MAX_BYTES`], or less
    /// under test.
    fn reader_cap(&self) -> usize {
        #[cfg(test)]
        {
            self.reader_cap_bytes
        }
        #[cfg(not(test))]
        {
            READ_AHEAD_MAX_BYTES
        }
    }

    /// The same scan with a smaller per-reader cap, so a test over a small
    /// file can pin how many row groups a batch takes.
    #[cfg(test)]
    pub(crate) fn with_reader_cap(mut self, bytes: usize) -> Self {
        self.reader_cap_bytes = bytes;
        self
    }

    /// Hand over the scan's filter and schema adapter (DataFusion's default if
    /// none), so readers prune row groups the way the scan does. The first
    /// call wins, since every partition runs the same plan.
    pub(crate) fn set_predicate(
        &self,
        predicate: Arc<dyn PhysicalExpr>,
        adapter: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    ) {
        let adapter = adapter.unwrap_or_else(|| Arc::new(DefaultPhysicalExprAdapterFactory));
        let _ = self.predicate.set((predicate, adapter));
    }

    /// The filter handed over, if any.
    #[cfg(test)]
    pub(crate) fn predicate(&self) -> Option<&Arc<dyn PhysicalExpr>> {
        self.predicate.get().map(|(predicate, _)| predicate)
    }

    /// Turn read-ahead off: readers created from now on never read ahead, and
    /// readers already reading start no more batches.
    pub(crate) fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Whether readers created now read ahead.
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// A reader for a file whose footer wasn't parsed at open. It reads the
    /// footer itself, like DataFusion's own reader, and never reads ahead.
    pub(crate) fn plain_reader(
        &self,
        file: &PartitionedFile,
        footer_size_hint: Option<usize>,
    ) -> ParquetObjectReader {
        let store: Arc<dyn ObjectStore> = Arc::clone(&self.store) as Arc<dyn ObjectStore>;
        let reader = ParquetObjectReader::new(store, file.object_meta.location.clone())
            .with_file_size(file.object_meta.size);
        match footer_size_hint {
            Some(hint) => reader.with_footer_size_hint(hint),
            None => reader,
        }
    }

    /// The reader for `file` in one partition, using the footer parsed at
    /// open. `file` carries the access plan and the partition's byte range.
    pub(crate) fn reader(
        self: &Arc<Self>,
        file: PartitionedFile,
        meta: Arc<ParquetMetaData>,
    ) -> ReadAhead {
        ReadAhead {
            scan: Arc::clone(self),
            file,
            meta,
            order: None,
            state: if self.is_enabled() {
                State::Unplanned(None)
            } else {
                State::Off
            },
            batches: Vec::new(),
            columns: BTreeSet::new(),
        }
    }

    /// The row-group pruning for one file ([`file_pruning`] of the filter
    /// handed over). `None` without a filter, or if it doesn't convert. Then
    /// the scan may skip row groups this reader planned, and read-ahead turns
    /// off when the reads go out of order.
    fn pruning(&self, meta: &ParquetMetaData) -> Option<(PruningPredicate, SchemaRef)> {
        let (predicate, adapter) = self.predicate.get()?;
        file_pruning(predicate, adapter, &self.schema, meta)
    }

    /// Reserve `bytes`, or `None` if that takes the budget past
    /// [`BUDGET_SHARE_PERCENT`] or the scan past [`READ_AHEAD_SCAN_MAX_BYTES`].
    /// The share is checked before and after reserving, so readers racing for
    /// the last of it can't all get through.
    fn reserve(self: &Arc<Self>, bytes: usize) -> Option<Hold> {
        self.registered.call_once(|| {
            let scan: Weak<ScanReadAhead> = Arc::downgrade(self);
            self.budget.register_reclaim(scan);
        });
        let over_share = |extra: usize| {
            self.budget.limit().is_some_and(|limit| {
                self.budget.used().saturating_add(extra).saturating_mul(100)
                    > limit.saturating_mul(BUDGET_SHARE_PERCENT)
            })
        };
        if over_share(bytes) {
            return None;
        }
        let held = Hold {
            scan: Arc::clone(self),
            bytes,
            _reservation: self.budget.try_reserve_without_reclaim(bytes).ok()?,
        };
        self.held.fetch_add(bytes, Ordering::Relaxed);
        let over_scan_cap = self.held.load(Ordering::Relaxed) > READ_AHEAD_SCAN_MAX_BYTES;
        (!over_scan_cap && !over_share(0)).then_some(held)
    }

    /// Track a reader's batch so [`Reclaim::reclaim`] can give it back, or give
    /// it back now if read-ahead was turned off while it was being set up.
    fn track(&self, held: &Arc<Mutex<Held>>) {
        let mut batches = self.batches.lock().unwrap_or_else(PoisonError::into_inner);
        if self.is_enabled() {
            batches.retain(|batch| batch.strong_count() > 0);
            batches.push(Arc::downgrade(held));
        } else {
            lock(held).give_back();
        }
    }
}

impl Reclaim for ScanReadAhead {
    /// Turn read-ahead off and give back what every reader holds; each reader
    /// then reads for itself. Off for the rest of the scan, so a join that
    /// grows step by step doesn't race read-ahead for the memory.
    fn reclaim(&self) {
        self.disable();
        let batches = mem::take(&mut *self.batches.lock().unwrap_or_else(PoisonError::into_inner));
        for held in batches.iter().filter_map(Weak::upgrade) {
            lock(&held).give_back();
        }
    }
}

/// Bytes held ahead: a budget reservation and the scan's count, both given
/// back on drop.
struct Hold {
    scan: Arc<ScanReadAhead>,
    bytes: usize,
    _reservation: Reservation,
}

impl Drop for Hold {
    fn drop(&mut self) {
        self.scan.held.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// Where a reader is in its file; each read moves it on:
///
/// ```text
///   Unplanned(None) ──any read──▶ Unplanned(that offset)
///
///   Unplanned(at):  every range of this read is local, nothing planned yet ──▶ Unplanned(this offset)
///                   bytes outside every row group ──────────────────────────▶ stay
///                   `at` was outside every row group, this read is inside ──▶ Unplanned(this offset)
///                   the same row group as `at` ─────────────────────────────▶ stay
///                   the planned row group after `at`'s ─────────────────────▶ InOrder { last: it }, look ahead
///                   any other row group ────────────────────────────────────▶ Off
///
///   InOrder:        bytes outside every row group ──────────────────────────▶ stay
///                   row group `last` again ─────────────────────────────────▶ stay
///                   the next planned row group ─────────────────────────────▶ last = it, serve it, look ahead
///                   any other row group ────────────────────────────────────▶ serve it if held, drop the rest, Off
///
///   Off:            terminal. Also where a reader under a LIMIT starts.
/// ```
///
/// "Look ahead" happens only if the read took most of each column (see
/// [`MIN_CHUNK_READ_PERCENT`]) and no batch is still downloading.
#[derive(Clone, Copy)]
enum State {
    /// Not reading in the planned order yet. Holds the offset to compare the
    /// next read with: the first read inside a row group, or the latest read
    /// while every read was local. `None` before any read.
    Unplanned(Option<u64>),
    /// Reading in the planned order; `last` is the row group read last.
    InOrder {
        last: usize,
    },
    Off,
}

/// The row groups this partition reads, and how.
struct Order {
    /// Every row group's byte span in the file, in file order (not only this
    /// partition's), so a binary search finds an offset's row group.
    spans: Vec<Range<u64>>,
    /// The access plan after repeating DataFusion's pruning.
    plan: ParquetAccessPlan,
    /// Per row group, whether another partition reads it (it survives the
    /// same pruning but is outside this partition's byte range).
    elsewhere: Vec<bool>,
}

impl Order {
    /// The row group holding `offset`, or `None` outside every row group (the
    /// footer, page indexes, bloom filters).
    fn row_group_of(&self, offset: u64) -> Option<usize> {
        let i = self.spans.partition_point(|span| span.end <= offset);
        self.spans
            .get(i)
            .filter(|span| span.contains(&offset))
            .map(|_| i)
    }

    /// Planned row groups after `row_group`, in order.
    fn after(&self, row_group: usize) -> impl Iterator<Item = usize> + '_ {
        (row_group + 1..self.plan.len()).filter(|&rg| self.plan.should_scan(rg))
    }

    /// Whether the scan reads `row_group` whole. A page-index selection that
    /// rules out no page still counts as whole.
    fn reads_whole(&self, row_group: usize, rows: usize) -> bool {
        match &self.plan.inner()[row_group] {
            RowGroupAccess::Scan => true,
            RowGroupAccess::Selection(selection) => selection.row_count() == rows,
            RowGroupAccess::Skip => false,
        }
    }
}

/// The next planned row groups that aren't all local, downloaded together.
/// Dropping it cancels the download and gives its bytes back to the budget
/// (dropping a tokio `JoinHandle` alone would leave the download running).
struct Batch {
    /// Each row group and its column byte ranges.
    row_groups: Vec<(usize, Vec<Range<u64>>)>,
    /// What the download fetches, in order (see [`fetch_spans`]).
    spans: Vec<Range<u64>>,
    /// The download, until the scan has waited for it.
    task: Option<JoinHandle<OsResult<Vec<Bytes>>>>,
    /// Which of `row_groups` the scan has read.
    served: Vec<bool>,
    /// Bytes of `spans`, counted against the reader's cap.
    charge: usize,
    /// What the batch holds, shared with the scan so it can give it back.
    held: Arc<Mutex<Held>>,
}

/// A batch's bytes and the memory reserved for them. The scan can empty it
/// from any thread when the connection needs the memory (see
/// [`Reclaim::reclaim`]); the reader then reads those row groups itself.
struct Held {
    /// The bytes of the batch's spans. `None` while downloading, or if the
    /// download failed or was given back.
    bytes: Option<Vec<Bytes>>,
    /// The memory reserved; `None` once given back.
    hold: Option<Hold>,
    /// Cancels the download.
    download: AbortHandle,
}

impl Held {
    /// Cancel the download, and drop the bytes and their reservation.
    fn give_back(&mut self) {
        self.download.abort();
        self.bytes = None;
        self.hold = None;
    }
}

fn lock(held: &Mutex<Held>) -> MutexGuard<'_, Held> {
    held.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Batch {
    fn position(&self, row_group: usize) -> Option<usize> {
        self.row_groups.iter().position(|(rg, _)| *rg == row_group)
    }

    fn is_running(&self) -> bool {
        self.task.is_some()
    }

    /// Whether the batch has bytes to serve: downloaded, and not given back.
    fn has_bytes(&self) -> bool {
        lock(&self.held).bytes.is_some()
    }

    /// Wait for the download, if it's still running. Bytes given back while
    /// it ran stay given back.
    async fn settle(&mut self) {
        if let Some(task) = &mut self.task {
            let fetched = match task.await {
                Ok(Ok(spans)) => Some(spans),
                Err(e) if e.is_panic() => resume_unwind(e.into_panic()),
                Ok(Err(_)) | Err(_) => None,
            };
            self.task = None;
            let mut held = lock(&self.held);
            if held.hold.is_some() {
                held.bytes = fetched;
            }
        }
    }

    /// `range` split, in order, into bytes the batch holds and the gaps it
    /// doesn't.
    fn pieces(&self, range: &Range<u64>) -> Vec<Piece> {
        let mut pieces = Vec::new();
        let mut at = range.start;
        if let Some(fetched) = &lock(&self.held).bytes {
            for (span, bytes) in self.spans.iter().zip(fetched) {
                let (start, end) = (
                    at.max(span.start),
                    range.end.min(span.start + bytes.len() as u64),
                );
                if start >= end {
                    continue;
                }
                if at < start {
                    pieces.push(Piece::Missing(at..start));
                }
                let slice = (start - span.start) as usize..(end - span.start) as usize;
                pieces.push(Piece::Held(bytes.slice(slice)));
                at = end;
            }
        }
        if at < range.end {
            pieces.push(Piece::Missing(at..range.end));
        }
        pieces
    }
}

/// Part of a requested range: bytes the batch holds, or a gap it doesn't.
enum Piece {
    Held(Bytes),
    Missing(Range<u64>),
}

impl Drop for Batch {
    fn drop(&mut self) {
        lock(&self.held).download.abort();
    }
}

/// How a batch downloads its row groups:
/// - each row group's byte ranges merge the way the scan would merge them
///   reading it alone, so nothing extra is downloaded;
/// - across row groups, spans that touch become one span;
/// - two spans whose cache downloads could overlap go in one cluster and
///   download one after the other, since two downloads running at once each
///   download the bytes they share;
/// - every other span starts its own cluster, and clusters download in
///   parallel.
///
/// Only an overlap puts two spans in order: ordering every span would
/// download a batch one span at a time, no faster than the scan itself.
fn fetch_spans(
    row_groups: &[(usize, Vec<Range<u64>>)],
    blocks: Option<CacheBlocks>,
) -> Vec<Vec<Range<u64>>> {
    let mut clusters: Vec<Vec<Range<u64>>> = Vec::new();
    for (_, ranges) in row_groups {
        for span in merged_ranges(ranges) {
            if let Some(cluster) = clusters.last_mut()
                && let Some(prev) = cluster.last_mut()
            {
                if span.start <= prev.end {
                    prev.end = prev.end.max(span.end);
                    continue;
                }
                if blocks.is_some_and(|blocks| blocks.reach(&span).start < blocks.reach(prev).end) {
                    cluster.push(span);
                    continue;
                }
            }
            clusters.push(vec![span]);
        }
    }
    clusters
}

/// Trim the batch so its ends stay clear of `others`, pieces another read may
/// be downloading at the same time. Only the ends can meet another read:
/// - before the batch, under a row filter the scan reads a row group's
///   columns in turns, so it can still be downloading the end of the row
///   group before the batch;
/// - after the batch, another partition can be reading the next row group.
///
/// ```text
///        others before          batch spans            others after
///   ...[ chunk ]---reach--->|  [  ]  [     ]  [ ]  |<-reach---[ chunk ]...
///                           ^ from                 ^ to
/// ```
///
/// The batch keeps only what lies between `from` and `to`, and drops an end
/// span whose own download would cross either line. The scan downloads what
/// is left out itself.
fn leave_shared_ends(
    mut clusters: Vec<Vec<Range<u64>>>,
    blocks: CacheBlocks,
    others: &[Range<u64>],
) -> Vec<Vec<Range<u64>>> {
    let (Some(start), Some(end)) = (
        clusters.first().and_then(|c| c.first()).map(|s| s.start),
        clusters.last().and_then(|c| c.last()).map(|s| s.end),
    ) else {
        return clusters;
    };
    let from = others
        .iter()
        .filter(|o| o.start < start)
        .map(|o| blocks.reach(o).end)
        .max();
    let to = others
        .iter()
        .filter(|o| o.start >= end)
        .map(|o| blocks.reach(o).start)
        .fold(u64::MAX, u64::min);
    let reaches_before =
        |span: &Range<u64>| from.is_some_and(|from| blocks.reach(span).start < from);
    for cluster in &mut clusters {
        for span in cluster.iter_mut() {
            *span = span.start.max(from.unwrap_or(0))..span.end.min(to);
        }
        cluster.retain(|span| span.start < span.end);
        while cluster.first().is_some_and(reaches_before) {
            cluster.remove(0);
        }
        while cluster
            .last()
            .is_some_and(|last| blocks.reach(last).end > to)
        {
            cluster.pop();
        }
    }
    clusters.retain(|cluster| !cluster.is_empty());
    clusters
}

/// The most bytes the cache could download for `clusters` on a cold file: the
/// union of the spans' [`CacheBlocks::reach`], or the sum of their
/// [`CacheBlocks::download_len`], whichever is smaller.
fn download_bytes(clusters: &[Vec<Range<u64>>], blocks: CacheBlocks) -> u64 {
    let spans = || clusters.iter().flatten();
    let mut reaches: Vec<Range<u64>> = spans().map(|s| blocks.reach(s)).collect();
    reaches.sort_by_key(|r| r.start);
    let (mut reached, mut covered) = (0, 0);
    for reach in reaches {
        reached += reach.end.saturating_sub(reach.start.max(covered));
        covered = covered.max(reach.end);
    }
    let windows = spans().map(|s| blocks.download_len(s)).sum();
    reached.min(windows)
}

/// The [`AsyncFileReader`] for one superfile in one scan partition. It serves
/// the scan's byte ranges, downloads the next row groups while the scan reads
/// in order, and returns the footer parsed at open. See the module docs.
pub(crate) struct ReadAhead {
    scan: Arc<ScanReadAhead>,
    /// The file as DataFusion handed it over: path, access plan, byte range.
    file: PartitionedFile,
    meta: Arc<ParquetMetaData>,
    /// Built on the first read that finds bytes not local yet.
    order: Option<Order>,
    state: State,
    /// Batches held, oldest first; at most one still downloading.
    batches: Vec<Batch>,
    /// The columns the scan has read since the order was built. Under a row
    /// filter that's more than one read's columns.
    columns: BTreeSet<usize>,
}

impl AsyncFileReader for ReadAhead {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        async move {
            self.read(vec![range])
                .await?
                .pop()
                .ok_or_else(|| ParquetError::General("read-ahead: no bytes for a range".into()))
        }
        .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
        self.read(ranges).boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<ParquetMetaData>>> {
        let meta = Arc::clone(&self.meta);
        async move { Ok(meta) }.boxed()
    }
}

impl ReadAhead {
    fn location(&self) -> &ObjPath {
        &self.file.object_meta.location
    }

    /// The bytes of `ranges`, in order: from a batch if one holds the row
    /// group, else downloaded now. If this read entered the next planned row
    /// group and took most of each column, it starts the next batch.
    async fn read(&mut self, ranges: Vec<Range<u64>>) -> ParquetResult<Vec<Bytes>> {
        let entered = self.advance(&ranges);
        if let Some(order) = &self.order {
            for range in &ranges {
                if let Some(column) = order
                    .row_group_of(range.start)
                    .and_then(|rg| column_of(self.meta.row_group(rg), range))
                {
                    self.columns.insert(column);
                }
            }
        }
        let held = entered.and_then(|rg| {
            let batch = self.batches.iter().position(|b| b.position(rg).is_some())?;
            Some((rg, batch))
        });
        let bytes = match held {
            Some((rg, batch)) => self.serve(rg, batch, &ranges).await?,
            None => self.fetch(&ranges).await?,
        };
        // Only a row group's first read starts a batch: under a row filter the
        // scan reads the filter's columns first and the rest after, and the
        // second read mustn't replace the batch the first one started. (Row
        // filters only run under a top-k today, where read-ahead is off.)
        if let Some(rg) = entered
            && matches!(self.state, State::InOrder { last } if last == rg)
            && let Some(columns) = self.mostly_read_columns(rg, &ranges)
        {
            self.start_ahead(rg, &columns);
        }
        Ok(bytes)
    }

    /// Move the state on for a read of `ranges` (see [`State`]). Returns the
    /// row group it enters: the next planned one, or the one it jumped to when
    /// it left the order, so a batch holding it can still serve it.
    fn advance(&mut self, ranges: &[Range<u64>]) -> Option<usize> {
        let offset = ranges.first()?.start;
        let (state, entered) = match self.state {
            State::Off => return None,
            State::Unplanned(None) => (State::Unplanned(Some(offset)), None),
            // Nothing is planned while every read is local: a warm file never
            // pays for planning, and a file cached only at the front starts
            // reading ahead at its first cold row group.
            State::Unplanned(Some(_))
                if self.order.is_none()
                    && ranges
                        .iter()
                        .all(|r| self.scan.store.is_local(self.location(), r)) =>
            {
                (State::Unplanned(Some(offset)), None)
            }
            State::Unplanned(Some(at)) => {
                let order = self.order();
                match (order.row_group_of(at), order.row_group_of(offset)) {
                    (_, None) => (State::Unplanned(Some(at)), None),
                    (None, Some(_)) => (State::Unplanned(Some(offset)), None),
                    (Some(was), Some(rg)) if rg == was => (State::Unplanned(Some(at)), None),
                    (Some(was), Some(rg)) if order.after(was).next() == Some(rg) => {
                        (State::InOrder { last: rg }, Some(rg))
                    }
                    (Some(_), Some(_)) => (State::Off, None),
                }
            }
            State::InOrder { last } => {
                let order = self.order();
                match order.row_group_of(offset) {
                    None => return None,
                    Some(rg) if rg == last => return None,
                    Some(rg) if order.after(last).next() == Some(rg) => {
                        (State::InOrder { last: rg }, Some(rg))
                    }
                    // Out of order: the scan pruned something this reader
                    // couldn't, so the rest may never be read. Keep only the
                    // batch holding this row group, to serve it.
                    Some(rg) => {
                        self.batches.retain(|b| b.position(rg).is_some());
                        (State::Off, Some(rg))
                    }
                }
            }
        };
        self.state = state;
        entered
    }

    /// The planned order, built on first use.
    fn order(&mut self) -> &Order {
        let Self {
            scan,
            file,
            meta,
            order,
            ..
        } = self;
        order.get_or_insert_with(|| planned_order(scan, file, meta))
    }

    /// The bytes of `ranges`, downloaded now. Cold kept pages of one column
    /// chunk a little apart come down in one span (see [`read_through_spans`]).
    async fn fetch(&self, ranges: &[Range<u64>]) -> ParquetResult<Vec<Bytes>> {
        let store = &self.scan.store;
        let location = self.location();
        let clamped = store.clamped(location, ranges)?;
        let in_one_chunk = |gap: &Range<u64>| {
            row_group_holding(&self.meta, gap.start)
                .is_some_and(|rg| column_of(self.meta.row_group(rg), gap).is_some())
        };
        let cold = |span: &Range<u64>| all_cold(store, location, span);
        let Some(spans) = read_through_spans(&clamped, in_one_chunk, cold) else {
            return Ok(store.get_ranges(location, ranges).await?);
        };
        // Counted as the ranges asked for, the way the store counts them: what
        // is read through depends on what the cache holds, and the counts must
        // not.
        store.count_page_reads(&clamped);
        let fetched = store.get_ranges_uncounted(location, &spans).await?;
        cut(&clamped, &spans, &fetched)
    }

    /// `ranges` of `row_group`: what `batch` holds comes from it, the rest is
    /// downloaded now. If the batch's download failed, `ranges` are downloaded
    /// as if nothing was read ahead. The batch is dropped once all its row
    /// groups are read, or once this reader is off.
    async fn serve(
        &mut self,
        row_group: usize,
        batch: usize,
        ranges: &[Range<u64>],
    ) -> ParquetResult<Vec<Bytes>> {
        self.batches[batch].settle().await;
        let held = &self.batches[batch];
        let (Some(position), true) = (held.position(row_group), held.has_bytes()) else {
            self.batches.remove(batch);
            return self.fetch(ranges).await;
        };
        // Clamp and count the request the way the store does, wherever its
        // bytes come from.
        let ranges = self
            .scan
            .store
            .clamped(&self.file.object_meta.location, ranges)?;
        self.scan.store.count_page_reads(&ranges);
        // The cache doesn't see reads served from here, so tell it where this
        // reader is, or it may take the reader's next miss for a jump back.
        self.scan.store.note_reads(self.location(), &ranges);
        let held = &mut self.batches[batch];
        let pieces: Vec<Vec<Piece>> = ranges.iter().map(|r| held.pieces(r)).collect();
        held.served[position] = true;
        if held.served.iter().all(|&s| s) || matches!(self.state, State::Off) {
            self.batches.remove(batch);
        }
        let missing: Vec<Range<u64>> = pieces
            .iter()
            .flatten()
            .filter_map(|piece| match piece {
                Piece::Missing(range) => Some(range.clone()),
                Piece::Held(_) => None,
            })
            .collect();
        // Not counted again: the whole request was counted above.
        let mut fetched = self
            .scan
            .store
            .get_ranges_uncounted(self.location(), &missing)
            .await?
            .into_iter();
        pieces
            .into_iter()
            .map(|pieces| {
                let mut parts = pieces
                    .into_iter()
                    .map(|piece| match piece {
                        Piece::Held(bytes) => Some(bytes),
                        // A short gap would shift every byte after it.
                        Piece::Missing(range) => fetched
                            .next()
                            .filter(|bytes| bytes.len() as u64 == range.end - range.start),
                    })
                    .collect::<Option<Vec<Bytes>>>()
                    .ok_or_else(|| {
                        ParquetError::General("read-ahead: a range was not served".into())
                    })?;
                Ok(match parts.len() {
                    1 => parts.pop().unwrap_or_default(),
                    _ => Bytes::from(parts.concat()),
                })
            })
            .collect()
    }

    /// Start downloading the next batch: `columns` of the planned row groups
    /// after the last one held, as many as fit the reader's cap.
    /// - It stops at a row group the scan reads only in part; the scan reads
    ///   that one itself.
    /// - It skips row groups whose columns are already local.
    /// - It drops row groups from the end until the batch fits the reader's
    ///   cap, and starts nothing if the budget or the cache's write queue
    ///   has no room.
    /// - It does nothing while a batch is still downloading.
    fn start_ahead(&mut self, row_group: usize, columns: &[usize]) {
        if !self.scan.is_enabled() || self.batches.iter().any(Batch::is_running) {
            return;
        }
        let Some(order) = &self.order else {
            return;
        };
        let after = self
            .batches
            .iter()
            .flat_map(|b| b.row_groups.iter().map(|(rg, _)| *rg))
            .max()
            .map_or(row_group, |held| held.max(row_group));
        let location = &self.file.object_meta.location;
        let cap = self.scan.reader_cap();
        // Half the reader's cap, so the next batch can download while the
        // scan decodes this one. Counted by the projected chunks alone; the
        // spans they merge into are measured below.
        let batch_cap = cap / READ_AHEAD_BATCHES_IN_FLIGHT;
        let mut row_groups: Vec<(usize, Vec<Range<u64>>)> = Vec::new();
        let mut planned = 0usize;
        for (walked, rg) in order.after(after).enumerate() {
            let meta = self.meta.row_group(rg);
            if !order.reads_whole(rg, meta.num_rows() as usize) {
                break;
            }
            let ranges: Vec<Range<u64>> = columns.iter().map(|&c| chunk_range(meta, c)).collect();
            let bytes: usize = ranges.iter().map(|r| (r.end - r.start) as usize).sum();
            if walked >= READ_AHEAD_MIN_ROW_GROUPS && planned + bytes > batch_cap {
                break;
            }
            planned += bytes;
            if !ranges.iter().all(|r| self.scan.store.is_local(location, r)) {
                row_groups.push((rg, ranges));
            }
        }
        // The batch holds whole spans, gaps included, so reserve those. Drop row
        // groups from the end until it fits its share and the reader's cap. A
        // batch of one row group may take the reader's whole cap: a row group
        // over half the cap is still worth downloading while the scan decodes.
        let held: usize = self.batches.iter().map(|b| b.charge).sum();
        let blocks = self.scan.store.cache_blocks(location);
        let (clusters, charge) = loop {
            if row_groups.is_empty() {
                return;
            }
            let mut clusters = fetch_spans(&row_groups, blocks);
            if let Some(blocks) = blocks {
                let others = self.others_near(row_groups[0].0, &clusters, blocks);
                clusters = leave_shared_ends(clusters, blocks, &others);
            }
            if clusters.is_empty() {
                return;
            }
            let charge: usize = clusters
                .iter()
                .flatten()
                .map(|span| (span.end - span.start) as usize)
                .sum();
            if (charge <= batch_cap || row_groups.len() == 1) && held + charge <= cap {
                break (clusters, charge);
            }
            row_groups.pop();
        };
        // Reserve room in the cache's write queue for what the cache will
        // download (not just the spans), so it keeps the batch's downloads.
        // The room is held until the download ends.
        let downloads = blocks.map_or(charge as u64, |blocks| download_bytes(&clusters, blocks));
        let Some(room) = self.scan.store.cache_room_ahead(location, downloads) else {
            return;
        };
        // Charged what the cache downloads, not just the spans: a read of a
        // whole download is served as a slice of it, which keeps it all alive.
        let Some(hold) = self.scan.reserve(charge.max(downloads as usize)) else {
            return;
        };
        let spans: Vec<Range<u64>> = clusters.iter().flatten().cloned().collect();
        let store = Arc::clone(&self.scan.store);
        let path = location.clone();
        let task = spawn(
            async move {
                let fetched = store.get_spans_ahead(&path, &clusters).await;
                drop(room);
                fetched
            }
            .in_current_span(),
        );
        let held = Arc::new(Mutex::new(Held {
            bytes: None,
            hold: Some(hold),
            download: task.abort_handle(),
        }));
        self.scan.track(&held);
        let served = vec![false; row_groups.len()];
        self.batches.push(Batch {
            row_groups,
            spans,
            task: Some(task),
            served,
            charge,
            held,
        });
    }

    /// The columns of `row_group` that `ranges` read, or `None` if a range is
    /// outside every column, or a column had less than
    /// [`MIN_CHUNK_READ_PERCENT`] of its bytes read.
    fn mostly_read_columns(&self, row_group: usize, ranges: &[Range<u64>]) -> Option<Vec<usize>> {
        let meta = self.meta.row_group(row_group);
        let chunks: Vec<Range<u64>> = (0..meta.num_columns())
            .map(|c| chunk_range(meta, c))
            .collect();
        let mut read: BTreeMap<usize, u64> = BTreeMap::new();
        for range in ranges {
            *read.entry(column_of(meta, range)?).or_default() += range.end - range.start;
        }
        read.iter()
            .all(|(&c, &bytes)| {
                bytes * 100 >= (chunks[c].end - chunks[c].start) * MIN_CHUNK_READ_PERCENT
            })
            .then(|| read.into_keys().collect())
    }

    /// What another read may be downloading near the batch while it downloads:
    /// the scan's columns, not local yet, of row groups this reader reads
    /// before the batch, or that another partition reads. Pruned row groups
    /// are read by no one, and later ones only after the batch is done.
    fn others_near(
        &self,
        first: usize,
        clusters: &[Vec<Range<u64>>],
        blocks: CacheBlocks,
    ) -> Vec<Range<u64>> {
        let (Some(order), Some(start), Some(end)) = (
            &self.order,
            clusters.first().and_then(|c| c.first()).map(|s| s.start),
            clusters.last().and_then(|c| c.last()).map(|s| s.end),
        ) else {
            return Vec::new();
        };
        // Anything further away can't overlap the batch's downloads.
        let near = start.saturating_sub(blocks.min_get_bytes + blocks.block_bytes)
            ..end + blocks.min_get_bytes + blocks.block_bytes;
        order
            .spans
            .iter()
            .enumerate()
            .filter(|&(rg, span)| {
                (order.elsewhere[rg] || (rg < first && order.plan.should_scan(rg)))
                    && span.start < near.end
                    && near.start < span.end
            })
            .flat_map(|(rg, _)| {
                self.columns
                    .iter()
                    .map(move |&c| chunk_range(self.meta.row_group(rg), c))
            })
            .filter(|chunk| {
                chunk.start < near.end
                    && near.start < chunk.end
                    && !self.scan.store.is_local(self.location(), chunk)
            })
            .collect()
    }
}

/// The row-group pruning of a scan filtered by `predicate`, written against
/// `table`, for the file `meta` describes, built the way DataFusion's opener
/// builds it:
/// - the filter is mapped by `adapter` onto the file's own schema (a column
///   the file lacks becomes null, a renamed one its stored name), then
///   simplified;
/// - it's built per file, since files written under older schemas map the
///   filter differently.
///
/// `None` if the filter doesn't convert; nothing can be pruned then.
pub(crate) fn file_pruning(
    predicate: &Arc<dyn PhysicalExpr>,
    adapter: &Arc<dyn PhysicalExprAdapterFactory>,
    table: &SchemaRef,
    meta: &ParquetMetaData,
) -> Option<(PruningPredicate, SchemaRef)> {
    let file_meta = meta.file_metadata();
    let file_schema = Arc::new(
        parquet_to_arrow_schema(file_meta.schema_descr(), file_meta.key_value_metadata()).ok()?,
    );
    let schema =
        apply_file_schema_type_coercions(table, &file_schema).map_or(file_schema, Arc::new);
    let mapped = adapter
        .create(Arc::clone(table), Arc::clone(&schema))
        .ok()?
        .rewrite(Arc::clone(predicate))
        .ok()?;
    let simplified = PhysicalExprSimplifier::new(&schema).simplify(mapped).ok()?;
    let pruning = PruningPredicate::try_new(simplified, Arc::clone(&schema)).ok()?;
    Some((pruning, schema))
}

/// `plan` with the row groups of the file `meta` describes, at `location`,
/// that their statistics show can't match `pruning` ([`file_pruning`])
/// skipped. DataFusion reports its own pruning counts; this repeat goes to
/// a throwaway set so they aren't counted twice.
pub(crate) fn statistics_pruned(
    plan: ParquetAccessPlan,
    meta: &ParquetMetaData,
    (predicate, file_schema): &(PruningPredicate, SchemaRef),
    location: &str,
) -> ParquetAccessPlan {
    let metrics = ParquetFileMetrics::new(0, location, &ExecutionPlanMetricsSet::new());
    let mut filter = RowGroupAccessPlanFilter::new(plan);
    filter.prune_by_statistics(
        file_schema,
        meta.file_metadata().schema_descr(),
        meta.row_groups(),
        predicate,
        &metrics,
    );
    filter.build()
}

/// The row groups `file` reads in this partition: the access plan, pruned by
/// the row-group statistics, the page index and the partition's byte range,
/// as DataFusion's opener prunes it after creating the reader.
fn planned_order(scan: &ScanReadAhead, file: &PartitionedFile, meta: &ParquetMetaData) -> Order {
    let groups = meta.row_groups();
    let plan = file
        .extension::<ParquetAccessPlan>()
        .cloned()
        .unwrap_or_else(|| ParquetAccessPlan::new_all(groups.len()));
    let file_meta = meta.file_metadata();
    let location = file.object_meta.location.as_ref();
    let pruning = scan.pruning(meta);
    let mut plan = match &pruning {
        Some(pruning) => statistics_pruned(plan, meta, pruning, location),
        None => plan,
    };
    // DataFusion already reports its pruning counts; this repeat goes to a
    // throwaway set so they aren't counted twice.
    let metrics = ParquetFileMetrics::new(0, location, &ExecutionPlanMetricsSet::new());
    if let Some((predicate, file_schema)) = &pruning
        && meta.offset_index().is_some()
    {
        plan = PagePruningAccessPlanFilter::new(predicate.orig_expr(), Arc::clone(file_schema))
            .prune_plan_with_page_index(
                plan,
                file_schema,
                file_meta.schema_descr(),
                meta,
                &metrics,
            );
    }
    // Prune for all partitions first, then narrow to this one's range, so we
    // also know which row groups the other partitions read.
    let everywhere: Vec<bool> = (0..groups.len()).map(|rg| plan.should_scan(rg)).collect();
    if let Some(range) = &file.range {
        let mut filter = RowGroupAccessPlanFilter::new(plan);
        filter.prune_by_range(groups, range);
        plan = filter.build();
    }
    let elsewhere = everywhere
        .iter()
        .enumerate()
        .map(|(rg, &read)| read && !plan.should_scan(rg))
        .collect();
    Order {
        spans: groups.iter().map(row_group_span).collect(),
        plan,
        elsewhere,
    }
}

/// The spans to download for `ranges`: the store's own merges (gaps up to 1
/// MiB, see [`merged_ranges`]), then two of those joined across a gap of up to
/// [`PAGE_READ_THROUGH_BYTES`] inside one column chunk, the skipped pages
/// included. `None` when nothing is joined, so the store fetches `ranges` as it
/// always has.
///
/// ```text
///   kept pages of one chunk:  [p0]---- > 1 MiB, <= 2 MiB ----[p10]
///   the store alone:          GET p0                         GET p10
///   read through:             GET p0 ................................ p10
/// ```
///
/// - Only inside one chunk (`in_one_chunk`): the bytes between two chunks are
///   columns the scan doesn't read.
/// - Only when both sides and the gap are cold (`cold`: no byte local). The
///   cache splits a download at bytes it holds, so a partly cached gap would
///   cost more GETs than not reading through, and a local side has no GET to
///   save.
/// - Joins chain, so kept pages each a little apart can come down as one span
///   over the whole chunk, as object_store already does for gaps up to 1 MiB.
fn read_through_spans(
    ranges: &[Range<u64>],
    in_one_chunk: impl Fn(&Range<u64>) -> bool,
    cold: impl Fn(&Range<u64>) -> bool,
) -> Option<Vec<Range<u64>>> {
    let merged = merged_ranges(ranges);
    let mut spans: Vec<Range<u64>> = Vec::with_capacity(merged.len());
    let mut prev: Option<&Range<u64>> = None;
    for span in &merged {
        // Merged spans are sorted and more than 1 MiB apart.
        let joins = prev.is_some_and(|prev| {
            let gap = prev.end..span.start;
            gap.end - gap.start <= PAGE_READ_THROUGH_BYTES
                && in_one_chunk(&gap)
                && cold(&(prev.start..span.end))
        });
        match spans.last_mut() {
            Some(last) if joins => last.end = span.end,
            _ => spans.push(span.clone()),
        }
        prev = Some(span);
    }
    (spans.len() < merged.len()).then_some(spans)
}

/// Whether no byte of `range` in `location` is local, checked one cache block
/// at a time: [`SuperfileObjectStore::is_local`] only says whether all of a
/// range is. `false` when no cache tells its blocks: nothing is read through
/// without one, as a store with no cache in front pays no GET to save.
fn all_cold(store: &SuperfileObjectStore, location: &ObjPath, range: &Range<u64>) -> bool {
    let Some(blocks) = store.cache_blocks(location) else {
        return false;
    };
    let block = blocks.block_bytes;
    (range.start / block..range.end.div_ceil(block)).all(|b| {
        let piece = (b * block).max(range.start)..((b + 1) * block).min(range.end);
        !store.is_local(location, &piece)
    })
}

/// The row group whose bytes hold `offset`, or `None` outside every row group
/// (the footer, page indexes, bloom filters).
fn row_group_holding(meta: &ParquetMetaData, offset: u64) -> Option<usize> {
    let groups = meta.row_groups();
    let i = groups.partition_point(|rg| row_group_span(rg).end <= offset);
    groups
        .get(i)
        .filter(|rg| row_group_span(rg).contains(&offset))
        .map(|_| i)
}

/// Each of `ranges`, copied out of the downloaded `spans`: `fetched[i]` holds
/// `spans[i]`, and the spans are sorted and apart. Copied, not sliced, so the
/// skipped pages a span read through are dropped here rather than kept alive
/// by the pages the scan holds.
fn cut(
    ranges: &[Range<u64>],
    spans: &[Range<u64>],
    fetched: &[Bytes],
) -> ParquetResult<Vec<Bytes>> {
    ranges
        .iter()
        .map(|range| {
            let i = spans.partition_point(|span| span.end < range.end);
            spans
                .get(i)
                .zip(fetched.get(i))
                .filter(|(span, bytes)| {
                    span.start <= range.start && bytes.len() as u64 == span.end - span.start
                })
                .map(|(span, bytes)| {
                    Bytes::copy_from_slice(
                        &bytes[(range.start - span.start) as usize
                            ..(range.end - span.start) as usize],
                    )
                })
                .ok_or_else(|| ParquetError::General("read-through: a range was not served".into()))
        })
        .collect()
}

/// The column whose chunk in `row_group` holds all of `range`, if one does.
fn column_of(row_group: &RowGroupMetaData, range: &Range<u64>) -> Option<usize> {
    (0..row_group.num_columns()).find(|&c| {
        let chunk = chunk_range(row_group, c);
        chunk.start <= range.start && range.end <= chunk.end
    })
}

/// A column chunk's bytes: its dictionary page, if any, through its last data page.
fn chunk_range(row_group: &RowGroupMetaData, column: usize) -> Range<u64> {
    let (start, len) = row_group.column(column).byte_range();
    start..start + len
}

/// A row group's bytes, from its first column chunk to the end of its last.
fn row_group_span(row_group: &RowGroupMetaData) -> Range<u64> {
    (0..row_group.num_columns())
        .map(|c| chunk_range(row_group, c))
        .reduce(|a, b| a.start.min(b.start)..a.end.max(b.end))
        .unwrap_or(0..0)
}

/// Test superfiles, shared with the provider's tests: an in-memory source that
/// records what it serves, files of a chosen shape, and a block-cached view.
#[cfg(test)]
pub(crate) mod fixture {
    use std::{
        ops::Range,
        path::Path,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    };

    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use async_trait::async_trait;
    use bytes::Bytes;
    use parquet::{
        arrow::ArrowWriter,
        file::{
            metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader},
            properties::WriterProperties,
        },
        schema::types::ColumnPath,
    };
    use tokio::time::sleep;

    use crate::{
        storage::LocalFsStorageProvider,
        superfile::{
            CacheRoom,
            lazy_source::{LazyByteSource, LazyByteSourceError},
        },
        supertable::{
            manifest::SuperfileUri,
            reader_cache::{
                ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy,
                block_source::BlockCachedSource,
            },
        },
    };

    /// Rows per row group in the small file.
    pub(crate) const ROWS_PER_GROUP: usize = 1_000;
    /// Row groups in every file.
    pub(crate) const GROUPS: usize = 8;

    /// A superfile on object storage. `range` records every read, unless
    /// `resident` makes them local.
    /// - `fail_at` fails the first read starting at that offset.
    /// - `latency` makes each read take that long, like a GET, so reads in
    ///   flight together really overlap.
    pub(crate) struct Superfile {
        pub(crate) bytes: Bytes,
        resident: bool,
        latency: Option<Duration>,
        pub(crate) fail_at: Mutex<Option<u64>>,
        /// Returns the first read starting at this offset one byte short.
        pub(crate) short_at: Mutex<Option<u64>>,
        /// Whether it reports room in the cache for downloads ahead.
        pub(crate) keeps: AtomicBool,
        reads: Mutex<Vec<Range<u64>>>,
        /// Reads noted without a download, see [`LazyByteSource::note_read`].
        noted: Mutex<Vec<Range<u64>>>,
    }

    #[async_trait]
    impl LazyByteSource for Superfile {
        fn size(&self) -> u64 {
            self.bytes.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            {
                let mut fail_at = self.fail_at.lock().expect("fail_at");
                if *fail_at == Some(start) {
                    *fail_at = None;
                    return Err(LazyByteSourceError::OutOfBounds {
                        start,
                        len,
                        size: 0,
                    });
                }
            }
            self.reads.lock().expect("reads").push(start..start + len);
            if let Some(latency) = self.latency {
                sleep(latency).await;
            }
            let short = self
                .short_at
                .lock()
                .expect("short_at")
                .take_if(|at| *at == start);
            let end = start + len - u64::from(short.is_some());
            Ok(self.bytes.slice(start as usize..end as usize))
        }

        fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
            self.resident
                .then(|| self.bytes.slice(start as usize..(start + len) as usize))
        }

        fn cache_room_ahead(&self, _bytes: u64) -> Option<CacheRoom> {
            self.keeps
                .load(Ordering::Relaxed)
                .then(|| Box::new(()) as CacheRoom)
        }

        fn note_read(&self, start: u64, len: u64) {
            self.noted.lock().expect("noted").push(start..start + len);
        }
    }

    impl Superfile {
        pub(crate) fn new(bytes: Bytes, resident: bool) -> Arc<Self> {
            Arc::new(Self {
                bytes,
                resident,
                latency: None,
                fail_at: Mutex::new(None),
                short_at: Mutex::new(None),
                keeps: AtomicBool::new(true),
                reads: Mutex::new(Vec::new()),
                noted: Mutex::new(Vec::new()),
            })
        }

        /// A cold superfile whose every read takes `latency`.
        pub(crate) fn with_latency(bytes: Bytes, latency: Duration) -> Arc<Self> {
            Arc::new(Self {
                bytes,
                resident: false,
                latency: Some(latency),
                fail_at: Mutex::new(None),
                short_at: Mutex::new(None),
                keeps: AtomicBool::new(true),
                reads: Mutex::new(Vec::new()),
                noted: Mutex::new(Vec::new()),
            })
        }

        pub(crate) fn reads(&self) -> Vec<Range<u64>> {
            self.reads.lock().expect("reads").clone()
        }

        /// Bytes served so far, overlapping reads counted each time.
        pub(crate) fn fetched_bytes(&self) -> u64 {
            self.reads().iter().map(|r| r.end - r.start).sum()
        }

        pub(crate) fn noted(&self) -> Vec<Range<u64>> {
            self.noted.lock().expect("noted").clone()
        }

        pub(crate) fn clear(&self) {
            self.reads.lock().expect("reads").clear();
        }

        pub(crate) fn slices(&self, ranges: &[Range<u64>]) -> Vec<Bytes> {
            ranges
                .iter()
                .map(|r| self.bytes.slice(r.start as usize..r.end as usize))
                .collect()
        }
    }

    /// The test table: `a` counts up from 0 (so min/max and page pruning on
    /// it work), `b` is text, `c` is `a` times 7 in the small file and
    /// scrambled in the wide one.
    pub(crate) fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Int64, false),
        ]))
    }

    /// A small file: [`GROUPS`] row groups of [`ROWS_PER_GROUP`] rows, one page
    /// per column. Its footer includes the page index, like the provider's.
    pub(crate) fn small_file() -> (Bytes, Arc<ParquetMetaData>) {
        write(ROWS_PER_GROUP, None, false)
    }

    /// A wide file: [`GROUPS`] row groups of `rows_per_group` rows, big enough
    /// to cross the cache's blocks, `b` and `c` scrambled so they don't
    /// compress away, and pages of `page_rows` rows.
    pub(crate) fn wide_file(
        rows_per_group: usize,
        page_rows: usize,
    ) -> (Bytes, Arc<ParquetMetaData>) {
        write(rows_per_group, Some(page_rows), true)
    }

    fn write(
        rows_per_group: usize,
        page_rows: Option<usize>,
        wide: bool,
    ) -> (Bytes, Arc<ParquetMetaData>) {
        let mut props =
            WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_group));
        if let Some(rows) = page_rows {
            props = props
                .set_data_page_row_count_limit(rows)
                .set_write_batch_size(rows);
        }
        if wide {
            // No dictionary for the scrambled text, so the page index can
            // narrow `b` too.
            props = props.set_column_dictionary_enabled(ColumnPath::from("b"), false);
        }
        let n = (rows_per_group * GROUPS) as i64;
        let scramble = |i: i64| {
            // splitmix64: cheap, deterministic, incompressible.
            let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..n)),
                Arc::new(StringArray::from_iter_values((0..n).map(|i| {
                    if wide {
                        // 64 characters, so `b` is wider than object_store's
                        // 1 MiB merge gap and a scan of `a` and `c` skips it.
                        format!(
                            "{:016x}{:016x}{:016x}{:016x}",
                            scramble(i),
                            scramble(-i),
                            scramble(i ^ 0x5555),
                            scramble(i ^ 0x2AAA)
                        )
                    } else {
                        format!("row {i}")
                    }
                }))),
                Arc::new(Int64Array::from_iter_values(
                    (0..n).map(|i| if wide { scramble(i) as i64 } else { i * 7 }),
                )),
            ],
        )
        .expect("batch");
        let mut buf = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut buf, schema(), Some(props.build())).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");
        let bytes = Bytes::from(buf);
        let meta = ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Required)
            .parse_and_finish(&bytes)
            .expect("metadata");
        (bytes, Arc::new(meta))
    }

    /// `inner` behind the disk cache, read cold: missing blocks are downloaded
    /// and kept. The store must outlive the source.
    pub(crate) fn block_cached(
        inner: &Arc<Superfile>,
        dir: &Path,
    ) -> (Arc<DiskCacheStore>, Arc<dyn LazyByteSource>) {
        let storage =
            Arc::new(LocalFsStorageProvider::new(dir.join("storage")).expect("local storage"));
        let config = DiskCacheConfig {
            cache_root: dir.join("cache"),
            disk_budget_bytes: u64::MAX,
            cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
            eviction: Box::new(LruPolicy::new()),
            ..DiskCacheConfig::default()
        };
        let store = DiskCacheStore::new_unpinned(storage, config).expect("disk cache");
        let uri = SuperfileUri::new_v4();
        let source = BlockCachedSource::new(
            Arc::clone(inner) as Arc<dyn LazyByteSource>,
            Arc::downgrade(&store),
            uri,
            dir.join("sf.blocks"),
        );
        store.install_block_entry_for_test(uri, Arc::clone(&source));
        (store, source as Arc<dyn LazyByteSource>)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        ops::Range,
        slice::from_ref,
        sync::{Arc, Mutex, atomic::Ordering},
        time::Duration,
    };

    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use bytes::Bytes;
    use datafusion::{
        common::DFSchema,
        datasource::{
            listing::{FileRange, PartitionedFile},
            physical_plan::parquet::ParquetAccessPlan,
        },
        execution::context::ExecutionProps,
        logical_expr::{Expr, col, lit, utils::conjunction},
        object_store::path::Path as ObjPath,
        physical_expr::{
            PhysicalExpr, create_physical_expr,
            expressions::{Column, DynamicFilterPhysicalExpr},
        },
    };
    use object_store::OBJECT_STORE_COALESCE_DEFAULT;
    use parquet::{
        arrow::{
            arrow_reader::{RowSelection, RowSelector},
            async_reader::AsyncFileReader,
        },
        file::metadata::ParquetMetaData,
    };
    use tokio::{spawn, task::yield_now, time::timeout};

    use crate::{
        memory::{ConnectionMemoryBudget, Reclaim},
        runtime_metrics::op_stats::with_op_stats,
        superfile::{CacheBlocks, lazy_source::LazyByteSource},
        supertable::query::{
            df_object_store::SuperfileObjectStore,
            read_ahead::{
                Batch, Held, PAGE_READ_THROUGH_BYTES, READ_AHEAD_BATCHES_IN_FLIGHT, ReadAhead,
                ScanReadAhead, all_cold, chunk_range, cut, download_bytes, fetch_spans,
                fixture::{
                    GROUPS, ROWS_PER_GROUP, Superfile, block_cached, schema, small_file, wide_file,
                },
                leave_shared_ends, lock, read_through_spans, row_group_span,
            },
        },
    };

    /// Columns `a` and `c`: the scan reads both, leaving `b` unread between them.
    const PROJECTED: [usize; 2] = [0, 2];
    /// The cache block size the span tests assume.
    const BLOCK: u64 = 64 * 1024;
    /// The fewest bytes one of the cache's downloads covers, in those tests.
    const MIN_GET: u64 = 512 * 1024;
    /// The cache the span tests assume.
    const BLOCKS: CacheBlocks = CacheBlocks {
        block_bytes: BLOCK,
        min_get_bytes: MIN_GET,
        group_bytes: MIN_GET,
        grouped: (0, 0),
    };
    /// How long a download takes in the test that drops a reader mid-download,
    /// so the batch is still running when the last read returns.
    const SLOW_DOWNLOAD: Duration = Duration::from_millis(200);
    /// How long a test waits for downloads ahead from an in-memory source.
    const SETTLE_DEADLINE: Duration = Duration::from_secs(10);
    /// A page's bytes in the read-through tests.
    const PAGE: u64 = 64 * 1024;

    /// The superfile, a store serving it, its path and its footer.
    fn setup(
        resident: bool,
    ) -> (
        Arc<Superfile>,
        Arc<SuperfileObjectStore>,
        ObjPath,
        Arc<ParquetMetaData>,
    ) {
        let (bytes, meta) = small_file();
        let superfile = Superfile::new(bytes, resident);
        let path = ObjPath::from("t/sf.parquet");
        let mut sources: HashMap<ObjPath, Arc<dyn LazyByteSource>> = HashMap::new();
        sources.insert(
            path.clone(),
            Arc::clone(&superfile) as Arc<dyn LazyByteSource>,
        );
        let (store, _) = with_op_stats(|| SuperfileObjectStore::from_sources(sources));
        (superfile, Arc::new(store), path, meta)
    }

    /// [`setup`] over a superfile whose every download takes
    /// [`SLOW_DOWNLOAD`], so a test can act while a batch downloads.
    fn setup_slow() -> (
        Arc<Superfile>,
        Arc<SuperfileObjectStore>,
        ObjPath,
        Arc<ParquetMetaData>,
    ) {
        let (bytes, meta) = small_file();
        let superfile = Superfile::with_latency(bytes, SLOW_DOWNLOAD);
        let path = ObjPath::from("t/sf.parquet");
        let sources = HashMap::from([(
            path.clone(),
            Arc::clone(&superfile) as Arc<dyn LazyByteSource>,
        )]);
        let store = Arc::new(SuperfileObjectStore::from_sources(sources));
        (superfile, store, path, meta)
    }

    /// A scan of the test table with `filters`, handed over the way the meter
    /// hands them over when the scan runs.
    fn scan(
        store: &Arc<SuperfileObjectStore>,
        budget: &Arc<ConnectionMemoryBudget>,
        filters: &[Expr],
    ) -> Arc<ScanReadAhead> {
        scan_over(store, budget, filters, schema())
    }

    /// A scan of a table with `table`'s schema and `filters`.
    fn scan_over(
        store: &Arc<SuperfileObjectStore>,
        budget: &Arc<ConnectionMemoryBudget>,
        filters: &[Expr],
        table: SchemaRef,
    ) -> Arc<ScanReadAhead> {
        // The test file's row groups are small, so the default cap would
        // take them all; these tests pin batches of two.
        let scan = Arc::new(
            ScanReadAhead::new(Arc::clone(store), Arc::clone(budget), Arc::clone(&table))
                .with_reader_cap(two_row_groups_cap()),
        );
        if let Some(filter) = conjunction(filters.iter().cloned()) {
            scan.set_predicate(predicate(&filter, table), None);
        }
        scan
    }

    /// `filter` planned over `table`.
    fn predicate(filter: &Expr, table: SchemaRef) -> Arc<dyn PhysicalExpr> {
        let schema = DFSchema::try_from(table).expect("schema");
        create_physical_expr(filter, &schema, &ExecutionProps::new()).expect("predicate")
    }

    /// The file as DataFusion hands it to the reader, with the provider's
    /// access `plan` and the partition's byte `range`.
    fn partitioned(
        path: &ObjPath,
        meta: &ParquetMetaData,
        plan: Option<ParquetAccessPlan>,
        range: Option<FileRange>,
    ) -> PartitionedFile {
        let size = meta
            .row_groups()
            .iter()
            .map(|rg| rg.compressed_size() as u64)
            .sum();
        let mut file = PartitionedFile::new(path.to_string(), size);
        if let Some(plan) = plan {
            file = file.with_extension(plan);
        }
        file.range = range;
        file
    }

    /// A reader over `scan` with no plan and no range.
    fn reader_of(
        scan: &Arc<ScanReadAhead>,
        path: &ObjPath,
        meta: &Arc<ParquetMetaData>,
    ) -> ReadAhead {
        scan.reader(partitioned(path, meta, None, None), Arc::clone(meta))
    }

    /// A reader over a plain scan: no LIMIT, no pruning, no plan, no range.
    fn plain_reader(
        store: &Arc<SuperfileObjectStore>,
        budget: &Arc<ConnectionMemoryBudget>,
        path: &ObjPath,
        meta: &Arc<ParquetMetaData>,
    ) -> ReadAhead {
        reader_of(&scan(store, budget, &[]), path, meta)
    }

    /// A reader cap under which a batch takes two of the test file's row
    /// groups but not three, with room for two such batches: a batch gets
    /// half the cap, one byte short of what three row groups take. The
    /// file's row groups are alike in size, so any two fit and no three do.
    fn two_row_groups_cap() -> usize {
        let (_, meta) = small_file();
        two_row_groups_cap_for(&meta)
    }

    /// [`two_row_groups_cap`] for a file other than the small one.
    fn two_row_groups_cap_for(meta: &ParquetMetaData) -> usize {
        (charge(meta, &[0, 1, 2]) - 1) * READ_AHEAD_BATCHES_IN_FLIGHT
    }

    /// The projected columns' whole chunks in row group `rg`.
    fn chunks(meta: &ParquetMetaData, rg: usize) -> Vec<Range<u64>> {
        PROJECTED
            .iter()
            .map(|&c| chunk_range(meta.row_group(rg), c))
            .collect()
    }

    /// What a batch fetching row groups `rgs` ahead is charged.
    fn charge(meta: &ParquetMetaData, rgs: &[usize]) -> usize {
        let row_groups: Vec<(usize, Vec<Range<u64>>)> =
            rgs.iter().map(|&rg| (rg, chunks(meta, rg))).collect();
        // The test sources cache nothing, so spans are never ordered.
        fetch_spans(&row_groups, None)
            .iter()
            .flatten()
            .map(|span| (span.end - span.start) as usize)
            .sum()
    }

    /// Whether any recorded read touches row group `rg`'s bytes.
    fn touched(reads: &[Range<u64>], meta: &ParquetMetaData, rg: usize) -> bool {
        let span = row_group_span(meta.row_group(rg));
        reads
            .iter()
            .any(|r| r.start < span.end && span.start < r.end)
    }

    /// Wait until every fetch `reader` holds has finished, so what they read
    /// is recorded before a test looks.
    async fn settle(reader: &ReadAhead) {
        timeout(SETTLE_DEADLINE, async {
            while reader
                .batches
                .iter()
                .any(|b| b.task.as_ref().is_some_and(|task| !task.is_finished()))
            {
                yield_now().await;
            }
        })
        .await
        .expect("fetches ahead finish");
    }

    /// Read the projected chunks of each row group in `rgs`, in turn.
    async fn read_row_groups(reader: &mut ReadAhead, meta: &ParquetMetaData, rgs: &[usize]) {
        for &rg in rgs {
            reader
                .get_byte_ranges(chunks(meta, rg))
                .await
                .expect("read");
        }
    }

    #[tokio::test]
    async fn an_in_order_scan_fetches_the_next_two_row_groups_and_serves_them() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        let reads = superfile.reads();
        assert!(touched(&reads, &meta, 2) && touched(&reads, &meta, 3));
        assert!(!touched(&reads, &meta, 4), "one batch of two, not more");

        superfile.clear();
        let got = reader
            .get_byte_ranges(chunks(&meta, 2))
            .await
            .expect("rg 2");
        assert_eq!(got, superfile.slices(&chunks(&meta, 2)));
        settle(&reader).await;
        assert!(!touched(&superfile.reads(), &meta, 2), "served from memory");
        assert_eq!(
            superfile.noted(),
            chunks(&meta, 2),
            "and the cache is told, so it still knows where the scan is"
        );
    }

    /// With the shipped cap, a batch takes every row group that fits: on a
    /// file of small row groups that is all of them, one round trip instead
    /// of one per pair.
    #[tokio::test]
    async fn a_batch_takes_as_many_row_groups_as_fit_the_cap() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let scan = Arc::new(ScanReadAhead::new(
            Arc::clone(&store),
            Arc::clone(&budget),
            schema(),
        ));
        let mut reader = reader_of(&scan, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        let reads = superfile.reads();
        for rg in 2..GROUPS {
            assert!(touched(&reads, &meta, rg), "row group {rg} fetched ahead");
        }
    }

    /// A row group over half the reader's cap, but under the cap, is read
    /// ahead on its own: the next one does not fit beside it.
    #[tokio::test]
    async fn a_row_group_over_half_the_cap_is_read_ahead_alone() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        // Two row groups do not fit the cap, one does, and a batch's share
        // (half the cap) is one byte short of one row group.
        let cap = charge(&meta, &[2]) * READ_AHEAD_BATCHES_IN_FLIGHT - 1;
        let scan = Arc::new(
            ScanReadAhead::new(Arc::clone(&store), Arc::clone(&budget), schema())
                .with_reader_cap(cap),
        );
        let mut reader = reader_of(&scan, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        let reads = superfile.reads();
        assert!(touched(&reads, &meta, 2), "row group 2 fetched ahead alone");
        assert!(
            !touched(&reads, &meta, 3),
            "row group 3 does not fit beside it"
        );
        let batch = reader.batches.first().expect("a batch");
        assert_eq!(batch.row_groups.len(), 1);
    }

    #[tokio::test]
    async fn the_next_batch_starts_only_once_the_last_one_is_read_from() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        assert!(!touched(&superfile.reads(), &meta, 4));
        read_row_groups(&mut reader, &meta, &[2]).await;
        settle(&reader).await;
        let reads = superfile.reads();
        assert!(touched(&reads, &meta, 4) && touched(&reads, &meta, 5));
        assert!(!touched(&reads, &meta, 6));
    }

    /// Row groups with the given `(start, end)` chunks, for [`fetch_spans`].
    fn with_chunks(rgs: &[&[(u64, u64)]]) -> Vec<(usize, Vec<Range<u64>>)> {
        rgs.iter()
            .enumerate()
            .map(|(rg, chunks)| (rg, chunks.iter().map(|&(s, e)| s..e).collect()))
            .collect()
    }

    /// The spans of each cluster as `(start, end)`.
    fn clusters_of(row_groups: &[(usize, Vec<Range<u64>>)]) -> Vec<Vec<(u64, u64)>> {
        fetch_spans(row_groups, Some(BLOCKS))
            .iter()
            .map(|cluster| cluster.iter().map(|r| (r.start, r.end)).collect())
            .collect()
    }

    #[test]
    fn a_batch_fetches_what_each_row_group_would_alone_and_joins_what_touches() {
        // Two row groups whose chunks touch at the boundary: one span.
        let touching = with_chunks(&[&[(0, 10), (10, 20)], &[(20, 30)]]);
        assert_eq!(clusters_of(&touching), vec![vec![(0, 30)]]);
        // The second span is inside the first one's 512 KiB window:
        //  - one cluster, downloaded in turn, so shared bytes come down once;
        //  - the gap between them isn't asked for.
        let close = with_chunks(&[&[(0, 10)], &[(MIN_GET - 10, MIN_GET)]]);
        assert_eq!(
            clusters_of(&close),
            vec![vec![(0, 10), (MIN_GET - 10, MIN_GET)]]
        );
        // Just past it, but the second span's window may reach back into it:
        // still one cluster.
        let behind = with_chunks(&[&[(0, 10)], &[(MIN_GET, MIN_GET + 10)]]);
        assert_eq!(clusters_of(&behind).len(), 1);
        // Out of reach of both windows: two clusters, downloaded in parallel.
        let far = 2 * MIN_GET - BLOCK;
        let apart = with_chunks(&[&[(0, 10)], &[(far, far + 10)]]);
        assert_eq!(
            clusters_of(&apart),
            vec![vec![(0, 10)], vec![(far, far + 10)]]
        );
        // A long span downloads just its own blocks, so the next span is
        // separate once it starts past the block the first one ends in.
        let long = with_chunks(&[&[(0, 2 * MIN_GET)], &[(2 * MIN_GET + BLOCK, 3 * MIN_GET)]]);
        assert_eq!(clusters_of(&long).len(), 2);
        let shared = with_chunks(&[&[(0, 2 * MIN_GET - 10)], &[(2 * MIN_GET - 5, 3 * MIN_GET)]]);
        assert_eq!(clusters_of(&shared).len(), 1);
    }

    #[test]
    fn a_batch_reserves_what_the_cache_downloads_for_it() {
        //  - a short span downloads a whole 512 KiB window;
        //  - two overlapping downloads count the overlap once;
        //  - a long span downloads just its blocks.
        let short = vec![vec![10..20]];
        assert_eq!(download_bytes(&short, BLOCKS), MIN_GET);
        let overlapping = vec![vec![10..20, BLOCK + 10..BLOCK + 20]];
        assert_eq!(download_bytes(&overlapping, BLOCKS), MIN_GET + BLOCK);
        let long = vec![vec![BLOCK + 10..3 * MIN_GET]];
        assert_eq!(download_bytes(&long, BLOCKS), 3 * MIN_GET - BLOCK);
        // A short span away from the file's start can reach either way, but
        // still downloads one window.
        let away = vec![vec![MIN_GET + 10..MIN_GET + 20]];
        assert_eq!(download_bytes(&away, BLOCKS), MIN_GET);
    }

    #[test]
    fn a_batch_without_a_cache_fetches_every_span_in_parallel() {
        let close = with_chunks(&[&[(0, 10)], &[(100, 110)]]);
        assert_eq!(fetch_spans(&close, None).len(), 2);
    }

    #[test]
    fn a_batch_keeps_clear_of_what_other_reads_may_download_at_its_ends() {
        let spans = |clusters: Vec<Vec<Range<u64>>>| -> Vec<Vec<(u64, u64)>> {
            clusters
                .iter()
                .map(|c| c.iter().map(|r| (r.start, r.end)).collect())
                .collect()
        };
        // Offsets in blocks from a block-aligned base.
        let base = 10 * MIN_GET;
        let at = |blocks: u64| base + blocks * BLOCK;
        let batch = || {
            vec![
                vec![at(0)..at(16)],
                vec![at(20)..at(30)],
                vec![at(36)..at(36) + 100],
            ]
        };
        // No other read near: fetched whole.
        assert_eq!(
            spans(leave_shared_ends(batch(), BLOCKS, &[])),
            spans(batch())
        );
        // Short chunks just before and just after the batch:
        //  - the batch starts where the chunk before can reach to;
        //  - it stops where the chunk after can reach back to;
        //  - its short last span would download into that, so it's left to
        //    the scan.
        let before = base - 100..base - 50;
        let after = at(40)..at(40) + 100;
        assert_eq!(BLOCKS.reach(&before).end, at(7));
        assert_eq!(BLOCKS.reach(&after).start, at(33));
        assert_eq!(
            spans(leave_shared_ends(batch(), BLOCKS, &[before.clone(), after])),
            vec![vec![(at(7), at(16))], vec![(at(20), at(30))]]
        );
        // A long chunk after the batch reaches only its own blocks, so the
        // batch keeps everything. A short one reaches back into the batch's
        // last span, so that span is left to the scan.
        let long_last = || vec![vec![at(0)..at(16)], vec![at(20)..at(30)]];
        let long_after = at(34)..at(42);
        assert_eq!(
            spans(leave_shared_ends(long_last(), BLOCKS, &[long_after])),
            spans(long_last())
        );
        let short_after = at(34)..at(34) + 100;
        assert_eq!(
            spans(leave_shared_ends(long_last(), BLOCKS, &[short_after])),
            vec![vec![(at(0), at(16))]]
        );
        // A short first span would download back into what the chunk before
        // may download, so it's left to the scan and the next span starts the
        // batch.
        let short_first = || {
            vec![
                vec![at(8)..at(8) + 100, at(9)..at(20)],
                vec![at(28)..at(36)],
            ]
        };
        assert_eq!(
            spans(leave_shared_ends(short_first(), BLOCKS, &[before])),
            vec![vec![(at(9), at(20))], vec![(at(28), at(36))]]
        );
        // With nothing before the batch, a short first span stays.
        assert_eq!(
            spans(leave_shared_ends(short_first(), BLOCKS, &[])),
            spans(short_first())
        );
    }

    /// `read_through_spans` over `pages` of a file whose column chunks are
    /// `chunks` and whose cache holds `local`.
    fn read_through(
        pages: &[Range<u64>],
        chunks: &[Range<u64>],
        local: &[Range<u64>],
    ) -> Option<Vec<Range<u64>>> {
        read_through_spans(
            pages,
            |gap| {
                chunks
                    .iter()
                    .any(|chunk| chunk.start <= gap.start && gap.end <= chunk.end)
            },
            |span| {
                !local
                    .iter()
                    .any(|held| held.start < span.end && span.start < held.end)
            },
        )
    }

    /// A page starting `gap` bytes after `before` ends.
    fn page_after(before: &Range<u64>, gap: u64) -> Range<u64> {
        let start = before.end + gap;
        start..start + PAGE
    }

    #[test]
    fn cold_kept_pages_of_one_chunk_a_little_apart_are_read_through() {
        let first = 0..PAGE;
        for gap in [OBJECT_STORE_COALESCE_DEFAULT + 1, PAGE_READ_THROUGH_BYTES] {
            let next = page_after(&first, gap);
            let both = first.start..next.end;
            assert_eq!(
                read_through(&[first.clone(), next.clone()], from_ref(&both), &[]),
                Some(vec![both]),
                "a gap of {gap} bytes is read through"
            );
        }
    }

    #[test]
    fn every_other_gap_is_left_to_the_store() {
        let first = 0..PAGE;
        let near = page_after(&first, OBJECT_STORE_COALESCE_DEFAULT + 1);
        let chunk = first.start..u64::MAX;
        let split = first.end + OBJECT_STORE_COALESCE_DEFAULT / 2;
        let island = split..split + PAGE;
        let cases = [
            (
                "the store merges it already",
                page_after(&first, OBJECT_STORE_COALESCE_DEFAULT),
                vec![chunk.clone()],
                vec![],
            ),
            (
                "it costs more than a GET",
                page_after(&first, PAGE_READ_THROUGH_BYTES + 1),
                vec![chunk.clone()],
                vec![],
            ),
            (
                "the gap crosses into another column's chunk",
                near.clone(),
                vec![0..split, split..u64::MAX],
                vec![],
            ),
            ("outside every chunk", near.clone(), vec![], vec![]),
            (
                "the first page is local",
                near.clone(),
                vec![chunk.clone()],
                vec![first.clone()],
            ),
            (
                "the next page is local",
                near.clone(),
                vec![chunk.clone()],
                vec![near.clone()],
            ),
            (
                "the cache holds bytes inside the gap",
                near,
                vec![chunk],
                vec![island],
            ),
        ];
        for (name, next, chunks, local) in cases {
            assert_eq!(
                read_through(&[first.clone(), next], &chunks, &local),
                None,
                "{name}"
            );
        }
    }

    #[test]
    fn read_through_keeps_the_stores_merges_in_any_order() {
        // p0 and p1 merge as the store would, p2 is read through after them,
        // and the other column's page far away stays a span of its own.
        let p0 = 0..PAGE;
        let p1 = page_after(&p0, PAGE);
        let p2 = page_after(&p1, OBJECT_STORE_COALESCE_DEFAULT + 1);
        let other = page_after(&p2, PAGE_READ_THROUGH_BYTES + 1);
        let pages = [p2.clone(), other.clone(), p0.clone(), p1];
        let chunks = [p0.start..p2.end, other.clone()];
        assert_eq!(
            read_through(&pages, &chunks, &[]),
            Some(vec![p0.start..p2.end, other])
        );
    }

    #[test]
    fn each_range_is_cut_from_its_span_or_fails_when_one_came_back_short() {
        let bytes = Bytes::from(
            (0..=u8::MAX)
                .cycle()
                .take(4 * PAGE as usize)
                .collect::<Vec<_>>(),
        );
        let spans = [0..2 * PAGE, 3 * PAGE..4 * PAGE];
        let fetched: Vec<Bytes> = spans
            .iter()
            .map(|span| bytes.slice(span.start as usize..span.end as usize))
            .collect();
        let ranges = [PAGE..2 * PAGE, 0..PAGE / 2, 3 * PAGE..4 * PAGE];
        let cut_out = cut(&ranges, &spans, &fetched).expect("every range is served");
        for (range, got) in ranges.iter().zip(&cut_out) {
            assert_eq!(got, &bytes.slice(range.start as usize..range.end as usize));
            assert!(
                !bytes.as_ptr_range().contains(&got.as_ptr()),
                "a copy, so the span can be dropped"
            );
        }
        let short = [fetched[0].clone(), fetched[1].slice(..PAGE as usize - 1)];
        assert!(cut(&ranges, &spans, &short).is_err());
    }

    #[tokio::test]
    async fn a_range_with_any_cached_block_is_not_all_cold() {
        // One short read leaves a window of blocks cached in the middle of a
        // range whose ends are still cold.
        let (bytes, _) = wide_file(WIDE_ROWS, WIDE_ROWS);
        let island = OBJECT_STORE_COALESCE_DEFAULT * 3;
        let range = 0..island + OBJECT_STORE_COALESCE_DEFAULT;
        assert!(bytes.len() as u64 >= range.end, "the file holds the range");
        let superfile = Superfile::new(bytes, false);
        let dir = tempfile::tempdir().expect("tempdir");
        let (_cache, source) = block_cached(&superfile, dir.path());
        source.range(island, PAGE).await.expect("warm");
        let path = ObjPath::from("t/sf.parquet");
        let store = SuperfileObjectStore::from_sources(HashMap::from([(path.clone(), source)]));
        assert!(!store.is_local(&path, &range), "the ends are cold");
        assert!(!all_cold(&store, &path, &range), "the middle is cached");
        assert!(all_cold(&store, &path, &(0..OBJECT_STORE_COALESCE_DEFAULT)));
    }

    #[tokio::test]
    async fn a_single_read_never_looks_ahead() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[3]).await;
        settle(&reader).await;
        assert_eq!(budget.used(), 0);
        assert!(!touched(&superfile.reads(), &meta, 4));
    }

    #[tokio::test]
    async fn a_read_narrowed_to_some_pages_does_not_look_ahead() {
        let (_superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0]).await;
        let half: Vec<_> = chunks(&meta, 1)
            .into_iter()
            .map(|r| r.start..r.start + (r.end - r.start) / 2)
            .collect();
        reader.get_byte_ranges(half).await.expect("rg 1 pages");
        assert_eq!(budget.used(), 0);
    }

    #[tokio::test]
    async fn a_read_taking_most_of_each_chunk_still_looks_ahead() {
        let (_superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0]).await;
        // Two page reads per chunk that skip a tenth of it.
        let pages: Vec<_> = chunks(&meta, 1)
            .into_iter()
            .flat_map(|r| {
                let len = r.end - r.start;
                [
                    r.start..r.start + len / 2,
                    r.start + len / 2 + len / 10..r.end,
                ]
            })
            .collect();
        reader.get_byte_ranges(pages).await.expect("rg 1 pages");
        assert!(budget.used() > 0);
    }

    #[tokio::test]
    async fn one_narrowed_chunk_is_enough_to_not_look_ahead() {
        let (_superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0]).await;
        let [a, c]: [Range<u64>; 2] = chunks(&meta, 1).try_into().expect("two chunks");
        reader
            .get_byte_ranges(vec![a, c.start..c.start + (c.end - c.start) / 2])
            .await
            .expect("rg 1");
        assert_eq!(budget.used(), 0);
    }

    #[tokio::test]
    async fn row_groups_the_plan_skips_are_not_fetched_and_keep_the_order() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut plan = ParquetAccessPlan::new_all(GROUPS);
        plan.skip(2);
        let mut reader = scan(&store, &budget, &[]).reader(
            partitioned(&path, &meta, Some(plan), None),
            Arc::clone(&meta),
        );
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        let reads = superfile.reads();
        assert!(!touched(&reads, &meta, 2));
        assert!(touched(&reads, &meta, 3) && touched(&reads, &meta, 4));

        // 1 then 3 is in order once 2 is skipped: 3 is served, not fetched.
        superfile.clear();
        read_row_groups(&mut reader, &meta, &[3]).await;
        settle(&reader).await;
        assert!(!touched(&superfile.reads(), &meta, 3));
    }

    #[tokio::test]
    async fn a_row_group_read_in_part_is_left_to_the_scan_and_not_fetched_past() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut plan = ParquetAccessPlan::new_all(GROUPS);
        plan.scan_selection(
            2,
            RowSelection::from(vec![
                RowSelector::select(1),
                RowSelector::skip(ROWS_PER_GROUP - 1),
            ]),
        );
        let mut reader = scan(&store, &budget, &[]).reader(
            partitioned(&path, &meta, Some(plan), None),
            Arc::clone(&meta),
        );
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        let reads = superfile.reads();
        assert!(
            !touched(&reads, &meta, 2),
            "one selected row: the scan reads its pages"
        );
        assert!(
            !touched(&reads, &meta, 3),
            "fetching past 2 would race the scan's own read of it"
        );
        // Once the scan has read 2 itself, the next row groups follow.
        read_row_groups(&mut reader, &meta, &[2]).await;
        settle(&reader).await;
        assert!(touched(&superfile.reads(), &meta, 3));
    }

    #[tokio::test]
    async fn row_groups_the_statistics_prune_are_not_fetched() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        // `a < 2000` keeps row groups 0 and 1 only.
        let kept = (2 * ROWS_PER_GROUP) as i64;
        let mut reader = reader_of(
            &scan(&store, &budget, &[col("a").lt(lit(kept))]),
            &path,
            &meta,
        );
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        assert_eq!(budget.used(), 0);
        assert!(!touched(&superfile.reads(), &meta, 2));
    }

    #[tokio::test]
    async fn a_partition_reads_ahead_only_within_its_byte_range() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        // This partition holds row groups 0 to 2: the rest start at or past its end.
        let range = FileRange {
            start: 0,
            end: row_group_span(meta.row_group(3)).start as i64,
        };
        let mut reader = scan(&store, &budget, &[]).reader(
            partitioned(&path, &meta, None, Some(range)),
            Arc::clone(&meta),
        );
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        let reads = superfile.reads();
        assert!(touched(&reads, &meta, 2));
        assert!(
            !touched(&reads, &meta, 3),
            "row group 3 is another partition's"
        );
    }

    #[tokio::test]
    async fn leaving_the_planned_order_turns_it_off_and_lets_go() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        assert!(budget.used() > 0, "row groups 2 and 3 are held");
        // The scan skipped 2 to 4 for a reason this reader can't see.
        read_row_groups(&mut reader, &meta, &[5]).await;
        assert_eq!(budget.used(), 0, "nothing held after leaving the order");
        superfile.clear();
        read_row_groups(&mut reader, &meta, &[6]).await;
        settle(&reader).await;
        assert!(
            !touched(&superfile.reads(), &meta, 7),
            "off for the rest of the file"
        );
    }

    #[tokio::test]
    async fn a_scan_turned_off_never_looks_ahead() {
        let (_superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let scan = scan(&store, &budget, &[]);
        scan.disable();
        let mut reader = reader_of(&scan, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1, 2]).await;
        assert_eq!(budget.used(), 0);
    }

    #[tokio::test]
    async fn a_local_file_is_never_planned() {
        let (_superfile, store, path, meta) = setup(true);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1, 2]).await;
        assert_eq!(
            budget.used(),
            0,
            "nothing reserved, so nothing was fetched ahead"
        );
        assert!(reader.order.is_none(), "a warm file is never planned");
    }

    #[tokio::test]
    async fn nothing_is_fetched_ahead_while_the_cache_cannot_keep_it() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        superfile.keeps.store(false, Ordering::Relaxed);
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        assert!(
            !touched(&superfile.reads(), &meta, 2),
            "no batch while it cannot"
        );
        // Room again: the next row group reached in order looks ahead.
        superfile.keeps.store(true, Ordering::Relaxed);
        read_row_groups(&mut reader, &meta, &[2]).await;
        settle(&reader).await;
        assert!(touched(&superfile.reads(), &meta, 3));
    }

    #[tokio::test]
    async fn a_fetch_that_would_pass_the_budget_share_is_not_made() {
        let (_superfile, store, path, meta) = setup(false);
        // The budget has room for one batch, but read-ahead's half of it
        // doesn't.
        let budget = ConnectionMemoryBudget::with_limit(charge(&meta, &[2, 3]) as u64 * 3 / 2);
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1, 2]).await;
        assert_eq!(budget.used(), 0);
    }

    #[tokio::test]
    async fn bytes_held_ahead_are_charged_and_given_back() {
        let (_superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        assert_eq!(budget.used(), charge(&meta, &[2, 3]));
        // Reading 2 leaves 3 held and fetches 4 and 5 ahead.
        read_row_groups(&mut reader, &meta, &[2]).await;
        assert_eq!(
            budget.used(),
            charge(&meta, &[2, 3]) + charge(&meta, &[4, 5])
        );
        // Reading 3 empties the first batch, which gives its bytes back.
        read_row_groups(&mut reader, &meta, &[3]).await;
        assert_eq!(budget.used(), charge(&meta, &[4, 5]));
        drop(reader);
        assert_eq!(budget.used(), 0);
    }

    #[tokio::test]
    async fn a_reader_dropped_while_its_batch_downloads_gives_the_bytes_back_at_once() {
        let (_superfile, store, path, meta) = setup_slow();
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        assert!(
            reader.batches.iter().any(Batch::is_running),
            "the batch is still downloading"
        );
        assert!(budget.used() > 0);
        drop(reader);
        assert_eq!(
            budget.used(),
            0,
            "given back without waiting for the download"
        );
    }

    // -- Giving memory back --------------------------------------------------

    /// A budget this many batches big: read-ahead's half of it fits one batch,
    /// with room to spare.
    const BUDGET_BATCHES: u64 = 4;
    /// Rows per row group in a file big enough that the cache downloads more
    /// than a batch's spans.
    const WIDE_ROWS: usize = 20_000;
    /// Bytes on each side of where a batch's bytes start, in the short-gap
    /// test.
    const GAP_BYTES: u64 = 10;

    /// The bytes of `rg`'s projected chunks as `reader` serves them, checked
    /// against the file.
    async fn read_and_check(reader: &mut ReadAhead, superfile: &Superfile, rg: usize) {
        let meta = Arc::clone(&reader.meta);
        let got = reader
            .get_byte_ranges(chunks(&meta, rg))
            .await
            .expect("read");
        assert_eq!(got, superfile.slices(&chunks(&meta, rg)), "row group {rg}");
    }

    #[tokio::test]
    async fn a_reservation_that_does_not_fit_takes_back_what_read_ahead_holds() {
        // A batch is held when something else, a join's build side say, wants
        // more than is left.
        let (superfile, store, path, meta) = setup(false);
        let batch = charge(&meta, &[2, 3]);
        let budget = ConnectionMemoryBudget::with_limit(batch as u64 * BUDGET_BATCHES);
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        assert_eq!(budget.used(), batch);
        let wanted = budget.limit().expect("bounded") - batch / 2;
        let build = budget
            .try_reserve(wanted)
            .expect("fits once read-ahead gives its batch back");
        assert_eq!(budget.used(), wanted, "nothing held ahead");

        // The scan reads 2 and 3 itself, the same bytes. Nothing more is read
        // ahead for the rest of the scan, even once the memory is free again.
        drop(build);
        superfile.clear();
        read_and_check(&mut reader, &superfile, 2).await;
        read_and_check(&mut reader, &superfile, 3).await;
        settle(&reader).await;
        let reads = superfile.reads();
        assert!(touched(&reads, &meta, 2) && touched(&reads, &meta, 3));
        assert!(!touched(&reads, &meta, 4), "off for the rest of the scan");
        assert!(reader.batches.is_empty(), "no batch even set up");
        assert_eq!(budget.used(), 0);
    }

    #[tokio::test]
    async fn a_batch_given_back_while_downloading_frees_its_memory_at_once() {
        let (superfile, store, path, meta) = setup_slow();
        let budget = ConnectionMemoryBudget::measured();
        let scan = scan(&store, &budget, &[]);
        let mut reader = reader_of(&scan, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        assert!(reader.batches.iter().any(Batch::is_running));
        scan.reclaim();
        assert_eq!(budget.used(), 0, "without waiting for the download");
        read_and_check(&mut reader, &superfile, 2).await;
        read_and_check(&mut reader, &superfile, 3).await;
    }

    #[tokio::test]
    async fn a_batch_given_back_after_part_of_it_was_read_leaves_the_rest_to_the_scan() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let scan = scan(&store, &budget, &[]);
        let mut reader = reader_of(&scan, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        // 2 comes from memory, and starts 4 and 5 ahead.
        read_and_check(&mut reader, &superfile, 2).await;
        settle(&reader).await;
        scan.reclaim();
        assert_eq!(budget.used(), 0);
        superfile.clear();
        read_and_check(&mut reader, &superfile, 3).await;
        read_and_check(&mut reader, &superfile, 4).await;
        assert!(touched(&superfile.reads(), &meta, 3), "3 read again");
    }

    #[tokio::test]
    async fn a_batch_set_up_while_read_ahead_turns_off_is_given_back() {
        // A reader reserves its batch, everything is taken back before it
        // tracks the batch, and then it tracks it.
        let (_superfile, store, _path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let scan = scan(&store, &budget, &[]);
        let hold = scan.reserve(charge(&meta, &[2, 3])).expect("room");
        let download = spawn(async {});
        let held = Arc::new(Mutex::new(Held {
            bytes: None,
            hold: Some(hold),
            download: download.abort_handle(),
        }));
        scan.reclaim();
        scan.track(&held);
        assert!(lock(&held).hold.is_none());
        assert_eq!(budget.used(), 0);
    }

    #[tokio::test]
    async fn every_scan_on_the_connection_gives_back() {
        let (_a, store_a, path_a, meta) = setup(false);
        let (_b, store_b, path_b, _) = setup(false);
        let batch = charge(&meta, &[2, 3]);
        let budget = ConnectionMemoryBudget::with_limit(batch as u64 * BUDGET_BATCHES * 2);
        let mut a = plain_reader(&store_a, &budget, &path_a, &meta);
        let mut b = plain_reader(&store_b, &budget, &path_b, &meta);
        read_row_groups(&mut a, &meta, &[0, 1]).await;
        read_row_groups(&mut b, &meta, &[0, 1]).await;
        assert_eq!(budget.used(), batch * 2);
        let wanted = budget.limit().expect("bounded") - batch / 2;
        let _build = budget
            .try_reserve(wanted)
            .expect("fits once both give back");
        assert_eq!(budget.used(), wanted);
    }

    #[tokio::test]
    async fn a_batch_is_charged_what_the_cache_downloads_for_it() {
        // The cache downloads whole blocks, and a read of a whole download is
        // served as a slice of it, which keeps all of it alive.
        let (bytes, meta) = wide_file(WIDE_ROWS, WIDE_ROWS);
        let superfile = Superfile::new(bytes, false);
        let dir = tempfile::tempdir().expect("tempdir");
        let (_cache, source) = block_cached(&superfile, dir.path());
        let path = ObjPath::from("t/sf.parquet");
        let store = Arc::new(SuperfileObjectStore::from_sources(HashMap::from([(
            path.clone(),
            source,
        )])));
        let budget = ConnectionMemoryBudget::measured();
        // The wide file's row groups are bigger than the small file's, so
        // the cap that pins two of them is its own.
        let scan = Arc::new(
            ScanReadAhead::new(Arc::clone(&store), Arc::clone(&budget), schema())
                .with_reader_cap(two_row_groups_cap_for(&meta)),
        );
        let mut reader = reader_of(&scan, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        let batch = reader.batches.first().expect("a batch");
        let blocks = store.cache_blocks(&path).expect("cached");
        let downloads = download_bytes(from_ref(&batch.spans), blocks) as usize;
        assert!(downloads > batch.charge, "the cache downloads more");
        assert_eq!(budget.used(), downloads);
    }

    #[tokio::test]
    async fn a_gap_that_comes_back_short_is_an_error_not_shifted_bytes() {
        // A request that starts before what the batch holds: a gap, then held
        // bytes. Superfiles never change, so a short gap can't happen today.
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        let held = reader.batches[0].spans[0].start;
        let gap = held - GAP_BYTES;
        *superfile.short_at.lock().expect("short_at") = Some(gap);
        let request = gap..held + GAP_BYTES;
        reader
            .serve(2, 0, from_ref(&request))
            .await
            .expect_err("a short gap");
    }

    #[tokio::test]
    async fn a_request_partly_fetched_ahead_gets_the_rest_from_the_store() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        superfile.clear();
        let [a, c]: [Range<u64>; 2] = chunks(&meta, 2).try_into().expect("two chunks");
        // Bytes past the batch (row groups 2 and 3): from the end of row group
        // 3's last projected column into row group 4.
        let batch_end = chunks(&meta, 3)[1].end;
        let unread = batch_end..batch_end + 64;
        // Part of a, part of c, and bytes the fetch did not cover.
        let ranges = vec![a.start + 1..a.end - 1, c.start..c.end - 1, unread.clone()];
        let got = reader.get_byte_ranges(ranges.clone()).await.expect("rg 2");
        assert_eq!(got, superfile.slices(&ranges));
        assert_eq!(
            superfile.reads(),
            vec![unread],
            "only the uncovered range is read"
        );
    }

    #[tokio::test]
    async fn a_failed_fetch_ahead_is_fetched_again() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        *superfile.fail_at.lock().expect("fail_at") = Some(chunks(&meta, 2)[0].start);
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        let got = reader
            .get_byte_ranges(chunks(&meta, 2))
            .await
            .expect("rg 2");
        assert_eq!(got, superfile.slices(&chunks(&meta, 2)));
    }

    #[tokio::test]
    async fn page_reads_count_the_same_with_or_without_read_ahead() {
        let mut counts = Vec::new();
        for resident in [false, true] {
            let (_superfile, store, path, meta) = setup(resident);
            let budget = ConnectionMemoryBudget::measured();
            let mut reader = plain_reader(&store, &budget, &path, &meta);
            read_row_groups(&mut reader, &meta, &(0..GROUPS).collect::<Vec<_>>()).await;
            let stats = store.op_stats().expect("collector").snapshot();
            counts.push((stats.sql_page_bytes, stats.planned_read_ranges));
        }
        assert_eq!(
            counts[0], counts[1],
            "cold (read ahead) and local count alike"
        );
    }

    #[tokio::test]
    async fn two_reads_of_the_first_row_group_keep_the_plan() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut plan = ParquetAccessPlan::new_all(GROUPS);
        plan.skip(2);
        let mut reader = scan(&store, &budget, &[]).reader(
            partitioned(&path, &meta, Some(plan), None),
            Arc::clone(&meta),
        );
        // A row filter reads the filter's column first, then the rest.
        let [a, c]: [Range<u64>; 2] = chunks(&meta, 0).try_into().expect("two chunks");
        reader.get_byte_ranges(vec![a]).await.expect("rg 0, a");
        reader.get_byte_ranges(vec![c]).await.expect("rg 0, c");
        read_row_groups(&mut reader, &meta, &[1]).await;
        settle(&reader).await;
        let reads = superfile.reads();
        assert!(!touched(&reads, &meta, 2), "the plan skips row group 2");
        assert!(touched(&reads, &meta, 3) && touched(&reads, &meta, 4));
    }

    #[tokio::test]
    async fn a_first_read_outside_every_row_group_does_not_count() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        let end = row_group_span(meta.row_group(GROUPS - 1)).end;
        reader
            .get_bytes(end..end + 8)
            .await
            .expect("bytes past the last row group");
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        assert!(touched(&superfile.reads(), &meta, 2));
    }

    #[tokio::test]
    async fn leaving_the_order_onto_a_held_row_group_serves_it() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let mut reader = plain_reader(&store, &budget, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        superfile.clear();
        // The scan skipped 2 for a reason this reader can't see; 3 is held.
        let got = reader
            .get_byte_ranges(chunks(&meta, 3))
            .await
            .expect("rg 3");
        assert_eq!(got, superfile.slices(&chunks(&meta, 3)));
        assert!(!touched(&superfile.reads(), &meta, 3), "served from memory");
        assert_eq!(budget.used(), 0, "off: nothing else held");
    }

    #[tokio::test]
    async fn statistics_pruning_reads_the_files_own_columns() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        // The table dropped `b`, which this file still holds between `a` and `c`.
        let table = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("c", DataType::Int64, false),
        ]));
        // `c < 7 * 2000` keeps row groups 0 and 1 only.
        let kept = (7 * 2 * ROWS_PER_GROUP) as i64;
        let scan = scan_over(&store, &budget, &[col("c").lt(lit(kept))], table);
        let mut reader = reader_of(&scan, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
        assert_eq!(budget.used(), 0);
        assert!(!touched(&superfile.reads(), &meta, 2));
    }

    #[tokio::test]
    async fn a_filter_on_a_column_added_after_the_file_was_written_plans_the_file() {
        let (_superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        // The table added `d` after this file was written; the file has no `d`.
        let table = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Int64, false),
            Field::new("d", DataType::Int64, true),
        ]));
        let scan = scan_over(
            &store,
            &budget,
            &[col("d").is_null().or(col("a").gt(lit(5i64)))],
            table,
        );
        let mut reader = reader_of(&scan, &path, &meta);
        read_row_groups(&mut reader, &meta, &[0, 1]).await;
        settle(&reader).await;
    }

    #[tokio::test]
    async fn a_dynamic_filter_prunes_with_its_values_when_the_file_is_planned() {
        let (superfile, store, path, meta) = setup(false);
        let budget = ConnectionMemoryBudget::measured();
        let scan = Arc::new(ScanReadAhead::new(
            Arc::clone(&store),
            Arc::clone(&budget),
            schema(),
        ));
        // A join's filter on `a`: nothing known when the plan is made.
        let a: Arc<dyn PhysicalExpr> = Arc::new(Column::new("a", 0));
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::clone(&a)],
            predicate(&lit(true), schema()),
        ));
        scan.set_predicate(Arc::clone(&dynamic) as Arc<dyn PhysicalExpr>, None);
        // A file planned before the filter knows anything prunes nothing.
        let mut early = reader_of(&scan, &path, &meta);
        read_row_groups(&mut early, &meta, &[0, 1]).await;
        settle(&early).await;
        assert!(touched(&superfile.reads(), &meta, 2), "nothing pruned yet");
        drop(early);
        // Once the join's build side fills in `a`'s range (row groups 0 and 1
        // only), a file planned after that prunes with it.
        let kept = (2 * ROWS_PER_GROUP) as i64;
        dynamic
            .update(predicate(&col("a").lt(lit(kept)), schema()))
            .expect("update");
        superfile.clear();
        let mut late = reader_of(&scan, &path, &meta);
        read_row_groups(&mut late, &meta, &[0, 1]).await;
        settle(&late).await;
        assert!(
            !touched(&superfile.reads(), &meta, 2),
            "the filter prunes 2"
        );
        drop(late);
        assert_eq!(budget.used(), 0);
    }
}
