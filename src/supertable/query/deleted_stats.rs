// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A superfile's statistics as its live rows have them.
//!
//! A superfile's statistics describe every row it was written with, deleted
//! ones included. An aggregate answered from them has to answer for the live
//! rows alone, so for a superfile with deletes each statistic is corrected
//! by the deleted rows, exactly, or the superfile is scanned:
//!
//!   * from the count of deleted rows alone, with no row read: a superfile
//!     whose rows are all deleted holds nothing; a column with no nulls
//!     holds a value in every live row, one null throughout in none; a
//!     column holding one value has it in every live row;
//!   * from the deleted rows' own statistics ([`ScalarStatsAgg`] over just
//!     those rows, read by row position), taken off the stored ones: a
//!     column's non-null count, an integer sum, its value counts, and a
//!     bound that no deleted row held. A float sum is never corrected: the
//!     difference of two float sums is not the sum of the rest. A bound a
//!     deleted row may have held is exact only through the value counts;
//!     otherwise the superfile still bounds the column from one side, which
//!     answers when another superfile's exact bound beats it.
//!
//! The deleted rows' statistics are kept per superfile and column in a
//! [`DeletedStatsCache`] on the table, keyed by the exact rows they cover. A
//! superfile's deletes only grow, so a later read takes only the rows
//! deleted since. Reading them costs the pages those rows sit on; when the
//! live rows sit on fewer, the superfile is scanned instead.

use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashMap, HashSet},
    sync::{Arc, Mutex},
};

use arrow::{
    compute::{cast, concat},
    datatypes::DataType,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::Schema;
use datafusion::{
    catalog::{Session, TableProvider},
    common::ScalarValue,
    error::{DataFusionError, Result as DfResult},
    physical_plan::collect,
};
use roaring::RoaringBitmap;
use tracing::debug;
use uuid::Uuid;

use crate::{
    runtime_bridge::run_on_pool,
    supertable::{
        manifest::{ColumnStats, ScalarStatsAgg, SuperfileEntry, sub_sum_arrays},
        query::{provider::SupertableProvider, value_count_carry::take_off},
        schema::FieldId,
    },
    utils::join::try_join_bounded,
};

/// Deleted-row statistics kept per table, at most. Each is one column's
/// bounds, counts and value table (at most the per-superfile cap of distinct
/// values) for one superfile.
const MAX_CACHED: usize = 65_536;

/// Bytes the kept statistics may take per table, about: their value tables,
/// and the deleted rows they cover, each superfile's counted once however
/// many of its columns share them.
const MAX_CACHED_BYTES: usize = 64 * 1024 * 1024;

/// Past either bound, the statistics used least recently are dropped until
/// the cache holds this share of it, so a run of inserts drops once rather
/// than on every insert.
const EVICT_TO: f64 = 0.75;

/// Statistics kept before a statement drops those of superfiles its
/// snapshot no longer lists.
const PRUNE_ABOVE: usize = MAX_CACHED / 2;

/// Share of a superfile's rows that may be read to correct its statistics
/// when its file carries no page index to weigh the read against the scan.
const MAX_READ_SHARE_WITHOUT_PAGES: f64 = 0.5;

/// The statistics of superfiles' deleted rows, per superfile and column,
/// shared by every reader of a table.
#[derive(Debug, Default)]
pub(crate) struct DeletedStatsCache {
    entries: Mutex<Entries>,
}

#[derive(Debug, Default)]
struct Entries {
    by_column: HashMap<(Uuid, FieldId), Kept>,
    /// The deleted rows the kept statistics cover, per superfile and bitmap
    /// (by address, as the entries holding it keep it alive): its bytes and
    /// how many entries hold it. A superfile's columns read in one statement
    /// share one bitmap, counted once.
    rows: HashMap<(Uuid, usize), (usize, usize)>,
    /// About the bytes the value tables and the rows hold.
    bytes: usize,
    /// A count of uses, the latest stamped on each entry used.
    clock: u64,
}

#[derive(Debug)]
struct Kept {
    cached: CachedRows,
    last_used: u64,
}

/// One column's statistics over one superfile's deleted rows.
#[derive(Debug, Clone)]
struct CachedRows {
    /// The type the rows were read as; a column the table now reads as
    /// another type starts over.
    data_type: DataType,
    /// The deleted rows the statistics cover.
    rows: Arc<RoaringBitmap>,
    stats: Arc<ScalarStatsAgg>,
}

impl CachedRows {
    /// About the bytes the entry's value table holds.
    fn value_bytes(&self) -> usize {
        self.stats
            .value_counts
            .as_ref()
            .map_or(0, |counts| counts.value_bytes())
    }
}

impl Entries {
    /// The entry for `superfile`'s column `field`, stamped as used now.
    fn used(&mut self, superfile: Uuid, field: FieldId) -> Option<&CachedRows> {
        self.clock += 1;
        let clock = self.clock;
        let kept = self.by_column.get_mut(&(superfile, field))?;
        kept.last_used = clock;
        Some(&kept.cached)
    }

    fn insert(&mut self, superfile: Uuid, field: FieldId, cached: CachedRows) {
        self.remove(superfile, field);
        self.bytes += cached.value_bytes();
        let (bytes, users) = self
            .rows
            .entry((superfile, Arc::as_ptr(&cached.rows) as usize))
            .or_insert_with(|| (cached.rows.serialized_size(), 0));
        if *users == 0 {
            self.bytes += *bytes;
        }
        *users += 1;
        self.clock += 1;
        self.by_column.insert(
            (superfile, field),
            Kept {
                cached,
                last_used: self.clock,
            },
        );
    }

    fn remove(&mut self, superfile: Uuid, field: FieldId) {
        let Some(Kept { cached, .. }) = self.by_column.remove(&(superfile, field)) else {
            return;
        };
        self.bytes -= cached.value_bytes();
        let key = (superfile, Arc::as_ptr(&cached.rows) as usize);
        if let Some((bytes, users)) = self.rows.get_mut(&key) {
            *users -= 1;
            if *users == 0 {
                self.bytes -= *bytes;
                self.rows.remove(&key);
            }
        }
    }

    /// Past either bound, drop the statistics used least recently until
    /// the cache holds [`EVICT_TO`] of it.
    fn evict(&mut self) {
        if self.by_column.len() <= MAX_CACHED && self.bytes <= MAX_CACHED_BYTES {
            return;
        }
        let keep_entries = (MAX_CACHED as f64 * EVICT_TO) as usize;
        let keep_bytes = (MAX_CACHED_BYTES as f64 * EVICT_TO) as usize;
        let mut by_age: Vec<(u64, (Uuid, FieldId))> = self
            .by_column
            .iter()
            .map(|(key, kept)| (kept.last_used, *key))
            .collect();
        by_age.sort_unstable_by_key(|(last_used, _)| *last_used);
        for (_, (superfile, field)) in by_age {
            if self.by_column.len() <= keep_entries && self.bytes <= keep_bytes {
                break;
            }
            self.remove(superfile, field);
        }
    }
}

impl DeletedStatsCache {
    /// The statistics of exactly `rows` of `superfile`'s column `field`,
    /// read as `data_type`.
    pub(crate) fn get(
        &self,
        superfile: Uuid,
        field: FieldId,
        data_type: &DataType,
        rows: &Arc<RoaringBitmap>,
    ) -> Option<Arc<ScalarStatsAgg>> {
        let mut entries = self.entries.lock().expect("deleted-stats cache poisoned");
        let cached = entries.used(superfile, field)?;
        // The tombstone cache hands out one bitmap per sidecar version, so
        // the rows the statistics were read for are most often the very
        // bitmap asked about; comparing them whole is the rare case.
        let same_rows = Arc::ptr_eq(&cached.rows, rows)
            || (cached.rows.len() == rows.len() && *cached.rows == **rows);
        (cached.data_type == *data_type && same_rows).then(|| Arc::clone(&cached.stats))
    }

    /// The cached statistics of some of `rows`, to add the rest to: a
    /// superfile's deletes only grow.
    fn subset_of(
        &self,
        superfile: Uuid,
        field: FieldId,
        data_type: &DataType,
        rows: &RoaringBitmap,
    ) -> Option<CachedRows> {
        let mut entries = self.entries.lock().expect("deleted-stats cache poisoned");
        let cached = entries.used(superfile, field)?;
        (cached.data_type == *data_type && cached.rows.is_subset(rows)).then(|| cached.clone())
    }

    fn insert(&self, superfile: Uuid, field: FieldId, cached: CachedRows) {
        let mut entries = self.entries.lock().expect("deleted-stats cache poisoned");
        entries.insert(superfile, field, cached);
        entries.evict();
    }

    /// Once many statistics are kept, drop those of superfiles `listed`
    /// does not name: a superfile compaction replaced is read no more.
    pub(crate) fn prune_unlisted(&self, listed: impl FnOnce() -> HashSet<Uuid>) {
        let mut entries = self.entries.lock().expect("deleted-stats cache poisoned");
        if entries.by_column.len() <= PRUNE_ABOVE {
            return;
        }
        let listed = listed();
        let unlisted: Vec<(Uuid, FieldId)> = entries
            .by_column
            .keys()
            .filter(|(superfile, _)| !listed.contains(superfile))
            .copied()
            .collect();
        for (superfile, field) in unlisted {
            entries.remove(superfile, field);
        }
    }

    /// The deleted rows the statistics of `superfile`'s column `field`
    /// cover, when kept.
    #[cfg(test)]
    pub(crate) fn rows_of(&self, superfile: Uuid, field: FieldId) -> Option<Arc<RoaringBitmap>> {
        self.entries
            .lock()
            .expect("deleted-stats cache poisoned")
            .by_column
            .get(&(superfile, field))
            .map(|kept| Arc::clone(&kept.cached.rows))
    }

    /// The statistics kept, one per superfile and column.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries
            .lock()
            .expect("deleted-stats cache poisoned")
            .by_column
            .len()
    }

    /// About the bytes the kept statistics hold.
    #[cfg(test)]
    fn bytes(&self) -> usize {
        self.entries
            .lock()
            .expect("deleted-stats cache poisoned")
            .bytes
    }
}

/// Why a superfile's live rows are not known from its statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Unknown {
    /// The deleted rows' statistics would tell, and are not read yet.
    DeletedRows,
    /// Nothing short of a scan tells.
    Scan,
}

/// A live bound of a column in one superfile.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LiveBound {
    /// The bound exactly; `None` when no live row holds a value.
    Exact(Option<ScalarValue>),
    /// Every live value lies on this side of `bound` (at or above it for a
    /// minimum, at or below for a maximum), or there is none: exact enough
    /// when another superfile's exact bound is no worse. `deleted_rows`:
    /// the deleted rows' statistics could make it exact.
    Beyond {
        bound: ScalarValue,
        deleted_rows: bool,
    },
}

/// How a superfile's live rows change its stored value counts.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CountsChange {
    /// Not at all.
    Unchanged,
    /// The rows are all deleted: take off every stored count.
    All,
    /// Take off these counts.
    Less(Vec<(ScalarValue, u64)>),
}

/// One superfile's statistics for one column, as its live rows have them.
pub(crate) struct LiveColumn<'a> {
    /// Rows written.
    rows: u64,
    /// Rows deleted.
    deleted: u64,
    /// Whether the superfile was written without the column: null throughout.
    lacks: bool,
    /// The statistics stored for the column.
    stored: Option<ColumnStats<'a>>,
    /// The deleted rows' statistics, when read.
    delta: Option<Arc<ScalarStatsAgg>>,
}

impl<'a> LiveColumn<'a> {
    /// `entry`'s column `field`, `deleted` of its rows deleted, `delta` the
    /// statistics of those, when read.
    pub(crate) fn new(
        entry: &'a SuperfileEntry,
        field: FieldId,
        lacks: bool,
        deleted: u64,
        delta: Option<Arc<ScalarStatsAgg>>,
    ) -> Self {
        Self {
            rows: entry.n_docs,
            deleted,
            lacks,
            stored: entry.scalar_stats.column(field),
            delta,
        }
    }

    fn all_deleted(&self) -> bool {
        self.deleted >= self.rows
    }

    /// Whether the column holds no value in any row written.
    fn no_values(&self) -> bool {
        self.lacks || self.stored.is_some_and(|stats| stats.all_null())
    }

    /// The stored null count.
    fn nulls(&self) -> Result<u64, Unknown> {
        self.stored
            .and_then(|stats| stats.null_count())
            .ok_or(Unknown::Scan)
    }

    fn delta(&self) -> Result<&ScalarStatsAgg, Unknown> {
        self.delta.as_deref().ok_or(Unknown::DeletedRows)
    }

    /// The deleted rows that hold a value.
    fn deleted_values(&self) -> Result<u64, Unknown> {
        let nulls = self.delta()?.null_count.ok_or(Unknown::Scan)?;
        self.deleted.checked_sub(nulls).ok_or(Unknown::Scan)
    }

    /// The live rows holding a value.
    pub(crate) fn non_null(&self) -> Result<u64, Unknown> {
        if self.lacks || self.all_deleted() {
            return Ok(0);
        }
        let nulls = self.nulls()?;
        let stored = self.rows.checked_sub(nulls).ok_or(Unknown::Scan)?;
        if self.deleted == 0 {
            return Ok(stored);
        }
        if nulls == self.rows {
            return Ok(0);
        }
        if nulls == 0 {
            return Ok(self.rows - self.deleted);
        }
        stored
            .checked_sub(self.deleted_values()?)
            .ok_or(Unknown::Scan)
    }

    /// The live rows holding no value.
    pub(crate) fn live_nulls(&self) -> Result<u64, Unknown> {
        if self.all_deleted() {
            return Ok(0);
        }
        if self.lacks {
            return Ok(self.rows - self.deleted);
        }
        let nulls = self.nulls()?;
        if self.deleted == 0 || nulls == 0 {
            return Ok(nulls);
        }
        if nulls == self.rows {
            return Ok(self.rows - self.deleted);
        }
        let deleted_nulls = self.delta()?.null_count.ok_or(Unknown::Scan)?;
        nulls.checked_sub(deleted_nulls).ok_or(Unknown::Scan)
    }

    /// The live values' sum; `None` when no live row holds a value.
    pub(crate) fn sum(&self) -> Result<Option<ScalarValue>, Unknown> {
        if self.no_values() || self.all_deleted() {
            return Ok(None);
        }
        let stored = self
            .stored
            .and_then(|stats| stats.sum())
            .ok_or(Unknown::Scan)?;
        if self.deleted > 0 {
            // A float sum less another is not the sum of the rest: the
            // larger terms' rounding swallows the smaller ones'.
            if matches!(stored.data_type(), DataType::Float64) {
                return Err(Unknown::Scan);
            }
            if self.non_null()? == 0 {
                return Ok(None);
            }
            if self.deleted_values()? > 0 {
                let deleted = self.delta()?.sum.as_ref().ok_or(Unknown::Scan)?;
                let live = sub_sum_arrays(&stored, deleted).ok_or(Unknown::Scan)?;
                return scalar(&live).map(Some);
            }
        }
        scalar(&stored).map(Some)
    }

    /// The live minimum (`max` false) or maximum (`max` true).
    pub(crate) fn bound(&self, max: bool) -> Result<LiveBound, Unknown> {
        if self.no_values() || self.all_deleted() {
            return Ok(LiveBound::Exact(None));
        }
        let stats = self.stored.ok_or(Unknown::Scan)?;
        let (min, max_value) = stats.min_max_values().ok_or(Unknown::Scan)?;
        let (bound, other) = match max {
            false => (min, max_value),
            true => (max_value, min),
        };
        if bound.is_null() || is_nan(&bound) || is_nan(&other) {
            return Err(Unknown::Scan);
        }
        if self.deleted == 0 {
            return Ok(LiveBound::Exact(Some(bound)));
        }
        // One value in every row: every live row holds it. Not for a float,
        // whose two zeros compare equal.
        if bound == other && !is_float(&bound) && self.nulls() == Ok(0) {
            return Ok(LiveBound::Exact(Some(bound)));
        }
        let Some(delta) = self.delta.as_deref() else {
            return Ok(LiveBound::Beyond {
                bound,
                deleted_rows: true,
            });
        };
        // No deleted row held a value: the stored bound is a live row's.
        if delta.all_null() {
            return Ok(LiveBound::Exact(Some(bound)));
        }
        if self.non_null()? == 0 {
            return Ok(LiveBound::Exact(None));
        }
        let deleted_bound = scalar(match max {
            false => &delta.min,
            true => &delta.max,
        })?;
        if is_nan(&deleted_bound) {
            return Err(Unknown::Scan);
        }
        // Every deleted value lies strictly inside the bound: a live row
        // holds it.
        let inside = match (deleted_bound.partial_cmp(&bound), max) {
            (Some(order), false) => order.is_gt(),
            (Some(order), true) => order.is_lt(),
            (None, _) => return Err(Unknown::Scan),
        };
        if inside {
            return Ok(LiveBound::Exact(Some(bound)));
        }
        // The value counts less the deleted rows' are the live ones.
        if let Some(stored) = stats.value_counts()
            && let Ok(CountsChange::Less(less)) = self.counts_change()
        {
            let live = take_off(
                stored.entries().iter().cloned().collect(),
                less.iter().map(|(value, count)| (value, *count)),
            )
            .ok_or(Unknown::Scan)?;
            let mut extreme: Option<&ScalarValue> = None;
            for value in live.keys() {
                extreme = Some(match extreme {
                    None => value,
                    Some(best) => {
                        let order = value.partial_cmp(best).ok_or(Unknown::Scan)?;
                        match (order, max) {
                            (Ordering::Less, false) | (Ordering::Greater, true) => value,
                            _ => best,
                        }
                    }
                });
            }
            return Ok(LiveBound::Exact(extreme.cloned()));
        }
        Ok(LiveBound::Beyond {
            bound,
            deleted_rows: false,
        })
    }

    /// How the live rows change the stored value counts. A superfile that
    /// holds values it recorded no counts of is scanned.
    pub(crate) fn counts_change(&self) -> Result<CountsChange, Unknown> {
        if self.no_values() {
            return Ok(CountsChange::Unchanged);
        }
        let stats = self.stored.ok_or(Unknown::Scan)?;
        if !stats.has_value_table() {
            return Err(Unknown::Scan);
        }
        if self.deleted == 0 {
            return Ok(CountsChange::Unchanged);
        }
        if self.all_deleted() {
            return Ok(CountsChange::All);
        }
        let stored = stats.value_counts().ok_or(Unknown::Scan)?;
        // One value and no null: every deleted row held it.
        if let [(value, _)] = stored.entries()
            && self.nulls() == Ok(0)
        {
            return Ok(CountsChange::Less(vec![(value.clone(), self.deleted)]));
        }
        let delta = self.delta()?;
        if delta.all_null() {
            return Ok(CountsChange::Unchanged);
        }
        let counts = delta.value_counts.as_ref().ok_or(Unknown::Scan)?;
        Ok(CountsChange::Less(counts.entries().to_vec()))
    }
}

fn scalar(array: &ArrayRef) -> Result<ScalarValue, Unknown> {
    ScalarValue::try_from_array(array.as_ref(), 0).map_err(|_| Unknown::Scan)
}

fn is_float(value: &ScalarValue) -> bool {
    matches!(
        value,
        ScalarValue::Float16(_) | ScalarValue::Float32(_) | ScalarValue::Float64(_)
    )
}

fn is_nan(value: &ScalarValue) -> bool {
    match value {
        ScalarValue::Float64(Some(v)) => v.is_nan(),
        ScalarValue::Float32(Some(v)) => v.is_nan(),
        ScalarValue::Float16(Some(v)) => v.is_nan(),
        _ => false,
    }
}

/// A column whose deleted rows' statistics a statement wants.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct WantedColumn {
    pub(crate) field: FieldId,
    pub(crate) name: String,
}

/// One superfile whose deleted rows' statistics a statement wants, for
/// some of its columns.
pub(crate) struct Wanted {
    pub(crate) entry: Arc<SuperfileEntry>,
    /// Its deleted rows, as the statement read them.
    pub(crate) deleted: Arc<RoaringBitmap>,
    pub(crate) columns: BTreeSet<WantedColumn>,
}

/// Read and cache the deleted rows' statistics `wanted` names, through
/// `provider`, where reading them costs no more pages than scanning the
/// live rows would. Each superfile reads only the rows its cached
/// statistics do not cover yet. A read that fails leaves the superfile to
/// the scan.
pub(crate) async fn read_wanted(
    provider: &SupertableProvider,
    state: &dyn Session,
    cache: &DeletedStatsCache,
    wanted: Vec<Wanted>,
) {
    let stored = provider.manifest().stored_schema();
    let in_flight = provider.manifest().options.superfiles_in_flight;
    // One read per superfile, of every column wanted of it.
    let mut by_superfile: HashMap<Uuid, Wanted> = HashMap::new();
    for wanted in wanted {
        match by_superfile.get_mut(&wanted.entry.superfile_id) {
            Some(seen) if *seen.deleted == *wanted.deleted => seen.columns.extend(wanted.columns),
            Some(_) => {}
            None => {
                by_superfile.insert(wanted.entry.superfile_id, wanted);
            }
        }
    }
    let reads = by_superfile.into_values().map(|wanted| {
        let stored = Arc::clone(&stored);
        async move {
            let superfile = wanted.entry.superfile_id;
            if let Err(error) = read_superfile(provider, state, cache, &stored, wanted).await {
                debug!(%superfile, %error, "deleted rows not read; the superfile is scanned");
            }
            Ok::<(), DataFusionError>(())
        }
    });
    let _ = try_join_bounded(reads, in_flight).await;
}

/// [`read_wanted`] for one superfile.
async fn read_superfile(
    provider: &SupertableProvider,
    state: &dyn Session,
    cache: &DeletedStatsCache,
    stored: &Schema,
    wanted: Wanted,
) -> DfResult<()> {
    let superfile = wanted.entry.superfile_id;
    // Columns grouped by the rows they still need, so columns cached over
    // the same deletes are read together.
    let mut groups: Vec<(
        RoaringBitmap,
        Vec<(WantedColumn, DataType, Option<CachedRows>)>,
    )> = Vec::new();
    for column in wanted.columns {
        let Ok(field) = stored.field_with_name(&column.name) else {
            continue;
        };
        let data_type = field.data_type().clone();
        if cache
            .get(superfile, column.field, &data_type, &wanted.deleted)
            .is_some()
        {
            continue;
        }
        let base = cache.subset_of(superfile, column.field, &data_type, &wanted.deleted);
        let mut unread = (*wanted.deleted).clone();
        if let Some(base) = &base {
            unread -= base.rows.as_ref();
        }
        match groups.iter_mut().find(|(rows, _)| *rows == unread) {
            Some((_, columns)) => columns.push((column, data_type, base)),
            None => groups.push((unread, vec![(column, data_type, base)])),
        }
    }
    for (unread, columns) in groups {
        let names: Vec<&str> = columns.iter().map(|(c, _, _)| c.name.as_str()).collect();
        let pays = match provider
            .page_costs(Arc::clone(&wanted.entry), &names, &unread, &wanted.deleted)
            .await?
        {
            Some((read, live)) => read <= live,
            None => {
                (unread.len() as f64) <= wanted.entry.n_docs as f64 * MAX_READ_SHARE_WITHOUT_PAGES
            }
        };
        if !pays {
            debug!(%superfile, rows = unread.len(), "deleted rows cost more pages than the live rows");
            continue;
        }
        let n_unread = unread.len();
        let read = read_rows(provider, state, &wanted.entry, Arc::new(unread), &names).await?;
        // The statistics are CPU work over every row read: the reader pool's.
        let folded = run_on_pool(
            Some(&provider.manifest().options.reader_pool),
            "deleted-row statistics",
            move || {
                columns
                    .into_iter()
                    .zip(read)
                    .filter_map(|((column, data_type, base), values)| {
                        // A read that missed some of the rows would cache
                        // statistics short of them; such a column is left
                        // to the scan.
                        let values = values.filter(|values| values.len() as u64 == n_unread)?;
                        let stats = fold_read(&values, &data_type, base.as_ref())?;
                        Some((column, data_type, stats))
                    })
                    .collect::<Vec<_>>()
            },
        )
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
        for (column, data_type, stats) in folded {
            cache.insert(
                superfile,
                column.field,
                CachedRows {
                    data_type,
                    rows: Arc::clone(&wanted.deleted),
                    stats: Arc::new(stats),
                },
            );
        }
    }
    Ok(())
}

/// The statistics of `values`, read as `data_type`, added to `base`'s: the
/// deleted rows read now beside those read before. `None` when they don't
/// fold (a type without statistics, or bounds of another type).
fn fold_read(
    values: &ArrayRef,
    data_type: &DataType,
    base: Option<&CachedRows>,
) -> Option<ScalarStatsAgg> {
    let values = cast(values, data_type).ok()?;
    let stats = ScalarStatsAgg::from_column(&values)?;
    match base {
        None => Some(stats),
        Some(base) => {
            let mut merged = (*base.stats).clone();
            merged.merge_with(&stats).ok()?;
            Some(merged)
        }
    }
}

/// The values `rows` of `entry` hold in each of `columns`, through the
/// provider's scan of just those rows; `None` for a column the scan does
/// not serve.
async fn read_rows(
    provider: &SupertableProvider,
    state: &dyn Session,
    entry: &Arc<SuperfileEntry>,
    rows: Arc<RoaringBitmap>,
    columns: &[&str],
) -> DfResult<Vec<Option<ArrayRef>>> {
    let reading = provider.reading_rows(Arc::clone(entry), rows);
    let schema = reading.schema();
    let projection: Vec<usize> = columns
        .iter()
        .filter_map(|name| schema.index_of(name).ok())
        .collect();
    let plan = reading.scan(state, Some(&projection), &[], None).await?;
    let batches = collect(plan, state.task_ctx()).await?;
    Ok(columns
        .iter()
        .map(|name| {
            let pieces: Vec<&dyn Array> = batches
                .iter()
                .filter_map(|batch| batch.column_by_name(name).map(|c| c.as_ref()))
                .collect();
            concat(&pieces).ok()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, ops::Range, slice::from_ref, sync::Arc};

    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};
    use roaring::RoaringBitmap;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            Supertable, SupertableOptions, SupertableReader, query::sql::value_count_schema,
        },
        test_helpers::default_disk_cache,
    };

    /// Rows of the superfile whose deleted rows sit on more pages than its
    /// live ones: ten pages of the writer's page row limit.
    const MANY_ROWS: i64 = 40_960;
    /// Rows of a small superfile.
    const FEW_ROWS: i64 = 1_000;
    /// Distinct values of `v`, beside its nulls: one row in this many.
    const V_VALUES: i64 = 7;
    /// The live rows left at the front of the superfile whose deletes
    /// cover most of it.
    const KEPT_AT_FRONT: i64 = 100;
    /// A run of rows a delete takes from a small superfile, and the next.
    const FIRST_RUN: Range<i64> = 10..20;
    const SECOND_RUN: Range<i64> = 30..40;
    /// The lowest `w` of the clean superfile in the bounds test, below every
    /// value of the other.
    const CLEAN_MIN: i64 = -1_000;
    /// Rows of the superfile with deletes in the bounds test: more distinct
    /// values of `w` than a superfile counts exactly.
    const WIDE_ROWS: i64 = 300;

    struct Table {
        table: Supertable,
        _dirs: (TempDir, TempDir),
    }

    /// A table on storage with deletes tracked: `k` a key, `v` a few values
    /// and nulls, `w` an integer of many values.
    fn table() -> Table {
        let dir = TempDir::new().expect("tempdir");
        let cache_dir = TempDir::new().expect("cache dir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("storage"));
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, true),
            Field::new("w", DataType::Int64, true),
        ]));
        let options = SupertableOptions::new(schema, vec![], vec![])
            .expect("options")
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(default_disk_cache(storage, cache_dir.path()));
        Table {
            table: Supertable::create(options).expect("create"),
            _dirs: (dir, cache_dir),
        }
    }

    impl Table {
        /// One superfile of keys `keys`, `w` from `w(k)`.
        fn append(&self, keys: Range<i64>, w: impl Fn(i64) -> i64) {
            let schema = Arc::clone(&self.table.options().schema);
            let k: Vec<i64> = keys.clone().collect();
            let v: Vec<Option<i64>> = keys
                .clone()
                .map(|k| (k % (V_VALUES + 1) != 0).then_some(k % V_VALUES))
                .collect();
            let w: Vec<Option<i64>> = keys.map(|k| Some(w(k))).collect();
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(k)),
                    Arc::new(Int64Array::from(v)),
                    Arc::new(Int64Array::from(w)),
                ],
            )
            .expect("batch");
            let mut writer = self.table.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }

        fn delete(&self, keys: Range<i64>) {
            self.table
                .delete(
                    col("k")
                        .gt_eq(lit(keys.start))
                        .and(col("k").lt(lit(keys.end))),
                )
                .expect("delete");
        }

        fn reader(&self) -> SupertableReader {
            self.table.reader().expect("reader")
        }
    }

    /// `sql`'s rows as text, sorted.
    fn rows(reader: &SupertableReader, sql: &str) -> Vec<String> {
        sorted_rows(&reader.query_sql(sql).expect("query"))
    }

    /// The rows of `batches` as text, sorted.
    fn sorted_rows(batches: &[RecordBatch]) -> Vec<String> {
        let mut rows = Vec::new();
        for batch in batches {
            for row in 0..batch.num_rows() {
                let cells: Vec<String> = batch
                    .columns()
                    .iter()
                    .map(|column| {
                        ScalarValue::try_from_array(column, row)
                            .expect("value")
                            .to_string()
                    })
                    .collect();
                rows.push(cells.join("|"));
            }
        }
        rows.sort();
        rows
    }

    fn explain(reader: &SupertableReader, sql: &str) -> String {
        rows(reader, &format!("EXPLAIN {sql}")).join("\n")
    }

    /// `sql` answered as its forced scan answers it (see the integration
    /// oracle): every row passes `random() >= 0`.
    fn assert_matches_scan(reader: &SupertableReader, sql: &str, forced: &str) {
        assert_eq!(rows(reader, sql), rows(reader, forced), "{sql}");
    }

    const GROUPED: &str = "SELECT v, COUNT(*) FROM supertable GROUP BY v";
    const GROUPED_SCAN: &str = "SELECT v, COUNT(*) FROM supertable WHERE random() >= 0 GROUP BY v";

    fn v_id(reader: &SupertableReader) -> FieldId {
        reader.manifest().field_id("v").expect("v")
    }

    fn superfile_ids(reader: &SupertableReader) -> Vec<Uuid> {
        reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|entry| entry.superfile_id)
            .collect()
    }

    /// A grouped count reads a superfile's deleted rows once and keeps
    /// their statistics; after more deletes, the kept statistics cover the
    /// new deleted rows too, and the count is answered with no scan each
    /// time.
    #[test]
    fn deleted_rows_are_read_once_and_then_only_the_new_ones() {
        let t = table();
        t.append(0..FEW_ROWS, |k| k);
        t.delete(FIRST_RUN);
        let reader = t.reader();
        assert_matches_scan(&reader, GROUPED, GROUPED_SCAN);
        assert!(!explain(&reader, GROUPED).contains("TableScan"));
        let superfile = superfile_ids(&reader)[0];
        let cache = reader.deleted_stats();
        let first: RoaringBitmap = FIRST_RUN.map(|k| k as u32).collect();
        assert_eq!(
            cache.rows_of(superfile, v_id(&reader)).as_deref(),
            Some(&first)
        );

        t.delete(SECOND_RUN);
        let reader = t.reader();
        assert_matches_scan(&reader, GROUPED, GROUPED_SCAN);
        assert!(!explain(&reader, GROUPED).contains("TableScan"));
        let both: RoaringBitmap = FIRST_RUN.chain(SECOND_RUN).map(|k| k as u32).collect();
        assert_eq!(
            cache.rows_of(superfile, v_id(&reader)).as_deref(),
            Some(&both)
        );
    }

    /// A superfile whose deleted rows sit on more pages than its live ones
    /// is scanned rather than corrected; one whose deletes are a short run
    /// is corrected.
    #[test]
    fn deleted_rows_on_more_pages_than_the_live_ones_are_scanned() {
        let t = table();
        t.append(0..MANY_ROWS, |k| k);
        t.append(MANY_ROWS..MANY_ROWS + FEW_ROWS, |k| k);
        t.delete(KEPT_AT_FRONT..MANY_ROWS);
        t.delete(MANY_ROWS + FIRST_RUN.start..MANY_ROWS + FIRST_RUN.end);
        let reader = t.reader();
        assert_matches_scan(&reader, GROUPED, GROUPED_SCAN);
        let explained = explain(&reader, GROUPED);
        assert!(
            explained.contains(
                "scans 1 superfiles (1 with deleted rows); 1 answered from statistics, 1 of them \
                 past their deleted rows"
            ),
            "{explained}"
        );
        let ids = superfile_ids(&reader);
        let cache = reader.deleted_stats();
        let field = v_id(&reader);
        let kept: HashSet<bool> = ids
            .iter()
            .map(|id| cache.rows_of(*id, field).is_some())
            .collect();
        assert_eq!(
            kept,
            [true, false].into_iter().collect(),
            "one corrected, one not"
        );
    }

    /// `MIN` over a superfile whose deleted row held its least value, with
    /// no value counts to say what is left, reads nothing when another
    /// superfile's minimum is lower; `MAX`, which that superfile decides,
    /// reads its deleted rows and is exact past them. Neither scans.
    #[test]
    fn a_bound_another_superfile_beats_reads_no_deleted_rows() {
        let t = table();
        t.append(0..FEW_ROWS, |k| CLEAN_MIN + k % V_VALUES);
        t.append(FEW_ROWS..FEW_ROWS + WIDE_ROWS, |k| k - FEW_ROWS);
        t.delete(FEW_ROWS..FEW_ROWS + 1);
        let reader = t.reader();
        let min = "SELECT MIN(w) FROM supertable";
        assert_eq!(rows(&reader, min), vec![CLEAN_MIN.to_string()]);
        assert!(!explain(&reader, min).contains("TableScan"));
        assert_eq!(reader.deleted_stats().len(), 0, "MIN read no deleted row");

        let max = "SELECT MAX(w) FROM supertable";
        assert_eq!(rows(&reader, max), vec![(WIDE_ROWS - 1).to_string()]);
        assert!(!explain(&reader, max).contains("TableScan"));
        assert_eq!(reader.deleted_stats().len(), 1, "MAX read the deleted row");
    }

    /// When every row holding a superfile's least value is deleted, its
    /// value counts less the deleted rows' give the live minimum, with no
    /// scan; the greatest value, still live, stays the maximum.
    #[test]
    fn value_counts_give_the_bound_the_deleted_rows_held() {
        let t = table();
        t.append(0..FEW_ROWS, |k| k);
        t.table.delete(col("v").eq(lit(0i64))).expect("delete");
        let reader = t.reader();
        let bounds = "SELECT MIN(v), MAX(v) FROM supertable";
        assert_eq!(rows(&reader, bounds), vec![format!("1|{}", V_VALUES - 1)]);
        assert!(!explain(&reader, bounds).contains("TableScan"));
    }

    /// The cache answers only for exactly the deleted rows it read, and
    /// offers what it read of fewer rows to add the rest to.
    #[test]
    fn the_cache_answers_for_exactly_the_rows_it_read() {
        let cache = DeletedStatsCache::default();
        let superfile = Uuid::new_v4();
        let field = FieldId::ID_COLUMN;
        let read: Arc<RoaringBitmap> = Arc::new(FIRST_RUN.map(|k| k as u32).collect());
        let more: RoaringBitmap = FIRST_RUN.chain(SECOND_RUN).map(|k| k as u32).collect();
        let values: ArrayRef = Arc::new(Int64Array::from_iter_values(FIRST_RUN));
        let stats = ScalarStatsAgg::from_column(&values).expect("stats");
        cache.insert(
            superfile,
            field,
            CachedRows {
                data_type: DataType::Int64,
                rows: Arc::clone(&read),
                stats: Arc::new(stats),
            },
        );
        let get = |data_type: &DataType, rows: &Arc<RoaringBitmap>| {
            cache.get(superfile, field, data_type, rows).is_some()
        };
        assert!(get(&DataType::Int64, &read), "the very bitmap read");
        assert!(
            get(&DataType::Int64, &Arc::new((*read).clone())),
            "an equal bitmap"
        );
        assert!(!get(&DataType::Int64, &Arc::new(more.clone())), "more rows");
        assert!(!get(&DataType::Int32, &read), "another type");
        let base = cache
            .subset_of(superfile, field, &DataType::Int64, &more)
            .expect("a subset of the rows deleted since");
        assert_eq!(base.rows, read);
        let fewer: RoaringBitmap = FIRST_RUN.skip(1).map(|k| k as u32).collect();
        assert!(
            cache
                .subset_of(superfile, field, &DataType::Int64, &fewer)
                .is_none()
        );
        cache.prune_unlisted(HashSet::new);
        assert_eq!(cache.len(), 1, "few kept: none dropped");
    }

    /// A superfile's columns that share the deleted rows they cover are
    /// charged for them once; past its bound the cache drops the statistics
    /// used least recently, keeping the rest.
    #[test]
    fn a_full_cache_drops_the_statistics_used_least_recently() {
        let cache = DeletedStatsCache::default();
        let rows: Arc<RoaringBitmap> = Arc::new(FIRST_RUN.map(|k| k as u32).collect());
        let values: ArrayRef = Arc::new(Int64Array::from_iter_values(FIRST_RUN));
        let stats = Arc::new(ScalarStatsAgg::from_column(&values).expect("stats"));
        let cached = || CachedRows {
            data_type: DataType::Int64,
            rows: Arc::clone(&rows),
            stats: Arc::clone(&stats),
        };
        let shared = Uuid::new_v4();
        cache.insert(shared, FieldId::ID_COLUMN, cached());
        cache.insert(shared, FieldId(1), cached());
        assert_eq!(
            cache.bytes(),
            2 * cached().value_bytes() + rows.serialized_size(),
            "the shared rows are charged once"
        );

        let cache = DeletedStatsCache::default();
        // One deleted row each, so the count of statistics, not their bytes,
        // is the bound reached.
        let one_row: Arc<RoaringBitmap> = Arc::new(RoaringBitmap::from([0]));
        let one_value: ArrayRef = Arc::new(Int64Array::from_iter_values([0]));
        let one_stats = Arc::new(ScalarStatsAgg::from_column(&one_value).expect("stats"));
        let cached = || CachedRows {
            data_type: DataType::Int64,
            rows: Arc::clone(&one_row),
            stats: Arc::clone(&one_stats),
        };
        let superfiles: Vec<Uuid> = (0..MAX_CACHED).map(|_| Uuid::new_v4()).collect();
        for superfile in &superfiles {
            cache.insert(*superfile, FieldId::ID_COLUMN, cached());
        }
        assert!(cache.bytes() < MAX_CACHED_BYTES);
        let kept = |superfile: Uuid| {
            cache
                .get(superfile, FieldId::ID_COLUMN, &DataType::Int64, &one_row)
                .is_some()
        };
        assert!(kept(superfiles[0]), "used just now");
        cache.insert(Uuid::new_v4(), FieldId::ID_COLUMN, cached());
        assert_eq!(cache.len(), (MAX_CACHED as f64 * EVICT_TO) as usize);
        assert!(kept(superfiles[0]), "used recently: kept");
        assert!(!kept(superfiles[1]), "used least recently: dropped");
        assert!(kept(superfiles[MAX_CACHED - 1]), "the newest: kept");
    }

    /// `value_counts` reads a superfile's deleted rows on its first call,
    /// through the grouped count it runs, and every later call is answered
    /// from the statistics corrected by them, with no statement planned.
    #[test]
    fn value_counts_answer_from_statistics_after_their_first_call() {
        let t = table();
        t.append(0..FEW_ROWS, |k| k);
        t.append(FEW_ROWS..2 * FEW_ROWS, |k| k);
        t.delete(FIRST_RUN);
        let reader = t.reader();
        let field = reader
            .sql_schemas()
            .scalar()
            .field_with_name("v")
            .cloned()
            .expect("v");
        let schema = value_count_schema(&field).expect("schema");
        let from_stats = |reader: &SupertableReader| {
            reader
                .value_counts_from_stats("v", &schema)
                .expect("value counts")
                .is_some()
        };
        assert!(!from_stats(&reader), "nothing read of the deleted rows yet");
        let counted = reader.value_counts("v").expect("value counts");
        assert!(from_stats(&reader), "the first call read them");
        let again = reader.value_counts("v").expect("value counts");
        assert_eq!(
            sorted_rows(from_ref(&again)),
            sorted_rows(from_ref(&counted))
        );
        assert_eq!(reader.deleted_stats().len(), 1);
    }
}
