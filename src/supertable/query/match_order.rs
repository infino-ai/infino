// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`MatchOrder`]: `token_match` / `exact_match` under `ORDER BY column
//! LIMIT n` open their superfiles in the column's order and stop once no
//! superfile left can place a row.
//!
//! An unranked match otherwise opens every superfile its terms route to and
//! hands every matching row to the sort, which keeps `n`. On a table written
//! in time order, the newest few superfiles hold the newest matches: the
//! manifest knows each superfile's range of the column, so the walk opens
//! them newest first (oldest first for an ascending sort), keeps the `n`
//! best keys among the rows it found that the `WHERE` keeps, and skips a
//! superfile whose whole range is worse than the `n`-th of them — and, the
//! walk being in order, every superfile after it:
//!
//! ```text
//!   Sort fetch=24: time DESC
//!     Filter lang = 'en'
//!       exact_match('title', 'Zyzzyva')     every superfile 'Zyzzyva' routes to
//!                                         → newest first, until the 24th best
//!                                           `time` passes every range left
//! ```
//!
//! The sort and the filter stay where they are, so the answer is the one
//! they give over the rows the walk returns; the walk only leaves out
//! superfiles and rows that cannot place. Each superfile returns at most `n`
//! rows, its best, and only those whose key is at least the `n`-th best found
//! so far. Rows tied at the cut are as good as each other, since the sort
//! orders by this one column only, so a key most rows share (a small count,
//! a year) does not hand up every tied row. A column whose ranges overlap in
//! every superfile (a count, a size) so materializes at most `n` rows per
//! superfile rather than every match; such a walk opens every superfile at
//! the fan-out's width, since none can be passed early, and is not built for
//! a `LIMIT` above [`MAX_OVERLAPPING_FETCH`]. It applies to a single sort column
//! of integer, date or timestamp type that every candidate superfile holds
//! current statistics for and no nulls in (a null sorts first under a
//! descending sort, outside every range), under a `WHERE` it can evaluate
//! from stored columns; any other shape opens every superfile, as before.

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    sync::{Arc, Mutex},
};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::{error::Result as DfResult, logical_expr::Expr};
use roaring::RoaringBitmap;

use crate::{
    runtime_bridge::run_on_pool,
    runtime_metrics::op_stats::OpStatsCollector,
    superfile::{
        SuperfileReader,
        range::{KeyKind, push_keys},
    },
    supertable::{
        error::QueryError,
        handle::SupertableReader,
        manifest::{ManifestSnapshot, SuperfileEntry},
        query::{
            SuperfileHit,
            candidate::literal_key,
            exec::common::{EXACT_SCOPE_TAKE_ROWS, PushedPredicate, RowCheck, take_file_columns},
            skip::ColumnTypeGuard,
        },
        schema::FieldId,
    },
};

/// Superfiles an ordered walk opens at once to begin with: enough to
/// overlap one open's round trip with the next, few enough that a walk
/// whose first superfiles place every row stops within one or two of them.
pub(crate) const ORDERED_WINDOW: usize = 2;

/// The walk runs only when at least one in this many pairs of superfiles
/// adjacent in its order hold ranges that do not overlap: a table written
/// in the column's order separates nearly every pair, one whose every
/// superfile spans the column's values separates none.
const SEPARATED_SHARE_DENOMINATOR: usize = 2;

/// The largest `LIMIT` a walk that cannot stop early is built for. Such a
/// walk reads the key of every match and folds up to `n` ranks per chunk
/// into one heap under a lock; past a few thousand that costs more than the
/// rows it leaves out, so the match runs as it would without an order. The
/// same cap as the scan's top-k walk
/// ([`MAX_WALKED_ROWS`](crate::supertable::query::sorted::MAX_WALKED_ROWS)).
pub(crate) const MAX_OVERLAPPING_FETCH: usize = 4096;

/// An `ORDER BY column LIMIT n` a match walk honours: the column, its
/// direction, the limit, and the `WHERE` it checks to know which rows count.
pub(crate) struct MatchOrder {
    manifest: Arc<ManifestSnapshot>,
    column: String,
    field: FieldId,
    data_type: DataType,
    kind: KeyKind,
    guard: ColumnTypeGuard,
    descending: bool,
    fetch: usize,
    check: Option<RowCheck>,
    key_schema: SchemaRef,
    op_stats: Option<Arc<OpStatsCollector>>,
    /// Whether superfiles adjacent in the walk's order hold ranges apart
    /// often enough for the walk to pass superfiles early (see
    /// [`Self::ranges_separate`]).
    stops_early: bool,
    /// The ranks of the `fetch` best keys found so far, the worst on top.
    found: Mutex<BinaryHeap<Reverse<u64>>>,
}

impl MatchOrder {
    /// The id, live type and key kind of `column` when the walk can order
    /// by it: a stored integer, date or timestamp column.
    pub(crate) fn sort_key(
        manifest: &ManifestSnapshot,
        column: &str,
    ) -> Option<(FieldId, DataType, KeyKind)> {
        let field = manifest.field_id(column)?;
        let data_type = manifest
            .stored_schema()
            .field_with_name(column)
            .ok()?
            .data_type()
            .clone();
        match KeyKind::of(&data_type)? {
            kind @ (KeyKind::Signed | KeyKind::Unsigned) => Some((field, data_type, kind)),
            KeyKind::Float => None,
        }
    }

    /// The order `ORDER BY column [DESC] LIMIT fetch` asks for over a match
    /// under `filters`, the `WHERE` pushed to the match. `None` when the
    /// walk cannot honour it: the column is not one [`Self::sort_key`]
    /// takes, or the `WHERE` is one the walk cannot evaluate from stored
    /// columns, so it could not tell which rows count, or the walk cannot
    /// stop early and `fetch` is above [`MAX_OVERLAPPING_FETCH`].
    pub(crate) fn new(
        reader: &SupertableReader,
        column: &str,
        descending: bool,
        fetch: usize,
        filters: &[Expr],
        output_schema: &SchemaRef,
    ) -> DfResult<Option<Self>> {
        let manifest = Arc::clone(reader.manifest());
        let Some((field, data_type, kind)) = Self::sort_key(&manifest, column) else {
            return Ok(None);
        };
        if fetch == 0 {
            return Ok(None);
        }
        let check = match filters.is_empty() {
            true => None,
            false => {
                let Some(predicate) = PushedPredicate::compile(filters, output_schema) else {
                    return Ok(None);
                };
                let Some(check) = RowCheck::new(reader, &predicate, output_schema)? else {
                    return Ok(None);
                };
                Some(check)
            }
        };
        let key_schema = Arc::new(Schema::new(vec![Field::new(
            column,
            data_type.clone(),
            true,
        )]));
        let mut order = Self {
            guard: ColumnTypeGuard::new(&manifest, field),
            manifest,
            column: column.to_owned(),
            field,
            data_type,
            kind,
            descending,
            fetch,
            check,
            key_schema,
            op_stats: reader.op_stats.clone(),
            stops_early: false,
            found: Mutex::new(BinaryHeap::new()),
        };
        order.stops_early = order.ranges_separate();
        if !order.stops_early && fetch > MAX_OVERLAPPING_FETCH {
            return Ok(None);
        }
        Ok(Some(order))
    }

    /// Whether at least one in [`SEPARATED_SHARE_DENOMINATOR`] pairs of
    /// superfiles adjacent in the walk's order hold ranges of the column
    /// that do not overlap, over the superfiles whose statistics rank it.
    fn ranges_separate(&self) -> bool {
        let mut walk_order: Vec<(u64, u64)> = self
            .manifest
            .superfiles
            .iter()
            .filter_map(|entry| self.rank_range(entry))
            .collect();
        walk_order.sort_unstable_by_key(|&(best, _)| Reverse(best));
        let separated = walk_order
            .windows(2)
            .filter(|pair| pair[0].1 >= pair[1].0)
            .count();
        separated * SEPARATED_SHARE_DENOMINATOR >= walk_order.len().saturating_sub(1)
    }

    /// A key's rank in the walk: higher is better, whichever the sort's
    /// direction.
    fn rank(&self, key: u64) -> u64 {
        rank_of(key, self.descending)
    }

    /// Each unit's best rank — of the best key its superfile can hold — in
    /// the units' order. `None` when a superfile holds no statistics for
    /// the column, may hold a null in it (a null sorts first under a
    /// descending sort, outside every range), lacks the column, or holds it
    /// in a type the table has since left: the walk cannot then know what
    /// it leaves out.
    pub(crate) fn best_ranks<P>(&self, units: &[(Arc<SuperfileEntry>, P)]) -> Option<Vec<u64>> {
        units
            .iter()
            .map(|(entry, _)| self.rank_range(entry).map(|(best, _)| best))
            .collect()
    }

    /// The best and worst rank a row of `entry` can take, from its
    /// statistics; `None` when they cannot be trusted (see
    /// [`Self::best_ranks`]).
    fn rank_range(&self, entry: &SuperfileEntry) -> Option<(u64, u64)> {
        if self.guard.absent_from(entry) || self.guard.stats_are_stale(entry) {
            return None;
        }
        let stats = entry.scalar_stats.column(self.field)?;
        if stats.null_count() != Some(0) {
            return None;
        }
        let (min, max) = stats.min_max_values()?;
        let (_, min) = literal_key(&min, &self.data_type)?;
        let (_, max) = literal_key(&max, &self.data_type)?;
        let (min, max) = (self.rank(min), self.rank(max));
        Some((min.max(max), min.min(max)))
    }

    /// Whether a superfile whose best rank is `best` can no longer place a
    /// row: `fetch` rows are in hand and its best key is worse than the
    /// worst of them.
    pub(crate) fn passed(&self, best: u64) -> bool {
        let found = self.found.lock().expect("match order mutex poisoned");
        found.len() >= self.fetch && found.peek().is_some_and(|Reverse(worst)| best < *worst)
    }

    /// Superfiles to keep open at once once `answered` have answered: as
    /// many as have answered, from [`ORDERED_WINDOW`] up to `widest`, and
    /// never past `widest`, the fan-out's cap, even when it is below
    /// [`ORDERED_WINDOW`]. A walk that stops early stops while its window
    /// is small, opening at most about twice the superfiles it needs; one
    /// that cannot stop — a `WHERE`
    /// few rows pass, or a column whose ranges overlap in every superfile —
    /// reaches the fan-out's width in a few steps instead of opening every
    /// superfile a fixed few at a time.
    pub(crate) fn window(answered: usize, widest: usize) -> usize {
        answered.max(ORDERED_WINDOW).min(widest)
    }

    /// [`Self::window`] for this walk: the fan-out's width from the start
    /// when the ranges overlap too often for any superfile to be passed.
    pub(crate) fn walk_window(&self, answered: usize, widest: usize) -> usize {
        match self.stops_early {
            true => Self::window(answered, widest),
            false => widest,
        }
    }

    /// Of `candidates`, rows of one superfile read through `r` that may
    /// match, those whose key is at least the walk's `fetch`-th best so far:
    /// the rest cannot place, so a caller need not check them further. Their
    /// keys are not folded in, since a candidate is not yet a match. Every
    /// candidate is kept until `fetch` rows are in hand.
    pub(crate) async fn prefilter(
        &self,
        r: &Arc<SuperfileReader>,
        candidates: Vec<u32>,
    ) -> Result<Vec<u32>, QueryError> {
        let Some(worst) = self.worst() else {
            return Ok(candidates);
        };
        let mut placing = Vec::new();
        for chunk in candidates.chunks(EXACT_SCOPE_TAKE_ROWS) {
            let (ranks, _) = self.chunk_ranks(r, chunk, 0).await?;
            placing.extend(
                self.ranked_at_least(chunk.to_vec(), ranks, Some(worst))
                    .await?,
            );
        }
        Ok(placing)
    }

    /// The rank of each of `rows`' keys in one superfile read through `r`,
    /// in `rows`' order, and the best `fetch` of them. The keys are reserved
    /// against the connection's budget while they are ranked, and ranked on
    /// the reader pool.
    async fn chunk_ranks(
        &self,
        r: &Arc<SuperfileReader>,
        rows: &[u32],
        fetch: usize,
    ) -> Result<(Vec<u64>, Vec<u64>), QueryError> {
        let batch = take_file_columns(
            &self.manifest,
            r,
            rows.to_vec(),
            &[self.column.as_str()],
            &self.key_schema,
            &self.op_stats,
        )
        .await
        .map_err(QueryError::DataFusion)?;
        let _held = self
            .manifest
            .options
            .connection_memory_budget
            .try_reserve(batch.get_array_memory_size())
            .map_err(|over| {
                QueryError::OverBudget(format!("during an ordered match walk, {over}"))
            })?;
        let (kind, descending) = (self.kind, self.descending);
        run_on_pool(
            Some(&self.manifest.options.reader_pool),
            "ordered match keys",
            move || ranks_of_chunk(&batch, kind, descending, fetch),
        )
        .await
        .map_err(QueryError::internal)?
    }

    /// The `fetch`-th best rank so far, once `fetch` rows are in hand.
    fn worst(&self) -> Option<u64> {
        let found = self.found.lock().expect("match order mutex poisoned");
        match found.len() >= self.fetch {
            true => found.peek().map(|Reverse(worst)| *worst),
            false => None,
        }
    }

    /// Of `hits`, the matches of one superfile read through `r`, at most
    /// `fetch` of those the `WHERE` keeps and whose key is at least the
    /// walk's `fetch`-th best once theirs are folded in: a row below it can
    /// never place, and the superfile's best `fetch` rows are all it can add
    /// to the answer.
    pub(crate) async fn keep(
        &self,
        r: &Arc<SuperfileReader>,
        hits: Vec<SuperfileHit>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if hits.is_empty() {
            return Ok(hits);
        }
        let rows: RoaringBitmap = hits.iter().map(|h| h.local_doc_id).collect();
        let kept = match &self.check {
            Some(check) => check.keep(r, &rows).await?,
            None => rows,
        };
        if kept.is_empty() {
            return Ok(Vec::new());
        }
        // The keys are read in chunks: a chunk's best `fetch` ranks are all
        // the walk's best can take from it, so only those reach the lock.
        let mut placing: Vec<(u32, u64)> = Vec::new();
        let mut remaining = kept.iter();
        loop {
            let chunk: Vec<u32> = remaining.by_ref().take(EXACT_SCOPE_TAKE_ROWS).collect();
            if chunk.is_empty() {
                break;
            }
            let (ranks, best) = self.chunk_ranks(r, &chunk, self.fetch).await?;
            let worst = {
                let mut found = self.found.lock().expect("match order mutex poisoned");
                for rank in best {
                    keep_best(&mut found, self.fetch, rank);
                }
                match found.len() >= self.fetch {
                    true => found.peek().map(|Reverse(worst)| *worst),
                    false => None,
                }
            };
            let fetch = self.fetch;
            placing = run_on_pool(
                Some(&self.manifest.options.reader_pool),
                "ordered match pruning",
                move || {
                    placing.extend(
                        chunk
                            .into_iter()
                            .zip(ranks)
                            .filter(|(_, rank)| worst.is_none_or(|worst| *rank >= worst)),
                    );
                    keep_best_rows(&mut placing, fetch);
                    placing
                },
            )
            .await
            .map_err(QueryError::internal)?;
        }
        run_on_pool(
            Some(&self.manifest.options.reader_pool),
            "ordered match pruning",
            move || {
                let placing: RoaringBitmap = placing.into_iter().map(|(row, _)| row).collect();
                hits.into_iter()
                    .filter(|h| placing.contains(h.local_doc_id))
                    .collect()
            },
        )
        .await
        .map_err(QueryError::internal)
    }

    /// Of `rows`, those whose rank in `ranks` is at least `worst`, all of
    /// them while there is no worst yet. Picked on the reader pool, since a
    /// chunk holds up to [`EXACT_SCOPE_TAKE_ROWS`] rows.
    async fn ranked_at_least(
        &self,
        rows: Vec<u32>,
        ranks: Vec<u64>,
        worst: Option<u64>,
    ) -> Result<Vec<u32>, QueryError> {
        run_on_pool(
            Some(&self.manifest.options.reader_pool),
            "ordered match pruning",
            move || {
                rows.into_iter()
                    .zip(ranks)
                    .filter(|(_, rank)| worst.is_none_or(|worst| *rank >= worst))
                    .map(|(row, _)| row)
                    .collect()
            },
        )
        .await
        .map_err(QueryError::internal)
    }
}

/// A key's rank in a walk sorting `descending`: higher is better, whichever
/// the direction.
fn rank_of(key: u64, descending: bool) -> u64 {
    match descending {
        true => key,
        false => !key,
    }
}

/// Each row's rank in `batch`'s first column, and the best `fetch` of them. A
/// row without a key ranks best; the walk runs only over superfiles with no
/// nulls in the column, so it meets none.
fn ranks_of_chunk(
    batch: &RecordBatch,
    kind: KeyKind,
    descending: bool,
    fetch: usize,
) -> Result<(Vec<u64>, Vec<u64>), QueryError> {
    let mut ranks = vec![u64::MAX; batch.num_rows()];
    let mut best: BinaryHeap<Reverse<u64>> = BinaryHeap::with_capacity(fetch.min(batch.num_rows()));
    push_keys(batch.column(0).as_ref(), kind, 0, |key, row| {
        let rank = rank_of(key, descending);
        ranks[row as usize] = rank;
        keep_best(&mut best, fetch, rank);
        Ok(())
    })
    .map_err(QueryError::internal)?;
    Ok((ranks, best.into_iter().map(|Reverse(rank)| rank).collect()))
}

/// Keep only the best `fetch` of `rows`, each a row and its rank. Which of
/// the rows tied at the cut stay does not matter: the sort above orders by
/// this one column only, so any of them is a right answer.
fn keep_best_rows(rows: &mut Vec<(u32, u64)>, fetch: usize) {
    if fetch > 0 && rows.len() > fetch {
        rows.select_nth_unstable_by_key(fetch - 1, |&(_, rank)| Reverse(rank));
        rows.truncate(fetch);
    }
}

/// Take `rank` into `best`, the `fetch` best ranks so far with the worst on
/// top.
fn keep_best(best: &mut BinaryHeap<Reverse<u64>>, fetch: usize, rank: u64) {
    if best.len() < fetch {
        best.push(Reverse(rank));
    } else if best.peek().is_some_and(|Reverse(worst)| rank > *worst) {
        best.pop();
        best.push(Reverse(rank));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::scalar::ScalarValue;

    use super::{MatchOrder, ORDERED_WINDOW, keep_best};
    use crate::{
        IndexSpec, connect,
        superfile::SuperfileReader,
        supertable::{
            handle::SupertableReader,
            manifest::SuperfileEntry,
            query::{SuperfileHit, candidate::literal_key},
        },
    };

    /// The fixture's table.
    const TABLE: &str = "views";
    /// Each row's `hour` in the fixture's one superfile: below, at and above
    /// the seeded threshold, with a tie at it.
    const HOURS: [i64; 5] = [10, 30, 40, 40, 60];
    /// The walk's limit.
    const FETCH: usize = 2;
    /// The hours of the matches already in hand when a test starts, best
    /// last: their worst, [`THRESHOLD_HOUR`], is the walk's `FETCH`-th best.
    const SEEDED: [i64; FETCH] = [THRESHOLD_HOUR, 50];
    /// The worst of [`SEEDED`].
    const THRESHOLD_HOUR: i64 = 40;
    /// A fan-out cap of one superfile at a time.
    const ONE_IN_FLIGHT: usize = 1;
    /// A fan-out cap above [`ORDERED_WINDOW`].
    const WIDE_IN_FLIGHT: usize = 16;
    /// Superfiles answered, past either cap.
    const MANY_ANSWERED: usize = 100;

    /// A walk ordering one superfile of [`HOURS`] by `hour DESC LIMIT FETCH`,
    /// the table's reader, and that superfile's entry and reader.
    fn fixture() -> (
        MatchOrder,
        SupertableReader,
        Arc<SuperfileEntry>,
        Arc<SuperfileReader>,
    ) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("hour", DataType::Int64, false),
        ]));
        let conn = connect("memory://").expect("connect");
        let table = conn
            .create_table(TABLE, Arc::clone(&schema), IndexSpec::new().fts("title"))
            .expect("create");
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(LargeStringArray::from(vec!["Zyzzyva"; HOURS.len()])),
                Arc::new(Int64Array::from(HOURS.to_vec())),
            ],
        )
        .expect("batch");
        table.append(&batch).expect("append");
        let reader = table.local_handle().reader().expect("reader");
        let entry = Arc::clone(&reader.manifest().superfiles[0]);
        let r = reader.open_superfile(&entry).expect("open");
        let order = MatchOrder::new(
            &reader,
            "hour",
            true,
            FETCH,
            &[],
            &reader.manifest().scalar_schema(),
        )
        .expect("order")
        .expect("the walk orders by hour");
        (order, reader, entry, r)
    }

    /// `hour`'s rank in `order`.
    fn rank(order: &MatchOrder, hour: i64) -> u64 {
        let (_, key) = literal_key(&ScalarValue::Int64(Some(hour)), &DataType::Int64).expect("key");
        order.rank(key)
    }

    /// Fold verified matches of `hours` into `order`'s best.
    fn seed(order: &MatchOrder, hours: &[i64]) {
        let mut found = order.found.lock().expect("match order mutex poisoned");
        for &hour in hours {
            keep_best(&mut found, order.fetch, rank(order, hour));
        }
    }

    /// The walk's `FETCH`-th best rank, which must be in hand.
    fn threshold(order: &MatchOrder) -> u64 {
        order.worst().expect("FETCH matches in hand")
    }

    /// The fixture's rows whose hour passes `keep`.
    fn rows_where(keep: impl Fn(i64) -> bool) -> Vec<u32> {
        (0..)
            .zip(HOURS)
            .filter(|&(_, hour)| keep(hour))
            .map(|(row, _)| row)
            .collect()
    }

    /// Hits on the fixture's `rows`.
    fn hits(entry: &SuperfileEntry, rows: &[u32]) -> Vec<SuperfileHit> {
        rows.iter()
            .map(|&local_doc_id| SuperfileHit {
                superfile: entry.uri,
                local_doc_id,
                score: 0.0,
                stable_id: None,
            })
            .collect()
    }

    /// With `FETCH` matches in hand, `keep` drops the hits whose key is below
    /// the `FETCH`-th best, keeps those tied with it, and leaves the
    /// threshold where hits no better than it put it.
    #[test]
    fn keep_drops_hits_below_the_threshold_and_keeps_ties() {
        let (order, reader, entry, r) = fixture();
        seed(&order, &SEEDED);
        let below_top: Vec<u32> = rows_where(|hour| hour <= THRESHOLD_HOUR);
        let kept = reader
            .block_on(order.keep(&r, hits(&entry, &below_top)))
            .expect("keep");
        let kept: Vec<u32> = kept.iter().map(|h| h.local_doc_id).collect();
        assert_eq!(kept, rows_where(|hour| hour == THRESHOLD_HOUR));
        assert_eq!(threshold(&order), rank(&order, THRESHOLD_HOUR));
    }

    /// A superfile hands up at most `FETCH` rows, its best: here the top row
    /// and one of the two tied at the `FETCH`-th best, not both.
    #[test]
    fn keep_hands_up_at_most_fetch_rows_of_ties() {
        let (order, reader, entry, r) = fixture();
        let kept = reader
            .block_on(order.keep(&r, hits(&entry, &rows_where(|_| true))))
            .expect("keep");
        let kept: Vec<i64> = kept
            .iter()
            .map(|h| HOURS[h.local_doc_id as usize])
            .collect();
        assert_eq!(kept.len(), FETCH);
        assert!(
            kept.contains(&60) && kept.contains(&THRESHOLD_HOUR),
            "{kept:?}"
        );
    }

    /// Until `FETCH` matches are in hand nothing is dropped: a hit worse
    /// than every other still counts towards the limit.
    #[test]
    fn keep_drops_nothing_before_fetch_matches_are_in_hand() {
        let (order, reader, entry, r) = fixture();
        let worst_row = rows_where(|hour| hour < THRESHOLD_HOUR);
        let kept = reader
            .block_on(order.keep(&r, hits(&entry, &worst_row[..1])))
            .expect("keep");
        assert_eq!(kept.len(), 1);
        assert_eq!(order.worst(), None);
    }

    /// `prefilter` drops candidates below the `FETCH`-th best and keeps ties,
    /// but a candidate is not yet a match: one better than every match in
    /// hand does not raise the threshold, while the same row as a verified
    /// match through `keep` does.
    #[test]
    fn prefilter_drops_candidates_below_the_threshold_without_raising_it() {
        let (order, reader, entry, r) = fixture();
        let every_row = rows_where(|_| true);
        let unseeded = reader
            .block_on(order.prefilter(&r, every_row.clone()))
            .expect("prefilter");
        assert_eq!(unseeded, every_row, "nothing to compare against yet");

        seed(&order, &SEEDED);
        let placing = reader
            .block_on(order.prefilter(&r, every_row))
            .expect("prefilter");
        assert_eq!(placing, rows_where(|hour| hour >= THRESHOLD_HOUR));
        assert_eq!(
            threshold(&order),
            rank(&order, THRESHOLD_HOUR),
            "candidate keys leave the threshold alone"
        );

        let top = rows_where(|hour| hour > *SEEDED.last().expect("seeded"));
        reader
            .block_on(order.keep(&r, hits(&entry, &top)))
            .expect("keep");
        assert_eq!(
            threshold(&order),
            rank(&order, *SEEDED.last().expect("seeded")),
            "a verified match raises it"
        );
    }

    /// The walk never keeps more superfiles open than the fan-out's cap,
    /// whether it can stop early or not, even a cap below
    /// [`ORDERED_WINDOW`]; a wider cap still starts a walk that can stop
    /// early at [`ORDERED_WINDOW`] and one that cannot at the cap.
    #[test]
    fn the_walk_window_never_passes_the_fan_out_cap() {
        let (mut order, _, _, _) = fixture();
        for stops_early in [true, false] {
            order.stops_early = stops_early;
            for answered in [0, MANY_ANSWERED] {
                assert_eq!(order.walk_window(answered, ONE_IN_FLIGHT), ONE_IN_FLIGHT);
            }
        }
        order.stops_early = true;
        assert_eq!(order.walk_window(0, WIDE_IN_FLIGHT), ORDERED_WINDOW);
        assert_eq!(
            order.walk_window(MANY_ANSWERED, WIDE_IN_FLIGHT),
            WIDE_IN_FLIGHT
        );
        order.stops_early = false;
        assert_eq!(order.walk_window(0, WIDE_IN_FLIGHT), WIDE_IN_FLIGHT);
    }
}
