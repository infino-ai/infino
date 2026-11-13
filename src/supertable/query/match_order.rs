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
//! superfiles that cannot place a row. It applies to a single sort column
//! of integer, date or timestamp type that every candidate superfile holds
//! current statistics for and no nulls in (a null sorts first under a
//! descending sort, outside every range), under a `WHERE` it can evaluate
//! from stored columns; any other shape opens every superfile, as before.

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    sync::{Arc, Mutex},
};

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::{error::Result as DfResult, logical_expr::Expr};
use roaring::RoaringBitmap;

use crate::{
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
            exec::common::{PushedPredicate, RowCheck, take_file_columns},
            skip::ColumnTypeGuard,
        },
        schema::FieldId,
    },
};

/// Superfiles an ordered walk opens at once once it holds its rows: enough
/// to overlap one open's round trip with the next, few enough that the
/// walk stops within a superfile or two of the last one that placed a row.
pub(crate) const ORDERED_WINDOW: usize = 2;

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
    /// columns, so it could not tell which rows count.
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
        Ok(Some(Self {
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
            found: Mutex::new(BinaryHeap::new()),
        }))
    }

    /// A key's rank in the walk: higher is better, whichever the sort's
    /// direction.
    fn rank(&self, key: u64) -> u64 {
        match self.descending {
            true => key,
            false => !key,
        }
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
            .map(|(entry, _)| {
                if self.guard.absent_from(entry) || self.guard.stats_are_stale(entry) {
                    return None;
                }
                let stats = entry.scalar_stats.column(self.field)?;
                if stats.null_count() != Some(0) {
                    return None;
                }
                let (min, max) = stats.min_max_values()?;
                let best = match self.descending {
                    true => max,
                    false => min,
                };
                let (_, key) = literal_key(&best, &self.data_type)?;
                Some(self.rank(key))
            })
            .collect()
    }

    /// Whether a superfile whose best rank is `best` can no longer place a
    /// row: `fetch` rows are in hand and its best key is worse than the
    /// worst of them.
    pub(crate) fn passed(&self, best: u64) -> bool {
        let found = self.found.lock().expect("match order mutex poisoned");
        found.len() >= self.fetch && found.peek().is_some_and(|Reverse(worst)| best < *worst)
    }

    /// Superfiles to keep open at once once `answered` have answered. While
    /// the walk holds fewer than `fetch` rows, as many as have answered, up
    /// to `widest`: under a `WHERE` few rows pass, a fixed window would open
    /// every superfile [`ORDERED_WINDOW`] at a time. Once it holds them,
    /// [`ORDERED_WINDOW`], so it stops soon after the last row placed.
    pub(crate) fn window(&self, answered: usize, widest: usize) -> usize {
        let short = self.found.lock().expect("match order mutex poisoned").len() < self.fetch;
        match short {
            true => answered.clamp(ORDERED_WINDOW, widest.max(ORDERED_WINDOW)),
            false => ORDERED_WINDOW,
        }
    }

    /// Of `hits`, the matches of one superfile read through `r`, those the
    /// `WHERE` keeps, with their keys folded into the walk's best.
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
        let hits: Vec<SuperfileHit> = hits
            .into_iter()
            .filter(|h| kept.contains(h.local_doc_id))
            .collect();
        if hits.is_empty() {
            return Ok(hits);
        }
        let locals: Vec<u32> = kept.iter().collect();
        let batch = take_file_columns(
            &self.manifest,
            r,
            locals,
            &[self.column.as_str()],
            &self.key_schema,
            &self.op_stats,
        )
        .await
        .map_err(QueryError::DataFusion)?;
        let mut found = self.found.lock().expect("match order mutex poisoned");
        push_keys(batch.column(0).as_ref(), self.kind, 0, |key, _| {
            let rank = self.rank(key);
            if found.len() < self.fetch {
                found.push(Reverse(rank));
            } else if found.peek().is_some_and(|Reverse(worst)| rank > *worst) {
                found.pop();
                found.push(Reverse(rank));
            }
            Ok(())
        })
        .map_err(QueryError::internal)?;
        Ok(hits)
    }
}
