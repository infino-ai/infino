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
//! statistics for and no nulls in (a null sorts first under a descending
//! sort, outside every range), under a `WHERE` it can check itself; any
//! other shape opens every superfile, as before.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    sync::{Arc, Mutex},
};

use arrow::compute::cast;
use arrow_array::{Array, Int64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::{error::Result as DfResult, scalar::ScalarValue};
use roaring::RoaringBitmap;

use crate::{
    runtime_metrics::op_stats::OpStatsCollector,
    superfile::SuperfileReader,
    supertable::{
        error::QueryError,
        handle::SupertableReader,
        manifest::{ManifestSnapshot, SuperfileEntry, SuperfileUri},
        query::{
            SuperfileHit,
            exec::common::{PushedPredicate, RowCheck, take_file_columns},
        },
        schema::FieldId,
    },
};

/// Superfiles an ordered walk opens at once: enough to overlap one open's
/// round trip with the next, few enough that the walk stops within a
/// superfile or two of the last one that placed a row.
pub(crate) const ORDERED_WINDOW: usize = 2;

/// An `ORDER BY column LIMIT n` a match walk honours: the column, its
/// direction, the limit, and the `WHERE` it checks to know which rows count.
pub(crate) struct MatchOrder {
    manifest: Arc<ManifestSnapshot>,
    column: String,
    field: FieldId,
    descending: bool,
    fetch: usize,
    check: Option<RowCheck>,
    key_schema: SchemaRef,
    op_stats: Option<Arc<OpStatsCollector>>,
    /// Each candidate superfile's best rank (see [`Self::rank`]), set by
    /// [`Self::order`].
    best_of: Mutex<HashMap<SuperfileUri, i128>>,
    /// The ranks of the `fetch` best keys found so far, the worst on top.
    found: Mutex<BinaryHeap<Reverse<i128>>>,
}

impl MatchOrder {
    /// The order `ORDER BY column [DESC] LIMIT fetch` asks for over a match
    /// under `predicate`, the `WHERE` pushed to the match (`None` when there
    /// is none). `None` when the walk cannot honour it: the column is not a
    /// stored integer, date or timestamp column, or the `WHERE` reads a
    /// column the walk cannot check before ranking.
    pub(crate) fn new(
        reader: &SupertableReader,
        column: &str,
        descending: bool,
        fetch: usize,
        predicate: Option<&PushedPredicate>,
        output_schema: &SchemaRef,
    ) -> DfResult<Option<Self>> {
        let manifest = Arc::clone(reader.manifest());
        let Some(field) = manifest.field_id(column) else {
            return Ok(None);
        };
        let stored = manifest.stored_schema();
        let Ok(data_type) = stored
            .field_with_name(column)
            .map(|f| f.data_type().clone())
        else {
            return Ok(None);
        };
        if !orderable(&data_type) || fetch == 0 {
            return Ok(None);
        }
        let check = match predicate {
            Some(predicate) => match RowCheck::new(reader, predicate, output_schema)? {
                Some(check) => Some(check),
                None => return Ok(None),
            },
            None => None,
        };
        let key_schema = Arc::new(Schema::new(vec![Field::new(column, data_type, true)]));
        Ok(Some(Self {
            manifest,
            column: column.to_owned(),
            field,
            descending,
            fetch,
            check,
            key_schema,
            op_stats: reader.op_stats.clone(),
            best_of: Mutex::new(HashMap::new()),
            found: Mutex::new(BinaryHeap::new()),
        }))
    }

    /// A key's rank in the walk: higher is better, whichever the sort's
    /// direction.
    fn rank(&self, key: i64) -> i128 {
        match self.descending {
            true => i128::from(key),
            false => -i128::from(key),
        }
    }

    /// `units` in the walk's order, best range first; `false`, leaving them
    /// as they are, when a superfile holds no statistics for the column or
    /// may hold a null in it: the walk cannot then know what it leaves out.
    pub(crate) fn order<P>(&self, units: &mut [(Arc<SuperfileEntry>, P)]) -> bool {
        let mut best_of = HashMap::with_capacity(units.len());
        for (entry, _) in units.iter() {
            let Some(stats) = entry.scalar_stats.column(self.field) else {
                return false;
            };
            if stats.null_count() != Some(0) {
                return false;
            }
            let Some((min, max)) = stats.min_max_values() else {
                return false;
            };
            let best = match self.descending {
                true => key_of(&max),
                false => key_of(&min),
            };
            let Some(best) = best else {
                return false;
            };
            best_of.insert(entry.uri, self.rank(best));
        }
        units.sort_by_key(|(entry, _)| Reverse(best_of[&entry.uri]));
        *self.best_of.lock().expect("match order mutex poisoned") = best_of;
        true
    }

    /// Whether superfile `uri` can no longer place a row: `fetch` rows are
    /// in hand and its best key is worse than the worst of them. A key that
    /// ties the worst is kept, since the sort may break the tie for it.
    pub(crate) fn passed(&self, uri: &SuperfileUri) -> bool {
        let Some(best) = self
            .best_of
            .lock()
            .expect("match order mutex poisoned")
            .get(uri)
            .copied()
        else {
            return false;
        };
        let found = self.found.lock().expect("match order mutex poisoned");
        if found.len() < self.fetch {
            return false;
        }
        found.peek().is_some_and(|Reverse(worst)| best < *worst)
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
        let keys = cast(batch.column(0), &DataType::Int64).map_err(QueryError::internal)?;
        let keys = keys
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| QueryError::Internal("an ordered match key is not an integer".into()))?;
        let mut found = self.found.lock().expect("match order mutex poisoned");
        for key in keys.iter().flatten() {
            let rank = self.rank(key);
            if found.len() < self.fetch {
                found.push(Reverse(rank));
            } else if found.peek().is_some_and(|Reverse(worst)| rank > *worst) {
                found.pop();
                found.push(Reverse(rank));
            }
        }
        Ok(hits)
    }
}

/// Whether a column of `data_type` orders as its 64-bit integer value.
fn orderable(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
    )
}

/// A statistic of an [`orderable`] column as the 64-bit value it orders as.
fn key_of(value: &ScalarValue) -> Option<i64> {
    match value {
        ScalarValue::Int8(Some(v)) => Some(i64::from(*v)),
        ScalarValue::Int16(Some(v)) => Some(i64::from(*v)),
        ScalarValue::Int32(Some(v)) => Some(i64::from(*v)),
        ScalarValue::Int64(Some(v)) => Some(*v),
        ScalarValue::UInt8(Some(v)) => Some(i64::from(*v)),
        ScalarValue::UInt16(Some(v)) => Some(i64::from(*v)),
        ScalarValue::UInt32(Some(v)) => Some(i64::from(*v)),
        ScalarValue::Date32(Some(v)) => Some(i64::from(*v)),
        ScalarValue::Date64(Some(v))
        | ScalarValue::TimestampSecond(Some(v), _)
        | ScalarValue::TimestampMillisecond(Some(v), _)
        | ScalarValue::TimestampMicrosecond(Some(v), _)
        | ScalarValue::TimestampNanosecond(Some(v), _) => Some(*v),
        _ => None,
    }
}
