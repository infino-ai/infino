// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use std::collections::HashMap;

use arrow_array::{Decimal128Array, RecordBatch};

use crate::{
    superfile::BuildError,
    supertable::{ScalarStatsAgg, schema::FieldId},
};

/// Statistics for a superfile, including the number of documents,
/// id range, and scalar statistics. Usually used during build time.
/// These stats are later stored in SuperfileEntry
#[derive(Debug)]
pub struct SuperfileStats {
    pub n_docs: u64,
    pub id_min: i128,
    pub id_max: i128,
    pub scalar_stats: HashMap<FieldId, ScalarStatsAgg>,
    // TODO: Vector & FTS related stats could also be added here
}

impl SuperfileStats {
    /// Stats of `batch`, some of a superfile's rows, its id column first.
    ///
    /// Its value tables are not held to the superfile's budget: which tables
    /// a superfile keeps depends on all of its rows, so the stats folded from
    /// such batches are fitted once they are all in ([`Self::fit_value_tables`],
    /// which [`Self::from_children`] does). A table past the budget alone is
    /// dropped here already, as it would be from the whole.
    pub fn try_compute_from_record_batch(batch: &RecordBatch) -> Result<Self, BuildError> {
        let schema = batch.schema();
        let id_idx = 0;

        let mut id_min = i128::MAX;
        let mut id_max = i128::MIN;
        let mut n_docs: u64 = 0;

        let id_col = batch
            .column(id_idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| {
                BuildError::IdColumnWrongType(
                    schema.fields[id_idx].name().clone(),
                    "<id column not Decimal128 at runtime>".to_string(),
                )
            })?;
        for i in 0..id_col.len() {
            let v = id_col.value(i);
            id_min = id_min.min(v);
            id_max = id_max.max(v);
        }
        n_docs += id_col.len() as u64;

        let scalar_stats = ScalarStatsAgg::from_batches_unbudgeted(&schema, &[batch]);
        Ok(Self {
            n_docs,
            id_min,
            id_max,
            scalar_stats,
        })
    }

    /// Stats of `batch`, some of a superfile's columns of some of its rows,
    /// the id column not among them: its columns' stats alone. The rows
    /// are counted, and the id range taken, from the batch that holds the
    /// id column. Its value tables are not held to the superfile's budget,
    /// as in [`Self::try_compute_from_record_batch`].
    pub(crate) fn of_columns(batch: &RecordBatch) -> Self {
        Self {
            n_docs: 0,
            id_min: i128::MAX,
            id_max: i128::MIN,
            scalar_stats: ScalarStatsAgg::from_batches_unbudgeted(&batch.schema(), &[batch]),
        }
    }

    /// One superfile's stats, folded from the stats of its parts, its value
    /// tables then fitted to the superfile's budget.
    pub fn from_children(stats: &[Self]) -> Self {
        let mut total = Self {
            n_docs: 0,
            id_min: i128::MAX,
            id_max: i128::MIN,
            scalar_stats: HashMap::new(),
        };
        for stat in stats {
            total.merge(stat);
        }
        total.fit_value_tables();
        total
    }

    /// Fold `other` into these stats, as if both covered one set of rows.
    pub fn merge(&mut self, other: &Self) {
        self.n_docs += other.n_docs;
        self.id_min = self.id_min.min(other.id_min);
        self.id_max = self.id_max.max(other.id_max);
        ScalarStatsAgg::merge(&mut self.scalar_stats, &other.scalar_stats);
    }

    /// Hold these stats' exact value tables to one superfile's budget, once
    /// every batch is folded in. The batches' tables are not budgeted, since
    /// which tables fit depends on every row: a table the budget would drop
    /// among one batch's columns may be one the whole superfile keeps. So the
    /// folded tables are fitted once, and a merge keeps the tables a fresh
    /// build of its rows keeps, however its batches and passes were cut.
    pub(crate) fn fit_value_tables(&mut self) {
        ScalarStatsAgg::fit_value_tables(&mut self.scalar_stats);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, LargeStringArray};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::{superfile::builder::stamp_ids_by_position, test_helpers::decimal128_ids};

    #[test]
    fn try_compute_from_record_batch_single_row() {
        let schema = stamp_ids_by_position(&Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
        ])));
        let ids = decimal128_ids(vec![42u64]);
        let titles = LargeStringArray::from(vec!["hello"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
            .expect("build RecordBatch");

        let stats = SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats");
        assert_eq!(stats.n_docs, 1);
        assert_eq!(stats.id_min, 42);
        assert_eq!(stats.id_max, 42);
        assert_eq!(stats.scalar_stats.len(), 2);
        assert!(stats.scalar_stats.contains_key(&FieldId::ID_COLUMN));
        assert!(stats.scalar_stats.contains_key(&FieldId(1)));
    }

    #[test]
    fn try_compute_from_record_batch_multiple_rows() {
        let schema = stamp_ids_by_position(&Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("text", DataType::LargeUtf8, false),
        ])));
        let ids = decimal128_ids(vec![10u64, 50, 30]);
        let text = LargeStringArray::from(vec!["a", "b", "c"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(text)])
            .expect("build RecordBatch");

        let stats = SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats");
        assert_eq!(stats.n_docs, 3);
        assert_eq!(stats.id_min, 10);
        assert_eq!(stats.id_max, 50);
    }

    #[test]
    fn try_compute_from_record_batch_non_decimal128_id_column_errors() {
        let schema = stamp_ids_by_position(&Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Int64, false),
            Field::new("text", DataType::LargeUtf8, false),
        ])));
        let ids = arrow_array::Int64Array::from(vec![1i64, 2, 3]);
        let text = LargeStringArray::from(vec!["a", "b", "c"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(text)])
            .expect("build RecordBatch");

        let err = SuperfileStats::try_compute_from_record_batch(&batch)
            .expect_err("expected error for non-Decimal128 id column");
        assert!(matches!(err, BuildError::IdColumnWrongType(_, _)));
    }

    #[test]
    fn try_compute_from_record_batch_computes_scalar_stats() {
        let schema = stamp_ids_by_position(&Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("count", DataType::Int64, false),
        ])));
        let ids = decimal128_ids(vec![5u64, 10, 15]);
        let titles = LargeStringArray::from(vec!["apple", "banana", "cherry"]);
        let counts = arrow_array::Int64Array::from(vec![1i64, 2, 3]);
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(ids), Arc::new(titles), Arc::new(counts)],
        )
        .expect("build RecordBatch");

        let stats = SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats");
        assert_eq!(stats.scalar_stats.len(), 3);
        assert!(stats.scalar_stats.contains_key(&FieldId::ID_COLUMN));
        assert!(stats.scalar_stats.contains_key(&FieldId(1)));
        assert!(stats.scalar_stats.contains_key(&FieldId(2)));
    }

    /// A merge folds its statistics a decoded batch at a time, and whether
    /// the merged superfile keeps a column's exact table must not hang on
    /// where those batches were cut. A rating cycling through a hundred
    /// values, in a batch of exactly a hundred rows, is a table of values
    /// that each appear once; folded with batches where they repeat, the
    /// merged superfile still counts the column exactly, so a filtered or
    /// grouped count over it reads the statistics, not the rows.
    #[test]
    fn a_merge_keeps_a_column_table_wherever_its_batches_are_cut() {
        const RATINGS: u64 = 100;
        let schema = stamp_ids_by_position(&Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("rating", DataType::Int64, false),
        ])));
        let batch = |rows: std::ops::Range<u64>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(decimal128_ids(rows.clone())),
                    Arc::new(arrow_array::Int64Array::from_iter_values(
                        rows.map(|doc| (doc % RATINGS) as i64),
                    )),
                ],
            )
            .expect("build RecordBatch")
        };
        let parts = [
            batch(0..8_192),
            batch(8_192..8_192 + RATINGS),
            batch(8_292..9_000),
        ];
        let mut merged = SuperfileStats::from_children(&[]);
        for part in &parts {
            merged.merge(&SuperfileStats::try_compute_from_record_batch(part).expect("stats"));
        }
        let table = merged.scalar_stats[&FieldId(1)]
            .value_counts
            .as_ref()
            .expect("the merged superfile counts the rating exactly");
        assert_eq!(table.entries().len(), RATINGS as usize);
        assert_eq!(
            table.entries().iter().map(|(_, n)| n).sum::<u64>(),
            9_000,
            "every row counted"
        );
    }

    /// A merge keeps the value tables a fresh build of its rows keeps, even
    /// where one batch alone would have dropped a table the whole keeps:
    /// which tables fit the superfile's budget depends on every row, so the
    /// batches' tables are fitted once, after they are folded.
    ///
    /// Setup, three integer columns, 16 bytes a value:
    ///   - `a` holds 4,000 values (64,000 bytes), all of them in the first
    ///     batch; `b` holds 625 there and 4,000 in all (64,000 bytes); `c`
    ///     holds 3,750 there and 4,375 in all (70,000 bytes).
    ///   - The whole keeps `a` and `b` (128,000 bytes) and sketches `c`. The
    ///     first batch alone, past the budget, would have dropped `a`, its
    ///     largest table there.
    #[test]
    fn a_merge_keeps_the_tables_a_fresh_build_keeps_whatever_one_batch_holds() {
        const ROWS: u64 = 4_000;
        let schema = stamp_ids_by_position(&Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Int64, false),
        ])));
        let int =
            |values: Vec<i64>| -> ArrayRef { Arc::new(arrow_array::Int64Array::from(values)) };
        let batch = |ids: std::ops::Range<u64>, a: Vec<i64>, b: Vec<i64>, c: Vec<i64>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(decimal128_ids(ids)), int(a), int(b), int(c)],
            )
            .expect("build RecordBatch")
        };
        let rows = || 0..ROWS as i64;
        let first = batch(
            0..ROWS,
            rows().collect(),
            rows().map(|r| r % 625).collect(),
            rows().map(|r| r % 3_750).collect(),
        );
        let second = batch(
            ROWS..2 * ROWS,
            rows().collect(),
            rows().map(|r| 625 + r % 3_375).collect(),
            rows().map(|r| 3_750 + r % 625).collect(),
        );
        let fresh = ScalarStatsAgg::from_batches(&schema, &[&first, &second]);
        let has_table = |id: u32| fresh[&FieldId(id)].value_counts.is_some();
        assert!(
            has_table(1) && has_table(2) && !has_table(3),
            "guard: a fresh build keeps `a` and `b` and sketches `c`"
        );
        let alone = ScalarStatsAgg::from_batches(&schema, &[&first]);
        assert!(
            alone[&FieldId(1)].value_counts.is_none(),
            "guard: the first batch alone, budgeted, drops `a`"
        );

        let merged = SuperfileStats::from_children(&[
            SuperfileStats::try_compute_from_record_batch(&first).expect("stats"),
            SuperfileStats::try_compute_from_record_batch(&second).expect("stats"),
        ]);
        assert_eq!(merged.scalar_stats, fresh);
    }

    #[test]
    fn from_children_empty_returns_defaults() {
        let result = SuperfileStats::from_children(&[]);
        assert_eq!(result.n_docs, 0);
        assert_eq!(result.id_min, i128::MAX);
        assert_eq!(result.id_max, i128::MIN);
        assert_eq!(result.scalar_stats.len(), 0);
    }

    #[test]
    fn from_children_single_stat_preserves_values() {
        let schema = stamp_ids_by_position(&Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
        ])));
        let ids = decimal128_ids(vec![100u64, 200]);
        let titles = LargeStringArray::from(vec!["first", "second"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
            .expect("build RecordBatch");

        let stat = SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats");
        let merged = SuperfileStats::from_children(&[stat]);

        assert_eq!(merged.n_docs, 2);
        assert_eq!(merged.id_min, 100);
        assert_eq!(merged.id_max, 200);
        assert_eq!(merged.scalar_stats.len(), 2);
    }

    #[test]
    fn from_children_multiple_stats_sums_n_docs() {
        let schema = stamp_ids_by_position(&Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
        ])));

        let stats1 = {
            let ids = decimal128_ids(vec![10u64, 20]);
            let titles = LargeStringArray::from(vec!["a", "b"]);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(titles)])
                .expect("build RecordBatch");
            SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats")
        };

        let stats2 = {
            let ids = decimal128_ids(vec![30u64, 40]);
            let titles = LargeStringArray::from(vec!["c", "d"]);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(titles)])
                .expect("build RecordBatch");
            SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats")
        };

        let merged = SuperfileStats::from_children(&[stats1, stats2]);
        assert_eq!(merged.n_docs, 4);
    }

    #[test]
    fn from_children_multiple_stats_computes_id_min_max() {
        let schema = stamp_ids_by_position(&Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("text", DataType::LargeUtf8, false),
        ])));

        let stats1 = {
            let ids = decimal128_ids(vec![50u64, 75]);
            let text = LargeStringArray::from(vec!["x", "y"]);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(text)])
                .expect("build RecordBatch");
            SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats")
        };

        let stats2 = {
            let ids = decimal128_ids(vec![25u64, 100]);
            let text = LargeStringArray::from(vec!["a", "b"]);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(text)])
                .expect("build RecordBatch");
            SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats")
        };

        let merged = SuperfileStats::from_children(&[stats1, stats2]);
        assert_eq!(merged.id_min, 25);
        assert_eq!(merged.id_max, 100);
    }

    #[test]
    fn from_children_multiple_stats_merges_scalar_stats() {
        let schema = stamp_ids_by_position(&Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("value", DataType::Int64, false),
        ])));

        let stats1 = {
            let ids = decimal128_ids(vec![1u64, 2]);
            let values = arrow_array::Int64Array::from(vec![10i64, 20]);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(values)])
                .expect("build RecordBatch");
            SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats")
        };

        let stats2 = {
            let ids = decimal128_ids(vec![3u64, 4]);
            let values = arrow_array::Int64Array::from(vec![5i64, 15]);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(values)])
                .expect("build RecordBatch");
            SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats")
        };

        let merged = SuperfileStats::from_children(&[stats1, stats2]);
        assert_eq!(merged.scalar_stats.len(), 2);
        assert!(merged.scalar_stats.contains_key(&FieldId::ID_COLUMN));
        assert!(merged.scalar_stats.contains_key(&FieldId(1)));
    }

    #[test]
    fn from_children_three_stats_maintains_correct_min_max() {
        let schema = stamp_ids_by_position(&Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("text", DataType::LargeUtf8, false),
        ])));

        let stats1 = {
            let ids = decimal128_ids(vec![50u64]);
            let text = LargeStringArray::from(vec!["first"]);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(text)])
                .expect("build RecordBatch");
            SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats")
        };

        let stats2 = {
            let ids = decimal128_ids(vec![10u64]);
            let text = LargeStringArray::from(vec!["second"]);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(text)])
                .expect("build RecordBatch");
            SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats")
        };

        let stats3 = {
            let ids = decimal128_ids(vec![100u64]);
            let text = LargeStringArray::from(vec!["third"]);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(ids), Arc::new(text)])
                .expect("build RecordBatch");
            SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats")
        };

        let merged = SuperfileStats::from_children(&[stats1, stats2, stats3]);
        assert_eq!(merged.n_docs, 3);
        assert_eq!(merged.id_min, 10);
        assert_eq!(merged.id_max, 100);
    }
}
