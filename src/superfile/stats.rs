// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use arrow_array::{Decimal128Array, RecordBatch};

use crate::{superfile::BuildError, supertable::ScalarStatsTable};

/// Statistics for a superfile, including the number of documents,
/// id range, and scalar statistics. Usually used during build time.
/// These stats are later stored in SuperfileEntry
#[derive(Debug)]
pub struct SuperfileStats {
    pub n_docs: u64,
    pub id_min: i128,
    pub id_max: i128,
    pub scalar_stats: ScalarStatsTable,
    // TODO: Vector & FTS related stats could also be added here
}

impl SuperfileStats {
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

        let scalar_stats = ScalarStatsTable::from_batch(&schema, batch);
        Ok(Self {
            n_docs,
            id_min,
            id_max,
            scalar_stats,
        })
    }

    pub fn from_children(stats: &[Self]) -> Self {
        let mut n_docs: u64 = 0;
        let mut id_min = i128::MAX;
        let mut id_max = i128::MIN;
        let mut scalar_stats = ScalarStatsTable::default();
        for stat in stats {
            n_docs += stat.n_docs;
            id_min = id_min.min(stat.id_min);
            id_max = id_max.max(stat.id_max);
            scalar_stats.merge(&stat.scalar_stats);
        }
        Self {
            n_docs,
            id_min,
            id_max,
            scalar_stats,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::decimal128_ids;
    use arrow_array::LargeStringArray;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn try_compute_from_record_batch_single_row() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let ids = decimal128_ids(vec![42u64]);
        let titles = LargeStringArray::from(vec!["hello"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
            .expect("build RecordBatch");

        let stats = SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats");
        assert_eq!(stats.n_docs, 1);
        assert_eq!(stats.id_min, 42);
        assert_eq!(stats.id_max, 42);
        assert_eq!(stats.scalar_stats.cols.len(), 2);
        assert!(stats.scalar_stats.cols.contains_key("doc_id"));
        assert!(stats.scalar_stats.cols.contains_key("title"));
    }

    #[test]
    fn try_compute_from_record_batch_multiple_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("text", DataType::LargeUtf8, false),
        ]));
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
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Int64, false),
            Field::new("text", DataType::LargeUtf8, false),
        ]));
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
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("count", DataType::Int64, false),
        ]));
        let ids = decimal128_ids(vec![5u64, 10, 15]);
        let titles = LargeStringArray::from(vec!["apple", "banana", "cherry"]);
        let counts = arrow_array::Int64Array::from(vec![1i64, 2, 3]);
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(ids), Arc::new(titles), Arc::new(counts)],
        )
        .expect("build RecordBatch");

        let stats = SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats");
        assert_eq!(stats.scalar_stats.cols.len(), 3);
        assert!(stats.scalar_stats.cols.contains_key("doc_id"));
        assert!(stats.scalar_stats.cols.contains_key("title"));
        assert!(stats.scalar_stats.cols.contains_key("count"));
    }

    #[test]
    fn from_children_empty_returns_defaults() {
        let result = SuperfileStats::from_children(&[]);
        assert_eq!(result.n_docs, 0);
        assert_eq!(result.id_min, i128::MAX);
        assert_eq!(result.id_max, i128::MIN);
        assert_eq!(result.scalar_stats.cols.len(), 0);
    }

    #[test]
    fn from_children_single_stat_preserves_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let ids = decimal128_ids(vec![100u64, 200]);
        let titles = LargeStringArray::from(vec!["first", "second"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
            .expect("build RecordBatch");

        let stat = SuperfileStats::try_compute_from_record_batch(&batch).expect("compute stats");
        let merged = SuperfileStats::from_children(&[stat]);

        assert_eq!(merged.n_docs, 2);
        assert_eq!(merged.id_min, 100);
        assert_eq!(merged.id_max, 200);
        assert_eq!(merged.scalar_stats.cols.len(), 2);
    }

    #[test]
    fn from_children_multiple_stats_sums_n_docs() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
        ]));

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
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("text", DataType::LargeUtf8, false),
        ]));

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
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("value", DataType::Int64, false),
        ]));

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
        assert_eq!(merged.scalar_stats.cols.len(), 2);
        assert!(merged.scalar_stats.cols.contains_key("doc_id"));
        assert!(merged.scalar_stats.cols.contains_key("value"));
    }

    #[test]
    fn from_children_three_stats_maintains_correct_min_max() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("text", DataType::LargeUtf8, false),
        ]));

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
