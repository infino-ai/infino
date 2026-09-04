// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Schema-driven SQL corpus: derives a queryable Arrow schema from a
//! parquet dataset's own columns (no fixed [`crate::harness::SqlRow`]
//! fixture) and streams the shards through it, converting `Binary` /
//! `LargeBinary` columns to `Utf8` since the SQL engines under test don't
//! index raw bytes, and the ClickBench `EventDate` column to `Date32`.

use std::{fs::File, str::from_utf8, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BinaryArray, Date32Array, Int16Array, Int32Array, Int64Array,
    LargeBinaryArray, RecordBatch, RecordBatchReader, StringArray, UInt16Array, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use infino::superfile::vector::distance::Metric;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::{
    corpus::{CorpusSource, PARQUET_BATCH_ROWS, PARQUET_VECTOR_COLUMNS, parquet_shards_for},
    harness::{SqlCorpusSpec, SqlVectorSpec},
};

/// The ClickBench `hits` dataset's day-count column. Named here (not
/// inline) so the int -> Date32 special case is visible in one place;
/// kept in parity with the upstream Infino ClickBench harness
/// (`infino/bench/src/main.rs`) so query results and row counts are
/// comparable to published numbers.
const CLICKBENCH_EVENT_DATE_COLUMN: &str = "EventDate";

/// A schema-driven SQL corpus: a dataset's own Arrow schema (binary
/// columns rewritten to text) plus the batches read up to `max_rows`.
pub struct ParquetSqlCorpus {
    spec: SqlCorpusSpec,
    batches: Vec<RecordBatch>,
    lossy_rows: usize,
}

impl ParquetSqlCorpus {
    pub fn spec(&self) -> &SqlCorpusSpec {
        &self.spec
    }

    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    pub fn n_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }

    /// Rows across the whole corpus that needed lossy UTF-8 replacement.
    pub fn lossy_rows(&self) -> usize {
        self.lossy_rows
    }
}

/// Map each field's type to what the SQL engines under test can index:
/// `EventDate` becomes `Date32` (ClickBench's `hits` stores it as an
/// integer day count), `Binary` / `LargeBinary` become `Utf8`, everything
/// else passes through unchanged. Nullability is preserved.
pub(crate) fn cast_schema_for_sql(schema: &Schema) -> SchemaRef {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| {
            if f.name() == CLICKBENCH_EVENT_DATE_COLUMN {
                Field::new(f.name(), DataType::Date32, f.is_nullable())
            } else {
                match f.data_type() {
                    DataType::Binary | DataType::LargeBinary => {
                        Field::new(f.name(), DataType::Utf8, f.is_nullable())
                    }
                    _ => f.as_ref().clone(),
                }
            }
        })
        .collect();
    Arc::new(Schema::new(fields))
}

/// Convert one `Binary`/`LargeBinary` column to `Utf8`. Null stays null;
/// valid UTF-8 passes through; invalid bytes are replaced lossily and
/// counted. Not the arrow `cast` kernel: `cast` errors on invalid UTF-8,
/// and real datasets (ClickBench `hits` included) carry some.
pub(crate) fn binary_to_utf8(array: &ArrayRef) -> (ArrayRef, usize) {
    let mut lossy_rows = 0;
    let mut convert = |bytes: Option<&[u8]>| -> Option<String> {
        let bytes = bytes?;
        Some(match from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(_) => {
                lossy_rows += 1;
                String::from_utf8_lossy(bytes).into_owned()
            }
        })
    };
    let values: Vec<Option<String>> = if let Some(a) = array.as_any().downcast_ref::<BinaryArray>()
    {
        a.iter().map(&mut convert).collect()
    } else if let Some(a) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        a.iter().map(&mut convert).collect()
    } else {
        panic!(
            "binary_to_utf8 expects Binary or LargeBinary, got {:?}",
            array.data_type()
        )
    };
    (Arc::new(StringArray::from(values)), lossy_rows)
}

/// Convert an integer day-count column (`EventDate`'s on-disk type) to
/// `Date32`, whose physical representation *is* a day count — no rounding
/// or timezone math, just a widening reinterpret. Mirrors datafusion's
/// `CAST(CAST(.. AS INTEGER) AS DATE)`, which the upstream harness relies
/// on to make the same source column queryable as a date. Idempotent: a
/// column already stored as `Date32` (or another already-supported target
/// type) passes through unchanged, since `cast_schema_for_sql` makes such
/// columns a schema no-op and this must not then panic on a downcast.
/// Out-of-range values become NULL (a SAFE cast, matching upstream's
/// `cast(.., Int32)`) instead of silently wrapping into a wrong date.
fn event_date_to_date32(array: &ArrayRef) -> ArrayRef {
    match array.data_type() {
        DataType::Date32 => Arc::clone(array),
        DataType::UInt16 => {
            let days: Vec<Option<i32>> = array
                .as_any()
                .downcast_ref::<UInt16Array>()
                .expect("invariant: data_type checked above")
                .iter()
                .map(|v| v.map(i32::from))
                .collect();
            Arc::new(Date32Array::from(days))
        }
        DataType::UInt32 => {
            let days: Vec<Option<i32>> = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .expect("invariant: data_type checked above")
                .iter()
                .map(|v| v.and_then(|v| i32::try_from(v).ok()))
                .collect();
            Arc::new(Date32Array::from(days))
        }
        DataType::Int16 => {
            let days: Vec<Option<i32>> = array
                .as_any()
                .downcast_ref::<Int16Array>()
                .expect("invariant: data_type checked above")
                .iter()
                .map(|v| v.map(i32::from))
                .collect();
            Arc::new(Date32Array::from(days))
        }
        DataType::Int32 => {
            let days: Vec<Option<i32>> = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("invariant: data_type checked above")
                .iter()
                .collect();
            Arc::new(Date32Array::from(days))
        }
        DataType::Int64 => {
            let days: Vec<Option<i32>> = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("invariant: data_type checked above")
                .iter()
                .map(|v| v.and_then(|v| i32::try_from(v).ok()))
                .collect();
            Arc::new(Date32Array::from(days))
        }
        other => panic!(
            "{CLICKBENCH_EVENT_DATE_COLUMN} column has unsupported type {other:?}: expected an \
             integer day count or an already-converted Date32"
        ),
    }
}

/// Convert every `Binary`/`LargeBinary` column of one batch to `Utf8` and
/// `EventDate` (int day count) to `Date32`, against the already-derived
/// `schema`. Returns the converted batch and the number of rows across
/// the batch that needed lossy UTF-8 replacement.
fn convert_batch(schema: &SchemaRef, batch: &RecordBatch) -> (RecordBatch, usize) {
    let mut lossy_rows = 0;
    let columns: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, column)| {
            if field.name() == CLICKBENCH_EVENT_DATE_COLUMN {
                event_date_to_date32(column)
            } else if field.data_type() == &DataType::Utf8
                && matches!(column.data_type(), DataType::Binary | DataType::LargeBinary)
            {
                let (converted, rows) = binary_to_utf8(column);
                lossy_rows += rows;
                converted
            } else {
                Arc::clone(column)
            }
        })
        .collect();
    let converted = RecordBatch::try_new(Arc::clone(schema), columns)
        .expect("converted columns match the derived schema");
    (converted, lossy_rows)
}

/// The `dim` of an embedding column, when `data_type` is a
/// `FixedSizeList<Float32>` — the only vector encoding this corpus reads.
fn fixed_size_float32_dim(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::FixedSizeList(item, size) if *item.data_type() == DataType::Float32 => {
            Some(*size as usize)
        }
        _ => None,
    }
}

/// Derive a [`SqlCorpusSpec`] from a dataset's own schema: no FTS columns
/// (see [`SqlCorpusSpec::fts_columns`]), and a vector spec only when one of
/// [`PARQUET_VECTOR_COLUMNS`] is present as `FixedSizeList<Float32>`.
pub(crate) fn spec_from_schema(schema: SchemaRef) -> SqlCorpusSpec {
    let vector = PARQUET_VECTOR_COLUMNS.iter().find_map(|name| {
        let field = schema.column_with_name(name)?.1;
        let dim = fixed_size_float32_dim(field.data_type())?;
        Some(SqlVectorSpec {
            column: (*name).to_string(),
            dim,
            metric: Metric::Cosine,
        })
    });
    SqlCorpusSpec {
        schema,
        fts_columns: Vec::new(),
        vector,
    }
}

/// Read a parquet dataset's shards in order, deriving the SQL-facing
/// schema from the first shard and streaming batches (converted to that
/// schema) until `max_rows` rows are collected or the shards are
/// exhausted. Prints one lossy-UTF-8 summary line for the whole corpus.
pub fn open(source: &CorpusSource, max_rows: usize) -> ParquetSqlCorpus {
    let shards = parquet_shards_for(source);
    let mut spec: Option<SqlCorpusSpec> = None;
    let mut batches = Vec::new();
    let mut lossy_rows = 0;
    let mut n_rows = 0;
    'shards: for shard in &shards {
        let file = File::open(shard).unwrap_or_else(|e| panic!("open {}: {e}", shard.display()));
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap_or_else(|e| panic!("read {}: {e}", shard.display()))
            .with_batch_size(PARQUET_BATCH_ROWS)
            .build()
            .unwrap_or_else(|e| panic!("build reader {}: {e}", shard.display()));
        let schema = spec
            .get_or_insert_with(|| spec_from_schema(cast_schema_for_sql(&reader.schema())))
            .schema
            .clone();
        for batch in reader {
            let batch = batch.unwrap_or_else(|e| panic!("read batch {}: {e}", shard.display()));
            // Slice to the retained prefix BEFORE converting: `lossy_rows` must count
            // only rows that end up in `batches`, never a discarded straddling tail.
            let keep = batch.num_rows().min(max_rows - n_rows);
            let batch = if keep < batch.num_rows() {
                batch.slice(0, keep)
            } else {
                batch
            };
            let (converted, rows) = convert_batch(&schema, &batch);
            lossy_rows += rows;
            n_rows += converted.num_rows();
            batches.push(converted);
            if n_rows >= max_rows {
                break 'shards;
            }
        }
    }
    if lossy_rows > 0 {
        eprintln!(
            "[corpus/sql] {lossy_rows} rows contained invalid UTF-8 and were replaced lossily"
        );
    }
    ParquetSqlCorpus {
        spec: spec.unwrap_or_else(|| panic!("parquet source has no shards")),
        batches,
        lossy_rows,
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Date32Array, UInt16Array};
    use chrono::NaiveDate;
    use parquet::arrow::ArrowWriter;
    use tempfile::TempDir;

    use super::*;

    /// Rows in the straddling-batch regression fixture's first (full) batch.
    const STRADDLE_FIRST_BATCH_ROWS: usize = PARQUET_BATCH_ROWS;
    /// Rows in the fixture's second batch, which straddles `max_rows`.
    const STRADDLE_SECOND_BATCH_ROWS: usize = 300;
    /// Rows kept from the second batch — the rest is discarded as over quota.
    const STRADDLE_RETAINED_FROM_SECOND_BATCH: usize = 100;
    /// `max_rows` for the fixture: exactly the retained-row count.
    const STRADDLE_MAX_ROWS: usize =
        STRADDLE_FIRST_BATCH_ROWS + STRADDLE_RETAINED_FROM_SECOND_BATCH;
    /// Embedding width for the vector-spec-derivation fixture below.
    const EMBEDDING_TEST_DIM: i32 = 8;
    /// Day count for the `EventDate` conversion test: 2013-07-14, verified
    /// independently against `chrono` in the test rather than asserted by
    /// comment alone (2013-07-01 is day 15887, not this constant).
    const EVENT_DATE_DAY_COUNT: u16 = 15900;

    #[test]
    fn binary_columns_become_utf8_in_the_derived_schema() {
        let schema = Schema::new(vec![
            Field::new("Title", DataType::Binary, true),
            Field::new("URL", DataType::LargeBinary, true),
            Field::new("UserID", DataType::Int64, false),
        ]);
        let out = cast_schema_for_sql(&schema);
        assert_eq!(out.field(0).data_type(), &DataType::Utf8);
        assert_eq!(out.field(1).data_type(), &DataType::Utf8);
        assert_eq!(
            out.field(2).data_type(),
            &DataType::Int64,
            "non-binary columns must pass through untouched"
        );
    }

    #[test]
    fn event_date_column_becomes_date32_in_the_derived_schema() {
        let schema = Schema::new(vec![
            Field::new(CLICKBENCH_EVENT_DATE_COLUMN, DataType::UInt16, false),
            Field::new("UserID", DataType::Int64, false),
        ]);
        let out = cast_schema_for_sql(&schema);
        assert_eq!(
            out.field(0).data_type(),
            &DataType::Date32,
            "EventDate must be cast to Date32 for parity with the upstream harness"
        );
        assert_eq!(out.field(1).data_type(), &DataType::Int64);
    }

    #[test]
    fn event_date_values_convert_from_integer_day_count_to_date32() {
        // Independently derive the expected day count from a real calendar
        // date so this test can actually catch an epoch or interpretation
        // error, rather than re-asserting the conversion's own identity map.
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("valid epoch date");
        let calendar_date = NaiveDate::from_ymd_opt(2013, 7, 14).expect("valid calendar date");
        let expected_days = i32::try_from((calendar_date - epoch).num_days())
            .expect("day offset fits in i32 for a 21st-century date");
        assert_eq!(
            EVENT_DATE_DAY_COUNT as i32, expected_days,
            "EVENT_DATE_DAY_COUNT must match the calendar date this test claims it represents"
        );

        let src_schema = Arc::new(Schema::new(vec![Field::new(
            CLICKBENCH_EVENT_DATE_COLUMN,
            DataType::UInt16,
            true,
        )]));
        let array: ArrayRef = Arc::new(UInt16Array::from(vec![
            Some(EVENT_DATE_DAY_COUNT),
            None,
            Some(u16::MAX),
        ]));
        let batch = RecordBatch::try_new(Arc::clone(&src_schema), vec![array]).expect("batch");

        let target_schema = cast_schema_for_sql(&src_schema);
        let (converted, lossy_rows) = convert_batch(&target_schema, &batch);

        assert_eq!(lossy_rows, 0);
        let dates = converted
            .column(0)
            .as_any()
            .downcast_ref::<Date32Array>()
            .expect("Date32 column");
        assert_eq!(dates.value(0), expected_days);
        assert!(dates.is_null(1), "a NULL input must stay NULL");
        assert_eq!(
            dates.value(2),
            u16::MAX as i32,
            "u16::MAX must widen exactly with no truncation"
        );
    }

    #[test]
    fn event_date_column_already_date32_passes_through_unchanged() {
        // A dataset that already stores `EventDate` as `Date32` makes
        // `cast_schema_for_sql` a schema no-op; `convert_batch` must not then
        // panic trying to downcast a Date32 array as an integer day count.
        let schema = Arc::new(Schema::new(vec![Field::new(
            CLICKBENCH_EVENT_DATE_COLUMN,
            DataType::Date32,
            true,
        )]));
        let array: ArrayRef = Arc::new(Date32Array::from(vec![
            Some(EVENT_DATE_DAY_COUNT as i32),
            None,
        ]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array]).expect("batch");

        let target_schema = cast_schema_for_sql(&schema);
        let (converted, lossy_rows) = convert_batch(&target_schema, &batch);

        assert_eq!(lossy_rows, 0);
        let dates = converted
            .column(0)
            .as_any()
            .downcast_ref::<Date32Array>()
            .expect("Date32 column");
        assert_eq!(dates.value(0), EVENT_DATE_DAY_COUNT as i32);
        assert!(dates.is_null(1), "a NULL input must stay NULL");
    }

    #[test]
    fn event_date_out_of_range_integers_become_null_instead_of_wrapping() {
        // A SAFE cast: values that don't fit in i32 must become NULL,
        // matching upstream's `cast(col, Int32)`, never a silently wrapped
        // (and wrong) date.
        let schema = Arc::new(Schema::new(vec![Field::new(
            CLICKBENCH_EVENT_DATE_COLUMN,
            DataType::Int64,
            true,
        )]));
        let array: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(EVENT_DATE_DAY_COUNT as i64),
            Some(i64::from(i32::MAX) + 1),
            Some(i64::from(i32::MIN) - 1),
        ]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array]).expect("batch");

        let target_schema = cast_schema_for_sql(&schema);
        let (converted, lossy_rows) = convert_batch(&target_schema, &batch);

        assert_eq!(lossy_rows, 0);
        let dates = converted
            .column(0)
            .as_any()
            .downcast_ref::<Date32Array>()
            .expect("Date32 column");
        assert_eq!(dates.value(0), EVENT_DATE_DAY_COUNT as i32);
        assert!(dates.is_null(1), "i32::MAX + 1 must not silently wrap");
        assert!(dates.is_null(2), "i32::MIN - 1 must not silently wrap");
    }

    #[test]
    fn invalid_utf8_is_replaced_and_counted_once_per_row() {
        // Second value is invalid UTF-8 (0xff is never a valid lead byte).
        let array: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(b"ok".as_slice()),
            Some(&[0xffu8, 0xfe][..]),
            None,
        ]));
        let (converted, replaced) = binary_to_utf8(&array);
        assert_eq!(replaced, 1, "exactly one row had invalid bytes");
        let strings = converted
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("utf8");
        assert_eq!(strings.value(0), "ok");
        assert!(strings.is_null(2), "nulls stay null");
    }

    #[test]
    fn spec_has_no_fts_columns_and_no_vector_without_an_embedding_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("Title", DataType::LargeUtf8, true),
            Field::new("UserID", DataType::Int64, false),
        ]));
        let spec = spec_from_schema(schema);
        assert!(
            spec.fts_columns.is_empty(),
            "schema-driven corpora index no FTS columns"
        );
        assert!(spec.vector.is_none());
    }

    /// A `FixedSizeList<Float32>` column named after one of
    /// [`PARQUET_VECTOR_COLUMNS`] must be picked up as the corpus's vector
    /// spec, with `dim` read from the list size.
    #[test]
    fn spec_has_a_vector_spec_when_an_embedding_column_is_present() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("Title", DataType::LargeUtf8, true),
            Field::new(
                "emb",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    EMBEDDING_TEST_DIM,
                ),
                true,
            ),
        ]));
        let spec = spec_from_schema(schema);
        let vector = spec
            .vector
            .expect("FixedSizeList<Float32> column must yield a vector spec");
        assert_eq!(vector.column, "emb");
        assert_eq!(vector.dim, EMBEDDING_TEST_DIM as usize);
    }

    #[test]
    fn lossy_rows_excludes_rows_discarded_by_the_max_rows_truncation() {
        // First batch fills PARQUET_BATCH_ROWS with valid ASCII. Second batch
        // straddles `max_rows`: its retained prefix is valid UTF-8, its
        // discarded tail is invalid UTF-8 that must never be counted.
        let mut values: Vec<Vec<u8>> = (0..STRADDLE_FIRST_BATCH_ROWS)
            .map(|i| format!("row{i}").into_bytes())
            .collect();
        for i in 0..STRADDLE_SECOND_BATCH_ROWS {
            let global = STRADDLE_FIRST_BATCH_ROWS + i;
            values.push(if i < STRADDLE_RETAINED_FROM_SECOND_BATCH {
                format!("row{global}").into_bytes()
            } else {
                vec![0xffu8, 0xfe]
            });
        }
        let array: ArrayRef = Arc::new(BinaryArray::from(
            values.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        ));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "junk",
            DataType::Binary,
            true,
        )]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array]).expect("batch");

        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("shard-0.parquet");
        let file = File::create(&path).expect("create parquet shard");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("arrow writer");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close writer");

        let source = CorpusSource::LocalParquet {
            dir: dir.path().to_path_buf(),
        };
        let corpus = open(&source, STRADDLE_MAX_ROWS);

        assert_eq!(
            corpus.n_rows(),
            STRADDLE_MAX_ROWS,
            "truncates exactly at max_rows"
        );
        assert_eq!(
            corpus.lossy_rows(),
            0,
            "invalid UTF-8 in the discarded tail must not be counted"
        );
    }
}
