// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Columns that hold nothing, shared rather than allocated one by one.
//!
//! A null array of `n` rows is not free in Arrow: a string column keeps
//! `n + 1` offsets and a validity bitmap whatever its values, so a null
//! `LargeUtf8` column costs about eight bytes a row. On a wide, sparse
//! table, where a batch carries thousands of columns and almost all of
//! them are null, allocating one such array per column makes a batch cost
//! rows × columns × eight bytes when its values cost a small fraction of
//! that. Arrow arrays are immutable and reference-counted, so every null
//! column of one type and length can be the same array: the batch then
//! costs one null array per type plus what its rows actually hold, and
//! anything that reads or encodes it sees exactly the values it saw before.

use std::sync::Arc;

use arrow::compute::{concat, concat_batches, take};
use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array, new_null_array};
use arrow_schema::{ArrowError, DataType, SchemaRef};

/// One all-null array per type, every one `len` rows long, made on first
/// use and handed out to every column of that type that holds nothing.
pub(crate) struct SharedNulls {
    len: usize,
    /// Few distinct types appear in one batch, so a scan beats a map.
    by_type: Vec<(DataType, ArrayRef)>,
}

impl SharedNulls {
    /// Null arrays of `len` rows.
    pub(crate) fn new(len: usize) -> Self {
        Self {
            len,
            by_type: Vec::new(),
        }
    }

    /// The shared null array of `data_type`.
    pub(crate) fn of(&mut self, data_type: &DataType) -> ArrayRef {
        if let Some((_, array)) = self.by_type.iter().find(|(t, _)| t == data_type) {
            return Arc::clone(array);
        }
        let array = new_null_array(data_type, self.len);
        self.by_type.push((data_type.clone(), Arc::clone(&array)));
        array
    }
}

/// Whether `array` is null in every row. Judged by its validity bitmap,
/// so an array that has none (a `Null` array, a union, a run-end encoded
/// array) never counts, and is kept as it is.
fn null_throughout(array: &dyn Array) -> bool {
    !array.is_empty() && array.null_count() == array.len()
}

/// The rows of `batch` at `indices`, as Arrow's `take` gathers them, except
/// that a column null throughout `batch` becomes the shared null array of
/// its type rather than a gathered array as long as `indices`. A column
/// null in every row is null in every row picked from it, so the values
/// are the same; a gather of a null string column would write an offset
/// per index all the same.
pub(crate) fn take_sharing_nulls(
    batch: &RecordBatch,
    indices: &UInt32Array,
) -> Result<RecordBatch, ArrowError> {
    let mut nulls = SharedNulls::new(indices.len());
    let columns = batch
        .columns()
        .iter()
        .map(|column| match null_throughout(column.as_ref()) {
            true => Ok(nulls.of(column.data_type())),
            false => take(column.as_ref(), indices, None),
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(batch.schema(), columns)
}

/// `batch` with every column that is null throughout replaced by the
/// shared null array of its type. The values are the same; the batch then
/// holds one null array per type instead of one per null column, as a merge
/// needs of the batches it decodes: a column an input stores but never set
/// decodes to a full array of nulls.
pub(crate) fn share_null_columns(batch: RecordBatch) -> Result<RecordBatch, ArrowError> {
    if !batch.columns().iter().any(|c| null_throughout(c.as_ref())) {
        return Ok(batch);
    }
    let mut nulls = SharedNulls::new(batch.num_rows());
    let columns = batch
        .columns()
        .iter()
        .map(|column| match null_throughout(column.as_ref()) {
            true => nulls.of(column.data_type()),
            false => Arc::clone(column),
        })
        .collect();
    RecordBatch::try_new(batch.schema(), columns)
}

/// Whether `arrays` share one type and hold no value among them: each is
/// empty or null throughout, and at least one has rows.
fn null_in_all(arrays: &[&dyn Array]) -> bool {
    let Some(first) = arrays.first() else {
        return false;
    };
    arrays.iter().any(|array| !array.is_empty())
        && arrays.iter().all(|array| {
            array.data_type() == first.data_type() && (array.is_empty() || null_throughout(*array))
        })
}

/// `batches` joined one after another under `schema`, as Arrow's
/// `concat_batches` joins them, except that a column null in every batch
/// becomes the shared null array of its type, as long as all their rows,
/// rather than a new array. Arrow's concat hands a lone array back as it
/// is but writes a new one for two or more, so batches whose absent
/// columns share one null array each would otherwise join into a batch
/// with a null array of its whole length for every such column.
pub(crate) fn concat_sharing_nulls<'a>(
    schema: &SchemaRef,
    batches: impl IntoIterator<Item = &'a RecordBatch>,
) -> Result<RecordBatch, ArrowError> {
    let batches: Vec<&RecordBatch> = batches.into_iter().collect();
    if batches.is_empty() || schema.fields().is_empty() {
        return concat_batches(schema, batches);
    }
    let mut nulls = SharedNulls::new(batches.iter().map(|batch| batch.num_rows()).sum());
    let columns = (0..schema.fields().len())
        .map(|i| {
            let arrays: Vec<&dyn Array> = batches
                .iter()
                .map(|batch| batch.column(i).as_ref())
                .collect();
            match null_in_all(&arrays) {
                true => Ok(nulls.of(arrays[0].data_type())),
                false => concat(&arrays),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(Arc::clone(schema), columns)
}

/// Helpers other modules' tests use to measure what a batch costs.
#[cfg(test)]
pub(crate) mod test_support {
    use std::collections::HashSet;

    use arrow::array::ArrayData;
    use arrow_array::RecordBatch;

    /// Bytes the batch's buffers take, each distinct buffer counted once:
    /// what the batch costs, where `get_array_memory_size` counts a shared
    /// buffer once per column it appears in.
    pub(crate) fn distinct_buffer_bytes(batch: &RecordBatch) -> usize {
        fn walk(data: &ArrayData, seen: &mut HashSet<usize>, total: &mut usize) {
            let buffers = data
                .buffers()
                .iter()
                .chain(data.nulls().map(|n| n.buffer()));
            for buffer in buffers {
                if seen.insert(buffer.as_ptr() as usize) {
                    *total += buffer.capacity();
                }
            }
            for child in data.child_data() {
                walk(child, seen, total);
            }
        }
        let mut seen = HashSet::new();
        let mut total = 0;
        for column in batch.columns() {
            walk(&column.to_data(), &mut seen, &mut total);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int64Array, LargeStringArray, NullArray};
    use arrow_schema::{Field, Schema};

    use super::{test_support::distinct_buffer_bytes, *};

    /// Rows in the test batches.
    const ROWS: usize = 1024;
    /// Null columns of each type in the wide test batch.
    const NULL_COLUMNS_PER_TYPE: usize = 500;
    /// Batches with rows that the concat test joins.
    const NARROW_BATCHES: usize = 3;

    /// A batch of `ROWS` rows: one filled string column, then
    /// [`NULL_COLUMNS_PER_TYPE`] null string columns and as many null
    /// integer columns, each allocated on its own as a reader decodes them.
    fn wide_batch() -> RecordBatch {
        let mut fields = vec![Field::new("filled", DataType::LargeUtf8, true)];
        let mut columns: Vec<ArrayRef> = vec![Arc::new(LargeStringArray::from_iter(
            (0..ROWS).map(|i| (i % 3 == 0).then(|| format!("v{i}"))),
        ))];
        for i in 0..NULL_COLUMNS_PER_TYPE {
            fields.push(Field::new(format!("s{i}"), DataType::LargeUtf8, true));
            columns.push(new_null_array(&DataType::LargeUtf8, ROWS));
            fields.push(Field::new(format!("n{i}"), DataType::Int64, true));
            columns.push(new_null_array(&DataType::Int64, ROWS));
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("wide batch")
    }

    /// `filled`, `late`, then [`NULL_COLUMNS_PER_TYPE`] string columns and
    /// as many integer columns, every one nullable.
    fn narrow_schema() -> SchemaRef {
        let mut fields = vec![
            Field::new("filled", DataType::LargeUtf8, true),
            Field::new("late", DataType::LargeUtf8, true),
        ];
        for i in 0..NULL_COLUMNS_PER_TYPE {
            fields.push(Field::new(format!("s{i}"), DataType::LargeUtf8, true));
            fields.push(Field::new(format!("n{i}"), DataType::Int64, true));
        }
        Arc::new(Schema::new(fields))
    }

    /// Batch `b` of the [`NARROW_BATCHES`], `rows` long, over
    /// [`narrow_schema`]: `filled` holds a value in every row, `late` in
    /// every other row of the last batch only, and every other column is
    /// null, filled with one array per type as a batch that lacks the
    /// columns is filled when it is conformed to its table.
    fn narrow_batch(schema: &SchemaRef, b: usize, rows: usize) -> RecordBatch {
        let mut nulls = SharedNulls::new(rows);
        let columns = schema
            .fields()
            .iter()
            .map(|field| -> ArrayRef {
                match field.name().as_str() {
                    "filled" => Arc::new(LargeStringArray::from_iter_values(
                        (0..rows).map(|r| format!("b{b} r{r}")),
                    )),
                    "late" if b == NARROW_BATCHES - 1 => Arc::new(LargeStringArray::from_iter(
                        (0..rows).map(|r| (r % 2 == 0).then(|| format!("late r{r}"))),
                    )),
                    _ => nulls.of(field.data_type()),
                }
            })
            .collect();
        RecordBatch::try_new(Arc::clone(schema), columns).expect("narrow batch")
    }

    #[test]
    fn shared_nulls_hand_out_one_array_per_type() {
        let mut nulls = SharedNulls::new(ROWS);
        let a = nulls.of(&DataType::LargeUtf8);
        let b = nulls.of(&DataType::LargeUtf8);
        let c = nulls.of(&DataType::Int64);
        assert!(Arc::ptr_eq(&a, &b));
        assert!(!Arc::ptr_eq(&a, &c));
        assert_eq!(a.len(), ROWS);
        assert_eq!(a.null_count(), ROWS);
        assert_eq!(c.data_type(), &DataType::Int64);
    }

    /// Gathering rows keeps every value `take` would give, and gathers no
    /// null column: each comes back as its type's one shared array, so
    /// the gathered batch costs its filled column, not its width.
    #[test]
    fn take_sharing_nulls_matches_take_and_gathers_no_null_column() {
        let batch = wide_batch();
        let indices = UInt32Array::from_iter_values((0..ROWS as u32).rev().step_by(2));
        let expected_columns = batch
            .columns()
            .iter()
            .map(|column| take(column.as_ref(), &indices, None))
            .collect::<Result<Vec<_>, _>>()
            .expect("take");
        let expected = RecordBatch::try_new(batch.schema(), expected_columns).expect("batch");
        let taken = take_sharing_nulls(&batch, &indices).expect("take sharing");
        assert_eq!(taken, expected);
        assert!(Arc::ptr_eq(taken.column(1), taken.column(3)));
        assert!(Arc::ptr_eq(taken.column(2), taken.column(4)));
        assert!(distinct_buffer_bytes(&taken) * 100 < distinct_buffer_bytes(&expected));
    }

    /// Joining batches keeps every value `concat_batches` would give, and
    /// writes no column that is null in all of them: each comes back as its
    /// type's one shared array of the joined length, so the joined batch
    /// costs its filled columns, not its width. A column that any batch
    /// fills is joined as Arrow joins it, the other batches' nulls included.
    #[test]
    fn concat_sharing_nulls_matches_concat_batches_and_writes_no_null_column() {
        let schema = narrow_schema();
        let mut batches: Vec<RecordBatch> = (0..NARROW_BATCHES)
            .map(|b| narrow_batch(&schema, b, ROWS))
            .collect();
        // A batch with no rows adds none, and leaves a null column null.
        batches.insert(1, narrow_batch(&schema, 0, 0));
        let expected = concat_batches(&schema, &batches).expect("concat");
        let joined = concat_sharing_nulls(&schema, &batches).expect("concat sharing");
        assert_eq!(joined, expected);
        assert_eq!(joined.num_rows(), NARROW_BATCHES * ROWS);

        let late = joined.column(1);
        assert_eq!(
            late.len() - late.null_count(),
            ROWS / 2,
            "late keeps its values"
        );
        let (strings, ints) = (joined.column(2), joined.column(3));
        assert_eq!(strings.null_count(), joined.num_rows());
        assert_eq!(ints.null_count(), joined.num_rows());
        for pair in joined.columns()[2..].chunks(2) {
            assert!(Arc::ptr_eq(&pair[0], strings), "one null string array");
            assert!(Arc::ptr_eq(&pair[1], ints), "one null integer array");
        }
        assert!(distinct_buffer_bytes(&joined) * 100 < distinct_buffer_bytes(&expected));
    }

    #[test]
    fn sharing_null_columns_keeps_every_value_and_drops_their_cost() {
        let batch = wide_batch();
        let before = distinct_buffer_bytes(&batch);
        let shared = share_null_columns(batch.clone()).expect("share");
        assert_eq!(shared, batch, "the values are the same");
        let after = distinct_buffer_bytes(&shared);
        // The filled column plus one null array per type: a null string
        // array is its offsets and bitmap, a null integer array its values
        // and bitmap, both about eight bytes a row.
        let filled = distinct_buffer_bytes(&batch.project(&[0]).expect("project"));
        let per_type_bound = 2 * (ROWS + 1) * size_of::<i64>();
        assert!(
            after <= filled + per_type_bound * 2,
            "{after} bytes after sharing, {before} before"
        );
        assert!(before > after * NULL_COLUMNS_PER_TYPE / 2);
    }

    #[test]
    fn sharing_keeps_a_column_without_a_validity_bitmap() {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Null, true)]));
        let column: ArrayRef = Arc::new(NullArray::new(ROWS));
        let batch = RecordBatch::try_new(schema, vec![Arc::clone(&column)]).expect("batch");
        let kept = share_null_columns(batch).expect("share");
        assert!(Arc::ptr_eq(kept.column(0), &column));
    }

    #[test]
    fn sharing_passes_a_batch_with_no_null_column_through() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let column: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::clone(&column)]).expect("batch");
        let kept = share_null_columns(batch).expect("share");
        assert!(Arc::ptr_eq(kept.column(0), &column));
    }
}
