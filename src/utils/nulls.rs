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

use arrow::compute::take;
use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array, new_null_array};
use arrow_schema::{ArrowError, DataType};

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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow::array::ArrayData;
    use arrow_array::LargeStringArray;
    use arrow_schema::{Field, Schema};

    use super::*;

    /// Rows in the test batches.
    const ROWS: usize = 1024;
    /// Null columns of each type in the wide test batch.
    const NULL_COLUMNS_PER_TYPE: usize = 500;

    /// Bytes the batch's buffers take, each distinct buffer counted once:
    /// what the batch costs, where `get_array_memory_size` counts a shared
    /// buffer once per column it appears in.
    fn distinct_buffer_bytes(batch: &RecordBatch) -> usize {
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
}
