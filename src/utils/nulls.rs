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

use std::{collections::HashSet, sync::Arc};

use arrow::compute::concat;
use arrow_array::{Array, ArrayRef, RecordBatch, new_null_array};
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};

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

/// The names of the columns of `batches` that hold a value in some row: a
/// column null throughout every batch is left out. Judged as
/// [`null_throughout`] judges, so a column with no validity bitmap counts
/// as holding values.
pub(crate) fn populated_columns<'a>(
    batches: impl IntoIterator<Item = &'a RecordBatch>,
) -> HashSet<String> {
    let mut populated = HashSet::new();
    for batch in batches {
        for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
            if !populated.contains(field.name()) && !null_throughout(column.as_ref()) {
                populated.insert(field.name().clone());
            }
        }
    }
    populated
}

/// `batch` with every column that is null throughout replaced by the
/// shared null array of its type. The values are the same; the batch then
/// holds one null array per type instead of one per null column.
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

/// `batches`, all of `schema`, joined into one batch, as
/// Arrow's `concat_batches` joins them, except that a
/// column null throughout every batch becomes the shared null array of its
/// type rather than a fresh array as long as all of them.
pub(crate) fn concat_sharing_nulls(
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<RecordBatch, ArrowError> {
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(Arc::clone(schema)));
    }
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let mut nulls = SharedNulls::new(rows);
    let columns = (0..schema.fields().len())
        .map(|i| -> Result<ArrayRef, ArrowError> {
            let parts: Vec<&dyn Array> = batches.iter().map(|b| b.column(i).as_ref()).collect();
            if rows > 0 && parts.iter().all(|p| p.is_empty() || null_throughout(*p)) {
                return Ok(nulls.of(schema.field(i).data_type()));
            }
            concat(&parts)
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(Arc::clone(schema), columns)
}

/// Bytes one row of `schema` takes decoded into Arrow when every column in
/// it is null: what each column keeps a row whatever it holds (offsets for
/// a string or list, the value slot of a fixed-width type), and nothing for
/// the values a row adds on top. A floor on a decoded row, and on a wide,
/// sparse table close to all of it: rows × this is what a batch of the
/// table decodes to before any of it is shared.
pub(crate) fn null_row_bytes(schema: &Schema) -> u64 {
    schema
        .fields()
        .iter()
        .map(|field| null_slot_bytes(field.data_type()))
        .sum()
}

/// Bytes a row of a column of `data_type` keeps when it is null.
fn null_slot_bytes(data_type: &DataType) -> u64 {
    match data_type {
        DataType::Null => 0,
        // Bit-packed values: under a byte a row, rounded up.
        DataType::Boolean => 1,
        DataType::Utf8 | DataType::Binary | DataType::List(_) | DataType::Map(..) => {
            size_of::<i32>() as u64
        }
        DataType::LargeUtf8 | DataType::LargeBinary | DataType::LargeList(_) => {
            size_of::<i64>() as u64
        }
        DataType::Utf8View | DataType::BinaryView => size_of::<u128>() as u64,
        DataType::FixedSizeBinary(width) => u64::try_from(*width).unwrap_or(0),
        DataType::FixedSizeList(child, len) => {
            u64::try_from(*len).unwrap_or(0) * null_slot_bytes(child.data_type())
        }
        DataType::Struct(children) => children
            .iter()
            .map(|child| null_slot_bytes(child.data_type()))
            .sum(),
        DataType::Dictionary(key, _) => null_slot_bytes(key),
        other => other
            .primitive_width()
            .map_or(size_of::<i64>() as u64, |w| w as u64),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow::{array::ArrayData, compute::concat_batches};
    use arrow_array::{Int64Array, LargeStringArray, NullArray};
    use arrow_schema::Field;

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
    fn a_column_without_a_validity_bitmap_is_kept() {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Null, true)]));
        let column: ArrayRef = Arc::new(NullArray::new(ROWS));
        let batch = RecordBatch::try_new(schema, vec![Arc::clone(&column)]).expect("batch");
        let kept = share_null_columns(batch).expect("share");
        assert!(Arc::ptr_eq(kept.column(0), &column));
    }

    #[test]
    fn a_batch_with_no_null_column_passes_through() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let column: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::clone(&column)]).expect("batch");
        let kept = share_null_columns(batch).expect("share");
        assert!(Arc::ptr_eq(kept.column(0), &column));
    }

    #[test]
    fn concat_sharing_nulls_matches_concat_batches() {
        let batch = wide_batch();
        let parts = vec![
            batch.slice(0, 100),
            batch.slice(100, 400),
            batch.slice(500, 0),
        ];
        let expected = concat_batches(&batch.schema(), &parts).expect("concat");
        let joined = concat_sharing_nulls(&batch.schema(), &parts).expect("concat sharing");
        assert_eq!(joined, expected);
        assert!(Arc::ptr_eq(joined.column(1), joined.column(3)));
        assert!(distinct_buffer_bytes(&joined) * 100 < distinct_buffer_bytes(&expected));
    }

    #[test]
    fn null_row_bytes_is_what_a_null_row_decodes_to() {
        let item = Arc::new(Field::new("item", DataType::Utf8, true));
        let schema = Schema::new(vec![
            Field::new("s", DataType::LargeUtf8, true),
            Field::new("n", DataType::Int64, true),
            Field::new("l", DataType::List(item), true),
            Field::new("u", DataType::Utf8, true),
            Field::new("b", DataType::Boolean, true),
        ]);
        let per_row = null_row_bytes(&schema);
        assert_eq!(per_row, 8 + 8 + 4 + 4 + 1);
        // Every column its own null array, as a reader decodes them.
        let columns: Vec<ArrayRef> = schema
            .fields()
            .iter()
            .map(|f| new_null_array(f.data_type(), ROWS))
            .collect();
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), columns).expect("batch");
        let decoded = distinct_buffer_bytes(&batch) as u64;
        let rows = ROWS as u64;
        // A floor, within the validity bitmaps and padding of the real cost.
        assert!(
            decoded >= rows * (per_row - 1),
            "{decoded} < {rows} x {per_row}"
        );
        assert!(
            decoded <= rows * (per_row + 2),
            "{decoded} > {rows} x {per_row}"
        );
    }

    #[test]
    fn concat_sharing_nulls_of_nothing_is_empty() {
        let batch = wide_batch();
        let joined = concat_sharing_nulls(&batch.schema(), &[]).expect("concat");
        assert_eq!(joined.num_rows(), 0);
        assert_eq!(joined.num_columns(), batch.num_columns());
    }
}
