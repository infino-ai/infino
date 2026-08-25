// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Classify ingested [`RecordBatch`] bytes into text/scalar vs vector columns.
//!
//! Used by the platform worker when attributing write meters for Stripe.
//! Vector columns are Arrow [`DataType::FixedSizeList`] (the engine's vector
//! column shape); every other column counts as text/scalar.

use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType;

/// Split `batch`'s visible column footprint into `(text_bytes, vector_bytes)`.
///
/// Sizes use Arrow's slice-aware array-data memory size so sliced arrays are
/// not billed for unused backing capacity (falls back to full array memory
/// size if slice sizing fails). This attributes the ingested payload, not the
/// on-storage superfile footprint after encode.
pub fn classify_batch_bytes(batch: &RecordBatch) -> (u64, u64) {
    let mut text_bytes = 0u64;
    let mut vector_bytes = 0u64;
    for column in batch.columns() {
        let size = visible_array_bytes(column.as_ref());
        if matches!(column.data_type(), DataType::FixedSizeList(_, _)) {
            vector_bytes = vector_bytes.saturating_add(size);
        } else {
            text_bytes = text_bytes.saturating_add(size);
        }
    }
    (text_bytes, vector_bytes)
}

/// One array's *visible* footprint: slice-aware, so an array that is a
/// zero-copy window into a larger buffer is billed for the rows it
/// actually carries rather than for its parent's whole allocation.
/// Falls back to the capacity-based size only if slice sizing fails.
///
/// This is the right measure wherever bytes are *priced*. It is the wrong
/// one for held-memory accounting (an auto-flush threshold, a scratch
/// reserve), where the parent allocation really is what is resident.
pub(crate) fn visible_array_bytes(column: &dyn Array) -> u64 {
    column
        .to_data()
        .get_slice_memory_size()
        .unwrap_or_else(|_| column.get_array_memory_size()) as u64
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use super::classify_batch_bytes;

    #[test]
    fn splits_text_and_vector_columns() {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("body", DataType::LargeUtf8, false),
            Field::new("emb", DataType::FixedSizeList(Arc::clone(&item), 2), false),
        ]));
        let body = LargeStringArray::from(vec!["hello", "world"]);
        let flat = Float32Array::from(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let emb = FixedSizeListArray::new(item, 2, Arc::new(flat), None);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(body), Arc::new(emb)]).expect("batch");
        let (text, vector) = classify_batch_bytes(&batch);
        assert!(text > 0, "text columns must contribute bytes");
        assert!(vector > 0, "vector columns must contribute bytes");
    }

    #[test]
    fn sliced_column_does_not_bill_full_backing_capacity() {
        let full = LargeStringArray::from(vec!["aaaa", "bbbb", "cccc", "dddd"]);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "body",
            DataType::LargeUtf8,
            false,
        )]));
        let full_batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(full.clone())]).expect("full");
        let sliced_batch =
            RecordBatch::try_new(schema, vec![Arc::new(full.slice(1, 1))]).expect("sliced");
        let (full_text, _) = classify_batch_bytes(&full_batch);
        let (sliced_text, _) = classify_batch_bytes(&sliced_batch);
        assert!(
            sliced_text < full_text,
            "slice should bill less than full array ({sliced_text} vs {full_text})"
        );
    }
}
