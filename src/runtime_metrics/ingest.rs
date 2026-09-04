// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Classify ingested [`RecordBatch`] bytes into text/scalar vs vector columns.
//!
//! Used by the platform worker when attributing write meters for Stripe.
//! Vector columns are Arrow [`DataType::FixedSizeList`] (the engine's vector
//! column shape); every other column counts as text/scalar.

use std::mem;

use arrow_array::{
    Array, BinaryViewArray, GenericListArray, OffsetSizeTrait, RecordBatch, StringViewArray,
};
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
///
/// This is the right measure wherever bytes are *priced*. It is the wrong
/// one for held-memory accounting (an auto-flush threshold, a scratch
/// reserve), where the parent allocation really is what is resident.
///
/// Two families need handling of their own, because Arrow's slice-aware
/// sizing is not uniformly slice-aware:
///
/// - **View types.** `get_slice_memory_size` walks only the buffers a
///   type's layout declares, and the view layout declares one 16-byte
///   view buffer with the payload buffers marked variadic — those are
///   never visited. A `Utf8View` column would price at 16 bytes a row
///   whatever it carries, roughly 13x under a `LargeUtf8` column of the
///   same strings. The capacity-based size is the closer answer, so these
///   use it. This matters here and not only in theory: the SQL provider
///   retypes non-FTS string columns to `Utf8View` for the scan.
/// - **Offset-addressed children.** `ArrayData::slice` recurses into
///   children only for `Struct`, so a sliced `List` bills its whole shared
///   values array — the over-count this function exists to avoid. Lists
///   size the child window their offsets actually name.
///
/// `Dictionary` keeps the over-count for a slice (every chunk counts the
/// whole dictionary) and is left alone deliberately: its values are
/// content-shared — arbitrary keys reference arbitrary value rows, so no
/// per-chunk window exists and amortizing needs a policy rather than a
/// measurement. `Map` also keeps the over-count today, but for the other
/// reason: its entries child is offset-addressed exactly like a list, so
/// the same window measurement would apply — it is simply not wired, and
/// no Map ingest path exists to feed it.
pub(crate) fn visible_array_bytes(column: &dyn Array) -> u64 {
    match column.data_type() {
        DataType::Utf8View | DataType::BinaryView => view_visible_bytes(column),
        DataType::List(_) => list_window_bytes::<i32>(column),
        DataType::LargeList(_) => list_window_bytes::<i64>(column),
        _ => flat_visible_bytes(column),
    }
}

/// A view column's visible footprint: the fixed 16-byte view per row plus
/// the payload of every value too long to live inline in it.
///
/// Neither of Arrow's two sizing calls works here. `get_slice_memory_size`
/// visits only the declared view buffer and reports 16 bytes a row however
/// much text the column carries; `get_array_memory_size` counts the
/// variadic buffers' whole *capacity*, and their allocator over-allocates
/// in large blocks — a four-row column of long strings measured 16,560
/// bytes against 196 for the same strings as `LargeUtf8`. Walking the
/// values is O(rows), which is affordable on a path that is about to
/// encode the batch anyway, and it is right for a slice as well as a whole
/// array.
fn view_visible_bytes(column: &dyn Array) -> u64 {
    /// Bytes each view occupies in the fixed-width buffer.
    const VIEW_STRIDE: usize = 16;
    /// Payload up to this length is stored inside the view itself.
    const INLINE_CAPACITY: usize = 12;

    let out_of_line = |lengths: &mut dyn Iterator<Item = usize>| -> u64 {
        lengths
            .filter(|len| *len > INLINE_CAPACITY)
            .map(|len| len as u64)
            .sum()
    };
    let payload = if let Some(view) = column.as_any().downcast_ref::<StringViewArray>() {
        out_of_line(&mut view.iter().flatten().map(str::len))
    } else if let Some(view) = column.as_any().downcast_ref::<BinaryViewArray>() {
        out_of_line(&mut view.iter().flatten().map(<[u8]>::len))
    } else {
        // Not a shape we can walk; the declared-buffer size is at least a
        // floor rather than a guess.
        return flat_visible_bytes(column);
    };
    ((column.len() * VIEW_STRIDE) as u64)
        .saturating_add(payload)
        .saturating_add(null_bitmap_visible_bytes(column))
}

/// Visible bytes of a column's null bitmap: one bit per visible row,
/// rounded up — never the backing buffer's length, which a slice shares
/// with its parent the same way it shares value buffers.
fn null_bitmap_visible_bytes(column: &dyn Array) -> u64 {
    if column.nulls().is_some() {
        column.len().div_ceil(8) as u64
    } else {
        0
    }
}

/// Slice-aware size for a type whose own layout describes it fully.
fn flat_visible_bytes(column: &dyn Array) -> u64 {
    column
        .to_data()
        .get_slice_memory_size()
        .unwrap_or_else(|_| column.get_array_memory_size()) as u64
}

/// A list column's own buffers plus only the slice of its values array
/// that this window addresses.
fn list_window_bytes<O: OffsetSizeTrait>(column: &dyn Array) -> u64 {
    let Some(list) = column.as_any().downcast_ref::<GenericListArray<O>>() else {
        return flat_visible_bytes(column);
    };
    let offsets = list.value_offsets();
    // An empty window addresses no child rows; its own offset buffer still
    // costs something, which `own` below covers.
    let (start, end) = match (offsets.first(), offsets.last()) {
        (Some(first), Some(last)) => (first.as_usize(), last.as_usize()),
        _ => (0, 0),
    };
    let own = mem::size_of_val(offsets) as u64 + null_bitmap_visible_bytes(column);
    let child = if end > start {
        visible_array_bytes(list.values().slice(start, end - start).as_ref())
    } else {
        0
    };
    own.saturating_add(child)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        Array, FixedSizeListArray, Float32Array, LargeStringArray, ListArray, RecordBatch,
        StringViewArray,
        builder::{ListBuilder, StringBuilder},
    };
    use arrow_schema::{DataType, Field, Schema};

    use super::{classify_batch_bytes, visible_array_bytes};

    /// Strings long enough to live outside a view's inline 12-byte prefix,
    /// which is the only case where the view layout hides its payload.
    const OUT_OF_LINE: [&str; 4] = [
        "a string comfortably past the inline prefix",
        "another string comfortably past the prefix",
        "a third string that will not fit inline",
        "a fourth string that will not fit inline",
    ];
    /// Widest gap tolerated between the two encodings of the same strings.
    const VIEW_VS_LARGE_UTF8_TOLERANCE: u64 = 4;

    #[test]
    fn a_view_column_is_not_billed_at_its_inline_stride() {
        // Arrow's slice-aware sizing visits only the buffers a layout
        // declares, and the view layout declares one 16-byte view buffer
        // with the payload buffers variadic — so a view column priced that
        // way reports 16 bytes a row however much text it carries. The SQL
        // provider retypes non-FTS string columns to Utf8View, so this is
        // a shape the engine really sees.
        let view = StringViewArray::from(OUT_OF_LINE.to_vec());
        let large = LargeStringArray::from(OUT_OF_LINE.to_vec());
        let view_bytes = visible_array_bytes(&view);
        let large_bytes = visible_array_bytes(&large);
        let payload: u64 = OUT_OF_LINE.iter().map(|s| s.len() as u64).sum();
        assert!(
            view_bytes > payload,
            "the same strings must cost at least their payload: {view_bytes} vs {payload}"
        );
        assert!(
            view_bytes * VIEW_VS_LARGE_UTF8_TOLERANCE > large_bytes
                && large_bytes * VIEW_VS_LARGE_UTF8_TOLERANCE > view_bytes,
            "two encodings of the same strings must price within a small \
             factor: view={view_bytes} large_utf8={large_bytes}"
        );
    }

    #[test]
    fn a_sliced_list_bills_only_the_child_window_it_addresses() {
        // `ArrayData::slice` recurses into children only for structs, so a
        // sliced list would otherwise carry its whole shared values array
        // — the exact over-count this sizing exists to avoid, on the
        // nested column shape the engine ingests.
        let mut builder = ListBuilder::new(StringBuilder::new());
        for row in 0..64 {
            for item in 0..8 {
                builder
                    .values()
                    .append_value(format!("row {row} item {item}"));
            }
            builder.append(true);
        }
        let full = builder.finish();
        let window = full.slice(0, 2);
        let window = window
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("still a list");
        let full_bytes = visible_array_bytes(&full);
        let window_bytes = visible_array_bytes(window);
        assert!(
            window_bytes * 4 < full_bytes,
            "two rows of sixty-four must cost a fraction of the whole: \
             window={window_bytes} full={full_bytes}"
        );
    }

    #[test]
    fn a_slice_taken_from_the_middle_is_billed_like_a_standalone_batch() {
        // Chunked ingest slices at 0, then at len, then at 2*len — every
        // chunk after the first carries a non-zero `ArrayData.offset`. A
        // sizing that honoured length but ignored offset would pass an
        // offset-zero test and still over-bill every later chunk.
        const OFFSET: usize = 200;
        const ROWS: usize = 10;
        let titles: Vec<String> = (0..1_000)
            .map(|i| format!("row {i} with some text"))
            .collect();
        let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
        let middle = LargeStringArray::from(refs.clone()).slice(OFFSET, ROWS);
        let standalone = LargeStringArray::from(refs[OFFSET..OFFSET + ROWS].to_vec());
        assert_eq!(
            visible_array_bytes(&middle),
            visible_array_bytes(&standalone),
            "a mid-array window must price like the same rows standing alone"
        );
    }

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
