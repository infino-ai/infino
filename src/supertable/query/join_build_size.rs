// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`RightSizeJoinBuild`]: a physical optimizer rule that copies a hash
//! join's build-side columns into buffers of their own size when the buffers
//! they point into are much larger than their rows.
//!
//! A hash join reserves memory for each build batch it collects by counting
//! the full capacity of every buffer the batch points into. A buffer is
//! counted once within a batch but again for every batch, so a buffer that
//! several build batches share is charged once per batch. Two common inputs
//! hand the join batches whose buffers are far larger than their rows:
//!
//! - A grouped aggregate emits its whole result as one large batch and passes
//!   it on in zero-copy slices of a few thousand rows, so each slice is
//!   charged for the entire buffer behind it:
//!
//! ```text
//!   aggregate result: 6.25M int64 keys (50 MB) in one buffer of 64 MiB capacity
//!   ┌──────────────────────────────────────────────────────────────┐
//!   │ slice 0 │ slice 1 │ slice 2 │ ...                │ slice 762  │  8,192 rows each
//!   └──────────────────────────────────────────────────────────────┘
//!   hash join build: asks for 64 MiB for slice 0, 64 MiB for slice 1, ...
//!                    763 x 64 MiB ≈ 48 GiB asked for 50 MB of keys
//! ```
//!
//!   Each further column adds its own full buffer to every slice's charge.
//! - A hash repartition or a filter assembles its output with arrow's batch
//!   coalescer. When an input's string buffers hold more than twice the bytes
//!   its rows use, as a `take` or filter result's do, the coalescer copies the
//!   strings into blocks that double up to 1 MiB and stay that size for later
//!   batches. An output batch of 8,192 names longer than 12 bytes (shorter
//!   strings live inline in the view) can own a 1 MiB block holding 150 KB of
//!   them, so every batch is charged several times what its strings take.
//!
//! Either way a build that easily fits fails with `ResourcesExhausted` under a
//! realistic budget.
//!
//! The rule puts a [`RightSizeBuildExec`] directly under every hash join's
//! build side. Each column of each batch is judged on its own: one charged
//! more than [`OVERCOUNT_FACTOR`] times the bytes its rows use is copied into
//! fresh buffers, and every other column passes through as the same array.
//! A struct's fields and a list's values are judged the same way, so a
//! nested column is right-sized like a top-level one. The copy has the same
//! values, nulls and order, so the join's result cannot change; it costs one
//! copy of the over-counted columns, and the larger buffers are freed once no
//! batch points into them. Until then a slice's source is charged to nobody
//! (an aggregate shrinks its reservation once it has emitted), so while the
//! slices are copied the budget sees less than the process holds.
//!
//! Nested-loop and cross joins also charge each build batch for the full
//! capacity of its buffers, are not wrapped, and are not covered by
//! DataFusion 55's fix below; an input like these feeding one of them can
//! still overrun the budget.
//!
//! TODO: revisit on the move to DataFusion 55. Its hash join counts a buffer
//! shared by several build batches once
//! (<https://github.com/apache/datafusion/pull/22862>), so aggregate slices
//! no longer need the copy; a batch that owns a part-filled buffer, like a
//! coalesced string batch, is still charged for all of it, so the rule then
//! has to earn its copy on that case alone.

use std::{fmt, sync::Arc};

use arrow::{
    array::{Capacities, MutableArrayData, make_array},
    buffer::{BooleanBuffer, NullBuffer, OffsetBuffer},
};
use arrow_array::{
    Array, ArrayRef, DictionaryArray, FixedSizeListArray, GenericListArray, OffsetSizeTrait,
    RecordBatch, StructArray, cast::AsArray, downcast_dictionary_array,
    types::ArrowDictionaryKeyType,
};
use arrow_schema::{DataType, FieldRef};
use datafusion::{
    common::{
        Statistics,
        config::ConfigOptions,
        tree_node::{Transformed, TransformedResult, TreeNode},
    },
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        execution_plan::CardinalityEffect,
        joins::HashJoinExec,
        metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::StreamExt;

use crate::runtime_metrics::ingest::{offset_window, visible_array_bytes};

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "RightSizeJoinBuild";

/// Metric counting the batches that had at least one column copied, as
/// `EXPLAIN ANALYZE` lists it.
pub(crate) const RIGHT_SIZED_BATCHES: &str = "right_sized_batches";

/// Metric summing, over copied columns, the charge before the copy minus
/// after. An accounting figure, not memory freed: slices of one buffer each
/// count it in full, so the sum can far exceed what the process ever held.
const UNCHARGED_BYTES: &str = "uncharged_bytes";

/// A column is copied when the buffers it points into hold more than this
/// many times the bytes of its own rows. Arrow grows buffers by doubling, so
/// a column that owns its buffers carries at most 2x slack and passes. An
/// aggregate's slice sits at result rows / slice rows or more (763 in the
/// module example), and a coalesced column of short strings above 2. A batch
/// of a few rows can pass 2 from 64-byte allocation rounding alone and is
/// copied for nothing, at negligible cost.
const OVERCOUNT_FACTOR: u64 = 2;

/// Puts a [`RightSizeBuildExec`] under every hash join's build side; see the
/// module docs.
#[derive(Debug, Default)]
pub(crate) struct RightSizeJoinBuild;

impl PhysicalOptimizerRule for RightSizeJoinBuild {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            let Some(join) = node.downcast_ref::<HashJoinExec>() else {
                return Ok(Transformed::no(node));
            };
            let build: Arc<dyn ExecutionPlan> =
                Arc::new(RightSizeBuildExec::new(Arc::clone(join.left())));
            let probe = Arc::clone(join.right());
            Ok(Transformed::yes(
                Arc::clone(&node).with_new_children(vec![build, probe])?,
            ))
        })
        .data()
    }

    fn name(&self) -> &str {
        RULE_NAME
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Passes its input through, copying each column whose buffers hold more
/// than [`OVERCOUNT_FACTOR`] times the bytes of its rows into fresh buffers.
/// It reports its input's partitioning, ordering and statistics, but like any
/// node that does not opt in it stops filter and projection pushdown, which
/// is why the rule runs after DataFusion's.
#[derive(Debug)]
struct RightSizeBuildExec {
    input: Arc<dyn ExecutionPlan>,
    metrics: ExecutionPlanMetricsSet,
}

impl RightSizeBuildExec {
    fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            input,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for RightSizeBuildExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RightSizeBuildExec")
    }
}

impl ExecutionPlan for RightSizeBuildExec {
    fn name(&self) -> &'static str {
        "RightSizeBuildExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "RightSizeBuildExec requires exactly one child; got {}",
                children.len()
            )));
        }
        Ok(Arc::new(Self::new(children.swap_remove(0))))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DfResult<Arc<Statistics>> {
        self.input.partition_statistics(partition)
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let inner = self.input.execute(partition, context)?;
        let sizer = BuildSizer::new(&self.metrics, partition);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            inner.schema(),
            inner.map(move |batch| sizer.right_size(batch?)),
        )))
    }
}

/// One partition's counters, and the copy they count.
struct BuildSizer {
    right_sized: Count,
    uncharged: Count,
}

impl BuildSizer {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            right_sized: MetricBuilder::new(metrics).counter(RIGHT_SIZED_BATCHES, partition),
            uncharged: MetricBuilder::new(metrics).counter(UNCHARGED_BYTES, partition),
        }
    }

    /// `batch` with each column passed through [`right_size_column`];
    /// `batch` itself when no column is copied. A batch with no rows passes
    /// through: it holds nothing to charge, and its columns' rounding
    /// capacity would read as over-counted.
    fn right_size(&self, batch: RecordBatch) -> DfResult<RecordBatch> {
        if batch.num_rows() == 0 {
            return Ok(batch);
        }
        let mut columns: Option<Vec<ArrayRef>> = None;
        for (i, column) in batch.columns().iter().enumerate() {
            let copy = right_size_column(column)?;
            if Arc::ptr_eq(&copy, column) {
                if let Some(columns) = columns.as_mut() {
                    columns.push(copy);
                }
                continue;
            }
            self.uncharged.add(
                column
                    .get_array_memory_size()
                    .saturating_sub(copy.get_array_memory_size()),
            );
            columns
                .get_or_insert_with(|| batch.columns()[..i].to_vec())
                .push(copy);
        }
        let Some(columns) = columns else {
            return Ok(batch);
        };
        self.right_sized.add(1);
        Ok(RecordBatch::try_new(batch.schema(), columns)?)
    }
}

/// `column` copied into buffers of its own when it [`is_overcounted`] and the
/// copy is charged less; `column` itself otherwise. The second test keeps a
/// column the size check misjudges, like a `NULL` literal (no bytes to
/// compare against) or a struct of strings it owns (the check counts a
/// struct's strings at 16 bytes a row), from being swapped for a copy that
/// saves nothing.
fn right_size_column(column: &ArrayRef) -> DfResult<ArrayRef> {
    if !is_overcounted(column.as_ref()) {
        return Ok(Arc::clone(column));
    }
    let copy = copy_column(column)?;
    if copy.get_array_memory_size() < column.get_array_memory_size() {
        Ok(copy)
    } else {
        Ok(Arc::clone(column))
    }
}

/// Whether a hash join would charge `column` more than [`OVERCOUNT_FACTOR`]
/// times the bytes its rows use.
fn is_overcounted(column: &dyn Array) -> bool {
    column.get_array_memory_size() as u64 > visible_array_bytes(column) * OVERCOUNT_FACTOR
}

/// `column` with the same values, nulls and order in buffers sized to its
/// rows. A struct is rebuilt from its fields, each through
/// [`right_size_column`], and a list from offsets starting at zero and only
/// the child values its rows address, so nested columns are right-sized the
/// same way as top-level ones. A dictionary gets keys of its own and keeps
/// its values, which every slice legitimately holds. Everything else is
/// copied by arrow's `MutableArrayData`, and a view column is then
/// compacted to only the out-of-line bytes its rows reference. Map, union,
/// run-end encoded and list-view columns come back as they are: their slices
/// are measured with their children in full and never read as over-counted.
fn copy_column(column: &ArrayRef) -> DfResult<ArrayRef> {
    Ok(match column.data_type() {
        DataType::Struct(fields) => {
            let record = column.as_struct();
            let fields_copied = record
                .columns()
                .iter()
                .map(right_size_column)
                .collect::<DfResult<Vec<_>>>()?;
            Arc::new(StructArray::try_new(
                fields.clone(),
                fields_copied,
                copy_nulls(record.nulls()),
            )?)
        }
        DataType::List(field) => copy_list(column.as_list::<i32>(), field)?,
        DataType::LargeList(field) => copy_list(column.as_list::<i64>(), field)?,
        DataType::FixedSizeList(field, size) => {
            // A sliced fixed-size list's values are already its own window.
            let list = column.as_fixed_size_list();
            Arc::new(FixedSizeListArray::try_new(
                Arc::clone(field),
                *size,
                right_size_column(list.values())?,
                copy_nulls(list.nulls()),
            )?)
        }
        DataType::Dictionary(..) => {
            let dictionary = column.as_ref();
            downcast_dictionary_array! {
                dictionary => copy_dictionary(dictionary)?,
                other => {
                    return Err(DataFusionError::Internal(format!(
                        "RightSizeBuildExec cannot copy a dictionary of {other}"
                    )));
                }
            }
        }
        DataType::Map(..)
        | DataType::Union(..)
        | DataType::RunEndEncoded(..)
        | DataType::ListView(_)
        | DataType::LargeListView(_) => Arc::clone(column),
        _ => {
            let data = column.to_data();
            let mut copy =
                MutableArrayData::with_capacities(vec![&data], false, flat_capacities(column));
            copy.extend(0, 0, data.len());
            let copy = make_array(copy.freeze());
            match copy.data_type() {
                DataType::Utf8View => Arc::new(copy.as_string_view().gc()),
                DataType::BinaryView => Arc::new(copy.as_binary_view().gc()),
                _ => copy,
            }
        }
    })
}

/// `list` with offsets of its own starting at zero, a null bitmap of its own,
/// and only the child values its rows address, right-sized.
fn copy_list<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    field: &FieldRef,
) -> DfResult<ArrayRef> {
    let (start, end) = offset_window(list.value_offsets());
    let offsets: Vec<O> = list
        .value_offsets()
        .iter()
        .map(|offset| O::usize_as(offset.as_usize() - start))
        .collect();
    let values = right_size_column(&list.values().slice(start, end - start))?;
    Ok(Arc::new(GenericListArray::try_new(
        Arc::clone(field),
        OffsetBuffer::new(offsets.into()),
        values,
        copy_nulls(list.nulls()),
    )?))
}

/// `dictionary` with keys of its own and the same values. Arrow's
/// `MutableArrayData` would reject a dictionary whose values fill its key
/// type (128 values for an `Int8` key), so only the keys go through it.
fn copy_dictionary<K: ArrowDictionaryKeyType>(
    dictionary: &DictionaryArray<K>,
) -> DfResult<ArrayRef> {
    let keys: ArrayRef = Arc::new(dictionary.keys().clone());
    let keys = copy_column(&keys)?.as_primitive::<K>().clone();
    Ok(Arc::new(DictionaryArray::try_new(
        keys,
        Arc::clone(dictionary.values()),
    )?))
}

/// A null bitmap of its own for `nulls`: a slice's bitmap points into its
/// source's, which is charged in full.
fn copy_nulls(nulls: Option<&NullBuffer>) -> Option<NullBuffer> {
    nulls.map(|nulls| {
        NullBuffer::new(BooleanBuffer::collect_bool(nulls.len(), |row| {
            nulls.is_valid(row)
        }))
    })
}

/// Capacities that fit a flat `column`'s rows exactly: for offset strings
/// and binaries, the value bytes its window addresses rather than arrow's
/// default of one byte a row, which a copy of longer values outgrows by
/// doubling.
fn flat_capacities(column: &dyn Array) -> Capacities {
    let len = column.len();
    let bytes = |offsets: (usize, usize)| Capacities::Binary(len, Some(offsets.1 - offsets.0));
    match column.data_type() {
        DataType::Utf8 => bytes(offset_window(column.as_string::<i32>().value_offsets())),
        DataType::LargeUtf8 => bytes(offset_window(column.as_string::<i64>().value_offsets())),
        DataType::Binary => bytes(offset_window(column.as_binary::<i32>().value_offsets())),
        DataType::LargeBinary => bytes(offset_window(column.as_binary::<i64>().value_offsets())),
        _ => Capacities::Array(len),
    }
}

#[cfg(test)]
mod tests {
    use arrow::{
        buffer::{NullBuffer, OffsetBuffer},
        compute::{BatchCoalescer, take_record_batch},
    };
    use arrow_array::{
        BinaryViewArray, BooleanArray, DictionaryArray, FixedSizeListArray, Float32Array,
        Int64Array, LargeStringArray, ListArray, NullArray, StringArray, StringViewArray,
        StructArray, UInt32Array,
        types::{Int8Type, Int32Type, Int64Type},
    };
    use arrow_schema::{Field, Schema};
    use datafusion::common::utils::memory::get_record_batch_memory_size;

    use super::*;

    /// Rows in the "aggregate result" the tests slice.
    const BIG_ROWS: usize = 1 << 16;
    /// Rows in a result large enough that a slice still pointing at its null
    /// bitmap or views would read as over-counted.
    const HUGE_ROWS: usize = 1 << 20;
    /// Rows in one slice of it: DataFusion's default batch size.
    const SLICE_ROWS: usize = 8192;
    /// Where the slice starts: not at offset 0 and not a multiple of 8, so
    /// offsets and bit offsets into boolean and null bitmaps are exercised.
    const SLICE_OFFSET: usize = 1001;
    /// Every this-many values is null, so null buffers are exercised.
    const NULL_EVERY: usize = 7;
    /// Distinct values in the dictionary column.
    const DICTIONARY_VALUES: usize = 16;
    /// The copy must be charged at least this many times less than the slice.
    const MIN_SHRINK: usize = 4;
    /// Output partitions of the simulated hash repartition: each takes one row
    /// in this many.
    const REPARTITION_FANOUT: usize = 32;
    /// Coalesced batches to build before checking one, so the coalescer's
    /// string blocks have grown to their full size.
    const COALESCED_BATCHES: usize = 8;
    /// Values a `Dictionary(Int8, _)` can index: every key from 0 to 127.
    const INT8_KEYS: usize = 128;
    /// Values in each list row of the list test.
    const LIST_WIDTH: usize = 4;
    /// In the sparse list test, one row in this many holds a value.
    const SPARSE_EVERY: usize = 4;
    /// Floats in each vector of the sparse list test.
    const VECTOR_WIDTH: i32 = 8;
    /// Distinct short keys in the inline-strings test: each at most 4 bytes,
    /// so it lives inline in its view.
    const SHORT_KEYS: usize = 1000;

    fn valid(i: usize) -> bool {
        !i.is_multiple_of(NULL_EVERY)
    }

    /// A string long enough to live out of line in a view.
    fn long_text(i: usize) -> Option<String> {
        valid(i).then(|| format!("a string long enough {i}"))
    }

    fn sizer() -> BuildSizer {
        BuildSizer::new(&ExecutionPlanMetricsSet::new(), 0)
    }

    /// One column of each common layout family a build side carries:
    /// primitive, string and binary views, offset strings (32- and 64-bit
    /// offsets), bit-packed boolean, dictionary, list and struct. Nullable
    /// where the type allows, strings long enough to live out of line.
    fn wide_batch(rows: usize) -> RecordBatch {
        let dictionary_text: Vec<Option<String>> = (0..rows)
            .map(|i| long_text(i % DICTIONARY_VALUES))
            .collect();
        let struct_fields = vec![
            Arc::new(Field::new("n", DataType::Int64, false)),
            Arc::new(Field::new("s", DataType::Utf8, true)),
        ];
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("view", DataType::Utf8View, true),
            Field::new("bytes", DataType::BinaryView, true),
            Field::new("utf8", DataType::Utf8, true),
            Field::new("large", DataType::LargeUtf8, true),
            Field::new("flag", DataType::Boolean, true),
            Field::new(
                "dict",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
            Field::new_list("list", Field::new_list_field(DataType::Int64, true), true),
            Field::new_struct("record", struct_fields.clone(), false),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter(
                (0..rows).map(|i| valid(i).then_some(i as i64)),
            )),
            Arc::new(StringViewArray::from_iter((0..rows).map(long_text))),
            Arc::new(BinaryViewArray::from_iter(
                (0..rows).map(|i| long_text(i).map(String::into_bytes)),
            )),
            Arc::new(StringArray::from_iter((0..rows).map(long_text))),
            Arc::new(LargeStringArray::from_iter((0..rows).map(long_text))),
            Arc::new(BooleanArray::from_iter(
                (0..rows).map(|i| valid(i).then_some(i % 2 == 0)),
            )),
            Arc::new(
                dictionary_text
                    .iter()
                    .map(Option::as_deref)
                    .collect::<DictionaryArray<Int32Type>>(),
            ),
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
                (0..rows).map(|i| valid(i).then(|| vec![Some(i as i64), None])),
            )),
            Arc::new(
                StructArray::try_new(
                    struct_fields.into(),
                    vec![
                        Arc::new(Int64Array::from_iter_values((0..rows).map(|i| i as i64))),
                        Arc::new(StringArray::from_iter((0..rows).map(long_text))),
                    ],
                    None,
                )
                .expect("struct"),
            ),
        ];
        RecordBatch::try_new(schema, columns).expect("batch")
    }

    /// Every column of a small slice of a large batch is over-counted; its
    /// right-sized copy holds the same rows, no column of it is over-counted,
    /// and it is charged a fraction of the slice. A dictionary's values stay
    /// shared with the large batch, which every slice legitimately holds.
    #[test]
    fn a_small_slice_of_a_large_batch_is_copied() {
        let big = wide_batch(BIG_ROWS);
        let slice = big.slice(SLICE_OFFSET, SLICE_ROWS);
        let copied = sizer().right_size(slice.clone()).expect("right_size");
        assert_eq!(copied, slice, "same rows, nulls and strings, same order");
        for (field, column) in copied.schema().fields().iter().zip(copied.columns()) {
            assert!(
                !is_overcounted(column.as_ref()),
                "{}: {} B charged for {} B of rows",
                field.name(),
                column.get_array_memory_size(),
                visible_array_bytes(column.as_ref())
            );
        }
        let before = get_record_batch_memory_size(&slice);
        let after = get_record_batch_memory_size(&copied);
        assert!(
            after * MIN_SHRINK < before,
            "the copy is charged {after} B, the slice {before} B"
        );
    }

    /// A view column's copy gets views and a null bitmap of its own, not the
    /// large result's: short strings, which live inline in the views, and
    /// nullable long strings alike.
    #[test]
    fn a_slice_of_short_or_nullable_strings_gets_views_and_nulls_of_its_own() {
        let short = Arc::new(StringViewArray::from_iter(
            (0..HUGE_ROWS).map(|i| valid(i).then(|| format!("k{}", i % SHORT_KEYS))),
        )) as ArrayRef;
        let long = Arc::new(StringViewArray::from_iter((0..HUGE_ROWS).map(long_text))) as ArrayRef;
        for (name, column) in [("short", short), ("long", long)] {
            let slice = column.slice(SLICE_OFFSET, SLICE_ROWS);
            assert!(
                is_overcounted(slice.as_ref()),
                "{name}: the slice is over-counted"
            );
            let copy = copy_column(&slice).expect("copy");
            assert_eq!(&copy, &slice, "{name}: same strings and nulls");
            assert!(
                !is_overcounted(copy.as_ref()),
                "{name}: the copy is charged for its rows"
            );
            let nulls = copy.nulls().expect("nulls");
            assert_eq!(
                nulls.buffer().len(),
                SLICE_ROWS.div_ceil(8),
                "{name}: the copy's null bitmap is its own"
            );
        }
    }

    /// A slice of a list column is over-counted by its share of the large
    /// result's child values, and its copy keeps only its own child window.
    #[test]
    fn a_slice_of_a_list_column_is_copied() {
        let list = Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
            (0..BIG_ROWS).map(|i| Some((0..LIST_WIDTH).map(move |j| Some((i + j) as i64)))),
        )) as ArrayRef;
        let slice = list.slice(SLICE_OFFSET, SLICE_ROWS);
        assert!(is_overcounted(slice.as_ref()), "the slice is over-counted");
        let copy = copy_column(&slice).expect("copy");
        assert_eq!(&copy, &slice, "same lists");
        assert!(
            !is_overcounted(copy.as_ref()),
            "the copy is charged for its rows"
        );
    }

    /// A dictionary whose values fill its key type (all 128 an `Int8` key can
    /// index) is copied, keys and all, without overflowing the key type.
    #[test]
    fn a_dictionary_that_fills_its_key_type_is_copied() {
        let values: Vec<String> = (0..BIG_ROWS)
            .map(|i| format!("value {}", i % INT8_KEYS))
            .collect();
        let dictionary = Arc::new(
            values
                .iter()
                .map(String::as_str)
                .collect::<DictionaryArray<Int8Type>>(),
        ) as ArrayRef;
        assert_eq!(dictionary.as_any_dictionary().values().len(), INT8_KEYS);
        let slice = dictionary.slice(SLICE_OFFSET, SLICE_ROWS);
        assert!(
            is_overcounted(slice.as_ref()),
            "the slice's keys are over-counted"
        );
        let copy = copy_column(&slice).expect("copy");
        assert_eq!(&copy, &slice, "same keys and values");
        assert!(
            !is_overcounted(copy.as_ref()),
            "the copy is charged for its rows"
        );
    }

    /// A slice of a sparse list column is copied with only the child values
    /// its rows address, not one per row: a list with a value in one row of
    /// four, an all-null list, and a sparse list of fixed-size lists.
    #[test]
    fn a_slice_of_a_sparse_list_column_is_copied() {
        let sparse = Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
            (0..BIG_ROWS).map(|i| Some((i % SPARSE_EVERY == 0).then_some(Some(i as i64)))),
        )) as ArrayRef;
        let all_null = Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
            (0..BIG_ROWS).map(|_| None::<Vec<Option<i64>>>),
        )) as ArrayRef;
        let vectors = FixedSizeListArray::try_new(
            Arc::new(Field::new_list_field(DataType::Float32, false)),
            VECTOR_WIDTH,
            Arc::new(Float32Array::from_iter_values(
                (0..BIG_ROWS / SPARSE_EVERY * VECTOR_WIDTH as usize).map(|i| i as f32),
            )),
            None,
        )
        .expect("vectors");
        let sparse_vectors = Arc::new(
            ListArray::try_new(
                Arc::new(Field::new_list_field(vectors.data_type().clone(), false)),
                OffsetBuffer::from_lengths(
                    (0..BIG_ROWS).map(|i| usize::from(i % SPARSE_EVERY == 0)),
                ),
                Arc::new(vectors),
                None,
            )
            .expect("list of vectors"),
        ) as ArrayRef;
        for (name, column) in [
            ("sparse", sparse),
            ("all null", all_null),
            ("sparse vectors", sparse_vectors),
        ] {
            let slice = column.slice(SLICE_OFFSET, SLICE_ROWS);
            assert!(
                is_overcounted(slice.as_ref()),
                "{name}: the slice is over-counted"
            );
            let copy = copy_column(&slice).expect("copy");
            assert_eq!(&copy, &slice, "{name}: same lists");
            assert!(
                !is_overcounted(copy.as_ref()),
                "{name}: {} B charged for {} B of rows",
                copy.get_array_memory_size(),
                visible_array_bytes(copy.as_ref())
            );
        }
    }

    /// A slice of nested columns is right-sized field by field: a struct of a
    /// dictionary whose values fill its key type, long string views and
    /// integers, and a list of long string views. The copy holds the same
    /// rows, no field or list of it is over-counted, and it is charged a
    /// fraction of the slice.
    #[test]
    fn a_slice_of_nested_dictionaries_and_views_is_copied() {
        let values: Vec<String> = (0..BIG_ROWS)
            .map(|i| format!("value {}", i % INT8_KEYS))
            .collect();
        let codes: ArrayRef = Arc::new(
            values
                .iter()
                .map(String::as_str)
                .collect::<DictionaryArray<Int8Type>>(),
        );
        let names: ArrayRef = Arc::new(StringViewArray::from_iter((0..BIG_ROWS).map(long_text)));
        let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(0..BIG_ROWS as i64));
        let record_fields = vec![
            Arc::new(Field::new("code", codes.data_type().clone(), false)),
            Arc::new(Field::new("name", DataType::Utf8View, true)),
            Arc::new(Field::new("id", DataType::Int64, false)),
        ];
        let record: ArrayRef = Arc::new(
            StructArray::try_new(
                record_fields.into(),
                vec![codes, Arc::clone(&names), ids],
                Some(NullBuffer::from_iter((0..BIG_ROWS).map(valid))),
            )
            .expect("struct"),
        );
        let name_lists: ArrayRef = Arc::new(
            ListArray::try_new(
                Arc::new(Field::new_list_field(DataType::Utf8View, true)),
                OffsetBuffer::from_lengths(std::iter::repeat_n(1, BIG_ROWS)),
                names,
                None,
            )
            .expect("list"),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("record", record.data_type().clone(), true),
            Field::new("names", name_lists.data_type().clone(), false),
        ]));
        let big = RecordBatch::try_new(schema, vec![record, name_lists]).expect("batch");
        let slice = big.slice(SLICE_OFFSET, SLICE_ROWS);

        let copied = sizer().right_size(slice.clone()).expect("right_size");
        assert_eq!(copied, slice, "same rows, keys, strings and nulls");
        let record = copied.column(0).as_struct();
        assert_eq!(
            record.nulls().expect("nulls").buffer().len(),
            SLICE_ROWS.div_ceil(8),
            "the struct's null bitmap is its own"
        );
        for (field, column) in record.fields().iter().zip(record.columns()) {
            assert!(
                !is_overcounted(column.as_ref()),
                "{}: {} B charged for {} B of rows",
                field.name(),
                column.get_array_memory_size(),
                visible_array_bytes(column.as_ref())
            );
        }
        assert!(
            !is_overcounted(copied.column(1).as_ref()),
            "the list of names is charged for its rows"
        );
        let before = get_record_batch_memory_size(&slice);
        let after = get_record_batch_memory_size(&copied);
        assert!(
            after * MIN_SHRINK < before,
            "the copy is charged {after} B, the slice {before} B"
        );
    }

    /// A struct of long strings it owns reads as over-counted, since a
    /// struct's strings are measured at 16 bytes a row, but copying it would
    /// save nothing: it passes through as the same array.
    #[test]
    fn a_struct_that_owns_its_strings_passes_through() {
        let names: ArrayRef = Arc::new(StringViewArray::from_iter(
            (0..SLICE_ROWS).map(|i| Some(format!("a string much longer than a view {i}"))),
        ));
        let field = Arc::new(Field::new("name", DataType::Utf8View, true));
        let record: ArrayRef =
            Arc::new(StructArray::try_new(vec![field].into(), vec![names], None).expect("struct"));
        assert!(
            is_overcounted(record.as_ref()),
            "the size check misjudges the struct"
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "record",
            record.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::clone(&record)]).expect("batch");
        let sizer = sizer();
        let out = sizer.right_size(batch).expect("right_size");
        assert!(Arc::ptr_eq(out.column(0), &record));
        assert_eq!(sizer.right_sized.value(), 0);
    }

    /// A `NULL` literal column uses no bytes and is never copied.
    #[test]
    fn a_null_column_passes_through() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Null, true)]));
        let nulls: ArrayRef = Arc::new(NullArray::new(BIG_ROWS));
        let sizer = sizer();
        for column in [nulls.slice(SLICE_OFFSET, SLICE_ROWS), nulls] {
            let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column]).expect("batch");
            let out = sizer.right_size(batch.clone()).expect("right_size");
            assert_eq!(out, batch);
        }
        assert_eq!(sizer.right_sized.value(), 0);
    }

    /// A batch arrow's coalescer assembles from sparse string views, as a hash
    /// repartition does for each output partition, owns string blocks it fills
    /// only in part: it is over-counted, and its copy is not.
    #[test]
    fn a_coalesced_batch_of_part_filled_string_blocks_is_copied() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Utf8View,
            false,
        )]));
        let source = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringViewArray::from_iter_values(
                (0..BIG_ROWS).map(|i| format!("customer name {i:09}")),
            ))],
        )
        .expect("batch");
        // One output partition's share of every input batch.
        let partition_rows =
            UInt32Array::from_iter_values((0..BIG_ROWS as u32).step_by(REPARTITION_FANOUT));
        let mut coalescer = BatchCoalescer::new(Arc::clone(&schema), SLICE_ROWS);
        let mut coalesced = Vec::new();
        while coalesced.len() < COALESCED_BATCHES {
            let share = take_record_batch(&source, &partition_rows).expect("take");
            coalescer.push_batch(share).expect("push");
            while let Some(batch) = coalescer.next_completed_batch() {
                coalesced.push(batch);
            }
        }
        let last = coalesced.last().expect("a coalesced batch");
        assert!(
            is_overcounted(last.column(0).as_ref()),
            "the coalesced batch is over-counted"
        );

        let copied = sizer().right_size(last.clone()).expect("right_size");
        assert_eq!(&copied, last, "same strings, same order");
        assert!(
            !is_overcounted(copied.column(0).as_ref()),
            "the copy is charged for its rows"
        );
    }

    /// Only the over-counted columns of a batch are copied; a column that owns
    /// its buffers comes back as the same array.
    #[test]
    fn only_the_over_counted_columns_are_copied() {
        let owned: ArrayRef = Arc::new(Int64Array::from_iter_values(0..SLICE_ROWS as i64));
        let big: ArrayRef = Arc::new(Int64Array::from_iter_values(0..BIG_ROWS as i64));
        let sliced = big.slice(SLICE_OFFSET, SLICE_ROWS);
        let schema = Arc::new(Schema::new(vec![
            Field::new("owned", DataType::Int64, false),
            Field::new("sliced", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![Arc::clone(&owned), sliced]).expect("batch");
        let out = sizer().right_size(batch.clone()).expect("right_size");
        assert_eq!(out, batch);
        assert!(
            Arc::ptr_eq(out.column(0), &owned),
            "the owned column is not copied"
        );
        assert!(
            !is_overcounted(out.column(1).as_ref()),
            "the slice is copied"
        );
    }

    /// A batch whose columns own their buffers, and a batch with no rows, pass
    /// through as they came, without counting a copy.
    #[test]
    fn a_batch_that_owns_its_buffers_passes_through() {
        let metrics = ExecutionPlanMetricsSet::new();
        let sizer = BuildSizer::new(&metrics, 0);
        for batch in [wide_batch(SLICE_ROWS), wide_batch(0)] {
            let out = sizer.right_size(batch.clone()).expect("right_size");
            assert_eq!(out, batch);
        }
        assert_eq!(sizer.right_sized.value(), 0);
        assert_eq!(sizer.uncharged.value(), 0);
    }
}
