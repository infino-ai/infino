// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A superfile's Parquet footer as row-group and page pruning may read it:
//! without the min/max of a float column that holds a NaN.
//!
//! A Parquet writer leaves NaN out of a float column's min/max, as the format
//! asks, so a column chunk holding `{-NaN, 1.0}` records `[1.0, 1.0]`. Arrow
//! compares floats in IEEE total order, where a negative NaN comes before
//! every number and a positive NaN after, and DataFusion sorts the same way.
//! Bounds that leave out the very rows sorting first or last then tell
//! pruning those rows are not there:
//!
//! ```text
//!   ORDER BY f ASC LIMIT 3      superfile 0: f = -1.0       → cutoff f < -1.0
//!                               superfile 2: f ∈ {-NaN, 1.0}, footer [1.0, 1.0]
//!                               pruned against the cutoff: its -NaN rows,
//!                               the true first three, are never read
//! ```
//!
//! The cutoff an `ORDER BY ... LIMIT` hands the scan moves as rows arrive, so
//! whether a row group is pruned depends on which superfiles were read first,
//! and the same statement can answer differently from run to run.
//!
//! The manifest records each superfile's float bounds in total order (Arrow's
//! min and max), so a NaN shows there as a NaN bound. A float column with one
//! keeps no bounds in the footer the scan reads, row groups and pages alike;
//! every other column, and every float column without a NaN, keeps them all.

use std::sync::Arc;

use datafusion::scalar::ScalarValue;
use parquet::{
    basic::{LogicalType, Type as PhysicalType},
    errors::Result as ParquetResult,
    file::{
        metadata::{ParquetMetaData, RowGroupMetaData},
        page_index::column_index::ColumnIndexMetaData,
    },
    schema::types::ColumnDescriptor,
};

use crate::supertable::schema::FieldId;

/// Parts in the path of a top-level column. Pruning compares only top-level
/// columns against a filter, so only their bounds can drop rows; a nested
/// leaf (a vector's elements, say) keeps its own.
const TOP_LEVEL_PATH_PARTS: usize = 1;

/// `meta`, a superfile's footer, without the min/max of every float column
/// that may hold a NaN (see the module docs), judged by `bounds`: the
/// manifest's `(min, max)` of a column in that superfile, by field id, or
/// `None` when it records none. `meta` itself, unchanged and uncopied, when
/// no column may.
pub(crate) fn without_nan_blind_bounds(
    meta: Arc<ParquetMetaData>,
    bounds: impl Fn(FieldId) -> Option<(ScalarValue, ScalarValue)>,
) -> ParquetResult<Arc<ParquetMetaData>> {
    let blind: Vec<usize> = meta
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .enumerate()
        .filter(|(_, leaf)| {
            leaf.path().parts().len() == TOP_LEVEL_PATH_PARTS
                && is_float(leaf)
                && may_hold_nan(leaf, &bounds)
        })
        .map(|(index, _)| index)
        .collect();
    if blind.is_empty() {
        return Ok(meta);
    }
    let mut builder = ParquetMetaData::clone(&meta).into_builder();
    let row_groups = builder
        .take_row_groups()
        .into_iter()
        .map(|row_group| without_chunk_bounds(row_group, &blind))
        .collect::<ParquetResult<Vec<_>>>()?;
    let column_index = builder.take_column_index().map(|mut row_groups| {
        for columns in &mut row_groups {
            for &leaf in &blind {
                if let Some(index) = columns.get_mut(leaf) {
                    *index = ColumnIndexMetaData::NONE;
                }
            }
        }
        row_groups
    });
    Ok(Arc::new(
        builder
            .set_row_groups(row_groups)
            .set_column_index(column_index)
            .build(),
    ))
}

/// `row_group` with the statistics of its `blind` column chunks cleared.
fn without_chunk_bounds(
    row_group: RowGroupMetaData,
    blind: &[usize],
) -> ParquetResult<RowGroupMetaData> {
    let columns = row_group
        .columns()
        .iter()
        .enumerate()
        .map(|(leaf, chunk)| {
            if blind.contains(&leaf) {
                chunk.clone().into_builder().clear_statistics().build()
            } else {
                Ok(chunk.clone())
            }
        })
        .collect::<ParquetResult<Vec<_>>>()?;
    row_group
        .into_builder()
        .set_column_metadata(columns)
        .build()
}

/// Whether `leaf` stores floats: Parquet's float and double, and the
/// half-precision float it stores as two fixed bytes.
fn is_float(leaf: &ColumnDescriptor) -> bool {
    match leaf.physical_type() {
        PhysicalType::FLOAT | PhysicalType::DOUBLE => true,
        PhysicalType::FIXED_LEN_BYTE_ARRAY => {
            matches!(leaf.logical_type_ref(), Some(LogicalType::Float16))
        }
        _ => false,
    }
}

/// Whether the top-level float column `leaf` may hold a NaN: unless `bounds`
/// gives it two that are numbers (or nulls, a column with no value), it may.
/// A leaf without a field id names no column the manifest bounds, so it may
/// too.
fn may_hold_nan(
    leaf: &ColumnDescriptor,
    bounds: impl Fn(FieldId) -> Option<(ScalarValue, ScalarValue)>,
) -> bool {
    let info = leaf.self_type().get_basic_info();
    if !info.has_id() {
        return true;
    }
    let Some((min, max)) = u32::try_from(info.id())
        .ok()
        .and_then(|id| bounds(FieldId(id)))
    else {
        return true;
    };
    !(is_number_or_null(&min) && is_number_or_null(&max))
}

/// Whether `bound` is a float that is not a NaN, or a null one.
fn is_number_or_null(bound: &ScalarValue) -> bool {
    match bound {
        ScalarValue::Float16(value) => value.is_none_or(|v| !v.is_nan()),
        ScalarValue::Float32(value) => value.is_none_or(|v| !v.is_nan()),
        ScalarValue::Float64(value) => value.is_none_or(|v| !v.is_nan()),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use parquet::{
        arrow::{
            ArrowWriter, PARQUET_FIELD_ID_META_KEY,
            arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions},
        },
        file::metadata::PageIndexPolicy,
    };

    use super::*;

    /// Field id of the float column in the test files.
    const FLOAT_ID: u32 = 1;
    /// Field id of the integer column in the test files.
    const INT_ID: u32 = 2;
    /// Leaf index of the float column.
    const FLOAT_LEAF: usize = 0;
    /// Leaf index of the integer column.
    const INT_LEAF: usize = 1;

    /// A one-row-group file of `floats` beside as many integers, with field
    /// ids and a page index; its footer, read back with the page index.
    fn footer(floats: Vec<f64>) -> Arc<ParquetMetaData> {
        let field = |name: &str, data_type, id: u32| {
            Field::new(name, data_type, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                id.to_string(),
            )]))
        };
        let schema = Arc::new(Schema::new(vec![
            field("f", DataType::Float64, FLOAT_ID),
            field("n", DataType::Int64, INT_ID),
        ]));
        let ints: Vec<i64> = (0..floats.len() as i64).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Float64Array::from(floats)) as ArrayRef,
                Arc::new(Int64Array::from(ints)),
            ],
        )
        .expect("batch");
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, None).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");
        let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        Arc::clone(
            ArrowReaderMetadata::load(&Bytes::from(bytes), options)
                .expect("footer")
                .metadata(),
        )
    }

    /// The manifest's bounds: `float` for the float column, numbers for the
    /// integer one.
    fn bounds(float: Option<(f64, f64)>) -> impl Fn(FieldId) -> Option<(ScalarValue, ScalarValue)> {
        move |id| match id {
            FieldId(FLOAT_ID) => float.map(|(min, max)| {
                (
                    ScalarValue::Float64(Some(min)),
                    ScalarValue::Float64(Some(max)),
                )
            }),
            FieldId(INT_ID) => Some((ScalarValue::Int64(Some(0)), ScalarValue::Int64(Some(1)))),
            _ => None,
        }
    }

    /// Whether `meta` bounds `leaf` in its row group.
    fn chunk_bounds(meta: &ParquetMetaData, leaf: usize) -> bool {
        meta.row_group(0).column(leaf).statistics().is_some()
    }

    /// Whether `meta` bounds `leaf`'s pages.
    fn page_bounds(meta: &ParquetMetaData, leaf: usize) -> bool {
        meta.column_index()
            .is_some_and(|index| !matches!(index[0][leaf], ColumnIndexMetaData::NONE))
    }

    /// A float column the manifest bounds by numbers keeps its footer
    /// bounds, and the footer is not copied.
    #[test]
    fn a_float_column_without_a_nan_keeps_its_bounds() {
        let meta = footer(vec![-1.0, 1.0]);
        let kept =
            without_nan_blind_bounds(Arc::clone(&meta), bounds(Some((-1.0, 1.0)))).expect("footer");
        assert!(Arc::ptr_eq(&meta, &kept));
        assert!(chunk_bounds(&kept, FLOAT_LEAF));
        assert!(page_bounds(&kept, FLOAT_LEAF));
    }

    /// A float column the manifest bounds by a NaN, at either end, loses its
    /// row-group and page bounds; the integer column keeps both.
    #[test]
    fn a_float_column_with_a_nan_loses_its_bounds() {
        for (min, max) in [(-f64::NAN, 1.0), (-1.0, f64::NAN)] {
            let meta = footer(vec![min, max]);
            assert!(chunk_bounds(&meta, FLOAT_LEAF), "{min} {max}");
            assert!(page_bounds(&meta, FLOAT_LEAF), "{min} {max}");
            let blind = without_nan_blind_bounds(meta, bounds(Some((min, max)))).expect("footer");
            assert!(!chunk_bounds(&blind, FLOAT_LEAF), "{min} {max}");
            assert!(!page_bounds(&blind, FLOAT_LEAF), "{min} {max}");
            assert!(chunk_bounds(&blind, INT_LEAF), "{min} {max}");
            assert!(page_bounds(&blind, INT_LEAF), "{min} {max}");
        }
    }

    /// A float column the manifest records no bounds for may hold a NaN, so
    /// it loses its footer bounds too.
    #[test]
    fn a_float_column_without_manifest_bounds_loses_its_bounds() {
        let blind =
            without_nan_blind_bounds(footer(vec![-1.0, 1.0]), bounds(None)).expect("footer");
        assert!(!chunk_bounds(&blind, FLOAT_LEAF));
        assert!(chunk_bounds(&blind, INT_LEAF));
    }

    /// Only a float bound that is a number, or null, counts as one.
    #[test]
    fn a_bound_is_a_number_only_when_a_float_holds_one() {
        assert!(is_number_or_null(&ScalarValue::Float64(Some(1.5))));
        assert!(is_number_or_null(&ScalarValue::Float64(None)));
        assert!(!is_number_or_null(&ScalarValue::Float64(Some(f64::NAN))));
        assert!(!is_number_or_null(&ScalarValue::Float32(Some(-f32::NAN))));
        assert!(!is_number_or_null(&ScalarValue::Int64(Some(1))));
    }
}
