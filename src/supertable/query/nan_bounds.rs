// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A superfile's Parquet footer as row-group and page pruning may read it:
//! without the min or max of a float column on the side its NaNs sort to.
//!
//! A Parquet writer leaves NaN out of a float column's min/max, as the format
//! asks, so a column chunk holding `{-NaN, 1.0}` records `[1.0, 1.0]`. Arrow
//! compares floats in IEEE total order, where a negative NaN comes before
//! every number and a positive NaN after, and DataFusion filters and sorts
//! the same way. Bounds that leave out the very rows sorting first or last
//! then tell pruning those rows are not there:
//!
//! ```text
//!   WHERE f < -1.0              superfile 2: f ∈ {-NaN, 1.0}, footer [1.0, 1.0]
//!                               pruned: its -NaN rows pass the filter, and
//!                               are never read
//!
//!   ORDER BY f ASC LIMIT 3      superfile 0: f = -1.0       → cutoff f < -1.0
//!                               superfile 2 pruned against the cutoff: its
//!                               -NaN rows, the true first three, are never read
//! ```
//!
//! A plain filter is pruned this way on every run. The cutoff an
//! `ORDER BY ... LIMIT` hands the scan moves as rows arrive, so there
//! whether a row group is pruned depends on which superfiles were read
//! first, and the same statement can answer differently from run to run.
//!
//! The manifest records each superfile's float bounds in total order (Arrow's
//! min and max, merged in that same order), so a NaN shows there as a NaN
//! bound: a negative NaN as the min, a positive NaN as the max. A float
//! column whose min is a NaN loses its row-group min in the footer the scan
//! reads, and one whose max is a NaN loses its row-group max; the other side
//! keeps pruning. A column whose only NaNs are positive (NaN standing for a
//! missing value, say) keeps its mins, so `f < 5` still skips row groups.
//! Pruning reads a missing side as unknown, never as a reason to skip.
//!
//! A page index holds a min and a max for every page with no way to leave
//! one out, so such a column loses its page bounds on both sides. Every
//! other column, and every float column without a NaN, keeps all its bounds.
//!
//! The manifest records no bounds for a half-precision (`Float16`) column,
//! so such a column counts as holding NaNs at both ends in every superfile:
//! it always loses its footer bounds, and every superfile's footer is copied
//! for it.

use std::sync::Arc;

use datafusion::scalar::ScalarValue;
use parquet::{
    basic::{LogicalType, Type as PhysicalType},
    errors::Result as ParquetResult,
    file::{
        metadata::{ParquetMetaData, ParquetMetaDataBuilder, RowGroupMetaData},
        page_index::column_index::ColumnIndexMetaData,
        statistics::{Statistics, ValueStatistics},
    },
    schema::types::ColumnDescriptor,
};

use crate::supertable::schema::FieldId;

/// Parts in the path of a top-level column. Pruning compares only top-level
/// columns against a filter, so only their bounds can drop rows; a nested
/// leaf (a vector's elements, say) keeps its own.
const TOP_LEVEL_PATH_PARTS: usize = 1;

/// The ends of a float column's total order its NaNs may sit at in one
/// superfile, each an end whose footer bound leaves those NaNs out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NanEnds {
    /// A NaN may sort before every number: a negative NaN. The footer's
    /// min leaves it out.
    low: bool,
    /// A NaN may sort after every number: a positive NaN. The footer's max
    /// leaves it out.
    high: bool,
}

impl NanEnds {
    /// No NaN: every footer bound holds.
    const NONE: Self = Self {
        low: false,
        high: false,
    };
    /// NaNs may sit at both ends: neither footer bound holds.
    const BOTH: Self = Self {
        low: true,
        high: true,
    };

    /// Whether a NaN may sit at either end.
    fn any(self) -> bool {
        self.low || self.high
    }
}

/// `meta`, a superfile's footer, without the bounds that leave out a NaN
/// (see the module docs), judged by `bounds`: the manifest's `(min, max)` of
/// a column in that superfile, by field id, or `None` when it records none.
/// `meta` itself, unchanged and uncopied, when no column may hold a NaN.
///
/// The copy is built straight from `meta`'s parts, so each column chunk and
/// page index is copied once.
pub(crate) fn without_nan_blind_bounds(
    meta: Arc<ParquetMetaData>,
    bounds: impl Fn(FieldId) -> Option<(ScalarValue, ScalarValue)>,
) -> ParquetResult<Arc<ParquetMetaData>> {
    let ends: Vec<NanEnds> = meta
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .map(|leaf| {
            if leaf.path().parts().len() == TOP_LEVEL_PATH_PARTS && is_float(leaf) {
                nan_ends(leaf, &bounds)
            } else {
                NanEnds::NONE
            }
        })
        .collect();
    if !ends.iter().any(|leaf| leaf.any()) {
        return Ok(meta);
    }
    let blind = |leaf: usize| ends.get(leaf).copied().unwrap_or(NanEnds::NONE);
    let row_groups = meta
        .row_groups()
        .iter()
        .map(|row_group| without_chunk_bounds(row_group, blind))
        .collect::<ParquetResult<Vec<_>>>()?;
    let column_index = meta.column_index().map(|row_groups| {
        row_groups
            .iter()
            .map(|columns| {
                columns
                    .iter()
                    .enumerate()
                    .map(|(leaf, index)| match blind(leaf).any() {
                        true => ColumnIndexMetaData::NONE,
                        false => index.clone(),
                    })
                    .collect()
            })
            .collect()
    });
    Ok(Arc::new(
        ParquetMetaDataBuilder::new(meta.file_metadata().clone())
            .set_row_groups(row_groups)
            .set_column_index(column_index)
            .set_offset_index(meta.offset_index().cloned())
            .build(),
    ))
}

/// A copy of `row_group` whose column chunks lose the bounds `blind` gives
/// for their leaf.
fn without_chunk_bounds(
    row_group: &RowGroupMetaData,
    blind: impl Fn(usize) -> NanEnds,
) -> ParquetResult<RowGroupMetaData> {
    let columns = row_group
        .columns()
        .iter()
        .enumerate()
        .map(|(leaf, chunk)| {
            let ends = blind(leaf);
            if !ends.any() {
                return Ok(chunk.clone());
            }
            let builder = chunk.clone().into_builder();
            match chunk
                .statistics()
                .and_then(|stats| without_ends(stats, ends))
            {
                Some(stats) => builder.set_statistics(stats),
                None => builder.clear_statistics(),
            }
            .build()
        })
        .collect::<ParquetResult<Vec<_>>>()?;
    let mut builder = RowGroupMetaData::builder(row_group.schema_descr_ptr())
        .set_num_rows(row_group.num_rows())
        .set_sorting_columns(row_group.sorting_columns().cloned())
        .set_total_byte_size(row_group.total_byte_size())
        .set_column_metadata(columns);
    if let Some(ordinal) = row_group.ordinal() {
        builder = builder.set_ordinal(ordinal);
    }
    if let Some(offset) = row_group.file_offset() {
        builder = builder.set_file_offset(offset);
    }
    builder.build()
}

/// A float column chunk's `stats` without the min when `ends` has a low NaN
/// and without the max when it has a high one; its null and distinct counts
/// stay. `None` for statistics that are not a float's, which lose all their
/// bounds.
fn without_ends(stats: &Statistics, ends: NanEnds) -> Option<Statistics> {
    let deprecated = stats.is_min_max_deprecated();
    Some(match stats {
        Statistics::Float(values) => Statistics::Float(keep_ends(values, ends, deprecated)),
        Statistics::Double(values) => Statistics::Double(keep_ends(values, ends, deprecated)),
        Statistics::FixedLenByteArray(values) => {
            Statistics::FixedLenByteArray(keep_ends(values, ends, deprecated))
        }
        _ => return None,
    })
}

/// `values` with the bound on each end of `ends` left out; `deprecated` is
/// whether they came from Parquet's old min/max fields.
fn keep_ends<T: Clone>(
    values: &ValueStatistics<T>,
    ends: NanEnds,
    deprecated: bool,
) -> ValueStatistics<T> {
    let min = values.min_opt().filter(|_| !ends.low).cloned();
    let max = values.max_opt().filter(|_| !ends.high).cloned();
    ValueStatistics::new(
        min,
        max,
        values.distinct_count(),
        values.null_count_opt(),
        deprecated,
    )
    .with_min_is_exact(values.min_is_exact())
    .with_max_is_exact(values.max_is_exact())
    .with_backwards_compatible_min_max(values.is_min_max_backwards_compatible())
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

/// The ends of its order at which the top-level float column `leaf` may
/// hold a NaN, by the manifest's `bounds` of it: low when its min is a NaN,
/// high when its max is. A NaN min is a negative NaN, unless every value is
/// a NaN, and a NaN max a positive one; either way the footer's bound on
/// that side leaves the NaN out. Both ends when `bounds` gives none, or none
/// that are floats, and for a leaf without a field id, which names no column
/// the manifest bounds.
fn nan_ends(
    leaf: &ColumnDescriptor,
    bounds: impl Fn(FieldId) -> Option<(ScalarValue, ScalarValue)>,
) -> NanEnds {
    let info = leaf.self_type().get_basic_info();
    if !info.has_id() {
        return NanEnds::BOTH;
    }
    let Some((min, max)) = u32::try_from(info.id())
        .ok()
        .and_then(|id| bounds(FieldId(id)))
    else {
        return NanEnds::BOTH;
    };
    match (is_nan_bound(&min), is_nan_bound(&max)) {
        (Some(low), Some(high)) => NanEnds { low, high },
        _ => NanEnds::BOTH,
    }
}

/// Whether `bound` is a NaN; `None` when it is not a float bound at all. A
/// null bound, a column with no value in the superfile, is not a NaN.
fn is_nan_bound(bound: &ScalarValue) -> Option<bool> {
    match bound {
        ScalarValue::Float16(value) => Some(value.is_some_and(|v| v.is_nan())),
        ScalarValue::Float32(value) => Some(value.is_some_and(f32::is_nan)),
        ScalarValue::Float64(value) => Some(value.is_some_and(f64::is_nan)),
        _ => None,
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

    /// The float column's row-group `(min, max)` in `meta`, each `None`
    /// where the footer leaves it out.
    fn float_chunk_bounds(meta: &ParquetMetaData) -> (Option<f64>, Option<f64>) {
        match meta.row_group(0).column(FLOAT_LEAF).statistics() {
            Some(Statistics::Double(values)) => {
                (values.min_opt().copied(), values.max_opt().copied())
            }
            _ => (None, None),
        }
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
        assert_eq!(float_chunk_bounds(&kept), (Some(-1.0), Some(1.0)));
        assert!(page_bounds(&kept, FLOAT_LEAF));
    }

    /// A float column loses the row-group bound on the side its NaNs sort
    /// to and keeps the other: a negative NaN takes the min, a positive one
    /// the max, both take both. Its page bounds go whole; its null count,
    /// the integer column and everything else in the footer stay as they
    /// were.
    #[test]
    fn a_float_column_loses_the_bound_on_the_side_of_its_nans() {
        let cases = [
            ((-f64::NAN, 1.0), (None, Some(1.0))),
            ((-1.0, f64::NAN), (Some(-1.0), None)),
            ((-f64::NAN, f64::NAN), (None, None)),
        ];
        for ((min, max), kept) in cases {
            let meta = footer(vec![min, max, 0.0]);
            assert!(page_bounds(&meta, FLOAT_LEAF), "{min} {max}");
            let blind = without_nan_blind_bounds(Arc::clone(&meta), bounds(Some((min, max))))
                .expect("footer");
            assert_eq!(float_chunk_bounds(&blind), kept, "{min} {max}");
            assert!(!page_bounds(&blind, FLOAT_LEAF), "{min} {max}");
            let (row_group, original) = (blind.row_group(0), meta.row_group(0));
            assert_eq!(
                row_group
                    .column(FLOAT_LEAF)
                    .statistics()
                    .and_then(Statistics::null_count_opt),
                original
                    .column(FLOAT_LEAF)
                    .statistics()
                    .and_then(Statistics::null_count_opt),
                "{min} {max}"
            );
            assert_eq!(
                row_group.column(INT_LEAF),
                original.column(INT_LEAF),
                "{min} {max}"
            );
            assert!(page_bounds(&blind, INT_LEAF), "{min} {max}");
            assert_eq!(row_group.num_rows(), original.num_rows());
            assert_eq!(row_group.ordinal(), original.ordinal());
            assert_eq!(row_group.file_offset(), original.file_offset());
            assert_eq!(blind.offset_index(), meta.offset_index());
            assert_eq!(blind.file_metadata(), meta.file_metadata());
        }
    }

    /// A float column the manifest records no bounds for may hold a NaN at
    /// either end, so it loses both footer bounds.
    #[test]
    fn a_float_column_without_manifest_bounds_loses_its_bounds() {
        let blind =
            without_nan_blind_bounds(footer(vec![-1.0, 1.0]), bounds(None)).expect("footer");
        assert_eq!(float_chunk_bounds(&blind), (None, None));
        assert!(!page_bounds(&blind, FLOAT_LEAF));
        assert!(page_bounds(&blind, INT_LEAF));
    }

    /// Only a float bound that holds a NaN counts as one; a null float bound
    /// is no NaN, and a bound of another type is no float.
    #[test]
    fn a_bound_is_a_nan_only_when_a_float_holds_one() {
        assert_eq!(is_nan_bound(&ScalarValue::Float64(Some(1.5))), Some(false));
        assert_eq!(is_nan_bound(&ScalarValue::Float64(None)), Some(false));
        assert_eq!(
            is_nan_bound(&ScalarValue::Float64(Some(f64::NAN))),
            Some(true)
        );
        assert_eq!(
            is_nan_bound(&ScalarValue::Float32(Some(-f32::NAN))),
            Some(true)
        );
        assert_eq!(is_nan_bound(&ScalarValue::Int64(Some(1))), None);
    }
}
