// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Split FixedSizeList vector columns out of an input `RecordBatch`.
//!
//! Supertable's append API takes a single `RecordBatch` carrying
//! both scalar columns and vector columns. The underlying
//! `SuperfileBuilder::add_batch` expects vectors out of band as
//! `&[&[f32]]`. `split_vectors` is the bridge.
//!
//! ## What gets validated at split time
//!
//! Schema-static checks live in `SupertableOptions::new` — types,
//! dims, name uniqueness, etc. The runtime checks here are the ones
//! that depend on the data of a specific `RecordBatch`:
//!
//! 1. Input batch's schema field-by-field equals the supertable's
//!    declared schema (every append must match what
//!    `Supertable::create` declared).
//! 2. Each vector column's underlying `FixedSizeListArray` has no
//!    null entries — null vectors aren't permitted, since the IVF
//!    index has no notion of "skip this row's vector".
//!
//! ## Zero-copy
//!
//! The returned `&[f32]` slices view directly into the input
//! batch's `Float32Array` buffers. No copy at split time. Lifetime:
//! tied to `&'a RecordBatch`, so callers must keep the batch alive
//! while using the slices. The writer side buffers
//! `Arc<Float32Array>` directly to extend the lifetime past a
//! single `append` call.

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch};

use crate::supertable::error::BuildError;
use crate::supertable::options::SupertableOptions;

/// Maximum number of null row offsets collected into a
/// `VectorColumnHasNulls` error. Bounds the error payload so a batch
/// that is null-heavy doesn't produce an unbounded diagnostic list.
const MAX_NULL_OFFSETS_IN_ERROR: usize = 5;

/// Split vector columns out of `batch`. Returns a `RecordBatch` of
/// scalar-only columns (matching `options.scalar_schema()`) plus a
/// `Vec<&[f32]>` parallel to `options.vector_columns` — one slice
/// per declared vector column, in declaration order.
///
/// See module docs for what's validated here vs at options time.
pub(crate) fn split_vectors<'a>(
    batch: &'a RecordBatch,
    options: &SupertableOptions,
) -> Result<(RecordBatch, Vec<&'a [f32]>), BuildError> {
    // 1. The input batch's schema must match the supertable's
    //    declared schema. We don't allow per-batch schema drift
    //    (per non-goal "Schema evolution"). Equality is by
    //    structural comparison.
    if batch.schema().as_ref() != options.schema.as_ref() {
        return Err(BuildError::BatchSchemaMismatch);
    }

    // 2. Pull each vector column out, validate FixedSizeList shape +
    //    no-nulls, view the inner Float32Array as &[f32].
    let mut vectors: Vec<&'a [f32]> = Vec::with_capacity(options.vector_columns.len());
    for vc in &options.vector_columns {
        let idx =
            batch
                .schema()
                .index_of(&vc.column)
                .map_err(|_| BuildError::VectorColumnMissing {
                    column: vc.column.clone(),
                })?;
        let col = batch.column(idx);

        let fsl = col
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| BuildError::VectorColumnNotFixedSizeList {
                column: vc.column.clone(),
                dim: vc.dim,
                actual: format!("{:?}", col.data_type()),
            })?;

        // The FixedSizeListArray's list_size must match dim.
        // (Options-time validation already checked this against the
        // schema, but a freshly constructed RecordBatch with a
        // schema arg whose FixedSizeList is mis-sized would slip
        // through — defensive check.)
        let list_size = usize::try_from(fsl.value_length()).unwrap_or(usize::MAX);
        if list_size != vc.dim {
            return Err(BuildError::VectorColumnDimMismatch {
                column: vc.column.clone(),
                expected: vc.dim,
                actual: list_size,
            });
        }

        // No-nulls check on the FSL itself. If any row's vector is
        // NULL, refuse the batch.
        if fsl.null_count() > 0 {
            let first_nulls = collect_first_nulls(fsl, MAX_NULL_OFFSETS_IN_ERROR);
            return Err(BuildError::VectorColumnHasNulls {
                column: vc.column.clone(),
                first_nulls,
            });
        }

        // Inner Float32Array. The FixedSizeList's flat values array
        // is exactly `n_rows * list_size` long; we view it as &[f32].
        let inner = fsl
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| BuildError::VectorColumnNotFixedSizeList {
                column: vc.column.clone(),
                dim: vc.dim,
                actual: format!("{:?}", col.data_type()),
            })?;

        // Reject if the inner Float32Array itself has nulls (each
        // f32 slot null). Distinct from FSL-level nulls above —
        // inner nulls would mean an individual lane is null inside
        // a non-null vector, which we also can't represent in IVF.
        if inner.null_count() > 0 {
            let first_nulls = collect_first_nulls_primitive(inner, MAX_NULL_OFFSETS_IN_ERROR);
            return Err(BuildError::VectorColumnHasNulls {
                column: vc.column.clone(),
                first_nulls,
            });
        }

        vectors.push(inner.values());
    }

    // 3. Project scalar-only RecordBatch by dropping vector columns.
    //    project_by_name preserves field order from the projection
    //    list, so collect the kept names in their original schema
    //    order.
    let scalar_field_names: Vec<&str> = options
        .schema
        .fields()
        .iter()
        .filter(|f| {
            !options
                .vector_columns
                .iter()
                .any(|vc| vc.column == *f.name())
        })
        .map(|f| f.name().as_str())
        .collect();
    let scalar_batch = batch
        .project(
            &scalar_field_names
                .iter()
                .map(|n| {
                    batch.schema().index_of(n).expect(
                        "invariant: name from options.schema is in batch.schema (checked above)",
                    )
                })
                .collect::<Vec<_>>(),
        )
        .map_err(|_| BuildError::BatchSchemaMismatch)?;

    Ok((scalar_batch, vectors))
}

fn collect_first_nulls(arr: &FixedSizeListArray, max: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(max);
    for i in 0..arr.len() {
        if arr.is_null(i) {
            out.push(i);
            if out.len() >= max {
                break;
            }
        }
    }
    out
}

fn collect_first_nulls_primitive(arr: &Float32Array, max: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(max);
    for i in 0..arr.len() {
        if arr.is_null(i) {
            out.push(i);
            if out.len() >= max {
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{Array, Float32Array, LargeStringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};

    use crate::superfile::builder::{FtsConfig, VectorConfig};

    use crate::superfile::vector::distance::Metric;

    fn fixed_list_f32(dim: usize) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        )
    }

    fn schema_id_title_emb(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), false),
        ]))
    }

    fn vc(name: &str, dim: usize) -> VectorConfig {
        VectorConfig {
            column: name.into(),
            dim,
            n_cent: 4,
            rot_seed: 0,
            metric: Metric::Cosine,
            rerank_codec: crate::superfile::vector::rerank_codec::RerankCodec::Fp32,
        }
    }

    fn fc(name: &str) -> FtsConfig {
        FtsConfig {
            column: name.into(),
        }
    }

    use crate::test_helpers::default_tokenizer as tok;

    /// Build a FixedSizeListArray of `n_rows` × `dim` f32s from a flat
    /// `Vec<f32>` of length `n_rows * dim`. No null entries.
    fn build_fsl(flat: Vec<f32>, dim: usize) -> FixedSizeListArray {
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(flat);
        FixedSizeListArray::try_new(item_field, dim as i32, Arc::new(values), None)
            .expect("build FixedSizeListArray")
    }

    fn build_batch(schema: Arc<Schema>, n: usize, dim: usize) -> RecordBatch {
        let titles = LargeStringArray::from((0..n).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let mut flat = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                flat.push((i * dim + j) as f32);
            }
        }
        let fsl = build_fsl(flat, dim);
        RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(fsl)])
            .expect("build RecordBatch")
    }

    #[test]
    fn split_extracts_vectors_and_drops_columns() {
        let dim = 16;
        let schema = schema_id_title_emb(dim);
        let opts = SupertableOptions::new(
            schema.clone(),
            vec![fc("title")],
            vec![vc("emb", dim)],
            Some(tok()),
        )
        .expect("valid options");

        let batch = build_batch(schema, 4, dim);
        let (scalar, vectors) = split_vectors(&batch, &opts).expect("split should succeed");

        // Scalar batch keeps only title in schema order
        // (vector columns dropped; the supertable's writer
        // prepends `_id` separately at append time, which
        // split_vectors does not).
        let names: Vec<_> = scalar
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, vec!["title"]);
        assert_eq!(scalar.num_rows(), 4);
        assert_eq!(scalar.num_columns(), 1);

        // One vector slice for emb, length n_rows * dim.
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].len(), 4 * dim);
        // Spot-check value ordering: row 2 col 3 = 2*16+3 = 35.
        assert_eq!(vectors[0][2 * dim + 3], 35.0);
    }

    #[test]
    fn split_rejects_batch_with_wrong_schema() {
        let dim = 16;
        let schema = schema_id_title_emb(dim);
        let opts = SupertableOptions::new(
            schema.clone(),
            vec![fc("title")],
            vec![vc("emb", dim)],
            Some(tok()),
        )
        .expect("valid options");

        // Build a batch with a different schema (no `title` column).
        let other_schema = Arc::new(Schema::new(vec![Field::new(
            "emb",
            fixed_list_f32(dim),
            false,
        )]));
        let fsl = build_fsl(vec![0.0; 2 * dim], dim);
        let other_batch =
            RecordBatch::try_new(other_schema, vec![Arc::new(fsl)]).expect("build batch");

        let err = split_vectors(&other_batch, &opts).expect_err("expected error");
        assert!(matches!(err, BuildError::BatchSchemaMismatch));
    }

    #[test]
    fn split_rejects_null_vector_row() {
        let dim = 16;
        // Schema must declare emb as nullable (true) so we can
        // construct a batch with a null entry at row 1; if the
        // schema said non-nullable, RecordBatch::try_new would
        // reject the batch before split_vectors ever sees it.
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), true),
        ]));
        let opts = SupertableOptions::new(
            schema.clone(),
            vec![fc("title")],
            vec![vc("emb", dim)],
            Some(tok()),
        )
        .expect("valid options");

        // Build a FSL with a NULL entry at row 1.
        use arrow::buffer::NullBuffer;
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(vec![0.0f32; 3 * dim]);
        let nulls = NullBuffer::from(vec![true, false, true]); // row 1 is null
        let fsl =
            FixedSizeListArray::try_new(item_field, dim as i32, Arc::new(values), Some(nulls))
                .expect("build FSL with nulls");

        let titles = LargeStringArray::from(vec!["a", "b", "c"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(fsl)])
            .expect("build batch");

        let err = split_vectors(&batch, &opts).expect_err("expected error");
        match err {
            BuildError::VectorColumnHasNulls {
                column,
                first_nulls,
            } => {
                assert_eq!(column, "emb");
                assert_eq!(first_nulls, vec![1]);
            }
            other => panic!("expected VectorColumnHasNulls, got {:?}", other),
        }
    }

    #[test]
    fn split_succeeds_with_zero_vector_columns() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]));
        let opts = SupertableOptions::new(schema.clone(), vec![fc("title")], vec![], Some(tok()))
            .expect("valid options");

        let titles = LargeStringArray::from(vec!["x", "y"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(titles)]).expect("build batch");

        let (scalar, vectors) = split_vectors(&batch, &opts).expect("split should succeed");
        assert_eq!(scalar.num_rows(), 2);
        assert_eq!(scalar.num_columns(), 1);
        assert_eq!(vectors.len(), 0);
    }

    #[test]
    fn split_preserves_scalar_column_order() {
        // Schema: a, vec_emb, b, vec_other, c — scalar projection
        // should produce [a, b, c] in that order.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::UInt64, false),
            Field::new("emb", fixed_list_f32(16), false),
            Field::new("b", DataType::UInt64, false),
            Field::new("other", fixed_list_f32(16), false),
            Field::new("c", DataType::UInt64, false),
        ]));
        let opts = SupertableOptions::new(
            schema.clone(),
            vec![],
            vec![vc("emb", 16), vc("other", 16)],
            None,
        )
        .expect("valid options");

        let n = 2;
        let dim = 16;
        let a = UInt64Array::from(vec![10u64, 20]);
        let b = UInt64Array::from(vec![30u64, 40]);
        let c = UInt64Array::from(vec![50u64, 60]);
        let fsl1 = build_fsl(vec![0.0; n * dim], dim);
        let fsl2 = build_fsl(vec![1.0; n * dim], dim);
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(a),
                Arc::new(fsl1),
                Arc::new(b),
                Arc::new(fsl2),
                Arc::new(c),
            ],
        )
        .expect("build batch");

        let (scalar, vectors) = split_vectors(&batch, &opts).expect("split should succeed");
        let names: Vec<_> = scalar
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0].len(), n * dim);
        assert_eq!(vectors[1].len(), n * dim);
        // The two vector slices are disjoint (different fill).
        assert_eq!(vectors[0][0], 0.0);
        assert_eq!(vectors[1][0], 1.0);
    }

    #[test]
    fn split_returns_zero_copy_view_into_batch() {
        let dim = 16;
        let schema = schema_id_title_emb(dim);
        let opts = SupertableOptions::new(
            schema.clone(),
            vec![fc("title")],
            vec![vc("emb", dim)],
            Some(tok()),
        )
        .expect("valid options");
        let batch = build_batch(schema, 4, dim);

        let (_scalar, vectors) = split_vectors(&batch, &opts).expect("split should succeed");
        // Compare the slice's pointer to the underlying Float32Array's
        // values pointer in the original batch — they must be the
        // same memory (zero-copy contract).
        // schema is [title, emb] — emb is at column index 1.
        let original = batch
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("FSL")
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("Float32Array")
            .values()
            .as_ptr();
        assert_eq!(vectors[0].as_ptr(), original);
    }
}
