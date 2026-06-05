//! Shared machinery for the search TVFs' custom `ExecutionPlan`s.
//!
//! All search TVFs (`vector_search`, `bm25_search`,
//! `bm25_search_prefix`, ...) produce a `Vec<SuperfileHit>` from a
//! kernel and then face the same two jobs:
//!
//!   1. **Resolve** each `(segment, local_doc_id)` hit to the
//!      supertable's `_id` + projected scalar columns via
//!      [`SuperfileReader::take_by_local_doc_ids`], preserving the
//!      kernel's rank order, and append a `score` column.
//!   2. **Parse** the literal SQL arguments (`column`, `k`, ...).
//!
//! [`SuperfileReader::take_by_local_doc_ids`]: crate::superfile::SuperfileReader::take_by_local_doc_ids

use std::sync::Arc;

use rayon::prelude::*;

use arrow::compute::{concat_batches, take};
use arrow_array::{ArrayRef, Float32Array, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::logical_expr::Expr;
use datafusion::scalar::ScalarValue;

use crate::supertable::handle::SupertableReader;
use crate::supertable::manifest::SuperfileUri;
use crate::supertable::query::SuperfileHit;

/// Output column carrying the per-hit score (vector distance or BM25
/// relevance — direction is the originating TVF's contract).
pub(crate) const SCORE_COLUMN: &str = "score";

/// Search-TVF output schema: the scalar schema with a trailing
/// non-null `score: Float32` appended.
pub(crate) fn output_schema_with_score(scalar_schema: &SchemaRef) -> SchemaRef {
    let mut fields: Vec<Field> = scalar_schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields.push(Field::new(SCORE_COLUMN, DataType::Float32, false));
    Arc::new(Schema::new(fields))
}

/// Resolve `hits` (in kernel rank order) to a `RecordBatch` matching
/// `output_schema` projected by `projection`, preserving rank order.
///
/// `output_schema` is the scalar schema with a trailing `score`
/// column ([`output_schema_with_score`]); `projection` indexes into
/// it, exactly as DataFusion hands to `scan`. **Only the scalar
/// columns the projection actually selects are decoded** — a query
/// that selects just `score` opens no segment readers and touches no
/// scalar bytes (cost-first: never decode a column the query did not
/// select). The `score` column is synthesized from the hits.
///
/// Selected scalar columns are read per segment (each
/// `take_by_local_doc_ids` is a column-projected read), concatenated,
/// then a single `take` reorders rows back into the global rank order
/// so row `i` is the `i`-th hit.
pub(crate) async fn resolve_hits(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
    scalar_schema: &SchemaRef,
    output_schema: &SchemaRef,
    projection: Option<&[usize]>,
) -> DfResult<RecordBatch> {
    let projected_schema = match projection {
        Some(indices) => Arc::new(
            output_schema
                .project(indices)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?,
        ),
        None => Arc::clone(output_schema),
    };
    if hits.is_empty() {
        return Ok(RecordBatch::new_empty(projected_schema));
    }

    // `score` is the trailing column of `output_schema`; every
    // smaller index is a scalar column.
    let score_idx = scalar_schema.fields().len();
    let requested: Vec<usize> = match projection {
        Some(indices) => indices.to_vec(),
        None => (0..output_schema.fields().len()).collect(),
    };

    // Distinct scalar columns the projection selects, in first-seen
    // order — the only columns we decode.
    let mut needed: Vec<&str> = Vec::new();
    for &p in &requested {
        if p != score_idx {
            let name = scalar_schema.field(p).name().as_str();
            if !needed.contains(&name) {
                needed.push(name);
            }
        }
    }

    let resolved = if needed.is_empty() {
        None
    } else {
        Some(resolve_columns(reader, hits, &needed).await?)
    };

    // Assemble output columns in the projection's emit order, each
    // drawn from the decoded scalar batch or the synthesized score.
    let score = Arc::new(Float32Array::from_iter_values(hits.iter().map(|h| h.score))) as ArrayRef;
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(requested.len());
    for &p in &requested {
        if p == score_idx {
            columns.push(Arc::clone(&score));
        } else {
            let name = scalar_schema.field(p).name();
            let rb = resolved
                .as_ref()
                .expect("a scalar column is projected => columns resolved");
            let idx = rb
                .schema()
                .index_of(name)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            columns.push(Arc::clone(rb.column(idx)));
        }
    }

    // `try_new_with_options` carries the row count so a projection
    // that selects no columns (e.g. `COUNT(*)`) still reports
    // `hits.len()` rows.
    RecordBatch::try_new_with_options(
        projected_schema,
        columns,
        &RecordBatchOptions::new().with_row_count(Some(hits.len())),
    )
    .map_err(|e| DataFusionError::Execution(e.to_string()))
}

/// Read `names` (scalar columns) at the `hits`' `(segment,
/// local_doc_id)` rows and return them in global rank order.
///
/// Hits are grouped by segment for one column-projected
/// [`take_by_local_doc_ids`] per segment; the per-segment batches are
/// concatenated and a single `take` restores rank order. Caller
/// guarantees `hits` and `names` are both non-empty.
///
/// [`take_by_local_doc_ids`]: crate::superfile::SuperfileReader::take_by_local_doc_ids
async fn resolve_columns(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
    names: &[&str],
) -> DfResult<RecordBatch> {
    // Group local_doc_ids by segment, preserving first-seen segment
    // order and recording where each global hit lands.
    let mut seg_order: Vec<SuperfileUri> = Vec::new();
    let mut seg_locals: Vec<Vec<u32>> = Vec::new();
    let mut placement: Vec<(usize, usize)> = Vec::with_capacity(hits.len());
    for hit in hits {
        let seg_idx = match seg_order.iter().position(|s| *s == hit.segment) {
            Some(i) => i,
            None => {
                seg_order.push(hit.segment);
                seg_locals.push(Vec::new());
                seg_order.len() - 1
            }
        };
        let row = seg_locals[seg_idx].len();
        seg_locals[seg_idx].push(hit.local_doc_id);
        placement.push((seg_idx, row));
    }

    // Per-segment column-projected reads, split by concern:
    //   1. **Open** every distinct segment reader **concurrently** on
    //      the tokio runtime — these are async I/O (in-memory cache
    //      lookups / disk-cache cold fetches), so overlapping them is
    //      the right model and they cost ~microseconds when warm.
    //   2. **Decode** every segment **in parallel** on
    //      `options.reader_pool` (rayon). `take_by_local_doc_ids` is a
    //      CPU-bound Parquet page decode over already-resident bytes;
    //      the SQL tokio runtime is single-worker by design (it drives
    //      the I/O state machine, not CPU), so CPU fan-out belongs on
    //      the reader pool — the same pool the search kernels and the
    //      writer's shard builds use. The work is bridged back to the
    //      async caller via a oneshot so the tokio worker is never
    //      blocked.
    let manifest = reader.manifest();
    let store = &manifest.options.store;
    let disk_cache = manifest.options.disk_cache.as_ref();
    let storage = manifest.options.storage.as_ref();

    let opened = futures::future::try_join_all(seg_order.iter().map(|uri| {
        crate::supertable::query::superfile_reader::superfile_reader(
            store, disk_cache, storage, uri, None,
        )
    }))
    .await
    .map_err(|e| DataFusionError::Execution(e.to_string()))?;

    // Owned inputs so the rayon closure is `'static`. `rayon`'s
    // ordered `collect` preserves segment order, so `per_segment[i]`
    // still lines up with `seg_order[i]` for the `offsets`/`placement`
    // reorder below.
    let owned_names: Vec<String> = names.iter().map(|s| (*s).to_string()).collect();
    let inputs: Vec<(Arc<crate::superfile::SuperfileReader>, Vec<u32>)> =
        opened.into_iter().zip(seg_locals).collect();
    let pool = Arc::clone(&manifest.options.reader_pool);
    let (tx, rx) = tokio::sync::oneshot::channel();
    pool.spawn(move || {
        let name_refs: Vec<&str> = owned_names.iter().map(String::as_str).collect();
        let result: Result<Vec<RecordBatch>, _> = inputs
            .into_par_iter()
            .map(|(sf, locals)| sf.take_by_local_doc_ids(&locals, &name_refs))
            .collect();
        let _ = tx.send(result);
    });
    let per_segment: Vec<RecordBatch> = rx
        .await
        .map_err(|_| {
            DataFusionError::Execution("resolve decode: reader pool dropped result".into())
        })?
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

    // Concatenate, then reorder rows into global rank order.
    let cat_schema = per_segment[0].schema();
    let combined = concat_batches(&cat_schema, &per_segment)
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

    let mut offsets: Vec<u32> = Vec::with_capacity(per_segment.len());
    let mut acc: u32 = 0;
    for batch in &per_segment {
        offsets.push(acc);
        acc += batch.num_rows() as u32;
    }
    let reorder =
        UInt32Array::from_iter_values(placement.iter().map(|(s, r)| offsets[*s] + *r as u32));

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(combined.num_columns());
    for column in combined.columns() {
        columns.push(
            take(column, &reorder, None).map_err(|e| DataFusionError::Execution(e.to_string()))?,
        );
    }
    RecordBatch::try_new(combined.schema(), columns)
        .map_err(|e| DataFusionError::Execution(e.to_string()))
}

/// Extract a string literal argument (a column name, query text, ...).
pub(crate) fn arg_to_string(expr: &Expr, what: &str) -> DfResult<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => Ok(s.clone()),
        other => Err(DataFusionError::Plan(format!(
            "{what} must be a string literal, got {other:?}"
        ))),
    }
}

/// Extract a non-negative integer literal argument (`k`).
pub(crate) fn arg_to_usize(expr: &Expr, what: &str) -> DfResult<usize> {
    let n: i64 = match expr {
        Expr::Literal(ScalarValue::Int64(Some(n)), _) => *n,
        Expr::Literal(ScalarValue::Int32(Some(n)), _) => i64::from(*n),
        Expr::Literal(ScalarValue::UInt64(Some(n)), _) => *n as i64,
        Expr::Literal(ScalarValue::UInt32(Some(n)), _) => i64::from(*n),
        other => {
            return Err(DataFusionError::Plan(format!(
                "{what} must be an integer literal, got {other:?}"
            )));
        }
    };
    usize::try_from(n).map_err(|_| DataFusionError::Plan(format!("{what} must be >= 0, got {n}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::lit;

    #[test]
    fn arg_to_string_accepts_utf8_literal_rejects_int() {
        assert_eq!(
            arg_to_string(&lit("emb"), "column").expect("utf8 literal"),
            "emb"
        );
        assert!(arg_to_string(&lit(3_i64), "column").is_err());
    }

    #[test]
    fn arg_to_usize_accepts_int_rejects_negative_and_nonint() {
        assert_eq!(arg_to_usize(&lit(10_i64), "k").expect("int literal"), 10);
        assert!(arg_to_usize(&lit(-1_i64), "k").is_err());
        assert!(arg_to_usize(&lit("nope"), "k").is_err());
    }

    #[test]
    fn output_schema_appends_score() {
        let s = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        let out = output_schema_with_score(&s);
        assert_eq!(out.fields().len(), 2);
        assert_eq!(out.field(1).name(), "score");
        assert_eq!(out.field(1).data_type(), &DataType::Float32);
    }
}
