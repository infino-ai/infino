// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared machinery for the search TVFs' custom `ExecutionPlan`s.
//!
//! All search TVFs (`vector_search`, `bm25_search`,
//! `bm25_search_prefix`, ...) produce a `Vec<SuperfileHit>` from a
//! kernel and then face the same two jobs:
//!
//!   1. **Resolve** each `(superfile, local_doc_id)` hit to the
//!      supertable's `_id` + projected scalar columns via
//!      [`SuperfileReader::take_by_local_doc_ids`], preserving the
//!      kernel's rank order, and append a `score` column.
//!   2. **Parse** the literal SQL arguments (`column`, `k`, ...).
//!
//! [`SuperfileReader::take_by_local_doc_ids`]: crate::superfile::SuperfileReader::take_by_local_doc_ids

use std::sync::Arc;

use crate::superfile::SuperfileReader;
use crate::superfile::reader::{rank_back_indices, row_selection_for_ids};
use crate::supertable::handle::SupertableReader;
use crate::supertable::manifest::SuperfileUri;
use crate::supertable::options::{DECIMAL128_PRECISION, DECIMAL128_SCALE};
use crate::supertable::query::SuperfileHit;
use arrow::compute::{concat_batches, take};
use arrow_array::{
    ArrayRef, Decimal128Array, Float32Array, RecordBatch, RecordBatchOptions, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::logical_expr::Expr;
use datafusion::scalar::ScalarValue;
use futures::TryStreamExt;
use parquet::arrow::ProjectionMask;
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use rayon::prelude::*;

/// Resolve `hits` to one `RecordBatch`, with `projection` naming the
/// output columns (any of `_id`, the visible scalar columns, or the
/// trailing `score`); `None` returns the engine-native `_id` + `score`
/// pair. Names are resolved to output-schema indices and forwarded to
/// [`resolve_hits`], which decodes only the projected columns. Shared
/// by every public row-returning search method (`bm25_search`,
/// `vector_search`, `token_match`, `exact_match`); `what` labels error
/// messages with the calling method.
pub(crate) async fn resolve_hits_named(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
    projection: Option<&[&str]>,
    what: &str,
) -> DfResult<RecordBatch> {
    let scalar_schema = reader.options().scalar_schema();
    let output_schema = output_schema_with_score(&scalar_schema);
    // `None` is the engine-native result: `_id` + `score` only.
    // `_id` decodes from its own dedicated id pages (cheap by
    // design) and `score` is synthesized from the hits, so the
    // bare call never touches user-column data pages — projecting
    // those is an explicit opt-in by name.
    let id_column = reader.options().id_column.clone();
    let bare: [&str; 2] = [id_column.as_str(), SCORE_COLUMN];
    let names: &[&str] = match projection {
        Some(names) => names,
        None => &bare,
    };
    let indices: Option<Vec<usize>> = Some(
        names
            .iter()
            .map(|name| {
                output_schema.index_of(name).map_err(|_| {
                    DataFusionError::Execution(format!("{what}: unknown column {name:?}"))
                })
            })
            .collect::<Result<_, _>>()?,
    );
    resolve_hits(
        reader,
        hits,
        &scalar_schema,
        &output_schema,
        indices.as_deref(),
    )
    .await
}

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
/// that selects just `score` opens no superfile readers and touches no
/// scalar bytes (cost-first: never decode a column the query did not
/// select). The `score` column is synthesized from the hits.
///
/// Selected scalar columns are read per superfile (each
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

    let id_column = reader.options().id_column.as_str();
    let resolved = if needed.is_empty() {
        None
    } else if needed == [id_column] {
        // Hit → `_id` translation without touching the file: ids are
        // minted in contiguous spans and the superfile body stores
        // rows in id order, so a segment whose manifest stats satisfy
        // `id_max - id_min + 1 == n_docs` maps `local_doc_id` to
        // `id_min + local_doc_id` by arithmetic. Falls back to the
        // id-page read for any segment where the span check fails
        // (multi-span commits can gap the range).
        match resolve_ids_arithmetic(reader, hits) {
            Some(batch) => Some(batch?),
            None => Some(resolve_columns(reader, hits, &needed).await?),
        }
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

/// Hit → stable-`_id` translation by manifest arithmetic — the
/// no-I/O fast path for the bare (`None`) projection.
///
/// Ids are minted in contiguous spans and the superfile body stores
/// rows in id order, so when a superfile's manifest stats satisfy
/// `id_max - id_min + 1 == n_docs` the stable id of row `local` is
/// exactly `id_min + local`. Returns the single-`_id`-column batch in
/// hit (rank) order, or `None` when any hit's superfile fails the span
/// check (e.g. a multi-span commit gapped the range) — the caller
/// then falls back to the id-page read.
fn resolve_ids_arithmetic(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
) -> Option<DfResult<RecordBatch>> {
    let manifest = reader.manifest();
    // Hit sets are top-k sized, so per-superfile memoization via a
    // linear scan is cheaper than building a map.
    let mut memo: Vec<(SuperfileUri, i128)> = Vec::new();
    let mut ids: Vec<i128> = Vec::with_capacity(hits.len());
    for hit in hits {
        let base = match memo.iter().find(|(uri, _)| *uri == hit.superfile) {
            Some((_, base)) => *base,
            None => {
                let entry = manifest
                    .superfiles
                    .iter()
                    .find(|e| e.uri == hit.superfile)?;
                let n_docs = i128::from(entry.n_docs);
                let span = entry.id_max.checked_sub(entry.id_min)?.checked_add(1)?;
                if n_docs == 0 || span != n_docs {
                    return None;
                }
                memo.push((hit.superfile, entry.id_min));
                entry.id_min
            }
        };
        ids.push(base + i128::from(hit.local_doc_id));
    }

    let array = match Decimal128Array::from_iter_values(ids)
        .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
    {
        Ok(a) => a,
        Err(e) => return Some(Err(DataFusionError::Execution(e.to_string()))),
    };
    let schema = Arc::new(Schema::new(vec![Field::new(
        reader.options().id_column.clone(),
        DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
        false,
    )]));
    Some(
        RecordBatch::try_new(schema, vec![Arc::new(array) as ArrayRef])
            .map_err(|e| DataFusionError::Execution(e.to_string())),
    )
}

/// Read `names` (scalar columns) at the `hits`' `(superfile,
/// local_doc_id)` rows and return them in global rank order.
///
/// Hits are grouped by superfile for one column-projected
/// [`take_by_local_doc_ids`] per superfile; the per-superfile batches are
/// concatenated and a single `take` restores rank order. Caller
/// guarantees `hits` and `names` are both non-empty.
///
/// [`take_by_local_doc_ids`]: crate::superfile::SuperfileReader::take_by_local_doc_ids
async fn resolve_columns(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
    names: &[&str],
) -> DfResult<RecordBatch> {
    // Group local_doc_ids by superfile, preserving first-seen superfile
    // order and recording where each global hit lands.
    let mut seg_order: Vec<SuperfileUri> = Vec::new();
    let mut seg_locals: Vec<Vec<u32>> = Vec::new();
    let mut placement: Vec<(usize, usize)> = Vec::with_capacity(hits.len());
    for hit in hits {
        let seg_idx = match seg_order.iter().position(|s| *s == hit.superfile) {
            Some(i) => i,
            None => {
                seg_order.push(hit.superfile);
                seg_locals.push(Vec::new());
                seg_order.len() - 1
            }
        };
        let row = seg_locals[seg_idx].len();
        seg_locals[seg_idx].push(hit.local_doc_id);
        placement.push((seg_idx, row));
    }

    // Open every distinct superfile reader concurrently on the tokio
    // runtime — these are async I/O (in-memory cache lookups /
    // disk-cache cold fetches), so overlapping them is the right
    // model and they cost ~microseconds when warm.
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

    // Materialize each superfile's projected hit rows, split by tier:
    //
    //   - **Resident readers** (in-memory tier / freshly written):
    //     `take_by_local_doc_ids` is a CPU-bound Parquet page decode
    //     over already-resident bytes, so the whole wave runs on
    //     `options.reader_pool` (rayon) — the same pool the search
    //     kernels and the writer's shard builds use — bridged back via
    //     a oneshot so no tokio worker blocks under the compute.
    //   - **Lazy readers** stream ONLY the projected hit rows through
    //     parquet's async `ParquetObjectReader` (footer + projected
    //     column pages via range GETs) — async I/O that belongs on the
    //     query runtime; a cold read never materializes the superfile.
    //
    // Both waves run concurrently and stitch back in `seg_order`
    // order. Superfile count here is bounded by the global top-k (one
    // entry per distinct hit-bearing superfile), so the fan-out is small.
    let mut warm_inputs: Vec<(usize, Arc<SuperfileReader>, Vec<u32>)> = Vec::new();
    let mut cold_units: Vec<(usize, &SuperfileUri, &Arc<SuperfileReader>, &[u32])> = Vec::new();
    for (i, ((uri, rd), locals)) in seg_order
        .iter()
        .zip(opened.iter())
        .zip(seg_locals.iter())
        .enumerate()
    {
        if rd.parquet_bytes().is_some() {
            warm_inputs.push((i, Arc::clone(rd), locals.clone()));
        } else {
            cold_units.push((i, uri, rd, locals.as_slice()));
        }
    }

    let warm_wave = async {
        if warm_inputs.is_empty() {
            return Ok::<Vec<(usize, RecordBatch)>, DataFusionError>(Vec::new());
        }
        // Owned inputs so the rayon closure is `'static`.
        let owned_names: Vec<String> = names.iter().map(|s| (*s).to_string()).collect();
        let pool = Arc::clone(&manifest.options.reader_pool);
        let inputs = warm_inputs;
        let (tx, rx) = tokio::sync::oneshot::channel();
        pool.spawn(move || {
            let name_refs: Vec<&str> = owned_names.iter().map(String::as_str).collect();
            let result: Result<Vec<(usize, RecordBatch)>, _> = inputs
                .into_par_iter()
                .map(|(i, sf, locals)| {
                    sf.take_by_local_doc_ids(&locals, &name_refs)
                        .map(|batch| (i, batch))
                })
                .collect();
            let _ = tx.send(result);
        });
        rx.await
            .map_err(|_| {
                DataFusionError::Execution("resolve decode: reader pool dropped result".into())
            })?
            .map_err(|e| DataFusionError::Execution(e.to_string()))
    };

    let cold_wave = futures::future::try_join_all(cold_units.into_iter().map(
        |(i, uri, reader, locals)| {
            let storage = storage.cloned();
            let file_size = manifest
                .superfiles
                .iter()
                .find(|e| e.uri == *uri)
                .and_then(|e| e.subsection_offsets.as_ref())
                .map(|o| o.total_size);
            async move {
                let storage = storage.ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "resolve_hits needs row bytes for {uri:?}, but the reader was lazy and no storage backend is attached"
                    ))
                })?;
                let (store, path) =
                    storage.object_store_handle(&uri.storage_path()).ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "resolve_hits: storage backend exposes no object_store handle for {uri:?}"
                        ))
                    })?;
                take_rows_object_store(
                    store,
                    path,
                    file_size,
                    reader.schema(),
                    reader.n_docs(),
                    locals,
                    names,
                )
                .await
                .map(|batch| (i, batch))
            }
        },
    ));

    let (warm_done, cold_done) = tokio::join!(warm_wave, cold_wave);
    let mut slots: Vec<Option<RecordBatch>> = vec![None; seg_order.len()];
    for (i, batch) in warm_done?.into_iter().chain(cold_done?) {
        slots[i] = Some(batch);
    }
    let per_superfile: Vec<RecordBatch> = slots
        .into_iter()
        .map(|s| s.expect("invariant: every superfile resolved by exactly one wave"))
        .collect();
    // Concatenate, then reorder rows into global rank order.
    let cat_schema = per_superfile[0].schema();
    let combined = concat_batches(&cat_schema, &per_superfile)
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

    let mut offsets: Vec<u32> = Vec::with_capacity(per_superfile.len());
    let mut acc: u32 = 0;
    for batch in &per_superfile {
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

/// Stream the projected `names` columns at `local_doc_ids` from a lazy
/// object-store superfile via parquet's async `ParquetObjectReader`
/// (footer + projected column pages fetched as range GETs). Mirrors
/// [`SuperfileReader::take_by_local_doc_ids`]'s row-selection + rank-back,
/// but never materializes the whole superfile — this is the cold/object-
/// store row-resolution path.
///
/// [`SuperfileReader::take_by_local_doc_ids`]: crate::superfile::SuperfileReader::take_by_local_doc_ids
async fn take_rows_object_store(
    store: Arc<dyn object_store::ObjectStore>,
    path: object_store::path::Path,
    file_size: Option<u64>,
    file_schema: &SchemaRef,
    n_docs: u64,
    local_doc_ids: &[u32],
    names: &[&str],
) -> DfResult<RecordBatch> {
    // Projected column indices (file order) + output fields (caller order).
    let mut col_indices = Vec::with_capacity(names.len());
    let mut out_fields: Vec<Field> = Vec::with_capacity(names.len());
    for &name in names {
        let idx = file_schema
            .index_of(name)
            .map_err(|_| DataFusionError::Execution(format!("unknown column {name}")))?;
        col_indices.push(idx);
        out_fields.push(file_schema.field(idx).clone());
    }
    let out_schema = Arc::new(Schema::new(out_fields));

    if local_doc_ids.is_empty() {
        return Ok(RecordBatch::new_empty(out_schema));
    }
    for &d in local_doc_ids {
        if u64::from(d) >= n_docs {
            return Err(DataFusionError::Execution(format!(
                "doc id {d} out of range (n_docs={n_docs})"
            )));
        }
    }

    // Distinct, sorted ids → monotonic skip/select runs (decode only the
    // rows the hits land on, not the whole column). Same selection
    // contract as `take_by_local_doc_ids` — shared helpers, different
    // I/O model (async range GETs here vs resident-bytes decode there).
    let (sorted, selection) = row_selection_for_ids(local_doc_ids);

    let mut object_reader = ParquetObjectReader::new(store, path);
    if let Some(size) = file_size.filter(|&s| s > 0) {
        // Skip the size-discovery HEAD when the manifest already knows it.
        object_reader = object_reader.with_file_size(size);
    }
    let builder = ParquetRecordBatchStreamBuilder::new(object_reader)
        .await
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), col_indices.iter().copied());
    let stream = builder
        .with_projection(mask)
        .with_row_selection(selection)
        .build()
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    let batches: Vec<RecordBatch> = stream
        .try_collect()
        .await
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(out_schema));
    }
    let read_schema = batches[0].schema();
    let selected = concat_batches(&read_schema, &batches)
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

    // Rank back into the caller's (possibly duplicated) order.
    let indices = rank_back_indices(local_doc_ids, &sorted);

    // Gather columns in caller projection order (parquet returns file order).
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(names.len());
    for &name in names {
        let idx = selected
            .schema()
            .index_of(name)
            .map_err(|_| DataFusionError::Execution(format!("unknown column {name}")))?;
        columns.push(
            take(selected.column(idx), &indices, None)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?,
        );
    }
    RecordBatch::try_new(out_schema, columns).map_err(|e| DataFusionError::Execution(e.to_string()))
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
