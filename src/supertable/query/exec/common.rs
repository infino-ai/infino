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

use std::{ops::Range, sync::Arc};

use arrow::compute::{concat_batches, interleave_record_batch, take};
use arrow_array::{ArrayRef, Decimal128Array, Float32Array, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use datafusion::{
    error::{DataFusionError, Result as DfResult},
    logical_expr::Expr,
    scalar::ScalarValue,
};
use futures::{
    FutureExt, TryStreamExt,
    future::{BoxFuture, try_join_all},
};
use object_store::{ObjectStore, path::Path as ObjPath};
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::ArrowReaderOptions,
        async_reader::{AsyncFileReader, ParquetObjectReader, ParquetRecordBatchStreamBuilder},
    },
    errors::{ParquetError, Result as ParquetResult},
    file::metadata::ParquetMetaData,
};
use rayon::prelude::*;
use tokio::sync::oneshot;

use crate::{
    superfile::{
        SuperfileReader,
        lazy_source::Source,
        reader::{rank_back_indices, row_selection_for_ids},
    },
    supertable::{
        error::QueryError,
        handle::SupertableReader,
        manifest::SuperfileUri,
        options::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
        query::{
            SuperfileHit, superfile_reader::superfile_reader, vector::row_id_from_manifest_entry,
        },
    },
};

/// Map a search TVF's `QueryError` into a DataFusion error at the
/// execution-node boundary.
///
/// The kNN runs off-SQL (in a custom `ExecutionPlan`), so its error must cross
/// back into DataFusion to bubble up through `collect()`. A connection-memory
/// budget refusal has to re-enter as `ResourcesExhausted`, the same channel
/// DataFusion's own memory pool uses, so the SQL error classifier routes it to
/// `InfinoError::OverBudget` rather than flattening it to a generic query
/// error. Every other failure is a plain execution error.
///
/// Shared by the `vector_search` and `hybrid_search` nodes; without it a hybrid
/// query would flatten the budget refusal that the plain vector query preserves.
pub(crate) fn search_query_df_error(e: QueryError) -> DataFusionError {
    match e.over_budget() {
        Some(msg) => DataFusionError::ResourcesExhausted(msg.to_string()),
        None => DataFusionError::Execution(e.to_string()),
    }
}

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
        // Hit → `_id` translation without touching the file: prefer
        // `hit.stable_id` (inline on MultiCell / hidden IVF), else
        // contiguous-span arithmetic. Falls back to the id-page read
        // only when a hit has neither.
        match resolve_ids_arithmetic(reader, hits) {
            Some(batch) => Some(batch?),
            None => Some(resolve_columns(reader, hits, &needed).await?),
        }
    } else if needed.contains(&id_column) {
        // `_id` + scalars: never Parquet-decode `_id` for identity.
        // MultiCell / hidden hits already carry stable `_id` on the hit;
        // id-ordered files use span arithmetic. Only the non-id columns
        // come from `take_by_local_doc_ids`.
        let other: Vec<&str> = needed
            .iter()
            .copied()
            .filter(|name| *name != id_column)
            .collect();
        let other_batch = resolve_columns(reader, hits, &other).await?;
        let id_batch = resolve_ids_arithmetic(reader, hits).ok_or_else(|| {
            DataFusionError::Execution(
                "resolve_hits: hit set missing stable _id and span arithmetic \
                 (cell-packed hits must carry stable_id from the search wave)"
                    .into(),
            )
        })??;
        let mut fields = vec![id_batch.schema().field(0).as_ref().clone()];
        fields.extend(
            other_batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone()),
        );
        let mut columns = vec![Arc::clone(id_batch.column(0))];
        columns.extend(other_batch.columns().iter().map(Arc::clone));
        Some(
            RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?,
        )
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
        if let Some(id) = hit.stable_id {
            ids.push(id);
            continue;
        }
        let base = match memo.iter().find(|(uri, _)| *uri == hit.superfile) {
            Some((_, base)) => *base,
            None => {
                let entry = manifest
                    .superfiles
                    .iter()
                    .find(|e| e.uri == hit.superfile)?;
                // `None` when the span is gapped (not contiguous single-append),
                // in which case arithmetic id resolution doesn't apply.
                let base = row_id_from_manifest_entry(entry, 0)?;
                memo.push((hit.superfile, base));
                base
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

/// Fill each hit's `stable_id` in place using the cheap, cache-backed
/// resolution path: inline id / contiguous-span arithmetic where they
/// apply, else a `_id`-column read routed through `resolve_columns` (and
/// thus `decoded_scalar_cache`), so a warm reader decodes each superfile's
/// `_id` column at most once and later queries hit the cache.
///
/// This is the resolution `resolve_hits` already uses for an id-only
/// projection. The scored fan-out uses it to stamp its final top-k instead
/// of a per-superfile `take_by_local_doc_ids`, which re-runs a scattered
/// Parquet page read (~one page + decompress per hit) on *every* query and
/// dominates large-k scored latency on real corpora (it is ~free only on a
/// tiny table with few pages, which is why synthetic benches never saw it).
pub(crate) async fn stamp_stable_ids(
    reader: &SupertableReader,
    hits: &mut [SuperfileHit],
) -> DfResult<()> {
    if hits.is_empty() || hits.iter().all(|h| h.stable_id.is_some()) {
        return Ok(());
    }
    let id_batch = match resolve_ids_arithmetic(reader, hits) {
        Some(batch) => batch?,
        None => {
            let id_column = reader.options().id_column.clone();
            resolve_columns(reader, hits, &[id_column.as_str()]).await?
        }
    };
    let ids = id_batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| {
            DataFusionError::Execution("stamp_stable_ids: _id column not Decimal128".into())
        })?;
    for (hit, id) in hits.iter_mut().zip(ids.values()) {
        hit.stable_id = Some(*id);
    }
    Ok(())
}

/// Build the engine-native `_id` + `score` batch directly from already-resolved
/// stable ids and per-hit scores — the same two-column shape `resolve_hits_named`
/// returns for a `None` projection, but synthesized without opening any
/// superfile. Used by the hidden-index id-only fast path, which holds the
/// stable `_id` after the remap's id-resolution step and so needs neither the
/// user-superfile lookup nor a data-page read. `ids` and `scores` are parallel
/// and already in global rank order.
pub(crate) fn id_score_batch(
    reader: &SupertableReader,
    ids: &[i128],
    scores: &[f32],
) -> DfResult<RecordBatch> {
    let id_array = Decimal128Array::from_iter_values(ids.iter().copied())
        .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    let score_array = Float32Array::from_iter_values(scores.iter().copied());
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            reader.options().id_column.clone(),
            DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
            false,
        ),
        Field::new(SCORE_COLUMN, DataType::Float32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array) as ArrayRef,
            Arc::new(score_array) as ArrayRef,
        ],
    )
    .map_err(|e| DataFusionError::Execution(e.to_string()))
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
    let decoded_cache = reader.decoded_scalar_cache();
    let mut slots: Vec<Option<RecordBatch>> = vec![None; seg_order.len()];
    let mut misses = Vec::new();
    for (index, (&uri, locals)) in seg_order.iter().zip(&seg_locals).enumerate() {
        if let Some(batch) = decoded_cache.get(uri, locals, names) {
            slots[index] = Some(batch);
        } else {
            misses.push((index, uri));
        }
    }

    let opened = try_join_all(misses.into_iter().map(|(index, uri)| async move {
        let entry = manifest
            .lookup_superfile_entry(uri)
            .await
            .map_err(|error| DataFusionError::Execution(error.to_string()))?
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "resolve_hits: superfile {uri:?} missing from manifest"
                ))
            })?;
        superfile_reader(
            store,
            disk_cache,
            storage,
            &entry.uri,
            entry.subsection_offsets.as_ref(),
            true,
        )
        .await
        .map(|reader| (index, reader))
        .map_err(|error| DataFusionError::Execution(error.to_string()))
    }))
    .await?;

    // Materialize each superfile's projected hit rows, split by tier:
    //
    //   - **Resident readers** (in-memory tier / freshly written):
    //     `take_by_local_doc_ids` is a CPU-bound Parquet page decode
    //     over already-resident bytes, so the whole wave runs on
    //     `options.reader_pool` (rayon) — the same pool the search
    //     kernels and the writer's shard builds use — bridged back via
    //     a oneshot so no tokio worker blocks under the compute.
    //   - **Lazy readers** stream only projected hit rows through their
    //     existing `LazyByteSource`. That preserves the disk cache's block
    //     layer: cold misses range-fetch, repeated warm reads hit local blocks.
    //     Async I/O stays on the query runtime and never materializes the file.
    //
    // Both waves run concurrently and stitch back in `seg_order`
    // order. Superfile count here is bounded by the global top-k (one
    // entry per distinct hit-bearing superfile), so the fan-out is small.
    let mut warm_inputs: Vec<(usize, Arc<SuperfileReader>, Vec<u32>)> = Vec::new();
    let mut cold_units: Vec<(usize, &Arc<SuperfileReader>, &[u32])> = Vec::new();
    for (i, rd) in &opened {
        let locals = &seg_locals[*i];
        if rd.can_take_by_local_doc_ids() {
            warm_inputs.push((*i, Arc::clone(rd), locals.clone()));
        } else {
            cold_units.push((*i, rd, locals.as_slice()));
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
        let (tx, rx) = oneshot::channel();
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

    let cold_wave = try_join_all(
        cold_units
            .into_iter()
            .map(|(i, reader, locals)| async move {
                take_rows_byte_source(reader, locals, names)
                    .await
                    .map(|batch| (i, batch))
            }),
    );

    let (warm_done, cold_done) = tokio::join!(warm_wave, cold_wave);
    for (i, batch) in warm_done?.into_iter().chain(cold_done?) {
        decoded_cache.insert(seg_order[i], &seg_locals[i], names, batch.clone());
        slots[i] = Some(batch);
    }
    let per_superfile: Vec<RecordBatch> = slots
        .into_iter()
        .map(|s| s.expect("invariant: every superfile resolved by exactly one wave"))
        .collect();
    // Assemble directly into global rank order. `interleave_record_batch`
    // gathers from the per-superfile arrays in one pass; the old
    // concatenate-then-take path allocated and copied every projected column
    // twice before producing the same top-k-sized output.
    let batches: Vec<&RecordBatch> = per_superfile.iter().collect();
    interleave_record_batch(&batches, &placement)
        .map_err(|error| DataFusionError::Execution(error.to_string()))
}

/// Parquet async reader backed by the `SuperfileReader`'s existing byte source.
/// For disk-cache readers this preserves the block-cache layer instead of
/// bypassing it with a new object-store handle on every scalar projection.
struct ByteSourceAsyncReader {
    source: Source,
    metadata: Arc<ParquetMetaData>,
}

impl AsyncFileReader for ByteSourceAsyncReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        // `range_async` resolves mmap/block-resident ranges synchronously
        // (zero-copy, no I/O) via `try_get_range_sync`, and only `await`s a
        // real fetch on a cold miss — the same fast path the FTS reader's
        // posting fetches ride.
        let source = self.source.clone();
        let range = range.start as usize..range.end as usize;
        async move {
            source
                .range_async(range)
                .await
                .map_err(|error| ParquetError::General(error.to_string()))
        }
        .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
        // `get_ranges_parallel_async` serves each resident range synchronously
        // and batches only the cold misses into one `try_join_all`, returning
        // bytes in input order.
        let source = self.source.clone();
        let ranges: Vec<Range<usize>> = ranges
            .into_iter()
            .map(|range| range.start as usize..range.end as usize)
            .collect();
        async move {
            source
                .get_ranges_parallel_async(&ranges)
                .await
                .map_err(|error| ParquetError::General(error.to_string()))
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<ParquetMetaData>>> {
        let metadata = Arc::clone(&self.metadata);
        async move { Ok(metadata) }.boxed()
    }
}

/// Stream projected rows through a reader's cache-aware byte source.
pub(crate) async fn take_rows_byte_source(
    reader: &SuperfileReader,
    local_doc_ids: &[u32],
    names: &[&str],
) -> DfResult<RecordBatch> {
    let metadata = reader
        .parquet_metadata_with_page_index()
        .await
        .map_err(|error| DataFusionError::Execution(error.to_string()))?;
    let input = ByteSourceAsyncReader {
        source: Source::Lazy(reader.byte_source()),
        metadata,
    };
    take_rows_async(
        input,
        reader.schema(),
        reader.n_docs(),
        local_doc_ids,
        names,
    )
    .await
}

/// Stream the projected `names` columns at `local_doc_ids` from an object-store
/// superfile. Used when a MultiCell tombstone resolve must read `_id` pages
/// without resident parquet bytes; also driven directly by parity tests.
///
/// [`SuperfileReader::take_by_local_doc_ids`]: crate::superfile::SuperfileReader::take_by_local_doc_ids
pub(crate) async fn take_rows_object_store(
    store: Arc<dyn ObjectStore>,
    path: ObjPath,
    file_size: Option<u64>,
    file_schema: &SchemaRef,
    n_docs: u64,
    local_doc_ids: &[u32],
    names: &[&str],
) -> DfResult<RecordBatch> {
    let mut object_reader = ParquetObjectReader::new(store, path).with_preload_offset_index(true);
    if let Some(size) = file_size.filter(|&s| s > 0) {
        // Skip the size-discovery HEAD when the manifest already knows it.
        object_reader = object_reader.with_file_size(size);
    }
    take_rows_async(object_reader, file_schema, n_docs, local_doc_ids, names).await
}

async fn take_rows_async<R>(
    input: R,
    file_schema: &SchemaRef,
    n_docs: u64,
    local_doc_ids: &[u32],
    names: &[&str],
) -> DfResult<RecordBatch>
where
    R: AsyncFileReader + Unpin + Send + 'static,
{
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

    let builder = ParquetRecordBatchStreamBuilder::new(input)
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

/// Shared test support for the exec-module tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use datafusion::{
        catalog::{TableFunctionArgs, TableFunctionImpl, TableProvider},
        error::Result as DfResult,
        logical_expr::Expr,
        prelude::SessionContext,
    };

    /// Invoke a table function's `call_with_args` with a throwaway session.
    /// The search TVFs read only the argument exprs, not the session, so a
    /// fresh empty context is enough to satisfy the DataFusion 54 signature.
    pub(crate) fn call_tvf<F: TableFunctionImpl>(
        func: &F,
        exprs: &[Expr],
    ) -> DfResult<Arc<dyn TableProvider>> {
        let ctx = SessionContext::new();
        let state = ctx.state();
        func.call_with_args(TableFunctionArgs::new(exprs, &state))
    }
}

#[cfg(test)]
mod tests {
    use std::{thread::sleep, time::Duration};

    use arrow_array::{Array, FixedSizeListArray, LargeStringArray};
    use arrow_schema::Field;
    use bytes::Bytes;
    use datafusion::prelude::lit;
    use object_store::{ObjectStore, ObjectStoreExt, PutPayload, memory, path::Path as ObjPath};
    use rayon::ThreadPoolBuilder;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig},
            fts::reader::BoolMode,
            vector::{distance::Metric, layout::VectorLayout, rerank_codec::RerankCodec},
        },
        supertable::{
            Supertable, SupertableOptions,
            reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore},
        },
        test_helpers::{
            build_title_batch, decimal128_id_field, decimal128_ids, default_supertable_options,
            default_tokenizer as tok,
        },
    };

    /// Force Snowflake ids in one committed superfile across an ms boundary.
    const ID_GAP_WAIT: Duration = Duration::from_millis(20);

    #[test]
    fn arg_to_string_accepts_utf8_literal_rejects_int() {
        assert_eq!(
            arg_to_string(&lit("emb"), "column").expect("utf8 literal"),
            "emb"
        );
        assert!(arg_to_string(&lit(3_i64), "column").is_err());
    }

    #[test]
    fn search_query_df_error_maps_over_budget_to_resources_exhausted() {
        // Pins the boundary contract both search nodes depend on: a budget
        // refusal maps to ResourcesExhausted (the shape the SQL classifier
        // routes back to OverBudget), any other failure to Execution.
        assert!(matches!(
            search_query_df_error(QueryError::OverBudget("vector search, over".into())),
            DataFusionError::ResourcesExhausted(_)
        ));
        assert!(matches!(
            search_query_df_error(QueryError::Plan("boom".into())),
            DataFusionError::Execution(_)
        ));
    }

    #[test]
    fn arg_to_string_accepts_large_utf8_and_utf8_view() {
        let large = Expr::Literal(ScalarValue::LargeUtf8(Some("body".into())), None);
        assert_eq!(arg_to_string(&large, "column").expect("large utf8"), "body");
        let view = Expr::Literal(ScalarValue::Utf8View(Some("title".into())), None);
        assert_eq!(arg_to_string(&view, "column").expect("utf8 view"), "title");
    }

    #[test]
    fn arg_to_usize_accepts_int_rejects_negative_and_nonint() {
        assert_eq!(arg_to_usize(&lit(10_i64), "k").expect("int literal"), 10);
        assert!(arg_to_usize(&lit(-1_i64), "k").is_err());
        assert!(arg_to_usize(&lit("nope"), "k").is_err());
    }

    #[test]
    fn arg_to_usize_accepts_all_integer_widths() {
        let i32e = Expr::Literal(ScalarValue::Int32(Some(7)), None);
        let u64e = Expr::Literal(ScalarValue::UInt64(Some(8)), None);
        let u32e = Expr::Literal(ScalarValue::UInt32(Some(9)), None);
        assert_eq!(arg_to_usize(&i32e, "k").expect("i32"), 7);
        assert_eq!(arg_to_usize(&u64e, "k").expect("u64"), 8);
        assert_eq!(arg_to_usize(&u32e, "k").expect("u32"), 9);
    }

    #[test]
    fn output_schema_appends_score() {
        let s = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        let out = output_schema_with_score(&s);
        assert_eq!(out.fields().len(), 2);
        assert_eq!(out.field(1).name(), "score");
        assert_eq!(out.field(1).data_type(), &DataType::Float32);
    }

    // ---- harness exercising resolve_hits_named / resolve_ids_arithmetic /
    //      resolve_columns through the public search methods ----

    fn fixed_list_f32(dim: usize) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        )
    }

    fn options_title_emb(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), false),
        ]));
        SupertableOptions::new(
            schema,
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
                provided_centroids: None,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    fn build_batch(titles: &[&str], dim: usize, schema: Arc<Schema>) -> RecordBatch {
        let n = titles.len();
        let title_arr = LargeStringArray::from(titles.to_vec());
        let mut flat = Vec::<f32>::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push(if d == i % dim { 1.0 } else { 0.0 });
            }
        }
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            Arc::new(Float32Array::from(flat)) as ArrayRef,
            None,
        )
        .expect("FSL");
        RecordBatch::try_new(schema, vec![Arc::new(title_arr), Arc::new(fsl)]).expect("batch")
    }

    fn demo(dim: usize) -> Supertable {
        let st = Supertable::create(options_title_emb(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_batch(
            &["rust async", "python data", "rust systems", "go routines"],
            dim,
            schema,
        ))
        .expect("append");
        w.commit().expect("commit");
        st
    }

    #[test]
    fn resolve_hits_named_id_only_takes_arithmetic_fast_path() {
        // `_id`-only projection drives resolve_hits_named → the
        // resolve_ids_arithmetic no-I/O fast path (single contiguous
        // span: id_max - id_min + 1 == n_docs).
        let st = demo(16);
        let batches = st
            .reader()
            .bm25_search("title", "rust", 10, BoolMode::Or, Some(&["_id"]))
            .expect("bm25_search _id");
        let b = &batches[0];
        assert_eq!(b.num_columns(), 1);
        assert_eq!(b.schema().field(0).name(), "_id");
        assert_eq!(b.num_rows(), 2, "two docs contain 'rust'");
    }

    #[test]
    fn resolve_hits_named_default_is_id_and_score() {
        // `None` projection is the engine-native `_id` + `score` pair.
        let st = demo(16);
        let batches = st
            .reader()
            .bm25_search("title", "rust", 10, BoolMode::Or, None)
            .expect("bm25_search default");
        let b = &batches[0];
        assert_eq!(b.num_columns(), 2);
        assert_eq!(b.schema().field(0).name(), "_id");
        assert_eq!(b.schema().field(1).name(), "score");
    }

    #[test]
    fn resolve_hits_named_scalar_column_decodes_via_resolve_columns() {
        // Naming a scalar column (`title`) forces resolve_columns to
        // decode the column bytes; `score` synthesized alongside.
        let st = demo(16);
        let batches = st
            .reader()
            .bm25_search(
                "title",
                "rust",
                10,
                BoolMode::Or,
                Some(&["_id", "title", "score"]),
            )
            .expect("bm25_search title");
        let b = &batches[0];
        assert_eq!(b.num_columns(), 3);
        let titles = b
            .column(1)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title col");
        for i in 0..titles.len() {
            assert!(titles.value(i).contains("rust"));
        }
    }

    #[test]
    fn resolve_hits_named_unknown_column_errors() {
        let st = demo(16);
        let res = st
            .reader()
            .bm25_search("title", "rust", 10, BoolMode::Or, Some(&["nope"]));
        assert!(res.is_err(), "unknown projected column must error");
    }

    #[test]
    fn resolve_hits_named_empty_hits_returns_empty_batch() {
        let st = demo(16);
        let batches = st
            .reader()
            .bm25_search("title", "nonexistentterm", 10, BoolMode::Or, Some(&["_id"]))
            .expect("bm25_search empty");
        let total: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total, 0);
    }

    // ---- resolve_hits direct: projection branches that the named
    //      wrapper never reaches (score-only, empty projection) ----

    /// Two hits against the demo superfile, with deliberately distinct
    /// scores so the synthesized `score` column can be checked.
    fn two_hits(reader: &SupertableReader) -> Vec<SuperfileHit> {
        let entry = Arc::clone(&reader.manifest().superfiles[0]);
        vec![
            SuperfileHit {
                superfile: entry.uri,
                local_doc_id: 0,
                score: 1.5,
                stable_id: None,
            },
            SuperfileHit {
                superfile: entry.uri,
                local_doc_id: (entry.n_docs - 1) as u32,
                score: 0.5,
                stable_id: None,
            },
        ]
    }

    #[test]
    fn resolve_hits_score_only_synthesizes_score_without_decoding_scalars() {
        // Projecting just the trailing `score` index decodes no scalar
        // columns (the cost-first "open no readers" branch): `needed`
        // is empty, `score` is synthesized straight from the hits.
        let st = demo(16);
        let reader = st.reader();
        let hits = two_hits(&reader);
        let scalar_schema = reader.options().scalar_schema();
        let output_schema = output_schema_with_score(&scalar_schema);
        let score_idx = scalar_schema.fields().len();

        let batch = reader
            .block_on(resolve_hits(
                &reader,
                &hits,
                &scalar_schema,
                &output_schema,
                Some(&[score_idx]),
            ))
            .expect("score-only resolve");

        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.schema().field(0).name(), SCORE_COLUMN);
        assert_eq!(batch.num_rows(), hits.len());
        let scores = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("score col");
        assert_eq!(scores.value(0), 1.5);
        assert_eq!(scores.value(1), 0.5);
    }

    #[test]
    fn resolve_hits_none_projection_returns_all_scalar_columns_and_score() {
        // `projection: None` (distinct from `resolve_hits_named`'s
        // `_id`+`score` default) materializes every scalar column plus
        // the trailing synthesized `score`, in schema order.
        let st = demo(16);
        let reader = st.reader();
        let mut hits = two_hits(&reader);
        reader
            .block_on(
                crate::supertable::query::dispatch::attach_stable_ids_to_hits(&reader, &mut hits),
            )
            .expect("stamp stable ids before scalar resolution");
        let scalar_schema = reader.options().scalar_schema();
        let output_schema = output_schema_with_score(&scalar_schema);

        let batch = reader
            .block_on(resolve_hits(
                &reader,
                &hits,
                &scalar_schema,
                &output_schema,
                None,
            ))
            .expect("none-projection resolve");

        assert_eq!(batch.num_columns(), output_schema.fields().len());
        assert_eq!(batch.num_rows(), hits.len());
        let last = batch.num_columns() - 1;
        assert_eq!(batch.schema().field(last).name(), SCORE_COLUMN);
        assert_eq!(batch.schema().field(0).name(), "_id");
    }

    #[test]
    fn resolve_hits_empty_projection_preserves_hit_row_count() {
        // A zero-column projection (the `COUNT(*)` shape) emits no
        // columns but must still report `hits.len()` rows — the
        // `with_row_count` path.
        let st = demo(16);
        let reader = st.reader();
        let hits = two_hits(&reader);
        let scalar_schema = reader.options().scalar_schema();
        let output_schema = output_schema_with_score(&scalar_schema);

        let batch = reader
            .block_on(resolve_hits(
                &reader,
                &hits,
                &scalar_schema,
                &output_schema,
                Some(&[]),
            ))
            .expect("empty-projection resolve");

        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), hits.len());
    }

    // ---- resolve_ids_arithmetic direct: the no-I/O span arithmetic
    //      and its fallback when the span check can't apply ----

    /// FTS-only table (no vector column): commits stay id-ordered, so the
    /// span-arithmetic fast path applies. Vector commits are cell-packed
    /// (`MultiCellIvf`, Parquet reordered by cell) and are covered by
    /// [`resolve_ids_arithmetic_declines_cell_packed_vector_superfiles`].
    fn demo_fts_only() -> Supertable {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]));
        let opts = SupertableOptions::new(
            schema.clone(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool);
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        let titles = LargeStringArray::from(vec![
            "rust async",
            "python data",
            "rust systems",
            "go routines",
        ]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(titles) as ArrayRef]).expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
        st
    }

    #[test]
    fn resolve_ids_arithmetic_maps_local_ids_via_manifest_span() {
        // Single contiguous id-ordered commit => `id_max - id_min + 1 ==
        // n_docs`, so row `local` resolves to `id_min + local` with no file
        // read.
        let st = demo_fts_only();
        let reader = st.reader();
        let entry = Arc::clone(&reader.manifest().superfiles[0]);
        let last = (entry.n_docs - 1) as u32;
        let hits = vec![
            SuperfileHit {
                superfile: entry.uri,
                local_doc_id: 0,
                score: 0.0,
                stable_id: None,
            },
            SuperfileHit {
                superfile: entry.uri,
                local_doc_id: last,
                score: 0.0,
                stable_id: None,
            },
        ];

        let batch = resolve_ids_arithmetic(&reader, &hits)
            .expect("contiguous span => Some")
            .expect("ok batch");
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.schema().field(0).name(), "_id");
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal id col");
        assert_eq!(ids.value(0), entry.id_min);
        assert_eq!(ids.value(1), entry.id_min + i128::from(last));
    }

    #[test]
    fn resolve_ids_arithmetic_declines_cell_packed_vector_superfiles() {
        // Vector commits write cell-packed MultiCell superfiles whose Parquet
        // rows are cell-ordered, not id-ordered — span arithmetic must bail
        // to the id-page read even though the id span looks contiguous.
        let st = demo(16);
        let reader = st.reader();
        let entry = Arc::clone(&reader.manifest().superfiles[0]);
        assert_eq!(entry.vector_layout, VectorLayout::MultiCellIvf);
        let hits = vec![SuperfileHit {
            superfile: entry.uri,
            local_doc_id: 0,
            score: 0.0,
            stable_id: None,
        }];
        assert!(
            resolve_ids_arithmetic(&reader, &hits).is_none(),
            "cell-packed superfiles must fall back to the id-page read",
        );
    }

    #[test]
    fn resolve_ids_arithmetic_returns_none_when_superfile_absent() {
        // A hit naming a superfile not in the manifest fails the
        // lookup, so arithmetic bails to `None` and the caller falls
        // back to the id-page read.
        let st = demo(16);
        let reader = st.reader();
        let hits = vec![SuperfileHit {
            superfile: SuperfileUri::new_v4(),
            local_doc_id: 0,
            score: 0.0,
            stable_id: None,
        }];
        assert!(
            resolve_ids_arithmetic(&reader, &hits).is_none(),
            "unknown superfile must abandon the arithmetic fast path",
        );
    }

    // ---- take_rows_object_store: the cold/object-store row-resolution
    //      path, exercised directly against an in-memory object store ----

    /// Titles for the standalone superfile the cold-path tests stream
    /// rows out of. Five rows so a sub-selection genuinely skips some.
    const TITLES: [&str; 5] = ["alpha", "bravo", "charlie", "delta", "echo"];

    /// Build a standalone superfile (id + `title` scalar columns, no
    /// indexes) whose `title` rows are [`TITLES`], in row order.
    fn titled_superfile_bytes() -> Bytes {
        let schema: Arc<Schema> = Arc::new(Schema::new(vec![
            decimal128_id_field("doc_id"),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![], None);
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let ids = decimal128_ids(0..TITLES.len() as u64);
        let title = LargeStringArray::from(TITLES.to_vec());
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title)])
            .expect("build batch");
        b.add_batch(&batch, &[]).expect("add_batch");
        Bytes::from(b.finish().expect("finish builder"))
    }

    /// Open `bytes` eagerly just to read the parquet schema + row count
    /// that the cold path passes to [`take_rows_object_store`].
    fn schema_and_n_docs(bytes: &Bytes) -> (SchemaRef, u64) {
        let reader = SuperfileReader::open(bytes.clone()).expect("open");
        (Arc::clone(reader.schema()), reader.n_docs())
    }

    /// Put `bytes` into a fresh in-memory object store and return the
    /// handle + path the cold reader will range-GET against.
    async fn object_store_with(bytes: &Bytes) -> (Arc<dyn ObjectStore>, ObjPath) {
        let store: Arc<dyn ObjectStore> = Arc::new(memory::InMemory::new());
        let path = ObjPath::from("data/seg.sf.parquet");
        store
            .put(&path, PutPayload::from_bytes(bytes.clone()))
            .await
            .expect("put superfile into object store");
        (store, path)
    }

    /// Downcast the single returned `title` column to a string array.
    fn titles_of(batch: &RecordBatch) -> Vec<String> {
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title col");
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    }

    #[tokio::test]
    async fn take_rows_object_store_streams_rows_in_caller_rank_order() {
        // Out-of-order ids: rows are decoded sorted but ranked back into
        // the caller's order, so output row i is the i-th requested id.
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let batch = take_rows_object_store(
            store,
            path,
            Some(bytes.len() as u64),
            &schema,
            n_docs,
            &[2, 0, 3],
            &["title"],
        )
        .await
        .expect("take rows");

        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.schema().field(0).name(), "title");
        assert_eq!(titles_of(&batch), vec!["charlie", "alpha", "delta"]);
    }

    #[tokio::test]
    async fn take_rows_object_store_ranks_back_duplicate_ids() {
        // The same id requested twice must appear twice in the output —
        // rows are decoded once (distinct) and gathered back per request.
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let batch = take_rows_object_store(
            store,
            path,
            Some(bytes.len() as u64),
            &schema,
            n_docs,
            &[1, 1, 0],
            &["title"],
        )
        .await
        .expect("take rows");

        assert_eq!(titles_of(&batch), vec!["bravo", "bravo", "alpha"]);
    }

    #[tokio::test]
    async fn take_rows_object_store_discovers_size_when_file_size_none() {
        // `None` file_size omits the `with_file_size` shortcut, so the
        // parquet reader discovers the size itself — same rows out.
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let batch = take_rows_object_store(store, path, None, &schema, n_docs, &[4], &["title"])
            .await
            .expect("take rows");

        assert_eq!(titles_of(&batch), vec!["echo"]);
    }

    #[tokio::test]
    async fn take_rows_object_store_empty_ids_returns_empty_batch() {
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let batch = take_rows_object_store(
            store,
            path,
            Some(bytes.len() as u64),
            &schema,
            n_docs,
            &[],
            &["title"],
        )
        .await
        .expect("empty ids");

        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.schema().field(0).name(), "title");
    }

    #[tokio::test]
    async fn take_rows_object_store_out_of_range_id_errors() {
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let err = take_rows_object_store(
            store,
            path,
            Some(bytes.len() as u64),
            &schema,
            n_docs,
            &[n_docs as u32],
            &["title"],
        )
        .await
        .expect_err("doc id past n_docs must error");
        assert!(
            err.to_string().contains("out of range"),
            "expected an out-of-range error, got {err}",
        );
    }

    #[tokio::test]
    async fn take_rows_object_store_unknown_column_errors() {
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let err = take_rows_object_store(
            store,
            path,
            Some(bytes.len() as u64),
            &schema,
            n_docs,
            &[0],
            &["nope"],
        )
        .await
        .expect_err("unknown column must error");
        assert!(
            err.to_string().contains("unknown column"),
            "expected an unknown-column error, got {err}",
        );
    }

    // ---- resolve_columns cold path: lazy (object-store) readers ----
    //
    // `demo()` keeps superfile bytes resident, so its readers are warm
    // and `resolve_columns` only ever takes the rayon decode branch. To
    // drive the cold branch (lazy readers → `take_rows_object_store`),
    // commit to durable storage, then *reopen* with a lazy disk cache:
    // the reopened handle's in-memory tier is empty, so every read cold-
    // fetches a lazy reader from object storage.

    /// Commit four titled docs to `storage` via a throwaway producer.
    fn commit_titles_to(storage: &Arc<dyn StorageProvider>) {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("single writer pool"),
        );
        let producer = Supertable::create(
            default_supertable_options()
                .with_storage(Arc::clone(storage))
                .with_writer_pool(pool),
        )
        .expect("create producer");
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["rust async", "python data"]))
            .expect("first append");
        sleep(ID_GAP_WAIT);
        w.append(&build_title_batch(&["rust systems", "go routines"]))
            .expect("second append");
        w.commit().expect("commit");
        let reader = producer.reader();
        let entry = reader
            .manifest()
            .superfiles
            .first()
            .expect("one committed superfile");
        assert!(
            row_id_from_manifest_entry(entry, 0).is_none(),
            "fixture must force a gapped id span"
        );
    }

    /// Reopen the supertable at `consumer_storage` with a lazy
    /// (`LazyForegroundWithBackgroundFill`) disk cache so reads resolve to
    /// lazy range-GET readers. The reopened handle's in-memory tier is
    /// empty, so every read cold-fetches.
    fn open_cold(
        consumer_storage: Arc<dyn StorageProvider>,
        cache_dir: &TempDir,
    ) -> (Arc<DiskCacheStore>, Supertable) {
        let cfg = DiskCacheConfig {
            cache_root: cache_dir.path().to_path_buf(),
            cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
            mmap_cold_threshold_secs: 0,
            mmap_sweep_interval_secs: 0,
            ..Default::default()
        };
        let cache =
            DiskCacheStore::new_unpinned(Arc::clone(&consumer_storage), cfg).expect("cache");
        let consumer = Supertable::open(
            default_supertable_options()
                .with_storage(consumer_storage)
                .with_disk_cache(Arc::clone(&cache)),
        )
        .expect("open consumer");
        (cache, consumer)
    }

    /// Producer commits to local-FS storage; consumer reopens cold over
    /// the same storage. Returns the temp dirs (kept alive as RAII
    /// guards), the cache (for stats), and the cold consumer handle.
    fn cold_consumer() -> (TempDir, TempDir, Arc<DiskCacheStore>, Supertable) {
        let storage_dir = TempDir::new().expect("storage tempdir");
        let cache_dir = TempDir::new().expect("cache tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
        commit_titles_to(&storage);
        let (cache, consumer) = open_cold(Arc::clone(&storage), &cache_dir);
        (storage_dir, cache_dir, cache, consumer)
    }

    #[test]
    fn resolve_columns_cold_path_streams_scalar_via_object_store() {
        // Naming a non-id scalar column forces resolve_columns; the lazy
        // readers route it through take_rows_object_store (range GETs),
        // and `score` is synthesized alongside in rank order.
        let (_sd, _cd, cache, consumer) = cold_consumer();

        let batches = consumer
            .reader()
            .bm25_search("title", "rust", 10, BoolMode::Or, Some(&["title", "score"]))
            .expect("cold bm25 with scalar projection");

        let b = &batches[0];
        assert_eq!(b.num_columns(), 2);
        assert_eq!(b.schema().field(0).name(), "title");
        assert_eq!(b.schema().field(1).name(), "score");
        let titles = b
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title col");
        assert_eq!(titles.len(), 2, "two docs contain 'rust'");
        for i in 0..titles.len() {
            assert!(titles.value(i).contains("rust"));
        }
        // The reopened consumer's in-memory tier is empty, so resolving
        // the rows genuinely cold-fetched through the disk cache — this
        // is what put us on the cold branch.
        assert!(
            cache.stats().n_cold_fetches >= 1,
            "scalar resolution must cold-fetch lazy readers; got {}",
            cache.stats().n_cold_fetches,
        );
    }

    #[test]
    fn resolve_id_and_scalar_cold_path_uses_final_hit_stamps() {
        let (_sd, _cd, _cache, consumer) = cold_consumer();

        let batches = consumer
            .reader()
            .bm25_search(
                "title",
                "rust",
                10,
                BoolMode::Or,
                Some(&["_id", "title", "score"]),
            )
            .expect("cold bm25 with id and scalar projection");

        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema().field(0).name(), "_id");
        assert_eq!(batch.schema().field(1).name(), "title");
        assert_eq!(batch.schema().field(2).name(), "score");
    }

    #[test]
    fn resolve_columns_cold_path_empty_hits_opens_no_readers() {
        // A query that matches nothing produces no hits, so resolve_hits
        // short-circuits before resolve_columns ever opens a reader — no
        // cold-fetch is issued for the (absent) scalar resolution.
        let (_sd, _cd, _cache, consumer) = cold_consumer();

        let batches = consumer
            .reader()
            .bm25_search(
                "title",
                "nonexistentterm",
                10,
                BoolMode::Or,
                Some(&["title", "score"]),
            )
            .expect("cold bm25 with no matches");

        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(rows, 0);
    }
}
