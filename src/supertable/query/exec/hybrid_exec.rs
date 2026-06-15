// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Hybrid retrieval (BM25 × vector) as a DataFusion table-valued
//! function.
//!
//! `hybrid_search(text_col, q_text, vec_col, q_vec, k)` registers via
//! `register_udtf` and lowers to [`HybridSearchExec`], a custom
//! `ExecutionPlan` that runs the existing BM25 and vector kernels
//! ([`SupertableReader::bm25_search`](crate::supertable::handle::SupertableReader::bm25_search)
//! / [`vector_search`](crate::supertable::handle::SupertableReader::vector_search))
//! *concurrently* inside `execute()`, fuses the two rankings with
//! **reciprocal-rank fusion** (RRF), and resolves the fused hits once
//! to the supertable's `_id` + projected scalar columns + `score`
//! through the shared [`resolve_hits`](super::common::resolve_hits).
//!
//! No new kernel — pure composition of the two existing Exec nodes'
//! kernels feeding a fusion operator.
//!
//! ## Query shape
//!
//! ```sql
//! SELECT _id, score
//! FROM hybrid_search('body', 'error timeout', 'embedding', '0.1,0.2, ... ,0.9', 10)
//! ORDER BY score DESC
//! ```
//!
//! `text_col` / `q_text` feed BM25 (OR mode); `vec_col` / `q_vec` feed
//! vector kNN. `q_vec` is parsed exactly like `vector_search` — a
//! comma-separated string or a SQL array literal — since the vector
//! column is stripped from the SQL schema at commit and lives in the
//! embedded blob, so it can never be a scanned column.
//!
//! ## Fusion + score direction
//!
//! Each retriever returns its hits best-first (BM25 descending
//! relevance; vector ascending distance). RRF ignores the raw scores
//! and fuses on **rank only**: a hit at 0-based position `r` in a list
//! contributes `1 / (RRF_K + r + 1)`, and the per-identity
//! contributions from the two lists are summed. The emitted `score`
//! column is the fused RRF score — **higher is better**, so `ORDER BY
//! score DESC` lists the best blended matches first. Identity is
//! `(superfile, local_doc_id)`, so a document surfaced by *both*
//! retrievers is boosted above one surfaced by a single retriever.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::TaskContext;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};

use super::common::{arg_to_string, arg_to_usize, output_schema_with_score, resolve_hits};
use super::vector_exec::arg_to_query_vector;
use crate::superfile::fts::reader::BoolMode;
use crate::superfile::reader::VectorSearchOptions;
use crate::supertable::QueryError;
use crate::supertable::handle::{SupertableReader, WeakReader};
use crate::supertable::manifest::SuperfileUri;
use crate::supertable::query::SuperfileHit;

/// SQL name the TVF is registered under.
pub(crate) const HYBRID_SEARCH_UDTF: &str = "hybrid_search";

/// Argument count for
/// `hybrid_search(text_col, q_text, vec_col, q_vec, k)`.
const HYBRID_SEARCH_ARG_COUNT: usize = 5;

/// RRF rank-bias constant. `60` is the value from the original
/// Cormack et al. reciprocal-rank-fusion paper and the de-facto
/// default across engines (Elasticsearch, Weaviate, …). Larger values
/// flatten the rank weighting; it never depends on the raw scores.
const RRF_K: f32 = 60.0;

/// Register `hybrid_search(text_col, q_text, vec_col, q_vec, k)` on
/// `ctx`, bound to the query's pinned `reader` + scalar `schema`.
/// Called from
/// [`Supertable::query_sql`](crate::supertable::handle::Supertable::query_sql).
pub(crate) fn register_hybrid_search(
    ctx: &SessionContext,
    reader: Arc<SupertableReader>,
    scalar_schema: SchemaRef,
) {
    ctx.register_udtf(
        HYBRID_SEARCH_UDTF,
        Arc::new(HybridSearchFunc::new(reader, scalar_schema)),
    );
}

impl SupertableReader {
    /// Async kernel: run BM25 (`mode`) and vector kNN concurrently over
    /// this reader's pinned snapshot and fuse the two rankings with
    /// reciprocal-rank fusion, returning the top-`k` fused hits (each
    /// carrying its RRF score; higher is better). The hits are
    /// unresolved — the caller resolves `_id` / scalar columns.
    /// `pub(crate)` kernel; the public surface is the sync
    /// [`SupertableReader::hybrid_search`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn hybrid_search_async(
        &self,
        text_col: &str,
        q_text: &str,
        mode: BoolMode,
        vec_col: &str,
        q_vec: &[f32],
        options: VectorSearchOptions,
        k: usize,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        // Both retrievers run concurrently on the query runtime; each
        // inherits its own manifest skip and returns hits best-first.
        let (bm25_res, vector_res) = futures::future::join(
            self.bm25_search_async(text_col, q_text, k, mode),
            self.vector_search_async(vec_col, q_vec, k, options),
        )
        .await;
        Ok(rrf_fuse(&bm25_res?, &vector_res?, k))
    }

    /// Hybrid BM25 + vector search fused with reciprocal-rank fusion
    /// over this reader's pinned snapshot. Returns up to `k` fused hits
    /// carrying the RRF score (higher is better). Sync wrapper over
    /// [`SupertableReader::hybrid_search_async`] — the same fusion the
    /// `hybrid_search` SQL table function runs.
    #[allow(clippy::too_many_arguments)]
    pub fn hybrid_search(
        &self,
        text_col: &str,
        q_text: &str,
        mode: BoolMode,
        vec_col: &str,
        q_vec: &[f32],
        options: VectorSearchOptions,
        k: usize,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.block_on(self.hybrid_search_async(text_col, q_text, mode, vec_col, q_vec, options, k))
    }
}

/// `TableFunctionImpl` for `hybrid_search`. Holds the query's pinned
/// snapshot; `call` parses the SQL arguments and hands back a
/// per-invocation [`HybridSearchTable`].
#[derive(Debug)]
pub(crate) struct HybridSearchFunc {
    reader: WeakReader,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl HybridSearchFunc {
    pub(crate) fn new(reader: Arc<SupertableReader>, scalar_schema: SchemaRef) -> Self {
        let output_schema = output_schema_with_score(&scalar_schema);
        Self {
            reader: WeakReader::from_reader(&reader),
            scalar_schema,
            output_schema,
        }
    }
}

impl TableFunctionImpl for HybridSearchFunc {
    fn call(&self, args: &[Expr]) -> DfResult<Arc<dyn TableProvider>> {
        if args.len() != HYBRID_SEARCH_ARG_COUNT {
            return Err(DataFusionError::Plan(format!(
                "hybrid_search expects {HYBRID_SEARCH_ARG_COUNT} arguments \
                 (text_col, q_text, vec_col, q_vec, k), got {}",
                args.len()
            )));
        }
        let text_col = arg_to_string(&args[0], "hybrid_search text_col")?;
        let q_text = arg_to_string(&args[1], "hybrid_search q_text")?;
        let vec_col = arg_to_string(&args[2], "hybrid_search vec_col")?;
        let q_vec = arg_to_query_vector(&args[3])?;
        let k = arg_to_usize(&args[4], "hybrid_search k")?;
        let reader = self.reader.upgrade().ok_or_else(|| {
            DataFusionError::Execution(
                "hybrid_search: supertable consumer dropped before execution".into(),
            )
        })?;
        Ok(Arc::new(HybridSearchTable {
            reader,
            text_col,
            q_text,
            mode: BoolMode::Or,
            vec_col,
            q_vec,
            options: VectorSearchOptions::new(),
            k,
            scalar_schema: Arc::clone(&self.scalar_schema),
            output_schema: Arc::clone(&self.output_schema),
        }))
    }
}

/// One parsed `hybrid_search(...)` invocation as a `TableProvider`.
/// `scan` lowers to [`HybridSearchExec`]; the TVF's `k` is the top-k
/// bound for *each* retriever and the final fused result.
struct HybridSearchTable {
    reader: Arc<SupertableReader>,
    text_col: String,
    q_text: String,
    mode: BoolMode,
    vec_col: String,
    q_vec: Vec<f32>,
    options: VectorSearchOptions,
    k: usize,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl fmt::Debug for HybridSearchTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HybridSearchTable")
            .field("text_col", &self.text_col)
            .field("vec_col", &self.vec_col)
            .field("k", &self.k)
            .field("dim", &self.q_vec.len())
            .finish()
    }
}

#[async_trait]
impl TableProvider for HybridSearchTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output_schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let exec = HybridSearchExec::try_new(
            Arc::clone(&self.reader),
            self.text_col.clone(),
            self.q_text.clone(),
            self.mode,
            self.vec_col.clone(),
            self.q_vec.clone(),
            self.options,
            self.k,
            Arc::clone(&self.scalar_schema),
            Arc::clone(&self.output_schema),
            projection.cloned(),
        )?;
        Ok(Arc::new(exec))
    }
}

/// Custom `ExecutionPlan` that runs the BM25 + vector kernels
/// concurrently on the query runtime inside `execute()`, fuses the
/// two rankings with RRF, and emits the resolved `_id` + scalar
/// columns + fused `score`.
struct HybridSearchExec {
    reader: Arc<SupertableReader>,
    text_col: String,
    q_text: String,
    mode: BoolMode,
    vec_col: String,
    q_vec: Vec<f32>,
    options: VectorSearchOptions,
    k: usize,
    /// Scalar schema, used as the resolve projection.
    scalar_schema: SchemaRef,
    /// Full (pre-projection) output schema: scalar columns + score.
    output_schema: SchemaRef,
    /// Optional projection into `output_schema`.
    projection: Option<Vec<usize>>,
    /// Output schema after `projection`.
    projected_schema: SchemaRef,
    cache: Arc<PlanProperties>,
}

impl HybridSearchExec {
    #[allow(clippy::too_many_arguments)]
    fn try_new(
        reader: Arc<SupertableReader>,
        text_col: String,
        q_text: String,
        mode: BoolMode,
        vec_col: String,
        q_vec: Vec<f32>,
        options: VectorSearchOptions,
        k: usize,
        scalar_schema: SchemaRef,
        output_schema: SchemaRef,
        projection: Option<Vec<usize>>,
    ) -> DfResult<Self> {
        let projected_schema = match &projection {
            Some(indices) => Arc::new(
                output_schema
                    .project(indices)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?,
            ),
            None => Arc::clone(&output_schema),
        };
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&projected_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            reader,
            text_col,
            q_text,
            mode,
            vec_col,
            q_vec,
            options,
            k,
            scalar_schema,
            output_schema,
            projection,
            projected_schema,
            cache,
        })
    }

    /// Concise one-line description shared by `Debug` + `DisplayAs`.
    fn describe(&self) -> String {
        format!(
            "HybridSearchExec: text_col={}, vec_col={}, k={}, dim={}",
            self.text_col,
            self.vec_col,
            self.k,
            self.q_vec.len()
        )
    }
}

impl fmt::Debug for HybridSearchExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl DisplayAs for HybridSearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl ExecutionPlan for HybridSearchExec {
    fn name(&self) -> &'static str {
        "HybridSearchExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "HybridSearchExec has a single partition; asked for {partition}"
            )));
        }
        let reader = Arc::clone(&self.reader);
        let text_col = self.text_col.clone();
        let q_text = self.q_text.clone();
        let mode = self.mode;
        let vec_col = self.vec_col.clone();
        let q_vec = self.q_vec.clone();
        let options = self.options;
        let k = self.k;
        let scalar_schema = Arc::clone(&self.scalar_schema);
        let output_schema = Arc::clone(&self.output_schema);
        let projection = self.projection.clone();
        let projected_schema = Arc::clone(&self.projected_schema);

        let fut = async move {
            // Run both retrievers concurrently and fuse with RRF via the
            // shared kernel, then resolve the fused set once.
            let fused = reader
                .hybrid_search_async(&text_col, &q_text, mode, &vec_col, &q_vec, options, k)
                .await
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            resolve_hits(
                &reader,
                &fused,
                &scalar_schema,
                &output_schema,
                projection.as_deref(),
            )
            .await
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_schema,
            stream,
        )))
    }
}

/// Fuse two rank-ordered hit lists into the top-`k` by reciprocal-rank
/// fusion.
///
/// Each list is assumed best-first. A hit at 0-based position `r`
/// contributes `1 / (RRF_K + r + 1)` to its `(superfile, local_doc_id)`
/// identity; contributions from the two lists are summed. The result
/// is sorted by fused score descending — `(superfile, local_doc_id)` as
/// a total tie-break so the order is deterministic regardless of the
/// `HashMap`'s iteration order — and truncated to `k`. The returned
/// hits carry the fused RRF score (higher is better).
fn rrf_fuse(bm25: &[SuperfileHit], vector: &[SuperfileHit], k: usize) -> Vec<SuperfileHit> {
    let mut acc: HashMap<(SuperfileUri, u32), f32> =
        HashMap::with_capacity(bm25.len() + vector.len());
    for list in [bm25, vector] {
        for (rank, hit) in list.iter().enumerate() {
            let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
            *acc.entry((hit.superfile, hit.local_doc_id)).or_insert(0.0) += contribution;
        }
    }

    let mut fused: Vec<SuperfileHit> = acc
        .into_iter()
        .map(|((superfile, local_doc_id), score)| SuperfileHit {
            superfile,
            local_doc_id,
            score,
        })
        .collect();
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.superfile.cmp(&b.superfile))
            .then_with(|| a.local_doc_id.cmp(&b.local_doc_id))
    });
    fused.truncate(k);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;

    use arrow_array::{
        Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray,
        RecordBatch,
    };
    use arrow_schema::{DataType, Field, Schema};

    use crate::superfile::builder::{FtsConfig, VectorConfig};
    use crate::superfile::vector::distance::Metric;
    use crate::superfile::vector::rerank_codec::RerankCodec;
    use crate::supertable::{Supertable, SupertableOptions};
    use crate::test_helpers::default_tokenizer as tok;

    // ---- supertable harness: title (FTS) + emb (vector) ----

    fn fixed_list_f32(dim: usize) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        )
    }

    fn options_title_emb(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
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
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Doc `i` gets `titles[i]` and a one-hot vector at dim `i % dim`.
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

    /// Demo corpus (single superfile). `async` is unique to doc 0;
    /// `rust` is in docs 0 + 4. Doc `i`'s vector is one-hot at dim `i`.
    fn demo(dim: usize) -> Supertable {
        let st = Supertable::create(options_title_emb(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        let titles = [
            "rust async",   // 0
            "python data",  // 1
            "java spring",  // 2
            "go routines",  // 3
            "rust systems", // 4
            "ruby rails",   // 5
            "scala akka",   // 6
            "kotlin flow",  // 7
        ];
        w.append(&build_batch(&titles, dim, schema))
            .expect("append");
        w.commit().expect("commit");
        st
    }

    fn csv_one_hot(dim: usize, active: usize) -> String {
        (0..dim)
            .map(|d| if d == active { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> &'a LargeStringArray {
        let idx = batch.schema().index_of(name).expect("column present");
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("large utf8 column")
    }

    fn col_f32<'a>(batch: &'a RecordBatch, name: &str) -> &'a Float32Array {
        let idx = batch.schema().index_of(name).expect("column present");
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("f32 column")
    }

    fn id_set(batches: &[RecordBatch]) -> HashSet<i128> {
        let mut out = HashSet::new();
        for b in batches {
            let idx = b.schema().index_of("_id").expect("_id column");
            let a = b
                .column(idx)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("decimal128 _id");
            for i in 0..a.len() {
                out.insert(a.value(i));
            }
        }
        out
    }

    fn first_title(batches: &[RecordBatch]) -> String {
        let b = &batches[0];
        col_str(b, "title").value(0).to_string()
    }

    fn scores(batches: &[RecordBatch]) -> Vec<f32> {
        let mut out = Vec::new();
        for b in batches {
            let c = col_f32(b, "score");
            out.extend((0..c.len()).map(|i| c.value(i)));
        }
        out
    }

    // ---- rrf_fuse unit (deterministic fusion math) ----

    #[test]
    fn rrf_fuse_boosts_shared_hits_and_orders_by_fused_score() {
        let seg = SuperfileUri::new_v4();
        let h = |doc: u32, score: f32| SuperfileHit {
            superfile: seg,
            local_doc_id: doc,
            score,
        };
        // BM25 best-first: doc1, doc2, doc3. Vector best-first: doc2, doc4.
        let bm25 = vec![h(1, 9.0), h(2, 8.0), h(3, 7.0)];
        let vector = vec![h(2, 0.1), h(4, 0.2)];
        let fused = rrf_fuse(&bm25, &vector, 10);

        // doc2 is in both lists → highest fused score.
        let ids: Vec<u32> = fused.iter().map(|x| x.local_doc_id).collect();
        assert_eq!(ids, vec![2, 1, 4, 3], "RRF order: shared hit first");

        let s2 = 1.0 / (RRF_K + 2.0) + 1.0 / (RRF_K + 1.0); // bm25 rank2 + vec rank1
        assert!(
            (fused[0].score - s2).abs() < 1e-6,
            "doc2 fused score must sum both contributions"
        );
    }

    #[test]
    fn rrf_fuse_truncates_to_k() {
        let seg = SuperfileUri::new_v4();
        let h = |doc: u32| SuperfileHit {
            superfile: seg,
            local_doc_id: doc,
            score: 0.0,
        };
        let bm25 = vec![h(1), h(2), h(3)];
        let vector = vec![h(4), h(5)];
        let fused = rrf_fuse(&bm25, &vector, 2);
        assert_eq!(fused.len(), 2, "fused list truncates to k");
    }

    #[test]
    fn rrf_fuse_distinguishes_same_doc_id_across_superfiles() {
        // local_doc_id alone is not a global identity: the same
        // doc id in two superfiles are *different* hits.
        let seg_a = SuperfileUri::new_v4();
        let seg_b = SuperfileUri::new_v4();
        let bm25 = vec![SuperfileHit {
            superfile: seg_a,
            local_doc_id: 0,
            score: 1.0,
        }];
        let vector = vec![SuperfileHit {
            superfile: seg_b,
            local_doc_id: 0,
            score: 0.1,
        }];
        let fused = rrf_fuse(&bm25, &vector, 10);
        assert_eq!(fused.len(), 2, "distinct superfiles → distinct hits");
    }

    // ---- end-to-end through query_sql ----

    #[test]
    fn hybrid_search_identity_set_is_union_of_subsearches() {
        // With k = n, the vector retriever returns every doc, so the
        // fused identity set must equal bm25 ∪ vector exactly.
        let dim = 16;
        let st = demo(dim);
        let qv = csv_one_hot(dim, 4);
        let k = 8;

        let hybrid = id_set(
            &st.reader()
                .query_sql(&format!(
                    "SELECT _id FROM hybrid_search('title', 'rust', 'emb', '{qv}', {k})"
                ))
                .expect("hybrid query_sql"),
        );
        let bm25 = id_set(
            &st.reader()
                .query_sql(&format!(
                    "SELECT _id FROM bm25_search('title', 'rust', {k})"
                ))
                .expect("bm25 query_sql"),
        );
        let vector = id_set(
            &st.reader()
                .query_sql(&format!(
                    "SELECT _id FROM vector_search('emb', '{qv}', {k})"
                ))
                .expect("vector query_sql"),
        );

        let expected: HashSet<i128> = bm25.union(&vector).copied().collect();
        assert_eq!(hybrid, expected, "hybrid identity set = bm25 ∪ vector");
    }

    #[test]
    fn hybrid_search_ranks_doc_top_in_both_retrievers_first() {
        // `async` is unique to doc 0 (BM25 rank 1) and the one-hot
        // query at dim 0 makes doc 0 the exact vector match (rank 1).
        // Top in both → highest RRF → emitted first.
        let dim = 16;
        let st = demo(dim);
        let res = st
            .reader()
            .query_sql(&format!(
                "SELECT title, score FROM hybrid_search('title', 'async', 'emb', '{}', 8)",
                csv_one_hot(dim, 0)
            ))
            .expect("query_sql");
        assert_eq!(first_title(&res), "rust async", "doc top in both ranks #1");

        // RRF score is descending (higher is better).
        let s = scores(&res);
        for w in s.windows(2) {
            assert!(w[0] >= w[1], "fused scores must be descending: {s:?}");
        }
    }

    #[test]
    fn hybrid_search_text_only_match_survives_fusion() {
        // The vector query targets dim 7 (doc 7), but `async` is only
        // in doc 0. Doc 0 must still appear — fusion unions the two
        // candidate sets, it does not intersect them.
        let dim = 16;
        let st = demo(dim);
        let res = st
            .reader()
            .query_sql(&format!(
                "SELECT title FROM hybrid_search('title', 'async', 'emb', '{}', 8)",
                csv_one_hot(dim, 7)
            ))
            .expect("query_sql");
        let titles: HashSet<String> = res
            .iter()
            .flat_map(|b| {
                let c = col_str(b, "title");
                (0..c.len())
                    .map(|i| c.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(
            titles.contains("rust async"),
            "text-only match must survive fusion; got {titles:?}"
        );
    }

    #[test]
    fn hybrid_search_star_projection_appends_score_column() {
        let dim = 16;
        let st = demo(dim);
        let batches = st
            .reader()
            .query_sql(&format!(
                "SELECT * FROM hybrid_search('title', 'rust', 'emb', '{}', 3)",
                csv_one_hot(dim, 0)
            ))
            .expect("query_sql");
        let b = &batches[0];
        // Scalar schema (_id, title) + score.
        assert_eq!(b.num_columns(), 3);
        assert_eq!(b.schema().field(0).name(), "_id");
        assert_eq!(b.schema().field(1).name(), "title");
        assert_eq!(b.schema().field(2).name(), "score");
    }

    #[test]
    fn hybrid_search_empty_supertable_returns_no_rows() {
        let dim = 16;
        let st = Supertable::create(options_title_emb(dim)).expect("create");
        let batches = st
            .reader()
            .query_sql(&format!(
                "SELECT _id, score FROM hybrid_search('title', 'rust', 'emb', '{}', 5)",
                csv_one_hot(dim, 0)
            ))
            .expect("query_sql");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn hybrid_search_arity_error() {
        let dim = 16;
        let st = demo(dim);
        // 4 args (missing k) → planning error.
        assert!(
            st.reader()
                .query_sql("SELECT _id FROM hybrid_search('title', 'rust', 'emb', '1,0')")
                .is_err()
        );
    }

    // ---- sql × vector × fts composed in ONE query ----
    //
    // The TVFs above each test one retriever in isolation. This block
    // exercises the composition the *pushdown contract* promises
    // but nothing else covered: a single SQL statement that JOINs an
    // FTS retriever (`bm25_search`) and a vector retriever
    // (`vector_search`) on the durable `_id` and post-filters with a
    // scalar `WHERE` — across two superfiles.

    /// Schema `[category (scalar), title (FTS), emb (vector)]`. Unlike
    /// `options_title_emb`, `category` is a plain scalar column (not in
    /// any FtsConfig/VectorConfig), so it is filterable from SQL.
    fn options_cat_title_emb(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("category", DataType::LargeUtf8, false),
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), false),
        ]));
        SupertableOptions::new(
            schema,
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Doc `i` (0-based within the batch) gets `cats[i]`, `titles[i]`,
    /// and a one-hot embedding at *global* dim `base_dim + i`.
    fn build_batch_cat(
        cats: &[&str],
        titles: &[&str],
        base_dim: usize,
        dim: usize,
        schema: Arc<Schema>,
    ) -> RecordBatch {
        let n = titles.len();
        let cat_arr = LargeStringArray::from(cats.to_vec());
        let title_arr = LargeStringArray::from(titles.to_vec());
        let mut flat = Vec::<f32>::with_capacity(n * dim);
        for i in 0..n {
            let active = base_dim + i;
            for d in 0..dim {
                flat.push(if d == active { 1.0 } else { 0.0 });
            }
        }
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            Arc::new(Float32Array::from(flat)) as ArrayRef,
            None,
        )
        .expect("FSL");
        RecordBatch::try_new(
            schema,
            vec![Arc::new(cat_arr), Arc::new(title_arr), Arc::new(fsl)],
        )
        .expect("batch")
    }

    /// Two-superfile corpus (docs 0–3, then 4–7) engineered so the three
    /// retrievers each drop a *different* doc. With dim = 8 and a graded
    /// query vector `[8,7,…,1]`, the cosine distance to one-hot doc `i`
    /// is strictly increasing in `i`, so `vector_search(k=5)` is exactly
    /// {0,1,2,3,4} (deterministic, no ties). Memberships:
    ///   - `rust` (FTS)              = {0,2,3,5,6}
    ///   - vector top-5              = {0,1,2,3,4}
    ///   - `category='systems'`      = {0,1,3,5,7}
    ///   - three-way intersection    = {0,3}
    /// Sole-reason witnesses: doc1 dropped only by FTS, doc2 only by the
    /// scalar filter, doc5 only by the vector cutoff.
    fn demo_cat_two_superfiles(dim: usize) -> Supertable {
        let st = Supertable::create(options_cat_title_emb(dim)).expect("create");
        let schema = st.options().schema.clone();
        // Each writer holds the single-writer lock until dropped, so
        // scope them: seg1's writer must drop before seg2's opens.
        {
            let mut w = st.writer().expect("writer seg1");
            w.append(&build_batch_cat(
                &["systems", "systems", "cooking", "systems"],
                &["rust alpha", "python beta", "rust gamma", "rust delta"],
                0,
                dim,
                schema.clone(),
            ))
            .expect("append seg1");
            w.commit().expect("commit seg1");
        }
        {
            let mut w = st.writer().expect("writer seg2");
            w.append(&build_batch_cat(
                &["cooking", "systems", "cooking", "systems"],
                &["python epsilon", "rust zeta", "rust eta", "python theta"],
                4,
                dim,
                schema,
            ))
            .expect("append seg2");
            w.commit().expect("commit seg2");
        }
        st
    }

    #[test]
    fn sql_join_of_bm25_and_vector_with_scalar_filter_matches_three_way_intersection() {
        let dim = 16;
        let st = demo_cat_two_superfiles(dim);
        // Graded query vector [dim, dim-1, …, 1] → cosine distance to
        // one-hot doc `i` is strictly increasing in `i`, so the vector
        // distance rank equals the doc id (no ties among the 8 docs).
        let qv: String = (0..dim)
            .map(|d| (dim - d).to_string())
            .collect::<Vec<_>>()
            .join(",");

        // The three single-retriever result sets — the oracle inputs.
        let fts = id_set(
            &st.reader()
                .query_sql("SELECT _id FROM bm25_search('title', 'rust', 8)")
                .expect("bm25 query_sql"),
        );
        let vector = id_set(
            &st.reader()
                .query_sql(&format!("SELECT _id FROM vector_search('emb', '{qv}', 5)"))
                .expect("vector query_sql"),
        );
        let scalar = id_set(
            &st.reader()
                .query_sql("SELECT _id FROM supertable WHERE category = 'systems'")
                .expect("scalar query_sql"),
        );
        // Guard against corpus drift: the witnesses below only hold for
        // these exact membership sizes.
        assert_eq!(fts.len(), 5, "'rust' should match 5 titles");
        assert_eq!(vector.len(), 5, "vector top-5");
        assert_eq!(scalar.len(), 5, "5 'systems' docs");

        // THE query under test: FTS ⋈ vector on the durable _id, then a
        // scalar SQL predicate — sql + vector + fts in one statement,
        // spanning two superfiles.
        let combined_batches = st
            .reader()
            .query_sql(&format!(
                "SELECT b._id, b.title AS title, b.category AS category, b.score AS score \
                 FROM bm25_search('title', 'rust', 8) AS b \
                 JOIN vector_search('emb', '{qv}', 5) AS v ON b._id = v._id \
                 WHERE b.category = 'systems' \
                 ORDER BY b.score DESC"
            ))
            .expect("combined sql+vector+fts query");
        let combined = id_set(&combined_batches);

        // 1. Exact correctness against the three-way intersection oracle.
        let fts_vec: HashSet<i128> = fts.intersection(&vector).copied().collect();
        let oracle: HashSet<i128> = fts_vec.intersection(&scalar).copied().collect();
        assert_eq!(
            combined, oracle,
            "combined query must equal fts ∩ vector ∩ scalar"
        );
        assert_eq!(combined.len(), 2, "intersection is exactly two docs");

        // 2. Each retriever is individually load-bearing: there is a doc
        //    dropped *only* because of it (kept by the other two). If any
        //    operator were silently a no-op, one of these would fail.
        let inter = |x: &HashSet<i128>, y: &HashSet<i128>| -> HashSet<i128> {
            x.intersection(y).copied().collect()
        };
        assert!(
            !inter(&vector, &scalar).is_subset(&fts),
            "FTS must be the sole reason ≥1 doc (kept by vector∧scalar) is dropped"
        );
        assert!(
            !inter(&fts, &scalar).is_subset(&vector),
            "vector cutoff must be the sole reason ≥1 doc (kept by fts∧scalar) is dropped"
        );
        assert!(
            !inter(&fts, &vector).is_subset(&scalar),
            "scalar WHERE must be the sole reason ≥1 doc (kept by fts∧vector) is dropped"
        );

        // 3. Every surviving row actually satisfies all three constraints.
        assert!(
            combined.is_subset(&vector),
            "every combined hit is within the vector top-k"
        );
        for b in &combined_batches {
            let cats = col_str(b, "category");
            let titles = col_str(b, "title");
            for i in 0..b.num_rows() {
                assert_eq!(cats.value(i), "systems", "scalar predicate holds on output");
                assert!(
                    titles.value(i).contains("rust"),
                    "FTS predicate holds on output row: {}",
                    titles.value(i)
                );
            }
        }

        // 4. ORDER BY b.score DESC honored (BM25 score: higher = better).
        let mut scores = Vec::new();
        for b in &combined_batches {
            let s = col_f32(b, "score");
            scores.extend((0..s.len()).map(|i| s.value(i)));
        }
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1],
                "combined scores must be descending: {scores:?}"
            );
        }
    }
}
