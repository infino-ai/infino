//! Combined FTS + vector supertable ingest to object storage.

use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::superfile::builder::{FtsConfig, VectorConfig};
use infino::superfile::fts::tokenize::Tokenizer;
use infino::superfile::vector::distance::Metric;
use infino::supertable::storage::StorageProvider;
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::default_tokenizer;

use crate::corpus::{self, DIM, SequentialSyntheticCorpus, SUPERTABLE_DOCS};
use crate::tiers;

pub const N_DOCS: usize = SUPERTABLE_DOCS;
/// Ingest commit chunks (not final superfile count).
pub const N_COMMIT_CHUNKS: usize = 16;
pub const TEXT_COLUMN: &str = "title";
pub const VEC_COLUMN: &str = "emb";

const CORPUS_VEC_SEED: u64 = 1;
const CORPUS_TEXT_SEED: u64 = 1;

/// Result of one object-storage ingest run.
pub struct IngestResult {
    pub storage: Arc<dyn StorageProvider>,
    pub storage_label: &'static str,
    pub n_superfiles: usize,
    pub total_index_bytes: u64,
}

pub fn combined_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(TEXT_COLUMN, DataType::LargeUtf8, false),
        Field::new(
            VEC_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]))
}

pub fn combined_options(
    storage: Option<Arc<dyn StorageProvider>>,
) -> SupertableOptions {
    let n_cent_total = corpus::n_cent(N_DOCS);
    let n_cent_per_segment = (n_cent_total / N_COMMIT_CHUNKS).max(1);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get().max(1))
            .build()
            .expect("pool"),
    );
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    let mut opts = SupertableOptions::new(
        combined_schema(),
        vec![FtsConfig {
            column: TEXT_COLUMN.into(),
        }],
        vec![VectorConfig {
            column: VEC_COLUMN.into(),
            dim: DIM,
            n_cent: n_cent_per_segment,
            rot_seed: 7,
            metric: Metric::Cosine,
            rerank_codec: infino::superfile::vector::rerank_codec::RerankCodec::Sq8Residual,
        }],
        Some(tk),
    )
    .expect("opts")
    .with_reader_pool(pool.clone())
    .with_commit_threshold_size_mb(1024)
    .with_writer_pool(pool);
    if let Some(s) = storage {
        opts = opts.with_storage(s);
    }
    opts
}

/// Stream synthetic corpus → append → commit → object storage.
pub fn build_combined_on_storage() -> IngestResult {
    let storage_backend = tiers::block_on(tiers::supertable_storage_fixture());
    let (cache_dir, cache) = tiers::fresh_disk_cache(Arc::clone(&storage_backend.storage));
    let n_cent_total = corpus::n_cent(N_DOCS);
    // Disk cache attached only to keep segment bytes out of the unbounded
    // in-memory store; this producer is dropped right after ingest, so skip
    // the post-commit warm-fill (pure waste + "budget exceeded" log spam).
    let opts = combined_options(Some(storage_backend.storage.clone()))
        .with_disk_cache(cache.clone())
        .with_memory_budget(8 * (1u64 << 30))
        .with_cache_prepopulation(false);
    let st = Supertable::create(opts);
    let mut w = st.writer().expect("writer");
    let chunk_size = N_DOCS.div_ceil(N_COMMIT_CHUNKS);
    let mut synth = SequentialSyntheticCorpus::new(
        n_cent_total,
        CORPUS_VEC_SEED,
        CORPUS_TEXT_SEED,
        true,
    );
    let schema = combined_schema();
    let mut titles = Vec::new();
    let mut flat = Vec::new();
    for start in (0..N_DOCS).step_by(chunk_size) {
        let end = (start + chunk_size).min(N_DOCS);
        let len = end - start;
        synth.fill_chunk(len, &mut titles, &mut flat);
        let title_arr: Vec<&str> = titles.iter().map(String::as_str).collect();
        let titles_col = LargeStringArray::from(title_arr);
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let values = Float32Array::from(std::mem::take(&mut flat));
        let fsl = FixedSizeListArray::try_new(
            item_field,
            DIM as i32,
            Arc::new(values) as Arc<dyn Array>,
            None,
        )
        .expect("FSL");
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(titles_col), Arc::new(fsl)])
            .expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    let reader = st.reader();
    let n_superfiles = reader.n_superfiles();
    let total_index_bytes: u64 = reader
        .manifest()
        .superfiles
        .iter()
        .filter_map(|e| e.subsection_offsets.as_ref())
        .map(|off| off.total_size)
        .sum();
    drop(reader);
    drop(st);
    drop(cache);
    drop(cache_dir);
    IngestResult {
        storage: storage_backend.storage,
        storage_label: storage_backend.storage_label,
        n_superfiles,
        total_index_bytes,
    }
}
