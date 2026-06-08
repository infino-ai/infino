// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

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

use crate::corpus::{self, DIM, SequentialSyntheticCorpus};
use crate::tiers;

/// Supertable-shape document count — the supplied parameter. Default 10M
/// ([`crate::corpus::supertable_docs`]); override with
/// `INFINO_BENCH_SUPERTABLE_DOCS`.
pub fn n_docs() -> usize {
    corpus::supertable_docs()
}
/// Ingest commit chunks (not final superfile count).
pub const N_COMMIT_CHUNKS: usize = 16;
pub const TEXT_COLUMN: &str = "title";
pub const VEC_COLUMN: &str = "emb";

const CORPUS_VEC_SEED: u64 = 1;
const CORPUS_TEXT_SEED: u64 = 1;

/// Random-rotation RNG seed for the bench vector index.
const ROT_SEED: u64 = 7;
/// Writer auto-flush threshold (MiB) per segment roll.
const COMMIT_THRESHOLD_SIZE_MB: u64 = 1024;
/// Producer memory budget (8 GiB) capping resident RSS during ingest.
const WRITER_MEMORY_BUDGET_BYTES: u64 = 8 * (1u64 << 30);

/// Result of one object-storage ingest run.
pub struct IngestResult {
    pub storage: Arc<dyn StorageProvider>,
    pub storage_label: &'static str,
    pub n_superfiles: usize,
    pub total_index_bytes: u64,
    /// Real-S3 prefix this build wrote under, to delete when the run ends.
    pub cleanup: Option<tiers::S3Cleanup>,
}

/// Which index shapes a supertable build includes. Drives apples-to-apples
/// ingest comparisons: `Fts` vs Tantivy (FTS-only), `Vector` vs Lance
/// (vector-only), `Combined` vs a combined Lance table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Modality {
    Fts,
    Vector,
    Combined,
}

impl Modality {
    pub fn has_text(self) -> bool {
        matches!(self, Modality::Fts | Modality::Combined)
    }
    pub fn has_vector(self) -> bool {
        matches!(self, Modality::Vector | Modality::Combined)
    }
}

fn schema_for(modality: Modality) -> Arc<Schema> {
    let mut fields = Vec::with_capacity(2);
    if modality.has_text() {
        fields.push(Field::new(TEXT_COLUMN, DataType::LargeUtf8, false));
    }
    if modality.has_vector() {
        fields.push(Field::new(
            VEC_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ));
    }
    Arc::new(Schema::new(fields))
}

pub fn combined_schema() -> Arc<Schema> {
    schema_for(Modality::Combined)
}

pub fn options_for(
    modality: Modality,
    storage: Option<Arc<dyn StorageProvider>>,
) -> SupertableOptions {
    let n_cent_total = corpus::n_cent(n_docs());
    let n_cent_per_segment = (n_cent_total / N_COMMIT_CHUNKS).max(1);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get().max(1))
            .build()
            .expect("pool"),
    );
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    let fts = if modality.has_text() {
        vec![FtsConfig {
            column: TEXT_COLUMN.into(),
        }]
    } else {
        vec![]
    };
    let vector = if modality.has_vector() {
        vec![VectorConfig {
            column: VEC_COLUMN.into(),
            dim: DIM,
            n_cent: n_cent_per_segment,
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: infino::superfile::vector::rerank_codec::RerankCodec::Sq8Residual,
        }]
    } else {
        vec![]
    };
    let mut opts = SupertableOptions::new(schema_for(modality), fts, vector, Some(tk))
        .expect("opts")
        .with_reader_pool(pool.clone())
        .with_commit_threshold_size_mb(COMMIT_THRESHOLD_SIZE_MB)
        .with_writer_pool(pool);
    if let Some(s) = storage {
        opts = opts.with_storage(s);
    }
    opts
}

pub fn combined_options(storage: Option<Arc<dyn StorageProvider>>) -> SupertableOptions {
    options_for(Modality::Combined, storage)
}

/// Stream synthetic corpus → append → commit → object storage, building only
/// the index shapes named by `modality`. The text/vector corpus is identical
/// across modalities (same seeds), so each shape is directly comparable to its
/// single-modality competitor.
pub fn build_on_storage(modality: Modality) -> IngestResult {
    let n_docs = n_docs();
    let storage_backend = tiers::block_on(tiers::supertable_storage_fixture());
    let cleanup = storage_backend.cleanup.clone();
    let (cache_dir, cache) = tiers::fresh_disk_cache(Arc::clone(&storage_backend.storage));
    let n_cent_total = corpus::n_cent(n_docs);
    // Disk cache attached only to keep segment bytes out of the unbounded
    // in-memory store; this producer is dropped right after ingest, so skip
    // the post-commit warm-fill (pure waste + "budget exceeded" log spam).
    let opts = options_for(modality, Some(storage_backend.storage.clone()))
        .with_disk_cache(cache.clone())
        .with_memory_budget(WRITER_MEMORY_BUDGET_BYTES)
        .with_cache_prepopulation(false);
    let st = Supertable::create(opts).expect("create supertable");
    let mut w = st.writer().expect("writer");
    let chunk_size = n_docs.div_ceil(N_COMMIT_CHUNKS);
    let mut synth =
        SequentialSyntheticCorpus::new(n_cent_total, CORPUS_VEC_SEED, CORPUS_TEXT_SEED, true);
    let schema = schema_for(modality);
    let mut titles = Vec::new();
    let mut flat = Vec::new();
    for start in (0..n_docs).step_by(chunk_size) {
        let end = (start + chunk_size).min(n_docs);
        let len = end - start;
        // Generate only the columns this modality ingests so the bench
        // process never holds (and the RSS sampler never counts) a corpus
        // column the build doesn't consume.
        synth.fill_chunk_modality(
            len,
            &mut titles,
            &mut flat,
            modality.has_text(),
            modality.has_vector(),
        );
        let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(2);
        if modality.has_text() {
            let title_arr: Vec<&str> = titles.iter().map(String::as_str).collect();
            columns.push(Arc::new(LargeStringArray::from(title_arr)));
            // The arrow array now owns the only copy of the text. Drop the
            // Vec<String> heap before append/commit so the long index-build
            // phase (where the RSS sampler dwells) measures the production
            // working set — arrow batch + writer + index — not the in-process
            // synthetic-corpus generator, which a real server never holds (it
            // receives the batch over the API).
            titles.clear();
            titles.shrink_to_fit();
        }
        if modality.has_vector() {
            let item_field = Arc::new(Field::new("item", DataType::Float32, true));
            let values = Float32Array::from(std::mem::take(&mut flat));
            let fsl = FixedSizeListArray::try_new(
                item_field,
                DIM as i32,
                Arc::new(values) as Arc<dyn Array>,
                None,
            )
            .expect("FSL");
            columns.push(Arc::new(fsl));
        }
        let batch = RecordBatch::try_new(schema.clone(), columns).expect("batch");
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
        cleanup,
    }
}

/// Combined FTS + vector build (search consumer + combined ingest row).
pub fn build_combined_on_storage() -> IngestResult {
    build_on_storage(Modality::Combined)
}
