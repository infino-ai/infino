// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Combined FTS + vector supertable ingest to object storage.

use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use infino::superfile::builder::{FtsConfig, VectorConfig};
use infino::superfile::fts::tokenize::Tokenizer;
use infino::superfile::vector::distance::Metric;
use infino::supertable::storage::StorageProvider;
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::default_tokenizer;

use crate::corpus::{self, DIM, MmapTextCorpus, SequentialSyntheticCorpus};
use crate::harness::{emb_for, scatter_key, sql_options, sql_schema};
use crate::markdown::fmt_count;
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
pub const SQL_CATEGORY_COLUMN: &str = "category";
pub const SQL_RATING_COLUMN: &str = "rating";

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
    /// Remote prefix this build wrote under, to delete when the run ends.
    pub cleanup: Option<tiers::PrefixCleanup>,
    pub sql_sample_title: Option<String>,
    pub sql_sample_key: Option<String>,
}

/// Which index shapes a supertable build includes. Drives apples-to-apples
/// ingest comparisons: `Fts` vs Tantivy (FTS-only), `Vector` vs Lance
/// (vector-only), `Combined` vs a combined Lance table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Modality {
    Fts,
    Vector,
    Sql,
    Combined,
}

pub fn modality_label(modality: Modality) -> &'static str {
    match modality {
        Modality::Fts => "FTS-only",
        Modality::Vector => "vector-only",
        Modality::Sql => "SQL",
        Modality::Combined => "combined FTS + vector",
    }
}

impl Modality {
    pub fn has_text(self) -> bool {
        matches!(self, Modality::Fts | Modality::Sql | Modality::Combined)
    }
    pub fn has_fts(self) -> bool {
        matches!(self, Modality::Fts | Modality::Combined)
    }
    pub fn has_vector(self) -> bool {
        matches!(self, Modality::Vector | Modality::Combined)
    }
    pub fn has_sql(self) -> bool {
        matches!(self, Modality::Sql)
    }
}

fn schema_for(modality: Modality) -> Arc<Schema> {
    let mut fields = Vec::with_capacity(3);
    if modality.has_text() {
        fields.push(Field::new(TEXT_COLUMN, DataType::LargeUtf8, false));
    }
    if modality.has_sql() {
        fields.push(Field::new(SQL_CATEGORY_COLUMN, DataType::LargeUtf8, false));
        fields.push(Field::new(SQL_RATING_COLUMN, DataType::Int64, false));
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
    // SQL uses the rich SQL bench schema (built by `build_sql_on_storage`
    // via `sql_options`). The consumer MUST open with the byte-identical
    // options, or `Supertable::open` rejects the table on an options-hash
    // mismatch. Route SQL here so ingest and read share one definition.
    if modality == Modality::Sql {
        let mut opts = sql_options(n_docs());
        if let Some(s) = storage {
            opts = opts.with_storage(s);
        }
        return opts;
    }
    let n_cent_total = corpus::n_cent(n_docs());
    let n_cent_per_segment = (n_cent_total / N_COMMIT_CHUNKS).max(1);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get().max(1))
            .build()
            .expect("pool"),
    );
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    let fts = if modality.has_fts() {
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
    eprintln!(
        "[supertable_ingest] ingesting {} docs ({}) in {} commits to object storage...",
        fmt_count(n_docs),
        modality_label(modality),
        N_COMMIT_CHUNKS,
    );
    let storage_backend = tiers::block_on(tiers::supertable_storage_fixture());
    let cleanup = storage_backend.cleanup.clone();
    let (cache_dir, cache) = tiers::fresh_disk_cache(Arc::clone(&storage_backend.storage));
    let n_cent_total = corpus::n_cent(n_docs);
    // Disk cache attached only to keep segment bytes out of the unbounded
    // in-memory store; this producer is dropped right after ingest, so skip
    // the post-commit warm-fill (pure waste + "budget exceeded" log spam).
    if modality == Modality::Sql {
        return build_sql_on_storage(storage_backend, cache_dir, cache);
    }

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
    let mut commit_idx = 0usize;
    for start in (0..n_docs).step_by(chunk_size) {
        commit_idx += 1;
        let end = (start + chunk_size).min(n_docs);
        let len = end - start;
        // Progress every ~4 commits (plus first + last) to keep the log
        // readable instead of one line per commit.
        if commit_idx == 1 || commit_idx == N_COMMIT_CHUNKS || commit_idx.is_multiple_of(4) {
            eprintln!(
                "[supertable_ingest] commit {commit_idx}/{N_COMMIT_CHUNKS} (docs {start}..{})...",
                end.saturating_sub(1),
            );
        }
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
        let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(3);
        if modality.has_text() {
            let title_arr: Vec<&str> = titles.iter().map(String::as_str).collect();
            columns.push(Arc::new(LargeStringArray::from(title_arr)));
        }
        if modality.has_sql() {
            let categories = (start..end)
                .map(|doc_id| match doc_id % 4 {
                    0 => "rust",
                    1 => "python",
                    2 => "go",
                    _ => "sql",
                })
                .collect::<Vec<_>>();
            columns.push(Arc::new(LargeStringArray::from(categories)));
            let ratings = (start..end)
                .map(|doc_id| (doc_id % 100) as i64)
                .collect::<Vec<_>>();
            columns.push(Arc::new(Int64Array::from(ratings)));
        }
        if modality.has_text() {
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
    eprintln!(
        "[supertable_ingest] ingest complete: {n_superfiles} superfiles, {:.2} GiB index bytes on {}",
        total_index_bytes as f64 / (1u64 << 30) as f64,
        storage_backend.storage_label,
    );
    IngestResult {
        storage: storage_backend.storage,
        storage_label: storage_backend.storage_label,
        n_superfiles,
        total_index_bytes,
        cleanup,
        sql_sample_title: None,
        sql_sample_key: None,
    }
}

fn build_sql_on_storage(
    storage_backend: tiers::StorageFixture,
    cache_dir: tempfile::TempDir,
    cache: Arc<infino::supertable::reader_cache::DiskCacheStore>,
) -> IngestResult {
    let n_docs = n_docs();
    let corpus = MmapTextCorpus::generate(n_docs, CORPUS_TEXT_SEED);
    let mid = n_docs / 2;
    let sample_title = corpus.doc(mid).replace('\'', "''");
    let sample_key = scatter_key(mid as u64);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get().max(1))
            .build()
            .expect("pool"),
    );
    let opts = sql_options(n_docs)
        .with_storage(Arc::clone(&storage_backend.storage))
        .with_disk_cache(cache.clone())
        .with_memory_budget(WRITER_MEMORY_BUDGET_BYTES)
        .with_cache_prepopulation(false)
        .with_commit_threshold_size_mb(COMMIT_THRESHOLD_SIZE_MB)
        .with_reader_pool(Arc::clone(&pool))
        .with_writer_pool(pool);
    let st = Supertable::create(opts).expect("create sql supertable");
    let schema = sql_schema();
    let mut w = st.writer().expect("writer");
    let chunk_size = n_docs.div_ceil(N_COMMIT_CHUNKS);
    let dim = emb_for(0).len();

    for start in (0..n_docs).step_by(chunk_size) {
        let commit_idx = start / chunk_size + 1;
        let end = (start + chunk_size).min(n_docs);
        let len = end - start;
        if commit_idx == 1 || commit_idx == N_COMMIT_CHUNKS || commit_idx.is_multiple_of(4) {
            eprintln!(
                "[supertable_ingest] commit {commit_idx}/{N_COMMIT_CHUNKS} (docs {start}..{})...",
                end.saturating_sub(1),
            );
        }
        let titles = corpus.chunk_strs(start, len);
        let titles_noidx = titles.clone();
        let bucket_vals: Vec<String> = (start..end)
            .map(|doc_id| format!("b{}", doc_id % 10))
            .collect();
        let key_vals: Vec<String> = (start..end)
            .map(|doc_id| scatter_key(doc_id as u64))
            .collect();
        let categories = (start..end)
            .map(|doc_id| match doc_id % 4 {
                0 => "rust",
                1 => "python",
                2 => "go",
                _ => "sql",
            })
            .collect::<Vec<_>>();
        let ratings = (start..end)
            .map(|doc_id| (doc_id % 100) as i64)
            .collect::<Vec<_>>();
        let mut flat = Vec::with_capacity(len * dim);
        for doc_id in start..end {
            flat.extend_from_slice(&emb_for(doc_id as u64));
        }
        let emb = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            Arc::new(Float32Array::from(flat)) as Arc<dyn Array>,
            None,
        )
        .expect("sql emb FixedSizeList");
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeStringArray::from(titles)),
                Arc::new(LargeStringArray::from(titles_noidx)),
                Arc::new(LargeStringArray::from(
                    bucket_vals.iter().map(String::as_str).collect::<Vec<_>>(),
                )),
                Arc::new(LargeStringArray::from(
                    bucket_vals.iter().map(String::as_str).collect::<Vec<_>>(),
                )),
                Arc::new(LargeStringArray::from(
                    key_vals.iter().map(String::as_str).collect::<Vec<_>>(),
                )),
                Arc::new(LargeStringArray::from(
                    key_vals.iter().map(String::as_str).collect::<Vec<_>>(),
                )),
                Arc::new(LargeStringArray::from(categories)),
                Arc::new(Int64Array::from(ratings)),
                Arc::new(emb),
            ],
        )
        .expect("sql batch");
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
    eprintln!(
        "[supertable_ingest] ingest complete: {n_superfiles} superfiles, {:.2} GiB index bytes on {}",
        total_index_bytes as f64 / (1u64 << 30) as f64,
        storage_backend.storage_label,
    );
    IngestResult {
        storage: storage_backend.storage,
        storage_label: storage_backend.storage_label,
        n_superfiles,
        total_index_bytes,
        cleanup: storage_backend.cleanup,
        sql_sample_title: Some(sample_title),
        sql_sample_key: Some(sample_key),
    }
}

/// Combined FTS + vector build (search consumer + combined ingest row).
pub fn build_combined_on_storage() -> IngestResult {
    build_on_storage(Modality::Combined)
}
