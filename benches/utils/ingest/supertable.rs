// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Combined FTS + vector supertable ingest to object storage.

use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::superfile::builder::{FtsConfig, VectorConfig};
use infino::superfile::fts::tokenize::Tokenizer;
use infino::superfile::vector::distance::Metric;
use infino::superfile::vector::rerank_codec::RerankCodec;
use infino::supertable::storage::StorageProvider;
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::default_tokenizer;

use crate::corpus::{self, DIM, MmapTextCorpus, MmapVectorCorpus};
use crate::harness::{emb_for, scatter_key, sql_options, sql_schema};
use crate::markdown::fmt_count;
use crate::tiers;

/// Supertable-shape document count — the supplied parameter. Default 10M
/// ([`crate::corpus::supertable_docs`]); override with
/// `INFINO_BENCH_SUPERTABLE_DOCS`.
pub fn n_docs() -> usize {
    corpus::supertable_docs()
}
/// Minimum ingest commit chunks (not final superfile count) — the
/// fixed shape every run ≤ 50M docs uses.
const MIN_COMMIT_CHUNKS: usize = 16;
/// Per-commit doc cap. The ingest working set (builder batch + index
/// accumulators held through finalize) scales with the commit's doc
/// count, not the table's — vector ingest measured 21.47 GiB peak at
/// 50M docs / 16 commits (3.125M docs per commit) on a 31 GiB host.
/// Capping docs-per-commit at that measured-safe size keeps ingest
/// RSS flat at any scale: larger runs commit more chunks, not bigger
/// ones.
const MAX_DOCS_PER_COMMIT: usize = 3_125_000;

/// Ingest commit count for this run's scale: the fixed 16-commit
/// shape up to 50M docs, growing past it so no commit exceeds
/// [`MAX_DOCS_PER_COMMIT`].
pub fn n_commits() -> usize {
    n_docs()
        .div_ceil(MAX_DOCS_PER_COMMIT)
        .max(MIN_COMMIT_CHUNKS)
}

/// Writer-pool thread count for ingest — the machine's logical core
/// count by default, overridable with `INFINO_BENCH_WRITERS` (same
/// knob the superfile build honors). Each commit's per-shard build
/// fans out across this pool.
pub fn n_writers() -> usize {
    corpus::parallel_writers()
}
pub const TEXT_COLUMN: &str = "title";
pub const VEC_COLUMN: &str = "emb";
pub const SQL_CATEGORY_COLUMN: &str = "category";
pub const SQL_RATING_COLUMN: &str = "rating";

const CORPUS_VEC_SEED: u64 = 1;
const CORPUS_TEXT_SEED: u64 = 1;

/// Random-rotation RNG seed for the bench vector index.
const ROT_SEED: u64 = 7;
/// Bytes in one gibibyte, for GiB-denominated memory and report values.
const GIB_BYTES: u64 = 1u64 << 30;

/// Distance metric for the bench vector index.
const BENCH_METRIC: Metric = Metric::Cosine;
/// Rerank residual codec for the bench vector index.
const BENCH_RERANK: RerankCodec = RerankCodec::Sq8ResidualEpsilon;
/// Writer auto-flush threshold (MiB) per superfile roll.
const COMMIT_THRESHOLD_SIZE_MB: u64 = 1024;
/// Producer memory budget in GiB, capping resident RSS during ingest.
const WRITER_MEMORY_BUDGET_GIB: u64 = 8;
/// Producer memory budget in bytes, derived from [`WRITER_MEMORY_BUDGET_GIB`].
const WRITER_MEMORY_BUDGET_BYTES: u64 = WRITER_MEMORY_BUDGET_GIB * GIB_BYTES;

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
    /// Path token namespacing a prepared dataset by modality.
    pub fn dataset_dir(self) -> &'static str {
        match self {
            Modality::Fts => "fts",
            Modality::Vector => "vector",
            Modality::Sql => "sql",
            Modality::Combined => "combined",
        }
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
    // SQL uses the rich SQL bench schema (via `sql_options`). The
    // consumer MUST open with the byte-identical options, or
    // `Supertable::open` rejects the table on an options-hash mismatch.
    // Route SQL here so ingest and read share one definition; the same
    // pool / commit-threshold tuning as the other modalities applies.
    if modality == Modality::Sql {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n_writers().max(1))
                .build()
                .expect("pool"),
        );
        let mut opts = sql_options(n_docs())
            .with_commit_threshold_size_mb(COMMIT_THRESHOLD_SIZE_MB)
            .with_reader_pool(Arc::clone(&pool))
            .with_writer_pool(pool);
        if let Some(s) = storage {
            opts = opts.with_storage(s);
        }
        return opts;
    }
    let n_cent_total = corpus::n_cent(n_docs());
    let n_cent_per_superfile = (n_cent_total / n_commits()).max(1);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_writers().max(1))
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
            n_cent: n_cent_per_superfile,
            rot_seed: ROT_SEED,
            metric: BENCH_METRIC,
            rerank_codec: BENCH_RERANK,
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

/// The corpus + index knobs this bench config builds with.
pub fn current_knobs(modality: Modality) -> crate::dataset::Knobs {
    crate::dataset::Knobs {
        doc_count: n_docs(),
        dim: DIM,
        n_cent_total: corpus::n_cent(n_docs()),
        vec_seed: CORPUS_VEC_SEED,
        text_seed: CORPUS_TEXT_SEED,
        rot_seed: ROT_SEED,
        metric: format!("{BENCH_METRIC:?}"),
        rerank_codec: format!("{BENCH_RERANK:?}"),
        modality: format!("{modality:?}"),
    }
}

/// Corpus artifacts for one supertable build, generated to disk and
/// mmapped **before** the measured build window. Callers create this
/// outside the RSS sampler so corpus generation is never billed to the
/// engine; the build loop then streams chunks straight off the mmap
/// (the harness never holds more than one chunk's Arrow batch on the
/// heap). Same shape for every modality: text for FTS/SQL/combined,
/// vectors for vector/combined.
pub struct PreparedCorpus {
    text: Option<MmapTextCorpus>,
    vectors: Option<MmapVectorCorpus>,
}

impl PreparedCorpus {
    /// The mmap-backed vector corpus (vector / combined builds) — also
    /// reusable for ground-truth recall measurement, so the search
    /// phase doesn't regenerate 10M×384 floats.
    pub fn vectors(&self) -> Option<&MmapVectorCorpus> {
        self.vectors.as_ref()
    }

    /// Logical size of the raw input corpus fed to ingest — text bytes
    /// plus vector f32 bytes. This is the *source* data size, distinct
    /// from the index bytes the supertable writes to object storage.
    pub fn byte_size(&self) -> u64 {
        let text = self.text.as_ref().map(|t| t.total_bytes()).unwrap_or(0);
        let vec = self
            .vectors
            .as_ref()
            .map(|v| std::mem::size_of_val(v.as_slice()) as u64)
            .unwrap_or(0);
        text + vec
    }
}

/// Generate (to disk) and mmap the corpus columns `modality` ingests.
/// Call this BEFORE starting the build RSS sampler.
pub fn prepare_corpus(modality: Modality) -> PreparedCorpus {
    let n_docs = n_docs();
    let text = modality.has_text().then(|| {
        eprintln!(
            "[supertable_ingest] generating {} -doc text corpus (mmap-backed)...",
            fmt_count(n_docs)
        );
        MmapTextCorpus::generate(n_docs, CORPUS_TEXT_SEED)
    });
    let vectors = modality.has_vector().then(|| {
        eprintln!(
            "[supertable_ingest] generating {} ×{DIM} vector corpus (mmap-backed)...",
            fmt_count(n_docs)
        );
        MmapVectorCorpus::generate(n_docs, corpus::n_cent(n_docs), CORPUS_VEC_SEED, true)
    });
    PreparedCorpus { text, vectors }
}

/// Stream the prepared on-disk corpus → append → commit → object
/// storage, building only the index shapes named by `modality`. One
/// loop for every modality — the corpus chunks are borrowed straight
/// off the mmap, and SQL's extra columns are derived inline from
/// `doc_id`. The text/vector corpus is identical across modalities
/// (same seeds), so each shape is directly comparable to its
/// single-modality competitor.
pub fn build_on_storage(modality: Modality, corpus: &PreparedCorpus) -> IngestResult {
    let n_docs = n_docs();
    let commits = n_commits();
    eprintln!(
        "[supertable_ingest] ingesting {} docs ({}) in {commits} commits to object storage...",
        fmt_count(n_docs),
        modality_label(modality),
    );
    let storage_backend = tiers::block_on(async {
        if crate::dataset::dataset_mode() {
            tiers::dataset_storage_fixture(modality.dataset_dir()).await
        } else {
            tiers::supertable_storage_fixture().await
        }
    });
    let cleanup = storage_backend.cleanup.clone();
    // Disk cache attached only to keep superfile bytes out of the unbounded
    // in-memory store; this producer is dropped right after ingest, so skip
    // the post-commit warm-fill (pure waste + "budget exceeded" log spam).
    let (cache_dir, cache) = tiers::fresh_disk_cache(Arc::clone(&storage_backend.storage));

    let opts = options_for(modality, Some(storage_backend.storage.clone()))
        .with_disk_cache(cache.clone())
        .with_memory_budget(WRITER_MEMORY_BUDGET_BYTES)
        .with_cache_prepopulation(false);
    let st = Supertable::create(opts).expect("create supertable");
    let mut w = st.writer().expect("writer");
    let chunk_size = n_docs.div_ceil(commits);
    let schema = if modality.has_sql() {
        sql_schema()
    } else {
        schema_for(modality)
    };
    let mut commit_idx = 0usize;
    for start in (0..n_docs).step_by(chunk_size) {
        commit_idx += 1;
        let end = (start + chunk_size).min(n_docs);
        let len = end - start;
        // Progress every ~4 commits (plus first + last) to keep the log
        // readable instead of one line per commit.
        if commit_idx == 1 || commit_idx == commits || commit_idx.is_multiple_of(4) {
            eprintln!(
                "[supertable_ingest] commit {commit_idx}/{commits} (docs {start}..{})...",
                end.saturating_sub(1),
            );
        }
        let batch = chunk_batch(modality, corpus, &schema, start, end, len);
        w.append(&batch).expect("append");
        w.commit().expect("commit");
        // The chunk is committed; drop its corpus pages from RSS so the
        // build sampler measures the engine, not the streamed harness
        // pages (clean file-backed pages — they'd re-fault if touched).
        if let Some(text) = &corpus.text {
            text.advise_consumed(start, len);
        }
        if let Some(vectors) = &corpus.vectors {
            vectors.advise_consumed(start, len);
        }
        // Anonymous-vs-file split per commit: a monotonic anonymous
        // climb = producer-side retention (heap); a file-backed climb
        // = freshly written cache mmaps staying resident.
        if commit_idx == 1 || commit_idx == commits || commit_idx.is_multiple_of(4) {
            crate::rss::log_rss_breakdown(&format!("ingest commit {commit_idx}"));
        }
    }
    drop(w);
    crate::rss::log_rss_breakdown("ingest writer dropped");
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
        total_index_bytes as f64 / GIB_BYTES as f64,
        storage_backend.storage_label,
    );
    // SQL query predicates sample the mid-corpus row (one mmap page
    // touch — not a corpus materialization).
    let mid = n_docs / 2;
    let (sql_sample_title, sql_sample_key) = if modality.has_sql() {
        let text = corpus.text.as_ref().expect("sql modality has text");
        (
            Some(text.doc(mid).replace('\'', "''")),
            Some(scatter_key(mid as u64)),
        )
    } else {
        (None, None)
    };
    if crate::dataset::dataset_mode() {
        write_sidecar(
            &storage_backend.storage,
            modality,
            n_superfiles,
            total_index_bytes,
            sql_sample_title.clone(),
            sql_sample_key.clone(),
        );
    }
    IngestResult {
        storage: storage_backend.storage,
        storage_label: storage_backend.storage_label,
        n_superfiles,
        total_index_bytes,
        cleanup,
        sql_sample_title,
        sql_sample_key,
    }
}

/// Write the dataset sidecar next to a freshly prepared dataset. Atomic
/// create: a pre-existing sidecar means the prefix already holds a dataset, so
/// this fails rather than clobbering it — re-prepare into a fresh prefix.
fn write_sidecar(
    storage: &Arc<dyn StorageProvider>,
    modality: Modality,
    n_superfiles: usize,
    total_index_bytes: u64,
    sql_sample_title: Option<String>,
    sql_sample_key: Option<String>,
) {
    let meta = crate::dataset::DatasetMeta {
        knobs: current_knobs(modality),
        n_superfiles,
        total_index_bytes,
        builder_id: infino::BUILDER_ID.to_string(),
        sql_sample_title,
        sql_sample_key,
    };
    let json = serde_json::to_vec_pretty(&meta).expect("serialize dataset sidecar");
    tiers::block_on(storage.put_atomic(crate::dataset::SIDECAR, Bytes::from(json)))
        .expect("write dataset sidecar");
    eprintln!(
        "[supertable_ingest] wrote {} for the {} dataset",
        crate::dataset::SIDECAR,
        modality.dataset_dir(),
    );
}

/// Open a pre-uploaded dataset for the read phases: resolve storage at the
/// fixed prefix, load and verify the sidecar, and return an [`IngestResult`]
/// the warm/cold runners consume exactly like a freshly built one — no corpus
/// generation, no ingest.
pub fn open_dataset(modality: Modality) -> IngestResult {
    let storage_backend = tiers::block_on(tiers::dataset_storage_fixture(modality.dataset_dir()));
    let meta = read_sidecar(&storage_backend.storage);
    crate::dataset::verify(&meta, &current_knobs(modality));
    eprintln!(
        "[supertable_dataset] opened {} dataset: {} superfiles, {:.2} GiB index bytes on {}",
        modality.dataset_dir(),
        meta.n_superfiles,
        meta.total_index_bytes as f64 / GIB_BYTES as f64,
        storage_backend.storage_label,
    );
    IngestResult {
        storage: storage_backend.storage,
        storage_label: storage_backend.storage_label,
        n_superfiles: meta.n_superfiles,
        total_index_bytes: meta.total_index_bytes,
        cleanup: None,
        sql_sample_title: meta.sql_sample_title,
        sql_sample_key: meta.sql_sample_key,
    }
}

/// Open an already-built supertable at `INFINO_BENCH_EXISTING_PREFIX` for the
/// read phases: no corpus, no ingest, no sidecar. A one-time manifest read
/// learns the superfile count + index bytes the warm/cold runners need to size
/// the search cache; the options the consumer reopens with must match the build
/// (same `INFINO_BENCH_SUPERTABLE_DOCS`), exactly like the dataset path.
pub(crate) fn open_existing(modality: Modality, fixture: tiers::StorageFixture) -> IngestResult {
    let (cache_dir, cache) = tiers::fresh_disk_cache(Arc::clone(&fixture.storage));
    let opts = options_for(modality, Some(Arc::clone(&fixture.storage))).with_disk_cache(cache);
    let st =
        Supertable::open(opts).expect("open existing supertable at INFINO_BENCH_EXISTING_PREFIX");
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
    drop(cache_dir);
    eprintln!(
        "[supertable_existing] opened {} supertable: {n_superfiles} superfiles, {:.2} GiB index bytes on {}",
        modality.dataset_dir(),
        total_index_bytes as f64 / GIB_BYTES as f64,
        fixture.storage_label,
    );
    IngestResult {
        storage: fixture.storage,
        storage_label: fixture.storage_label,
        n_superfiles,
        total_index_bytes,
        cleanup: None,
        sql_sample_title: None,
        sql_sample_key: None,
    }
}

/// Whether a prepared dataset (its sidecar) exists for `modality` at the
/// configured prefix.
pub fn dataset_exists(modality: Modality) -> bool {
    let fixture = tiers::block_on(tiers::dataset_storage_fixture(modality.dataset_dir()));
    tiers::block_on(fixture.storage.head(crate::dataset::SIDECAR)).is_ok()
}

fn read_sidecar(storage: &Arc<dyn StorageProvider>) -> crate::dataset::DatasetMeta {
    let (bytes, _) = tiers::block_on(storage.get(crate::dataset::SIDECAR))
        .expect("read dataset sidecar — is the dataset prepared at this prefix?");
    serde_json::from_slice(&bytes).expect("parse dataset sidecar")
}

/// One commit chunk's `RecordBatch` for `modality`, borrowing the text
/// / vector payload straight off the prepared corpus mmaps. SQL's
/// derived columns (buckets, keys, category, rating, small `emb`) are
/// pure functions of `doc_id`.
fn chunk_batch(
    modality: Modality,
    corpus: &PreparedCorpus,
    schema: &Arc<Schema>,
    start: usize,
    end: usize,
    len: usize,
) -> RecordBatch {
    let titles: Option<Vec<&str>> = corpus
        .text
        .as_ref()
        .filter(|_| modality.has_text())
        .map(|c| c.chunk_strs(start, len));

    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(schema.fields().len());
    if let Some(titles) = &titles {
        columns.push(Arc::new(LargeStringArray::from(titles.clone())));
    }
    if modality.has_sql() {
        let _ = titles.as_ref().expect("sql modality has text");
        let bucket_vals: Vec<String> = (start..end)
            .map(|doc_id| format!("b{}", doc_id % 10))
            .collect();
        let key_vals: Vec<String> = (start..end)
            .map(|doc_id| scatter_key(doc_id as u64))
            .collect();
        for vals in [&bucket_vals, &key_vals] {
            columns.push(Arc::new(LargeStringArray::from(
                vals.iter().map(String::as_str).collect::<Vec<_>>(),
            )));
        }
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
        // Small deterministic embedding column (SQL_DIM, not the
        // planted-cluster corpus) — kept for the vector/hybrid TVFs.
        let dim = emb_for(0).len();
        let mut flat = Vec::with_capacity(len * dim);
        for doc_id in start..end {
            flat.extend_from_slice(&emb_for(doc_id as u64));
        }
        columns.push(Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
                Arc::new(Float32Array::from(flat)) as Arc<dyn Array>,
                None,
            )
            .expect("sql emb FixedSizeList"),
        ));
    }
    if modality.has_vector() {
        let all = corpus
            .vectors
            .as_ref()
            .expect("vector modality has a vector corpus")
            .as_slice();
        let flat = &all[start * DIM..end * DIM];
        columns.push(Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
                Arc::new(Float32Array::from(flat.to_vec())) as Arc<dyn Array>,
                None,
            )
            .expect("FSL"),
        ));
    }
    RecordBatch::try_new(schema.clone(), columns).expect("batch")
}

/// Combined FTS + vector build (search consumer + combined ingest row).
/// Prepares the corpus inline — callers that need the corpus generated
/// outside a measurement window use `prepare_corpus` + `build_on_storage`
/// directly.
pub fn build_combined_on_storage() -> IngestResult {
    let corpus = prepare_corpus(Modality::Combined);
    build_on_storage(Modality::Combined, &corpus)
}
