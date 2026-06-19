// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Single combined supertable ingest + search consumer for `supertable_all`.

use std::{
    sync::{Arc, OnceLock},
    time::Instant,
};

use infino::supertable::{Supertable, reader_cache::DiskCacheStore, storage::StorageProvider};
use tempfile::TempDir;

use crate::{
    ingest::supertable::{self, IngestResult, Modality},
    tiers,
};

/// Seconds-to-nanoseconds factor for recording build time for the
/// Criterion replay path.
const SEC_TO_NANOS: f64 = 1e9;

static INGEST: OnceLock<IngestResult> = OnceLock::new();
static BUILD_NS: OnceLock<f64> = OnceLock::new();

static FTS_INGEST: OnceLock<IngestResult> = OnceLock::new();
static FTS_BUILD_NS: OnceLock<f64> = OnceLock::new();

static VEC_INGEST: OnceLock<IngestResult> = OnceLock::new();
static VEC_BUILD_NS: OnceLock<f64> = OnceLock::new();

struct SearchConsumer {
    st: Supertable,
    _cache_dir: TempDir,
    _cache: Arc<DiskCacheStore>,
}
static SEARCH_CONSUMER: OnceLock<SearchConsumer> = OnceLock::new();

/// Run (or reuse) the one object-storage ingest. Used by the ingest timing group.
pub fn ensure_ingest(reason: &str) -> &'static IngestResult {
    if INGEST.get().is_none() {
        eprintln!(
            "[supertable_all] ingesting {} docs ({} commits) to object storage for {reason}...",
            supertable::n_docs(),
            supertable::n_commits()
        );
    }
    INGEST.get_or_init(|| {
        let t0 = Instant::now();
        let built = supertable::build_combined_on_storage();
        let _ = BUILD_NS.set(t0.elapsed().as_secs_f64() * SEC_TO_NANOS);
        eprintln!(
            "[supertable_all] ingest OK: {} superfiles ({})",
            built.n_superfiles, built.storage_label
        );
        built
    })
}

/// Search benches use the shared combined fixture. If an ingest group already
/// ran in this process, reuse it; otherwise build it here.
pub fn ensure_ingest_for_search(reason: &str) -> &'static IngestResult {
    if let Some(built) = INGEST.get() {
        return built;
    }
    ensure_ingest(reason)
}

pub fn ingest_build_nanos() -> f64 {
    ensure_ingest("build timing");
    *BUILD_NS.get().expect("build timing recorded")
}

pub fn ingest_recorded() -> bool {
    INGEST.get().is_some()
}

/// FTS-only supertable ingest (apples-to-apples vs Tantivy). Separate storage
/// prefix + fixture from the combined build.
pub fn ensure_fts_ingest(reason: &str) -> &'static IngestResult {
    if FTS_INGEST.get().is_none() {
        eprintln!(
            "[supertable_fts] ingesting {} docs (FTS-only) to object storage for {reason}...",
            supertable::n_docs()
        );
    }
    FTS_INGEST.get_or_init(|| {
        // Corpus generated before the timed window — engine-only timing.
        let corpus = supertable::prepare_corpus(Modality::Fts);
        let t0 = Instant::now();
        let built = supertable::build_on_storage(Modality::Fts, &corpus);
        let _ = FTS_BUILD_NS.set(t0.elapsed().as_secs_f64() * SEC_TO_NANOS);
        eprintln!(
            "[supertable_fts] ingest OK: {} superfiles ({})",
            built.n_superfiles, built.storage_label
        );
        built
    })
}

pub fn fts_ingest_build_nanos() -> f64 {
    ensure_fts_ingest("build timing");
    *FTS_BUILD_NS.get().expect("fts build timing recorded")
}

pub fn fts_ingest_recorded() -> bool {
    FTS_INGEST.get().is_some()
}

/// Vector-only supertable ingest (apples-to-apples vs Lance vector-only).
pub fn ensure_vector_ingest(reason: &str) -> &'static IngestResult {
    if VEC_INGEST.get().is_none() {
        eprintln!(
            "[supertable_vec] ingesting {} docs (vector-only) to object storage for {reason}...",
            supertable::n_docs()
        );
    }
    VEC_INGEST.get_or_init(|| {
        // Corpus generated before the timed window — engine-only timing.
        let corpus = supertable::prepare_corpus(Modality::Vector);
        let t0 = Instant::now();
        let built = supertable::build_on_storage(Modality::Vector, &corpus);
        let _ = VEC_BUILD_NS.set(t0.elapsed().as_secs_f64() * SEC_TO_NANOS);
        eprintln!(
            "[supertable_vec] ingest OK: {} superfiles ({})",
            built.n_superfiles, built.storage_label
        );
        built
    })
}

pub fn vector_ingest_build_nanos() -> f64 {
    ensure_vector_ingest("build timing");
    *VEC_BUILD_NS.get().expect("vector build timing recorded")
}

pub fn vector_ingest_recorded() -> bool {
    VEC_INGEST.get().is_some()
}

pub fn ingest() -> &'static IngestResult {
    INGEST.get().expect("ingest must run before ingest()")
}

pub fn search_table() -> &'static Supertable {
    ensure_ingest_for_search("search");
    &search_consumer().st
}

fn search_consumer() -> &'static SearchConsumer {
    SEARCH_CONSUMER.get_or_init(|| {
        let built = INGEST
            .get()
            .expect("ensure_ingest_for_search must run first");
        let (cache_dir, cache) = tiers::fresh_supertable_search_cache(
            Arc::clone(&built.storage),
            Some(built.total_index_bytes),
        );
        let opts = tiers::consumer_options(
            supertable::combined_options(None),
            Arc::clone(&built.storage),
            cache.clone(),
        );
        let st = tiers::open_consumer(opts);
        SearchConsumer {
            st,
            _cache_dir: cache_dir,
            _cache: cache,
        }
    })
}

pub fn storage() -> Arc<dyn StorageProvider> {
    Arc::clone(&ensure_ingest_for_search("storage").storage)
}

pub fn storage_label() -> &'static str {
    ensure_ingest_for_search("storage label").storage_label
}

pub fn total_index_bytes() -> u64 {
    ensure_ingest_for_search("index bytes").total_index_bytes
}
