//! Single combined supertable ingest + search consumer for `supertable_all`.

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use infino::supertable::reader_cache::DiskCacheStore;
use infino::supertable::storage::StorageProvider;
use infino::supertable::Supertable;
use tempfile::TempDir;

use crate::ingest::supertable::{self, IngestResult};
use crate::tiers;

static INGEST: OnceLock<IngestResult> = OnceLock::new();
static BUILD_NS: OnceLock<f64> = OnceLock::new();

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
            supertable::N_DOCS,
            supertable::N_COMMIT_CHUNKS
        );
    }
    INGEST.get_or_init(|| {
        let t0 = Instant::now();
        let built = supertable::build_combined_on_storage();
        let _ = BUILD_NS.set(t0.elapsed().as_secs_f64() * 1e9);
        eprintln!(
            "[supertable_all] ingest OK: {} superfiles ({})",
            built.n_superfiles, built.storage_label
        );
        built
    })
}

/// Search benches: reuse ingest from an earlier group in this process, or fail fast.
///
/// Avoids starting a surprise 10M ingest when you only filter to `supertable_fts_search`
/// or `supertable_vec_search`. Run ingest first in the same invocation, e.g.
/// `cargo bench --bench supertable_all -- supertable_all_build supertable_fts_search`.
///
/// Set `INFINO_SUPERTABLE_ALLOW_SEARCH_INGEST=1` to allow search-only invocations to
/// trigger ingest (old behaviour).
pub fn ensure_ingest_for_search(reason: &str) -> &'static IngestResult {
    if let Some(built) = INGEST.get() {
        return built;
    }
    if std::env::var("INFINO_SUPERTABLE_ALLOW_SEARCH_INGEST").is_ok() {
        eprintln!("[supertable_all] INFINO_SUPERTABLE_ALLOW_SEARCH_INGEST=1: building for {reason}");
        return ensure_ingest(reason);
    }
    panic!(
        "supertable not ingested in this process ({reason}).\n\
         Run ingest first in the same bench invocation:\n\
           cargo bench --bench supertable_all -- supertable_all_build supertable_fts_search\n\
         or run without filters:\n\
           cargo bench --bench supertable_all\n\
         To allow search-only runs to trigger ingest, set INFINO_SUPERTABLE_ALLOW_SEARCH_INGEST=1"
    );
}

pub fn ingest_build_nanos() -> f64 {
    ensure_ingest("build timing");
    *BUILD_NS.get().expect("build timing recorded")
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
        let built = INGEST.get().expect("ensure_ingest_for_search must run first");
        let (cache_dir, cache) = tiers::fresh_supertable_search_cache(
            Arc::clone(&built.storage),
            Some(built.total_index_bytes),
        );
        let opts = tiers::consumer_options(
            supertable::combined_options(None),
            Arc::clone(&built.storage),
            cache.clone(),
        );
        let st = tiers::block_on(tiers::open_consumer(opts));
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
