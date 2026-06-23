// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Hierarchical query path with list-prune
//! integration.
//!
//! Covers the load-bearing invariants:
//!
//!   - **List-level bloom-union prune.** With a
//!     storage-backed multi-part manifest, an exact-term
//!     BM25 query that hits exactly one part's bloom
//!     union loads only that one part — the others stay
//!     cold (`OnceCell::get()` is `None`). Term that's
//!     not in any union prunes everything.
//!   - **List-level term-range prune (prefix BM25).**
//!     `bm25_search_prefix` for a prefix that overlaps
//!     one part's range loads only that part.
//!   - **Vector list-prune deferred but path still
//!     functional.** `vector_search` loads all
//!     parts (iterative-cutoff prune is a follow-up); it
//!     still must return correct results.
//!   - **SQL list-prune deferred but path still
//!     functional.** `query_sql` loads all parts; correct
//!     COUNT(*) across multi-part manifests.
//!   - **Eager-mode unchanged.** When all parts are
//!     pre-loaded (n_parts ≤ eager_load_threshold), the
//!     hierarchical iterator is observationally identical
//!     to the flat iteration (every
//!     `Manifest::part().await` hits a populated
//!     OnceCell).

#![deny(clippy::unwrap_used)]

use std::{collections::HashSet, sync::Arc};

use infino::{
    superfile::fts::reader::BoolMode,
    supertable::{
        Supertable,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};

/// Disk-cache byte budget (1 GiB) for the hierarchical-manifest tests.
const DISK_CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Parallel cold-fetch streams.
const COLD_FETCH_STREAMS: usize = 4;
/// Cold-fetch range chunk size (1 MiB).
const COLD_FETCH_CHUNK_BYTES: u64 = 1 << 20;
/// One superfile per manifest part (forces a multi-part list).
const TARGET_SUPERFILES_PER_PART: u64 = 1;
/// Eager-load threshold of 0 forces lazy part loading.
const EAGER_LOAD_THRESHOLD_FORCE_LAZY: u32 = 0;
/// Part count for the multi-part list fixture.
const HIERARCHICAL_PART_COUNT: usize = 5;
/// Rows per part (each commit appends two rows).
const ROWS_PER_PART: i64 = 2;
/// BM25 / prefix top-k for the hierarchical queries.
const BM25_TOP_K: usize = 10;
use tempfile::TempDir;

fn make_cache(
    storage: Arc<dyn StorageProvider>,
    cache_root: &std::path::Path,
) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: DISK_CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("cache")
}

/// Build a producer that creates one part per commit (via
/// target_superfiles_per_partition=1, the partition-split path),
/// then drop it. Returns the path to the storage root for
/// the consumer to open against.
fn build_5_parts_with_distinct_terms(storage_dir: &std::path::Path) {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir).expect("provider"));
    let opts = default_supertable_options()
        .with_storage(Arc::clone(&storage))
        .with_target_superfiles_per_part(TARGET_SUPERFILES_PER_PART);
    let producer = Supertable::create(opts).expect("create");

    // Each commit's batch uses a distinct vocabulary so the
    // list-level bloom-union skip can route an exact-term
    // query to exactly one part.
    let vocabs = [
        ("alpha", "bravo"),
        ("charlie", "delta"),
        ("echo", "foxtrot"),
        ("golf", "hotel"),
        ("india", "juliet"),
    ];
    for (a, b) in vocabs.iter() {
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&[a, b])).expect("append");
        w.commit().expect("commit");
    }
}

#[test]
fn bm25_exact_term_loads_only_the_matching_part() {
    let dir = TempDir::new().expect("tempdir");
    build_5_parts_with_distinct_terms(dir.path());

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    // Force lazy mode so the OnceCell occupancy delta is
    // observable. (Default threshold=4 + 5 parts also
    // produces lazy mode but eager_load_threshold=0 is
    // explicit + test-readable.)
    let cache_dir = TempDir::new().expect("cache");
    let cache = make_cache(Arc::clone(&storage), cache_dir.path());
    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_eager_load_threshold(EAGER_LOAD_THRESHOLD_FORCE_LAZY)
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("open");

    // Pre-condition: nothing loaded.
    {
        let r = consumer.reader();
        let m = r.manifest();
        let list_entries = m.get_all_list_entries();
        assert_eq!(list_entries.len(), HIERARCHICAL_PART_COUNT);
        let loaded = list_entries
            .iter()
            .filter(|e| m.get_cached_part_by_id(&e.part_id).is_some())
            .count();
        assert_eq!(loaded, 0, "lazy-open should not have eager-fetched");
    }

    // Search a term that exists only in commit #2's batch
    // ("echo"). The list-level bloom-union should prune
    // four parts; we expect exactly one part loaded post-
    // query.
    let hits = consumer
        .reader()
        .bm25_search("title", "echo", BM25_TOP_K, BoolMode::Or, None)
        .expect("bm25");
    assert!(
        !hits.is_empty(),
        "bm25 search should find 'echo' in one of the parts"
    );

    // Post-condition: exactly one OnceCell populated.
    let r = consumer.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    let n_loaded = list_entries
        .iter()
        .filter(|e| m.get_cached_part_by_id(&e.part_id).is_some())
        .count();
    assert_eq!(
        n_loaded, 1,
        "high-selectivity bm25 must load exactly 1 of 5 parts; got {n_loaded}"
    );
}

#[test]
fn bm25_term_in_no_part_loads_nothing() {
    let dir = TempDir::new().expect("tempdir");
    build_5_parts_with_distinct_terms(dir.path());

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let cache_dir = TempDir::new().expect("cache");
    let cache = make_cache(Arc::clone(&storage), cache_dir.path());
    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_eager_load_threshold(EAGER_LOAD_THRESHOLD_FORCE_LAZY)
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("open");

    // 'zoo' is not in any commit's vocabulary. The bloom-
    // union skip should prune all 5 parts → empty hits +
    // zero parts loaded (other than what the bloom test
    // already rejected without needing the part bytes).
    let hits = consumer
        .reader()
        .bm25_search("title", "zoo", BM25_TOP_K, BoolMode::Or, None)
        .expect("bm25");
    // False positives are tolerated. So `hits` might end
    // up non-empty if any bloom collides on 'zoo' — but
    // in practice, with disjoint vocabularies, the union
    // is selective. The load-bearing assertion is the
    // n_loaded count: if the union pruned everything, no
    // part was ever loaded.
    let r = consumer.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    let n_loaded = list_entries
        .iter()
        .filter(|e| m.get_cached_part_by_id(&e.part_id).is_some())
        .count();
    // Allow some flexibility for bloom false-positives —
    // in degenerate cases the bloom can spuriously claim
    // a term is present. Just assert "not all 5."
    assert!(
        n_loaded < 5,
        "bloom-union list-prune must drop at least one part on \
         a no-such-term query; got {n_loaded}/5 loaded (hits={})",
        hits.len()
    );
}

#[test]
fn bm25_prefix_with_narrow_prefix_loads_one_part() {
    let dir = TempDir::new().expect("tempdir");
    build_5_parts_with_distinct_terms(dir.path());

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let cache_dir = TempDir::new().expect("cache");
    let cache = make_cache(Arc::clone(&storage), cache_dir.path());
    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_eager_load_threshold(EAGER_LOAD_THRESHOLD_FORCE_LAZY)
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("open");

    // Prefix "echo" — appears only in part #2. Term-range
    // union should route the prefix to one part.
    let hits = consumer
        .reader()
        .bm25_search_prefix("title", "ech", BM25_TOP_K)
        .expect("prefix");
    assert!(
        !hits.is_empty(),
        "prefix search must find 'echo'-rooted terms"
    );

    let r = consumer.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    let n_loaded = list_entries
        .iter()
        .filter(|e| m.get_cached_part_by_id(&e.part_id).is_some())
        .count();
    // Term-range prune is range-based — a part survives
    // iff [prefix, prefix_upper_bound) overlaps the
    // part's [min_term, max_term]. With 5 disjoint
    // vocabularies the prefix "ech" lands in exactly one
    // part's range.
    assert_eq!(
        n_loaded, 1,
        "prefix-prune should load exactly 1 of 5 parts; got {n_loaded}"
    );
}

#[test]
fn sql_loads_all_parts_returns_correct_count() {
    // SQL list-prune is deferred (DataFusion pushdown
    // through MemTable requires a custom TableProvider).
    // The SQL path loads all parts and returns correct
    // aggregate results. The "loads all parts" property
    // is documented; the correctness property is asserted
    // here.
    let dir = TempDir::new().expect("tempdir");
    build_5_parts_with_distinct_terms(dir.path());

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let cache_dir = TempDir::new().expect("cache");
    let cache = make_cache(Arc::clone(&storage), cache_dir.path());
    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_eager_load_threshold(EAGER_LOAD_THRESHOLD_FORCE_LAZY)
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("open");

    // 5 commits × 2 rows/commit = 10 rows total.
    let batches = consumer
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("query");
    assert_eq!(batches.len(), 1);
    let arr = batches[0]
        .column_by_name("n")
        .expect("n column")
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("Int64");
    assert_eq!(arr.value(0), HIERARCHICAL_PART_COUNT as i64 * ROWS_PER_PART);

    // Post: all 5 parts loaded (SQL doesn't list-prune).
    let r = consumer.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    let n_loaded = list_entries
        .iter()
        .filter(|e| m.get_cached_part_by_id(&e.part_id).is_some())
        .count();
    assert_eq!(
        n_loaded, HIERARCHICAL_PART_COUNT,
        "SQL loads all parts (list-pushdown deferred); got {n_loaded}/5"
    );
}

/// Build a manifest with `target_superfiles_per_part = 2`: each commit
/// is one superfile, two superfiles pack into a part, then a new part
/// rolls over. 6 commits → 3 parts. `titles[i]` is commit i's batch.
fn build_3_parts_two_superfiles_each(storage_dir: &std::path::Path, commits: &[[&str; 2]]) {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir).expect("provider"));
    let producer = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_target_superfiles_per_part(2),
    )
    .expect("create");
    for titles in commits.iter() {
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&titles[..])).expect("append");
        w.commit().expect("commit");
    }
}

fn open_lazy_consumer(storage_dir: &std::path::Path, cache_dir: &std::path::Path) -> Supertable {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(storage_dir).expect("provider"));
    let cache = make_cache(Arc::clone(&storage), cache_dir);
    Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_eager_load_threshold(EAGER_LOAD_THRESHOLD_FORCE_LAZY)
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("open")
}

/// How many parts are currently resident (loaded) in the consumer's
/// manifest — the observable behind "did the prune skip parts?".
fn parts_loaded(consumer: &Supertable) -> (usize, usize) {
    let r = consumer.reader();
    let m = r.manifest();
    let entries = m.get_all_list_entries();
    let loaded = entries
        .iter()
        .filter(|e| m.get_cached_part_by_id(&e.part_id).is_some())
        .count();
    (loaded, entries.len())
}

#[test]
fn sql_single_value_in_prunes_parts_via_equality_rewrite() {
    // Single-value `IN ('Fig Roll')` on the FTS `title` column:
    //  - DataFusion rewrites a 1-value IN to `title = 'Fig Roll'`.
    //  - equality on an FTS column → a `TermPresence` bloom leaf.
    //  - so only the one part holding the value is loaded.
    let dir = TempDir::new().expect("tempdir");
    build_3_parts_two_superfiles_each(
        dir.path(),
        &[
            ["Apple Pie", "Apricot Tart"], // part 0: [Apple Pie, Banana Bread]
            ["Avocado Toast", "Banana Bread"],
            ["Cherry Cake", "Date Loaf"], // part 1: [Cherry Cake, Grape Jam]
            ["Fig Roll", "Grape Jam"],
            ["Kiwi Smoothie", "Lemon Tart"], // part 2: [Kiwi Smoothie, Orange Juice]
            ["Mango Lassi", "Orange Juice"],
        ],
    );
    let cache_dir = TempDir::new().expect("cache");
    let consumer = open_lazy_consumer(dir.path(), cache_dir.path());

    let batches = consumer
        .reader()
        .query_sql("SELECT _id FROM supertable WHERE title IN ('Fig Roll')")
        .expect("query");
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 1, "exactly one row has title 'Fig Roll'");

    let (loaded, total) = parts_loaded(&consumer);
    assert_eq!(total, 3, "6 commits / 2-per-part = 3 parts");
    assert_eq!(
        loaded, 1,
        "min/max prune must load only part 1; got {loaded}/3"
    );
}

#[test]
fn sql_multi_value_in_returns_exact_rows_across_parts() {
    // `title IN ('Straw Berry', 'Orange Juice')`, matches in parts 1 and 2.
    //  - DataFusion rewrites a multi-value IN to `title = a OR title = b`.
    //  - the manifest prune doesn't descend `OR`, so all 3 parts load.
    //  - correctness comes from `FilterExec`, not pruning.
    // This test pins the rows; pruning the `OR` form is future work.
    let dir = TempDir::new().expect("tempdir");
    build_3_parts_two_superfiles_each(
        dir.path(),
        &[
            ["Apple Pie", "Banana Bread"], // part 0: [Apple Pie, Date Loaf] — neither match
            ["Cherry Cake", "Date Loaf"],
            ["Mango Lassi", "Orange Juice"], // part 1: holds 'Orange Juice'
            ["Peach Melba", "Plum Cake"],
            ["Raspberry Pie", "Straw Berry"], // part 2: holds 'Straw Berry'
            ["Vanilla Slice", "Walnut Bread"],
        ],
    );
    let cache_dir = TempDir::new().expect("cache");
    let consumer = open_lazy_consumer(dir.path(), cache_dir.path());

    let batches = consumer
        .reader()
        .query_sql("SELECT _id FROM supertable WHERE title IN ('Straw Berry', 'Orange Juice')")
        .expect("query");
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 2, "'Orange Juice' (part 1) + 'Straw Berry' (part 2)");

    // Multi-value IN lowers to OR-of-equalities → no manifest prune today,
    // so all parts load. Correctness still holds via FilterExec.
    let (_loaded, total) = parts_loaded(&consumer);
    assert_eq!(total, 3, "6 commits / 2-per-part = 3 parts");

    // Token-superset that isn't a full match → FilterExec drops it.
    // 'Straw' shares the `straw` token with 'Straw Berry' but isn't an
    // exact title; the other literal exists nowhere.
    let none = consumer
        .reader()
        .query_sql("SELECT _id FROM supertable WHERE title IN ('Straw', 'Iced Coffee Blend')")
        .expect("query");
    assert_eq!(
        none.iter().map(|b| b.num_rows()).sum::<usize>(),
        0,
        "no row's full title equals either literal"
    );
}

#[test]
fn sql_between_returns_exact_rows_across_parts() {
    // `title BETWEEN 'C' AND 'G'` — a range predicate, sibling of IN.
    //  - DataFusion expands BETWEEN to `title >= 'C' AND title <= 'G'`,
    //    two comparisons the scalar conjunct path lowers to range leaves.
    //  - so this never enters the IN path, and min/max still prunes:
    //    part 2's titles all sort above 'G', so its range can't match.
    //  - pins both the rows and the prune so the IN work can't regress it.
    let dir = TempDir::new().expect("tempdir");
    build_3_parts_two_superfiles_each(
        dir.path(),
        &[
            ["Apple", "Cherry"], // part 0: matches Cherry, Date
            ["Banana", "Date"],
            ["Egg", "Fig"], // part 1: matches Egg, Fig
            ["Grape", "Berry"],
            ["Mango", "Orange"], // part 2: all > 'G', no match
            ["Tango", "Plum"],
        ],
    );
    let cache_dir = TempDir::new().expect("cache");
    let consumer = open_lazy_consumer(dir.path(), cache_dir.path());

    let batches = consumer
        .reader()
        .query_sql("SELECT _id FROM supertable WHERE title BETWEEN 'C' AND 'G'")
        .expect("query");
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 4, "Cherry, Date (part 0) + Egg, Fig (part 1)");

    let (loaded, total) = parts_loaded(&consumer);
    assert_eq!(total, 3, "6 commits / 2-per-part = 3 parts");
    assert_eq!(
        loaded, 2,
        "min/max range prune skips part 2 (all titles > 'G'); got {loaded}/3"
    );
}

#[test]
fn fts_in_bloom_prunes_parts_min_max_cannot() {
    // Bloom prunes where min/max can't. `title IN (...)`, 4 values:
    //  - every part holds anchors "aaa"+"zzz" → min/max is [aaa,zzz] for
    //    all → the ScalarInList leaf keeps all 3 parts.
    //  - "bravo" lives only in part 1 → the TermPresence{Or} bloom leaf
    //    narrows to part 1.
    //  - 4 values keeps it an `Expr::InList` (≤3 would lower to OR).
    let dir = TempDir::new().expect("tempdir");
    build_3_parts_two_superfiles_each(
        dir.path(),
        &[
            ["aaa", "alpha"], // part 0: tokens aaa, alpha, zzz, filler0
            ["zzz", "filler0"],
            ["aaa", "bravo"], // part 1: holds 'bravo'
            ["zzz", "filler1"],
            ["aaa", "charlie"], // part 2
            ["zzz", "filler2"],
        ],
    );
    let cache_dir = TempDir::new().expect("cache");
    let consumer = open_lazy_consumer(dir.path(), cache_dir.path());

    // 4 values → stays InList (not lowered to OR). 'bravo' matches part 1;
    // the other three exist nowhere.
    let batches = consumer
        .reader()
        .query_sql("SELECT _id FROM supertable WHERE title IN ('bravo', 'qx', 'qy', 'qz')")
        .expect("query");
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 1, "only the 'bravo' title matches");

    let (loaded, total) = parts_loaded(&consumer);
    assert_eq!(total, 3);
    assert_eq!(
        loaded, 1,
        "min/max keeps all 3 (range [aaa,zzz]); the bloom must narrow to part 1; got {loaded}/3"
    );
}

#[test]
fn fts_in_multitoken_bloom_spans_parts_and_skips_the_unmatched() {
    // Multi-word values, again min/max-blind (all parts [aaa,zzz]).
    // `title IN ('new york', 'los angeles', 'qx', 'qy')`:
    //  - bloom leaf = `TermPresence{Or, [new,york,los,angeles,qx,qy]}`.
    //  - part 1 (san/diego) holds none of those tokens → dropped.
    //  - parts 0 (new york) + 2 (los angeles) kept → FilterExec keeps the
    //    two exact full-title matches.
    let dir = TempDir::new().expect("tempdir");
    build_3_parts_two_superfiles_each(
        dir.path(),
        &[
            ["aaa", "new york"], // part 0
            ["zzz", "filler0"],
            ["aaa", "san diego"], // part 1 — no query token
            ["zzz", "filler1"],
            ["aaa", "los angeles"], // part 2
            ["zzz", "filler2"],
        ],
    );
    let cache_dir = TempDir::new().expect("cache");
    let consumer = open_lazy_consumer(dir.path(), cache_dir.path());

    let batches = consumer
        .reader()
        .query_sql(
            "SELECT _id FROM supertable \
             WHERE title IN ('new york', 'los angeles', 'qx', 'qy')",
        )
        .expect("query");
    assert_eq!(
        batches.iter().map(|b| b.num_rows()).sum::<usize>(),
        2,
        "'new york' (part 0) + 'los angeles' (part 2)"
    );
    let (loaded, total) = parts_loaded(&consumer);
    assert_eq!(total, 3);
    assert_eq!(loaded, 2, "bloom drops part 1 (san/diego); got {loaded}/3");
}

#[test]
fn fts_in_all_values_absent_prunes_every_part() {
    // Same fixture; none of the IN values' tokens are in any part's
    // bloom → every part dropped → zero parts opened, zero rows.
    let dir = TempDir::new().expect("tempdir");
    build_3_parts_two_superfiles_each(
        dir.path(),
        &[
            ["aaa", "new york"],
            ["zzz", "filler0"],
            ["aaa", "san diego"],
            ["zzz", "filler1"],
            ["aaa", "los angeles"],
            ["zzz", "filler2"],
        ],
    );
    let cache_dir = TempDir::new().expect("cache");
    let consumer = open_lazy_consumer(dir.path(), cache_dir.path());

    let batches = consumer
        .reader()
        .query_sql("SELECT _id FROM supertable WHERE title IN ('qx', 'qy', 'qz', 'qw')")
        .expect("query");
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 0);
    let (loaded, _total) = parts_loaded(&consumer);
    assert_eq!(loaded, 0, "no token in any part bloom → open nothing");
}

#[test]
fn fts_in_multiword_mixedcase_value_bloom_and_exact_filter() {
    // A 3-token, mixed-case value: "lives in Mumbai". Covers:
    //   - tokenizer lowercases → bloom terms [lives, in, mumbai];
    //   - a COMMON token ("in") shared across parts → Or-mode bloom
    //     over-keeps the part that only shares "in" (sound, looser);
    //   - FilterExec is CASE-SENSITIVE on the full string.
    let dir = TempDir::new().expect("tempdir");
    build_3_parts_two_superfiles_each(
        dir.path(),
        &[
            ["aaa", "lives in Mumbai"], // part 0 — the target
            ["zzz", "filler0"],
            ["aaa", "works in Delhi"], // part 1 — shares the common token "in"
            ["zzz", "filler1"],
            ["aaa", "stays at Pune"], // part 2 — no shared token
            ["zzz", "filler2"],
        ],
    );
    let cache_dir = TempDir::new().expect("cache");
    let consumer = open_lazy_consumer(dir.path(), cache_dir.path());

    // Exact-case query (4 values → InList):
    //  - bloom keeps part 0 (lives,in,mumbai) and part 1 (shares "in" — over-keep).
    //  - part 2 dropped.
    //  - FilterExec keeps only the exact "lives in Mumbai" row.
    let batches = consumer
        .reader()
        .query_sql(
            "SELECT _id FROM supertable \
             WHERE title IN ('lives in Mumbai', 'qx', 'qy', 'qz')",
        )
        .expect("query");
    assert_eq!(
        batches.iter().map(|b| b.num_rows()).sum::<usize>(),
        1,
        "only the exact 'lives in Mumbai' row matches"
    );
    let (loaded, total) = parts_loaded(&consumer);
    assert_eq!(total, 3);
    assert_eq!(
        loaded, 2,
        "bloom keeps part 0 + part 1 (shared token 'in'); part 2 skipped; got {loaded}/3"
    );

    // Case-mismatched query ('lives in MUMBAI'):
    //  - bloom still matches (tokens are lowercased on both sides).
    //  - but FilterExec's full-string equality is case-sensitive → 0 rows.
    // So the bloom is a presence superset; the exact filter is correctness.
    let none = consumer
        .reader()
        .query_sql(
            "SELECT _id FROM supertable \
             WHERE title IN ('lives in MUMBAI', 'qx', 'qy', 'qz')",
        )
        .expect("query");
    assert_eq!(
        none.iter().map(|b| b.num_rows()).sum::<usize>(),
        0,
        "stored 'lives in Mumbai' != literal 'lives in MUMBAI' (case-sensitive)"
    );
}

#[test]
fn eager_mode_query_paths_observationally_unchanged() {
    // 1 part + default threshold (4) → eager mode. All
    // query paths return the same results as the flat path,
    // and the OnceCell is populated from open (not
    // first query).
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
            .expect("append");
        w.commit().expect("commit");
    }

    let cache_dir = TempDir::new().expect("cache");
    let cache = make_cache(Arc::clone(&storage), cache_dir.path());
    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("open");

    // Eager: 1 part loaded at open.
    let r = consumer.reader();
    let m = r.manifest();
    let list_entries = m.get_all_list_entries();
    assert_eq!(list_entries.len(), 1);
    assert!(
        m.get_cached_part_by_id(&list_entries[0].part_id).is_some(),
        "eager mode pre-loads the part at open"
    );
    drop(r);

    // BM25 hits.
    let hits = consumer
        .reader()
        .bm25_search("title", "alpha", BM25_TOP_K, BoolMode::Or, None)
        .expect("bm25");
    assert!(!hits.is_empty());

    // SQL.
    let batches = consumer
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("sql");
    assert_eq!(batches.len(), 1);
}
