// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! ManifestSnapshot-level skip pruning end-to-end.
//!
//! These tests are the load-bearing perf claim of the skip
//! layer: a superfile that doesn't match a query must never
//! trigger a [`SuperfileReaderCache::reader`] call. We assert that by
//! wrapping the in-memory store in a counting decorator and
//! comparing per-URI reader-call counts taken before and after
//! the query.
//!
//! Two query shapes are exercised:
//!
//!   1. **Exact-term BM25** — `nimblefox` is planted in exactly
//!      one of N superfiles. After running `bm25_search`, only that
//!      superfile's reader has been opened. The N-1 pruned superfiles
//!      stay cold.
//!   2. **Prefix BM25** — terms beginning with `quokka` are
//!      planted in exactly one superfile (the others contain only
//!      `apple`/`banana`/etc. — no overlap with the lex range
//!      `[quokka, quokka_upper_bound)`). `bm25_search_prefix`
//!      opens only the matching superfile.
//!
//! Vector centroid skip is not asserted here — the v1
//! `vector_centroid_skip` returns all-keep (cutoff-driven skip
//! is deferred), so the test would just confirm "every superfile
//! is opened" which isn't a useful invariant. Scalar skip via
//! SQL is similarly deferred: the SQL path uses a `MemTable`
//! that opens every superfile at registration time; a future
//! custom `TableProvider` will integrate `PruningPredicate`
//! and revisit this.

#![deny(clippy::unwrap_used)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use bytes::Bytes;
use infino::{
    superfile::{SuperfileReader, builder::FtsConfig, fts::tokenize::Tokenizer},
    supertable::{
        Supertable, SupertableOptions,
        manifest::SuperfileUri,
        reader_cache::{InMemoryReaderCache, ReaderCacheError, SuperfileReaderCache},
    },
    test_helpers::{build_title_batch, default_tokenizer, schema_id_title},
};

/// Single-thread rayon pool for deterministic skip-pruning.
const RAYON_POOL_THREADS: usize = 1;
/// Four-superfile corpus for the exact-term skip tests.
const EXACT_TERM_SUPERFILE_COUNT: usize = 4;
/// BM25 / prefix top-k used across the skip-pruning queries.
const BM25_TOP_K: usize = 5;
/// Superfiles with no matching term (bloom-prune-all fixture).
const NO_MATCH_SUPERFILE_COUNT: u64 = 3;

/// Decorator over an `InMemoryReaderCache` that counts
/// per-URI `reader` calls. Wraps without behavior change.
#[derive(Default)]
struct CountingStore {
    inner: InMemoryReaderCache,
    reader_calls: Mutex<HashMap<SuperfileUri, usize>>,
}

impl CountingStore {
    fn new() -> Self {
        Self::default()
    }

    /// Snapshot of reader-call counts. Used to compute the delta
    /// across a single query.
    fn snapshot(&self) -> HashMap<SuperfileUri, usize> {
        self.reader_calls
            .lock()
            .expect("reader_calls mutex")
            .clone()
    }

    /// `after - before` for each URI. Missing keys count as 0 on
    /// either side.
    fn delta(&self, before: &HashMap<SuperfileUri, usize>) -> HashMap<SuperfileUri, usize> {
        let after = self.snapshot();
        let mut out = HashMap::new();
        for (uri, n_after) in &after {
            let n_before = before.get(uri).copied().unwrap_or(0);
            if *n_after > n_before {
                out.insert(*uri, n_after - n_before);
            }
        }
        out
    }
}

impl SuperfileReaderCache for CountingStore {
    fn reader(&self, uri: &SuperfileUri) -> Result<Arc<SuperfileReader>, ReaderCacheError> {
        *self
            .reader_calls
            .lock()
            .expect("reader_calls mutex")
            .entry(*uri)
            .or_insert(0) += 1;
        self.inner.reader(uri)
    }

    fn insert(&self, uri: SuperfileUri, bytes: Bytes) -> Result<(), ReaderCacheError> {
        self.inner.insert(uri, bytes)
    }

    fn resident_bytes(&self) -> usize {
        self.inner.resident_bytes()
    }
}

fn options_with_counting_store(store: Arc<CountingStore>) -> SupertableOptions {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("build pool"),
    );
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    SupertableOptions::new(
        schema_id_title(),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![],
        Some(tk),
    )
    .expect("opts")
    .with_writer_pool(pool)
    .with_store(store)
}

#[test]
fn bm25_exact_term_skip_opens_only_matching_superfile() {
    let store = Arc::new(CountingStore::new());
    let st = Supertable::create(options_with_counting_store(Arc::clone(&store))).expect("create");

    // Four superfiles. Plant the rare term `nimblefox` in superfile 0
    // only; the other three superfiles share generic terms only.
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&[
        "lookup nimblefox special token",
        "ordinary common everyday text",
    ]))
    .expect("append");
    w.commit().expect("commit");

    w.append(&build_title_batch(&[
        "another generic page",
        "more filler text",
    ]))
    .expect("append");
    w.commit().expect("commit");

    w.append(&build_title_batch(&[
        "yet another normal title",
        "wrapping up the corpus",
    ]))
    .expect("append");
    w.commit().expect("commit");

    w.append(&build_title_batch(&["filler bin", "extra padding"]))
        .expect("append");
    w.commit().expect("commit");
    drop(w);

    let r = st.reader();
    assert_eq!(r.n_superfiles(), EXACT_TERM_SUPERFILE_COUNT);

    // Identify the URI of superfile 0 (the planted superfile).
    let manifest = r.manifest();
    let target_uri = manifest.superfiles[0].uri;

    // Snapshot reader-call counts AFTER commits (writer publishes
    // each superfile via one reader call to derive summaries). We
    // measure the delta over the query alone.
    let before = store.snapshot();

    let hits = r
        .bm25_hits(
            "title",
            "nimblefox",
            BM25_TOP_K,
            infino::supertable::query::fts::BoolMode::Or,
        )
        .expect("query");
    assert_eq!(hits.len(), 1, "exactly one doc matches `nimblefox`");
    assert_eq!(hits[0].superfile, target_uri);

    let delta = store.delta(&before);
    assert_eq!(
        delta.len(),
        1,
        "skip should open exactly one superfile for an exact-term query \
         where 3 of 4 superfiles have the term definitively absent — got {delta:?}"
    );
    assert!(
        delta.contains_key(&target_uri),
        "the one opened superfile must be the planted one"
    );
}

#[test]
fn bm25_prefix_skip_opens_only_superfiles_overlapping_prefix_range() {
    let store = Arc::new(CountingStore::new());
    let st = Supertable::create(options_with_counting_store(Arc::clone(&store))).expect("create");

    // Four superfiles. Superfile 1 contains terms starting with
    // `quokka`; the other three contain only terms strictly
    // lex-less-than `quokka` so each superfile's lex term range
    // is fully below `[quokka, quokka_upper_bound)` and the
    // term-range skip prunes them.
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["apple bagel", "banana bread"]))
        .expect("append");
    w.commit().expect("commit");

    w.append(&build_title_batch(&["quokka cuddle", "quokkateer reviews"]))
        .expect("append");
    w.commit().expect("commit");

    w.append(&build_title_batch(&["cherry coke", "date butter"]))
        .expect("append");
    w.commit().expect("commit");

    w.append(&build_title_batch(&["edam fondue", "gouda henna"]))
        .expect("append");
    w.commit().expect("commit");
    drop(w);

    let r = st.reader();
    assert_eq!(r.n_superfiles(), EXACT_TERM_SUPERFILE_COUNT);

    let manifest = r.manifest();
    let quokka_uri = manifest.superfiles[1].uri;

    let before = store.snapshot();
    let hits = r
        .bm25_search_prefix("title", "quokka", BM25_TOP_K)
        .expect("prefix query");
    assert_eq!(hits.len(), 2, "two docs in superfile 1 begin with `quokka`");
    for h in &hits {
        assert_eq!(h.superfile, quokka_uri);
    }

    let delta = store.delta(&before);
    assert_eq!(
        delta.len(),
        1,
        "term-range skip should open exactly the one superfile whose \
         lex term range overlaps [quokka, quokka_upper_bound) — got {delta:?}"
    );
    assert!(delta.contains_key(&quokka_uri));
}

#[test]
fn bm25_search_with_no_matching_superfiles_opens_no_superfiles_at_all() {
    let store = Arc::new(CountingStore::new());
    let st = Supertable::create(options_with_counting_store(Arc::clone(&store))).expect("create");

    // Three superfiles — none contains the rare query term.
    let mut w = st.writer().expect("writer");
    for _i in 0..NO_MATCH_SUPERFILE_COUNT {
        w.append(&build_title_batch(&[
            "ordinary term filler",
            "another mundane title",
        ]))
        .expect("append");
        w.commit().expect("commit");
    }
    drop(w);

    let before = store.snapshot();
    let hits = st
        .reader()
        .bm25_hits(
            "title",
            "definitelynotpresent",
            BM25_TOP_K,
            infino::supertable::query::fts::BoolMode::Or,
        )
        .expect("query");
    assert!(hits.is_empty());

    // Bloom skip should prune all 3 superfiles — no reader calls.
    let delta = store.delta(&before);
    assert!(
        delta.is_empty(),
        "an absent rare term should prune all superfiles — got {delta:?}"
    );
}

#[test]
fn bm25_and_mode_skip_requires_all_terms_present_in_superfile() {
    let store = Arc::new(CountingStore::new());
    let st = Supertable::create(options_with_counting_store(Arc::clone(&store))).expect("create");

    // Two superfiles. Superfile 0 has both `alpha` and `beta`; superfile
    // 1 has `alpha` only. AND-mode for "alpha beta" must prune
    // superfile 1 (missing `beta`) but keep superfile 0.
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["alpha beta gamma", "doc with beta"]))
        .expect("append");
    w.commit().expect("commit");
    w.append(&build_title_batch(&[
        "alpha only here",
        "no betas whatever",
    ]))
    .expect("append");
    w.commit().expect("commit");
    drop(w);

    let r = st.reader();
    let manifest = r.manifest();
    let kept_uri = manifest.superfiles[0].uri;

    let before = store.snapshot();
    let _hits = r
        .bm25_hits(
            "title",
            "alpha beta",
            BM25_TOP_K,
            infino::supertable::query::fts::BoolMode::And,
        )
        .expect("AND query");

    let delta = store.delta(&before);
    assert_eq!(
        delta.len(),
        1,
        "AND mode should prune the superfile missing one of the terms"
    );
    assert!(delta.contains_key(&kept_uri));
}
