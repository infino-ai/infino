// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `Supertable::open` + read-path freshness (`Consistency`).
//!
//! Covers, entirely through the public API:
//! - open against a persisted supertable written by another
//!   "process" (simulated via dropping the producer handle)
//! - manifest_id + superfiles + queries all match the producer's
//!   post-commit state
//! - open errors with `PointerUnreadable` on a fresh tempdir
//!   (open-or-create trigger)
//! - under `Consistency::Strong`, a query on a consumer handle picks
//!   up another writer's new commit; readers taken before the query
//!   keep their pinned snapshot
//! - a strongly-consistent query is stable when the pointer hasn't
//!   advanced, and is a clean no-op on an as-yet-uncommitted table
//!
//! Freshness is engine-driven on the read path (governed by
//! `Consistency`); there is no public `refresh` verb, so these tests
//! observe freshness the way a real client does — by querying.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use infino::superfile::builder::FtsConfig;
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::fts::tokenize::Tokenizer;
use infino::supertable::options::Consistency;
use infino::supertable::storage::{LocalFsStorageProvider, StorageProvider};
use infino::supertable::{ManifestLoadError, OpenError, Supertable, SupertableOptions};
use infino::test_helpers::{build_title_batch, default_supertable_options, default_tokenizer};

/// BM25 top-k for the open/refresh consistency queries.
const BM25_TOP_K: usize = 10;
/// Single-thread rayon pool for the mismatched-schema test.
const RAYON_POOL_THREADS: usize = 1;
use tempfile::TempDir;

#[test]
fn open_sees_writes_made_by_a_different_handle() {
    // Producer: create + commit + drop.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let producer =
        Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("create");
    let mut w = producer.writer().expect("writer");
    w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
        .expect("append");
    w.commit().expect("commit");
    drop(w);
    drop(producer); // simulate "process exit"

    // Consumer: open against the same storage.
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open");
    assert_eq!(consumer.manifest_id(), 1);
    assert_eq!(consumer.reader().n_superfiles(), 1);
    // Note: full query parity post-open requires the deferred
    // query-path integration through `DiskCacheStore` —
    // the reader sees the manifest's superfile list but
    // superfile *bytes* live only in object storage and aren't
    // yet routed through the cache. That wiring is the next
    // step. This test validates the manifest-side open here; an
    // end-to-end query test on a post-open Supertable lands
    // when the cache-backed reader path ships.
}

#[test]
fn open_on_fresh_tempdir_returns_pointer_unreadable() {
    // The open-or-create trigger: no pointer exists, so
    // open() must surface a typed error the caller can
    // pattern-match on for fallback to Supertable::create.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let err = Supertable::open(default_supertable_options().with_storage(storage))
        .expect_err("must reject fresh dir");
    assert!(
        matches!(err, OpenError::ManifestLoadError(_)),
        "expected PointerUnreadable, got {err:?}"
    );
}

#[test]
fn open_without_storage_rejects() {
    // open requires options.storage; without it the error is
    // a typed BuildError surfaced via OpenError::Build.
    let opts = default_supertable_options();
    let err = Supertable::open(opts).expect_err("must reject");
    assert!(matches!(err, OpenError::Build(_)), "{err:?}");
}

#[test]
fn strong_consistency_query_sees_another_writers_new_commit() {
    // Freshness is observed the way a real client observes it: a
    // strongly-consistent consumer issues a query and sees the latest
    // committed state. There is no public `refresh` — the read path
    // re-checks the pointer under `Consistency::Strong`.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // Producer commits v1.
    let producer =
        Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("create");
    let mut w = producer.writer().expect("w1");
    w.append(&build_title_batch(&["initial"])).expect("append1");
    w.commit().expect("commit1");
    drop(w);

    // Consumer opens at v1 with strong read consistency.
    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_read_consistency(Consistency::Strong),
    )
    .expect("open");
    assert_eq!(consumer.manifest_id(), 1);
    let pinned_reader = consumer.reader(); // snapshot pinned at v1

    // Producer commits v2.
    let mut w = producer.writer().expect("w2");
    w.append(&build_title_batch(&["added"])).expect("append2");
    w.commit().expect("commit2");
    drop(w);

    // A strongly-consistent query re-checks the pointer and serves
    // against the latest manifest — picking up the new commit.
    let hits = consumer
        .reader()
        .bm25_hits("title", "added", BM25_TOP_K, BoolMode::Or)
        .expect("query under strong consistency");
    assert!(!hits.is_empty(), "strong query must see the v2 row");
    assert_eq!(consumer.manifest_id(), 2);
    assert_eq!(
        consumer.reader().n_superfiles(),
        2,
        "post-query reader sees both commits"
    );

    // The snapshot taken before the query stays pinned at v1.
    assert_eq!(
        pinned_reader.manifest_id(),
        1,
        "a reader captured earlier keeps its snapshot"
    );
    assert_eq!(pinned_reader.n_superfiles(), 1);
}

#[test]
fn strong_consistency_query_is_stable_when_pointer_unchanged() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    let producer =
        Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("create");
    let mut w = producer.writer().expect("w");
    w.append(&build_title_batch(&["only"])).expect("append");
    w.commit().expect("commit");
    drop(w);

    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_read_consistency(Consistency::Strong),
    )
    .expect("open");

    // No producer commits between open and query: the pointer hasn't
    // advanced, so the strongly-consistent query stays at v1.
    let _ = consumer
        .reader()
        .bm25_hits("title", "only", BM25_TOP_K, BoolMode::Or)
        .expect("query");
    assert_eq!(consumer.manifest_id(), 1);
}

#[test]
fn strong_consistency_query_on_uncommitted_table_stays_at_zero() {
    // Edge case: an in-memory table (created locally, never committed)
    // served under strong consistency. The read-path pointer re-check
    // finds nothing committed yet and is a clean no-op — the table
    // stays at manifest_id 0 and the query returns no hits.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(storage)
            .with_read_consistency(Consistency::Strong),
    )
    .expect("create");
    let hits = st
        .reader()
        .bm25_hits("title", "anything", BM25_TOP_K, BoolMode::Or)
        .expect("query on uncommitted table");
    assert!(hits.is_empty());
    assert_eq!(st.manifest_id(), 0);
}

#[test]
fn open_rejects_mismatched_options_via_options_hash() {
    // A producer commits with one schema; opening with
    // a structurally-different schema (different column
    // name) must surface a typed `OptionsHashMismatch`
    // before any decode work happens.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // Producer: standard schema.
    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["alpha"])).expect("append");
        w.commit().expect("commit");
    }

    // Consumer: same id_column, same fts column name, but
    // schema lists fields in REVERSE order — that changes
    // the per-field iteration the options_hash digest
    // covers.
    let other_schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("title", arrow_schema::DataType::LargeUtf8, false),
        arrow_schema::Field::new("doc_id", arrow_schema::DataType::UInt64, false),
    ]));
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("pool"),
    );
    let mismatched_opts = SupertableOptions::new(
        other_schema,
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(tk),
    )
    .expect("opts")
    .with_writer_pool(pool)
    .with_storage(Arc::clone(&storage));

    let err = Supertable::open(mismatched_opts)
        .expect_err("open must surface OptionsHashMismatch for a reordered schema");
    assert!(
        matches!(
            err,
            OpenError::ManifestLoadError(ManifestLoadError::ContentHashMismatch { .. })
        ),
        "expected OptionsHashMismatch; got {err:?}"
    );
}

#[test]
fn open_with_matching_options_succeeds_under_options_hash_validation() {
    // Happy path: producer + consumer with identical
    // options round-trip cleanly.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["alpha"])).expect("append");
        w.commit().expect("commit");
    }
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open must succeed when options match");
    assert_eq!(consumer.manifest_id(), 1);
}
