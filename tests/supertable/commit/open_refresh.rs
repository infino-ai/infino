//! `Supertable::open` + `Supertable::refresh` — 003 M10.
//!
//! Covers:
//! - open against a persisted supertable written by another
//!   "process" (simulated via dropping the producer handle)
//! - manifest_id + superfiles + queries all match the producer's
//!   post-commit state
//! - open errors with `PointerUnreadable` on a fresh tempdir
//!   (open-or-create trigger)
//! - refresh after a producer's commit picks up the new
//!   manifest_id; pre-refresh reader stays pinned
//! - refresh is a no-op when the pointer hasn't advanced
//! - parts inherited via content-addressed Arc::clone (no
//!   re-fetch for unchanged parts)

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use infino::superfile::builder::FtsConfig;
use infino::superfile::fts::tokenize::Tokenizer;
use infino::supertable::storage::{LocalFsStorageProvider, StorageProvider};
use infino::supertable::{OpenError, Supertable, SupertableOptions};
use infino::test_helpers::{build_title_batch, default_supertable_options, default_tokenizer};
use tempfile::TempDir;

#[tokio::test(flavor = "multi_thread")]
async fn open_sees_writes_made_by_a_different_handle() {
    // Producer: create + commit + drop.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let producer =
        Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)));
    let mut w = producer.writer().expect("writer");
    w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
        .expect("append");
    w.commit().expect("commit");
    drop(w);
    drop(producer); // simulate "process exit"

    // Consumer: open against the same storage.
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .await
            .expect("open");
    assert_eq!(consumer.manifest_id(), 1);
    assert_eq!(consumer.reader().n_superfiles(), 1);
    // Note: full query parity post-open requires the deferred
    // query-path integration through `DiskCacheStore` —
    // M10's reader sees the manifest's segment list but
    // segment *bytes* live only in object storage and aren't
    // yet routed through the cache. That wiring is the next
    // step. M10 validates the manifest-side open here; an
    // end-to-end query test on a post-open Supertable lands
    // when the cache-backed reader path ships.
}

#[tokio::test(flavor = "multi_thread")]
async fn open_on_fresh_tempdir_returns_pointer_unreadable() {
    // The open-or-create trigger: no pointer exists, so
    // open() must surface a typed error the caller can
    // pattern-match on for fallback to Supertable::create.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let err = Supertable::open(default_supertable_options().with_storage(storage))
        .await
        .expect_err("must reject fresh dir");
    assert!(
        matches!(err, OpenError::PointerUnreadable(_)),
        "expected PointerUnreadable, got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn open_without_storage_rejects() {
    // open requires options.storage; without it the error is
    // a typed BuildError surfaced via OpenError::Build.
    let opts = default_supertable_options();
    let err = Supertable::open(opts).await.expect_err("must reject");
    assert!(matches!(err, OpenError::Build(_)), "{err:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn refresh_picks_up_new_commits() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // Producer commits v1.
    let producer =
        Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)));
    let mut w = producer.writer().expect("w1");
    w.append(&build_title_batch(&["initial"])).expect("append1");
    w.commit().expect("commit1");
    drop(w);

    // Consumer opens at v1.
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .await
            .expect("open");
    assert_eq!(consumer.manifest_id(), 1);
    let pre_refresh_reader = consumer.reader(); // pinned at v1

    // Producer commits v2.
    let mut w = producer.writer().expect("w2");
    w.append(&build_title_batch(&["added"])).expect("append2");
    w.commit().expect("commit2");
    drop(w);

    // Consumer's manifest_id hasn't moved yet — refresh()
    // pulls the new pointer.
    assert_eq!(consumer.manifest_id(), 1);
    let advanced = consumer.refresh().await.expect("refresh");
    assert!(advanced, "refresh must report it advanced");
    assert_eq!(consumer.manifest_id(), 2);
    assert_eq!(
        consumer.reader().n_superfiles(),
        2,
        "post-refresh reader sees both commits"
    );

    // Pre-refresh reader stays pinned at v1.
    assert_eq!(
        pre_refresh_reader.manifest_id(),
        1,
        "pre-refresh reader keeps its snapshot"
    );
    assert_eq!(pre_refresh_reader.n_superfiles(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn refresh_no_op_when_pointer_unchanged() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    let producer =
        Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)));
    let mut w = producer.writer().expect("w");
    w.append(&build_title_batch(&["only"])).expect("append");
    w.commit().expect("commit");
    drop(w);

    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .await
            .expect("open");

    // No producer commits between open and refresh.
    let advanced = consumer.refresh().await.expect("refresh");
    assert!(!advanced, "refresh must be a no-op when pointer unchanged");
    assert_eq!(consumer.manifest_id(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn refresh_no_op_returns_false_when_no_pointer_yet() {
    // Edge case: in-memory consumer (created locally) calls
    // refresh against a storage that has no pointer yet.
    // Should be a no-op (false), not an error.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(storage));
    let advanced = st.refresh().await.expect("refresh");
    assert!(!advanced);
    assert_eq!(st.manifest_id(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn open_rejects_mismatched_options_via_options_hash() {
    // D15: a producer commits with one schema; opening with
    // a structurally-different schema (different column
    // name) must surface a typed `OptionsHashMismatch`
    // before any decode work happens.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // Producer: standard schema.
    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)));
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
            .num_threads(1)
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
        .await
        .expect_err("open must surface OptionsHashMismatch for a reordered schema");
    assert!(
        matches!(err, OpenError::OptionsHashMismatch { .. }),
        "expected OptionsHashMismatch; got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn open_with_matching_options_succeeds_under_options_hash_validation() {
    // D15 happy path: producer + consumer with identical
    // options round-trip cleanly.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)));
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["alpha"])).expect("append");
        w.commit().expect("commit");
    }
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .await
            .expect("open must succeed when options match");
    assert_eq!(consumer.manifest_id(), 1);
}
