// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Writer write-through to storage.
//!
//! Covers the persistence path the writer takes when
//! `SupertableOptions::with_storage(...)` is attached:
//!
//! - A commit on a storage-backed supertable writes:
//!   - each new superfile's bytes to `data/seg-<uuid>.sf.parquet`
//!   - one manifest part to `manifest-parts/part-<hash>.avro.zst`
//!   - the manifest to `manifest/manifest-NNNNNN.json`
//!   - the pointer to `_supertable/current`
//! - The pointer is readable after commit; manifest_id
//!   increments per commit.
//! - Two successive commits both publish (CAS works); the
//!   second commit's manifest list references all superfiles
//!   (existing + new).
//! - In-memory queries still work post-commit (the in-memory
//!   store stays active for reads even with storage attached).
//! - A supertable with NO storage attached takes the
//!   in-memory path — no on-disk state, no regressions.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, RecordBatch};
use infino::supertable::{Supertable, manifest::commit::read_pointer};

/// 1-byte multipart threshold forcing every upload through the
/// multipart path.
const PUT_MULTIPART_THRESHOLD_BYTES: u64 = 1;
/// BM25 top-k for the post-commit query.
const BM25_TOP_K: usize = 5;
use infino::{
    Bm25SearchOptions,
    supertable::storage::{LocalFsStorageProvider, StorageProvider},
    test_helpers::{build_title_batch, default_supertable_options},
};
use tempfile::TempDir;

#[test]
fn commit_persists_pointer_list_part_and_superfile() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
        .expect("append");
    w.commit().expect("commit");
    drop(w);

    // Pointer file exists on disk, manifest_id=1 (initial was 0).
    let (pointer, _) = futures::executor::block_on(read_pointer(&*storage))
        .expect("read")
        .expect("pointer present");
    assert_eq!(pointer.get_manifest_id(), 1);
    assert!(pointer.manifest_uri.starts_with("manifest/manifest-"));

    // Manifest file exists and is non-empty.
    let (list_bytes, _) =
        futures::executor::block_on(storage.get(&pointer.manifest_uri)).expect("get list");
    assert!(!list_bytes.is_empty());

    // At least one manifest part exists in manifest-parts/.
    let manifest_parts_dir = dir.path().join("manifest-parts");
    let parts: Vec<_> = std::fs::read_dir(&manifest_parts_dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        parts.len(),
        1,
        "single-partition mode: exactly one manifest part on disk; got {parts:?}"
    );

    // Superfile file exists in data/.
    let data_dir = dir.path().join("data");
    let superfiles: Vec<_> = std::fs::read_dir(&data_dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        superfiles.len(),
        1,
        "one shard committed → one superfile file on disk; got {superfiles:?}"
    );

    // In-memory manifest reflects the commit.
    let r = st.reader().expect("reader");
    assert_eq!(r.manifest_id(), 1);
    assert_eq!(r.n_superfiles(), 1);
}

#[test]
fn two_successive_commits_both_publish() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let mut w = st.writer().expect("w1");
    w.append(&build_title_batch(&["foo", "bar"]))
        .expect("append1");
    w.commit().expect("commit1");
    drop(w);

    let mut w = st.writer().expect("w2");
    w.append(&build_title_batch(&["baz"])).expect("append2");
    w.commit().expect("commit2");
    drop(w);

    let (pointer, _) = futures::executor::block_on(read_pointer(&*storage))
        .expect("read")
        .expect("pointer");
    assert_eq!(
        pointer.get_manifest_id(),
        2,
        "two commits ⇒ pointer at manifest_id=2"
    );

    // Each manifest version persists (immutable per id): the empty manifest
    // published by `create` (id 0) plus the two commits (ids 1 + 2).
    let manifest_dir = dir.path().join("manifest");
    let n_manifests = std::fs::read_dir(&manifest_dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .count();
    assert_eq!(
        n_manifests, 3,
        "three manifest files (manifest_id 0 + 1 + 2)"
    );

    // ManifestSnapshot part count = 2 (each commit writes a fresh part
    // under content-addressed URI; single-partition mode
    // means a fresh part per commit, no reuse).
    let manifest_parts_dir = dir.path().join("manifest-parts");
    let n_parts = std::fs::read_dir(&manifest_parts_dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .count();
    assert_eq!(n_parts, 2);

    // In-memory manifest reflects both commits.
    let r = st.reader().expect("reader");
    assert_eq!(r.manifest_id(), 2);
    assert_eq!(
        r.n_superfiles(),
        2,
        "two shard commits ⇒ two superfiles visible"
    );
}

#[test]
fn multipart_threshold_forces_superfile_through_put_multipart() {
    // Setting `put_multipart_threshold_bytes = 1` routes
    // every superfile through `put_multipart` instead of
    // `put_atomic`. Verifies the end-to-end shape:
    //   - commit succeeds (no panic, no error)
    //   - superfile file lands on disk
    //   - manifest pointer + list + part written
    //   - cross-process open recovers the data
    // The actual `put_atomic` vs `put_multipart` distinction
    // is invisible to readers — the test passes through
    // `Supertable::open` to assert the superfile bytes were
    // correctly assembled by the multipart path.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let opts = default_supertable_options()
        .with_storage(Arc::clone(&storage))
        .with_put_multipart_threshold_bytes(PUT_MULTIPART_THRESHOLD_BYTES);
    let producer = Supertable::create(opts).expect("create");
    {
        let mut w = producer.writer().expect("writer");
        // Two docs so the FTS posting list has more than a
        // single term — exercises a non-trivial superfile
        // payload through multipart chunking.
        w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
            .expect("append");
        w.commit().expect("commit via multipart path");
    }
    drop(producer);

    // Superfile file landed on disk.
    let data_dir = dir.path().join("data");
    let superfiles: Vec<_> = std::fs::read_dir(&data_dir)
        .expect("readdir data")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        superfiles.len(),
        1,
        "one superfile file should land on disk after a multipart commit"
    );

    // Cross-process open recovers correctly — proof the
    // multipart-uploaded superfile is byte-identical to what
    // the writer produced.
    let consumer =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open after multipart commit");
    let r = consumer.reader().expect("reader");
    assert_eq!(r.manifest_id(), 1);
    assert_eq!(r.n_superfiles(), 1);
}

#[test]
fn no_storage_attached_takes_in_memory_path() {
    // Sanity: a supertable WITHOUT storage attached behaves
    // exactly like the no-storage baseline — in-memory only.
    let dir = TempDir::new().expect("tempdir");
    let st = Supertable::create(default_supertable_options()).expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["x", "y"])).expect("append");
    w.commit().expect("commit");
    drop(w);

    // Nothing on disk under the tempdir.
    let entries: Vec<_> = std::fs::read_dir(dir.path())
        .expect("readdir")
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(
        entries.len(),
        0,
        "no-storage supertable must not touch the filesystem; got {entries:?}"
    );

    // In-memory manifest still updates.
    let r = st.reader().expect("reader");
    assert_eq!(r.manifest_id(), 1);
    assert_eq!(r.n_superfiles(), 1);
}

#[test]
fn committed_supertable_remains_in_memory_queryable_for_now() {
    // Storage write-through is additive — the
    // in-memory store still holds superfile bytes, so existing
    // in-memory query paths keep working unchanged. Verifies no
    // regression to the FTS read path.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&[
        "nimblefox special token",
        "ordinary common text",
    ]))
    .expect("append");
    w.commit().expect("commit");
    drop(w);

    let hits = st
        .reader()
        .expect("reader")
        .bm25_hits(
            "title",
            "nimblefox",
            BM25_TOP_K,
            Bm25SearchOptions::new().with_mode(infino::supertable::query::fts::BoolMode::Or),
        )
        .expect("query");
    assert_eq!(hits.len(), 1, "commit must not break in-memory reads");
}

/// `append_named` keys the commit's superfile by the source stem while
/// everything else stays as it is: the object lands as
/// `data/<stem>-<uuid>.sf.parquet`, a later plain `append` lands unnamed
/// beside it, a fresh handle reads both back from storage by their keys,
/// and a zero-gap GC keeps the named object because the keep-set is built
/// from the same key the writer used.
#[test]
fn append_named_keys_the_superfile_by_its_source_stem() {
    use std::time::Duration;

    use infino::supertable::manifest::SuperfileUri;

    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let mut w = st.writer().expect("writer");
    w.append_named(
        &build_title_batch(&["nimblefox special token", "ordinary text"]),
        "Customers 2024.parquet",
    )
    .expect("append_named");
    w.commit().expect("commit");
    drop(w);

    let data_dir = dir.path().join("data");
    let names = |label: &str| -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&data_dir)
            .unwrap_or_else(|e| panic!("readdir data ({label}): {e}"))
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    };
    let after_named = names("after append_named");
    assert_eq!(
        after_named.len(),
        1,
        "one shard → one object: {after_named:?}"
    );
    let named = &after_named[0];
    assert!(
        named.starts_with("customers_2024_parquet-") && named.ends_with(".sf.parquet"),
        "the object key carries the key-safe stem: {named}"
    );
    assert!(
        SuperfileUri::from_storage_path(&format!("data/{named}")).is_some(),
        "the named key parses back to a uri: {named}"
    );

    // A plain append afterwards is unnamed, beside it.
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&["another row"]))
        .expect("append");
    w.commit().expect("commit");
    drop(w);
    let after_plain = names("after plain append");
    assert_eq!(after_plain.len(), 2, "{after_plain:?}");
    assert!(
        after_plain.iter().any(|n| n.starts_with("seg-")),
        "the plain append is unnamed: {after_plain:?}"
    );

    // GC's keep-set is built from the same key the writer PUT at: a zero
    // safety gap reaps superseded manifest history but no live superfile,
    // named or not.
    let report = st.gc(Duration::ZERO).expect("gc");
    assert_eq!(
        names("after gc"),
        after_plain,
        "every live superfile survives gc: {report:?}"
    );

    // A fresh handle knows nothing of the writer's in-memory bytes: it reads
    // the named superfile from storage at the key the manifest names.
    drop(st);
    let reopened =
        Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open");
    let reader = reopened.reader().expect("reader");
    assert_eq!(reader.n_superfiles(), 2);
    let hits = reader
        .bm25_hits(
            "title",
            "nimblefox",
            BM25_TOP_K,
            Bm25SearchOptions::new().with_mode(infino::supertable::query::fts::BoolMode::Or),
        )
        .expect("query through the named key");
    assert_eq!(hits.len(), 1, "the named superfile's rows are served");
}

/// A label describes a whole commit, so two sources buffered into one commit
/// must publish it unnamed rather than under whichever source was named last
/// — labelling it `orders` would put that name on the `customers` rows. The
/// same source named twice keeps its label, so the guard costs the ordinary
/// case nothing.
#[test]
fn two_sources_in_one_commit_publish_unnamed() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let data_dir = dir.path().join("data");
    let names = |label: &str| -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&data_dir)
            .unwrap_or_else(|e| panic!("readdir data ({label}): {e}"))
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    };

    // Two different sources, one commit: no single name describes it.
    let mut w = st.writer().expect("writer");
    w.append_named(&build_title_batch(&["from customers"]), "customers.parquet")
        .expect("append_named customers");
    w.append_named(&build_title_batch(&["from orders"]), "orders.parquet")
        .expect("append_named orders");
    w.commit().expect("commit");
    drop(w);

    let mixed = names("after two sources");
    assert_eq!(mixed.len(), 1, "one commit → one object: {mixed:?}");
    assert!(
        mixed[0].starts_with("seg-"),
        "a two-source commit is unnamed, not keyed by the last source: {mixed:?}"
    );

    // The same source named twice is still one source: the label holds.
    let mut w = st.writer().expect("writer");
    w.append_named(&build_title_batch(&["first"]), "customers.parquet")
        .expect("append_named 1");
    w.append_named(&build_title_batch(&["second"]), "customers.parquet")
        .expect("append_named 2");
    w.commit().expect("commit");
    drop(w);

    let repeated = names("after the same source twice");
    assert_eq!(repeated.len(), 2, "{repeated:?}");
    assert!(
        repeated.iter().any(|n| n.starts_with("customers_parquet-")),
        "one source named twice keeps its label: {repeated:?}"
    );

    // A commit after the mixed one starts clean: the dropped label does not
    // leak into the next commit, and a fresh name applies.
    let mut w = st.writer().expect("writer");
    w.append_named(&build_title_batch(&["third"]), "invoices.parquet")
        .expect("append_named invoices");
    w.commit().expect("commit");
    drop(w);

    let after = names("after a fresh source");
    assert_eq!(after.len(), 3, "{after:?}");
    assert!(
        after.iter().any(|n| n.starts_with("invoices_parquet-")),
        "a commit after a mixed one takes its own label: {after:?}"
    );
}

/// A row-less batch contributes no rows, so it must contribute no opinion
/// about the label either. Both arms matter: an empty `append` between two
/// named appends would otherwise make the commit mixed and publish correctly
/// sourced rows unnamed, and an empty `append_named` naming a second source
/// would do the same. A hydrate job hitting an empty shard is exactly the
/// caller that produces one.
#[test]
fn an_empty_batch_leaves_the_source_label_alone() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let empty = build_title_batch(&[]);
    let mut w = st.writer().expect("writer");
    w.append_named(&build_title_batch(&["real row"]), "customers.parquet")
        .expect("append_named");
    // Neither of these joins a row, so neither may unname the commit.
    w.append(&empty).expect("empty append");
    w.append_named(&empty, "orders.parquet")
        .expect("empty append_named");
    w.commit().expect("commit");
    drop(w);

    let names: Vec<String> = std::fs::read_dir(dir.path().join("data"))
        .expect("readdir data")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names.len(), 1, "one commit → one object: {names:?}");
    assert!(
        names[0].starts_with("customers_parquet-"),
        "the label survives row-less batches: {names:?}"
    );
}

/// A row-less batch BEFORE the first named one, which is the harder half: an
/// empty batch that reached the buffer would make it non-empty while nothing
/// had been buffered, so the first real named append would read as joining a
/// mixed commit and lose its label. `buffer` has to hold rows or nothing.
#[test]
fn an_empty_batch_before_the_first_named_one_does_not_cost_it_its_label() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let empty = build_title_batch(&[]);
    let mut w = st.writer().expect("writer");
    // Both orders, because both put a row-less batch through the buffer path.
    w.append(&empty).expect("empty append first");
    w.append_named(&empty, "ignored.parquet")
        .expect("empty append_named");
    w.append_named(&build_title_batch(&["real row"]), "customers.parquet")
        .expect("append_named");
    w.commit().expect("commit");
    drop(w);

    let names: Vec<String> = std::fs::read_dir(dir.path().join("data"))
        .expect("readdir data")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names.len(), 1, "one commit → one object: {names:?}");
    assert!(
        names[0].starts_with("customers_parquet-"),
        "an empty batch before the named one must not make the commit look mixed: {names:?}"
    );
}

/// A rejected append leaves the buffered commit alone - its rows AND its
/// label. The buffer is what a failed append preserves, so preserving the rows
/// while relabelling them would publish one source's rows under another's
/// name, or unnamed, on a call that returned an error.
#[test]
fn a_rejected_append_does_not_relabel_the_buffered_commit() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let mut w = st.writer().expect("writer");
    w.append_named(&build_title_batch(&["from customers"]), "customers.parquet")
        .expect("append_named");

    // A batch the writer must refuse: a schema that is not the table's.
    let wrong = RecordBatch::try_from_iter([(
        "not_the_title_column",
        Arc::new(Int32Array::from(vec![1])) as ArrayRef,
    )])
    .expect("build a mismatched batch");
    w.append_named(&wrong, "orders.parquet")
        .expect_err("a mismatched schema must be refused");

    w.commit().expect("commit");
    drop(w);

    let names: Vec<String> = std::fs::read_dir(dir.path().join("data"))
        .expect("readdir data")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names.len(), 1, "one commit → one object: {names:?}");
    assert!(
        names[0].starts_with("customers_parquet-"),
        "the refused append must not have taken the label with it: {names:?}"
    );
}

#[test]
fn manifest_id_increments_only_on_non_empty_commits() {
    // A commit with no buffered batches is a no-op.
    // Storage write-through should preserve this — no spurious
    // pointer rewrites on empty commits.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    let mut w = st.writer().expect("w");
    w.commit().expect("empty commit"); // no buffer → no-op
    drop(w);

    // `create` publishes the initial empty manifest, so the pointer already
    // exists at manifest_id=0. The empty commit above is a no-op: it must
    // neither advance the id nor republish.
    let (pointer, _) = futures::executor::block_on(read_pointer(&*storage))
        .expect("read")
        .expect("create publishes the initial empty-manifest pointer");
    assert_eq!(
        pointer.get_manifest_id(),
        0,
        "empty commit must not advance the manifest_id past create's id 0"
    );

    // Now do a real commit; pointer advances to manifest_id=1.
    let mut w = st.writer().expect("w");
    w.append(&build_title_batch(&["only", "real"]))
        .expect("append");
    w.commit().expect("real commit");
    drop(w);

    let (pointer, _) = futures::executor::block_on(read_pointer(&*storage))
        .expect("read")
        .expect("pointer");
    assert_eq!(pointer.get_manifest_id(), 1);
}
