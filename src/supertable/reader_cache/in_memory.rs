// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! In-memory reader cache.
//!
//! Holds every inserted superfile's bytes in RAM for the
//! supertable's lifetime. No eviction — total RAM is the sum of
//! every published superfile's bytes. Suitable for the in-memory
//! supertable shape; not for corpora that exceed the host's RAM
//! budget.
//!
//! ## Concurrency
//!
//! `RwLock<HashMap<...>>`. The `reader()` hot path takes a *read*
//! lock so a fan-out (rayon `par_iter` across N non-pruned
//! superfiles) can resolve all N URIs in parallel — readers never
//! serialize on each other. `insert()` takes a write lock; its
//! window is bounded by the in-flight `HashMap::entry +
//! or_insert` (the bytes parse runs outside both locks, in the
//! optimistic-then-recheck flow below).
//!
//! ## Idempotent insert
//!
//! `insert(uri, bytes)` is a no-op if the URI already exists —
//! superfiles are immutable, so the same URI always names the
//! same bytes (caller's contract). Re-inserting doesn't re-parse
//! the bytes or re-build the reader.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use bytes::Bytes;

use super::{ReaderCacheError, SuperfileReaderCache};
use crate::{superfile::SuperfileReader, supertable::manifest::SuperfileUri};

/// Per-URI entry. Holds the raw bytes alongside the parsed reader
/// so `resident_bytes()` can attribute storage back to specific
/// URIs cheaply (without re-walking each `SuperfileReader`'s
/// internal state).
struct Entry {
    bytes: Bytes,
    reader: Arc<SuperfileReader>,
}

/// `RwLock<HashMap<SuperfileUri, Entry>>`-backed superfile store.
///
/// Cheap to clone *as `Arc<dyn SuperfileReaderCache>`* — but `Clone` is
/// not implemented directly because cloning the inner map (vs the
/// `Arc`) almost always indicates a bug. Wrap `Self` in `Arc` at
/// construction and pass that around instead.
pub struct InMemoryReaderCache {
    inner: RwLock<HashMap<SuperfileUri, Entry>>,
}

impl InMemoryReaderCache {
    /// Empty store.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    /// Number of superfiles currently registered. Read lock — does
    /// not serialize against concurrent `reader()` calls.
    pub fn n_superfiles(&self) -> usize {
        self.inner
            .read()
            .expect("InMemoryReaderCache rwlock poisoned")
            .len()
    }
}

impl Default for InMemoryReaderCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SuperfileReaderCache for InMemoryReaderCache {
    fn reader(&self, uri: &SuperfileUri) -> Result<Arc<SuperfileReader>, ReaderCacheError> {
        let map = self
            .inner
            .read()
            .expect("InMemoryReaderCache rwlock poisoned");
        map.get(uri)
            .map(|entry| Arc::clone(&entry.reader))
            .ok_or(ReaderCacheError::NotFound { uri: *uri })
    }

    fn insert(&self, uri: SuperfileUri, bytes: Bytes) -> Result<(), ReaderCacheError> {
        // Optimistic read first — same-URI re-puts are the hot
        // path for the writer's idempotent-publish flow.
        {
            let map = self
                .inner
                .read()
                .expect("InMemoryReaderCache rwlock poisoned");
            if map.contains_key(&uri) {
                return Ok(());
            }
        }

        // Open + reader-build OUTSIDE both locks so concurrent
        // puts of different URIs don't serialize on the parse.
        let reader = SuperfileReader::open(bytes.clone())
            .map_err(|source| ReaderCacheError::OpenFailed { source })?;

        let mut map = self
            .inner
            .write()
            .expect("InMemoryReaderCache rwlock poisoned");
        // Re-check under the write lock — between the read above
        // and now, another caller may have raced the same URI in.
        // First-writer-wins; we drop our parsed reader on the
        // floor (cheap; just an Arc + the parsed Bytes view).
        map.entry(uri).or_insert(Entry {
            bytes,
            reader: Arc::new(reader),
        });
        Ok(())
    }

    fn resident_bytes(&self) -> usize {
        self.inner
            .read()
            .expect("InMemoryReaderCache rwlock poisoned")
            .values()
            .map(|e| e.bytes.len())
            .sum()
    }

    fn remove(&self, uri: &SuperfileUri) {
        self.inner
            .write()
            .expect("InMemoryReaderCache rwlock poisoned")
            .remove(uri);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{Field, Schema};

    use super::*;
    use crate::{
        superfile::builder::{BuilderOptions, SuperfileBuilder},
        test_helpers::{decimal128_id_field, decimal128_ids},
    };

    /// Build minimal valid superfile bytes (no FTS, no vectors —
    /// just the parquet body + KV metadata that
    /// `SuperfileReader::open` requires).
    fn minimal_superfile_bytes() -> Bytes {
        let schema: Arc<Schema> = Arc::new(Schema::new(vec![
            decimal128_id_field("doc_id"),
            Field::new("title", arrow_schema::DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![], None);
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let ids = decimal128_ids(vec![1u64, 2, 3]);
        let title = LargeStringArray::from(vec!["a", "b", "c"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title)])
            .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
        Bytes::from(b.finish().expect("finish builder"))
    }

    fn fresh_uri() -> SuperfileUri {
        SuperfileUri::new_v4()
    }

    // ---- empty / round-trip -------------------------------------------

    #[test]
    fn new_store_is_empty() {
        let store = InMemoryReaderCache::new();
        assert_eq!(store.n_superfiles(), 0);
        assert_eq!(store.resident_bytes(), 0);
    }

    #[test]
    fn insert_then_reader_round_trips() {
        let store = InMemoryReaderCache::new();
        let bytes = minimal_superfile_bytes();
        let uri = fresh_uri();

        store
            .insert(uri, bytes.clone())
            .expect("insert should succeed");

        let r = store.reader(&uri).expect("reader should find uri");
        assert_eq!(r.n_docs(), 3, "minimal superfile carries 3 docs");
        assert_eq!(store.n_superfiles(), 1);
        assert_eq!(store.resident_bytes(), bytes.len());
    }

    #[test]
    fn reader_on_unknown_uri_returns_not_found() {
        let store = InMemoryReaderCache::new();
        let unknown = fresh_uri();
        let err = store.reader(&unknown).expect_err("expected error");
        match err {
            ReaderCacheError::NotFound { uri } => {
                assert_eq!(uri, unknown);
            }
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn insert_with_invalid_bytes_returns_open_failed() {
        let store = InMemoryReaderCache::new();
        let uri = fresh_uri();
        let garbage = Bytes::from(vec![0u8; 16]);
        let err = store.insert(uri, garbage).expect_err("expected error");
        assert!(matches!(err, ReaderCacheError::OpenFailed { .. }));
        // Failed put should leave the store empty.
        assert_eq!(store.n_superfiles(), 0);
        assert_eq!(store.resident_bytes(), 0);
        assert!(store.reader(&uri).is_err());
    }

    // ---- idempotent put ------------------------------------------------

    #[test]
    fn insert_is_idempotent_on_duplicate_uri() {
        let store = InMemoryReaderCache::new();
        let bytes = minimal_superfile_bytes();
        let uri = fresh_uri();

        store.insert(uri, bytes.clone()).expect("first insert");
        let bytes_after_first = store.resident_bytes();
        let r1 = store.reader(&uri).expect("first reader");

        // Second put with same URI: no-op semantics. Even if we
        // pass DIFFERENT bytes, the contract says same URI →
        // same bytes (caller invariant); the store doesn't
        // re-parse.
        store.insert(uri, bytes.clone()).expect("idempotent insert");
        let r2 = store.reader(&uri).expect("second reader");

        assert_eq!(store.n_superfiles(), 1, "still one superfile");
        assert_eq!(
            store.resident_bytes(),
            bytes_after_first,
            "byte accounting unchanged",
        );
        // Same Arc<SuperfileReader> on both reads — re-put didn't
        // replace the parsed reader.
        assert!(Arc::ptr_eq(&r1, &r2));
    }

    // ---- multi-superfile accounting --------------------------------------

    #[test]
    fn resident_bytes_sums_across_superfiles() {
        let store = InMemoryReaderCache::new();
        let bytes_a = minimal_superfile_bytes();
        let bytes_b = minimal_superfile_bytes();
        store
            .insert(fresh_uri(), bytes_a.clone())
            .expect("insert a");
        store
            .insert(fresh_uri(), bytes_b.clone())
            .expect("insert b");
        assert_eq!(store.n_superfiles(), 2);
        assert_eq!(store.resident_bytes(), bytes_a.len() + bytes_b.len());
    }

    // ---- reader Arc identity -------------------------------------------

    #[test]
    fn concurrent_reader_clones_share_arc() {
        // Two reader() calls for the same URI return Arcs to the
        // same underlying SuperfileReader.
        let store = InMemoryReaderCache::new();
        let uri = fresh_uri();
        store
            .insert(uri, minimal_superfile_bytes())
            .expect("insert");
        let a = store.reader(&uri).expect("a");
        let b = store.reader(&uri).expect("b");
        assert!(Arc::ptr_eq(&a, &b));
    }

    // ---- threading -----------------------------------------------------

    #[test]
    fn store_is_send_and_sync() {
        // Compile-time check that the trait + impl are both
        // Send + Sync, which is the contract that lets multiple
        // reader threads share `Arc<dyn SuperfileReaderCache>`.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InMemoryReaderCache>();
        assert_send_sync::<Arc<dyn SuperfileReaderCache>>();
    }

    #[test]
    fn concurrent_reads_after_put_succeed() {
        // Smoke test: pre-populate a few URIs, then spawn several
        // threads each doing many reader() lookups. Verify all
        // succeed and return Arcs that share the underlying
        // SuperfileReader per URI.
        use std::thread;

        let store: Arc<dyn SuperfileReaderCache> = Arc::new(InMemoryReaderCache::new());
        let bytes = minimal_superfile_bytes();
        let uris: Vec<SuperfileUri> = (0..4).map(|_| fresh_uri()).collect();
        for u in &uris {
            store.insert(*u, bytes.clone()).expect("insert");
        }

        let mut handles = Vec::new();
        for _ in 0..4 {
            let store = Arc::clone(&store);
            let uris = uris.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    for u in &uris {
                        let r = store.reader(u).expect("reader");
                        assert_eq!(r.n_docs(), 3);
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    #[test]
    fn concurrent_puts_of_distinct_uris_all_succeed() {
        // Each put builds its parsed SuperfileReader OUTSIDE the
        // mutex; this test exercises that doesn't break under
        // concurrent puts.
        use std::thread;

        let store: Arc<dyn SuperfileReaderCache> = Arc::new(InMemoryReaderCache::new());
        let mut handles = Vec::new();
        let n_threads = 4;
        let n_per_thread = 4;
        for _ in 0..n_threads {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for _ in 0..n_per_thread {
                    let uri = fresh_uri();
                    store
                        .insert(uri, minimal_superfile_bytes())
                        .expect("insert should succeed");
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }
        // All puts should land — every UUID is unique.
        let store_concrete = InMemoryReaderCache::new();
        let _ = store_concrete; // just type-check
        // n_superfiles check via the trait.
        let n = (0..n_threads * n_per_thread).map(|_| ()).count();
        // We can't downcast Arc<dyn SuperfileReaderCache> easily; instead
        // verify resident_bytes is positive and grows
        // proportionally.
        let resident = store.resident_bytes();
        assert!(resident > 0);
        // Each superfile is the same size (minimal_superfile_bytes
        // is deterministic).
        let expected_per_seg = minimal_superfile_bytes().len();
        assert_eq!(resident, expected_per_seg * n);
    }

    #[test]
    fn concurrent_puts_of_same_uri_resolve_to_one_entry() {
        // Idempotent semantics under contention: if N threads
        // race the same URI in, exactly one entry lands and all
        // subsequent reader() calls see the same Arc.
        use std::thread;

        let store: Arc<dyn SuperfileReaderCache> = Arc::new(InMemoryReaderCache::new());
        let uri = fresh_uri();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                store
                    .insert(uri, minimal_superfile_bytes())
                    .expect("insert should succeed");
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }

        // Exactly one superfile in the store.
        assert_eq!(store.resident_bytes(), minimal_superfile_bytes().len());
        // All reads return the same Arc.
        let r1 = store.reader(&uri).expect("r1");
        let r2 = store.reader(&uri).expect("r2");
        assert!(Arc::ptr_eq(&r1, &r2));
    }
}
