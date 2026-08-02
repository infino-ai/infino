// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `create` is create-or-open: a creator that loses the initial-pointer
//! race adopts the winner's manifest instead of failing.
//!
//! The loser's pre-built hidden vector-index handle used a freshly
//! generated storage prefix the durable manifest does not track, so the
//! adopt path must also RECONCILE: drop the loser's hidden handle and
//! reopen the hidden table at the prefix stamped in the adopted manifest.
//! A regression here silently splits the hidden index across two
//! prefixes — maintenance writes one subtree while every fresh handle
//! reads the other. The guard therefore routes vectors THROUGH the
//! adopted handle's hidden machinery (commit + drain) and asserts a
//! FRESH handle — which resolves its hidden prefix from the stamped
//! manifest alone — sees them. User-table adoption is asserted
//! separately; user-side queries cannot stand in for the hidden index,
//! because undrained searches are served from user superfiles.

#![deny(clippy::unwrap_used)]

use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use bytes::Bytes;
use infino::{
    storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions, manifest::commit::POINTER_PATH},
    test_helpers::{default_tokenizer, default_vector_config},
};
use tempfile::TempDir;

/// Hides the user-table pointer from its first `get` probe. `create`
/// pre-probes the pointer and opens when one exists; masking that probe
/// walks the loser down the full create path until the initial pointer
/// PUT collides with the winner's — the deterministic reproduction of
/// losing the create race inside the probe→publish window. The hidden
/// table's pointer lives under the sibling prefix, so only the exact
/// user pointer path is masked, and only once.
#[derive(Debug)]
struct PointerHiddenOnce {
    inner: Arc<dyn StorageProvider>,
    hidden_probes_left: AtomicUsize,
}

impl PointerHiddenOnce {
    fn new(inner: Arc<dyn StorageProvider>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            hidden_probes_left: AtomicUsize::new(1),
        })
    }
}

#[async_trait]
impl StorageProvider for PointerHiddenOnce {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }
    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        if uri == POINTER_PATH
            && self
                .hidden_probes_left
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
        {
            return Err(StorageError::NotFound { uri: uri.into() });
        }
        self.inner.get(uri).await
    }
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.inner.get_range(uri, range).await
    }
    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.inner.put_atomic(uri, bytes).await
    }
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        self.inner.put_if_match(uri, bytes, expected_etag).await
    }
    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.inner.put_multipart(uri).await
    }
    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(uri).await
    }
}

/// Matches `default_vector_config`'s dimension.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 31;
/// The winner's committed corpus.
const TITLES: &[&str] = &["alpha document", "bravo document", "charlie document"];

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

fn vector_options() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(DIM), false),
    ]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
        Some(default_tokenizer()),
    )
    .expect("valid options")
}

fn one_hot_batch(schema: Arc<Schema>) -> RecordBatch {
    let n = TITLES.len();
    let mut flat = Vec::<f32>::with_capacity(n * DIM);
    for i in 0..n {
        for d in 0..DIM {
            flat.push(if d == i % DIM { 1.0 } else { 0.0 });
        }
    }
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(TITLES.to_vec())),
            Arc::new(fsl),
        ],
    )
    .expect("batch")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn losing_creator_adopts_winner_manifest_and_hidden_index() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    // Winner: creates the table (publishes the id-0 pointer and the
    // hidden index bootstrap) and commits real vector data.
    let winner = Supertable::create(vector_options().with_storage(Arc::clone(&storage)))
        .expect("winner create");
    let schema = winner.options().schema.clone();
    let mut w = winner.writer().expect("writer");
    w.append(&one_hot_batch(schema)).expect("append");
    w.commit().expect("commit");
    assert_eq!(winner.manifest_id(), 1);

    // Loser: same storage, but its create-time pointer probe is masked so
    // it walks the full create path — building its own hidden handle at a
    // fresh prefix — until the initial pointer PUT collides with the
    // winner's. Create must then adopt the durable manifest and reconcile
    // its hidden handle to the stamped prefix — not error, and not create
    // a second table.
    let racing: Arc<dyn StorageProvider> = PointerHiddenOnce::new(Arc::clone(&storage));
    let loser = Supertable::create(vector_options().with_storage(racing))
        .expect("create-or-open must adopt, not fail");
    assert_eq!(
        loser.manifest_id(),
        1,
        "the adopted view is the winner's committed manifest"
    );
    assert_eq!(
        loser.reader().expect("reader").n_docs_total(),
        TITLES.len() as u64
    );

    // The reconciliation guard — the two-prefix-split detector. Route
    // vectors through the ADOPTED handle's hidden machinery (a commit's
    // dual-write plus an explicit drain), then read the hidden index
    // through a FRESH handle, which resolves its hidden prefix from the
    // stamped manifest alone. Had the loser kept its pre-built hidden
    // handle (a fresh prefix the manifest never records), this batch's
    // vectors would land under that orphaned subtree and the fresh
    // handle's hidden index would come up short. A user-side query can't
    // stand in here: undrained searches are answered from user
    // superfiles and pass regardless of which prefix the hidden handle
    // points at.
    let schema = loser.options().schema.clone();
    let mut w = loser.writer().expect("loser writer");
    w.append(&one_hot_batch(schema))
        .expect("append via adopted");
    w.commit().expect("commit via adopted");
    drop(w);
    loser
        .drain_vectors_to_cells_sync()
        .expect("drain through the adopted handle");

    let fresh =
        Supertable::open(vector_options().with_storage(Arc::clone(&storage))).expect("fresh open");
    assert_eq!(
        fresh.reader().expect("reader").n_docs_total(),
        2 * TITLES.len() as u64,
        "both writers' commits are visible"
    );
    let fresh_hidden = fresh.vector_index_table().expect("hidden handle");
    assert_eq!(
        fresh_hidden.reader().expect("hidden reader").n_docs_total(),
        2 * TITLES.len() as u64,
        "vectors routed through the adopted handle must land under the \
         manifest-stamped hidden prefix; a shortfall means the loser \
         drained into its own orphaned prefix"
    );
}
