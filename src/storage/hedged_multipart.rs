// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A multipart upload whose stuck parts are sent again.
//!
//! A store answers most parts in well under a second and now and then holds
//! one for most of its request timeout. An upload waits for every part, so
//! one held part holds the whole object: a 1 GiB merge output of 128 parts
//! once took 46 s where its siblings took two to five, and the commit waiting
//! on it idled the host for 42 s. A part that has run past
//! [`stuck_after`] its size is sent again under the same part number; the
//! store keeps one copy of a part number, and both copies hold the same
//! bytes, so either answer completes the upload.

use std::{
    fmt, mem,
    pin::pin,
    sync::{Arc, Mutex, PoisonError},
};

use async_trait::async_trait;
use object_store::{
    Error, MultipartUpload, PutPayload, PutResult, Result, UploadPart,
    multipart::{MultipartStore, PartId},
    path::Path,
};
use tokio::{select, time::sleep};

use super::stuck_after;

/// A [`MultipartUpload`] over a [`MultipartStore`] that sends a part again
/// once it has run long enough to be taken for stuck.
pub(crate) struct HedgedUpload {
    store: Arc<dyn MultipartStore>,
    path: Path,
    id: String,
    /// Each part's id once it has landed, by part number.
    parts: Arc<Mutex<Vec<Option<PartId>>>>,
    next: usize,
}

impl fmt::Debug for HedgedUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HedgedUpload")
            .field("path", &self.path)
            .field("parts", &self.next)
            .finish()
    }
}

impl HedgedUpload {
    /// Start a multipart upload of `path` on `store`.
    pub(crate) async fn start(store: Arc<dyn MultipartStore>, path: Path) -> Result<Self> {
        let id = store.create_multipart(&path).await?;
        Ok(Self {
            store,
            path,
            id,
            parts: Arc::new(Mutex::new(Vec::new())),
            next: 0,
        })
    }
}

#[async_trait]
impl MultipartUpload for HedgedUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        let index = self.next;
        self.next += 1;
        let (store, path, id, parts) = (
            Arc::clone(&self.store),
            self.path.clone(),
            self.id.clone(),
            Arc::clone(&self.parts),
        );
        Box::pin(async move {
            let part = hedged_part(store.as_ref(), &path, &id, index, data).await?;
            let mut parts = parts.lock().unwrap_or_else(PoisonError::into_inner);
            if parts.len() <= index {
                parts.resize(index + 1, None);
            }
            parts[index] = Some(part);
            Ok(())
        })
    }

    async fn complete(&mut self) -> Result<PutResult> {
        let parts = mem::take(&mut *self.parts.lock().unwrap_or_else(PoisonError::into_inner));
        let parts = (0..self.next)
            .map(|index| {
                parts
                    .get(index)
                    .cloned()
                    .flatten()
                    .ok_or_else(|| Error::Generic {
                        store: "hedged multipart",
                        source: format!("part {index} of {} never landed", self.path).into(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        self.store
            .complete_multipart(&self.path, &self.id, parts)
            .await
    }

    async fn abort(&mut self) -> Result<()> {
        self.store.abort_multipart(&self.path, &self.id).await
    }
}

/// Put part `index`, sending it again once the first attempt has run
/// [`stuck_after`] its size. The first attempt to land wins; one that fails
/// waits for the other, so the part fails only if both do.
async fn hedged_part(
    store: &dyn MultipartStore,
    path: &Path,
    id: &str,
    index: usize,
    data: PutPayload,
) -> Result<PartId> {
    let delay = stuck_after(data.content_length() as u64);
    let id = id.to_string();
    let mut first = pin!(store.put_part(path, &id, index, data.clone()));
    select! {
        part = &mut first => part,
        () = sleep(delay) => {
            let mut second = pin!(store.put_part(path, &id, index, data));
            select! {
                part = &mut first => match part {
                    Ok(part) => Ok(part),
                    Err(_) => second.await,
                },
                part = &mut second => match part {
                    Ok(part) => Ok(part),
                    Err(_) => first.await,
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        future::pending,
        sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    use bytes::Bytes;
    use futures::future::try_join_all;
    use object_store::{MultipartId, ObjectStoreExt, memory::InMemory};

    use super::*;

    /// Bytes per part in the tests.
    const PART_BYTES: usize = 64;
    /// Parts in the tests' uploads.
    const PARTS: usize = 4;
    /// The part whose first attempt is held.
    const HELD_PART: usize = 2;

    /// A store whose first attempt at one part never answers, and which
    /// counts every attempt at a part. It keeps its own parts by number
    /// (object_store's in-memory store takes parts only in order, and a held
    /// part lands last) and writes the finished object to an in-memory store.
    #[derive(Debug)]
    struct HoldsOnePart {
        inner: InMemory,
        parts: Mutex<BTreeMap<usize, Bytes>>,
        held: AtomicBool,
        attempts: AtomicUsize,
    }

    impl HoldsOnePart {
        fn new(held: bool) -> Self {
            Self {
                inner: InMemory::new(),
                parts: Mutex::new(BTreeMap::new()),
                held: AtomicBool::new(held),
                attempts: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl MultipartStore for HoldsOnePart {
        async fn create_multipart(&self, _path: &Path) -> Result<MultipartId> {
            Ok("upload".to_string())
        }
        async fn put_part(
            &self,
            _path: &Path,
            _id: &MultipartId,
            part_idx: usize,
            data: PutPayload,
        ) -> Result<PartId> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            if part_idx == HELD_PART && self.held.swap(false, Ordering::SeqCst) {
                pending::<()>().await;
            }
            let bytes: Bytes = data.into();
            self.parts.lock().expect("parts").insert(part_idx, bytes);
            Ok(PartId {
                content_id: part_idx.to_string(),
            })
        }
        async fn complete_multipart(
            &self,
            path: &Path,
            _id: &MultipartId,
            parts: Vec<PartId>,
        ) -> Result<PutResult> {
            let object = {
                let stored = self.parts.lock().expect("parts");
                let mut object = Vec::new();
                for part in &parts {
                    let index: usize = part.content_id.parse().expect("part number");
                    object.extend_from_slice(&stored[&index]);
                }
                object
            };
            self.inner.put(path, PutPayload::from(object)).await
        }
        async fn abort_multipart(&self, _path: &Path, _id: &MultipartId) -> Result<()> {
            Ok(())
        }
    }

    fn part(index: usize) -> Bytes {
        Bytes::from(vec![index as u8; PART_BYTES])
    }

    /// An upload whose part never answers on its first attempt completes on
    /// its second, and the object holds every part's bytes in order.
    #[tokio::test(start_paused = true)]
    async fn a_held_part_is_answered_by_its_second_attempt() {
        let store = Arc::new(HoldsOnePart::new(true));
        let path = Path::from("hedged/object");
        let mut upload = HedgedUpload::start(store.clone(), path.clone())
            .await
            .expect("start");
        let puts: Vec<UploadPart> = (0..PARTS)
            .map(|index| upload.put_part(PutPayload::from_bytes(part(index))))
            .collect();
        try_join_all(puts).await.expect("every part lands");
        upload.complete().await.expect("complete");
        assert_eq!(
            store.attempts.load(Ordering::SeqCst),
            PARTS + 1,
            "one part sent twice"
        );
        let stored = store
            .inner
            .get(&path)
            .await
            .expect("get")
            .bytes()
            .await
            .expect("bytes");
        let expected: Vec<u8> = (0..PARTS).flat_map(|index| part(index).to_vec()).collect();
        assert_eq!(stored.as_ref(), expected.as_slice());
    }

    /// An upload whose parts all answer in time sends each part once.
    #[tokio::test(start_paused = true)]
    async fn parts_that_answer_in_time_are_sent_once() {
        let store = Arc::new(HoldsOnePart::new(false));
        let mut upload = HedgedUpload::start(store.clone(), Path::from("hedged/once"))
            .await
            .expect("start");
        let puts: Vec<UploadPart> = (0..PARTS)
            .map(|index| upload.put_part(PutPayload::from_bytes(part(index))))
            .collect();
        try_join_all(puts).await.expect("every part lands");
        upload.complete().await.expect("complete");
        assert_eq!(store.attempts.load(Ordering::SeqCst), PARTS);
    }
}
