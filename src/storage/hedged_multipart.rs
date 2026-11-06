// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A multipart upload whose stuck parts are sent again.
//!
//! A store answers most parts in well under a second and now and then holds
//! one for most of its request timeout. An upload waits for every part, so
//! one held part holds the whole object: a 1 GiB merge output of 128 parts
//! once took 46 s where its siblings took two to five, and the commit waiting
//! on it idled the host for 42 s. A part that has run past
//! [`stuck_after`](super::stuck_after) its size is sent again under the same
//! part number (see [`hedged`]); the store keeps one copy of a part number,
//! and both copies hold the same bytes, so either answer completes the
//! upload. One upload has at most [`HEDGES_IN_FLIGHT`] second attempts out
//! at once: on a link slow enough that every part passes its delay, a
//! second copy of each would only compete with the first copies for it.

use std::{
    fmt, mem,
    sync::{Arc, Mutex, PoisonError},
};

use async_trait::async_trait;
use object_store::{
    Error, MultipartUpload, PutPayload, PutResult, Result, UploadPart,
    multipart::{MultipartStore, PartId},
    path::Path,
};
use tokio::sync::Semaphore;

use super::hedged;

/// Second attempts one upload has in flight at once. One is enough for the
/// case hedging is for, a part the store holds while the others land; a
/// part that passes its delay while another's second attempt runs waits
/// for that one to settle before it sends its own.
const HEDGES_IN_FLIGHT: usize = 1;

/// A [`MultipartUpload`] over a [`MultipartStore`] that sends a part again
/// once it has run long enough to be taken for stuck.
pub(crate) struct HedgedUpload {
    store: Arc<dyn MultipartStore>,
    path: Path,
    id: String,
    /// Each part's id once it has landed, by part number.
    parts: Arc<Mutex<Vec<Option<PartId>>>>,
    /// The permits a part's second attempt is sent under.
    hedges: Arc<Semaphore>,
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
            hedges: Arc::new(Semaphore::new(HEDGES_IN_FLIGHT)),
            next: 0,
        })
    }
}

#[async_trait]
impl MultipartUpload for HedgedUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        let index = self.next;
        self.next += 1;
        let (store, path, id, parts, hedges) = (
            Arc::clone(&self.store),
            self.path.clone(),
            self.id.clone(),
            Arc::clone(&self.parts),
            Arc::clone(&self.hedges),
        );
        Box::pin(async move {
            let part = hedged_part(store.as_ref(), &path, &id, index, &hedges, data).await?;
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

/// Put part `index`, sending it again under one of `hedges`' permits once
/// the first attempt has run [`stuck_after`](super::stuck_after) its size
/// (see [`hedged`]).
async fn hedged_part(
    store: &dyn MultipartStore,
    path: &Path,
    id: &str,
    index: usize,
    hedges: &Semaphore,
    data: PutPayload,
) -> Result<PartId> {
    let id = id.to_string();
    hedged(data.content_length() as u64, Some(hedges), || {
        store.put_part(path, &id, index, data.clone())
    })
    .await
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        future::pending,
        sync::atomic::{AtomicUsize, Ordering as AtomicOrdering},
        time::Duration,
    };

    use bytes::Bytes;
    use futures::future::{join_all, try_join_all};
    use object_store::{MultipartId, ObjectStoreExt, memory::InMemory};
    use tokio::time::{sleep, timeout};

    use super::*;
    use crate::{
        runtime_metrics::io::UsageMeter,
        storage::{counting::wrap_multipart, stuck_after},
    };

    /// Bytes per part in the tests.
    const PART_BYTES: usize = 64;
    /// Parts in the tests' uploads.
    const PARTS: usize = 4;
    /// The part whose attempts the tests script.
    const SCRIPTED_PART: usize = 2;

    /// How one attempt at a part answers.
    #[derive(Debug, Clone, Copy)]
    enum Answer {
        /// Never answers.
        Hold,
        /// Lands after the given time.
        LandAfter(Duration),
        /// Fails after the given time.
        FailAfter(Duration),
    }

    /// How long a part of the tests' size runs before its hedge is sent.
    fn hedge_delay() -> Duration {
        stuck_after(PART_BYTES as u64)
    }

    /// A store that answers the attempts at [`SCRIPTED_PART`] in the order
    /// its script gives, lands every other attempt at once, and counts the
    /// attempts at each part. It keeps its own parts by number
    /// (object_store's in-memory store takes parts only in order, and a held
    /// part lands last) and writes the finished object to an in-memory store.
    #[derive(Debug)]
    struct ScriptedStore {
        inner: InMemory,
        parts: Mutex<BTreeMap<usize, Bytes>>,
        script: Mutex<Vec<Answer>>,
        /// When set, every part's first attempt lands after the first time
        /// and every later attempt after the second, and the script is not
        /// read.
        slow: Option<(Duration, Duration)>,
        attempts: Mutex<BTreeMap<usize, usize>>,
        /// Second or later attempts running now, and the most there were.
        hedges_running: AtomicUsize,
        most_hedges_running: AtomicUsize,
    }

    /// Counts one second attempt as running for as long as it lives, so an
    /// attempt dropped unanswered stops counting too.
    struct RunningHedge<'a>(&'a AtomicUsize);

    impl<'a> RunningHedge<'a> {
        fn start(store: &'a ScriptedStore) -> Self {
            let running = store.hedges_running.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            store
                .most_hedges_running
                .fetch_max(running, AtomicOrdering::SeqCst);
            Self(&store.hedges_running)
        }
    }

    impl Drop for RunningHedge<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, AtomicOrdering::SeqCst);
        }
    }

    impl ScriptedStore {
        /// A store whose attempts at [`SCRIPTED_PART`] answer as `script`
        /// says, first attempt first; attempts past the script land at once.
        fn new(script: &[Answer]) -> Arc<Self> {
            Self::build(script, None)
        }

        /// A store on which every part's first attempt lands after `first`,
        /// and every later attempt after `again`.
        fn slow(first: Duration, again: Duration) -> Arc<Self> {
            Self::build(&[], Some((first, again)))
        }

        fn build(script: &[Answer], slow: Option<(Duration, Duration)>) -> Arc<Self> {
            Arc::new(Self {
                inner: InMemory::new(),
                parts: Mutex::new(BTreeMap::new()),
                script: Mutex::new(script.iter().rev().copied().collect()),
                slow,
                attempts: Mutex::new(BTreeMap::new()),
                hedges_running: AtomicUsize::new(0),
                most_hedges_running: AtomicUsize::new(0),
            })
        }

        fn attempts_at(&self, part: usize) -> usize {
            self.attempts
                .lock()
                .expect("attempts")
                .get(&part)
                .copied()
                .unwrap_or(0)
        }

        fn total_attempts(&self) -> usize {
            self.attempts.lock().expect("attempts").values().sum()
        }
    }

    #[async_trait]
    impl MultipartStore for ScriptedStore {
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
            let attempt = {
                let mut attempts = self.attempts.lock().expect("attempts");
                let attempt = attempts.entry(part_idx).or_default();
                *attempt += 1;
                *attempt
            };
            let _hedge = (attempt > 1).then(|| RunningHedge::start(self));
            let answer = match (self.slow, part_idx == SCRIPTED_PART) {
                (Some((first, _)), _) if attempt == 1 => Some(Answer::LandAfter(first)),
                (Some((_, again)), _) => Some(Answer::LandAfter(again)),
                (None, true) => self.script.lock().expect("script").pop(),
                (None, false) => None,
            };
            match answer {
                None => {}
                Some(Answer::Hold) => pending::<()>().await,
                Some(Answer::LandAfter(after)) => sleep(after).await,
                Some(Answer::FailAfter(after)) => {
                    sleep(after).await;
                    return Err(Error::Generic {
                        store: "scripted",
                        source: format!("part {part_idx} refused").into(),
                    });
                }
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

    /// Upload [`PARTS`] parts to `path` on `store` and complete the upload.
    async fn upload(store: &Arc<ScriptedStore>, path: &Path) -> Result<PutResult> {
        let mut upload = HedgedUpload::start(Arc::clone(store) as _, path.clone()).await?;
        let puts: Vec<UploadPart> = (0..PARTS)
            .map(|index| upload.put_part(PutPayload::from_bytes(part(index))))
            .collect();
        try_join_all(puts).await?;
        upload.complete().await
    }

    /// The object at `path` holds every part's bytes in order.
    async fn assert_holds_every_part(store: &ScriptedStore, path: &Path) {
        let stored = store
            .inner
            .get(path)
            .await
            .expect("get")
            .bytes()
            .await
            .expect("bytes");
        let expected: Vec<u8> = (0..PARTS).flat_map(|index| part(index).to_vec()).collect();
        assert_eq!(stored.as_ref(), expected.as_slice());
    }

    /// An upload whose part never answers on its first attempt completes on
    /// its second, and the object holds every part's bytes in order.
    #[tokio::test(start_paused = true)]
    async fn a_held_part_is_answered_by_its_second_attempt() {
        let store = ScriptedStore::new(&[Answer::Hold]);
        let path = Path::from("hedged/object");
        upload(&store, &path).await.expect("upload");
        assert_eq!(store.total_attempts(), PARTS + 1, "one part sent twice");
        assert_holds_every_part(&store, &path).await;
    }

    /// A slow first attempt that lands after its hedge was sent answers the
    /// part, though the hedge never does: the upload completes in order. The
    /// time limit turns a wait on the held hedge into a failure, not a hang.
    #[tokio::test(start_paused = true)]
    async fn a_first_attempt_landing_after_the_hedge_answers_the_part() {
        let store = ScriptedStore::new(&[
            Answer::LandAfter(hedge_delay() + Duration::from_secs(1)),
            Answer::Hold,
        ]);
        let path = Path::from("hedged/first-lands-late");
        timeout(hedge_delay() * 10, upload(&store, &path))
            .await
            .expect("the first attempt answers the part without the hedge")
            .expect("upload");
        assert_eq!(store.attempts_at(SCRIPTED_PART), 2, "the hedge was sent");
        assert_holds_every_part(&store, &path).await;
    }

    /// On a link slow enough that every part passes its delay, the upload
    /// has one part's second attempt out at a time, not one for every part,
    /// and still completes in order.
    #[tokio::test(start_paused = true)]
    async fn a_slow_upload_has_one_hedge_in_flight_at_a_time() {
        let store = ScriptedStore::slow(hedge_delay() * 4, Duration::from_secs(1));
        let path = Path::from("hedged/slow-link");
        upload(&store, &path).await.expect("upload");
        assert_eq!(
            store.most_hedges_running.load(AtomicOrdering::SeqCst),
            HEDGES_IN_FLIGHT,
            "second attempts in flight at once"
        );
        assert!(
            store.total_attempts() > PARTS,
            "the parts past their delay were hedged"
        );
        assert_holds_every_part(&store, &path).await;
    }

    /// An upload whose parts all answer in time sends each part once.
    #[tokio::test(start_paused = true)]
    async fn parts_that_answer_in_time_are_sent_once() {
        let store = ScriptedStore::new(&[]);
        upload(&store, &Path::from("hedged/once"))
            .await
            .expect("upload");
        assert_eq!(store.total_attempts(), PARTS);
    }

    /// A first attempt that fails after the hedge was sent waits for the
    /// hedge, which lands the part: the upload completes in order.
    #[tokio::test(start_paused = true)]
    async fn a_first_attempt_failing_after_the_hedge_waits_for_the_hedge() {
        let store = ScriptedStore::new(&[
            Answer::FailAfter(hedge_delay() + Duration::from_secs(1)),
            Answer::LandAfter(Duration::from_secs(3)),
        ]);
        let path = Path::from("hedged/first-fails");
        upload(&store, &path).await.expect("the hedge lands");
        assert_eq!(store.attempts_at(SCRIPTED_PART), 2);
        assert_holds_every_part(&store, &path).await;
    }

    /// A hedge that fails at once does not cancel a slow first attempt
    /// that lands: the upload waits for it and completes in order.
    #[tokio::test(start_paused = true)]
    async fn a_failed_hedge_waits_for_the_first_attempt() {
        let store = ScriptedStore::new(&[
            Answer::LandAfter(hedge_delay() * 4),
            Answer::FailAfter(Duration::ZERO),
        ]);
        let path = Path::from("hedged/hedge-fails");
        upload(&store, &path)
            .await
            .expect("the first attempt lands");
        assert_eq!(store.attempts_at(SCRIPTED_PART), 2);
        assert_holds_every_part(&store, &path).await;
    }

    /// A part whose both attempts fail fails, and no third is sent.
    #[tokio::test(start_paused = true)]
    async fn a_part_whose_both_attempts_fail_fails() {
        let store = ScriptedStore::new(&[
            Answer::FailAfter(hedge_delay() + Duration::from_secs(1)),
            Answer::FailAfter(Duration::ZERO),
        ]);
        let mut upload = HedgedUpload::start(Arc::clone(&store) as _, Path::from("hedged/both"))
            .await
            .expect("start");
        let puts: Vec<UploadPart> = (0..PARTS)
            .map(|index| upload.put_part(PutPayload::from_bytes(part(index))))
            .collect();
        let answers = join_all(puts).await;
        for (index, answer) in answers.iter().enumerate() {
            assert_eq!(answer.is_err(), index == SCRIPTED_PART, "part {index}");
        }
        assert_eq!(store.attempts_at(SCRIPTED_PART), 2, "no third attempt");
    }

    /// A first attempt that fails before the hedge is due is the part's
    /// answer: the part fails and no hedge is sent.
    #[tokio::test(start_paused = true)]
    async fn a_first_attempt_failing_before_the_delay_is_not_hedged() {
        let store = ScriptedStore::new(&[Answer::FailAfter(hedge_delay() / 2)]);
        let mut upload = HedgedUpload::start(Arc::clone(&store) as _, Path::from("hedged/early"))
            .await
            .expect("start");
        for _ in 0..SCRIPTED_PART {
            upload
                .put_part(PutPayload::from_bytes(part(0)))
                .await
                .expect("part");
        }
        upload
            .put_part(PutPayload::from_bytes(part(SCRIPTED_PART)))
            .await
            .expect_err("the first attempt's failure is the answer");
        assert_eq!(store.attempts_at(SCRIPTED_PART), 1, "no hedge was sent");
    }

    /// `complete` refuses an upload one of whose parts never landed, and
    /// names it, rather than completing the object without it.
    #[tokio::test(start_paused = true)]
    async fn complete_reports_a_part_that_never_landed() {
        let store = ScriptedStore::new(&[Answer::FailAfter(Duration::ZERO)]);
        let path = Path::from("hedged/gap");
        let mut upload = HedgedUpload::start(Arc::clone(&store) as _, path.clone())
            .await
            .expect("start");
        let puts: Vec<UploadPart> = (0..PARTS)
            .map(|index| upload.put_part(PutPayload::from_bytes(part(index))))
            .collect();
        // The caller drops the failed part's error and completes anyway.
        let _ = join_all(puts).await;
        let err = upload.complete().await.expect_err("a part is missing");
        assert!(
            err.to_string()
                .contains(&format!("part {SCRIPTED_PART} of {path} never landed")),
            "{err}"
        );
        assert!(store.inner.get(&path).await.is_err(), "no object was made");
    }

    /// A hedged part is metered as the one PUT that landed it, the same as
    /// every other request in the ledger: a request that fails or is
    /// dropped unanswered records nothing.
    #[tokio::test(start_paused = true)]
    async fn a_hedged_part_is_metered_once() {
        let store = ScriptedStore::new(&[Answer::Hold]);
        let meter = UsageMeter::new();
        let hedged_upload =
            HedgedUpload::start(Arc::clone(&store) as _, Path::from("hedged/meter"))
                .await
                .expect("start");
        let mut upload = wrap_multipart(Box::new(hedged_upload), Arc::clone(&meter));
        let puts: Vec<UploadPart> = (0..PARTS)
            .map(|index| upload.put_part(PutPayload::from_bytes(part(index))))
            .collect();
        try_join_all(puts).await.expect("every part lands");
        upload.complete().await.expect("complete");
        assert_eq!(store.total_attempts(), PARTS + 1, "one part sent twice");
        let usage = meter.snapshot();
        assert_eq!(
            usage.put_count,
            PARTS as u64 + 1,
            "a PUT per part, and complete"
        );
        assert_eq!(usage.put_bytes, (PARTS * PART_BYTES) as u64);
    }
}
