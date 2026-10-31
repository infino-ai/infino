// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A table's kept term-index layer files: the slices and tier objects a
//! rebuild downloaded or wrote, on the table's disk and named by content
//! hash, so the next rebuild links them instead of downloading them again.
//!
//! A file takes its final name only after its bytes are synced, by a rename
//! from a temp name, so a crash never leaves a partial file under it. All of
//! it is best effort: an object the store cannot keep or link is downloaded,
//! as it would be without a store.

use std::{
    collections::HashSet,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

use tokio::{fs as async_fs, task};
use tracing::debug;

use super::{BuiltBytes, LAYER_FETCH_CONCURRENCY, Root, run_bounded};
use crate::supertable::manifest::part::ContentHash;

/// Extension of a kept slice.
pub(super) const SLICE_EXTENSION: &str = "slice";
/// Extension of a kept tier object.
pub(super) const TIER_EXTENSION: &str = "tier";
/// Extension of a file on its way into the store.
const TMP_EXTENSION: &str = "tmp";

/// Numbers each temp file in this process, so two keeps of one object
/// never write the same temp file.
static NEXT_TMP: AtomicU64 = AtomicU64::new(0);

/// Where the store keeps the object `hash`.
fn kept_path(store: &Path, hash: &ContentHash, extension: &str) -> PathBuf {
    store.join(format!("{}.{extension}", hash.to_hex()))
}

/// A temp name in `store` for one keep of the object `hash`.
fn tmp_path(store: &Path, hash: &ContentHash, extension: &str) -> PathBuf {
    let n = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
    store.join(format!(
        "{}.{extension}.{}.{n:x}.{TMP_EXTENSION}",
        hash.to_hex(),
        process::id()
    ))
}

/// Link the kept object `hash` to `dest`; false when the store does not
/// hold it.
pub(super) async fn link_kept(
    store: &Path,
    hash: &ContentHash,
    extension: &str,
    dest: &Path,
) -> bool {
    async_fs::hard_link(kept_path(store, hash, extension), dest)
        .await
        .is_ok()
}

/// Keep `file`, the object `hash`, in `store`.
pub(super) async fn keep_file(
    store: PathBuf,
    file: PathBuf,
    hash: ContentHash,
    extension: &'static str,
) {
    let kept = task::spawn_blocking(move || {
        keep_with(&store, &hash, extension, |tmp| {
            File::open(&file)?.sync_all()?;
            fs::hard_link(&file, tmp)
        })
    })
    .await;
    log_failure(kept, &hash);
}

/// Keep every slice and tier object of a build in `store`.
pub(crate) async fn keep_built(
    store: &Path,
    slices: &[(ContentHash, BuiltBytes)],
    tiers: &[(ContentHash, BuiltBytes)],
) {
    let objects = slices
        .iter()
        .map(|object| (object, SLICE_EXTENSION))
        .chain(tiers.iter().map(|object| (object, TIER_EXTENSION)));
    let jobs = objects.map(|((hash, built), extension)| {
        let (store, hash, built) = (store.to_path_buf(), *hash, built.clone());
        async move {
            let kept = task::spawn_blocking(move || {
                keep_with(&store, &hash, extension, |tmp| match &built {
                    BuiltBytes::Spilled { path, .. } => {
                        File::open(path)?.sync_all()?;
                        fs::hard_link(path, tmp)
                    }
                    BuiltBytes::Resident(bytes) => {
                        let mut file = File::create(tmp)?;
                        file.write_all(bytes)?;
                        file.sync_all()
                    }
                })
            })
            .await;
            log_failure(kept, &hash);
            Ok(())
        }
    });
    // Every job succeeds; a failed keep is logged inside it.
    let _ = run_bounded(LAYER_FETCH_CONCURRENCY, "layer keep", jobs).await;
    sync_dir(store).await;
}

/// Remove every file in `store` that `root` does not name.
pub(crate) async fn prune_layer_store(store: &Path, root: &Root) {
    let wanted: HashSet<String> = root
        .layers
        .iter()
        .flat_map(|layer| &layer.slices)
        .flat_map(|slice| {
            [
                Some((slice.content_hash, SLICE_EXTENSION)),
                slice.tier.map(|tier| (tier, TIER_EXTENSION)),
            ]
        })
        .flatten()
        .map(|(hash, extension)| format!("{}.{extension}", hash.to_hex()))
        .collect();
    let store = store.to_path_buf();
    let pruned = task::spawn_blocking(move || -> io::Result<()> {
        for entry in fs::read_dir(&store)? {
            let entry = entry?;
            if !wanted.contains(entry.file_name().to_string_lossy().as_ref()) {
                fs::remove_file(entry.path())?;
            }
        }
        Ok(())
    })
    .await;
    match pruned {
        Ok(Ok(())) => {}
        Ok(Err(e)) if e.kind() == io::ErrorKind::NotFound => {}
        Ok(Err(e)) => debug!("term-index layer store: prune failed: {e}"),
        Err(e) => debug!("term-index layer store: prune task failed: {e}"),
    }
}

/// Make the names this process gave files in `store` durable.
pub(super) async fn sync_dir(store: &Path) {
    let store = store.to_path_buf();
    let _ = task::spawn_blocking(move || File::open(store)?.sync_all()).await;
}

/// Place the object `hash` in `store`: `write` puts its synced bytes at a
/// temp path, which is then renamed to the object's name. Does nothing when
/// the store already holds it.
fn keep_with(
    store: &Path,
    hash: &ContentHash,
    extension: &str,
    write: impl FnOnce(&Path) -> io::Result<()>,
) -> io::Result<()> {
    let kept = kept_path(store, hash, extension);
    if kept.exists() {
        return Ok(());
    }
    fs::create_dir_all(store)?;
    let tmp = tmp_path(store, hash, extension);
    let placed = write(&tmp).and_then(|()| fs::rename(&tmp, &kept));
    if placed.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    placed
}

fn log_failure(kept: Result<io::Result<()>, task::JoinError>, hash: &ContentHash) {
    match kept {
        Ok(Ok(())) => {}
        Ok(Err(e)) => debug!(hash = %hash.to_hex(), "term-index layer store: keep failed: {e}"),
        Err(e) => debug!(hash = %hash.to_hex(), "term-index layer store: keep task failed: {e}"),
    }
}
