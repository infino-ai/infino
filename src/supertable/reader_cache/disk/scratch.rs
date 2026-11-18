// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The scratch folder beside a disk cache, where a table's builds keep their
//! temporary files.
//!
//! ```text
//! <cache_root>/scratch/
//!     run-<pid>-<n>/        one open store's temporary files
//!     run-<pid>-<n>.lock    held for as long as that store is open
//!     ...                   work a later process resumes (a drain's spill)
//! ```
//!
//! A store keeps its temporary files in its own `run-` folder and holds a
//! lock on the file beside it. A process that dies (killed, out of memory)
//! leaves both behind with the lock free, which is how the next store to
//! open tells the folder is no one's and removes it. A folder whose lock is
//! held belongs to a live store, in this process or another, and is kept.
//!
//! On a filesystem without file locks a store keeps no lock file, so its
//! folder is never taken for an exited process's; it is removed when the
//! store is dropped and stays behind if the process dies first.

use std::{
    fs::{self, File, TryLockError},
    io,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

/// Directory, under the cache root, where the table's builds keep their
/// temporary files.
const SCRATCH_SUBDIR: &str = "scratch";

/// Name prefix of a store's own folder of temporary files, and of the lock
/// file beside it.
const RUN_PREFIX: &str = "run-";

/// Name suffix of the lock file beside a store's `run-` folder.
const LOCK_SUFFIX: &str = ".lock";

/// Name prefix of a lock file being created, before it is locked and given
/// its `run-` name, so a sweep never sees an unlocked lock of a live store.
const NEW_LOCK_PREFIX: &str = ".new-";

/// How many times a store tries to make its lock file. A sweep in another
/// process can hold or remove the file for a moment while it is being made.
const LOCK_ATTEMPTS: usize = 3;

/// Numbers the `run-` folders of the stores this process opens.
static NEXT_RUN: AtomicU64 = AtomicU64::new(0);

/// The scratch folder beside the cache at `cache_root`.
pub(super) fn scratch_root(cache_root: &Path) -> PathBuf {
    cache_root.join(SCRATCH_SUBDIR)
}

/// One open store's folder of temporary files, removed when the store is
/// dropped.
pub(crate) struct RunScratch {
    dir: PathBuf,
    lock_path: PathBuf,
    /// Holds the lock that marks the folder as a live store's. `None` on a
    /// filesystem without file locks.
    _lock: Option<File>,
}

impl RunScratch {
    /// Remove the `run-` folders under `scratch` that exited processes left,
    /// then make this store's own.
    pub(crate) fn open(scratch: &Path) -> io::Result<Self> {
        fs::create_dir_all(scratch)?;
        remove_orphaned_runs(scratch);
        let name = format!(
            "{RUN_PREFIX}{}-{}",
            process::id(),
            NEXT_RUN.fetch_add(1, Ordering::Relaxed)
        );
        let lock_path = scratch.join(format!("{name}{LOCK_SUFFIX}"));
        let lock = create_lock(scratch, &name, &lock_path)?;
        let dir = scratch.join(name);
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            lock_path,
            _lock: lock,
        })
    }

    /// The folder. Made on each call, so a build still has it after the
    /// cache folder was cleared under an open table. A failure shows up when
    /// the build creates its first file there.
    pub(crate) fn dir(&self) -> PathBuf {
        let _ = fs::create_dir_all(&self.dir);
        self.dir.clone()
    }
}

impl Drop for RunScratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
        let _ = fs::remove_file(&self.lock_path);
    }
}

/// Make the lock file at `lock_path` and return it locked, or `None` when
/// the filesystem has no file locks. The file is locked under a `.new-` name
/// and then renamed, so a sweep never sees it unlocked under its `run-`
/// name.
fn create_lock(scratch: &Path, name: &str, lock_path: &Path) -> io::Result<Option<File>> {
    let new_lock_path = scratch.join(format!("{NEW_LOCK_PREFIX}{name}{LOCK_SUFFIX}"));
    for _ in 0..LOCK_ATTEMPTS {
        let lock = File::create(&new_lock_path)?;
        match lock.try_lock() {
            Ok(()) => {}
            // A sweep is looking at the file and may remove it.
            Err(TryLockError::WouldBlock) => continue,
            Err(TryLockError::Error(error)) => {
                tracing::debug!(
                    path = %scratch.display(),
                    %error,
                    "no file locks here; temporary files left by a crash will not be removed"
                );
                let _ = fs::remove_file(&new_lock_path);
                return Ok(None);
            }
        }
        match fs::rename(&new_lock_path, lock_path) {
            Ok(()) => return Ok(Some(lock)),
            // A sweep removed the file before it was locked.
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    let _ = fs::remove_file(&new_lock_path);
    Ok(None)
}

/// Delete every `run-` folder under `scratch` whose lock is free, meaning
/// the process that made it is gone, and every lock file such a process
/// left half-made. Best-effort: a lock that cannot be opened or taken, and
/// a folder that cannot be removed, are left for a later call.
fn remove_orphaned_runs(scratch: &Path) {
    let Ok(listing) = fs::read_dir(scratch) else {
        return;
    };
    for entry in listing.flatten() {
        let file_name = entry.file_name();
        let Some(locked_name) = file_name
            .to_str()
            .and_then(|name| name.strip_suffix(LOCK_SUFFIX))
        else {
            continue;
        };
        let half_made = locked_name.starts_with(NEW_LOCK_PREFIX);
        if !half_made && !locked_name.starts_with(RUN_PREFIX) {
            continue;
        }
        let lock_path = entry.path();
        let Ok(lock) = File::open(&lock_path) else {
            continue;
        };
        if lock.try_lock().is_err() {
            continue;
        }
        if half_made {
            let _ = fs::remove_file(&lock_path);
            continue;
        }
        let run = locked_name;
        let dir = scratch.join(run);
        match fs::remove_dir_all(&dir) {
            Ok(()) => tracing::info!(
                path = %dir.display(),
                "removed the temporary files an exited process left"
            ),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                tracing::debug!(
                    path = %dir.display(),
                    %error,
                    "could not remove an exited process's temporary files"
                );
                continue;
            }
        }
        // The lock is still held here, so no other sweep is removing the
        // same folder; the lock file goes last.
        let _ = fs::remove_file(&lock_path);
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    /// A store's folder exists while it is open, holds no file of its own,
    /// and is removed with its lock when the store is dropped.
    #[test]
    fn a_run_folder_lives_as_long_as_its_store() {
        let root = tempdir().expect("tempdir");
        let scratch = scratch_root(root.path());
        let run = RunScratch::open(&scratch).expect("open");
        let dir = run.dir();
        assert!(dir.starts_with(&scratch));
        assert_eq!(fs::read_dir(&dir).expect("read run folder").count(), 0);

        drop(run);
        assert!(!dir.exists(), "the folder goes with the store");
        assert_eq!(
            fs::read_dir(&scratch).expect("read scratch").count(),
            0,
            "the lock goes too"
        );
    }

    /// Opening a store removes the folder of a store that is gone (its lock
    /// is free) and keeps the folder of one that is still open, along with
    /// anything that is not a `run-` folder.
    #[test]
    fn opening_removes_the_folders_exited_processes_left() {
        let root = tempdir().expect("tempdir");
        let scratch = scratch_root(root.path());
        fs::create_dir_all(&scratch).expect("scratch");
        // What a killed process leaves: its folder, a file in it, and its
        // lock file with no one holding the lock.
        let orphan = scratch.join(format!("{RUN_PREFIX}orphan"));
        fs::create_dir_all(&orphan).expect("orphan folder");
        fs::write(orphan.join("half-written"), b"bytes").expect("orphan file");
        let orphan_lock = scratch.join(format!("{RUN_PREFIX}orphan{LOCK_SUFFIX}"));
        fs::write(&orphan_lock, b"").expect("orphan lock");
        // A process killed while making its lock leaves the half-made file.
        let half_made_lock =
            scratch.join(format!("{NEW_LOCK_PREFIX}{RUN_PREFIX}orphan{LOCK_SUFFIX}"));
        fs::write(&half_made_lock, b"").expect("half-made lock");
        // Work a later process resumes lives beside the `run-` folders.
        let resumable = scratch.join("resumable");
        fs::create_dir_all(&resumable).expect("resumable folder");
        let live = RunScratch::open(&scratch).expect("live store");
        let live_dir = live.dir();
        fs::write(live_dir.join("in-use"), b"bytes").expect("live file");

        let second = RunScratch::open(&scratch).expect("second store");

        assert!(!orphan.exists(), "an exited process's folder is removed");
        assert!(!orphan_lock.exists(), "and its lock file");
        assert!(!half_made_lock.exists(), "and a half-made lock file");
        assert!(live_dir.join("in-use").exists(), "a live store's is kept");
        assert!(resumable.exists(), "other folders are kept");
        assert_ne!(second.dir(), live_dir, "each store has its own folder");
    }
}
