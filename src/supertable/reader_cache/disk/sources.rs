// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! mmap primitives: map a finished cache file read-only and hand it out as
//! shared [`Bytes`] that both the reader and the cache entry hold.

use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use bytes::Bytes;
use memmap2::Mmap;
use tempfile::NamedTempFile;

/// Newtype around `Arc<Mmap>` that delegates `AsRef<[u8]>`
/// to the underlying `Mmap`. Lets the cache's `mmap: Arc<Mmap>`
/// field and the reader's `Bytes::from_owner(...)` share the
/// same `Arc<Mmap>` — both refer to the same OS mapping, so
/// `madvise` on the cache's handle affects the reader's
/// resident pages (the idle-threshold sweep relies on this).
pub(crate) struct ArcMmapOwner(pub(crate) Arc<Mmap>);

impl AsRef<[u8]> for ArcMmapOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

fn open_readonly_mmap(path: &Path) -> io::Result<Mmap> {
    let file = fs::File::open(path)?;
    // SAFETY: a mapping is only unsound if the file's bytes change or the
    // file shrinks under it, and every file mapped here is complete and
    // immutable once mapped. A cache file or a merge's temp output is
    // created, filled and fsync'd by this process before the map, and never
    // written again. A superfile or hidden-index file in a table's local
    // storage directory is written by the storage provider to a staged file
    // and renamed into place, so a reader only ever sees a finished file; a
    // later write to the same path is another rename, which leaves the
    // mapped inode as it was, and removal is an unlink. Unlinking or
    // renaming over a mapped file is safe under POSIX: the inode stays until
    // the last mapping drops, so eviction, publish and garbage collection
    // can all proceed while a reader holds the map. Nothing truncates or
    // writes a file in place.
    unsafe { Mmap::map(&file) }
}

/// Open a completed local file as zero-copy mmap-backed [`Bytes`].
///
/// For a finished file outside the cache: a superfile or hidden-index file in
/// a table's local storage directory, which compaction and the vector index
/// read where it lies rather than copying it into the cache first. Keeping
/// the unsafe mmap construction in this module preserves the repository's
/// documented mmap safety boundary; see [`open_readonly_mmap`] for why a file
/// there may be mapped.
pub(crate) fn mmap_readonly_bytes(path: &Path) -> io::Result<Bytes> {
    Ok(mmap_readonly_with_handle(path)?.1)
}

/// Read-only mmap `path`, returning both the mapping handle and `Bytes` that own a clone of it.
///
/// The promote path keeps the `Arc<Mmap>` for the entry's `mmap` field (so the idle sweep's
/// `madvise` reaches the reader's resident pages) and hands the `Bytes` to the reader. Both refer
/// to one OS mapping.
pub(crate) fn mmap_readonly_with_handle(path: &Path) -> io::Result<(Arc<Mmap>, Bytes)> {
    let mmap = Arc::new(open_readonly_mmap(path)?);
    let bytes = Bytes::from_owner(ArcMmapOwner(Arc::clone(&mmap)));
    Ok((mmap, bytes))
}

/// The sealed files behind live temp-backed [`Bytes`], keyed by the
/// mapping's address, with the mapping's length: where a publish to
/// local-filesystem storage finds the file to link into place instead of
/// copying the bytes out again. An entry lives exactly as long as the
/// bytes do.
static TEMP_BACKED: OnceLock<Mutex<HashMap<usize, (PathBuf, usize)>>> = OnceLock::new();

fn temp_backed() -> &'static Mutex<HashMap<usize, (PathBuf, usize)>> {
    TEMP_BACKED.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Owner of temp-backed bytes: the mapping and the file it maps, which is
/// unlinked only when the last clone of the bytes drops.
struct TempBackedOwner {
    mmap: Arc<Mmap>,
    file: NamedTempFile,
}

impl AsRef<[u8]> for TempBackedOwner {
    fn as_ref(&self) -> &[u8] {
        self.mmap.as_ref()
    }
}

impl Drop for TempBackedOwner {
    fn drop(&mut self) {
        if let Ok(mut map) = temp_backed().lock() {
            map.remove(&(self.mmap.as_ptr() as usize));
        }
        // `file` unlinks itself after this.
    }
}

/// Map a finished temp file read-only as zero-copy [`Bytes`] that keep the
/// file alive, so a publish to local-filesystem storage can link that file
/// into place ([`temp_file_behind`]) rather than write its bytes a second
/// time. The ordinary [`mmap_readonly_bytes`] drops the file at once.
pub(crate) fn mmap_temp_file_bytes(file: NamedTempFile) -> io::Result<Bytes> {
    let mmap = Arc::new(open_readonly_mmap(file.path())?);
    let owner = TempBackedOwner { mmap, file };
    if let Ok(mut map) = temp_backed().lock() {
        map.insert(
            owner.mmap.as_ptr() as usize,
            (owner.file.path().to_path_buf(), owner.mmap.len()),
        );
    }
    Ok(Bytes::from_owner(owner))
}

/// The still-present temp file `bytes` were mapped from by
/// [`mmap_temp_file_bytes`], for the whole (unsliced) bytes; `None` for any
/// other bytes, a slice of the mapping included: a prefix starts at the
/// same address, so the length is checked too.
pub(crate) fn temp_file_behind(bytes: &Bytes) -> Option<PathBuf> {
    let map = temp_backed().lock().ok()?;
    let (path, len) = map.get(&(bytes.as_ptr() as usize))?;
    (*len == bytes.len()).then(|| path.clone())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::{NamedTempFile, TempDir};

    use super::*;

    /// The sealed file is found behind the whole mapping only — a slice
    /// of it would link the whole file for a part — and is forgotten with
    /// the bytes.
    #[test]
    fn temp_file_behind_the_whole_mapping_only_while_the_bytes_live() {
        let dir = TempDir::new().expect("tempdir");
        let mut file = NamedTempFile::new_in(dir.path()).expect("temp file");
        file.write_all(b"sealed superfile bytes").expect("write");
        file.flush().expect("flush");
        let path = file.path().to_path_buf();

        let bytes = mmap_temp_file_bytes(file).expect("map");
        assert_eq!(temp_file_behind(&bytes).as_deref(), Some(path.as_path()));
        assert!(
            temp_file_behind(&bytes.clone()).is_some(),
            "a clone is the same mapping"
        );
        assert!(
            temp_file_behind(&bytes.slice(1..)).is_none(),
            "an offset slice"
        );
        assert!(
            temp_file_behind(&bytes.slice(..bytes.len() - 1)).is_none(),
            "a prefix starts at the same address"
        );
        assert!(temp_file_behind(&Bytes::from_static(b"other")).is_none());

        drop(bytes);
        assert!(!path.exists(), "the temp file goes with the bytes");
    }
}
