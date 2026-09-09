// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! mmap primitives: map a finished cache file read-only and hand it out as
//! shared [`Bytes`] that both the reader and the cache entry hold.

use std::{fs, io, path::Path, sync::Arc};

use bytes::Bytes;
use memmap2::Mmap;

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
    // SAFETY: the cache file is created + filled + fsync'd
    // before this mmap call. The file is owned by us; no
    // other process modifies it. Once mmap'd we never write
    // to it (eviction unlinks + drops the Arc<Mmap>, which
    // unmaps cleanly under POSIX even if the file's already
    // unlinked).
    unsafe { Mmap::map(&file) }
}

/// Open a completed local superfile as zero-copy mmap-backed [`Bytes`].
///
/// Drain assembles very large packed shards in temporary files and maps the
/// finished file through this helper before handing it to the ordinary
/// `prepare_superfile`/publish path. Keeping the unsafe mmap construction in
/// this module preserves the repository's documented mmap safety boundary.
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
