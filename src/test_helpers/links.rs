// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Assertions about superfiles published to a directory on this host.

use std::{fs, os::unix::fs::MetadataExt};

use crate::{
    storage::StorageProvider,
    supertable::{manifest::SuperfileEntry, reader_cache::DiskCacheStore},
};

/// Every entry's storage copy and cache copy are one file: the sealed
/// superfile was linked into both places, never written again.
pub fn assert_cache_copies_are_links<'a>(
    storage: &dyn StorageProvider,
    cache: &DiskCacheStore,
    entries: impl IntoIterator<Item = &'a SuperfileEntry>,
) {
    for entry in entries {
        let stored = storage
            .local_path(&entry.storage_path())
            .expect("a directory on this host");
        let stored = fs::metadata(&stored).expect("storage file");
        let cached = fs::metadata(cache.cache_path(&entry.uri)).expect("cache file");
        assert_eq!(stored.ino(), cached.ino(), "one file behind both copies");
        assert!(stored.nlink() >= 2, "linked, not copied");
    }
}
