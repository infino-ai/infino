// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Bounded caches of decoded data, and the byte-bounded LRU they share.
//!
//! Superfiles are immutable, so a `(uri, projected columns, local rows)`
//! materialization is stable for the process lifetime. Caching that small
//! result avoids decompressing the same Parquet pages on repeated search
//! fetches while keeping memory independent of table size. The SQL scan's
//! decoded columns ([`super::column_cache`]) are held the same way.

use std::{
    hash::Hash,
    sync::{Arc, Mutex},
};

use arrow_array::RecordBatch;
use rustc_hash::FxHashMap;

use crate::supertable::manifest::SuperfileUri;

/// Process-local decoded top-k cache budget per supertable handle.
const DEFAULT_DECODED_SCALAR_CACHE_BYTES: usize = 64 * 1024 * 1024;

/// Decoded values by key, bounded by their bytes: an insert that would pass
/// `max_bytes` evicts the least recently used entries first, and a value
/// larger than the whole budget is not kept.
///
/// A lookup, an insert and each eviction cost the same however many values
/// are kept, so the lock every reader takes is held only that long: the
/// values sit in slots linked from most to least recently used, found by key
/// through a hash index.
#[derive(Debug)]
pub(crate) struct ByteLru<K, V> {
    max_bytes: usize,
    state: Mutex<LruState<K, V>>,
}

#[derive(Debug)]
struct LruState<K, V> {
    /// Each key's slot. Fx-hashed: a decoded-row key holds every row of a
    /// take, millions on a wide match, where SipHash over them was 6% of a
    /// 2.26M-row match's time.
    index: FxHashMap<Arc<K>, usize>,
    /// The kept values; a freed slot is `None` until reused.
    slots: Vec<Option<LruSlot<K, V>>>,
    /// Slots freed by evictions, reused before `slots` grows.
    free: Vec<usize>,
    /// The most recently used slot, and the least.
    newest: Option<usize>,
    oldest: Option<usize>,
    bytes: usize,
}

#[derive(Debug)]
struct LruSlot<K, V> {
    /// Shared with the index, so a key is held once however large.
    key: Arc<K>,
    value: V,
    bytes: usize,
    /// The slot used just before this one, and just after.
    newer: Option<usize>,
    older: Option<usize>,
}

impl<K: Eq + Hash, V: Clone> ByteLru<K, V> {
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            state: Mutex::new(LruState {
                index: FxHashMap::default(),
                slots: Vec::new(),
                free: Vec::new(),
                newest: None,
                oldest: None,
                bytes: 0,
            }),
        }
    }

    /// The most bytes the values may hold together.
    pub(crate) fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// The value under `key`, marking it used.
    pub(crate) fn get(&self, key: &K) -> Option<V> {
        let mut state = self.state.lock().expect("decoded cache poisoned");
        let at = *state.index.get(key)?;
        state.unlink(at);
        state.link_newest(at);
        state.slots[at].as_ref().map(|slot| slot.value.clone())
    }

    /// Keep `value`, which holds `bytes`, under `key`.
    pub(crate) fn insert(&self, key: K, value: V, bytes: usize) {
        if bytes > self.max_bytes {
            return;
        }
        let mut state = self.state.lock().expect("decoded cache poisoned");
        if let Some(&at) = state.index.get(&key) {
            state.remove(at);
        }
        state.push(key, value, bytes, self.max_bytes);
    }

    /// The value already under `key`, marking it used, or else `value`,
    /// kept under `key` as [`Self::insert`] keeps it. The first value kept
    /// wins, so callers racing to fill a key all share one value.
    pub(crate) fn get_or_insert(&self, key: K, value: V, bytes: usize) -> V {
        let mut state = self.state.lock().expect("decoded cache poisoned");
        if let Some(&at) = state.index.get(&key) {
            state.unlink(at);
            state.link_newest(at);
            if let Some(slot) = state.slots[at].as_ref() {
                return slot.value.clone();
            }
        }
        if bytes <= self.max_bytes {
            state.push(key, value.clone(), bytes, self.max_bytes);
        }
        value
    }

    /// How many values are kept.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.state
            .lock()
            .expect("decoded cache poisoned")
            .index
            .len()
    }
}

impl<K: Eq + Hash, V> LruState<K, V> {
    /// Keeps `value` under `key`, which is not kept yet, as the newest,
    /// first evicting the least recently used until `bytes` fits in
    /// `max_bytes`.
    fn push(&mut self, key: K, value: V, bytes: usize, max_bytes: usize) {
        while self.bytes.saturating_add(bytes) > max_bytes {
            let Some(oldest) = self.oldest else {
                break;
            };
            self.remove(oldest);
        }
        let key = Arc::new(key);
        let slot = LruSlot {
            key: Arc::clone(&key),
            value,
            bytes,
            newer: None,
            older: None,
        };
        let at = match self.free.pop() {
            Some(at) => {
                self.slots[at] = Some(slot);
                at
            }
            None => {
                self.slots.push(Some(slot));
                self.slots.len() - 1
            }
        };
        self.link_newest(at);
        self.index.insert(key, at);
        self.bytes = self.bytes.saturating_add(bytes);
    }

    /// Takes slot `at` out of the recency list, joining its neighbours.
    fn unlink(&mut self, at: usize) {
        let Some((newer, older)) = self.slots[at].as_ref().map(|slot| (slot.newer, slot.older))
        else {
            return;
        };
        match newer {
            Some(newer) => self.set_older(newer, older),
            None => self.newest = older,
        }
        match older {
            Some(older) => self.set_newer(older, newer),
            None => self.oldest = newer,
        }
    }

    /// Puts slot `at` at the newest end of the recency list.
    fn link_newest(&mut self, at: usize) {
        let previous = self.newest;
        if let Some(slot) = self.slots[at].as_mut() {
            slot.newer = None;
            slot.older = previous;
        }
        if let Some(previous) = previous {
            self.set_newer(previous, Some(at));
        }
        self.newest = Some(at);
        if self.oldest.is_none() {
            self.oldest = Some(at);
        }
    }

    /// Drops the value in slot `at`, freeing the slot.
    fn remove(&mut self, at: usize) {
        self.unlink(at);
        if let Some(slot) = self.slots[at].take() {
            self.index.remove(&slot.key);
            self.bytes = self.bytes.saturating_sub(slot.bytes);
            self.free.push(at);
        }
    }

    fn set_newer(&mut self, at: usize, newer: Option<usize>) {
        if let Some(slot) = self.slots[at].as_mut() {
            slot.newer = newer;
        }
    }

    fn set_older(&mut self, at: usize, older: Option<usize>) {
        if let Some(slot) = self.slots[at].as_mut() {
            slot.older = older;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    uri: SuperfileUri,
    /// The schema the rows were decoded under. A column's name, type and
    /// identity all belong to a generation of the schema, so a cached
    /// batch answers only for the generation that produced it.
    schema_id: u32,
    columns: Box<[String]>,
    local_doc_ids: Box<[u32]>,
}

#[derive(Debug)]
pub(crate) struct DecodedScalarCache {
    lru: ByteLru<CacheKey, RecordBatch>,
}

impl Default for DecodedScalarCache {
    fn default() -> Self {
        Self::new(DEFAULT_DECODED_SCALAR_CACHE_BYTES)
    }
}

impl DecodedScalarCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            lru: ByteLru::new(max_bytes),
        }
    }

    fn key(uri: SuperfileUri, schema_id: u32, local_doc_ids: &[u32], columns: &[&str]) -> CacheKey {
        CacheKey {
            uri,
            schema_id,
            columns: columns.iter().map(|column| (*column).to_string()).collect(),
            local_doc_ids: local_doc_ids.into(),
        }
    }

    pub(crate) fn get(
        &self,
        uri: SuperfileUri,
        schema_id: u32,
        local_doc_ids: &[u32],
        columns: &[&str],
    ) -> Option<RecordBatch> {
        self.lru
            .get(&Self::key(uri, schema_id, local_doc_ids, columns))
    }

    pub(crate) fn insert(
        &self,
        uri: SuperfileUri,
        schema_id: u32,
        local_doc_ids: &[u32],
        columns: &[&str],
        batch: RecordBatch,
    ) {
        let bytes = batch.get_array_memory_size();
        self.lru.insert(
            Self::key(uri, schema_id, local_doc_ids, columns),
            batch,
            bytes,
        );
    }
}

#[cfg(test)]
mod tests {
    /// The schema generation these tests decode under; the cache keys on
    /// it, and one generation is enough to exercise everything else.
    const SCHEMA_ID: u32 = 1;

    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;

    /// Ample budget for the small cache-hit fixture.
    const TEST_CACHE_BYTES: usize = 1024;

    fn batch(values: &[i64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(values.to_vec())) as ArrayRef],
        )
        .expect("batch")
    }

    #[test]
    fn repeated_key_returns_cached_batch() {
        let cache = DecodedScalarCache::new(TEST_CACHE_BYTES);
        let uri = SuperfileUri::new_v4();
        cache.insert(uri, SCHEMA_ID, &[7, 3], &["value"], batch(&[70, 30]));
        let cached = cache
            .get(uri, SCHEMA_ID, &[7, 3], &["value"])
            .expect("cache hit");
        assert_eq!(cached.num_rows(), 2);
        assert!(cache.get(uri, SCHEMA_ID, &[3, 7], &["value"]).is_none());
    }

    /// Bytes each value of the recency tests counts as.
    const VALUE_BYTES: usize = 10;
    /// Values the recency tests' budget holds.
    const KEPT_VALUES: usize = 3;

    /// A lookup makes a value the newest, so the next eviction takes the
    /// value used longest ago instead; replacing a key frees its old bytes;
    /// and slots freed by evictions are reused.
    #[test]
    fn eviction_takes_the_least_recently_used_value() {
        let lru: ByteLru<u32, u32> = ByteLru::new(VALUE_BYTES * KEPT_VALUES);
        for key in 0..3 {
            lru.insert(key, key, VALUE_BYTES);
        }
        assert_eq!(lru.get(&0), Some(0), "0 is now the newest");
        lru.insert(3, 3, VALUE_BYTES);
        assert_eq!(lru.get(&1), None, "1 was used longest ago");
        assert_eq!(lru.get(&0), Some(0));
        assert_eq!(lru.get(&2), Some(2));
        assert_eq!(lru.get(&3), Some(3));

        lru.insert(2, 20, VALUE_BYTES);
        assert_eq!(lru.len(), KEPT_VALUES, "a replaced key is kept once");
        assert_eq!(lru.get(&2), Some(20));

        // One value the size of the whole budget evicts every other.
        lru.insert(9, 9, VALUE_BYTES * KEPT_VALUES);
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.get(&9), Some(9));
        for key in 10..13 {
            lru.insert(key, key, VALUE_BYTES);
        }
        assert_eq!(lru.len(), KEPT_VALUES);
        assert_eq!(
            lru.state.lock().expect("lock").slots.len(),
            KEPT_VALUES,
            "freed slots are reused"
        );
    }

    /// `get_or_insert` keeps the first value under a key, so racing fills
    /// all share it, marks a hit used, and evicts the least recently used
    /// past the bound, as the lazy reader cache relies on with unit weights.
    #[test]
    fn get_or_insert_keeps_the_first_value_and_drops_the_least_recently_used() {
        let lru: ByteLru<u32, Arc<u32>> = ByteLru::new(2);
        assert!(lru.get(&1).is_none());
        let first = lru.get_or_insert(1, Arc::new(10), 1);
        // A racing fill's value is not kept: every caller gets the first.
        let raced = lru.get_or_insert(1, Arc::new(11), 1);
        assert!(Arc::ptr_eq(&first, &raced));
        assert!(Arc::ptr_eq(&first, &lru.get(&1).expect("kept")));
        lru.get_or_insert(2, Arc::new(20), 1);
        // The racing fill above made 1 the newest, so 2 goes next.
        lru.get_or_insert(1, Arc::new(12), 1);
        lru.get_or_insert(3, Arc::new(30), 1);
        assert!(lru.get(&2).is_none(), "the least recently used goes");
        assert_eq!(lru.get(&1).as_deref(), Some(&10));
        assert_eq!(lru.get(&3).as_deref(), Some(&30));
        // A value larger than the whole budget is handed back but not kept.
        assert_eq!(*lru.get_or_insert(4, Arc::new(40), 3), 40);
        assert!(lru.get(&4).is_none());
        assert_eq!(lru.len(), 2);
    }

    #[test]
    fn insertion_evicts_oldest_entry_at_budget() {
        let one = batch(&[1]);
        let budget = one.get_array_memory_size();
        let cache = DecodedScalarCache::new(budget);
        let first = SuperfileUri::new_v4();
        let second = SuperfileUri::new_v4();
        cache.insert(first, SCHEMA_ID, &[0], &["value"], one);
        cache.insert(second, SCHEMA_ID, &[0], &["value"], batch(&[2]));
        assert!(cache.get(first, SCHEMA_ID, &[0], &["value"]).is_none());
        assert!(cache.get(second, SCHEMA_ID, &[0], &["value"]).is_some());
    }
}
