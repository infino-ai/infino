// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Bounded caches of decoded data, and the byte-bounded LRU they share.
//!
//! Superfiles are immutable, so a `(uri, projected columns, local rows)`
//! materialization is stable for the process lifetime. Caching that small
//! result avoids decompressing the same Parquet pages on repeated search
//! fetches while keeping memory independent of table size. The SQL scan's
//! decoded columns ([`super::column_cache`]) are held the same way.

use std::{collections::HashMap, hash::Hash, sync::Mutex};

use arrow_array::RecordBatch;

use crate::supertable::manifest::SuperfileUri;

/// Process-local decoded top-k cache budget per supertable handle.
const DEFAULT_DECODED_SCALAR_CACHE_BYTES: usize = 64 * 1024 * 1024;

/// Decoded values by key, bounded by their bytes: an insert that would pass
/// `max_bytes` evicts the least recently used entries first, and a value
/// larger than the whole budget is not kept.
#[derive(Debug)]
pub(crate) struct ByteLru<K, V> {
    max_bytes: usize,
    state: Mutex<LruState<K, V>>,
}

#[derive(Debug)]
struct LruState<K, V> {
    entries: HashMap<K, LruEntry<V>>,
    bytes: usize,
    tick: u64,
}

#[derive(Debug)]
struct LruEntry<V> {
    value: V,
    bytes: usize,
    last_access: u64,
}

impl<K: Clone + Eq + Hash, V: Clone> ByteLru<K, V> {
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            state: Mutex::new(LruState {
                entries: HashMap::new(),
                bytes: 0,
                tick: 0,
            }),
        }
    }

    /// The value under `key`, marking it used.
    pub(crate) fn get(&self, key: &K) -> Option<V> {
        let mut state = self.state.lock().expect("decoded cache poisoned");
        state.tick = state.tick.wrapping_add(1);
        let tick = state.tick;
        let entry = state.entries.get_mut(key)?;
        entry.last_access = tick;
        Some(entry.value.clone())
    }

    /// Keep `value`, which holds `bytes`, under `key`.
    pub(crate) fn insert(&self, key: K, value: V, bytes: usize) {
        if bytes > self.max_bytes {
            return;
        }
        let mut state = self.state.lock().expect("decoded cache poisoned");
        state.tick = state.tick.wrapping_add(1);
        let tick = state.tick;
        if let Some(previous) = state.entries.remove(&key) {
            state.bytes = state.bytes.saturating_sub(previous.bytes);
        }
        while state.bytes.saturating_add(bytes) > self.max_bytes && !state.entries.is_empty() {
            let oldest = state
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(key, _)| key.clone())
                .expect("non-empty cache has an oldest entry");
            if let Some(evicted) = state.entries.remove(&oldest) {
                state.bytes = state.bytes.saturating_sub(evicted.bytes);
            }
        }
        state.bytes = state.bytes.saturating_add(bytes);
        state.entries.insert(
            key,
            LruEntry {
                value,
                bytes,
                last_access: tick,
            },
        );
    }

    /// How many values are kept.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.state
            .lock()
            .expect("decoded cache poisoned")
            .entries
            .len()
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
