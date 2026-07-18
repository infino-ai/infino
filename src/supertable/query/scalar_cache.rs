// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Bounded cache of decoded top-k scalar rows.
//!
//! Superfiles are immutable, so a `(uri, projected columns, local rows)`
//! materialization is stable for the process lifetime. Caching that small
//! result avoids decompressing the same Parquet pages on repeated search
//! fetches while keeping memory independent of table size.

use std::{collections::HashMap, sync::Mutex};

use arrow_array::RecordBatch;

use crate::supertable::manifest::SuperfileUri;

/// Process-local decoded top-k cache budget per supertable handle.
const DEFAULT_DECODED_SCALAR_CACHE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    uri: SuperfileUri,
    columns: Box<[String]>,
    local_doc_ids: Box<[u32]>,
}

#[derive(Debug)]
struct CacheEntry {
    batch: RecordBatch,
    bytes: usize,
    last_access: u64,
}

#[derive(Debug, Default)]
struct CacheState {
    entries: HashMap<CacheKey, CacheEntry>,
    bytes: usize,
    tick: u64,
}

#[derive(Debug)]
pub(crate) struct DecodedScalarCache {
    max_bytes: usize,
    state: Mutex<CacheState>,
}

impl Default for DecodedScalarCache {
    fn default() -> Self {
        Self::new(DEFAULT_DECODED_SCALAR_CACHE_BYTES)
    }
}

impl DecodedScalarCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            state: Mutex::new(CacheState::default()),
        }
    }

    fn key(uri: SuperfileUri, local_doc_ids: &[u32], columns: &[&str]) -> CacheKey {
        CacheKey {
            uri,
            columns: columns.iter().map(|column| (*column).to_string()).collect(),
            local_doc_ids: local_doc_ids.into(),
        }
    }

    pub(crate) fn get(
        &self,
        uri: SuperfileUri,
        local_doc_ids: &[u32],
        columns: &[&str],
    ) -> Option<RecordBatch> {
        let key = Self::key(uri, local_doc_ids, columns);
        let mut state = self.state.lock().expect("decoded scalar cache poisoned");
        state.tick = state.tick.wrapping_add(1);
        let tick = state.tick;
        let entry = state.entries.get_mut(&key)?;
        entry.last_access = tick;
        Some(entry.batch.clone())
    }

    pub(crate) fn insert(
        &self,
        uri: SuperfileUri,
        local_doc_ids: &[u32],
        columns: &[&str],
        batch: RecordBatch,
    ) {
        let bytes = batch.get_array_memory_size();
        if bytes > self.max_bytes {
            return;
        }
        let key = Self::key(uri, local_doc_ids, columns);
        let mut state = self.state.lock().expect("decoded scalar cache poisoned");
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
            CacheEntry {
                batch,
                bytes,
                last_access: tick,
            },
        );
    }
}

#[cfg(test)]
mod tests {
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
        cache.insert(uri, &[7, 3], &["value"], batch(&[70, 30]));
        let cached = cache.get(uri, &[7, 3], &["value"]).expect("cache hit");
        assert_eq!(cached.num_rows(), 2);
        assert!(cache.get(uri, &[3, 7], &["value"]).is_none());
    }

    #[test]
    fn insertion_evicts_oldest_entry_at_budget() {
        let one = batch(&[1]);
        let budget = one.get_array_memory_size();
        let cache = DecodedScalarCache::new(budget);
        let first = SuperfileUri::new_v4();
        let second = SuperfileUri::new_v4();
        cache.insert(first, &[0], &["value"], one);
        cache.insert(second, &[0], &["value"], batch(&[2]));
        assert!(cache.get(first, &[0], &["value"]).is_none());
        assert!(cache.get(second, &[0], &["value"]).is_some());
    }
}
