// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The lazy readers a table keeps between queries.
//!
//! A table with durable storage and no disk cache (every connection without
//! `cache_dir`, the default) opens each superfile a query reads as a lazy
//! reader over range reads. Keeping that reader saves the next query the
//! footer read and parse, and on its first scan the page indexes. But a kept
//! reader also keeps what later reads loaded into it: the FTS doc-id map (4
//! bytes a document), the norms, the vector columns' norms, the range
//! directory. So the readers are weighed by what they hold
//! ([`SuperfileReader::held_bytes`]), bounded by bytes, and charged to the
//! connection's memory budget, which can take them all back.
//!
//! A reader is weighed when it is kept and again each time it is handed out,
//! so what a query loads into it counts from the next open of its superfile.
//! Past the bound, or when the budget refuses the charge, the least recently
//! used readers are dropped; a caller holding one keeps it alive until it is
//! done.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex, PoisonError, Weak},
};

use crate::{
    memory::{ConnectionMemoryBudget, Reclaim, Reservation},
    superfile::SuperfileReader,
    supertable::manifest::SuperfileUri,
};

/// Most bytes one table's kept readers hold, by
/// [`SuperfileReader::held_bytes`].
pub(super) const KEPT_LAZY_MAX_BYTES: usize = 256 << 20;

/// One table's kept lazy readers.
pub(super) struct KeptLazyReaders {
    /// Most bytes the readers hold together.
    max_bytes: usize,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    readers: HashMap<SuperfileUri, Kept>,
    /// The kept readers by when each was last used, oldest first.
    recency: BTreeMap<u64, SuperfileUri>,
    /// Ticks once per use; orders `recency`.
    clock: u64,
    /// Sum of the kept readers' weights.
    bytes: usize,
    /// The budget the readers are charged to, once the table attached one.
    budget: Option<Arc<ConnectionMemoryBudget>>,
}

struct Kept {
    reader: Arc<SuperfileReader>,
    /// What the reader held when last weighed.
    bytes: usize,
    /// Its key in `recency`.
    used: u64,
    /// Its weight, held against the budget; `None` without one.
    charge: Option<Reservation>,
}

impl KeptLazyReaders {
    /// Readers holding at most `max_bytes` together.
    pub(super) fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            state: Mutex::new(State::default()),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Charge the readers to `budget` from now on, moving what they hold
    /// off any budget charged before, and let it take them back when a
    /// reservation does not fit.
    pub(super) fn attach(self: &Arc<Self>, budget: &Arc<ConnectionMemoryBudget>) {
        {
            let mut state = self.state();
            if state
                .budget
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, budget))
            {
                return;
            }
            state.budget = Some(Arc::clone(budget));
            let uris: Vec<SuperfileUri> = state.readers.keys().copied().collect();
            for uri in uris {
                let kept = state.readers.get_mut(&uri).expect("listed above");
                kept.charge = None;
                match budget.try_reserve_without_reclaim(kept.bytes) {
                    Ok(charge) => kept.charge = Some(charge),
                    Err(_) => state.drop_reader(&uri),
                }
            }
        }
        let holder: Weak<Self> = Arc::downgrade(self);
        budget.register_reclaim(holder);
    }

    /// The reader kept for `uri`, weighed again and marked used.
    pub(super) fn get(&self, uri: &SuperfileUri) -> Option<Arc<SuperfileReader>> {
        let mut state = self.state();
        let reader = Arc::clone(&state.readers.get(uri)?.reader);
        state.touch(uri);
        state.reweigh(uri, reader.held_bytes(), self.max_bytes);
        Some(reader)
    }

    /// Keep `reader` for `uri` and return the reader kept for it: the first
    /// one kept wins, so opens racing on one superfile all share a reader.
    /// A reader that does not fit the bound or the budget is handed back
    /// without being kept.
    pub(super) fn keep(
        &self,
        uri: SuperfileUri,
        reader: Arc<SuperfileReader>,
    ) -> Arc<SuperfileReader> {
        let mut state = self.state();
        if let Some(kept) = state.readers.get(&uri) {
            let kept = Arc::clone(&kept.reader);
            state.touch(&uri);
            return kept;
        }
        let used = state.tick();
        state.recency.insert(used, uri);
        state.readers.insert(
            uri,
            Kept {
                reader: Arc::clone(&reader),
                bytes: 0,
                used,
                charge: None,
            },
        );
        if let Some(budget) = state.budget.clone() {
            // Empty for now: the weighing below grows it to the reader's
            // weight, or drops the reader when the budget refuses.
            let charge = budget
                .try_reserve_without_reclaim(0)
                .expect("an empty reservation always fits");
            state.readers.get_mut(&uri).expect("kept above").charge = Some(charge);
        }
        state.reweigh(&uri, reader.held_bytes(), self.max_bytes);
        reader
    }

    /// Drop the reader kept for `uri`, if any.
    pub(super) fn remove(&self, uri: &SuperfileUri) {
        self.state().drop_reader(uri);
    }

    /// How many readers are kept.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.state().readers.len()
    }

    /// The bytes the kept readers weigh together.
    #[cfg(test)]
    pub(super) fn bytes(&self) -> usize {
        self.state().bytes
    }
}

impl State {
    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Mark `uri`'s reader the most recently used.
    fn touch(&mut self, uri: &SuperfileUri) {
        let used = self.tick();
        if let Some(kept) = self.readers.get_mut(uri) {
            self.recency.remove(&kept.used);
            kept.used = used;
            self.recency.insert(used, *uri);
        }
    }

    /// Set `uri`'s weight to `bytes`, charging the difference to the budget,
    /// then drop the least recently used readers until the rest fit
    /// `max_bytes`. A reader that alone outweighs the bound, or whose growth
    /// the budget refuses, is dropped itself.
    fn reweigh(&mut self, uri: &SuperfileUri, bytes: usize, max_bytes: usize) {
        let Some(kept) = self.readers.get_mut(uri) else {
            return;
        };
        let before = kept.bytes;
        if bytes > before {
            if let Some(charge) = kept.charge.as_mut()
                && charge.try_grow_without_reclaim(bytes - before).is_err()
            {
                self.drop_reader(uri);
                return;
            }
        } else if let Some(charge) = kept.charge.as_mut() {
            charge.shrink(before - bytes);
        }
        kept.bytes = bytes;
        self.bytes = self.bytes - before + bytes;
        if bytes > max_bytes {
            self.drop_reader(uri);
            return;
        }
        while self.bytes > max_bytes {
            let Some((_, oldest)) = self.recency.first_key_value().map(|(t, u)| (*t, *u)) else {
                break;
            };
            self.drop_reader(&oldest);
        }
    }

    /// Stop keeping `uri`'s reader, giving its charge back.
    fn drop_reader(&mut self, uri: &SuperfileUri) {
        if let Some(kept) = self.readers.remove(uri) {
            self.recency.remove(&kept.used);
            self.bytes -= kept.bytes;
        }
    }
}

impl Reclaim for KeptLazyReaders {
    /// Give back every kept reader. A later open of a superfile opens it
    /// again and keeps that one.
    fn reclaim(&self) {
        let dropped = {
            let mut state = self.state();
            state.recency.clear();
            state.bytes = 0;
            std::mem::take(&mut state.readers)
        };
        // Readers drop outside the lock.
        drop(dropped);
    }
}
