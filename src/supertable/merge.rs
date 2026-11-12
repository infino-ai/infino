// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Merging one supertable into another without rewriting its rows: the
//! source's superfiles are committed into the destination as they are, and
//! the source's term-index layers are stacked onto the destination's.
//!
//! The caller places the objects (`objects_to_copy` names them) before
//! `graft` commits, so the destination's gc never sees a key it lists.

use std::{
    collections::HashSet,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use futures::{StreamExt, TryStreamExt, stream};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::{
    Consistency,
    storage::{SplitStorageProvider, StorageProvider},
    supertable::{
        SupertableReader,
        error::{BuildError, QueryError},
        gc::term_index_slices,
        handle::Supertable,
        manifest::{
            ManifestSnapshot, SuperfileEntry,
            term_index::{self, Root, format::Layer},
        },
        writer::{
            CommitListMetadata, persist_commit_async, persist_graft_async, scoring_by_superfile,
            stamp_term_index, stamp_with_retries,
        },
    },
};

/// Object checks in flight before a commit: enough to hide object-store
/// latency over tens of thousands of superfiles.
const HEADS_IN_FLIGHT: usize = 64;

/// Replica manifest objects [`detach_replicas`] deletes at once.
const DELETES_IN_FLIGHT: usize = 64;

fn internal(e: impl std::fmt::Display) -> QueryError {
    QueryError::Internal(e.to_string())
}

fn store(e: impl std::fmt::Display) -> BuildError {
    BuildError::Store(e.to_string())
}

/// Fails when `INFINO_GRAFT_FAILPOINT` names `step`, so a test can stop a
/// graft where a crash would.
#[cfg(any(test, feature = "test-helpers"))]
fn failpoint(step: &str) -> Result<(), QueryError> {
    match std::env::var("INFINO_GRAFT_FAILPOINT") {
        Ok(v) if v == step => Err(internal(format!("graft: failpoint {step}"))),
        _ => Ok(()),
    }
}

#[cfg(not(any(test, feature = "test-helpers")))]
fn failpoint(_step: &str) -> Result<(), QueryError> {
    Ok(())
}

/// The pinned term index's shape: layers, listed superfiles, and whether
/// the manifest marks it complete.
pub fn term_index_layout(reader: &SupertableReader) -> Result<Value, QueryError> {
    let m = reader.manifest();
    let index = reader.block_on(m.term_index()).map_err(internal)?;
    Ok(match index {
        None => json!(null),
        Some(ix) => json!({
            "layers": ix.root().layers.len(),
            "covered": ix.root().superfiles.len(),
            "complete": m.term_index_complete(),
        }),
    })
}

/// Objects `graft` needs under the destination, at the same keys as in the
/// source: the superfiles the destination does not list yet, with sizes,
/// and with `reuse_term_index` every slice and tier the source's root names.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GraftObjects {
    pub superfiles: Vec<(String, u64)>,
    pub term_index: Vec<String>,
    /// Rows the superfiles hold.
    pub rows: u64,
    /// Whether the source's term-index layers can be stacked; when not, a
    /// graft makes the destination read every superfile to index it.
    pub spliceable: bool,
}

/// What a `graft` did.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GraftReport {
    /// Superfiles committed into the destination by this call.
    pub merged: usize,
    /// Source superfiles the destination already listed, from an earlier
    /// run or an earlier merge of the same source.
    pub already_merged: usize,
    /// Whether the source's term-index layers were stacked onto the
    /// destination's (by this call or an earlier one).
    pub term_index_spliced: bool,
    /// Commits the superfiles took.
    pub commits: usize,
    /// Source superfiles left out because the destination already lists a
    /// superfile of the same name: the same batch, loaded by another source.
    pub duplicates: usize,
    /// Rows the duplicates hold.
    pub duplicate_rows: u64,
}

/// The source's state a graft works from, read once.
struct Source {
    todo: Vec<Arc<SuperfileEntry>>,
    already_merged: usize,
    duplicates: usize,
    duplicate_rows: u64,
    /// Every live source superfile, merged or not, but for the duplicates.
    ids: HashSet<Uuid>,
    /// The source's root, when its layers can be stacked.
    root: Option<Root>,
}

/// Refuses a source whose superfiles would not mean the same here, or
/// that holds state a graft does not carry.
fn check(dst: &ManifestSnapshot, src: &ManifestSnapshot) -> Result<(), QueryError> {
    let refuse = |what: &str| Err(internal(format!("graft: {what}")));
    if src.table_schema() != dst.table_schema() {
        return refuse("the tables' schemas differ (columns, field ids or index declarations)");
    }
    if src.get_partition_strategy() != dst.get_partition_strategy()
        || src.get_opts().id_column != dst.get_opts().id_column
    {
        return refuse("the tables differ in partitioning or id column");
    }
    if src.get_tombstone_seqs().is_some_and(|t| !t.is_empty()) {
        return refuse("the source has deleted rows; optimize it first");
    }
    for t in [src, dst] {
        if !t.vector_configs().is_empty()
            || !t.companions().is_empty()
            || t.slow_vector_state_blob().is_some()
        {
            return refuse(
                "a table has vector columns or companion tables, which a graft does not carry",
            );
        }
    }
    Ok(())
}

async fn read_source(
    dst: &ManifestSnapshot,
    src: &ManifestSnapshot,
    reuse_term_index: bool,
) -> Result<Source, QueryError> {
    check(dst, src)?;
    let dst_entries = dst.get_all_superfiles_loaded().await.map_err(internal)?;
    let listed: HashSet<Uuid> = dst_entries.iter().map(|e| e.superfile_id).collect();
    // A batch's superfiles share its name. One whose name the destination already lists came from a resent
    // batch, so the first copy grafted is kept.
    let names: HashSet<&str> = dst_entries
        .iter()
        .filter_map(|e| e.stem.as_deref())
        .collect();
    let mut todo = vec![];
    let mut already_merged = 0;
    let mut duplicates = 0;
    let mut duplicate_rows = 0;
    let mut ids = HashSet::new();
    for e in src
        .get_all_superfiles_loaded()
        .await
        .map_err(internal)?
        .iter()
    {
        if listed.contains(&e.superfile_id) {
            ids.insert(e.superfile_id);
            already_merged += 1;
            continue;
        }
        if e.stem.as_deref().is_some_and(|n| names.contains(n)) {
            duplicates += 1;
            duplicate_rows += e.n_docs;
            continue;
        }
        ids.insert(e.superfile_id);
        // A summaries row of the source's part means nothing in ours.
        if e.scalar_stats.is_unavailable() {
            return Err(internal("graft: a source superfile has no statistics"));
        }
        // The commit stamps our partition and refuses an entry already stamped.
        todo.push(Arc::new(SuperfileEntry {
            partition_key: vec![],
            ..(**e).clone()
        }));
    }
    let root = match reuse_term_index && src.term_index_complete() {
        true => src
            .term_index()
            .await
            .map_err(internal)?
            .map(|ix| ix.root().clone())
            .filter(|r| r.layers.iter().all(|l| l.ordinals.is_some())),
        false => None,
    };
    Ok(Source {
        todo,
        already_merged,
        duplicates,
        duplicate_rows,
        ids,
        root,
    })
}

/// Names the objects to place under `dst` before [`graft`].
pub fn objects_to_copy(
    dst: &Supertable,
    src: &SupertableReader,
    reuse_term_index: bool,
) -> Result<GraftObjects, QueryError> {
    let dst_m = dst.inner().manifest.load_full();
    let src_m = Arc::clone(src.manifest());
    let src_storage = src_m
        .get_opts()
        .storage
        .clone()
        .ok_or_else(|| internal("graft: the source has no storage"))?;
    dst.block_on_query(async {
        let s = read_source(&dst_m, &src_m, reuse_term_index).await?;
        let superfiles = s
            .todo
            .iter()
            .map(|e| {
                (
                    e.storage_path(),
                    e.subsection_offsets.as_ref().map_or(0, |o| o.total_size),
                )
            })
            .collect();
        let term_index = match (&s.root, src_m.term_index_ref()) {
            (Some(_), Some(reference)) => term_index_slices(src_storage.as_ref(), reference)
                .await
                .map_err(internal)?,
            _ => vec![],
        };
        let rows = s.todo.iter().map(|e| e.n_docs).sum();
        Ok(GraftObjects {
            superfiles,
            term_index,
            rows,
            spliceable: s.root.is_some(),
        })
    })
}

/// Commits every superfile of `src` that `dst` does not list yet, the
/// objects [`objects_to_copy`] named being in `dst`'s storage already.
///
/// Each commit adds superfiles together with the source layers that index them, so the index is never
/// incomplete and no superfile is read; without a reusable source index the closing refresh reads them. Safe to
/// re-run after a failure at any step.
///
/// Errors when the tables are not compatible (see `check`), when an object
/// is missing under `dst`, or when a commit fails.
pub fn graft(
    dst: &Supertable,
    src: &SupertableReader,
    reuse_term_index: bool,
) -> Result<GraftReport, QueryError> {
    let inner = dst.inner();
    let storage = inner
        .options
        .storage
        .clone()
        .ok_or_else(|| internal("graft: the destination has no storage"))?;
    let src_m = Arc::clone(src.manifest());
    // Keeps a compaction in this process from rewriting grafted superfiles
    // before their layers are stacked, which would leave them to be read.
    let _slot = dst
        .try_hold_compaction_slot()
        .ok_or_else(|| internal("graft: a compaction or reindex is running"))?;
    dst.block_on_query(async {
        // Another coordinator may have grafted since this handle last looked.
        inner.refresh().await.map_err(internal)?;
        let s = read_source(&inner.manifest.load_full(), &src_m, reuse_term_index).await?;
        let mut report = GraftReport {
            already_merged: s.already_merged,
            duplicates: s.duplicates,
            duplicate_rows: s.duplicate_rows,
            ..GraftReport::default()
        };
        // One part per commit: a commit re-encodes the parts it adds to.
        let chunk = inner.options.target_superfiles_per_part.max(1) as usize;
        // A splice may list source superfiles the destination already holds and those this call commits, never
        // one still to come: a fold could drop its postings while it is not live.
        let mut live: HashSet<Uuid> = s.ids.clone();
        for e in &s.todo {
            live.remove(&e.superfile_id);
        }
        for entries in s.todo.chunks(chunk) {
            // Checked right before the commit that lists them.
            stream::iter(entries)
                .map(|e| {
                    let key = e.storage_path();
                    let storage = &storage;
                    async move {
                        storage.head(&key).await.map(|_| ()).map_err(|e| {
                            internal(format!("graft: {key} is not under the destination: {e}"))
                        })
                    }
                })
                .buffer_unordered(HEADS_IN_FLIGHT)
                .try_collect::<Vec<()>>()
                .await?;
            live.extend(entries.iter().map(|e| e.superfile_id));
            match &s.root {
                Some(root) => {
                    let splice = |current: &Root| spliced_root(current, root, &live);
                    persist_graft_async(inner, Arc::clone(&storage), entries.to_vec(), &splice)
                        .await
                        .map_err(internal)?;
                }
                None => {
                    persist_commit_async(
                        inner,
                        Arc::clone(&storage),
                        entries.to_vec(),
                        &[],
                        vec![],
                        vec![],
                        CommitListMetadata::empty(),
                        vec![],
                    )
                    .await
                    .map_err(internal)?;
                }
            }
            report.merged += entries.len();
            report.commits += 1;
            failpoint("commit")?;
        }
        if let Some(root) = &s.root {
            report.term_index_spliced = splice_term_index(dst, &storage, root, &s.ids).await?;
        }
        failpoint("splice")?;
        stamp_term_index(inner, 0).await.map_err(internal)?;
        Ok(report)
    })
}

/// Opens replica `name` of `table`, creating it if absent: `table`'s schema and settings, its
/// manifest under [`crate::storage::REPLICAS_DIR`]`<name>/` and its objects in `table`, with
/// replica folds on, reads that see other handles' commits, and no disk cache of its own.
pub fn open_replica(table: &Supertable, name: &str) -> Result<Supertable, QueryError> {
    let mut options = (*table.inner().options).clone();
    let storage = options
        .storage
        .clone()
        .ok_or_else(|| internal("replica: the table has no storage"))?;
    options.storage = Some(Arc::new(SplitStorageProvider::for_replica(storage, name)));
    // The table's caches are bound to its own storage, where the replica's manifest is not.
    options.disk_cache = None;
    options.metadata_disk_cache = None;
    options.term_index_replica = true;
    options.read_consistency = Consistency::Strong;
    Supertable::create(options).map_err(internal)
}

/// Names of the replicas attached to `table` (see [`crate::storage::REPLICAS_DIR`]).
pub fn attached_replicas(table: &Supertable) -> Result<Vec<String>, QueryError> {
    let Some(storage) = table.inner().options.storage.clone() else {
        return Ok(vec![]);
    };
    table.block_on_query(async {
        crate::storage::attached_replicas(storage.as_ref())
            .await
            .map_err(internal)
    })
}

/// Deletes every replica manifest under `table`, ending the attachment: gc, compaction and
/// optimize run on it again, and the next gc reclaims what replicas wrote but never committed.
/// Call it once the last graft is in and the rows are checked; returns the objects deleted.
pub fn detach_replicas(table: &Supertable) -> Result<usize, QueryError> {
    let storage = table
        .inner()
        .options
        .storage
        .clone()
        .ok_or_else(|| internal("detach: the table has no storage"))?;
    table.block_on_query(async {
        let keys = storage
            .list_with_prefix(crate::storage::REPLICAS_DIR)
            .await
            .map_err(internal)?;
        stream::iter(&keys)
            .map(|key| {
                let storage = &storage;
                async move { storage.delete(key).await.map_err(internal) }
            })
            .buffer_unordered(DELETES_IN_FLIGHT)
            .try_collect::<Vec<()>>()
            .await?;
        Ok(keys.len())
    })
}

/// Folds the next run of a replica's term-index layers and publishes it in place of the run; returns whether it
/// published. Commits only append, so the run stays put; if it moved, this returns false and the next call retries.
pub fn fold_replica(table: &Supertable) -> Result<bool, QueryError> {
    fold_replica_runs(table, false)
}

/// Folds every layer a replica holds into one; run it after the replica's last commit, so the grafted table ends
/// with about one layer per replica.
pub fn fold_replica_all(table: &Supertable) -> Result<bool, QueryError> {
    fold_replica_runs(table, true)
}

fn fold_replica_runs(table: &Supertable, whole: bool) -> Result<bool, QueryError> {
    let inner = table.inner();
    let storage = inner
        .options
        .storage
        .clone()
        .ok_or_else(|| internal("fold: the table has no storage"))?;
    table.block_on_query(async {
        {
            // The loader commits through other handles; refresh so the fold and its publish start from the
            // committed state.
            inner.refresh().await.map_err(internal)?;
            let manifest = inner.manifest.load_full();
            let Some(index) = manifest.term_index().await.map_err(internal)? else {
                return Ok(false);
            };
            let root = index.root().clone();
            let scoring = scoring_by_superfile(&manifest);
            let policy = term_index::BuildPolicy {
                scoring_by_ordinal: Arc::new(
                    root.superfiles
                        .iter()
                        .map(|id| scoring.get(id).cloned().unwrap_or_default())
                        .collect(),
                ),
                ..inner.options.term_index_build_policy()
            };
            let foldable = vec![true; root.layers.len()];
            let pool = Some(&*inner.options.reader_pool);
            let Some((run, merged)) =
                term_index::fold_next(&storage, &root, &foldable, &policy, pool, whole)
                    .await
                    .map_err(internal)?
            else {
                return Ok(false);
            };
            let replaced = &root.layers[run.clone()];
            let published = AtomicBool::new(false);
            stamp_with_retries(
                inner,
                &storage,
                "replica fold",
                |old: Arc<ManifestSnapshot>| {
                    let (published, merged, run, storage) =
                        (&published, merged.clone(), run.clone(), &storage);
                    async move {
                        published.store(false, Ordering::Relaxed);
                        let Some(current) = old.term_index().await.map_err(store)? else {
                            return Ok(None);
                        };
                        let mut next = current.root().clone();
                        if next.layers.get(run.clone()) != Some(replaced) {
                            return Ok(None);
                        }
                        next.layers.splice(run, [merged]);
                        let reference = term_index::write_root(storage.as_ref(), &next)
                            .await
                            .map_err(store)?;
                        published.store(true, Ordering::Relaxed);
                        Ok(Some(
                            old.with_term_index(reference, old.term_index_complete()),
                        ))
                    }
                },
            )
            .await
            .map_err(internal)?;
            Ok(published.load(Ordering::Relaxed))
        }
    })
}

/// `dst`'s root with every `src` layer it does not hold yet appended, each listing its own span of `src` ordinals;
/// `None` when nothing moves. A superfile listed again is set to nil in its earlier slot, and a `dst` layer left
/// with only nil slots is dropped, so a replica's fold replaces the layers it merged. Superfiles outside
/// `grafted` are listed as nil.
fn spliced_root(dst: &Root, src: &Root, grafted: &HashSet<Uuid>) -> Option<Root> {
    if src.layers.iter().any(|l| l.ordinals.is_none()) {
        return None;
    }
    let key = |l: &Layer| {
        (
            l.slices
                .iter()
                .map(|s| s.content_hash.0)
                .collect::<Vec<_>>(),
            l.ordinals.as_ref().map_or(0, |r| r.len()),
        )
    };
    let held: HashSet<_> = dst.layers.iter().map(key).collect();
    let listed: HashSet<Uuid> = dst
        .superfiles
        .iter()
        .copied()
        .filter(|u| !u.is_nil())
        .collect();
    let span = |l: &Layer| {
        let r = l.ordinals.clone().unwrap_or(0..0);
        // A fold written with absolute ordinals has its base below its range.
        l.base.min(r.start) as usize..r.end as usize
    };
    let moves: Vec<&Layer> = src
        .layers
        .iter()
        .filter(|l| {
            let names = &src.superfiles[span(l)];
            names.iter().any(|u| grafted.contains(u))
                && (!held.contains(&key(l))
                    || names
                        .iter()
                        .any(|u| grafted.contains(u) && !listed.contains(u)))
        })
        .collect();
    if moves.is_empty() {
        return None;
    }
    let mut root = dst.clone();
    let mut relisted: HashSet<Uuid> = HashSet::new();
    let mut moved_layers = vec![];
    for layer in moves {
        let lo = span(layer).start as u32;
        let offset = root.superfiles.len() as u32;
        for (u, id_min) in src.superfiles[span(layer)]
            .iter()
            .zip(&src.id_mins[span(layer)])
        {
            let keep = grafted.contains(u);
            root.superfiles.push(if keep { *u } else { Uuid::nil() });
            root.id_mins.push(*id_min);
            if keep {
                relisted.insert(*u);
            }
        }
        let range = layer.ordinals.clone()?;
        let mut moved = layer.clone();
        moved.base = layer.base - lo + offset;
        moved.ordinals = Some(range.start - lo + offset..range.end - lo + offset);
        moved_layers.push(moved);
    }
    // Earlier slots of what was just listed again.
    let first_new = dst.superfiles.len();
    for slot in &mut root.superfiles[..first_new] {
        if relisted.contains(slot) {
            *slot = Uuid::nil();
        }
    }
    let superfiles = &root.superfiles;
    root.layers.retain(|l| {
        l.ordinals.as_ref().is_none_or(|r| {
            r.is_empty()
                || superfiles[r.start as usize..r.end as usize]
                    .iter()
                    .any(|u| !u.is_nil())
        })
    });
    root.layers.extend(moved_layers);
    Some(root)
}

/// Publishes `dst`'s root with `src`'s layers stacked on it; `true` when
/// every grafted superfile is listed in it, by this call or an earlier one.
async fn splice_term_index(
    dst: &Supertable,
    storage: &Arc<dyn StorageProvider>,
    src_root: &Root,
    grafted: &HashSet<Uuid>,
) -> Result<bool, QueryError> {
    let spliced = AtomicBool::new(false);
    stamp_with_retries(
        dst.inner(),
        storage,
        "term-index graft",
        |old: Arc<ManifestSnapshot>| {
            let spliced = &spliced;
            async move {
                spliced.store(false, Ordering::Relaxed);
                let dst_root = match old.term_index().await.map_err(store)? {
                    Some(ix) => ix.root().clone(),
                    None if old.term_index_ref().is_none() => Root::default(),
                    // Unreadable for now: leave it to the refresh to index.
                    None => return Ok(None),
                };
                let Some(root) = spliced_root(&dst_root, src_root, grafted) else {
                    // Nothing left to list: an earlier run spliced it, or the
                    // source's layers cannot be stacked.
                    let done = !grafted.is_empty()
                        && grafted.iter().all(|u| dst_root.superfiles.contains(u));
                    spliced.store(done, Ordering::Relaxed);
                    return Ok(None);
                };
                // Left incomplete: the refresh that follows checks every live
                // superfile is listed and its objects present before marking it.
                let reference = term_index::write_root(storage.as_ref(), &root)
                    .await
                    .map_err(store)?;
                spliced.store(true, Ordering::Relaxed);
                Ok(Some(old.with_term_index(reference, false)))
            }
        },
    )
    .await
    .map_err(internal)?;
    Ok(spliced.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CompactionSettings,
        supertable::{
            manifest::term_index::format::Layer, optimize::compact::max_merge_input_bytes,
        },
    };

    fn layer(base: u32, range: std::ops::Range<u32>) -> Layer {
        Layer {
            slices: vec![slice(base as u8)],
            ordinals: Some(range),
            base,
        }
    }

    /// A slice whose content hash is `tag`, so layers built apart differ.
    fn slice(tag: u8) -> crate::supertable::manifest::term_index::format::SliceRef {
        crate::supertable::manifest::term_index::format::SliceRef {
            first_key: vec![],
            last_key: vec![],
            content_hash: crate::supertable::manifest::part::ContentHash([tag; 32]),
            tier: None,
            len: 0,
        }
    }

    fn ids(n: u128) -> Vec<Uuid> {
        (1..=n).map(Uuid::from_u128).collect()
    }

    /// Each live id once, and every layer's range inside the list.
    fn assert_sound(root: &Root) {
        let live: Vec<_> = root.superfiles.iter().filter(|u| !u.is_nil()).collect();
        let unique: HashSet<_> = live.iter().collect();
        assert_eq!(
            live.len(),
            unique.len(),
            "an id listed twice: {:?}",
            root.superfiles
        );
        assert_eq!(root.superfiles.len(), root.id_mins.len());
        for l in &root.layers {
            let r = l.ordinals.clone().expect("range");
            assert!(r.end as usize <= root.superfiles.len() && l.base <= r.start);
        }
    }

    #[test]
    fn spliced_root_moves_new_deltas_and_lists_only_their_span() {
        let u = ids(4);
        let src = Root {
            superfiles: u.clone(),
            id_mins: vec![10, 20, 30, 40],
            layers: vec![layer(0, 0..1), layer(1, 1..2), layer(2, 2..4)],
        };
        let first = spliced_root(&Root::default(), &src, &HashSet::from([u[0], u[1]]))
            .expect("something to move");
        assert_eq!(first.superfiles, u[..2]);
        assert_eq!(first.layers.len(), 2);
        let all: HashSet<Uuid> = u.iter().copied().collect();
        let root = spliced_root(&first, &src, &all).expect("something to move");
        assert_eq!(root.superfiles[2..], [u[2], u[3]]);
        assert_eq!(root.id_mins[2..], [30, 40]);
        assert_eq!(
            (root.layers[2].base, root.layers[2].ordinals.clone()),
            (2, Some(2..4))
        );
        assert_sound(&root);
        assert!(
            spliced_root(&root, &src, &all).is_none(),
            "nothing left to move"
        );
    }

    #[test]
    fn a_fold_of_grafted_layers_replaces_them() {
        let u = ids(3);
        let all: HashSet<Uuid> = u.iter().copied().collect();
        let deltas = Root {
            superfiles: u.clone(),
            id_mins: vec![1, 2, 3],
            layers: vec![layer(0, 0..1), layer(1, 1..2), layer(2, 2..3)],
        };
        let other = Root {
            superfiles: vec![Uuid::from_u128(99)],
            id_mins: vec![9],
            layers: vec![layer(0, 0..1)],
        };
        let t = spliced_root(&Root::default(), &deltas, &all).expect("something to move");
        let t = spliced_root(&t, &other, &HashSet::from([Uuid::from_u128(99)]))
            .expect("something to move");
        // The replica folds its first two layers into one kept relative to its range.
        let mut fold = layer(0, 0..2);
        fold.slices = vec![];
        fold.base = 0;
        let folded = Root {
            layers: vec![
                Layer {
                    slices: vec![slice(7)],
                    ..fold
                },
                deltas.layers[2].clone(),
            ],
            ..deltas.clone()
        };
        let root = spliced_root(&t, &folded, &all).expect("something to move");
        assert_sound(&root);
        // The two replaced layers are gone; the other replica's and the third delta stay.
        assert_eq!(root.layers.len(), 3);
        assert_eq!(root.superfiles.iter().filter(|x| **x == u[0]).count(), 1);
        assert!(spliced_root(&root, &folded, &all).is_none(), "idempotent");
    }

    #[test]
    fn a_fold_spanning_grafted_and_new_superfiles_lists_both() {
        let u = ids(2);
        let deltas = Root {
            superfiles: u.clone(),
            id_mins: vec![1, 2],
            layers: vec![layer(0, 0..1), layer(1, 1..2)],
        };
        let t = spliced_root(&Root::default(), &deltas, &HashSet::from([u[0]]))
            .expect("something to move");
        assert_eq!(t.superfiles, vec![u[0]]);
        let folded = Root {
            layers: vec![Layer {
                slices: vec![slice(8)],
                ..layer(0, 0..2)
            }],
            ..deltas
        };
        let all: HashSet<Uuid> = u.iter().copied().collect();
        let root = spliced_root(&t, &folded, &all).expect("something to move");
        assert_sound(&root);
        assert_eq!(root.layers.len(), 1, "the old delta is replaced");
        let live: Vec<_> = root
            .superfiles
            .iter()
            .filter(|x| !x.is_nil())
            .copied()
            .collect();
        assert_eq!(live, u);
    }

    #[test]
    fn spliced_root_refuses_layers_without_ranges() {
        let src = Root {
            superfiles: ids(1),
            id_mins: vec![1],
            layers: vec![Layer {
                slices: vec![],
                ordinals: None,
                base: 0,
            }],
        };
        assert!(spliced_root(&Root::default(), &src, &ids(1).into_iter().collect()).is_none());
    }

    /// A table in its own temp dir with one term-index layer per commit.
    fn table(tag: &str, commits: usize) -> (tempfile::TempDir, Supertable) {
        table_as(tag, commits, false)
    }

    fn options_at(
        dir: &std::path::Path,
        schema: &arrow_schema::SchemaRef,
        replica: bool,
    ) -> crate::supertable::SupertableOptions {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(crate::storage::LocalFsStorageProvider::new(dir).expect("local fs"));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        crate::supertable::SupertableOptions::new(
            Arc::clone(schema),
            vec![crate::superfile::builder::FtsConfig::new("title")],
            vec![],
        )
        .expect("options")
        .with_writer_pool(pool)
        .with_term_index_replica(replica)
        // As the loader and the coordinator open tables: a reader sees what other handles committed.
        .with_read_consistency(crate::Consistency::Strong)
        .with_storage(storage)
    }

    fn table_as(tag: &str, commits: usize, replica: bool) -> (tempfile::TempDir, Supertable) {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        let dir = tempfile::TempDir::new().expect("tempdir");
        let schema = Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]));
        let st = Supertable::create(options_at(dir.path(), &schema, replica)).expect("create");
        for c in 0..commits {
            let titles: Vec<String> = (0..50).map(|i| format!("shared {tag}c{c}d{i}")).collect();
            let arr: ArrayRef = Arc::new(LargeStringArray::from(
                titles.iter().map(String::as_str).collect::<Vec<_>>(),
            ));
            let mut w = st.writer().expect("writer");
            w.append(&RecordBatch::try_new(Arc::clone(&schema), vec![arr]).expect("batch"))
                .expect("append");
            w.commit().expect("commit");
        }
        (dir, st)
    }

    fn layout(st: &Supertable) -> (u64, bool) {
        let v = term_index_layout(&st.reader().expect("reader")).expect("layout");
        (
            v["layers"].as_u64().expect("an index"),
            v["complete"].as_bool().expect("complete flag"),
        )
    }

    #[test]
    fn grafting_folding_replicas_keeps_the_index_short_and_whole() {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
        let (dst_dir, dst) = table("t", 0);
        let replicas: Vec<_> = ["p", "q"]
            .iter()
            .map(|tag| (tag.to_string(), table_as(tag, 0, true)))
            .collect();
        // Each replica's fold thread keeps one handle of its own for the whole load, as the loader does.
        let folders: Vec<Supertable> = replicas
            .iter()
            .map(|(_, (dir, src))| {
                let schema = src
                    .reader()
                    .expect("reader")
                    .manifest()
                    .get_opts()
                    .schema
                    .clone();
                Supertable::open(options_at(dir.path(), &schema, true)).expect("open")
            })
            .collect();
        let mut committed = 0;
        for round in 0..10 {
            for ((tag, (dir, src)), folder) in replicas.iter().zip(&folders) {
                let schema = src
                    .reader()
                    .expect("reader")
                    .manifest()
                    .get_opts()
                    .schema
                    .clone();
                for c in 0..7 {
                    let titles: Vec<String> = (0..20)
                        .map(|i| format!("shared {tag}r{round}c{c}d{i}"))
                        .collect();
                    let arr: ArrayRef = Arc::new(LargeStringArray::from(
                        titles.iter().map(String::as_str).collect::<Vec<_>>(),
                    ));
                    let mut w = src.writer().expect("writer");
                    w.append(&RecordBatch::try_new(schema.clone(), vec![arr]).expect("batch"))
                        .expect("append");
                    w.commit().expect("commit");
                    committed += 20;
                }
                while fold_replica(folder).expect("fold") {}
                // Replicas write into t's storage; here a copy stands in for that.
                let reader = src.reader().expect("reader");
                for key in std::fs::read_dir(dir.path().join("term-index"))
                    .expect("ls")
                    .map(|e| {
                        format!(
                            "term-index/{}",
                            e.expect("entry").file_name().to_string_lossy()
                        )
                    })
                    .chain(
                        objects_to_copy(&dst, &reader, true)
                            .expect("objects")
                            .superfiles
                            .into_iter()
                            .map(|(k, _)| k),
                    )
                {
                    let to = dst_dir.path().join(&key);
                    if !to.exists() {
                        std::fs::create_dir_all(to.parent().expect("parent")).expect("mkdir");
                        std::fs::copy(dir.path().join(&key), to).expect("copy");
                    }
                }
                graft(&dst, &reader, true).expect("graft");
                let (layers, complete) = layout(&dst);
                assert!(complete, "complete after every graft");
                // Each replica keeps under eight layers per size level: at 70 commits
                // at most 7 + 7 + 1, against one per commit without folds.
                assert!(layers <= 2 * 15, "round {round}: {layers} layers");
            }
        }
        // The end of the load: each replica folds everything it holds into one layer.
        for ((_, (dir, src)), folder) in replicas.iter().zip(&folders) {
            assert!(fold_replica_all(folder).expect("final fold"));
            assert!(
                !fold_replica_all(folder).expect("final fold"),
                "one layer left"
            );
            let reader = src.reader().expect("reader");
            for key in std::fs::read_dir(dir.path().join("term-index"))
                .expect("ls")
                .map(|e| {
                    format!(
                        "term-index/{}",
                        e.expect("entry").file_name().to_string_lossy()
                    )
                })
            {
                let to = dst_dir.path().join(&key);
                if !to.exists() {
                    std::fs::copy(dir.path().join(&key), to).expect("copy");
                }
            }
            graft(&dst, &reader, true).expect("graft");
        }
        let (layers, complete) = layout(&dst);
        assert!(complete && layers <= 2, "one layer per replica: {layers}");
        let hits = dst
            .reader()
            .expect("reader")
            .token_match(
                "title",
                "shared",
                crate::superfile::fts::reader::BoolMode::Or,
            )
            .expect("search");
        assert_eq!(hits.len(), committed, "every row once");
    }

    #[test]
    fn a_batch_loaded_by_two_sources_is_grafted_once() {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
        let (dst_dir, dst) = table("d", 0);
        let load = |tag: &str, names: &[&str]| {
            let (dir, t) = table(tag, 0);
            let schema = t
                .reader()
                .expect("reader")
                .manifest()
                .get_opts()
                .schema
                .clone();
            for name in names {
                let titles: Vec<String> = (0..50).map(|i| format!("shared {name}d{i}")).collect();
                let arr: ArrayRef = Arc::new(LargeStringArray::from(
                    titles.iter().map(String::as_str).collect::<Vec<_>>(),
                ));
                t.append_named(
                    &RecordBatch::try_new(schema.clone(), vec![arr]).expect("batch"),
                    name,
                )
                .expect("append");
            }
            (dir, t)
        };
        let (a_dir, a) = load("a", &["x", "y"]);
        let (b_dir, b) = load("b", &["y", "z"]);
        let mut duplicates = 0;
        for (dir, src) in [(&a_dir, &a), (&b_dir, &b)] {
            let reader = src.reader().expect("reader");
            let objects = objects_to_copy(&dst, &reader, true).expect("objects");
            for key in objects
                .superfiles
                .iter()
                .map(|(k, _)| k)
                .chain(&objects.term_index)
            {
                let to = dst_dir.path().join(key);
                std::fs::create_dir_all(to.parent().expect("parent")).expect("mkdir");
                std::fs::copy(dir.path().join(key), to).expect("copy");
            }
            duplicates += graft(&dst, &reader, true).expect("graft").duplicates;
        }
        assert_eq!(duplicates, 1, "b's copy of y");
        let r = dst.reader().expect("reader");
        assert_eq!(
            r.manifest()
                .superfiles
                .iter()
                .map(|e| e.n_docs)
                .sum::<u64>(),
            3 * 50
        );
        let hits = r
            .token_match(
                "title",
                "shared",
                crate::superfile::fts::reader::BoolMode::Or,
            )
            .expect("search");
        assert_eq!(hits.len(), 3 * 50);
        assert!(layout(&dst).1, "complete");
    }

    #[test]
    fn a_replica_writes_into_the_table_and_grafts_without_copies() {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
        let (_dir, t) = table("t", 1);
        let schema = t
            .reader()
            .expect("reader")
            .manifest()
            .get_opts()
            .schema
            .clone();
        let replica = open_replica(&t, "r0").expect("replica");
        for name in ["b1", "b2"] {
            let titles: Vec<String> = (0..50).map(|i| format!("shared {name}d{i}")).collect();
            let arr: ArrayRef = Arc::new(LargeStringArray::from(
                titles.iter().map(String::as_str).collect::<Vec<_>>(),
            ));
            replica
                .append_named(
                    &RecordBatch::try_new(schema.clone(), vec![arr]).expect("batch"),
                    name,
                )
                .expect("append");
        }
        assert!(t.has_attached_replicas().expect("attached"));
        let again = open_replica(&t, "r0").expect("reopen");
        let report = graft(&t, &again.reader().expect("reader"), true).expect("graft");
        assert_eq!(report.merged, 2);
        let hits = t
            .reader()
            .expect("reader")
            .token_match(
                "title",
                "shared",
                crate::superfile::fts::reader::BoolMode::Or,
            )
            .expect("search");
        assert_eq!(hits.len(), 3 * 50, "the table's own rows and both batches");
        assert!(layout(&t).1, "complete");
    }

    #[test]
    fn a_second_coordinator_with_a_stale_view_adds_no_rows() {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
        let (dir, t) = table("t", 1);
        let schema = t
            .reader()
            .expect("reader")
            .manifest()
            .get_opts()
            .schema
            .clone();
        // Coordinator A opens the table before B grafts, so its view is stale.
        let a = Supertable::open(options_at(dir.path(), &schema, false)).expect("open");
        let replica = open_replica(&t, "r0").expect("replica");
        let arr: ArrayRef = Arc::new(LargeStringArray::from(vec!["shared x"; 50]));
        replica
            .append_named(
                &RecordBatch::try_new(schema.clone(), vec![arr]).expect("batch"),
                "b1",
            )
            .expect("append");
        let reader = replica.reader().expect("reader");
        graft(&t, &reader, true).expect("B grafts");
        // A's graft either fails or adds nothing; the next round re-reads and moves on.
        if let Ok(report) = graft(&a, &reader, true) {
            assert_eq!(report.merged, 0, "{report:?}");
        }
        let rows = |h: &Supertable| {
            h.reader()
                .expect("reader")
                .manifest()
                .superfiles
                .iter()
                .map(|e| e.n_docs)
                .sum::<u64>()
        };
        assert_eq!(rows(&t), 100);
        assert_eq!(rows(&a), 100);
    }

    #[test]
    fn detach_replicas_ends_the_attachment() {
        let (dir, t) = table("a", 1);
        for name in ["r0", "r1"] {
            let pointer = dir
                .path()
                .join(format!("_replicas/{name}/_supertable/current"));
            std::fs::create_dir_all(pointer.parent().expect("parent")).expect("mkdir");
            std::fs::write(&pointer, b"x").expect("pointer");
        }
        let mut names = attached_replicas(&t).expect("list");
        names.sort();
        assert_eq!(names, ["r0", "r1"]);
        assert!(t.has_attached_replicas().expect("attached"));
        assert_eq!(detach_replicas(&t).expect("detach"), 2);
        assert!(attached_replicas(&t).expect("list").is_empty());
        assert!(!t.has_attached_replicas().expect("detached"));
    }

    #[test]
    fn attached_replicas_keep_gc_and_compaction_off_the_table() {
        let (dir, t) = table("a", 9);
        let orphan = dir.path().join("data/seg-orphan.sf.parquet");
        std::fs::write(&orphan, b"replica superfile not grafted yet").expect("orphan");
        let pointer = dir.path().join("_replicas/r0/_supertable/current");
        std::fs::create_dir_all(pointer.parent().expect("parent")).expect("mkdir");
        std::fs::write(&pointer, b"x").expect("pointer");
        let files = |t: &Supertable| t.reader().expect("reader").manifest().superfiles.len();
        let before = files(&t);

        t.gc(std::time::Duration::ZERO).expect("gc");
        assert!(orphan.exists(), "kept while a replica is attached");
        let mut cfg = CompactionSettings {
            target_superfile_size_mb: 1,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        };
        cfg.steps.recalibrate = crate::RecalibratePolicy::Skip;
        t.compact(&cfg).expect("compact");
        assert_eq!(files(&t), before, "no compaction while attached");

        // optimize stands down as a whole: its term-index fold would merge replicas' layers too.
        assert!(t.has_attached_replicas().expect("attached"));
        let layers = layout(&t).0;
        t.optimize(&crate::OptimizeOptions::compact(cfg.clone()))
            .expect("optimize");
        assert_eq!(
            (files(&t), layout(&t).0),
            (before, layers),
            "optimize changed nothing"
        );

        std::fs::remove_dir_all(dir.path().join("_replicas")).expect("detach");
        assert!(!t.has_attached_replicas().expect("detached"));
        t.gc(std::time::Duration::ZERO).expect("gc");
        assert!(!orphan.exists(), "reclaimed once detached");
        t.compact(&cfg).expect("compact");
        assert!(files(&t) < before, "compacts once detached");
    }

    #[test]
    fn a_graft_commit_alone_leaves_the_index_complete() {
        let (dst_dir, dst) = table("a", 2);
        let (src_dir, src) = table("b", 3);
        let reader = src.reader().expect("reader");
        let objects = objects_to_copy(&dst, &reader, true).expect("objects");
        for key in objects
            .superfiles
            .iter()
            .map(|(k, _)| k)
            .chain(&objects.term_index)
        {
            let to = dst_dir.path().join(key);
            std::fs::create_dir_all(to.parent().expect("parent")).expect("mkdir");
            std::fs::copy(src_dir.path().join(key), to).expect("copy");
        }
        let inner = dst.inner();
        let storage = inner.options.storage.clone().expect("storage");
        dst.block_on_query(async {
            let s = read_source(&inner.manifest.load_full(), reader.manifest(), true)
                .await
                .expect("source");
            let root = s.root.expect("a stackable source index");
            let splice = |current: &Root| spliced_root(current, &root, &s.ids);
            persist_graft_async(inner, storage, s.todo, &splice)
                .await
                .expect("commit");
        });
        // No refresh ran: the commit itself listed the superfiles and their postings.
        let (_, complete) = layout(&dst);
        assert!(complete, "complete right after the commit");
        let hits = dst
            .reader()
            .expect("reader")
            .token_match(
                "title",
                "shared",
                crate::superfile::fts::reader::BoolMode::Or,
            )
            .expect("search");
        assert_eq!(hits.len(), 5 * 50);
    }

    #[test]
    fn graft_with_zero_fold_cap_stacks_layers_and_completes_the_index() {
        // Four layers each stays under the eight a fold needs; the eight stacked
        // by the graft fold whenever folds may run.
        let (dst_dir, dst) = table("a", 4);
        let (src_dir, src) = table("b", 4);
        let (dst_layers, _) = layout(&dst);
        let (src_layers, src_complete) = layout(&src);
        assert!(
            dst_layers > 1 && src_layers > 1 && src_complete,
            "multi-layer inputs: {dst_layers}, {src_layers}"
        );

        let reader = src.reader().expect("reader");
        let objects = objects_to_copy(&dst, &reader, true).expect("objects");
        assert!(objects.spliceable && !objects.term_index.is_empty());
        for key in objects
            .superfiles
            .iter()
            .map(|(k, _)| k)
            .chain(&objects.term_index)
        {
            let to = dst_dir.path().join(key);
            std::fs::create_dir_all(to.parent().expect("parent")).expect("mkdir");
            std::fs::copy(src_dir.path().join(key), to).expect("copy");
        }
        let report = graft(&dst, &reader, true).expect("graft");
        assert!(report.term_index_spliced);
        assert_eq!(
            layout(&dst),
            (dst_layers + src_layers, true),
            "no fold ran and the index is complete"
        );

        // Every grafted row is found through the stacked index.
        let mode = crate::superfile::fts::reader::BoolMode::Or;
        let hits = dst
            .reader()
            .expect("reader")
            .token_match("title", "shared", mode)
            .expect("search");
        assert_eq!(hits.len(), 8 * 50);

        // The next optimize folds them.
        dst.refresh_term_index_sync(max_merge_input_bytes(&CompactionSettings::default()))
            .expect("refresh");
        let (layers, complete) = layout(&dst);
        assert!(
            layers < dst_layers + src_layers && complete,
            "folded: {layers}"
        );
    }
}
