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
use uuid::Uuid;

use crate::{
    storage::StorageProvider,
    supertable::{
        SupertableReader,
        error::{BuildError, QueryError},
        gc::term_index_slices,
        handle::Supertable,
        manifest::{
            ManifestSnapshot, SuperfileEntry,
            term_index::{self, Root},
        },
        writer::{CommitListMetadata, persist_commit_async, stamp_term_index, stamp_with_retries},
    },
};

/// Object checks in flight before a commit: enough to hide object-store
/// latency over tens of thousands of superfiles.
const HEADS_IN_FLIGHT: usize = 64;

/// Copies [`merge_tables`] keeps in flight when the store copies server side: each is one small request.
const SERVER_COPIES_IN_FLIGHT: usize = 256;

/// Copies [`merge_tables`] keeps in flight when they stream through this process: each holds a
/// superfile in memory.
const STREAMED_COPIES_IN_FLIGHT: usize = 4;

fn internal(e: impl std::fmt::Display) -> QueryError {
    QueryError::Internal(e.to_string())
}

fn store(e: impl std::fmt::Display) -> BuildError {
    BuildError::Store(e.to_string())
}

/// Fails when `INFINO_GRAFT_FAILPOINT` names `step`, so a test can stop a
/// graft where a crash would.
fn failpoint(step: &str) -> Result<(), QueryError> {
    match std::env::var("INFINO_GRAFT_FAILPOINT") {
        Ok(v) if v == step => Err(internal(format!("graft: failpoint {step}"))),
        _ => Ok(()),
    }
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

/// What [`Supertable::merge_tables`](crate::Supertable::merge_tables) did.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct MergeReport {
    /// Superfiles committed into the table by this call.
    pub merged: usize,
    /// Rows those superfiles hold.
    pub rows: u64,
    /// Objects copied into the table's storage.
    pub copied: usize,
    /// Objects the table's storage held already, from an earlier run.
    pub present: usize,
    /// Sources whose term-index layers were stacked; the table indexed the others' superfiles itself.
    pub spliced: usize,
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
}

/// The source's state a graft works from, read once.
struct Source {
    todo: Vec<Arc<SuperfileEntry>>,
    already_merged: usize,
    /// Every live source superfile, merged or not.
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
    let listed: HashSet<Uuid> = dst
        .get_all_superfiles_loaded()
        .await
        .map_err(internal)?
        .iter()
        .map(|e| e.superfile_id)
        .collect();
    let mut todo = vec![];
    let mut already_merged = 0;
    let mut ids = HashSet::new();
    for e in src
        .get_all_superfiles_loaded()
        .await
        .map_err(internal)?
        .iter()
    {
        ids.insert(e.superfile_id);
        if listed.contains(&e.superfile_id) {
            already_merged += 1;
            continue;
        }
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
/// The superfiles are committed first, so they are live before the index
/// names them: a fold anywhere drops postings only of superfiles not live.
/// With `reuse_term_index` and a complete source index, the source's layers
/// are then stacked onto `dst`'s, so no superfile is read; otherwise the
/// refresh run last reads them. The refresh runs no folds: the layers stay
/// stacked for the next optimize to fold. That
/// refresh is also what marks the index complete again. Safe to re-run
/// after a failure at any step: listed superfiles and stacked layers are
/// skipped.
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
        let s = read_source(&inner.manifest.load_full(), &src_m, reuse_term_index).await?;
        let mut report = GraftReport {
            already_merged: s.already_merged,
            ..GraftReport::default()
        };
        // One part per commit: a commit re-encodes the parts it adds to.
        let chunk = inner.options.target_superfiles_per_part.max(1) as usize;
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

/// `dst`'s root with `src`'s layers stacked on it: `src`'s superfiles are
/// listed after `dst`'s, and each of its layers moves up by `dst`'s length,
/// which the layer base makes a renumbering of no stored posting. `None`
/// when no grafted superfile is left to list, or a `src` layer predates
/// ordinal ranges and so may name any ordinal.
fn spliced_root(dst: &Root, src: &Root, grafted: &HashSet<Uuid>) -> Option<Root> {
    let offset = dst.superfiles.len() as u32;
    let listed: HashSet<Uuid> = dst.superfiles.iter().copied().collect();
    // A superfile not being grafted, or one `dst` already lists, is nil
    // here: no live entry matches nil, so its postings are ignored like a
    // dead one's and no superfile is listed twice.
    let added: Vec<Uuid> = src
        .superfiles
        .iter()
        .map(|u| {
            if grafted.contains(u) && !listed.contains(u) {
                *u
            } else {
                Uuid::nil()
            }
        })
        .collect();
    if added.iter().all(Uuid::is_nil) {
        return None;
    }
    let mut root = dst.clone();
    root.superfiles.extend(added);
    root.id_mins.extend(&src.id_mins);
    for layer in &src.layers {
        let range = layer.ordinals.clone()?;
        let mut moved = layer.clone();
        moved.base = layer.base + offset;
        moved.ordinals = Some(range.start + offset..range.end + offset);
        root.layers.push(moved);
    }
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

fn storage_of_table(table: &Supertable) -> Result<Arc<dyn StorageProvider>, QueryError> {
    table
        .inner()
        .options
        .storage
        .clone()
        .ok_or_else(|| internal("merge: the table has no storage"))
}

/// Moves every row of `sources` into `table`: copies their superfiles and term-index layers into
/// `table`'s storage (server side where the store can), then commits them with their index, one
/// source at a time. Safe to run again after a failure; what `table` holds already is skipped.
/// Sources are left as they were; drop them after. Errors when a source differs from `table` in
/// schema, partitioning or vector cell grid, or when a source changes during the merge.
pub fn merge_tables(
    table: &Supertable,
    sources: &[&Supertable],
) -> Result<MergeReport, QueryError> {
    let storage = storage_of_table(table)?;
    let mut report = MergeReport::default();
    for source in sources {
        let reader = source.reader().map_err(internal)?;
        let mut objects = objects_to_copy(table, &reader, true)?;
        let spliced = objects.spliceable;
        if !spliced {
            objects = objects_to_copy(table, &reader, false)?;
        }
        let source_storage = storage_of_table(source)?;
        let server_side = matches!(
            (storage.server_side_key(""), source_storage.server_side_key("")),
            (Some((a, _)), Some((b, _))) if a == b
        );
        let keys: Vec<&String> = objects
            .superfiles
            .iter()
            .map(|(key, _)| key)
            .chain(&objects.term_index)
            .collect();
        let copied: Vec<bool> = table
            .block_on_query(
                stream::iter(&keys)
                    .map(|key| {
                        let (storage, source_storage) = (&storage, &source_storage);
                        async move { storage.copy_in(key, source_storage.as_ref(), key).await }
                    })
                    .buffer_unordered(match server_side {
                        true => SERVER_COPIES_IN_FLIGHT,
                        false => STREAMED_COPIES_IN_FLIGHT,
                    })
                    .try_collect(),
            )
            .map_err(internal)?;
        let grafted = graft(table, &reader, spliced)?;
        report.merged += grafted.merged;
        report.rows += objects.rows;
        report.copied += copied.iter().filter(|&&c| c).count();
        report.present += copied.iter().filter(|&&c| !c).count();
        report.spliced += usize::from(spliced);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        CompactionSettings,
        supertable::{
            manifest::term_index::format::Layer, optimize::compact::max_merge_input_bytes,
        },
    };

    fn layer(base: u32, range: std::ops::Range<u32>) -> Layer {
        Layer {
            slices: vec![],
            ordinals: Some(range),
            base,
        }
    }

    #[test]
    fn spliced_root_moves_src_layers_past_dst() {
        let a = Uuid::from_u128(1);
        let (b1, b2) = (Uuid::from_u128(2), Uuid::from_u128(3));
        let dst = Root {
            superfiles: vec![a],
            id_mins: vec![10],
            layers: vec![layer(0, 0..1)],
        };
        let src = Root {
            superfiles: vec![b1, b2],
            id_mins: vec![20, 30],
            layers: vec![layer(0, 0..2), layer(1, 1..2)],
        };
        let root = spliced_root(&dst, &src, &HashSet::from([b2])).expect("spliced");
        assert_eq!(root.superfiles, vec![a, Uuid::nil(), b2]);
        assert_eq!(root.id_mins, vec![10, 20, 30]);
        assert_eq!(
            (root.layers[1].base, root.layers[1].ordinals.clone()),
            (1, Some(1..3))
        );
        assert_eq!(
            (root.layers[2].base, root.layers[2].ordinals.clone()),
            (2, Some(2..3))
        );
        // Already listed: nothing to add.
        assert!(spliced_root(&root, &src, &HashSet::from([b2])).is_none());
        // Shares b1 with dst: b1 is nil in the stacked part, b2 is added.
        let dst = Root {
            superfiles: vec![a, b1],
            id_mins: vec![10, 20],
            layers: vec![layer(0, 0..2)],
        };
        let root = spliced_root(&dst, &src, &HashSet::from([b1, b2])).expect("spliced");
        assert_eq!(root.superfiles, vec![a, b1, Uuid::nil(), b2]);
        let unranged = Root {
            layers: vec![Layer {
                slices: vec![],
                ordinals: None,
                base: 0,
            }],
            ..src
        };
        assert!(spliced_root(&dst, &unranged, &HashSet::from([b2])).is_none());
    }

    /// A table in its own temp dir with one term-index layer per commit.
    fn table(tag: &str, commits: usize) -> (tempfile::TempDir, Supertable) {
        use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};

        use crate::{
            storage::LocalFsStorageProvider, superfile::builder::FtsConfig,
            supertable::SupertableOptions,
        };

        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs"));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]));
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("pool"),
        );
        let options =
            SupertableOptions::new(Arc::clone(&schema), vec![FtsConfig::new("title")], vec![])
                .expect("options")
                .with_writer_pool(pool)
                .with_storage(storage);
        let st = Supertable::create(options).expect("create");
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

    /// The pinned term index's shape: layers, listed superfiles, and whether
    /// the manifest marks it complete.
    fn term_index_layout(reader: &SupertableReader) -> Result<Value, QueryError> {
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

    fn layout(st: &Supertable) -> (u64, bool) {
        let v = term_index_layout(&st.reader().expect("reader")).expect("layout");
        (
            v["layers"].as_u64().expect("an index"),
            v["complete"].as_bool().expect("complete flag"),
        )
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
