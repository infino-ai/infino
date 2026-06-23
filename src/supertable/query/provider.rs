// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `SupertableProvider` — a DataFusion [`TableProvider`] that owns
//! superfile selection and hands the rest to DataFusion.
//!
//! ## Two-tier pruning
//!
//! This is the SQL counterpart to the dedicated BM25 / vector
//! entry points: **infino decides which superfiles are relevant;
//! DataFusion executes over them.** Concretely, [`scan`] performs
//! two tiers of skipping:
//!
//!   1. **Superfile skip (infino).** The `WHERE` clause's simple
//!      `column <op> literal` conjuncts are lowered to
//!      [`ScalarPredicate`]s and run through
//!      [`scalar_skip`] against each superfile's persisted
//!      `scalar_stats` min/max. Definitely-irrelevant superfiles
//!      are dropped before any bytes are decoded. This is the same
//!      manifest-level skip philosophy as `fts_bloom_skip` /
//!      `vector_centroid_skip`.
//!   2. **Row-group / page skip (DataFusion).** The surviving
//!      superfiles' Parquet bytes are exposed to a DataFusion
//!      `ParquetSource` via an in-memory object store. The same
//!      predicate is handed to DataFusion as a physical expression
//!      so `PruningPredicate` prunes row groups and pages, then
//!      projects + limits. We deliberately do **not** reimplement
//!      this commodity layer.
//!
//! Correctness is independent of either tier: every pushed filter
//! is reported [`TableProviderFilterPushDown::Inexact`], so
//! DataFusion always re-applies the full predicate in a
//! `FilterExec` above the scan. Both skip tiers are pure
//! *conservative* optimizations — they may keep a non-matching
//! superfile/row group, never drop a matching one.
//!
//! ## Why an in-memory object store
//!
//! The reader cache already holds warm superfiles as resident Parquet bytes
//! (`SuperfileReader::parquet_bytes`, an `Arc`-backed `Bytes` — cloning is
//! a refcount bump, not a copy). Registering those bytes into a DataFusion
//! [`InMemory`] object store lets us reuse DataFusion's full `ParquetSource`
//! (lazy row-group decode, projection/limit pushdown, row-group pruning)
//! without adding another object-store implementation.

use std::{
    any::Any,
    cmp,
    collections::{HashMap, HashSet},
    fmt,
    ops::Range,
    sync::{Arc, atomic},
    time::Instant,
};

use arrow_array::ArrayRef;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::{
    catalog::{Session, TableProvider},
    common::{ColumnStatistics, DFSchema, Statistics, stats::Precision},
    datasource::{
        listing::PartitionedFile,
        physical_plan::{
            FileScanConfigBuilder, ParquetFileReaderFactory, ParquetSource,
            parquet::ParquetAccessPlan,
        },
        source::DataSourceExec,
    },
    error::{DataFusionError, Result as DfResult},
    execution::object_store::ObjectStoreUrl,
    logical_expr::{Expr, Operator, TableProviderFilterPushDown, TableType},
    object_store::path::Path as ObjPath,
    physical_expr::PhysicalExpr,
    physical_plan::{ExecutionPlan, empty::EmptyExec, metrics::ExecutionPlanMetricsSet},
    scalar::ScalarValue,
};
use futures::{FutureExt, future::BoxFuture};
use object_store::ObjectStore as OsObjectStore;
use parquet::{
    arrow::{
        arrow_reader::{
            ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection, RowSelector,
        },
        async_reader::{AsyncFileReader, ParquetObjectReader},
    },
    errors,
    file::metadata::ParquetMetaData,
};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::{
    superfile::{
        LazyByteSource,
        fts::{
            reader::BoolMode,
            tokenize::{Tokenizer, unique_tokens},
        },
    },
    supertable::{
        SuperfileEntry,
        manifest::{ManifestSnapshot, add_sum_arrays, hll::HllSketch},
        options::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
        query::{
            candidate::CandidatePlan,
            df_object_store::SuperfileObjectStore,
            prune::{PruneLeaf, select_superfiles},
            skip::{ScalarOp, ScalarPredicate},
            superfile_reader::superfile_reader,
        },
        reader_cache::{DiskCacheStore, SuperfileReaderCache},
        tombstones::SidecarCache,
    },
};

/// Logical name the supertable is registered under in the
/// DataFusion `SessionContext`. Callers reference it as
/// `FROM supertable`; we also use it as the schema qualifier when
/// resolving filter columns to a physical pruning predicate.
pub(crate) const TABLE_NAME: &str = "supertable";

/// Object-store URL *prefix* the surviving superfiles are registered under
/// for a scan. The authority is arbitrary — only a key into the session's
/// object-store registry — but it must be **unique per provider**: a
/// multi-table catalog query (`Connection::query_sql`) registers several
/// providers into one DataFusion session, and a shared key would let one
/// table's store overwrite another's, so the shadowed table's superfiles
/// would read "not found". A process-global counter makes each provider's
/// URL distinct while staying stable across that provider's own scans (so
/// the cached single-table session re-registers the same key, no growth).
const SUPERFILE_STORE_URL_PREFIX: &str = "superfile://supertable-";

/// Monotonic source of per-provider object-store authorities.
static STORE_URL_SEQ: atomic::AtomicU64 = atomic::AtomicU64::new(0);

/// Selectivity gate for the FTS `WHERE` pushdown: only push an index
/// candidate set into the scan when the estimated match count is at
/// most this fraction of the superfile's rows. Above it, matches saturate
/// the Parquet data pages so an index `RowSelection` can't skip any —
/// a plain scan is cheaper than the posting walk + selection overhead.
const PUSHDOWN_MAX_FRACTION: f64 = 0.01;

/// Floor for the gate so the pushdown stays active on small superfiles
/// (where `n_docs * fraction` rounds to ~0 but there's no page-skip
/// tradeoff to lose anyway).
const PUSHDOWN_MIN_ROWS: u64 = 4096;

/// Density ceiling that binds even under [`PUSHDOWN_MIN_ROWS`]: when
/// the estimate covers at least this fraction of a superfile's rows, a
/// selection can't skip anything no matter how small the superfile is,
/// so the posting walk + selection build is pure overhead. Measured at
/// 1M docs × 256 superfiles (each under the floor), an all-matching
/// `IN` aggregate ran 2.5× slower through the index path than the
/// plain scan.
const PUSHDOWN_MAX_DENSITY: f64 = 0.5;

/// A [`TableProvider`] over a pinned supertable snapshot.
///
/// Cheap to build (just `Arc` clones); all real work happens in
/// [`scan`](TableProvider::scan), which is invoked per physical
/// plan. See the module docs for the two-tier pruning model.
pub(crate) struct SupertableProvider {
    /// User-visible scalar schema (`_id` + scalar + FTS columns).
    /// Matches the Parquet body each superfile was written with.
    schema: SchemaRef,
    /// Pinned manifest snapshot for this query.
    manifest: Arc<ManifestSnapshot>,
    /// In-memory superfile-bytes tier.
    store: Arc<dyn SuperfileReaderCache>,
    /// Optional disk cache (storage-backed supertables).
    disk_cache: Option<Arc<DiskCacheStore>>,
    /// Per-superfile soft-delete (tombstone) overlay. `None` for
    /// in-memory tables with no WAL/mutation surface. When present,
    /// [`scan`](TableProvider::scan) pushes the tombstoned rows into
    /// each superfile's Parquet read as a [`ParquetAccessPlan`] row
    /// selection — the *lazy* delete path: deleted rows are skipped
    /// during decode rather than materialized then dropped. This
    /// keeps the analytical SELECT path's projection/limit/row-group
    /// pushdown intact while still honoring deletes.
    tombstone_cache: Option<Arc<SidecarCache>>,
    /// Per-provider object-store registry key (see
    /// [`SUPERFILE_STORE_URL_PREFIX`]). Unique across providers so a
    /// multi-table query's registrations don't collide.
    store_url: ObjectStoreUrl,
    /// Restriction to a segment subset, set only by the
    /// covered/residual aggregate rewrite (the residual scan reads
    /// boundary segments only; covered segments were answered from
    /// manifest statistics). `None` = the whole snapshot. Also the
    /// rewrite's idempotency guard: a restricted provider is never
    /// rewritten again.
    segment_filter: Option<HashSet<Uuid>>,
}

/// Manual `Debug` (required by `TableProvider`): the cache /
/// disk-cache fields are trait objects without a `Debug` bound, so
/// we print a structural summary instead.
impl fmt::Debug for SupertableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SupertableProvider")
            .field("schema", &self.schema)
            .field("n_superfiles", &self.manifest.superfiles.len())
            .field("has_disk_cache", &self.disk_cache.is_some())
            .field("has_tombstone_cache", &self.tombstone_cache.is_some())
            .finish()
    }
}

impl SupertableProvider {
    /// Build a provider over a pinned snapshot. The arguments
    /// mirror what `Supertable::query_sql` already pins.
    pub(crate) fn new(
        schema: SchemaRef,
        manifest: Arc<ManifestSnapshot>,
        store: Arc<dyn SuperfileReaderCache>,
        disk_cache: Option<Arc<DiskCacheStore>>,
        tombstone_cache: Option<Arc<SidecarCache>>,
    ) -> Self {
        let seq = STORE_URL_SEQ.fetch_add(1, atomic::Ordering::Relaxed);
        let store_url = ObjectStoreUrl::parse(format!("{SUPERFILE_STORE_URL_PREFIX}{seq}/"))
            .expect("invariant: a counter-derived store URL is always valid");
        Self {
            schema,
            manifest,
            store,
            disk_cache,
            tombstone_cache,
            store_url,
            segment_filter: None,
        }
    }

    /// Clone of this provider restricted to `segments` — used by the
    /// covered/residual aggregate rewrite for the residual (boundary
    /// segment) scan. Gets its own object-store registry key so the
    /// restricted scan's registration can't collide with the parent's.
    pub(crate) fn restricted_to(&self, segments: HashSet<Uuid>) -> Self {
        let mut restricted = Self::new(
            Arc::clone(&self.schema),
            Arc::clone(&self.manifest),
            Arc::clone(&self.store),
            self.disk_cache.clone(),
            self.tombstone_cache.clone(),
        );
        restricted.segment_filter = Some(segments);
        restricted
    }

    /// `true` when this provider is a covered/residual residual scan —
    /// the rewrite's idempotency guard.
    pub(crate) fn is_segment_restricted(&self) -> bool {
        self.segment_filter.is_some()
    }

    /// The pinned manifest snapshot (covered/residual rewrite input).
    pub(crate) fn manifest(&self) -> &Arc<ManifestSnapshot> {
        &self.manifest
    }

    /// Whether `entry` currently has a clean (empty, resolvable from
    /// cache) tombstone view — the precondition for answering any part
    /// of an aggregate from its manifest statistics.
    pub(crate) fn entry_is_clean(&self, entry: &SuperfileEntry) -> bool {
        match self.tombstone_cache.as_ref() {
            None => true,
            Some(cache) => cache
                .bitmap_for(entry.superfile_id, Instant::now())
                .map(|bitmap| bitmap.is_empty())
                .unwrap_or(false),
        }
    }

    /// Lower scalar predicates to prune leaves. Each predicate yields a
    /// `Scalar` leaf; additionally, an equality on an FTS-indexed text
    /// column also yields a `TermPresence` leaf so the superfile's term
    /// bloom prunes it. Sound: a row matching `col = 'a b'` has a value
    /// whose tokens include every token of the literal, so requiring all
    /// of them possibly-present (`BoolMode::And`) never drops a match —
    /// bloom false positives can only keep a superfile, never drop one.
    fn predicates_to_prune_leaves(&self, predicates: Vec<ScalarPredicate>) -> Vec<PruneLeaf> {
        let opts = &self.manifest.options;
        let mut leaves = Vec::with_capacity(predicates.len());
        for pred in predicates {
            if pred.op == ScalarOp::Eq
                && opts.fts_columns.iter().any(|c| c.column == pred.column)
                && let Some(tok) = opts.tokenizer.as_ref()
                && let Some(literal) = scalar_as_str(&pred.value)
            {
                let terms: Vec<String> = tok.tokenize(literal).collect();
                if !terms.is_empty() {
                    leaves.push(PruneLeaf::TermPresence {
                        column: pred.column.clone(),
                        terms,
                        mode: BoolMode::And,
                    });
                }
            }
            leaves.push(PruneLeaf::Scalar(pred));
        }
        leaves
    }

    // Lower `filters` to prune leaves and select the superfiles that
    // survive the two-tier prune — per-part aggregates (ManifestPartEntry)
    // first, then per-superfile stats (SuperfileEntry).
    //
    // Pure manifest work: reads stats only, opens no superfile. Returns the
    // survivor entries; `scan` is what opens and reads them.
    async fn select_survivors(&self, filters: &[Expr]) -> DfResult<Vec<Arc<SuperfileEntry>>> {
        let predicates = exprs_to_scalar_predicates(filters, &self.schema);
        let mut leaves = self.predicates_to_prune_leaves(predicates);

        leaves.extend(exprs_to_in_list_leaves(
            filters,
            &self.schema,
            &self.fts_cols_set(),
            self.manifest.options.tokenizer.as_deref(),
        ));

        let mut survivors = select_superfiles(self.manifest.as_ref(), &leaves)
            .await
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

        // Covered/residual residual scans read only their boundary
        // superfiles; everything else was answered from statistics.
        if let Some(allowed) = self.segment_filter.as_ref() {
            survivors.retain(|entry| allowed.contains(&entry.superfile_id));
        }

        Ok(survivors)
    }

    /// The set of FTS-indexed column names — used by the candidate
    /// planner and by `supports_filters_pushdown` to decide which
    /// filters the index can resolve.
    fn fts_cols_set(&self) -> HashSet<&str> {
        self.manifest
            .options
            .fts_columns
            .iter()
            .map(|c| c.column.as_str())
            .collect()
    }

    /// Test hook: how many superfiles survive pruning for `filters` — the
    /// observable behind "did the index prune more than min/max?".
    #[cfg(test)]
    pub(crate) async fn surviving_superfile_count(&self, filters: &[Expr]) -> usize {
        self.select_survivors(filters)
            .await
            .expect("select survivors")
            .len()
    }

    /// Statistics over `entries`, assembled entirely from the manifest
    /// (plus cached tombstone views) — no I/O, no scan:
    ///
    ///   * `num_rows` is **Exact** (Σ `n_docs` − Σ tombstone-bitmap
    ///     cardinalities) when every entry's tombstone view resolves
    ///     from cache; Inexact (Σ `n_docs`) otherwise.
    ///   * per-column min/max come from the manifest's per-segment
    ///     skip stats (`scalar_stats`, plus the dedicated `_id`
    ///     range) and are **Exact only on a tombstone-free view** — a
    ///     deleted row may hold the extremum — Inexact otherwise.
    ///
    /// Exact statistics let DataFusion's `AggregateStatistics` rule
    /// fold `COUNT(*)` / `MIN` / `MAX` into literals, eliminating the
    /// scan entirely; Inexact ones still feed planner estimates.
    fn statistics_for(&self, entries: &[Arc<SuperfileEntry>]) -> Statistics {
        let total_rows: u64 = entries.iter().map(|e| e.n_docs).sum();
        let now = Instant::now();

        // Tombstone view: resolved-from-cache only (this path must be
        // sync and I/O-free). A missing/stale view degrades to
        // Inexact, never blocks.
        let mut deleted: u64 = 0;
        let mut views_resolved = true;
        if let Some(cache) = self.tombstone_cache.as_ref() {
            for entry in entries {
                match cache.bitmap_for(entry.superfile_id, now) {
                    Ok(bitmap) => deleted += bitmap.len(),
                    Err(_) => {
                        views_resolved = false;
                        break;
                    }
                }
            }
        }
        let num_rows = if views_resolved {
            Precision::Exact((total_rows - deleted) as usize)
        } else {
            Precision::Inexact(total_rows as usize)
        };
        let clean = views_resolved && deleted == 0;

        // Wrap a known stat in the exactness the tombstone view
        // allows: deleted rows may hold the extremum / contribute to
        // a sum, so anything value-derived is only Exact on a clean
        // view.
        let wrap = |v: ScalarValue| {
            if clean {
                Precision::Exact(v)
            } else {
                Precision::Inexact(v)
            }
        };
        let id_column = self.manifest.options.id_column.as_str();
        let column_statistics = self
            .schema
            .fields()
            .iter()
            .map(|field| {
                let name = field.name().as_str();
                if name == id_column {
                    // `_id` is engine-injected: non-null and unique by
                    // construction, range tracked in the manifest.
                    let mut stats = ColumnStatistics::new_unknown();
                    if let Some((min, max)) = id_min_max(entries) {
                        stats.min_value = wrap(min);
                        stats.max_value = wrap(max);
                    }
                    stats.null_count = Precision::Exact(0);
                    stats.distinct_count = num_rows;
                    return stats;
                }
                let mut stats = ColumnStatistics::new_unknown();
                if let Some((min, max)) = scalar_min_max(entries, name) {
                    stats.min_value = wrap(min);
                    stats.max_value = wrap(max);
                }
                if let Some(nulls) = scalar_null_count(entries, name) {
                    stats.null_count = if clean {
                        Precision::Exact(nulls as usize)
                    } else {
                        Precision::Inexact(nulls as usize)
                    };
                }
                if let Some(sum) = scalar_sum(entries, name) {
                    stats.sum_value = wrap(sum);
                }
                if let Some(distinct) = scalar_distinct(entries, name) {
                    // A sketch estimate — never exact.
                    stats.distinct_count = Precision::Inexact(distinct);
                }
                stats
            })
            .collect();

        Statistics {
            num_rows,
            total_byte_size: Precision::Absent,
            column_statistics,
        }
    }
}

/// Min/max of the supertable-injected `_id` column across `entries`,
/// from the manifest's dedicated id range fields.
fn id_min_max(entries: &[Arc<SuperfileEntry>]) -> Option<(ScalarValue, ScalarValue)> {
    let min = entries.iter().map(|e| e.id_min).min()?;
    let max = entries.iter().map(|e| e.id_max).max()?;
    Some((
        ScalarValue::Decimal128(Some(min), DECIMAL128_PRECISION, DECIMAL128_SCALE),
        ScalarValue::Decimal128(Some(max), DECIMAL128_PRECISION, DECIMAL128_SCALE),
    ))
}

/// Total null count of column `name` across `entries`; `None` unless
/// every entry carries the stat (a missing side makes the total
/// unknowable).
fn scalar_null_count(entries: &[Arc<SuperfileEntry>], name: &str) -> Option<u64> {
    entries.iter().try_fold(0u64, |acc, entry| {
        acc.checked_add(entry.scalar_stats.get(name)?.null_count?)
    })
}

/// Exact sum of column `name` across `entries`; `None` unless every
/// entry carries it and the fold doesn't overflow.
fn scalar_sum(entries: &[Arc<SuperfileEntry>], name: &str) -> Option<ScalarValue> {
    let mut acc: Option<ArrayRef> = None;
    for entry in entries {
        let part = entry.scalar_stats.get(name)?.sum.as_ref()?;
        acc = Some(match acc {
            None => Arc::clone(part),
            Some(total) => add_sum_arrays(&total, part)?,
        });
    }
    ScalarValue::try_from_array(&acc?, 0).ok()
}

/// HLL distinct-count estimate for column `name` across `entries`;
/// `None` unless every entry carries a sketch. Sketch unions are
/// exact, so the merged estimate has single-sketch accuracy.
fn scalar_distinct(entries: &[Arc<SuperfileEntry>], name: &str) -> Option<usize> {
    let mut merged: Option<HllSketch> = None;
    for entry in entries {
        let sketch = HllSketch::from_bytes(entry.scalar_stats.get(name)?.hll.as_ref()?)?;
        merged = Some(match merged {
            None => sketch,
            Some(mut acc) => {
                acc.merge(&sketch);
                acc
            }
        });
    }
    Some(merged?.estimate().round() as usize)
}

fn scalar_min_max(
    entries: &[Arc<SuperfileEntry>],
    name: &str,
) -> Option<(ScalarValue, ScalarValue)> {
    let mut acc: Option<(ScalarValue, ScalarValue)> = None;
    for entry in entries {
        let agg = entry.scalar_stats.get(name)?;
        let min = ScalarValue::try_from_array(&agg.min, 0).ok()?;
        let max = ScalarValue::try_from_array(&agg.max, 0).ok()?;
        if min.is_null() || max.is_null() {
            return None;
        }
        acc = match acc {
            None => Some((min, max)),
            Some((cur_min, cur_max)) => {
                let new_min = match min.partial_cmp(&cur_min)? {
                    cmp::Ordering::Less => min,
                    _ => cur_min,
                };
                let new_max = match max.partial_cmp(&cur_max)? {
                    cmp::Ordering::Greater => max,
                    _ => cur_max,
                };
                Some((new_min, new_max))
            }
        };
    }
    acc
}

/// Extract a UTF-8 string literal from a scalar value, if it is one.
/// Used to tokenize an equality literal for FTS-bloom pruning.
fn scalar_as_str(v: &ScalarValue) -> Option<&str> {
    match v {
        ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => Some(s.as_str()),
        _ => None,
    }
}

#[async_trait]
impl TableProvider for SupertableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Report every filter as `Inexact`: DataFusion hands us the
    /// predicates (for both pruning tiers) **and** keeps a
    /// `FilterExec` above the scan, so correctness never depends on
    /// our conservative pruning. The `FilterExec` also does the
    /// candidate-superset verification in the same scan pass as the
    /// projection (one decode), which a self-verifying `exact_match`
    /// candidate would split into an extra pass — measured slower.
    /// Returning `Unsupported` (the default) would withhold the filters
    /// from [`scan`] entirely, disabling superfile + row-group skip.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    /// Whole-table statistics from the manifest (no I/O) — feeds
    /// logical planning; the physical fold reads the scan node's
    /// statistics, attached in [`scan`](Self::scan).
    fn statistics(&self) -> Option<Statistics> {
        // Hierarchical manifests keep segments inside lazily-loaded
        // parts; the flat `superfiles` view may be empty or partial
        // there, so whole-table claims would be wrong. The scan-level
        // statistics (computed from the actually-flattened survivors)
        // still apply in that mode.
        if !self.manifest.is_in_process_only() {
            return None;
        }
        Some(self.statistics_for(&self.manifest.superfiles))
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        // Superfile selection via the shared two-tier prune (the same
        // path FTS search uses); see `select_survivors`. Survivors go to
        // DataFusion.
        let survivor_entries = self.select_survivors(filters).await?;
        let survivors: Vec<&Arc<SuperfileEntry>> = survivor_entries.iter().collect();

        // Nothing survived (empty table, or every superfile pruned):
        // a schema-correct empty scan. EmptyExec yields one
        // partition / zero rows, so `COUNT(*)` is 0 and `SELECT *`
        // returns the right empty shape. The projection must be
        // honored here too — `COUNT(*)` projects zero columns, and
        // DataFusion checks the physical schema against the logical
        // one.
        if survivors.is_empty() {
            let projected = match projection {
                Some(indices) => Arc::new(self.schema.project(indices)?),
                None => Arc::clone(&self.schema),
            };
            return Ok(Arc::new(EmptyExec::new(projected)));
        }

        // One `Instant::now()` for the whole scan so every per-superfile
        // tombstone lookup shares the same `SidecarCache` TTL
        // reference.
        let now = Instant::now();

        // Warm every surviving superfile's tombstone bitmap in one
        // batched fetch before the per-superfile sweep below, mirroring
        // the bm25 / vector fan-out (see `SidecarCache::prefetch`);
        // each `bitmap_for` in the loop then resolves from cache.
        if let Some(cache) = self.tombstone_cache.as_ref() {
            let ids: Vec<_> = survivors.iter().map(|e| e.superfile_id).collect();
            cache.prefetch(&ids, now).await;
        }

        // Pass 1 — build the index candidate plan once for this scan. It
        // lowers the FTS-resolvable part of the `WHERE` clause to a
        // boolean tree over `token_match`; evaluated per superfile below
        // it yields a candidate row-id superset (or `Unbounded` = scan
        // the superfile). See `crate::supertable::query::candidate`.
        let candidate_plan = CandidatePlan::from_filters(
            filters,
            &self.fts_cols_set(),
            self.manifest.options.tokenizer.as_ref(),
        );

        // Every surviving superfile is read through the one byte path the
        // FTS/vector resolve path uses: `superfile_reader(...)` ->
        // `SuperfileReader::byte_source()`. Warm / mmap superfiles slice
        // their resident bytes zero-copy; cold superfiles range-fetch from
        // object storage through the same source. The provider never
        // branches on which — that distinction lives entirely inside the
        // byte source.
        let mut sources: HashMap<ObjPath, Arc<dyn LazyByteSource>> = HashMap::new();

        // Per-superfile scan inputs, resolved into PartitionedFiles once the
        // store is built (row-group counts are read from each superfile's
        // footer through the same byte source).
        struct SuperfileScan {
            path: ObjPath,
            size: u64,
            candidates: Option<RoaringBitmap>,
            tombstones: Arc<RoaringBitmap>,
            /// Footer metadata the segment's reader parsed at open —
            /// row-group counts and DataFusion's opener both read from
            /// this; the scan never re-parses a footer.
            parquet_meta: Arc<ParquetMetaData>,
        }
        let mut superfiles: Vec<SuperfileScan> = Vec::with_capacity(survivors.len());

        for entry in &survivors {
            let reader = superfile_reader(
                &self.store,
                self.disk_cache.as_ref(),
                self.manifest.options.storage.as_ref(),
                &entry.uri,
                entry.subsection_offsets.as_ref(),
            )
            .await
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

            // Pass 1 (per superfile): resolve candidate rows from the
            // index. `None` => no usable bound, scan the superfile.
            //
            // Selectivity gate: estimate the match count from per-term
            // `df` first (cheap, header-only). If a predicate would match
            // more than `PUSHDOWN_MAX_FRACTION` of this superfile, skip the
            // index path and let DataFusion scan: at that match density
            // the rows saturate the data pages, so an index `RowSelection`
            // can't skip any page and only adds posting-walk + selection
            // overhead. The floor keeps the pushdown active on small
            // superfiles; the density cap binds even under the floor so
            // an all-matching predicate never takes the index path.
            let est = candidate_plan
                .estimate(reader.as_ref())
                .await
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            let gate =
                ((reader.n_docs() as f64 * PUSHDOWN_MAX_FRACTION) as u64).max(PUSHDOWN_MIN_ROWS);
            let density_cap = (reader.n_docs() as f64 * PUSHDOWN_MAX_DENSITY) as u64;
            let candidates = if est > gate || est >= density_cap {
                None
            } else {
                candidate_plan
                    .evaluate(reader.as_ref())
                    .await
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?
            };

            // This superfile's tombstoned rows (empty when no overlay).
            let tombstones = match self.tombstone_cache.as_ref() {
                Some(cache) => cache
                    .bitmap_for(entry.superfile_id, now)
                    .map_err(|e| DataFusionError::Execution(format!("tombstone cache: {e}")))?,
                None => Arc::new(RoaringBitmap::new()),
            };

            let source = reader.byte_source();
            let size = source.size();
            let path = ObjPath::from(entry.uri.storage_path());
            sources.insert(path.clone(), source);
            superfiles.push(SuperfileScan {
                path,
                size,
                candidates,
                tombstones,
                parquet_meta: Arc::clone(reader.parquet_metadata()),
            });
        }

        // The single object store DataFusion reads every survivor through.
        let store: Arc<dyn OsObjectStore> = Arc::new(SuperfileObjectStore::from_sources(sources));

        // Build each superfile's PartitionedFile + access plan. An access
        // plan exists only when the index bounded the rows or tombstones
        // must be skipped; a plain full scan needs no row-group map at
        // all. Row-group counts come from the footer metadata the
        // reader parsed at open — never re-read per query.
        let mut files: Vec<PartitionedFile> = Vec::with_capacity(superfiles.len());
        for seg in &superfiles {
            let access_plan = if seg.candidates.is_some() || !seg.tombstones.is_empty() {
                let row_counts: Vec<u32> = seg
                    .parquet_meta
                    .row_groups()
                    .iter()
                    .map(|rg| rg.num_rows() as u32)
                    .collect();
                build_access_plan(&row_counts, &seg.candidates, &seg.tombstones)
            } else {
                None
            };
            let mut file = PartitionedFile::new(seg.path.to_string(), seg.size);
            if let Some(plan) = access_plan {
                file = file.with_extensions(Arc::new(plan));
            }
            files.push(file);
        }

        // Tier 2 - DataFusion-owned row-group / page pruning + row-level
        // filter pushdown, used **only when the index could not bound the
        // rows** (`Unbounded` candidate plan). In that fallback the
        // predicate becomes a Parquet `RowFilter` (`with_pushdown_filters`)
        // so the predicate columns are decoded first and only surviving
        // rows materialize.
        //
        // When the index *did* bound the rows, the per-superfile access plan
        // already selects exactly the candidate rows and the `FilterExec`
        // above (filters are `Inexact`) verifies the exact predicate over
        // that tiny set. So we attach the pushdown predicate only on the
        // unbounded path.
        let index_bounded = !matches!(candidate_plan, CandidatePlan::Unbounded);
        let predicate = if !index_bounded {
            row_group_predicate(state, filters, &self.schema)
        } else {
            None
        };

        // Only push the LIMIT into the scan when there are no filters:
        // with an `Inexact` filter re-applied above, a scan-level limit
        // could stop before enough matching rows are produced. With no
        // filters, DataFusion's own limit and a scan-level limit agree.
        let effective_limit = if filters.is_empty() { limit } else { None };

        let mut source = ParquetSource::new(Arc::clone(&self.schema));
        if let Some(predicate) = predicate.as_ref() {
            source = source
                .with_predicate(Arc::clone(predicate))
                .with_pushdown_filters(true)
                .with_reorder_filters(true);
        }
        // Serve DataFusion's opener the footers the readers already
        // parsed — without this the opener re-reads + re-parses every
        // superfile's footer on every query (~half the warm flat cost
        // at 256 superfiles).
        let metas: HashMap<ObjPath, Arc<ParquetMetaData>> = superfiles
            .iter()
            .map(|s| (s.path.clone(), Arc::clone(&s.parquet_meta)))
            .collect();
        source = source.with_parquet_file_reader_factory(Arc::new(CachedMetadataReaderFactory {
            store: Arc::clone(&store),
            metas,
        }));

        // Manifest-derived statistics for the scan node. Exact only
        // when the scan emits exactly the survivor rows minus
        // tombstones — i.e. no filters (no index candidate bounding,
        // no FilterExec re-verification above). With filters the scan
        // may emit fewer rows than the manifest totals, so everything
        // degrades to inexact estimates.
        let scan_stats = {
            let stats = self.statistics_for(&survivor_entries);
            if filters.is_empty() {
                stats
            } else {
                stats.to_inexact()
            }
        };

        let url = self.store_url.clone();
        state
            .runtime_env()
            .register_object_store(url.as_ref(), store);
        let mut builder = FileScanConfigBuilder::new(url, Arc::new(source));
        for file in files {
            builder = builder.with_file(file);
        }
        let config = builder
            .with_statistics(scan_stats)
            .with_projection_indices(projection.cloned())?
            .with_limit(effective_limit)
            .build();
        Ok(DataSourceExec::from_data_source(config))
    }
}

/// Build a [`ParquetAccessPlan`] that skips this superfile's
/// tombstoned rows during decode, or `None` if none of the deleted
/// `local_doc_id`s fall inside the file (so a plain full scan is
/// correct and cheaper than attaching an all-`Scan` plan).
///
/// `bitmap` holds the tombstoned `local_doc_id`s, where a row's
/// `local_doc_id` is its 0-based global position within the superfile's
/// Parquet body (row groups are laid out in append order, so global
/// position partitions contiguously across them). For each row group
/// we translate the deleted positions into a [`RowSelection`] of
/// alternating select/skip runs; fully-deleted row groups are skipped
/// outright and clean ones are left as `Scan`.
///
/// Parsing the footer via [`ParquetRecordBatchReaderBuilder`] only
/// touches metadata, not column data, and only happens when the
/// superfile actually has tombstones — clean tables pay nothing.
/// Byte-sourced wrapper over [`tombstone_access_plan_from_counts`]. The
/// scan paths call the counts core directly via [`build_access_plan`];
/// this wrapper serves callers that hold the raw Parquet bytes — the
/// superfile reader's deleted-docs batching
/// ([`SuperfileReader`](crate::superfile::SuperfileReader)) and the
/// resident-bytes unit tests. In-crate callers only, so `pub(crate)`.
pub(crate) fn tombstone_access_plan(
    parquet_bytes: &Bytes,
    bitmap: &RoaringBitmap,
) -> DfResult<Option<ParquetAccessPlan>> {
    Ok(tombstone_access_plan_from_counts(
        &row_group_rows_from_bytes(parquet_bytes)?,
        bitmap,
    ))
}

/// Counts-based core of [`tombstone_access_plan`]: `row_counts[i]` is the
/// row count of row group `i`. Lets the object-store scan path build the
/// plan from a lazily-fetched footer (no whole-superfile bytes).
fn tombstone_access_plan_from_counts(
    row_counts: &[u32],
    bitmap: &RoaringBitmap,
) -> Option<ParquetAccessPlan> {
    // Sorted ascending — `RoaringBitmap::iter` yields in order, which
    // lets each row group binary-search its slice of deleted ids.
    let deleted: Vec<u32> = bitmap.iter().collect();

    let mut plan = ParquetAccessPlan::new_all(row_counts.len());
    let mut base: u32 = 0;
    let mut any = false;
    for (idx, &n) in row_counts.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let lo = deleted.partition_point(|&x| x < base);
        let hi = deleted.partition_point(|&x| x < base + n);
        let rg_deleted = &deleted[lo..hi];
        if rg_deleted.is_empty() {
            base += n;
            continue;
        }
        any = true;
        if rg_deleted.len() as u32 == n {
            plan.skip(idx);
            base += n;
            continue;
        }
        // Coalesce consecutive deleted positions into single skip runs,
        // emitting the live gaps between them as select runs.
        let mut selectors: Vec<RowSelector> = Vec::new();
        let mut cursor: u32 = 0; // next un-emitted position, relative to row group
        let mut i = 0usize;
        while i < rg_deleted.len() {
            let start_rel = rg_deleted[i] - base;
            if start_rel > cursor {
                selectors.push(RowSelector::select((start_rel - cursor) as usize));
            }
            let mut j = i;
            while j + 1 < rg_deleted.len() && rg_deleted[j + 1] == rg_deleted[j] + 1 {
                j += 1;
            }
            let run = (rg_deleted[j] - rg_deleted[i] + 1) as usize;
            selectors.push(RowSelector::skip(run));
            cursor = (rg_deleted[j] - base) + 1;
            i = j + 1;
        }
        if cursor < n {
            selectors.push(RowSelector::select((n - cursor) as usize));
        }
        plan.scan_selection(idx, RowSelection::from(selectors));
        base += n;
    }

    any.then_some(plan)
}

/// Row counts per row group, parsed from a resident parquet footer.
/// The scan path reads row-group counts from each reader's open-time
/// footer parse; this byte-sourced variant backs
/// [`tombstone_access_plan`] — the superfile reader's deleted-docs
/// batching and the resident-bytes unit tests.
fn row_group_rows_from_bytes(parquet_bytes: &Bytes) -> DfResult<Vec<u32>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_bytes.clone())
        .map_err(|e| DataFusionError::Execution(format!("parquet metadata: {e}")))?;
    Ok(builder
        .metadata()
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows() as u32)
        .collect())
}

/// Serves DataFusion's parquet opener the footer metadata each
/// segment's [`SuperfileReader`] parsed at open, so a scan never
/// re-reads or re-parses footers. Byte ranges still flow through the
/// unified [`SuperfileObjectStore`] (zero-copy slices on warm
/// segments, range GETs on cold ones).
///
/// [`SuperfileReader`]: crate::superfile::SuperfileReader
struct CachedMetadataReaderFactory {
    store: Arc<dyn OsObjectStore>,
    metas: HashMap<ObjPath, Arc<ParquetMetaData>>,
}

impl fmt::Debug for CachedMetadataReaderFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CachedMetadataReaderFactory")
            .field("superfiles", &self.metas.len())
            .finish()
    }
}

/// [`AsyncFileReader`] handed to DataFusion's opener: byte ranges
/// delegate to the plain object-store reader; `get_metadata` returns
/// the open-time parse instead of re-fetching the footer.
struct CachedMetadataReader {
    inner: ParquetObjectReader,
    /// `None` only for a file outside the survivors loop (never
    /// expected) — falls back to the inner reader's footer fetch.
    meta: Option<Arc<ParquetMetaData>>,
}

impl AsyncFileReader for CachedMetadataReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, errors::Result<Bytes>> {
        self.inner.get_bytes(range)
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, errors::Result<Vec<Bytes>>> {
        self.inner.get_byte_ranges(ranges)
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, errors::Result<Arc<ParquetMetaData>>> {
        match self.meta.clone() {
            Some(meta) => async move { Ok(meta) }.boxed(),
            None => self.inner.get_metadata(options),
        }
    }
}

impl ParquetFileReaderFactory for CachedMetadataReaderFactory {
    fn create_reader(
        &self,
        _partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        _metrics: &ExecutionPlanMetricsSet,
    ) -> DfResult<Box<dyn AsyncFileReader + Send>> {
        let location = &partitioned_file.object_meta.location;
        let mut inner = ParquetObjectReader::new(Arc::clone(&self.store), location.clone())
            .with_file_size(partitioned_file.object_meta.size);
        if let Some(hint) = metadata_size_hint {
            inner = inner.with_footer_size_hint(hint);
        }
        Ok(Box::new(CachedMetadataReader {
            meta: self.metas.get(location).cloned(),
            inner,
        }))
    }
}

/// Assemble the per-superfile [`ParquetAccessPlan`] from row-group counts:
/// index candidates (minus tombstones) drive a selection plan; otherwise
/// tombstones alone drive a delete-skip plan; a clean full scan is `None`.
fn build_access_plan(
    row_counts: &[u32],
    candidates: &Option<RoaringBitmap>,
    tombstones: &RoaringBitmap,
) -> Option<ParquetAccessPlan> {
    match candidates {
        Some(keep) => {
            let mut keep = keep.clone();
            keep -= tombstones;
            Some(selection_access_plan_from_counts(row_counts, &keep))
        }
        None => {
            if tombstones.is_empty() {
                None
            } else {
                tombstone_access_plan_from_counts(row_counts, tombstones)
            }
        }
    }
}

/// Build a [`ParquetAccessPlan`] that decodes **only** the rows in
/// `keep`, skipping everything else - the inverse of the tombstone
/// plan. Used for index-driven row selection: the candidate planner
/// yields a small set of `local_doc_id`s (already minus tombstones),
/// and we want the Parquet reader to materialize the payload columns
/// for just those rows rather than scanning the superfile.
///
/// `keep`'s ids are `local_doc_id`s - global row positions in the
/// Parquet body - which partition contiguously across row groups laid
/// out in append order. An empty `keep` produces an all-skip plan (zero
/// rows decoded), the correct result for a superfile with no candidate.
/// `row_counts[i]` is the row count of row group `i`, read from the
/// superfile footer through the unified store via [`build_access_plan`].
fn selection_access_plan_from_counts(
    row_counts: &[u32],
    keep: &RoaringBitmap,
) -> ParquetAccessPlan {
    // Ascending — `RoaringBitmap::iter` yields sorted, so each row group
    // binary-searches its contiguous slice of kept ids.
    let kept: Vec<u32> = keep.iter().collect();

    let mut plan = ParquetAccessPlan::new_all(row_counts.len());
    let mut base: u32 = 0;
    for (idx, &n) in row_counts.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let lo = kept.partition_point(|&x| x < base);
        let hi = kept.partition_point(|&x| x < base + n);
        let rg_kept = &kept[lo..hi];
        if rg_kept.is_empty() {
            plan.skip(idx);
            base += n;
            continue;
        }
        if rg_kept.len() as u32 == n {
            // Every row in this group is a candidate — leave it as Scan.
            base += n;
            continue;
        }
        // Emit alternating skip(gap) / select(run) selectors so only the
        // kept rows are decoded.
        let mut selectors: Vec<RowSelector> = Vec::new();
        let mut cursor: u32 = 0; // next un-emitted position within the row group
        let mut i = 0usize;
        while i < rg_kept.len() {
            let start_rel = rg_kept[i] - base;
            if start_rel > cursor {
                selectors.push(RowSelector::skip((start_rel - cursor) as usize));
            }
            let mut j = i;
            while j + 1 < rg_kept.len() && rg_kept[j + 1] == rg_kept[j] + 1 {
                j += 1;
            }
            let run = (rg_kept[j] - rg_kept[i] + 1) as usize;
            selectors.push(RowSelector::select(run));
            cursor = (rg_kept[j] - base) + 1;
            i = j + 1;
        }
        if cursor < n {
            selectors.push(RowSelector::skip((n - cursor) as usize));
        }
        plan.scan_selection(idx, RowSelection::from(selectors));
        base += n;
    }
    plan
}

/// Lower a conjunction of DataFusion filter `Expr`s into infino's
/// [`ScalarPredicate`]s for superfile skip.
///
/// Each top-level filter is treated as a conjunct; nested `AND`s
/// are flattened. Only `column <op> literal` (and the mirrored
/// `literal <op> column`) shapes over a column present in `schema`
/// are recognized — everything else is silently dropped (it just
/// doesn't contribute pruning; `FilterExec` still applies it).
pub(crate) fn exprs_to_scalar_predicates(
    filters: &[Expr],
    schema: &SchemaRef,
) -> Vec<ScalarPredicate> {
    let mut out = Vec::new();
    for filter in filters {
        collect_conjuncts(filter, schema, &mut out);
    }
    out
}

/// Build prune leaves for the `column IN (...)` filters in `filters`. Each yields:
///  i)  a `ScalarInList` (min/max) leaf — on every column;
///  ii) plus, on an FTS-indexed column, a `TermPresence{Or}` leaf over the
///      values' tokens, which prunes on which superfiles hold the term.
///
/// A `NOT IN`, a non-literal item, a function-wrapped or unknown column
/// yields no leaf — that filter just isn't pruned (the scan stays correct).
///
/// Handles any `Expr::InList`, whatever its length. In practice SQL
/// planning rewrites a short `IN` first — a 1-value to `=`, a 2–3 value
/// to `OR` — so only 4+ value lists usually reach here, but that's the
/// optimizer's behavior, not a precondition this helper relies on.
pub(crate) fn exprs_to_in_list_leaves(
    filters: &[Expr],
    schema: &SchemaRef,
    fts_cols: &HashSet<&str>,
    tokenizer: Option<&dyn Tokenizer>,
) -> Vec<PruneLeaf> {
    let mut out = Vec::new();

    for filter in filters {
        collect_in_list_leaves(filter, schema, fts_cols, tokenizer, &mut out);
    }

    out
}

/// e.g. `product IN ('Orange Juice', 'Pineapple')` on an FTS column →
/// `[ TermPresence{Or, [orange, juice, pineapple]},
///    ScalarInList{product, ['Orange Juice', 'Pineapple']} ]`
/// — bloom leaf holds the lowercased tokens, min/max leaf the raw values.
///
/// Note: the bloom flattens all tokens into one `Or`, so a superfile with
/// only `orange` is kept though no value matches. A tighter prune would be
/// per-value `(orange AND juice) OR pineapple`; FilterExec verifies either way.
fn collect_in_list_leaves(
    expr: &Expr,
    schema: &SchemaRef,
    fts_cols: &HashSet<&str>,
    tokenizer: Option<&dyn Tokenizer>,
    out: &mut Vec<PruneLeaf>,
) {
    match expr {
        // Filters reach us alias-free (Filter::try_new runs unalias_nested),
        // but an alias is a pure rename; descend it so pruning is unaffected
        // if one ever survives (e.g. a metadata-carrying alias).
        Expr::Alias(a) => collect_in_list_leaves(&a.expr, schema, fts_cols, tokenizer, out),
        // Descend AND; an IN can sit on either side.
        Expr::BinaryExpr(be) if be.op == Operator::And => {
            collect_in_list_leaves(&be.left, schema, fts_cols, tokenizer, out);
            collect_in_list_leaves(&be.right, schema, fts_cols, tokenizer, out);
        }
        Expr::InList(il) if !il.negated => {
            // Only a bare column maps to a min/max or bloom; else skip.
            let Expr::Column(c) = il.expr.as_ref() else {
                return;
            };

            if schema.field_with_name(&c.name).is_err() {
                return;
            }

            // Every item must be a literal to bound min/max; else skip.
            let mut values = Vec::with_capacity(il.list.len());
            for item in &il.list {
                let Expr::Literal(v, _) = item else {
                    return;
                };
                values.push(v.clone());
            }

            // Empty IN list, nothing to bound.
            if values.is_empty() {
                return;
            }

            // Only for FTS indexed columns
            if fts_cols.contains(c.name.as_str())
                && let Some(tok) = tokenizer
            {
                // One deduped term set across all values — a word shared
                // by several values (e.g. 'Orange Juice', 'Apple Juice'
                // → one `juice`) is probed once.
                let terms = unique_tokens(tok, values.iter().filter_map(scalar_as_str));
                if !terms.is_empty() {
                    // one bloom leaf holding all the values' tokens
                    out.push(PruneLeaf::TermPresence {
                        column: c.name.clone(),
                        terms,
                        mode: BoolMode::Or,
                    });
                }
            }

            // The min/max leaf — on every column.
            out.push(PruneLeaf::ScalarInList {
                column: c.name.clone(),
                values,
            });
        }
        _ => {}
    }
}

/// Recurse through `AND` nodes, pushing any recognized
/// `column <op> literal` leaf into `out`.
fn collect_conjuncts(expr: &Expr, schema: &SchemaRef, out: &mut Vec<ScalarPredicate>) {
    if let Expr::BinaryExpr(be) = expr {
        if be.op == Operator::And {
            collect_conjuncts(&be.left, schema, out);
            collect_conjuncts(&be.right, schema, out);
        } else if let Some(p) = leaf_to_predicate(&be.left, be.op, &be.right, schema) {
            out.push(p);
        }
    }
}

/// Convert a single `left <op> right` comparison into a
/// [`ScalarPredicate`] when it's `column <op> literal` or
/// `literal <op> column` over a known column; else `None`.
fn leaf_to_predicate(
    left: &Expr,
    op: Operator,
    right: &Expr,
    schema: &SchemaRef,
) -> Option<ScalarPredicate> {
    let (column, value, scalar_op) = match (left, right) {
        (Expr::Column(c), Expr::Literal(v, _)) => (&c.name, v, map_op(op)?),
        (Expr::Literal(v, _), Expr::Column(c)) => (&c.name, v, flip_op(map_op(op)?)),
        _ => return None,
    };
    // Guard against columns not in the scalar schema (e.g. a typo
    // would already fail planning, but be defensive).
    schema.field_with_name(column).ok()?;
    Some(ScalarPredicate {
        column: column.clone(),
        op: scalar_op,
        value: value.clone(),
    })
}

/// Map a DataFusion comparison [`Operator`] to a [`ScalarOp`].
/// Non-comparison operators return `None` (no pruning).
fn map_op(op: Operator) -> Option<ScalarOp> {
    match op {
        Operator::Eq => Some(ScalarOp::Eq),
        Operator::NotEq => Some(ScalarOp::NotEq),
        Operator::Lt => Some(ScalarOp::Lt),
        Operator::LtEq => Some(ScalarOp::LtEq),
        Operator::Gt => Some(ScalarOp::Gt),
        Operator::GtEq => Some(ScalarOp::GtEq),
        _ => None,
    }
}

/// Flip a comparison so `literal <op> column` becomes the
/// equivalent `column <flipped> literal` (e.g. `5 < x` ⟺ `x > 5`).
fn flip_op(op: ScalarOp) -> ScalarOp {
    match op {
        ScalarOp::Eq => ScalarOp::Eq,
        ScalarOp::NotEq => ScalarOp::NotEq,
        ScalarOp::Lt => ScalarOp::Gt,
        ScalarOp::LtEq => ScalarOp::GtEq,
        ScalarOp::Gt => ScalarOp::Lt,
        ScalarOp::GtEq => ScalarOp::LtEq,
    }
}

/// Lower the conjunction of `filters` into a single physical
/// predicate for DataFusion's row-group pruning, or `None` if the
/// filters are empty or can't be lowered (column-resolution /
/// planning failure → skip pruning, never incorrect).
fn row_group_predicate(
    state: &dyn Session,
    filters: &[Expr],
    schema: &SchemaRef,
) -> Option<Arc<dyn PhysicalExpr>> {
    let combined = filters.iter().cloned().reduce(|a, b| a.and(b))?;
    // Filter columns may arrive qualified (`supertable.col`) or
    // bare depending on the plan; try the qualified schema first,
    // then the unqualified one.
    let df_schema = DFSchema::try_from_qualified_schema(TABLE_NAME, schema.as_ref())
        .or_else(|_| DFSchema::try_from(schema.as_ref().clone()))
        .ok()?;
    state.create_physical_expr(combined, &df_schema).ok()
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int64Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        prelude::{col, lit},
        scalar::ScalarValue,
    };
    use object_store::memory::InMemory;
    use rayon::ThreadPoolBuilder;
    use tokio::runtime;

    use super::*;
    use crate::{
        superfile::builder::FtsConfig,
        supertable::{
            Supertable, SupertableOptions,
            manifest::{ScalarStatsAgg, SuperfileUri},
        },
        test_helpers::default_tokenizer,
    };

    /// Build an in-memory Parquet file of `Int64` values `0..total`
    /// split into row groups of `rg_size` rows each.
    fn parquet_with_row_groups(total: i64, rg_size: usize) -> Bytes {
        use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let arr = Int64Array::from((0..total).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(arr)]).expect("batch");
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rg_size))
            .build();
        let mut buf = Vec::new();
        {
            let mut w =
                ArrowWriter::try_new(&mut buf, Arc::clone(&schema), Some(props)).expect("writer");
            w.write(&batch).expect("write");
            w.close().expect("close");
        }
        Bytes::from(buf)
    }

    /// Decode `bytes` honoring `plan`'s row-group + row selection and
    /// return the surviving `v` values in order.
    fn read_with_plan(bytes: &Bytes, plan: ParquetAccessPlan) -> Vec<i64> {
        let meta = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .expect("meta")
            .metadata()
            .clone();
        let row_groups = plan.row_group_indexes();
        let selection = plan
            .into_overall_row_selection(meta.row_groups())
            .expect("overall selection");
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .expect("builder")
            .with_row_groups(row_groups);
        if let Some(sel) = selection {
            builder = builder.with_row_selection(sel);
        }
        let reader = builder.build().expect("reader");
        let mut got = Vec::new();
        for b in reader {
            let b = b.expect("batch");
            let c = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int64 col");
            for i in 0..c.len() {
                got.push(c.value(i));
            }
        }
        got
    }

    #[test]
    fn tombstone_access_plan_none_when_no_deletes_in_file() {
        let bytes = parquet_with_row_groups(12, 4);
        // Tombstone an id past the end of the file → nothing selected.
        let mut bm = RoaringBitmap::new();
        bm.insert(99);
        assert!(
            tombstone_access_plan(&bytes, &bm).expect("plan").is_none(),
            "no deleted id falls inside the file → full scan (None)"
        );
    }

    #[test]
    fn tombstone_access_plan_skips_deleted_across_row_groups() {
        // 3 row groups of 4 rows: rg0=0..4, rg1=4..8, rg2=8..12.
        let bytes = parquet_with_row_groups(12, 4);

        // rg0: delete 0,1 (consecutive run at the start)
        // rg1: delete 4,5,6,7 (whole row group → Skip)
        // rg2: delete 10 (single row mid-group)
        let mut bm = RoaringBitmap::new();
        for id in [0u32, 1, 4, 5, 6, 7, 10] {
            bm.insert(id);
        }

        let plan = tombstone_access_plan(&bytes, &bm)
            .expect("plan")
            .expect("some deletes");

        // Whole-deleted row group is skipped entirely.
        assert!(!plan.should_scan(1), "fully-tombstoned row group 1 skipped");
        assert!(plan.should_scan(0));
        assert!(plan.should_scan(2));

        let survivors = read_with_plan(&bytes, plan);
        assert_eq!(survivors, vec![2, 3, 8, 9, 11]);
    }

    #[test]
    fn tombstone_access_plan_handles_alternating_and_boundary_deletes() {
        // Single row group of 8 rows with an alternating pattern plus
        // the last row deleted (exercises the trailing-select branch).
        let bytes = parquet_with_row_groups(8, 8);
        let mut bm = RoaringBitmap::new();
        for id in [0u32, 2, 4, 7] {
            bm.insert(id);
        }
        let plan = tombstone_access_plan(&bytes, &bm)
            .expect("plan")
            .expect("some deletes");
        let survivors = read_with_plan(&bytes, plan);
        assert_eq!(survivors, vec![1, 3, 5, 6]);
    }

    fn schema_xy() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("y", DataType::Int64, true),
        ]))
    }

    #[test]
    fn col_op_lit_maps_directly() {
        let s = schema_xy();
        let preds = exprs_to_scalar_predicates(&[col("x").gt(lit(5_i64))], &s);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].column, "x");
        assert_eq!(preds[0].op, ScalarOp::Gt);
        assert_eq!(preds[0].value, ScalarValue::Int64(Some(5)));
    }

    #[test]
    fn lit_op_col_flips_operator() {
        // `5 < x`  ⟺  `x > 5`
        let s = schema_xy();
        let preds = exprs_to_scalar_predicates(&[lit(5_i64).lt(col("x"))], &s);
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].column, "x");
        assert_eq!(preds[0].op, ScalarOp::Gt);
        assert_eq!(preds[0].value, ScalarValue::Int64(Some(5)));
    }

    #[test]
    fn and_is_flattened_into_two_predicates() {
        let s = schema_xy();
        let expr = col("x").gt_eq(lit(5_i64)).and(col("x").lt_eq(lit(8_i64)));
        let preds = exprs_to_scalar_predicates(&[expr], &s);
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].op, ScalarOp::GtEq);
        assert_eq!(preds[1].op, ScalarOp::LtEq);
    }

    #[test]
    fn multiple_top_level_filters_each_contribute() {
        let s = schema_xy();
        let preds =
            exprs_to_scalar_predicates(&[col("x").gt(lit(1_i64)), col("y").lt(lit(9_i64))], &s);
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].column, "x");
        assert_eq!(preds[1].column, "y");
    }

    #[test]
    fn col_op_col_is_ignored() {
        let s = schema_xy();
        let preds = exprs_to_scalar_predicates(&[col("x").gt(col("y"))], &s);
        assert!(preds.is_empty());
    }

    /// The `(column, value-count)` of each `ScalarInList` leaf, for asserting
    /// extraction without matching on the full enum.
    fn in_list_leaves(filters: &[Expr], schema: &SchemaRef) -> Vec<(String, usize)> {
        // No FTS columns / tokenizer → only the scalar min/max leaf.
        exprs_to_in_list_leaves(filters, schema, &HashSet::new(), None)
            .into_iter()
            .map(|l| match l {
                PruneLeaf::ScalarInList { column, values } => (column, values.len()),
                _ => panic!("expected a ScalarInList leaf"),
            })
            .collect()
    }

    #[test]
    fn in_list_lowers_to_one_leaf_with_all_values() {
        let s = schema_xy();
        let expr = col("x").in_list(vec![lit(1_i64), lit(2_i64), lit(3_i64)], false);
        assert_eq!(in_list_leaves(&[expr], &s), vec![("x".to_string(), 3)]);
    }

    #[test]
    fn in_list_under_and_is_found() {
        let s = schema_xy();
        let expr = col("x")
            .gt(lit(0_i64))
            .and(col("y").in_list(vec![lit(7_i64)], false));
        assert_eq!(in_list_leaves(&[expr], &s), vec![("y".to_string(), 1)]);
    }

    #[test]
    fn in_list_under_alias_is_found() {
        // Filters reach us unaliased, but the descent must still find an
        // IN wrapped in an alias if one ever survives (a pure rename
        // doesn't change the column the leaf prunes on).
        let s = schema_xy();
        let expr = col("x")
            .in_list(vec![lit(1_i64), lit(2_i64)], false)
            .alias("k");
        assert_eq!(in_list_leaves(&[expr], &s), vec![("x".to_string(), 2)]);
    }

    #[test]
    fn negated_in_list_emits_no_leaf() {
        let s = schema_xy();
        let expr = col("x").in_list(vec![lit(1_i64)], true);
        assert!(exprs_to_in_list_leaves(&[expr], &s, &HashSet::new(), None).is_empty());
    }

    #[test]
    fn in_list_with_non_literal_item_emits_no_leaf() {
        let s = schema_xy();
        // `x IN (1, y)` — `y` is a column, not a literal; can't bound min/max.
        let expr = col("x").in_list(vec![lit(1_i64), col("y")], false);
        assert!(exprs_to_in_list_leaves(&[expr], &s, &HashSet::new(), None).is_empty());
    }

    #[test]
    fn in_list_on_unknown_column_emits_no_leaf() {
        let s = schema_xy();
        let expr = col("z").in_list(vec![lit(1_i64)], false);
        assert!(exprs_to_in_list_leaves(&[expr], &s, &HashSet::new(), None).is_empty());
    }

    #[test]
    fn in_list_on_fts_column_also_emits_term_presence_bloom() {
        use crate::superfile::fts::tokenize::AsciiLowerTokenizer;
        let s = Arc::new(Schema::new(vec![Field::new("title", DataType::Utf8, true)]));
        let fts = HashSet::from(["title"]);
        let tok = AsciiLowerTokenizer;
        // 'Foo Bar' → [foo, bar]; 'Bar Baz' → [bar, baz]. The shared `bar`
        // is deduped, and the terms come out sorted-unique.
        let expr = col("title").in_list(vec![lit("Foo Bar"), lit("Bar Baz")], false);
        let leaves = exprs_to_in_list_leaves(&[expr], &s, &fts, Some(&tok));

        assert!(
            leaves
                .iter()
                .any(|l| matches!(l, PruneLeaf::ScalarInList { .. })),
            "scalar min/max leaf still emitted"
        );
        let (col_name, terms, mode) = leaves
            .iter()
            .find_map(|l| match l {
                PruneLeaf::TermPresence {
                    column,
                    terms,
                    mode,
                } => Some((column.as_str(), terms, *mode)),
                _ => None,
            })
            .expect("FTS column also emits a TermPresence bloom leaf");
        assert_eq!(col_name, "title");
        assert_eq!(mode, BoolMode::Or);
        assert_eq!(
            terms,
            &vec!["bar".to_string(), "baz".to_string(), "foo".to_string()],
            "tokens are deduped (shared `bar`) and sorted"
        );
    }

    #[test]
    fn in_list_on_non_fts_column_emits_only_scalar_leaf() {
        let s = schema_xy();
        let fts = HashSet::from(["title"]); // "x" not in the set
        let tok = crate::superfile::fts::tokenize::AsciiLowerTokenizer;
        let expr = col("x").in_list(vec![lit(1_i64), lit(2_i64), lit(3_i64), lit(4_i64)], false);
        let leaves = exprs_to_in_list_leaves(&[expr], &s, &fts, Some(&tok));
        assert_eq!(leaves.len(), 1);
        assert!(matches!(leaves[0], PruneLeaf::ScalarInList { .. }));
    }

    #[test]
    fn unknown_column_is_ignored() {
        let s = schema_xy();
        let preds = exprs_to_scalar_predicates(&[col("z").gt(lit(1_i64))], &s);
        assert!(preds.is_empty());
    }

    #[test]
    fn non_comparison_operator_is_ignored() {
        let s = schema_xy();
        // x + 1 (arithmetic) — not a comparison, no predicate.
        let preds = exprs_to_scalar_predicates(&[col("x") + lit(1_i64)], &s);
        assert!(preds.is_empty());
    }

    // ---- Superfile-prune contrast: index helps vs. doesn't ----------
    //
    // End-to-end through a real multi-superfile supertable: count how many
    // superfiles survive the scan's prune for different predicates. This
    // is the observable proof that the embedded FTS index prunes more
    // than the scalar min/max a plain Parquet scan relies on — and,
    // honestly, where it doesn't (full scans, non-FTS predicates).

    fn cat_title_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("category", DataType::LargeUtf8, false),
            Field::new("title", DataType::LargeUtf8, false),
        ]))
    }

    fn cat_title_opts() -> SupertableOptions {
        // One writer thread → one superfile per commit (deterministic
        // superfile counts).
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(
            cat_title_schema(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        )
        .expect("opts")
        .with_writer_pool(pool)
    }

    fn cat_title_batch(cats: &[&str], titles: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            cat_title_schema(),
            vec![
                Arc::new(LargeStringArray::from(cats.to_vec())),
                Arc::new(LargeStringArray::from(titles.to_vec())),
            ],
        )
        .expect("batch")
    }

    #[test]
    fn superfile_prune_index_helps_vs_does_not() {
        let st = Supertable::create(cat_title_opts()).expect("create");
        let mut w = st.writer().expect("writer");
        // Three superfiles. Every superfile's `title` lexicographic range
        // spans "mango", so scalar min/max can prune none of them — but
        // only the middle superfile actually holds the token.
        w.append(&cat_title_batch(&["lang", "lang"], &["aardvark", "zebra"]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&cat_title_batch(&["lang"], &["mango"]))
            .expect("a2");
        w.commit().expect("c2");
        w.append(&cat_title_batch(&["lang", "lang"], &["delta", "sigma"]))
            .expect("a3");
        w.commit().expect("c3");
        assert_eq!(st.reader().n_superfiles(), 3);

        let reader = st.reader();
        let provider = SupertableProvider::new(
            st.options().scalar_schema(),
            reader.manifest().clone(),
            st.options().store.clone(),
            st.options().disk_cache.clone(),
            reader.tombstone_cache.clone(),
        );
        let rt = runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("rt");

        // Index HELPS: the term bloom prunes the two wide-range superfiles
        // that min/max could not, leaving only the real holder.
        assert_eq!(
            rt.block_on(provider.surviving_superfile_count(&[col("title").eq(lit("mango"))])),
            1,
            "FTS bloom prunes to the single token holder"
        );

        // Index can't help a full scan — every superfile is read.
        assert_eq!(
            rt.block_on(provider.surviving_superfile_count(&[])),
            3,
            "no predicate → full scan, nothing pruned"
        );

        // Non-FTS predicate present in every superfile: no bloom to use,
        // and min/max can't prune (all categories equal) → nothing
        // pruned. This is the honest "index doesn't help" case.
        assert_eq!(
            rt.block_on(provider.surviving_superfile_count(&[col("category").eq(lit("lang"))])),
            3,
            "non-FTS predicate matching all superfiles prunes nothing"
        );
    }

    /// Build a provider over a freshly-committed two-superfile table
    /// (the `cat_title` schema), returning the provider and a runtime to
    /// drive its async surface.
    fn provider_over_two_superfiles() -> (SupertableProvider, runtime::Runtime) {
        let st = Supertable::create(cat_title_opts()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&cat_title_batch(&["a", "a"], &["alpha beta", "gamma"]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&cat_title_batch(&["b"], &["delta"])).expect("a2");
        w.commit().expect("c2");

        let reader = st.reader();
        let provider = SupertableProvider::new(
            st.options().scalar_schema(),
            reader.manifest().clone(),
            st.options().store.clone(),
            st.options().disk_cache.clone(),
            reader.tombstone_cache.clone(),
        );
        let rt = runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("rt");
        (provider, rt)
    }

    #[test]
    fn trait_accessors_and_debug() {
        let (provider, _rt) = provider_over_two_superfiles();

        // as_any downcasts back to the concrete provider.
        assert!(
            provider
                .as_any()
                .downcast_ref::<SupertableProvider>()
                .is_some()
        );
        // table_type is a base table.
        assert!(matches!(provider.table_type(), TableType::Base));
        // schema() returns the scalar schema (category + title + _id).
        let sch = provider.schema();
        assert!(sch.field_with_name("category").is_ok());
        assert!(sch.field_with_name("title").is_ok());

        // Debug renders a structural summary, not the trait-object fields.
        let dbg = format!("{provider:?}");
        assert!(dbg.contains("SupertableProvider"));
        assert!(dbg.contains("n_superfiles"));
    }

    #[test]
    fn supports_filters_pushdown_is_always_inexact() {
        let (provider, _rt) = provider_over_two_superfiles();
        let f1 = col("category").eq(lit("a"));
        let f2 = col("title").eq(lit("alpha"));
        let filters = [&f1, &f2];
        let pushdown = provider
            .supports_filters_pushdown(&filters)
            .expect("pushdown");
        assert_eq!(pushdown.len(), 2);
        assert!(
            pushdown
                .iter()
                .all(|p| matches!(p, TableProviderFilterPushDown::Inexact))
        );
    }

    #[test]
    fn statistics_exact_on_clean_in_memory_flat_manifest() {
        let (provider, _rt) = provider_over_two_superfiles();
        // In-memory (no manifest list) flat manifest → whole-table
        // statistics are available and the row count is Exact (3 docs,
        // no tombstones).
        let stats = provider.statistics().expect("flat-manifest statistics");
        assert!(matches!(stats.num_rows, Precision::Exact(3)));
        // One ColumnStatistics per scalar-schema field.
        assert_eq!(
            stats.column_statistics.len(),
            provider.schema().fields().len()
        );
    }

    #[test]
    fn manifest_accessor_and_restricted_to_idempotency_guard() {
        let (provider, _rt) = provider_over_two_superfiles();
        // Unrestricted provider is not a residual scan.
        assert!(!provider.is_segment_restricted());

        // manifest() exposes the pinned snapshot.
        let ids: Vec<Uuid> = provider
            .manifest()
            .superfiles
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        assert_eq!(ids.len(), 2);

        // restricted_to keeps only the named segment and flips the guard.
        let only_first: HashSet<Uuid> = [ids[0]].into_iter().collect();
        let restricted = provider.restricted_to(only_first);
        assert!(restricted.is_segment_restricted());
        // Both providers see the same pinned manifest (Arc::clone).
        assert!(Arc::ptr_eq(restricted.manifest(), provider.manifest()));
    }

    #[test]
    fn entry_is_clean_true_without_tombstone_overlay() {
        let (provider, _rt) = provider_over_two_superfiles();
        // In-memory tables carry no tombstone overlay, so every entry is
        // trivially clean.
        for entry in provider.manifest().superfiles.iter() {
            assert!(provider.entry_is_clean(entry));
        }
    }

    #[test]
    fn restricted_provider_scans_only_its_segment() {
        let (provider, rt) = provider_over_two_superfiles();
        let first = provider.manifest().superfiles[0].superfile_id;
        let only_first: HashSet<Uuid> = [first].into_iter().collect();
        let restricted = provider.restricted_to(only_first);
        // With no filters, the unrestricted provider keeps both
        // superfiles; the restricted one keeps only the allowed segment.
        assert_eq!(rt.block_on(provider.surviving_superfile_count(&[])), 2);
        assert_eq!(rt.block_on(restricted.surviving_superfile_count(&[])), 1);
    }

    // ---- Scalar aggregate helpers over a numeric column -------------

    fn num_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]))
    }

    fn num_opts() -> SupertableOptions {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        // No FTS column — this fixture exercises the scalar-stats path
        // (min/max/sum/null/distinct), not the term bloom.
        SupertableOptions::new(num_schema(), vec![], vec![], None)
            .expect("opts")
            .with_writer_pool(pool)
    }

    fn num_batch(vals: &[Option<i64>]) -> RecordBatch {
        RecordBatch::try_new(
            num_schema(),
            vec![Arc::new(Int64Array::from(vals.to_vec()))],
        )
        .expect("batch")
    }

    #[test]
    fn statistics_for_aggregates_scalar_stats_across_superfiles() {
        let st = Supertable::create(num_opts()).expect("create");
        let mut w = st.writer().expect("writer");
        // Superfile 1: 1,2,3 (one null). Superfile 2: 10, 20.
        w.append(&num_batch(&[Some(1), Some(2), Some(3), None]))
            .expect("a1");
        w.commit().expect("c1");
        w.append(&num_batch(&[Some(10), Some(20)])).expect("a2");
        w.commit().expect("c2");

        let reader = st.reader();
        let provider = SupertableProvider::new(
            st.options().scalar_schema(),
            reader.manifest().clone(),
            st.options().store.clone(),
            st.options().disk_cache.clone(),
            reader.tombstone_cache.clone(),
        );

        let stats = provider.statistics().expect("statistics");
        // 6 rows total across both superfiles, clean view → Exact.
        assert!(matches!(stats.num_rows, Precision::Exact(6)));

        // Find the `n` column's statistics and assert aggregated min/max.
        let sch = provider.schema();
        let n_idx = sch.index_of("n").expect("n column");
        let cs = &stats.column_statistics[n_idx];
        assert_eq!(cs.min_value, Precision::Exact(ScalarValue::Int64(Some(1))));
        assert_eq!(cs.max_value, Precision::Exact(ScalarValue::Int64(Some(20))));
        // One null planted in superfile 1.
        assert_eq!(cs.null_count, Precision::Exact(1));
    }

    /// A superfile entry carrying only min/max for `col` (no null count,
    /// sum, or HLL) — the shape a non-summable column (e.g. Utf8) produces.
    fn entry_minmax_only(col: &str, min: &str, max: &str) -> Arc<SuperfileEntry> {
        let mn: ArrayRef = Arc::new(LargeStringArray::from(vec![min]));
        let mx: ArrayRef = Arc::new(LargeStringArray::from(vec![max]));
        let mut scalar_stats = HashMap::new();
        scalar_stats.insert(col.to_string(), ScalarStatsAgg::from_min_max(mn, mx));
        Arc::new(SuperfileEntry {
            superfile_id: Uuid::new_v4(),
            uri: SuperfileUri::new_v4(),
            n_docs: 1,
            id_min: 0,
            id_max: 0,
            scalar_stats,
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    /// The statistics fold helpers return `None` when a column carries no
    /// usable stat — both when the column has min/max only (no additive
    /// stat) and when it's absent entirely. Exercises the `get(col)?` and
    /// `.<stat>.as_ref()?` short-circuit branches of the new map-based access.
    #[test]
    fn scalar_statistics_helpers_return_none_when_stat_absent() {
        let entries = vec![entry_minmax_only("s", "alpha", "omega")];
        // Column present, but the additive stats are absent → None.
        assert!(scalar_sum(&entries, "s").is_none(), "no sum stat → None");
        assert!(
            scalar_distinct(&entries, "s").is_none(),
            "no hll stat → None"
        );
        assert!(
            scalar_null_count(&entries, "s").is_none(),
            "no null_count stat → None"
        );
        // min/max IS present for the column.
        assert!(scalar_min_max(&entries, "s").is_some());
        // A column absent from every entry yields None for all helpers.
        assert!(scalar_sum(&entries, "missing").is_none());
        assert!(scalar_min_max(&entries, "missing").is_none());
        assert!(scalar_null_count(&entries, "missing").is_none());
    }

    /// `CachedMetadataReaderFactory`'s `Debug` reports the superfile
    /// (cached-meta) count and is otherwise unreachable from a normal
    /// scan. Build one with an empty meta map over an in-memory store
    /// and render it.
    #[test]
    fn cached_metadata_reader_factory_debug_reports_superfile_count() {
        let store: Arc<dyn OsObjectStore> = Arc::new(InMemory::new());
        let factory = CachedMetadataReaderFactory {
            store,
            metas: HashMap::new(),
        };
        let dbg = format!("{factory:?}");
        assert!(
            dbg.contains("CachedMetadataReaderFactory") && dbg.contains("superfiles: 0"),
            "Debug missing fields: {dbg}"
        );
    }
}
