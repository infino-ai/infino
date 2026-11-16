// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`Supertable::head`]: the first `k` live rows of the table in storage
//! order, without SQL; and [`Supertable::head_where`], the first `k` a
//! filter keeps.
//!
//! A page of rows in no particular order, `SELECT … LIMIT k` with no
//! `ORDER BY`, plans a statement whose planning costs more than reading the
//! rows. The rows are the superfiles' own, in the order the manifest lists
//! the superfiles and each superfile holds its rows, less those deleted:
//! only the superfiles that can hold the page are read, and only the `k`
//! rows kept are resolved.
//!
//! Under a `WHERE`, the filter is checked a row group at a time over the
//! columns it names, taken from the decoded columns kept across statements
//! ([`column_cache`](super::column_cache)) or decoded and kept there, until
//! `k` rows pass. A superfile is passed over unread when its exact value
//! table shows that no value it holds passes one of the filter's conditions
//! on a single column, so a filter on values no superfile holds reads no
//! row.

use std::{sync::Arc, time::Instant};

use arrow::{
    array::{Array, AsArray, new_null_array},
    compute::{cast, concat},
    datatypes::SchemaRef,
    record_batch::RecordBatch,
};
use arrow_array::{ArrayRef, RecordBatchOptions};
use datafusion::{
    common::{DFSchema, ScalarValue, tree_node::TreeNode},
    execution::context::ExecutionProps,
    logical_expr::{Expr, simplify::SimplifyContext, utils::split_conjunction},
    optimizer::{
        analyzer::type_coercion::TypeCoercionRewriter, simplify_expressions::ExprSimplifier,
    },
    physical_expr::{PhysicalExpr, create_physical_expr},
};
use roaring::RoaringBitmap;
use tracing::{Instrument, debug};

use crate::{
    InfinoError,
    runtime_bridge::run_on_pool,
    runtime_metrics::op_stats::{OpStatsCollector, timed_section},
    superfile::SuperfileReader,
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::{ManifestSnapshot, SuperfileEntry, SuperfileUri, load_section_groups},
        query::{
            SuperfileHit,
            column_cache::{ColumnKey, DecodedColumnCache, process_decoded_column_cache},
            dispatch::{self, tag_hits, tombstone_deny_set},
            exec::common::{read_file_row_group, resolve_hits_named},
            skip::{ColumnTypeGuard, file_lacks_column},
        },
        reader_cache::disk::ForegroundQueryGuard,
        schema::FieldId,
        tombstones::SidecarCache,
    },
    utils::trace::detail_span,
};

/// The most row groups a [`Supertable::head_where`] walk reads before it
/// leaves the page to the caller's statement, which scans the rest in
/// parallel: a filter that keeps so few rows is found sooner that way.
const MAX_WALKED_ROW_GROUPS: usize = 64;

/// Rows of a row group a walk checks at a time, so the check stops near
/// the row that fills the page rather than at the row group's end.
const CHECK_ROWS: usize = 4096;

/// The most superfiles a walk reads at once. It reads the first alone,
/// which holds the page when the filter keeps a fair share of its rows,
/// and twice as many each round after, up to this.
const MAX_SUPERFILES_AT_ONCE: usize = 4;

/// What [`Supertable::head_where`] reads a filter as.
enum HeadFilter {
    /// It keeps every row: the page is [`Supertable::head`]'s.
    Every,
    /// It keeps no row.
    Nothing,
    /// It is checked on the rows.
    Check(CheckedFilter),
}

/// A filter checked on the rows of the columns it names.
struct CheckedFilter {
    /// Those columns, by name, as the scan types them.
    columns: SchemaRef,
    /// The filter over `columns`.
    predicate: Arc<dyn PhysicalExpr>,
    /// Its conditions that name one column each, for the value tables.
    conditions: Vec<OneColumnCondition>,
}

/// One condition of a filter over one column, checked against a
/// superfile's exact value table of the column.
struct OneColumnCondition {
    field: FieldId,
    /// The column alone, as the scan types it.
    column: SchemaRef,
    predicate: Arc<dyn PhysicalExpr>,
}

impl OneColumnCondition {
    /// Whether the condition keeps none of `values`, every value the
    /// column holds in a superfile (a null among them where it holds one).
    /// `false` when the values do not take the scan's type or the condition
    /// does not evaluate over them.
    fn keeps_none_of(&self, values: ArrayRef) -> bool {
        let field = self.column.field(0);
        let values = match values.data_type() == field.data_type() {
            true => values,
            false => match cast(&values, field.data_type()) {
                Ok(values) => values,
                Err(_) => return false,
            },
        };
        let rows = values.len();
        let Ok(batch) = RecordBatch::try_new(Arc::clone(&self.column), vec![values]) else {
            return false;
        };
        let Ok(kept) = self
            .predicate
            .evaluate(&batch)
            .and_then(|v| v.into_array(rows))
        else {
            return false;
        };
        let Some(kept) = kept.as_boolean_opt() else {
            return false;
        };
        (0..kept.len()).all(|i| !(kept.is_valid(i) && kept.value(i)))
    }
}

/// What a walk of one superfile found.
struct Walked {
    hits: Vec<SuperfileHit>,
    /// Row groups it read.
    read: usize,
    /// Whether it read on to the page or the superfile's end, not stopping
    /// at its row-group budget first.
    complete: bool,
}

/// What every superfile's walk reads with.
struct WalkContext {
    manifest: Arc<ManifestSnapshot>,
    op_stats: Option<Arc<OpStatsCollector>>,
    cache: Option<Arc<DecodedColumnCache>>,
    schema_id: u32,
    columns: SchemaRef,
    predicate: Arc<dyn PhysicalExpr>,
}

impl WalkContext {
    /// The cache's key for `column` of `row_group` of superfile `uri`.
    fn key(&self, uri: SuperfileUri, row_group: usize, column: &str) -> ColumnKey {
        ColumnKey {
            uri,
            schema_id: self.schema_id,
            column: column.to_owned(),
            row_group,
        }
    }

    /// The filter's columns of `row_group` of the superfile, `rows` rows,
    /// each taken from the cache, or decoded and kept there.
    async fn row_group(
        &self,
        reader: &Arc<SuperfileReader>,
        uri: SuperfileUri,
        row_group: usize,
        rows: usize,
    ) -> Result<RecordBatch, QueryError> {
        let fields = self.columns.fields();
        let mut arrays: Vec<Option<ArrayRef>> = Vec::with_capacity(fields.len());
        for field in fields.iter() {
            let kept = self
                .cache
                .as_ref()
                .and_then(|cache| cache.get(&self.key(uri, row_group, field.name())));
            arrays.push(match kept {
                Some(kept) if kept.data_type() != field.data_type() => {
                    Some(cast(&kept, field.data_type()).map_err(QueryError::internal)?)
                }
                kept => kept,
            });
        }
        let missing: Vec<usize> = (0..arrays.len()).filter(|&i| arrays[i].is_none()).collect();
        if !missing.is_empty() {
            let names: Vec<&str> = missing.iter().map(|&i| fields[i].name().as_str()).collect();
            let typed = Arc::new(
                self.columns
                    .project(&missing)
                    .map_err(QueryError::internal)?,
            );
            let read = read_file_row_group(
                &self.manifest,
                reader,
                row_group,
                &names,
                &typed,
                &self.op_stats,
            )
            .await
            .map_err(QueryError::DataFusion)?;
            for (&i, array) in missing.iter().zip(read.columns()) {
                let array = match array.data_type() == fields[i].data_type() {
                    true => Arc::clone(array),
                    false => cast(array, fields[i].data_type()).map_err(QueryError::internal)?,
                };
                if let Some(cache) = &self.cache {
                    cache.insert(
                        self.key(uri, row_group, fields[i].name()),
                        Arc::clone(&array),
                        array.get_array_memory_size(),
                    );
                }
                arrays[i] = Some(array);
            }
        }
        let arrays = arrays
            .into_iter()
            .map(|a| a.ok_or_else(|| QueryError::internal("a column was not read")))
            .collect::<Result<Vec<_>, _>>()?;
        RecordBatch::try_new_with_options(
            Arc::clone(&self.columns),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map_err(QueryError::internal)
    }

    /// The rows of `batch` (a row group whose first row is `first` in the
    /// superfile) the filter keeps, less the `deleted`, the first `want`
    /// of them, by position in the superfile. `None` when the filter does
    /// not evaluate to booleans over them.
    async fn kept(
        &self,
        batch: RecordBatch,
        first: u32,
        deleted: Option<Arc<RoaringBitmap>>,
        want: usize,
    ) -> Result<Option<Vec<u32>>, QueryError> {
        let predicate = Arc::clone(&self.predicate);
        // The check is compute: the reader pool runs it, and its time is the
        // query's, as a statement's predicate is.
        let (kept, ns) = run_on_pool(
            Some(&self.manifest.options.reader_pool),
            "head condition",
            move || {
                timed_section(|| {
                    let mut found = Vec::new();
                    let rows = batch.num_rows();
                    let mut at = 0;
                    while at < rows && found.len() < want {
                        let n = CHECK_ROWS.min(rows - at);
                        let kept = predicate
                            .evaluate(&batch.slice(at, n))
                            .and_then(|v| v.into_array(n))
                            .ok()?;
                        let kept = kept.as_boolean_opt()?;
                        // A condition a row's nulls leave unknown does not
                        // keep it, as in a statement's WHERE.
                        let passed = match kept.nulls() {
                            Some(valid) => kept.values() & valid.inner(),
                            None => kept.values().clone(),
                        };
                        for i in passed.set_indices() {
                            let row = first + (at + i) as u32;
                            if deleted.as_ref().is_none_or(|d| !d.contains(row)) {
                                found.push(row);
                                if found.len() == want {
                                    break;
                                }
                            }
                        }
                        at += n;
                    }
                    Some(found)
                })
            },
        )
        .await
        .map_err(QueryError::internal)?;
        if let Some(stats) = &self.op_stats {
            stats.add_kernel_cpu_ns(ns);
        }
        Ok(kept)
    }

    /// The first `want` rows of superfile `entry` the filter keeps, in its
    /// order, reading at most `budget` row groups. `None` when the filter
    /// does not evaluate over its rows.
    async fn walk(
        &self,
        reader: Arc<SuperfileReader>,
        entry: &SuperfileEntry,
        deleted: Option<Arc<RoaringBitmap>>,
        want: usize,
        budget: usize,
    ) -> Result<Option<Walked>, QueryError> {
        let footer = reader.parquet_metadata();
        let mut rows: Vec<u32> = Vec::new();
        let mut first = 0u32;
        let mut read = 0;
        let mut complete = true;
        for (row_group, meta) in footer.row_groups().iter().enumerate() {
            if rows.len() >= want {
                break;
            }
            let n = u32::try_from(meta.num_rows()).unwrap_or(u32::MAX);
            let group = first..first.saturating_add(n);
            first = group.end;
            let gone = deleted
                .as_ref()
                .map_or(0, |d| d.range_cardinality(group.clone()));
            if n == 0 || gone >= u64::from(n) {
                continue;
            }
            if read == budget {
                complete = false;
                break;
            }
            read += 1;
            let batch = self
                .row_group(&reader, entry.uri, row_group, n as usize)
                .await?;
            let Some(kept) = self
                .kept(batch, group.start, deleted.clone(), want - rows.len())
                .await?
            else {
                return Ok(None);
            };
            rows.extend(kept);
        }
        Ok(Some(Walked {
            hits: tag_hits(entry, rows.into_iter().map(|row| (row, 0.0)).collect()),
            read,
            complete,
        }))
    }
}

impl SupertableReader {
    /// The first `k` live rows in storage order, as hits: the superfiles in
    /// the manifest's order, each one's rows in its own order, deleted rows
    /// skipped. Superfiles are read a group at a time, each group holding
    /// enough rows for the rest of the page were none of them deleted, so a
    /// page of a large table reads the first superfile or two.
    pub(crate) async fn head_hits_async(&self, k: usize) -> Result<Vec<SuperfileHit>, QueryError> {
        let mut hits: Vec<SuperfileHit> = Vec::with_capacity(k);
        let manifest = self.manifest();
        let mut entries = manifest.superfiles.iter().cloned();
        while hits.len() < k {
            let want = k - hits.len();
            let mut units: Vec<(Arc<SuperfileEntry>, usize)> = Vec::new();
            let mut held = 0u64;
            for entry in entries.by_ref() {
                held = held.saturating_add(entry.n_docs);
                units.push((entry, want));
                if held >= want as u64 {
                    break;
                }
            }
            if units.is_empty() {
                break;
            }
            let body = |_: Arc<SuperfileReader>,
                        entry: Arc<SuperfileEntry>,
                        tombstone_cache: Option<Arc<SidecarCache>>,
                        now: Instant,
                        want: usize| async move {
                let deleted = match &tombstone_cache {
                    Some(cache) => tombstone_deny_set(cache, entry.superfile_id, now)?,
                    None => None,
                };
                let rows = u32::try_from(entry.n_docs).unwrap_or(u32::MAX);
                let live: Vec<(u32, f32)> = (0..rows)
                    .filter(|row| deleted.as_ref().is_none_or(|d| !d.contains(*row)))
                    .take(want)
                    .map(|row| (row, 0.0))
                    .collect();
                Ok::<Vec<SuperfileHit>, QueryError>(tag_hits(&entry, live))
            };
            let per_superfile = dispatch::fanout_with(self, units, true, self.open_intent(), body)
                .instrument(detail_span!("head.fanout"))
                .await?;
            hits.extend(per_superfile.into_iter().flatten().take(want));
        }
        dispatch::attach_stable_ids_to_hits(self, &mut hits).await?;
        Ok(hits)
    }

    /// How [`Supertable::head_where`] reads `filter`: simplified over the
    /// scan's columns, then checked over the columns it names, typed as the
    /// scan types them. `None` when it names a column the table does not
    /// store, or does not evaluate outside a statement (a subquery, a
    /// function only a statement registers).
    fn head_filter(&self, filter: &Expr) -> Result<Option<HeadFilter>, QueryError> {
        let scan = Arc::clone(self.sql_schemas().scan());
        let simplifier = ExprSimplifier::new(
            SimplifyContext::builder()
                .with_schema(Arc::new(
                    DFSchema::try_from(scan.as_ref().clone()).map_err(QueryError::DataFusion)?,
                ))
                .build(),
        );
        let Ok(filter) = simplifier.simplify(filter.clone()) else {
            return Ok(None);
        };
        let mut names: Vec<String> = filter
            .column_refs()
            .into_iter()
            .map(|c| c.name.clone())
            .collect();
        names.sort();
        names.dedup();
        if names.is_empty() {
            return Ok(match filter {
                Expr::Literal(ScalarValue::Boolean(Some(true)), _) => Some(HeadFilter::Every),
                Expr::Literal(ScalarValue::Boolean(_), _) => Some(HeadFilter::Nothing),
                _ => None,
            });
        }
        let stored = self.manifest().stored_schema();
        let mut at = Vec::with_capacity(names.len());
        for name in &names {
            match (stored.field_with_name(name), scan.index_of(name)) {
                (Ok(_), Ok(i)) => at.push(i),
                _ => return Ok(None),
            }
        }
        let columns = Arc::new(scan.project(&at).map_err(QueryError::internal)?);
        let schema =
            DFSchema::try_from(columns.as_ref().clone()).map_err(QueryError::DataFusion)?;
        let Ok(filter) = filter
            .rewrite(&mut TypeCoercionRewriter::new(&schema))
            .map(|t| t.data)
        else {
            return Ok(None);
        };
        let props = ExecutionProps::new();
        let Ok(predicate) = create_physical_expr(&filter, &schema, &props) else {
            return Ok(None);
        };
        let table = self.manifest().table_schema();
        let mut conditions = Vec::new();
        for condition in split_conjunction(&filter) {
            let refs = condition.column_refs();
            let mut refs = refs.iter();
            let (Some(only), None) = (refs.next(), refs.next()) else {
                continue;
            };
            let (Some(field), Ok(i)) = (table.id_of(&only.name), columns.index_of(&only.name))
            else {
                continue;
            };
            let column = Arc::new(columns.project(&[i]).map_err(QueryError::internal)?);
            let Ok(alone) = DFSchema::try_from(column.as_ref().clone()) else {
                continue;
            };
            if let Ok(predicate) = create_physical_expr(condition, &alone, &props) {
                conditions.push(OneColumnCondition {
                    field,
                    column,
                    predicate,
                });
            }
        }
        Ok(Some(HeadFilter::Check(CheckedFilter {
            columns,
            predicate,
            conditions,
        })))
    }

    /// Whether the table's statistics prove that `predicate` keeps no row:
    /// it is false or null whatever the row, or every superfile holds none
    /// of the values a condition of it on one column needs
    /// ([`Self::holds_none`]). `false` when they prove nothing.
    pub(crate) fn keeps_no_row(&self, predicate: &Expr) -> Result<bool, QueryError> {
        Ok(match self.head_filter(predicate)? {
            Some(HeadFilter::Nothing) => true,
            Some(HeadFilter::Check(filter)) if !filter.conditions.is_empty() => {
                let manifest = self.manifest();
                let guards: Vec<ColumnTypeGuard> = filter
                    .conditions
                    .iter()
                    .map(|c| ColumnTypeGuard::new(manifest, c.field))
                    .collect();
                manifest
                    .superfiles
                    .iter()
                    .all(|entry| Self::holds_none(entry, &filter, &guards))
            }
            _ => false,
        })
    }

    /// Whether superfile `entry` holds no row `filter` keeps, by its exact
    /// value tables: a condition on one column keeps none of the values
    /// the column holds there. A column the superfile lacks, or holds only
    /// nulls in, holds one value, null. A superfile without a table for the
    /// column, or whose statistics are in a type the column has since left
    /// (`guards`, one per condition), may hold any row.
    fn holds_none(
        entry: &SuperfileEntry,
        filter: &CheckedFilter,
        guards: &[ColumnTypeGuard],
    ) -> bool {
        filter
            .conditions
            .iter()
            .zip(guards)
            .any(|(condition, guard)| {
                let data_type = condition.column.field(0).data_type();
                let values = if file_lacks_column(entry, condition.field) {
                    new_null_array(data_type, 1)
                } else {
                    if guard.stats_are_stale(entry) {
                        return false;
                    }
                    let Some(stats) = entry.scalar_stats.column(condition.field) else {
                        return false;
                    };
                    if stats.all_null() {
                        new_null_array(data_type, 1)
                    } else {
                        let Some(values) = stats.exact_values() else {
                            return false;
                        };
                        match stats.null_count() {
                            Some(0) => values,
                            _ => {
                                let null = new_null_array(values.data_type(), 1);
                                match concat(&[values.as_ref(), null.as_ref()]) {
                                    Ok(values) => values,
                                    Err(_) => return false,
                                }
                            }
                        }
                    }
                };
                condition.keeps_none_of(values)
            })
    }

    /// The first `k` live rows `filter` keeps in storage order, as hits:
    /// the superfiles in the manifest's order, less those
    /// [`Self::holds_none`] passes over, each one's row groups in order,
    /// checked a row group at a time. The first superfile is read alone,
    /// and more at once each round after while the page is short. `None`
    /// when the filter does not evaluate over the rows, or when about
    /// [`MAX_WALKED_ROW_GROUPS`] row groups do not fill the page.
    async fn head_where_hits_async(
        &self,
        filter: &CheckedFilter,
        k: usize,
    ) -> Result<Option<Vec<SuperfileHit>>, QueryError> {
        let manifest = self.manifest();
        let mut fields: Vec<FieldId> = filter.conditions.iter().map(|c| c.field).collect();
        fields.sort_unstable();
        fields.dedup();
        let entries: Vec<Arc<SuperfileEntry>> = manifest.superfiles.to_vec();
        load_section_groups(&entries, &fields).await;
        let guards: Vec<ColumnTypeGuard> = filter
            .conditions
            .iter()
            .map(|c| ColumnTypeGuard::new(manifest, c.field))
            .collect();
        let mut candidates = entries
            .into_iter()
            .filter(|entry| !Self::holds_none(entry, filter, &guards));
        let context = Arc::new(WalkContext {
            manifest: Arc::clone(manifest),
            op_stats: self.op_stats.clone(),
            cache: process_decoded_column_cache(),
            schema_id: manifest.table_schema().schema_id(),
            columns: Arc::clone(&filter.columns),
            predicate: Arc::clone(&filter.predicate),
        });
        let mut hits: Vec<SuperfileHit> = Vec::new();
        let mut budget = MAX_WALKED_ROW_GROUPS;
        let mut at_once = 1;
        while hits.len() < k {
            let want = k - hits.len();
            let units: Vec<(Arc<SuperfileEntry>, (usize, usize))> = candidates
                .by_ref()
                .take(at_once)
                .map(|entry| (entry, (want, budget)))
                .collect();
            if units.is_empty() {
                break;
            }
            let context = Arc::clone(&context);
            let body = move |reader: Arc<SuperfileReader>,
                             entry: Arc<SuperfileEntry>,
                             tombstone_cache: Option<Arc<SidecarCache>>,
                             now: Instant,
                             (want, budget): (usize, usize)| {
                let context = Arc::clone(&context);
                async move {
                    let deleted = match &tombstone_cache {
                        Some(cache) => tombstone_deny_set(cache, entry.superfile_id, now)?,
                        None => None,
                    };
                    context.walk(reader, &entry, deleted, want, budget).await
                }
            };
            let walked = dispatch::fanout_with(self, units, true, self.open_intent(), body)
                .instrument(detail_span!("head_where.fanout"))
                .await?;
            for walked in walked {
                let Some(walked) = walked else {
                    return Ok(None);
                };
                budget = budget.saturating_sub(walked.read);
                hits.extend(walked.hits.into_iter().take(k - hits.len()));
                if hits.len() == k {
                    break;
                }
                if !walked.complete {
                    return Ok(None);
                }
            }
            at_once = (at_once * 2).min(MAX_SUPERFILES_AT_ONCE);
        }
        dispatch::attach_stable_ids_to_hits(self, &mut hits).await?;
        Ok(Some(hits))
    }
}

impl Supertable {
    /// The first `k` live rows of the table in storage order, without
    /// planning a statement: rows `SELECT … LIMIT k` with no `WHERE` and no
    /// `ORDER BY` could return, in the order the superfiles hold them, with
    /// deleted rows left out. Fewer come back when the table holds fewer.
    /// Returns Arrow rows like [`Supertable::bm25_search`], with `score`
    /// fixed at `0.0`; `projection` follows the same rules.
    pub fn head(
        &self,
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(k, "head");
        let close_out = self.close_out();
        let reader = self.reader()?;
        let context = |e: QueryError| InfinoError::from(e).with_context("head", None);
        reader.check_projection(projection).map_err(context)?;
        let hits = {
            let _foreground = ForegroundQueryGuard::enter();
            reader
                .block_on(reader.head_hits_async(k))
                .map_err(context)?
        };
        let batches = vec![
            self.block_on_query(
                resolve_hits_named(&reader, &hits, projection)
                    .instrument(detail_span!("search.resolve", hits = hits.len())),
            )
            .map_err(context)?,
        ];
        close_out.finish_batches(&batches);
        Ok(batches)
    }

    /// The first `k` live rows `filter` keeps, in storage order, without
    /// planning a statement: rows `SELECT … WHERE filter LIMIT k` with no
    /// `ORDER BY` could return, in the order [`Supertable::head`] reads
    /// them. Fewer come back when fewer pass. The filter is checked a row
    /// group at a time over the columns it names, kept decoded across calls
    /// and statements; a superfile whose exact value table shows that no
    /// value it holds passes a condition on one column is not read. `None`,
    /// for the caller to plan the statement, when the filter names a column
    /// the table does not store, needs a statement to evaluate (a subquery,
    /// say), or keeps so few rows that about [`MAX_WALKED_ROW_GROUPS`] row
    /// groups do not fill the page. Returns Arrow rows like
    /// [`Supertable::bm25_search`], with `score` fixed at `0.0`;
    /// `projection` follows the same rules.
    pub fn head_where(
        &self,
        filter: &Expr,
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Option<Vec<RecordBatch>>, InfinoError> {
        debug!(k, "head_where");
        let close_out = self.close_out();
        let reader = self.reader()?;
        let context = |e: QueryError| InfinoError::from(e).with_context("head_where", None);
        reader.check_projection(projection).map_err(context)?;
        let hits = match reader.head_filter(filter).map_err(context)? {
            None => return Ok(None),
            Some(HeadFilter::Every) => return self.head(k, projection).map(Some),
            Some(HeadFilter::Nothing) => Vec::new(),
            Some(HeadFilter::Check(filter)) => {
                let _foreground = ForegroundQueryGuard::enter();
                match reader
                    .block_on(reader.head_where_hits_async(&filter, k))
                    .map_err(context)?
                {
                    Some(hits) => hits,
                    None => return Ok(None),
                }
            }
        };
        let batches = vec![
            self.block_on_query(
                resolve_hits_named(&reader, &hits, projection)
                    .instrument(detail_span!("search.resolve", hits = hits.len())),
            )
            .map_err(context)?,
        ];
        close_out.finish_batches(&batches);
        Ok(Some(batches))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, sync::Arc};

    use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{Expr, col, lit};
    use tempfile::TempDir;

    use super::HeadFilter;
    use crate::{
        config::OptimizeOptions,
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            Supertable, SupertableOptions, handle::SupertableReader, query::skip::ColumnTypeGuard,
        },
    };

    /// Superfiles the table is written as, one commit each.
    const FILES: i64 = 3;
    /// Rows in each.
    const ROWS_PER_FILE: i64 = 1000;
    /// The rows the first delete removes: every third one.
    const DELETED_EVERY: i64 = 3;
    /// Rows the second delete removes: the whole first superfile from this
    /// `seq` on, so the first superfile alone cannot fill a larger page.
    const FIRST_FILE_DELETED_FROM: i64 = 100;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("seq", DataType::Int64, false)]))
    }

    fn table(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage);
        let st = Supertable::create(options).expect("create");
        for file in 0..FILES {
            let seqs: Vec<i64> = (file * ROWS_PER_FILE..(file + 1) * ROWS_PER_FILE).collect();
            let batch = RecordBatch::try_new(schema(), vec![Arc::new(Int64Array::from(seqs))])
                .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete((col("seq") % lit(DELETED_EVERY)).eq(lit(0_i64)))
            .expect("delete");
        st.delete(
            col("seq")
                .gt_eq(lit(FIRST_FILE_DELETED_FROM))
                .and(col("seq").lt(lit(ROWS_PER_FILE))),
        )
        .expect("delete");
        st
    }

    fn seqs(batches: &[RecordBatch]) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| {
                let c = b
                    .column_by_name("seq")
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .expect("seq")
                    .clone();
                (0..c.len()).map(move |i| c.value(i))
            })
            .collect()
    }

    /// The live rows, as the table's own statement reads them.
    fn live(st: &Supertable) -> BTreeSet<i64> {
        let reader = st.reader().expect("reader");
        seqs(&reader.query_sql("SELECT seq FROM supertable").expect("sql"))
            .into_iter()
            .collect()
    }

    /// Every page is distinct live rows, as many as asked or as the table
    /// holds, before and after an optimize: deleted rows never come back, a
    /// page past the first superfile's live rows reads on into the next,
    /// and a page past every row is every live row.
    #[test]
    fn head_returns_distinct_live_rows_up_to_k() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        for pass in ["appended", "optimized"] {
            if pass == "optimized" {
                st.optimize(&OptimizeOptions::default()).expect("optimize");
            }
            let live = live(&st);
            for k in [0, 1, 10, 66, 67, 500, live.len(), live.len() + 10] {
                let got = seqs(&st.head(k, Some(&["seq"])).expect("head"));
                let what = format!("{pass}, k {k}");
                assert_eq!(
                    got.len(),
                    k.min(live.len()),
                    "{what}: as many as asked or held"
                );
                let distinct: BTreeSet<i64> = got.iter().copied().collect();
                assert_eq!(distinct.len(), got.len(), "{what}: no row twice");
                assert!(distinct.is_subset(&live), "{what}: only live rows");
                if k >= live.len() {
                    assert_eq!(distinct, live, "{what}: every live row");
                }
            }
        }
    }

    /// With no projection the rows carry `_id` and a zero `score`, as the
    /// other search calls return them; an unknown column is refused.
    #[test]
    fn a_bare_head_returns_ids_and_a_zero_score() {
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir);
        let batches = st.head(5, None).expect("head");
        let names: Vec<String> = batches[0]
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["_id", "score"]);
        let score = batches[0]
            .column_by_name("score")
            .expect("score")
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
            .expect("f32");
        assert!(score.values().iter().all(|s| *s == 0.0));
        assert!(st.head(5, Some(&["nope"])).is_err());
    }

    /// Rows in each superfile of the tagged table.
    const TAGGED_ROWS: i64 = 1000;
    /// In the first superfile, every this-many rows has no `tag`.
    const NULL_TAG_EVERY: i64 = 50;

    fn tagged_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("tag", DataType::Utf8, true),
        ]))
    }

    /// `tag` of row `seq`: `even` and `odd` in the first superfile, with a
    /// null now and then; `odd` throughout the second; `late` in the third.
    fn tag(seq: i64) -> Option<&'static str> {
        match seq / TAGGED_ROWS {
            0 if seq % NULL_TAG_EVERY == 0 => None,
            0 if seq % 2 == 0 => Some("even"),
            0 | 1 => Some("odd"),
            _ => Some("late"),
        }
    }

    /// Three superfiles of [`TAGGED_ROWS`] rows, one commit each, every
    /// third row deleted: each superfile holds its own few `tag` values, so
    /// each keeps an exact value table of them.
    fn tagged(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(tagged_schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage);
        let st = Supertable::create(options).expect("create");
        for file in 0..3 {
            let seqs: Vec<i64> = (file * TAGGED_ROWS..(file + 1) * TAGGED_ROWS).collect();
            let tags: Vec<Option<&str>> = seqs.iter().map(|s| tag(*s)).collect();
            let batch = RecordBatch::try_new(
                tagged_schema(),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(StringArray::from(tags)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete((col("seq") % lit(DELETED_EVERY)).eq(lit(0_i64)))
            .expect("delete");
        st
    }

    /// The filters the tests ask for, each with what it keeps of a row.
    #[allow(clippy::type_complexity)]
    fn filters() -> Vec<(Expr, Box<dyn Fn(i64) -> bool>)> {
        vec![
            (
                (col("seq") % lit(7_i64)).eq(lit(0_i64)),
                Box::new(|s| s % 7 == 0),
            ),
            (col("seq").gt_eq(lit(2500_i64)), Box::new(|s| s >= 2500)),
            (
                col("tag").eq(lit("late")),
                Box::new(|s| tag(s) == Some("late")),
            ),
            (
                col("tag")
                    .eq(lit("odd"))
                    .and((col("seq") % lit(5_i64)).eq(lit(1_i64))),
                Box::new(|s| tag(s) == Some("odd") && s % 5 == 1),
            ),
            (col("tag").is_null(), Box::new(|s| tag(s).is_none())),
            (col("tag").eq(lit("absent")), Box::new(|_| false)),
            (lit(true), Box::new(|_| true)),
            (lit(false), Box::new(|_| false)),
        ]
    }

    /// Every page is the rows [`Supertable::head`] reads, in its order,
    /// those the filter keeps, the first `k` of them, before and after an
    /// optimize: deleted rows never come back, a page reads on past a
    /// superfile that runs out, and a page past every row that passes is
    /// every such row.
    #[test]
    fn head_where_is_heads_order_less_the_rows_the_filter_drops() {
        let dir = TempDir::new().expect("tempdir");
        let st = tagged(&dir);
        for pass in ["appended", "optimized"] {
            if pass == "optimized" {
                st.optimize(&OptimizeOptions::default()).expect("optimize");
            }
            let every = seqs(
                &st.head(10 * TAGGED_ROWS as usize, Some(&["seq"]))
                    .expect("head"),
            );
            for (filter, keeps) in filters() {
                let passing: Vec<i64> = every.iter().copied().filter(|s| keeps(*s)).collect();
                for k in [0, 1, 10, 150, passing.len(), passing.len() + 10] {
                    let what = format!("{pass}, {filter}, k {k}");
                    let got = st
                        .head_where(&filter, k, Some(&["seq"]))
                        .expect("head_where")
                        .unwrap_or_else(|| panic!("{what}: answered"));
                    let want: Vec<i64> = passing.iter().copied().take(k).collect();
                    assert_eq!(seqs(&got), want, "{what}");
                }
            }
        }
    }

    /// A superfile is passed over when a condition on one column keeps
    /// none of the values its exact value table holds, the null it holds
    /// counted as one of them; a superfile holding a passing value, or a
    /// condition over two columns, is read.
    #[test]
    fn a_superfile_none_of_whose_values_pass_is_not_read() {
        let dir = TempDir::new().expect("tempdir");
        let st = tagged(&dir);
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let passed_over = |filter: &Expr| -> Vec<bool> {
            let Some(HeadFilter::Check(filter)) = reader.head_filter(filter).expect("filter")
            else {
                panic!("{filter}: checked on the rows");
            };
            let guards: Vec<_> = filter
                .conditions
                .iter()
                .map(|c| ColumnTypeGuard::new(manifest, c.field))
                .collect();
            manifest
                .superfiles
                .iter()
                .map(|entry| SupertableReader::holds_none(entry, &filter, &guards))
                .collect()
        };
        assert_eq!(
            passed_over(&col("tag").eq(lit("late"))),
            [true, true, false]
        );
        assert_eq!(passed_over(&col("tag").eq(lit("absent"))), [true; 3]);
        assert_eq!(
            passed_over(&col("tag").in_list(vec![lit("even"), lit("late")], false)),
            [false, true, false]
        );
        assert_eq!(passed_over(&col("tag").is_null()), [false, true, true]);
        assert_eq!(
            passed_over(&col("tag").eq(lit("odd")).and(col("seq").gt(lit(5_i64)))),
            [false, false, true]
        );
        assert_eq!(
            passed_over(&col("tag").is_null().or(col("seq").gt(lit(0_i64)))),
            [false; 3]
        );
    }

    /// A filter the call cannot check alone is left to the caller's
    /// statement: a column the table does not have.
    #[test]
    fn a_filter_naming_an_unknown_column_is_left_to_the_statement() {
        let dir = TempDir::new().expect("tempdir");
        let st = tagged(&dir);
        assert!(
            st.head_where(&col("nope").eq(lit(1_i64)), 10, Some(&["seq"]))
                .expect("head_where")
                .is_none()
        );
    }

    /// The statistics prove a predicate keeps no row when no superfile holds
    /// a value one of its single-column conditions needs, and `count_where`
    /// counts such a predicate 0, as its statement does.
    #[test]
    fn a_predicate_no_superfile_holds_a_value_for_keeps_no_row() {
        let dir = TempDir::new().expect("tempdir");
        let st = tagged(&dir);
        let reader = st.reader().expect("reader");
        for (predicate, none) in [
            (col("tag").eq(lit("absent")), true),
            (col("tag").eq(lit("late")), false),
            (col("tag").is_not_null(), false),
            (lit(false), true),
            (
                col("tag").eq(lit("absent")).or(col("seq").eq(lit(1_i64))),
                false,
            ),
        ] {
            assert_eq!(
                reader.keeps_no_row(&predicate).expect("keeps_no_row"),
                none,
                "{predicate}"
            );
        }
        assert_eq!(
            st.count_where(&[col("tag").eq(lit("absent")), lit(false)])
                .expect("count_where"),
            [0, 0]
        );
        let late = (2 * TAGGED_ROWS..3 * TAGGED_ROWS)
            .filter(|s| s % DELETED_EVERY != 0)
            .count() as u64;
        assert_eq!(
            st.count_where(&[col("tag").eq(lit("late")), col("tag").eq(lit("absent"))])
                .expect("count_where"),
            [late, 0]
        );
    }
}
