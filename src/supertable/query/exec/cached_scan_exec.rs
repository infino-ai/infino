// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`CachedScanExec`]: a table scan served from decoded columns kept across
//! statements ([`column_cache`](crate::supertable::query::column_cache)).
//!
//! The provider's scan otherwise hands its files to DataFusion's Parquet
//! reader, which decompresses and decodes every row group's column chunks on
//! every statement. With a decoded-column cache configured, the provider builds
//! this node instead whenever it can. Each partition walks its share of the
//! files' row groups; for each it takes every projected column from the cache,
//! decodes the ones the cache lacks from the row group whole
//! ([`read_file_row_group`], with the schema adaptation the search paths
//! use), keeps them, and drops the row group's deleted rows. When an index
//! chose a file's rows (an `ORDER BY ... LIMIT` over a range, or a scan the
//! full-text index narrowed in some superfiles only), only those rows come
//! out, and a row group holding none of them is not read.
//!
//! A row group whose chosen rows are fewer than half its live rows, and whose
//! columns the cache lacks, has just those rows read from its pages, as the
//! Parquet scan reads them, and nothing kept: one selective statement does not
//! decode a row group whole or push out the columns other statements read
//! whole. The cache remembers the row group, and the next scan that chooses
//! rows in it decodes and keeps its columns
//! ([`DecodedColumnCache::read_in_part_before`]). A row group whose columns
//! are kept hands over its chosen rows from them.
//!
//! A parent hands down filters that tighten as the statement runs
//! (`DynamicFilterPhysicalExpr`): a top-k its running bound (the k-th best
//! sort value so far), a hash join a filter on its probe side built from the
//! rows of its build side, a `MIN` or `MAX` aggregate the best value so far.
//! Each rules out only rows that can't change its parent's answer, and the
//! parent still applies its own check, so the scan keeps every such filter.
//! Every filter skips a row group its statistics rule out, just before the
//! row group is read. When the scan reads best first for a sort above it —
//! what a top-k asks for — its bound also drops the rows it rules out from
//! each batch the scan hands over; a hash join's or aggregate's filter, which
//! asks for no order, does not, matching the Parquet scan, which filters rows
//! only under `ORDER BY ... LIMIT`. Asked for a sort on one of its columns,
//! the scan reads the row groups whose statistics start best first, so a
//! top-k settles early and the rest go unread.
//!
//! How much that saves depends on how the sort column's values cluster by
//! row group. On a column whose values follow the order rows are written in
//! (as `_id`'s do), few row groups reach a settled bound and the rest are
//! skipped; on one whose values spread across every row group, the bound
//! skips none, and each row group and batch is checked for nothing.
//!
//! A sort on `_id` reads finer than row groups. Ids are minted in order by
//! each writer, so a file's run of [`BLOCK_DOCS`] rows spans a narrow band of
//! ids even where its row groups, mixing writers, span nearly all of them.
//! The stable-id sidecar records each block's bounds in its header
//! ([`SuperfileReader::id_blocks`]); the scan reads the blocks in order
//! of those bounds, skips each block the top-k's bound rules out, and stops
//! once it rules out every block still to come. A page of `ORDER BY _id`
//! under a filter that keeps a sizeable share of the rows reads a block or
//! two instead of every row the filter keeps.
//!
//! It answers what the Parquet scan would: the same columns, typed as the scan
//! schema types them, each file's rows in file order, without the rows a
//! tombstone deletes. A `WHERE` stays above it in a `FilterExec`, as above the
//! Parquet scan, and like the Parquet scan it skips a row group whose
//! statistics show the `WHERE` can't match ([`RowGroupFilter`]): such a row
//! group is neither decoded nor kept, so a first run decodes what the Parquet
//! scan would, and a table larger than the cache doesn't decode the rest of
//! it on every run.
//!
//! The Parquet scan serves the statement instead when it reads less:
//! - the columns this scan would decode don't fit the cache. Kept, they
//!   would push each other out, and every run would decode them all again;
//! - a filtered `ORDER BY ... LIMIT` or plain `LIMIT` whose other columns
//!   are wide. `RowFilterUnderTopK` takes back the Parquet scan this one
//!   stands in for, whose row filter decodes those columns only for the rows
//!   that pass.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    fmt,
    ops::Range,
    option::IntoIter,
    sync::{Arc, Mutex, PoisonError},
};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Decimal128Array, RecordBatch, RecordBatchOptions, UInt32Array,
    UInt64Array, builder::BooleanBufferBuilder, cast::AsArray, new_null_array, types::Int16Type,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
use datafusion::{
    arrow::{
        buffer::BooleanBuffer,
        compute::{
            can_cast_types, cast_with_options, concat, filter, filter_record_batch,
            prep_null_mask_filter, sort_to_indices, take,
        },
    },
    common::{
        Column as StatsColumn, ScalarValue, Statistics, config::ConfigOptions,
        pruning::PruningStatistics,
    },
    datasource::physical_plan::parquet::ParquetAccessPlan,
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    physical_expr::{
        EquivalenceProperties, PhysicalExpr, PhysicalSortExpr,
        expressions::{Column, DynamicFilterPhysicalExpr, Literal},
        utils::{collect_columns, reassign_expr_columns, split_conjunction},
    },
    physical_expr_adapter::PhysicalExprAdapterFactory,
    physical_expr_common::{
        physical_expr::{snapshot_generation, snapshot_physical_expr},
        utils::scatter,
    },
    physical_optimizer::pruning::PruningPredicate,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        filter_pushdown::{
            ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
        },
        metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet},
        sort_pushdown::SortOrderPushdownResult,
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{StreamExt, TryStreamExt, future, stream};
use parquet::{
    arrow::{arrow_reader::statistics::StatisticsConverter, parquet_to_arrow_schema},
    file::metadata::{ParquetMetaData, RowGroupMetaData},
};
use roaring::RoaringBitmap;
use tokio::{
    spawn,
    sync::{Mutex as AsyncMutex, OnceCell},
};
use tracing::Instrument;

use crate::{
    runtime_metrics::op_stats::OpStatsCollector,
    superfile::{SuperfileReader, ids::BLOCK_DOCS, reader::IdBlocks},
    supertable::{
        error::QueryError,
        manifest::{ManifestSnapshot, SuperfileUri, hll::HllSketch},
        query::{
            column_cache::{ColumnKey, DecodedColumnCache, RowGroupKey},
            exec::common::{hit_output_schema, read_file_row_group, take_file_columns},
            read_ahead::{file_pruning, file_schema, pruning_for, statistics_pruned},
            schema_adapter::TableExprAdapterFactory,
        },
        schema::{FieldId, cast::CAST_OPTIONS},
    },
};

/// Bytes a decoded string or binary value takes per row beside its data: a
/// view, as the scan reads non-indexed strings, and the most any of Arrow's
/// variable-width layouts takes.
const VIEW_BYTES_PER_ROW: usize = 16;

/// Bytes per row a string column kept as a dictionary takes: its `Int16` key.
const DICTIONARY_KEY_BYTES_PER_ROW: usize = 2;

/// A superfile's string column is kept as a dictionary when it holds at most
/// one value for this many rows ([`dictionary_columns`]): its keys then take
/// 2 bytes a row where views take 16, and its values a small share of the
/// rows' bytes.
const DICTIONARY_ROWS_PER_VALUE: usize = 16;

/// How far under the most values `Int16` keys reach a distinct-sketch
/// estimate must sit for its column to be kept as a dictionary: the sketch
/// counts within about 3.25% (one standard error), so half the keys' reach
/// leaves a row group whose values outrun its keys no real chance. An exact
/// value table needs no margin.
const SKETCH_KEY_MARGIN: usize = 2;

/// Metric counting the row groups a scan decoded because the cache lacked
/// one of their projected columns, as `EXPLAIN ANALYZE` lists it. A run over
/// columns the cache keeps decodes none.
const ROW_GROUPS_DECODED: &str = "row_groups_decoded";

/// Metric counting the row groups a scan read only the chosen rows of, from
/// their pages, keeping nothing, as `EXPLAIN ANALYZE` lists it.
const ROW_GROUPS_READ_IN_PART: &str = "row_groups_read_in_part";

/// How `EXPLAIN ANALYZE` names the rows of the units a scan read.
const ROWS_READ: &str = "rows_read";

/// How `EXPLAIN ANALYZE` names the row groups the filters that tighten as
/// the statement runs ruled out before they were read.
const ROW_GROUPS_SKIPPED: &str = "row_groups_skipped";

/// How `EXPLAIN ANALYZE` names the rows those filters dropped from the
/// batches the scan read.
const ROWS_DROPPED: &str = "rows_dropped";

/// One file of the scan: what its row groups are read from.
#[derive(Debug)]
pub(crate) struct CachedFile {
    pub(crate) uri: SuperfileUri,
    pub(crate) reader: Arc<SuperfileReader>,
    /// The footer [`RowGroupFilter`] prunes by: the file's own, without the
    /// bounds that leave out a float column's NaNs, as the Parquet scan
    /// prunes by
    /// ([`without_nan_blind_bounds`](crate::supertable::query::nan_bounds::without_nan_blind_bounds)).
    pub(crate) footer: Arc<ParquetMetaData>,
    /// Rows in each row group, in file order.
    pub(crate) row_counts: Arc<[u32]>,
    /// The file's deleted rows, by position in the file.
    pub(crate) tombstones: Arc<RoaringBitmap>,
    /// The only rows to read, by position in the file, when an index chose
    /// them (an `ORDER BY ... LIMIT` over a range reads the rows up to its
    /// cutoff); `None` reads every live row. Live rows only: none of them is
    /// in [`Self::tombstones`].
    pub(crate) selected: Option<RoaringBitmap>,
}

/// The scan's `WHERE`, for skipping the row groups whose statistics show it
/// can't match, as the Parquet scan skips them.
#[derive(Debug)]
pub(crate) struct RowGroupFilter {
    /// The `WHERE`, written against `table`.
    pub(crate) predicate: Arc<dyn PhysicalExpr>,
    pub(crate) table: SchemaRef,
    /// Maps the `WHERE` onto each file's own schema, as the Parquet scan's
    /// adapter does.
    pub(crate) adapter: Arc<dyn PhysicalExprAdapterFactory>,
}

impl RowGroupFilter {
    /// Which of `file`'s row groups the statistics leave to read; `None`
    /// when the `WHERE` doesn't map onto the file, and every row group is
    /// read.
    fn plan(&self, file: &CachedFile) -> Option<ParquetAccessPlan> {
        let pruning = file_pruning(&self.predicate, &self.adapter, &self.table, &file.footer)?;
        let all = ParquetAccessPlan::new_all(file.row_counts.len());
        Some(statistics_pruned(
            all,
            &file.footer,
            &pruning,
            &file.uri.0.to_string(),
        ))
    }
}

/// One row group of one file, or one id block of it: the unit a partition
/// reads.
#[derive(Debug, Clone)]
struct RowGroupUnit {
    file: Arc<CachedFile>,
    row_group: usize,
    /// The rows to read, by position in the file: the row group's, or one
    /// id block's.
    rows: Range<u32>,
    /// Whether a read that lacks a column decodes the row group whole and
    /// keeps its columns: it is read whole, the scan chose at least half its
    /// live rows, or a scan read some of its rows before. Otherwise such a
    /// read takes the chosen rows from the pages and keeps nothing.
    keep: bool,
    /// The row group's rows, by position in the file.
    group: Range<u32>,
    /// The lowest and highest `_id` the rows can hold, for an id block.
    id_bounds: Option<(i128, i128)>,
}

/// What every partition reads with.
#[derive(Debug)]
struct ScanContext {
    /// The scan's output: the projected columns, typed as the scan schema
    /// types them.
    schema: SchemaRef,
    manifest: Arc<ManifestSnapshot>,
    op_stats: Option<Arc<OpStatsCollector>>,
    cache: Arc<DecodedColumnCache>,
    schema_id: u32,
    /// For each file, which of `schema`'s columns are kept as dictionaries
    /// ([`dictionary_columns`]).
    dictionaries: HashMap<SuperfileUri, Arc<[bool]>>,
    /// The `WHERE` the scan checks before it hands rows on
    /// ([`CachedScanExec::with_predicate`]): over the kept dictionaries
    /// themselves, so only the rows that pass are made strings.
    predicate: Option<Arc<dyn PhysicalExpr>>,
    /// For each row group reads are decoding, a lock held while one decodes
    /// over the columns decoded so far, by their place in `schema`: reads of
    /// one row group that miss the cache together decode it once, the rest
    /// taking the first's columns from here. A row group read whole leaves
    /// its entry once no read waits on it. A row group read by id block keeps
    /// its entry until its last block is read or skipped, so its blocks share
    /// one decode even when the cache can't hold its columns; the bytes those
    /// entries hold across the scan are bounded ([`BlockDecoding`]).
    decoding: Mutex<HashMap<(SuperfileUri, usize), Arc<AsyncMutex<Vec<Option<ArrayRef>>>>>>,
}

/// A table scan served from the decoded-column cache; see the module docs.
#[derive(Debug, Clone)]
pub(crate) struct CachedScanExec {
    context: Arc<ScanContext>,
    units: Arc<[RowGroupUnit]>,
    /// Row groups the `WHERE` ruled out by their statistics.
    pruned: usize,
    partitions: usize,
    fetch: Option<usize>,
    statistics: Arc<Statistics>,
    properties: Arc<PlanProperties>,
    /// The Parquet scan this one stands in for, metered as the provider
    /// builds it. A plan that reads better through the Parquet scan, a
    /// `LIMIT` whose row filter decodes the wide columns only for the rows
    /// that pass, takes it back ([`Self::parquet`]).
    parquet: Arc<dyn ExecutionPlan>,
    /// Filters a parent pushed down that tighten while the statement runs
    /// ([`LiveFilters`]); `None` without any.
    live: Option<Arc<LiveFilters>>,
    /// Set when the scan took a sort-order pushdown, so it reads its row
    /// groups best first for a sort above it — what a top-k asks for. Only
    /// then are rows dropped per batch by the tightening bound; a hash join
    /// or aggregate pushes a filter but asks for no order, so its filter
    /// skips whole row groups ([`LiveFilters`]) but drops no rows.
    reads_best_first: bool,
    /// The `EXPLAIN ANALYZE` counters: the rows each partition hands on, the
    /// row groups decoded ([`ROW_GROUPS_DECODED`]) and read in part
    /// ([`ROW_GROUPS_READ_IN_PART`]), the rows of the units read
    /// ([`ROWS_READ`]), the row groups skipped ([`ROW_GROUPS_SKIPPED`]) and
    /// the rows dropped ([`ROWS_DROPPED`]).
    /// Set when a sort on `_id` was pushed down: the rows are read by id
    /// block ([`IdOrder`]).
    id_order: Option<IdOrder>,
    /// The rows of the units read ([`ROWS_READ`]), the row groups decoded
    /// ([`ROW_GROUPS_DECODED`]), the row groups skipped
    /// ([`ROW_GROUPS_SKIPPED`]) and the rows dropped ([`ROWS_DROPPED`]), for
    /// `EXPLAIN ANALYZE`.
    metrics: ExecutionPlanMetricsSet,
}

/// A read in `_id` order: the scan's row groups cut into id blocks, taken
/// in order of their lowest id (their highest, descending) from a queue
/// built by the first partition to start and shared by the rest, each taking
/// the next block as it is ready for one. `None` in the queue's cell when a
/// file has no sidecar bounds to order by; the row groups are read as they
/// are then.
#[derive(Debug, Clone)]
struct IdOrder {
    descending: bool,
    queue: Arc<OnceCell<Option<Arc<IdRead>>>>,
}

/// One file's part of an [`IdBlockQueue`].
#[derive(Debug)]
struct FileBlocks {
    /// The file's blocks, their bounds and orders, as its reader keeps them.
    blocks: Arc<IdBlocks>,
    /// For each of the file's row groups, its unit among the queue's units;
    /// `None` for a row group the scan doesn't read.
    units: Vec<Option<usize>>,
    /// Each row group's first row, and the file's row count last.
    starts: Vec<u32>,
    /// How many of the file's blocks, in order, have been taken.
    taken: usize,
}

/// The scan's id blocks in order, taken one at a time: the files' block
/// orders, sorted once per reader ([`SuperfileReader::id_blocks`]), merged as
/// blocks are taken, so a read that stops after a few blocks orders only
/// those.
#[derive(Debug)]
struct IdBlockQueue {
    units: Arc<[RowGroupUnit]>,
    files: Vec<FileBlocks>,
    descending: bool,
    /// Each file's next block, by its place in the order: the smallest
    /// first.
    next: BinaryHeap<Reverse<(u128, usize)>>,
    /// Pieces of a taken block not yet handed out: a block that spans two
    /// row groups is read as one piece of each.
    pending: VecDeque<RowGroupUnit>,
}

impl IdBlockQueue {
    /// The queue over `units`, row groups of the scan; `None` when a file
    /// has no sidecar bounds to order by, or fewer blocks than its rows
    /// fill.
    async fn new(units: Arc<[RowGroupUnit]>, descending: bool, in_flight: usize) -> Option<Self> {
        // The scan's distinct files, in the order they first appear among the
        // units; a file's place here is its place in `files`.
        let mut order: Vec<&Arc<CachedFile>> = Vec::new();
        let mut placed: HashMap<SuperfileUri, usize> = HashMap::new();
        for unit in units.iter() {
            placed.entry(unit.file.uri).or_insert_with(|| {
                order.push(&unit.file);
                order.len() - 1
            });
        }
        // Read the files' id blocks concurrently rather than awaiting each in
        // turn, so a first read over many files pays one fetch latency and not
        // one per file — but no more than `in_flight` at once, the cap the
        // other per-file I/O waves keep, so a table of thousands of files does
        // not open every sidecar at the same time. `buffered` preserves input
        // order, so `fetched` still lines up with `order`.
        let reads: Vec<_> = order
            .iter()
            .map(|file| {
                let file = Arc::clone(file);
                async move { file.reader.id_blocks().await }
            })
            .collect();
        let fetched: Vec<_> = stream::iter(reads)
            .buffered(in_flight.max(1))
            .collect()
            .await;
        let mut files: Vec<FileBlocks> = Vec::with_capacity(order.len());
        for (file, blocks) in order.iter().zip(fetched) {
            let blocks = blocks?;
            let mut starts = Vec::with_capacity(file.row_counts.len() + 1);
            let mut first = 0u32;
            for &n in file.row_counts.iter() {
                starts.push(first);
                first = first.checked_add(n)?;
            }
            starts.push(first);
            let needed = u64::from(first).div_ceil(BLOCK_DOCS as u64);
            if (blocks.bounds.len() as u64) < needed {
                return None;
            }
            let groups = file.row_counts.len();
            files.push(FileBlocks {
                blocks,
                units: vec![None; groups],
                starts,
                taken: 0,
            });
        }
        for (at, unit) in units.iter().enumerate() {
            let file = *placed.get(&unit.file.uri)?;
            *files[file].units.get_mut(unit.row_group)? = Some(at);
        }
        let mut queue = Self {
            units,
            files,
            descending,
            next: BinaryHeap::new(),
            pending: VecDeque::new(),
        };
        for file in 0..queue.files.len() {
            queue.push_next(file);
        }
        Some(queue)
    }

    /// `file`'s blocks in the order of the read.
    fn order(&self, file: usize) -> &[u32] {
        let blocks = &self.files[file].blocks;
        match self.descending {
            false => &blocks.by_lowest,
            true => &blocks.by_highest,
        }
    }

    /// Queues `file`'s next block, if it has one, by its place in the
    /// order: its lowest id ascending, or its highest descending, as an
    /// unsigned key that sorts the same way smallest first.
    fn push_next(&mut self, file: usize) {
        let taken = self.files[file].taken;
        let Some(&block) = self.order(file).get(taken) else {
            return;
        };
        let (lo, hi) = self.files[file].blocks.bounds[block as usize];
        let sign = 1u128 << 127;
        let key = match self.descending {
            false => (lo as u128) ^ sign,
            true => !((hi as u128) ^ sign),
        };
        self.next.push(Reverse((key, file)));
    }

    /// The next block to read, as a unit of the rows it holds of one row
    /// group; `None` once every block is taken. A block holding no row to
    /// read (in a row group the scan doesn't read, every row deleted, or
    /// none the index chose) is passed over. A block passed over in a row
    /// group the scan reads counts against that row group's block tally, as
    /// one read of it would, so the tally still reaches zero.
    fn take(&mut self, decoding: &BlockDecoding) -> Option<RowGroupUnit> {
        loop {
            if let Some(unit) = self.pending.pop_front() {
                return Some(unit);
            }
            let Reverse((_, file)) = self.next.pop()?;
            let block = self.order(file)[self.files[file].taken];
            self.files[file].taken += 1;
            self.push_next(file);
            let blocks = &self.files[file];
            let bounds = blocks.blocks.bounds[block as usize];
            let Some(end_of_file) = blocks.starts.last().copied() else {
                continue;
            };
            let start = block.saturating_mul(BLOCK_DOCS as u32);
            let end = start.saturating_add(BLOCK_DOCS as u32).min(end_of_file);
            // The row groups the block's rows fall in.
            let mut group = blocks
                .starts
                .partition_point(|&first| first <= start)
                .saturating_sub(1);
            while group + 1 < blocks.starts.len() && blocks.starts[group] < end {
                let rows = start.max(blocks.starts[group])..end.min(blocks.starts[group + 1]);
                if let Some(unit) = blocks.units[group].map(|at| &self.units[at]) {
                    let chosen = unit
                        .file
                        .selected
                        .as_ref()
                        .is_none_or(|selected| selected.range_cardinality(rows.clone()) > 0);
                    let deleted = unit.file.tombstones.range_cardinality(rows.clone());
                    if !rows.is_empty() && chosen && deleted < u64::from(rows.end - rows.start) {
                        self.pending.push_back(RowGroupUnit {
                            rows,
                            id_bounds: Some(bounds),
                            ..unit.clone()
                        });
                    } else {
                        decoding.done((unit.file.uri, unit.row_group));
                    }
                }
                group += 1;
            }
        }
    }
}

/// How many id blocks overlap `group`, a row group's rows by position in its
/// file: the blocks that read any of its rows. Blocks are [`BLOCK_DOCS`] rows
/// from the file's start, so this is a count over `group`'s ends, not a walk.
fn blocks_overlapping(group: &Range<u32>) -> usize {
    if group.is_empty() {
        return 0;
    }
    let block = BLOCK_DOCS as u32;
    let first = group.start / block;
    let last = (group.end - 1) / block;
    (last - first) as usize + 1
}

/// A read by id block: the queue the partitions take blocks from, and the
/// bookkeeping ([`BlockDecoding`]) that keeps each row group's shared decode
/// alive until its last block is read and bounds the bytes those decodes
/// hold. Built once by the first partition to start and shared by the rest.
#[derive(Debug)]
struct IdRead {
    queue: Mutex<IdBlockQueue>,
    decoding: BlockDecoding,
}

impl IdRead {
    /// The id-block read over `units`, descending or not; `None` when no file
    /// has the sidecar bounds to order by, as [`IdBlockQueue::new`] gives.
    async fn new(
        units: Arc<[RowGroupUnit]>,
        descending: bool,
        context: Arc<ScanContext>,
    ) -> Option<Self> {
        let in_flight = context.manifest.options.superfiles_in_flight;
        let queue = IdBlockQueue::new(Arc::clone(&units), descending, in_flight).await?;
        // Each row group the scan reads, by how many of its blocks are still
        // to come: read or skipped, each takes one off, and the shared decode
        // leaves once none remain.
        let remaining = units
            .iter()
            .map(|unit| {
                (
                    (unit.file.uri, unit.row_group),
                    RowGroupDecode {
                        blocks_left: blocks_overlapping(&unit.group),
                        bytes: 0,
                    },
                )
            })
            .collect();
        let cap = context.cache.max_bytes();
        Some(Self {
            queue: Mutex::new(queue),
            decoding: BlockDecoding {
                context,
                cap,
                inner: Mutex::new(BlockDecodingInner { remaining, held: 0 }),
            },
        })
    }
}

/// One row group's place in a [`BlockDecoding`].
#[derive(Debug)]
struct RowGroupDecode {
    /// Blocks of the row group the scan has still to read or skip; its shared
    /// decode is kept until this reaches zero.
    blocks_left: usize,
    /// Bytes the row group's kept decoded columns hold, counted in `held`;
    /// zero when its decode isn't kept.
    bytes: usize,
}

/// The id-block decodes a scan keeps: for each row group read by block, how
/// many of its blocks are left and how many bytes its kept columns hold.
#[derive(Debug)]
struct BlockDecodingInner {
    remaining: HashMap<(SuperfileUri, usize), RowGroupDecode>,
    /// Bytes the kept row-group decodes hold in sum.
    held: usize,
}

/// Keeps each row group's shared decode in [`ScanContext::decoding`] alive
/// until its last block is read or skipped, and bounds the bytes the kept
/// decodes hold across the scan at the decoded-column cache's own budget. A
/// row group whose columns the cache won't hold is still decoded once for its
/// blocks; past the budget a further row group's decode isn't kept, and its
/// later blocks decode it again.
#[derive(Debug)]
struct BlockDecoding {
    context: Arc<ScanContext>,
    /// The most bytes the kept decodes may hold in sum.
    cap: usize,
    inner: Mutex<BlockDecodingInner>,
}

impl BlockDecoding {
    /// Accounts a block's decode of `group`, `fresh` bytes of newly decoded
    /// columns: keeps the row group's decode for its other blocks while the
    /// kept decodes fit the cap, and always while it is the only one kept.
    /// Returns whether the decode isn't kept, when its entry is to be removed.
    fn account(&self, group: (SuperfileUri, usize), fresh: usize) -> bool {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let inner = &mut *inner;
        let Some(rg) = inner.remaining.get_mut(&group) else {
            // Its last block is already accounted for; nothing reads the
            // entry again, so this decode isn't kept.
            return true;
        };
        rg.bytes = rg.bytes.saturating_add(fresh);
        inner.held = inner.held.saturating_add(fresh);
        let others = inner.held - rg.bytes;
        if inner.held > self.cap && others > 0 {
            inner.held -= rg.bytes;
            rg.bytes = 0;
            true
        } else {
            false
        }
    }

    /// One block of `group` read or skipped: removes the row group's shared
    /// decode once its last block is done.
    fn done(&self, group: (SuperfileUri, usize)) {
        let drop_entry = {
            let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            let inner = &mut *inner;
            match inner.remaining.get_mut(&group) {
                Some(rg) => {
                    rg.blocks_left = rg.blocks_left.saturating_sub(1);
                    if rg.blocks_left == 0 {
                        inner.held = inner.held.saturating_sub(rg.bytes);
                        inner.remaining.remove(&group);
                        true
                    } else {
                        false
                    }
                }
                None => false,
            }
        };
        if drop_entry {
            self.context
                .decoding
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&group);
        }
    }
}

impl Drop for BlockDecoding {
    /// A read that stops early leaves blocks unread; their row groups' decodes
    /// never count down to zero, so the entries they may still hold leave with
    /// the scan.
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(PoisonError::into_inner);
        if inner.remaining.is_empty() {
            return;
        }
        let mut decoding = self
            .context
            .decoding
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for group in inner.remaining.keys() {
            decoding.remove(group);
        }
    }
}

/// One span of ids as the statistics of a single container, for pruning:
/// the `_id` column lies between its two ends and holds no null; nothing is
/// known of any other column.
struct IdSpan<'a> {
    column: &'a str,
    data_type: &'a DataType,
    lo: i128,
    hi: i128,
    rows: u64,
}

impl IdSpan<'_> {
    fn bound(&self, value: i128) -> Option<ArrayRef> {
        let DataType::Decimal128(precision, scale) = *self.data_type else {
            return None;
        };
        Decimal128Array::from(vec![value])
            .with_precision_and_scale(precision, scale)
            .ok()
            .map(|array| Arc::new(array) as ArrayRef)
    }
}

impl PruningStatistics for IdSpan<'_> {
    fn min_values(&self, column: &StatsColumn) -> Option<ArrayRef> {
        (column.name == self.column)
            .then(|| self.bound(self.lo))
            .flatten()
    }

    fn max_values(&self, column: &StatsColumn) -> Option<ArrayRef> {
        (column.name == self.column)
            .then(|| self.bound(self.hi))
            .flatten()
    }

    fn num_containers(&self) -> usize {
        1
    }

    fn null_counts(&self, column: &StatsColumn) -> Option<ArrayRef> {
        (column.name == self.column).then(|| Arc::new(UInt64Array::from(vec![0])) as ArrayRef)
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        Some(Arc::new(UInt64Array::from(vec![self.rows])))
    }

    fn contained(
        &self,
        _column: &StatsColumn,
        _values: &HashSet<ScalarValue>,
    ) -> Option<BooleanArray> {
        None
    }
}

/// [`LiveFilters`] checked against spans of `_id`, for a read by id block:
/// each filter's pruning is rebuilt only when the filter has changed since
/// the last check.
struct IdGate {
    live: Arc<LiveFilters>,
    schema: SchemaRef,
    column: String,
    data_type: DataType,
    /// Each filter's generation when its pruning was built, and the pruning.
    /// The generation is `None` until the pruning is first built, so a filter
    /// whose generation reads 0 still builds; the pruning is `None` for a
    /// filter that rules nothing out (a top-k's before its first `k` rows) or
    /// doesn't convert.
    built: Vec<(Option<u64>, Option<PruningPredicate>)>,
}

impl IdGate {
    /// A gate over `live` for the scan output `schema`'s `_id` column
    /// `column`; `None` when the scan doesn't output it as a decimal.
    fn new(live: Arc<LiveFilters>, schema: SchemaRef, column: &str) -> Option<Self> {
        let data_type = schema.field_with_name(column).ok()?.data_type().clone();
        matches!(data_type, DataType::Decimal128(..)).then(|| Self {
            built: Vec::new(),
            live,
            schema,
            column: column.to_owned(),
            data_type,
        })
    }

    /// Whether the filters as they stand rule out every row whose `_id` lies
    /// between `lo` and `hi`.
    fn rules_out(&mut self, lo: i128, hi: i128, rows: u64) -> bool {
        if self.built.len() != self.live.filters.len() {
            self.built = self.live.filters.iter().map(|_| (None, None)).collect();
        }
        let span = IdSpan {
            column: &self.column,
            data_type: &self.data_type,
            lo,
            hi,
            rows,
        };
        let schema = &self.schema;
        self.live.filters.iter().zip(self.built.iter_mut()).any(
            |(filter, (generation, pruning))| {
                let now = snapshot_generation(filter);
                if *generation != Some(now) {
                    *generation = Some(now);
                    *pruning = snapshot_physical_expr(Arc::clone(filter))
                        .ok()
                        .and_then(|expr| PruningPredicate::try_new(expr, Arc::clone(schema)).ok())
                        .filter(|pruning| !pruning.always_true());
                }
                pruning
                    .as_ref()
                    .and_then(|pruning| pruning.prune(&span).ok())
                    .is_some_and(|keep| keep.first() == Some(&false))
            },
        )
    }
}

/// Filters pushed down from above that tighten as the statement runs: a
/// top-k's bound on its sort key (the k-th best value so far), a hash join's
/// filter on its probe side, a `MIN` or `MAX` aggregate's best value so far.
/// Each is checked against a row group's statistics just before the row
/// group is read ([`RowGroupGate`]), and a row group that can no longer
/// reach the result is skipped. When the scan reads best first for a sort
/// above it — what a top-k asks for — the tightening bound is also checked
/// against each batch the scan hands over, and the rows it rules out are
/// dropped before the node above sees them; a join's or aggregate's filter,
/// which asks for no order, only skips row groups. A row such a filter rules
/// out can't change its parent's answer, so dropping it changes no answer;
/// the rows that pass are still checked above.
#[derive(Debug)]
struct LiveFilters {
    filters: Vec<Arc<dyn PhysicalExpr>>,
    /// Maps a filter onto each file's own schema, as the scan's `WHERE` is.
    adapter: Arc<dyn PhysicalExprAdapterFactory>,
}

impl LiveFilters {
    /// `batch` without the rows the filters as they stand rule out. A filter
    /// that rules nothing out yet (a top-k's before its first `k` rows) is
    /// not evaluated.
    fn keep(&self, mut batch: RecordBatch) -> DfResult<RecordBatch> {
        for filter in &self.filters {
            let now = snapshot_physical_expr(Arc::clone(filter))?;
            if now
                .downcast_ref::<Literal>()
                .is_some_and(|literal| literal.value() == &ScalarValue::Boolean(Some(true)))
            {
                continue;
            }
            let mask = now.evaluate(&batch)?.into_array(batch.num_rows())?;
            let mask = mask
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal("a live filter is not boolean".to_owned())
                })?;
            batch = filter_record_batch(&batch, mask)?;
        }
        Ok(batch)
    }
}

/// What [`RowGroupGate`] holds for one file: its schema, read once, and for
/// each filter the generation its pruning was built at and the pruning.
struct FileGate {
    /// The file's schema its statistics are read by ([`file_schema`]); `None`
    /// when it doesn't parse, and nothing on the file is pruned.
    schema: Option<SchemaRef>,
    /// Each filter's generation when its pruning was built, and the pruning:
    /// `None` for a filter that rules nothing out yet (a top-k's before its
    /// first `k` rows) or doesn't map onto the file.
    filters: Vec<(Option<u64>, Option<PruningPredicate>)>,
}

/// [`LiveFilters`] checked against the statistics of one row group at a
/// time, for one partition of the scan. A file's schema is parsed once; each
/// filter's pruning is built for a file once, and again only when the filter
/// has tightened since. A check reads the statistics of the one row group it
/// decides.
struct RowGroupGate {
    live: Arc<LiveFilters>,
    /// The scan's output schema, the filters are written against.
    table: SchemaRef,
    /// Each file's schema and per-filter pruning ([`FileGate`]), filled in as
    /// the file's row groups first come up and the filters tighten.
    built: HashMap<SuperfileUri, FileGate>,
}

impl RowGroupGate {
    fn new(live: Arc<LiveFilters>, table: SchemaRef) -> Self {
        Self {
            live,
            table,
            built: HashMap::new(),
        }
    }

    /// Whether the filters as they stand rule `unit`'s row group out by its
    /// statistics.
    fn skip(&mut self, unit: &RowGroupUnit) -> bool {
        let Self { live, table, built } = self;
        let live: &LiveFilters = live;
        let table: &SchemaRef = table;
        let file = &unit.file;
        let gate = built.entry(file.uri).or_insert_with(|| FileGate {
            schema: file_schema(table, &file.footer),
            filters: live.filters.iter().map(|_| (None, None)).collect(),
        });
        let Some(schema) = &gate.schema else {
            // The file's schema didn't parse; nothing on it is pruned.
            return false;
        };
        live.filters
            .iter()
            .zip(gate.filters.iter_mut())
            .any(|(filter, (generation, pruning))| {
                let now = snapshot_generation(filter);
                if *generation != Some(now) {
                    *generation = Some(now);
                    *pruning = snapshot_physical_expr(Arc::clone(filter))
                        .ok()
                        .and_then(|expr| pruning_for(&expr, &live.adapter, table, schema))
                        .filter(|pruning| !pruning.always_true());
                }
                pruning.as_ref().is_some_and(|pruning| {
                    let statistics = OneRowGroup {
                        footer: &file.footer,
                        row_group: unit.row_group,
                        schema,
                    };
                    pruning
                        .prune(&statistics)
                        .is_ok_and(|keep| keep.first() == Some(&false))
                })
            })
    }
}

/// The statistics of one row group of a file, for pruning, read as
/// DataFusion's Parquet scan reads a file's row groups.
struct OneRowGroup<'a> {
    footer: &'a ParquetMetaData,
    row_group: usize,
    /// The file's schema, as its pruning was built against.
    schema: &'a Schema,
}

impl<'a> OneRowGroup<'a> {
    fn converter(&self, column: &StatsColumn) -> Option<StatisticsConverter<'a>> {
        StatisticsConverter::try_new(
            &column.name,
            self.schema,
            self.footer.file_metadata().schema_descr(),
        )
        .ok()
        .map(|converter| converter.with_missing_null_counts_as_zero(true))
    }

    fn group(&self) -> IntoIter<&'a RowGroupMetaData> {
        self.footer.row_groups().get(self.row_group).into_iter()
    }
}

impl PruningStatistics for OneRowGroup<'_> {
    fn min_values(&self, column: &StatsColumn) -> Option<ArrayRef> {
        self.converter(column)?.row_group_mins(self.group()).ok()
    }

    fn max_values(&self, column: &StatsColumn) -> Option<ArrayRef> {
        self.converter(column)?.row_group_maxes(self.group()).ok()
    }

    fn num_containers(&self) -> usize {
        1
    }

    fn null_counts(&self, column: &StatsColumn) -> Option<ArrayRef> {
        self.converter(column)?
            .row_group_null_counts(self.group())
            .ok()
            .map(|counts| Arc::new(counts) as ArrayRef)
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        let rows = self.footer.row_groups().get(self.row_group)?.num_rows();
        let rows = u64::try_from(rows).ok()?;
        Some(Arc::new(UInt64Array::from(vec![rows])) as ArrayRef)
    }

    fn contained(
        &self,
        _column: &StatsColumn,
        _values: &HashSet<ScalarValue>,
    ) -> Option<BooleanArray> {
        None
    }
}

impl CachedScanExec {
    /// A scan of `schema`'s columns over `files`, read as up to
    /// `target_partitions` partitions, skipping the row groups `filter` rules
    /// out, standing in for `parquet`. `None`, and the Parquet scan serves the
    /// statement, when:
    /// - a column is not one the files store in a type that converts to
    ///   `schema`'s (the Parquet scan's schema adapter serves those);
    /// - the columns it would decode don't fit the cache ([`decoded_bytes`]):
    ///   kept, they would push each other out, and every run would decode
    ///   them all again where the Parquet scan decodes only what it reads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        schema: SchemaRef,
        files: Vec<CachedFile>,
        filter: Option<RowGroupFilter>,
        manifest: Arc<ManifestSnapshot>,
        op_stats: Option<Arc<OpStatsCollector>>,
        cache: Arc<DecodedColumnCache>,
        target_partitions: usize,
        statistics: Statistics,
        parquet: Arc<dyn ExecutionPlan>,
    ) -> Option<Self> {
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        let stored = hit_output_schema(&manifest.stored_schema(), &names).ok()?;
        let convertible = stored
            .fields()
            .iter()
            .zip(schema.fields())
            .all(|(from, to)| can_cast_types(from.data_type(), to.data_type()));
        if !convertible {
            return None;
        }
        let schema_id = manifest.table_schema().schema_id();
        let mut units = Vec::new();
        let mut pruned = 0;
        for file in files.into_iter().map(Arc::new) {
            let plan = filter.as_ref().and_then(|filter| filter.plan(&file));
            let mut first = 0u32;
            for (row_group, &n) in file.row_counts.iter().enumerate() {
                let rows = first..first + n;
                first += n;
                if plan
                    .as_ref()
                    .is_some_and(|plan| !plan.should_scan(row_group))
                {
                    pruned += 1;
                    continue;
                }
                // A row group every row of which is deleted reads nothing, nor
                // does one holding none of the rows the index chose.
                let live = u64::from(n) - file.tombstones.range_cardinality(rows.clone());
                let chosen = file
                    .selected
                    .as_ref()
                    .map_or(live, |selected| selected.range_cardinality(rows.clone()));
                if chosen == 0 {
                    continue;
                }
                let keep = file.selected.is_none()
                    || chosen.saturating_mul(2) >= live
                    || cache.read_in_part_before(&RowGroupKey {
                        uri: file.uri,
                        schema_id,
                        row_group,
                    });
                units.push(RowGroupUnit {
                    file: Arc::clone(&file),
                    row_group,
                    group: rows.clone(),
                    rows,
                    keep,
                    id_bounds: None,
                });
            }
        }
        let dictionaries = dictionary_columns(&manifest, &schema, &units);
        if decoded_bytes(
            units.iter().filter(|unit| unit.keep),
            &schema,
            &dictionaries,
        ) > cache.max_bytes()
        {
            return None;
        }
        let partitions = target_partitions.clamp(1, units.len().max(1));
        let properties = plan_properties(&schema, partitions);
        Some(Self {
            context: Arc::new(ScanContext {
                schema,
                manifest,
                op_stats,
                cache,
                schema_id,
                dictionaries,
                predicate: None,
                decoding: Mutex::default(),
            }),
            units: units.into(),
            pruned,
            partitions,
            fetch: None,
            statistics: Arc::new(statistics),
            properties,
            parquet,
            live: None,
            reads_best_first: false,
            id_order: None,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// The Parquet scan this one stands in for, metered as the provider
    /// builds it.
    pub(crate) fn parquet(&self) -> &Arc<dyn ExecutionPlan> {
        &self.parquet
    }

    /// This scan reading by id block ([`IdOrder`]) for a sort on its output
    /// column `column`, descending or not; `None` when that column isn't the
    /// table's `_id`.
    pub(crate) fn in_id_order(
        &self,
        column: usize,
        descending: bool,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        let field = self.context.schema.fields().get(column)?;
        (field.name() == &self.context.manifest.options.id_column).then(|| {
            Arc::new(Self {
                id_order: Some(IdOrder {
                    descending,
                    queue: Arc::default(),
                }),
                ..self.clone()
            }) as Arc<dyn ExecutionPlan>
        })
    }
}

/// About how many bytes the columns of `schema` take decoded over `units`'
/// row groups (those a scan would keep), from the footers:
/// - a fixed-width column, its width per row;
/// - any other, [`VIEW_BYTES_PER_ROW`] per row plus its values' bytes, which
///   the footer records unencoded (a dictionary-encoded column stores far
///   fewer bytes than it decodes to), or failing that its chunk's
///   uncompressed size.
///
/// A column a file holds under another name, after a rename, isn't found by
/// this name and counts as nothing; the estimate decides only which scan
/// serves the statement, never what it returns.
fn decoded_bytes<'a>(
    units: impl Iterator<Item = &'a RowGroupUnit>,
    schema: &Schema,
    dictionaries: &HashMap<SuperfileUri, Arc<[bool]>>,
) -> usize {
    units
        .map(|unit| {
            let dictionary = dictionaries.get(&unit.file.uri);
            let meta = unit.file.reader.parquet_metadata();
            let Some(row_group) = meta.row_groups().get(unit.row_group) else {
                return 0;
            };
            let rows = usize::try_from(row_group.num_rows()).unwrap_or(0);
            schema
                .fields()
                .iter()
                .enumerate()
                .map(|(i, field)| {
                    let Some(chunk) = row_group.columns().iter().find(|chunk| {
                        chunk
                            .column_descr()
                            .path()
                            .parts()
                            .first()
                            .is_some_and(|root| root == field.name())
                    }) else {
                        return 0;
                    };
                    if dictionary.is_some_and(|kept| kept[i]) {
                        // The keys, and at most one value in
                        // `DICTIONARY_ROWS_PER_VALUE` rows.
                        let values = chunk
                            .unencoded_byte_array_data_bytes()
                            .unwrap_or_else(|| chunk.uncompressed_size());
                        return rows
                            .saturating_mul(DICTIONARY_KEY_BYTES_PER_ROW)
                            .saturating_add(
                                usize::try_from(values).unwrap_or(0) / DICTIONARY_ROWS_PER_VALUE,
                            );
                    }
                    match field.data_type().primitive_width() {
                        Some(width) => rows.saturating_mul(width),
                        None => {
                            let values = chunk
                                .unencoded_byte_array_data_bytes()
                                .unwrap_or_else(|| chunk.uncompressed_size());
                            rows.saturating_mul(VIEW_BYTES_PER_ROW)
                                .saturating_add(usize::try_from(values).unwrap_or(0))
                        }
                    }
                })
                .sum::<usize>()
        })
        .sum()
}

/// For each file `units` read, which of `schema`'s columns the cache keeps as
/// dictionaries: the string columns whose superfile statistics count at most
/// one value per [`DICTIONARY_ROWS_PER_VALUE`] rows, and no more values than
/// `Int16` keys reach. The superfile's exact value table counts them where
/// it keeps one, and its distinct sketch estimates them otherwise; a column
/// with neither is kept as the scan reads it. The superfiles are written
/// without Parquet dictionaries, so the dictionary is built once, when the
/// row group is first decoded, and every later statement reads it kept.
fn dictionary_columns(
    manifest: &ManifestSnapshot,
    schema: &Schema,
    units: &[RowGroupUnit],
) -> HashMap<SuperfileUri, Arc<[bool]>> {
    let table = manifest.table_schema();
    let fields: Vec<Option<FieldId>> = schema
        .fields()
        .iter()
        .map(|field| {
            matches!(
                field.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            )
            .then(|| table.id_of(field.name()))
            .flatten()
        })
        .collect();
    if fields.iter().all(Option::is_none) {
        return HashMap::new();
    }
    let read: HashSet<SuperfileUri> = units.iter().map(|unit| unit.file.uri).collect();
    manifest
        .superfiles
        .iter()
        .filter(|entry| read.contains(&entry.uri))
        .map(|entry| {
            let rows = usize::try_from(entry.n_docs).unwrap_or(usize::MAX);
            let kept: Arc<[bool]> = fields
                .iter()
                .map(|field| {
                    let Some(stats) = field.and_then(|id| entry.scalar_stats.column(id)) else {
                        return false;
                    };
                    // The values, and what they must leave under the keys'
                    // reach: an estimate keeps a margin, an exact count none.
                    let distinct = match stats.value_table_len() {
                        Some(exact) => Some((exact, exact)),
                        None => stats
                            .sketch()
                            .and_then(HllSketch::from_bytes)
                            .map(|sketch| sketch.estimate().ceil() as usize)
                            .map(|estimate| (estimate, estimate.saturating_mul(SKETCH_KEY_MARGIN))),
                    };
                    distinct.is_some_and(|(distinct, reach)| {
                        reach <= i16::MAX as usize
                            && distinct.saturating_mul(DICTIONARY_ROWS_PER_VALUE) <= rows
                    })
                })
                .collect();
            (entry.uri, kept)
        })
        .collect()
}

/// `array` as the cache keeps a column of few values: a dictionary of
/// `Int16` keys over its strings. `None` when it does not take that type.
fn as_dictionary(array: &ArrayRef) -> Option<ArrayRef> {
    let keep = DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8));
    cast_with_options(array, &keep, &CAST_OPTIONS).ok()
}

/// A kept column as the scan types it: a kept dictionary as the scan's
/// strings, every other column as it is.
fn as_scanned(kept: ArrayRef, scanned: &DataType) -> DfResult<ArrayRef> {
    match kept.data_type() {
        DataType::Dictionary(..) if kept.data_type() != scanned => {
            Ok(cast_with_options(&kept, scanned, &CAST_OPTIONS)?)
        }
        _ => Ok(kept),
    }
}

/// `units` ordered by the row-group statistics of `column`, smallest minimum
/// first (largest maximum first when `descending`); a row group without a
/// statistic for the column, and one whose bounds don't cast, ordered where
/// `nulls_first` puts them, so a `NULLS FIRST` sort reads the row groups that
/// may hold its leading nulls first. `None` when no file has statistics for
/// it. Each file's bounds are cast to `data_type`, the type the scan reads
/// the column as, so files written before the column was widened order with
/// the rest.
fn best_first(
    units: &[RowGroupUnit],
    column: &str,
    data_type: &DataType,
    descending: bool,
    nulls_first: bool,
) -> Option<Vec<RowGroupUnit>> {
    // Each file's bounds, read and cast once, laid back to back in `all`,
    // then one trailing null row the units with no usable bound point at: a
    // file whose bounds don't cast, and a row group past the ones its file
    // has statistics for. The key column is one `take` from `all`, so each
    // file allocates once however many row groups it holds.
    let mut offsets: HashMap<SuperfileUri, Option<(usize, usize)>> = HashMap::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    let mut len = 0usize;
    for unit in units {
        offsets.entry(unit.file.uri).or_insert_with(|| {
            let array = row_group_bounds(&unit.file.footer, column, descending)
                .and_then(|array| cast_with_options(&array, data_type, &CAST_OPTIONS).ok())?;
            let at = len;
            let rows = array.len();
            len += rows;
            arrays.push(array);
            Some((at, rows))
        });
    }
    let null_at = len;
    arrays.push(new_null_array(data_type, 1));
    let arrays: Vec<&dyn Array> = arrays.iter().map(|array| array.as_ref()).collect();
    let all = concat(&arrays).ok()?;
    let indices = UInt64Array::from_iter_values(units.iter().map(|unit| {
        let at = offsets
            .get(&unit.file.uri)
            .copied()
            .flatten()
            .filter(|&(_, rows)| unit.row_group < rows)
            .map_or(null_at, |(at, _)| at + unit.row_group);
        at as u64
    }));
    let keys = take(all.as_ref(), &indices, None).ok()?;
    if keys.null_count() == keys.len() {
        return None;
    }
    // Arrow's sort orders every value of one type, NaNs and all.
    let options = SortOptions {
        descending,
        nulls_first,
    };
    let order = sort_to_indices(&keys, Some(options), None).ok()?;
    Some(
        order
            .values()
            .iter()
            .map(|&at| units[at as usize].clone())
            .collect(),
    )
}

/// Each row group's minimum (maximum when `descending`) of `column` in a
/// file with `footer`; `None` when the file does not hold it or records no
/// statistics a converter reads.
fn row_group_bounds(footer: &ParquetMetaData, column: &str, descending: bool) -> Option<ArrayRef> {
    let file = footer.file_metadata();
    let schema = parquet_to_arrow_schema(file.schema_descr(), file.key_value_metadata()).ok()?;
    let converter = StatisticsConverter::try_new(column, &schema, file.schema_descr()).ok()?;
    match descending {
        false => converter.row_group_mins(footer.row_groups().iter()).ok(),
        true => converter.row_group_maxes(footer.row_groups().iter()).ok(),
    }
}

/// The plan properties of a scan of `schema` read as `partitions` partitions.
fn plan_properties(schema: &SchemaRef, partitions: usize) -> Arc<PlanProperties> {
    Arc::new(PlanProperties::new(
        EquivalenceProperties::new(Arc::clone(schema)),
        Partitioning::UnknownPartitioning(partitions),
        EmissionType::Incremental,
        Boundedness::Bounded,
    ))
}

/// Which of `rows` (positions in the file) are in `selected`, one entry per
/// row of the range.
fn selected_mask(selected: &RoaringBitmap, rows: &Range<u32>) -> BooleanArray {
    let n = rows.len();
    let mut mask = BooleanBufferBuilder::new(n);
    mask.append_n(n, false);
    for row in selected.range(rows.clone()) {
        mask.set_bit((row - rows.start) as usize, true);
    }
    BooleanArray::new(mask.finish(), None)
}

/// `array` typed as `target`, the scan schema's type for its column.
fn as_scan_type(array: &ArrayRef, target: &DataType) -> DfResult<ArrayRef> {
    Ok(if array.data_type() == target {
        Arc::clone(array)
    } else {
        cast_with_options(array, target, &CAST_OPTIONS)?
    })
}

/// `batch` as slices of at most `rows` rows each, sharing its buffers.
fn sliced(batch: RecordBatch, rows: usize) -> Vec<RecordBatch> {
    let rows = rows.max(1);
    let n = batch.num_rows();
    (0..n)
        .step_by(rows)
        .map(|at| batch.slice(at, rows.min(n - at)))
        .collect()
}

impl ScanContext {
    /// The cache's key for `unit`'s row group.
    fn row_group_key(&self, unit: &RowGroupUnit) -> RowGroupKey {
        RowGroupKey {
            uri: unit.file.uri,
            schema_id: self.schema_id,
            row_group: unit.row_group,
        }
    }

    /// The cache's key for `column` of `unit`'s row group.
    fn key(&self, unit: &RowGroupUnit, column: &str) -> ColumnKey {
        ColumnKey {
            uri: unit.file.uri,
            schema_id: self.schema_id,
            column: column.to_owned(),
            row_group: unit.row_group,
        }
    }

    /// The `missing` projected columns of `unit`'s row group, decoded, typed
    /// as the scan schema types them, and kept in the cache. A read of the
    /// same row group decoding already is waited for, and the columns it
    /// decoded are taken from it.
    async fn decode_and_keep(
        self: Arc<Self>,
        unit: RowGroupUnit,
        missing: Vec<usize>,
        decoded_groups: Count,
        id_read: Option<Arc<IdRead>>,
    ) -> DfResult<Vec<ArrayRef>> {
        let group = (unit.file.uri, unit.row_group);
        let shared = Arc::clone(
            self.decoding
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(group)
                .or_insert_with(|| {
                    Arc::new(AsyncMutex::new(vec![None; self.schema.fields().len()]))
                }),
        );
        let read = self
            .decode_shared(&unit, &missing, &shared, decoded_groups)
            .await;
        match &id_read {
            // Read by id block: the entry stays for the row group's other
            // blocks, removed when its last block is read or skipped
            // ([`BlockDecoding::done`]). Past the budget this decode isn't
            // kept, and the entry goes now so the bytes aren't held.
            Some(id_read) => {
                if read
                    .as_ref()
                    .is_ok_and(|(_, fresh)| id_read.decoding.account(group, *fresh))
                {
                    self.decoding
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .remove(&group);
                }
            }
            // Read whole: the entry leaves once no read waits on it, and a
            // later read of the row group finds the columns kept until then. A
            // whole row group is read once a scan, so its entry then goes.
            // Dropping this read's clone before the count is taken, and taking
            // it under the map's lock, leaves the map the sole holder exactly
            // when no read waits on the entry: two reads finishing at once
            // can't each count the other's clone and both keep it, and no read
            // holding the entry can be dropped from under.
            None => {
                drop(shared);
                let mut decoding = self.decoding.lock().unwrap_or_else(PoisonError::into_inner);
                let alone = decoding
                    .get(&group)
                    .is_some_and(|entry| Arc::strong_count(entry) == 1);
                if alone {
                    decoding.remove(&group);
                }
            }
        }
        read.map(|(columns, _)| columns)
    }

    /// The `missing` projected columns of `unit`'s row group from `shared`,
    /// the columns reads of it decoded, else from the cache, else decoded,
    /// kept, and left in `shared` for the reads waiting on it; and the bytes
    /// of the columns this read decoded afresh, which it alone added to
    /// `shared`.
    async fn decode_shared(
        &self,
        unit: &RowGroupUnit,
        missing: &[usize],
        shared: &AsyncMutex<Vec<Option<ArrayRef>>>,
        decoded_groups: Count,
    ) -> DfResult<(Vec<ArrayRef>, usize)> {
        let mut decoded = shared.lock().await;
        let fields = self.schema.fields();
        let mut columns: Vec<Option<ArrayRef>> = missing
            .iter()
            .map(|&i| {
                decoded[i]
                    .clone()
                    .or_else(|| self.cache.get(&self.key(unit, fields[i].name())))
            })
            .collect();
        let at: Vec<usize> = (0..missing.len())
            .filter(|&k| columns[k].is_none())
            .collect();
        let mut fresh = 0usize;
        if !at.is_empty() {
            let still: Vec<usize> = at.iter().map(|&k| missing[k]).collect();
            decoded_groups.add(1);
            for (k, array) in at.into_iter().zip(self.decode(unit, &still).await?) {
                fresh = fresh.saturating_add(array.get_array_memory_size());
                decoded[missing[k]] = Some(Arc::clone(&array));
                columns[k] = Some(array);
            }
        }
        let columns = columns
            .into_iter()
            .map(|c| c.ok_or_else(|| DataFusionError::Internal("a column was not read".into())))
            .collect::<DfResult<_>>()?;
        Ok((columns, fresh))
    }

    /// The `missing` projected columns of `unit`'s row group, decoded, typed
    /// as the scan schema types them, and kept in the cache.
    async fn decode(&self, unit: &RowGroupUnit, missing: &[usize]) -> DfResult<Vec<ArrayRef>> {
        let fields = self.schema.fields();
        let names: Vec<&str> = missing.iter().map(|&i| fields[i].name().as_str()).collect();
        // Read as the scan types them, so a string the scan works on as a
        // view decodes into one directly.
        let typed = Arc::new(self.schema.project(missing).map_err(QueryError::internal)?);
        let read = read_file_row_group(
            &self.manifest,
            &unit.file.reader,
            unit.row_group,
            &names,
            &typed,
            &self.op_stats,
        )
        .await?;
        let mut decoded = Vec::with_capacity(missing.len());
        let dictionary = self.dictionaries.get(&unit.file.uri);
        for (&i, array) in missing.iter().zip(read.columns()) {
            let array = as_scan_type(array, fields[i].data_type())?;
            // A column of few values is kept as a dictionary; this statement
            // reads the strings it decoded.
            let kept = match dictionary.is_some_and(|kept| kept[i]) {
                true => as_dictionary(&array).unwrap_or_else(|| Arc::clone(&array)),
                false => Arc::clone(&array),
            };
            self.cache.insert(
                self.key(unit, fields[i].name()),
                Arc::clone(&kept),
                kept.get_array_memory_size(),
            );
            decoded.push(array);
        }
        Ok(decoded)
    }

    /// `unit`'s rows, every projected column, without its deleted rows and,
    /// when the index chose the file's rows, without the rows it didn't. For a
    /// whole row group the columns are decoded and kept, counted in
    /// `decoded_groups`; when the index chose few rows and the group isn't
    /// kept, just those rows are read from the pages, counted in
    /// `read_in_part`.
    async fn read(
        self: &Arc<Self>,
        unit: &RowGroupUnit,
        decoded_groups: Count,
        read_in_part: &Count,
        id_read: Option<Arc<IdRead>>,
    ) -> DfResult<RecordBatch> {
        let fields = self.schema.fields();
        // As kept: a column kept as a dictionary stays one until the rows
        // the predicate keeps are known.
        let mut columns: Vec<Option<ArrayRef>> = fields
            .iter()
            .map(|f| self.cache.get(&self.key(unit, f.name())))
            .collect();
        let missing: Vec<usize> = (0..columns.len())
            .filter(|&i| columns[i].is_none())
            .collect();
        if !missing.is_empty()
            && !unit.keep
            && id_read.is_none()
            && let Some(selected) = &unit.file.selected
        {
            read_in_part.add(1);
            return self
                .read_chosen_rows(unit, selected, columns, missing)
                .await;
        }
        if !missing.is_empty() {
            // The decode runs as its own task: a `LIMIT` above drops this
            // partition's stream once it has its rows, and a decode dropped
            // halfway would have spent its CPU without keeping the columns.
            // Run to the end, it keeps them for the next statement.
            let decoded = spawn(
                Arc::clone(self)
                    .decode_and_keep(unit.clone(), missing.clone(), decoded_groups, id_read)
                    .in_current_span(),
            )
            .await
            .map_err(QueryError::internal)??;
            for (i, array) in missing.into_iter().zip(decoded) {
                columns[i] = Some(array);
            }
        }
        let columns = columns
            .into_iter()
            .map(|c| c.ok_or_else(|| DataFusionError::Internal("a column was not read".into())))
            .collect::<DfResult<Vec<_>>>()?;
        let n = unit.rows.len();
        let start = unit.rows.start;
        // An id-block unit reads a contiguous slice of a row group; the decoded
        // columns span the whole group, so cut them to the unit's rows before
        // the masks, which are sized to the unit.
        let columns: Vec<ArrayRef> = match unit.rows == unit.group {
            true => columns,
            false => columns
                .iter()
                .map(|column| column.slice((start - unit.group.start) as usize, n))
                .collect(),
        };
        let keep = match &unit.file.selected {
            Some(selected) => Some(selected_mask(selected, &unit.rows)),
            None => {
                let mut keep: Option<BooleanBufferBuilder> = None;
                for row in unit.file.tombstones.range(unit.rows.clone()) {
                    let mask = keep.get_or_insert_with(|| {
                        let mut mask = BooleanBufferBuilder::new(n);
                        mask.append_n(n, true);
                        mask
                    });
                    mask.set_bit((row - start) as usize, false);
                }
                keep.map(|mut mask| BooleanArray::new(mask.finish(), None))
            }
        };
        let live = keep;
        let keep = match &self.predicate {
            Some(predicate) => passing_rows(predicate, &self.schema, &columns, live, n),
            None => live,
        };
        let rows = keep.as_ref().map_or(n, BooleanArray::true_count);
        let columns = match &keep {
            Some(mask) => columns
                .iter()
                .map(|column| filter(column.as_ref(), mask))
                .collect::<Result<Vec<_>, _>>()?,
            None => columns,
        };
        let columns = columns
            .into_iter()
            .zip(fields)
            .map(|(column, field)| as_scanned(column, field.data_type()))
            .collect::<DfResult<Vec<_>>>()?;
        Ok(RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )?)
    }

    /// The `selected` rows of `unit`'s row group: the `missing` columns read
    /// from the pages for just those rows and not kept, the others taken from
    /// the kept `columns`. The cache remembers the row group, so the next scan
    /// that chooses rows in it decodes and keeps its columns.
    async fn read_chosen_rows(
        &self,
        unit: &RowGroupUnit,
        selected: &RoaringBitmap,
        columns: Vec<Option<ArrayRef>>,
        missing: Vec<usize>,
    ) -> DfResult<RecordBatch> {
        self.cache.note_read_in_part(self.row_group_key(unit));
        let fields = self.schema.fields();
        let rows: Vec<u32> = selected.range(unit.rows.clone()).collect();
        let chosen = rows.len();
        let names: Vec<&str> = missing.iter().map(|&i| fields[i].name().as_str()).collect();
        let stored = hit_output_schema(&self.manifest.stored_schema(), &names)?;
        let taken = take_file_columns(
            &self.manifest,
            &unit.file.reader,
            rows,
            &names,
            &stored,
            &self.op_stats,
        )
        .await?;
        if taken.num_rows() != chosen {
            return Err(DataFusionError::Internal(format!(
                "read {} rows of a row group for {chosen} chosen ones",
                taken.num_rows()
            )));
        }
        let mask = selected_mask(selected, &unit.rows);
        // `missing` lists the absent columns in schema order, as the take
        // returned them.
        let mut taken = taken.columns().iter();
        let columns = columns
            .into_iter()
            .zip(fields)
            .map(|(column, field)| match column {
                // A column kept as a dictionary comes out as the scan's
                // strings.
                Some(kept) => as_scanned(filter(kept.as_ref(), &mask)?, field.data_type()),
                None => {
                    let array = taken
                        .next()
                        .ok_or_else(|| DataFusionError::Internal("a column was not read".into()))?;
                    as_scan_type(array, field.data_type())
                }
            })
            .collect::<DfResult<Vec<_>>>()?;
        RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(chosen)),
        )
        .map_err(DataFusionError::from)
    }
}

/// Which of a row group's `rows` rows the scan hands on: those `live` keeps
/// (every row without it) that pass the conjuncts of `predicate` the scan
/// checks, over `columns` as kept. Never null.
///
/// The filter above checks every row it gets, and DataFusion's `AND` checks
/// its right side on every row of the batch, not only on the rows its left
/// side keeps, once the left keeps more than a fifth of them. Handed fewer
/// rows than the files hold, the filter can so check a conjunct on a row an
/// earlier conjunct drops, where without a cache it wouldn't: a guard like
/// `x <> 0 AND 10 / x > 1` would fail. So when a conjunct reads a column kept
/// as a dictionary, the scan
/// - checks every conjunct, in order, each on the rows that passed the ones
///   before it, as `AND` would, and never on a deleted or unselected row. The
///   rows it hands on pass all of them, so the filter checks each conjunct
///   only on rows it has already been checked on;
/// - leaves the last conjunct to the filter when it reads no kept
///   dictionary: the filter checks it once, on rows that pass every conjunct
///   before it;
/// - checks a conjunct on one kept dictionary once per value the rows use,
///   and any other on the rows themselves;
/// - stops at a conjunct that fails to evaluate on the rows, and hands on the
///   rows it has: the filter then checks that conjunct on the same rows, and
///   fails as it does with no scan below it.
///
/// A row group whose dictionaries no conjunct reads is handed on as it is.
fn passing_rows(
    predicate: &Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
    columns: &[ArrayRef],
    live: Option<BooleanArray>,
    rows: usize,
) -> Option<BooleanArray> {
    let conjuncts: Vec<(&Arc<dyn PhysicalExpr>, Vec<usize>, bool)> = split_conjunction(predicate)
        .into_iter()
        .map(|conjunct| {
            let mut used: Vec<usize> = collect_columns(conjunct)
                .iter()
                .map(|column| column.index())
                .collect();
            used.sort_unstable();
            used.dedup();
            let on_dictionary = used
                .iter()
                .any(|&i| kept_as_dictionary(&columns[i], schema.field(i)));
            (conjunct, used, on_dictionary)
        })
        .collect();
    if !conjuncts.iter().any(|(_, _, on_dictionary)| *on_dictionary) {
        return live;
    }
    let last = conjuncts.len() - 1;
    let mut passing = live.unwrap_or_else(|| BooleanArray::new(BooleanBuffer::new_set(rows), None));
    for (at, (conjunct, used, on_dictionary)) in conjuncts.iter().enumerate() {
        if at == last && !on_dictionary {
            break;
        }
        if passing.true_count() == 0 {
            break;
        }
        let on_rows = || conjunct_on_rows(conjunct, schema, columns, used, &passing);
        let checked = match used.as_slice() {
            [only] if *on_dictionary => {
                dictionary_conjunct(conjunct, schema.field(*only), &columns[*only], &passing)
                    .or_else(|_| on_rows())
            }
            _ => on_rows(),
        };
        match checked {
            Ok(next) => passing = next,
            Err(_) => break,
        }
    }
    Some(passing)
}

/// Whether `column`, as kept, is a dictionary of a column the scan returns
/// as something else.
fn kept_as_dictionary(column: &ArrayRef, field: &Field) -> bool {
    matches!(column.data_type(), DataType::Dictionary(..))
        && !matches!(field.data_type(), DataType::Dictionary(..))
}

/// The rows of `passing` that pass `conjunct`, which reads only the column
/// `field`, kept as the dictionary `kept`: the conjunct is checked once on
/// each value the passing rows use (cast as the scan types the column), and
/// once on a null when a passing row has none, then taken for each row by its
/// key. Values only other rows use, deleted ones among them, are not checked.
fn dictionary_conjunct(
    conjunct: &Arc<dyn PhysicalExpr>,
    field: &Field,
    kept: &ArrayRef,
    passing: &BooleanArray,
) -> DfResult<BooleanArray> {
    let Some(dictionary) = kept.as_dictionary_opt::<Int16Type>() else {
        return Err(DataFusionError::Internal(format!(
            "column {} is kept as {}, not a dictionary of Int16 keys",
            field.name(),
            kept.data_type()
        )));
    };
    // The passing rows' keys, in order.
    let keys = filter(dictionary.keys(), passing)?;
    let keys = keys.as_primitive::<Int16Type>();
    // Each value a passing row uses, once, and where it sits among them.
    let mut slot: Vec<Option<usize>> = vec![None; dictionary.values().len()];
    let mut used: Vec<u32> = Vec::new();
    for value in keys.iter().flatten() {
        let value = value as usize;
        if slot[value].is_none() {
            slot[value] = Some(used.len());
            used.push(value as u32);
        }
    }
    let null_row = keys.null_count() > 0;
    let one = Arc::new(Schema::new(vec![field.clone().with_nullable(true)]));
    let expr = reassign_expr_columns(Arc::clone(conjunct), &one)?;
    let check = |values: ArrayRef| -> DfResult<BooleanArray> {
        let n = values.len();
        let batch = RecordBatch::try_new(Arc::clone(&one), vec![values])?;
        boolean(expr.evaluate(&batch)?.into_array(n)?)
    };
    let values = take(dictionary.values().as_ref(), &UInt32Array::from(used), None)?;
    let on_values = check(cast_with_options(
        &values,
        field.data_type(),
        &CAST_OPTIONS,
    )?)?;
    let on_null = match null_row {
        true => {
            let on_null = check(new_null_array(field.data_type(), 1))?;
            on_null.is_valid(0) && on_null.value(0)
        }
        false => false,
    };
    // Each dictionary value's answer (false for a value no passing row
    // uses), taken for each passing row by its key. A row with no key gets
    // the null's answer.
    let answers: BooleanArray = slot
        .iter()
        .map(|at| Some(at.is_some_and(|at| on_values.is_valid(at) && on_values.value(at))))
        .collect();
    let on_rows = boolean(take(&answers, keys, None)?)?;
    let on_rows = match (on_null, on_rows.nulls()) {
        (true, Some(present)) => BooleanArray::new(on_rows.values() | &!present.inner(), None),
        _ => on_rows,
    };
    in_place(passing, &on_rows)
}

/// The rows of `passing` that pass `conjunct`, which reads the columns
/// `used`, checked on those rows alone, a column kept as a dictionary among
/// them cast to the scan's type first.
fn conjunct_on_rows(
    conjunct: &Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
    columns: &[ArrayRef],
    used: &[usize],
    passing: &BooleanArray,
) -> DfResult<BooleanArray> {
    let projected = Arc::new(Schema::new(
        used.iter()
            .map(|&i| schema.field(i).clone())
            .collect::<Vec<_>>(),
    ));
    let arrays = used
        .iter()
        .map(|&i| {
            as_scanned(
                filter(columns[i].as_ref(), passing)?,
                schema.field(i).data_type(),
            )
        })
        .collect::<DfResult<Vec<_>>>()?;
    let n = passing.true_count();
    let batch = RecordBatch::try_new_with_options(
        Arc::clone(&projected),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(n)),
    )?;
    let expr = reassign_expr_columns(Arc::clone(conjunct), &projected)?;
    let checked = boolean(expr.evaluate(&batch)?.into_array(n)?)?;
    in_place(passing, &checked)
}

/// The rows of `passing` whose answer in `checked`, which holds one per
/// passing row in order, is true. A null answer does not pass. Never null.
fn in_place(passing: &BooleanArray, checked: &BooleanArray) -> DfResult<BooleanArray> {
    // Rows that don't pass come back null, as do null answers.
    let placed = boolean(scatter(passing, checked)?)?;
    Ok(match placed.null_count() {
        0 => placed,
        _ => prep_null_mask_filter(&placed),
    })
}

/// `array` as the booleans a predicate gives.
fn boolean(array: ArrayRef) -> DfResult<BooleanArray> {
    array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .cloned()
        .ok_or_else(|| DataFusionError::Internal(format!("a predicate gave {}", array.data_type())))
}

impl CachedScanExec {
    /// This scan, checking `predicate` (a `WHERE` over its output, as the
    /// filter above it checks it) on each row group that keeps a column it
    /// reads as a dictionary, before handing the row group's rows on: its
    /// conjuncts in order, on the live rows, each on the rows the ones before
    /// it kept ([`passing_rows`]). A conjunct on one kept dictionary is
    /// checked once per value the rows use rather than once per row, and only
    /// the rows that pass are cast to the strings the scan returns. The
    /// filter above still checks every row it gets, so the answer is the one
    /// it gives.
    pub(crate) fn with_predicate(&self, predicate: Arc<dyn PhysicalExpr>) -> Self {
        let context = &self.context;
        Self {
            context: Arc::new(ScanContext {
                schema: Arc::clone(&context.schema),
                manifest: Arc::clone(&context.manifest),
                op_stats: context.op_stats.clone(),
                cache: Arc::clone(&context.cache),
                schema_id: context.schema_id,
                dictionaries: context.dictionaries.clone(),
                predicate: Some(predicate),
                decoding: Mutex::default(),
            }),
            metrics: ExecutionPlanMetricsSet::new(),
            ..self.clone()
        }
    }

    /// Whether the scan checks a `WHERE` itself ([`Self::with_predicate`]).
    pub(crate) fn has_predicate(&self) -> bool {
        self.context.predicate.is_some()
    }
}

impl DisplayAs for CachedScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let columns: Vec<&str> = self
            .context
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        write!(
            f,
            "CachedScanExec: row_groups={}, pruned={}, partitions={}, projection=[{}]",
            self.units.len(),
            self.pruned,
            self.partitions,
            columns.join(", ")
        )?;
        if let Some(predicate) = &self.context.predicate {
            write!(f, ", predicate={predicate}")?;
        }
        if let Some(fetch) = self.fetch {
            write!(f, ", fetch={fetch}")?;
        }
        if let Some(order) = &self.id_order {
            let direction = if order.descending { "desc" } else { "asc" };
            write!(f, ", id_blocks={direction}")?;
        }
        Ok(())
    }
}

impl ExecutionPlan for CachedScanExec {
    fn name(&self) -> &'static str {
        "CachedScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    /// A fresh id block queue, so a plan run again (a recursive query's
    /// term, once per step) reads every block again instead of what the
    /// last run left in the queue.
    fn reset_state(self: Arc<Self>) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self {
            id_order: self.id_order.as_ref().map(|order| IdOrder {
                descending: order.descending,
                queue: Arc::default(),
            }),
            ..(*self).clone()
        }))
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DfResult<Arc<Statistics>> {
        Ok(match partition {
            None => Arc::clone(&self.statistics),
            Some(_) => Arc::new(Statistics::new_unknown(&self.context.schema)),
        })
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        Some(Arc::new(Self {
            fetch: limit,
            metrics: ExecutionPlanMetricsSet::new(),
            ..self.clone()
        }))
    }

    /// Keeps the filters a parent pushed down that tighten as the statement
    /// runs ([`LiveFilters`]: a top-k's, a hash join's, a `MIN` or `MAX`
    /// aggregate's), to skip row groups and drop rows by; every filter stays
    /// checked above.
    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> DfResult<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let filters: Vec<Arc<dyn PhysicalExpr>> = child_pushdown_result
            .parent_filters
            .iter()
            .map(|result| Arc::clone(&result.filter))
            .collect();
        let none = vec![PushedDown::No; filters.len()];
        let live: Vec<Arc<dyn PhysicalExpr>> = filters
            .into_iter()
            .filter(|filter| filter.downcast_ref::<DynamicFilterPhysicalExpr>().is_some())
            .collect();
        let propagation = FilterPushdownPropagation::with_parent_pushdown_result(none);
        if phase != FilterPushdownPhase::Post || live.is_empty() {
            return Ok(propagation);
        }
        Ok(propagation.with_updated_node(Arc::new(Self {
            live: Some(Arc::new(LiveFilters {
                filters: live,
                adapter: Arc::new(TableExprAdapterFactory::new(&self.context.manifest)),
            })),
            ..self.clone()
        })))
    }

    /// For a sort on one of the scan's columns, reads the row groups that
    /// statistics say start best first, so a top-k above settles its bound
    /// early and [`LiveFilters`] skip the rest; for a sort on `_id`, reads by
    /// id block ([`IdOrder`]). The order is only a reading order: the sort
    /// stays. DataFusion asks only through nodes that pass a sort on, which a
    /// `FilterExec` doesn't; under one, `IdOrderUnderTopK` asks for the id
    /// order instead.
    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> DfResult<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        let Some(first) = order.first() else {
            return Ok(SortOrderPushdownResult::Unsupported);
        };
        let Some(column) = first.expr.downcast_ref::<Column>() else {
            return Ok(SortOrderPushdownResult::Unsupported);
        };
        let descending = first.options.descending;
        let id_order =
            (column.name() == self.context.manifest.options.id_column).then(|| IdOrder {
                descending,
                queue: Arc::default(),
            });
        let units = self
            .context
            .schema
            .field_with_name(column.name())
            .ok()
            .and_then(|field| {
                best_first(
                    &self.units,
                    column.name(),
                    field.data_type(),
                    descending,
                    first.options.nulls_first,
                )
            });
        if units.is_none() && id_order.is_none() {
            return Ok(SortOrderPushdownResult::Unsupported);
        }
        Ok(SortOrderPushdownResult::Inexact {
            inner: Arc::new(Self {
                units: units.map_or_else(|| Arc::clone(&self.units), Into::into),
                reads_best_first: true,
                id_order,
                ..self.clone()
            }),
        })
    }

    fn execute(
        &self,
        partition: usize,
        task: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition >= self.partitions {
            return Err(DataFusionError::Internal(format!(
                "CachedScanExec has {} partitions; asked for {partition}",
                self.partitions
            )));
        }
        let partitions = self.partitions;
        let all = Arc::clone(&self.units);
        // By id block only under a filter that tightens: without one, no
        // block is skipped, and the row groups read in fewer, larger batches.
        let id_order = self.id_order.clone().filter(|_| self.live.is_some());
        let descending = id_order.as_ref().is_some_and(|order| order.descending);
        // The id-block read, built by the first partition to start and shared
        // by the rest through the queue's cell: the blocks it hands out, and
        // the bookkeeping that keeps each row group's shared decode alive until
        // its last block is read and bounds the bytes those decodes hold.
        let context_for_queue = Arc::clone(&self.context);
        let id_read_cell = id_order.as_ref().map(|order| Arc::clone(&order.queue));
        let units = stream::once(async move {
            let id_read = match &id_order {
                Some(order) => order
                    .queue
                    .get_or_init(|| async {
                        IdRead::new(
                            Arc::clone(&all),
                            order.descending,
                            Arc::clone(&context_for_queue),
                        )
                        .await
                        .map(Arc::new)
                    })
                    .await
                    .clone(),
                None => None,
            };
            match id_read {
                Some(id_read) => stream::unfold(id_read, |id_read| {
                    let next = id_read
                        .queue
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .take(&id_read.decoding);
                    future::ready(next.map(|unit| (unit, id_read)))
                })
                .boxed(),
                None => {
                    let mine: Vec<RowGroupUnit> = all
                        .iter()
                        .skip(partition)
                        .step_by(partitions)
                        .cloned()
                        .collect();
                    stream::iter(mine).boxed()
                }
            }
        })
        .flatten();
        let context = Arc::clone(&self.context);
        let output_rows = MetricBuilder::new(&self.metrics).output_rows(partition);
        let read_in_part =
            MetricBuilder::new(&self.metrics).counter(ROW_GROUPS_READ_IN_PART, partition);
        let rows_read = MetricBuilder::new(&self.metrics).counter(ROWS_READ, partition);
        let decoded_groups =
            MetricBuilder::new(&self.metrics).counter(ROW_GROUPS_DECODED, partition);
        let skipped = MetricBuilder::new(&self.metrics).counter(ROW_GROUPS_SKIPPED, partition);
        let dropped = MetricBuilder::new(&self.metrics).counter(ROWS_DROPPED, partition);
        // Rows are dropped per batch only when the scan reads best first for
        // a sort above it, which a top-k asks for and a hash join or an
        // aggregate does not; their filters still skip whole row groups but
        // dropping their rows here, where most pass, is work for nothing.
        let kept = self.reads_best_first.then(|| self.live.clone()).flatten();
        let mut row_gate = self
            .live
            .clone()
            .map(|live| RowGroupGate::new(live, Arc::clone(&self.context.schema)));
        let id_gate = self.live.clone().and_then(|live| {
            IdGate::new(
                live,
                Arc::clone(&self.context.schema),
                &self.context.manifest.options.id_column,
            )
        });
        // A row group goes out in batch-size slices, as the Parquet scan's
        // rows do: the filter above then works a batch at a time, and a
        // `LIMIT` met in the first slice leaves the rest of the row group.
        let batch_size = task.session_config().batch_size();
        // The shared id-block read, for the stages that count a block done:
        // the id gate when a block is ruled out, and the read when it is read.
        // Resolved from the cell the first partition filled.
        let gate_cell = id_read_cell.clone();
        let then_cell = id_read_cell;
        let batches = units
            // An id block is checked against the filters' bounds as it comes
            // up. Blocks come in order of their lowest id (highest,
            // descending), so the ids of every block still to come lie at or
            // above this one's lowest: once the filters rule out all of those,
            // the partition is done. A block ruled out here is read no further,
            // so it counts against its row group's block tally as a read would.
            .scan(id_gate, move |gate, unit| {
                let id_read = gate_cell
                    .as_ref()
                    .and_then(|cell| cell.get().cloned().flatten());
                let next = match (gate.as_mut(), unit.id_bounds) {
                    (Some(gate), Some((lo, hi))) => {
                        let group = (unit.file.uri, unit.row_group);
                        let rows = u64::from(unit.rows.end - unit.rows.start);
                        let rest = match descending {
                            false => (lo, i128::MAX),
                            true => (i128::MIN, hi),
                        };
                        if gate.rules_out(rest.0, rest.1, u64::MAX) {
                            if let Some(id_read) = &id_read {
                                id_read.decoding.done(group);
                            }
                            None
                        } else if gate.rules_out(lo, hi, rows) {
                            if let Some(id_read) = &id_read {
                                id_read.decoding.done(group);
                            }
                            Some(None)
                        } else {
                            Some(Some(unit))
                        }
                    }
                    _ => Some(Some(unit)),
                };
                future::ready(next)
            })
            .filter_map(future::ready)
            .filter(move |unit| {
                // Checked as each row group comes up, so a bound tightened by
                // the rows read so far skips it. An id block was checked by
                // its own bounds above.
                let skip = unit.id_bounds.is_none()
                    && row_gate.as_mut().is_some_and(|gate| gate.skip(unit));
                if skip {
                    skipped.add(1);
                }
                future::ready(!skip)
            })
            .then(move |unit| {
                let context = Arc::clone(&context);
                let read_in_part = read_in_part.clone();
                rows_read.add(unit.rows.len());
                let decoded_groups = decoded_groups.clone();
                let id_read = then_cell
                    .as_ref()
                    .and_then(|cell| cell.get().cloned().flatten());
                async move {
                    let group = (unit.file.uri, unit.row_group);
                    let block = unit.id_bounds.is_some();
                    let batch = context
                        .read(&unit, decoded_groups, &read_in_part, id_read.clone())
                        .await;
                    // The block is read: count it against its row group's tally
                    // once its decode has had its chance to be kept, so the
                    // shared entry leaves after the last block, not before.
                    if block && let Some(id_read) = &id_read {
                        id_read.decoding.done(group);
                    }
                    batch
                }
            })
            .map_ok(move |batch| stream::iter(sliced(batch, batch_size).into_iter().map(Ok)))
            .try_flatten()
            // Each slice as it goes out, so a bound the rows before it
            // tightened drops what can no longer reach the result.
            .and_then(move |batch| {
                future::ready(match &kept {
                    Some(live) => {
                        let rows = batch.num_rows();
                        live.keep(batch)
                            .inspect(|left| dropped.add(rows - left.num_rows()))
                    }
                    None => Ok(batch),
                })
            })
            .try_filter(|batch| future::ready(batch.num_rows() > 0))
            // A partition stops once it has the scan's fetch: the limit above
            // takes at most that many rows from any one of them.
            .scan(self.fetch.unwrap_or(usize::MAX), |remaining, batch| {
                future::ready(match batch {
                    Err(error) => Some(Err(error)),
                    Ok(_) if *remaining == 0 => None,
                    Ok(batch) => {
                        let rows = batch.num_rows().min(*remaining);
                        *remaining -= rows;
                        Some(Ok(batch.slice(0, rows)))
                    }
                })
            })
            .inspect_ok(move |batch| output_rows.add(batch.num_rows()));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.context.schema),
            batches,
        )))
    }
}

/// The scan schema's projection by `indices`, every column when `None`.
pub(crate) fn projected_schema(
    schema: &SchemaRef,
    indices: Option<&Vec<usize>>,
) -> DfResult<SchemaRef> {
    Ok(match indices {
        Some(indices) => Arc::new(schema.project(indices).map_err(QueryError::internal)?),
        None => Arc::clone(schema),
    })
}

/// Whether `schema` has a column for the cache to keep. A scan of no column
/// (`COUNT(*)`) has nothing to decode, so the Parquet scan serves it.
pub(crate) fn has_columns(schema: &Schema) -> bool {
    !schema.fields().is_empty()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use arrow::{compute::cast, util::pretty::pretty_format_batches};
    use arrow_array::{
        Array, Decimal128Array, DictionaryArray, Int16Array, Int64Array, LargeStringArray,
        StringArray,
    };
    use arrow_schema::{DataType, Field};
    use bytes::Bytes;
    use datafusion::{
        common::DFSchema,
        execution::context::ExecutionProps,
        logical_expr::{Expr, cast as cast_to},
        physical_expr::create_physical_expr,
        prelude::{col, lit},
    };
    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::builder::{FtsConfig, with_raw_id_sidecar_from_self},
        supertable::{Supertable, SupertableOptions, SupertableReader},
    };

    /// Superfiles the dictionary test's table is written as.
    const FILES: i64 = 3;
    /// Rows in each.
    const ROWS_PER_FILE: i64 = 2000;
    /// Rows per row group, so each file has several.
    const ROW_GROUP_ROWS: usize = 500;
    /// Partitions the table's statements run on: no more than its row groups, or a repartition
    /// spreads the scan and leaves the `WHERE` to the filter above it.
    const PARTITIONS: usize = 4;
    const _: () = assert!(PARTITIONS as i64 <= FILES * ROWS_PER_FILE / ROW_GROUP_ROWS as i64);
    /// The few values `region` takes.
    const REGIONS: [&str; 5] = ["af", "ap", "eu", "sa", "us"];
    /// Rows the delete removes: those with `seq` below this.
    const DELETED_BELOW: i64 = 300;
    /// Kept bytes: room for every column of every row group.
    const KEPT_BYTES: usize = 64 << 20;
    /// What `code` holds on a deleted row: no number, so a cast of it fails.
    const NOT_A_NUMBER: &str = "x";
    /// Every this many rows `code` is null.
    const NULL_EVERY: i64 = 7;
    /// The numbers `code` and `tag` take otherwise, as strings: few values.
    const CODES: i64 = 5;
    /// The region whose rows hold [`NOT_A_NUMBER`] in `tag`, live rows among
    /// them, so a cast of `tag` on them fails.
    const GUARDED_REGION: &str = "af";
    /// Statements over `region` (few values), `msg` (one per row) and `seq`.
    const STATEMENTS: &[&str] = &[
        "SELECT region, count(*) AS n FROM supertable WHERE seq < 4500 GROUP BY region ORDER BY region",
        "SELECT region, msg FROM supertable WHERE region = 'eu' ORDER BY seq LIMIT 5",
        "SELECT count(*) AS n FROM supertable WHERE region IN ('af', 'us') AND seq > 1000",
        "SELECT region, min(seq) AS lo, max(seq) AS hi FROM supertable GROUP BY region ORDER BY region DESC",
    ];

    /// A condition that fails on a deleted row's `code`, which holds no
    /// number: a scan that checked it on deleted rows would fail.
    const FAILS_ON_DELETED_ROWS: &str = "CAST(code AS INT) > 2";
    /// A guard ahead of a division by zero, after a condition on a kept
    /// dictionary. Without a cache the division sees only the rows both
    /// conditions before it keep; a scan that dropped the other regions'
    /// rows and left the guard to the filter would hand it a batch where the
    /// guard keeps more than a fifth of the rows, and the filter's `AND`
    /// would then divide every row by its `seq % 4`, zeros included.
    const GUARDS_A_DIVISION: &str = "region = 'eu' AND seq % 4 <> 0 AND 100 / (seq % 4) > 1";
    /// A condition that fails on live rows, a cast of the guarded region's
    /// `tag`, with or without a cache.
    const FAILS_ON_LIVE_ROWS: &str = "region = 'af' AND CAST(tag AS INT) > 2";
    /// Conditions over `code` and `tag`, each run grouped by `region` so the
    /// statement reads the scan rather than the files' statistics.
    const CONDITIONS: &[&str] = &[
        "code != '2'",
        "code NOT IN ('1', '3')",
        "code IS NULL",
        "code IS NOT NULL AND region LIKE 'a%'",
        "region LIKE '%u%' AND seq > 2500",
        "region NOT LIKE 'e%' AND code = '4'",
        "code IN ('1', '5') OR code IS NULL",
        FAILS_ON_DELETED_ROWS,
        "100 / seq >= 0 AND region = 'eu'",
        "region <> 'af' AND msg LIKE '%7'",
        GUARDS_A_DIVISION,
        FAILS_ON_LIVE_ROWS,
    ];

    fn region_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("msg", DataType::Utf8, false),
            Field::new("code", DataType::Utf8, true),
            Field::new("tag", DataType::Utf8, false),
        ]))
    }

    /// The table, keeping decoded columns in `kept` (none with `None`), its
    /// rows below [`DELETED_BELOW`] deleted, its statements on [`PARTITIONS`].
    fn region_table(dir: &TempDir, kept: Option<Arc<DecodedColumnCache>>) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut options = SupertableOptions::new(region_schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage);
        options.decoded_columns = kept;
        options.row_group_size = Some(ROW_GROUP_ROWS);
        let st = Supertable::create(options).expect("create");
        for file in 0..FILES {
            let seqs: Vec<i64> = (file * ROWS_PER_FILE..(file + 1) * ROWS_PER_FILE).collect();
            let regions: Vec<&str> = seqs
                .iter()
                .map(|s| REGIONS[(*s as usize * 7) % REGIONS.len()])
                .collect();
            let msgs: Vec<String> = seqs.iter().map(|s| format!("message {s}")).collect();
            let codes: Vec<Option<String>> = seqs
                .iter()
                .map(|&s| match s {
                    s if s % NULL_EVERY == 0 => None,
                    s if s < DELETED_BELOW => Some(NOT_A_NUMBER.to_string()),
                    s => Some(((s * 3) % CODES + 1).to_string()),
                })
                .collect();
            let tags: Vec<String> = seqs
                .iter()
                .zip(&regions)
                .map(|(&s, &region)| match region == GUARDED_REGION {
                    true => NOT_A_NUMBER.to_string(),
                    false => (s % CODES + 1).to_string(),
                })
                .collect();
            let batch = RecordBatch::try_new(
                region_schema(),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(StringArray::from(regions)),
                    Arc::new(StringArray::from(msgs)),
                    Arc::new(StringArray::from(codes)),
                    Arc::new(StringArray::from(tags)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete(col("seq").lt(lit(DELETED_BELOW)))
            .expect("delete");
        // Set on the session the table's readers share, which lasts until its next commit.
        let ctx = st
            .reader()
            .expect("reader")
            .sql_session_context()
            .expect("context");
        ctx.state_ref()
            .write()
            .config_mut()
            .options_mut()
            .execution
            .target_partitions = PARTITIONS;
        st
    }

    fn answer(st: &Supertable, sql: &str) -> String {
        outcome(st, sql).expect(sql)
    }

    /// `sql`'s answer on `st`, or the error it fails with.
    fn outcome(st: &Supertable, sql: &str) -> Result<String, String> {
        let batches = st
            .reader()
            .expect("reader")
            .query_sql(sql)
            .map_err(|error| error.to_string())?;
        Ok(pretty_format_batches(&batches).expect("format").to_string())
    }

    /// Whether the table `answer` prints has a row of exactly `cells`.
    fn has_row(answer: &str, cells: &[&str]) -> bool {
        answer.lines().any(|line| {
            let row: Vec<&str> = line
                .split('|')
                .map(str::trim)
                .filter(|cell| !cell.is_empty())
                .collect();
            row == cells
        })
    }

    /// The scan checks a `WHERE` on a column it keeps as a dictionary, and a
    /// statement over such columns answers as without a cache: with nulls,
    /// `!=`, `NOT IN`, `IS NULL` and `LIKE`; with a condition that fails on a
    /// deleted row, which the scan never checks; and with a guard ahead of a
    /// condition that fails on the rows the guard drops. A statement that
    /// fails without a cache fails with one.
    #[test]
    fn a_where_checked_on_kept_dictionaries_answers_as_without_a_cache() {
        let plain_dir = TempDir::new().expect("tempdir");
        let plain = region_table(&plain_dir, None);
        let kept_dir = TempDir::new().expect("tempdir");
        let kept = region_table(
            &kept_dir,
            Some(Arc::new(DecodedColumnCache::new(KEPT_BYTES))),
        );
        let grouped = |condition: &str| {
            format!(
                "SELECT region, count(*) AS n FROM supertable WHERE {condition} \
                 GROUP BY region ORDER BY region"
            )
        };
        for pass in ["decoded", "kept", "kept again"] {
            for condition in CONDITIONS {
                let sql = grouped(condition);
                match outcome(&plain, &sql) {
                    Ok(expected) => {
                        assert_eq!(outcome(&kept, &sql), Ok(expected), "{pass}: {sql}");
                    }
                    Err(error) => {
                        assert_eq!(*condition, FAILS_ON_LIVE_ROWS, "{sql}: {error}");
                        assert!(outcome(&kept, &sql).is_err(), "{pass}: {sql}");
                    }
                }
            }
            // The guarded division keeps the live eu rows whose `seq % 4` is
            // not 0 (100 divided by 1, 2 or 3 is over 1): three in four of
            // the 1,140 live eu rows.
            let guarded = answer(&kept, &grouped(GUARDS_A_DIVISION));
            assert!(has_row(&guarded, &["eu", "855"]), "{pass}: {guarded}");
            // `100 / seq` fails only on `seq = 0`, which is deleted: every
            // live row passes, with or without a cache.
            let all_live = "SELECT count(*) AS n FROM supertable WHERE 100 / seq >= 0";
            let live = (FILES * ROWS_PER_FILE - DELETED_BELOW).to_string();
            for st in [&plain, &kept] {
                let counted = answer(st, all_live);
                assert!(has_row(&counted, &[live.as_str()]), "{pass}: {counted}");
            }
        }
        let fails = grouped(FAILS_ON_LIVE_ROWS);
        assert!(outcome(&plain, &fails).is_err(), "{fails}");
        // The filter above the scan hands its `WHERE` to the scan.
        let plan = explained(
            &kept.reader().expect("reader"),
            "SELECT region, max(msg) AS m FROM supertable WHERE region = 'eu' GROUP BY region",
        );
        assert!(
            plan.lines()
                .any(|line| line.contains("CachedScanExec") && line.contains("predicate=")),
            "the scan checks the WHERE:\n{plan}"
        );
    }

    /// The values the kernel test's dictionary holds: [`NOT_A_NUMBER`] fails
    /// a cast to a number.
    const KERNEL_VALUES: [&str; 4] = ["1", "2", "3", NOT_A_NUMBER];
    /// The kernel test's keys, before the slice; `None` is a row with no key.
    const KERNEL_KEYS: [Option<i16>; 12] = [
        Some(3),
        Some(0),
        Some(1),
        None,
        Some(0),
        Some(2),
        None,
        Some(3),
        Some(1),
        Some(2),
        Some(0),
        Some(1),
    ];
    /// The kernel test's other column, before the slice: a number or null.
    const KERNEL_NUMBERS: [Option<i64>; 12] = [
        Some(9),
        None,
        Some(5),
        Some(1),
        None,
        Some(7),
        Some(3),
        Some(8),
        None,
        Some(6),
        Some(2),
        Some(4),
    ];
    /// Where the kernel test's slice starts, so its keys and nulls start
    /// mid-buffer.
    const KERNEL_OFFSET: usize = 2;
    /// Rows in the kernel test's slice.
    const KERNEL_ROWS: usize = 10;
    /// The rows still passing in the kernel test's partial case. The one row
    /// holding [`NOT_A_NUMBER`] (5) does not pass, nor does one with no key (4).
    const KERNEL_PASSING: [bool; KERNEL_ROWS] = [
        true, true, false, true, false, false, true, true, true, false,
    ];
    /// The number `n > ` compares with in the kernel test.
    const KERNEL_NUMBER_BOUND: i64 = 4;
    /// The number a cast `name` is compared with in the kernel test.
    const KERNEL_CAST_BOUND: i32 = 1;

    /// `name` as a sliced dictionary with null keys, and `n` as numbers with
    /// nulls, with the schema the scan reads them as.
    fn kernel_columns() -> (SchemaRef, Vec<ArrayRef>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("n", DataType::Int64, true),
        ]));
        let dictionary = DictionaryArray::<Int16Type>::try_new(
            Int16Array::from(KERNEL_KEYS.to_vec()),
            Arc::new(StringArray::from(KERNEL_VALUES.to_vec())),
        )
        .expect("dictionary");
        let names: ArrayRef = Arc::new(dictionary.slice(KERNEL_OFFSET, KERNEL_ROWS));
        let numbers: ArrayRef =
            Arc::new(Int64Array::from(KERNEL_NUMBERS.to_vec()).slice(KERNEL_OFFSET, KERNEL_ROWS));
        (schema, vec![names, numbers])
    }

    /// The rows of `passing` that pass `conjunct`, checked one row at a time
    /// on the columns as the scan reads them: a null answer does not pass,
    /// and a row that does not pass is never checked.
    fn passing_row_by_row(
        conjunct: &Arc<dyn PhysicalExpr>,
        schema: &SchemaRef,
        columns: &[ArrayRef],
        passing: &BooleanArray,
    ) -> DfResult<Vec<bool>> {
        let scanned = columns
            .iter()
            .zip(schema.fields())
            .map(|(column, field)| cast(column, field.data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        (0..passing.len())
            .map(|row| {
                if !passing.value(row) {
                    return Ok(false);
                }
                let one = scanned.iter().map(|column| column.slice(row, 1)).collect();
                let batch = RecordBatch::try_new(Arc::clone(schema), one)?;
                let answer = boolean(conjunct.evaluate(&batch)?.into_array(1)?)?;
                Ok(answer.is_valid(0) && answer.value(0))
            })
            .collect()
    }

    /// `dictionary_conjunct` and `conjunct_on_rows` give the rows a check of
    /// each passing row on its own gives, on a sliced dictionary with null
    /// keys: with every row passing (where a null answer must not pass) and
    /// with some, where a value only the other rows use, which fails a cast,
    /// is never checked. A conjunct that fails on a passing row fails both.
    #[test]
    fn conjunct_kernels_answer_as_a_check_of_each_passing_row() {
        let (schema, columns) = kernel_columns();
        let df_schema = DFSchema::try_from(Arc::clone(&schema)).expect("schema");
        let conjuncts: Vec<Expr> = vec![
            col("name").eq(lit(KERNEL_VALUES[1])),
            col("name").not_eq(lit(KERNEL_VALUES[1])),
            col("name").is_null(),
            col("name").is_not_null(),
            cast_to(col("name"), DataType::Int32).gt(lit(KERNEL_CAST_BOUND)),
            col("n").gt(lit(KERNEL_NUMBER_BOUND)),
        ];
        let every_row = BooleanArray::new(BooleanBuffer::new_set(KERNEL_ROWS), None);
        let some_rows = BooleanArray::from(KERNEL_PASSING.to_vec());
        for conjunct in &conjuncts {
            let expr =
                create_physical_expr(conjunct, &df_schema, &ExecutionProps::new()).expect("expr");
            let used: Vec<usize> = collect_columns(&expr)
                .iter()
                .map(|column| column.index())
                .collect();
            for passing in [&every_row, &some_rows] {
                let expected = passing_row_by_row(&expr, &schema, &columns, passing);
                let case = format!("{conjunct} on {passing:?}");
                let on_rows = conjunct_on_rows(&expr, &schema, &columns, &used, passing);
                let mut answers = vec![on_rows];
                if let [only] = used.as_slice()
                    && kept_as_dictionary(&columns[*only], schema.field(*only))
                {
                    answers.push(dictionary_conjunct(
                        &expr,
                        schema.field(*only),
                        &columns[*only],
                        passing,
                    ));
                }
                for answer in answers {
                    match &expected {
                        Ok(expected) => {
                            let answer = answer.expect(&case);
                            assert_eq!(answer.null_count(), 0, "{case}");
                            assert_eq!(
                                &answer.iter().flatten().collect::<Vec<_>>(),
                                expected,
                                "{case}"
                            );
                        }
                        Err(_) => assert!(answer.is_err(), "{case}"),
                    }
                }
            }
        }
    }

    /// A string column of few values is kept as a dictionary of `Int16`
    /// keys, and statements over it answer as without a cache: on the run
    /// that decodes and keeps it, and on the ones that read it kept.
    #[test]
    fn a_column_of_few_strings_is_kept_as_a_dictionary_and_answers_the_same() {
        let plain_dir = TempDir::new().expect("tempdir");
        let plain = region_table(&plain_dir, None);
        let kept_dir = TempDir::new().expect("tempdir");
        let cache = Arc::new(DecodedColumnCache::new(KEPT_BYTES));
        let kept = region_table(&kept_dir, Some(Arc::clone(&cache)));
        for pass in ["decoded", "kept", "kept again"] {
            for sql in STATEMENTS {
                assert_eq!(answer(&kept, sql), answer(&plain, sql), "{pass}: {sql}");
            }
        }
        let reader = kept.reader().expect("reader");
        let manifest = reader.manifest();
        let schema_id = manifest.table_schema().schema_id();
        let dictionary = DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8));
        for entry in manifest.superfiles.iter() {
            let region = cache
                .get(&ColumnKey {
                    uri: entry.uri,
                    schema_id,
                    column: "region".to_string(),
                    row_group: 0,
                })
                .expect("region kept");
            assert_eq!(region.data_type(), &dictionary, "{:?}", entry.uri);
            // A column of one value per row is kept as the scan reads it.
            let msg = cache
                .get(&ColumnKey {
                    uri: entry.uri,
                    schema_id,
                    column: "msg".to_string(),
                    row_group: 0,
                })
                .expect("msg kept");
            assert_eq!(msg.data_type(), &DataType::Utf8View, "{:?}", entry.uri);
        }
    }

    /// Rows in the batch the slicing test cuts.
    const ROWS: i64 = 10;
    /// Rows per slice in the slicing test: not a divisor of [`ROWS`].
    const SLICE: usize = 4;

    #[test]
    fn a_batch_is_cut_into_slices_of_the_batch_size_in_order() {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        let values: ArrayRef = Arc::new(Int64Array::from((0..ROWS).collect::<Vec<_>>()));
        let batch = RecordBatch::try_new(schema, vec![values]).expect("batch");
        let slices = sliced(batch, SLICE);
        let sizes: Vec<usize> = slices.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(sizes, vec![4, 4, 2]);
        let back: Vec<i64> = slices
            .iter()
            .flat_map(|s| {
                let n = s
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("ints")
                    .clone();
                (0..n.len()).map(move |i| n.value(i))
            })
            .collect();
        assert_eq!(back, (0..ROWS).collect::<Vec<_>>());
        let empty = RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
            "n",
            DataType::Int64,
            false,
        )])));
        assert!(sliced(empty, SLICE).is_empty());
    }

    /// Rows in each superfile of the narrowed table, one row group each.
    const NARROWED_ROWS: i64 = 5_000;
    /// Rows of the first superfile holding `x`: few enough for the index to
    /// select them, and far fewer than half the row group.
    const SPARSE: i64 = 100;
    /// Rows of the second superfile holding `x`: more than half of it, so
    /// the index does not select them and the superfile is read whole.
    const DENSE: i64 = 3_000;
    /// Rows deleted: those whose `seq` leaves [`DELETED_REMAINDER`] divided
    /// by this, some of each superfile's `x` rows among them.
    const DELETED_EVERY: i64 = 10;
    /// See [`DELETED_EVERY`].
    const DELETED_REMAINDER: i64 = 3;
    /// Kept bytes: room for every column of the narrowed table.
    const NARROWED_KEPT_BYTES: usize = 64 << 20;
    /// Rows the `ORDER BY ... LIMIT` statements take.
    const TOP_ROWS: usize = 10;

    /// A scan the index narrows in the first superfile only.
    const PARTLY_NARROWED: &str = "SELECT seq FROM supertable WHERE k = 'x' ORDER BY seq";
    /// A scan the index narrows in both superfiles, to one row each.
    const FULLY_NARROWED: &str =
        "SELECT seq FROM supertable WHERE k IN ('v200', 'v8200') ORDER BY seq";

    fn narrowed_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("k", DataType::LargeUtf8, false),
            Field::new("pad", DataType::LargeUtf8, false),
        ]))
    }

    /// The `seq` of row `i` of superfile `file`, and its `k`.
    fn narrowed_row(file: i64, i: i64) -> (i64, String) {
        let seq = file * NARROWED_ROWS + i;
        let x = match file {
            0 => i < SPARSE,
            _ => i < DENSE,
        };
        (seq, if x { "x".to_owned() } else { format!("v{seq}") })
    }

    /// Every live `x` row's `seq`, ascending.
    fn live_x() -> Vec<i64> {
        (0..2)
            .flat_map(|file| (0..NARROWED_ROWS).map(move |i| narrowed_row(file, i)))
            .filter(|(seq, k)| k == "x" && seq % DELETED_EVERY != DELETED_REMAINDER)
            .map(|(seq, _)| seq)
            .collect()
    }

    /// Two superfiles of [`NARROWED_ROWS`] rows, `k` indexed, its rows
    /// deleted as [`DELETED_EVERY`] says, keeping decoded columns in `kept`
    /// when given.
    fn narrowed_table(dir: &TempDir, kept: Option<Arc<DecodedColumnCache>>) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut options =
            SupertableOptions::new(narrowed_schema(), vec![FtsConfig::new("k")], vec![])
                .expect("options")
                .with_storage(storage);
        options.decoded_columns = kept;
        options.row_group_size = Some(NARROWED_ROWS as usize);
        let st = Supertable::create(options).expect("create");
        for file in 0..2 {
            let (seqs, ks): (Vec<i64>, Vec<String>) =
                (0..NARROWED_ROWS).map(|i| narrowed_row(file, i)).unzip();
            let pads: Vec<String> = seqs.iter().map(|s| format!("{s:0>48}")).collect();
            let batch = RecordBatch::try_new(
                narrowed_schema(),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(LargeStringArray::from(ks)),
                    Arc::new(LargeStringArray::from(pads)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete((col("seq") % lit(DELETED_EVERY)).eq(lit(DELETED_REMAINDER)))
            .expect("delete");
        st
    }

    /// The `seq` column of `sql`'s rows, in order.
    fn seqs_of(reader: &SupertableReader, sql: &str) -> Vec<i64> {
        reader
            .query_sql(sql)
            .unwrap_or_else(|error| panic!("{sql}: {error}"))
            .iter()
            .flat_map(|b| {
                let seqs = b
                    .column_by_name("seq")
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .expect("seq")
                    .clone();
                (0..seqs.len()).map(move |i| seqs.value(i))
            })
            .collect()
    }

    /// The `_id` column of `sql`'s rows, in order.
    fn ids_of(reader: &SupertableReader, sql: &str) -> Vec<i128> {
        reader
            .query_sql(sql)
            .unwrap_or_else(|error| panic!("{sql}: {error}"))
            .iter()
            .flat_map(|b| {
                let ids = b
                    .column_by_name("_id")
                    .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
                    .expect("_id")
                    .clone();
                (0..ids.len()).map(move |i| ids.value(i))
            })
            .collect()
    }

    /// The `EXPLAIN` of `sql` on `reader`, every plan line joined.
    fn explained(reader: &SupertableReader, sql: &str) -> String {
        reader
            .query_sql(&format!("EXPLAIN {sql}"))
            .expect("explain")
            .iter()
            .flat_map(|b| {
                let plans = b
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("plan column")
                    .clone();
                (0..plans.len()).map(move |i| plans.value(i).to_owned())
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// How many of `reader`'s superfiles have `column` of their first row
    /// group, the only one each holds, kept in `kept`.
    fn superfiles_keeping(
        reader: &SupertableReader,
        kept: &DecodedColumnCache,
        column: &str,
    ) -> usize {
        let manifest = reader.manifest();
        let schema_id = manifest.table_schema().schema_id();
        manifest
            .superfiles
            .iter()
            .filter(|entry| {
                kept.get(&ColumnKey {
                    uri: entry.uri,
                    schema_id,
                    column: column.to_owned(),
                    row_group: 0,
                })
                .is_some()
            })
            .count()
    }

    /// A scan the index narrows in some superfiles only reads them through
    /// kept columns, and answers as the same table keeping no columns does:
    /// the plain scan and `ORDER BY ... LIMIT` both ways (the path that can
    /// hand back the Parquet scan), on the first pass and the kept ones. The
    /// narrowed row group's few rows are read from its pages and kept only
    /// on its second read, so the first statement keeps the columns of the
    /// superfile read whole alone.
    #[test]
    fn a_partly_narrowed_scan_answers_as_the_table_without_kept_columns() {
        let kept = Arc::new(DecodedColumnCache::new(NARROWED_KEPT_BYTES));
        let (with_dir, without_dir) = (
            TempDir::new().expect("tempdir"),
            TempDir::new().expect("tempdir"),
        );
        let with = narrowed_table(&with_dir, Some(Arc::clone(&kept)));
        let without = narrowed_table(&without_dir, None);
        let (with, without) = (
            with.reader().expect("reader"),
            without.reader().expect("reader"),
        );
        let plan = explained(&with, PARTLY_NARROWED);
        assert!(plan.contains("CachedScanExec"), "{plan}");
        let plan = explained(&without, PARTLY_NARROWED);
        assert!(!plan.contains("CachedScanExec"), "{plan}");

        // First read: `k` is kept for the superfile read whole only. (The
        // delete that built the table may have kept `seq` already.)
        assert_eq!(superfiles_keeping(&with, &kept, "k"), 0);
        assert_eq!(seqs_of(&with, PARTLY_NARROWED), live_x());
        assert_eq!(
            superfiles_keeping(&with, &kept, "k"),
            1,
            "the narrowed row group's first read keeps nothing"
        );
        // Second read of the narrowed row group: its columns are kept too.
        assert_eq!(seqs_of(&with, PARTLY_NARROWED), live_x());
        assert_eq!(
            superfiles_keeping(&with, &kept, "k"),
            2,
            "the second read keeps the narrowed row group"
        );

        let statements = [
            PARTLY_NARROWED.to_owned(),
            format!(
                "SELECT seq, pad FROM supertable WHERE k = 'x' ORDER BY seq DESC LIMIT {TOP_ROWS}"
            ),
            format!(
                "SELECT seq, k, pad FROM supertable WHERE k = 'x' ORDER BY seq LIMIT {TOP_ROWS}"
            ),
        ];
        let mut descending = live_x();
        descending.reverse();
        let wanted = [
            live_x(),
            descending.into_iter().take(TOP_ROWS).collect::<Vec<_>>(),
            live_x().into_iter().take(TOP_ROWS).collect::<Vec<_>>(),
        ];
        for pass in ["first", "kept", "kept again"] {
            for (sql, want) in statements.iter().zip(&wanted) {
                let got = seqs_of(&with, sql);
                assert_eq!(got, seqs_of(&without, sql), "{pass}: {sql}");
                assert_eq!(&got, want, "{pass}: {sql}");
            }
        }
    }

    /// A narrowed row group none of whose rows were read in part before
    /// hands over the columns kept for it and reads the others for its
    /// selected rows from its pages, keeping them only for the row group read
    /// whole; and answers as the table keeping no columns does.
    #[test]
    fn a_narrowed_row_group_takes_kept_columns_and_reads_the_rest_from_its_pages() {
        let kept = Arc::new(DecodedColumnCache::new(NARROWED_KEPT_BYTES));
        let (with_dir, without_dir) = (
            TempDir::new().expect("tempdir"),
            TempDir::new().expect("tempdir"),
        );
        let with = narrowed_table(&with_dir, Some(Arc::clone(&kept)));
        let without = narrowed_table(&without_dir, None);
        let (with, without) = (
            with.reader().expect("reader"),
            without.reader().expect("reader"),
        );
        // A scan narrowed nowhere keeps `seq` of both row groups.
        let every = "SELECT seq FROM supertable ORDER BY seq";
        assert_eq!(seqs_of(&with, every), seqs_of(&without, every));
        assert_eq!(superfiles_keeping(&with, &kept, "seq"), 2);
        assert_eq!(superfiles_keeping(&with, &kept, "k"), 0);
        // `k` is kept for the row group read whole only; the narrowed one's
        // `seq` comes from the kept column, its `k` from its pages.
        let got = seqs_of(&with, PARTLY_NARROWED);
        assert_eq!(got, seqs_of(&without, PARTLY_NARROWED));
        assert_eq!(got, live_x());
        assert_eq!(
            superfiles_keeping(&with, &kept, "k"),
            1,
            "k of the row group read whole only"
        );
    }

    /// A scan the index narrows in every superfile reads their selected
    /// pages through the Parquet scan, keeping none of its columns, and answers as
    /// the table keeping no columns does.
    #[test]
    fn a_scan_narrowed_in_every_superfile_reads_the_files() {
        let kept = Arc::new(DecodedColumnCache::new(NARROWED_KEPT_BYTES));
        let (with_dir, without_dir) = (
            TempDir::new().expect("tempdir"),
            TempDir::new().expect("tempdir"),
        );
        let with = narrowed_table(&with_dir, Some(Arc::clone(&kept)));
        let without = narrowed_table(&without_dir, None);
        let (with, without) = (
            with.reader().expect("reader"),
            without.reader().expect("reader"),
        );
        let plan = explained(&with, FULLY_NARROWED);
        assert!(!plan.contains("CachedScanExec"), "{plan}");
        assert!(plan.contains("DataSourceExec"), "{plan}");
        for pass in ["first", "again"] {
            let got = seqs_of(&with, FULLY_NARROWED);
            assert_eq!(got, seqs_of(&without, FULLY_NARROWED), "{pass}");
            assert_eq!(got, vec![200, 8200], "{pass}");
        }
        assert_eq!(superfiles_keeping(&with, &kept, "k"), 0, "k is not kept");
    }

    /// The `id_blocks=asc` the rule prints is a reading order the scan asks
    /// its files for, not a promise every file can give it. A superfile a
    /// writer left before the packed layout carries its ids raw, with no
    /// blocks to read in order, so the scan falls back to reading it whole
    /// and sorting. This rewrites the table's superfiles to that older raw
    /// form and checks the plan still engages the rule (`id_blocks=asc`)
    /// while an `ORDER BY _id ... LIMIT` answers exactly as the full sort
    /// truncated — the fallback is correct, not merely reachable.
    #[test]
    fn a_scan_without_packed_id_blocks_answers_as_a_full_sort() {
        let kept = Arc::new(DecodedColumnCache::new(NARROWED_KEPT_BYTES));
        let dir = TempDir::new().expect("tempdir");
        let st = narrowed_table(&dir, Some(kept));
        let reader = st.reader().expect("reader");
        let store = Arc::clone(&reader.manifest().options.store);

        // Replace every superfile's resident bytes with its raw-sidecar form,
        // the layout with no id blocks. The on-disk bytes carry a manifest
        // layout hint the storage fallback would read a stale footer through,
        // so the swap stays in the in-memory tier the live handle serves
        // from. `insert` keeps the first bytes for a uri, so remove first.
        let mut rewritten = 0;
        for entry in &reader.manifest().superfiles {
            let packed = Bytes::from(
                std::fs::read(dir.path().join(entry.uri.storage_path())).expect("read superfile"),
            );
            if !SuperfileReader::open(packed.clone())
                .expect("open superfile")
                .id_sidecar_is_packed()
            {
                continue;
            }
            let raw = with_raw_id_sidecar_from_self(&packed);
            assert!(
                !SuperfileReader::open(raw.clone())
                    .expect("open rewritten superfile")
                    .id_sidecar_is_packed(),
                "the rewrite must drop the packed layout",
            );
            store.remove(&entry.uri);
            store.insert(entry.uri, raw).expect("insert raw superfile");
            rewritten += 1;
        }
        assert!(rewritten >= 1, "at least one packed superfile to rewrite");

        for (keyword, label) in [("ASC", "asc"), ("DESC", "desc")] {
            let sorted =
                format!("SELECT _id FROM supertable WHERE seq % 7 = 0 ORDER BY _id {keyword}");
            let topk = format!("{sorted} LIMIT {TOP_ROWS}");
            let plan = explained(&reader, &topk);
            assert!(plan.contains("CachedScanExec"), "{keyword}: {plan}");
            assert!(
                plan.contains(&format!("id_blocks={label}")),
                "{keyword}: {plan}"
            );
            let full = ids_of(&reader, &sorted);
            assert!(full.len() > TOP_ROWS, "{keyword}: {} rows", full.len());
            let want: Vec<i128> = full.into_iter().take(TOP_ROWS).collect();
            for pass in ["first", "again"] {
                assert_eq!(ids_of(&reader, &topk), want, "{pass} {keyword}: {topk}");
            }
        }
    }

    /// A top-k on `_id` reads its files by id block in that order and the
    /// answer is the full sort truncated — on packed files that read by id
    /// block, not the raw-sidecar fallback. Descending reads by its own block
    /// order, merge key and stop range, so it is checked beside ascending
    /// rather than assumed from it: a `k` of 1, 10 and 500, and a `k` past
    /// every matching row so every block is read and the merge runs the whole
    /// way down, not only at the top.
    #[test]
    fn a_top_k_on_id_reads_by_id_block_and_answers_as_the_full_sort() {
        let kept = Arc::new(DecodedColumnCache::new(NARROWED_KEPT_BYTES));
        let dir = TempDir::new().expect("tempdir");
        let st = narrowed_table(&dir, Some(kept));
        let reader = st.reader().expect("reader");

        for (keyword, label) in [("ASC", "asc"), ("DESC", "desc")] {
            let sorted =
                format!("SELECT _id FROM supertable WHERE seq % 7 = 0 ORDER BY _id {keyword}");
            let full = ids_of(&reader, &sorted);
            assert!(full.len() > 500, "{keyword}: {} rows", full.len());
            for &k in &[1usize, 10, 500, full.len() + 50] {
                let topk = format!("{sorted} LIMIT {k}");
                let plan = explained(&reader, &topk);
                assert!(plan.contains("CachedScanExec"), "{keyword} k={k}: {plan}");
                assert!(
                    plan.contains(&format!("id_blocks={label}")),
                    "{keyword} k={k}: {plan}"
                );
                let want: Vec<i128> = full.iter().copied().take(k).collect();
                for pass in ["first", "again"] {
                    assert_eq!(
                        ids_of(&reader, &topk),
                        want,
                        "{pass} {keyword} k={k}: {topk}"
                    );
                }
            }
        }
    }

    /// A recursive query runs its term once per step, so the id-block read
    /// runs again each step and each run needs a fresh queue: a run reading
    /// from the one the last step left drained would return nothing and the
    /// step would drop out. This runs the read twice — the anchor, then the
    /// recursive term joined to the ids the anchor found — and checks both
    /// steps return that same top-k, so a queue that was not reset fails here.
    #[test]
    fn a_recursive_query_reads_by_id_block_fresh_each_step() {
        let kept = Arc::new(DecodedColumnCache::new(NARROWED_KEPT_BYTES));
        let dir = TempDir::new().expect("tempdir");
        let st = narrowed_table(&dir, Some(kept));
        let reader = st.reader().expect("reader");

        let top = format!(
            "SELECT _id AS id FROM supertable WHERE seq % 7 = 0 ORDER BY _id LIMIT {TOP_ROWS}"
        );
        let sql = format!(
            "WITH RECURSIVE r AS ( \
                 SELECT 1 AS step, a.id FROM ({top}) a \
                 UNION ALL \
                 SELECT r.step + 1, s.id FROM r JOIN ({top}) s ON r.id = s.id \
                  WHERE r.step < 2) \
             SELECT step, id FROM r ORDER BY step, id"
        );
        let rows = reader.query_sql(&sql).expect("recursive CTE");

        let mut want: Vec<i128> = ids_of(
            &reader,
            &format!("SELECT _id FROM supertable WHERE seq % 7 = 0 ORDER BY _id LIMIT {TOP_ROWS}"),
        );
        want.sort_unstable();
        assert_eq!(
            want.len(),
            TOP_ROWS,
            "the id-block top-k has {TOP_ROWS} rows"
        );

        let mut by_step: BTreeMap<i64, Vec<i128>> = BTreeMap::new();
        for batch in &rows {
            let steps = batch
                .column_by_name("step")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                .expect("step");
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
                .expect("id");
            for i in 0..steps.len() {
                by_step
                    .entry(steps.value(i))
                    .or_default()
                    .push(ids.value(i));
            }
        }
        for step in [1i64, 2] {
            let mut got = by_step.get(&step).cloned().unwrap_or_default();
            got.sort_unstable();
            assert_eq!(
                got, want,
                "step {step} must read the same id-block top-k afresh"
            );
        }
    }
}
