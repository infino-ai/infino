// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared vocabulary for the tracing spans the engine emits.
//!
//! The same functions run on behalf of very different callers:
//! `ManifestSnapshot::load` can serve a foreground query, a commit, an
//! explicit `optimize()`, or a detached background sweep, and it can run
//! against either the user table or the derived vector-index table. A
//! span name alone can't tell those apart, so each operation's root span
//! carries two low-cardinality tags:
//!
//! * [`OpOrigin`] — what kind of operation is driving the work.
//! * [`TableRole`] — which of the two tables the work is touching.
//!
//! Note these are recorded on the root, not repeated on every descendant:
//! `tracing` fields do not inherit. A descendant is attributed by walking
//! its parent chain, which is what the `fmt` subscriber prints as the
//! span stack. The one root without a `role` is the connection-level SQL
//! entry, which has no table handle yet.
//!
//! Both render as `&'static str`, so recording one is a pointer copy and
//! never allocates. The field names are spelled `origin` and `role` at
//! each span site: `tracing`'s macros take field names as literal
//! tokens, so a shared constant can't stand in for them.
//!
//! What is deliberately *not* here is any notion of the calling
//! application's own roles. A caller that wants its own label installs a
//! span before calling in; because the public API is synchronous, that
//! span is the ambient parent and `#[instrument]` picks it up with no
//! engine-side code. The engine's job is only to never orphan the chain
//! — see the thread hand-off helpers in `runtime_bridge`.

use std::sync::Arc;
#[cfg(feature = "detailed-tracing")]
use std::{any::Any, sync::Once};

use arrow_array::RecordBatch;
#[cfg(feature = "detailed-tracing")]
use datafusion::common::runtime::{JoinSetTracer, set_join_set_tracer};
#[cfg(feature = "detailed-tracing")]
use futures::{FutureExt, future::BoxFuture};
#[cfg(feature = "detailed-tracing")]
use tracing::Instrument;
use tracing::{Span, field::Value};

use crate::runtime_metrics::{
    io::{UsageMeter, UsageSnapshot},
    op_stats::OpStatsCollector,
};

/// What kind of operation a span's work is being done for.
///
/// Recorded on the root of each operation, so a shared helper's span can
/// be attributed to the caller that triggered it by walking up to the
/// root. The distinction that motivates the enum is
/// `Optimize` vs `Maintenance`: both run the identical compaction code,
/// but one blocks a caller and one does not.
///
/// Only `Maintenance` is constructed without `detailed-tracing`: the
/// detached background spans are always compiled (they cost a disabled
/// callsite check once per cold fetch), while the per-operation spans
/// that carry the other variants are behind the feature.
#[cfg_attr(not(feature = "detailed-tracing"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpOrigin {
    /// A read: search or SQL, driven by a caller waiting on the result.
    Query,
    /// A write: append, update, delete, or the commit that publishes it.
    Ingest,
    /// An explicit `optimize()` call. Runs the same compaction and sweep
    /// code as [`Self::Maintenance`], but synchronously, with a caller
    /// blocked on it — so its latency is user-visible and its cost
    /// belongs to the caller.
    Optimize,
    /// Detached background work: compaction, gc, hidden-index drain, and
    /// the disk cache's background fills. Nobody is waiting on it, but it
    /// competes for the same pools as the foreground.
    Maintenance,
    /// Connect, create, or open — including the open-time recovery and gc
    /// sweeps that run before a handle is returned.
    Open,
}

impl OpOrigin {
    /// The span-field rendering. `&'static str` so recording is free.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Ingest => "ingest",
            Self::Optimize => "optimize",
            Self::Maintenance => "maintenance",
            Self::Open => "open",
        }
    }
}

/// Which table a span's work is touching.
///
/// A table with vector columns owns a second, derived supertable holding
/// the cell-ordered vector index. It runs the same code as the user
/// table, so without this tag its spans are indistinguishable from the
/// user table's — two `manifest.load` spans per query, same name,
/// different table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TableRole {
    /// The user's own table: the append-only, time-ordered rows.
    User,
    /// The derived, cell-ordered vector index that accelerates vector
    /// search over [`Self::User`].
    VectorIndex,
}

impl TableRole {
    /// The span-field rendering. `&'static str` so recording is free.
    /// Only read by the span sites, which are behind `detailed-tracing`.
    #[cfg_attr(not(feature = "detailed-tracing"), allow(dead_code))]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::VectorIndex => "vector_index",
        }
    }
}

/// Turn `span` into the root of a *detached* unit of work that was
/// merely triggered by the current span, rather than awaited by it.
///
/// The distinction matters for timing. A fire-and-forget task that
/// inherits the triggering span as its parent keeps that span alive
/// until the task finishes, so the span's recorded duration absorbs
/// background work the caller never waited for — a query that returned
/// in 5 ms reports the seconds its background cache fill went on to
/// take. `follows_from` records the same causal link without the
/// parent-child timing relationship, so both durations stay honest.
///
/// Use it at every `tokio::spawn` whose `JoinHandle` is dropped. Awaited
/// fan-out is the opposite case and should keep `in_current_span`: the
/// caller really is waiting, so the time really is the caller's.
pub(crate) fn detached(span: Span) -> Span {
    span.follows_from(Span::current());
    span
}

/// Create an `info` span, or nothing at all without `detailed-tracing`.
///
/// The inline counterpart to `#[cfg_attr(feature = "detailed-tracing",
/// tracing::instrument(...))]`, for work that is a region inside a
/// function rather than a whole function. Feature off, this expands to
/// [`Span::none()`], and the field expressions are not compiled at all
/// — so a field naming a variable that only exists for the span will
/// fail the `detailed-tracing` build while the default build stays
/// green. That is what `make check`'s `cargo check --features
/// metering,detailed-tracing` line is there to catch.
///
/// Entering a `Span::none()` is a genuine no-op: it never touches the
/// dispatcher's span stack, so spans created inside the guard's scope
/// keep the ambient parent rather than being orphaned. That makes the
/// guard safe to hold across arbitrary nested work in either build.
///
/// Fields are `name = value` pairs only. `tracing`'s own sigils (`?x`,
/// `%x`) and dotted field names are not expressible here: for the
/// sigils use `tracing::field::debug(&x)` / `display(&x)`, and for a
/// dotted name reach for `info_span!` under a `cfg_attr`.
macro_rules! detail_span {
    ($name:literal $(, $field:ident = $value:expr)* $(,)?) => {{
        #[cfg(feature = "detailed-tracing")]
        {
            ::tracing::info_span!($name $(, $field = $value)*)
        }
        #[cfg(not(feature = "detailed-tracing"))]
        {
            ::tracing::Span::none()
        }
    }};
}
pub(crate) use detail_span;

/// A [`detail_span!`] for a phase that opens superfiles. It also declares the
/// fields `OpenTierCounts::record_on` fills (`memory`, `disk`, `lazy`,
/// `source`, `coalesced`, `streamed`): how many of the phase's opens each
/// cache tier served, as one line of counts rather than a span per file.
macro_rules! tiered_span {
    ($name:literal $(, $field:ident = $value:expr)* $(,)?) => {
        $crate::utils::trace::detail_span!(
            $name
            $(, $field = $value)*,
            memory = ::tracing::field::Empty,
            disk = ::tracing::field::Empty,
            lazy = ::tracing::field::Empty,
            source = ::tracing::field::Empty,
            coalesced = ::tracing::field::Empty,
            streamed = ::tracing::field::Empty,
        )
    };
}
pub(crate) use tiered_span;

/// A search's phase span, or [`Span::none()`] when `on` is false.
///
/// Every exported span costs the query a few microseconds, so a search only
/// splits into phases when there is something to split: see
/// `SupertableReader::phase_spans`. `span` is only called when `on`, so its
/// field expressions cost nothing otherwise. A child of a skipped phase
/// nests under the nearest span that was made.
pub(crate) fn phase(on: bool, span: impl FnOnce() -> Span) -> Span {
    if on { span() } else { Span::none() }
}

/// End `span` here, for every consumer.
///
/// An OpenTelemetry exporter dates a span's end from its last exit, and a
/// log layer from its close. A phase span that only instruments some of its
/// awaits would end at the last of those in one and wherever the handle is
/// dropped in the other. One empty enter and exit, then the drop, puts both
/// at this line.
pub(crate) fn end(span: Span) {
    span.in_scope(|| {});
}

/// Record `value` into `field` on the currently-entered span.
///
/// For the outcome of an operation — a cache hit, a byte count, which of
/// three branches a refresh took — which isn't known until the work is
/// done. The enclosing `#[instrument]` declares the field as
/// `tracing::field::Empty` and this fills it in before the span closes,
/// so the span carries both its duration and what it did.
///
/// Compiles to nothing without `detailed-tracing`: the body is behind a
/// `cfg!` so it still type-checks in every configuration (a field/value
/// mistake can't hide in the feature-off build) while folding away in a
/// release build that doesn't want it. Recording into a field the
/// enclosing span never declared — or with no span entered — is a
/// silent no-op, which is what makes the call sites safe to leave
/// unconditional.
pub(crate) fn record<V: Value>(field: &'static str, value: V) {
    if cfg!(feature = "detailed-tracing") {
        Span::current().record(field, value);
    }
}

/// Carry the current span into the tasks DataFusion spawns, so a query's
/// spans stay in its trace whatever plan DataFusion runs.
///
/// DataFusion executes an operator's input on a spawned task when it
/// repartitions, merges partitions or builds a join side. Without this, a
/// search table function planned under such an operator would open its span
/// on that task with no parent and start a trace of its own. Installed once
/// per process, and only with `detailed-tracing`. The tracer is
/// process-wide: if the embedding application has installed one, that one
/// stays.
pub(crate) fn follow_spans_into_datafusion_tasks() {
    #[cfg(feature = "detailed-tracing")]
    {
        static INSTALLED: Once = Once::new();
        INSTALLED.call_once(|| {
            // Already set by the application: theirs is the one to keep.
            let _ = set_join_set_tracer(&CurrentSpanTracer);
        });
    }
}

/// Runs each spawned DataFusion task in the span that spawned it.
#[cfg(feature = "detailed-tracing")]
struct CurrentSpanTracer;

#[cfg(feature = "detailed-tracing")]
impl JoinSetTracer for CurrentSpanTracer {
    fn trace_future(
        &self,
        fut: BoxFuture<'static, Box<dyn Any + Send>>,
    ) -> BoxFuture<'static, Box<dyn Any + Send>> {
        fut.in_current_span().boxed()
    }

    fn trace_block(
        &self,
        f: Box<dyn FnOnce() -> Box<dyn Any + Send> + Send>,
    ) -> Box<dyn FnOnce() -> Box<dyn Any + Send> + Send> {
        let span = Span::current();
        Box::new(move || span.in_scope(f))
    }
}

/// What a read's root span records once the read has run: the rows it
/// returned, the counters of the op-stats collector it ran under, and the
/// object-store requests and bytes issued meanwhile.
///
/// The store numbers are a delta from [`Self::begin`]. The ledger is the
/// connection's, not the read's, so a second read on the same connection at
/// the same time lands in it too. The op-stats numbers are the collector's
/// totals, so they are the read's own only when its `with_op_stats` scope
/// wraps just this read.
///
/// Fields the span does not declare are ignored, so a root declares the ones
/// it wants as `tracing::field::Empty`; [`Self::finish`] lists them all.
pub(crate) struct CloseOut {
    op_stats: Option<Arc<OpStatsCollector>>,
    store: Option<(Arc<UsageMeter>, UsageSnapshot)>,
}

impl CloseOut {
    /// Snapshot what [`Self::finish`] subtracts from. Without
    /// `detailed-tracing` neither input is called, so a read pays nothing.
    pub(crate) fn begin(
        op_stats: impl FnOnce() -> Option<Arc<OpStatsCollector>>,
        meter: impl FnOnce() -> Option<Arc<UsageMeter>>,
    ) -> Self {
        if !cfg!(feature = "detailed-tracing") {
            return Self {
                op_stats: None,
                store: None,
            };
        }
        Self {
            op_stats: op_stats(),
            store: meter().map(|meter| {
                let before = meter.snapshot();
                (meter, before)
            }),
        }
    }

    /// Record the read's outcome on the current span.
    pub(crate) fn finish(self, rows_out: u64) {
        if !cfg!(feature = "detailed-tracing") {
            return;
        }
        let span = Span::current();
        span.record("rows_out", rows_out);
        if let Some(stats) = self.op_stats {
            let stats = stats.snapshot();
            span.record("sql_page_bytes", stats.sql_page_bytes);
            span.record("planned_read_ranges", stats.planned_read_ranges);
            span.record("rows_materialized", stats.rows_materialized);
            span.record("kernel_cpu_ns", stats.kernel_cpu_ns);
            span.record("fts_postings_bytes", stats.fts_postings_bytes);
            span.record("vector_cells_scanned", stats.vector_cells_scanned);
            span.record("vector_candidates_scanned", stats.vector_candidates_scanned);
            span.record("vector_rows_reranked", stats.vector_rows_reranked);
        }
        if let Some((meter, before)) = self.store {
            let used = meter.snapshot().since(&before);
            span.record("store_heads", used.head_count);
            span.record("store_gets", used.get_count);
            span.record("store_get_bytes", used.get_bytes);
            span.record("store_bg_gets", used.bg_get_count);
            span.record("store_bg_get_bytes", used.bg_get_bytes);
        }
    }

    /// [`Self::finish`] with `rows_out` counted from the read's batches.
    pub(crate) fn finish_batches(self, batches: &[RecordBatch]) {
        self.finish(batches.iter().map(|b| b.num_rows() as u64).sum());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The span-field vocabulary is an observability contract: dashboards
    /// and log filters key on these exact strings, so a rename is a
    /// breaking change to anything consuming the spans. Both renderings
    /// are `const fn` returning `&'static str`, and only the span sites
    /// call them — which are behind `detailed-tracing`, so nothing else
    /// in a default build pins the spelling.
    #[test]
    fn span_field_renderings_are_stable() {
        assert_eq!(OpOrigin::Query.as_str(), "query");
        assert_eq!(OpOrigin::Ingest.as_str(), "ingest");
        assert_eq!(OpOrigin::Optimize.as_str(), "optimize");
        assert_eq!(OpOrigin::Maintenance.as_str(), "maintenance");
        assert_eq!(OpOrigin::Open.as_str(), "open");
        assert_eq!(TableRole::User.as_str(), "user");
        assert_eq!(TableRole::VectorIndex.as_str(), "vector_index");
    }
}
