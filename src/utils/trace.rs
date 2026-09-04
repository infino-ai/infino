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

use tracing::{Span, field::Value};

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
