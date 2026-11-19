// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! When each cached file was cached, and when each kind of reader last read it. Recorded only:
//! nothing here changes what the cache keeps or evicts.

use std::{
    fmt,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use tokio::task::futures::TaskLocalFuture;

/// Microseconds in a millisecond: a file's stamp is written at most once a millisecond.
const STAMP_EVERY_US: u64 = 1_000;

/// Why a file's bytes were read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadKind {
    /// Any open of the file. A whole copy's reads skip the cache, so for it this is the only stamp.
    Open,
    /// Opening a file to plan a query: footer, page index, statistics.
    Planning,
    /// A query reading the rows or index sections it needs.
    Scan,
    /// Reading ahead of a scan, for the row groups it reads next.
    ReadAhead,
    /// Sections fetched ahead of the reads that need them, as the term-index build does.
    Prefetch,
    /// A background download of the whole file.
    Fill,
    /// Not set where the read began. Stays last: [`READ_KINDS`] counts up to it.
    Unknown,
}

/// How many [`ReadKind`]s there are, so a new kind can't index past a record's stamps.
const READ_KINDS: usize = ReadKind::Unknown as usize + 1;

impl ReadKind {
    const ALL: [ReadKind; READ_KINDS] = [
        ReadKind::Open,
        ReadKind::Planning,
        ReadKind::Scan,
        ReadKind::ReadAhead,
        ReadKind::Prefetch,
        ReadKind::Fill,
        ReadKind::Unknown,
    ];
}

/// The kind of read a task makes, and the one time its reads are stamped at, so a read under a
/// scope never reads the clock; a read outside one stamps `Unknown` now.
#[derive(Clone, Copy)]
pub(crate) struct ReadScope {
    kind: ReadKind,
    at: Instant,
}

tokio::task_local! {
    static READ_SCOPE: ReadScope;
}

/// `fut`, with its reads stamped as `kind` at the enclosing scope's time, else now: one clock read
/// per outermost scope.
pub(crate) fn with_read_kind<F: Future>(kind: ReadKind, fut: F) -> TaskLocalFuture<ReadScope, F> {
    let at = READ_SCOPE
        .try_with(|scope| scope.at)
        .unwrap_or_else(|_| Instant::now());
    with_read_kind_at(kind, at, fut)
}

/// `fut`, with its reads stamped as `kind` at `at`, the time its statement started.
pub(crate) fn with_read_kind_at<F: Future>(
    kind: ReadKind,
    at: Instant,
    fut: F,
) -> TaskLocalFuture<ReadScope, F> {
    READ_SCOPE.scope(ReadScope { kind, at }, fut)
}

/// When the cache first saw a file, and when each read kind last read it, in microseconds since the
/// store started (the clock an entry's last access uses); 0 for a kind that never read it.
#[derive(Debug)]
pub(crate) struct FileUse {
    store_started: Instant,
    cached_at_us: AtomicU64,
    last_read_us: [AtomicU64; READ_KINDS],
}

impl FileUse {
    pub(crate) fn new(store_started: Instant) -> Arc<Self> {
        Arc::new(Self {
            store_started,
            cached_at_us: AtomicU64::new(micros_since(store_started, Instant::now())),
            last_read_us: Default::default(),
        })
    }

    /// Stamp a read under the current task's scope, else as `Unknown` now (such as a read on the
    /// reader pool).
    pub(crate) fn record_read(&self) {
        match READ_SCOPE.try_with(|scope| *scope) {
            Ok(scope) => self.record_at(scope.kind, micros_since(self.store_started, scope.at)),
            Err(_) => self.record_read_by(ReadKind::Unknown),
        }
    }

    /// Stamp a read by `kind` now.
    pub(crate) fn record_read_by(&self, kind: ReadKind) {
        self.record_at(kind, micros_since(self.store_started, Instant::now()));
    }

    /// Stamp `kind` at `now_us` on the store's clock, writing only when it moves the stamp forward
    /// into a later millisecond, so reads of one file rarely write the same memory.
    pub(crate) fn record_at(&self, kind: ReadKind, now_us: u64) {
        let now = now_us.max(1);
        let stamp = &self.last_read_us[kind as usize];
        let last = stamp.load(Ordering::Relaxed);
        if last == 0 || now / STAMP_EVERY_US > last / STAMP_EVERY_US {
            stamp.fetch_max(now, Ordering::Relaxed);
        }
    }

    /// Take over `older`'s history, as a whole copy replacing an entry does: the earliest first
    /// seen time, and each kind's latest read.
    pub(crate) fn absorb(&self, older: &FileUse) {
        self.cached_at_us.fetch_min(
            older.cached_at_us.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        for kind in ReadKind::ALL {
            self.last_read_us[kind as usize].fetch_max(older.last_read_us(kind), Ordering::Relaxed);
        }
    }

    /// When `kind` last read the file; 0 if it never has.
    pub(crate) fn last_read_us(&self, kind: ReadKind) -> u64 {
        self.last_read_us[kind as usize].load(Ordering::Relaxed)
    }
}

impl fmt::Display for FileUse {
    /// One line: when cached, and when each read kind that read the file last did.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cached_at_us={}",
            self.cached_at_us.load(Ordering::Relaxed)
        )?;
        for kind in ReadKind::ALL {
            let last = self.last_read_us(kind);
            if last > 0 {
                write!(f, " {kind:?}={last}")?;
            }
        }
        Ok(())
    }
}

/// Microseconds from `start` to `at`, at least 1 so 0 can mean "never".
fn micros_since(start: Instant, at: Instant) -> u64 {
    (at.saturating_duration_since(start).as_micros() as u64).max(1)
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::Ordering,
        time::{Duration, Instant},
    };

    use tokio::time::sleep;

    use crate::supertable::reader_cache::file_use::{
        FileUse, ReadKind, micros_since, with_read_kind, with_read_kind_at,
    };

    /// A statement's start, this far after its store started.
    const STATEMENT_AT: Duration = Duration::from_millis(5);

    /// How long a store runs before a planning, so a stamp at the store's start can't pass.
    const STORE_AGE: Duration = Duration::from_millis(2);

    #[tokio::test]
    async fn a_read_stamps_only_its_read_kind() {
        // One file read under a scan, under a fill inside the scan, and by a task with no kind.
        //  - each read stamps its own kind and leaves the others as they were.
        //  - an inner scope's kind wins over the outer one.
        // This test pins the kind each read is stamped under.
        let file = FileUse::new(Instant::now());
        let stamped = |file: &FileUse| ReadKind::ALL.map(|kind| file.last_read_us(kind) > 0);
        let only = |kinds: &[ReadKind]| ReadKind::ALL.map(|kind| kinds.contains(&kind));

        with_read_kind(ReadKind::Scan, async { file.record_read() }).await;
        assert_eq!(stamped(&file), only(&[ReadKind::Scan]));
        with_read_kind(
            ReadKind::Scan,
            with_read_kind(ReadKind::Fill, async { file.record_read() }),
        )
        .await;
        assert_eq!(stamped(&file), only(&[ReadKind::Scan, ReadKind::Fill]));
        file.record_read();
        assert_eq!(
            stamped(&file),
            only(&[ReadKind::Scan, ReadKind::Fill, ReadKind::Unknown])
        );
    }

    #[tokio::test]
    async fn a_scoped_read_is_stamped_at_its_statements_start() {
        // A statement started 5 ms after the store, its reads made later, one through an inner
        // scope of another kind.
        //  - every read is stamped at the statement's start, not when it ran;
        //  - the inner scope keeps the outer scope's time.
        // This test pins that a read under a scope never reads the clock.
        let started = Instant::now();
        let file = FileUse::new(started);
        let statement = started + STATEMENT_AT;
        let at_us = STATEMENT_AT.as_micros() as u64;

        with_read_kind_at(ReadKind::Scan, statement, async {
            file.record_read();
            with_read_kind(ReadKind::Planning, async { file.record_read() }).await;
        })
        .await;

        assert_eq!(file.last_read_us(ReadKind::Scan), at_us);
        assert_eq!(file.last_read_us(ReadKind::Planning), at_us);
    }

    #[tokio::test]
    async fn a_scope_with_no_statement_is_stamped_when_it_begins() {
        // A planning open with no enclosing scope, as the SQL provider's file open is.
        //  - its read is stamped at the scope's own start: after the store started, not a constant.
        // This test pins that a planning stamp moves with each planning, not with the store's age.
        let started = Instant::now();
        let file = FileUse::new(started);
        sleep(STORE_AGE).await;
        let before = micros_since(started, Instant::now());

        with_read_kind(ReadKind::Planning, async { file.record_read() }).await;

        let after = micros_since(started, Instant::now());
        let planned = file.last_read_us(ReadKind::Planning);
        assert!(
            (before..=after).contains(&planned),
            "{before} <= {planned} <= {after}"
        );
    }

    #[test]
    fn a_replacing_record_keeps_the_earliest_first_seen_and_latest_reads() {
        // An older record read by a scan, a newer one read by a fill; the newer absorbs the older.
        //  - first seen is the older record's.
        //  - each kind keeps its latest read: the scan from the older, the fill from the newer.
        // This test pins what a whole copy keeps from the entry it replaces.
        let started = Instant::now();
        let older = FileUse::new(started);
        older.record_at(ReadKind::Scan, 5_000);
        let newer = FileUse::new(started);
        newer.record_at(ReadKind::Fill, 9_000);
        newer.record_at(ReadKind::Scan, 2_000);
        let older_seen = older.cached_at_us.load(Ordering::Relaxed);
        newer.absorb(&older);

        assert_eq!(newer.cached_at_us.load(Ordering::Relaxed), older_seen);
        assert_eq!(newer.last_read_us(ReadKind::Scan), 5_000);
        assert_eq!(newer.last_read_us(ReadKind::Fill), 9_000);
    }
}
