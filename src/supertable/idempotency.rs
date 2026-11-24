// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Idempotency tokens: what lets a writer retry a commit, or resume after a
//! crash, without applying it twice.
//!
//! A caller passes an [`IdempotencyToken`] with a write. The commit records
//! the token in the manifest list it publishes, in the same object as the
//! rows, so the pointer swap that makes the rows visible also makes the
//! token visible. A later commit carrying a token the list already holds
//! succeeds without publishing anything.
//!
//! Two forms:
//!  - a **key** is an opaque string, one entry per write. It suits
//!    independent requests, such as an HTTP client retrying an append.
//!  - a **sequence** is a scope and a number that rises with each commit
//!    under that scope, one entry per scope however many commits it makes.
//!    It suits a single writer that commits many times, such as a loader or
//!    a streaming sink, and lets a resumed writer ask how far it got. The
//!    engine requires only that the number rises within a scope; the writer
//!    must not commit two numbers of one scope at once.
//!
//! A token names one exact write: each entry records how many rows its write
//! carried, and a write reusing a held token with a different row count is
//! refused rather than skipped, since skipping it would drop the rows the
//! first write did not carry.
//!
//! Entries expire after the table's token retention, which the list
//! records: the `idempotency_token_ttl` of the latest commit that recorded a
//! token, so every handle, whatever its own setting, expires by the same
//! retention. Key entries are also capped at [`MAX_KEY_ENTRIES`], so under
//! heavy keyed traffic the cap, not the retention, bounds how long a key is
//! kept; a writer that commits often holds one sequence entry instead. The
//! list remembers the newest expiry it applied, so [`TokenStatus::Expired`]
//! tells "may have committed, no longer known" apart from
//! [`TokenStatus::Absent`].

use std::{
    collections::BTreeMap,
    fmt,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

/// Most key entries a list keeps. Past it the oldest are dropped first, as
/// if they had expired. Sequence entries are not capped: there is one per
/// writer.
pub(crate) const MAX_KEY_ENTRIES: usize = 10_000;

/// Identifies a write so that applying it twice commits it once.
///
/// Passed as the trailing argument of the write calls; `None` there commits
/// as a write without a token always has, and records nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdempotencyToken<'a> {
    /// One write, named by an opaque string the caller chooses.
    Key(&'a str),
    /// The `seq`th commit of the writer named `scope`. A commit is skipped
    /// when the table already holds `seq` or a later one for `scope`.
    Sequence {
        /// The writer.
        scope: &'a str,
        /// Rises with each commit the writer makes; need not be contiguous.
        seq: u64,
    },
}

impl<'a> From<&'a str> for IdempotencyToken<'a> {
    fn from(key: &'a str) -> Self {
        IdempotencyToken::Key(key)
    }
}

impl<'a> From<&'a String> for IdempotencyToken<'a> {
    fn from(key: &'a String) -> Self {
        IdempotencyToken::Key(key)
    }
}

impl IdempotencyToken<'_> {
    /// An owned copy for a write of `rows` rows, carried with its commit
    /// across threads and retries.
    pub(crate) fn to_owned_token(self, rows: u64) -> OwnedToken {
        let kind = match self {
            IdempotencyToken::Key(key) => OwnedKind::Key(key.to_owned()),
            IdempotencyToken::Sequence { scope, seq } => OwnedKind::Sequence {
                scope: scope.to_owned(),
                seq,
            },
        };
        OwnedToken { kind, rows }
    }
}

/// Whether a token's write committed, as the table's current list records
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TokenStatus {
    /// The write committed: the table holds the key, or a number at or past
    /// the sequence's under its scope. For a sequence the fields describe the
    /// scope's latest commit.
    Committed {
        /// The manifest the token was recorded in.
        manifest_id: u64,
        /// When that commit was made, in milliseconds since the Unix epoch.
        committed_ms: u64,
        /// For a sequence token, the scope's latest committed number: the one
        /// asked about or a later one. `None` for a key.
        seq: Option<u64>,
    },
    /// The table never held the token, or held it before `since_ms`. For a
    /// sequence token whose scope is held at an earlier number, the number
    /// asked about has not committed.
    Absent,
    /// The table does not hold the token, but may have held it after
    /// `since_ms` and dropped it since: retention expired entries that were
    /// at least that recent. Whether the write committed is unknown.
    Expired,
}

/// [`IdempotencyToken`] owned, with the row count of the write it names, as
/// a commit carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OwnedToken {
    kind: OwnedKind,
    /// Rows the write carries.
    pub(crate) rows: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OwnedKind {
    Key(String),
    Sequence { scope: String, seq: u64 },
}

impl OwnedToken {
    pub(crate) fn as_token(&self) -> IdempotencyToken<'_> {
        match &self.kind {
            OwnedKind::Key(key) => IdempotencyToken::Key(key),
            OwnedKind::Sequence { scope, seq } => IdempotencyToken::Sequence { scope, seq: *seq },
        }
    }
}

/// A held token reused by a write of a different row count: not a retry of
/// the write that recorded it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenReused {
    /// The token, as the caller passed it.
    pub(crate) token: String,
    /// Rows the recorded write carried.
    pub(crate) recorded_rows: u64,
    /// Rows the reusing write carries.
    pub(crate) rows: u64,
}

impl fmt::Display for TokenReused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "idempotency token {} committed a write of {} rows, and this write carries {}: a \
             token names one exact write, so retry that write alone and give new rows a new token",
            self.token, self.recorded_rows, self.rows
        )
    }
}

/// A key token's entry. Written in the list as one string,
/// `"<manifest_id> <committed_ms> <rows>"`: a table can hold thousands, and
/// every commit carries them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub struct TokenEntry {
    /// The manifest the token was recorded in.
    pub manifest_id: u64,
    /// When, in milliseconds since the Unix epoch.
    pub committed_ms: u64,
    /// Rows the write carried.
    pub rows: u64,
}

impl From<TokenEntry> for String {
    fn from(e: TokenEntry) -> Self {
        format!("{} {} {}", e.manifest_id, e.committed_ms, e.rows)
    }
}

impl TryFrom<String> for TokenEntry {
    type Error = String;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        let mut fields = text.split(' ').map(str::parse::<u64>);
        match (fields.next(), fields.next(), fields.next(), fields.next()) {
            (Some(Ok(manifest_id)), Some(Ok(committed_ms)), Some(Ok(rows)), None) => Ok(Self {
                manifest_id,
                committed_ms,
                rows,
            }),
            _ => Err(format!(
                "idempotency key entry {text:?} is not \"<manifest_id> <committed_ms> <rows>\""
            )),
        }
    }
}

/// A sequence scope's entry: its latest committed number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeqEntry {
    /// The latest number committed under the scope.
    pub seq: u64,
    /// The manifest it was recorded in.
    pub manifest_id: u64,
    /// When, in milliseconds since the Unix epoch.
    pub committed_ms: u64,
    /// Rows the write of `seq` carried.
    pub rows: u64,
}

/// The tokens a manifest list holds. Written in the list, so it lands
/// atomically with the rows of the commit that recorded it, and carried to
/// every list after it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Idempotency {
    /// Key tokens.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub keys: BTreeMap<String, TokenEntry>,
    /// Sequence tokens, by scope.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sequences: BTreeMap<String, SeqEntry>,
    /// The newest `committed_ms` among the entries retention dropped. A
    /// token absent from the list may have committed at or before it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expired_through_ms: Option<u64>,
    /// How long entries are kept, in milliseconds; zero keeps them until the
    /// key cap drops them. Set by every commit that records a token, from
    /// its handle's `idempotency_token_ttl`, and applied by every commit.
    /// `None` while no token is held.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_ms: Option<u64>,
}

impl Idempotency {
    /// Nothing recorded and nothing expired: the list carries no field.
    pub fn is_empty(&self) -> bool {
        !self.holds_tokens() && self.expired_through_ms.is_none() && self.ttl_ms.is_none()
    }

    /// Whether any token is held. A list holding one is written at the
    /// format major that engines unaware of tokens refuse (see
    /// [`super::manifest::list::FORMAT_VERSION`]).
    pub fn holds_tokens(&self) -> bool {
        !self.keys.is_empty() || !self.sequences.is_empty()
    }

    /// Whether manifest `manifest_id` recorded a token: the commit that
    /// wrote it carried one.
    pub(crate) fn recorded_in(&self, manifest_id: u64) -> bool {
        self.keys.values().any(|e| e.manifest_id == manifest_id)
            || self
                .sequences
                .values()
                .any(|e| e.manifest_id == manifest_id)
    }

    /// Whether the write `token` names was already applied: the key is held,
    /// or the scope holds its number or a later one. Refused when the held
    /// entry is for that very write but recorded a different row count:
    /// the call is not a retry of it. A scope already past the number keeps
    /// no count for it, so there is nothing to compare.
    pub(crate) fn committed(&self, token: &OwnedToken) -> Result<bool, TokenReused> {
        let recorded_rows = match token.as_token() {
            IdempotencyToken::Key(key) => self.keys.get(key).map(|e| e.rows),
            IdempotencyToken::Sequence { scope, seq } => match self.sequences.get(scope) {
                Some(e) if e.seq > seq => return Ok(true),
                Some(e) if e.seq == seq => Some(e.rows),
                _ => None,
            },
        };
        match recorded_rows {
            None => Ok(false),
            Some(recorded_rows) if recorded_rows == token.rows => Ok(true),
            Some(recorded_rows) => Err(TokenReused {
                token: match token.as_token() {
                    IdempotencyToken::Key(key) => format!("{key:?}"),
                    IdempotencyToken::Sequence { scope, seq } => format!("{scope:?} seq {seq}"),
                },
                recorded_rows,
                rows: token.rows,
            }),
        }
    }

    /// Record `token` as committed in manifest `manifest_id` at `now_ms`,
    /// and `ttl` as the retention of every entry.
    pub(crate) fn record(
        &mut self,
        token: &OwnedToken,
        manifest_id: u64,
        now_ms: u64,
        ttl: Duration,
    ) {
        let rows = token.rows;
        match token.as_token() {
            IdempotencyToken::Key(key) => {
                self.keys.insert(
                    key.to_owned(),
                    TokenEntry {
                        manifest_id,
                        committed_ms: now_ms,
                        rows,
                    },
                );
            }
            IdempotencyToken::Sequence { scope, seq } => {
                self.sequences.insert(
                    scope.to_owned(),
                    SeqEntry {
                        seq,
                        manifest_id,
                        committed_ms: now_ms,
                        rows,
                    },
                );
            }
        }
        self.ttl_ms = Some(u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX));
    }

    /// Drop the entries older than [`Self::ttl_ms`] at `now_ms` (none when
    /// it is zero), then the oldest keys past `key_cap`, raising
    /// [`Self::expired_through_ms`] to the newest one dropped. Entries
    /// recorded in manifest `writing_id`, the list being written, always
    /// stay: the commit carrying a token must publish it, however short the
    /// ttl or however far another writer's clock runs ahead.
    pub(crate) fn expire(&mut self, now_ms: u64, key_cap: usize, writing_id: u64) {
        let mut newest_dropped = self.expired_through_ms;
        let mut note = |committed_ms: u64| {
            newest_dropped = Some(newest_dropped.map_or(committed_ms, |n| n.max(committed_ms)));
        };
        let ttl_ms = self.ttl_ms.unwrap_or(0);
        if ttl_ms > 0 {
            let live = |manifest_id: u64, committed_ms: u64| {
                manifest_id >= writing_id || now_ms.saturating_sub(committed_ms) < ttl_ms
            };
            self.keys.retain(|_, e| {
                let keep = live(e.manifest_id, e.committed_ms);
                if !keep {
                    note(e.committed_ms);
                }
                keep
            });
            self.sequences.retain(|_, e| {
                let keep = live(e.manifest_id, e.committed_ms);
                if !keep {
                    note(e.committed_ms);
                }
                keep
            });
        }
        if self.keys.len() > key_cap {
            let mut by_age: Vec<(u64, String)> = self
                .keys
                .iter()
                .filter(|(_, e)| e.manifest_id < writing_id)
                .map(|(key, e)| (e.committed_ms, key.clone()))
                .collect();
            by_age.sort_unstable();
            let excess = self.keys.len() - key_cap;
            for (committed_ms, key) in by_age.into_iter().take(excess) {
                self.keys.remove(&key);
                note(committed_ms);
            }
        }
        self.expired_through_ms = newest_dropped;
        if !self.holds_tokens() {
            self.ttl_ms = None;
        }
    }

    /// What the list knows of `token`, for a caller whose write could not
    /// have committed before `since_ms`.
    pub(crate) fn status(&self, token: IdempotencyToken<'_>, since_ms: u64) -> TokenStatus {
        let held = match token {
            IdempotencyToken::Key(key) => self.keys.get(key).map(|e| TokenStatus::Committed {
                manifest_id: e.manifest_id,
                committed_ms: e.committed_ms,
                seq: None,
            }),
            IdempotencyToken::Sequence { scope, seq } => match self.sequences.get(scope) {
                Some(e) if e.seq >= seq => Some(TokenStatus::Committed {
                    manifest_id: e.manifest_id,
                    committed_ms: e.committed_ms,
                    seq: Some(e.seq),
                }),
                // The scope is held at an earlier number, and its numbers
                // rise: this one has not committed.
                Some(_) => return TokenStatus::Absent,
                None => None,
            },
        };
        match held {
            Some(status) => status,
            None if self.expired_through_ms.is_some_and(|w| w >= since_ms) => TokenStatus::Expired,
            None => TokenStatus::Absent,
        }
    }
}

/// Milliseconds since the Unix epoch, as entries record them.
pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_MS: u64 = 24 * 60 * 60 * 1000;
    const WEEK: Duration = Duration::from_millis(7 * DAY_MS);

    fn key(key: &str, rows: u64) -> OwnedToken {
        IdempotencyToken::Key(key).to_owned_token(rows)
    }

    fn seq(scope: &str, seq: u64, rows: u64) -> OwnedToken {
        IdempotencyToken::Sequence { scope, seq }.to_owned_token(rows)
    }

    #[test]
    fn a_key_is_committed_once_recorded() {
        let mut map = Idempotency::default();
        assert_eq!(map.committed(&key("k", 3)), Ok(false));
        assert_eq!(map.status("k".into(), 0), TokenStatus::Absent);
        map.record(&key("k", 3), 7, 1_000, WEEK);
        assert_eq!(map.committed(&key("k", 3)), Ok(true));
        assert_eq!(
            map.status("k".into(), 0),
            TokenStatus::Committed {
                manifest_id: 7,
                committed_ms: 1_000,
                seq: None
            }
        );
    }

    #[test]
    fn a_held_key_reused_with_other_rows_is_refused() {
        let mut map = Idempotency::default();
        map.record(&key("order-1", 1_000), 7, 1_000, WEEK);
        assert_eq!(
            map.committed(&key("order-1", 1_500)),
            Err(TokenReused {
                token: "\"order-1\"".to_owned(),
                recorded_rows: 1_000,
                rows: 1_500,
            }),
            "a retry carrying more rows would drop the 500 the first write did not"
        );
        assert_eq!(map.committed(&key("order-1", 1_000)), Ok(true));
    }

    #[test]
    fn a_sequence_skips_its_number_and_every_earlier_one() {
        let mut map = Idempotency::default();
        map.record(&seq("s", 3, 10), 1, 10, WEEK);
        assert_eq!(
            map.committed(&seq("s", 1, 99)),
            Ok(true),
            "an earlier number keeps no row count to compare"
        );
        assert_eq!(map.committed(&seq("s", 3, 10)), Ok(true));
        assert_eq!(
            map.committed(&seq("s", 3, 11)),
            Err(TokenReused {
                token: "\"s\" seq 3".to_owned(),
                recorded_rows: 10,
                rows: 11,
            })
        );
        assert_eq!(map.committed(&seq("s", 4, 10)), Ok(false));
        assert_eq!(
            map.committed(&seq("t", 1, 10)),
            Ok(false),
            "scopes are independent"
        );
        map.record(&seq("s", 5, 10), 2, 20, WEEK);
        assert_eq!(
            map.status(IdempotencyToken::Sequence { scope: "s", seq: 1 }, 0),
            TokenStatus::Committed {
                manifest_id: 2,
                committed_ms: 20,
                seq: Some(5)
            },
            "status names the scope's latest number"
        );
    }

    #[test]
    fn a_sequence_number_past_the_scope_reads_absent() {
        let mut map = Idempotency::default();
        map.record(&seq("loader", 4, 10), 1, 10, WEEK);
        assert_eq!(
            map.status(
                IdempotencyToken::Sequence {
                    scope: "loader",
                    seq: 5
                },
                0
            ),
            TokenStatus::Absent,
            "the scope holds 4, so 5 has not committed"
        );
        map.expired_through_ms = Some(5);
        assert_eq!(
            map.status(
                IdempotencyToken::Sequence {
                    scope: "loader",
                    seq: 5
                },
                0
            ),
            TokenStatus::Absent,
            "a held scope's numbers rise, so expiry cannot hide 5"
        );
    }

    #[test]
    fn expiry_drops_old_entries_and_raises_the_watermark() {
        let mut map = Idempotency::default();
        map.record(&key("old", 1), 1, DAY_MS, WEEK);
        map.record(&seq("s", 1, 1), 1, 2 * DAY_MS, WEEK);
        map.record(&key("new", 1), 2, 9 * DAY_MS, WEEK);
        map.expire(10 * DAY_MS, MAX_KEY_ENTRIES, 3);
        assert_eq!(map.committed(&key("old", 1)), Ok(false));
        assert!(map.sequences.is_empty());
        assert_eq!(map.committed(&key("new", 1)), Ok(true));
        assert_eq!(map.expired_through_ms, Some(2 * DAY_MS));
        assert_eq!(map.status("old".into(), 0), TokenStatus::Expired);
        assert_eq!(
            map.status("old".into(), 2 * DAY_MS),
            TokenStatus::Expired,
            "a write marked at the newest expired entry may be that entry"
        );
        assert_eq!(
            map.status("never".into(), 3 * DAY_MS),
            TokenStatus::Absent,
            "nothing at or after since_ms was dropped"
        );
    }

    #[test]
    fn the_latest_recording_commit_sets_the_retention_for_every_entry() {
        let mut map = Idempotency::default();
        map.record(&key("a", 1), 1, 0, WEEK);
        map.record(&key("b", 1), 2, 0, Duration::from_millis(DAY_MS));
        assert_eq!(map.ttl_ms, Some(DAY_MS));
        map.expire(2 * DAY_MS, MAX_KEY_ENTRIES, 3);
        assert!(
            !map.holds_tokens(),
            "the list's one-day retention drops a, recorded under a week"
        );
        assert_eq!(map.ttl_ms, None, "no token left to keep");
    }

    #[test]
    fn retention_is_the_lists_not_the_committing_handles() {
        let mut map = Idempotency::default();
        map.record(&key("k", 1), 1, 0, WEEK);
        // A commit that records no token applies the list's retention,
        // whatever its own handle's.
        map.expire(2 * DAY_MS, MAX_KEY_ENTRIES, 2);
        assert_eq!(map.committed(&key("k", 1)), Ok(true));
    }

    #[test]
    fn a_zero_ttl_keeps_entries_forever() {
        let mut map = Idempotency::default();
        map.record(&key("k", 1), 1, 0, Duration::ZERO);
        map.expire(1_000 * DAY_MS, MAX_KEY_ENTRIES, 2);
        assert_eq!(map.committed(&key("k", 1)), Ok(true));
        assert_eq!(map.expired_through_ms, None);
    }

    #[test]
    fn the_key_cap_drops_the_oldest_first() {
        let mut map = Idempotency::default();
        for (i, k) in ["a", "b", "c", "d"].into_iter().enumerate() {
            map.record(&key(k, 1), 1, 100 - i as u64, Duration::ZERO);
        }
        map.record(&seq("s", 1, 1), 1, 1, Duration::ZERO);
        map.expire(200, 2, 2);
        assert_eq!(
            map.keys.keys().map(String::as_str).collect::<Vec<_>>(),
            ["a", "b"],
            "the two newest keys stay"
        );
        assert_eq!(map.sequences.len(), 1, "sequences are not capped");
        assert_eq!(map.expired_through_ms, Some(98));
    }

    #[test]
    fn the_list_being_written_keeps_the_entries_it_records() {
        let ms = Duration::from_millis(1);
        let mut map = Idempotency::default();
        map.record(&key("older", 1), 4, 0, ms);
        map.record(&key("mine", 1), 5, 0, ms);
        map.record(&seq("s", 3, 1), 5, 0, ms);
        map.expire(DAY_MS, MAX_KEY_ENTRIES, 5);
        assert_eq!(map.committed(&key("older", 1)), Ok(false));
        assert_eq!(
            map.committed(&key("mine", 1)),
            Ok(true),
            "past its ttl, but this list records it"
        );
        assert_eq!(map.committed(&seq("s", 3, 1)), Ok(true));
        assert_eq!(map.expired_through_ms, Some(0));

        // A writer whose clock runs behind records the oldest key; the cap
        // drops another.
        let mut map = Idempotency::default();
        map.record(&key("ahead-1", 1), 1, 500, Duration::ZERO);
        map.record(&key("ahead-2", 1), 2, 600, Duration::ZERO);
        map.record(&key("behind", 1), 3, 100, Duration::ZERO);
        map.expire(100, 2, 3);
        assert_eq!(
            map.keys.keys().map(String::as_str).collect::<Vec<_>>(),
            ["ahead-2", "behind"]
        );
    }

    #[test]
    fn an_emptied_map_keeps_its_watermark_but_holds_no_tokens() {
        let mut map = Idempotency::default();
        map.record(&key("k", 1), 1, 0, Duration::from_millis(1));
        map.expire(DAY_MS, MAX_KEY_ENTRIES, 2);
        assert!(!map.holds_tokens());
        assert!(!map.is_empty());
        assert_eq!(map.status("k".into(), 0), TokenStatus::Expired);
    }

    #[test]
    fn an_empty_map_serializes_to_nothing() {
        assert_eq!(
            serde_json::to_string(&Idempotency::default()).expect("encode"),
            "{}"
        );
    }

    #[test]
    fn a_key_entry_is_one_string_in_the_list() {
        let mut map = Idempotency::default();
        map.record(&key("k", 12), 7, 1_000, WEEK);
        let text = serde_json::to_string(&map).expect("encode");
        assert_eq!(text, r#"{"keys":{"k":"7 1000 12"},"ttl_ms":604800000}"#);
        assert_eq!(
            serde_json::from_str::<Idempotency>(&text).expect("decode"),
            map
        );
        for bad in [r#""7 1000""#, r#""7 1000 12 9""#, r#""7 x 12""#, r#""""#] {
            assert!(
                serde_json::from_str::<TokenEntry>(bad).is_err(),
                "{bad} is not an entry"
            );
        }
    }
}
