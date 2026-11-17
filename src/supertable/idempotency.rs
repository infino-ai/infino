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
//! Entries expire after the table's `idempotency_token_ttl`, and key entries
//! are also capped at [`MAX_KEY_ENTRIES`]. The list remembers the newest
//! expiry it applied, so [`TokenStatus::Expired`] tells "may have committed,
//! no longer known" apart from [`TokenStatus::Absent`].

use std::{
    collections::BTreeMap,
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

impl<'a> IdempotencyToken<'a> {
    /// The token as an `Idempotency-Token` HTTP header value: the key itself,
    /// or `<scope>; seq=<n>` for a sequence. `None` when the token cannot
    /// travel as a header value: empty, padded with spaces, holding anything
    /// but visible ASCII and spaces, or a key that would read back as a
    /// sequence.
    pub fn to_header_value(&self) -> Option<String> {
        let (text, value) = match self {
            IdempotencyToken::Key(key) => (*key, (*key).to_owned()),
            IdempotencyToken::Sequence { scope, seq } => (*scope, format!("{scope}; seq={seq}")),
        };
        let travels = !text.is_empty()
            && text.trim() == text
            && text.bytes().all(|b| b == b' ' || b.is_ascii_graphic());
        let reads_back = IdempotencyToken::from_header_value(&value) == Some(*self);
        (travels && reads_back).then_some(value)
    }

    /// Read an `Idempotency-Token` header value written by
    /// [`Self::to_header_value`]: a value ending in `; seq=<n>` is a
    /// sequence under the scope before it, anything else a key. `None` for
    /// an empty value.
    pub fn from_header_value(value: &'a str) -> Option<Self> {
        let value = value.trim();
        if value.is_empty() {
            return None;
        }
        if let Some((scope, seq)) = value.rsplit_once("; seq=")
            && !scope.is_empty()
            && let Ok(seq) = seq.parse::<u64>()
        {
            return Some(IdempotencyToken::Sequence { scope, seq });
        }
        Some(IdempotencyToken::Key(value))
    }

    /// An owned copy, carried with a commit across threads and retries.
    pub(crate) fn to_owned_token(self) -> OwnedToken {
        match self {
            IdempotencyToken::Key(key) => OwnedToken::Key(key.to_owned()),
            IdempotencyToken::Sequence { scope, seq } => OwnedToken::Sequence {
                scope: scope.to_owned(),
                seq,
            },
        }
    }
}

/// Whether a token's write committed, as the table's current list records
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TokenStatus {
    /// The table holds the token: the key, or for a sequence token its
    /// scope. A sequence's number committed only when `seq` is at or past
    /// it; the fields describe the scope's latest commit.
    Committed {
        /// The manifest the token was recorded in.
        manifest_id: u64,
        /// When that commit was made, in milliseconds since the Unix epoch.
        committed_ms: u64,
        /// For a sequence token, the scope's latest committed number, which
        /// may be before or after the one asked about. `None` for a key.
        seq: Option<u64>,
    },
    /// The table never held the token, or held it before `since_ms`.
    Absent,
    /// The table does not hold the token, but may have held it after
    /// `since_ms` and dropped it since: retention expired entries that were
    /// at least that recent. Whether the write committed is unknown.
    Expired,
}

/// [`IdempotencyToken`] owned, as a commit carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OwnedToken {
    Key(String),
    Sequence { scope: String, seq: u64 },
}

impl OwnedToken {
    pub(crate) fn as_token(&self) -> IdempotencyToken<'_> {
        match self {
            OwnedToken::Key(key) => IdempotencyToken::Key(key),
            OwnedToken::Sequence { scope, seq } => IdempotencyToken::Sequence { scope, seq: *seq },
        }
    }
}

/// A key token's entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenEntry {
    /// The manifest the token was recorded in.
    pub manifest_id: u64,
    /// When, in milliseconds since the Unix epoch.
    pub committed_ms: u64,
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
}

impl Idempotency {
    /// Nothing recorded and nothing expired: the list carries no field.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty() && self.sequences.is_empty() && self.expired_through_ms.is_none()
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

    /// Whether a commit carrying `token` was already applied: the key is
    /// held, or the scope holds `seq` or a later number.
    pub(crate) fn committed(&self, token: IdempotencyToken<'_>) -> bool {
        match token {
            IdempotencyToken::Key(key) => self.keys.contains_key(key),
            IdempotencyToken::Sequence { scope, seq } => {
                self.sequences.get(scope).is_some_and(|e| e.seq >= seq)
            }
        }
    }

    /// Record `token` as committed in manifest `manifest_id` at `now_ms`.
    pub(crate) fn record(&mut self, token: IdempotencyToken<'_>, manifest_id: u64, now_ms: u64) {
        match token {
            IdempotencyToken::Key(key) => {
                self.keys.insert(
                    key.to_owned(),
                    TokenEntry {
                        manifest_id,
                        committed_ms: now_ms,
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
                    },
                );
            }
        }
    }

    /// Drop the entries older than `ttl` at `now_ms` (none when `ttl` is
    /// zero), then the oldest keys past `key_cap`, raising
    /// [`Self::expired_through_ms`] to the newest one dropped. Entries
    /// recorded in manifest `writing_id`, the list being written, always
    /// stay: the commit carrying a token must publish it, however short the
    /// ttl or however far another writer's clock runs ahead.
    pub(crate) fn expire(&mut self, now_ms: u64, ttl: Duration, key_cap: usize, writing_id: u64) {
        let mut newest_dropped = self.expired_through_ms;
        let mut note = |committed_ms: u64| {
            newest_dropped = Some(newest_dropped.map_or(committed_ms, |n| n.max(committed_ms)));
        };
        let ttl_ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX);
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
            IdempotencyToken::Sequence { scope, .. } => {
                self.sequences.get(scope).map(|e| TokenStatus::Committed {
                    manifest_id: e.manifest_id,
                    committed_ms: e.committed_ms,
                    seq: Some(e.seq),
                })
            }
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

    #[test]
    fn a_key_is_committed_once_recorded() {
        let mut map = Idempotency::default();
        assert!(!map.committed(IdempotencyToken::Key("k")));
        assert_eq!(map.status("k".into(), 0), TokenStatus::Absent);
        map.record("k".into(), 7, 1_000);
        assert!(map.committed("k".into()));
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
    fn a_sequence_skips_its_number_and_every_earlier_one() {
        let mut map = Idempotency::default();
        let at = |seq| IdempotencyToken::Sequence { scope: "s", seq };
        map.record(at(3), 1, 10);
        assert!(map.committed(at(1)));
        assert!(map.committed(at(3)));
        assert!(!map.committed(at(4)));
        assert!(
            !map.committed(IdempotencyToken::Sequence { scope: "t", seq: 1 }),
            "scopes are independent"
        );
        map.record(at(5), 2, 20);
        assert_eq!(
            map.status(at(1), 0),
            TokenStatus::Committed {
                manifest_id: 2,
                committed_ms: 20,
                seq: Some(5)
            },
            "status names the scope's latest number"
        );
    }

    #[test]
    fn expiry_drops_old_entries_and_raises_the_watermark() {
        let mut map = Idempotency::default();
        map.record("old".into(), 1, DAY_MS);
        map.record(
            IdempotencyToken::Sequence { scope: "s", seq: 1 },
            1,
            2 * DAY_MS,
        );
        map.record("new".into(), 2, 9 * DAY_MS);
        map.expire(
            10 * DAY_MS,
            Duration::from_millis(7 * DAY_MS),
            MAX_KEY_ENTRIES,
            3,
        );
        assert!(!map.committed("old".into()));
        assert!(map.sequences.is_empty());
        assert!(map.committed("new".into()));
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
    fn a_zero_ttl_keeps_entries_forever() {
        let mut map = Idempotency::default();
        map.record("k".into(), 1, 0);
        map.expire(1_000 * DAY_MS, Duration::ZERO, MAX_KEY_ENTRIES, 2);
        assert!(map.committed("k".into()));
        assert_eq!(map.expired_through_ms, None);
    }

    #[test]
    fn the_key_cap_drops_the_oldest_first() {
        let mut map = Idempotency::default();
        for (i, key) in ["a", "b", "c", "d"].into_iter().enumerate() {
            map.record(key.into(), 1, 100 - i as u64);
        }
        map.record(IdempotencyToken::Sequence { scope: "s", seq: 1 }, 1, 1);
        map.expire(200, Duration::ZERO, 2, 2);
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
        let mut map = Idempotency::default();
        map.record("older".into(), 4, 0);
        map.record("mine".into(), 5, 0);
        map.record(IdempotencyToken::Sequence { scope: "s", seq: 3 }, 5, 0);
        map.expire(DAY_MS, Duration::from_millis(1), MAX_KEY_ENTRIES, 5);
        assert!(!map.committed("older".into()));
        assert!(
            map.committed("mine".into()),
            "past its ttl, but this list records it"
        );
        assert!(map.committed(IdempotencyToken::Sequence { scope: "s", seq: 3 }));
        assert_eq!(map.expired_through_ms, Some(0));

        // A writer whose clock runs behind records the oldest key; the cap
        // drops another.
        let mut map = Idempotency::default();
        map.record("ahead-1".into(), 1, 500);
        map.record("ahead-2".into(), 2, 600);
        map.record("behind".into(), 3, 100);
        map.expire(100, Duration::ZERO, 2, 3);
        assert_eq!(
            map.keys.keys().map(String::as_str).collect::<Vec<_>>(),
            ["ahead-2", "behind"]
        );
    }

    #[test]
    fn an_emptied_map_keeps_its_watermark_but_holds_no_tokens() {
        let mut map = Idempotency::default();
        map.record("k".into(), 1, 0);
        map.expire(DAY_MS, Duration::from_millis(1), MAX_KEY_ENTRIES, 2);
        assert!(!map.holds_tokens());
        assert!(!map.is_empty());
        assert_eq!(map.status("k".into(), 0), TokenStatus::Expired);
    }

    #[test]
    fn a_header_value_reads_back_as_the_token_it_came_from() {
        for token in [
            IdempotencyToken::Key("order-42"),
            IdempotencyToken::Key("a key with spaces"),
            IdempotencyToken::Key("seq=7"),
            IdempotencyToken::Sequence {
                scope: "loader/3",
                seq: 0,
            },
            IdempotencyToken::Sequence {
                scope: "a; seq=1",
                seq: u64::MAX,
            },
        ] {
            let value = token.to_header_value().expect("travels");
            assert_eq!(IdempotencyToken::from_header_value(&value), Some(token));
        }
        assert_eq!(
            IdempotencyToken::Sequence {
                scope: "loader",
                seq: 9
            }
            .to_header_value()
            .as_deref(),
            Some("loader; seq=9")
        );
    }

    #[test]
    fn a_token_that_cannot_travel_has_no_header_value() {
        for token in [
            IdempotencyToken::Key(""),
            IdempotencyToken::Key(" padded"),
            IdempotencyToken::Key("padded "),
            IdempotencyToken::Key("line\nbreak"),
            IdempotencyToken::Key("caf\u{e9}"),
            IdempotencyToken::Key("looks like; seq=5"),
            IdempotencyToken::Sequence { scope: "", seq: 1 },
        ] {
            assert_eq!(token.to_header_value(), None, "{token:?}");
        }
    }

    #[test]
    fn a_header_value_is_a_sequence_only_with_a_scope_and_a_number() {
        assert_eq!(IdempotencyToken::from_header_value("  "), None);
        assert_eq!(
            IdempotencyToken::from_header_value(" k1 "),
            Some(IdempotencyToken::Key("k1"))
        );
        assert_eq!(
            IdempotencyToken::from_header_value("; seq=3"),
            Some(IdempotencyToken::Key("; seq=3"))
        );
        assert_eq!(
            IdempotencyToken::from_header_value("w; seq=x"),
            Some(IdempotencyToken::Key("w; seq=x"))
        );
        assert_eq!(
            IdempotencyToken::from_header_value("w; seq=12"),
            Some(IdempotencyToken::Sequence {
                scope: "w",
                seq: 12
            })
        );
    }

    #[test]
    fn an_empty_map_serializes_to_nothing() {
        assert_eq!(
            serde_json::to_string(&Idempotency::default()).expect("encode"),
            "{}"
        );
    }
}
