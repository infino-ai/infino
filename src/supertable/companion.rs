// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Hidden companion tables: tables a caller declares beside a table when it
//! creates it, each with its own schema and indexes, stored under the
//! table's location, recorded on its manifest and never listed in the
//! catalog.
//!
//! This module holds the naming rules: what a companion's purpose may be,
//! the storage prefix a companion is created at, and how a directory name
//! under the table reads back as a companion of a purpose. Creating,
//! opening and re-attaching companions is in [`super::handle`], by the same
//! code that makes the hidden vector index.

use arrow_schema::SchemaRef;
use uuid::Uuid;

use crate::{
    superfile::builder::{FtsConfig, VectorConfig},
    supertable::manifest::UUID_TEXT_LEN,
};

/// What the storage prefix of every hidden table under a table starts with:
/// the vector index's and each companion's.
pub(crate) const HIDDEN_TABLE_PREFIX: &str = "_infino_";

/// Suffix of the hidden vector index's prefix
/// (`_infino_<uuid>_vector_index`). Reserved, so no companion can take it
/// as its purpose and a vector index never reads back as a companion.
pub(crate) const VECTOR_INDEX_SUFFIX: &str = "vector_index";

/// Longest purpose a companion may have. The purpose is part of an object
/// key under the table, so this keeps the key legible; it is not a storage
/// limit.
pub(crate) const MAX_PURPOSE_LEN: usize = 64;

/// A companion declared at create, lowered to what the table's options
/// carry: its purpose, its Arrow schema and the indexes over it.
#[derive(Debug, Clone)]
pub(crate) struct CompanionSpec {
    /// Short name, unique per table, that the companion is reached by.
    pub(crate) purpose: String,
    /// The companion's columns, as the caller declared them.
    pub(crate) schema: SchemaRef,
    /// The companion's full-text indexed columns.
    pub(crate) fts_columns: Vec<FtsConfig>,
    /// The companion's vector indexed columns.
    pub(crate) vector_columns: Vec<VectorConfig>,
}

/// Refuses a purpose that cannot name a companion: empty, longer than
/// [`MAX_PURPOSE_LEN`], holding anything but `[a-z0-9_]`, or the vector
/// index's reserved suffix. The error says which.
pub(crate) fn check_purpose(purpose: &str) -> Result<(), String> {
    if purpose.is_empty() {
        return Err("a companion purpose must not be empty".to_string());
    }
    if purpose.len() > MAX_PURPOSE_LEN {
        return Err(format!(
            "companion purpose {purpose:?} is longer than {MAX_PURPOSE_LEN} characters"
        ));
    }
    if !purpose
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    {
        return Err(format!(
            "companion purpose {purpose:?} may hold only lowercase letters, digits and '_'"
        ));
    }
    if purpose == VECTOR_INDEX_SUFFIX {
        return Err(format!(
            "companion purpose {purpose:?} is reserved for the hidden vector index"
        ));
    }
    Ok(())
}

/// A fresh storage prefix for a hidden table whose prefix ends in `suffix`:
/// `_infino_<uuid>_<suffix>`. The uuid keeps two creations apart; the
/// suffix says what the table is.
pub(crate) fn hidden_table_prefix(suffix: &str) -> String {
    format!("{HIDDEN_TABLE_PREFIX}{}_{suffix}", Uuid::new_v4())
}

/// The purpose a directory named `name` under a table holds a companion of,
/// when the name has exactly the shape [`hidden_table_prefix`] gives a
/// companion: `_infino_`, a lowercase hyphenated uuid, `_`, and a valid
/// purpose. `None` for every other name, the hidden vector index's
/// included. The uuid has a fixed width, so the purpose after it is read
/// unambiguously even when it holds underscores.
pub(crate) fn companion_purpose_of(name: &str) -> Option<&str> {
    let rest = name.strip_prefix(HIDDEN_TABLE_PREFIX)?;
    let uuid_text = rest.get(..UUID_TEXT_LEN)?;
    let uuid = Uuid::parse_str(uuid_text).ok()?;
    if uuid.hyphenated().to_string() != uuid_text {
        return None;
    }
    let purpose = rest.get(UUID_TEXT_LEN..)?.strip_prefix('_')?;
    check_purpose(purpose).is_ok().then_some(purpose)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supertable::manifest::DEFAULT_VECTOR_INDEX_PREFIX;

    /// A purpose longer than the bound by one character.
    const TOO_LONG: usize = MAX_PURPOSE_LEN + 1;

    #[test]
    fn a_purpose_is_short_lowercase_and_not_reserved() {
        assert!(check_purpose("sync").is_ok());
        assert!(check_purpose("sync_state_2").is_ok());
        assert!(check_purpose(&"a".repeat(MAX_PURPOSE_LEN)).is_ok());
        for bad in ["", "Sync", "sync-state", "sync/state", "sync state", "é"] {
            assert!(check_purpose(bad).is_err(), "{bad:?} is refused");
        }
        assert!(check_purpose(&"a".repeat(TOO_LONG)).is_err());
        assert!(check_purpose(VECTOR_INDEX_SUFFIX).is_err());
    }

    /// A generated prefix reads back as its purpose, underscores and all.
    #[test]
    fn a_generated_prefix_reads_back_as_its_purpose() {
        for purpose in ["sync", "sync_state", "a_b_c"] {
            assert_eq!(
                companion_purpose_of(&hidden_table_prefix(purpose)),
                Some(purpose)
            );
        }
    }

    /// Nothing but a companion's shape reads as a companion: not the vector
    /// index, not the legacy vector-index prefix, not the table's own
    /// directories, not a near miss on the uuid.
    #[test]
    fn only_a_companion_prefix_reads_as_a_companion() {
        let uuid = Uuid::new_v4();
        let not_companions = [
            hidden_table_prefix(VECTOR_INDEX_SUFFIX),
            DEFAULT_VECTOR_INDEX_PREFIX.to_string(),
            "_supertable".to_string(),
            "manifest".to_string(),
            format!("{HIDDEN_TABLE_PREFIX}{uuid}"),
            format!("{HIDDEN_TABLE_PREFIX}{uuid}_"),
            format!("{HIDDEN_TABLE_PREFIX}{uuid}sync"),
            format!("{HIDDEN_TABLE_PREFIX}{}_sync", uuid.simple()),
            format!(
                "{HIDDEN_TABLE_PREFIX}{}_sync",
                uuid.hyphenated().to_string().to_uppercase()
            ),
            format!("{HIDDEN_TABLE_PREFIX}{uuid}_Sync"),
        ];
        for name in &not_companions {
            assert_eq!(companion_purpose_of(name), None, "{name:?}");
        }
    }
}
