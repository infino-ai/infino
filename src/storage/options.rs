// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Backend-agnostic storage configuration.
//!
//! Credentials and tuning travel as a flat map keyed by `object_store`'s
//! config strings (`aws_*` / `azure_*`); [`apply`] folds it onto any
//! backend's builder. A new cloud backend reuses the same map — no new
//! API. Infino reads no credentials from the environment; ambient cloud
//! identity is `object_store`'s concern, resolved at request time.

use std::{collections::HashMap, fmt::Display, str::FromStr};

use super::StorageError;

/// Storage config keyed by `object_store` config strings. Empty → the
/// backend's ambient identity.
pub(crate) type StorageOptions = HashMap<String, String>;

/// Fold `opts` onto `builder`, parsing each key into the backend's config
/// key `K`. An unknown or cross-backend key fails loudly rather than being
/// silently dropped.
pub(crate) fn apply<K, B>(
    builder: B,
    opts: &StorageOptions,
    uri: &str,
    fold: impl Fn(B, K, &str) -> B,
) -> Result<B, StorageError>
where
    K: FromStr,
    <K as FromStr>::Err: Display,
{
    let mut builder = builder;
    for (key, value) in opts {
        let parsed = K::from_str(key).map_err(|e| StorageError::Permanent {
            uri: uri.to_string(),
            source: format!("invalid storage option '{key}': {e}").into(),
        })?;
        builder = fold(builder, parsed, value);
    }
    Ok(builder)
}
