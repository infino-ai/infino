// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Prepared-dataset metadata for the search-on-pre-uploaded-data path.
//!
//! A bench normally generates its corpus and ingests it every run. When
//! `INFINO_BENCH_DATASET_PREFIX` is set ("dataset mode"), it instead opens a
//! frozen supertable already in object storage and runs only the search/query
//! groups. This module owns the sidecar that travels with such a dataset
//! ([`DatasetMeta`]) and the guard that refuses to benchmark one built with a
//! different corpus shape ([`verify`]).
//!
//! Everything here is inert unless `INFINO_BENCH_DATASET_PREFIX` is set.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// Object key of the metadata sidecar, relative to the dataset's
/// prefix-scoped storage provider.
pub const SIDECAR: &str = "dataset.json";

/// Env var naming the dataset prefix.
pub const PREFIX_ENV: &str = "INFINO_BENCH_DATASET_PREFIX";

static CONFIGURED_PREFIX: OnceLock<String> = OnceLock::new();

/// Set the dataset prefix for this process. First call wins; overrides the
/// env var.
pub fn set_prefix(prefix: &str) {
    let _ = CONFIGURED_PREFIX.set(prefix.trim_matches('/').to_string());
}

/// The corpus + index knobs that determine a dataset's bytes. The synthetic
/// corpus is fully seeded, so two datasets with equal knobs are byte-identical
/// — making structural equality a sound reuse gate (see [`verify`]).
///
/// Deliberately excludes the writer's `BUILDER_ID`: it embeds a git hash that
/// changes every commit, so gating on it would reject every dataset across any
/// two builds. On-disk format compatibility is the reader's concern
/// (`ReadError::UnsupportedVersion` on open), not this guard's.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub struct Knobs {
    pub doc_count: usize,
    pub dim: usize,
    pub n_cent_total: usize,
    pub vec_seed: u64,
    pub text_seed: u64,
    pub rot_seed: u64,
    pub metric: String,
    pub rerank_codec: String,
    pub modality: String,
}

/// Sidecar written next to a prepared dataset. `knobs` is the reuse gate;
/// the rest is provenance + what the consumer needs to size its disk cache
/// without a probe open.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DatasetMeta {
    pub knobs: Knobs,
    pub n_superfiles: usize,
    pub total_index_bytes: u64,
    /// Which build wrote the data. Provenance only — never a reuse gate.
    pub builder_id: String,
    /// SQL query sample row (SQL datasets only). Deterministic from the corpus,
    /// persisted so the consumer needn't regenerate it to run SQL predicates.
    #[serde(default)]
    pub sql_sample_title: Option<String>,
    #[serde(default)]
    pub sql_sample_key: Option<String>,
}

/// The dataset prefix, or `None` when not in dataset mode.
pub fn dataset_prefix() -> Option<String> {
    if let Some(p) = CONFIGURED_PREFIX.get() {
        return Some(p.clone());
    }
    std::env::var(PREFIX_ENV).ok().filter(|s| !s.is_empty())
}

/// Whether the bench should open a pre-uploaded dataset instead of ingesting.
pub fn dataset_mode() -> bool {
    dataset_prefix().is_some()
}

/// Panic unless the dataset's corpus shape matches what this bench expects.
/// A mismatch means the queries / ground truth would be measured against the
/// wrong bytes, so there is no safe way to proceed.
pub fn verify(meta: &DatasetMeta, expected: &Knobs) {
    assert!(
        &meta.knobs == expected,
        "prepared dataset knob mismatch — re-prepare the dataset.\n  dataset: {:?}\n  bench:   {expected:?}",
        meta.knobs
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knobs() -> Knobs {
        Knobs {
            doc_count: 10_000_000,
            dim: 384,
            n_cent_total: 4096,
            vec_seed: 1,
            text_seed: 1,
            rot_seed: 7,
            metric: "Cosine".into(),
            rerank_codec: "Sq8Residual".into(),
            modality: "Combined".into(),
        }
    }

    #[test]
    fn set_prefix_wins_and_first_call_sticks() {
        set_prefix("/datasets/a/");
        set_prefix("datasets/b");
        assert_eq!(dataset_prefix().as_deref(), Some("datasets/a"));
        assert!(dataset_mode());
    }

    #[test]
    fn meta_round_trips() {
        let meta = DatasetMeta {
            knobs: knobs(),
            n_superfiles: 13,
            total_index_bytes: 1_234_567_890,
            builder_id: "infino/0.1.0+abc1234".into(),
            sql_sample_title: None,
            sql_sample_key: None,
        };
        let bytes = serde_json::to_vec(&meta).expect("serialize");
        let back: DatasetMeta = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(back.knobs, meta.knobs);
        assert_eq!(back.total_index_bytes, meta.total_index_bytes);
    }

    #[test]
    fn verify_accepts_matching_knobs() {
        let meta = DatasetMeta {
            knobs: knobs(),
            n_superfiles: 1,
            total_index_bytes: 0,
            builder_id: "x".into(),
            sql_sample_title: None,
            sql_sample_key: None,
        };
        verify(&meta, &knobs());
    }

    #[test]
    #[should_panic(expected = "knob mismatch")]
    fn verify_rejects_mismatched_knobs() {
        let meta = DatasetMeta {
            knobs: knobs(),
            n_superfiles: 1,
            total_index_bytes: 0,
            builder_id: "x".into(),
            sql_sample_title: None,
            sql_sample_key: None,
        };
        let mut expected = knobs();
        expected.dim = 256;
        verify(&meta, &expected);
    }

    #[test]
    fn builder_id_is_not_part_of_the_gate() {
        // Same knobs, different builder ⇒ still accepted: the gate is knobs,
        // not provenance.
        let meta = DatasetMeta {
            knobs: knobs(),
            n_superfiles: 1,
            total_index_bytes: 0,
            builder_id: "infino/9.9.9+deadbee".into(),
            sql_sample_title: None,
            sql_sample_key: None,
        };
        verify(&meta, &knobs());
    }
}
