// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Global term-statistics sidecar: gross `df` per (column, term) summed
//! over a recorded set of superfiles, so a global-stats BM25 query reads
//! corpus-wide document frequency in one artifact lookup instead of
//! fanning a dictionary probe over every superfile at query time.
//!
//! The artifact is content-addressed (`term-stats/stats-<blake3>.bin`)
//! and referenced from the manifest list ([`Manifest::term_stats`]);
//! maintenance (optimize) builds it over the manifest's current
//! superfiles, appends leave it valid (new superfiles are uncovered
//! *tail* the query tops up from their own dictionaries), and any commit
//! that removes superfiles drops the reference — a removed superfile's
//! contribution is baked into the sums and cannot be attributed, so only
//! a fresh maintenance pass may republish (see the carry rule in
//! `ManifestSnapshot::update_inner`).
//!
//! Layout: a fixed header (magic, version, covered-superfile ids) then a
//! standard FST map — the same `column <FST_SEPARATOR> term → u64`
//! shape as a superfile dictionary, values holding summed gross df.
//! Gross means tombstoned docs still count until compaction rewrites the
//! underlying dictionaries — exactly the semantics of the query-time
//! gather this sidecar replaces (consumers clamp df to `n_docs_total`).
//!
//! [`Manifest::term_stats`]: super::list::Manifest::term_stats

use std::{collections::BTreeMap, future::Future, str::from_utf8, sync::Arc};

use bytes::Bytes;
use fst::Map;
use thiserror::Error;
use uuid::Uuid;

use crate::{
    storage::StorageProvider,
    superfile::SuperfileReader,
    supertable::manifest::{RoutingRef, SuperfileEntry, part::ContentHash},
    utils::terms::{DictBuilder, make_key},
};

/// Object-store directory prefix for term-stats artifacts, sibling to
/// the superfile data and slow-vector-state prefixes.
pub(crate) const STORAGE_PREFIX: &str = "term-stats/";

/// Artifact magic: identifies the file and its major layout family.
const MAGIC: &[u8; 8] = b"INFTSTA1";
/// Layout version within the magic family; bump on any layout change.
const FORMAT_VERSION: u32 = 1;
/// Header size before the covered-id array: magic + version + count.
const HEADER_FIXED_LEN: usize = 8 + 4 + 4;
/// Terms per `term_dfs` batch while building — bounds the coalesced
/// header-fetch wave and the per-batch scratch.
const BUILD_DF_BATCH_TERMS: usize = 8_192;

#[derive(Debug, Error)]
pub enum TermStatsError {
    #[error("term-stats storage error: {0}")]
    Storage(String),
    #[error("term-stats artifact malformed: {0}")]
    Malformed(String),
    #[error("term-stats artifact hash mismatch")]
    HashMismatch,
    #[error("term-stats build error: {0}")]
    Build(String),
}

/// One decoded term-stats artifact: which superfiles its sums cover,
/// and the `(column, term) → gross df` map.
pub(crate) struct TermStatsSidecar {
    covered: Vec<Uuid>,
    map: Map<Bytes>,
}

impl TermStatsSidecar {
    /// Superfile ids whose dictionaries are summed into this artifact.
    pub(crate) fn covered(&self) -> &[Uuid] {
        &self.covered
    }

    /// Summed gross df for `term` in `column` across the covered set
    /// (0 when the term appears in none of them).
    pub(crate) fn df(&self, column: &str, term: &str) -> u64 {
        self.map.get(make_key(column, term)).unwrap_or(0)
    }

    /// Decode an artifact, verifying layout only (the content hash is
    /// checked against the manifest reference by [`load`]).
    pub(crate) fn decode(bytes: Bytes) -> Result<Self, TermStatsError> {
        let too_short = || TermStatsError::Malformed("truncated header".into());
        if bytes.len() < HEADER_FIXED_LEN {
            return Err(too_short());
        }
        if &bytes[..8] != MAGIC {
            return Err(TermStatsError::Malformed("bad magic".into()));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
        if version != FORMAT_VERSION {
            return Err(TermStatsError::Malformed(format!(
                "unsupported version {version} (expected {FORMAT_VERSION})"
            )));
        }
        let n_covered = u32::from_le_bytes(bytes[12..16].try_into().expect("4 bytes")) as usize;
        let ids_end = HEADER_FIXED_LEN + n_covered * 16;
        if bytes.len() < ids_end {
            return Err(too_short());
        }
        let covered: Vec<Uuid> = bytes[HEADER_FIXED_LEN..ids_end]
            .chunks_exact(16)
            .map(|c| Uuid::from_bytes(c.try_into().expect("16 bytes")))
            .collect();
        let map = Map::new(bytes.slice(ids_end..))
            .map_err(|e| TermStatsError::Malformed(format!("fst: {e}")))?;
        Ok(Self { covered, map })
    }
}

/// Encode an artifact from sorted `(key, df)` entries (dictionary key
/// order — [`DictBuilder`] enforces it) and the covered id set.
fn encode(covered: &[Uuid], entries: &BTreeMap<Vec<u8>, u64>) -> Vec<u8> {
    let mut dict = DictBuilder::new();
    for (key, df) in entries {
        dict.insert(key, *df);
    }
    let fst_bytes = dict.finish();
    let mut out = Vec::with_capacity(HEADER_FIXED_LEN + covered.len() * 16 + fst_bytes.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(covered.len() as u32).to_le_bytes());
    for id in covered {
        out.extend_from_slice(id.as_bytes());
    }
    out.extend_from_slice(&fst_bytes);
    out
}

/// Build the artifact bytes over `readers`: for every FTS column of
/// every superfile, walk its dictionary terms and sum gross df. The df
/// reads are the batched header probes [`SuperfileReader::term_dfs`]
/// performs (one dictionary parse + coalesced header fetches per batch)
/// — no posting bodies are read, which is what makes this a *light*
/// stats-only pass rather than a compaction.
/// `open` is called once per entry and its reader is dropped before the next
/// one opens, so the pass costs one superfile's open-time state rather than
/// the whole table's. That matters because a lazy reader pins its term
/// dictionary for its lifetime — megabytes each — so holding every reader at
/// once made this scale with table size instead of with the work: on a
/// 30,000-superfile table it reached roughly 100 GB and could not run at all.
/// Entries are still visited in order and one at a time, so throughput is
/// unchanged; only the lifetime is.
pub(crate) async fn build<F, Fut>(
    entries: &[Arc<SuperfileEntry>],
    mut open: F,
) -> Result<Vec<u8>, TermStatsError>
where
    F: FnMut(&Arc<SuperfileEntry>) -> Fut,
    Fut: Future<Output = Result<Arc<SuperfileReader>, TermStatsError>>,
{
    let mut merged: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    let mut covered: Vec<Uuid> = Vec::with_capacity(entries.len());
    for entry in entries {
        covered.push(entry.superfile_id);
        let reader = open(entry).await?;
        let Some(fts) = reader.fts() else { continue };
        let columns: Vec<String> = fts.fts_columns_config().map(|c| c.name.clone()).collect();
        for column in &columns {
            let term_bytes = fts
                .iter_column_terms(column)
                .map_err(|e| TermStatsError::Build(format!("term walk: {e}")))?;
            let terms: Vec<&str> = term_bytes
                .iter()
                .map(|t| from_utf8(t).map_err(|_| TermStatsError::Build("non-utf8 term".into())))
                .collect::<Result<_, _>>()?;
            for chunk in terms.chunks(BUILD_DF_BATCH_TERMS) {
                let (dfs, _work) = reader
                    .term_dfs(column, chunk)
                    .await
                    .map_err(|e| TermStatsError::Build(format!("df batch: {e}")))?;
                for (term, df) in chunk.iter().zip(dfs) {
                    *merged.entry(make_key(column, term)).or_insert(0) += df;
                }
            }
        }
    }
    covered.sort_unstable();
    Ok(encode(&covered, &merged))
}

/// Content-address and persist artifact bytes; returns the manifest
/// reference. Idempotent: an object that already exists under its hash
/// name is the same bytes.
pub(crate) async fn write(
    storage: &dyn StorageProvider,
    bytes: Vec<u8>,
) -> Result<RoutingRef, TermStatsError> {
    crate::supertable::writer::put_content_addressed(
        storage,
        |hash| format!("{STORAGE_PREFIX}stats-{}.bin", hash.to_hex()),
        bytes,
    )
    .await
    .map_err(|e| TermStatsError::Storage(e.to_string()))
}

/// Fetch + verify + decode the artifact a manifest references.
pub(crate) async fn load(
    storage: &dyn StorageProvider,
    reference: &RoutingRef,
) -> Result<TermStatsSidecar, TermStatsError> {
    let (bytes, _meta) = storage
        .get(&reference.uri)
        .await
        .map_err(|e| TermStatsError::Storage(e.to_string()))?;
    if ContentHash::of(bytes.as_ref()) != reference.content_hash {
        return Err(TermStatsError::HashMismatch);
    }
    TermStatsSidecar::decode(bytes)
}

#[cfg(test)]
mod tests {
    use std::sync::Weak;

    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use uuid::Uuid as TestUuid;

    use super::*;
    use crate::{
        superfile::{
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
            reader::SuperfileReader,
        },
        supertable::{
            manifest::{SuperfileUri, VectorLayout},
            reader_cache::disk::test_support::tiny_superfile_bytes,
        },
        test_helpers::{decimal128_id_field, decimal128_ids},
    };

    fn entry() -> Arc<SuperfileEntry> {
        Arc::new(SuperfileEntry {
            stem: None,
            birth_version: 0,
            superfile_id: TestUuid::new_v4(),
            uri: SuperfileUri::new_v4(),
            n_docs: 1,
            id_min: 0,
            id_max: 0,
            scalar_stats: Default::default(),
            fts_summary: Default::default(),
            vector_summary: Default::default(),
            partition_key: vec![],
            partition_hint: None,
            vector_layout: VectorLayout::Ivf,
            subsection_offsets: None,
        })
    }

    /// The pass must cost one superfile's open-time state, not the table's.
    /// A reader pins its term dictionary for its lifetime, so holding every
    /// reader at once scaled this with table size; opening them all was what
    /// made the artifact unbuildable on a large table.
    ///
    /// Asserts the lifetime directly: by the time the opener is called for
    /// the next entry, the previous reader must already be unreachable.
    #[tokio::test]
    async fn build_holds_one_reader_at_a_time() {
        let entries: Vec<Arc<SuperfileEntry>> = (0..5).map(|_| entry()).collect();
        let mut previous: Option<Weak<SuperfileReader>> = None;
        let mut opened = 0usize;

        let bytes = build(&entries, |_entry| {
            if let Some(prior) = previous.as_ref() {
                assert!(
                    prior.upgrade().is_none(),
                    "the previous reader must be dropped before the next opens"
                );
            }
            let reader = Arc::new(
                SuperfileReader::open(tiny_superfile_bytes()).expect("open tiny superfile"),
            );
            previous = Some(Arc::downgrade(&reader));
            opened += 1;
            async move { Ok(reader) }
        })
        .await
        .expect("build");

        assert_eq!(opened, entries.len(), "every entry must be visited");
        let sidecar = TermStatsSidecar::decode(Bytes::from(bytes)).expect("decode");
        assert_eq!(
            sidecar.covered().len(),
            entries.len(),
            "the artifact must record every superfile it covers"
        );
    }

    /// Build a superfile carrying an FTS index, so the df walk actually runs.
    /// `tiny_superfile_bytes` has no FTS blob and is skipped by the build.
    fn indexed_superfile_bytes() -> Bytes {
        let schema = Arc::new(Schema::new(vec![
            decimal128_id_field("doc_id"),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(
            Arc::clone(&schema),
            "doc_id",
            vec![FtsConfig::new("title")],
            vec![],
        );
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let ids = decimal128_ids(vec![1u64, 2]);
        let titles = LargeStringArray::from(vec!["rust async runtime", "rust embedded system"]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)]).expect("batch");
        b.add_batch(&batch, &[]).expect("add_batch");
        Bytes::from(b.finish().expect("finish"))
    }

    /// Dropping each reader before the next opens must not lose a
    /// contribution: `df` is the sum over superfiles, so building over N
    /// copies of the same superfile must report N times the single-superfile
    /// figure, and must cover all N.
    #[tokio::test]
    async fn build_sums_df_across_superfiles() {
        let open_indexed = |_e: &Arc<SuperfileEntry>| async {
            SuperfileReader::open(indexed_superfile_bytes())
                .map(Arc::new)
                .map_err(|e| TermStatsError::Build(e.to_string()))
        };

        let one = vec![entry()];
        let side_one = TermStatsSidecar::decode(Bytes::from(
            build(&one, open_indexed).await.expect("build one"),
        ))
        .expect("decode one");
        let df_one = side_one.df("title", "rust");
        assert!(
            df_one > 0,
            "the fixture must contribute a df for the walked term"
        );

        let n = 4;
        let many: Vec<Arc<SuperfileEntry>> = (0..n).map(|_| entry()).collect();
        let side_many = TermStatsSidecar::decode(Bytes::from(
            build(&many, open_indexed).await.expect("build many"),
        ))
        .expect("decode many");

        assert_eq!(
            side_many.df("title", "rust"),
            df_one * n as u64,
            "df must be the sum over superfiles, so every reader's contribution counts"
        );
        assert_eq!(
            side_many.covered().len(),
            n,
            "every superfile must be recorded as covered"
        );
        assert_eq!(side_many.df("title", "absent"), 0);
    }

    /// An opener failure must surface rather than yield a sidecar that
    /// silently claims coverage it never read.
    #[tokio::test]
    async fn build_propagates_an_open_failure() {
        let entries = vec![entry()];
        let result = build(&entries, |_entry| async {
            Err(TermStatsError::Build("open refused".into()))
        })
        .await;
        assert!(
            matches!(result, Err(TermStatsError::Build(m)) if m.contains("open refused")),
            "the opener's error must propagate"
        );
    }

    #[test]
    fn round_trips_covered_ids_and_dfs() {
        let ids = vec![Uuid::from_u128(7), Uuid::from_u128(3)];
        let mut entries = BTreeMap::new();
        entries.insert(make_key("title", "alpha"), 41);
        entries.insert(make_key("title", "beta"), 1);
        entries.insert(make_key("body", "alpha"), 9);
        let bytes = encode(&ids, &entries);
        let side = TermStatsSidecar::decode(Bytes::from(bytes)).expect("decode");
        assert_eq!(side.covered(), ids.as_slice());
        assert_eq!(side.df("title", "alpha"), 41);
        assert_eq!(side.df("title", "beta"), 1);
        assert_eq!(side.df("body", "alpha"), 9);
        assert_eq!(side.df("title", "missing"), 0);
        assert_eq!(side.df("other", "alpha"), 0);
    }

    #[test]
    fn decode_rejects_bad_magic_and_version() {
        let good = encode(&[], &BTreeMap::new());
        let mut bad_magic = good.clone();
        bad_magic[0] ^= 0xFF;
        assert!(matches!(
            TermStatsSidecar::decode(Bytes::from(bad_magic)),
            Err(TermStatsError::Malformed(_))
        ));
        let mut bad_version = good;
        bad_version[8] ^= 0xFF;
        assert!(matches!(
            TermStatsSidecar::decode(Bytes::from(bad_version)),
            Err(TermStatsError::Malformed(_))
        ));
        assert!(matches!(
            TermStatsSidecar::decode(Bytes::from_static(b"short")),
            Err(TermStatsError::Malformed(_))
        ));
    }
}
