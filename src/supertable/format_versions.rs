// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! On-disk format versions of a table's manifest and superfiles.
//!
//! A superfile has three independently versioned layers — the Parquet
//! container (`inf.format_version`), the embedded full-text section header,
//! and the embedded vector section header — and the manifest list that names
//! the superfiles has a version and an options hash of its own. The reader
//! accepts several versions of each layer, so a long-lived table can hold a
//! mix. [`crate::Supertable::format_versions`] reports that mix, per superfile and
//! for the manifest, and says which pieces are behind what the running engine
//! writes today.
//!
//! The report is read-only and takes no writer or compaction slot. It reads
//! the manifest the handle holds (refreshed to the committed one when the
//! table has storage) and, per superfile, only the Parquet footer and the
//! first bytes of each index section header. No dictionary, posting list,
//! IVF block, or Parquet row group is fetched: the cost is a few small ranged
//! reads per superfile.

use std::{collections::BTreeMap, sync::Arc};

use futures::{StreamExt, stream};
use uuid::Uuid;

use super::{
    handle::{Supertable, SupertableInner},
    lazy_source::StorageRangeSource,
    manifest::{ManifestLoadError, SuperfileEntry, options_hash},
};
use crate::{
    runtime_bridge::bridge_on_runtime,
    storage::StorageProvider,
    superfile::{
        LazyByteSource, LazyByteSourceError,
        format::{self, footer, kv},
        reader::{DEFAULT_TAIL_SPECULATIVE_BYTES, vector_layout_from_kv},
        vector::layout::VectorLayout,
    },
    supertable::error::FormatVersionsError,
};

/// Superfile header reads in flight at once while building the report.
const HEADER_READ_CONCURRENCY: usize = 16;

/// Bytes read from the start of an index section to learn its version: the
/// eight-byte magic followed by the little-endian `u32` version word.
const SECTION_VERSION_PROBE_BYTES: u64 = (format::fts::MAGIC_BYTES + format::fts::U32_BYTES) as u64;

/// Which rule the manifest list's stored options hash verifies under.
///
/// The engine recomputes the hash from the table's options on every open and
/// compares it with the stored one. This names the comparison that passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OptionsHashRule {
    /// The stored hash equals the hash the running engine computes.
    Current,
    /// The stored hash is the all-zero sentinel, which skips verification.
    /// Older manifests and synthetic fixtures carry it; a manifest rewritten
    /// by the running engine would carry a real hash instead.
    ZeroSentinel,
}

impl OptionsHashRule {
    /// Stable lower-case name, for logs and language bindings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::ZeroSentinel => "zero_sentinel",
        }
    }
}

/// Format facts about the table's persisted manifest list.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ManifestFormatVersions {
    /// The list's `format_version` as stored, e.g. `"1.0"`.
    pub format_version: String,
    /// Which rule the stored options hash verifies under.
    pub options_hash_rule: OptionsHashRule,
    /// `true` when the list is exactly what the running engine would write:
    /// the current format version and a hash under the current rule.
    pub current: bool,
}

/// Format facts about one superfile.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SuperfileFormatVersions {
    /// The superfile's id, as it appears in its storage path.
    pub superfile_id: String,
    /// `true` when the superfile belongs to the table's derived vector index
    /// rather than to the user's rows. Both are part of the table's storage
    /// and both are reported.
    pub vector_index: bool,
    /// Total size of the superfile in bytes.
    pub size_bytes: u64,
    /// The container's `inf.format_version`, e.g. `"1.1.0"`.
    pub container_version: String,
    /// Version word of the embedded full-text section header, or `None`
    /// when the superfile has no full-text section.
    pub fts_version: Option<u32>,
    /// Version word of the embedded vector section header, or `None` when
    /// the superfile has no vector section. Cell-posting vector blobs carry a
    /// single magic rather than a numbered header and report `None`.
    pub vector_version: Option<u32>,
    /// `true` when the superfile carries the packed stable-id sidecar that
    /// lets `_id` resolve without decoding Parquet id pages.
    pub id_sidecar: bool,
    /// `true` when every layer is what the running engine writes today.
    pub current: bool,
}

/// What [`crate::Supertable::format_versions`] returns.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FormatVersionsReport {
    /// The persisted manifest list, or `None` for an in-process table that
    /// has no persisted manifest.
    pub manifest: Option<ManifestFormatVersions>,
    /// One row per superfile, in manifest order: the user table's superfiles
    /// first, then the derived vector index's.
    pub superfiles: Vec<SuperfileFormatVersions>,
}

impl FormatVersionsReport {
    /// `true` when the manifest (if persisted) and every superfile are
    /// current.
    pub fn is_current(&self) -> bool {
        self.manifest.as_ref().is_none_or(|m| m.current)
            && self.superfiles.iter().all(|row| row.current)
    }

    /// Superfiles with at least one layer behind the current format.
    pub fn stale_superfiles(&self) -> usize {
        self.superfiles.iter().filter(|row| !row.current).count()
    }

    /// Superfile count per container version.
    pub fn container_versions(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for row in &self.superfiles {
            *out.entry(row.container_version.clone()).or_insert(0) += 1;
        }
        out
    }

    /// Superfile count per full-text section version, over superfiles that
    /// have one.
    pub fn fts_versions(&self) -> BTreeMap<u32, usize> {
        let mut out = BTreeMap::new();
        for version in self.superfiles.iter().filter_map(|row| row.fts_version) {
            *out.entry(version).or_insert(0) += 1;
        }
        out
    }

    /// Superfile count per vector section version, over superfiles that
    /// have one.
    pub fn vector_versions(&self) -> BTreeMap<u32, usize> {
        let mut out = BTreeMap::new();
        for version in self.superfiles.iter().filter_map(|row| row.vector_version) {
            *out.entry(version).or_insert(0) += 1;
        }
        out
    }
}

impl Supertable {
    /// Report the on-disk format versions of the table's manifest and of
    /// every superfile, and whether each is what the running engine writes.
    ///
    /// Read-only: takes no writer or compaction slot and is safe to call at
    /// any time. Per superfile it reads the Parquet footer and the first
    /// bytes of each index section header, nothing more.
    pub fn format_versions(&self) -> Result<FormatVersionsReport, FormatVersionsError> {
        bridge_on_runtime(self.format_versions_async(), &self.inner().query_runtime())
    }

    pub(crate) async fn format_versions_async(
        &self,
    ) -> Result<FormatVersionsReport, FormatVersionsError> {
        let inner = self.inner();
        // Reflect the committed table, not this handle's memory of it. An
        // in-process table has no pointer to refresh against.
        if inner.options.storage.is_some() {
            inner.refresh().await?;
        }

        let manifest = inner.manifest.load_full();
        let manifest_report = manifest_format_versions(inner, &manifest)?;

        let mut superfiles = superfile_rows(inner, false).await?;
        if let Some(hidden) = inner.vector_index_table.as_ref() {
            let hidden_inner = hidden.inner();
            if hidden_inner.options.storage.is_some() {
                hidden_inner.refresh().await?;
            }
            superfiles.extend(superfile_rows(hidden_inner, true).await?);
        }

        Ok(FormatVersionsReport {
            manifest: manifest_report,
            superfiles,
        })
    }
}

fn manifest_format_versions(
    inner: &SupertableInner,
    manifest: &super::manifest::ManifestSnapshot,
) -> Result<Option<ManifestFormatVersions>, FormatVersionsError> {
    let Some((format_version, stored_hash, strategy)) = manifest.list_format_identity() else {
        return Ok(None);
    };
    let options_hash_rule = if stored_hash.0 == [0u8; 32] {
        OptionsHashRule::ZeroSentinel
    } else {
        let expected = options_hash::compute_options_hash(&inner.options, strategy);
        if expected.0 == stored_hash.0 {
            OptionsHashRule::Current
        } else {
            // The table opened, so its hash verified then; options cannot
            // change on an open handle. Reaching here means the manifest
            // moved under rules this engine does not compute.
            return Err(FormatVersionsError::Manifest(
                ManifestLoadError::ContentHashMismatch {
                    expected: expected.to_hex(),
                    actual: stored_hash.to_hex(),
                },
            ));
        }
    };
    let current = format_version == super::manifest::list::FORMAT_VERSION
        && options_hash_rule == OptionsHashRule::Current;
    Ok(Some(ManifestFormatVersions {
        format_version: format_version.to_string(),
        options_hash_rule,
        current,
    }))
}

async fn superfile_rows(
    inner: &SupertableInner,
    vector_index: bool,
) -> Result<Vec<SuperfileFormatVersions>, FormatVersionsError> {
    let manifest = inner.manifest.load_full();
    let entries = manifest.get_all_superfiles_loaded().await?;
    let storage = inner.options.storage.clone();
    let store = Arc::clone(&inner.options.store);

    stream::iter(entries)
        .map(|entry| {
            let storage = storage.clone();
            let store = Arc::clone(&store);
            async move {
                let source = byte_source_for(&entry, storage.as_ref(), store.as_ref())?;
                superfile_row(entry.superfile_id, source.as_ref(), vector_index).await
            }
        })
        .buffered(HEADER_READ_CONCURRENCY)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect()
}

/// A byte source for one superfile that costs nothing to construct. With
/// storage attached this is a ranged reader over the object; an in-process
/// table serves the bytes it already holds.
fn byte_source_for(
    entry: &SuperfileEntry,
    storage: Option<&Arc<dyn StorageProvider>>,
    store: &dyn super::reader_cache::SuperfileReaderCache,
) -> Result<Arc<dyn LazyByteSource>, FormatVersionsError> {
    match storage {
        Some(storage) => {
            let path = entry.uri.storage_path();
            let known_size = entry
                .subsection_offsets
                .as_ref()
                .map(|o| o.total_size)
                .filter(|&size| size > 0);
            Ok(match known_size {
                Some(size) => Arc::new(StorageRangeSource::with_known_size(
                    Arc::clone(storage),
                    path,
                    size,
                )),
                None => Arc::new(StorageRangeSource::with_unknown_size(
                    Arc::clone(storage),
                    path,
                )),
            })
        }
        None => store
            .reader(&entry.uri)
            .map(|reader| reader.byte_source())
            .map_err(|e| superfile_error(entry.superfile_id, e.to_string())),
    }
}

/// One report row from a superfile's bytes: the Parquet footer and the
/// twelve-byte magic-plus-version probe of each index section present.
async fn superfile_row(
    superfile_id: Uuid,
    source: &dyn LazyByteSource,
    vector_index: bool,
) -> Result<SuperfileFormatVersions, FormatVersionsError> {
    let metadata = footer::read_parquet_metadata_lazy(source, DEFAULT_TAIL_SPECULATIVE_BYTES)
        .await
        .map_err(|e| superfile_error(superfile_id, e.to_string()))?;
    let kv_map = footer::extract_kv_map(&metadata)
        .map_err(|e| superfile_error(superfile_id, e.to_string()))?;
    let size_bytes = source.size();

    let container_version = kv_map
        .get(kv::FORMAT_VERSION)
        .cloned()
        .ok_or_else(|| superfile_error(superfile_id, format!("missing {}", kv::FORMAT_VERSION)))?;

    let fts_version = match section_start(&kv_map, kv::FTS_KEYS, kv::FTS_OFFSET, kv::FTS_LENGTH)
        .map_err(|reason| superfile_error(superfile_id, reason))?
    {
        Some(start) => Some(
            section_version(source, start, format::fts::MAGIC, "full-text")
                .await
                .map_err(|reason| superfile_error(superfile_id, reason))?,
        ),
        None => None,
    };

    let vector_layout = vector_layout_from_kv(&kv_map);
    let vector_version = match section_start(&kv_map, kv::VEC_KEYS, kv::VEC_OFFSET, kv::VEC_LENGTH)
        .map_err(|reason| superfile_error(superfile_id, reason))?
    {
        Some(_) if vector_layout == VectorLayout::CellPosting => None,
        Some(start) => Some(
            section_version(source, start, format::vec::OUTER_MAGIC, "vector")
                .await
                .map_err(|reason| superfile_error(superfile_id, reason))?,
        ),
        None => None,
    };

    let id_sidecar = kv::IDS_KEYS.iter().all(|k| kv_map.contains_key(*k));

    let container_current = container_version == format::FORMAT_VERSION;
    let fts_current = fts_version.is_none_or(|v| v == format::fts::VERSION_V5);
    // The vector layout recorded in the footer decides which header the
    // builder writes: the single-column header for `ivf` blobs and the
    // multi-cell header for `multi_cell_ivf` blobs. Both are current for
    // their layout, so the verdict is per layout.
    let vector_current = match (vector_layout, vector_version) {
        (_, None) => true,
        (VectorLayout::MultiCellIvf, Some(v)) => v == format::vec::VERSION_MULTI_CELL,
        (_, Some(v)) => v == format::vec::VERSION,
    };

    Ok(SuperfileFormatVersions {
        superfile_id: superfile_id.to_string(),
        vector_index,
        size_bytes,
        container_version,
        fts_version,
        vector_version,
        id_sidecar,
        current: container_current && fts_current && vector_current,
    })
}

/// Start offset of an index section named by an all-or-none KV key set, or
/// `None` when the section is absent. A partial key set is malformed.
fn section_start(
    kv_map: &footer::KvMap,
    keys: &[&str],
    offset_key: &'static str,
    length_key: &'static str,
) -> Result<Option<u64>, String> {
    let present = keys.iter().filter(|k| kv_map.contains_key(**k)).count();
    if present == 0 {
        return Ok(None);
    }
    if present != keys.len() {
        return Err(format!(
            "partial {} keys present",
            keys[0]
                .rsplit_once('.')
                .map_or(keys[0], |(prefix, _)| prefix)
        ));
    }
    let offset = parse_u64(kv_map, offset_key)?;
    let length = parse_u64(kv_map, length_key)?;
    if length < SECTION_VERSION_PROBE_BYTES {
        return Err(format!("{offset_key} section too short to carry a header"));
    }
    Ok(Some(offset))
}

fn parse_u64(kv_map: &footer::KvMap, key: &'static str) -> Result<u64, String> {
    kv_map
        .get(key)
        .ok_or_else(|| format!("missing {key}"))?
        .parse()
        .map_err(|_| format!("{key} not a u64"))
}

/// Read a section's magic and version word.
async fn section_version(
    source: &dyn LazyByteSource,
    start: u64,
    magic: &[u8; 8],
    section: &str,
) -> Result<u32, String> {
    let probe = source
        .range(start, SECTION_VERSION_PROBE_BYTES)
        .await
        .map_err(|e: LazyByteSourceError| e.to_string())?;
    if &probe[..format::fts::MAGIC_BYTES] != magic {
        return Err(format!("{section} section magic mismatch"));
    }
    let word: [u8; 4] = probe[format::fts::MAGIC_BYTES..]
        .try_into()
        .map_err(|_| format!("{section} section header truncated"))?;
    Ok(u32::from_le_bytes(word))
}

fn superfile_error(superfile_id: Uuid, reason: String) -> FormatVersionsError {
    FormatVersionsError::Superfile {
        superfile_id: superfile_id.to_string(),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use uuid::Uuid;

    use super::*;
    use crate::{
        superfile::{
            BytesLazyByteSource,
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
            format,
        },
        supertable::Supertable,
        test_helpers::{
            build_title_batch, decimal128_id_field, decimal128_ids, default_supertable_options,
        },
    };

    /// Superfile bytes with one full-text column and no vectors.
    fn fts_superfile_bytes() -> Bytes {
        let schema: Arc<Schema> = Arc::new(Schema::new(vec![
            decimal128_id_field("doc_id"),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(
            schema.clone(),
            "doc_id",
            vec![FtsConfig::new("title")],
            vec![],
        );
        let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
        let ids = decimal128_ids(vec![1u64, 2, 3]);
        let title = LargeStringArray::from(vec!["alpha", "beta", "gamma"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title)])
            .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
        Bytes::from(b.finish().expect("finish builder"))
    }

    fn row_for(bytes: Bytes) -> SuperfileFormatVersions {
        let source = BytesLazyByteSource::new(bytes);
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime");
        rt.block_on(superfile_row(Uuid::new_v4(), &source, false))
            .expect("row")
    }

    #[test]
    fn freshly_built_superfile_is_current_on_every_layer() {
        let bytes = fts_superfile_bytes();
        let row = row_for(bytes.clone());
        assert_eq!(row.container_version, format::FORMAT_VERSION);
        assert_eq!(row.fts_version, Some(format::fts::VERSION_V5));
        assert_eq!(row.vector_version, None);
        assert_eq!(row.size_bytes, bytes.len() as u64);
        assert!(row.current, "{row:?}");
    }

    #[test]
    fn older_fts_header_word_marks_the_superfile_stale() {
        // Patch only the version word of the full-text header. The report
        // reads that word and nothing else in the section, so the rest of
        // the blob need not be a valid older layout.
        let bytes = fts_superfile_bytes();
        let kv = footer::read_kv_metadata(&bytes).expect("kv");
        let fts_off: usize = kv[kv::FTS_OFFSET].parse().expect("fts offset");
        let mut patched = bytes.to_vec();
        let word = fts_off + format::fts::hdr::VERSION_OFF;
        patched[word..word + format::fts::U32_BYTES]
            .copy_from_slice(&format::fts::VERSION_V4.to_le_bytes());

        let row = row_for(Bytes::from(patched));
        assert_eq!(row.fts_version, Some(format::fts::VERSION_V4));
        assert!(!row.current, "{row:?}");
    }

    #[test]
    fn in_process_table_reports_every_superfile_and_no_manifest() {
        let st = Supertable::create(default_supertable_options()).expect("create");
        for titles in [&["alpha", "beta"][..], &["gamma"][..]] {
            let mut w = st.writer().expect("writer");
            w.append(&build_title_batch(titles)).expect("append");
            w.commit().expect("commit");
        }

        let report = st.format_versions().expect("format_versions");
        assert!(
            report.manifest.is_none(),
            "an in-process table has no persisted manifest list"
        );
        assert_eq!(report.superfiles.len(), 2);
        assert!(report.superfiles.iter().all(|r| !r.vector_index));
        assert!(report.is_current(), "{report:?}");
        assert_eq!(report.stale_superfiles(), 0);
        assert_eq!(
            report.fts_versions(),
            BTreeMap::from([(format::fts::VERSION_V5, 2)])
        );
        assert_eq!(
            report.container_versions(),
            BTreeMap::from([(format::FORMAT_VERSION.to_string(), 2)])
        );
        assert!(report.vector_versions().is_empty());
    }

    #[test]
    fn report_helpers_summarize_rows() {
        let row = |fts: Option<u32>, current: bool| SuperfileFormatVersions {
            superfile_id: Uuid::new_v4().to_string(),
            vector_index: false,
            size_bytes: 1,
            container_version: format::FORMAT_VERSION.to_string(),
            fts_version: fts,
            vector_version: None,
            id_sidecar: true,
            current,
        };
        let report = FormatVersionsReport {
            manifest: Some(ManifestFormatVersions {
                format_version: "1.0".into(),
                options_hash_rule: OptionsHashRule::ZeroSentinel,
                current: false,
            }),
            superfiles: vec![
                row(Some(format::fts::VERSION_V5), true),
                row(Some(format::fts::VERSION_V3), false),
            ],
        };
        assert!(!report.is_current());
        assert_eq!(report.stale_superfiles(), 1);
        assert_eq!(
            report.fts_versions(),
            BTreeMap::from([(format::fts::VERSION_V3, 1), (format::fts::VERSION_V5, 1)])
        );
        assert_eq!(OptionsHashRule::ZeroSentinel.as_str(), "zero_sentinel");
    }
}
