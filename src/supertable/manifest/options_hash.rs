//! Canonical digest of `SupertableOptions` (D15).
//!
//! [`compute_options_hash`] produces a deterministic
//! `ContentHash` over the load-bearing options fields — the
//! Arrow schema, id column, FTS / vector column declarations,
//! and the resolved partition strategy. Stamped onto
//! [`ManifestList::options_hash`] at commit time; verified
//! at [`Supertable::open`] against the caller's options so a
//! schema mismatch surfaces as a clean
//! [`OpenError::OptionsHashMismatch`] instead of a parquet /
//! arrow decode failure on first query.
//!
//! ## Encoding
//!
//! Hand-rolled length-prefixed byte stream, blake3'd. Each
//! field is preceded by a fixed string tag so two
//! structurally-different shapes with overlapping byte
//! patterns can't collide:
//!
//! ```text
//! "schema"       | n_fields u64 | for each field: name_len u64 | name | dt_str_len u64 | dt_str | nullable u8
//! "id_column"    | len u64 | bytes
//! "fts_columns"  | count u64 | for each: name_len u64 | name
//! "vector_columns" | count u64 | for each: name_len u64 | name | dim u64 | n_cent u64 | rot_seed u64 | metric_len u64 | metric_str
//! "partition_strategy" | variant_tag | per-variant fields
//! ```
//!
//! Determinism is bounded by:
//! - [`arrow_schema::DataType::Debug`] formatting (stable
//!   across arrow patch versions; a minor-version Debug
//!   format change would invalidate the hash — accepted
//!   trade-off vs Arrow IPC's larger encoding surface).
//! - `format!("{:?}", metric).to_lowercase()` for vector
//!   metric — matches the same encoding the manifest list's
//!   `VectorColumnInfo.metric` uses, so list ⇄ hash stay in
//!   lockstep.
//!
//! Legacy / synthetic-manifest escape hatch: a stored
//! `options_hash` of all zeros is treated as "validation
//! skipped" by [`verify_options_hash`] — pre-D15 manifests
//! + test fixtures that construct lists manually keep
//! opening cleanly.
//!
//! [`ManifestList::options_hash`]: super::list::ManifestList
//! [`Supertable::open`]: crate::supertable::Supertable::open
//! [`OpenError::OptionsHashMismatch`]: crate::supertable::OpenError::OptionsHashMismatch

use crate::supertable::manifest::list::PartitionStrategy;
use crate::supertable::manifest::part::ContentHash;
use crate::supertable::options::SupertableOptions;

/// Compute the canonical options-hash from `opts` + the
/// resolved `strategy`. See the module-level docs for the
/// encoding layout.
pub fn compute_options_hash(opts: &SupertableOptions, strategy: &PartitionStrategy) -> ContentHash {
    let mut buf: Vec<u8> = Vec::with_capacity(256);

    // 1. schema (field-by-field).
    push_tag(&mut buf, b"schema");
    let fields = opts.schema.fields();
    buf.extend_from_slice(&(fields.len() as u64).to_le_bytes());
    for f in fields {
        push_str(&mut buf, f.name());
        let dt_str = format!("{:?}", f.data_type());
        push_str(&mut buf, &dt_str);
        buf.push(f.is_nullable() as u8);
    }

    // 2. id_column.
    push_tag(&mut buf, b"id_column");
    push_str(&mut buf, &opts.id_column);

    // 3. fts_columns (declared order — order is part of the
    //    schema identity since FtsBuilder assigns column ids
    //    by position).
    push_tag(&mut buf, b"fts_columns");
    buf.extend_from_slice(&(opts.fts_columns.len() as u64).to_le_bytes());
    for c in &opts.fts_columns {
        push_str(&mut buf, &c.column);
    }

    // 4. vector_columns (same declared-order rationale).
    push_tag(&mut buf, b"vector_columns");
    buf.extend_from_slice(&(opts.vector_columns.len() as u64).to_le_bytes());
    for v in &opts.vector_columns {
        push_str(&mut buf, &v.column);
        buf.extend_from_slice(&(v.dim as u64).to_le_bytes());
        buf.extend_from_slice(&(v.n_cent as u64).to_le_bytes());
        buf.extend_from_slice(&v.rot_seed.to_le_bytes());
        // Match the manifest list's metric encoding
        // (`VectorColumnInfo.metric` writer site) — lowercased
        // Debug form — so the hash stays in lockstep.
        let metric_str = format!("{:?}", v.metric).to_lowercase();
        push_str(&mut buf, &metric_str);
    }

    // 5. partition_strategy.
    push_tag(&mut buf, b"partition_strategy");
    match strategy {
        PartitionStrategy::TimeRange {
            column,
            granularity_secs,
        } => {
            push_tag(&mut buf, b"time_range");
            push_str(&mut buf, column);
            buf.extend_from_slice(&granularity_secs.to_le_bytes());
        }
        PartitionStrategy::Hash { column, n_buckets } => {
            push_tag(&mut buf, b"hash");
            push_str(&mut buf, column);
            buf.extend_from_slice(&n_buckets.to_le_bytes());
        }
        PartitionStrategy::ColumnRange { column, boundaries } => {
            push_tag(&mut buf, b"column_range");
            push_str(&mut buf, column);
            buf.extend_from_slice(&(boundaries.len() as u64).to_le_bytes());
            for b in boundaries {
                buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
                buf.extend_from_slice(b);
            }
        }
    }

    let h = blake3::hash(&buf);
    ContentHash(*h.as_bytes())
}

/// Compare `expected` (caller-side recomputed from current
/// options) against `actual` (stored on the manifest list).
///
/// Returns `Ok(())` if the two match, OR if `actual` is the
/// all-zero sentinel (pre-D15 manifests + synthetic test
/// fixtures bypass validation).
pub fn verify_options_hash(
    expected: ContentHash,
    actual: ContentHash,
) -> Result<(), OptionsHashMismatch> {
    if actual.0 == [0u8; 32] {
        // Legacy / synthetic — skip validation.
        return Ok(());
    }
    if expected.0 == actual.0 {
        return Ok(());
    }
    Err(OptionsHashMismatch {
        expected: expected.to_hex(),
        actual: actual.to_hex(),
    })
}

/// Mismatch between the caller's options-derived hash and
/// the manifest list's stored hash. Carries hex strings so
/// the variant's `Display` impl can render them without
/// pulling the raw bytes into the public error surface.
#[derive(Debug, Clone)]
pub struct OptionsHashMismatch {
    pub expected: String,
    pub actual: String,
}

impl std::fmt::Display for OptionsHashMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "options_hash mismatch: caller=blake3:{} list=blake3:{}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for OptionsHashMismatch {}

#[inline]
fn push_tag(buf: &mut Vec<u8>, tag: &[u8]) {
    // Tags are short string literals controlled by this
    // crate, not user input, so we don't bother with the
    // length prefix the variable-length string fields use.
    buf.push(tag.len() as u8);
    buf.extend_from_slice(tag);
}

#[inline]
fn push_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}
