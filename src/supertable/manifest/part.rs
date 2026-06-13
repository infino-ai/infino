// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `ManifestPart` — one node of the two-tier manifest.
//!
//! A part is a bounded collection of `SuperfileEntry` records,
//! serialized as Avro inside a zstd frame and addressed by
//! the blake3 hash of the compressed bytes. The blake3 hash
//! is the part's URI (modulo backend prefix); content
//! addressing means commits that don't change a part's
//! superfile set reuse it across manifest versions without a
//! re-PUT.

use std::collections::HashMap;
use std::sync::Arc;

use apache_avro::Schema as AvroSchema;
use apache_avro::types::Value as AvroValue;
use apache_avro::{from_avro_datum, to_avro_datum};
use thiserror::Error;
use uuid::Uuid;

use crate::supertable::manifest::encoding::{
    self, DecodeError, decode_fts_summary_map, decode_scalar_stats, decode_vector_summary_map,
    encode_fts_summary_map, encode_scalar_stats, encode_vector_summary_map,
};
use crate::supertable::manifest::{SubsectionOffsets, SuperfileEntry};

/// The format version stamped into every emitted part.
///
/// Major-version-incompatible readers must reject; minor-
/// version-newer readers must ignore unknown minor fields
/// (see [`PartParseError::IncompatibleMajorVersion`]). The
/// supported range is `>=1.0 <2.0`.
pub const FORMAT_VERSION: &str = "1.0";

/// Blake3 digest width in bytes. Blake3 emits a 256-bit (32-byte)
/// digest; this is the length of a [`ContentHash`]'s payload and the
/// buffer size when decoding one from hex.
pub(crate) const BLAKE3_DIGEST_BYTES: usize = 32;

/// Length of a Blake3 digest in lowercase hex — two characters per
/// digest byte. Used to pre-size and validate `blake3:<hex>` strings.
pub(crate) const BLAKE3_HEX_LEN: usize = BLAKE3_DIGEST_BYTES * 2;

/// Number of leading hex characters of a content hash shown in
/// `Debug` output, trading full identification for readable logs.
const CONTENT_HASH_DEBUG_HEX_PREFIX_LEN: usize = 8;

/// Width of a little-endian `u32` field in the packed
/// subsection-offsets blob.
const U32_BYTES: usize = 4;
/// Width of a little-endian `u64` field in the packed
/// subsection-offsets blob.
const U64_BYTES: usize = 8;

/// Current `subsection_offsets` encoding version. Version 3 appends
/// the inline open-batch blob; version 2 (still accepted on read)
/// has none.
const SUBSECTION_OFFSETS_VERSION_CURRENT: u8 = 3;
/// Oldest `subsection_offsets` encoding version still accepted on
/// read (no open-batch blob).
const SUBSECTION_OFFSETS_VERSION_LEGACY: u8 = 2;

/// Presence flag byte: the optional subsection (vec or fts) is
/// present and its `(offset, length)` pair follows.
const SUBSECTION_FLAG_PRESENT: u8 = 1;
/// Presence flag byte: the optional subsection is absent.
const SUBSECTION_FLAG_ABSENT: u8 = 0;

/// Width of the Avro `fixed` id columns (`id_min` / `id_max`): a
/// 128-bit id stored big-endian as 16 bytes. Must match the `size`
/// in the Avro schema string.
const ID_COLUMN_FIXED_BYTES: usize = 16;

/// Avro optional-field union arm index for the `null` branch.
const AVRO_UNION_NULL_INDEX: u32 = 0;
/// Avro optional-field union arm index for the present-value branch.
const AVRO_UNION_VALUE_INDEX: u32 = 1;

/// Content hash of a manifest part — blake3 of the
/// compressed (zstd) Avro bytes. The hex form is the URI
/// suffix used in the storage layer.
///
/// Two parts with identical byte content always have
/// identical `ContentHash` — that's the property the
/// "reuse-by-uri across manifest versions" optimization
/// rides on.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ContentHash(pub [u8; BLAKE3_DIGEST_BYTES]);

impl ContentHash {
    /// Hash a byte slice.
    pub fn of(bytes: &[u8]) -> Self {
        let hash = blake3::hash(bytes);
        Self(*hash.as_bytes())
    }

    /// Hex representation, lower-case, 64 chars.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_hex(&self) -> String {
        let mut out = String::with_capacity(BLAKE3_HEX_LEN);
        for byte in self.0 {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }
}

impl std::fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Show only the first 8 hex chars in Debug to keep
        // logs readable. Use `to_hex()` for the full form.
        write!(
            f,
            "blake3:{}…",
            &self.to_hex()[..CONTENT_HASH_DEBUG_HEX_PREFIX_LEN]
        )
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "blake3:{}", self.to_hex())
    }
}

/// Identifier for a manifest part. UUID v4 (random); not
/// derived from content hash so part-id stays stable while
/// the bytes evolve under it. (Content addressing operates
/// at the URI level, not the part-id level.)
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct PartId(pub Uuid);

impl PartId {
    pub fn new_v4() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for PartId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// One node of the hierarchical manifest. Holds a bounded
/// set of `SuperfileEntry`s (default cap: 10K per part).
///
/// `ManifestPart` is the in-memory shape. The wire shape is
/// the Avro DTO emitted by [`encode`] and consumed by
/// [`decode`]. Reader-pinning semantics: parts are immutable
/// once written — content-addressing makes that invariant
/// load-bearing.
#[derive(Debug, Clone)]
pub struct ManifestPart {
    /// Format version of the part. Set to [`FORMAT_VERSION`]
    /// at encode time; verified at decode time.
    pub format_version: String,
    /// Identifier for this part — UUID v4, **not** derived
    /// from content.
    pub part_id: PartId,
    /// The superfiles this part references. Order is
    /// preserved across encode/decode for determinism
    /// (content addressing requires bit-stable output).
    pub superfiles: Vec<Arc<SuperfileEntry>>,
}

/// Errors from the Avro+zstd decode path.
#[derive(Debug, Error)]
pub enum PartParseError {
    #[error("zstd decompress failed: {0}")]
    Zstd(String),
    #[error("avro decode failed: {0}")]
    Avro(String),
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),
    #[error("per-summary decode failed: {0}")]
    SummaryDecode(#[from] DecodeError),
    #[error("malformed superfile_id uuid: {0}")]
    BadSuperfileId(String),
    #[error("incompatible major version: got {got}, supported {supported}")]
    IncompatibleMajorVersion { got: String, supported: String },
    #[error("missing field: {0}")]
    MissingField(&'static str),
    #[error("wrong avro field type for {0}")]
    WrongFieldType(&'static str),
}

/// The Avro schema for a `ManifestPart`.
///
/// Kept in one place so encoder + decoder stay in sync. The
/// schema is parsed once on first use and cached via
/// `std::sync::OnceLock`.
fn schema() -> &'static AvroSchema {
    use std::sync::OnceLock;
    static SCHEMA: OnceLock<AvroSchema> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        let schema_str = r#"
        {
          "type": "record",
          "name": "ManifestPart",
          "fields": [
            {"name": "format_version", "type": "string"},
            {"name": "part_id", "type": "string"},
            {"name": "superfiles", "type": {"type": "array", "items": {
              "type": "record",
              "name": "SuperfileEntry",
              "fields": [
                {"name": "superfile_id", "type": "string"},
                {"name": "uri", "type": "string"},
                {"name": "n_docs", "type": "long"},
                {"name": "id_min", "type": {"type": "fixed", "name": "IdMin", "size": 16}},
                {"name": "id_max", "type": {"type": "fixed", "name": "IdMax", "size": 16}},
                {"name": "partition_key", "type": "bytes"},
                {"name": "partition_hint", "type": ["null", "int"], "default": null},
                {"name": "scalar_stats", "type": "bytes"},
                {"name": "fts_summary", "type": "bytes"},
                {"name": "vector_summary", "type": "bytes"},
                {"name": "subsection_offsets", "type": ["null", "bytes"], "default": null}
              ]
            }}}
          ]
        }
        "#;
        AvroSchema::parse_str(schema_str).expect("ManifestPart Avro schema parses")
    })
}

/// Encode a [`ManifestPart`] to Avro bytes wrapped in a zstd
/// frame, returning the bytes + their `ContentHash`.
///
/// The hash is the blake3 of the **compressed** bytes — the
/// URI uses the same form, so a re-write of bit-identical
/// content produces the same URI (the load-bearing property
/// for cross-version part sharing).
///
/// `zstd_level` is the compression level (1..=22); v1 default
/// is 3 (matches Iceberg's manifest-file default; good
/// time/space trade for sub-MB Avro payloads).
pub fn encode(part: &ManifestPart, zstd_level: i32) -> Vec<u8> {
    // Use schemaless Avro datum encoding (no OCF container).
    // The OCF wrapper carries a random 16-byte sync marker, which
    // would break content-addressing: encoding the same logical
    // part twice would produce different bytes → different
    // blake3 → different URI. Iceberg manifest files take the
    // same approach for the same reason.
    let superfile_records: Vec<AvroValue> = part
        .superfiles
        .iter()
        .map(|seg| {
            let scalar_bytes = encode_scalar_stats(&seg.scalar_stats);
            let fts_bytes = encode_fts_summary_map(&seg.fts_summary);
            let vector_bytes = encode_vector_summary_map(&seg.vector_summary);

            AvroValue::Record(vec![
                (
                    "superfile_id".into(),
                    AvroValue::String(seg.superfile_id.to_string()),
                ),
                ("uri".into(), AvroValue::String(seg.uri.0.to_string())),
                ("n_docs".into(), AvroValue::Long(seg.n_docs as i64)),
                (
                    "id_min".into(),
                    AvroValue::Fixed(ID_COLUMN_FIXED_BYTES, seg.id_min.to_be_bytes().to_vec()),
                ),
                (
                    "id_max".into(),
                    AvroValue::Fixed(ID_COLUMN_FIXED_BYTES, seg.id_max.to_be_bytes().to_vec()),
                ),
                (
                    "partition_key".into(),
                    AvroValue::Bytes(seg.partition_key.clone()),
                ),
                (
                    "partition_hint".into(),
                    match seg.partition_hint {
                        Some(b) => AvroValue::Union(
                            AVRO_UNION_VALUE_INDEX,
                            Box::new(AvroValue::Int(b as i32)),
                        ),
                        None => AvroValue::Union(AVRO_UNION_NULL_INDEX, Box::new(AvroValue::Null)),
                    },
                ),
                ("scalar_stats".into(), AvroValue::Bytes(scalar_bytes)),
                ("fts_summary".into(), AvroValue::Bytes(fts_bytes)),
                ("vector_summary".into(), AvroValue::Bytes(vector_bytes)),
                (
                    "subsection_offsets".into(),
                    match &seg.subsection_offsets {
                        Some(off) => AvroValue::Union(
                            AVRO_UNION_VALUE_INDEX,
                            Box::new(AvroValue::Bytes(encode_subsection_offsets(off))),
                        ),
                        None => AvroValue::Union(AVRO_UNION_NULL_INDEX, Box::new(AvroValue::Null)),
                    },
                ),
            ])
        })
        .collect();

    let record = AvroValue::Record(vec![
        (
            "format_version".into(),
            AvroValue::String(part.format_version.clone()),
        ),
        (
            "part_id".into(),
            AvroValue::String(part.part_id.0.to_string()),
        ),
        ("superfiles".into(), AvroValue::Array(superfile_records)),
    ]);

    let avro_bytes = to_avro_datum(schema(), record).expect("avro datum encode");
    zstd::stream::encode_all(avro_bytes.as_slice(), zstd_level).expect("zstd encode")
}

/// Decode a manifest-part byte buffer (zstd-wrapped Avro)
/// back into a [`ManifestPart`].
///
/// Verifies format-version compatibility (major must match
/// the constant [`FORMAT_VERSION`]; minor differences are
/// accepted).
pub fn decode(bytes: &[u8]) -> Result<ManifestPart, PartParseError> {
    let avro_bytes =
        zstd::stream::decode_all(bytes).map_err(|e| PartParseError::Zstd(e.to_string()))?;
    // Schemaless datum decode — mirrors `to_avro_datum` in
    // `encode`. The schema is in-source (compiled in), so the
    // reader doesn't need a wire-side schema.
    let mut cursor = std::io::Cursor::new(avro_bytes.as_slice());
    let value = from_avro_datum(schema(), &mut cursor, None)
        .map_err(|e| PartParseError::Avro(e.to_string()))?;

    let fields = match value {
        AvroValue::Record(r) => r,
        _ => {
            return Err(PartParseError::SchemaMismatch(
                "top-level not a record".into(),
            ));
        }
    };
    let mut map: HashMap<String, AvroValue> = fields.into_iter().collect();

    let format_version = take_string(&mut map, "format_version")?;
    check_major(&format_version)?;

    let part_id_str = take_string(&mut map, "part_id")?;
    let part_id = PartId(
        Uuid::parse_str(&part_id_str).map_err(|e| PartParseError::BadSuperfileId(e.to_string()))?,
    );

    let superfiles_val = map
        .remove("superfiles")
        .ok_or(PartParseError::MissingField("superfiles"))?;
    let segs = match superfiles_val {
        AvroValue::Array(a) => a,
        _ => return Err(PartParseError::WrongFieldType("superfiles")),
    };
    let mut superfiles = Vec::with_capacity(segs.len());
    for seg_val in segs {
        superfiles.push(Arc::new(decode_superfile(seg_val)?));
    }

    Ok(ManifestPart {
        format_version,
        part_id,
        superfiles,
    })
}

fn decode_superfile(v: AvroValue) -> Result<SuperfileEntry, PartParseError> {
    let fields = match v {
        AvroValue::Record(r) => r,
        _ => {
            return Err(PartParseError::SchemaMismatch(
                "superfile not a record".into(),
            ));
        }
    };
    let mut map: HashMap<String, AvroValue> = fields.into_iter().collect();

    let superfile_id = Uuid::parse_str(&take_string(&mut map, "superfile_id")?)
        .map_err(|e| PartParseError::BadSuperfileId(e.to_string()))?;
    let uri = Uuid::parse_str(&take_string(&mut map, "uri")?)
        .map_err(|e| PartParseError::BadSuperfileId(e.to_string()))?;
    let n_docs = take_long(&mut map, "n_docs")? as u64;
    let id_min = take_i128_be(&mut map, "id_min")?;
    let id_max = take_i128_be(&mut map, "id_max")?;
    let partition_key = take_bytes(&mut map, "partition_key")?;
    let partition_hint = take_optional_int(&mut map, "partition_hint")?.map(|i| i as u32);
    let scalar_bytes = take_bytes(&mut map, "scalar_stats")?;
    let fts_bytes = take_bytes(&mut map, "fts_summary")?;
    let vector_bytes = take_bytes(&mut map, "vector_summary")?;

    // `subsection_offsets` lands as a separate
    // optional bytes field below. Parsed if present; defaulted to
    // None for older manifests so old parts decode losslessly
    // (the cold-open path falls back to the 2-RTT shape).
    let subsection_offsets = take_optional_bytes(&mut map, "subsection_offsets")?
        .map(|b| decode_subsection_offsets(&b))
        .transpose()?;

    Ok(SuperfileEntry {
        superfile_id,
        uri: crate::supertable::manifest::SuperfileUri(uri),
        n_docs,
        id_min,
        id_max,
        scalar_stats: decode_scalar_stats(&scalar_bytes)?,
        fts_summary: decode_fts_summary_map(&fts_bytes)?,
        vector_summary: decode_vector_summary_map(&vector_bytes)?,
        partition_key,
        partition_hint,
        subsection_offsets,
    })
}

fn check_major(fv: &str) -> Result<(), PartParseError> {
    let supported_major = FORMAT_VERSION
        .split('.')
        .next()
        .expect("constant has a dot");
    let got_major = fv.split('.').next().unwrap_or("");
    if got_major != supported_major {
        return Err(PartParseError::IncompatibleMajorVersion {
            got: fv.to_string(),
            supported: FORMAT_VERSION.to_string(),
        });
    }
    Ok(())
}

fn take_string(
    map: &mut HashMap<String, AvroValue>,
    name: &'static str,
) -> Result<String, PartParseError> {
    match map.remove(name).ok_or(PartParseError::MissingField(name))? {
        AvroValue::String(s) => Ok(s),
        _ => Err(PartParseError::WrongFieldType(name)),
    }
}

fn take_long(
    map: &mut HashMap<String, AvroValue>,
    name: &'static str,
) -> Result<i64, PartParseError> {
    match map.remove(name).ok_or(PartParseError::MissingField(name))? {
        AvroValue::Long(v) => Ok(v),
        _ => Err(PartParseError::WrongFieldType(name)),
    }
}

fn take_bytes(
    map: &mut HashMap<String, AvroValue>,
    name: &'static str,
) -> Result<Vec<u8>, PartParseError> {
    match map.remove(name).ok_or(PartParseError::MissingField(name))? {
        AvroValue::Bytes(b) => Ok(b),
        _ => Err(PartParseError::WrongFieldType(name)),
    }
}

fn take_i128_be(
    map: &mut HashMap<String, AvroValue>,
    name: &'static str,
) -> Result<i128, PartParseError> {
    let bytes = match map.remove(name).ok_or(PartParseError::MissingField(name))? {
        AvroValue::Fixed(ID_COLUMN_FIXED_BYTES, b) => b,
        _ => return Err(PartParseError::WrongFieldType(name)),
    };
    let arr: [u8; ID_COLUMN_FIXED_BYTES] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| PartParseError::WrongFieldType(name))?;
    Ok(i128::from_be_bytes(arr))
}

fn take_optional_int(
    map: &mut HashMap<String, AvroValue>,
    name: &'static str,
) -> Result<Option<i32>, PartParseError> {
    match map.remove(name).ok_or(PartParseError::MissingField(name))? {
        AvroValue::Union(_, boxed) => match *boxed {
            AvroValue::Null => Ok(None),
            AvroValue::Int(v) => Ok(Some(v)),
            _ => Err(PartParseError::WrongFieldType(name)),
        },
        AvroValue::Null => Ok(None),
        AvroValue::Int(v) => Ok(Some(v)),
        _ => Err(PartParseError::WrongFieldType(name)),
    }
}

/// Pull an optional bytes field. Missing-key returns `Ok(None)` so
/// new schema fields stay backward-compatible with parts emitted
/// before they were added (e.g. `subsection_offsets`).
fn take_optional_bytes(
    map: &mut HashMap<String, AvroValue>,
    name: &'static str,
) -> Result<Option<Vec<u8>>, PartParseError> {
    match map.remove(name) {
        None => Ok(None),
        Some(AvroValue::Union(_, boxed)) => match *boxed {
            AvroValue::Null => Ok(None),
            AvroValue::Bytes(b) => Ok(Some(b)),
            _ => Err(PartParseError::WrongFieldType(name)),
        },
        Some(AvroValue::Null) => Ok(None),
        Some(AvroValue::Bytes(b)) => Ok(Some(b)),
        Some(_) => Err(PartParseError::WrongFieldType(name)),
    }
}

/// Encode [`SubsectionOffsets`] as a flat byte string: a 1-byte
/// version tag, then `total_size`, optional `(vec_off, vec_len)`,
/// optional `(fts_off, fts_len)`, and exact open-time range lists.
///
/// Stays in lock-step with [`decode_subsection_offsets`].
fn encode_subsection_offsets(off: &SubsectionOffsets) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        1 + 8
            + 1
            + 16
            + 1
            + 16
            + 4
            + off.vec_open_ranges.len() * 16
            + 4
            + off.fts_open_ranges.len() * 16,
    );
    out.push(SUBSECTION_OFFSETS_VERSION_CURRENT);
    out.extend_from_slice(&off.total_size.to_le_bytes());
    match off.vec {
        Some((o, l)) => {
            out.push(SUBSECTION_FLAG_PRESENT);
            out.extend_from_slice(&o.to_le_bytes());
            out.extend_from_slice(&l.to_le_bytes());
        }
        None => out.push(SUBSECTION_FLAG_ABSENT),
    }
    match off.fts {
        Some((o, l)) => {
            out.push(SUBSECTION_FLAG_PRESENT);
            out.extend_from_slice(&o.to_le_bytes());
            out.extend_from_slice(&l.to_le_bytes());
        }
        None => out.push(SUBSECTION_FLAG_ABSENT),
    }
    encode_range_list(&mut out, &off.vec_open_ranges);
    encode_range_list(&mut out, &off.fts_open_ranges);
    encode_open_blob(&mut out, &off.open_blob);
    out
}

/// Encode the inline open-batch blob: a u32 count, then each
/// entry as `(u64 absolute_offset, u32 byte_len, bytes)`. Stays
/// in lock-step with [`decode_open_blob`].
fn encode_open_blob(out: &mut Vec<u8>, blob: &[(u64, Vec<u8>)]) {
    out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
    for (off, bytes) in blob {
        out.extend_from_slice(&off.to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
}

fn encode_range_list(out: &mut Vec<u8>, ranges: &[(u64, u64)]) {
    out.extend_from_slice(&(ranges.len() as u32).to_le_bytes());
    for &(off, len) in ranges {
        out.extend_from_slice(&off.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
    }
}

fn decode_subsection_offsets(bytes: &[u8]) -> Result<SubsectionOffsets, PartParseError> {
    let mut cur = bytes;
    let take = |cur: &mut &[u8], n: usize| -> Result<Vec<u8>, PartParseError> {
        if cur.len() < n {
            return Err(PartParseError::SchemaMismatch(
                "subsection_offsets truncated".into(),
            ));
        }
        let (head, tail) = cur.split_at(n);
        let out = head.to_vec();
        *cur = tail;
        Ok(out)
    };
    let read_u64 = |cur: &mut &[u8]| -> Result<u64, PartParseError> {
        let b = take(cur, U64_BYTES)?;
        let arr: [u8; U64_BYTES] = b
            .as_slice()
            .try_into()
            .map_err(|_| PartParseError::SchemaMismatch("subsection_offsets u64 read".into()))?;
        Ok(u64::from_le_bytes(arr))
    };
    let ver = take(&mut cur, 1)?[0];
    if ver != SUBSECTION_OFFSETS_VERSION_LEGACY && ver != SUBSECTION_OFFSETS_VERSION_CURRENT {
        return Err(PartParseError::SchemaMismatch(format!(
            "subsection_offsets unknown version {ver}"
        )));
    }
    let total_size = read_u64(&mut cur)?;
    let vec_flag = take(&mut cur, 1)?[0];
    let vec = if vec_flag == SUBSECTION_FLAG_PRESENT {
        let o = read_u64(&mut cur)?;
        let l = read_u64(&mut cur)?;
        Some((o, l))
    } else {
        None
    };
    let fts_flag = take(&mut cur, 1)?[0];
    let fts = if fts_flag == SUBSECTION_FLAG_PRESENT {
        let o = read_u64(&mut cur)?;
        let l = read_u64(&mut cur)?;
        Some((o, l))
    } else {
        None
    };
    let vec_open_ranges = decode_range_list(&mut cur, &read_u64, &take)?;
    let fts_open_ranges = decode_range_list(&mut cur, &read_u64, &take)?;
    // Version 3 appends the inline open-batch blob; version 2 has
    // none (and leaves `cur` empty here).
    let open_blob = if ver >= SUBSECTION_OFFSETS_VERSION_CURRENT {
        decode_open_blob(&mut cur, &read_u64, &take)?
    } else {
        Vec::new()
    };
    if !cur.is_empty() {
        return Err(PartParseError::SchemaMismatch(
            "subsection_offsets has trailing bytes".into(),
        ));
    }
    Ok(SubsectionOffsets {
        total_size,
        vec,
        fts,
        vec_open_ranges,
        fts_open_ranges,
        open_blob,
    })
}

fn decode_open_blob(
    cur: &mut &[u8],
    read_u64: &impl Fn(&mut &[u8]) -> Result<u64, PartParseError>,
    take: &impl Fn(&mut &[u8], usize) -> Result<Vec<u8>, PartParseError>,
) -> Result<Vec<(u64, Vec<u8>)>, PartParseError> {
    let count_bytes = take(cur, U32_BYTES)?;
    let count = u32::from_le_bytes(
        count_bytes
            .as_slice()
            .try_into()
            .map_err(|_| PartParseError::SchemaMismatch("open_blob count read".into()))?,
    ) as usize;
    let mut blob = Vec::with_capacity(count);
    for _ in 0..count {
        let off = read_u64(cur)?;
        let len_bytes = take(cur, U32_BYTES)?;
        let len = u32::from_le_bytes(
            len_bytes
                .as_slice()
                .try_into()
                .map_err(|_| PartParseError::SchemaMismatch("open_blob len read".into()))?,
        ) as usize;
        let bytes = take(cur, len)?;
        blob.push((off, bytes));
    }
    Ok(blob)
}

fn decode_range_list(
    cur: &mut &[u8],
    read_u64: &impl Fn(&mut &[u8]) -> Result<u64, PartParseError>,
    take: &impl Fn(&mut &[u8], usize) -> Result<Vec<u8>, PartParseError>,
) -> Result<Vec<(u64, u64)>, PartParseError> {
    let count_bytes = take(cur, U32_BYTES)?;
    let count = u32::from_le_bytes(count_bytes.as_slice().try_into().map_err(|_| {
        PartParseError::SchemaMismatch("subsection_offsets range count read".into())
    })?) as usize;
    let mut ranges = Vec::with_capacity(count);
    for _ in 0..count {
        ranges.push((read_u64(cur)?, read_u64(cur)?));
    }
    Ok(ranges)
}

// Silence "unused" if Schema isn't consumed yet on its
// type-only path during cfg(test) gates.
#[allow(dead_code)]
fn _schema_handle() -> &'static AvroSchema {
    schema()
}

#[allow(dead_code)]
fn _encoding_used() {
    let _ = encoding::encode_scalar_stats;
}

#[cfg(test)]
mod tests {
    //! Avro+zstd round-trip tests for `ManifestPart`.
    //!
    //! Covers: empty / single / multi-superfile round-trip;
    //! every per-superfile summary type (scalar stats, fts
    //! summary, vector summary) survives bit-exactly through
    //! encode → decode; centroid f32 values are bit-identical
    //! (no decimal-string round-trip); content_hash covers
    //! the entire compressed byte buffer; same logical
    //! content → same bytes + same content_hash (the
    //! property cross-version part-reuse rides on);
    //! format_version major/minor compat; corrupt zstd
    //! surfaces a typed error.
    use super::*;
    use crate::supertable::manifest::bloom::BloomBuilder;
    use crate::supertable::manifest::{FtsSummary, ScalarStatsTable, VectorSummary};
    use crate::supertable::{SuperfileEntry, SuperfileUri};
    use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
    use bytes::Bytes;
    use std::collections::HashMap;
    use std::sync::Arc;
    use uuid::Uuid;

    fn fresh_superfile(n_docs: u64) -> Arc<SuperfileEntry> {
        let id = Uuid::new_v4();
        Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs,
            id_min: 0,
            id_max: n_docs.saturating_sub(1) as i128,
            scalar_stats: ScalarStatsTable::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        })
    }

    fn fresh_part(superfiles: Vec<Arc<SuperfileEntry>>) -> ManifestPart {
        ManifestPart {
            format_version: FORMAT_VERSION.into(),
            part_id: PartId::new_v4(),
            superfiles,
        }
    }

    fn make_fts_summary(seed: u8, n_terms: u32, range: (Vec<u8>, Vec<u8>)) -> FtsSummary {
        let mut builder = BloomBuilder::with_n_blocks(16);
        for i in 0..n_terms {
            let key = format!("term_{}_{i}", seed);
            builder.insert(key.as_bytes());
        }
        FtsSummary {
            term_bloom: builder.finish(),
            n_terms_distinct: n_terms,
            term_range: range,
        }
    }

    fn make_vector_summary(dim: usize, seed: f32) -> VectorSummary {
        let centroid: Vec<f32> = (0..dim).map(|i| seed + i as f32 * 0.001).collect();
        VectorSummary {
            centroid,
            radius: seed * 1.7,
            clusters: crate::supertable::manifest::ClusterCentroids::empty(),
        }
    }

    fn make_scalar_stats() -> ScalarStatsTable {
        // Cover Int64, Float64, Boolean, Utf8 — the four
        // shapes the existing skip path supports.
        let mut cols: HashMap<String, (ArrayRef, ArrayRef)> = HashMap::new();
        cols.insert(
            "ts".into(),
            (
                Arc::new(Int64Array::from(vec![1_715_000_000_i64])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1_715_086_400_i64])) as ArrayRef,
            ),
        );
        cols.insert(
            "score".into(),
            (
                Arc::new(Float64Array::from(vec![0.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![0.999_999])) as ArrayRef,
            ),
        );
        cols.insert(
            "active".into(),
            (
                Arc::new(BooleanArray::from(vec![false])) as ArrayRef,
                Arc::new(BooleanArray::from(vec![true])) as ArrayRef,
            ),
        );
        cols.insert(
            "category".into(),
            (
                Arc::new(StringArray::from(vec!["alpha"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["zulu"])) as ArrayRef,
            ),
        );
        ScalarStatsTable {
            cols,
            ..Default::default()
        }
    }

    fn make_rich_superfile() -> Arc<SuperfileEntry> {
        let id = Uuid::new_v4();
        let mut fts = HashMap::new();
        fts.insert(
            "title".into(),
            make_fts_summary(1, 50, (b"alpha".to_vec(), b"zulu".to_vec())),
        );
        fts.insert(
            "body".into(),
            make_fts_summary(2, 30, (b"".to_vec(), b"\xff\xff".to_vec())),
        );

        let mut vec_summary = HashMap::new();
        vec_summary.insert("emb".into(), make_vector_summary(8, 0.5));
        vec_summary.insert("img".into(), make_vector_summary(16, 1.25));

        Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: 12_345,
            id_min: 1_000,
            id_max: 13_344,
            scalar_stats: make_scalar_stats(),
            fts_summary: fts,
            vector_summary: vec_summary,
            partition_key: vec![0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            partition_hint: Some(13),
            subsection_offsets: Some(SubsectionOffsets {
                total_size: 12_345_678,
                vec: Some((123_456, 78_910)),
                fts: Some((11_111, 22_222)),
                vec_open_ranges: vec![(123_456, 96), (200_000, 4096)],
                fts_open_ranges: vec![(11_111, 1024), (30_000, 2048)],
                open_blob: vec![(12_345_614, vec![0xAB; 64]), (123_456, vec![0xCD; 96])],
            }),
        })
    }

    fn assert_superfiles_equal(a: &SuperfileEntry, b: &SuperfileEntry) {
        assert_eq!(a.superfile_id, b.superfile_id, "superfile_id");
        assert_eq!(a.uri, b.uri, "uri");
        assert_eq!(a.n_docs, b.n_docs, "n_docs");
        assert_eq!(a.id_min, b.id_min, "id_min");
        assert_eq!(a.id_max, b.id_max, "id_max");
        assert_eq!(a.partition_key, b.partition_key, "partition_key");
        assert_eq!(a.partition_hint, b.partition_hint, "partition_hint");

        assert_eq!(
            a.scalar_stats.cols.len(),
            b.scalar_stats.cols.len(),
            "scalar_stats column count"
        );
        for (k, (a_min, a_max)) in &a.scalar_stats.cols {
            let (b_min, b_max) = b
                .scalar_stats
                .cols
                .get(k)
                .unwrap_or_else(|| panic!("missing scalar col {k}"));
            assert_eq!(a_min.data_type(), b_min.data_type(), "scalar {k} min type");
            assert_eq!(a_max.data_type(), b_max.data_type(), "scalar {k} max type");
            assert_eq!(a_min.to_data(), b_min.to_data(), "scalar {k} min data");
            assert_eq!(a_max.to_data(), b_max.to_data(), "scalar {k} max data");
        }

        assert_eq!(a.fts_summary.len(), b.fts_summary.len(), "fts col count");
        for (k, av) in &a.fts_summary {
            let bv = b
                .fts_summary
                .get(k)
                .unwrap_or_else(|| panic!("missing fts col {k}"));
            assert_eq!(
                av.n_terms_distinct, bv.n_terms_distinct,
                "fts {k} n_terms_distinct"
            );
            assert_eq!(av.term_range, bv.term_range, "fts {k} term_range");
            assert_eq!(
                av.term_bloom.to_bytes(),
                bv.term_bloom.to_bytes(),
                "fts {k} bloom bytes"
            );
        }

        // Bit-exact float compare via to_bits() — catches
        // any decimal-string round-trip.
        assert_eq!(a.vector_summary.len(), b.vector_summary.len(), "vec count");
        for (k, av) in &a.vector_summary {
            let bv = b
                .vector_summary
                .get(k)
                .unwrap_or_else(|| panic!("missing vec col {k}"));
            assert_eq!(
                av.radius.to_bits(),
                bv.radius.to_bits(),
                "vec {k} radius bits"
            );
            assert_eq!(av.centroid.len(), bv.centroid.len(), "vec {k} dim");
            for (i, (af, bf)) in av.centroid.iter().zip(bv.centroid.iter()).enumerate() {
                assert_eq!(
                    af.to_bits(),
                    bf.to_bits(),
                    "vec {k} centroid[{i}] bits ({af} vs {bf})"
                );
            }
        }
    }

    #[test]
    fn empty_part_roundtrip() {
        let part = fresh_part(vec![]);
        let bytes = encode(&part, 3);
        let decoded = decode(&bytes).expect("decode empty");
        assert_eq!(decoded.format_version, FORMAT_VERSION);
        assert_eq!(decoded.part_id, part.part_id);
        assert_eq!(decoded.superfiles.len(), 0);
    }

    #[test]
    fn single_minimal_superfile_roundtrip() {
        let part = fresh_part(vec![fresh_superfile(100)]);
        let bytes = encode(&part, 3);
        let decoded = decode(&bytes).expect("decode minimal");
        assert_eq!(decoded.superfiles.len(), 1);
        assert_superfiles_equal(&decoded.superfiles[0], &part.superfiles[0]);
    }

    #[test]
    fn multi_superfile_with_full_summaries_roundtrip() {
        let superfiles: Vec<Arc<SuperfileEntry>> = (0..5).map(|_| make_rich_superfile()).collect();
        let part = fresh_part(superfiles);
        let bytes = encode(&part, 3);
        let decoded = decode(&bytes).expect("decode rich");
        assert_eq!(decoded.superfiles.len(), 5);
        for (a, b) in decoded.superfiles.iter().zip(part.superfiles.iter()) {
            assert_superfiles_equal(a, b);
        }
    }

    #[test]
    fn content_hash_covers_all_bytes() {
        let part = fresh_part(vec![make_rich_superfile()]);
        let bytes = encode(&part, 3);
        let hash = ContentHash::of(&bytes);

        let mut tampered = bytes.clone();
        let mid = tampered.len() / 2;
        tampered[mid] ^= 0xff;
        let tampered_hash = ContentHash::of(&tampered);
        assert_ne!(
            hash, tampered_hash,
            "blake3 must change when any byte changes"
        );
    }

    #[test]
    fn same_logical_content_produces_same_bytes_and_hash() {
        // Same superfiles + same part_id ⇒ bit-identical Avro
        // output, bit-identical zstd output, same blake3 —
        // the property cross-version part-reuse rides on.
        let superfiles = vec![make_rich_superfile(), make_rich_superfile()];
        let part_a = ManifestPart {
            format_version: FORMAT_VERSION.into(),
            part_id: PartId(Uuid::nil()),
            superfiles: superfiles.clone(),
        };
        let part_b = ManifestPart {
            format_version: FORMAT_VERSION.into(),
            part_id: PartId(Uuid::nil()),
            superfiles,
        };

        let bytes_a = encode(&part_a, 3);
        let bytes_b = encode(&part_b, 3);
        assert_eq!(bytes_a, bytes_b, "same logical content → same bytes");
        assert_eq!(
            ContentHash::of(&bytes_a),
            ContentHash::of(&bytes_b),
            "same logical content → same content_hash"
        );
    }

    #[test]
    fn partition_hint_some_and_none_both_roundtrip() {
        let id = Uuid::new_v4();
        let seg_with = Arc::new(SuperfileEntry {
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: 1,
            id_min: 0,
            id_max: 0,
            scalar_stats: ScalarStatsTable::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: vec![0xab, 0xcd],
            partition_hint: Some(0xdead_beef),
            subsection_offsets: None,
        });
        let id2 = Uuid::new_v4();
        let seg_without = Arc::new(SuperfileEntry {
            superfile_id: id2,
            uri: SuperfileUri(id2),
            n_docs: 1,
            id_min: 0,
            id_max: 0,
            scalar_stats: ScalarStatsTable::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
        });
        let part = fresh_part(vec![seg_with.clone(), seg_without.clone()]);
        let bytes = encode(&part, 3);
        let decoded = decode(&bytes).expect("decode mixed-hint");
        assert_eq!(decoded.superfiles.len(), 2);
        assert_eq!(decoded.superfiles[0].partition_hint, Some(0xdead_beef));
        assert_eq!(decoded.superfiles[0].partition_key, vec![0xab, 0xcd]);
        assert_eq!(decoded.superfiles[1].partition_hint, None);
        assert_eq!(decoded.superfiles[1].partition_key, Vec::<u8>::new());
    }

    #[test]
    fn incompatible_major_version_rejected() {
        let mut part = fresh_part(vec![fresh_superfile(1)]);
        part.format_version = "2.0".into();
        let bytes = encode(&part, 3);
        let err = decode(&bytes).expect_err("major 2 must reject");
        assert!(
            matches!(err, PartParseError::IncompatibleMajorVersion { .. }),
            "expected IncompatibleMajorVersion, got {err:?}"
        );
    }

    #[test]
    fn minor_version_compatible() {
        let mut part = fresh_part(vec![fresh_superfile(7)]);
        part.format_version = "1.99".into();
        let bytes = encode(&part, 3);
        let decoded = decode(&bytes).expect("minor 99 must accept");
        assert_eq!(decoded.format_version, "1.99");
        assert_eq!(decoded.superfiles.len(), 1);
    }

    #[test]
    fn zstd_corruption_surfaces_typed_error() {
        let part = fresh_part(vec![fresh_superfile(1)]);
        let mut bytes = encode(&part, 3);
        bytes[0] ^= 0xff;
        bytes[1] ^= 0xff;
        let err = decode(&bytes).expect_err("corrupt zstd must fail");
        assert!(
            matches!(err, PartParseError::Zstd(_) | PartParseError::Avro(_)),
            "expected Zstd or Avro error, got {err:?}"
        );
    }

    #[test]
    fn bytes_payload_is_well_formed_use_via_bytes_type() {
        // Sanity: wire shape is acceptable to bytes::Bytes
        // for the storage layer downstream.
        let part = fresh_part(vec![make_rich_superfile()]);
        let raw = encode(&part, 3);
        let wrapped = Bytes::from(raw.clone());
        let decoded = decode(&wrapped).expect("decode from Bytes");
        assert_eq!(decoded.superfiles.len(), 1);
    }
}
