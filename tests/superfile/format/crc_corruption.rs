// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! CRC32C corruption oracle.
//!
//! Builds a real superfile (FTS + vector + parquet body), then for each
//! CRC-protected region flips a byte inside it and asserts that
//! `SuperfileReader::open` rejects the result. The test does not pin
//! the exact error variant — different sections surface different
//! errors (FTS `ChecksumMismatch`, vector `ChecksumMismatch`, parquet
//! footer corruption may surface as `Footer(Parquet(...))` from the
//! parquet crate, etc.). The contract this test enforces is:
//! corrupted bytes must never silently return wrong data.
//!
//! Why this matters: bit rot on disk, partial S3 multipart uploads,
//! filesystem checksum gaps — all real failure modes. The reader's job
//! is to surface them as errors, not to plough on with truncated
//! posting lists or scrambled centroids.

use std::sync::Arc;

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        vector::distance::normalize,
    },
    test_helpers::{decimal128_ids, default_tokenizer, default_vector_config},
};

/// Decimal128 precision / scale for the `doc_id` column.
const ID_DECIMAL_PRECISION: u8 = 38;
const ID_DECIMAL_SCALE: i8 = 0;
/// Random-rotation seed for the corruptable superfile's vector index.
const CRC_TEST_ROT_SEED: u64 = 31;
/// Doc count for the small corruptable superfile.
const CRC_TEST_N_DOCS: u32 = 12;
/// Embedding dimension (matches `default_vector_config`'s dim).
const CRC_TEST_EMB_DIM: usize = 16;
/// Secondary one-hot axis weight planted in each doc vector.
const CRC_TEST_SECONDARY_WEIGHT: f32 = 0.5;
/// Offset of the planted secondary axis relative to the primary.
const CRC_TEST_SECONDARY_AXIS_OFFSET: usize = 3;
/// XOR mask used to flip a byte when corrupting a CRC-protected region.
const CORRUPTION_FLIP_MASK: u8 = 0xFF;

fn build_corruptable_superfile() -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "doc_id",
            DataType::Decimal128(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE),
            false,
        ),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![default_vector_config("emb", CRC_TEST_ROT_SEED)],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let n = CRC_TEST_N_DOCS;
    let ids = decimal128_ids(0..n as u64);
    let titles = LargeStringArray::from(
        (0..n)
            .map(|i| format!("doc {i} rust async runtime systems"))
            .collect::<Vec<_>>(),
    );
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
        .expect("build RecordBatch");

    // n × dim unit-norm vectors.
    let mut flat = Vec::<f32>::with_capacity(n as usize * CRC_TEST_EMB_DIM);
    for i in 0..n {
        let mut v = vec![0.0f32; CRC_TEST_EMB_DIM];
        v[(i as usize) % CRC_TEST_EMB_DIM] = 1.0;
        v[((i as usize) + CRC_TEST_SECONDARY_AXIS_OFFSET) % CRC_TEST_EMB_DIM] =
            CRC_TEST_SECONDARY_WEIGHT;
        normalize(&mut v);
        flat.extend_from_slice(&v);
    }

    b.add_batch(&batch, &[flat.as_slice()]).expect("add_batch");
    b.finish().expect("finish builder")
}

/// Read `inf.fts.offset` / `inf.vec.offset` to find each blob's
/// absolute byte range within the file.
fn locate_blobs(bytes: &[u8]) -> ((usize, usize), (usize, usize)) {
    use infino::superfile::format::footer::read_kv_metadata;
    let kv = read_kv_metadata(bytes).expect("read kv metadata");
    let fts_off: usize = kv["inf.fts.offset"].parse().expect("parse");
    let fts_len: usize = kv["inf.fts.length"].parse().expect("parse");
    let vec_off: usize = kv["inf.vec.offset"].parse().expect("parse");
    let vec_len: usize = kv["inf.vec.length"].parse().expect("parse");
    ((fts_off, fts_len), (vec_off, vec_len))
}

/// Flip a single byte in the buffer (XOR with 0xFF) and try to reopen
/// the superfile. Asserts that open returns Err.
fn assert_corruption_rejected(mut bytes: Vec<u8>, position: usize, label: &str) {
    bytes[position] ^= CORRUPTION_FLIP_MASK;
    let result = SuperfileReader::open(Bytes::from(bytes));
    assert!(
        result.is_err(),
        "corruption at {label} (byte {position}) should be rejected, got Ok"
    );
}

#[test]
fn corrupt_fts_fst_body_rejected() {
    // Flip a byte well inside the FTS FST. The FST sits right after
    // the 48-byte header, so byte (fts_off + 100) is firmly inside
    // FST data — guaranteed to be CRC-protected.
    let bytes = build_corruptable_superfile();
    let ((fts_off, _), _) = locate_blobs(&bytes);
    let target = fts_off + 100;
    assert_corruption_rejected(bytes, target, "fts/fst body");
}

#[test]
fn corrupt_fts_postings_region_rejected() {
    // The postings region follows the FST. Read the FTS header at
    // bytes [fts_off+32..+40] for postings_offset (relative to blob
    // start). Flip a byte inside postings.
    let bytes = build_corruptable_superfile();
    let ((fts_off, _), _) = locate_blobs(&bytes);
    let postings_offset_rel = u64::from_le_bytes(
        bytes[fts_off + 32..fts_off + 40]
            .try_into()
            .expect("try_into"),
    ) as usize;
    let target = fts_off + postings_offset_rel + 8;
    assert_corruption_rejected(bytes, target, "fts/postings region");
}

#[test]
fn corrupt_fts_doc_lengths_directory_rejected() {
    // The doc-lengths directory follows postings. FTS header bytes
    // [fts_off+40..+48] hold doc_lengths_table_offset (relative).
    let bytes = build_corruptable_superfile();
    let ((fts_off, _), _) = locate_blobs(&bytes);
    let doc_lengths_off_rel = u64::from_le_bytes(
        bytes[fts_off + 40..fts_off + 48]
            .try_into()
            .expect("try_into"),
    ) as usize;
    // First directory entry is column_id (4) + doc_lengths_offset (8)
    // + avgdl_x1000 (4) = 16 B; flip a byte inside doc_lengths_offset
    // (8 B into the directory) to corrupt the directory.
    let target = fts_off + doc_lengths_off_rel + 8;
    assert_corruption_rejected(bytes, target, "fts/doc-lengths directory");
}

#[test]
fn corrupt_fts_doc_lengths_array_rejected() {
    // The per-column doc-lengths arrays sit after the directory CRC.
    // The FTS blob ends with the last column's array+CRC, so flipping
    // a byte 8 bytes before the FTS blob end hits the last column's
    // doc-lengths array.
    let bytes = build_corruptable_superfile();
    let ((fts_off, fts_len), _) = locate_blobs(&bytes);
    let target = fts_off + fts_len - 8;
    assert_corruption_rejected(bytes, target, "fts/doc-lengths array");
}

#[test]
fn corrupt_vector_outer_header_rejected() {
    // Vector outer header starts with 8-byte magic; corrupting bytes
    // 16..24 (n_docs in outer header) sits past the magic but inside
    // the outer-CRC-protected region.
    let bytes = build_corruptable_superfile();
    let (_, (vec_off, _)) = locate_blobs(&bytes);
    let target = vec_off + 16;
    assert_corruption_rejected(bytes, target, "vector/outer header");
}

#[test]
fn corrupt_vector_subsection_rejected() {
    // The first subsection lives after the outer header (32) +
    // directory (1 entry × 64) + dir CRC (4) = 100 bytes from outer
    // start. Each subsection has its own CRC at its tail; corrupting
    // 64 bytes into the subsection (well past its 56-byte sub-header)
    // hits cached centroid / code data.
    let bytes = build_corruptable_superfile();
    let (_, (vec_off, _)) = locate_blobs(&bytes);
    let target = vec_off + 100 + 64;
    assert_corruption_rejected(bytes, target, "vector/subsection body");
}

#[test]
fn corrupt_vector_outer_trailing_crc_rejected() {
    // Last 4 bytes of the vector blob hold the outer CRC; corrupting
    // it makes the body+CRC mismatch.
    let bytes = build_corruptable_superfile();
    let (_, (vec_off, vec_len)) = locate_blobs(&bytes);
    let target = vec_off + vec_len - 2;
    assert_corruption_rejected(bytes, target, "vector/outer trailing CRC");
}

#[test]
fn corrupt_parquet_footer_rejected() {
    // Last 8 bytes of the file are footer_len (4) + b"PAR1" (4); the
    // u32 footer_len locates the start of the thrift-encoded footer.
    // We corrupt the first byte of the thrift encoding, which is the
    // root struct's field-id+type header — any flip there changes
    // field type or id and Parquet's decoder rejects the result.
    let bytes = build_corruptable_superfile();
    let n = bytes.len();
    let footer_len =
        u32::from_le_bytes([bytes[n - 8], bytes[n - 7], bytes[n - 6], bytes[n - 5]]) as usize;
    let target = n - 8 - footer_len;
    assert_corruption_rejected(bytes, target, "parquet/footer thrift");
}

#[test]
fn corrupt_parquet_trailing_magic_rejected() {
    // The b"PAR1" magic at file end. Flip the last byte.
    let bytes = build_corruptable_superfile();
    let target = bytes.len() - 1;
    assert_corruption_rejected(bytes, target, "parquet/trailing magic");
}

#[test]
fn untouched_bytes_open_succeeds() {
    // Sanity baseline: the unmodified bytes do open. Catches a bug
    // where `build_corruptable_superfile` produces an already-broken
    // file.
    let bytes = build_corruptable_superfile();
    let r = SuperfileReader::open(Bytes::from(bytes));
    assert!(r.is_ok(), "uncorrupted file must open: {:?}", r.err());
}

#[test]
fn corruption_at_random_interior_positions_rejected() {
    // Stronger property check: pick 32 deterministic positions
    // spanning from byte 100 (past the parquet header magic) to
    // file_len - 12 (before parquet trailer). Each must reject.
    // Some may land on padding bytes that aren't CRC-checked
    // (e.g., reserved fields between sections); we accept that
    // a small fraction may not reject. Assert at least 75%
    // rejection rate.
    let bytes = build_corruptable_superfile();
    let n = bytes.len();
    let lo = 100usize;
    let hi = n - 12;
    let span = hi - lo;
    let n_samples = 32usize;
    let mut rejected = 0;
    for i in 0..n_samples {
        let pos = lo + (i * span) / n_samples;
        let mut copy = bytes.clone();
        copy[pos] ^= CORRUPTION_FLIP_MASK;
        if SuperfileReader::open(Bytes::from(copy)).is_err() {
            rejected += 1;
        }
    }
    assert!(
        rejected * 4 >= n_samples * 3,
        "expected ≥75% of random interior corruptions to be rejected; got {rejected}/{n_samples}"
    );
}
