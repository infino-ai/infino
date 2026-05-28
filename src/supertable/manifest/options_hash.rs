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
//! Legacy / synthetic-manifest escape hatch: a stored `options_hash` of
//! all zeros is treated as "validation skipped" by
//! [`verify_options_hash`] — pre-D15 manifests + test fixtures that
//! construct lists manually keep opening cleanly.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::builder::{FtsConfig, VectorConfig};
    use crate::superfile::vector::distance::Metric;
    use crate::supertable::manifest::list::PartitionStrategy;
    use crate::supertable::manifest::part::ContentHash;
    use crate::supertable::options::SupertableOptions;
    use crate::test_helpers::default_tokenizer;
    use arrow_array::FixedSizeListArray;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn schema_title_only() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn schema_title_emb(dim: usize) -> Arc<Schema> {
        let list_field = Field::new("item", DataType::Float32, false);
        let list_type = DataType::FixedSizeList(Arc::new(list_field), dim as i32);
        Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", list_type, false),
        ]))
    }

    fn fts_opts() -> SupertableOptions {
        SupertableOptions::new(
            schema_title_only(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        )
        .expect("opts")
    }

    fn time_range() -> PartitionStrategy {
        PartitionStrategy::TimeRange {
            column: "_id".into(),
            granularity_secs: 86_400,
        }
    }

    // ---- compute_options_hash determinism --------------------------------

    #[test]
    fn compute_options_hash_deterministic() {
        // Same options + strategy yield byte-identical hashes
        // across calls. Guards against accidental
        // nondeterminism from HashMap iteration order or
        // similar.
        let h1 = compute_options_hash(&fts_opts(), &time_range());
        let h2 = compute_options_hash(&fts_opts(), &time_range());
        assert_eq!(h1.0, h2.0);
    }

    #[test]
    fn compute_options_hash_changes_with_schema() {
        // Renaming a column changes the schema field name, which
        // is part of the hash. Same column type, different name.
        let opts_a = fts_opts();
        let opts_b = SupertableOptions::new(
            Arc::new(Schema::new(vec![Field::new(
                "body",
                DataType::LargeUtf8,
                false,
            )])),
            vec![FtsConfig {
                column: "body".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        )
        .expect("opts");
        let h_a = compute_options_hash(&opts_a, &time_range());
        let h_b = compute_options_hash(&opts_b, &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    #[test]
    fn compute_options_hash_changes_with_nullability() {
        // The nullable byte is included in the schema encoding,
        // so flipping nullable changes the hash even when
        // names and types match.
        let opts_a = fts_opts();
        let opts_b = SupertableOptions::new(
            Arc::new(Schema::new(vec![Field::new(
                "title",
                DataType::LargeUtf8,
                true, // nullable
            )])),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(default_tokenizer()),
        )
        .expect("opts");
        let h_a = compute_options_hash(&opts_a, &time_range());
        let h_b = compute_options_hash(&opts_b, &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    #[test]
    fn compute_options_hash_changes_with_fts_column_set() {
        // Adding another FTS column changes the fts_columns
        // length prefix + content. The schema must still be
        // compatible, so the second variant adds a `subtitle`
        // field.
        let opts_a = fts_opts();
        let schema_two = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("subtitle", DataType::LargeUtf8, false),
        ]));
        let opts_b = SupertableOptions::new(
            schema_two,
            vec![
                FtsConfig {
                    column: "title".into(),
                },
                FtsConfig {
                    column: "subtitle".into(),
                },
            ],
            vec![],
            Some(default_tokenizer()),
        )
        .expect("opts");
        let h_a = compute_options_hash(&opts_a, &time_range());
        let h_b = compute_options_hash(&opts_b, &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    #[test]
    fn compute_options_hash_changes_with_fts_column_order() {
        // FTS column order is part of the schema identity
        // (FtsBuilder assigns ids by position). Swapping the
        // two FTS column declarations must produce a different
        // hash even though the underlying set is the same.
        let schema_two = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("subtitle", DataType::LargeUtf8, false),
        ]));
        let opts_a = SupertableOptions::new(
            schema_two.clone(),
            vec![
                FtsConfig {
                    column: "title".into(),
                },
                FtsConfig {
                    column: "subtitle".into(),
                },
            ],
            vec![],
            Some(default_tokenizer()),
        )
        .expect("opts");
        let opts_b = SupertableOptions::new(
            schema_two,
            vec![
                FtsConfig {
                    column: "subtitle".into(),
                },
                FtsConfig {
                    column: "title".into(),
                },
            ],
            vec![],
            Some(default_tokenizer()),
        )
        .expect("opts");
        let h_a = compute_options_hash(&opts_a, &time_range());
        let h_b = compute_options_hash(&opts_b, &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    #[test]
    fn compute_options_hash_changes_with_vector_columns() {
        // Adding a vector column changes the vector_columns
        // count + per-column field bytes (dim, n_cent, rot_seed,
        // metric).
        let opts_a = fts_opts();
        let opts_b = SupertableOptions::new(
            schema_title_emb(16),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim: 16,
                n_cent: 4,
                rot_seed: 0,
                metric: Metric::Cosine,
            }],
            Some(default_tokenizer()),
        )
        .expect("opts");
        let h_a = compute_options_hash(&opts_a, &time_range());
        let h_b = compute_options_hash(&opts_b, &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    #[test]
    fn compute_options_hash_changes_with_vector_metric() {
        // The metric is encoded via lowercased `format!("{:?}",
        // metric)`, so changing Cosine → NegDot at otherwise
        // equal options must produce a different hash. Verifies
        // the per-metric encoding actually contributes.
        let mk = |metric: Metric| {
            SupertableOptions::new(
                schema_title_emb(16),
                vec![],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim: 16,
                    n_cent: 4,
                    rot_seed: 0,
                    metric,
                }],
                Some(default_tokenizer()),
            )
            .expect("opts")
        };
        let h_a = compute_options_hash(&mk(Metric::Cosine), &time_range());
        let h_b = compute_options_hash(&mk(Metric::NegDot), &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    // ---- PartitionStrategy variants ------------------------------------

    #[test]
    fn compute_options_hash_distinguishes_partition_strategy_variants() {
        // Same options, different partition-strategy variants
        // must produce different hashes — the variant tag is
        // pushed before any per-variant fields. Covers all three
        // arms of the match in compute_options_hash.
        let opts = fts_opts();
        let h_time = compute_options_hash(
            &opts,
            &PartitionStrategy::TimeRange {
                column: "_id".into(),
                granularity_secs: 86_400,
            },
        );
        let h_hash = compute_options_hash(
            &opts,
            &PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 16,
            },
        );
        let h_range = compute_options_hash(
            &opts,
            &PartitionStrategy::ColumnRange {
                column: "_id".into(),
                boundaries: vec![vec![1, 2, 3], vec![4, 5, 6]],
            },
        );
        assert_ne!(h_time.0, h_hash.0);
        assert_ne!(h_hash.0, h_range.0);
        assert_ne!(h_time.0, h_range.0);
    }

    #[test]
    fn compute_options_hash_partition_field_changes_propagate() {
        // Within each PartitionStrategy variant, mutating a
        // per-variant field must change the hash. Catches the
        // case where a field is added to the enum but forgotten
        // in the hash encoding.
        let opts = fts_opts();

        // TimeRange: granularity differs.
        let h_t1 = compute_options_hash(
            &opts,
            &PartitionStrategy::TimeRange {
                column: "_id".into(),
                granularity_secs: 86_400,
            },
        );
        let h_t2 = compute_options_hash(
            &opts,
            &PartitionStrategy::TimeRange {
                column: "_id".into(),
                granularity_secs: 3600,
            },
        );
        assert_ne!(h_t1.0, h_t2.0);

        // Hash: bucket count differs.
        let h_h1 = compute_options_hash(
            &opts,
            &PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 16,
            },
        );
        let h_h2 = compute_options_hash(
            &opts,
            &PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 32,
            },
        );
        assert_ne!(h_h1.0, h_h2.0);

        // ColumnRange: one extra boundary.
        let h_r1 = compute_options_hash(
            &opts,
            &PartitionStrategy::ColumnRange {
                column: "_id".into(),
                boundaries: vec![vec![1, 2]],
            },
        );
        let h_r2 = compute_options_hash(
            &opts,
            &PartitionStrategy::ColumnRange {
                column: "_id".into(),
                boundaries: vec![vec![1, 2], vec![3, 4]],
            },
        );
        assert_ne!(h_r1.0, h_r2.0);
    }

    // ---- verify_options_hash --------------------------------------------

    #[test]
    fn verify_options_hash_accepts_matching_pair() {
        let opts = fts_opts();
        let h = compute_options_hash(&opts, &time_range());
        verify_options_hash(h, h).expect("matching pair accepted");
    }

    #[test]
    fn verify_options_hash_skips_zero_sentinel() {
        // Pre-D15 manifests + synthetic test fixtures with an
        // all-zero stored hash bypass validation: the caller's
        // computed hash can be anything.
        let opts = fts_opts();
        let computed = compute_options_hash(&opts, &time_range());
        let zero = ContentHash([0u8; 32]);
        verify_options_hash(computed, zero).expect("zero sentinel bypasses verification");
    }

    #[test]
    fn verify_options_hash_rejects_mismatch_with_hex_payload() {
        // Two clearly different hashes must produce
        // OptionsHashMismatch whose Display includes both hex
        // strings prefixed with `blake3:`.
        let h_a = ContentHash([1u8; 32]);
        let h_b = ContentHash([2u8; 32]);
        let err = verify_options_hash(h_a, h_b).expect_err("mismatch must error");
        let rendered = format!("{err}");
        assert!(
            rendered.contains("options_hash mismatch"),
            "got: {rendered}"
        );
        assert!(rendered.contains("blake3:"), "got: {rendered}");
        // 32 bytes of 0x01 → 64-char hex string.
        assert!(rendered.contains(&"01".repeat(32)), "got: {rendered}");
        assert!(rendered.contains(&"02".repeat(32)), "got: {rendered}");
    }

    #[test]
    fn options_hash_mismatch_is_error_impl() {
        // Trait-object usage exercises the
        // `impl std::error::Error for OptionsHashMismatch` — a
        // no-op body but the impl block needs to compile and
        // the dyn-error conversion needs to succeed.
        let h_a = ContentHash([3u8; 32]);
        let h_b = ContentHash([4u8; 32]);
        let err = verify_options_hash(h_a, h_b).expect_err("mismatch");
        let dyn_err: Box<dyn std::error::Error> = Box::new(err);
        assert!(dyn_err.to_string().contains("options_hash mismatch"));
    }

    // ---- helpers (light coverage on push_tag / push_str) ----------------

    #[test]
    fn push_helpers_emit_length_prefixed_bytes() {
        // The hash encoding's correctness rests on these
        // helpers; cover them directly so a regression in
        // either is caught at unit-test scale rather than at
        // an integration mismatch later.
        let mut buf = Vec::new();
        push_tag(&mut buf, b"schema");
        assert_eq!(buf, vec![6u8, b's', b'c', b'h', b'e', b'm', b'a']);

        let mut buf = Vec::new();
        push_str(&mut buf, "ok");
        // 8-byte LE length prefix + 2 ASCII bytes.
        assert_eq!(buf, vec![2u8, 0, 0, 0, 0, 0, 0, 0, b'o', b'k']);
    }

    // Compiler-only smoke that the FixedSizeListArray import
    // is exercised (the schema-builder uses it via DataType,
    // not the array itself). Keeps the import non-dead even if
    // some future test removes its only use.
    #[allow(dead_code)]
    fn _silence_fixed_size_list_array(_arr: FixedSizeListArray) {}
}
