// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The table's identity hash.
//!
//! [`compute_options_hash`] digests what a caller must supply to open a
//! table that the manifest list cannot tell them: the id column and the
//! partition strategy. The list is the authority for the schema and the
//! index config, so those are not part of the hash; a caller whose seed
//! schema differs from the list's simply reads the list's. The hash is
//! stamped onto `Manifest::options_hash` at commit and verified by
//! [`verify_options_hash`] on load, so an open with the wrong id column or
//! partitioning surfaces as `OpenError::OptionsHashMismatch` instead of a
//! decode failure on the first query.
//!
//! ## Encoding
//!
//! A length-prefixed byte stream, blake3'd. Each field is preceded by a
//! fixed tag so two shapes with overlapping bytes cannot collide:
//!
//! ```text
//! "id_column"          | len u64 | bytes
//! "partition_strategy" | variant_tag | per-variant fields
//! ```
//!
//! ## The creation-record stream
//!
//! A list written before it carried the schema was stamped with a hash
//! over the whole creation record — the Arrow schema, the id column, the
//! FTS and vector declarations and the strategy. [`verify_options_hash`]
//! accepts that hash too, computed from the caller's seed options (the
//! catalog's creation record), so such a table opens read-only without a
//! commit; its first commit re-stamps it with the identity hash. The old
//! stream contains the identity fields, so accepting it never admits a
//! caller the identity hash would refuse.
//!
//! A stored hash of all zeros means "validation skipped": synthetic lists
//! and the oldest manifests open without a check.

use std::{error::Error, fmt};

use crate::supertable::{
    manifest::{encoding::encode_cluster_centroids, list::PartitionStrategy, part::ContentHash},
    options::SupertableOptions,
};

/// The identity hash: the id column and the partition strategy.
pub fn compute_options_hash(opts: &SupertableOptions, strategy: &PartitionStrategy) -> ContentHash {
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    push_identity(&mut buf, opts, strategy);
    ContentHash(*blake3::hash(&buf).as_bytes())
}

/// The hash a list written before it carried the schema bears.
fn creation_record_hash(opts: &SupertableOptions, strategy: &PartitionStrategy) -> ContentHash {
    let mut buf: Vec<u8> = Vec::with_capacity(256);

    push_tag(&mut buf, b"schema");
    let fields = opts.schema.fields();
    buf.extend_from_slice(&(fields.len() as u64).to_le_bytes());
    for f in fields {
        push_str(&mut buf, f.name());
        let dt_str = format!("{:?}", f.data_type());
        push_str(&mut buf, &dt_str);
        buf.push(f.is_nullable() as u8);
    }

    push_tag(&mut buf, b"id_column");
    push_str(&mut buf, &opts.id_column);

    push_tag(&mut buf, b"fts_columns");
    buf.extend_from_slice(&(opts.fts_columns.len() as u64).to_le_bytes());
    for c in &opts.fts_columns {
        push_str(&mut buf, &c.column);
    }
    if opts.fts_columns.iter().any(|c| c.positions) {
        push_tag(&mut buf, b"fts_positions");
        for c in &opts.fts_columns {
            buf.push(c.positions as u8);
        }
    }
    if !opts.fts_columns.is_empty() {
        push_tag(&mut buf, b"fts_analyzers");
        for c in &opts.fts_columns {
            push_str(&mut buf, c.chain_name());
        }
    }
    if opts.fts_columns.iter().any(|c| !c.stored) {
        push_tag(&mut buf, b"fts_stored");
        for c in &opts.fts_columns {
            buf.push(c.stored as u8);
        }
    }

    push_tag(&mut buf, b"vector_columns");
    buf.extend_from_slice(&(opts.vector_columns.len() as u64).to_le_bytes());
    for v in &opts.vector_columns {
        push_str(&mut buf, &v.column);
        buf.extend_from_slice(&(v.dim as u64).to_le_bytes());
        buf.extend_from_slice(&v.rot_seed.to_le_bytes());
        push_str(&mut buf, v.metric.name());
    }

    push_tag(&mut buf, b"partition_strategy");
    push_strategy(&mut buf, strategy);

    ContentHash(*blake3::hash(&buf).as_bytes())
}

fn push_identity(buf: &mut Vec<u8>, opts: &SupertableOptions, strategy: &PartitionStrategy) {
    push_tag(buf, b"id_column");
    push_str(buf, &opts.id_column);
    push_tag(buf, b"partition_strategy");
    push_strategy(buf, strategy);
}

fn push_strategy(buf: &mut Vec<u8>, strategy: &PartitionStrategy) {
    match strategy {
        PartitionStrategy::TimeRange {
            column,
            granularity_secs,
        } => {
            push_tag(buf, b"time_range");
            push_str(buf, column);
            buf.extend_from_slice(&granularity_secs.to_le_bytes());
        }
        PartitionStrategy::Hash { column, n_buckets } => {
            push_tag(buf, b"hash");
            push_str(buf, column);
            buf.extend_from_slice(&n_buckets.to_le_bytes());
        }
        PartitionStrategy::ColumnRange { column, boundaries } => {
            push_tag(buf, b"column_range");
            push_str(buf, column);
            buf.extend_from_slice(&(boundaries.len() as u64).to_le_bytes());
            for b in boundaries {
                buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
                buf.extend_from_slice(b);
            }
        }
        PartitionStrategy::VectorCell {
            column,
            clusters,
            routing,
        } => {
            push_tag(buf, b"vector_cell");
            push_str(buf, column);
            let enc = encode_cluster_centroids(clusters);
            buf.extend_from_slice(&(enc.len() as u64).to_le_bytes());
            buf.extend_from_slice(&enc);
            buf.extend_from_slice(&(routing.nprobe_min as u64).to_le_bytes());
            buf.extend_from_slice(&(routing.nprobe_max as u64).to_le_bytes());
            buf.extend_from_slice(&routing.slack.to_le_bytes());
        }
        PartitionStrategy::IngestionTime { granularity_secs } => {
            push_tag(buf, b"ingestion_time");
            buf.extend_from_slice(&granularity_secs.to_le_bytes());
        }
    }
}

/// Accepts `stored` when it is the identity hash of `opts`, a
/// creation-record hash of `opts` (a list from before the list carried the
/// schema), or the zero sentinel.
pub fn verify_options_hash(
    opts: &SupertableOptions,
    strategy: &PartitionStrategy,
    stored: ContentHash,
) -> Result<(), OptionsHashMismatch> {
    if stored.0 == [0u8; 32] {
        return Ok(());
    }
    let expected = compute_options_hash(opts, strategy);
    if expected.0 == stored.0 || creation_record_hash(opts, strategy).0 == stored.0 {
        return Ok(());
    }
    Err(OptionsHashMismatch {
        expected: expected.to_hex(),
        actual: stored.to_hex(),
    })
}

#[derive(Debug, Clone)]
pub struct OptionsHashMismatch {
    pub expected: String,
    pub actual: String,
}

impl fmt::Display for OptionsHashMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "options_hash mismatch: caller=blake3:{} list=blake3:{}",
            self.expected, self.actual
        )
    }
}

impl Error for OptionsHashMismatch {}

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
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::{
        Stemmer, Stopwords,
        superfile::{
            builder::{FtsConfig, VectorConfig},
            vector::{distance::Metric, rerank_codec::RerankCodec},
        },
        supertable::{
            manifest::{ClusterCentroids, list::PartitionStrategy, part::ContentHash},
            options::SupertableOptions,
        },
    };

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
        SupertableOptions::new(schema_title_only(), vec![FtsConfig::new("title")], vec![])
            .expect("opts")
    }

    fn time_range() -> PartitionStrategy {
        PartitionStrategy::TimeRange {
            column: "_id".into(),
            granularity_secs: 86_400,
        }
    }

    /// The creation-record stream a list from before the schema slot was
    /// stamped with.
    fn record(opts: &SupertableOptions, strategy: &PartitionStrategy) -> ContentHash {
        creation_record_hash(opts, strategy)
    }

    /// The list carries the schema and the index config, so the identity
    /// hash does not move with them: only the id column and the strategy
    /// are the caller's to get wrong.
    #[test]
    fn identity_hash_ignores_schema_and_index_config() {
        let strat = time_range();
        let plain = SupertableOptions::new(schema_title_only(), vec![], vec![]).expect("opts");
        let indexed = SupertableOptions::new(
            schema_title_emb(16),
            vec![FtsConfig::new("title").positions(true)],
            vec![VectorConfig::new("emb".into(), 16, 1, Metric::L2Sq)],
        )
        .expect("opts");
        assert_eq!(
            compute_options_hash(&plain, &strat),
            compute_options_hash(&indexed, &strat)
        );
        assert_ne!(record(&plain, &strat), record(&indexed, &strat));

        let mut other_id =
            SupertableOptions::new(schema_title_only(), vec![], vec![]).expect("opts");
        other_id.id_column = "doc".into();
        assert_ne!(
            compute_options_hash(&plain, &strat),
            compute_options_hash(&other_id, &strat)
        );
    }

    /// A list stamped by an engine that hashed the whole creation record
    /// still verifies against the same creation record, and not against
    /// another table's.
    #[test]
    fn verify_options_hash_accepts_the_creation_record_streams() {
        let strat = time_range();
        let opts = fts_opts();
        verify_options_hash(&opts, &strat, record(&opts, &strat)).expect("creation record");
        let other = SupertableOptions::new(
            schema_title_only(),
            vec![FtsConfig::new("title").positions(true)],
            vec![],
        )
        .expect("opts");
        verify_options_hash(&other, &strat, record(&opts, &strat))
            .expect_err("another table's creation record must mismatch");
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
    fn creation_record_hash_changes_with_schema() {
        // Renaming a column changes the schema field name, which
        // is part of the hash. Same column type, different name.
        let opts_a = fts_opts();
        let opts_b = SupertableOptions::new(
            Arc::new(Schema::new(vec![Field::new(
                "body",
                DataType::LargeUtf8,
                false,
            )])),
            vec![FtsConfig::new("body")],
            vec![],
        )
        .expect("opts");
        let h_a = record(&opts_a, &time_range());
        let h_b = record(&opts_b, &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    #[test]
    fn creation_record_hash_changes_with_nullability() {
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
            vec![FtsConfig::new("title")],
            vec![],
        )
        .expect("opts");
        let h_a = record(&opts_a, &time_range());
        let h_b = record(&opts_b, &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    #[test]
    fn creation_record_hash_changes_with_fts_column_set() {
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
            vec![FtsConfig::new("title"), FtsConfig::new("subtitle")],
            vec![],
        )
        .expect("opts");
        let h_a = record(&opts_a, &time_range());
        let h_b = record(&opts_b, &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    #[test]
    fn creation_record_hash_changes_with_fts_column_order() {
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
            vec![FtsConfig::new("title"), FtsConfig::new("subtitle")],
            vec![],
        )
        .expect("opts");
        let opts_b = SupertableOptions::new(
            schema_two,
            vec![FtsConfig::new("subtitle"), FtsConfig::new("title")],
            vec![],
        )
        .expect("opts");
        let h_a = record(&opts_a, &time_range());
        let h_b = record(&opts_b, &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    /// The positions flag is hashed via a tagged block appended ONLY
    /// when some column opts in: flipping a column to positional changes
    /// the hash, and WHICH column is positional matters (per-column
    /// bytes, not a single any() bit).
    #[test]
    fn creation_record_hash_positions_flag() {
        let schema_two = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("subtitle", DataType::LargeUtf8, false),
        ]));
        let opts = |title_pos: bool, subtitle_pos: bool| {
            SupertableOptions::new(
                schema_two.clone(),
                vec![
                    FtsConfig::new("title").positions(title_pos),
                    FtsConfig::new("subtitle").positions(subtitle_pos),
                ],
                vec![],
            )
            .expect("opts")
        };
        let h_ff = record(&opts(false, false), &time_range());
        let h_tf = record(&opts(true, false), &time_range());
        let h_ft = record(&opts(false, true), &time_range());
        assert_ne!(h_ff.0, h_tf.0, "positional column must change the hash");
        assert_ne!(h_tf.0, h_ft.0, "which column is positional must matter");
    }

    /// The stored flag follows the same only-when-non-default rule as
    /// positions: an index-only column changes the hash, and WHICH column
    /// is index-only matters.
    #[test]
    fn creation_record_hash_stored_flag() {
        let schema_two = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("subtitle", DataType::LargeUtf8, false),
        ]));
        let opts = |title_stored: bool, subtitle_stored: bool| {
            SupertableOptions::new(
                schema_two.clone(),
                vec![
                    FtsConfig::new("title").stored(title_stored),
                    FtsConfig::new("subtitle").stored(subtitle_stored),
                ],
                vec![],
            )
            .expect("opts")
        };
        let h_tt = record(&opts(true, true), &time_range());
        let h_ft = record(&opts(false, true), &time_range());
        let h_tf = record(&opts(true, false), &time_range());
        assert_ne!(h_tt.0, h_ft.0, "index-only column must change the hash");
        assert_ne!(h_ft.0, h_tf.0, "which column is index-only must matter");
    }

    #[test]
    fn creation_record_hash_changes_with_vector_columns() {
        // Adding a vector column changes the vector_columns
        // count + per-column field bytes (dim, n_cent, rot_seed,
        // metric).
        let opts_a = fts_opts();
        let opts_b = SupertableOptions::new(
            schema_title_emb(16),
            vec![FtsConfig::new("title")],
            vec![VectorConfig {
                column: "emb".into(),
                dim: 16,
                rot_seed: 0,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::default(),
                provided_centroids: None,
            }],
        )
        .expect("opts");
        let h_a = record(&opts_a, &time_range());
        let h_b = record(&opts_b, &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    #[test]
    fn creation_record_hash_changes_with_vector_metric() {
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
                    rot_seed: 0,
                    metric,
                    rerank_codec: RerankCodec::Sq8Residual,
                    provided_centroids: None,
                }],
            )
            .expect("opts")
        };
        let h_a = record(&mk(Metric::Cosine), &time_range());
        let h_b = record(&mk(Metric::NegDot), &time_range());
        assert_ne!(h_a.0, h_b.0);
    }

    #[test]
    fn creation_record_hash_ignores_rerank_codec() {
        let mk = |rerank_codec: RerankCodec| {
            SupertableOptions::new(
                schema_title_emb(16),
                vec![],
                vec![VectorConfig {
                    column: "emb".into(),
                    dim: 16,
                    rot_seed: 0,
                    metric: Metric::Cosine,
                    rerank_codec,
                    provided_centroids: None,
                }],
            )
            .expect("opts")
        };
        // rerank_codec is data-determined (on-disk `codec_id`, dispatched by the
        // reader), not an identity input — changing only the codec must NOT move
        // the hash, so a default flip cannot break reopening an existing table.
        let base = record(&mk(RerankCodec::Sq8FixedResidual), &time_range());
        for codec in [
            RerankCodec::Sq16,
            RerankCodec::Sq8Residual,
            RerankCodec::Fp32,
            RerankCodec::RabitqOnly,
        ] {
            assert_eq!(
                record(&mk(codec), &time_range()).0,
                base.0,
                "options hash must not depend on rerank_codec ({codec:?})"
            );
        }
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
    fn compute_options_hash_distinguishes_vector_cell() {
        let opts = fts_opts();
        let clusters = ClusterCentroids::from_fp32(2, 4, &[0.0; 8], vec![1, 1]);
        let h_vc = compute_options_hash(
            &opts,
            &PartitionStrategy::VectorCell {
                column: "emb".into(),
                clusters: clusters.clone(),
                routing: Default::default(),
            },
        );
        let h_hash = compute_options_hash(
            &opts,
            &PartitionStrategy::Hash {
                column: "_id".into(),
                n_buckets: 16,
            },
        );
        assert_ne!(h_vc.0, h_hash.0);
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
        verify_options_hash(&opts, &time_range(), h).expect("matching pair accepted");
    }

    #[test]
    fn verify_options_hash_skips_zero_sentinel() {
        // Older manifests + synthetic test fixtures with an
        // all-zero stored hash bypass validation: the caller's
        // computed hash can be anything.
        let opts = fts_opts();
        let zero = ContentHash([0u8; 32]);
        verify_options_hash(&opts, &time_range(), zero)
            .expect("zero sentinel bypasses verification");
    }

    #[test]
    fn verify_options_hash_rejects_mismatch_with_hex_payload() {
        // Two clearly different hashes must produce
        // OptionsHashMismatch whose Display includes both hex
        // strings prefixed with `blake3:`.
        let opts = fts_opts();
        let expected = compute_options_hash(&opts, &time_range());
        let stored = ContentHash([2u8; 32]);
        let err =
            verify_options_hash(&opts, &time_range(), stored).expect_err("mismatch must error");
        let rendered = format!("{err}");
        assert!(
            rendered.contains("options_hash mismatch"),
            "got: {rendered}"
        );
        assert!(rendered.contains("blake3:"), "got: {rendered}");
        assert!(rendered.contains(&expected.to_hex()), "got: {rendered}");
        // 32 bytes of 0x02 → 64-char hex string.
        assert!(rendered.contains(&"02".repeat(32)), "got: {rendered}");
    }

    /// A column's analysis filters join the table's identity, and a
    /// filterless table's identity does not move.
    ///
    /// Both halves matter. The first: two tables differing only in a
    /// stopword set hold different terms, so reopening one with the
    /// other's options must mismatch rather than silently query a
    /// differently-analyzed index. The second: the filters ride the
    /// *derived* analyzer identity rather than a block of their own, and
    /// a column with no filter derives to its plain tokenizer name — so
    /// the byte stream for every table that predates the filters is
    /// unchanged and its stored hash still verifies.
    #[test]
    fn analysis_filters_join_the_hash_and_a_filterless_table_is_unchanged() {
        let strategy = time_range();
        let hash_of = |fts: FtsConfig| {
            let opts =
                SupertableOptions::new(schema_title_only(), vec![fts], vec![]).expect("options");
            record(&opts, &strategy)
        };

        let plain = hash_of(FtsConfig::new("title"));
        let stopped = hash_of(FtsConfig::new("title").stopwords(Stopwords::English));
        let stemmed = hash_of(FtsConfig::new("title").stemmer(Stemmer::English));
        let both = hash_of(
            FtsConfig::new("title")
                .stopwords(Stopwords::English)
                .stemmer(Stemmer::English),
        );
        for (label, h) in [
            ("stopwords", &stopped),
            ("stemmer", &stemmed),
            ("both", &both),
        ] {
            assert_ne!(
                &plain, h,
                "{label}: a filtered column must not hash like an unfiltered one"
            );
        }
        assert_ne!(&stopped, &stemmed, "the two filters are distinguishable");
        assert_ne!(&stopped, &both);
        assert_ne!(&stemmed, &both);

        // Declaring the filters off explicitly is the same table as not
        // mentioning them, so an existing table's stored hash still
        // verifies after this feature ships.
        assert_eq!(
            plain,
            hash_of(
                FtsConfig::new("title")
                    .stopwords(Stopwords::None)
                    .stemmer(Stemmer::None)
            ),
            "a filterless column must hash exactly as it did before \
             filters existed"
        );
    }

    #[test]
    fn options_hash_mismatch_is_error_impl() {
        // Trait-object usage exercises the
        // `impl std::error::Error for OptionsHashMismatch` — a
        // no-op body but the impl block needs to compile and
        // the dyn-error conversion needs to succeed.
        let opts = fts_opts();
        let err = verify_options_hash(&opts, &time_range(), ContentHash([4u8; 32]))
            .expect_err("mismatch");
        let dyn_err: Box<dyn Error> = Box::new(err);
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
}
