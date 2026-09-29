// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS compaction-merge diagnostic: times the streaming
//! `SuperfileBuilder::build_from_readers_fts_merge_to` (the path compaction
//! uses) over N prebuilt inputs, writing to a byte-counting sink.
//!
//! The corpus has one positional, index-only FTS column. Each doc carries Zipfian
//! tokens from a shared vocabulary plus a few doc-unique tokens, so most
//! terms appear in exactly one document. Merge cost tracks term count
//! rather than row count, so this is the shape that matters.
//!
//! The default size is well past the FTS spill threshold, the large-merge
//! case; set `INFINO_BENCH_FTS_MERGE_INPUTS` low (e.g. 4) for a small merge.
//!
//! ```text
//! cargo bench -- fts-merge
//! INFINO_BENCH_FTS_MERGE_INPUTS=32 INFINO_BENCH_FTS_MERGE_DOCS=50000 cargo bench -- fts-merge
//! INFINO_BENCH_FTS_MERGE_DELETE_EVERY=10 cargo bench -- fts-merge   # tombstone every 10th row
//! ```

use std::{
    collections::HashMap,
    io::{self, Write},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    roaring::RoaringBitmap,
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
    },
};
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::{corpus::ZipfDistribution, diag_common::env_usize};

/// Default number of merge inputs.
const DEFAULT_INPUTS: usize = 40;
/// Default docs per input.
const DEFAULT_DOCS_PER_INPUT: usize = 50_000;
/// Default timed merge runs; the median is reported.
const DEFAULT_RUNS: usize = 3;
/// Shared vocabulary size.
const VOCAB_SIZE: usize = 20_000;
/// Zipfian tokens per doc, drawn from the shared vocabulary.
const SHARED_TOKENS_PER_DOC: usize = 60;
/// Doc-unique tokens per doc; these become `df = 1` terms.
const UNIQUE_TOKENS_PER_DOC: usize = 5;
/// Base RNG seed; each input adds its index.
const SEED: u64 = 0x05ee_df75;
/// Decimal128 precision of the id column.
const ID_PRECISION: u8 = 38;

const ID_COLUMN: &str = "doc_id";
const TEXT_COLUMN: &str = "text";

pub fn run() {
    let n_inputs = env_usize("INFINO_BENCH_FTS_MERGE_INPUTS", DEFAULT_INPUTS);
    let docs = env_usize("INFINO_BENCH_FTS_MERGE_DOCS", DEFAULT_DOCS_PER_INPUT);
    let runs = env_usize("INFINO_BENCH_FTS_MERGE_RUNS", DEFAULT_RUNS).max(1);
    let delete_every = env_usize("INFINO_BENCH_FTS_MERGE_DELETE_EVERY", 0);

    let schema = Arc::new(Schema::new(vec![
        Field::new(ID_COLUMN, DataType::Decimal128(ID_PRECISION, 0), false),
        Field::new(TEXT_COLUMN, DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        ID_COLUMN,
        vec![FtsConfig::new(TEXT_COLUMN).positions(true).stored(false)],
        vec![],
    );

    let build_start = Instant::now();
    let input_bytes = AtomicUsize::new(0);
    let zipf = ZipfDistribution::new(VOCAB_SIZE);
    let inputs: Vec<(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)> = (0..n_inputs)
        .into_par_iter()
        .map(|i| {
            let bytes = build_input(&opts, &schema, &zipf, i, docs);
            input_bytes.fetch_add(bytes.len(), Ordering::Relaxed);
            let reader = SuperfileReader::open(Bytes::from(bytes)).expect("open input");
            let deleted = (delete_every > 0).then(|| {
                Arc::new(
                    (0..docs as u32)
                        .step_by(delete_every)
                        .collect::<RoaringBitmap>(),
                )
            });
            (Arc::new(reader), deleted)
        })
        .collect();

    let input_bytes = input_bytes.into_inner();
    println!(
        "fts-merge: {n_inputs} inputs x {docs} docs, delete_every={delete_every}, \
         built in {:.1}s",
        build_start.elapsed().as_secs_f64()
    );
    println!("  inputs {} MiB", input_bytes >> 20);

    let deleted_per_input = if delete_every > 0 {
        docs.div_ceil(delete_every)
    } else {
        0
    };
    let expected_docs = (n_inputs * (docs - deleted_per_input)) as u64;

    let mut times = Vec::with_capacity(runs);
    for run in 0..runs {
        let mut sink = ByteCounter(0);
        let start = Instant::now();
        // Empty corpus stats: a standalone merge averages lengths over its own docs.
        let stats =
            SuperfileBuilder::build_from_readers_fts_merge_to(&inputs, &HashMap::new(), &mut sink)
                .expect("merge");
        let secs = start.elapsed().as_secs_f64();
        assert_eq!(stats.n_docs, expected_docs, "merged doc count");
        println!("  run {}: {secs:.2}s, output {} MiB", run + 1, sink.0 >> 20);
        times.push(secs);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "fts-merge median {:.2}s (min {:.2}s, max {:.2}s)",
        times[times.len() / 2],
        times[0],
        times[times.len() - 1]
    );
}

/// Build one input superfile. Doc-unique tokens embed the global doc
/// number, so they are unique across inputs too.
fn build_input(
    opts: &BuilderOptions,
    schema: &Arc<Schema>,
    zipf: &ZipfDistribution,
    input: usize,
    docs: usize,
) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(SEED + input as u64);
    let first = input * docs;
    let texts: Vec<String> = (first..first + docs)
        .map(|doc| {
            let mut text = String::new();
            for k in 0..UNIQUE_TOKENS_PER_DOC {
                text.push_str(&format!("u{doc:08}x{k} "));
            }
            for _ in 0..SHARED_TOKENS_PER_DOC {
                text.push_str(&format!("w{:05} ", zipf.sample(&mut rng)));
            }
            text
        })
        .collect();
    let ids: Decimal128Array = (first..first + docs)
        .map(|i| Some(i as i128))
        .collect::<Decimal128Array>()
        .with_precision_and_scale(ID_PRECISION, 0)
        .expect("decimal128 precision");
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids), Arc::new(LargeStringArray::from(texts))],
    )
    .expect("build RecordBatch");
    let mut b = SuperfileBuilder::new(opts.clone()).expect("new SuperfileBuilder");
    b.add_batch(&batch, &[]).expect("add_batch");
    b.finish().expect("finish input")
}

/// Write sink that only counts bytes, so the timing leaves out disk I/O.
struct ByteCounter(usize);

impl Write for ByteCounter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0 += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
