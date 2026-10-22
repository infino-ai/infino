//! Test-only census of a built index's postings under alternative block
//! encodings. It streams every term of a real superfile, re-encodes each
//! long-form posting list under a set of policies with the same planners
//! the writer uses, and reports bytes and block shapes per document
//! frequency bucket. The baseline policy is checked block by block against
//! [`encode_block`], so the other policies are measured with a model that
//! reproduces today's bytes exactly.
//!
//! ```text
//! INFINO_CENSUS_FILE=/path/to/superfile \
//!   cargo test --release --lib posting_census -- --ignored --nocapture
//! ```

use std::{env, fs};

use bytes::Bytes;

use super::*;
use crate::superfile::{
    bits::{ExceptionPlan, PackScratch, plan_exceptions},
    format::fts::{COARSE_BLOCK_MAX_SPAN, coarse_slot, skip_entry},
    reader::SuperfileReader,
};

/// One encoding policy to price.
#[derive(Clone, Copy)]
struct Policy {
    name: &'static str,
    /// Docs per block.
    n: usize,
    /// Store `tf - 1`, so an all-ones block packs at width 0.
    tf_minus_one: bool,
    /// A block of consecutive doc ids carries no doc payload.
    run: bool,
    /// Take the bitset whenever it is at most `bitset_bias` times the
    /// tight packed delta size (1.0 is the size-only rule)...
    bitset_bias: f32,
    /// ...for terms with at least this many documents; 1.0 below it.
    bias_min_df: u64,
}

const fn policy(
    name: &'static str,
    run: bool,
    tf_minus_one: bool,
    bias: f32,
    min_df: u64,
) -> Policy {
    Policy {
        name,
        n: BLOCK_LEN,
        tf_minus_one,
        run,
        bitset_bias: bias,
        bias_min_df: min_df,
    }
}

const POLICIES: &[Policy] = &[
    policy(
        "before (size-only bitset, tf stored as is)",
        false,
        false,
        1.0,
        0,
    ),
    policy("tf-1 + run (the encoder)", true, true, 1.0, 0),
    policy(
        "tf-1 + run + bitset 1.6x at df>=256K",
        true,
        true,
        1.6,
        262_144,
    ),
    policy(
        "tf-1 + run + bitset 2x at df>=256K",
        true,
        true,
        2.0,
        262_144,
    ),
    policy(
        "tf-1 + run + bitset 3x at df>=256K",
        true,
        true,
        3.0,
        262_144,
    ),
    policy(
        "tf-1 + run + bitset 1.6x at df>=16K",
        true,
        true,
        1.6,
        16_384,
    ),
    policy("tf-1 + run + bitset 2x at df>=16K", true, true, 2.0, 16_384),
];

/// The policy [`encode_block`] implements, checked block by block.
const VALIDATED: usize = 1;

/// Document-frequency buckets the report splits by.
const BUCKETS: &[(u64, u64, &str)] = &[
    (129, 1_024, "129..1K"),
    (1_025, 16_384, "1K..16K"),
    (16_385, 262_144, "16K..256K"),
    (262_145, u64::MAX, ">256K"),
];

/// Bytes the positional column spends per block outside the block itself.
const SKIP_BYTES: usize = skip_entry::POSITIONAL_BYTES;

#[derive(Default, Clone)]
struct Tally {
    terms: u64,
    postings: u64,
    blocks: u64,
    header: u64,
    docs: u64,
    tfs: u64,
    skip: u64,
    coarse: u64,
    bitset_blocks: u64,
    patched_blocks: u64,
    run_blocks: u64,
    ones_blocks: u64,
}

impl Tally {
    fn total(&self) -> u64 {
        self.header + self.docs + self.tfs + self.skip + self.coarse
    }
}

fn width_of(v: u32) -> usize {
    (32 - v.leading_zeros()) as usize
}

/// Packed size of `n` lanes at `width` bits.
fn packed(n: usize, width: usize) -> usize {
    n * width / 8
}

/// The cheapest patched size of `lanes`, as the writer plans it.
fn patched_size(lanes: &[u32], plan: &mut ExceptionPlan, cand: &mut Vec<(u32, u32)>) -> usize {
    let n = lanes.len();
    plan_exceptions(
        lanes,
        PATCHED_MAX_EXCEPTIONS,
        |w| n * w as usize / 8,
        |_, hi| 1 + varint_len(hi),
        plan,
        cand,
    );
    plan.bytes
}

/// What one block costs under `p`: (header, doc bytes, tf bytes, kind).
/// `kind`: 0 packed, 1 bitset, 2 patched, 3 run.
fn model_block(
    p: &Policy,
    df: u64,
    docs: &[u32],
    tfs: &[u32],
    prev: Option<u32>,
    patchable: bool,
    s: &mut PackScratch,
) -> (usize, usize, usize, u8) {
    let n = p.n;
    let count = docs.len();
    let last = docs[count - 1];
    let base = prev.unwrap_or(0);
    let mut deltas = vec![0u32; n];
    deltas[0] = docs[0] - base;
    for i in 1..count {
        deltas[i] = docs[i] - docs[i - 1];
    }
    let mut tf_lanes = vec![0u32; n];
    for (l, &t) in tf_lanes.iter_mut().zip(tfs) {
        *l = if p.tf_minus_one { t - 1 } else { t };
    }
    let delta_bits = width_of(deltas.iter().copied().max().unwrap_or(0));
    let tf_bits = width_of(tf_lanes.iter().copied().max().unwrap_or(0));
    let deltas_size = packed(n, delta_bits);
    let tfs_size = packed(n, tf_bits);
    let tight = width_of(deltas[1..].iter().copied().max().unwrap_or(0).max(1));
    let aligned = bitset_block_base(docs[0]);
    let bitset_size = ((last - aligned) as usize / 64 + 1) * 8;
    let consecutive = (last - docs[0]) as usize + 1 == count;

    if p.run && consecutive {
        // Header plus the first doc's offset from the previous block.
        return (COMPACT_HEADER_SIZE + ORIGIN_BYTES, 0, tfs_size, 3);
    }
    let bias = match df >= p.bias_min_df {
        true => p.bitset_bias,
        false => 1.0,
    };
    let use_bitset = bitset_size as f32 <= packed(n, tight) as f32 * bias;
    if patchable && !use_bitset {
        let mut explicit = deltas.clone();
        explicit[count..].fill(0);
        let d = patched_size(&explicit, &mut s.plan_a, &mut s.candidates);
        let t = patched_size(&tf_lanes, &mut s.plan_b, &mut s.candidates);
        if d + t < deltas_size + tfs_size {
            return (COMPACT_HEADER_SIZE, d, t, 2);
        }
    }
    match use_bitset {
        true => (COMPACT_HEADER_SIZE + ORIGIN_BYTES, bitset_size, tfs_size, 1),
        false => (COMPACT_HEADER_SIZE, deltas_size, tfs_size, 0),
    }
}

fn bucket_of(df: u64) -> Option<usize> {
    BUCKETS
        .iter()
        .position(|&(lo, hi, _)| (lo..=hi).contains(&df))
}

#[test]
#[ignore = "reads a real superfile named by INFINO_CENSUS_FILE"]
fn posting_census() {
    let Ok(path) = env::var("INFINO_CENSUS_FILE") else {
        panic!("set INFINO_CENSUS_FILE to a superfile path");
    };
    let bytes = Bytes::from(fs::read(&path).expect("read superfile"));
    let sf = SuperfileReader::open(bytes).expect("open superfile");
    let fts = sf.fts().expect("superfile has an FTS index");
    let (dict, postings, positions) = fts.region_lens_for_census();
    println!("regions: dict {dict} B, postings {postings} B, positions {positions} B");

    let mut tallies = vec![vec![Tally::default(); BUCKETS.len()]; POLICIES.len()];
    let mut short_terms = 0u64;
    let mut inline_terms = 0u64;
    let mut mismatches = 0u64;
    let mut scratch = PackScratch::default();
    let mut real_scratch = PackScratch::default();

    let mut term: Vec<u8> = Vec::new();
    let mut docs: Vec<u32> = Vec::new();
    let mut tfs: Vec<u32> = Vec::new();
    let mut flush = |term_for_report: &[u8], docs: &mut Vec<u32>, tfs: &mut Vec<u32>| {
        let df = docs.len() as u64;
        match df {
            0 => return,
            1 => inline_terms += 1,
            2..=128 => short_terms += 1,
            _ => {}
        }
        let Some(b) = bucket_of(df) else {
            docs.clear();
            tfs.clear();
            return;
        };
        let patchable = docs.len() <= 16 * 1024;
        if df > 1_000_000 {
            // How today's rule stores the biggest terms, block by block: the
            // dense-union count folds bitset blocks as words but decodes and
            // scatters packed ones.
            let mut kinds = [0u64; 4];
            let mut prev = None;
            let mut span = 0u64;
            for (dc, tc) in docs.chunks(BLOCK_LEN).zip(tfs.chunks(BLOCK_LEN)) {
                let (_, _, _, kind) =
                    model_block(&POLICIES[0], df, dc, tc, prev, patchable, &mut scratch);
                kinds[kind as usize] += 1;
                span += u64::from(dc[dc.len() - 1] - dc[0] + 1);
                prev = Some(dc[dc.len() - 1]);
            }
            let blocks = kinds.iter().sum::<u64>().max(1);
            println!(
                "big term {:?}: df {df}, blocks {blocks}, packed {:.1}% bitset {:.1}% patched {:.1}%, mean span {:.0} ids",
                String::from_utf8_lossy(term_for_report),
                100.0 * kinds[0] as f64 / blocks as f64,
                100.0 * kinds[1] as f64 / blocks as f64,
                100.0 * kinds[2] as f64 / blocks as f64,
                span as f64 / blocks as f64,
            );
        }
        for (pi, p) in POLICIES.iter().enumerate() {
            let t = &mut tallies[pi][b];
            t.terms += 1;
            t.postings += df;
            let mut prev = None;
            let mut nblocks = 0usize;
            for (dc, tc) in docs.chunks(p.n).zip(tfs.chunks(p.n)) {
                let (h, d, f, kind) = model_block(p, df, dc, tc, prev, patchable, &mut scratch);
                if pi == VALIDATED {
                    let real = encode_block(
                        &Block {
                            doc_ids: dc.to_vec(),
                            tfs: tc.to_vec(),
                        },
                        prev,
                        patchable,
                        &mut real_scratch,
                    );
                    if real.bytes.len() != h + d + f {
                        mismatches += 1;
                    }
                }
                t.header += h as u64;
                t.docs += d as u64;
                t.tfs += f as u64;
                match kind {
                    1 => t.bitset_blocks += 1,
                    2 => t.patched_blocks += 1,
                    3 => t.run_blocks += 1,
                    _ => {}
                }
                if tc.iter().all(|&x| x == 1) {
                    t.ones_blocks += 1;
                }
                prev = Some(dc[dc.len() - 1]);
                nblocks += 1;
            }
            t.blocks += nblocks as u64;
            t.skip += (nblocks * SKIP_BYTES) as u64;
            t.coarse += (nblocks.div_ceil(COARSE_BLOCK_MAX_SPAN) * coarse_slot::BYTES) as u64;
        }
        docs.clear();
        tfs.clear();
    };

    fts.for_each_term_posting(0, |t, doc, tf, _positions| {
        if t != term.as_slice() {
            flush(&term, &mut docs, &mut tfs);
            term.clear();
            term.extend_from_slice(t);
        }
        docs.push(doc);
        tfs.push(tf);
        Ok(())
    })
    .expect("walk postings");
    flush(&term, &mut docs, &mut tfs);

    println!("inline terms {inline_terms}, short-form terms {short_terms}");
    println!("baseline model vs encode_block mismatches: {mismatches}");
    let base_total: u64 = tallies[0].iter().map(Tally::total).sum();
    for (pi, p) in POLICIES.iter().enumerate() {
        let total: u64 = tallies[pi].iter().map(Tally::total).sum();
        println!(
            "\n== {} : long-form {} B ({:+.2}% vs today)",
            p.name,
            total,
            (total as f64 / base_total as f64 - 1.0) * 100.0
        );
        println!(
            "  bucket      terms   postings   blocks    header      docs       tfs      skip+coarse  bitset% patched% run% ones%"
        );
        for (b, t) in tallies[pi].iter().enumerate() {
            let pct = |x: u64| 100.0 * x as f64 / t.blocks.max(1) as f64;
            println!(
                "  {:10} {:6} {:10} {:8} {:9} {:10} {:9} {:12} {:6.1} {:7.1} {:5.1} {:5.1}",
                BUCKETS[b].2,
                t.terms,
                t.postings,
                t.blocks,
                t.header,
                t.docs,
                t.tfs,
                t.skip + t.coarse,
                pct(t.bitset_blocks),
                pct(t.patched_blocks),
                pct(t.run_blocks),
                pct(t.ones_blocks)
            );
        }
    }
}
