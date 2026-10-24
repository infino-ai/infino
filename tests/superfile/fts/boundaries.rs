// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Oracle tests at the posting-layout boundaries the planted corpora in
//! [`super::brute_force_oracle`] and the random ones in
//! [`super::fuzz_oracle`] do not reach on purpose.
//!
//! Every test here plants a corpus whose term frequencies put one list
//! exactly on an encoding or kernel boundary — the short-form cap, the
//! patched-block exception cap, a word-aligned bitset origin, a density
//! change inside one list, the last partial block, the second coarse
//! span, a phrase member many blocks past its driver, a negated block's
//! last document — and grades every relevant query shape against the
//! textbook reference, match set and score by value. Document lengths
//! stay under the length quantizer's exact region, so a score that
//! differs is a kernel disagreement, not a rounding one.
//!
//! The reader exposes no accessor for the encoding a block took, so
//! each corpus is planted from the encoder's own selection rules
//! (`posting::encode_block`): a block is a bitset when its presence
//! words are no larger than its tightest delta packing, patched when
//! the term is granted patching (document frequency at or under the
//! builder's cap) and the packed-plus-exceptions form is smaller than
//! plain packing, and short-form when the whole list fits one block.
//! Each test's doc states the arithmetic that puts its list where it
//! says.

use std::{collections::HashSet, iter::repeat_n};

use infino::{
    superfile::{
        SuperfileReader,
        builder::FtsConfig,
        fts::{posting::BLOCK_LEN, reader::BoolMode},
    },
    test_helpers::{brute_force_bm25::BruteForceBm25, default_tokenizer},
};

use super::{
    brute_force_oracle::build_infino_superfile_with_fts,
    corpus_truth::docs_with,
    phrase::{assert_matches_oracle, search_hits},
};

/// The one indexed column of every fixture.
const COLUMN: &str = "title";

/// Documents in the short-form boundary corpus. `s128` and `s129` are
/// planted every eleventh document, so both lists fit well inside it.
const SHORT_FORM_DOCS: u64 = 1_500;
/// Planting stride of the two boundary terms.
const SHORT_FORM_STRIDE: u64 = 11;
/// Offset of `s129`'s stride from `s128`'s so the two lists interleave
/// rather than coincide.
const SHORT_FORM_OFFSET: u64 = 5;
/// `tail` follows the boundary term in every third of its documents,
/// so each boundary term is a phrase's first member with a partial
/// phrase match set.
const SHORT_FORM_TAIL_EVERY: u64 = 3;

/// Documents in the delta-exception corpus: one list of consecutive
/// documents with a single gap of [`DELTA_EXCEPTION_GAP`] in the middle
/// of its first block.
const DELTA_EXCEPTION_DOCS: u64 = 100_200;
/// The gap. Seventeen bits as a delta, against one bit for every other
/// lane of the block.
const DELTA_EXCEPTION_GAP: u64 = 100_000;
/// Documents before the gap: one short of a block, so the gap's delta
/// is the block's last lane.
const DELTA_EXCEPTION_HEAD: u64 = BLOCK_LEN as u64 - 1;

/// Documents in the tf-exception corpus.
const TF_EXCEPTION_DOCS: u64 = 1_000;
/// Planting stride of `spiky`: three, so its blocks are not dense
/// enough for the bitset form (six presence words against four bytes
/// of two-bit deltas).
const TF_EXCEPTION_STRIDE: u64 = 3;
/// The one document with a repeated term.
const TF_EXCEPTION_DOC: u64 = 300;
/// Its term frequency: four bits against one for every other lane, and
/// still a document under the exact-length region.
const TF_EXCEPTION_TF: usize = 12;

/// Gap between the outlier lanes of the exception-cap corpus: ten bits
/// as a delta.
const EXCEPTION_CAP_GAP: u64 = 1_000;
/// Consecutive documents before the outliers, so a block with
/// [`EXCEPTION_CAP_MAX`] outliers is exactly full.
const EXCEPTION_CAP_HEAD: u64 = BLOCK_LEN as u64 - EXCEPTION_CAP_MAX;
/// The most exceptions a patched block may carry (the header field is
/// five bits). Restated from the encoder so the fixture is pinned to
/// the number, not to a name that could move.
const EXCEPTION_CAP_MAX: u64 = 31;
/// Postings after the first block, so each list has two blocks (a
/// list of one block would take the short form).
const EXCEPTION_CAP_TAIL: u64 = 10;

/// The largest document frequency the builder grants patching to.
/// Restated from the builder for the same reason.
const PATCHED_DF_CAP: u64 = 16 * 1024;
/// Gap planted every block in the df-cap corpus so each block has one
/// delta outlier and the patched form is the smaller one.
const PATCHED_DF_CAP_GAP: u64 = 300;

/// Documents in the unaligned-bitset corpus.
const UNALIGNED_BITSET_DOCS: u64 = 600;
/// `dense` is in every document except these two: skipping document 0
/// puts the first block's origin below its first document, skipping
/// [`BLOCK_LEN`] gives that block one two-wide delta so its presence
/// words (three) fit under its tight packing (four bytes per lane
/// pair).
const UNALIGNED_BITSET_SKIPPED: [u64; 2] = [0, BLOCK_LEN as u64];
/// The rare driver's documents, chosen around word and block edges.
const UNALIGNED_DRIVER_DOCS: [u64; 6] = [1, 64, 65, 128, 129, 300];

/// Documents in the density-transition corpus.
const DENSITY_DOCS: u64 = 6_000;
/// Where the dense run of `frontdense` ends (sixteen full blocks of
/// consecutive documents, all bitset).
const DENSITY_FRONT_END: u64 = 2_048;
/// The dense run of `backdense`, placed after a sparse head.
const DENSITY_BACK_START: u64 = 2_048;
const DENSITY_BACK_END: u64 = 4_096;
/// Stride of both terms outside their dense run: sparse enough that
/// those blocks pack (thirteen presence words against two-bit deltas).
const DENSITY_SPARSE_STRIDE: u64 = 10;
/// Stride of the rarer driver.
const DENSITY_DRIVER_STRIDE: u64 = 37;

/// Full blocks in the anchor corpus: one whole coarse span plus one
/// block, then a partial block.
const ANCHOR_FULL_BLOCKS: u64 = 33;
/// Documents in the partial last block.
const ANCHOR_PARTIAL: u64 = 50;
/// Uniform document length, so score is monotonic in term frequency.
const ANCHOR_DOC_LEN: usize = 6;
/// `(document, term frequency)` of the planted anchors: two in the
/// partial last block, two in the block that opens the second coarse
/// span. Frequencies strictly decrease in this order.
const ANCHORS: [(u64, usize); 4] = [(4_270, 5), (4_250, 4), (4_100, 3), (4_200, 2)];

/// Documents in the far-member phrase corpus.
const FAR_PHRASE_DOCS: u64 = 8_000;
/// Documents where `aa` occurs, and how: the phrase in order, reversed,
/// or with a token between.
const FAR_PHRASE_IN_ORDER: [u64; 2] = [3_100, 6_100];
const FAR_PHRASE_REVERSED: u64 = 100;
const FAR_PHRASE_SPLIT: u64 = 7_900;

/// Documents in the negation-edge corpus.
const NEGATION_EDGE_DOCS: u64 = 1_000;
/// `neg`'s two full blocks: documents `0..=127` and `300..=427`, so its
/// blocks' last documents are 127 and 427.
const NEGATION_EDGE_SECOND_BLOCK_START: u64 = 300;
/// `pos` sits on and just past each edge, and past the list's end.
const NEGATION_EDGE_POS: [u64; 7] = [127, 128, 299, 300, 427, 428, 600];
/// `neg2` is one short-form list, documents `0..=99`.
const NEGATION_EDGE_SHORT_END: u64 = 99;
const NEGATION_EDGE_POS2: [u64; 3] = [99, 100, 700];

/// A planted corpus: `tokens(doc)` for every document, plus a token
/// unique to the document so no two rows are identical.
fn planted(n_docs: u64, tokens: impl Fn(u64) -> Vec<&'static str>) -> Vec<(u64, String)> {
    (0..n_docs)
        .map(|d| {
            let mut toks = tokens(d);
            let unique = format!("d{d}");
            let mut text = toks.join(" ");
            if !toks.is_empty() {
                text.push(' ');
            }
            text.push_str(&unique);
            toks.clear();
            (d, text)
        })
        .collect()
}

/// The reader and the reference over one corpus, positional or not.
fn fixture(corpus: &[(u64, String)], positions: bool) -> (SuperfileReader, BruteForceBm25) {
    let refs: Vec<(u64, &str)> = corpus.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let reader =
        build_infino_superfile_with_fts(&refs, FtsConfig::new(COLUMN).positions(positions));
    let tok = default_tokenizer();
    (reader, BruteForceBm25::index(&refs, tok.as_ref()))
}

/// Grade `queries` under `mode` with `k` covering every match.
async fn assert_all(
    reader: &SuperfileReader,
    oracle: &BruteForceBm25,
    queries: &[&str],
    mode: BoolMode,
    k: usize,
) {
    for q in queries {
        assert_matches_oracle(reader, oracle, q, mode, k).await;
    }
}

/// The reader's hit ids for `query` in score order — the shared
/// `search_hits` adapter with the scores dropped, for the tests that
/// assert placement rather than value.
async fn ordered_hits(reader: &SuperfileReader, query: &str, k: usize) -> Vec<u64> {
    search_hits(reader, query, k, BoolMode::Or)
        .await
        .into_iter()
        .map(|(d, _)| d)
        .collect()
}

// ── the short-form cap ────────────────────────────────────────────────

/// Two lists straddle the one-block cap: `s128` fits the short form
/// (a varint body, decoded whole into the single-block cursor), `s129`
/// is one posting past it and takes the long form with a skip table
/// and a second, one-document block. Both must behave identically as a
/// scored term, a union and intersection partner, a phrase member and
/// a negated term. A short-form decoder that dropped or duplicated a
/// posting, or a long-form list whose second block held only its
/// padding, would fail here and nowhere in the small oracles, whose
/// lists never reach the cap.
#[tokio::test]
async fn short_form_cap_terms_agree_with_the_oracle() {
    let corpus = planted(SHORT_FORM_DOCS, |d| {
        let mut t = vec!["every"];
        let s128 = d % SHORT_FORM_STRIDE == 0 && d / SHORT_FORM_STRIDE < BLOCK_LEN as u64;
        let s129 = d % SHORT_FORM_STRIDE == SHORT_FORM_OFFSET
            && d / SHORT_FORM_STRIDE < BLOCK_LEN as u64 + 1;
        if s128 {
            t.push("s128");
        }
        if s129 {
            t.push("s129");
        }
        if (s128 || s129) && (d / SHORT_FORM_STRIDE).is_multiple_of(SHORT_FORM_TAIL_EVERY) {
            t.push("tail");
        }
        t
    });
    assert_eq!(docs_with(&corpus, "s128").len(), BLOCK_LEN);
    assert_eq!(docs_with(&corpus, "s129").len(), BLOCK_LEN + 1);
    let (reader, oracle) = fixture(&corpus, true);
    let k = corpus.len();
    assert_all(
        &reader,
        &oracle,
        &[
            "s128",
            "s129",
            "s128 s129",
            "s128 every",
            "s129 every",
            "+s128 +every",
            "+s129 +every",
            "+every s128",
            "+every s129",
            "\"s128 tail\"",
            "\"s129 tail\"",
            "every -s128",
            "every -s129",
            "s128 -tail",
            "s129 -\"s129 tail\"",
        ],
        BoolMode::Or,
        k,
    )
    .await;
    assert_all(
        &reader,
        &oracle,
        &["s128 every", "s129 every"],
        BoolMode::And,
        k,
    )
    .await;
}

// ── patched blocks ────────────────────────────────────────────────────

/// A list of consecutive documents with one gap of a hundred thousand
/// in its first block. Plain packing would spend seventeen bits on all
/// 128 lanes; the patched form packs one bit per lane and carries the
/// gap as its only exception, so the encoder takes it (the term's
/// frequency is far under the patching cap, and a block spanning a
/// hundred thousand ids is nowhere near dense enough for a bitset). A
/// decoder that patched the wrong lane, or prefix-summed before
/// patching, would misplace every document after the gap.
#[tokio::test]
async fn patched_block_with_one_delta_exception_agrees_with_the_oracle() {
    let corpus = planted(DELTA_EXCEPTION_DOCS, |d| {
        let mut t = vec!["every"];
        if d % 2 == 1 {
            t.push("odd");
        }
        if !(DELTA_EXCEPTION_HEAD..DELTA_EXCEPTION_HEAD + DELTA_EXCEPTION_GAP).contains(&d) {
            t.push("gapped");
        }
        t
    });
    let (reader, oracle) = fixture(&corpus, false);
    let k = corpus.len();
    assert_all(
        &reader,
        &oracle,
        &["gapped", "+gapped +every", "gapped -odd", "+every gapped"],
        BoolMode::Or,
        k,
    )
    .await;
}

/// One document repeats `spiky` twelve times where every other carries
/// it once: the block's tf lanes pack at one bit with a single
/// four-bit exception, which is smaller than four bits for all 128
/// lanes, and the list's stride keeps it out of the bitset form. A
/// tf-stream patch applied to the wrong lane would score two documents
/// wrongly and rank the repeat under its neighbours.
#[tokio::test]
async fn patched_block_with_one_tf_exception_agrees_with_the_oracle() {
    let corpus = planted(TF_EXCEPTION_DOCS, |d| {
        let mut t = Vec::new();
        if d % TF_EXCEPTION_STRIDE == 0 {
            let tf = if d == TF_EXCEPTION_DOC {
                TF_EXCEPTION_TF
            } else {
                1
            };
            t.extend(repeat_n("spiky", tf));
        }
        t.push("every");
        t
    });
    let (reader, oracle) = fixture(&corpus, false);
    let k = corpus.len();
    assert_all(
        &reader,
        &oracle,
        &["spiky", "spiky every", "+spiky +every"],
        BoolMode::Or,
        k,
    )
    .await;
    // The repeat is the best document by construction; the top-1 must
    // be it rather than a tied one-occurrence neighbour.
    assert_eq!(
        ordered_hits(&reader, "spiky", 1).await,
        vec![TF_EXCEPTION_DOC]
    );
}

/// Two lists whose first blocks have 31 and 32 ten-bit outlier lanes
/// among one-bit ones. Thirty-one is the most exceptions a patched
/// block can carry, so `cap31`'s block is patched at width one with a
/// full exception list; `cap32`'s cannot be patched at any width below
/// the plain one (every narrower width leaves all 32 outliers as
/// exceptions) and stays plain. Both must decode to the planted
/// documents: an exception list read one entry short, or a header
/// whose five-bit count overflowed, would drop or misplace documents.
#[tokio::test]
async fn patched_blocks_at_the_exception_cap_agree_with_the_oracle() {
    let outlier_docs = |head: u64, outliers: u64| -> HashSet<u64> {
        let mut set: HashSet<u64> = (0..head).collect();
        set.extend((1..=outliers).map(|i| head - 1 + i * EXCEPTION_CAP_GAP));
        let last = head - 1 + outliers * EXCEPTION_CAP_GAP;
        set.extend((1..=EXCEPTION_CAP_TAIL).map(|i| last + i));
        set
    };
    let cap31 = outlier_docs(EXCEPTION_CAP_HEAD, EXCEPTION_CAP_MAX);
    let cap32 = outlier_docs(EXCEPTION_CAP_HEAD - 1, EXCEPTION_CAP_MAX + 1);
    let n_docs = cap31.iter().chain(&cap32).max().copied().expect("planted") + 1;
    assert_eq!(cap31.len(), BLOCK_LEN + EXCEPTION_CAP_TAIL as usize);
    assert_eq!(cap32.len(), BLOCK_LEN + EXCEPTION_CAP_TAIL as usize);
    let corpus = planted(n_docs, |d| {
        let mut t = vec!["every"];
        if cap31.contains(&d) {
            t.push("cap31");
        }
        if cap32.contains(&d) {
            t.push("cap32");
        }
        t
    });
    let (reader, oracle) = fixture(&corpus, false);
    let k = corpus.len();
    assert_all(
        &reader,
        &oracle,
        &[
            "cap31",
            "cap32",
            "cap31 cap32",
            "+cap31 +cap32",
            "cap31 -cap32",
            "+every cap32",
        ],
        BoolMode::Or,
        k,
    )
    .await;
    assert_all(
        &reader,
        &oracle,
        &["cap31 every", "cap32 cap31"],
        BoolMode::And,
        k,
    )
    .await;
}

/// Two lists on either side of the builder's patching cap: `pmax` has
/// exactly the cap's document frequency and may be patched, `pover` has
/// one more and may not. Both are planted with one long gap per block,
/// so `pmax`'s blocks all take the patched form and `pover`'s all stay
/// plain, and the same query shapes must agree on both. A grant that
/// applied at the wrong frequency changes only the bytes, so this pins
/// that the two forms decode identically at scale (128 blocks each).
#[tokio::test]
async fn lists_at_the_patched_df_cap_agree_with_the_oracle() {
    // Postings `i` map to document `i + (i / BLOCK_LEN) * GAP`: 127
    // consecutive documents, then a jump, block after block.
    let doc_of = |i: u64| i + (i / BLOCK_LEN as u64) * PATCHED_DF_CAP_GAP;
    let pover: HashSet<u64> = (0..=PATCHED_DF_CAP).map(doc_of).collect();
    let pmax: HashSet<u64> = (0..PATCHED_DF_CAP).map(doc_of).collect();
    let extra = doc_of(PATCHED_DF_CAP);
    let n_docs = extra + 1;
    let corpus = planted(n_docs, |d| {
        let mut t = vec!["every"];
        if pmax.contains(&d) {
            t.push("pmax");
        }
        if pover.contains(&d) {
            t.push("pover");
        }
        t
    });
    assert_eq!(docs_with(&corpus, "pmax").len() as u64, PATCHED_DF_CAP);
    assert_eq!(docs_with(&corpus, "pover").len() as u64, PATCHED_DF_CAP + 1);
    let (reader, oracle) = fixture(&corpus, false);
    let k = corpus.len();
    assert_all(
        &reader,
        &oracle,
        &[
            "pmax",
            "pover",
            "+pmax +pover",
            "pover -pmax",
            "pmax -pover",
        ],
        BoolMode::Or,
        k,
    )
    .await;
    assert_eq!(ordered_hits(&reader, "pover -pmax", k).await, vec![extra]);
}

// ── bitset blocks ─────────────────────────────────────────────────────

/// `dense`'s first block holds documents 1..=127 and 129: its presence
/// bitset starts at the word-aligned origin 0, one below its first
/// document, and spans three words (24 bytes) against 32 bytes of
/// two-bit deltas, so the encoder stores it as a bitset. Every walk
/// that bit-tests the block — the intersection's membership probe, the
/// phrase member check, the negation filter — must translate a document
/// id through that origin rather than through the block's first
/// document, or document 1 reads as document 0 and 129 as 128. The
/// driver's documents sit on both sides of every such edge.
#[tokio::test]
async fn bitset_block_with_an_unaligned_origin_agrees_with_the_oracle() {
    let driver: HashSet<u64> = UNALIGNED_DRIVER_DOCS.into_iter().collect();
    let corpus = planted(UNALIGNED_BITSET_DOCS, |d| {
        let mut t = vec!["filler"];
        let dense = !UNALIGNED_BITSET_SKIPPED.contains(&d);
        if driver.contains(&d) {
            // In order at 1, 65 and 129; reversed at 64 and 300; alone at 128.
            match d {
                1 | 65 | 129 => t.extend(["drv", "dense"]),
                64 | 300 => t.extend(["dense", "drv"]),
                _ => t.push("drv"),
            }
        } else if dense {
            t.push("dense");
        }
        t
    });
    assert_eq!(
        docs_with(&corpus, "dense").len() as u64,
        UNALIGNED_BITSET_DOCS - 2
    );
    let (reader, oracle) = fixture(&corpus, true);
    let k = corpus.len();
    assert_all(
        &reader,
        &oracle,
        &[
            "dense",
            "+drv +dense",
            "\"drv dense\"",
            "\"dense drv\"",
            "drv -dense",
            "filler -dense",
            "+filler -\"drv dense\" drv",
        ],
        BoolMode::Or,
        k,
    )
    .await;
    assert_eq!(
        ordered_hits(&reader, "drv -dense", k).await,
        vec![BLOCK_LEN as u64]
    );
}

/// Two lists that change encoding mid-way: `frontdense` is every
/// document for sixteen blocks and then every tenth, `backdense` every
/// tenth, then every document for sixteen blocks, then every tenth
/// again. A cursor seeking across the change must switch between the
/// bitset probe and the packed decode in both directions; a stale
/// block cache or a probe helper applied to a packed block would skip
/// or duplicate documents at the seam. Graded as an intersection with
/// a rarer driver, as a negated list, and as a union of the two.
#[tokio::test]
async fn lists_that_change_density_mid_way_agree_with_the_oracle() {
    let corpus = planted(DENSITY_DOCS, |d| {
        let mut t = Vec::new();
        if d < DENSITY_FRONT_END || d % DENSITY_SPARSE_STRIDE == 0 {
            t.push("frontdense");
        }
        if (DENSITY_BACK_START..DENSITY_BACK_END).contains(&d) || d % DENSITY_SPARSE_STRIDE == 0 {
            t.push("backdense");
        }
        if d % DENSITY_DRIVER_STRIDE == 0 {
            t.push("drv");
        }
        t.push("filler");
        t
    });
    let (reader, oracle) = fixture(&corpus, false);
    let k = corpus.len();
    assert_all(
        &reader,
        &oracle,
        &[
            "+drv +frontdense",
            "+drv +backdense",
            "+drv +frontdense +backdense",
            "drv -frontdense",
            "drv -backdense",
            "frontdense backdense",
            "filler -frontdense",
        ],
        BoolMode::Or,
        k,
    )
    .await;
    assert_all(
        &reader,
        &oracle,
        &["drv frontdense", "drv backdense"],
        BoolMode::And,
        k,
    )
    .await;
}

// ── block-max skipping at the list's edges ────────────────────────────

/// `hot` is in every document of 33 full blocks and a 50-document
/// partial block; every document is the same length, so the score is
/// strictly monotonic in term frequency and the four planted repeats
/// form a tie-free head. Two of them are in the partial last block and
/// two in block 32, the first block of the second coarse span, while
/// blocks 0..31 hold only single occurrences that fill the heap to the
/// bulk score first. The walk must read the partial block's real
/// document count rather than the padded lanes, and must read the
/// second span's coarse maximum and the block maxima under it instead
/// of skipping the span on the first one; either fault drops an anchor.
#[tokio::test]
async fn top_k_anchors_in_the_last_partial_block_and_the_second_coarse_span() {
    let n_docs = ANCHOR_FULL_BLOCKS * BLOCK_LEN as u64 + ANCHOR_PARTIAL;
    let corpus = planted(n_docs, |d| {
        let tf = ANCHORS
            .iter()
            .find(|(a, _)| *a == d)
            .map_or(1, |(_, tf)| *tf);
        let mut t = vec!["hot"; tf];
        // One token is the unique id appended by the corpus builder.
        t.extend(repeat_n("pad", ANCHOR_DOC_LEN - tf - 1));
        t
    });
    let (reader, oracle) = fixture(&corpus, false);
    let tok = default_tokenizer();
    let expected: Vec<u64> = ANCHORS.iter().map(|(d, _)| *d).collect();
    for k in [ANCHORS.len(), 10, 200] {
        let head: Vec<u64> = ordered_hits(&reader, "hot", k)
            .await
            .into_iter()
            .take(ANCHORS.len())
            .collect();
        assert_eq!(head, expected, "k={k}: anchors missing or misordered");
        let oracle_head: Vec<u64> = oracle
            .top_k("hot", k, tok.as_ref())
            .into_iter()
            .take(ANCHORS.len())
            .map(|(d, _)| d)
            .collect();
        assert_eq!(head, oracle_head, "k={k}: disagrees with the reference");
    }
    assert_matches_oracle(&reader, &oracle, "hot", BoolMode::Or, corpus.len()).await;
}

// ── phrase member far past the driver ─────────────────────────────────

/// `bb` is in every document (63 blocks); `aa` is in four, so the
/// phrase walk drives on `aa` and positions `bb`'s cursor by block seek
/// to each candidate — 23 blocks past its previous position for the
/// second candidate, another 23 for the third. Adjacency holds at two
/// candidates, is reversed at one and split at one. A seek that landed
/// a block early or late would verify positions against the wrong
/// document and accept or reject the wrong candidate.
#[tokio::test]
async fn phrase_member_seeks_many_blocks_past_the_driver() {
    let corpus = planted(FAR_PHRASE_DOCS, |d| match d {
        _ if FAR_PHRASE_IN_ORDER.contains(&d) => vec!["aa", "bb"],
        FAR_PHRASE_REVERSED => vec!["bb", "aa"],
        FAR_PHRASE_SPLIT => vec!["aa", "gap", "bb"],
        _ => vec!["bb"],
    });
    let (reader, oracle) = fixture(&corpus, true);
    let k = corpus.len();
    assert_all(
        &reader,
        &oracle,
        &["\"aa bb\"", "\"bb aa\"", "+aa -\"aa bb\"", "\"aa bb\" aa"],
        BoolMode::Or,
        k,
    )
    .await;
    let mut got = ordered_hits(&reader, "\"aa bb\"", k).await;
    got.sort_unstable();
    assert_eq!(got, FAR_PHRASE_IN_ORDER.to_vec());
}

// ── negation probes at block edges ────────────────────────────────────

/// The negation filter skip-probes the negated list toward each
/// candidate. `neg` has two full blocks ending at documents 127 and
/// 427; `pos` sits on each of those last documents, one past each, on
/// and before the second block's first document, and past the list's
/// end. A probe that treated a block's last document as already passed
/// would admit 127 and 427; one that did not stop at the list's end
/// would drop 600 or read past the block. `neg2` is a short-form list,
/// so the same edges are probed through the single-block cursor.
#[tokio::test]
async fn negation_probes_at_block_edges_agree_with_the_oracle() {
    let corpus = planted(NEGATION_EDGE_DOCS, |d| {
        let mut t = vec!["filler"];
        let first_block = d < BLOCK_LEN as u64;
        let second_block = (NEGATION_EDGE_SECOND_BLOCK_START
            ..NEGATION_EDGE_SECOND_BLOCK_START + BLOCK_LEN as u64)
            .contains(&d);
        if first_block || second_block {
            t.push("neg");
        }
        if NEGATION_EDGE_POS.contains(&d) {
            t.push("pos");
        }
        if d <= NEGATION_EDGE_SHORT_END {
            t.push("neg2");
        }
        if NEGATION_EDGE_POS2.contains(&d) {
            t.push("pos2");
        }
        t
    });
    assert_eq!(docs_with(&corpus, "neg").len(), 2 * BLOCK_LEN);
    let (reader, oracle) = fixture(&corpus, false);
    let k = corpus.len();
    assert_all(
        &reader,
        &oracle,
        &[
            "pos -neg",
            "pos2 -neg2",
            "pos -neg -neg2",
            "filler -neg",
            "+pos -neg2",
        ],
        BoolMode::Or,
        k,
    )
    .await;
    let mut got = ordered_hits(&reader, "pos -neg", k).await;
    got.sort_unstable();
    assert_eq!(got, vec![128, 299, 428, 600]);
    let mut got = ordered_hits(&reader, "pos2 -neg2", k).await;
    got.sort_unstable();
    assert_eq!(got, vec![100, 700]);
}
