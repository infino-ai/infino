//! FTS blob builder. Multi-column FTS index assembly.
//!
//! `FtsBuilder` accumulates posting lists across all FTS-indexed columns
//! into a `Vec<(u32, u32)>` per `(column, term)`, then `finish()`
//! emits the on-disk FTS blob laid out as:
//!
//! ```text
//!   header (56 bytes)
//!   FST term dictionary  + CRC32C
//!   postings region      + CRC32C
//!   doc-lengths directory   + CRC32C
//!   per-column doc-lengths arrays  (each + its own CRC32C)
//! ```
//!
//! See `docs/architecture/superfile.md` for the full byte-level spec.
//!
//! ## Builder lifecycle
//!
//! 1. `FtsBuilder::new(tokenizer)` — empty builder.
//! 2. `register_column(name)` per FTS column, in declaration order.
//! 3. `add_doc(column_id, local_doc_id, text)` per `(doc, column)` pair.
//!    Caller passes monotonically-increasing `local_doc_id`s.
//! 4. `finish()` consumes the builder, returns the FTS blob bytes.

use crate::superfile::BuildError;
use crate::superfile::format::checksum::crc32c;
use crate::superfile::format::{self, FST_SEPARATOR};
use crate::superfile::fts::dict::DictBuilder;
use crate::superfile::fts::fst_value::FstValue;
use crate::superfile::fts::posting::{BLOCK_LEN, Block, encode_block};
use crate::superfile::fts::tokenize::Tokenizer;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Per-(column, term) metadata header — 20 bytes, written immediately
/// before the term's skip table + posting blocks in the postings region.
/// `term_metadata_offset` (referenced from the FST value) points at the
/// start of this struct.
///
/// Layout:
///   off  0 ..  4 : df (u32) — bounded by n_docs per segment
///   off  4 .. 12 : postings_offset (u64) — equals the term's metadata_offset;
///                  self-describing. u64 supports segments past 4 GiB
///                  (e.g. the 16 GB target).
///   off 12 .. 16 : postings_length (u32) — single term's bytes, well under
///                  4 G even at high df (≤ ~1 MB for the most common term in
///                  a 16 GB segment).
///   off 16 .. 20 : num_blocks (u32)
///
/// `df`, `postings_length`, and `num_blocks` stay u32; only the absolute
/// offset into the postings region needs the full u64 range.
const TERM_META_SIZE: usize = 20;

/// Skip-table entry size in bytes.
const SKIP_ENTRY_SIZE: usize = 16;

/// Doc-lengths directory entry size in bytes (per column).
///
/// Layout:
///   off  0 ..  4 : column_id (u32)
///   off  4 .. 12 : doc_lengths_offset (u64) — absolute offset of this column's
///                  doc-lengths array in the FTS blob. u64 supports segments
///                  past 4 GiB.
///   off 12 .. 16 : avgdl_x1000 (u32) — avgdl × 1000, as an integer
///
/// Only the absolute offset needs u64; column_id and avgdl_x1000 stay
/// u32 (bounded by column count and doc length respectively).
const DOC_LENGTHS_ENTRY_SIZE: usize = 16;

/// Per-column build-time state.
struct ColumnState {
    name: String,
    /// One u32 per doc (token count for this column), push order
    /// matches local_doc_id order.
    doc_lengths: Vec<u32>,
    /// Total token count across every doc in this column. Used for
    /// `avgdl = total_tokens / n_docs`.
    total_tokens: u64,
}

/// Per-term posting accumulator: a `Vec` of `(local_doc_id, tf)`
/// pairs in ascending doc-id order. Pairs are pushed in `add_doc`
/// (monotonic `local_doc_id` per API contract) and consumed in
/// `finish()`. The owning column is implied by the position in
/// [`FtsBuilder::postings`] — the outer `Vec` is indexed by
/// `column_id`.
struct PostingAcc {
    list: Vec<(u32, u32)>,
}

pub struct FtsBuilder {
    tokenizer: Arc<dyn Tokenizer>,
    columns: Vec<ColumnState>,
    /// Per-column posting tables. `postings[column_id]` maps term
    /// → posting accumulator for that column. Keyed by `Box<str>`
    /// — heap-allocated copy of the term taken once on first
    /// sight; the steady-state per-doc lookup is
    /// `FxHashMap::get_mut(&str)` (via `Box<str>: Borrow<str>`),
    /// hashing only the term bytes instead of the
    /// `<col_name>\x1F<term>` byte string the prior single-map
    /// layout hashed every time. At Zipfian-multi-column scale this
    /// is the dominant cost in `add_doc`.
    ///
    /// A chained-chunk arena variant was 3.5× slower than the
    /// mimalloc-backed `Vec`-per-term layout on the same push +
    /// iter workload; mimalloc's small-class freelists already
    /// absorb the realloc churn.
    postings: Vec<FxHashMap<Box<str>, PostingAcc>>,
    /// Tracks the number of distinct local_doc_ids ever seen by add_doc.
    /// Used as `n_docs` for the FTS blob header.
    n_docs: u32,
    /// Per-shard bump arena reused across every `add_doc` call.
    /// Holds the transient `&str` keys of the per-doc tf hashmap.
    /// Reset at the top of each `add_doc` so the leftover bytes are
    /// invalidated before the next allocation; `Bump::reset` keeps
    /// the largest chunk so subsequent docs allocate in-place
    /// without going back to the system allocator.
    bump: bumpalo::Bump,
}

impl FtsBuilder {
    pub fn new(tokenizer: Arc<dyn Tokenizer>) -> Self {
        Self {
            tokenizer,
            columns: Vec::new(),
            postings: Vec::new(),
            n_docs: 0,
            bump: bumpalo::Bump::new(),
        }
    }

    /// Register an FTS column up-front. Returns its `column_id` (its
    /// index in declaration order).
    pub fn register_column(&mut self, name: String) -> Result<u32, BuildError> {
        if name.as_bytes().contains(&FST_SEPARATOR) {
            return Err(BuildError::ReservedSeparatorInColumnName(name));
        }
        if name.starts_with(format::RESERVED_PREFIX) {
            return Err(BuildError::ReservedPrefixInColumnName(name));
        }
        if self.columns.iter().any(|c| c.name == name) {
            return Err(BuildError::DuplicateColumnName(name));
        }
        let column_id = self.columns.len() as u32;
        self.columns.push(ColumnState {
            name,
            doc_lengths: Vec::new(),
            total_tokens: 0,
        });
        self.postings.push(FxHashMap::default());
        Ok(column_id)
    }

    /// Index `text` for `(column_id, local_doc_id)`.
    ///
    /// Caller must call this once per (doc, registered FTS column) pair,
    /// with monotonically increasing `local_doc_id` per column. Multiple
    /// occurrences of the same term in `text` increment the term-frequency
    /// for that doc.
    pub fn add_doc(
        &mut self,
        column_id: u32,
        local_doc_id: u32,
        text: &str,
    ) -> Result<(), BuildError> {
        let col_idx = column_id as usize;
        if col_idx >= self.columns.len() {
            return Err(BuildError::FtsColumnTypeInvalid {
                column: format!("(unregistered column_id {column_id})"),
                actual: "n/a".to_string(),
            });
        }

        // The contract is that `local_doc_id` increments by 1 per
        // (per-column) call, starting at 0. `finish()` indexes
        // `col.doc_lengths[doc_id]` with a doc_id from the posting list,
        // so the doc_lengths vec must be in sync with the local_doc_id
        // axis. Catch contract violations early in debug builds;
        // release skips the check.
        debug_assert!(
            local_doc_id as usize == self.columns[col_idx].doc_lengths.len(),
            "FtsBuilder::add_doc: local_doc_id ({local_doc_id}) must equal \
             this column's next index ({}); doc_ids must be consecutive \
             from 0 within a column",
            self.columns[col_idx].doc_lengths.len(),
        );

        // Reset the per-shard bump arena so leftover token bytes
        // from the prior `add_doc` call are invalidated before we
        // reuse the chunk. `Bump::reset` keeps the largest chunk
        // (no system-allocator round trip on the typical
        // steady-state doc) and frees any extra chunks the
        // pathological-long doc grew.
        self.bump.reset();

        // Split borrows: `tokenize_each` calls into `self.tokenizer`
        // and the closure captures `&self.bump` to alloc per-token
        // copies. Disjoint immutable borrows of two fields — Rust's
        // borrow split allows this when each field is named
        // explicitly rather than reached through `self.method`.
        let tokenizer = &self.tokenizer;
        let bump = &self.bump;

        let mut tf_per_term: HashMap<&str, u32> = HashMap::new();
        let mut tokens_in_doc: u64 = 0;
        tokenizer.tokenize_each(text, &mut |tok| {
            tokens_in_doc += 1;
            // alloc_str copies the borrowed token bytes into the
            // bump. The returned `&str` outlives the next callback
            // call (bump-arena lifetime), unlike the input `tok`
            // which doesn't.
            let bumped: &str = bump.alloc_str(tok);
            // SAFETY-equivalent: widen the lifetime from the bump's
            // borrow to a `'static` tag tied to the HashMap's
            // lifetime. `tf_per_term` is a local that drops at the
            // end of `add_doc` — well before `self.bump` is reset
            // on the next call — so every key in the HashMap stays
            // valid for the HashMap's full lifetime.
            let extended: &'static str = unsafe { std::mem::transmute(bumped) };
            *tf_per_term.entry(extended).or_insert(0) += 1;
        });

        // Update column doc-lengths + accounting.
        let col = &mut self.columns[col_idx];
        let dl_clamped: u32 = tokens_in_doc.min(u32::MAX as u64) as u32;
        col.doc_lengths.push(dl_clamped);
        col.total_tokens = col.total_tokens.saturating_add(tokens_in_doc);

        // Update n_docs (max local_doc_id + 1 across all columns).
        let docs_now = local_doc_id.saturating_add(1);
        if docs_now > self.n_docs {
            self.n_docs = docs_now;
        }

        // Push (doc_id, tf) into the per-column posting table. The
        // steady-state lookup is `get_mut(&str)` on a Box<str>-keyed
        // FxHashMap — hashes only the term bytes (vs the prior
        // `<col_name>\x1F<term>` byte-string layout's hash over
        // every key). On first sight we allocate one Box<str> per
        // unique term in this column; that's amortized away by
        // Zipfian repetition in any realistic corpus.
        let col_postings = &mut self.postings[col_idx];
        for (term, tf) in tf_per_term {
            match col_postings.get_mut(term) {
                Some(acc) => acc.list.push((local_doc_id, tf)),
                None => {
                    col_postings.insert(
                        Box::<str>::from(term),
                        PostingAcc {
                            list: vec![(local_doc_id, tf)],
                        },
                    );
                }
            }
        }

        Ok(())
    }

    /// Finalise and emit the FTS blob bytes. Consumes the builder.
    pub fn finish(self) -> Vec<u8> {
        let n_columns = self.columns.len() as u32;
        let n_docs = self.n_docs;
        let n_terms_total_usize: usize = self.postings.iter().map(|m| m.len()).sum();
        debug_assert!(
            n_terms_total_usize <= u32::MAX as usize,
            "term count overflows u32"
        );
        let n_terms_total = n_terms_total_usize as u32;

        // 1. Canonical FST order is lex-sorted full keys
        //    `<col_name>\x1F<term>`. Since `\x1F` < every printable
        //    ASCII byte the v1 tokenizer emits, that ordering is
        //    equivalently (col_name lex, then term lex). With
        //    per-column posting tables we already have terms
        //    grouped by column; build a column iteration order by
        //    name and sort terms within each column independently.
        let mut col_iter_order: Vec<usize> = (0..self.columns.len()).collect();
        col_iter_order.sort_by(|&a, &b| self.columns[a].name.cmp(&self.columns[b].name));

        // 2. Pre-compute per-column avgdl (in fixed-point ×1000 per spec).
        //    avgdl == 0 if the column has zero docs (pathological — guarded).
        let avgdl_per_col: Vec<f32> = self
            .columns
            .iter()
            .map(|c| {
                let n = c.doc_lengths.len() as u64;
                if n == 0 {
                    0.0
                } else {
                    (c.total_tokens as f32) / (n as f32)
                }
            })
            .collect();

        // 3. Encode postings region. For each (column, term):
        //    - Drain posting list into (doc_ids, tfs)
        //    - Split into 128-doc blocks
        //    - Encode each block; track skip-table entries
        //    - Write per-(col, term) metadata + skip table + blocks
        //    - Record metadata_offset (relative to postings region start)
        //      for the FST value.
        let mut postings_buf: Vec<u8> = Vec::new();
        let mut fst_entries: Vec<(Vec<u8>, u64)> = Vec::with_capacity(n_terms_total_usize);
        // Scratch buffer for the per-(col, term) FST key bytes —
        // reused across every term to avoid one alloc per insert.
        let mut key_buf: Vec<u8> = Vec::with_capacity(64);

        // Move postings out so we can drain per column without
        // holding `self.postings` borrowed across the per-column
        // loop body.
        let mut postings_by_col = self.postings;

        for col_idx in col_iter_order {
            let col_name_bytes = self.columns[col_idx].name.as_bytes();
            let avgdl = avgdl_per_col[col_idx];

            // Drain this column's terms and sort lex by term bytes.
            let mut term_entries: Vec<(Box<str>, PostingAcc)> =
                std::mem::take(&mut postings_by_col[col_idx])
                    .into_iter()
                    .collect();
            term_entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

            for (term, acc) in &term_entries {
                key_buf.clear();
                key_buf.extend_from_slice(col_name_bytes);
                key_buf.push(FST_SEPARATOR);
                key_buf.extend_from_slice(term.as_bytes());

                // The list is doc-id-sorted by the API contract
                // (caller passes monotonic local_doc_ids, we push in
                // order).
                let pairs: &[(u32, u32)] = &acc.list;
                debug_assert!(
                    pairs.windows(2).all(|w| w[0].0 < w[1].0),
                    "posting list not sorted by doc_id"
                );
                let df = pairs.len() as u64;

                // df=1 inline short-circuit: pack (doc_id, tf)
                // directly into the FST value's bits and write
                // nothing to the postings region. No metadata
                // header, no skip table, no PFOR block — the reader
                // recovers everything from the FST value alone.
                // Saves ~50–60 B of postings + the PFOR encode call
                // per df=1 term.
                if df == 1 {
                    let (doc_id, tf) = pairs[0];
                    fst_entries.push((key_buf.clone(), FstValue::pack_inline(doc_id, tf)));
                    continue;
                }

                let idf_t = crate::superfile::fts::bm25::idf(n_docs as u64, df);

                // Split into BLOCK_LEN-sized chunks; encode each.
                let mut encoded_blocks: Vec<crate::superfile::fts::posting::EncodedBlock> =
                    Vec::new();
                let mut min_dl_per_block: Vec<u32> = Vec::new();
                for chunk in pairs.chunks(BLOCK_LEN) {
                    let doc_ids: Vec<u32> = chunk.iter().map(|(d, _)| *d).collect();
                    let tfs: Vec<u32> = chunk.iter().map(|(_, t)| *t).collect();
                    // min_dl across this block — determines BMW upper bound.
                    let col_doc_lengths = &self.columns[col_idx].doc_lengths;
                    let min_dl = doc_ids
                        .iter()
                        .map(|d| col_doc_lengths[*d as usize])
                        .min()
                        .unwrap_or(0);
                    min_dl_per_block.push(min_dl);
                    encoded_blocks.push(encode_block(&Block { doc_ids, tfs }));
                }
                let num_blocks = encoded_blocks.len() as u32;

                // Per-(col, term) metadata starts here.
                let metadata_offset = postings_buf.len() as u64;

                // Skip-table size = num_blocks × SKIP_ENTRY_SIZE.
                let skip_table_size = encoded_blocks.len() * SKIP_ENTRY_SIZE;
                // Total per-term posting bytes = metadata + skip table + blocks.
                let blocks_total_size: usize = encoded_blocks.iter().map(|b| b.bytes.len()).sum();
                let postings_length = (TERM_META_SIZE + skip_table_size + blocks_total_size) as u64;

                // Metadata header (20 bytes). df, postings_length,
                // and num_blocks fit u32 even at the 16 GB segment
                // target (df ≤ ~16M; postings_length ≤ ~1 MB;
                // num_blocks ≤ ~125K). Only postings_offset needs
                // u64 — it's the absolute offset into the postings
                // region which can exceed 4 GiB.
                debug_assert!(df <= u32::MAX as u64, "df overflows u32");
                debug_assert!(
                    postings_length <= u32::MAX as u64,
                    "single-term posting > 4 GiB"
                );
                postings_buf.extend_from_slice(&(df as u32).to_le_bytes()); // df: 4
                // postings_offset is the absolute offset where the
                // metadata starts — i.e., metadata_offset itself.
                // Structurally redundant (the reader already knows
                // this from the FST value) but kept for
                // self-description / future format checks.
                postings_buf.extend_from_slice(&metadata_offset.to_le_bytes()); // 8
                postings_buf.extend_from_slice(&(postings_length as u32).to_le_bytes()); // 4
                postings_buf.extend_from_slice(&num_blocks.to_le_bytes()); // 4
                debug_assert_eq!(
                    postings_buf.len() - metadata_offset as usize,
                    TERM_META_SIZE
                );

                // Skip table — block_offset is relative to
                // (col, term)'s posting region (i.e., relative to
                // the start of metadata).
                let mut block_offset: u32 = (TERM_META_SIZE + skip_table_size) as u32;
                for (i, blk) in encoded_blocks.iter().enumerate() {
                    let last_doc_id = blk.last_doc_id;
                    // BMW per-block max BM25 contribution. Uses
                    // this column's avgdl + this block's min_dl +
                    // this block's max_tf.
                    let max_bm25 = crate::superfile::fts::bm25::block_upper_bound(
                        idf_t,
                        blk.max_tf,
                        min_dl_per_block[i],
                        avgdl,
                    );
                    let max_bm25_x1000 = (max_bm25 * 1000.0).max(0.0).min(u32::MAX as f32) as u32;
                    postings_buf.extend_from_slice(&last_doc_id.to_le_bytes()); // 4
                    postings_buf.extend_from_slice(&block_offset.to_le_bytes()); // 4
                    postings_buf.extend_from_slice(&max_bm25_x1000.to_le_bytes()); // 4
                    postings_buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
                    block_offset += blk.bytes.len() as u32;
                }

                // Posting blocks.
                for blk in encoded_blocks {
                    postings_buf.extend_from_slice(&blk.bytes);
                }

                // FST value: PFOR-form — `metadata_offset << 1`
                // (low bit clear). The reader right-shifts to
                // recover the offset; the low bit distinguishes
                // this from the df=1 inline form above.
                fst_entries.push((key_buf.clone(), FstValue::pack_pfor(metadata_offset)));
            }
        }

        let postings_crc = crc32c(&postings_buf);
        postings_buf.extend_from_slice(&postings_crc.to_le_bytes());

        // 4. Build FST.
        let mut dict_builder = DictBuilder::new();
        for (k, v) in fst_entries {
            dict_builder.insert(&k, v);
        }
        let mut fst_bytes = dict_builder.finish();
        let fst_crc = crc32c(&fst_bytes);
        fst_bytes.extend_from_slice(&fst_crc.to_le_bytes());

        // 5. Doc-lengths directory. n_columns × DOC_LENGTHS_ENTRY_SIZE bytes.
        //    Each entry: column_id (u32) + doc_lengths_offset (u64) + avgdl_x1000 (u32).
        //    We compute doc_lengths_offsets only after we know the directory's
        //    own size — they're absolute offsets in the final blob.

        // Precompute layout offsets:
        //   header_size = 48
        //   fst_offset = 48
        //   postings_offset = fst_offset + fst_bytes.len()
        //   doc_lengths_table_offset = postings_offset + postings_buf.len()
        //   doc_lengths_arrays_start = doc_lengths_table_offset + (n_columns * 16) + 4 /* dir crc */
        let header_size: u64 = 48;
        let fst_offset: u64 = header_size;
        let postings_offset: u64 = fst_offset + fst_bytes.len() as u64;
        let doc_lengths_table_offset: u64 = postings_offset + postings_buf.len() as u64;
        let mut doc_lengths_array_offset: u64 =
            doc_lengths_table_offset + (n_columns as u64) * (DOC_LENGTHS_ENTRY_SIZE as u64) + 4 /* dir CRC */;

        let mut dir_buf: Vec<u8> = Vec::with_capacity(n_columns as usize * DOC_LENGTHS_ENTRY_SIZE);
        let mut arrays_buf: Vec<u8> = Vec::new();
        for (i, col) in self.columns.iter().enumerate() {
            let avgdl_x1000 = (avgdl_per_col[i] * 1000.0).max(0.0).min(u32::MAX as f32) as u32;
            dir_buf.extend_from_slice(&(i as u32).to_le_bytes()); // column_id: 4
            dir_buf.extend_from_slice(&doc_lengths_array_offset.to_le_bytes()); // doc_lengths_offset: 8
            dir_buf.extend_from_slice(&avgdl_x1000.to_le_bytes()); // avgdl_x1000: 4

            // Serialize this column's doc_lengths (u32 LE), append CRC32C.
            let array_start = arrays_buf.len();
            for &dl in &col.doc_lengths {
                arrays_buf.extend_from_slice(&dl.to_le_bytes());
            }
            let array_bytes = &arrays_buf[array_start..];
            let array_crc = crc32c(array_bytes);
            arrays_buf.extend_from_slice(&array_crc.to_le_bytes());
            // Advance for next column. Length = 4 * n_docs + 4 (CRC).
            doc_lengths_array_offset += (col.doc_lengths.len() as u64) * 4 + 4;
        }
        let dir_crc = crc32c(&dir_buf);
        dir_buf.extend_from_slice(&dir_crc.to_le_bytes());

        // 6. Concatenate everything into the final blob.
        let mut blob: Vec<u8> = Vec::with_capacity(
            header_size as usize
                + fst_bytes.len()
                + postings_buf.len()
                + dir_buf.len()
                + arrays_buf.len(),
        );

        // Header (48 bytes).
        blob.extend_from_slice(format::fts::MAGIC); // 8
        blob.extend_from_slice(&format::fts::VERSION.to_le_bytes()); // 4
        blob.extend_from_slice(&n_columns.to_le_bytes()); // 4
        blob.extend_from_slice(&n_docs.to_le_bytes()); // 4 (u32)
        blob.extend_from_slice(&n_terms_total.to_le_bytes()); // 4 (u32)
        blob.extend_from_slice(&fst_offset.to_le_bytes()); // 8
        blob.extend_from_slice(&postings_offset.to_le_bytes()); // 8
        blob.extend_from_slice(&doc_lengths_table_offset.to_le_bytes()); // 8
        debug_assert_eq!(blob.len(), header_size as usize, "header size mismatch");

        blob.extend_from_slice(&fst_bytes);
        blob.extend_from_slice(&postings_buf);
        blob.extend_from_slice(&dir_buf);
        blob.extend_from_slice(&arrays_buf);

        blob
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::default_tokenizer as tokenizer;

    #[test]
    fn register_column_returns_sequential_ids() {
        let mut b = FtsBuilder::new(tokenizer());
        assert_eq!(
            b.register_column("title".into()).expect("register column"),
            0
        );
        assert_eq!(
            b.register_column("body".into()).expect("register column"),
            1
        );
        assert_eq!(b.register_column("tag".into()).expect("register column"), 2);
    }

    #[test]
    fn register_column_rejects_separator_byte() {
        let mut b = FtsBuilder::new(tokenizer());
        let bad = String::from("ti\x1Ftle");
        let err = b.register_column(bad).expect_err("expected error");
        assert!(matches!(err, BuildError::ReservedSeparatorInColumnName(_)));
    }

    #[test]
    fn register_column_rejects_reserved_prefix() {
        let mut b = FtsBuilder::new(tokenizer());
        let err = b
            .register_column("inf.title".into())
            .expect_err("expected error");
        assert!(matches!(err, BuildError::ReservedPrefixInColumnName(_)));
    }

    #[test]
    fn register_column_rejects_duplicates() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register column");
        let err = b
            .register_column("title".into())
            .expect_err("expected error");
        assert!(matches!(err, BuildError::DuplicateColumnName(_)));
    }

    #[test]
    fn add_doc_unknown_column_id_errors() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register column");
        let err = b.add_doc(99, 0, "text").expect_err("expected error");
        assert!(matches!(err, BuildError::FtsColumnTypeInvalid { .. }));
    }

    #[test]
    fn add_doc_accumulates_tf_within_doc() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register column");
        b.add_doc(0, 0, "rust rust rust async").expect("add doc");

        // Posting tables are per-column; look up by term &str. The
        // old `make_key("title", "rust")` byte-string layout is
        // gone, replaced by `postings[col_idx].get(&str)`.
        let col_postings = &b.postings[0];
        let acc = col_postings.get("rust").expect("posting exists");
        assert_eq!(acc.list, vec![(0, 3)]);

        // (title, "async") tf=1.
        let acc2 = col_postings.get("async").expect("posting exists");
        assert_eq!(acc2.list, vec![(0, 1)]);
    }

    #[test]
    fn cross_column_same_term_stays_isolated_through_round_trip() {
        // A term that appears in two different columns must keep
        // its posting lists scoped per column — both in the
        // in-memory `FtsBuilder.postings` accumulator (separate
        // hashmaps per column_id) AND in the emitted FST + posting
        // region. Regression check for the
        // `Vec<u8>(<col>\x1F<term>)` → per-column `Box<str>` key
        // change: confirms the rename of in-memory key shape didn't
        // introduce a cross-column collision the FST round-trip
        // would otherwise hide.
        use crate::superfile::fts::reader::{BoolMode, FtsReader};
        use bytes::Bytes;

        let mut b = FtsBuilder::new(tokenizer());
        let title_id = b.register_column("title".into()).expect("register title");
        let body_id = b.register_column("body".into()).expect("register body");

        // Doc 0: "rust" + "tokio" in title, "rust" + "async" in body.
        // Doc 1: only in body — "rust".
        // Doc 2: only in title — "rust".
        b.add_doc(title_id, 0, "rust tokio")
            .expect("add title doc 0");
        b.add_doc(body_id, 0, "rust async").expect("add body doc 0");
        b.add_doc(body_id, 1, "rust").expect("add body doc 1");
        b.add_doc(title_id, 1, "rust").expect("add title doc 1");

        // (A) In-memory check: the per-column hashmaps must NOT
        //     observe each other's "rust" entries.
        let title_postings = &b.postings[title_id as usize];
        let body_postings = &b.postings[body_id as usize];
        assert_eq!(
            title_postings.get("rust").map(|a| &a.list),
            Some(&vec![(0u32, 1u32), (1u32, 1u32)]),
            "title's 'rust' should hold doc_ids 0 and 1 only"
        );
        assert_eq!(
            body_postings.get("rust").map(|a| &a.list),
            Some(&vec![(0u32, 1u32), (1u32, 1u32)]),
            "body's 'rust' should hold doc_ids 0 and 1 only (its own axis)"
        );
        // "tokio" lives only in title; "async" lives only in body.
        assert!(title_postings.contains_key("tokio"));
        assert!(!body_postings.contains_key("tokio"));
        assert!(body_postings.contains_key("async"));
        assert!(!title_postings.contains_key("async"));

        // (B) Round-trip through finish() + FtsReader::search. The
        //     reader looks up via `dict::make_key(column, term)`,
        //     so this is the strict on-disk equivalent of "two
        //     columns share a term — does each see its own
        //     postings?"
        let blob = Bytes::from(b.finish());
        let json = r#"[{"name":"title","tokenizer":"ascii_lower"},{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // "rust" in title returns title's docs (0, 1) and no others.
        let hits_t = r
            .search("title", &["rust"], 10, BoolMode::Or)
            .expect("title search");
        let ids_t: Vec<u32> = hits_t.iter().map(|(d, _)| *d).collect();
        assert_eq!(ids_t.len(), 2, "title 'rust' hit count");
        assert!(ids_t.contains(&0));
        assert!(ids_t.contains(&1));

        // "rust" in body also returns its own docs (0, 1). Same ids
        // by coincidence; what matters is the search is scoped to
        // body's posting list, not title's.
        let hits_b = r
            .search("body", &["rust"], 10, BoolMode::Or)
            .expect("body search");
        let ids_b: Vec<u32> = hits_b.iter().map(|(d, _)| *d).collect();
        assert_eq!(ids_b.len(), 2, "body 'rust' hit count");
        assert!(ids_b.contains(&0));
        assert!(ids_b.contains(&1));

        // Cross-leak negative: a term that lives only in body
        // (`async`) must NOT be findable in title, and vice versa
        // (`tokio` in body).
        let hits_async_in_title = r
            .search("title", &["async"], 10, BoolMode::Or)
            .expect("title async search");
        assert!(
            hits_async_in_title.is_empty(),
            "title must not return 'async' (lives only in body)"
        );
        let hits_tokio_in_body = r
            .search("body", &["tokio"], 10, BoolMode::Or)
            .expect("body tokio search");
        assert!(
            hits_tokio_in_body.is_empty(),
            "body must not return 'tokio' (lives only in title)"
        );
    }

    #[test]
    fn add_doc_tracks_doc_lengths_clamped() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("body".into()).expect("register column");
        b.add_doc(0, 0, "alpha beta gamma").expect("add doc");
        b.add_doc(0, 1, "").expect("add doc"); // zero-token doc
        b.add_doc(0, 2, "delta").expect("add doc");
        let col = &b.columns[0];
        assert_eq!(col.doc_lengths, vec![3, 0, 1]);
        assert_eq!(col.total_tokens, 4);
    }

    #[test]
    fn add_doc_updates_n_docs_per_call() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("body".into()).expect("register column");
        // Contract: local_doc_id is consecutive from 0 (per column).
        // n_docs ends up == max(local_doc_id) + 1 == call count.
        b.add_doc(0, 0, "a").expect("add doc");
        b.add_doc(0, 1, "b").expect("add doc");
        b.add_doc(0, 2, "c").expect("add doc");
        assert_eq!(b.n_docs, 3);
    }

    #[test]
    fn finish_emits_valid_header() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register column");
        b.add_doc(0, 0, "hello world").expect("add doc");
        let blob = b.finish();

        // Magic.
        assert_eq!(&blob[0..8], format::fts::MAGIC);
        // Version.
        let version = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
        assert_eq!(version, format::fts::VERSION);
        // n_columns.
        let n_cols = u32::from_le_bytes([blob[12], blob[13], blob[14], blob[15]]);
        assert_eq!(n_cols, 1);
        // n_docs (u32 at 16..20).
        let n_docs = u32::from_le_bytes([blob[16], blob[17], blob[18], blob[19]]);
        assert_eq!(n_docs, 1);
        // n_terms_total = 2 ("hello", "world") (u32 at 20..24).
        let n_terms = u32::from_le_bytes([blob[20], blob[21], blob[22], blob[23]]);
        assert_eq!(n_terms, 2);
        // fst_offset == 48 (u64 at 24..32).
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&blob[24..32]);
        let fst_off = u64::from_le_bytes(buf);
        assert_eq!(fst_off, 48);
    }

    #[test]
    fn finish_with_no_docs_still_produces_valid_blob() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("title".into()).expect("register column");
        let blob = b.finish();
        assert_eq!(&blob[0..8], format::fts::MAGIC);
        // n_docs == 0 (u32 at 16..20), n_terms_total == 0 (u32 at 20..24).
        assert_eq!(
            u32::from_le_bytes([blob[16], blob[17], blob[18], blob[19]]),
            0
        );
        assert_eq!(
            u32::from_le_bytes([blob[20], blob[21], blob[22], blob[23]]),
            0
        );
    }

    #[test]
    fn finish_offsets_are_consistent() {
        let mut b = FtsBuilder::new(tokenizer());
        b.register_column("body".into()).expect("register column");
        for i in 0..10 {
            b.add_doc(0, i, &format!("term{i} common"))
                .expect("add doc");
        }
        let blob = b.finish();

        // Header layout post-u32-narrowing: fst_offset at 24..32,
        // postings_offset at 32..40, doc_lengths_table_offset at 40..48.
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&blob[24..32]);
        let fst_off = u64::from_le_bytes(buf) as usize;
        buf.copy_from_slice(&blob[32..40]);
        let postings_off = u64::from_le_bytes(buf) as usize;
        buf.copy_from_slice(&blob[40..48]);
        let dir_off = u64::from_le_bytes(buf) as usize;

        assert_eq!(fst_off, 48);
        assert!(postings_off > fst_off, "postings after FST");
        assert!(dir_off > postings_off, "directory after postings");
        assert!(dir_off <= blob.len(), "directory offset within blob");
    }
}
