// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS index metadata + open configuration: the per-doc length-norm
//! table ([`NormTable`]), per-column metadata ([`ColumnMeta`]) and its
//! JSON config ([`FtsColumnConfig`]), and the reader [`OpenOptions`].

use std::{
    fmt,
    ops::Range,
    sync::{Arc, OnceLock},
};

use serde::Deserialize;

use crate::superfile::{
    ReadError,
    error::FtsError,
    format::checksum::crc32c,
    fts::{
        analysis::{Base, Stemmer, Stopwords},
        bm25,
        reader::core::read_doc_length,
        tokenize::Tokenizer,
    },
    lazy_source::Source,
};

/// Per-doc BM25 length normalizer, quantized to one byte per doc.
///
/// The scorer needs `dl_norm_k1[doc] = K1·(1 - B + B·dl/avgdl)` for
/// every scored doc. Held as an `f32` per doc, that table is 4 bytes ×
/// n_docs — at multi-million-doc scale too large to stay cache-resident,
/// so each scored doc pays a scattered load from a table that overflows
/// cache. Instead the doc length is quantized to one byte
/// ([`bm25::quantize_len`]) and a 256-entry table decodes each bucket to
/// its norm value: the per-doc table is 4× smaller (one byte), and the
/// decode table is 1 KiB (L1-resident). A scored doc reads
/// `lut[bytes[doc]]` — one load from the small per-doc table plus one L1
/// lookup — instead of one load from a 4×-larger table.
#[derive(Debug, Clone)]
pub struct NormTable {
    /// Per-doc quantized length bucket. Empty for a column with no docs.
    /// Parameter-free — the buckets are lengths, not norms — and shared
    /// rather than copied so [`NormTable::rescored`] costs one `Arc`
    /// bump plus a 1 KiB table instead of a pass over every doc.
    bytes: Arc<[u8]>,
    /// Bucket → `k1·(1 - b + b·dequantize_len(bucket)/avgdl)`. A fixed
    /// 256-entry table, boxed so `ColumnMeta` stays pointer-sized (it is
    /// scanned by non-scoring paths — column lookup, listing) while the
    /// `u8` bucket index into a fixed-length array lets the compiler drop
    /// the bounds check in `get`. This is the only parameter-dependent
    /// part of the table.
    lut: Arc<[f32; 256]>,
    /// Lowest and highest bucket any doc in this column actually
    /// occupies, tracked at build so [`NormTable::bound_scale`] takes
    /// its supremum over lengths that occur rather than over all 256
    /// representable ones. `(0, 0)` for an empty column.
    occupied: (u8, u8),
    /// The average document length `lut` was built at. Kept because it
    /// is the other half of what determines the table — two tables can
    /// share a parameter pair and still decode differently — so
    /// [`NormTable::bound_scale`] can tell "nothing changed" from "the
    /// average moved" without walking 256 buckets to find out.
    avgdl: f32,
}

/// A column's document-length totals, summed over the stored
/// doc-lengths array when the reader opens.
///
/// Both numbers are counted over the documents BM25's corpus
/// statistics are actually defined over — the ones carrying at least
/// one indexed token. A row whose cell was null occupies a slot in the
/// doc-lengths array (the array is indexed by local doc id, so it
/// cannot skip rows) with a length of zero, and a row whose text
/// produced no tokens is indistinguishable from it; neither can match
/// a term, and counting either would deflate the average and inflate
/// the collection size for every term in the column.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ColumnLengthStats {
    /// Exact sum of every document's token count in this column.
    pub total_tokens: u64,
    /// Documents carrying at least one indexed token in this column.
    pub n_scored_docs: u64,
}

impl ColumnLengthStats {
    /// Average length over the documents that carry tokens. `0.0` for a
    /// column no document contributes to, which is the value
    /// [`NormTable::new`] reads as "never scored".
    pub fn avgdl(&self) -> f32 {
        if self.n_scored_docs == 0 {
            return 0.0;
        }
        (self.total_tokens as f64 / self.n_scored_docs as f64) as f32
    }

    /// Fold another column's totals in — the sum across superfiles that
    /// turns per-file totals into table-wide ones.
    pub fn merge_with(&mut self, other: &ColumnLengthStats) {
        self.total_tokens = self.total_tokens.saturating_add(other.total_tokens);
        self.n_scored_docs = self.n_scored_docs.saturating_add(other.n_scored_docs);
    }

    /// Count one document of `dl` tokens. A zero-length document — a null
    /// or empty cell indexed to keep the per-doc arrays row-aligned — is
    /// not a document this column has and contributes to neither total.
    #[inline]
    pub fn add(&mut self, dl: u32) {
        if dl > 0 {
            self.total_tokens += u64::from(dl);
            self.n_scored_docs += 1;
        }
    }

    /// The statistics of a column whose per-doc lengths are `lengths`.
    pub fn from_lengths(lengths: impl IntoIterator<Item = u32>) -> Self {
        let mut stats = Self::default();
        for dl in lengths {
            stats.add(dl);
        }
        stats
    }

    /// Fold two contributions where either may be unknown. Statistics
    /// exist to be summed — the whole point of rolling them up is that
    /// the table's average and collection size come out as they would
    /// for one unfragmented file — and a contributor that predates the
    /// totals has nothing to add: folding it in as zero would silently
    /// shrink both. So an unknown side makes the result unknown.
    pub fn fold(acc: Option<Self>, next: Option<Self>) -> Option<Self> {
        let (mut acc, next) = (acc?, next?);
        acc.merge_with(&next);
        Some(acc)
    }
}

impl NormTable {
    /// Build from a column's per-doc lengths, returning the table
    /// alongside the totals the pass produced.
    ///
    /// The same pass that quantizes each length sums them, so the
    /// column's statistics cost nothing extra; `decode_avgdl` then picks
    /// the average the table decodes at, given those statistics — the
    /// average the file declares, or one corrected from them. A column
    /// no document contributes to yields an empty table; it is never
    /// indexed because `search` short-circuits on empty columns.
    pub(super) fn new(
        doc_lengths: impl Iterator<Item = u32>,
        n_docs: usize,
        params: bm25::Bm25Params,
        decode_avgdl: impl FnOnce(&ColumnLengthStats) -> f32,
    ) -> (Self, ColumnLengthStats) {
        let mut bytes = Vec::with_capacity(n_docs);
        let mut lo = u8::MAX;
        let mut hi = u8::MIN;
        let mut stats = ColumnLengthStats::default();
        for dl in doc_lengths {
            let bucket = bm25::quantize_len(dl);
            lo = lo.min(bucket);
            hi = hi.max(bucket);
            bytes.push(bucket);
            stats.add(dl);
        }
        if stats.n_scored_docs == 0 {
            return (Self::empty(), stats);
        }
        let avgdl = decode_avgdl(&stats);
        let table = Self {
            bytes: Arc::from(bytes),
            lut: build_lut(avgdl, params),
            occupied: (lo, hi),
            avgdl,
        };
        (table, stats)
    }

    /// The same per-doc buckets decoded at a different average length
    /// and/or parameter pair — for a query that overrides what the
    /// column declared, and for an older file whose declared average the
    /// reader corrects. Shares `bytes`, so the cost is one 256-entry
    /// table.
    pub(super) fn rescored(&self, avgdl: f32, params: bm25::Bm25Params) -> Self {
        if self.bytes.is_empty() {
            return Self::empty();
        }
        Self {
            bytes: Arc::clone(&self.bytes),
            lut: build_lut(avgdl, params),
            occupied: self.occupied,
            avgdl,
        }
    }

    /// The average document length this table decodes at.
    pub(super) fn avgdl(&self) -> f32 {
        self.avgdl
    }

    /// The factor `R >= 1` by which every bound built at `baked` must be
    /// inflated to stay an upper bound under `query`'s parameters, where
    /// `self` is the norm table at `baked` and `other` the one at
    /// `query`.
    ///
    /// The stored per-block bound is the block's true max of
    /// `idf·tf / (tf + dl_norm_k1)`. Between two parameter sets the
    /// per-doc ratio is
    ///
    /// ```text
    ///   (tf + A) / (tf + B),
    ///       A = lut_baked[bucket],  B = lut_query[bucket]
    /// ```
    ///
    /// with `idf` cancelling — it carries no parameters. For a fixed
    /// bucket that is monotone in `tf` and tends to 1, so its supremum
    /// over `tf >= 1` is `max(1, (1+A)/(1+B))`; taking the max over the
    /// buckets docs actually occupy gives the supremum over the column.
    /// Loosening, never under-bounding, and exactly `1.0` when the two
    /// tables decode identically.
    ///
    /// The short-circuit tests the average as well as the parameters,
    /// because either one moves the `lut` on its own: two tables at the
    /// same `k1`/`b` but different averages produce different norms for
    /// every bucket, and returning `1.0` for that pair would leave the
    /// stored bounds below the scores they are supposed to cap — which
    /// prunes documents out of the top-k with nothing to show for it.
    pub(super) fn bound_scale(
        &self,
        other: &NormTable,
        baked: bm25::Bm25Params,
        query: bm25::Bm25Params,
    ) -> f32 {
        if (baked == query && self.avgdl == other.avgdl) || self.bytes.is_empty() {
            return 1.0;
        }
        let (lo, hi) = self.occupied;
        let mut worst = 1.0_f32;
        for bucket in lo..=hi {
            let a = self.lut[bucket as usize];
            let b = other.lut[bucket as usize];
            worst = worst.max((1.0 + a) / (1.0 + b));
        }
        worst
    }

    /// `dl_norm_k1` for a doc (length quantized): one per-doc byte load
    /// plus one L1 decode-table lookup. Hot path — keep it inlined.
    #[inline(always)]
    pub(super) fn get(&self, doc: u32) -> f32 {
        self.lut[self.bytes[doc as usize] as usize]
    }

    /// Number of docs in the table. Test-only: the query path indexes
    /// by doc id and never needs the count.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.bytes.len()
    }

    /// An empty table: `bytes` is empty, so `get` must never be called on
    /// it. For call sites that need a `&NormTable` but provably never index
    /// it — an unranked (`bar == NEG_INFINITY`) phrase seek, which does no
    /// scoring. The `lut` is a zeroed 256-entry table, allocated but never
    /// read.
    pub(super) fn empty() -> Self {
        Self {
            bytes: Arc::from(Vec::new()),
            lut: Arc::new([0.0; 256]),
            occupied: (0, 0),
            avgdl: 0.0,
        }
    }
}

/// Decode table for one parameter set: bucket → `dl_norm_k1`. Filled in
/// place on the heap so the 256 `f32`s are not built on the stack and
/// moved.
fn build_lut(avgdl: f32, params: bm25::Bm25Params) -> Arc<[f32; 256]> {
    let mut lut = Box::new([0.0_f32; 256]);
    for (bucket, slot) in lut.iter_mut().enumerate() {
        let dl = bm25::dequantize_len(bucket as u8);
        *slot = params.dl_norm_k1(dl, avgdl);
    }
    Arc::from(lut)
}

/// Per-column metadata, indexed by column_id (declaration order).
/// What scoring needs from a column's per-document length array: the
/// norm table, the length statistics it was computed from, and the factor
/// that keeps the stored bounds upper bounds. Built on first scored use —
/// see [`ColumnMeta::norms`] — so a query that only matches, or that a
/// table-level term index resolves without this superfile's dictionary,
/// never reads the array.
#[derive(Debug, Clone)]
pub struct ColumnNorms {
    pub dl_norm_k1: NormTable,
    pub length_stats: ColumnLengthStats,
    /// `1.0` for a file whose bounds were baked at the average it declares;
    /// the older-file correction otherwise; composed with the override
    /// factor when the column is scored at other parameters.
    pub bound_scale: f32,
}

impl ColumnNorms {
    /// Build from a column's length array, at `params` (the pair the stored
    /// bounds were baked at) and the average the file declares.
    pub(super) fn from_array(
        array: &[u8],
        n_docs: usize,
        doc_length_bytes: usize,
        params: bm25::Bm25Params,
        baked_avgdl: f32,
        declared: bool,
    ) -> Self {
        let (dl_norm_k1, length_stats) = NormTable::new(
            (0..n_docs).map(|d| read_doc_length(array, d, doc_length_bytes)),
            n_docs,
            params,
            |stats| match declared {
                true => baked_avgdl,
                false => bm25::stored_avgdl(stats.avgdl()),
            },
        );
        // A current-version file is scored at the average it declares, so
        // its bounds are exact as stored. An older file is scored at the
        // average over the documents that carry tokens, computed from the
        // array being walked, and its bounds owe two corrections: that
        // average can only be higher than the row-count one it was baked
        // at (no more documents carry tokens than there are rows), which
        // lowers the norm and raises every score above the bound meant to
        // cap it, so the bound is inflated by the supremum of that move;
        // and the `(k1 + 1)` factor those files carry is divided out, which
        // restores exactly the pruning they had.
        let bound_scale = match declared {
            true => 1.0,
            false => {
                let baked = dl_norm_k1.rescored(baked_avgdl, params);
                baked.bound_scale(&dl_norm_k1, params, params) / (params.k1 + 1.0)
            }
        };
        Self {
            dl_norm_k1,
            length_stats,
            bound_scale,
        }
    }

    /// The norms for a column that has none to read (a failed or empty
    /// array): every score computes against an empty table.
    fn empty() -> Self {
        Self {
            dl_norm_k1: NormTable::empty(),
            length_stats: ColumnLengthStats::default(),
            bound_scale: 1.0,
        }
    }

    /// These norms re-derived at `params`, for a view that scores with
    /// parameters other than `declared` — the ones the stored bounds were
    /// baked at. The per-doc length buckets are shared, not copied; the
    /// bound factor composes rather than replaces, since an older file's
    /// bounds already owe the correction applied above and this move is
    /// owed on top of it. The product of the two suprema cannot under-bound.
    fn rescored(&self, declared: bm25::Bm25Params, params: bm25::Bm25Params) -> Self {
        let dl_norm_k1 = self.dl_norm_k1.rescored(self.dl_norm_k1.avgdl(), params);
        Self {
            bound_scale: self.bound_scale
                * self.dl_norm_k1.bound_scale(&dl_norm_k1, declared, params),
            dl_norm_k1,
            length_stats: self.length_stats,
        }
    }
}

/// One FTS column as the reader sees it: its configuration, where its
/// length array sits, and the norms scoring needs — built lazily, see
/// [`Self::norms`].
#[derive(Clone)]
pub struct ColumnMeta {
    pub name: String,
    pub doc_lengths_range: Range<usize>,
    /// The parameters this column is scored at: the declared pair, or an
    /// override's (see `FtsReader::with_bm25_override`).
    pub params: bm25::Bm25Params,
    pub positions: bool,
    pub tokenizer: Arc<dyn Tokenizer>,
    pub(crate) base: Base,
    pub stopwords: Stopwords,
    pub stemmer: Stemmer,
    pub stored: bool,
    /// Where the length array is read from when the norms are first needed.
    pub(super) source: Source,
    pub(super) n_docs: u32,
    pub(super) doc_length_bytes: usize,
    pub(super) baked_avgdl: f32,
    /// Whether the file declares the average its bounds were baked at.
    pub(super) declared: bool,
    /// The pair the stored bounds were baked at.
    pub(super) declared_params: bm25::Bm25Params,
    /// Whether the length array's CRC is checked when it is read.
    pub(super) verify_crc: bool,
    /// Norms at the declared parameters, shared by every view of the reader.
    pub(super) base_norms: Arc<OnceLock<ColumnNorms>>,
    /// Norms at `params` when they differ from the declared pair, derived
    /// from the base on first use; each override view has its own.
    pub(super) view_norms: Arc<OnceLock<ColumnNorms>>,
}

impl fmt::Debug for ColumnMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColumnMeta")
            .field("name", &self.name)
            .field("doc_lengths_range", &self.doc_lengths_range)
            .field("params", &self.params)
            .field("positions", &self.positions)
            .field("stored", &self.stored)
            .field("norms_loaded", &self.base_norms.get().is_some())
            .finish_non_exhaustive()
    }
}

impl ColumnMeta {
    /// The norms at the declared parameters, reading the length array now
    /// if no scoring entry point has done so yet. The entry points prewarm
    /// through `FtsReader::ensure_norms` on the async path; this synchronous
    /// fallback exists so a kernel can never find the norms absent — on an
    /// in-memory source it costs nothing, on a lazy one it is the read the
    /// prewarm would have made. A read that fails here logs and scores
    /// against an empty table rather than aborting mid-kernel.
    fn base_norms(&self) -> &ColumnNorms {
        self.base_norms.get_or_init(|| {
            let n = self.n_docs as usize;
            let array_len = n * self.doc_length_bytes;
            let start = self.doc_lengths_range.start;
            let fetched = self
                .source
                .get_range(start..start + array_len + 4)
                .map_err(|e| e.to_string())
                .and_then(|array| {
                    self.check_array_crc(&array)
                        .map(|()| array)
                        .map_err(|e| e.to_string())
                });
            match fetched {
                Ok(array) => self.norms_from_array(&array[..array_len]),
                Err(error) => {
                    tracing::error!(column = %self.name, %error, "doc-length array unreadable; scoring against empty norms");
                    ColumnNorms::empty()
                }
            }
        })
    }

    /// Check the CRC that trails the length array in `array_with_crc`, when
    /// verification is on. The array is `n_docs × doc_length_bytes` long
    /// and its CRC32C follows it.
    pub(super) fn check_array_crc(&self, array_with_crc: &[u8]) -> Result<(), FtsError> {
        if !self.verify_crc {
            return Ok(());
        }
        let len = self.n_docs as usize * self.doc_length_bytes;
        let Some(crc_bytes) = array_with_crc.get(len..len + 4) else {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "doc-lengths array shorter than its CRC".into(),
            )));
        };
        let expected = u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
        if expected != crc32c(&array_with_crc[..len]) {
            return Err(FtsError::Read(ReadError::ChecksumMismatch {
                section: "fts/doc_lengths_array",
                column: format!(" (column '{}')", self.name),
            }));
        }
        Ok(())
    }

    /// Build the declared-parameter norms from the column's length array.
    pub(super) fn norms_from_array(&self, array: &[u8]) -> ColumnNorms {
        ColumnNorms::from_array(
            array,
            self.n_docs as usize,
            self.doc_length_bytes,
            self.declared_params,
            self.baked_avgdl,
            self.declared,
        )
    }

    /// The norms this column scores with: the declared ones, or the
    /// override view's, derived from them on first use.
    pub(super) fn norms(&self) -> &ColumnNorms {
        if self.params == self.declared_params {
            return self.base_norms();
        }
        self.view_norms.get_or_init(|| {
            self.base_norms()
                .rescored(self.declared_params, self.params)
        })
    }

    /// Whether the declared-parameter norms have been built.
    pub(super) fn norms_loaded(&self) -> bool {
        self.base_norms.get().is_some()
    }

    /// This column with its bound factor replaced — for tests that probe the
    /// decoder's scaling. Marks the file as not declaring its average so the
    /// exact-`1.0` shortcut in [`Self::bound_scale`] does not bypass the
    /// replaced value.
    #[cfg(test)]
    pub(super) fn with_bound_scale_for_test(mut self, bound_scale: f32) -> Self {
        let mut norms = self.base_norms().clone();
        norms.bound_scale = bound_scale;
        self.base_norms = Arc::new(OnceLock::from(norms));
        self.view_norms = Arc::new(OnceLock::new());
        self.declared = false;
        self
    }

    /// The BM25 length-normalization table; see [`Self::norms`].
    pub fn dl_norm_k1(&self) -> &NormTable {
        &self.norms().dl_norm_k1
    }

    /// The length statistics the norms were computed from.
    pub fn length_stats(&self) -> ColumnLengthStats {
        self.norms().length_stats
    }

    /// The factor that keeps this column's stored bounds upper bounds under
    /// the parameters it is scored at. Exactly `1.0` for a current-version
    /// file scored as declared, known without reading anything.
    pub fn bound_scale(&self) -> f32 {
        if self.declared && self.params == self.declared_params {
            return 1.0;
        }
        self.norms().bound_scale
    }
}

impl ColumnMeta {
    /// The collection size this column's inverse document frequency is
    /// computed against: the documents that carry tokens here, not the
    /// superfile's row count.
    ///
    /// The two differ by however many rows are null for this column,
    /// and using the row count does not merely shift every term's
    /// weight by a constant — the document frequency it is compared
    /// against is not rescaled with it, so a column that is null for
    /// most rows ends up weighting its common terms too heavily against
    /// its rare ones. Counting only documents that could match keeps
    /// each column's weighting independent of how often it is filled.
    pub fn scored_doc_count(&self) -> u64 {
        self.length_stats().n_scored_docs
    }

    /// The average document length this column is scored at — the one
    /// its norm table decodes with.
    pub fn avgdl(&self) -> f32 {
        self.dl_norm_k1().avgdl()
    }
}

/// JSON-deserialized form of one entry in `inf.fts.columns`. The KV
/// value is a JSON array of these, in declaration order.
#[derive(Debug, Clone, Deserialize)]
pub struct FtsColumnConfig {
    pub name: String,
    /// The column's analyzer name: `"ascii_lower"` or `"standard"`.
    /// Required — the builder has always emitted it, so a column entry
    /// without it is a malformed footer and open fails rather than
    /// guessing which analyzer produced the postings.
    pub tokenizer: String,
    /// Whether this column's index records token positions (phrase
    /// support). Files written before positions existed lack the
    /// field, which can only mean no positions — so a missing field
    /// deserializes to `false`.
    #[serde(default)]
    pub positions: bool,
    /// Whether the raw text is kept in the Parquet body. Files written
    /// before index-only columns existed lack the field, which can only
    /// mean the text is stored — so a missing field deserializes to
    /// `true` (the writer emits it only when `false`).
    #[serde(default = "default_stored")]
    pub stored: bool,
    /// BM25 term-frequency saturation this column's stored block-max
    /// bounds were built with. Files written before the parameters were
    /// recordable lack the field, and can only have been built with the
    /// standard value — so the default here is frozen at
    /// [`bm25::K1`] and must not follow a change to what the API
    /// recommends. The writer emits it unconditionally, defaults
    /// included, so no reader of a current file has to fall back on
    /// this.
    #[serde(default = "default_k1")]
    pub k1: f32,
    /// BM25 length normalization, same provenance and same frozen
    /// default ([`bm25::B`]) as [`FtsColumnConfig::k1`].
    #[serde(default = "default_b")]
    pub b: f32,
    /// Stopword set applied to this column, by name. Absent means no
    /// set — the one thing a file written before the filter existed can
    /// mean, so a missing field needs no guess. A *present* name this
    /// engine does not ship is a different matter and fails the open:
    /// there is no sound way to analyze without a set the index was
    /// built with.
    #[serde(default)]
    pub stopwords: Option<String>,
    /// Stemmer applied to this column, by name; same absent-means-off
    /// and unknown-name-fails rules as [`FtsColumnConfig::stopwords`].
    #[serde(default)]
    pub stemmer: Option<String>,
}

impl FtsColumnConfig {
    /// The parameters this column's bounds were baked at.
    pub fn params(&self) -> bm25::Bm25Params {
        bm25::Bm25Params::new(self.k1, self.b)
    }

    /// This column's analysis filters. `Err` carries the offending
    /// field name and value for an entry naming a filter this engine
    /// cannot reproduce — the caller turns that into a read error
    /// rather than analyzing the column some other way.
    pub fn filters(&self) -> Result<(Stopwords, Stemmer), (&'static str, &str)> {
        let stopwords = match &self.stopwords {
            None => Stopwords::None,
            Some(name) => Stopwords::from_name(name).ok_or(("stopwords", name.as_str()))?,
        };
        let stemmer = match &self.stemmer {
            None => Stemmer::None,
            Some(name) => Stemmer::from_name(name).ok_or(("stemmer", name.as_str()))?,
        };
        Ok((stopwords, stemmer))
    }
}

pub(super) fn default_stored() -> bool {
    true
}

pub(super) fn default_k1() -> f32 {
    bm25::K1
}

pub(super) fn default_b() -> f32 {
    bm25::B
}

/// Per-open knobs for [`FtsReader::open_with`]. Mirrors the
/// vector reader's `OpenOptions` so the superfile layer can
/// pass a single `verify_crc` flag through to both
/// sub-readers.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Verify the four per-section CRC32C checks (FST,
    /// postings region, doc-lengths directory, per-column
    /// doc-lengths arrays). Defaults to `true`; flip to
    /// `false` only when the underlying storage already
    /// validates checksums (content-addressed object
    /// store, ZFS, etc.) to skip the scan on cold open.
    pub verify_crc: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { verify_crc: true }
    }
}

impl OpenOptions {
    pub fn for_object_store() -> Self {
        Self { verify_crc: false }
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    /// The two boundary rules the analysis fields live by, asserted on
    /// the deserializer directly because both are invisible in a
    /// round-trip through our own writer.
    ///
    /// Absent means off: a file written before the filters existed has
    /// no such field, and that can only mean it was built unfiltered —
    /// so a current reader infers the right analysis with no guess. An
    /// unrecognized *value* is the opposite case and must not be
    /// tolerated: analyzing without a set the postings were built with
    /// is a different index, not a degraded one.
    #[test]
    fn absent_analysis_fields_mean_off_and_unknown_values_are_refused() {
        let entry: FtsColumnConfig =
            serde_json::from_str(r#"{"name":"body","tokenizer":"standard"}"#).expect("parse");
        assert_eq!(entry.stopwords, None);
        assert_eq!(entry.stemmer, None);
        assert_eq!(
            entry.filters().expect("no filters resolves"),
            (Stopwords::None, Stemmer::None)
        );

        let entry: FtsColumnConfig = serde_json::from_str(
            r#"{"name":"body","tokenizer":"standard","stopwords":"english","stemmer":"english"}"#,
        )
        .expect("parse");
        assert_eq!(
            entry.filters().expect("known filters resolve"),
            (Stopwords::English, Stemmer::English)
        );

        // A filter this engine does not ship, in either field.
        let entry: FtsColumnConfig =
            serde_json::from_str(r#"{"name":"body","tokenizer":"standard","stopwords":"german"}"#)
                .expect("the field parses; resolving it is what fails");
        assert_eq!(entry.filters(), Err(("stopwords", "german")));
        let entry: FtsColumnConfig =
            serde_json::from_str(r#"{"name":"body","tokenizer":"standard","stemmer":"porter"}"#)
                .expect("parse");
        assert_eq!(entry.filters(), Err(("stemmer", "porter")));
    }

    use super::{super::test_util::*, *};
    use crate::superfile::fts::{
        bm25,
        builder::{BlobEra, FtsBuilder},
        reader::FtsReader,
        tokenize::AsciiLowerTokenizer,
    };

    // ── Column length totals ──────────────────────────────────────────

    #[test]
    fn length_totals_sum_and_average_over_documents_that_carry_tokens() {
        let mut a = ColumnLengthStats {
            total_tokens: 40,
            n_scored_docs: 4,
        };
        assert_eq!(a.avgdl(), 10.0);
        a.merge_with(&ColumnLengthStats {
            total_tokens: 20,
            n_scored_docs: 1,
        });
        // 60 tokens over 5 documents — the fold is a plain sum, which is
        // what makes a fragmented table average the same as one file.
        assert_eq!(a.total_tokens, 60);
        assert_eq!(a.n_scored_docs, 5);
        assert_eq!(a.avgdl(), 12.0);
    }

    #[test]
    fn a_column_no_document_contributes_to_has_no_average() {
        // Not a division by zero and not a 1.0 default: zero is the value
        // `NormTable::new` reads as "never scored", which keeps an empty
        // column off the scoring path entirely.
        let empty = ColumnLengthStats::default();
        assert_eq!(empty.avgdl(), 0.0);
        let mut a = empty;
        a.merge_with(&empty);
        assert_eq!(a.avgdl(), 0.0);
    }

    #[test]
    fn merging_an_empty_contributor_changes_nothing() {
        // A superfile that indexes the column but holds no document with
        // tokens in it must leave the table-wide average alone rather
        // than pulling it toward zero.
        let mut a = ColumnLengthStats {
            total_tokens: 99,
            n_scored_docs: 9,
        };
        let before = a.avgdl();
        a.merge_with(&ColumnLengthStats::default());
        assert_eq!(a.avgdl(), before);
        assert_eq!(a.n_scored_docs, 9);
    }

    // ── Corpus statistics over a sparse column ────────────────────────

    /// Number of rows carrying text in the sparse fixture below.
    const SPARSE_FILLED_ROWS: u32 = 2;
    const SPARSE_EMPTY_ROWS: u32 = 6;

    /// Two documents of two tokens each, then rows this column is null
    /// for: four tokens over two documents, eight rows.
    fn sparse_builder(era: BlobEra) -> FtsBuilder {
        let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
        b.era = era;
        b.register_column("body".into(), false).expect("register");
        b.add_doc(0, 0, "alpha beta").expect("doc 0");
        b.add_doc(0, 1, "alpha gamma").expect("doc 1");
        for row in 0..SPARSE_EMPTY_ROWS {
            b.add_doc(0, SPARSE_FILLED_ROWS + row, "")
                .expect("null row");
        }
        b
    }

    fn sparse_reader() -> FtsReader {
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        FtsReader::open(
            Bytes::from(sparse_builder(BlobEra::V6).finish().expect("finish")),
            json,
        )
        .expect("open")
    }

    #[test]
    fn folding_an_unknown_contributor_makes_the_total_unknown() {
        let known = Some(ColumnLengthStats {
            total_tokens: 100,
            n_scored_docs: 10,
        });
        let more = Some(ColumnLengthStats {
            total_tokens: 50,
            n_scored_docs: 15,
        });
        assert_eq!(
            ColumnLengthStats::fold(known, more),
            Some(ColumnLengthStats {
                total_tokens: 150,
                n_scored_docs: 25,
            })
        );
        assert_eq!(ColumnLengthStats::fold(known, None), None);
        assert_eq!(ColumnLengthStats::fold(None, more), None);
        assert_eq!(
            ColumnLengthStats::from_lengths([3, 0, 1, 0]),
            ColumnLengthStats {
                total_tokens: 4,
                n_scored_docs: 2,
            }
        );
    }

    #[test]
    fn statistics_count_documents_with_tokens_not_rows() {
        let r = sparse_reader();
        let col = &r.columns[0];
        // Four tokens over the two documents that have any.
        assert_eq!(col.length_stats().total_tokens, 4);
        assert_eq!(
            col.length_stats().n_scored_docs,
            u64::from(SPARSE_FILLED_ROWS),
            "empty rows are not documents this column has"
        );
        assert_eq!(col.scored_doc_count(), u64::from(SPARSE_FILLED_ROWS));
        // 4 / 2, not 4 / 8 — dividing by the row count would deflate the
        // average by the fill rate and over-reward short documents.
        assert_eq!(col.avgdl(), 2.0);
        // The doc-lengths array still has one slot per row: the array is
        // indexed by local doc id and cannot skip rows.
        assert_eq!(
            col.dl_norm_k1().len(),
            (SPARSE_FILLED_ROWS + SPARSE_EMPTY_ROWS) as usize
        );
    }

    #[test]
    fn a_sparse_column_declares_the_average_over_its_documents() {
        // The builder divides by the documents that carry tokens, so the
        // file declares the corrected average and its bounds — exact
        // scores at that average — need no inflation.
        let r = sparse_reader();
        let col = &r.columns[0];
        assert_eq!(col.avgdl(), 2.0);
        assert_eq!(col.bound_scale(), 1.0);
    }

    #[test]
    fn an_older_sparse_file_is_corrected_and_its_bounds_inflated() {
        // A pre-current file divided by its row count. Correcting the
        // average raises it, which lowers the norm and raises every
        // score, so the bounds baked at the row-count average sit below
        // the scores they exist to cap and owe the supremum factor —
        // on top of the `(k1 + 1)` those files carry. If this regresses,
        // block-max pruning silently drops documents from the top-k.
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        for era in [BlobEra::V5, BlobEra::V2ToV4] {
            let blob = Bytes::from(sparse_builder(era).finish().expect("finish"));
            let r = FtsReader::open(blob, json).expect("open");
            let col = &r.columns[0];
            assert_eq!(
                col.avgdl(),
                2.0,
                "{era:?}: scored at the average over documents"
            );
            let legacy_scale = 1.0 / (col.params.k1 + 1.0);
            assert!(
                col.bound_scale() > legacy_scale,
                "{era:?}: a corrected average owes an inflation factor beyond the scale change, got {}",
                col.bound_scale()
            );
            // What that build recorded: the same token total over every row.
            let rows = (SPARSE_FILLED_ROWS + SPARSE_EMPTY_ROWS) as f32;
            let baked = col
                .dl_norm_k1()
                .rescored(col.length_stats().total_tokens as f32 / rows, col.params);
            for doc in 0..SPARSE_FILLED_ROWS {
                for tf in 1..8u32 {
                    let at_baked =
                        bm25::score_with_dl_norm_k1(col.params.k1 + 1.0, tf, baked.get(doc));
                    let at_scored = bm25::score_with_dl_norm_k1(1.0, tf, col.dl_norm_k1().get(doc));
                    assert!(
                        at_baked * col.bound_scale() >= at_scored - f32::EPSILON,
                        "{era:?} doc {doc} tf {tf}: {at_baked} * {} < {at_scored}",
                        col.bound_scale()
                    );
                }
            }
        }
    }

    #[test]
    fn bound_scale_reacts_to_the_average_alone() {
        // The regression that made the correction above possible: the
        // short-circuit compared only the parameter pair, so two tables
        // at the same k1/b but different averages returned 1.0 and
        // under-bounded every score.
        let r = sparse_reader();
        let col = &r.columns[0];
        let wider = col.dl_norm_k1().rescored(col.avgdl() * 2.0, col.params);
        let factor = col.dl_norm_k1().bound_scale(&wider, col.params, col.params);
        assert!(
            factor > 1.0,
            "same parameters, larger average: expected an inflation factor, got {factor}"
        );
        // And it is still exactly 1.0 when nothing moves at all.
        let same = col.dl_norm_k1().rescored(col.avgdl(), col.params);
        assert_eq!(
            col.dl_norm_k1().bound_scale(&same, col.params, col.params),
            1.0
        );
    }

    #[test]
    fn a_column_with_no_text_is_never_scored() {
        // Every row null: no document carries a token, so there is no
        // average to normalize against and the table stays empty rather
        // than dividing by zero.
        let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
        b.register_column("body".into(), false).expect("register");
        for row in 0..4 {
            b.add_doc(0, row, "").expect("null row");
        }
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let col = &r.columns[0];
        assert_eq!(col.length_stats().n_scored_docs, 0);
        assert_eq!(col.avgdl(), 0.0);
        assert_eq!(col.dl_norm_k1().len(), 0);
    }

    // ── Additional coverage ───────────────────────────────────────────

    #[test]
    fn open_with_verify_crc_off_succeeds() {
        // The trusted-storage fast path skips the four CRC scans but must
        // still produce a fully usable reader.
        let (blob, json) = build_blob();
        let r = FtsReader::open_with(blob, &json, OpenOptions { verify_crc: false })
            .expect("open with crc off");
        assert_eq!(r.n_docs(), 3);
        assert_eq!(r.fts_columns().collect::<Vec<_>>(), vec!["body"]);
    }

    #[test]
    fn open_with_object_store_options_matches_crc_off() {
        // `for_object_store` is the named constructor for the crc-off
        // OpenOptions the lazy/object-store path uses.
        let opts = OpenOptions::for_object_store();
        assert!(!opts.verify_crc);
        let (blob, json) = build_blob();
        FtsReader::open_with(blob, &json, opts).expect("open object-store options");
    }

    #[test]
    fn default_open_options_verifies_crc() {
        assert!(OpenOptions::default().verify_crc);
    }

    #[test]
    fn fts_column_config_without_tokenizer_is_rejected() {
        // The analyzer name is load-bearing: query terms must be
        // tokenized the way the postings were. A column entry missing it
        // is a malformed footer, so open fails instead of picking an
        // analyzer for the caller.
        let (blob, _) = build_blob();
        let json = r#"[{"name":"body"}]"#;
        let err = FtsReader::open(blob, json).expect_err("missing tokenizer must fail open");
        assert!(
            err.to_string().contains("tokenizer"),
            "error should name the missing field: {err}"
        );
    }

    #[test]
    fn fts_columns_config_exposes_per_column_metadata() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let cols: Vec<&ColumnMeta> = r.fts_columns_config().collect();
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0].name, "body");
        // Three non-empty docs ⇒ a positive average doc length and a
        // populated per-doc normalization table.
        assert!(cols[0].avgdl() > 0.0);
        assert_eq!(cols[0].dl_norm_k1().len(), 3);
    }

    #[test]
    fn norm_table_footprint_is_one_byte_per_doc() {
        // Memory guard: the resident length-norm table must stay at one
        // byte per doc (plus the fixed 256-entry decode LUT), not the
        // 4-byte-per-doc `f32` table it replaced. Build enough
        // varied-length docs that the per-doc term dominates the LUT.
        const N: u32 = 5_000;
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false)
            .expect("register column");
        for d in 0..N {
            // Lengths cycle 1..=40 tokens so norms span many buckets and
            // the table isn't a degenerate single value.
            let words = (d % 40) + 1;
            let text: String = (0..words).map(|w| format!("t{}x{w} ", d % 97)).collect();
            b.add_doc(0, d, text.trim()).expect("add doc");
        }
        let bytes = b.finish().expect("finish");
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(Bytes::from(bytes), json).expect("open");
        let nt = r.columns[0].dl_norm_k1();

        let per_doc = nt.bytes.len(); // 1 byte/doc
        let lut = std::mem::size_of_val(&*nt.lut); // 256 * 4 = 1 KiB
        let m2_bytes = per_doc + lut;
        let f32_baseline = N as usize * std::mem::size_of::<f32>();

        assert_eq!(nt.bytes.len(), N as usize, "one bucket byte per doc");
        assert_eq!(nt.lut.len(), 256, "fixed 256-entry decode table");
        // The whole point: strictly smaller than the old f32 table, and
        // asymptotically 4× smaller (per-doc term is 1 byte vs 4).
        assert!(
            m2_bytes < f32_baseline,
            "norm table {m2_bytes} B not smaller than f32 baseline {f32_baseline} B"
        );
        assert_eq!(
            per_doc * 4,
            f32_baseline,
            "per-doc term is exactly 4× smaller"
        );
    }
}
