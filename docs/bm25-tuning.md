# Tuning BM25 relevance

How to make full-text results better on your corpus: what Infino's BM25
actually computes, how a column's analysis (tokenizer, stopwords,
stemmer) decides what is searchable at all, and the two ways to set `k1`
and `b`, per search for experimentation and per column at
table-creation time, where the engine bakes them into the index and the
query pays nothing for them.

Code pointers use paths relative to the crate root. Start with
[`src/superfile/fts/bm25.rs`](../src/superfile/fts/bm25.rs), which holds
the scoring math and the parameter type.

## What Infino scores

```text
  idf(N, df)         = ln( 1 + (N - df + 0.5) / (df + 0.5) )

  norm(dl, avgdl)    = 1 - b + b * dl / avgdl

  score(t, d)        = idf(t) * tf / ( tf + k1 * norm(dl, avgdl) )
```

A document's score for a query is the sum over its matching terms. Higher
is better (the opposite direction from `vector_search`, which returns a
distance).

Four properties of this implementation are worth knowing before you tune
anything, because each one shows up as "my scores do not match what I
expected":

1. **The IDF is the smoothed `+0.5` form**, so it is never negative: a
   term present in every document contributes about zero rather than
   pulling a score down (`idf` in `src/superfile/fts/bm25.rs`).
2. **There is no `(k1 + 1)` factor in the numerator.** It is a constant
   multiplier on every score a query produces, so it cannot change a
   ranking, and Infino omits it. Ordering is unaffected; absolute scores
   are `1 / (k1 + 1)` of the textbook-scaled value (about 0.45x at the
   default `k1`). This matters only if you compare raw scores against a
   fixed threshold, against another engine's numbers, or fuse them with
   vector distances by hand.
3. **Document length is quantized to one byte** for scoring
   (`quantize_len` / `dequantize_len`), and the average document length
   is stored in thousandths (`avgdl_x1000`). Lengths under 16 tokens are
   exact; above that the stored length is truncated downward by at most
   12.5%, and `b` scales only part of that error, since the `1 - b` term
   carries none. A reference scorer that wants to reproduce engine
   scores bit for bit feeds `stored_len(len)` and `stored_avgdl(avgdl)`
   into its own formula rather than the raw values.
4. **`avgdl` is table-wide as of the commit that wrote each superfile**,
   not a per-file average (`ColumnState::stored_average` in
   [`src/superfile/fts/builder.rs`](../src/superfile/fts/builder.rs)), and it
   is computed over the documents that carry at least one token in that
   column. Null and empty cells occupy a row slot but are not documents
   the column has, so a sparsely populated column is not penalized with a
   deflated average.

## The two knobs

`k1` controls **term-frequency saturation**: how quickly repeated
occurrences of a term stop adding score.

| `k1` | Effect |
| --- | --- |
| Low (0.3 to 0.8) | Near-binary. One occurrence is worth almost as much as ten. Good when repetition is noise (logs, boilerplate, machine-generated text) or when documents vary wildly in verbosity. |
| Default (1.2) | Standard saturation. A safe start for prose. |
| High (1.6 to 2.0) | Repetition keeps paying. Good when a term occurring often genuinely signals aboutness (long-form articles, transcripts). |

`b` controls **length normalization**: how much a long document is
penalized for being long.

| `b` | Effect |
| --- | --- |
| `0.0` | Off. Length is ignored entirely. Right for short, uniform fields where length differences are meaningless (titles, names, tags, identifiers). |
| 0.3 to 0.5 | Mild. A common pick for titles and short descriptions, and for corpora where long documents are long because they are thorough, not because they are padded. |
| Default (0.75) | Standard. A good start for mixed prose. |
| 0.9 to 1.0 | Aggressive. Use when long documents dominate results for the wrong reason (concatenated pages, appendix-heavy documents). |

Both are **per column**, so a `title` column and a `body` column in the
same table can carry different pairs. Validation is the same in both
places they can be set: `k1` must be finite and greater than 0, `b` must
be finite and in `[0, 1]`.

Starting points that are usually better than the defaults:

| Column shape | `k1` | `b` |
| --- | --- | --- |
| Titles, product names, tags, identifiers | 1.0 to 1.2 | 0.0 to 0.4 |
| Short descriptions, abstracts, summaries | 1.2 | 0.4 to 0.6 |
| Long prose (articles, docs, tickets) | 1.2 to 1.6 | 0.75 |
| Log lines, machine-generated text | 0.4 to 0.9 | 0.3 to 0.75 |
| Highly variable lengths, long docs winning too often | 1.2 | 0.85 to 1.0 |

Treat these as places to start a sweep, not as answers. The pair that
wins on your judgments beats the pair that wins on someone else's corpus.

## Setting `k1` and `b` per search

For experimentation, override the pair on the search itself. Nothing is
rebuilt, no data is rewritten, and **results stay exact**: the engine
corrects its stored pruning bounds for the new parameters rather than
pruning with bounds that no longer hold.

Rust ([`Bm25SearchOptions`](../src/superfile/fts/reader/options.rs)):

```rust
use infino::Bm25SearchOptions;

let opts = Bm25SearchOptions::new().with_bm25(0.9, 0.4);
let hits = posts.bm25_search("body", "climate policy", 10, opts, Some(&["_id", "score"]))?;
```

Python:

```python
hits = posts.bm25_search("body", "climate policy", 10, k1=0.9, b=0.4)
```

Node:

```js
const hits = posts.bm25Search("body", "climate policy", 10, {
  k1: 0.9,
  b: 0.4,
  projection: ["_id", "score"],
});
```

Pass `k1` and `b` together or neither; the bindings reject half a pair.

Two limits are worth planning around:

- **SQL cannot override the pair.** The `bm25_search` table function
  takes `(column, query, k [, mode])` and scores with each column's
  declared pair
  ([`src/supertable/query/exec/fts_exec.rs`](../src/supertable/query/exec/fts_exec.rs)).
- **`hybrid_search` cannot override it either.** It takes a boolean mode
  and nothing else, so its BM25 half always scores with the declared
  pair.

Both are reasons to bake a pair you have settled on: it is the only way
every query path gets it.

## Baking the pair into the index

Declare the pair on the column when you create the table
([`FtsField::bm25`](../src/catalog/index_spec.rs)):

```rust
use infino::{FtsField, IndexSpec};

let spec = IndexSpec::new()
    .fts(FtsField::new("title").bm25(1.2, 0.3))
    .fts(FtsField::new("body").bm25(1.6, 0.75));
let posts = db.create_table("posts", schema, spec)?;
```

Python:

```python
spec = IndexSpec().fts("title", k1=1.2, b=0.3).fts("body", k1=1.6, b=0.75)
posts = db.create_table("posts", schema, spec)
```

Node:

```js
const spec = new IndexSpec()
  .fts("title", { k1: 1.2, b: 0.3 })
  .fts("body", { k1: 1.6, b: 0.75 });
const posts = db.createTable("posts", schema, spec);
```

What declaring it does:

- The pair is recorded in the catalog entry (`fts_k1` / `fts_b` in
  [`src/catalog/manifest.rs`](../src/catalog/manifest.rs)) and read back on
  `open_table`, so every process that opens the table scores the same way.
- Every superfile written from then on records the pair alongside its
  index, and **its per-block score bounds are computed under that pair**
  (`ColumnState::params` in `src/superfile/fts/builder.rs`).
- Later appends and compactions bake bounds at the declared pair too, so
  the table does not drift as it grows or merges.

The pair is fixed at `create_table`: there is no alter path for it. To
adopt a new pair on an existing table, create a table declaring the pair
and re-ingest into it. (Unlike the analysis options below, this is a
performance migration, not a correctness one. A query-time override
already gives you the exact ranking a rebuild would; the rebuild buys
back the pruning.)

## Why a baked pair is faster

Infino's ranked kernels are block-max walks: each posting block carries
an upper bound on the scores it can produce, and the kernel skips any
block whose bound cannot beat the current top-k floor
([`src/superfile/fts/reader/bounds.rs`](../src/superfile/fts/reader/bounds.rs)).
A stored bound is the block's **true maximum under the pair the build
baked in**, which is as tight as the format allows.

When a query overrides the pair
(`FtsReader::with_bm25_override` in
[`src/superfile/fts/reader/core.rs`](../src/superfile/fts/reader/core.rs)),
two things happen per column:

1. The 256-entry length-norm decode table is rebuilt at the new pair. The
   per-document length buckets are shared, not copied, so this is one
   1 KiB table per column plus a reference-count bump, not a pass over
   every document.
2. Each column picks up a `bound_scale`: the supremum over the length
   buckets documents actually occupy of `(1 + A) / (1 + B)`, where `A`
   and `B` are the baked and query normalizers
   (`NormTable::bound_scale` in
   [`src/superfile/fts/reader/metadata.rs`](../src/superfile/fts/reader/metadata.rs)).
   It is always at least 1, so every stored bound is inflated. Results
   stay exact, because an inflated bound is still an upper bound; what is
   lost is pruning power. Blocks that would have been skipped now get
   decoded and scored.

The size of that loss grows with how far the pair moved and with how wide
the column's length spread is, since the supremum is taken across the
occupied buckets. A column whose declared pair already equals the
override is short-circuited and costs nothing, so passing the pair a
column already uses is free rather than wasteful.

One more effect worth knowing if you are serving several pairs at once:
each superfile reader memoizes exactly one derived view, keyed by the
override pair (`fts_scored` in
[`src/superfile/reader.rs`](../src/superfile/reader.rs)). Repeated queries
at the same pair reuse it; queries alternating between pairs rebuild the
view each time. An experiment sweeping pairs is fine. A production
workload where every tenant sends a different pair is not what this path
is for, and is another argument for baking.

To see the difference on your own data, run the full-text benches before
and after baking:

```sh
cargo bench --bench bench -- superfile fts warm
cargo bench --bench bench -- supertable fts warm
```

Recorded numbers live in [`benches/README.md`](../benches/README.md); each run
also writes `target/infino-bench/<bench>.json`, which the next run uses as
its delta baseline.

## Tokenizer, stopwords, and stemming

`k1` and `b` weigh the terms an index holds. Analysis decides **which
terms it holds at all**, which is why it is the part to get right first:
a pair of parameters can reorder results, but only analysis can make a
document findable or unfindable.

A column's analysis is a chain of up to three stages, applied in a fixed
order ([`src/superfile/fts/analysis.rs`](../src/superfile/fts/analysis.rs)):

```text
  text -> base tokenizer -> stopword removal -> stemmer -> indexed terms
```

Three properties hold for the whole chain:

- **Both sides run it.** The column's chain tokenizes the documents at
  ingest and the query string at search time, so the two can never
  disagree about what a word is.
- **It is per column.** A `title` column and a `body` column in one
  table can use different chains.
- **It is fixed at `create_table`.** The chain is recorded with the
  table, and there is no migration: a filter's effect is not recoverable
  from the index it produced, because the tokens it removed or rewrote
  were never written. Changing analysis means creating a new table and
  re-ingesting from the source text. On a `stored(false)` column the
  source text is never kept, so not even that is possible. Declare a
  filter with `stored(false)` only when you are sure.

Stopwords come before stemming and are matched against the unstemmed
token, because stopword lists are written in surface forms (`are`,
`their`, `these`). The order is not configurable.

### Base tokenizer

Two ship today, named by `FtsField::analyzer` (the option is spelled
"analyzer" but the value names a tokenizer; the chain is the analyzer).
See [`src/superfile/fts/tokenize.rs`](../src/superfile/fts/tokenize.rs).

| | `standard` (default) | `ascii_lower` |
| --- | --- | --- |
| Splitting | Unicode UAX #29 word boundaries; segments containing an alphanumeric are kept | Runs of `[A-Za-z0-9]`; every other ASCII byte separates |
| Case | Full Unicode case folding | ASCII `A-Z` to `a-z` |
| Non-ASCII | Preserved, so accented and non-Latin scripts stay searchable | Any token containing a non-ASCII byte is dropped silently |
| Use it for | Natural language, anything multilingual, anything user-written | ASCII-only identifiers, codes, SKUs, log lines |

Two details that surprise people:

- **No Unicode normalization.** A token is indexed in the code points it
  arrived in, so precomposed `é` and `e` plus a combining acute are
  different terms. This matches Lucene's standard analyzer, which
  normalizes only through a separate opt-in filter. If your data mixes
  encodings, normalize it upstream.
- **Tokens are capped at 255 characters.** A longer run is chopped into
  255-character pieces, each occupying its own position, rather than
  being truncated or dropped.

Word boundaries are worth checking rather than assuming, because they
decide what a query can ask for. `Supertable::tokenize(column, text)`
returns exactly what the column's chain produces, which settles most
"why did this not match" questions in one call.

### Stopwords

`FtsField::stopwords(Stopwords::English)` (`stopwords="english"` in the
bindings) removes Lucene's 33-word Snowball English set from both index
and query:

```text
a an and are as at be but by for if in into is it no not of on or such
that the their then there these they this to was will with
```

English is the only set today. The list is frozen on purpose: a column
persists the name `english`, not the words, so editing the list would
silently re-analyze every column already built under that name. A
different list would arrive as a new name rather than an edit.

What you gain: a smaller index, and faster queries containing common
words (the engine never walks a posting list holding most of the
corpus).

What it costs:

- **Queries about a stopword become unanswerable.** A query whose terms
  are all removed has no clause left and returns no rows. `"to be or not
  to be"` finds nothing on a stopworded column.
- **Document lengths shrink**, because length counts the tokens the
  chain emits. That changes what `b` normalizes against, so re-sweep
  `b` after turning stopwords on.
- **Document frequencies shift** for surviving terms relative to the
  corpus, which moves IDF slightly.

What it does not cost, which is the part worth knowing: **phrases stay
correct**. A removed token leaves a hole in the position sequence, so
`"new york"` does not match `new the york`, and `"end of the world"`
matches only text with exactly two words between `end` and `world`. The
holes reach both the index and the query side, and are covered by
[`tests/superfile/fts/analysis_chain.rs`](../tests/superfile/fts/analysis_chain.rs).

Declare stopwords on prose columns where common words carry no signal.
Do not declare them on short identifiers, titles, or codes, where every
token is doing work and the index is small anyway.

### Stemming

`FtsField::stemmer(Stemmer::English)` (`stemmer="english"` in the
bindings) is Snowball English, which is Porter2, via `rust-stemmers`. It
is named `english` rather than `porter` because Lucene's
`PorterStemFilter` is the *original* Porter, a different algorithm; if
you are comparing against Lucene, compare against its
`EnglishAnalyzer`, not its `PorterStemFilter`.

It folds inflections onto one term on both sides, so `running`, `runs`
and `run` are one term and any of them finds all of them. Irregular
forms it has no rule for (`ran`, `went`) stay distinct.

What it costs:

- **Precision.** A stemmed column conflates words a reader would not,
  and there is no way to ask for the unstemmed form: the index holds
  `run`, never the `running` it came from.
- **IDF.** Folding inflections together raises the merged term's
  document frequency, which lowers its IDF. A stemmed term is slightly
  less discriminating than any of the surface forms it replaced, so
  scores shift even for queries whose result set does not change. This
  is the third reason to re-sweep `k1` and `b` after changing analysis.

Stemming is usually right for long-form English prose where recall
matters more than precision (support tickets, articles, documentation),
and usually wrong for names, identifiers, and short catalog fields.

### Costs that are not about relevance

Three consequences of declaring any filter, none of which are obvious
from the relevance side:

- **`LIKE` and `ILIKE` superfile pruning turns off for that column.**
  The SQL lowering bounds a `LIKE` fragment by the terms it tokenizes
  to, which is sound only while an indexed term is a substring-preserving
  image of the text. With a stemmer it is not: `LIKE '%runni%'` matches
  the text `running`, whose indexed term is `run`. So a column carrying
  a chain keeps every superfile for such a predicate
  ([`Analyzer::of` in `src/supertable/query/candidate.rs`](../src/supertable/query/candidate.rs)).
  Results stay correct; the skip is what is lost.
- **Analysis is part of the table's identity.** A merge refuses inputs
  whose per-column analysis differs, because a merged file records one
  chain per column and carried postings would silently stop matching
  (`check_fts_carry_compat` in
  [`src/superfile/builder.rs`](../src/superfile/builder.rs)). This is also
  why there is no in-place way to change the chain.
- **Prefix search walks indexed terms.** The prefix is lowercased and
  expanded against the dictionary, which on a stemmed column holds
  stems. A prefix of a surface form can therefore expand to nothing
  where the same prefix would match on an unstemmed column.

Recording is additive and degrades safely in one direction only: an
engine predating a filter ignores the field and analyzes the column
unfiltered (wrong answers on that column until it rolls forward,
recoverable), while a filter *value* the engine does not know, such as a
stopword set it does not ship, is a typed error at open rather than a
guess.

### Picking a chain

| Column | Tokenizer | Stopwords | Stemmer | Typical `k1` / `b` |
| --- | --- | --- | --- | --- |
| Titles, product names | `standard` | off | off | 1.0 to 1.2 / 0.0 to 0.4 |
| Long English prose | `standard` | `english` | `english` | 1.2 to 1.6 / 0.75 |
| Multilingual or mixed-script text | `standard` | off | off | 1.2 / 0.75 |
| Identifiers, SKUs, codes | `ascii_lower` | off | off | 1.2 / 0.0 |
| Log lines, machine output | `ascii_lower` | off | off | 0.4 to 0.9 / 0.3 |
| Code or symbol-heavy text | `ascii_lower` | off | off | 1.2 / 0.3 to 0.5 |

Declaring a chain in each language:

```rust
use infino::{FtsField, IndexSpec, Stemmer, Stopwords};

let spec = IndexSpec::new()
    .fts(FtsField::new("title").bm25(1.1, 0.3))
    .fts(
        FtsField::new("body")
            .stopwords(Stopwords::English)
            .stemmer(Stemmer::English)
            .positions(true)
            .bm25(1.4, 0.75),
    );
```

```python
spec = (
    IndexSpec()
    .fts("title", k1=1.1, b=0.3)
    .fts("body", k1=1.4, b=0.75, stopwords="english", stemmer="english", positions=True)
)
```

```js
const spec = new IndexSpec()
  .fts("title", { k1: 1.1, b: 0.3 })
  .fts("body", {
    k1: 1.4,
    b: 0.75,
    stopwords: "english",
    stemmer: "english",
    positions: true,
  });
```

## Levers that usually matter more than `k1` and `b`

Parameter tuning is the last 10% of relevance work. These come first, and
most of them are decided at `create_table`, where they are recorded with
the table and cannot be changed without re-ingesting from the source
text.

| Lever | Where | Changeable later | What it does |
| --- | --- | --- | --- |
| Corpus statistics scope (`Bm25Stats`) | Per search | Yes | `Global` (the default) scores every superfile against table-wide document counts and document frequencies, so a fragmented table ranks like one corpus. `PerSuperfile` uses each file's own statistics: faster, but a term's IDF depends on which file a document landed in. On a fragmented table this is usually the single biggest ranking lever. See [`src/superfile/fts/reader/options.rs`](../src/superfile/fts/reader/options.rs). |
| Base tokenizer | `FtsField::analyzer` | No | `standard` or `ascii_lower`. See [Tokenizer, stopwords, and stemming](#tokenizer-stopwords-and-stemming). |
| Stopwords | `FtsField::stopwords` | No | Drops very common words from index and query. See [Stopwords](#stopwords). |
| Stemmer | `FtsField::stemmer` | No | Folds inflections onto one term, so one form finds the others. See [Stemming](#stemming). |
| Positions | `FtsField::positions` | No | Required for exact phrase queries. Roughly doubles the column's index footprint. A phrase against a positionless column is a typed error, never a silent bag-of-words fallback. |
| Boolean mode and sigils | Per search | Yes | `BoolMode::And` makes bare terms mandatory, `Or` makes them scoring-only once any `+must` exists. `+term` is a must, `-term` a hard exclusion, and `"quoted words"` is a phrase atom scored as one term whose `tf` is the phrase count and whose IDF is the sum of its members'. |
| Query shape | Per search | Yes | Prefix search expands a trailing prefix to the indexed terms beginning with it and runs an OR over them. `hybrid_search` fuses text and vector results; do not fuse raw BM25 scores with distances by hand. |

Two debugging helpers repay the time they take:

- `Supertable::tokenize(column, text)` returns the terms a column's
  analyzer produces, which settles most "why did this not match"
  questions in one call.
- `token_match` and `exact_match` return the candidate set without
  ranking, which separates "the wrong documents matched" from "the right
  documents ranked badly". Only the second is a `k1`/`b` problem.

## A tuning loop that works

1. **Build a judgment set first.** Twenty to fifty real queries, each
   with the ids that should come back. Without it you are tuning on
   anecdotes.
2. **Fix analysis before parameters.** Tokenizer, stopwords, stemmer and
   positions decide what is in the index, and changing them later means
   re-ingesting. Get them right (see
   [Tokenizer, stopwords, and stemming](#tokenizer-stopwords-and-stemming)),
   then tune scoring on top. Every analysis change moves document
   lengths and document frequencies, so a pair tuned before it is worth
   re-sweeping after.
3. **Check the match set, then the order.** Use `token_match` to confirm
   the right documents are candidates at all.
4. **Sweep `k1` and `b` at query time** against the real table, scoring
   each pair on your judgments (nDCG@10, MRR, or recall@k, whichever
   matches how results are consumed). Sweep `b` first: it usually moves
   more than `k1`.
5. **Watch the score scale** if anything downstream uses raw scores.
   Changing `k1` moves absolute scores even where it barely moves the
   order.
6. **Bake the winner** into a table declaring it, re-ingest, and confirm
   two things: the ranking matches what the query-time override produced
   (the same pair yields the same scores), and latency improved.
7. **Revisit after the corpus grows.** `avgdl` and document frequencies
   move as the table grows, and a pair tuned on 100K documents is worth
   rechecking at 10M.

## Verifying a change

- The scoring oracle in
  [`src/test_helpers/brute_force_bm25.rs`](../src/test_helpers/brute_force_bm25.rs)
  computes textbook BM25 on a planted corpus with no index, no skip
  table and no pruning. It accepts a parameter pair, so a tuned column
  or an overridden query can be graded against the same formula the
  engine runs.
- [`tests/supertable/query/brute_force_oracle.rs`](../tests/supertable/query/brute_force_oracle.rs)
  exercises exactly this: a column declaring a non-default pair, and
  queries overriding the pair, both compared against the oracle. It is
  the test to extend if you change anything in the scoring or bound path.
- [`tests/superfile/fts/analysis_chain.rs`](../tests/superfile/fts/analysis_chain.rs)
  covers the filter chains against the same oracle: stopword and stem
  behaviour on both sides, position holes in phrases, and the document
  lengths a chain produces. Its siblings
  [`standard_tokenizer.rs`](../tests/superfile/fts/standard_tokenizer.rs) and
  [`token_cap.rs`](../tests/superfile/fts/token_cap.rs) cover the base
  tokenizers.
- Run the benches from the section above before and after any change that
  touches scoring, and report the comparison. A relevance win that costs
  latency is a trade to state explicitly, not to discover later.

## Common surprises

| Symptom | Cause |
| --- | --- |
| Scores are smaller than another engine reports | The missing `(k1 + 1)` numerator. Ranking is unaffected; multiply by `k1 + 1` to compare numbers. |
| Scores differ slightly from a hand-written BM25 | One-byte length quantization and the thousandths-resolution `avgdl`. Feed `stored_len` and `stored_avgdl` into the reference formula. |
| Ranking shifts as the table is written to | Per-superfile corpus statistics. Use the default `Bm25Stats::Global`. |
| A tuned pair works in Rust or a binding but not in SQL | SQL and `hybrid_search` have no override. Declare the pair on the column. |
| `create_table` rejects the pair | `k1` must be finite and greater than 0, `b` finite and in `[0, 1]`. The error names the column ([`src/supertable/error.rs`](../src/supertable/error.rs)). |
| A query-time override made things slower | Expected: looser bounds prune fewer blocks. Bake the pair once you have settled on it. |
| Phrase query returns an error | The column was not declared with `positions(true)`, and that is decided at `create_table`. |
| A query of only common words returns nothing | The column declares `stopwords`, so no clause survived parsing. An empty result, not an error. |
| A `LIKE` predicate got slower after declaring a filter | Superfile skip pruning for `LIKE` / `ILIKE` turns off on a column carrying a stopword set or stemmer. |
| Prefix search finds nothing on a stemmed column | The dictionary holds stems and the prefix is not stemmed. |
| Accented text matches inconsistently | `standard` applies no Unicode normalization. Normalize upstream. |

## File map

| Area | Path |
| --- | --- |
| Scoring math, `Bm25Params`, length quantization | [`src/superfile/fts/bm25.rs`](../src/superfile/fts/bm25.rs) |
| Base tokenizers and query parsing | [`src/superfile/fts/tokenize.rs`](../src/superfile/fts/tokenize.rs) |
| Analysis chain (stopwords, stemmer, position holes) | [`src/superfile/fts/analysis.rs`](../src/superfile/fts/analysis.rs) |
| Analysis compatibility on merge | [`src/superfile/builder.rs`](../src/superfile/builder.rs) |
| `LIKE` lowering and analyzer recognition | [`src/supertable/query/candidate.rs`](../src/supertable/query/candidate.rs) |
| Per-search options (`Bm25SearchOptions`, `BoolMode`, `Bm25Stats`) | [`src/superfile/fts/reader/options.rs`](../src/superfile/fts/reader/options.rs) |
| Column declaration (`FtsField`, `IndexSpec`) | [`src/catalog/index_spec.rs`](../src/catalog/index_spec.rs) |
| Declared pair recorded in the catalog | [`src/catalog/manifest.rs`](../src/catalog/manifest.rs), [`src/catalog/mod.rs`](../src/catalog/mod.rs) |
| Bound baking at build time | [`src/superfile/fts/builder.rs`](../src/superfile/fts/builder.rs) |
| Override view and bound correction | [`src/superfile/fts/reader/core.rs`](../src/superfile/fts/reader/core.rs), [`src/superfile/fts/reader/metadata.rs`](../src/superfile/fts/reader/metadata.rs) |
| Bound decoding for the ranked kernels | [`src/superfile/fts/reader/bounds.rs`](../src/superfile/fts/reader/bounds.rs) |
| Per-superfile derived-view memo | [`src/superfile/reader.rs`](../src/superfile/reader.rs) |
| Query fan-out, statistics scope, query syntax | [`src/supertable/query/fts.rs`](../src/supertable/query/fts.rs) |
| SQL search table functions | [`src/supertable/query/exec/fts_exec.rs`](../src/supertable/query/exec/fts_exec.rs), [`src/catalog/search_tvf.rs`](../src/catalog/search_tvf.rs) |
| Public search surface | [`src/catalog/table.rs`](../src/catalog/table.rs) |

See also [`docs/architecture/superfile.md`](./architecture/superfile.md) for the
on-disk layout the bounds live in, and
[`docs/architecture/supertable.md`](./architecture/supertable.md) for how a
query fans out across superfiles.
