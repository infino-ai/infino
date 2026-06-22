# Hybrid search

**Hybrid search runs keyword (BM25) and vector (semantic) retrieval over the
same query and fuses their two rankings into one result list.** Keyword search
nails exact terms — error codes, identifiers, names, rare words; vector search
captures meaning — paraphrase, synonyms, intent. Hybrid returns what each finds
that the other misses.

## Why combine them

- **Keyword alone** misses paraphrase: a query for "cancel subscription" won't
  rank a doc that says "end your plan."
- **Vector alone** misses exact tokens: a semantic query can rank a near-miss
  above the row that contains the literal error code or product name.
- **RAG and agent retrieval need both.** The fix is to retrieve with each and
  fuse the rankings — commonly with **reciprocal-rank fusion (RRF)**, which
  merges two ranked lists by rank position, so you don't have to reconcile a
  BM25 score with a vector distance.

## How Infino does it — one engine, one copy of the data

In Infino the BM25 and vector indexes live in the same superfile over a single
copy of your rows, and the retrievers are **SQL table-valued functions**. So
hybrid search is ordinary SQL composition inside one engine — not a separate
keyword system and vector system with results stitched together in the client:

- `hybrid_search(...)` packages the common RRF fusion as a single call; or
- compose `bm25_search(...)` and `vector_search(...)` yourself and rank by a
  fusion score (a `FULL OUTER JOIN` on `_id` plus an RRF expression).

```sql
-- Keyword + vector over one table, fused (reciprocal-rank fusion) into one
-- ranked list — one query, one snapshot, no client-side stitching.
SELECT _id, score
FROM hybrid_search('docs', 'body', 'how do I cancel my subscription?',
                   'embedding', '<query-vector>', 10);
```

Because it runs over one snapshot of one copy of the data, there is no second
store to keep in sync and no result-merging glue in your application.

## When it fits

Hybrid is the default for retrieval where queries are natural language but
exact terms still matter — support search, RAG over mixed content, agent
retrieval. When a query is purely lexical (codes, exact phrases) keyword search
alone is enough; when it's purely conceptual, vector search alone is enough.

## See also

- [Object-storage-native retrieval](object-storage-native-retrieval.md) — the
  storage model the indexes sit on.
- [Retrieval for agents](retrieval-for-agents.md) — why hybrid retrieval
  matters for agent workloads.
- [Supertable layer → Queries](../architecture/supertable.md#queries) — the
  search table functions and how they compose in SQL.
- [FAQ](../faq.md) and [Tradeoffs and limits](../tradeoffs.md).
