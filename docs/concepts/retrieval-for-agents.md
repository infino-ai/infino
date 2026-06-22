# Retrieval for agents

**Retrieval for agents is the read path an AI agent calls — often hundreds of
times per task — to fetch the context it needs to ground its next step.**
Because an agent issues so many retrievals, every millisecond of latency and
every fraction of a cent per query *compounds*: a margin too small for a human
to notice becomes decisive across a single agent task, and dominant across many.

## What agent workloads need from retrieval

- **Speed-per-dollar that compounds.** Hundreds of calls per task means the
  per-query cost and latency, not a one-off query time, set the bill and the
  wall-clock.
- **Multi-modal over one corpus.** Agents ask lexical and semantic questions
  over the same data — keyword, vector, and structured filters — ideally in one
  query rather than three systems.
- **Embeddable, no server to operate.** An agent (or the code it writes) should
  be able to stand the engine up and run a query unattended, with no service to
  provision or keep warm.
- **Runs on data you already keep.** Agent memory and RAG corpora are large and
  cheap to store; retrieval should run directly on object storage, not require
  loading the data into a separate cluster.

## How Infino fits

- **In-process, no server.** Infino is an embedded library — add it, open a
  connection, query. There is no daemon to deploy, so it stands up unattended
  in a sandbox or inside an agent runtime.
- **Multi-modal in one engine.** BM25, vector, and SQL run over one copy of the
  data, and hybrid (keyword + vector) retrieval is a single query — see
  [Hybrid search](hybrid-search.md).
- **Object-storage-native economics.** Data lives as Parquet on S3, Azure, or
  local disk; storage is flat-priced and compute is stateless — see
  [Object-storage-native retrieval](object-storage-native-retrieval.md).
- **Fast where it counts.** On a 1-million-document index a warm BM25 query
  returns in the **microsecond range**; repeated retrievals run from cached
  bytes. See [`benches/README.md`](../../benches/README.md) for current figures
  and the hardware.

```python
# Embedded — no server. Open a catalog and retrieve in a few lines.
import infino

db = infino.connect("s3://my-bucket/agent-memory")
hits = db.query_sql("""
    SELECT body FROM bm25_search('memory', 'body', 'what did the user ask about billing?', 10)
""")
```

## When it fits — and when it doesn't

This is the sweet spot: agent memory, RAG over a document corpus, and tool-call
retrieval where the data lives in (or can live in) object storage. It is not a
transactional database for high-rate row-level updates; see
[Tradeoffs and limits](../tradeoffs.md).

## See also

- [Hybrid search](hybrid-search.md) and
  [Object-storage-native retrieval](object-storage-native-retrieval.md).
- [FAQ](../faq.md) — operational answers for evaluating Infino.
