"""RAG core for the chat-with-docs app — all the Infino logic, no UI.

Kept separate from `app.py` so it can be imported and tested headlessly. The
app layer only does Streamlit widgets; everything that touches the engine lives
here. The table is durable on local disk, so ingested documents survive across
app restarts.
"""

import os
import sys

import infino
import pyarrow as pa

# Import the shared example helpers (this file lives in examples/chat_app/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _shared.embedding import DIM, METRIC, embed, embed_query, as_vector_column  # noqa: E402
from _shared.chunking import chunk_text  # noqa: E402

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_data")
TABLE = "docs"
RRF_K = 10  # candidates pulled per retrieval


def open_db():
    """Open (or lazily create) the durable local catalog."""
    return infino.connect(DB_DIR)


def get_table(db):
    """Return the docs table, creating it on first use."""
    try:
        return db.open_table(TABLE)
    except Exception:
        schema = pa.schema([
            pa.field("title", pa.large_utf8(), nullable=False),
            pa.field("text", pa.large_utf8(), nullable=False),
            pa.field("source", pa.large_utf8(), nullable=False),
            pa.field("emb", pa.list_(pa.float32(), DIM), nullable=False),
        ])
        spec = infino.IndexSpec().fts("text").vector("emb", DIM, n_cent=64, metric=METRIC)
        return db.create_table(TABLE, schema, spec)


def count(db) -> int:
    res = db.query_sql(f"SELECT COUNT(*) AS n FROM {TABLE}")
    return res.to_pydict()["n"][0] if res.num_rows else 0


def ingest(table, docs: list[dict]) -> int:
    """Chunk, embed, and append documents.

    `docs` is a list of `{"title", "text", "source"}`. Returns the chunk count.
    """
    titles, texts, sources = [], [], []
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            titles.append(doc.get("title", ""))
            texts.append(chunk)
            sources.append(doc.get("source", ""))
    if not texts:
        return 0
    table.append(pa.record_batch([
        pa.array(titles, type=pa.large_utf8()),
        pa.array(texts, type=pa.large_utf8()),
        pa.array(sources, type=pa.large_utf8()),
        as_vector_column(embed(texts)),
    ], schema=table.schema()))
    return len(texts)


def _sql_lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def retrieve(db, query: str, k: int = 4) -> list[dict]:
    """Hybrid (BM25 + vector) retrieval, fused in-engine via one SQL statement."""
    qvec = ",".join(str(x) for x in embed_query(query))
    sql = (
        f"SELECT p.title, p.text, p.source, h.score "
        f"FROM hybrid_search('{TABLE}', 'text', {_sql_lit(query)}, 'emb', '{qvec}', {max(k, RRF_K)}) h "
        f"JOIN {TABLE} p ON h._id = p._id "
        f"ORDER BY h.score DESC LIMIT {k}"
    )
    res = db.query_sql(sql)
    if not res.num_rows:
        return []
    d = res.to_pydict()
    return [
        {"title": t, "text": x, "source": s, "score": sc}
        for t, x, s, sc in zip(d["title"], d["text"], d["source"], d["score"])
    ]


def build_prompt(query: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[{i + 1}] {h['title']}: {h['text']}" for i, h in enumerate(hits))
    return (
        "Answer the question using only the context. Cite sources as [n].\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )


def answer(query: str, hits: list[dict]) -> str | None:
    """Generate an answer if an OpenAI key is configured, else return None.

    Returning None lets the UI fall back to showing the retrieved sources, so
    the app is useful with no API key at all.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from openai import OpenAI

    reply = OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": build_prompt(query, hits)}],
    )
    return reply.choices[0].message.content
