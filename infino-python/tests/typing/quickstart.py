"""Type-check sample for the infino stubs (checked under `mypy --strict`).

Not executed — its job is to fail the type-check gate if the published
stubs drift from the surface or lose their `Literal` constraints. The
`type: ignore` lines double as assertions: `warn_unused_ignores` (part of
`--strict`) fails the run if mypy stops flagging the rejected values, i.e.
if a `Literal` silently widens to plain `str`.
"""

import pyarrow as pa

import infino


def quickstart() -> None:
    db: infino.Connection = infino.connect("memory://")

    schema = pa.schema([pa.field("title", pa.large_utf8(), nullable=False)])
    spec = infino.IndexSpec().fts("title")
    docs: infino.Table = db.create_table("docs", schema, spec)

    docs.append([{"title": "the quick brown fox"}, {"title": "a lazy dog"}])

    hits = docs.bm25_search("title", "fox", k=10, mode="and")
    names: list[str] = hits.column_names

    counts = db.query_sql("SELECT COUNT(*) AS n FROM docs")
    tables: list[str] = db.list_tables()
    _ = (names, counts, tables)


def vectors() -> None:
    db = infino.connect("memory://")
    schema = pa.schema([pa.field("emb", pa.list_(pa.float32(), 16), nullable=False)])
    spec = infino.IndexSpec().vector("emb", 16, 1, "cosine")
    vecs: infino.Table = db.create_table("vecs", schema, spec)
    vecs.vector_search("emb", [0.0] * 16, k=5, nprobe=8)
    vecs.vector_search(
        "emb",
        [0.0] * 16,
        k=5,
        filter_column="title",
        filter_query="fox",
        filter_mode="and",
    )


def mutations() -> None:
    db = infino.connect("./data")
    docs = db.open_table("docs")
    stats: infino.MutationStats = docs.delete("title = 'spam'")
    matched: int = stats.matched
    docs.update("title = 'draft'", [{"title": "published"}])
    docs.optimize(infino.OptimizeOptions(min_fill_percent=50))
    _ = matched


def rejects_invalid_literals() -> None:
    db = infino.connect("memory://")
    schema = pa.schema([pa.field("emb", pa.list_(pa.float32(), 16), nullable=False)])
    db.create_table(
        "vecs",
        schema,
        infino.IndexSpec().vector("emb", 16, 1, "euclidean"),  # type: ignore[arg-type]
    )
    db.create_table("docs", schema, infino.IndexSpec()).bm25_search(
        "emb", "q", k=1, mode="xor"  # type: ignore[arg-type]
    )
    db.create_table(
        "vecs2", schema, infino.IndexSpec().vector("emb", 16, 1, "cosine")
    ).vector_search(
        "emb",
        [0.0] * 16,
        k=1,
        filter_column="title",
        filter_query="fox",
        filter_mode="nand",  # type: ignore[arg-type]
    )
    infino.connect("memory://", cold_fetch_mode="warp_speed")  # type: ignore[arg-type]
