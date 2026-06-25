# Open-format proof: a superfile is a spec-compliant Parquet file.
#
# Shortest end-to-end demo for "anything that reads Parquet can read your data":
#
#   1. write a small corpus with infino, persisted to a local directory;
#   2. run BM25, vector, and SQL/hybrid retrieval against that same table;
#   3. locate the produced superfile on disk;
#   4. read that file back with DuckDB *and* pyarrow, no infino in the
#      read path, no export step.
#
# The embedded BM25/vector index regions live ahead of a standard Parquet
# footer and are referenced by `inf.*` key/value metadata, which a
# conformant Parquet reader simply ignores.
#
# Run:  pip install infino pyarrow duckdb pandas && python parquet_interop.py

import glob
import shutil

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

import infino

# Where the table is persisted. A real directory on disk so there is an
# actual file to open with third-party tools afterwards. Cleaned first so
# re-runs start from an empty catalog.
DATA_DIR = "./parquet_interop_data"

# 16-dim one-hot "embeddings" by topic so the demo runs with no model:
#   0 = billing, 1 = appearance. Real embeddings are dense and far larger.
DIM = 16


def embed(topic: int) -> list[float]:
    v = [0.0] * DIM
    v[topic] = 1.0
    return v


def main() -> None:
    shutil.rmtree(DATA_DIR, ignore_errors=True)

    # 1. Write a corpus with infino, indexed for BM25 + vector.
    db = infino.connect(DATA_DIR)
    schema = pa.schema([
        pa.field("source", pa.large_utf8(), nullable=False),
        pa.field("body", pa.large_utf8(), nullable=False),
        pa.field("embedding", pa.list_(pa.float32(), DIM), nullable=False),
    ])
    docs = db.create_table(
        "docs",
        schema,
        infino.IndexSpec().fts("body").vector("embedding", DIM, 1, "cosine"),
    )
    docs.append([
        {"source": "help-center", "body": "To cancel a subscription, open Settings then Billing.", "embedding": embed(0)},
        {"source": "help-center", "body": "Refunds return to the original payment method.",         "embedding": embed(0)},
        {"source": "blog",        "body": "Enable dark mode under Settings then Appearance.",        "embedding": embed(1)},
    ])

    # 2. Retrieve against the same table: BM25, vector, and a hybrid SQL
    #    query that fuses both rankings (reciprocal-rank fusion).
    print("== infino retrieval ==")
    bm25 = docs.bm25_search("body", "cancel subscription", 5, projection=["body"])
    print(f"  BM25  'cancel subscription' -> {bm25.num_rows} hit(s): {bm25.column('body').to_pylist()}")

    knn = docs.vector_search("embedding", embed(0), 5, projection=["body"])
    print(f"  kNN   topic=billing         -> {knn.num_rows} hit(s)")

    qvec = ",".join(str(x) for x in embed(0))
    hybrid = db.query_sql(f"""
        WITH lexical AS (
            SELECT _id, ROW_NUMBER() OVER (ORDER BY score DESC) AS rank
            FROM bm25_search('docs', 'body', 'cancel subscription', 50)
        ),
        semantic AS (
            SELECT _id, ROW_NUMBER() OVER (ORDER BY score ASC) AS rank
            FROM vector_search('docs', 'embedding', '{qvec}', 50)
        )
        SELECT COALESCE(l._id, s._id) AS id,
               COALESCE(1.0/(60+l.rank), 0.0) + COALESCE(1.0/(60+s.rank), 0.0) AS relevance
        FROM lexical l
        FULL OUTER JOIN semantic s ON l._id = s._id
        ORDER BY relevance DESC
        LIMIT 5
    """)
    print(f"  hybrid (RRF over BM25+vector) -> {hybrid.num_rows} ranked row(s)")

    # 3. Locate the superfiles on disk. They are ordinary files; the table
    #    lives under a per-table directory and superfiles carry a
    #    `.sf.parquet` suffix. A single write can shard into several
    #    superfiles, so read them as a set.
    glob_pat = f"{DATA_DIR}/**/*.sf.parquet"
    files = sorted(glob.glob(glob_pat, recursive=True))
    print(f"\n== on disk ({len(files)} superfile(s)) ==")
    for f in files:
        print(f"  {f}")

    # 4a. Read them back with DuckDB, no infino in this path.
    print("\n== DuckDB reads the same files ==")
    duckdb.sql(
        f"SELECT source, count(*) AS n FROM read_parquet('{glob_pat}') GROUP BY source ORDER BY source"
    ).show()

    # 4b. ...and with pyarrow/pandas, likewise no infino.
    print("== pyarrow reads the same files ==")
    table = pa.concat_tables([pq.read_table(f) for f in files])
    counts = table.select(["source"]).to_pandas().groupby("source").size()
    print(counts.to_string())

    # The index regions are invisible to a standard reader; only the
    # columnar body comes through. The vector column (`embedding`) is
    # consumed into the embedded index, not stored as a Parquet column, so
    # a standard reader sees `_id`, `source`, and `body`. The `inf.*` keys
    # live in the file metadata but carry no columns.
    print("\n== Parquet sees only the stored columns, indexes ignored ==")
    print(f"  columns: {table.column_names}")


if __name__ == "__main__":
    main()
