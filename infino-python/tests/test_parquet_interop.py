"""Open-format proof, wired into CI.

A superfile is a spec-compliant Parquet file: third-party readers
(DuckDB, pyarrow) open its columnar body with no infino in the read
path. This mirrors the README "Open format" snippet and the
`examples/parquet_interop.py` demo.
"""

import glob
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import infino

DIM = 16  # infino requires vector dim in [16, 4096]


def _onehot(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def _write_corpus(uri: str):
    db = infino.connect(uri)
    schema = pa.schema(
        [
            pa.field("source", pa.large_utf8(), nullable=False),
            pa.field("body", pa.large_utf8(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), DIM), nullable=False),
        ]
    )
    docs = db.create_table(
        "docs",
        schema,
        infino.IndexSpec().fts("body").vector("embedding", DIM, 1, "cosine"),
    )
    docs.append(
        [
            {
                "source": "help-center",
                "body": "To cancel a subscription, open Settings then Billing.",
                "embedding": _onehot(0),
            },
            {
                "source": "help-center",
                "body": "Refunds return to the original payment method.",
                "embedding": _onehot(0),
            },
            {
                "source": "blog",
                "body": "Enable dark mode under Settings then Appearance.",
                "embedding": _onehot(0),
            },
        ]
    )
    return db, docs


# Stored Parquet columns. The auto-injected `_id` and the user's scalar
# columns are stored columnar; the vector column (`embedding`) is consumed
# into the embedded index, not written as a Parquet column.
STORED_COLUMNS = {"_id", "source", "body"}


def _superfile_glob(uri: str) -> str:
    # The catalog manifest (`_catalog/current`) and per-table state are not
    # `.parquet`, so this matches only superfiles.
    return f"{uri}/**/*.sf.parquet"


def _superfile_paths(uri: str) -> list[str]:
    # A single write can shard into several superfiles (the commit path
    # builds shards in parallel), so the table is the union of all of them.
    matches = sorted(glob.glob(_superfile_glob(uri), recursive=True))
    assert matches, f"no superfile written under {uri}"
    return matches


def test_infino_retrieval_still_works(tmp_path):
    # Sanity: the writer side (the thing being proven open) actually
    # indexes BM25 + vector before we read it as plain Parquet.
    _, docs = _write_corpus(str(tmp_path / "data"))
    assert docs.bm25_search("body", "cancel subscription", 5).num_rows >= 1
    assert docs.vector_search("embedding", _onehot(0), 5).num_rows >= 1


def test_superfile_is_on_disk(tmp_path):
    uri = str(tmp_path / "data")
    _write_corpus(uri)
    for path in _superfile_paths(uri):
        assert path.endswith(".sf.parquet")


def test_pyarrow_reads_superfile_as_parquet(tmp_path):
    uri = str(tmp_path / "data")
    _write_corpus(uri)
    paths = _superfile_paths(uri)

    # No infino in the read path: pyarrow's own Parquet reader, unioning
    # the shards a single write produced.
    table = pa.concat_tables([pq.read_table(p) for p in paths])
    # Only the columnar body is visible; the embedded index regions and
    # `inf.*` metadata keys are ignored by a standard reader.
    assert set(table.column_names) == STORED_COLUMNS

    counts = dict(Counter(table.column("source").to_pylist()))
    assert counts == {"help-center": 2, "blog": 1}


def test_duckdb_reads_superfile_as_parquet(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    uri = str(tmp_path / "data")
    _write_corpus(uri)

    # The exact shape the README advertises: read_parquet over the glob +
    # GROUP BY, no infino import in this query.
    rows = duckdb.sql(
        f"SELECT source, count(*) AS n FROM read_parquet('{_superfile_glob(uri)}') "
        "GROUP BY source ORDER BY source"
    ).fetchall()
    assert dict(rows) == {"blog": 1, "help-center": 2}


def test_round_trip_through_generic_writer_drops_indexes(tmp_path):
    # The honest limit: rewriting a superfile through a generic Parquet
    # writer yields valid Parquet that is no longer a superfile. We can't
    # cheaply assert "indexes gone" from Python, but we can show the
    # round-trip produces a plain file that re-opens with the same
    # columns, i.e. the columnar body survives, which is all a generic
    # writer preserves.
    uri = str(tmp_path / "data")
    _write_corpus(uri)
    table = pa.concat_tables([pq.read_table(p) for p in _superfile_paths(uri)])

    rewritten = tmp_path / "rewritten.parquet"
    pq.write_table(table, rewritten)

    reopened = pq.read_table(str(rewritten))
    assert reopened.num_rows == 3
    assert set(reopened.column_names) == STORED_COLUMNS
