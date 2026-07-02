"""End-to-end tests for the infino Python bindings over Google Cloud Storage.

These exercise the full table lifecycle through the real GCS wire protocol,
mirroring the Rust `supertable_real_gcs_round_trip` test and the Azure e2e
suite.

Gating (matches the Rust integration test):

- `INFINO_TEST_REAL_GCS=1`
- `INFINO_REAL_GCS_BUCKET` set, plus `GOOGLE_SERVICE_ACCOUNT_KEY` (the inline
  service-account key JSON). The test passes the key to `connect` as
  `storage_options` — infino reads nothing from the env.

The bucket must already exist; each test session scopes itself under a random
prefix (`gs://<bucket>/<prefix>`) and purges its tables on teardown, so runs
never collide and leave nothing behind.

Run:

    INFINO_TEST_REAL_GCS=1 \
    INFINO_REAL_GCS_BUCKET=... \
    GOOGLE_SERVICE_ACCOUNT_KEY="$(cat sa-key.json)" \
    pytest tests/test_gcs_e2e.py
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator

import infino
import pyarrow as pa
import pytest

_REQUIRED_ENV = ("INFINO_REAL_GCS_BUCKET", "GOOGLE_SERVICE_ACCOUNT_KEY")
_missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]

pytestmark = pytest.mark.skipif(
    os.environ.get("INFINO_TEST_REAL_GCS") != "1" or _missing,
    reason="set INFINO_TEST_REAL_GCS=1, INFINO_REAL_GCS_BUCKET, and GOOGLE_SERVICE_ACCOUNT_KEY to run",
)

DIM = 16  # infino requires vector dim in [16, 4096]


def _title_schema() -> pa.Schema:
    return pa.schema([pa.field("title", pa.large_utf8(), nullable=False)])


def _onehot(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def _count(db: infino.Connection, table: str) -> int:
    return db.query_sql(f"SELECT COUNT(*) AS n FROM {table}").column("n")[0].as_py()


def _storage_options() -> dict[str, str]:
    return {"google_service_account_key": os.environ["GOOGLE_SERVICE_ACCOUNT_KEY"]}


@pytest.fixture
def gcs_uri() -> str:
    bucket = os.environ["INFINO_REAL_GCS_BUCKET"]
    # Overridable so CI scopes objects per run.
    root = os.environ.get("INFINO_E2E_PREFIX", "infino-py-e2e")
    return f"gs://{bucket}/{root}/{secrets.token_hex(8)}"


@pytest.fixture
def db(gcs_uri: str) -> Iterator[infino.Connection]:
    conn = infino.connect(gcs_uri, storage_options=_storage_options())
    try:
        yield conn
    finally:
        for name in conn.list_tables():
            conn.drop_table(name, True)


def test_fts_lifecycle(db: infino.Connection) -> None:
    assert db.list_tables() == []

    table = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    table.append([{"title": "the quick brown fox"}, {"title": "a lazy dog"}])

    assert db.list_tables() == ["docs"]
    assert table.bm25_search("title", "fox", 10).num_rows == 1
    assert table.token_match("title", "dog").num_rows == 1
    assert _count(db, "docs") == 2

    tvf = db.query_sql("SELECT _id, score FROM bm25_search('docs', 'title', 'fox', 10)")
    assert tvf.num_rows == 1


def test_persists_across_reconnect(gcs_uri: str, db: infino.Connection) -> None:
    table = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    table.append([{"title": "a lazy sleeping fox"}])

    reopened = infino.connect(gcs_uri, storage_options=_storage_options())
    assert reopened.list_tables() == ["docs"]
    assert reopened.open_table("docs").bm25_search("title", "fox", 10).num_rows == 1


def test_update_delete_optimize(db: infino.Connection) -> None:
    table = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    table.append([{"title": "draft"}, {"title": "keep"}, {"title": "obsolete"}])

    assert table.delete("title = 'obsolete'").matched == 1
    assert _count(db, "docs") == 2

    assert table.update("title = 'draft'", [{"title": "published"}]).matched == 1
    assert table.token_match("title", "published").num_rows == 1
    assert table.token_match("title", "draft").num_rows == 0

    table.optimize(infino.OptimizeOptions(target_superfile_size_mb=256, min_fill_percent=50))
    assert _count(db, "docs") == 2


def test_vector_search(db: infino.Connection) -> None:
    schema = pa.schema([pa.field("emb", pa.list_(pa.float32(), DIM), nullable=False)])
    table = db.create_table("vecs", schema, infino.IndexSpec().vector("emb", DIM, 1, "cosine"))
    vecs = pa.array([_onehot(0), _onehot(1), _onehot(2)], type=pa.list_(pa.float32(), DIM))
    table.append(pa.record_batch([vecs], schema=schema))

    hits = table.vector_search("emb", _onehot(0), 10)
    assert hits.num_rows >= 1
    assert "_id" in hits.column_names and "score" in hits.column_names


def test_hybrid_search(db: infino.Connection) -> None:
    schema = pa.schema([
        pa.field("title", pa.large_utf8(), nullable=False),
        pa.field("emb", pa.list_(pa.float32(), DIM), nullable=False),
    ])
    table = db.create_table(
        "docs", schema, infino.IndexSpec().fts("title").vector("emb", DIM, 1, "cosine")
    )
    table.append(
        pa.record_batch(
            [
                pa.array(["rust async", "python data", "rust systems"], type=pa.large_utf8()),
                pa.array([_onehot(0), _onehot(1), _onehot(2)], type=pa.list_(pa.float32(), DIM)),
            ],
            schema=schema,
        )
    )

    hits = table.hybrid_search("title", "rust", "emb", _onehot(0), 10)
    assert hits.num_rows >= 1
    assert "_id" in hits.column_names and "score" in hits.column_names

    # The SQL TVF fixes mode="or" and default nprobe, so the direct call matches.
    csv = ",".join("1" if d == 0 else "0" for d in range(DIM))
    tvf = db.query_sql(f"SELECT _id FROM hybrid_search('docs', 'title', 'rust', 'emb', '{csv}', 10)")
    assert tvf.num_rows >= 1


def test_bad_credentials_fail_at_connect(gcs_uri: str) -> None:
    # A service-account path that doesn't exist fails when the provider is
    # built at connect, surfacing bad config immediately (GCS's analogue of
    # the Azure wrong-key probe — deterministic, no live auth needed).
    with pytest.raises(RuntimeError):
        infino.connect(
            gcs_uri,
            storage_options={"google_service_account": "/nonexistent/infino-gcs-key.json"},
        )


def test_drop_purge_removes_table(db: infino.Connection) -> None:
    table = db.create_table("docs", _title_schema(), infino.IndexSpec().fts("title"))
    table.append([{"title": "ephemeral"}])
    assert db.list_tables() == ["docs"]

    db.drop_table("docs", True)
    assert db.list_tables() == []
