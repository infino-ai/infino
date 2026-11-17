"""Idempotency tokens: a retried append commits once."""

import infino
import pyarrow as pa
import pytest


def _docs(tmp_path):
    db = infino.connect(str(tmp_path / "catalog"))
    schema = pa.schema([pa.field("title", pa.large_string(), nullable=False)])
    return db.create_table("docs", schema, infino.IndexSpec().fts("title"))


def test_a_retried_keyed_append_commits_once(tmp_path):
    docs = _docs(tmp_path)
    assert docs.idempotency_token_status("order-1") == {"status": "absent"}

    docs.append([{"title": "a quick fox"}], idempotency_token="order-1")
    docs.append([{"title": "a quick fox"}], idempotency_token="order-1")

    assert docs.count("title", "fox") == 1
    status = docs.idempotency_token_status("order-1")
    assert status["status"] == "committed"
    assert status["seq"] is None
    assert status["manifest_id"] > 0 and status["committed_ms"] > 0


def test_a_sequence_skips_its_number_and_every_earlier_one(tmp_path):
    docs = _docs(tmp_path)
    docs.append_named([{"title": "fox one"}], "loader", idempotency_token=("loader", 2))
    docs.append([{"title": "fox two"}], idempotency_token=("loader", 1))
    docs.append([{"title": "fox three"}], idempotency_token=("loader", 3))

    assert docs.count("title", "fox") == 2
    status = docs.idempotency_token_status(("loader", 1))
    assert status["status"] == "committed" and status["seq"] == 3


def test_a_token_is_refused_on_a_table_with_no_storage():
    db = infino.connect("memory://")
    schema = pa.schema([pa.field("title", pa.large_string(), nullable=False)])
    docs = db.create_table("docs", schema, infino.IndexSpec().fts("title"))
    with pytest.raises(NotImplementedError, match="storage"):
        docs.append([{"title": "fox"}], idempotency_token="k")
    docs.append([{"title": "fox"}])
    assert docs.count("title", "fox") == 1
