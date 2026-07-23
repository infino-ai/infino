# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Infino Authors
"""Live smoke test for the remote (hosted) transport.

Skipped unless INFINO_REMOTE_URL and INFINO_API_KEY are set, so `pytest`
passes with no server. Run manually against a locally-running hosted service:

    cd infino-python
    maturin develop
    export INFINO_REMOTE_URL=http://localhost:8080/mydb
    export INFINO_API_KEY=ik_...
    pytest tests/test_remote_smoke.py

The only difference from a local connection is the connect target: an
``https://host/db`` (or ``http://localhost/db``) URL plus an API key.
"""

import os

import infino
import pyarrow as pa
import pytest

REMOTE_URL = os.environ.get("INFINO_REMOTE_URL")
API_KEY = os.environ.get("INFINO_API_KEY")

pytestmark = pytest.mark.skipif(
    not REMOTE_URL or not API_KEY,
    reason="set INFINO_REMOTE_URL and INFINO_API_KEY to run the remote smoke",
)


def test_remote_round_trip():
    db = infino.connect(REMOTE_URL, api_key=API_KEY)
    schema = pa.schema([pa.field("title", pa.large_utf8(), nullable=False)])

    posts = db.create_table("posts", schema, infino.IndexSpec().fts("title"))
    batch = pa.record_batch(
        [pa.array(["the quick brown fox", "a lazy dog"], type=pa.large_utf8())],
        schema=schema,
    )
    posts.append(batch)

    assert "posts" in db.list_tables()

    hits = posts.bm25_search("title", "fox", 10)
    assert hits.num_rows == 1

    db.drop_table("posts")
