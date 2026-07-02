// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// End-to-end tests for the infino Node bindings over Google Cloud Storage.
// Mirrors infino-python/tests/test_gcs_e2e.py.
//
// Gating (matches the Python + Rust integration tests):
//
//   INFINO_TEST_REAL_GCS=1
//   INFINO_REAL_GCS_BUCKET=... GOOGLE_SERVICE_ACCOUNT_KEY="$(cat sa.json)" npm test
//
// Credentials are passed to `connect` as `storageOptions` — infino reads
// nothing from the environment. Each test scopes itself under a random
// prefix and purges its tables on teardown, so runs never collide.

import test from "node:test";
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";

import { connect, IndexSpec } from "../infino/index.js";

const DIM = 16; // infino requires vector dim in [16, 4096]

const REQUIRED = ["INFINO_REAL_GCS_BUCKET", "GOOGLE_SERVICE_ACCOUNT_KEY"];
const skip =
  process.env.INFINO_TEST_REAL_GCS === "1" && REQUIRED.every((v) => process.env[v])
    ? false
    : "set INFINO_TEST_REAL_GCS=1, INFINO_REAL_GCS_BUCKET, and GOOGLE_SERVICE_ACCOUNT_KEY to run";

const storageOptions = () => ({
  google_service_account_key: process.env.GOOGLE_SERVICE_ACCOUNT_KEY,
});

// Overridable so CI scopes objects per run.
const prefixRoot = process.env.INFINO_E2E_PREFIX ?? "infino-node-e2e";
const gcsUri = () => `gs://${process.env.INFINO_REAL_GCS_BUCKET}/${prefixRoot}/${randomUUID()}`;

const onehot = (i) => {
  const v = new Array(DIM).fill(0);
  v[i] = 1.0;
  return v;
};

// Connect to a fresh prefix, run `body(db, uri)`, then purge every table.
const withDb = (body) => {
  const uri = gcsUri();
  const db = connect(uri, { storageOptions: storageOptions() });
  try {
    body(db, uri);
  } finally {
    for (const name of db.listTables()) db.dropTable(name, true);
  }
};

test("fts lifecycle", { skip }, () => {
  withDb((db) => {
    assert.deepEqual(db.listTables(), []);

    const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
    docs.append([{ title: "the quick brown fox" }, { title: "a lazy dog" }]);

    assert.deepEqual(db.listTables(), ["docs"]);
    assert.equal(docs.bm25Search("title", "fox", 10).length, 1);
    assert.equal(docs.tokenMatch("title", "dog").length, 1);
    assert.equal(Number(db.querySql("SELECT COUNT(*) AS n FROM docs")[0].n), 2);

    const tvf = db.querySql("SELECT _id, score FROM bm25_search('docs', 'title', 'fox', 10)");
    assert.equal(tvf.length, 1);
  });
});

test("persists across reconnect", { skip }, () => {
  withDb((db, uri) => {
    const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
    docs.append([{ title: "a lazy sleeping fox" }]);

    const reopened = connect(uri, { storageOptions: storageOptions() });
    assert.deepEqual(reopened.listTables(), ["docs"]);
    assert.equal(reopened.openTable("docs").bm25Search("title", "fox", 10).length, 1);
  });
});

test("update, delete, optimize", { skip }, () => {
  withDb((db) => {
    const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
    docs.append([{ title: "draft" }, { title: "keep" }, { title: "obsolete" }]);

    assert.equal(docs.delete("title = 'obsolete'").matched, 1);
    assert.equal(Number(db.querySql("SELECT COUNT(*) AS n FROM docs")[0].n), 2);

    assert.equal(docs.update("title = 'draft'", [{ title: "published" }]).matched, 1);
    assert.equal(docs.tokenMatch("title", "published").length, 1);
    assert.equal(docs.tokenMatch("title", "draft").length, 0);

    docs.optimize({ targetSuperfileSizeMb: 256, minFillPercent: 50 });
    assert.equal(Number(db.querySql("SELECT COUNT(*) AS n FROM docs")[0].n), 2);
  });
});

test("vector search", { skip }, () => {
  withDb((db) => {
    const vecs = db.createTable("vecs", { emb: { vector: DIM } }, new IndexSpec().vector("emb", DIM, 1, "cosine"));
    vecs.append([{ emb: onehot(0) }, { emb: onehot(1) }, { emb: onehot(2) }]);

    const hits = vecs.vectorSearch("emb", onehot(0), 10);
    assert.ok(hits.length >= 1);
    assert.equal(typeof hits[0]._id, "bigint");
  });
});

test("hybridSearch", { skip }, () => {
  withDb((db) => {
    const docs = db.createTable(
      "docs",
      { title: "large_utf8", emb: { vector: DIM } },
      new IndexSpec().fts("title").vector("emb", DIM, 1, "cosine"),
    );
    docs.append([
      { title: "rust async", emb: onehot(0) },
      { title: "python data", emb: onehot(1) },
      { title: "rust systems", emb: onehot(2) },
    ]);

    const hits = docs.hybridSearch("title", "rust", "emb", onehot(0), 10);
    assert.ok(hits.length >= 1);
    assert.equal(typeof hits[0]._id, "bigint");

    // The SQL TVF fixes mode="or" and default nprobe, so the direct call matches.
    const qvec = onehot(0).join(",");
    const tvf = db.querySql(`SELECT _id FROM hybrid_search('docs', 'title', 'rust', 'emb', '${qvec}', 10)`);
    assert.ok(tvf.length >= 1);
  });
});

test("bad config fails at connect", { skip }, () => {
  // A service-account path that doesn't exist fails when the provider is
  // built at connect (GCS's analogue of the Azure wrong-key probe —
  // deterministic, no live auth needed).
  assert.throws(() =>
    connect(gcsUri(), {
      storageOptions: { google_service_account: "/nonexistent/infino-gcs-key.json" },
    }),
  );
});

test("drop purge removes the table", { skip }, () => {
  withDb((db) => {
    const docs = db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
    docs.append([{ title: "ephemeral" }]);
    assert.deepEqual(db.listTables(), ["docs"]);

    db.dropTable("docs", true);
    assert.deepEqual(db.listTables(), []);
  });
});
