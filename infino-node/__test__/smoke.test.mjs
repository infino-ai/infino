// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// End-to-end smoke tests for the infino Node.js bindings.
// Run after `npm run build` (which generates index.js + the .node addon):
//
//     cd infino-node
//     npm install && npm run build && npm test
//
// Mirrors infino-python/tests/test_smoke.py. Arrow data crosses the
// boundary as IPC bytes, so the helpers below build typed apache-arrow
// tables and serialize them with `tableToIPC`.

import test from "node:test";
import assert from "node:assert/strict";

import { connect, IndexSpec } from "../index.js";
import {
  Table,
  Field,
  Float32,
  FixedSizeList,
  makeData,
  makeVector,
  vectorFromArray,
  tableToIPC,
  tableFromIPC,
  LargeUtf8,
} from "apache-arrow";

// Build a `title` table and return its IPC stream bytes as a Buffer. The
// core requires FTS columns to be LargeUtf8 (not Utf8), so build that type
// explicitly — the declared schema and appended data must match.
function titleIpc(titles) {
  const table = new Table({ title: vectorFromArray(titles, new LargeUtf8()) });
  return Buffer.from(tableToIPC(table, "stream"));
}

// An empty `title` table — carries just the schema for createTable.
function titleSchemaIpc() {
  return titleIpc([]);
}

// Build an `emb` fixed_size_list<float32, dim> table from rows of numbers
// and return its IPC stream bytes. `vectorFromArray` doesn't build nested
// types, so assemble the FixedSizeList Data by hand: a flat Float32 child
// wrapped in the list type. Pass `rows = []` for an empty (schema-only)
// table to hand to createTable.
function embIpc(rows, dim) {
  const flat = Float32Array.from(rows.flat());
  const child = makeData({ type: new Float32(), length: flat.length, data: flat });
  const type = new FixedSizeList(dim, new Field("item", new Float32(), true));
  const data = makeData({ type, length: rows.length, nullCount: 0, child });
  const table = new Table({ emb: makeVector(data) });
  return Buffer.from(tableToIPC(table, "stream"));
}

// A one-hot vector of length `dim` with a 1.0 at index `i`.
function onehot(i, dim) {
  const v = new Array(dim).fill(0);
  v[i] = 1.0;
  return v;
}

function count(db, tableName) {
  const ipc = db.querySql(`SELECT COUNT(*) AS n FROM ${tableName}`);
  const out = tableFromIPC(ipc);
  return Number(out.getChild("n").get(0));
}

test("memory roundtrip: create, append, search, drop", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchemaIpc(), new IndexSpec().fts("title"));
  docs.append(titleIpc(["the quick brown fox", "a lazy dog"]));

  assert.deepEqual(db.listTables(), ["docs"]);

  const reopened = db.openTable("docs");
  const hits = reopened.bm25Search("title", "fox", 10);
  assert.equal(hits.length, 1);
  assert.equal(typeof hits[0].id, "bigint"); // _id is a bigint, not number
  assert.equal(typeof hits[0].score, "number");

  db.dropTable("docs");
  assert.deepEqual(db.listTables(), []);
});

test("querySql returns an Arrow IPC table", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchemaIpc(), new IndexSpec().fts("title"));
  docs.append(titleIpc(["alpha", "beta", "gamma"]));
  assert.equal(count(db, "docs"), 3);
});

test("querySql exposes the bm25 TVF", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchemaIpc(), new IndexSpec().fts("title"));
  docs.append(titleIpc(["the quick brown fox", "a lazy dog"]));

  const ipc = db.querySql("SELECT _id, score FROM bm25_search('docs', 'title', 'fox', 10)");
  const out = tableFromIPC(ipc);
  assert.equal(out.numRows, 1);
});

test("unknown table throws", () => {
  const db = connect("memory://");
  assert.throws(() => db.openTable("nope"));
});

test("localfs persists across reconnect", (t) => {
  const dir = `${process.env.TMPDIR ?? "/tmp"}/infino-node-smoke-${process.pid}`;
  const db = connect(dir);
  const docs = db.createTable("docs", titleSchemaIpc(), new IndexSpec().fts("title"));
  docs.append(titleIpc(["a lazy sleeping fox"]));

  const db2 = connect(dir);
  assert.deepEqual(db2.listTables(), ["docs"]);
  assert.equal(db2.openTable("docs").bm25Search("title", "fox", 10).length, 1);
});

test("vector search end-to-end", () => {
  const db = connect("memory://");
  const dim = 16; // infino requires vector dim in [16, 4096]

  const docs = db.createTable(
    "vecs",
    embIpc([], dim), // schema-only: emb fixed_size_list<float32, 16>
    new IndexSpec().vector("emb", dim, 1, "cosine"),
  );
  docs.append(embIpc([onehot(0, dim), onehot(1, dim), onehot(2, dim)], dim));

  // Query vector crosses as a Float32Array (zero-copy, by reference).
  const hits = docs.vectorSearch("emb", Float32Array.from(onehot(0, dim)), 10);
  assert.ok(hits.length >= 1);
  assert.equal(typeof hits[0].id, "bigint");
  assert.equal(typeof hits[0].score, "number");
});
