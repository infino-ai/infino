// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// `bm25SearchFields` through the public wrapper: each clause scores a row by
// its best column, a row's score is the sum over the clauses it matches, and
// options (mode, weights, projection) reach the addon.

import test from "node:test";
import assert from "node:assert/strict";

import { connect, IndexSpec } from "../infino/index.js";
import { Schema, Field, LargeUtf8 } from "apache-arrow";

const seeded = () => {
  const db = connect("memory://");
  const schema = new Schema([new Field("title", new LargeUtf8(), false), new Field("body", new LargeUtf8(), false)]);
  const docs = db.createTable("docs", schema, new IndexSpec().fts("title").fts("body"));
  docs.append([
    { title: "quick fox", body: "a lazy dog sleeps" },
    { title: "lazy dog", body: "the quick brown fox jumps over the lazy dog" },
    { title: "hound", body: "the fox and the hound" },
    { title: "nothing here", body: "nothing here either" },
  ]);
  return docs;
};

const byId = (rows) => new Map(rows.map((r) => [String(r._id), r.score]));

const expected = (docs, clauses, weights = {}) => {
  const total = new Map();
  for (const { query, fields } of clauses) {
    const best = new Map();
    for (const f of fields) {
      for (const [id, s] of byId(docs.bm25Search(f, query, 100))) {
        best.set(id, Math.max(best.get(id) ?? 0, s * (weights[f] ?? 1)));
      }
    }
    for (const [id, s] of best) total.set(id, (total.get(id) ?? 0) + s);
  }
  return total;
};

const assertScores = (got, want) => {
  assert.deepEqual([...got.keys()].sort(), [...want.keys()].sort());
  for (const [id, s] of want) assert.ok(Math.abs(got.get(id) - s) <= 1e-5 * Math.max(1, Math.abs(s)), `${id}: ${got.get(id)} vs ${s}`);
};

test("a row scores its best column per clause, summed over clauses", () => {
  const docs = seeded();
  const clauses = [
    { query: "fox", fields: ["title", "body"] },
    { query: "hound", fields: ["title", "body"] },
  ];
  assertScores(byId(docs.bm25SearchFields(clauses, 10)), expected(docs, clauses));
});

test("weights multiply a column's scores", () => {
  const docs = seeded();
  const clauses = [{ query: "fox", fields: ["title", "body"] }];
  const weights = { title: 3 };
  assertScores(byId(docs.bm25SearchFields(clauses, 10, { weights })), expected(docs, clauses, weights));
});

test("and mode keeps rows matching every clause", () => {
  const docs = seeded();
  const clauses = [
    { query: "fox", fields: ["title", "body"] },
    { query: "hound", fields: ["title", "body"] },
  ];
  assert.equal(docs.bm25SearchFields(clauses, 10).length, 3);
  assert.equal(docs.bm25SearchFields(clauses, 10, { mode: "and" }).length, 1);
});

test("projection returns the named columns", () => {
  const docs = seeded();
  const rows = docs.bm25SearchFields([{ query: "hound", fields: ["title"] }], 10, { projection: ["title", "score"] });
  assert.deepEqual(rows.map((r) => r.title), ["hound"]);
  assert.deepEqual(Object.keys(rows[0]).sort(), ["score", "title"]);
});

test("bad arguments are refused", () => {
  const docs = seeded();
  assert.throws(() => docs.bm25SearchFields([], 10));
  assert.throws(() => docs.bm25SearchFields([{ query: "fox", fields: ["missing"] }], 10));
  assert.throws(() => docs.bm25SearchFields([{ query: "fox", fields: ["title"] }], 10, { weights: { title: 0 } }));
});
