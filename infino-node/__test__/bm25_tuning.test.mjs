// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Per-search BM25 tuning through the public wrapper. The addon has taken
// `k1` / `b` since the pair became overridable per search, but the wrapper
// forwarded only mode / stats / projection, so the options were silently
// dropped at the boundary: a caller passing them got the column's declared
// pair and no error. These tests hold the wrapper to the addon's contract.

import test from "node:test";
import assert from "node:assert/strict";

import { connect, IndexSpec } from "../infino/index.js";
import { Schema, Field, LargeUtf8 } from "apache-arrow";

const titleSchema = () => new Schema([new Field("title", new LargeUtf8(), false)]);

const seeded = () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  docs.append([
    { title: "the quick brown fox" },
    { title: "the fox and the hound" },
    { title: "a lazy dog" },
  ]);
  return docs;
};

test("k1 and b together are accepted and the search still ranks", () => {
  const docs = seeded();
  const tuned = docs.bm25Search("title", "fox", 10, { k1: 1.2, b: 0.75, projection: ["_id", "score"] });
  assert.equal(tuned.length, 2);
  assert.ok(tuned.every((r) => typeof r.score === "number"));
});

test("a different pair changes the scores, proving the override reached the engine", () => {
  const docs = seeded();
  const scores = (opts) => docs.bm25Search("title", "fox", 10, { ...opts, projection: ["_id", "score"] }).map((r) => r.score);
  const declared = scores({});
  // Near-zero saturation with full length normalization: the two "fox" rows
  // differ in length, so their scores must move away from the declared pair's.
  const retuned = scores({ k1: 0.1, b: 1.0 });
  assert.equal(declared.length, retuned.length);
  assert.notDeepEqual(declared, retuned);
});

test("an out-of-range pair is refused by the engine's own validation", () => {
  const docs = seeded();
  assert.throws(() => docs.bm25Search("title", "fox", 10, { k1: 0, b: 0 }), /k1 must be finite and > 0/);
});

test("half a pair is rejected by the addon, not swallowed by the wrapper", () => {
  const docs = seeded();
  for (const half of [{ k1: 1.2 }, { b: 0.75 }]) {
    assert.throws(() => docs.bm25Search("title", "fox", 10, half), /pass k1 and b together, or neither/);
  }
});
