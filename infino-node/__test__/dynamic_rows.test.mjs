// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Rows as documents: nested objects, arrays and new keys grow the schema,
// and the document the engine freezes for the shared corpus is the one the
// Rust and Python suites freeze. Mirrors infino-python/tests/test_dynamic_rows.py.

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { connect, IndexSpec } from "../infino/index.js";
import { Schema, Field, LargeUtf8, Binary, Table, vectorFromArray } from "apache-arrow";

const FIXTURES = fileURLToPath(new URL("../../tests/fixtures/dynamic/", import.meta.url));
const titleSchema = () => new Schema([new Field("title", new LargeUtf8(), false)]);
const corpus = () =>
  readFileSync(FIXTURES + "rows.jsonl", "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));

test("the corpus freezes the shared document", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  docs.append(corpus());
  const frozen = JSON.parse(readFileSync(FIXTURES + "schema.json", "utf8"));
  assert.deepEqual(db.schema("docs"), frozen);
  const rows = db.querySql('SELECT title, views, "author.name" FROM docs ORDER BY _id');
  assert.deepEqual(
    rows.map((r) => r.views),
    [10n, 25n, null, 7n],
  );
  assert.deepEqual(
    rows.map((r) => r["author.name"]),
    ["ann", "bob", "cy", null],
  );
});

test("documents grow the schema and disagreements are refused", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchema(), new IndexSpec());
  docs.append([{ title: "a", n: 1, tags: ["x", "y"], who: { name: "ann" } }]);
  const byName = Object.fromEntries(db.schema("docs").fields.map((f) => [f.name, f]));
  assert.equal(byName.n.type, "i64");
  assert.equal(byName.tags.type, "list");
  assert.equal(byName["who.name"].type, "large_utf8");

  // A JS float literal does not fit the Int64 column; a JS `5` does.
  assert.throws(() => docs.append([{ title: "b", n: 1.5 }]), /Int64/);
  docs.append([{ title: "c", n: 5, extra: true }]);
  const rows = db.querySql("SELECT title, n, extra FROM docs ORDER BY _id");
  assert.deepEqual(
    rows.map((r) => [r.title, r.n, r.extra]),
    [
      ["a", 1n, null],
      ["c", 5n, true],
    ],
  );
});

// A Table is the route the document path points callers at for values JSON
// cannot spell, so appending one has to carry the types a table may declare.
// Rebuilding the consumer's schema went through the createTable helper,
// which knew only the handful of types that helper needed: it threw on a
// declared `Binary` column, and on the dictionary-encoded strings
// `vectorFromArray` produces for a plain array of strings.
test("appending a Table carries the types a document cannot", () => {
  const db = connect("memory://");
  const t = db.createTable(
    "docs",
    new Schema([
      new Field("title", new LargeUtf8(), false),
      new Field("payload", new Binary(), true),
    ]),
    new IndexSpec(),
  );

  t.append(
    new Table({
      title: vectorFromArray(["a", "b"]),
      payload: vectorFromArray(
        [Uint8Array.from([0, 1]), Uint8Array.from([2, 3])],
        new Binary(),
      ),
    }),
  );

  const rows = db.querySql("SELECT title, payload FROM docs ORDER BY _id");
  assert.deepEqual(
    rows.map((r) => r.title),
    ["a", "b"],
  );
  assert.deepEqual(Array.from(rows[0].payload), [0, 1]);
  assert.deepEqual(Array.from(rows[1].payload), [2, 3]);
});
