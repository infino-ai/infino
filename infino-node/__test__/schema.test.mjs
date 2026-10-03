// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// The schema document: read, grown from data, and changed through one write.
// Mirrors infino-python/tests/test_schema.py.

import test from "node:test";
import assert from "node:assert/strict";

import { connect, IndexSpec } from "../infino/index.js";
import { Schema, Field, LargeUtf8, Int64, Table, tableToIPC, vectorFromArray } from "apache-arrow";

const titleSchema = () => new Schema([new Field("title", new LargeUtf8(), false)]);
const names = (doc) => doc.fields.map((f) => [f.name, f.id]);

test("the schema grows from data and reads back", () => {
  const db = connect("memory://");
  const docs = db.createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  const before = db.schema("docs");
  assert.deepEqual(names(before), [["title", 1]]);
  assert.equal(before.schema_id, 1);
  assert.equal(before.fields[0].type, "large_utf8");
  assert.equal(before.fields[0].index.kind, "fts");

  docs.append([{ title: "a" }]);
  // Arrow-typed input carries columns the table has not seen; record input
  // is shaped by the table's schema.
  const grown = new Table({
    title: vectorFromArray(["b"], new LargeUtf8()),
    score: vectorFromArray([7n], new Int64()),
  });
  docs.append(Buffer.from(tableToIPC(grown, "stream")));
  const after = db.schema("docs");
  assert.deepEqual(names(after), [["title", 1], ["score", 2]]);
  assert.equal(after.fields[1].nullable, true);
  assert.equal(after.schema_id, 2);

  const rows = db.querySql("SELECT title, score FROM docs ORDER BY title");
  assert.equal(rows.length, 2);
  assert.equal(rows[0].title, "a");
  assert.equal(rows[0].score, null);
  assert.equal(rows[1].title, "b");
});

test("the schema write adds, renames, drops and guards with the expected id", () => {
  const db = connect("memory://");
  db.createTable("docs", titleSchema(), new IndexSpec().fts("title"));

  const doc = db.schema("docs", { fields: [{ name: "score", type: "i64" }] });
  assert.deepEqual(names(doc), [["title", 1], ["score", 2]]);
  // The read document applies as a no-op.
  assert.deepEqual(db.schema("docs", doc), doc);

  const renamed = db.schema("docs", { fields: [{ id: 2, name: "points" }] }, doc.schema_id);
  assert.deepEqual(names(renamed), [["title", 1], ["points", 2]]);

  assert.throws(
    () => db.schema("docs", { fields: [{ name: "tag", type: "large_utf8" }] }, doc.schema_id),
    /Conflict/,
  );

  const dropped = db.schema("docs", { fields: [{ name: "points", dropped: true }] });
  assert.deepEqual(names(dropped), [["title", 1]]);
  assert.deepEqual(dropped.tombstoned, [2]);
});

test("the schema write creates an absent table", () => {
  const db = connect("memory://");
  const doc = db.schema("fresh", {
    fields: [
      { name: "title", type: "large_utf8", nullable: false },
      { name: "score", type: "i64" },
    ],
    max_fields: 50,
  });
  assert.deepEqual(names(doc), [["title", 1], ["score", 2]]);
  assert.equal(doc.max_fields, 50);
  assert.deepEqual(db.listTables(), ["fresh"]);
  assert.throws(() => db.createTable("fresh", titleSchema(), new IndexSpec()), /AlreadyExists/);
});
