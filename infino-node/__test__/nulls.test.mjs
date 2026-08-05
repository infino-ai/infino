// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Null handling on the row-object append path: a nullable field that is
// omitted from a row must store SQL NULL, exactly like passing `null`
// explicitly — never a type's zero value. Also covers the Boolean column
// edge where every value in a batch is null.

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { connect, IndexSpec } from "../infino/index.js";
import { Schema, Field, LargeUtf8, Int32, Int64, Float64, Bool } from "apache-arrow";

// One non-null FTS column (createTable requires an index) plus one nullable
// column of the type under test.
const schemaWith = (name, type) =>
  new Schema([new Field("id", new LargeUtf8(), false), new Field(name, type, true)]);

const makeTable = (db, name, colName, type) =>
  db.createTable(name, schemaWith(colName, type), new IndexSpec().fts("id"));

// Read back the row for `id` and the table-wide IS NULL count for `col`.
const nullCount = (db, table, col) =>
  Number(db.querySql(`SELECT count(*) AS c FROM ${table} WHERE ${col} IS NULL`)[0].c);

test("omitted nullable Int32 stores null, not 0", () => {
  const db = connect("memory://");
  const t = makeTable(db, "t", "i", new Int32());
  t.append([{ id: "omitted" }, { id: "explicit", i: null }, { id: "present", i: 7 }]);

  const [row] = db.querySql("SELECT i FROM t WHERE id = 'omitted'");
  assert.equal(row.i, null);
  assert.equal(nullCount(db, "t", "i"), 2);
});

test("omitted nullable LargeUtf8 stores null, not empty string", () => {
  const db = connect("memory://");
  const t = makeTable(db, "t", "s", new LargeUtf8());
  t.append([{ id: "omitted" }, { id: "explicit", s: null }, { id: "present", s: "x" }]);

  const [row] = db.querySql("SELECT s FROM t WHERE id = 'omitted'");
  assert.equal(row.s, null);
  assert.equal(nullCount(db, "t", "s"), 2);
});

test("omitted nullable Float64 stores null, not NaN", () => {
  const db = connect("memory://");
  const t = makeTable(db, "t", "f", new Float64());
  t.append([{ id: "omitted" }, { id: "explicit", f: null }, { id: "present", f: 1.5 }]);

  const [row] = db.querySql("SELECT f FROM t WHERE id = 'omitted'");
  assert.equal(row.f, null);
  assert.equal(nullCount(db, "t", "f"), 2);
});

test("omitted nullable Int64 stores null instead of throwing", () => {
  const db = connect("memory://");
  const t = makeTable(db, "t", "n", new Int64());
  t.append([{ id: "omitted" }, { id: "explicit", n: null }, { id: "present", n: 9n }]);

  const [row] = db.querySql("SELECT n FROM t WHERE id = 'omitted'");
  assert.equal(row.n, null);
  assert.equal(nullCount(db, "t", "n"), 2);
});

test("omitted nullable Bool stores null, not false", () => {
  const db = connect("memory://");
  const t = makeTable(db, "t", "flag", new Bool());
  t.append([{ id: "omitted" }, { id: "present", flag: true }]);

  const [row] = db.querySql("SELECT flag FROM t WHERE id = 'omitted'");
  assert.equal(row.flag, null);
  assert.equal(nullCount(db, "t", "flag"), 1);
});

test("all-null nullable Bool batch appends and reads back null", () => {
  const db = connect("memory://");
  const t = makeTable(db, "t", "flag", new Bool());
  t.append([{ id: "a", flag: null }, { id: "b", flag: null }]);

  assert.deepEqual(
    db.querySql("SELECT id, flag FROM t ORDER BY id"),
    [{ id: "a", flag: null }, { id: "b", flag: null }],
  );
  assert.equal(nullCount(db, "t", "flag"), 2);
});

test("single-row null Bool batch appends and reads back null", () => {
  const db = connect("memory://");
  const t = makeTable(db, "t", "flag", new Bool());
  t.append([{ id: "a", flag: null }]);

  assert.deepEqual(db.querySql("SELECT flag FROM t"), [{ flag: null }]);
});

test("mixed null and non-null Bool batch still round-trips", () => {
  const db = connect("memory://");
  const t = makeTable(db, "t", "flag", new Bool());
  t.append([{ id: "a", flag: null }, { id: "b", flag: true }, { id: "c", flag: false }]);

  assert.deepEqual(
    db.querySql("SELECT id, flag FROM t ORDER BY id"),
    [{ id: "a", flag: null }, { id: "b", flag: true }, { id: "c", flag: false }],
  );
});

test("update replacement rows null omitted fields", () => {
  const dir = mkdtempSync(join(tmpdir(), "infino-node-nulls-"));
  const db = connect(dir);
  const t = makeTable(db, "t", "i", new Int32());
  t.append([{ id: "a", i: 1 }]);

  // Replacement row omits `i`: it must become null, not 0.
  t.update("id = 'a'", [{ id: "a" }]);
  assert.deepEqual(db.querySql("SELECT i FROM t WHERE id = 'a'"), [{ i: null }]);
});

test("update accepts an all-null Bool replacement batch", () => {
  const dir = mkdtempSync(join(tmpdir(), "infino-node-nulls-"));
  const db = connect(dir);
  const t = makeTable(db, "t", "flag", new Bool());
  t.append([{ id: "a", flag: true }]);

  t.update("id = 'a'", [{ id: "a", flag: null }]);
  assert.deepEqual(db.querySql("SELECT flag FROM t WHERE id = 'a'"), [{ flag: null }]);
});
