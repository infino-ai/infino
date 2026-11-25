// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Idempotency tokens: a retried append commits once. Mirrors
// infino-python/tests/test_idempotency.py.

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { connect, IndexSpec } from "../infino/index.js";
import { Schema, Field, LargeUtf8 } from "apache-arrow";

const titleSchema = () => new Schema([new Field("title", new LargeUtf8(), false)]);
const docsTable = () =>
  connect(mkdtempSync(join(tmpdir(), "infino-node-idem-"))).createTable("docs", titleSchema(), new IndexSpec().fts("title"));

test("a retried keyed append commits once", () => {
  const docs = docsTable();
  const since = Date.now();
  assert.deepEqual(docs.idempotencyTokenStatus("order-1", since), { status: "absent" });

  docs.append([{ title: "a quick fox" }], { idempotencyToken: "order-1" });
  docs.append([{ title: "a quick fox" }], { idempotencyToken: "order-1" });

  assert.equal(docs.count("title", "fox"), 1);
  const status = docs.idempotencyTokenStatus("order-1", since);
  assert.equal(status.status, "committed");
  assert.ok(status.manifestId > 0 && status.committedMs > 0);
  assert.equal(status.seq, undefined);
});

test("a key reused with other rows is refused", () => {
  const docs = docsTable();
  docs.append([{ title: "a quick fox" }], { idempotencyToken: "order-1" });
  assert.throws(
    () => docs.append([{ title: "a quick fox" }, { title: "a lazy fox" }], { idempotencyToken: "order-1" }),
    /one exact write/,
  );
  assert.equal(docs.count("title", "fox"), 1);
});

test("a sequence skips its number and every earlier one", () => {
  const docs = docsTable();
  const since = Date.now();
  docs.appendNamed([{ title: "fox one" }], "loader", { idempotencyToken: { scope: "loader", seq: 2 } });
  docs.append([{ title: "fox two" }], { idempotencyToken: { scope: "loader", seq: 1 } });
  docs.append([{ title: "fox three" }], { idempotencyToken: { scope: "loader", seq: 3 } });

  assert.equal(docs.count("title", "fox"), 2);
  const status = docs.idempotencyTokenStatus({ scope: "loader", seq: 1 }, since);
  assert.equal(status.status, "committed");
  assert.equal(status.seq, 3);
  assert.deepEqual(docs.idempotencyTokenStatus({ scope: "loader", seq: 4 }, since), { status: "absent" });
});

test("a token is refused on a table with no storage", () => {
  const docs = connect("memory://").createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  assert.throws(() => docs.append([{ title: "fox" }], { idempotencyToken: "k" }));
  docs.append([{ title: "fox" }]);
  assert.equal(docs.count("title", "fox"), 1);
});

test("a negative seq is refused", () => {
  const docs = docsTable();
  assert.throws(() => docs.append([{ title: "fox" }], { idempotencyToken: { scope: "w", seq: -1 } }), /seq/);
});

test("a seq that is not a safe integer is refused", () => {
  const docs = docsTable();
  for (const seq of [1.5, 2 ** 53]) {
    assert.throws(() => docs.append([{ title: "fox" }], { idempotencyToken: { scope: "w", seq } }), /seq/);
  }
  assert.equal(docs.count("title", "fox"), 0);
});
