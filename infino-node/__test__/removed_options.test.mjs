// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Options the engine no longer takes throw at the wrapper, since the addon
// would otherwise drop them without a word.

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { connect, IndexSpec } from "../infino/index.js";
import { Schema, Field, LargeUtf8 } from "apache-arrow";

const titleSchema = () => new Schema([new Field("title", new LargeUtf8(), false)]);

const durableTable = () =>
  connect(mkdtempSync(join(tmpdir(), "infino-node-removed-"))).createTable(
    "docs",
    titleSchema(),
    new IndexSpec().fts("title"),
  );

test("IndexSpec.fts refuses `analyzer`", () => {
  assert.throws(() => new IndexSpec().fts("title", { analyzer: "standard" }), (e) => {
    assert.ok(e instanceof TypeError);
    assert.match(e.message, /IndexSpec\.fts: option `analyzer` is not supported/);
    return true;
  });
});

test("IndexSpec.fts refuses `analyzer` on a chained call", () => {
  const spec = new IndexSpec().fts("title");
  assert.throws(() => spec.fts("body", { analyzer: "ascii_lower" }), /option `analyzer`/);
});

test("IndexSpec.fts still takes its current options", () => {
  assert.doesNotThrow(() => new IndexSpec().fts("title", { stopwords: "english", stemmer: "english" }));
});

test("bm25Search refuses `stats`", () => {
  const docs = connect("memory://").createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  docs.append([{ title: "the quick brown fox" }]);
  assert.throws(() => docs.bm25Search("title", "fox", 10, { stats: "per_superfile" }), (e) => {
    assert.ok(e instanceof TypeError);
    assert.match(e.message, /bm25Search: option `stats` is not supported/);
    return true;
  });
});

for (const call of ["reindex", "reindexPlan", "indexStaleness"]) {
  test(`${call} refuses \`trustWriterAnalysis\``, () => {
    const docs = durableTable();
    assert.throws(() => docs[call]({ trustWriterAnalysis: true }), (e) => {
      assert.ok(e instanceof TypeError);
      assert.match(e.message, new RegExp(`${call}: option \`trustWriterAnalysis\` is not supported`));
      return true;
    });
  });
}
