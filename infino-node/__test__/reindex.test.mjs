// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// The index-repair surface: `indexStaleness`, `reindexPlan` and `reindex`.
// Mirrors infino-python/tests/test_reindex.py.

import test from "node:test";
import assert from "node:assert/strict";
import { cpSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { connect, IndexSpec } from "../infino/index.js";
import { Schema, Field, LargeUtf8 } from "apache-arrow";

// A three-superfile table whose full-text index an older engine wrote, so it
// is genuinely behind what this engine writes. Shared with the crate's tests.
const OLD_FORMAT_FIXTURE = fileURLToPath(
  new URL("../../tests/fixtures/old_format_fts_table", import.meta.url),
);
// Superfiles in that fixture.
const OLD_FORMAT_SUPERFILES = 3;
// A term every row of the fixture carries.
const SHARED_TERM = "shared";
// Upper bound on hits for a fixture-wide search; above its row count.
const ALL_HITS = 1000;
// A seal-takeover age past what 32 bits of milliseconds can hold (~49 days).
const LONG_SEAL_TIMEOUT_MS = 2 ** 40;
// Where the catalog keeps its own records, beside each table's subtree.
const CATALOG_DIR = "_catalog";

const titleSchema = () => new Schema([new Field("title", new LargeUtf8(), false)]);

const tempRoot = () => mkdtempSync(join(tmpdir(), "infino-node-reindex-"));

// Register a `title` table under `root`, then swap its subtree for a copy of
// the old-format fixture, so the catalog opens the stale table.
function oldFormatTable(root) {
  connect(root).createTable("legacy", titleSchema(), new IndexSpec().fts("title"));
  const [location] = readdirSync(root).filter((name) => name !== CATALOG_DIR);
  rmSync(join(root, location), { recursive: true });
  cpSync(OLD_FORMAT_FIXTURE, join(root, location), { recursive: true });
  // A fresh connection, so no handle cached over the replaced subtree.
  return connect(root).openTable("legacy");
}

const hits = (table, term) => table.bm25Search("title", term, ALL_HITS).length;

for (const call of ["reindex", "reindexPlan", "indexStaleness"]) {
  test(`${call} refuses a memory table`, () => {
    const table = connect("memory://").createTable("docs", titleSchema(), new IndexSpec().fts("title"));
    assert.throws(() => table[call](), (e) => {
      assert.equal(e.code, "InvalidArg");
      assert.match(e.message, /requires durable storage/);
      return true;
    });
  });
}

test("an unknown mode is rejected", () => {
  const table = connect(tempRoot()).createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  assert.throws(() => table.reindexPlan({ mode: "rebuild" }), /unknown reindex mode/);
});

test("a table this engine wrote is current", () => {
  const table = connect(tempRoot()).createTable("docs", titleSchema(), new IndexSpec().fts("title"));
  table.append([{ title: "a fresh row" }]);
  table.append([{ title: "another fresh row" }]);

  assert.deepEqual(table.indexStaleness(), {
    superfiles: 2,
    needingRewrite: 0,
    awaitingReanalysis: 0,
    bytesToRewrite: 0,
    unrepairableColumns: [],
    inconsistentFooters: [],
    isCurrent: true,
  });
  assert.deepEqual(table.reindexPlan(), []);
  const report = table.reindex();
  assert.equal(report.rewritten, 0);
  assert.equal(report.alreadyCurrent, 2);
});

test("the plan names exactly what the run repairs", () => {
  const table = oldFormatTable(tempRoot());
  const before = hits(table, SHARED_TERM);
  assert.ok(before > 0);

  const staleness = table.indexStaleness();
  assert.equal(staleness.isCurrent, false);
  assert.equal(staleness.superfiles, OLD_FORMAT_SUPERFILES);

  // Every fixture superfile holds terms an older analysis produced, so the
  // default mode resolves each one to a re-analysis.
  const plan = table.reindexPlan();
  assert.equal(plan.length, OLD_FORMAT_SUPERFILES);
  assert.equal(plan.length, staleness.awaitingReanalysis);
  assert.equal(new Set(plan.map((p) => p.superfileId)).size, plan.length);
  assert.ok(plan.every((p) => p.mode === "reanalyze"));
  assert.ok(plan.every((p) => p.liveBytes > 0));

  const report = table.reindex();
  assert.deepEqual(report, {
    rewritten: plan.length,
    alreadyCurrent: OLD_FORMAT_SUPERFILES - plan.length,
    awaitingReanalysis: 0,
    heldByAnotherRun: 0,
    unrepairableColumns: [],
    inconsistentFooters: [],
  });

  // Repaired in place: nothing left to plan, and every row still answers.
  assert.equal(table.indexStaleness().isCurrent, true);
  assert.deepEqual(table.reindexPlan(), []);
  assert.equal(hits(table, SHARED_TERM), before);
});

test("a rewrite leaves analysis-stale superfiles reported", () => {
  // The fixture's layout is current and its terms are not, so a layout-only
  // run has nothing it can repair and must say so rather than rewrite.
  const table = oldFormatTable(tempRoot());
  const rewriting = { mode: "rewrite" };

  const staleness = table.indexStaleness(rewriting);
  assert.equal(staleness.needingRewrite, 0);
  assert.equal(staleness.bytesToRewrite, 0);
  assert.equal(staleness.awaitingReanalysis, OLD_FORMAT_SUPERFILES);

  assert.deepEqual(table.reindexPlan(rewriting), []);
  const report = table.reindex(rewriting);
  assert.equal(report.rewritten, 0);
  assert.equal(report.awaitingReanalysis, OLD_FORMAT_SUPERFILES);
  assert.equal(table.indexStaleness().isCurrent, false);
});

test("trusting writer analysis reaches the engine and is off by default", () => {
  // The fixture's superfiles record no analysis revision, but the engine that
  // wrote them emitted the current one — so trusting the writer reads the
  // table as current, and the default must not.
  const table = oldFormatTable(tempRoot());
  const trusting = { trustWriterAnalysis: true };

  assert.equal(table.indexStaleness(trusting).isCurrent, true);
  assert.deepEqual(table.reindexPlan(trusting), []);

  assert.equal(table.indexStaleness().isCurrent, false);
  assert.equal(table.indexStaleness({}).isCurrent, false);
  assert.equal(table.reindexPlan({}).length, OLD_FORMAT_SUPERFILES);
});

test("a long seal timeout is passed through", () => {
  const table = oldFormatTable(tempRoot());
  const report = table.reindex({ staleSealTimeoutMs: LONG_SEAL_TIMEOUT_MS });
  assert.equal(report.rewritten, OLD_FORMAT_SUPERFILES);
});

test("a negative seal timeout is rejected", () => {
  const table = oldFormatTable(tempRoot());
  assert.throws(() => table.reindexPlan({ staleSealTimeoutMs: -1 }), (e) => {
    assert.equal(e.code, "InvalidArg");
    assert.match(e.message, /must not be negative/);
    return true;
  });
});
