// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Dropped table handles are collected as their native memory adds up.
//
// Each handle owns engine memory the JS heap does not see (hundreds of KiB
// for an open table), freed only when V8 collects the handle. The addon
// reports that memory to V8, which then collects as it grows. Unreported, V8
// saw a few hundred bytes per handle, so a process that kept opening and
// dropping tables went thousands of handles without a collection, and its
// RSS grew by the native size of every table it had opened.

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setImmediate as yieldToEventLoop } from "node:timers/promises";

import { connect, IndexSpec } from "../infino/index.js";
import { Schema, Field, LargeUtf8, Float32, FixedSizeList } from "apache-arrow";

/** Vector width of the test table. */
const DIM = 16;
/** Rows in the test table. */
const ROWS = 200;
/** Tables opened and dropped. */
const OPENS = 1000;
/** Opens between yields to the event loop, where finalizers run. */
const YIELD_EVERY = 50;
/** Share of the dropped tables that must have been collected by the end:
 * reported at their native size, V8 collects every few hundred of them;
 * unreported, it collected none of them. */
const MIN_COLLECTED_SHARE = 0.5;

const schema = () =>
  new Schema([
    new Field("title", new LargeUtf8(), false),
    new Field("emb", new FixedSizeList(DIM, new Field("item", new Float32(), true)), false),
  ]);

const row = (i) => ({
  title: `row ${i}`,
  emb: Array.from({ length: DIM }, (_, d) => Math.sin(i * (d + 1))),
});

test("dropped table handles are collected without an explicit gc", async () => {
  const dir = mkdtempSync(join(tmpdir(), "infino-handles-"));
  {
    const db = connect(dir);
    const docs = db.createTable("docs", schema(), new IndexSpec().fts("title").vector("emb", DIM, "cosine"));
    docs.append(Array.from({ length: ROWS }, (_, i) => row(i)));
  }

  let collected = 0;
  const registry = new FinalizationRegistry(() => {
    collected += 1;
  });
  for (let i = 0; i < OPENS; i++) {
    registry.register(connect(dir).openTable("docs"), i);
    if (i % YIELD_EVERY === 0) await yieldToEventLoop();
  }
  await yieldToEventLoop();
  assert.ok(
    collected >= OPENS * MIN_COLLECTED_SHARE,
    `${collected} of ${OPENS} dropped tables collected`,
  );
});
