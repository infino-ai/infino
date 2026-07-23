// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Live smoke test for the remote (hosted) transport. Skipped unless
// INFINO_REMOTE_URL and INFINO_API_KEY are set, so `npm test` passes with no
// server. Run manually against a locally-running hosted service:
//
//     cd infino-node
//     npm install && npm run build
//     INFINO_REMOTE_URL=http://localhost:8080/mydb INFINO_API_KEY=ik_... npm test
//
// The only difference from a local connection is the connect target: an
// `https://host/db` (or `http://localhost/db`) URL plus an API key.

import test from "node:test";
import assert from "node:assert/strict";
import { Field, LargeUtf8, Schema } from "apache-arrow";

import { connect, IndexSpec } from "../infino/index.js";

const url = process.env.INFINO_REMOTE_URL;
const apiKey = process.env.INFINO_API_KEY;
const skip =
  !url || !apiKey ? "set INFINO_REMOTE_URL and INFINO_API_KEY to run" : false;

test("remote round-trip against a hosted endpoint", { skip }, () => {
  const db = connect(url, { apiKey });
  const schema = new Schema([new Field("title", new LargeUtf8(), false)]);

  const posts = db.createTable("posts", schema, new IndexSpec().fts("title"));
  posts.append([{ title: "the quick brown fox" }, { title: "a lazy dog" }]);

  assert.ok(db.listTables().includes("posts"));

  const ranked = posts.bm25Search("title", "fox", 10);
  assert.equal(ranked.length, 1);

  db.dropTable("posts");
});
