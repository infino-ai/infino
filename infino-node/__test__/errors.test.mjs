// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Hosted-API error status. A failure the hosted service reported is
// rethrown unchanged except for one added property: `status`, the HTTP
// status the API actually returned, so callers branch on a number instead
// of regexing message text. Local (embedded) connections are untouched.
//
// The hosted-path tests run a stub API server on a worker thread — the
// SDK's calls are synchronous and would deadlock a same-thread server.

import test from "node:test";
import assert from "node:assert/strict";
import { Worker } from "node:worker_threads";

import { connect, IndexSpec } from "../infino/index.js";

// A one-route HTTP stub: every request gets `status` + `body` back.
const SERVER_SRC = `
  const { parentPort, workerData } = require("node:worker_threads");
  const http = require("node:http");
  const server = http.createServer((req, res) => {
    res.writeHead(workerData.status, { "content-type": "application/json" });
    res.end(workerData.body);
  });
  server.listen(0, "127.0.0.1", () => parentPort.postMessage(server.address().port));
`;

async function withStubApi(status, body, run) {
  const worker = new Worker(SERVER_SRC, { eval: true, workerData: { status, body } });
  const port = await new Promise((resolve) => worker.once("message", resolve));
  try {
    run(connect(`http://127.0.0.1:${port}/yelp`, { apiKey: "test-key" }));
  } finally {
    await worker.terminate();
  }
}

const throws = (fn) => {
  try {
    fn();
  } catch (e) {
    return e;
  }
  assert.fail("expected the call to throw");
};

// --- hosted path: `.status` = what the API returned ---

test("hosted 409 create carries status 409 and the API's message", async () => {
  await withStubApi(409, '{"error":"database already exists: cust_x/yelp"}', (db) => {
    const e = throws(() => db.createDatabase());
    assert.equal(e.status, 409);
    // The stable message prefix separates a duplicate from a bad argument.
    assert.match(e.message, /^AlreadyExists: /);
    // The API's own words stay visible.
    assert.match(e.message, /database already exists/);
  });
});

test("hosted 404 carries status 404", async () => {
  await withStubApi(404, '{"error":"database does not exist"}', (db) => {
    const e = throws(() => db.openTable("t"));
    assert.equal(e.status, 404);
  });
});

test("hosted 503 carries status 503 (transient — caller can retry)", async () => {
  await withStubApi(503, '{"error":"no capacity to activate database"}', (db) => {
    const e = throws(() => db.listTables());
    assert.equal(e.status, 503);
    assert.match(e.message, /no capacity/);
  });
});

test("hosted 500 carries status 500", async () => {
  await withStubApi(500, '{"error":"boom"}', (db) => {
    const e = throws(() => db.listTables());
    assert.equal(e.status, 500);
  });
});

test("hosted 401 keeps the unauthorized message (transport carries no number)", async () => {
  await withStubApi(401, '{"error":"invalid api key"}', (db) => {
    const e = throws(() => db.listTables());
    assert.equal(e.status, undefined);
    assert.match(e.message, /unauthorized \(check the API key\)/);
  });
});

// --- local path: untouched — no status, addon error rethrown as-is ---

test("local duplicate createTable throws the addon error with no status", () => {
  const db = connect("memory://");
  const make = () => db.createTable("docs", { title: "large_utf8" }, new IndexSpec().fts("title"));
  make();
  const e = throws(make);
  assert.equal(e.status, undefined);
  assert.equal(e.code, "InvalidArg");
  assert.match(e.message, /^AlreadyExists: /);
});

test("local openTable of a missing table throws the addon error with no status", () => {
  const db = connect("memory://");
  const e = throws(() => db.openTable("missing"));
  assert.equal(e.status, undefined);
  assert.equal(e.code, "GenericFailure");
  assert.match(e.message, /^NotFound: /);
});

test("local query failures are untouched", () => {
  const db = connect("memory://");
  const e = throws(() => db.querySql("definitely not sql"));
  assert.equal(e.status, undefined);
  assert.equal(typeof e.code, "string");
});
