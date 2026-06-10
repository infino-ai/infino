// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Friendly Node-idiomatic API over the raw napi addon (./index.js).
//
// The addon traffics in Arrow IPC `Buffer`s; this layer hides that. You
// pass arrays of plain objects (or apache-arrow Tables / RecordBatches)
// in, and get arrays of plain records out — mirroring the ergonomics of
// LanceDB / nodejs-polars (and the Python binding). Pass `{ arrow: true }`
// to a query/search to get an apache-arrow `Table` instead of records.

const arrow = require("apache-arrow");
const native = require("./index.js");

const STREAM = "stream";

// ---------------------------------------------------------------------------
// Arrow <-> IPC helpers (the boundary this layer hides)
// ---------------------------------------------------------------------------

// An apache-arrow Schema -> the IPC bytes the addon's createTable wants (an
// empty table that carries just the schema). A Buffer passes through, for
// callers that already have IPC bytes.
function schemaToIpc(schema) {
  if (Buffer.isBuffer(schema) || schema instanceof Uint8Array) return schema;
  if (!(schema instanceof arrow.Schema)) {
    throw new TypeError("createTable: schema must be an apache-arrow Schema");
  }
  const children = schema.fields.map((f) => arrow.makeData({ type: f.type, length: 0 }));
  const structData = arrow.makeData({
    type: new arrow.Struct(schema.fields),
    length: 0,
    nullCount: 0,
    children,
  });
  const empty = new arrow.Table(new arrow.RecordBatch(schema, structData));
  return Buffer.from(arrow.tableToIPC(empty, STREAM));
}

// Build one typed Arrow column from row objects. `vectorFromArray` handles
// scalars; FixedSizeList<Float32> (vector columns) need the nested Data
// built by hand.
function buildColumn(field, rows) {
  const values = rows.map((r) => r[field.name]);
  if (field.type instanceof arrow.FixedSizeList) {
    const flat = Float32Array.from(values.flat());
    const child = arrow.makeData({ type: field.type.children[0].type, length: flat.length, data: flat });
    const data = arrow.makeData({ type: field.type, length: rows.length, nullCount: 0, child });
    return arrow.makeVector(data);
  }
  return arrow.vectorFromArray(values, field.type);
}

// Normalize append input -> IPC bytes. Accepts an array of objects (typed
// against the table's declared schema), an apache-arrow Table/RecordBatch,
// or raw IPC bytes.
function dataToIpc(data, getSchema) {
  if (Buffer.isBuffer(data) || data instanceof Uint8Array) return data;
  if (data instanceof arrow.Table) return Buffer.from(arrow.tableToIPC(data, STREAM));
  if (data instanceof arrow.RecordBatch) {
    return Buffer.from(arrow.tableToIPC(new arrow.Table([data]), STREAM));
  }
  if (Array.isArray(data)) {
    const schema = getSchema();
    const cols = {};
    for (const field of schema.fields) cols[field.name] = buildColumn(field, data);
    return Buffer.from(arrow.tableToIPC(new arrow.Table(cols), STREAM));
  }
  throw new TypeError(
    "append: expected an array of objects, an apache-arrow Table / RecordBatch, or an Arrow IPC Buffer",
  );
}

// IPC result bytes -> records (default) or an apache-arrow Table.
function decode(buf, asArrow) {
  const table = arrow.tableFromIPC(buf);
  return asArrow ? table : table.toArray().map((row) => row.toJSON());
}

// ---------------------------------------------------------------------------
// Friendly handles
// ---------------------------------------------------------------------------

class Table {
  constructor(inner) {
    this._inner = inner;
  }

  // The table's Arrow schema (apache-arrow `Schema`).
  schema() {
    return arrow.tableFromIPC(this._inner.schema()).schema;
  }

  // Append rows. Accepts an array of objects, an apache-arrow Table /
  // RecordBatch, or raw IPC bytes. Durable on return; one append == one
  // commit.
  append(data) {
    this._inner.append(dataToIpc(data, () => this.schema()));
  }

  // Ranked BM25 search. Returns matching rows as records (or an
  // apache-arrow Table with `{ arrow: true }`). Options: { mode,
  // materialize, arrow }.
  bm25Search(column, query, k, opts = {}) {
    const buf = this._inner.bm25Search(column, query, k, opts.mode, opts.materialize);
    return decode(buf, opts.arrow);
  }

  // Vector kNN. `query` is a number[] or Float32Array. Returns rows as
  // records (or a Table with `{ arrow: true }`). Options: { nprobe,
  // materialize, arrow }.
  vectorSearch(column, query, k, opts = {}) {
    const q = query instanceof Float32Array ? query : Float32Array.from(query);
    const buf = this._inner.vectorSearch(column, q, k, opts.nprobe, opts.materialize);
    return decode(buf, opts.arrow);
  }

  // Unranked token match — `[{ id, score }]` (`id` is a bigint, score 0).
  tokenMatch(column, query, mode) {
    return this._inner.tokenMatch(column, query, mode);
  }

  // Unranked exact match — `[{ id, score }]` (`id` is a bigint, score 0).
  exactMatch(column, value) {
    return this._inner.exactMatch(column, value);
  }
}

class Connection {
  constructor(inner) {
    this._inner = inner;
  }

  // Create a table from an apache-arrow `Schema` and an `IndexSpec`.
  createTable(name, schema, indexes) {
    return new Table(this._inner.createTable(name, schemaToIpc(schema), indexes));
  }

  openTable(name) {
    return new Table(this._inner.openTable(name));
  }

  dropTable(name) {
    this._inner.dropTable(name);
  }

  listTables() {
    return this._inner.listTables();
  }

  // Run SQL across the catalog. Returns rows as records (or an apache-arrow
  // Table with `{ arrow: true }`).
  querySql(sql, opts = {}) {
    return decode(this._inner.querySql(sql), opts.arrow);
  }
}

// Open (or create) a catalog rooted at `uri`.
function connect(uri, options) {
  return new Connection(native.connect(uri, options));
}

module.exports = { connect, Connection, Table, IndexSpec: native.IndexSpec };
