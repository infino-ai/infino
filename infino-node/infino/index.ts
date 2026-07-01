// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Public Node.js API for infino. Pass arrays of objects (or apache-arrow
// Tables) in; get plain records out. `{ arrow: true }` on a search or
// query returns an apache-arrow `Table` instead of records.

import * as arrow from "apache-arrow";
import { connect as nativeConnect, IndexSpec, builderId } from "./native.js";

export { IndexSpec };

/** Infino's build identifier (version + build hash). */
export const BUILDER_ID: string = builderId();

const STREAM = "stream";

// --- public types ---

/** Vector distance metric. */
export type Metric = "cosine" | "l2sq" | "negdot";
/** Boolean mode for multi-term FTS queries. */
export type BoolMode = "or" | "and";
/** A row from a query/search when not materializing to Arrow. */
export type RowRecord = Record<string, unknown>;
/** A plain `{ column: type }` schema descriptor for `createTable`. */
export type SchemaDescriptor = Record<string, string | { vector: number }>;
/** Accepted shapes for `Table.append`. */
export type AppendData = RowRecord[] | arrow.Table | arrow.RecordBatch | Buffer | Uint8Array;

/** Storage and cache config the `connect` URI can't carry. All optional. */
export interface ConnectOptions {
  /**
   * Credentials/tuning for the URI-selected backend, keyed by `object_store`
   * config strings (`aws_*` / `azure_*`). An unknown key is rejected at
   * `connect`.
   */
  storageOptions?: Record<string, string>;
  /** Local disk-cache directory for remote-backed tables. */
  cacheDir?: string;
  /** Disk-cache budget in bytes. */
  cacheBudgetBytes?: number;
  /** How cold misses are serviced. */
  coldFetchMode?: "hybrid_with_prefetch" | "range_only" | "lazy_foreground_with_background_fill";
  /**
   * Probe the object store at `connect` (default `false`). `true` fails fast
   * on bad credentials instead of on the first table operation.
   */
  validate?: boolean;
}

/** Row counts returned by `update` / `delete`. */
export interface MutationStats {
  /** Rows the predicate matched. */
  matched: number;
  /** Rows tombstoned (removed from the live set). */
  nTombstoned: number;
  /** Matched rows not found in any live segment. */
  nNotFound: number;
}

/** Counts from a `gc` sweep. */
export interface GcReport {
  /** Bytes reclaimed by deleting orphaned objects. */
  bytesFreed: number;
  /** Orphaned objects deleted. */
  objectsDeleted: number;
  /** Objects kept because they are still referenced by the live set. */
  objectsSkippedLive: number;
  /** Objects kept because they are younger than the grace period. */
  objectsSkippedTooNew: number;
  /** Objects that failed to delete (left for the next sweep). */
  deleteErrors: number;
}

/** Tuning for `optimize`; all fields optional (omitted ⇒ engine default). */
export interface OptimizeOptions {
  /** Build-time memory budget, in MB. */
  maxMemoryMb?: number;
  /** Only compact superfiles below this fill percent (0–100). */
  minFillPercent?: number;
  /** Target merged-superfile size, in MB. */
  targetSuperfileSizeMb?: number;
}

export interface Bm25SearchOptions {
  mode?: BoolMode;
  /** Columns to return, e.g. `["_id", "score"]`; omit for full rows. */
  projection?: string[];
  arrow?: boolean;
}
/** Text-predicate filter for `vectorSearch` (a pushdown pre-filter, not a
 * post-filter): kNN ranks only among rows whose FTS-indexed `column` matches
 * `query`. */
export interface VectorFilter {
  /** FTS-indexed column the predicate applies to. */
  column: string;
  /** Query terms, tokenized by the index tokenizer. */
  query: string;
  /** `"or"` (default) or `"and"`. */
  mode?: BoolMode;
}
export interface VectorSearchOptions {
  /** IVF partitions to probe (higher = better recall, more work). */
  nprobe?: number;
  /** Over-fetch multiplier for the exact-rerank stage (higher = better recall). */
  rerankMult?: number;
  projection?: string[];
  arrow?: boolean;
  /** Restrict the kNN to rows matching a text predicate (pushdown pre-filter). */
  filter?: VectorFilter;
}
/** Options for `hybridSearch`. `mode` applies to the BM25 side; `nprobe` to
 * the vector side. */
export interface HybridSearchOptions {
  /** BM25 boolean mode: `"or"` (default) or `"and"`. */
  mode?: BoolMode;
  /** IVF partitions to probe on the vector side (higher = better recall). */
  nprobe?: number;
  projection?: string[];
  arrow?: boolean;
}
export interface TokenMatchOptions {
  mode?: BoolMode;
  projection?: string[];
  arrow?: boolean;
}
export interface MatchOptions {
  projection?: string[];
  arrow?: boolean;
}
export interface CountOptions {
  mode?: BoolMode;
}
export interface QueryOptions {
  arrow?: boolean;
}

// --- Arrow <-> IPC helpers (the boundary this layer hides) ---

// Rebuild an arrow type in our instance from the consumer's type. When
// apache-arrow is loaded as two module instances, our `makeData` can't
// dispatch on the consumer's type object, but its numeric `typeId` reads
// across instances.
function nativeTypeFromForeign(t: any): arrow.DataType {
  switch (t.typeId) {
    case arrow.Type.Utf8: return new arrow.Utf8();
    case arrow.Type.LargeUtf8: return new arrow.LargeUtf8();
    case arrow.Type.Bool: return new arrow.Bool();
    case arrow.Type.Int: return new arrow.Int(t.isSigned, t.bitWidth);
    case arrow.Type.Float: return new arrow.Float(t.precision);
    case arrow.Type.FixedSizeList:
      return new arrow.FixedSizeList(
        t.listSize,
        new arrow.Field("item", nativeTypeFromForeign(t.children[0].type), true),
      );
    default:
      throw new TypeError(`createTable: unsupported column type (typeId ${t.typeId})`);
  }
}

// Build an arrow type from a descriptor value: a type-name string, or
// `{ vector: dim }` for a FixedSizeList<Float32, dim> column.
function nativeTypeFromSpec(spec: string | { vector: number }): arrow.DataType {
  if (spec && typeof spec === "object" && typeof spec.vector === "number") {
    return new arrow.FixedSizeList(spec.vector, new arrow.Field("item", new arrow.Float32(), true));
  }
  switch (String(spec).toLowerCase()) {
    case "utf8": case "string": return new arrow.Utf8();
    case "large_utf8": case "largeutf8": return new arrow.LargeUtf8();
    case "bool": case "boolean": return new arrow.Bool();
    case "int32": return new arrow.Int32();
    case "int64": return new arrow.Int64();
    case "float32": return new arrow.Float32();
    case "float64": case "double": return new arrow.Float64();
    default:
      throw new TypeError(`createTable: unknown column type ${JSON.stringify(spec)}`);
  }
}

// An apache-arrow `Schema`, a plain `{ column: type }` descriptor, or raw
// IPC bytes -> the IPC the addon's createTable wants (an empty table that
// carries just the schema). Types are rebuilt in OUR arrow instance.
function schemaToIpc(schema: any): Buffer {
  if (Buffer.isBuffer(schema)) return schema;
  if (schema instanceof Uint8Array) return Buffer.from(schema);
  let fields: arrow.Field[];
  if (schema && Array.isArray(schema.fields)) {
    fields = schema.fields.map(
      (f: any) => new arrow.Field(f.name, nativeTypeFromForeign(f.type), f.nullable),
    );
  } else if (schema && typeof schema === "object") {
    fields = Object.entries(schema as SchemaDescriptor).map(
      ([name, spec]) => new arrow.Field(name, nativeTypeFromSpec(spec), false),
    );
  } else {
    throw new TypeError(
      "createTable: schema must be an apache-arrow Schema or a { column: type } descriptor",
    );
  }
  const nativeSchema = new arrow.Schema(fields);
  const children = fields.map((f) => arrow.makeData({ type: f.type, length: 0 }));
  const structData = arrow.makeData({
    type: new arrow.Struct(fields),
    length: 0,
    nullCount: 0,
    children,
  });
  const empty = new arrow.Table(new arrow.RecordBatch(nativeSchema, structData));
  return Buffer.from(arrow.tableToIPC(empty, STREAM));
}

// Build one typed Arrow column from row objects. `vectorFromArray` handles
// scalars; FixedSizeList<Float32> (vector columns) need the nested Data
// built by hand. The schema here is ours (from the addon), so its types
// are same-instance.
function buildColumn(field: arrow.Field, rows: RowRecord[]): arrow.Vector {
  const values = rows.map((r) => r[field.name]);
  const t = field.type as any;
  if (t && typeof t.listSize === "number") {
    const flat = Float32Array.from((values as number[][]).flat());
    const child = arrow.makeData({ type: t.children[0].type, length: flat.length, data: flat });
    const data = arrow.makeData({ type: t, length: rows.length, nullCount: 0, child });
    return arrow.makeVector(data);
  }
  return arrow.vectorFromArray(values, field.type);
}

// Normalize append input -> IPC bytes. An array of objects, or an
// apache-arrow Table / RecordBatch (normalized to rows via its own
// `toArray()`/`toJSON()`); either way the columns are rebuilt in our arrow
// instance from the declared schema. (We can't feed the consumer's Table
// straight into our `tableToIPC` — a different module instance isn't
// recognized.)
function dataToIpc(data: AppendData, getSchema: () => arrow.Schema): Buffer {
  if (Buffer.isBuffer(data)) return data;
  if (data instanceof Uint8Array) return Buffer.from(data);

  let rows: RowRecord[];
  const d = data as any;
  if (Array.isArray(data)) {
    rows = data as RowRecord[];
  } else if (d && (Array.isArray(d.batches) || (d.schema && typeof d.numRows === "number"))) {
    rows = Array.from(d).map((r: any) => r.toJSON() as RowRecord);
  } else {
    throw new TypeError(
      "append: expected an array of objects, an apache-arrow Table / RecordBatch, or an Arrow IPC Buffer",
    );
  }

  const schema = getSchema();
  const cols: Record<string, arrow.Vector> = {};
  for (const field of schema.fields) cols[field.name] = buildColumn(field, rows);
  return Buffer.from(arrow.tableToIPC(new arrow.Table(cols), STREAM));
}

// A Decimal128 value renders as a 4×u32 little-endian array in records.
// Convert (scale-0 -> integer) to a `bigint`, matching token/exact match.
function decimalToBigInt(words: Uint32Array): bigint {
  let v = 0n;
  for (let i = words.length - 1; i >= 0; i--) v = (v << 32n) | BigInt(words[i] >>> 0);
  const bits = BigInt(words.length * 32);
  if (v >= 1n << (bits - 1n)) v -= 1n << bits; // two's-complement sign
  return v;
}

// IPC result bytes -> records (default) or an apache-arrow Table. In record
// form, scale-0 Decimal columns (notably `_id`) become `bigint`.
function decode(buf: Buffer, asArrow?: boolean): RowRecord[] | arrow.Table {
  const table = arrow.tableFromIPC(buf);
  if (asArrow) return table;
  const intCols = table.schema.fields
    .filter((f) => (f.type as any).typeId === arrow.Type.Decimal && (f.type as any).scale === 0)
    .map((f) => f.name);
  return table.toArray().map((row: any) => {
    const obj = row.toJSON() as RowRecord;
    for (const name of intCols) {
      const cell = obj[name];
      if (cell != null && typeof cell !== "bigint") obj[name] = decimalToBigInt(cell as Uint32Array);
    }
    return obj;
  });
}

// --- friendly handles ---

export class Table {
  private inner: any;
  constructor(inner: any) {
    this.inner = inner;
  }

  /** The table's Arrow schema. */
  schema(): arrow.Schema {
    return arrow.tableFromIPC(this.inner.schema()).schema;
  }

  /**
   * Append rows. Accepts an array of objects, an apache-arrow
   * Table/RecordBatch, or raw Arrow IPC bytes. Durable on return; one
   * append == one commit.
   */
  append(data: AppendData): void {
    this.inner.append(dataToIpc(data, () => this.schema()));
  }

  /** Ranked BM25 search; rows as records (or an Arrow `Table`). */
  bm25Search(column: string, query: string, k: number, opts: Bm25SearchOptions & { arrow: true }): arrow.Table;
  bm25Search(column: string, query: string, k: number, opts?: Bm25SearchOptions): RowRecord[];
  bm25Search(column: string, query: string, k: number, opts: Bm25SearchOptions = {}): RowRecord[] | arrow.Table {
    const buf = this.inner.bm25Search(column, query, k, opts.mode, opts.projection);
    return decode(buf, opts.arrow);
  }

  /** Vector kNN; rows as records (or an Arrow `Table`). */
  vectorSearch(column: string, query: number[] | Float32Array, k: number, opts: VectorSearchOptions & { arrow: true }): arrow.Table;
  vectorSearch(column: string, query: number[] | Float32Array, k: number, opts?: VectorSearchOptions): RowRecord[];
  vectorSearch(column: string, query: number[] | Float32Array, k: number, opts: VectorSearchOptions = {}): RowRecord[] | arrow.Table {
    const q = query instanceof Float32Array ? query : Float32Array.from(query);
    const buf = this.inner.vectorSearch(column, q, k, opts.nprobe, opts.rerankMult, opts.projection, opts.filter);
    return decode(buf, opts.arrow);
  }

  /** Hybrid BM25 + vector search, fused with reciprocal-rank fusion; rows as
   * records (or an Arrow `Table`). `score` is the fused RRF score (higher is
   * better). */
  hybridSearch(textColumn: string, textQuery: string, vectorColumn: string, vectorQuery: number[] | Float32Array, k: number, opts: HybridSearchOptions & { arrow: true }): arrow.Table;
  hybridSearch(textColumn: string, textQuery: string, vectorColumn: string, vectorQuery: number[] | Float32Array, k: number, opts?: HybridSearchOptions): RowRecord[];
  hybridSearch(textColumn: string, textQuery: string, vectorColumn: string, vectorQuery: number[] | Float32Array, k: number, opts: HybridSearchOptions = {}): RowRecord[] | arrow.Table {
    const q = vectorQuery instanceof Float32Array ? vectorQuery : Float32Array.from(vectorQuery);
    const buf = this.inner.hybridSearch(textColumn, textQuery, vectorColumn, q, k, opts.mode, opts.nprobe, opts.projection);
    return decode(buf, opts.arrow);
  }

  /** Unranked token match; matching rows as records (or an Arrow `Table`). */
  tokenMatch(column: string, query: string, opts: TokenMatchOptions & { arrow: true }): arrow.Table;
  tokenMatch(column: string, query: string, opts?: TokenMatchOptions): RowRecord[];
  tokenMatch(column: string, query: string, opts: TokenMatchOptions = {}): RowRecord[] | arrow.Table {
    const buf = this.inner.tokenMatch(column, query, opts.mode, opts.projection);
    return decode(buf, opts.arrow);
  }

  /** Unranked exact match; matching rows as records (or an Arrow `Table`). */
  exactMatch(column: string, value: string, opts: MatchOptions & { arrow: true }): arrow.Table;
  exactMatch(column: string, value: string, opts?: MatchOptions): RowRecord[];
  exactMatch(column: string, value: string, opts: MatchOptions = {}): RowRecord[] | arrow.Table {
    const buf = this.inner.exactMatch(column, value, opts.projection);
    return decode(buf, opts.arrow);
  }

  /** Count rows matching a BM25 keyword `query` over `column`, without
   * fetching them. `mode` is `"or"` (default) or `"and"`. */
  count(column: string, query: string, opts: CountOptions = {}): number {
    return this.inner.count(column, query, opts.mode);
  }

  /** Replace rows matching a SQL predicate (e.g. `"status = 'spam'"`) with
   * `data` (same shapes as `append`), 1:1 — the matched count must equal the
   * replacement-row count. Requires durable storage (not `memory://`). */
  update(predicate: string, data: AppendData): MutationStats {
    return this.inner.update(predicate, dataToIpc(data, () => this.schema()));
  }

  /** Delete rows matching a SQL predicate (e.g. `"status = 'spam'"`).
   * Requires durable storage (not `memory://`). */
  delete(predicate: string): MutationStats {
    return this.inner.delete(predicate);
  }

  /** Merge small / underfilled superfiles into larger ones (omit `settings`
   * for engine defaults). */
  optimize(settings?: OptimizeOptions): void {
    this.inner.optimize(settings);
  }

  /** Delete orphaned storage objects left by compaction or interrupted writes.
   * Only objects older than `graceSecs` (a safety window against racing
   * readers/writers) are removed. Requires durable storage (not `memory://`). */
  gc(graceSecs: number): GcReport {
    return this.inner.gc(graceSecs);
  }
}

export class Connection {
  private inner: any;
  constructor(inner: any) {
    this.inner = inner;
  }

  /** Create a table from an apache-arrow `Schema` or `{ column: type }`. */
  createTable(name: string, schema: arrow.Schema | SchemaDescriptor | Buffer, indexes: IndexSpec): Table {
    return new Table(this.inner.createTable(name, schemaToIpc(schema), indexes));
  }

  openTable(name: string): Table {
    return new Table(this.inner.openTable(name));
  }

  dropTable(name: string, purge?: boolean): void {
    this.inner.dropTable(name, purge);
  }

  listTables(): string[] {
    return this.inner.listTables();
  }

  /** SQL across the catalog; rows as records (or an Arrow `Table`). */
  querySql(sql: string, opts: QueryOptions & { arrow: true }): arrow.Table;
  querySql(sql: string, opts?: QueryOptions): RowRecord[];
  querySql(sql: string, opts: QueryOptions = {}): RowRecord[] | arrow.Table {
    return decode(this.inner.querySql(sql), opts.arrow);
  }
}

/** Open (or create) a catalog rooted at `uri`. */
export function connect(uri: string, options?: ConnectOptions): Connection {
  return new Connection(nativeConnect(uri, options as any));
}
