// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//
// Types for the friendly Node-idiomatic API (infino.js). The low-level
// addon types are generated in index.d.ts; this is the surface users
// program against.

import type { Schema, Table as ArrowTable, RecordBatch } from "apache-arrow";

/** Vector distance metric. */
export type Metric = "cosine" | "l2sq" | "negdot";

/** Boolean mode for multi-term FTS queries. */
export type BoolMode = "or" | "and";

/** A row from a query/search when not materializing to Arrow. */
export type RowRecord = Record<string, unknown>;

/** Accepted shapes for `Table.append`. */
export type AppendData =
  | RowRecord[]
  | ArrowTable
  | RecordBatch
  | Buffer
  | Uint8Array;

/** An unranked match — the public `_id` (as a `bigint`) plus a score. */
export interface SearchHit {
  id: bigint;
  score: number;
}

/** Storage config the `connect` URI can't carry (S3-compatible creds). */
export interface ConnectOptions {
  endpoint?: string;
  region?: string;
  accessKey?: string;
  secretKey?: string;
}

export interface Bm25SearchOptions {
  /** `"or"` (default) or `"and"`. */
  mode?: BoolMode;
  /** Include the table's scalar columns (default: true for BM25). */
  materialize?: boolean;
  /** Return an apache-arrow `Table` instead of plain records. */
  arrow?: boolean;
}

export interface VectorSearchOptions {
  nprobe?: number;
  /** Include the table's scalar columns (default: false for vector). */
  materialize?: boolean;
  /** Return an apache-arrow `Table` instead of plain records. */
  arrow?: boolean;
}

export interface QueryOptions {
  /** Return an apache-arrow `Table` instead of plain records. */
  arrow?: boolean;
}

/**
 * Declares which columns are full-text (BM25) and which are vector (IVF
 * kNN) indexed. Built fluently:
 * `new IndexSpec().fts("body").vector("emb", 384, 256, "cosine")`.
 */
export class IndexSpec {
  constructor();
  fts(column: string): IndexSpec;
  vector(column: string, dim: number, nCent: number, metric: Metric): IndexSpec;
}

/** A single-table handle. */
export class Table {
  /** The table's Arrow schema. */
  schema(): Schema;

  /**
   * Append rows. Accepts an array of objects, an apache-arrow
   * `Table`/`RecordBatch`, or raw Arrow IPC bytes. Durable on return;
   * one append == one commit.
   */
  append(data: AppendData): void;

  /** Ranked BM25 search; matching rows as records (or an Arrow `Table`). */
  bm25Search(column: string, query: string, k: number, opts: Bm25SearchOptions & { arrow: true }): ArrowTable;
  bm25Search(column: string, query: string, k: number, opts?: Bm25SearchOptions): RowRecord[];

  /** Vector kNN; matching rows as records (or an Arrow `Table`). */
  vectorSearch(column: string, query: number[] | Float32Array, k: number, opts: VectorSearchOptions & { arrow: true }): ArrowTable;
  vectorSearch(column: string, query: number[] | Float32Array, k: number, opts?: VectorSearchOptions): RowRecord[];

  /** Unranked token match — `_id` + score list (`score` is `0`). */
  tokenMatch(column: string, query: string, mode?: BoolMode): SearchHit[];

  /** Unranked exact match — `_id` + score list (`score` is `0`). */
  exactMatch(column: string, value: string): SearchHit[];
}

/** A catalog connection. */
export class Connection {
  /** Create a table from an apache-arrow `Schema` and an `IndexSpec`. */
  createTable(name: string, schema: Schema, indexes: IndexSpec): Table;
  openTable(name: string): Table;
  dropTable(name: string): void;
  listTables(): string[];

  /** SQL across the catalog; rows as records (or an Arrow `Table`). */
  querySql(sql: string, opts: QueryOptions & { arrow: true }): ArrowTable;
  querySql(sql: string, opts?: QueryOptions): RowRecord[];
}

/** Open (or create) a catalog rooted at `uri`. */
export function connect(uri: string, options?: ConnectOptions): Connection;
