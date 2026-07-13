# On-disk layout

The files infino writes under a database root, and how a read walks from the root down to the [superfile](./superfile.md) that holds the data. The single-file superfile format is described in [superfile](./superfile.md); the table model and commit protocol in [supertable](./supertable.md). This document is the physical view: what is on disk, and which file points to which.

Everything on disk is immutable except the two `current` pointer files. A commit writes new immutable files first, then atomically renames one pointer to publish them, so a reader sees either the state before a commit or the state after it, never a partial one.

## Directory tree

Legend: `dir/` is a directory, a bare name is a file. `stores` is its main contents, `used` is what reads it.

```text
mydb/                                     database root (the connect() path)
│
├── _catalog/
│   └── current                           JSON · the catalog
│       ├ stores : catalog_id; tables{ name -> { location, schema_ipc,
│       │          fts[], vectors[], created_at_unix } }
│       └ used   : read on connect() to resolve a table name to its subtree
│
├── table1-18bf675d94bf6360-0/            one table (name "table1")
│   │                                     dir = {name}-{nanos:x}-{seq:x}; a re-created
│   │                                     table lands on a fresh subtree (drop-safe)
│   │
│   ├── _supertable/
│   │   └── current                       text KV · the live-version pointer
│   │       ├ stores : manifest_id, manifest_uri, content_hash=blake3:...
│   │       └ used   : the only file ever atomically renamed = the commit barrier
│   │
│   ├── manifest/
│   │   └── manifest-000004.json          JSON · immutable manifest version
│   │       ├ stores : parts[] (each: uri + content_hash), schema, fts/vector
│   │       │          column specs, partition strategy
│   │       └ used   : one written per commit (NNNNNN bumps); names the live parts
│   │
│   ├── manifest-parts/
│   │   └── part-<blake3>.avro.zst        Avro+zstd · immutable, content-addressed
│   │       ├ stores : superfiles[] entries, each with uri, n_docs, id_min/max,
│   │       │          bloom + min/max summaries
│   │       └ used   : skip-pruning; reject superfiles before reading their bytes
│   │
│   ├── data/
│   │   └── <superfile>.sf.parquet        the superfiles (files prefixed seg-)
│   │       ├ stores : valid Parquet columns + embedded BM25 + vector index,
│   │       │          all in one file
│   │       └ used   : the actual data + search; immutable, never rewritten
│   │
│   ├── superfiles/                       not superfiles: tombstone sidecars only
│   │   └── <superfile_id>.tombstones     binary · magic "INFTOMB\0"
│   │       ├ stores : a RoaringBitmap of deleted row positions in the
│   │       │          superfile of that id (+ optional compaction seal record)
│   │       └ used   : update/delete tombstones rows instead of rewriting a superfile
│   │
│   └── wal/
│       └── mutations/                    write-ahead log for update/delete
│           ├── <walid>.json              state-doc: the pending mutation + its state
│           └── <walid>.arrow             IPC sidecar with the new rows (UPDATE only)
│               ├ stores : one entry per in-flight mutation (walid is a time-ordered id)
│               └ used   : crash recovery; drained after commit, so empty at rest
│
└── table2-18bf9efdf7479d90-0/           second table, identical shape
    └── ...                               (a table with no vector column has vectors[]
                                          empty in the catalog and no vector index)
```

## Logical view

Who points to whom, and how many. The catalog lists many tables; each table pins exactly one live manifest; that manifest lists many parts; each part carries many superfile entries; each entry names exactly one superfile. table2 has the same shape and is collapsed here to save room.

```text
                        ┌─────────────────────┐
                        │  _catalog/current   │
                        └──────────┬──────────┘
                                   │ tables[name].location
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
      ┌──────────────────┐                    ┌──────────────────┐
      │ table1/          │                    │ table2/          │
      │  _supertable/    │                    │  _supertable/    │
      │  current         │                    │  current         │
      └────────┬─────────┘                    └────────┬─────────┘
               |                                       |
               │ manifest_uri                          │ manifest_uri
               ▼                                       ▼
   ┌───────────────────────┐                ┌───────────────────────┐
   │ manifest-000004.json  │                │ manifest-000001.json  │
   └───────────┬───────────┘                └───────────────────────┘
               │ parts[].uri                    (same shape below)
               │
               |
               ├──▶ ┌──────────────────┐  superfiles[].uri
               │    │ manifest part 1  │─ entry ━━━━━▶ seg-A.sf.parquet ╌╌╌╌╌▶ A.tombstones
               │    │  (avro.zst)      │─ entry ━━━━━▶ seg-B.sf.parquet ╌╌╌╌╌▶ B.tombstones
               │    └──────────────────┘
               │
               └──▶ ┌──────────────────┐
                    │ manifest part 2  │─ entry ━━━━━▶ seg-C.sf.parquet ╌╌╌╌╌▶ C.tombstones
                    │  (avro.zst)      │─ entry ━━━━━▶ seg-D.sf.parquet ╌╌╌╌╌▶ (none yet, as nothing deleted/updated)
                    └──────────────────┘

   Legend
     seg-X            = data/seg-<id>.sf.parquet   (the superfile: Parquet + BM25 + vector)
     X.tombstones     = superfiles/<id>.tombstones (shares the superfile's id)

     ━━━━━▶   stored pointer: the labelled field names the next file
     ╌╌╌╌╌▶   NOT a stored pointer: derived from the superfile id; the
              tombstone file appears only after a row in it is deleted
```

Every hop below the two pointer files is content-addressed with a blake3 hash, so identical bytes always resolve to the same uri. A commit that does not touch a part reuses its existing entries by uri, so successive manifests can point at the same superfile bytes.

## Notes

- `data/` holds the superfiles; `superfiles/` holds only their tombstones. The directory name is misleading.
- The live set is whatever the current manifest names. An update writes a new superfile and tombstones the old one, so `data/` accumulates more `.sf.parquet` files than the table's live row count until GC removes the dead ones.
- `wal/mutations/` is normally empty. A `<walid>.json` (and, for UPDATE, its `.arrow` sidecar) exists only while a mutation is in flight or was interrupted; the next recovery sweep drains it.

## Source references

Anchored by symbol so they survive line moves:

| On disk | Defined by | File |
| --- | --- | --- |
| table subtree name `{name}-{nanos:x}-{seq:x}` | `unique_location` | [src/catalog/mod.rs](../../src/catalog/mod.rs) |
| `_supertable/current` pointer text format + atomic-rename commit | `PointerFile`, `POINTER_PATH`, `commit_manifest` | [src/supertable/manifest/commit.rs](../../src/supertable/manifest/commit.rs) |
| `manifest/manifest-NNNNNN.json` and `manifest-parts/part-<hash>.avro.zst` names | `MANIFEST_DIR`, `MANIFEST_PARTS_DIR`, `manifest_uri` | [src/supertable/manifest/commit.rs](../../src/supertable/manifest/commit.rs) |
| `data/seg-<uuid>.sf.parquet` name | `SuperfileUri::storage_path` | [src/supertable/manifest/mod.rs](../../src/supertable/manifest/mod.rs) |
| manifest-part Avro schema (`superfiles[].uri`, summaries) | `SuperfileEntry` schema | [src/supertable/manifest/part.rs](../../src/supertable/manifest/part.rs) |
| `wal/mutations/` state-doc + `.arrow` sidecar | `WAL_DIR`, `WalStore` | [src/supertable/wal/persistence.rs](../../src/supertable/wal/persistence.rs) |
| `superfiles/<id>.tombstones` sidecar paths | `SUPERFILES_DIR`, `tombstones_path` | [src/supertable/wal/persistence.rs](../../src/supertable/wal/persistence.rs) |
| tombstone binary format (`INFTOMB\0` + RoaringBitmap) | `MAGIC`, layout doc | [src/supertable/wal/tombstones_codec.rs](../../src/supertable/wal/tombstones_codec.rs) |
