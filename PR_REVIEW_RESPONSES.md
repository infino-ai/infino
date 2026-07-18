# PR #3 code-review responses

This document records the disposition of every finding against
`global-vector`. A local code observation is not treated as a product bug unless
the complete production path can reach it. Infino has not shipped this vector
architecture, so compatibility with superseded branch-local layouts and
configuration behavior is not a requirement.

## Functional findings

### F1 — compaction crossing the drain watermark

**Response:** Valid correctness bug.

A compacted user superfile inherits the minimum input `birth_version`, while
drain skips birth versions already covered by `drained_ranges`. Mixing drained
and undrained inputs could therefore hide the undrained rows permanently.

**Action:** Fixed by `796faa3`. User compaction selects jobs independently on
the drained and undrained sides. The focused unit regression proves unsplit
selection mixes the fixture and partitioned selection cannot.

### F2 — repeated no-op cell splits

**Response:** The loop-progress observation is valid, but the stated
delete-heavy production scenario is not.

Cell splitting runs only on the hidden table. User deletes update the hidden
resident deleted-id set; they do not create hidden tombstones. Consequently,
normal user deletes do not produce the claimed fully-tombstoned hidden cell.

**Action:** `beefc1c` is retained as defensive progress protection. Source
comments now describe it as defensive rather than a confirmed delete-path
blocker.

### F3 — unchecked subsection offset subtraction

**Response:** Valid corruption-handling bug.

CRC-disabled lazy opens parse untrusted offsets. The old code used raw
subtraction after an incomplete guard and could panic on inverted offsets.

**Action:** Fixed by `0e92814` with checked subtraction and explicit malformed
layout errors. New CRC-disabled tests cover both inverted offset cases.

### F4 — missing v2 outer CRC verification

**Response:** Valid integrity gap.

The v2 writer emits an outer CRC, but the reader did not verify it. The outer
`n_docs` field was also not covered by the directory CRC.

**Action:** Fixed by `0e92814`: always cross-check outer `n_docs` against cell
subsections and verify the outer CRC when CRC verification is enabled. Lazy
object-store opens intentionally disable full CRC scans, so this adds no lazy
query GETs. Four new F3/F4 corruption tests pass. Eager-open checksum CPU should
still be reported in benchmark results.

### F5 — hidden deleted-id set growth

**Response:** This is an intentional delete-design trade-off, not a correctness
or query-I/O bug. The suggested pruning fix is invalid for the current design.

The delete path is:

1. Tombstone the user-table row.
2. Leave the immutable hidden row physically present.
3. Store the deleted stable id inline in the hidden manifest.
4. Filter ranked candidates by binary search over the resident set.

The set causes no per-query object-store GET and is not scanned in full per
candidate. It is decoded once per manifest version and cached; candidate checks
are `O(log deleted_ids)`.

Drain and hidden compaction do not currently remove rows named only by this
set. Pruning their ids afterward would allow those physical rows to reappear.

**Action:** No pruning change. Treat full-set manifest rewrite as a known
write-amplification/storage trade-off of the chosen zero-GET query design.

### F6 — repeated delete refill fan-out

**Response:** The control flow repeats fan-out, but the review's repeated-GET
claim does not hold on the default cached path.

Lazy readers sit on `BlockCachedSource`. The first cold probe writes the
selected cluster ranges into the sparse block cache; a larger refill probes the
same routed clusters from local blocks rather than issuing the same object-store
GETs again. `RangeOnly` can re-fetch, but it is the explicit no-cache fallback.

The proposed one-shot `k + deleted_ids.len()` request is unsafe because the
resident set grows with lifetime deletes and could create an enormous query.
Filtering every scored candidate by stable id would also add substantially more
query CPU than the intentional post-top-k design.

**Action:** Restore and retain the adaptive refill loop. No query algorithm
change.

### F7 — segmented NegDot norms length

**Response:** Valid latent decoder defect.

Segmented NegDot encoding omits per-document norms, while the old decoder added
`n_docs * 4` to every segment length.

**Action:** Fixed in `51ed0d7`. Decoder length uses the already-stored metric.
No new on-disk `has_norms` bit is required. NegDot regression test passes.

### F8 — NaN ordering and non-finite codec metadata

**Response:** Partially valid corruption hardening.

`partial_cmp(...).unwrap_or(Equal)` and `total_cmp` differ for NaN. Normal
builders emit finite metadata; non-finite values require malformed or corrupt
input.

**Action:** Fixed in `51ed0d7`. Use `total_cmp` consistently and validate
finite scale/offset values in the shared quantizer-metadata validator used by
eager and normal lazy opens. A CRC-disabled malformed-metadata regression test
covers the path.

### F9 — unchecked stable-id index in IVF splice

**Response:** Valid latent corruption panic.

`src_local` is decoded from stored bytes and must be checked before indexing
the stable-id slice.

**Action:** Fixed in `51ed0d7`. One shared bounds-checked helper returns
`VectorSchemaMismatch`. Malformed-input regression test covers the path.

### F10 — hidden deleted-id wire ordering

**Response:** No current production bug; useful future invariant hardening.

The sole production caller already sorts and deduplicates. The original fix
unnecessarily copied the complete already-canonical set on every delete and
claimed deduplication without enforcing it.

**Action:** Corrected in place: borrow canonical input without copying; only
allocate, sort, and deduplicate noncanonical input. Wire-order tests pass.

## Efficiency and cleanup findings

### A1 — synchronous summary methods on nonresident cells

**Response:** Local behavior is real but unreachable in current production
callers, which build these summaries from resident readers. A synchronous
method should not initiate asynchronous object-store I/O.

**Action:** No change.

### A2 — unkeyed v1 inline-id stash

**Response:** No demonstrated bug. All vector columns in one superfile share
the same local row space and stable-id sequence. There is no product
compatibility requirement for superseded branch-local layouts.

**Action:** No change.

### A3 — cloned split payload

**Response:** Valid scale inefficiency.

The split path cloned every encoded row and then cloned rows again into split
shards.

**Action:** Fixed in `51ed0d7`. Pass references through the existing split
planner and medoid helper. Focused split tests pass.

### A4 — repeated split recount and packed-entry decode

**Response:** Valid efficiency observation.

The keep-cell republish path decoded the same packed entry once per retained
cell. The outer split loop also recomputes physical counts after each committed
split.

**Action:** Fixed in `51ed0d7`. Decode each packed entry once for all retained
cells, compute the physical count table once, and apply each committed split's
returned two-cell count delta in memory instead of reopening every file for
another complete recount.

### A5 — background cache-fill behavior

**Response:** Three separate efficiency observations:

- A paused fill previously restarted from byte zero.
- `FOREGROUND_QUERIES` is process-wide.
- Cache-root restoration runs synchronously.

These are not one correctness defect. Process-wide quiescence is a scheduling
policy, and asynchronous restore changes open semantics. Separately, vector
queries were spawning full-object fill and competing with cold query GETs.

**Action:** Fixed across `51ed0d7` and `9521716`:

- Preserve completed chunks across pauses (resume cursor).
- Pause fill only when the same URI's lazy reader is held; unrelated fills
  continue.
- Vector opens do not start background fill (block-cache retention only);
  FTS/SQL opens may start fill and skip the vector blob range.
- Fill GETs run under `io_counters::scope_background` so cold-query meters
  stay foreground-only.
- Cache-root restore remains available; focused tests cover resume, same-URI
  pause, unrelated-URI progress, vector-skip-fill, and restored-cache reuse.

### A6 — duplicated residual-norm formula

**Response:** Valid cleanup and drift prevention.

The committed implementation routes norm recomputation, fresh encoding, and
transcode through the same `sq8_residual_norm_sq` kernel, preventing the three
formulas from diverging.

**Action:** Keep `b222da1` unchanged, including the relative tolerance that
fixes the demonstrated cross-architecture CI flake.

## Legacy and configuration findings

### L1 — `_id` plus scalar fallback

**Response:** Not applicable. There is no shipped legacy product contract. The
current writer owns the invariant that ranked packed hits carry stable ids.

**Action:** No compatibility fallback.

### L2 — unresolved gapped v1 hits

**Response:** Not applicable. This concerns a superseded branch-local plain-v1
layout, not a supported external format.

**Action:** No legacy fixture or fallback.

### L3 — missing/empty vector summary fallback

**Response:** No legacy fallback requirement. Current builders must either
produce valid routing summaries or reject unsupported input; query should fail
loud on malformed current output.

**Action:** No blind-probe fallback.

### L4 — environment overrides removed

**Response:** Intentional. Configuration is YAML-only, and there is no shipped
deployment migration to preserve.

**Action:** No rollback and no migration document.

## Refuted findings

### X1 — filtered search returns deleted rows

Refuted. Filtered user candidates subtract user-table tombstones before
routing. **No action.**

### X2 — inline region and scalar `_id` mismatch

Refuted. MultiCell Parquet rows and inline stable-id regions use the same cell
directory order. **No action.**

### X3 — vector codec verification breaks FTS/filtered paths

Refuted. Configured superfiles carry the vector blob under immutable table
options. **No action.**

### X4 — watermark interval prefix assumption

Not an independent issue. Mixed-birth compaction was the upstream violation;
F1 prevents it. **No separate action.**

### X5 — reused `_id` remains invisible

Refuted. `_id` values are monotonic Snowflake ids and are not re-minted.
**No action beyond the acknowledged F5 cost trade-off.**

## Verification status

- Library and test targets type-check.
- Four new v2 F3/F4 corruption tests pass.
- Corrected hidden deleted-id wire tests pass.
- Focused F7, F8, F9, A3, A4, and A5 cache-behavior tests pass.
- Accepted second-review fixes that landed in `51ed0d7` (*vector: finish
  review hardening and cache fixes*): S2/S3/S14/S16–S23, A3/A4, and the
  F7–F9 / cache pieces named above.
- A5 modality gate + fill metering landed in `9521716` (*supertable/cache:
  modality-gate background fill; meter fill I/O*); needless-borrow nit in
  `724e39d`.
- S1/S4/S9–S12 atomicity group fixed in code after `724e39d` (see disposition
  below); `make check` / lib clippy clean on the change.
- 1M@64 Azure vector gate after `9521716`: cold GETs 16/4/5/5, repeats 0,
  fill 0, recall unchanged vs baseline.

## Second-review disposition

Earlier drafts incorrectly listed S1/S4/S9–S12 as fixed in `51ed0d7`; that
commit's `writer.rs` work is A3/A4 (per-cell packed decode + split count
bookkeeping) plus related hardening. The atomicity group is fixed in the
working tree after `724e39d` (see below).

### Fixed in this pass (atomicity group; was incorrectly listed under `51ed0d7`)

- **S1:** Drain passes grid + drained watermark as `CommitListMetadata` into
  `persist_commit_async`, applied on every OCC attempt with membership — no
  pre-`store`.
- **S4:** Cell split likewise publishes the updated grid only via
  `CommitListMetadata` in the same OCC attempt as replacement membership.
- **S9:** `commit_appends_internal` restores the taken append buffer and byte
  counters on any flush failure so the writer can retry.
- **S10:** First-commit global-grid bootstrap is local `pending_gvi` stamped
  through `CommitListMetadata` on the membership commit — not a bare
  `ArcSwap` store ahead of CAS.
- **S11:** The zero-replica-budget fast path dedups by `stable_id` before
  spilling; only re-assign is skipped.
- **S12:** In-memory `store.insert` is deferred until after durable (or local)
  membership publish succeeds.

### Fixed (committed in `51ed0d7` unless noted)

- **S2:** Multi-cell lazy open threads the caller's `OpenOptions`; 
  `verify_crc=true` fetches the full blob so outer/subsection CRC checks run.
- **S3:** Deleted-id decode rejects noncanonical wire order in release builds
  (`HiddenDeletedError::NonCanonical`); encode still sorts+dedups.
- **S14:** Packed cells are sorted before both Parquet ids and vector
  subsections are emitted.
- **S16:** Open-range capture rejects unknown vector versions.
- **S17:** Hidden membership commits (`try_commit_attempt`) write the
  content-addressed slow-state blob and stamp it via
  `with_slow_vector_state_ref` on the same successor before the list/pointer
  CAS (closes the post-`update` clear window).
- **S18:** A concurrent-create loser adopts the winner's manifest and
  reopens the hidden index at the winner's stamped prefix (drops the loser's
  process-local UUID prefix).
- **S19/F8:** Lazy Sq8 metadata validation is a real release error on the
  cold parse path and on per-cluster on-demand metadata.
- **S20:** Hidden bootstrap-create failures are recorded as Broken rather than
  silently treated as Absent.
- **S21:** Vector directory geometry uses checked multiplication/addition on
  eager and lazy paths.
- **S22:** Cluster-centroid wire lengths use checked arithmetic and release
  length validation.
- **S23:** A non-empty source cluster mapping to zero destination cells is an
  error.
- **A3/A4:** Split path stops cloning encoded rows; packed keep-cells decode
  once per entry; physical counts update from the committed split delta
  instead of a full recount (`51ed0d7`).
- **A5 (follow-up in `9521716`, clippy nit in `724e39d`):** Vector opens skip
  background fill; FTS/SQL may fill non-vector bytes; fill I/O is metered
  separately.

### Deferred or ratified

- **S5:** No checkpoint fix. `splice` is vestigial and non-default; remove the
  mode in a separate cleanup PR.
- **S6:** Lazy foreground is the intentional pre-product default.
- **S7:** Bounded PUT fan-out is intentional; benchmark before changing it.
- **S8:** No current bug while user vectors are retained; route hybrid through
  hidden before any reclaim work.
- **S13:** Watermark-side fragmentation is the accepted cost of F1 correctness.
- **S15:** Already superseded by canonical no-copy encode behavior.
