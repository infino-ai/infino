// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Concurrency stress for the shared query fan-out
//! (`supertable::query::dispatch::fanout`) — the tokio-I/O /
//! rayon-CPU orchestration that backs `bm25_search`, `vector_search`,
//! and the SQL TVFs.
//!
//! ## The hazard this pins
//!
//! Every public search call is **sync**: it `block_on`s the
//! supertable's shared query runtime. Underneath, `fanout` spawns one
//! tokio task per superfile for the I/O wave, and each per-superfile kernel
//! offloads its scoring onto the global **rayon** pool. Driving that
//! from many OS threads at once is the classic deadlock surface:
//!
//!   * many concurrent `block_on`s contending for a bounded-worker
//!     runtime,
//!   * tokio tasks awaiting results whose CPU is on rayon, while rayon
//!     workers are themselves saturated by other queries' fan-outs.
//!
//! If the runtime/pool split is wrong (e.g. CPU on the tokio workers,
//! or a `block_on` nested inside a rayon worker), this either deadlocks
//! or starves. The test fires a mix of BM25 / vector / hybrid-SQL
//! queries from `N_THREADS` threads in a tight loop and asserts:
//!
//!   1. **Liveness.** The whole burst finishes well inside a watchdog
//!      window — a hang (deadlock/starvation) trips the watchdog
//!      instead of wedging the suite forever.
//!   2. **Correctness under contention.** Every concurrent result is
//!      byte-identical to the single-threaded golden result; the
//!      shared snapshot + fan-out must not interleave or corrupt
//!      across threads.

#![deny(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use arrow_array::{
    ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};

use infino::superfile::builder::FtsConfig;
use infino::superfile::fts::reader::BoolMode;
use infino::supertable::query::SuperfileHit;
use infino::supertable::query::vector::VectorSearchOptions;
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::{default_tokenizer, default_vector_config};

const DIM: usize = 16;
const SUPERFILES: usize = 4;
const DOCS_PER_SUPERFILE: usize = 8;

const N_THREADS: usize = 16;
const ITERS_PER_THREAD: usize = 25;
/// Random-rotation seed for the fanout fixture's vector index.
const VECTOR_ROT_SEED: u64 = 7;
/// Rayon CPU pool size used to orchestrate cross-superfile fanout.
const RAYON_POOL_THREADS: usize = 4;
/// Number of distinct query kinds cycled in the stress loop.
const QUERY_KIND_COUNT: usize = 4;
/// Generous upper bound: the single-threaded golden takes well under a
/// second; if `N_THREADS × ITERS` worth of bursts can't finish in this
/// window the fan-out has deadlocked or is pathologically starved.
const WATCHDOG: Duration = Duration::from_secs(120);

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

fn options_title_emb() -> SupertableOptions {
    // Multi-thread pools on *both* sides so the test actually exercises
    // rayon parallelism against the tokio I/O fan-out, not a 1-thread
    // degenerate pool.
    let writer_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("writer pool"),
    );
    let reader_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("reader pool"),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(DIM), false),
    ]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
        Some(default_tokenizer()),
    )
    .expect("valid options")
    .with_writer_pool(writer_pool)
    .with_reader_pool(reader_pool)
}

/// Doc `i` within the batch gets a "rust …" title (so BM25 `rust`
/// fans out across every superfile) and a one-hot embedding at global
/// dim `base_dim + i`.
fn build_batch(base_dim: usize, schema: Arc<Schema>) -> RecordBatch {
    let titles: Vec<String> = (0..DOCS_PER_SUPERFILE)
        .map(|i| format!("rust topic {} variant", base_dim + i))
        .collect();
    let title_arr = LargeStringArray::from(titles.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let mut flat = Vec::<f32>::with_capacity(DOCS_PER_SUPERFILE * DIM);
    for i in 0..DOCS_PER_SUPERFILE {
        let active = (base_dim + i) % DIM;
        for d in 0..DIM {
            flat.push(if d == active { 1.0 } else { 0.0 });
        }
    }
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(schema, vec![Arc::new(title_arr), Arc::new(fsl)]).expect("batch")
}

fn build_supertable() -> Supertable {
    let st = Supertable::create(options_title_emb()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    for seg in 0..SUPERFILES {
        w.append(&build_batch(seg * DOCS_PER_SUPERFILE, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

fn csv_one_hot(active: usize) -> String {
    (0..DIM)
        .map(|d| if d == active { "1" } else { "0" })
        .collect::<Vec<_>>()
        .join(",")
}

/// Stable, order-independent identity for a hit list:
/// sorted `(superfile, local_doc_id)`. Superfile is rendered via its
/// `Debug` form so the test needs no extra type imports and is immune
/// to fan-out / merge ordering.
fn hit_key(hits: &[SuperfileHit]) -> Vec<(String, u32)> {
    let mut v: Vec<(String, u32)> = hits
        .iter()
        .map(|h| (format!("{:?}", h.superfile), h.local_doc_id))
        .collect();
    v.sort();
    v
}

fn id_vec(batches: &[RecordBatch]) -> Vec<i128> {
    let mut set: HashSet<i128> = HashSet::new();
    for b in batches {
        let idx = b.schema().index_of("_id").expect("_id column");
        let a = b
            .column(idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal128 _id");
        for i in 0..a.len() {
            set.insert(a.value(i));
        }
    }
    let mut v: Vec<i128> = set.into_iter().collect();
    v.sort_unstable();
    v
}

/// The four query shapes each worker re-runs; bundled so golden +
/// workers stay in lockstep.
struct Golden {
    bm25: Vec<(String, u32)>,
    vector: Vec<(String, u32)>,
    hybrid_ids: Vec<i128>,
    count: i64,
}

const K: usize = 10;

fn run_bm25(st: &Supertable) -> Vec<(String, u32)> {
    hit_key(
        &st.reader()
            .bm25_hits("title", "rust", K, BoolMode::Or)
            .expect("bm25"),
    )
}

fn run_vector(st: &Supertable) -> Vec<(String, u32)> {
    let mut q = [0.0f32; DIM];
    q[0] = 1.0;
    hit_key(
        &st.reader()
            .vector_hits("emb", &q, K, VectorSearchOptions::new(), None)
            .expect("vector"),
    )
}

fn run_hybrid(st: &Supertable) -> Vec<i128> {
    id_vec(
        &st.reader()
            .query_sql(&format!(
                "SELECT _id FROM hybrid_search('title', 'rust', 'emb', '{}', {K})",
                csv_one_hot(0)
            ))
            .expect("hybrid query_sql"),
    )
}

fn run_count(st: &Supertable) -> i64 {
    let batches = st
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("count query_sql");
    batches[0]
        .column_by_name("n")
        .expect("n column")
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("Int64 count")
        .value(0)
}

fn golden(st: &Supertable) -> Golden {
    Golden {
        bm25: run_bm25(st),
        vector: run_vector(st),
        hybrid_ids: run_hybrid(st),
        count: run_count(st),
    }
}

/// Single-threaded determinism gate for the floored fan-out: the
/// cross-segment threshold (`SharedTopK`) evolves with unit completion
/// order, which varies run to run — the *result* must not. This pins
/// the regression where a truncated fixed-point block-max let the
/// floor skip blocks holding score-tied hits, making the top-k depend
/// on completion order.
#[test]
fn repeated_bm25_is_deterministic_under_threshold_sharing() {
    let st = Arc::new(build_supertable());
    let golden = run_bm25(&st);
    for it in 0..200 {
        let got = run_bm25(&st);
        assert_eq!(got, golden, "bm25 result diverged on iteration {it}");
    }
}

#[test]
fn fanout_under_concurrency_is_live_and_deterministic() {
    let st = Arc::new(build_supertable());
    let gold = Arc::new(golden(&st));

    // Sanity: the golden is non-degenerate, otherwise "matches golden"
    // is a vacuous assertion.
    assert!(!gold.bm25.is_empty(), "bm25 golden must be non-empty");
    assert!(!gold.vector.is_empty(), "vector golden must be non-empty");
    assert!(
        !gold.hybrid_ids.is_empty(),
        "hybrid golden must be non-empty"
    );
    assert_eq!(gold.count, (SUPERFILES * DOCS_PER_SUPERFILE) as i64);

    // Run the whole stress on a coordinator thread and gate it behind a
    // watchdog so a deadlock trips the timeout instead of hanging the
    // test binary forever.
    let (tx, rx) = mpsc::channel::<Result<(), String>>();
    let coordinator = {
        let st = Arc::clone(&st);
        let gold = Arc::clone(&gold);
        thread::spawn(move || {
            let mut handles = Vec::with_capacity(N_THREADS);
            for t in 0..N_THREADS {
                let st = Arc::clone(&st);
                let gold = Arc::clone(&gold);
                handles.push(thread::spawn(move || -> Result<(), String> {
                    for it in 0..ITERS_PER_THREAD {
                        // Rotate the query order per (thread, iter) so
                        // the four shapes interleave differently across
                        // threads — maximizes runtime/pool contention.
                        match (t + it) % QUERY_KIND_COUNT {
                            0 => {
                                let got = run_bm25(&st);
                                if got != gold.bm25 {
                                    return Err(format!(
                                        "bm25 mismatch t={t} it={it}\n  gold: {:?}\n  got:  {:?}",
                                        gold.bm25, got
                                    ));
                                }
                            }
                            1 => {
                                if run_vector(&st) != gold.vector {
                                    return Err(format!("vector mismatch t={t} it={it}"));
                                }
                            }
                            2 => {
                                if run_hybrid(&st) != gold.hybrid_ids {
                                    return Err(format!("hybrid mismatch t={t} it={it}"));
                                }
                            }
                            _ => {
                                if run_count(&st) != gold.count {
                                    return Err(format!("count mismatch t={t} it={it}"));
                                }
                            }
                        }
                    }
                    Ok(())
                }));
            }
            // Join all workers; surface the first error (or a panic).
            let mut result = Ok(());
            for h in handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) if result.is_ok() => result = Err(e),
                    Ok(Err(_)) => {}
                    Err(_) if result.is_ok() => {
                        result = Err("a worker thread panicked".to_string())
                    }
                    Err(_) => {}
                }
            }
            let _ = tx.send(result);
        })
    };

    match rx.recv_timeout(WATCHDOG) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("fan-out concurrency mismatch: {e}"),
        Err(mpsc::RecvTimeoutError::Timeout) => panic!(
            "fan-out concurrency test did not finish within {WATCHDOG:?} — \
             likely a tokio/rayon deadlock or runtime starvation"
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("coordinator dropped the channel without sending a result")
        }
    }
    coordinator.join().expect("coordinator thread joined");
}
