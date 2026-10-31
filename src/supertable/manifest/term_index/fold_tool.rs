// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Offline driver for the term index's optimize-time merge: opens an
//! existing table, fetches its root's layers once into a scratch
//! directory, and times `merge_layers` over them exactly as optimize's
//! rebuild merges them — so a change to the merge can be measured in a
//! minute, without re-ingesting the table or going back to object storage.
//!
//! Run as an ignored test, from a host with credentials for the table:
//!
//! ```sh
//! FOLD_TABLE=gs://bucket/prefix FOLD_SCRATCH=/data/fold-scratch \
//!   cargo test --release --lib term_index::fold_tool -- --ignored --nocapture
//! ```
//!
//! Knobs: `FOLD_THREADS` sizes the writer pool the merge runs on (default:
//! the engine's own), `FOLD_RANGES` fixes how many key ranges the merge is cut into (default:
//! chosen by the engine from the sources and the pool), `FOLD_RUNS` repeats
//! the merge (default 1), and `FOLD_UPLOAD=<uri>` also times writing the
//! merged output to that prefix. Slices already in the scratch directory
//! are not fetched again. Per-range timing is logged at debug level under
//! the `term_index::fold` target.

use std::{
    collections::HashMap,
    env, fs, io,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::Instant,
};

use arrow::datatypes::{DataType, Field, Schema};
use rayon::ThreadPoolBuilder;
use tracing_subscriber::{EnvFilter, fmt};

use super::{
    build::{BuildPolicy, BuiltBytes, SliceFile, merge_layers, tier_path},
    download_layers,
    format::Layer,
    slice_files, write_slices, write_tiers,
};
use crate::{
    runtime_metrics::rss::{available_memory_bytes, status_field_bytes},
    storage::{GcsStorageProvider, LocalFsStorageProvider, StorageOptions, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{
        Supertable, SupertableOptions, manifest::list::PartitionStrategy,
        writer::scoring_by_ordinal,
    },
};

/// The text column of the tables this drives (the FineWeb scale tables).
const COLUMN: &str = "text";
/// Where the layers go when `FOLD_SCRATCH` is not set.
const DEFAULT_SCRATCH: &str = "/tmp/infino-fold-scratch";
/// The scratch subdirectory a run's merged output is spilled to.
const OUT_DIR: &str = "out";
const MIB: f64 = 1024.0 * 1024.0;

fn env_num<T: FromStr>(key: &str) -> Option<T> {
    env::var(key).ok().and_then(|v| v.trim().parse().ok())
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / MIB
}

fn storage_for(target: &str) -> Arc<dyn StorageProvider> {
    match target.strip_prefix("gs://") {
        Some(rest) => {
            let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
            let opts: StorageOptions = HashMap::new();
            Arc::new(
                GcsStorageProvider::new_with_prefix(bucket, prefix, &opts)
                    .expect("build GCS storage provider"),
            )
        }
        None => Arc::new(LocalFsStorageProvider::new(target).expect("open local storage")),
    }
}

/// `layers`' slices as files under `dir`, when every slice and tier
/// object of the root is already there; `None` otherwise.
fn present_sources(dir: &Path, layers: &[Layer]) -> Option<Vec<Vec<SliceFile>>> {
    let files = slice_files(dir, layers);
    let complete = layers.iter().zip(&files).all(|(layer, files)| {
        layer.slices.iter().zip(files).all(|(slice, file)| {
            file.path.is_file()
                && (slice.tier.is_none() || tier_path(dir, &slice.content_hash).is_file())
        })
    });
    complete.then_some(files)
}

fn bytes_of(built: &BuiltBytes) -> u64 {
    match built {
        BuiltBytes::Resident(bytes) => bytes.len() as u64,
        BuiltBytes::Spilled { path, .. } => fs::metadata(path).map(|m| m.len()).unwrap_or(0),
    }
}

fn file_len(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

#[test]
#[ignore = "offline timing driver: needs FOLD_TABLE and credentials"]
fn fold_timing() {
    let table = env::var("FOLD_TABLE").expect("FOLD_TABLE=<table uri>");
    let scratch =
        PathBuf::from(env::var("FOLD_SCRATCH").unwrap_or_else(|_| DEFAULT_SCRATCH.to_owned()));
    fs::create_dir_all(&scratch).expect("scratch dir");
    let threads: Option<usize> = env_num("FOLD_THREADS");
    let n_ranges: Option<usize> = env_num("FOLD_RANGES");
    let runs: usize = env_num("FOLD_RUNS").unwrap_or(1);
    let _ = fmt()
        .with_env_filter(EnvFilter::new("term_index::fold=debug"))
        .with_writer(io::stderr)
        .with_target(false)
        .try_init();

    let storage = storage_for(&table);
    let schema = Arc::new(Schema::new(vec![Field::new(
        COLUMN,
        DataType::LargeUtf8,
        false,
    )]));
    let mut options = SupertableOptions::new(
        schema,
        vec![FtsConfig::new(COLUMN).positions(true).stored(false)],
        vec![],
    )
    .expect("options")
    .with_partition_strategy(PartitionStrategy::Hash {
        column: "_id".to_owned(),
        n_buckets: 1,
    })
    .with_storage(Arc::clone(&storage));
    if let Some(n) = threads {
        options = options.with_writer_pool(Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(|i| format!("fold-writer-{i}"))
                .build()
                .expect("pool"),
        ));
    }
    let t_open = Instant::now();
    let st = Supertable::open(options).expect("open table");
    let manifest = st.manifest_snapshot();
    let index = st
        .block_on_query(manifest.term_index())
        .expect("load term index")
        .expect("table has a term index");
    let root = index.root().clone();
    let slices_in: usize = root.layers.iter().map(|s| s.slices.len()).sum();
    let bytes_in: u64 = root
        .layers
        .iter()
        .flat_map(|s| &s.slices)
        .map(|s| s.len)
        .sum();
    eprintln!(
        "FOLD open {:.1}s table={table} superfiles={} root_superfiles={} layers={} slices={slices_in} slice_bytes={:.1}MiB complete={}",
        t_open.elapsed().as_secs_f64(),
        manifest.superfiles.len(),
        root.superfiles.len(),
        root.layers.len(),
        mib(bytes_in),
        manifest.term_index_complete(),
    );
    for (i, layer) in root.layers.iter().enumerate() {
        let bytes: u64 = layer.slices.iter().map(|s| s.len).sum();
        eprintln!(
            "FOLD layer[{i}] slices={} bytes={:.1}MiB tiers={}",
            layer.slices.len(),
            mib(bytes),
            layer.slices.iter().filter(|s| s.tier.is_some()).count()
        );
    }

    let t_fetch = Instant::now();
    let sources = match present_sources(&scratch, &root.layers) {
        Some(sources) => {
            eprintln!("FOLD sources present in {}", scratch.display());
            sources
        }
        None => {
            let sources = st
                .block_on_query(download_layers(&storage, &root.layers, &scratch, None))
                .expect("download layers");
            eprintln!(
                "FOLD download {:.1}s ({} objects)",
                t_fetch.elapsed().as_secs_f64(),
                root.layers
                    .iter()
                    .flat_map(|s| &s.slices)
                    .map(|s| 1 + usize::from(s.tier.is_some()))
                    .sum::<usize>()
            );
            sources
        }
    };
    let local_bytes: u64 = root
        .layers
        .iter()
        .zip(&sources)
        .flat_map(|(layer, files)| layer.slices.iter().zip(files))
        .map(|(slice, file)| {
            file_len(&file.path) + file_len(&tier_path(&scratch, &slice.content_hash))
        })
        .sum();

    // Exactly the merge optimize's rebuild runs: ordinals keep their
    // place for a superfile still live, and are dropped otherwise.
    let live: HashMap<_, u32> = manifest
        .superfiles
        .iter()
        .enumerate()
        .map(|(i, e)| (e.superfile_id, i as u32))
        .collect();
    let ordinals: Vec<Option<u32>> = root
        .superfiles
        .iter()
        .map(|id| live.get(id).copied())
        .collect();
    let dropped = ordinals.iter().filter(|o| o.is_none()).count();
    let new_superfiles: Vec<_> = manifest.superfiles.iter().map(|e| e.superfile_id).collect();
    let policy = BuildPolicy {
        scoring_by_ordinal: Arc::new(scoring_by_ordinal(&manifest, &new_superfiles)),
        merge_ranges: n_ranges,
        scratch_root: Some(scratch.join(OUT_DIR)),
        ..manifest.options.term_index_build_policy()
    };
    let pool_threads = policy
        .pool
        .as_deref()
        .map_or(rayon::current_num_threads(), |p| p.current_num_threads());
    eprintln!(
        "FOLD inputs local_bytes={:.1}MiB ordinals={} dropped={dropped} pool_threads={pool_threads} ranges={n_ranges:?} avail_mem={:.1}GiB",
        mib(local_bytes),
        ordinals.len(),
        available_memory_bytes().unwrap_or(0) as f64 / (MIB * 1024.0),
    );

    for run in 0..runs {
        let _ = fs::remove_dir_all(scratch.join(OUT_DIR));
        let t = Instant::now();
        let merged = merge_layers(&sources, &ordinals, &policy).expect("merge");
        let secs = t.elapsed().as_secs_f64();
        let out_bytes: u64 = merged.slices.iter().map(|(_, b)| bytes_of(b)).sum();
        let tier_bytes: u64 = merged.tiers.iter().map(|(_, b)| bytes_of(b)).sum();
        eprintln!(
            "FOLD run[{run}] merge {secs:.1}s slices={} slice_bytes={:.1}MiB tiers={} tier_bytes={:.1}MiB in_rate={:.1}MiB/s rss_peak={:.0}MiB",
            merged.slices.len(),
            mib(out_bytes),
            merged.tiers.len(),
            mib(tier_bytes),
            mib(local_bytes) / secs,
            mib(status_field_bytes("VmHWM:").unwrap_or(0)),
        );
        // Time the upload of the merged output to a scratch prefix, never
        // the table's own, so the fixture keeps no unreferenced objects.
        if let Ok(upload_to) = env::var("FOLD_UPLOAD") {
            let sink = storage_for(&upload_to);
            let t = Instant::now();
            st.block_on_query(write_slices(&sink, merged.slices))
                .expect("upload slices");
            st.block_on_query(write_tiers(&sink, merged.tiers))
                .expect("upload tiers");
            let secs = t.elapsed().as_secs_f64();
            eprintln!(
                "FOLD run[{run}] upload {secs:.1}s {:.1}MiB to {upload_to} ({:.1}MiB/s)",
                mib(out_bytes + tier_bytes),
                mib(out_bytes + tier_bytes) / secs,
            );
        }
    }
}
