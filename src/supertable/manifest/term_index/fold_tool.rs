// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Offline driver for the term index's optimize-time fold: opens an
//! existing table, fetches its root's segments once into a scratch
//! directory, and times `merge_segments` over them exactly as optimize's
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
//! the engine's own), `FOLD_SHARDS` fixes the merge's shard count (default:
//! chosen by the engine from the sources and the pool), `FOLD_RUNS` repeats
//! the merge (default 1). Slices already in the scratch directory are not
//! fetched again.

use std::{
    collections::HashMap,
    env, fs, io,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use arrow::datatypes::{DataType, Field, Schema};
use rayon::ThreadPoolBuilder;
use tracing_subscriber::{EnvFilter, fmt};

use super::{
    build::{BuildPolicy, BuiltBytes, SliceFile, merge_segments, tier_path},
    download_segments,
    format::Segment,
};
use crate::{
    runtime_metrics::rss::available_memory_bytes,
    storage::{GcsStorageProvider, LocalFsStorageProvider, StorageOptions, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{
        Supertable, SupertableOptions,
        manifest::{list::PartitionStrategy, part::ContentHash},
        writer::scoring_by_ordinal,
    },
};

/// The text column of the tables this drives (the FineWeb scale tables).
const COLUMN: &str = "text";

fn env_num<T: std::str::FromStr>(key: &str) -> Option<T> {
    env::var(key).ok().and_then(|v| v.trim().parse().ok())
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

/// `segments`' slices as files under `dir`, when every slice and tier
/// object of the root is already there; `None` otherwise.
fn present_sources(dir: &Path, segments: &[Segment]) -> Option<Vec<Vec<SliceFile>>> {
    let path_of = |hash: &ContentHash| dir.join(format!("{}.slice", hash.to_hex()));
    let mut out = Vec::with_capacity(segments.len());
    for segment in segments {
        let mut files = Vec::with_capacity(segment.slices.len());
        for slice in &segment.slices {
            let path = path_of(&slice.content_hash);
            if !path.is_file() {
                return None;
            }
            if slice.tier.is_some() && !tier_path(dir, &slice.content_hash).is_file() {
                return None;
            }
            files.push(SliceFile {
                path,
                first_key: slice.first_key.clone(),
                last_key: slice.last_key.clone(),
            });
        }
        out.push(files);
    }
    Some(out)
}

fn bytes_of(built: &BuiltBytes) -> u64 {
    match built {
        BuiltBytes::Resident(bytes) => bytes.len() as u64,
        BuiltBytes::Spilled { path, .. } => fs::metadata(path).map(|m| m.len()).unwrap_or(0),
    }
}

fn rss_peak_mb() -> u64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map_or(0, |kb| kb / 1024)
}

#[test]
#[ignore = "offline timing driver: needs FOLD_TABLE and credentials"]
fn fold_timing() {
    let table = env::var("FOLD_TABLE").expect("FOLD_TABLE=<table uri>");
    let scratch = PathBuf::from(
        env::var("FOLD_SCRATCH").unwrap_or_else(|_| "/tmp/infino-fold-scratch".to_owned()),
    );
    fs::create_dir_all(&scratch).expect("scratch dir");
    let threads: Option<usize> = env_num("FOLD_THREADS");
    let shards: Option<usize> = env_num("FOLD_SHARDS");
    let runs: usize = env_num("FOLD_RUNS").unwrap_or(1);

    let _ = fmt()
        .with_env_filter(EnvFilter::new("term_index::fold=info"))
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
    let n_superfiles = manifest.superfiles.len();
    let slices_in: usize = root.segments.iter().map(|s| s.slices.len()).sum();
    let bytes_in: u64 = root
        .segments
        .iter()
        .flat_map(|s| &s.slices)
        .map(|s| s.len)
        .sum();
    eprintln!(
        "FOLD open {:.1}s table={table} superfiles={n_superfiles} root_superfiles={} segments={} slices={slices_in} slice_bytes={:.1}MiB complete={}",
        t_open.elapsed().as_secs_f64(),
        root.superfiles.len(),
        root.segments.len(),
        bytes_in as f64 / (1024.0 * 1024.0),
        manifest.term_index_complete(),
    );
    for (i, segment) in root.segments.iter().enumerate() {
        let bytes: u64 = segment.slices.iter().map(|s| s.len).sum();
        eprintln!(
            "FOLD segment[{i}] slices={} bytes={:.1}MiB tiers={}",
            segment.slices.len(),
            bytes as f64 / (1024.0 * 1024.0),
            segment.slices.iter().filter(|s| s.tier.is_some()).count()
        );
    }

    let t_fetch = Instant::now();
    let sources = match present_sources(&scratch, &root.segments) {
        Some(sources) => {
            eprintln!("FOLD sources present in {}", scratch.display());
            sources
        }
        None => {
            let sources = st
                .block_on_query(download_segments(
                    storage.as_ref(),
                    &root.segments,
                    &scratch,
                ))
                .expect("download segments");
            eprintln!(
                "FOLD download {:.1}s ({} objects)",
                t_fetch.elapsed().as_secs_f64(),
                root.segments
                    .iter()
                    .flat_map(|s| &s.slices)
                    .map(|s| 1 + usize::from(s.tier.is_some()))
                    .sum::<usize>()
            );
            sources
        }
    };
    let local_bytes: u64 = sources
        .iter()
        .flatten()
        .map(|f| {
            fs::metadata(&f.path).map(|m| m.len()).unwrap_or(0)
                + fs::metadata(f.path.with_extension("tier"))
                    .map(|m| m.len())
                    .unwrap_or(0)
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
        merge_shards: shards,
        scratch_root: Some(scratch.join("out")),
        ..manifest.options.term_index_build_policy()
    };
    let pool_threads = policy
        .pool
        .as_deref()
        .map_or(rayon::current_num_threads(), |p| p.current_num_threads());
    eprintln!(
        "FOLD inputs local_bytes={:.1}MiB ordinals={} dropped={dropped} pool_threads={pool_threads} shards={shards:?} avail_mem={:.1}GiB",
        local_bytes as f64 / (1024.0 * 1024.0),
        ordinals.len(),
        available_memory_bytes().unwrap_or(0) as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    for run in 0..runs {
        let out_dir = scratch.join("out");
        let _ = fs::remove_dir_all(&out_dir);
        let t = Instant::now();
        let merged = merge_segments(&sources, &ordinals, &policy).expect("merge");
        let secs = t.elapsed().as_secs_f64();
        let out_slices = merged.slices.len();
        let out_bytes: u64 = merged.slices.iter().map(|(_, b)| bytes_of(b)).sum();
        let out_tiers = merged.tiers.len();
        let tier_bytes: u64 = merged.tiers.iter().map(|(_, b)| bytes_of(b)).sum();
        eprintln!(
            "FOLD run[{run}] merge {secs:.1}s slices={out_slices} slice_bytes={:.1}MiB tiers={out_tiers} tier_bytes={:.1}MiB in_rate={:.1}MiB/s rss_peak={}MiB",
            out_bytes as f64 / (1024.0 * 1024.0),
            tier_bytes as f64 / (1024.0 * 1024.0),
            local_bytes as f64 / (1024.0 * 1024.0) / secs,
            rss_peak_mb(),
        );
        drop(merged);
    }
}
