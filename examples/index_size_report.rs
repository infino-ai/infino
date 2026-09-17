// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Where do an index's bytes go? Reports every superfile under a path
//! region by region — Parquet columns, id sidecar, vector index, FTS
//! blob — and, inside the FTS blob, per document-frequency band and per
//! layout part (fixed per-term and per-block overhead, lane padding,
//! doc-id / tf payload, positions). Optionally builds the index first
//! from a newline-delimited JSON corpus, one text field per line, into
//! a single compacted superfile — the shape a benchmark index has.
//!
//! ```text
//! # report existing superfiles (a file, or a directory searched recursively)
//! cargo run --release --features test-helpers --example index_size_report -- <path>
//!
//! # build from a corpus, then report
//! cargo run --release --features test-helpers --example index_size_report -- \
//!     --corpus docs.jsonl --field text --out /tmp/idx [--limit N] [--no-positions]
//! ```

use std::{
    env, fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
    time::Duration,
};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    CompactionSettings, GcSettings, OptimizeOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::{builder::FtsConfig, reader::SuperfileReader},
    supertable::{Supertable, SupertableOptions, manifest::list::PartitionStrategy},
};

/// Rows per appended batch while streaming the corpus in.
const BATCH_ROWS: usize = 50_000;
/// Compaction target: large enough that a benchmark-scale corpus lands in
/// one superfile.
const COMPACT_TARGET_MB: u64 = 8 * 1024;
/// Extra memory the compaction may use over its target.
const COMPACT_MEMORY_HEADROOM_MB: u64 = 2048;
/// Every buffered row is committed in one go; auto-flush is set past
/// any corpus this tool is pointed at so a build is one commit.
const COMMIT_THRESHOLD_MB: u64 = 64 * 1024;
/// Bytes per mebibyte.
const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

struct BuildArgs {
    corpus: PathBuf,
    field: String,
    out: PathBuf,
    limit: Option<usize>,
    positions: bool,
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: index_size_report <superfile-or-dir>...\n       index_size_report --corpus <jsonl> --field <name> --out <dir> [--limit N] [--no-positions]"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        return usage();
    }
    if args[0].starts_with("--") {
        let mut b = BuildArgs {
            corpus: PathBuf::new(),
            field: "text".to_string(),
            out: PathBuf::new(),
            limit: None,
            positions: true,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--corpus" => b.corpus = PathBuf::from(&args[i + 1]),
                "--field" => b.field = args[i + 1].clone(),
                "--out" => b.out = PathBuf::from(&args[i + 1]),
                "--limit" => b.limit = args[i + 1].parse().ok(),
                "--no-positions" => {
                    b.positions = false;
                    i += 1;
                    continue;
                }
                _ => return usage(),
            }
            i += 2;
        }
        if b.corpus.as_os_str().is_empty() || b.out.as_os_str().is_empty() {
            return usage();
        }
        if let Err(e) = build(&b) {
            eprintln!("build failed: {e}");
            return ExitCode::FAILURE;
        }
        return report(&[b.out]);
    }
    report(&args.iter().map(PathBuf::from).collect::<Vec<_>>())
}

fn build(b: &BuildArgs) -> Result<(), Box<dyn std::error::Error>> {
    let _ = fs::remove_dir_all(&b.out);
    fs::create_dir_all(&b.out)?;
    let schema = Arc::new(Schema::new(vec![Field::new(
        &b.field,
        DataType::LargeUtf8,
        false,
    )]));
    let storage: Arc<dyn StorageProvider> = Arc::new(LocalFsStorageProvider::new(&b.out)?);
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build()?);
    let opts = SupertableOptions::new(
        schema.clone(),
        vec![
            FtsConfig::new(&b.field)
                .analyzer("standard")
                .positions(b.positions)
                .stored(false),
        ],
        vec![],
    )?
    .with_partition_strategy(PartitionStrategy::Hash {
        column: "_id".to_string(),
        n_buckets: 1,
    })
    .with_writer_pool(pool)
    .with_commit_threshold_size_mb(COMMIT_THRESHOLD_MB)
    .with_storage(storage);
    let st = Supertable::create(opts)?;
    let mut writer = st.writer()?;
    let file = fs::File::open(&b.corpus)?;
    let mut buf: Vec<String> = Vec::with_capacity(BATCH_ROWS);
    let mut total = 0usize;
    let flush = |writer: &mut infino::supertable::SupertableWriter,
                 buf: &mut Vec<String>|
     -> Result<(), Box<dyn std::error::Error>> {
        let arr = LargeStringArray::from(buf.iter().map(String::as_str).collect::<Vec<_>>());
        writer.append(&RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])?)?;
        buf.clear();
        Ok(())
    };
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)?;
        let Some(text) = v.get(&b.field).and_then(|t| t.as_str()) else {
            continue;
        };
        buf.push(text.to_string());
        total += 1;
        if buf.len() == BATCH_ROWS {
            flush(&mut writer, &mut buf)?;
            eprintln!("  {total} docs");
        }
        if b.limit.is_some_and(|n| total >= n) {
            break;
        }
    }
    if !buf.is_empty() {
        flush(&mut writer, &mut buf)?;
    }
    writer.commit()?;
    drop(writer);
    eprintln!("indexed {total} docs; compacting to one superfile");
    st.optimize(
        &OptimizeOptions::compact(CompactionSettings {
            target_superfile_size_mb: COMPACT_TARGET_MB,
            min_fill_percent: 1,
            max_memory_mb: COMPACT_TARGET_MB + COMPACT_MEMORY_HEADROOM_MB,
            ..Default::default()
        })
        .with_gc(GcSettings {
            safety_gap: Duration::ZERO,
        }),
    )?;
    Ok(())
}

fn superfiles_under(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = fs::read_dir(path)
            .map(|rd| rd.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        entries.sort();
        for e in entries {
            superfiles_under(&e, out);
        }
    } else if path.extension().is_some_and(|x| x == "parquet") {
        out.push(path.to_path_buf());
    }
}

fn report(paths: &[PathBuf]) -> ExitCode {
    let mut files = Vec::new();
    for p in paths {
        superfiles_under(p, &mut files);
    }
    if files.is_empty() {
        eprintln!("no superfiles found");
        return ExitCode::FAILURE;
    }
    // Everything in the tree counts toward what a benchmark measures —
    // manifests and sidecars included — so report the tree total too.
    let mut tree_total = 0u64;
    for p in paths {
        tree_total += dir_bytes(p);
    }
    for f in &files {
        let bytes = match fs::read(f) {
            Ok(b) => Bytes::from(b),
            Err(e) => {
                eprintln!("{}: {e}", f.display());
                return ExitCode::FAILURE;
            }
        };
        println!("==== {} ====", f.display());
        match SuperfileReader::open(bytes).and_then(|r| r.size_breakdown()) {
            Ok(Some(b)) => print!("{b}"),
            Ok(None) => println!("(not resident)"),
            Err(e) => {
                eprintln!("{}: {e}", f.display());
                return ExitCode::FAILURE;
            }
        }
    }
    println!(
        "tree total (every file under the given paths): {tree_total} B  {:.2} MiB",
        tree_total as f64 / BYTES_PER_MIB
    );
    ExitCode::SUCCESS
}

fn dir_bytes(p: &Path) -> u64 {
    if p.is_dir() {
        fs::read_dir(p)
            .map(|rd| rd.flatten().map(|e| dir_bytes(&e.path())).sum())
            .unwrap_or(0)
    } else {
        fs::metadata(p).map(|m| m.len()).unwrap_or(0)
    }
}
