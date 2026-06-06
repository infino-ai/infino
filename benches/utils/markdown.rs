//! Shared markdown summary emitter for the infino-only bench harnesses.
//!
//! After criterion finishes timing, each topic's bench function builds
//! a markdown block summarizing the infino numbers. The block is always
//! written to stderr framed by sentinel comments
//! (`<!-- BEGIN: <anchor_id> -->` / `<!-- END: <anchor_id> -->`).
//! When `INFINO_BENCH_UPDATE_README=1` is set, the same block also
//! replaces the matching section in `benches/README.md` in place.
//!
//! The markdown is purely for human readers. Programmatic consumers
//! should read criterion's own
//! `target/criterion/<group>/<bench>/new/estimates.json` directly —
//! that's the structured source of truth this markdown is derived
//! from.

use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// One markdown section to emit. `anchor_id` is the stable key that
/// matches the `<!-- BEGIN/END: ... -->` markers in
/// `benches/README.md`. `body` is the inner markdown (markers
/// themselves are added by [`emit`]).
pub struct MarkdownSection {
    pub anchor_id: String,
    pub body: String,
}

/// Emit `section` to stderr framed by sentinel markers. When
/// `INFINO_BENCH_UPDATE_README=1`, additionally replace the matching
/// block in `benches/README.md`.
pub fn emit(section: &MarkdownSection) {
    let stderr = std::io::stderr();
    let mut out = stderr.lock();
    let _ = writeln!(out);
    let _ = writeln!(out, "<!-- BEGIN: {} -->", section.anchor_id);
    let _ = writeln!(out, "{}", section.body);
    let _ = writeln!(out, "<!-- END: {} -->", section.anchor_id);
    let _ = writeln!(out);

    if std::env::var_os("INFINO_BENCH_UPDATE_README").is_some() {
        let path = std::path::PathBuf::from("benches/README.md");
        if let Err(e) = update_readme(&path, section) {
            eprintln!("[markdown] failed to update {}: {e}", path.display());
        } else {
            eprintln!(
                "[markdown] updated {} ({})",
                path.display(),
                section.anchor_id
            );
        }
    }
}

fn update_readme(path: &Path, section: &MarkdownSection) -> std::io::Result<()> {
    let begin = format!("<!-- BEGIN: {} -->", section.anchor_id);
    let end = format!("<!-- END: {} -->", section.anchor_id);
    let content = fs::read_to_string(path)?;

    let begin_pos = content.find(&begin).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("marker not found: {begin}"),
        )
    })?;
    let after_begin = begin_pos + begin.len();
    let end_pos = content[after_begin..].find(&end).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("end marker not found after begin: {end}"),
        )
    })? + after_begin;

    let mut new = String::with_capacity(content.len() + section.body.len());
    new.push_str(&content[..after_begin]);
    new.push('\n');
    new.push_str(&section.body);
    new.push('\n');
    new.push_str(&content[end_pos..]);
    fs::write(path, new)?;
    Ok(())
}

// ─── Number formatting ────────────────────────────────────────────────

/// Human-readable duration with magnitude-selected units (ns / µs / ms / s).
pub fn fmt_time(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.0} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} s", ns / 1_000_000_000.0)
    }
}

/// Throughput (elements per second) with K/M units.
pub fn fmt_throughput(elements_per_sec: f64) -> String {
    if elements_per_sec >= 1_000_000.0 {
        format!("{:.2} M/s", elements_per_sec / 1_000_000.0)
    } else if elements_per_sec >= 1_000.0 {
        format!("{:.1} K/s", elements_per_sec / 1_000.0)
    } else {
        format!("{elements_per_sec:.0}/s")
    }
}

/// Ingest bandwidth (bytes per second) with MB/s / GB/s units. Decimal
/// (1e6 / 1e9) to match the conventional "MB/s" reading. The byte count
/// is the logical input payload processed (FTS: corpus text bytes;
/// vector: `n_docs × dim × 4`), not the output artifact size.
pub fn fmt_bandwidth(bytes_per_sec: f64) -> String {
    if bytes_per_sec >= 1_000_000_000.0 {
        format!("{:.2} GB/s", bytes_per_sec / 1_000_000_000.0)
    } else {
        format!("{:.1} MB/s", bytes_per_sec / 1_000_000.0)
    }
}

// ─── estimates.json reader ────────────────────────────────────────────

/// Read criterion's `mean.point_estimate` (in nanoseconds) for a given
/// group + bench id from the criterion artifacts tree. Honors
/// `CARGO_TARGET_DIR` the same way criterion itself does, so the read
/// path tracks where criterion actually wrote (matters on CI / hosts
/// where the target dir is redirected outside the workspace).
///
/// Returns `None` if the file doesn't exist (bench was filtered out or
/// hasn't run yet) or the JSON can't be parsed.
/// Read a tiered search group (`{family}_hot_search` or `{family}_{tier}_search_{s3s_fs|real_s3}`).
pub fn read_tier_mean_ns(family: &str, tier: &str, bench: &str) -> Option<f64> {
    if tier == "hot" {
        return read_mean_ns(&format!("{family}_hot_search"), bench);
    }
    for storage in ["s3s_fs", "real_s3"] {
        let group = format!("{family}_{tier}_search_{storage}");
        if let Some(ns) = read_mean_ns(&group, bench) {
            return Some(ns);
        }
    }
    None
}

pub fn read_mean_ns(group: &str, bench: &str) -> Option<f64> {
    let path = criterion_target_dir()
        .join(group)
        .join(bench)
        .join("new")
        .join("estimates.json");
    let text = fs::read_to_string(&path).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    v.get("mean")?.get("point_estimate")?.as_f64()
}

/// `$CARGO_TARGET_DIR/criterion` if set (criterion writes there when
/// the env var is exported), else workspace-relative `target/criterion`.
fn criterion_target_dir() -> PathBuf {
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    base.join("criterion")
}

/// Mean time + throughput per second given a per-iteration element
/// count. `None` if the bench result isn't on disk.
#[allow(dead_code)]
pub fn read_mean_with_throughput(group: &str, bench: &str, elements: u64) -> Option<(f64, f64)> {
    let ns = read_mean_ns(group, bench)?;
    if ns <= 0.0 {
        return None;
    }
    let throughput = (elements as f64) / (ns / 1_000_000_000.0);
    Some((ns, throughput))
}
