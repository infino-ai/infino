// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared markdown summary emitter for the infino-only bench harnesses.
//!
//! The custom report layer builds markdown blocks summarizing measured
//! results. Blocks are written to stderr framed by sentinel comments
//! (`<!-- BEGIN: <anchor_id> -->` / `<!-- END: <anchor_id> -->`).
//! When `INFINO_BENCH_UPDATE_README=1` is set, the same block also
//! replaces the matching section in `benches/README.md` in place.

use std::fs;
use std::io::Write;
use std::path::Path;

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

    maybe_update_readme(section);
}

/// Replace the matching README section iff `INFINO_BENCH_UPDATE_README`
/// is set. Unlike [`emit`], this does **not** echo to stderr — callers
/// that do their own (e.g. colored, delta-annotated) terminal rendering
/// use this to avoid a double print.
pub fn maybe_update_readme(section: &MarkdownSection) {
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

/// Nanoseconds per microsecond — unit boundary + divisor in [`fmt_time`].
const NS_PER_US: f64 = 1_000.0;
/// Nanoseconds per millisecond.
const NS_PER_MS: f64 = 1_000_000.0;
/// Nanoseconds per second.
const NS_PER_SEC: f64 = 1_000_000_000.0;
/// Elements per "K/s" unit in [`fmt_throughput`].
const ELEMS_PER_K: f64 = 1_000.0;
/// Elements per "M/s" unit.
const ELEMS_PER_M: f64 = 1_000_000.0;
/// Decimal bytes per "MB/s" in [`fmt_bandwidth`] (1e6, matching the
/// conventional MB/s reading rather than MiB).
const BYTES_PER_MB_DECIMAL: f64 = 1_000_000.0;
/// Decimal bytes per "GB/s" (1e9).
const BYTES_PER_GB_DECIMAL: f64 = 1_000_000_000.0;

/// Human-readable duration with magnitude-selected units (ns / µs / ms / s).
pub fn fmt_time(ns: f64) -> String {
    if ns < NS_PER_US {
        format!("{ns:.0} ns")
    } else if ns < NS_PER_MS {
        format!("{:.2} µs", ns / NS_PER_US)
    } else if ns < NS_PER_SEC {
        format!("{:.2} ms", ns / NS_PER_MS)
    } else {
        format!("{:.2} s", ns / NS_PER_SEC)
    }
}

/// Human-readable count with K/M/B suffixes — `1000000` → `1M`,
/// `500000` → `500K`, `12345` → `12.3K`. For doc-scale labels so a
/// reader doesn't have to count zeros.
pub fn fmt_count(n: usize) -> String {
    let f = n as f64;
    let (v, suffix) = if f >= 1e9 {
        (f / 1e9, "B")
    } else if f >= 1e6 {
        (f / 1e6, "M")
    } else if f >= 1e3 {
        (f / 1e3, "K")
    } else {
        return format!("{n}");
    };
    if (v.fract()).abs() < f64::EPSILON {
        format!("{v:.0}{suffix}")
    } else {
        format!("{v:.1}{suffix}")
    }
}

/// Throughput (elements per second) with K/M units.
pub fn fmt_throughput(elements_per_sec: f64) -> String {
    if elements_per_sec >= ELEMS_PER_M {
        format!("{:.2} M/s", elements_per_sec / ELEMS_PER_M)
    } else if elements_per_sec >= ELEMS_PER_K {
        format!("{:.1} K/s", elements_per_sec / ELEMS_PER_K)
    } else {
        format!("{elements_per_sec:.0}/s")
    }
}

/// Ingest bandwidth (bytes per second) with MB/s / GB/s units. Decimal
/// (1e6 / 1e9) to match the conventional "MB/s" reading. The byte count
/// is the logical input payload processed (FTS: corpus text bytes;
/// vector: `n_docs × dim × 4`), not the output artifact size.
pub fn fmt_bandwidth(bytes_per_sec: f64) -> String {
    if bytes_per_sec >= BYTES_PER_GB_DECIMAL {
        format!("{:.2} GB/s", bytes_per_sec / BYTES_PER_GB_DECIMAL)
    } else {
        format!("{:.1} MB/s", bytes_per_sec / BYTES_PER_MB_DECIMAL)
    }
}
