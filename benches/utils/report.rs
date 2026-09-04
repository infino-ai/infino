// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Fact reporting + pretty rendering for benches.
//!
//! A bench builds [`Section`]s of [`Block`]s of [`Cell`]s; every metric
//! cell carries its raw comparable value and its human form. [`Report`]:
//!
//!   - persists **every** metric this run produced to one JSON file per
//!     bench (in the target dir) — the structured source of truth,
//!   - renders each table of measured values to the terminal, and
//!   - when `INFINO_BENCH_UPDATE_README=1`, writes the same tables into
//!     `benches/README.md`.
//!
//! Tables report the run's numbers as-is: no run-over-run comparison,
//! no delta annotations. Each section is stamped with the host
//! (CPU / cores / RAM / OS) so a committed table says what machine
//! produced it.

use std::{collections::HashMap, path::PathBuf};

use serde_json::Value;

use crate::{
    corpus,
    markdown::{self, MarkdownSection},
};

/// Metrics-file key holding [`corpus_fingerprint`]. Underscore-prefixed so
/// it cannot collide with a metric key, which is always
/// `anchor|subtitle|label|header`.
const CORPUS_FINGERPRINT_KEY: &str = "__corpus";

/// FNV-1a 64-bit offset basis and prime — a short, stable, dependency-free
/// string hash. Only identity matters here (same corpus ⇒ same value), not
/// distribution.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Bits of a `u64` that survive a round trip through `f64` (and therefore
/// through JSON) exactly. The fingerprint is masked to this width so the
/// value read back compares equal to the value written.
const F64_EXACT_INT_BITS: u32 = 53;

/// Stable fingerprint of the active corpus label, as an integer-valued
/// `f64` so it rides in the existing `HashMap<String, f64>` metrics map
/// without widening the file format.
fn corpus_fingerprint() -> f64 {
    let mut h = FNV_OFFSET_BASIS;
    for b in corpus::corpus_label().as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    (h & ((1u64 << F64_EXACT_INT_BITS) - 1)) as f64
}

/// Which direction is an improvement for a metric. Retained as
/// metric-direction metadata on [`Cell::Metric`]; the report itself no
/// longer annotates deltas.
#[derive(Clone, Copy)]
pub enum Better {
    /// Smaller is better — latency, build time, memory.
    Lower,
    /// Larger is better — throughput, bandwidth.
    Higher,
}

/// One table cell.
pub enum Cell {
    /// A row/label cell with no tracked metric.
    Text(String),
    /// A measured value: `raw` is the comparable quantity (ns, bytes,
    /// items/s …) persisted to the JSON source of truth; `shown` is its
    /// human form displayed in the table.
    Metric {
        raw: f64,
        shown: String,
        better: Better,
        gate: bool,
    },
}

/// A label cell (no delta).
pub fn text(s: impl Into<String>) -> Cell {
    Cell::Text(s.into())
}

/// A gate metric cell — Δ-tracked and surfaced (min latency, build time,
/// peak RSS, stored size).
pub fn metric(raw: f64, shown: impl Into<String>, better: Better) -> Cell {
    Cell::Metric {
        raw,
        shown: shown.into(),
        better,
        gate: true,
    }
}

/// A context metric cell — shown and saved, but not Δ-tracked (spread,
/// secondary, derived numbers).
pub fn context(raw: f64, shown: impl Into<String>, better: Better) -> Cell {
    Cell::Metric {
        raw,
        shown: shown.into(),
        better,
        gate: false,
    }
}

/// One titled table within a section.
pub struct Block {
    pub subtitle: String,
    pub headers: Vec<String>,
    pub rows: Vec<Vec<Cell>>,
}

/// One README-anchored section, possibly several tables (e.g. OR / AND /
/// per-algorithm probes all under one anchor).
pub struct Section {
    pub anchor: String,
    pub title: String,
    pub note: String,
    pub blocks: Vec<Block>,
}

const C_DIM: &str = "\x1b[2m";
const C_RESET: &str = "\x1b[0m";

/// A rendered cell — just its displayed text. (No delta column: the
/// report shows measured values, not run-over-run comparisons.)
struct Rendered {
    value: String,
}

pub struct Report {
    bench: String,
    cur: HashMap<String, f64>,
    host: String,
}

impl Report {
    /// Start a report for `bench`. The run's metrics are recorded and
    /// persisted to the JSON source of truth by [`Report::save`].
    pub fn load(bench: &str) -> Self {
        Self {
            bench: bench.to_string(),
            cur: HashMap::new(),
            host: machine_info(),
        }
    }

    /// Render `section` to the terminal and, when
    /// `INFINO_BENCH_UPDATE_README` is set, into `benches/README.md`.
    /// Records every metric for [`Report::save`].
    pub fn emit(&mut self, section: &Section) {
        let mut md = String::new();
        md.push_str(&format!("### {}\n\n", section.title));
        md.push_str(&format!("_{}_\n\n", self.host));
        if !section.note.is_empty() {
            md.push_str(&section.note);
            md.push_str("\n\n");
        }

        eprintln!();
        eprintln!("══ {} ══", section.title);
        eprintln!("{}{}{}", C_DIM, self.host, C_RESET);

        for block in &section.blocks {
            let grid = self.render_block(section, block);
            if !block.subtitle.is_empty() {
                md.push_str(&format!("**{}**\n\n", block.subtitle));
                eprintln!("\n{}", block.subtitle);
            }
            // Markdown: compact GFM (GitHub aligns columns itself, so no
            // manual padding — keeps the committed source clean).
            md.push_str(&assemble_markdown(&block.headers, &grid));
            md.push('\n');
            // Terminal: padded for monospace readability.
            eprint!("{}", assemble_terminal(&block.headers, &grid));
        }

        markdown::maybe_update_readme(&MarkdownSection {
            anchor_id: section.anchor.clone(),
            body: md.trim_end().to_string(),
        });
    }

    /// Render one block into a grid of [`Rendered`] cells, recording each
    /// metric under a stable `anchor|subtitle|label|header` key for the
    /// JSON source of truth.
    fn render_block(&mut self, section: &Section, block: &Block) -> Vec<Vec<Rendered>> {
        let mut grid = Vec::with_capacity(block.rows.len());
        for row in &block.rows {
            let label = row
                .first()
                .and_then(|c| match c {
                    Cell::Text(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("");
            let mut rrow = Vec::with_capacity(row.len());
            for (ci, cell) in row.iter().enumerate() {
                match cell {
                    Cell::Text(s) => rrow.push(Rendered { value: s.clone() }),
                    Cell::Metric { raw, shown, .. } => {
                        let header = block.headers.get(ci).map(String::as_str).unwrap_or("");
                        let key =
                            format!("{}|{}|{}|{}", section.anchor, block.subtitle, label, header);
                        self.cur.insert(key, *raw);
                        rrow.push(Rendered {
                            value: shown.clone(),
                        });
                    }
                }
            }
            grid.push(rrow);
        }
        grid
    }

    /// Persist this run's metrics to the JSON source of truth.
    ///
    /// Merges over the existing file rather than overwriting, so a
    /// partial run (e.g. `-- superfile_fts_build`) updates only the
    /// metrics it measured and leaves the rest of the file intact.
    ///
    /// The merge is only valid within one corpus. Metrics measured on
    /// different data are not comparable, and merging them silently
    /// produces a file whose untouched keys still hold the previous
    /// corpus's numbers — read later as regressions (or improvements)
    /// that no code change caused. So the file records which corpus wrote
    /// it, and a run under a different one starts a fresh accumulator
    /// instead of merging. Keyed by fingerprint rather than by filename
    /// so the path CI copies for its A/B baseline stays exactly as it is.
    pub fn save(&self) {
        let path = store_path(&self.bench);
        let corpus = corpus_fingerprint();
        let existing = read_map(&path).unwrap_or_default();
        let same_corpus = existing
            .get(CORPUS_FINGERPRINT_KEY)
            .is_none_or(|&prev| prev == corpus);
        let mut merged = if same_corpus {
            existing
        } else {
            eprintln!(
                "[report] {} was last written under a different corpus; starting a \
                 fresh metrics file for {} rather than merging incomparable numbers",
                path.display(),
                corpus::corpus_label()
            );
            HashMap::new()
        };
        for (k, v) in &self.cur {
            merged.insert(k.clone(), *v);
        }
        merged.insert(CORPUS_FINGERPRINT_KEY.to_string(), corpus);
        if let Err(e) = write_map(&path, &merged) {
            eprintln!("[report] failed to persist metrics for {}: {e}", self.bench);
        }
    }
}

/// Compact GFM table for markdown. No manual alignment padding (GitHub
/// renders the columns aligned); each cell is just its value. Clean
/// committed source.
fn assemble_markdown(headers: &[String], grid: &[Vec<Rendered>]) -> String {
    let mut s = String::new();
    s.push('|');
    for h in headers {
        s.push_str(&format!(" {h} |"));
    }
    s.push('\n');
    s.push('|');
    for _ in headers {
        s.push_str(" --- |");
    }
    s.push('\n');
    for row in grid {
        s.push('|');
        for cell in row {
            s.push_str(&format!(" {} |", cell.value));
        }
        s.push('\n');
    }
    s
}

/// Assemble an aligned table for the terminal: values right-aligned per
/// column under a left-aligned header. Widths are computed from
/// **visible** length (multibyte glyphs like `µ` counted as one).
fn assemble_terminal(headers: &[String], grid: &[Vec<Rendered>]) -> String {
    let ncol = headers.len();
    let mut value_w = vec![0usize; ncol];
    for row in grid {
        for (c, cell) in row.iter().enumerate().take(ncol) {
            value_w[c] = value_w[c].max(visible_len(&cell.value));
        }
    }
    // Column width = max(header, value).
    let col_w: Vec<usize> = (0..ncol)
        .map(|c| visible_len(&headers[c]).max(value_w[c]))
        .collect();

    let mut s = String::new();
    // Header (left-aligned).
    s.push('|');
    for (c, w) in col_w.iter().enumerate() {
        s.push_str(&format!(" {} |", pad_right(&headers[c], *w)));
    }
    s.push('\n');
    s.push('|');
    for w in &col_w {
        s.push_str(&format!(" {} |", "-".repeat(*w)));
    }
    s.push('\n');
    // Data rows: values right-aligned.
    for row in grid {
        s.push('|');
        for (c, w) in col_w.iter().enumerate() {
            s.push_str(&format!(" {} |", pad_left(&row[c].value, *w)));
        }
        s.push('\n');
    }
    s
}

fn pad_right(s: &str, width: usize) -> String {
    format!("{s}{}", " ".repeat(width.saturating_sub(visible_len(s))))
}

fn pad_left(s: &str, width: usize) -> String {
    format!("{}{s}", " ".repeat(width.saturating_sub(visible_len(s))))
}

/// Visible length ignoring ANSI escape sequences and counting chars (not
/// bytes), so multibyte glyphs (`µ`) and color codes don't skew padding.
fn visible_len(s: &str) -> usize {
    let mut n = 0;
    let mut in_escape = false;
    for c in s.chars() {
        if in_escape {
            if c == 'm' {
                in_escape = false;
            }
            continue;
        }
        if c == '\x1b' {
            in_escape = true;
            continue;
        }
        n += 1;
    }
    n
}

/// `CPU · physical/logical cores · RAM · OS/arch`, best-effort.
fn machine_info() -> String {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let physical = num_cpus::get_physical();
    let cpu = read_cpu_model().unwrap_or_else(|| "unknown CPU".into());
    let ram = read_mem_total_gib()
        .map(|g| format!(" · {g:.0} GiB RAM"))
        .unwrap_or_default();
    format!(
        "Host: {cpu} · {physical}C/{logical}T{ram} · {}/{}",
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

fn read_cpu_model() -> Option<String> {
    let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            return rest.split_once(':').map(|(_, v)| v.trim().to_string());
        }
    }
    None
}

fn read_mem_total_gib() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: f64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb / (1024.0 * 1024.0));
        }
    }
    None
}

fn store_path(bench: &str) -> PathBuf {
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    base.join("infino-bench").join(format!("{bench}.json"))
}

fn read_map(path: &PathBuf) -> Option<HashMap<String, f64>> {
    let text = std::fs::read_to_string(path).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let obj = v.as_object()?;
    Some(
        obj.iter()
            .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?)))
            .collect(),
    )
}

fn write_map(path: &PathBuf, map: &HashMap<String, f64>) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_vec_pretty(map).expect("serialize bench metrics");
    std::fs::write(path, body)
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    /// The metrics file is a merge accumulator, which is only sound within
    /// one corpus: a run that measures a subset of the keys must not leave
    /// the rest holding numbers measured on different data. The fingerprint
    /// gate is what prevents that, so pin both directions.
    #[test]
    fn a_corpus_switch_replaces_the_metrics_file_instead_of_merging() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("superfile_fts.json");

        // A first corpus writes two metrics plus its fingerprint.
        let first_corpus = 111.0;
        let mut first = HashMap::new();
        first.insert("anchor|sub|row|warm p50".to_string(), 10.0);
        first.insert("anchor|sub|row|build".to_string(), 20.0);
        first.insert(CORPUS_FINGERPRINT_KEY.to_string(), first_corpus);
        write_map(&path, &first).expect("write first");

        // Same fingerprint ⇒ the accumulator is kept, so a partial run's
        // untouched keys survive (the behaviour `save` relies on).
        let existing = read_map(&path).expect("read back");
        assert!(
            existing
                .get(CORPUS_FINGERPRINT_KEY)
                .is_some_and(|&p| p == first_corpus),
            "same corpus keeps its accumulator"
        );
        assert_eq!(existing.get("anchor|sub|row|build"), Some(&20.0));

        // Different fingerprint ⇒ `save` starts fresh. Emulate that write
        // and assert the previous corpus's untouched metric is gone rather
        // than lingering as a stale number.
        let second_corpus = 222.0;
        let mut fresh = HashMap::new();
        fresh.insert("anchor|sub|row|warm p50".to_string(), 99.0);
        fresh.insert(CORPUS_FINGERPRINT_KEY.to_string(), second_corpus);
        write_map(&path, &fresh).expect("write second");

        let after = read_map(&path).expect("read back");
        assert_eq!(after.get("anchor|sub|row|warm p50"), Some(&99.0));
        assert!(
            !after.contains_key("anchor|sub|row|build"),
            "the previous corpus's untouched metric must be gone, not stale"
        );
        assert!(
            after
                .get(CORPUS_FINGERPRINT_KEY)
                .is_some_and(|&p| p == second_corpus),
            "the file records the corpus that wrote it"
        );
    }

    /// The fingerprint has to round-trip through JSON's `f64` exactly, or
    /// the gate would fire on every run and no partial run would ever merge.
    #[test]
    fn the_corpus_fingerprint_round_trips_through_the_metrics_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("roundtrip.json");
        let fp = corpus_fingerprint();
        assert_eq!(fp, fp.trunc(), "fingerprint must be integer-valued");
        assert!(
            (fp as u64) < (1u64 << F64_EXACT_INT_BITS),
            "fingerprint must fit the exactly-representable width"
        );
        let mut m = HashMap::new();
        m.insert(CORPUS_FINGERPRINT_KEY.to_string(), fp);
        write_map(&path, &m).expect("write");
        assert_eq!(
            read_map(&path).expect("read").get(CORPUS_FINGERPRINT_KEY),
            Some(&fp),
            "written and read fingerprints must compare equal"
        );
    }
}
