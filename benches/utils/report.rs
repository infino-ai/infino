// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Run-to-run delta tracking + pretty rendering for benches.
//!
//! A bench builds [`Section`]s of [`Block`]s of [`Cell`]s; every metric
//! cell carries its raw comparable
//! value and a "lower/higher is better" direction. [`Report`]:
//!
//!   - persists **every** metric this run produced to one JSON file per
//!     bench (in the target dir),
//!   - diffs each metric against the **previous run's** file,
//!   - renders each table to the terminal with a per-cell delta
//!     (`+6.5% better` / `-4.1% worse` / `~`; `(new)` on first run),
//!     coloring the delta when stderr is a TTY,
//!   - and, when `INFINO_BENCH_UPDATE_README=1`, writes the same tables
//!     (plain) into `benches/README.md`.
//!
//! Each section is stamped with the host (CPU / cores / RAM / OS) so a
//! committed table says what machine produced it.
//!
//! Baseline is the machine-local previous run — the edit/build/compare
//! loop. There is no bootstrap-statistics layer: build, RSS, and cold-tier
//! numbers need "is this run better or worse than my last one, on every
//! number".

use std::{collections::HashMap, io::IsTerminal, path::PathBuf};

use serde_json::Value;

use crate::markdown::{self, MarkdownSection};

/// Which direction is an improvement for a metric.
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
    /// items/s …) used for the delta; `shown` is its human form. `gate`
    /// metrics carry a Δ and are surfaced; context metrics (`gate = false`)
    /// are shown and saved but not Δ-tracked — spread/derived numbers that
    /// would otherwise drown the signal.
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

/// Percent band inside which a change is reported as noise (`~`).
const NOISE_BAND_PCT: f64 = 3.0;

const C_GREEN: &str = "\x1b[32m";
const C_RED: &str = "\x1b[31m";
const C_DIM: &str = "\x1b[2m";
const C_RESET: &str = "\x1b[0m";

/// A rendered cell, split into the value and its delta so the assembler
/// can right-align values and left-align deltas into aligned columns.
struct Rendered {
    value: String,
    delta: String,
    delta_color: &'static str,
}

fn compute_delta(prev: Option<f64>, new: f64, better: Better) -> (String, &'static str) {
    let Some(base) = prev.filter(|&b| b != 0.0) else {
        return ("new".into(), C_DIM);
    };
    let pct = (new - base) / base * 100.0;
    if pct.abs() < NOISE_BAND_PCT {
        return (format!("{pct:+.1}% ~"), C_DIM);
    }
    let improved = match better {
        Better::Lower => pct < 0.0,
        Better::Higher => pct > 0.0,
    };
    (
        format!("{pct:+.1}% {}", if improved { "better" } else { "worse" }),
        if improved { C_GREEN } else { C_RED },
    )
}

pub struct Report {
    bench: String,
    prev: HashMap<String, f64>,
    cur: HashMap<String, f64>,
    color: bool,
    host: String,
    /// When false (comparison sections), cells render as plain values —
    /// no run-over-run delta annotations, no baseline persistence.
    track_deltas: bool,
}

impl Report {
    /// Load the previous run's metrics for `bench` (empty on first run).
    pub fn load(bench: &str) -> Self {
        Self {
            bench: bench.to_string(),
            prev: read_map(&store_path(bench)).unwrap_or_default(),
            cur: HashMap::new(),
            color: std::io::stderr().is_terminal(),
            host: machine_info(),
            track_deltas: true,
        }
    }

    /// A report without run-over-run deltas: cells render as plain
    /// values, nothing is loaded or persisted as a baseline. Used by the
    /// cross-engine comparison sections, whose only meaningful delta is
    /// engine-vs-engine (computed by the caller as its own column).
    pub fn load_plain(bench: &str) -> Self {
        Self {
            bench: bench.to_string(),
            prev: HashMap::new(),
            cur: HashMap::new(),
            color: std::io::stderr().is_terminal(),
            host: machine_info(),
            track_deltas: false,
        }
    }

    /// Render `section` to the terminal (with per-cell deltas) and, when
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
            // Terminal: padded + colored for monospace readability.
            eprint!("{}", assemble_terminal(&block.headers, &grid, self.color));
        }

        markdown::maybe_update_readme(&MarkdownSection {
            anchor_id: section.anchor.clone(),
            body: md.trim_end().to_string(),
        });
    }

    /// Render one block into a grid of [`Rendered`] cells, recording each
    /// metric under a stable `anchor|subtitle|label|header` key so the
    /// next run can diff against it.
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
                    Cell::Text(s) => rrow.push(Rendered {
                        value: s.clone(),
                        delta: String::new(),
                        delta_color: "",
                    }),
                    Cell::Metric {
                        raw,
                        shown,
                        better,
                        gate,
                    } => {
                        let header = block.headers.get(ci).map(String::as_str).unwrap_or("");
                        let key =
                            format!("{}|{}|{}|{}", section.anchor, block.subtitle, label, header);
                        // Only gate metrics carry a Δ; context metrics render
                        // as a plain value (still saved, for the full picture).
                        let (delta, color) = if self.track_deltas && *gate {
                            compute_delta(self.prev.get(&key).copied(), *raw, *better)
                        } else {
                            (String::new(), "")
                        };
                        self.cur.insert(key, *raw);
                        rrow.push(Rendered {
                            value: shown.clone(),
                            delta,
                            delta_color: color,
                        });
                    }
                }
            }
            grid.push(rrow);
        }
        grid
    }

    /// Persist this run's metrics, becoming the next run's baseline.
    ///
    /// Merges over the previous file rather than overwriting, so a
    /// partial run (e.g. `-- superfile_fts_build`) updates only the
    /// metrics it measured and leaves the rest of the baseline intact.
    pub fn save(&self) {
        // Plain (comparison) reports keep no baseline: nothing to diff
        // against next run, so don't write one.
        if !self.track_deltas {
            return;
        }
        let mut merged = self.prev.clone();
        for (k, v) in &self.cur {
            merged.insert(k.clone(), *v);
        }
        if let Err(e) = write_map(&store_path(&self.bench), &merged) {
            eprintln!(
                "[report] failed to persist baseline for {}: {e}",
                self.bench
            );
        }
    }
}

/// Compact GFM table for markdown. No manual alignment padding (GitHub
/// renders the columns aligned); each metric cell is `value (delta)`,
/// each text cell is just its value. Clean committed source.
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
            let c = if cell.delta.is_empty() {
                cell.value.clone()
            } else {
                format!("{} ({})", cell.value, cell.delta)
            };
            s.push_str(&format!(" {c} |"));
        }
        s.push('\n');
    }
    s
}

/// Assemble an aligned table for the terminal. Per column: values are
/// right-aligned in a value sub-field, deltas left-aligned after a
/// 2-space gutter, so both line up vertically. Widths are computed from
/// **visible** length (ANSI escapes and multibyte glyphs excluded).
fn assemble_terminal(headers: &[String], grid: &[Vec<Rendered>], color: bool) -> String {
    let ncol = headers.len();
    let mut value_w = vec![0usize; ncol];
    let mut delta_w = vec![0usize; ncol];
    for row in grid {
        for (c, cell) in row.iter().enumerate().take(ncol) {
            value_w[c] = value_w[c].max(visible_len(&cell.value));
            delta_w[c] = delta_w[c].max(visible_len(&cell.delta));
        }
    }
    // Column width = max(header, value field + gutter + delta field).
    let col_w: Vec<usize> = (0..ncol)
        .map(|c| {
            let content = if delta_w[c] > 0 {
                value_w[c] + 2 + delta_w[c]
            } else {
                value_w[c]
            };
            visible_len(&headers[c]).max(content)
        })
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
    // Data rows.
    for row in grid {
        s.push('|');
        for (c, w) in col_w.iter().enumerate() {
            let cell = &row[c];
            let inner = render_cell(cell, value_w[c], delta_w[c], color);
            s.push_str(&format!(" {} |", pad_right(&inner, *w)));
        }
        s.push('\n');
    }
    s
}

/// Build one cell's inner text: value right-aligned in `value_w`, then a
/// 2-space gutter and the (optionally colored) delta left-aligned in
/// `delta_w`. Text cells with no delta just get the right-aligned value.
fn render_cell(cell: &Rendered, value_w: usize, delta_w: usize, color: bool) -> String {
    let value = pad_left(&cell.value, value_w);
    if delta_w == 0 {
        return value;
    }
    let delta = if color && !cell.delta_color.is_empty() {
        format!(
            "{}{}{}{}",
            cell.delta_color,
            cell.delta,
            C_RESET,
            " ".repeat(delta_w.saturating_sub(visible_len(&cell.delta)))
        )
    } else {
        pad_right(&cell.delta, delta_w)
    };
    format!("{value}  {delta}")
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
