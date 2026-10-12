// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! How many bytes do term blooms written before the table-level term index
//! still occupy in a table's manifest? A storage-backed table has written no
//! bloom since the index landed and drops stored ones when it loads, but the
//! bytes stay in every part no commit has touched since. This reads a table's
//! current manifest as stored — list and every part it references — and
//! reports those bytes next to the manifest's size, so a rewrite of the
//! untouched parts can be decided on numbers.
//!
//! Each argument is searched recursively for tables (a directory holding
//! `_supertable/current`), so a mirrored catalog can be passed whole. Only the
//! pointer, the current list and its parts are read; superfiles are not.
//!
//! ```text
//! cargo run --release --features test-helpers --example manifest_bloom_report -- <dir>...
//! ```

use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use infino::supertable::{
    manifest::{
        commit::{POINTER_PATH, PointerFile},
        list, part,
    },
    schema::LegacyNames,
};

/// Step between binary byte units.
const BYTES_PER_KIB: f64 = 1024.0;
/// Byte units, smallest first.
const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];

/// `bytes` in the largest unit that keeps the value at least 1.
fn human(bytes: u64) -> String {
    let mut v = bytes as f64;
    let mut unit = 0;
    while v >= BYTES_PER_KIB && unit + 1 < UNITS.len() {
        v /= BYTES_PER_KIB;
        unit += 1;
    }
    format!("{v:.1} {}", UNITS[unit])
}

/// One table's manifest, as stored.
#[derive(Default)]
struct Report {
    /// List plus every part it references.
    manifest_bytes: u64,
    entries: u64,
    /// Entries carrying at least one term bloom: written before the index.
    entries_with_bloom: u64,
    entry_bloom_bytes: u64,
    part_union_bloom_bytes: u64,
    /// Inline open-time bytes on the entries that carry a bloom. Older
    /// writers inlined the FTS dictionary and doc lengths here.
    old_entry_open_blob_bytes: u64,
    /// The same on the entries that carry no bloom, for comparison.
    new_entry_open_blob_bytes: u64,
}

impl Report {
    fn add(&mut self, other: &Report) {
        self.manifest_bytes += other.manifest_bytes;
        self.entries += other.entries;
        self.entries_with_bloom += other.entries_with_bloom;
        self.entry_bloom_bytes += other.entry_bloom_bytes;
        self.part_union_bloom_bytes += other.part_union_bloom_bytes;
        self.old_entry_open_blob_bytes += other.old_entry_open_blob_bytes;
        self.new_entry_open_blob_bytes += other.new_entry_open_blob_bytes;
    }

    fn print(&self, name: &str) {
        let bloom = self.entry_bloom_bytes + self.part_union_bloom_bytes;
        println!(
            "{name}\n  manifest {}, {} entries, {} with blooms\n  \
             blooms {} ({} entries, {} part unions)\n  \
             open blobs {} on entries with blooms, {} on the rest",
            human(self.manifest_bytes),
            self.entries,
            self.entries_with_bloom,
            human(bloom),
            human(self.entry_bloom_bytes),
            human(self.part_union_bloom_bytes),
            human(self.old_entry_open_blob_bytes),
            human(self.new_entry_open_blob_bytes),
        );
    }
}

fn report_table(root: &Path) -> Result<Report, Box<dyn Error>> {
    let pointer = PointerFile::from_bytes(&fs::read(root.join(POINTER_PATH))?)?;
    let list_bytes = fs::read(root.join(&pointer.manifest_uri))?;
    // A name-keyed list or part resolves its summary keys through the table
    // schema, which the list itself carries; decode once to read it, then
    // again with it, or an older table's summaries would be dropped and
    // undercounted.
    let peek = list::decode(&list_bytes, &LegacyNames::none())?;
    let schema = peek
        .schema
        .ok_or("the list predates carrying the schema (no commit since); its names cannot be resolved here")?;
    let legacy = LegacyNames::new(schema, peek.id_column);
    let list = list::decode(&list_bytes, &legacy)?;

    let mut r = Report {
        manifest_bytes: list_bytes.len() as u64,
        ..Report::default()
    };
    for entry in &list.parts {
        r.part_union_bloom_bytes += entry
            .fts_summary_agg
            .values()
            .filter_map(|agg| agg.term_bloom.as_ref())
            .map(|b| b.len() as u64)
            .sum::<u64>();
        let part_bytes = fs::read(root.join(&entry.uri))?;
        r.manifest_bytes += part_bytes.len() as u64;
        for sf in part::decode(&part_bytes, &legacy)?.superfiles {
            let bloom: u64 = sf
                .fts_summary
                .values()
                .filter_map(|s| s.term_bloom.as_ref())
                .map(|b| b.len() as u64)
                .sum();
            let blob: u64 = sf
                .subsection_offsets
                .iter()
                .flat_map(|o| &o.open_blob)
                .map(|(_, bytes)| bytes.len() as u64)
                .sum();
            r.entries += 1;
            if bloom > 0 {
                r.entries_with_bloom += 1;
                r.entry_bloom_bytes += bloom;
                r.old_entry_open_blob_bytes += blob;
            } else {
                r.new_entry_open_blob_bytes += blob;
            }
        }
    }
    Ok(r)
}

fn tables_under(path: &Path, out: &mut Vec<PathBuf>) {
    if path.join(POINTER_PATH).is_file() {
        out.push(path.to_path_buf());
        return;
    }
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = fs::read_dir(path)
            .map(|rd| rd.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        entries.sort();
        for e in entries {
            tables_under(&e, out);
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<PathBuf> = env::args().skip(1).map(PathBuf::from).collect();
    if args.is_empty() {
        eprintln!("usage: manifest_bloom_report <dir>...");
        return ExitCode::from(2);
    }
    let mut tables = Vec::new();
    for a in &args {
        tables_under(a, &mut tables);
    }
    let mut total = Report::default();
    let mut failed = 0;
    for t in &tables {
        match report_table(t) {
            Ok(r) => {
                r.print(&t.display().to_string());
                total.add(&r);
            }
            Err(e) => {
                eprintln!("{}: {e}", t.display());
                failed += 1;
            }
        }
    }
    total.print(&format!(
        "total: {} tables, {failed} unreadable",
        tables.len()
    ));
    if failed > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
