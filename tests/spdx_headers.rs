// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! License-header gate. Every Rust source file in the crate must carry
//! the SPDX identifier near the top; this runs under plain `cargo test`
//! so CI rejects any new file that ships without it.

use std::path::{Path, PathBuf};

/// The line every `.rs` file must declare, within its opening lines.
const EXPECTED_SPDX: &str = "// SPDX-License-Identifier: Apache-2.0";
/// Crate subtrees scanned for source files.
const SCAN_DIRS: &[&str] = &["src", "benches", "examples", "tests"];
/// How many leading lines may precede the SPDX line (shebang / blank
/// lines / tooling directives); the header is always at the very top in
/// this repo, so a small window is plenty.
const HEADER_WINDOW_LINES: usize = 5;

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn every_rust_file_carries_the_spdx_header() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for dir in SCAN_DIRS {
        collect_rs_files(&root.join(dir), &mut files);
    }
    assert!(
        !files.is_empty(),
        "no .rs files found under {SCAN_DIRS:?} — scan paths are wrong"
    );

    let mut missing = Vec::new();
    for path in &files {
        let contents = std::fs::read_to_string(path).expect("read source file");
        let has_header = contents
            .lines()
            .take(HEADER_WINDOW_LINES)
            .any(|line| line.trim_end() == EXPECTED_SPDX);
        if !has_header {
            let rel = path.strip_prefix(root).unwrap_or(path);
            missing.push(rel.display().to_string());
        }
    }

    assert!(
        missing.is_empty(),
        "{} source file(s) missing the `{EXPECTED_SPDX}` header:\n{}",
        missing.len(),
        missing.join("\n")
    );
}
