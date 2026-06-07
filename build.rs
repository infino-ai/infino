// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use std::process::Command;

/// Number of hex characters in the embedded git short hash
/// (`INFINO_GIT_HASH`).
const GIT_REV_SHORT_LEN: usize = 12;

fn main() {
    let hash = Command::new("git")
        .args(["rev-parse", &format!("--short={GIT_REV_SHORT_LEN}"), "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    let dirty = Command::new("git")
        .args(["diff-index", "--quiet", "HEAD", "--"])
        .status()
        .map(|s| !s.success())
        .unwrap_or(false);
    let suffix = if dirty { "-dirty" } else { "" };

    println!("cargo:rustc-env=INFINO_GIT_HASH={hash}{suffix}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
}
