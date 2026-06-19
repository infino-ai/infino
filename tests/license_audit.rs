// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! License audit: fails the test suite if any dependency in the
//! resolved tree is licensed under anything that isn't either:
//!
//!   1. An option in [`ALLOWED_LICENSES`] (the license-class
//!      allowlist — currently MIT, Apache-2.0, and a curated set
//!      of cloud-deploy- and MIT/Apache-release-friendly permissive
//!      licenses). A multi-license expression like
//!      `Zlib OR Apache-2.0 OR MIT` passes if **any** option is on
//!      the list — that's how SPDX-OR semantics work.
//!
//!   2. An entry in [`ALLOWED_PACKAGES`] (the per-package allow
//!      list). For specific (name, version) pairs whose license
//!      isn't on the class list but has been individually reviewed
//!      and accepted. Each entry's comment MUST name the actual
//!      license so a reviewer auditing the dep tree can judge each
//!      acceptance without re-deriving from `cargo metadata`.
//!
//! Three tests run:
//!
//!   1. [`every_dep_has_an_allowed_license_or_is_in_the_allow_list`]
//!      — fails if a new dep slips in without an allowed-license
//!      option and isn't on the per-package list.
//!   2. [`unused_allow_list_entries_should_be_pruned`] — fails if
//!      an [`ALLOWED_PACKAGES`] entry no longer matches anything
//!      in the tree (likely a version bump dropped it). Forces the
//!      per-package list to stay tight.
//!   3. [`license_parser_handles_common_separators`] — sanity
//!      check on the SPDX expression parser.
//!
//! Run via the regular `cargo test`; `cargo metadata` is invoked
//! as a subprocess.

use std::{collections::HashSet, process::Command};

#[derive(serde::Deserialize)]
struct Metadata {
    packages: Vec<Package>,
}

#[derive(serde::Deserialize)]
struct Package {
    name: String,
    version: String,
    license: Option<String>,
}

/// License-class allowlist. Any package whose SPDX expression
/// contains at least one of these options as an `OR` choice is
/// accepted unconditionally. All entries here are reviewed to be:
///
///   - **Cloud-deploy friendly.** No clause that turns network use
///     into source-disclosure (rules out AGPL / SSPL / Commons
///     Clause / BUSL).
///   - **Compatible with releasing infino under MIT or Apache-2.0.**
///     The license imposes attribution-only (or weaker) obligations
///     on consumers and lets us pick our own license for our own
///     code.
///
/// MPL-2.0 is intentionally NOT here. It's weakly copyleft and
/// only acceptable under a per-package review (see the
/// [`ALLOWED_PACKAGES`] entry for `option-ext`). Future MPL-2.0
/// deps therefore land in the failing-test offenders list and
/// force a reviewer to add a per-package entry with reasoning
/// rather than slipping in via the class list.
const ALLOWED_LICENSES: &[&str] = &[
    // The two intended targets.
    "MIT",
    "Apache-2.0",
    // BSD family — attribution; no copyleft. 3-clause adds a
    // no-endorsement clause; 2-clause is attribution-only;
    // 0-clause is public-domain-equivalent.
    "BSD-2-Clause",
    "BSD-3-Clause",
    "0BSD",
    // ISC — functionally MIT/BSD-2-equivalent.
    "ISC",
    // Zlib — preserve notice + don't misrepresent authorship +
    // mark modifications. No copyleft.
    "Zlib",
    // Unicode-3.0 — used by ICU project crates. Permissive.
    "Unicode-3.0",
    // BSL-1.0 (Boost) — permissive; notice obligation only on
    // source distribution, not binary, so SaaS-friendly.
    "BSL-1.0",
    // CC0-1.0 — public-domain dedication; no obligations.
    "CC0-1.0",
    // bzip2-1.0.6 — BSD-style permissive license shipped with
    // the bzip2 source release.
    "bzip2-1.0.6",
    // Unlicense — public-domain-equivalent.
    "Unlicense",
];

/// Per-package allow list for licenses outside [`ALLOWED_LICENSES`]
/// that have been individually reviewed and accepted. Each entry
/// MUST name the actual license in the comment.
const ALLOWED_PACKAGES: &[(&str, &str, &str)] = &[
    // Our own crate — license field unset in Cargo.toml. Set
    // `license = "Apache-2.0"` (or the chosen license) before any
    // open-source release.
    (
        "infino",
        "0.1.0",
        "UNSPECIFIED — local crate, set in Cargo.toml before OSS release",
    ),
    // Our own crate — license field unset in Cargo.toml. Set
    // `license = "Apache-2.0"` (or the chosen license) before any
    // open-source release.
    (
        "infino-bench-utils",
        "0.1.0",
        "UNSPECIFIED — local crate, set in Cargo.toml before OSS release",
    ),
];

/// True iff at least one option in the SPDX expression is exactly
/// one of [`ALLOWED_LICENSES`]. Handles `OR`, `AND`, parentheses,
/// and the older slash separator (`MIT/Apache-2.0`).
///
/// `Apache-2.0 WITH LLVM-exception` matches via the bare
/// `Apache-2.0` token. `MIT-0` is intentionally NOT matched as
/// `MIT` — it's a different SPDX id.
fn license_has_allowed_option(license: &str) -> bool {
    let normalized = license
        .replace(['(', ')'], " ")
        .replace(" OR ", " ")
        .replace(" AND ", " ")
        .replace('/', " ");
    normalized
        .split_whitespace()
        .any(|tok| ALLOWED_LICENSES.contains(&tok))
}

fn read_metadata() -> Metadata {
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1"])
        .output()
        .expect("cargo metadata invocation");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("cargo metadata returned valid JSON")
}

#[test]
fn every_dep_has_an_allowed_license_or_is_in_the_allow_list() {
    let meta = read_metadata();

    let allowed_packages: HashSet<(&str, &str)> = ALLOWED_PACKAGES
        .iter()
        .map(|(name, version, _reason)| (*name, *version))
        .collect();

    let mut offenders: Vec<(String, String, String)> = Vec::new();
    for p in &meta.packages {
        if allowed_packages.contains(&(p.name.as_str(), p.version.as_str())) {
            continue;
        }
        let license_str = p.license.as_deref().unwrap_or("UNSPECIFIED");
        if !license_has_allowed_option(license_str) {
            offenders.push((p.name.clone(), p.version.clone(), license_str.to_string()));
        }
    }

    if !offenders.is_empty() {
        let mut msg = String::new();
        msg.push_str(
            "\nDependencies whose license is not on the license-class allow list\n\
             (ALLOWED_LICENSES) and whose (name, version) is not on the per-package\n\
             allow list (ALLOWED_PACKAGES):\n\n",
        );
        for (name, version, lic) in &offenders {
            msg.push_str(&format!("  {name} {version} :: {lic}\n"));
        }
        msg.push_str(
            "\nIf the license is broadly safe (permissive, cloud-deploy friendly,\n\
             compatible with an MIT/Apache release), add it to ALLOWED_LICENSES in\n\
             tests/license_audit.rs with a one-line comment naming the obligation.\n\n\
             If only this specific package should be allowed (typical for copyleft\n\
             licenses), add a (name, version, reason) entry to ALLOWED_PACKAGES.\n\
             The reason MUST name the actual license. Pay special attention to\n\
             copyleft licenses — GPL / AGPL / LGPL / MPL / SSPL / BUSL / EPL /\n\
             CDDL all impose obligations that may be incompatible with this\n\
             project's licensing goals.\n",
        );
        panic!("{msg}");
    }
}

#[test]
fn unused_allow_list_entries_should_be_pruned() {
    let meta = read_metadata();

    let live: HashSet<(&str, &str)> = meta
        .packages
        .iter()
        .map(|p| (p.name.as_str(), p.version.as_str()))
        .collect();

    let stale: Vec<&(&str, &str, &str)> = ALLOWED_PACKAGES
        .iter()
        .filter(|(name, version, _)| !live.contains(&(*name, *version)))
        .collect();

    if !stale.is_empty() {
        let mut msg = String::new();
        msg.push_str(
            "\nALLOWED_PACKAGES entries that no longer match anything in the dep tree\n\
             (a version bump or a dropped dep removed them). Remove from\n\
             ALLOWED_PACKAGES in tests/license_audit.rs to keep the allow list tight:\n\n",
        );
        for (name, version, _reason) in &stale {
            msg.push_str(&format!("  {name} {version}\n"));
        }
        panic!("{msg}");
    }
}

#[test]
fn license_parser_handles_common_separators() {
    // Sanity check on `license_has_allowed_option`'s SPDX expression
    // parser.
    assert!(license_has_allowed_option("MIT"));
    assert!(license_has_allowed_option("Apache-2.0"));
    assert!(license_has_allowed_option("MIT OR Apache-2.0"));
    assert!(license_has_allowed_option("Apache-2.0 OR MIT"));
    assert!(license_has_allowed_option("MIT/Apache-2.0"));
    assert!(license_has_allowed_option("Apache-2.0/MIT"));
    assert!(license_has_allowed_option("Unlicense OR MIT"));
    assert!(license_has_allowed_option("Zlib OR Apache-2.0 OR MIT"));
    assert!(license_has_allowed_option(
        "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT"
    ));
    assert!(license_has_allowed_option("BSD-3-Clause AND MIT"));
    assert!(license_has_allowed_option("BSD-3-Clause"));
    assert!(license_has_allowed_option("ISC"));
    assert!(license_has_allowed_option("Unicode-3.0"));

    // Copyleft-family licenses are NOT on ALLOWED_LICENSES; if any
    // future dep brings one in it must be reviewed via
    // ALLOWED_PACKAGES with explicit reasoning.
    assert!(!license_has_allowed_option("MPL-2.0"));
    assert!(!license_has_allowed_option("GPL-3.0"));
    assert!(!license_has_allowed_option("AGPL-3.0"));
    assert!(!license_has_allowed_option("LGPL-2.1-or-later"));
    // MIT-0 is intentionally NOT matched as MIT — it's a different
    // SPDX id (MIT No Attribution). With no other allowed option in
    // the expression, the parser correctly says "not allowed".
    assert!(!license_has_allowed_option("MIT-0"));
    assert!(!license_has_allowed_option("MIT-0 OR MPL-2.0"));
}
