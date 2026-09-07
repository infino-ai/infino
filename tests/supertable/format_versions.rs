// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `Supertable::format_versions` over durable storage: the manifest facts
//! come from the persisted list, every committed superfile is reported, and
//! the report costs only footer and header reads — no dictionary, posting
//! list, or row group is fetched.

use std::sync::Arc;

use infino::{
    OptionsHashRule,
    supertable::{
        Supertable,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};
use tempfile::TempDir;

/// Commits, one superfile each.
const N_COMMITS: u64 = 4;

/// Object reads the report may issue per superfile: one footer read (the
/// size is known from the manifest, so the tail is a single ranged GET) and
/// one header probe for the full-text section.
const GETS_PER_SUPERFILE: u64 = 2;

/// Object reads the report may issue outside the superfiles: the manifest
/// pointer refresh and the list and part loads behind it.
const GETS_FOR_MANIFEST: u64 = 4;

/// Speculative footer tail the reader fetches per superfile.
const FOOTER_TAIL_BYTES: u64 = 64 * 1024;

/// Bytes a header probe reads: eight-byte magic plus the version word.
const HEADER_PROBE_BYTES: u64 = 12;

/// Slack for the manifest pointer, list, and parts, which are small.
const MANIFEST_BYTES_SLACK: u64 = 256 * 1024;

fn commit_titles(st: &Supertable, titles: &[&str]) {
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(titles)).expect("append");
    w.commit().expect("commit");
}

#[test]
fn durable_table_reports_manifest_and_superfiles_within_a_header_read_budget() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    for i in 0..N_COMMITS {
        let title = format!("document number {i}");
        commit_titles(&st, &[title.as_str(), "filler alpha", "filler bravo"]);
    }

    let before = storage.usage_meter().snapshot();
    let report = st.format_versions().expect("format_versions");
    let after = storage.usage_meter().snapshot();

    let manifest = report
        .manifest
        .as_ref()
        .expect("durable table has a persisted list");
    assert_eq!(manifest.options_hash_rule, OptionsHashRule::Current);
    assert!(manifest.current, "{manifest:?}");

    assert_eq!(report.superfiles.len() as u64, N_COMMITS);
    assert!(report.is_current(), "{report:?}");
    assert!(report.superfiles.iter().all(|r| r.size_bytes > 0));
    assert!(report.superfiles.iter().all(|r| r.fts_version.is_some()));

    // The report never touches a dictionary, a posting list, or a row group:
    // its reads fit a footer-and-header budget per superfile.
    let gets = after.get_count - before.get_count;
    let bytes = after.get_bytes - before.get_bytes;
    assert!(
        gets <= N_COMMITS * GETS_PER_SUPERFILE + GETS_FOR_MANIFEST,
        "{gets} object reads for {N_COMMITS} superfiles"
    );
    assert!(
        bytes <= N_COMMITS * (FOOTER_TAIL_BYTES + HEADER_PROBE_BYTES) + MANIFEST_BYTES_SLACK,
        "{bytes} bytes read for {N_COMMITS} superfiles"
    );

    // A second call is the same picture.
    assert_eq!(st.format_versions().expect("again"), report);
}
