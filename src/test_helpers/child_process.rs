// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Runs one test alone in a child process, for a test that measures or blocks the whole process
//! (its RSS, the shared runtime's workers), which tests running beside it would disturb or be
//! disturbed by.

use std::{env, process::Command};

/// What the test harness prints when it ran one test and it passed.
const ONE_PASSED: &str = "1 passed";

/// In the test process, reruns the test at `path` alone in a child process with `env_var` set,
/// asserts it passed, and returns `false`; in the child, returns `true` so the test runs its body.
/// `path` is the test's `module_path!()` path, crate first.
pub(crate) fn in_child_process(path: &str, env_var: &str) -> bool {
    if env::var_os(env_var).is_some() {
        return true;
    }
    // The harness names a test by its path below the crate.
    let (_, name) = path.split_once("::").expect("a crate-rooted path");
    let output = Command::new(env::current_exe().expect("current_exe"))
        .args(["--exact", "--test-threads=1", "--nocapture", name])
        .env(env_var, "1")
        .output()
        .expect("spawn child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    assert!(
        output.status.success() && stdout.contains(ONE_PASSED),
        "the child failed: {stdout}"
    );
    false
}
