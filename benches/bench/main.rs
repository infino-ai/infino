// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Unified Infino bench entry point.
//!
//! Select one or more tier×modality tests, then optionally select phases:
//!
//! ```text
//! cargo bench --bench bench
//! cargo bench --bench bench -- superfile_fts
//! cargo bench --bench bench -- supertable_sql hot
//! cargo bench --bench bench -- superfile_vector supertable_vector build cold
//! ```
//!
//! Scale (`INFINO_BENCH_SUPERFILE_DOCS`, `INFINO_BENCH_SUPERTABLE_DOCS`),
//! object-store backend (`INFINO_BENCH_STORE`), and tombstone/update
//! diagnostics remain separate knobs / benches.

use infino_bench_utils::supertable::Phases;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Test {
    SuperfileFts,
    SuperfileVector,
    SuperfileSql,
    SupertableFts,
    SupertableVector,
    SupertableSql,
}

impl Test {
    const ALL: [Test; 6] = [
        Test::SuperfileFts,
        Test::SuperfileVector,
        Test::SuperfileSql,
        Test::SupertableFts,
        Test::SupertableVector,
        Test::SupertableSql,
    ];

    fn key(self) -> &'static str {
        match self {
            Test::SuperfileFts => "superfile_fts",
            Test::SuperfileVector => "superfile_vector",
            Test::SuperfileSql => "superfile_sql",
            Test::SupertableFts => "supertable_fts",
            Test::SupertableVector => "supertable_vector",
            Test::SupertableSql => "supertable_sql",
        }
    }

    fn from_arg(arg: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|test| test.key() == arg)
    }

    fn run(self, phases: Phases) {
        match self {
            Test::SuperfileFts => infino_bench_utils::superfile::fts::run(phases),
            Test::SuperfileVector => infino_bench_utils::superfile::vector::run(phases),
            Test::SuperfileSql => infino_bench_utils::superfile::sql::run(phases),
            Test::SupertableFts => infino_bench_utils::supertable::fts::run(phases),
            Test::SupertableVector => infino_bench_utils::supertable::vector::run(phases),
            Test::SupertableSql => infino_bench_utils::supertable::sql::run(phases),
        }
    }
}

fn parse_args() -> (Vec<Test>, Phases) {
    let mut tests = Vec::new();
    let mut build = false;
    let mut hot = false;
    let mut cold = false;

    for arg in std::env::args().skip(1).filter(|arg| !arg.starts_with('-')) {
        if let Some(test) = Test::from_arg(&arg) {
            if !tests.contains(&test) {
                tests.push(test);
            }
            continue;
        }

        match arg.as_str() {
            "build" => build = true,
            "hot" => hot = true,
            "cold" => cold = true,
            "search" => {
                hot = true;
                cold = true;
            }
            other => {
                eprintln!("[bench] ignoring unknown selector {other:?}");
            }
        }
    }

    if tests.is_empty() {
        tests.extend(Test::ALL);
    }

    let phases = if build || hot || cold {
        Phases { build, hot, cold }
    } else {
        Phases::ALL
    };

    (tests, phases)
}

fn main() {
    let (tests, phases) = parse_args();
    for test in tests {
        eprintln!(
            "[bench] === {} (build={}, hot={}, cold={}) ===",
            test.key(),
            phases.build,
            phases.hot,
            phases.cold
        );
        test.run(phases);
    }
}
