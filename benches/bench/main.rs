// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The single Infino benchmark binary.
//!
//! Everything is selected by positional tokens after `cargo bench --`.
//! A bare `cargo bench` is identical to `cargo bench -- all`.
//!
//! ```text
//! cargo bench                              # everything, all phases
//! cargo bench -- all                       # same as above
//! cargo bench -- supertable                # all 3 supertable modalities
//! cargo bench -- superfile                 # all 3 superfile modalities
//! cargo bench -- superfile fts             # one cell
//! cargo bench -- supertable sql warm       # one cell, one phase
//! cargo bench -- supertable vector build cold
//!
//! # Diagnostics (standalone programs, same binary):
//! cargo bench -- diagnostic              # all five
//! cargo bench -- diagnostic scale        # a subset, grouped
//! cargo bench -- tombstone               # bare names also work
//!
//! # Prepared datasets (supertable, real object store only):
//! cargo bench -- dataset prepare datasets/bench-10m          # ingest + sidecar
//! cargo bench -- dataset bench   datasets/bench-10m vector   # read phases only
//! cargo bench -- dataset run     datasets/bench-10m          # prepare if absent, then bench
//! ```
//!
//! Token vocabulary:
//!   tier        : `superfile` | `supertable`        (omitted => both)
//!   modality    : `fts` | `vector` | `sql`          (omitted => all three)
//!   phase       : `build` | `warm` | `cold` | `search` (= warm+cold)
//!                 (omitted => all three phases)
//!   `all`       : explicit "every tier × modality × phase" (the default).
//!                 Matrix only — diagnostics are NEVER implied by `all` or
//!                 by a bare `cargo bench`.
//!   diagnostic  : `scale` | `tombstone` | `update` | `sql-diag` | `object-store`,
//!                 by name, or grouped under the `diagnostic` keyword —
//!                 `cargo bench -- diagnostic` runs all five,
//!                 `cargo bench -- diagnostic scale tombstone` a subset.
//!
//! The matrix tests run = (selected tiers) × (selected modalities).
//!
//! Scale (`INFINO_BENCH_SUPERFILE_DOCS`, `INFINO_BENCH_SUPERTABLE_DOCS` —
//! plain integers) and object-store backend (`INFINO_BENCH_STORE`) are env
//! knobs.

use infino_bench_utils::supertable::Phases;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier {
    Superfile,
    Supertable,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Modality {
    Fts,
    Vector,
    Sql,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Diagnostic {
    Scale,
    Tombstone,
    Update,
    SqlDiag,
    ObjectStore,
}

impl Diagnostic {
    fn label(self) -> &'static str {
        match self {
            Diagnostic::Scale => "scale",
            Diagnostic::Tombstone => "tombstone",
            Diagnostic::Update => "update",
            Diagnostic::SqlDiag => "sql-diag",
            Diagnostic::ObjectStore => "object-store",
        }
    }

    fn run(self) {
        match self {
            Diagnostic::Scale => infino_bench_utils::scale::run(),
            Diagnostic::Tombstone => infino_bench_utils::tombstone_overhead::run(),
            Diagnostic::Update => infino_bench_utils::supertable_update::run(),
            Diagnostic::SqlDiag => infino_bench_utils::sql_diag::run(),
            Diagnostic::ObjectStore => infino_bench_utils::unified_object_store::run(),
        }
    }
}

fn run_cell(tier: Tier, modality: Modality, phases: Phases) {
    let label = match (tier, modality) {
        (Tier::Superfile, Modality::Fts) => "superfile_fts",
        (Tier::Superfile, Modality::Vector) => "superfile_vector",
        (Tier::Superfile, Modality::Sql) => "superfile_sql",
        (Tier::Supertable, Modality::Fts) => "supertable_fts",
        (Tier::Supertable, Modality::Vector) => "supertable_vector",
        (Tier::Supertable, Modality::Sql) => "supertable_sql",
    };
    eprintln!(
        "[bench] === {label} (build={}, warm={}, cold={}) ===",
        phases.build, phases.warm, phases.cold
    );
    match (tier, modality) {
        (Tier::Superfile, Modality::Fts) => infino_bench_utils::superfile::fts::run(phases),
        (Tier::Superfile, Modality::Vector) => infino_bench_utils::superfile::vector::run(phases),
        (Tier::Superfile, Modality::Sql) => infino_bench_utils::superfile::sql::run(phases),
        (Tier::Supertable, Modality::Fts) => infino_bench_utils::supertable::fts::run(phases),
        (Tier::Supertable, Modality::Vector) => infino_bench_utils::supertable::vector::run(phases),
        (Tier::Supertable, Modality::Sql) => infino_bench_utils::supertable::sql::run(phases),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DatasetVerb {
    Prepare,
    Bench,
    Run,
}

fn dataset_usage_and_exit(code: i32) -> ! {
    eprintln!(
        "Usage:\n  cargo bench -- dataset <prepare|bench|run> <prefix> [fts|vector|sql ...] [warm|cold|search]\n\
         \n\
         prepare : ingest the corpus to <prefix> and write the sidecar (fails if already there)\n\
         bench   : open the dataset at <prefix> and run the read phases (fails if absent)\n\
         run     : prepare if absent, then bench\n\
         \n\
         Modality defaults to all three; phase (bench/run only) defaults to search (warm+cold).\n\
         Doc count: INFINO_BENCH_SUPERTABLE_DOCS. Store: INFINO_BENCH_STORE (s3 | azure).\n"
    );
    std::process::exit(code);
}

fn ingest_modality(m: Modality) -> infino_bench_utils::ingest::supertable::Modality {
    use infino_bench_utils::ingest::supertable::Modality as M;
    match m {
        Modality::Fts => M::Fts,
        Modality::Vector => M::Vector,
        Modality::Sql => M::Sql,
    }
}

/// `dataset <verb> <prefix> [modality ...] [phase ...]` — prepare a reusable
/// dataset on object storage, benchmark an existing one, or both.
fn run_dataset_command(tokens: &[String]) {
    use infino_bench_utils::{dataset, ingest::supertable as ingest, tiers};

    let verb = match tokens.first().map(String::as_str) {
        Some("prepare") => DatasetVerb::Prepare,
        Some("bench") => DatasetVerb::Bench,
        Some("run") => DatasetVerb::Run,
        _ => dataset_usage_and_exit(2),
    };
    let mut prefix: Option<&str> = None;
    let mut modalities: Vec<Modality> = Vec::new();
    let (mut warm, mut cold) = (false, false);
    for tok in &tokens[1..] {
        match tok.as_str() {
            "fts" if !modalities.contains(&Modality::Fts) => modalities.push(Modality::Fts),
            "vector" if !modalities.contains(&Modality::Vector) => {
                modalities.push(Modality::Vector)
            }
            "sql" if !modalities.contains(&Modality::Sql) => modalities.push(Modality::Sql),
            "fts" | "vector" | "sql" => {}
            "warm" => warm = true,
            "cold" => cold = true,
            "search" => {
                warm = true;
                cold = true;
            }
            other if prefix.is_none() => prefix = Some(other),
            other => {
                eprintln!("[dataset] unexpected token {other:?}");
                dataset_usage_and_exit(2);
            }
        }
    }
    let Some(prefix) = prefix else {
        dataset_usage_and_exit(2)
    };
    if let Err(reason) = tiers::supertable_backend_check() {
        eprintln!("[dataset] {reason}");
        std::process::exit(2);
    }
    dataset::set_prefix(prefix);
    if modalities.is_empty() {
        modalities = vec![Modality::Fts, Modality::Vector, Modality::Sql];
    }
    if !(warm || cold) {
        warm = true;
        cold = true;
    }

    for &m in &modalities {
        let dir = ingest_modality(m).dataset_dir();
        let exists = ingest::dataset_exists(ingest_modality(m));
        let phases = match (verb, exists) {
            (DatasetVerb::Prepare, true) => {
                eprintln!(
                    "[dataset] {prefix}/{dir} already exists — bench it or pick a new prefix"
                );
                std::process::exit(1);
            }
            (DatasetVerb::Bench, false) => {
                eprintln!("[dataset] {prefix}/{dir} not found — prepare it first");
                std::process::exit(1);
            }
            (DatasetVerb::Prepare, false) => Phases {
                build: true,
                warm: false,
                cold: false,
            },
            (DatasetVerb::Bench, true) => Phases {
                build: false,
                warm,
                cold,
            },
            (DatasetVerb::Run, exists) => {
                if exists {
                    eprintln!("[dataset] {prefix}/{dir} exists — skipping prepare");
                }
                Phases {
                    build: !exists,
                    warm,
                    cold,
                }
            }
        };
        run_cell(Tier::Supertable, m, phases);
    }
}

fn print_usage_and_exit(code: i32) -> ! {
    eprintln!(
        "Usage:\n  cargo bench -- [tier] [modality] [phase ...]\n  cargo bench -- <diagnostic>\n\
         \x20 cargo bench -- dataset <prepare|bench|run> <prefix> [modality ...] [phase]\n\
         \n\
         Tier      : superfile | supertable        (omitted => both)\n\
         Modality  : fts | vector | sql            (omitted => all three)\n\
         Phase     : build | warm | cold | search  (search = warm+cold; omitted => all)\n\
         all       : every tier x modality x phase (the default for a bare\n\
         \x20           `cargo bench`); matrix only — never implies diagnostics\n\
         Diagnostic: scale | tombstone | update | sql-diag | object-store,\n\
         \x20           or `diagnostic` for all five / `diagnostic <names>` for a subset\n\
         \n\
         Examples:\n\
         \x20 cargo bench\n\
         \x20 cargo bench -- supertable\n\
         \x20 cargo bench -- superfile fts\n\
         \x20 cargo bench -- supertable sql warm\n\
         \x20 cargo bench -- tombstone\n"
    );
    std::process::exit(code);
}

struct Selection {
    tiers: Vec<Tier>,
    modalities: Vec<Modality>,
    phases: Phases,
    diagnostics: Vec<Diagnostic>,
    /// Explicit `all` token.
    want_all: bool,
    /// Any of `build` / `warm` / `cold` / `search` was given.
    phase_selected: bool,
    /// No tokens at all → the bare `cargo bench` "run everything" case.
    empty: bool,
}

fn parse_args() -> Selection {
    // Drop harness flags (e.g. a stray `--bench`); only positional tokens
    // are ours.
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .collect();

    if std::env::args().any(|a| matches!(a.as_str(), "help" | "-h" | "--help")) {
        print_usage_and_exit(0);
    }

    let mut tiers: Vec<Tier> = Vec::new();
    let mut modalities: Vec<Modality> = Vec::new();
    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    let mut build = false;
    let mut warm = false;
    let mut cold = false;
    let mut want_all = false;
    let mut want_diagnostics = false;
    let mut unknown: Vec<String> = Vec::new();

    let push_tier = |t: Tier, tiers: &mut Vec<Tier>| {
        if !tiers.contains(&t) {
            tiers.push(t);
        }
    };

    for arg in &args {
        match arg.as_str() {
            "all" => want_all = true,
            "superfile" => push_tier(Tier::Superfile, &mut tiers),
            "supertable" => push_tier(Tier::Supertable, &mut tiers),
            "fts" => {
                if !modalities.contains(&Modality::Fts) {
                    modalities.push(Modality::Fts);
                }
            }
            "vector" => {
                if !modalities.contains(&Modality::Vector) {
                    modalities.push(Modality::Vector);
                }
            }
            "sql" => {
                if !modalities.contains(&Modality::Sql) {
                    modalities.push(Modality::Sql);
                }
            }
            "build" => build = true,
            "warm" => warm = true,
            "cold" => cold = true,
            "search" => {
                warm = true;
                cold = true;
            }
            "scale" => diagnostics.push(Diagnostic::Scale),
            "tombstone" | "tombstone-overhead" => diagnostics.push(Diagnostic::Tombstone),
            "update" | "supertable-update" => diagnostics.push(Diagnostic::Update),
            "sql-diag" | "sql_diag" => diagnostics.push(Diagnostic::SqlDiag),
            "object-store" | "object_store" => diagnostics.push(Diagnostic::ObjectStore),
            "diagnostic" | "diagnostics" => want_diagnostics = true,
            other => unknown.push(other.to_string()),
        }
    }

    if !unknown.is_empty() {
        eprintln!("[bench] unknown selector(s): {}", unknown.join(", "));
        print_usage_and_exit(2);
    }

    // Bare `diagnostic` (no names) selects every diagnostic; with names it
    // is a plain grouping word (`diagnostic scale tombstone`). Keeps the
    // matrix vocabulary (`all`, tiers, modalities) disjoint from the
    // diagnostics namespace.
    if want_diagnostics && diagnostics.is_empty() {
        diagnostics = vec![
            Diagnostic::Scale,
            Diagnostic::Tombstone,
            Diagnostic::Update,
            Diagnostic::SqlDiag,
            Diagnostic::ObjectStore,
        ];
    }

    let phase_selected = build || warm || cold;
    let phases = if phase_selected {
        Phases { build, warm, cold }
    } else {
        Phases::ALL
    };

    Selection {
        tiers,
        modalities,
        phases,
        diagnostics,
        want_all,
        phase_selected,
        empty: args.is_empty(),
    }
}

fn main() {
    // Isolated per-shape supertable ingest child (`INFINO_BENCH_SUPERTABLE_SHAPE`).
    if infino_bench_utils::supertable::handle_shape_child_from_env() {
        return;
    }

    // `dataset <verb> ...` is its own grammar, separate from the matrix.
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .collect();
    if args.first().map(String::as_str) == Some("dataset") {
        run_dataset_command(&args[1..]);
        return;
    }

    let sel = parse_args();

    // Diagnostics are standalone programs that share this binary.
    for diag in &sel.diagnostics {
        eprintln!("[bench] === {} ===", diag.label());
        diag.run();
    }

    // Decide whether to run the tier × modality matrix. A bare
    // `cargo bench` (no tokens) runs everything; otherwise the matrix
    // runs when any matrix token was given (`all`, a tier, a modality,
    // or a phase). A pure-diagnostic invocation (only diagnostic tokens)
    // skips the matrix.
    let run_matrix = sel.empty
        || sel.want_all
        || !sel.tiers.is_empty()
        || !sel.modalities.is_empty()
        || sel.phase_selected;

    if !run_matrix {
        return;
    }

    let tiers = if sel.tiers.is_empty() {
        vec![Tier::Superfile, Tier::Supertable]
    } else {
        sel.tiers.clone()
    };
    let modalities = if sel.modalities.is_empty() {
        vec![Modality::Fts, Modality::Vector, Modality::Sql]
    } else {
        sel.modalities.clone()
    };

    for tier in tiers {
        for &modality in &modalities {
            run_cell(tier, modality, sel.phases);
        }
    }
}
