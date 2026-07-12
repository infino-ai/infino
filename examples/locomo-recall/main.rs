// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Reproduce — and debug — infino's retrieval on the LOCOMO long-term-memory
//! benchmark, against the engine core, with no embedder in the loop.
//!
//! `embed.mjs` writes a fixture once: the corpus and every question already
//! embedded (`fixture.json`). This program loads those FROZEN vectors, ingests
//! them into an in-memory infino table, and runs each question through all three
//! retrieval modes — vector kNN, BM25 keyword, and the native single-pass
//! `hybrid_search` SQL function — scoring whether the gold-supporting memory
//! (LOCOMO's `evidence`) lands in the top-k.
//!
//! Because the vectors never change between runs, the ONLY variable is the
//! engine code. So after you change BM25 scoring, the vector codec, or the
//! hybrid fusion, re-running this tells you — apples to apples — whether a
//! specific miss is fixed and whether anything regressed.
//!
//! ```text
//! cargo run --example locomo-recall                 # full report: recall@k + every miss
//! cargo run --example locomo-recall -- --id=D6:3    # focus the case(s) whose evidence is D6:3
//! cargo run --example locomo-recall -- --case=42    # focus one question by index
//! cargo run --example locomo-recall -- --fail-under=0.68   # exit non-zero if hybrid recall@10 drops below a tolerance floor (CI gate)
//! ```
//!
//! Determinism: vector and keyword modes are stable run-to-run on a fixed
//! fixture; hybrid jitters ~1pt (recall@10) because RRF tie-break ordering isn't
//! deterministic, so a CI `--fail-under` should sit a few points below baseline.
//!
//! A LOCOMO id like `D6:3` is the dataset's own `dia_id` — session 6, turn 3 of
//! the conversation. We use it verbatim as the memory id, and the fixture
//! carries each id's text, so every id printed below is resolved to its source
//! turn — no cross-referencing the raw dataset.

use std::{
    collections::{HashMap, HashSet},
    env,
    error::Error,
    fs::File,
    io::BufReader,
    process::exit,
    sync::Arc,
};

use infino::{
    BoolMode, IndexSpec, Metric, VectorSearchOptions,
    arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch},
    arrow_schema::{DataType, Field, Schema},
    connect,
};
use serde::Deserialize;

/// Top-k retrieved per query (headline is recall@10).
const DEFAULT_K: usize = 10;
/// recall@ cutoffs to report (filtered to those <= k).
const CUTOFFS: [usize; 4] = [1, 3, 5, 10];
/// The cutoff the `--fail-under` gate is calibrated against — the published
/// baseline is recall@10, so the floor only means anything when at least this
/// many rows were retrieved (we require `k >= GATE_CUTOFF` when a floor is set).
const GATE_CUTOFF: usize = 10;
/// IVF centroid count. The canonical slice is one ~400-row conversation — a
/// single list searches it exhaustively, so vector recall isn't itself lossy
/// and a miss is attributable to ranking/fusion, not ANN approximation.
const N_CENTROIDS: usize = 1;
/// The three retrieval modes, hybrid first (the default a memory product uses).
const MODES: [&str; 3] = ["hybrid", "vector", "keyword"];
/// Truncate resolved memory text to this many chars when printing.
const TEXT_WIDTH: usize = 96;

#[derive(Deserialize)]
struct Mem {
    id: String,
    text: String,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct Case {
    question: String,
    #[serde(default)]
    category: String,
    #[serde(default)]
    gold: String,
    #[serde(rename = "queryVector")]
    query_vector: Vec<f32>,
    expected: Vec<String>,
}

#[derive(Deserialize)]
struct Fixture {
    dim: usize,
    #[serde(default)]
    embedder: String,
    #[serde(default)]
    conversation: String,
    corpus: Vec<Mem>,
    cases: Vec<Case>,
}

/// Per-question retrieval result: the ranked id list each mode returned.
struct Scored<'a> {
    case: &'a Case,
    expected_present: Vec<String>, // evidence ids that exist in the corpus
    ranked: HashMap<&'static str, Vec<String>>, // mode -> ids best-first
}

fn arg_map() -> HashMap<String, String> {
    let mut m = HashMap::new();
    for a in env::args().skip(1) {
        if let Some(rest) = a.strip_prefix("--") {
            let mut it = rest.splitn(2, '=');
            let k = it.next().unwrap_or("").to_string();
            let v = it.next().unwrap_or("true").to_string();
            m.insert(k, v);
        }
    }
    m
}

/// Arrow type for a `dim`-wide Float32 vector column.
fn vector_field(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// Pull the `id` column out of search results, preserving rank order.
fn ids_in_order(batches: &[RecordBatch]) -> Result<Vec<String>, Box<dyn Error>> {
    let mut out = Vec::new();
    for b in batches {
        let col = b
            .column_by_name("id")
            .ok_or("search result has no `id` column")?;
        let ids = col
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or("`id` column is not LargeUtf8")?;
        for i in 0..ids.len() {
            out.push(ids.value(i).to_string());
        }
    }
    Ok(out)
}

/// Escape a string for a single-quoted SQL literal.
fn sql_lit(s: &str) -> String {
    s.replace('\'', "''")
}

/// 1-based rank of each id in a ranked list (first occurrence wins).
fn positions(ranked: &[String]) -> HashMap<&str, usize> {
    let mut pos = HashMap::new();
    for (i, id) in ranked.iter().enumerate() {
        pos.entry(id.as_str()).or_insert(i + 1);
    }
    pos
}

fn truncate(s: &str) -> String {
    if s.chars().count() <= TEXT_WIDTH {
        s.to_string()
    } else {
        let cut: String = s.chars().take(TEXT_WIDTH).collect();
        format!("{cut}…")
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = arg_map();
    let k: usize = args
        .get("k")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_K);
    let cutoffs: Vec<usize> = CUTOFFS.into_iter().filter(|c| *c <= k).collect();
    let fixture_path = args.get("fixture").cloned().unwrap_or_else(|| {
        format!(
            "{}/examples/locomo-recall/fixture.json",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let focus_id = args.get("id").cloned();
    let focus_case: Option<usize> = args.get("case").and_then(|s| s.parse().ok());
    let fail_under: Option<f64> = args.get("fail-under").and_then(|s| s.parse().ok());
    if fail_under.is_some() && k < GATE_CUTOFF {
        return Err(format!(
            "--fail-under is calibrated on recall@{GATE_CUTOFF}; pass --k>={GATE_CUTOFF} (got k={k})"
        )
        .into());
    }

    // --- load the frozen fixture ---------------------------------------------
    let file = File::open(&fixture_path).map_err(|e| {
        format!("open {fixture_path}: {e} (generate it with examples/locomo-recall/embed.mjs)")
    })?;
    let fx: Fixture = serde_json::from_reader(BufReader::new(file))?;
    let dim = fx.dim;
    for m in &fx.corpus {
        if m.vector.len() != dim {
            return Err(format!(
                "memory {} has {} dims, fixture declares {dim}",
                m.id,
                m.vector.len()
            )
            .into());
        }
    }
    eprintln!(
        "fixture: {} · {} · {} memories · {} questions · k={k}",
        fx.conversation,
        fx.embedder,
        fx.corpus.len(),
        fx.cases.len()
    );

    // --- build the table with the EXACT fixture vectors ----------------------
    let db = connect("memory://")?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::LargeUtf8, false),
        Field::new("text", DataType::LargeUtf8, false),
        Field::new("vector", vector_field(dim), false),
    ]));
    let table = db.create_table(
        "mem",
        schema.clone(),
        IndexSpec::new()
            .fts("text")
            .vector("vector", dim, N_CENTROIDS, Metric::Cosine),
    )?;
    let ids = LargeStringArray::from(fx.corpus.iter().map(|m| m.id.as_str()).collect::<Vec<_>>());
    let texts = LargeStringArray::from(
        fx.corpus
            .iter()
            .map(|m| m.text.as_str())
            .collect::<Vec<_>>(),
    );
    let mut flat = Vec::with_capacity(fx.corpus.len() * dim);
    for m in &fx.corpus {
        flat.extend_from_slice(&m.vector);
    }
    let vectors = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(Float32Array::from(flat)) as Arc<dyn Array>,
        None,
    )?;
    table.append(&RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(texts), Arc::new(vectors)],
    )?)?;

    let corpus_ids: HashSet<&str> = fx.corpus.iter().map(|m| m.id.as_str()).collect();
    let text_of: HashMap<&str, &str> = fx
        .corpus
        .iter()
        .map(|m| (m.id.as_str(), m.text.as_str()))
        .collect();

    // --- retrieve every question through all three modes ---------------------
    let mut scored: Vec<Scored> = Vec::new();
    for case in &fx.cases {
        let csv = case
            .query_vector
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let vector = ids_in_order(&table.vector_search(
            "vector",
            &case.query_vector,
            k,
            VectorSearchOptions::new(),
            None,
            Some(&["id", "score"]),
        )?)?;
        let keyword = ids_in_order(&table.bm25_search(
            "text",
            &case.question,
            k,
            BoolMode::Or,
            Some(&["id", "score"]),
        )?)?;
        let hybrid = ids_in_order(&db.query_sql(&format!(
            "SELECT id, score FROM hybrid_search('mem', 'text', '{}', 'vector', '{}', {k}) ORDER BY score DESC",
            sql_lit(&case.question),
            csv,
        ))?)?;

        let mut ranked: HashMap<&'static str, Vec<String>> = HashMap::new();
        ranked.insert("hybrid", hybrid);
        ranked.insert("vector", vector);
        ranked.insert("keyword", keyword);
        let expected_present = case
            .expected
            .iter()
            .filter(|e| corpus_ids.contains(e.as_str()))
            .cloned()
            .collect();
        scored.push(Scored {
            case,
            expected_present,
            ranked,
        });
    }

    // --- recall@k + MRR per mode ---------------------------------------------
    println!("\n=== recall (n = questions with evidence in the corpus) ===");
    let header_cuts: Vec<String> = cutoffs.iter().map(|c| format!("r@{c}")).collect();
    println!(
        "  {:<9}{:>8}  {:>8}",
        "mode",
        header_cuts.join("  "),
        "mrr  n"
    );
    for mode in MODES {
        let mut recall_sum = vec![0.0f64; cutoffs.len()];
        let mut mrr_sum = 0.0f64;
        let mut n = 0usize;
        for s in &scored {
            if s.expected_present.is_empty() {
                continue; // adversarial / no-evidence — not scored
            }
            n += 1;
            let pos = positions(&s.ranked[mode]);
            for (ci, c) in cutoffs.iter().enumerate() {
                let hit = s
                    .expected_present
                    .iter()
                    .filter(|e| pos.get(e.as_str()).is_some_and(|p| p <= c))
                    .count();
                recall_sum[ci] += hit as f64 / s.expected_present.len() as f64;
            }
            let first = s
                .expected_present
                .iter()
                .filter_map(|e| pos.get(e.as_str()))
                .min();
            mrr_sum += first.map_or(0.0, |r| 1.0 / *r as f64);
        }
        let denom = n.max(1) as f64;
        let cells: Vec<String> = recall_sum
            .iter()
            .map(|v| format!("{:.3}", v / denom))
            .collect();
        println!(
            "  {:<9}{:>8}  {:>6.3}  {n}",
            mode,
            cells.join("  "),
            mrr_sum / denom
        );
    }

    // --- the drill-down: which questions miss, and why -----------------------
    let scored_n = scored
        .iter()
        .filter(|s| !s.expected_present.is_empty())
        .count();
    let mut gate_recall_sum = 0.0f64;

    let selected: Vec<(usize, &Scored)> = scored
        .iter()
        .enumerate()
        .filter(|(i, s)| {
            if let Some(ci) = focus_case {
                return *i == ci;
            }
            if let Some(id) = &focus_id {
                return s.case.expected.iter().any(|e| e == id);
            }
            // default: show only the hybrid misses
            if s.expected_present.is_empty() {
                return false;
            }
            let pos = positions(&s.ranked["hybrid"]);
            !s.expected_present
                .iter()
                .all(|e| pos.contains_key(e.as_str()))
        })
        .collect();

    let heading = match (&focus_id, focus_case) {
        (Some(id), _) => format!("=== cases whose evidence includes {id} ==="),
        (_, Some(ci)) => format!("=== case #{ci} ==="),
        _ => "=== hybrid misses (evidence not fully in top-k) ===".to_string(),
    };
    println!("\n{heading}");
    if selected.is_empty() {
        println!("  (none)");
    }

    let mut focus_missing = false;
    for (i, s) in &selected {
        println!("\n[#{i}] ({}) {}", s.case.category, s.case.question);
        if !s.case.gold.is_empty() {
            println!("  gold answer: {}", s.case.gold);
        }
        if s.expected_present.is_empty() {
            println!("  no evidence in the corpus — not scored (adversarial / open inference)");
            continue;
        }
        let hpos = positions(&s.ranked["hybrid"]);
        let vpos = positions(&s.ranked["vector"]);
        let kpos = positions(&s.ranked["keyword"]);
        println!("  expected evidence:");
        for e in &s.expected_present {
            let where_ = match hpos.get(e.as_str()) {
                Some(r) => format!("hybrid #{r}"),
                None => "NOT in hybrid top-k".to_string(),
            };
            let v = vpos
                .get(e.as_str())
                .map_or("—".to_string(), |r| format!("#{r}"));
            let kr = kpos
                .get(e.as_str())
                .map_or("—".to_string(), |r| format!("#{r}"));
            println!("    {e}  [{where_} · vector {v} · keyword {kr}]");
            println!(
                "        {}",
                truncate(text_of.get(e.as_str()).copied().unwrap_or(""))
            );
            if focus_id.as_deref() == Some(e.as_str()) && !hpos.contains_key(e.as_str()) {
                focus_missing = true;
            }
        }
        println!("  hybrid top-{k}:");
        for (rank, id) in s.ranked["hybrid"].iter().enumerate() {
            let star = if s.expected_present.iter().any(|e| e == id) {
                " ★"
            } else {
                ""
            };
            println!(
                "    #{:<2} {id}{star}  {}",
                rank + 1,
                truncate(text_of.get(id.as_str()).copied().unwrap_or(""))
            );
        }
    }

    for s in &scored {
        if s.expected_present.is_empty() {
            continue;
        }
        let pos = positions(&s.ranked["hybrid"]);
        let hit = s
            .expected_present
            .iter()
            .filter(|e| pos.get(e.as_str()).is_some_and(|p| *p <= GATE_CUTOFF))
            .count();
        gate_recall_sum += hit as f64 / s.expected_present.len() as f64;
    }
    let gate_recall = if scored_n > 0 {
        gate_recall_sum / scored_n as f64
    } else {
        0.0
    };

    // --- exit code: a CI / regression guard ----------------------------------
    if focus_id.is_some() && focus_missing {
        eprintln!("\nFAIL: focused evidence is absent from hybrid top-{k}");
        exit(1);
    }
    if let Some(floor) = fail_under {
        if gate_recall < floor {
            eprintln!("\nFAIL: hybrid recall@{GATE_CUTOFF} {gate_recall:.4} < floor {floor:.4}");
            exit(1);
        }
        eprintln!("\nok: hybrid recall@{GATE_CUTOFF} {gate_recall:.4} >= floor {floor:.4}");
    }
    Ok(())
}
