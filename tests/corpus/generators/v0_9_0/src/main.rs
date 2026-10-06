// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Reindexes a corpus table in place with the 0.9.0 engine.
//!
//! 0.9.0's reindex carried a superfile's vector subsection but also copied
//! the input's vector region keys into the output footer, ahead of the
//! output's own. The engine reads the last copy and is unaffected; a reader
//! that takes the first resolves the input's offsets. This is the shape a
//! table repaired by that release is in, so a later engine can be held to
//! detecting and repairing it.
//!
//! Usage: `cargo run -- <table-dir> <table-name>`, where `<table-dir>`
//! already holds the table to reindex.

use std::{env, time::Duration};

use infino::{ReindexOptions, connect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let dir = args.next().ok_or("usage: <table-dir> <table-name>")?;
    let table = args.next().ok_or("usage: <table-dir> <table-name>")?;

    let handle = connect(&dir)?.open_table(&table)?;
    let report = handle.reindex(&ReindexOptions::default())?;
    // Only the rewritten files stay, so the table holds the shape alone.
    handle.gc(Duration::ZERO)?;

    println!("reindexed {} superfile(s) in {dir}/{table}", report.rewritten);
    Ok(())
}
