// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Remote (hosted) quickstart — the local→cloud one-line diff in action.
//!
//! Point `INFINO_REMOTE_URL` at a hosted endpoint and `INFINO_API_KEY` at a
//! key for it, then run:
//!
//! ```sh
//! export INFINO_REMOTE_URL="https://your-endpoint/mydb"  # http:// only for a loopback address
//! export INFINO_API_KEY="ik_…"
//! cargo run --features remote --example remote_quickstart
//! ```
//!
//! It creates a table, appends a batch, runs a keyword search and a SQL query,
//! then drops the table — the same calls as a local `connect("./data")`, only
//! the connect target changed.

#[cfg(feature = "remote")]
mod remote_example {
    use std::{error::Error, sync::Arc};

    use infino::{
        Bm25SearchOptions, IndexSpec,
        arrow_array::{Int32Array, LargeStringArray, RecordBatch},
        arrow_schema::{DataType, Field, Schema},
        connect,
    };

    pub fn run() -> Result<(), Box<dyn Error>> {
        let url = std::env::var("INFINO_REMOTE_URL")
            .map_err(|_| "set INFINO_REMOTE_URL, e.g. http://localhost:8080/mydb")?;
        // The API key is read from INFINO_API_KEY by `connect`; fail early with
        // a clear message if it is missing.
        std::env::var("INFINO_API_KEY").map_err(|_| "set INFINO_API_KEY")?;

        // One argument different from a local `connect("./data")`.
        let db = connect(&url)?;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("body", DataType::LargeUtf8, false),
        ]));
        let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
        println!("created table `posts`");

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(LargeStringArray::from(vec![
                    "cancel my subscription",
                    "reset my password",
                    "cancel the order",
                ])),
            ],
        )?;
        posts.append(&batch)?;
        println!("appended {} rows", batch.num_rows());

        let hits = posts.bm25_search(
            "body",
            "cancel",
            10,
            Bm25SearchOptions::new(),
            Some(&["_id", "body"]),
        )?;
        let matched: usize = hits.iter().map(RecordBatch::num_rows).sum();
        println!("bm25_search('cancel') matched {matched} rows");

        let sql = db.query_sql("SELECT COUNT(*) AS n FROM posts")?;
        println!("query_sql returned {} batch(es)", sql.len());

        db.drop_table("posts", true)?;
        println!("dropped table `posts` — done");
        Ok(())
    }
}

fn main() {
    #[cfg(feature = "remote")]
    {
        if let Err(e) = remote_example::run() {
            eprintln!("remote quickstart failed: {e}");
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "remote"))]
    {
        eprintln!("build with `--features remote` to run this example");
    }
}
