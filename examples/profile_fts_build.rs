//! Single-shot profile harness for `FtsBuilder` at the same 1M-doc
//! Zipfian corpus the Criterion superfile FTS build bench uses.
//!
//! Run with:
//! ```text
//! INFINO_FTS_PROFILE=1 cargo run --release --example profile_fts_build
//! ```

use infino::superfile::fts::builder::FtsBuilder;
use infino::test_helpers::default_tokenizer;
use infino_bench_utils::corpus;
use std::time::Instant;

const N_DOCS: usize = 1_000_000;

fn main() {
    eprintln!("[profile] generating corpus n_docs={N_DOCS} ...");
    let t0 = Instant::now();
    let docs: Vec<String> = corpus::generate_text_corpus(N_DOCS, 1);
    let avg_bytes = if docs.is_empty() {
        0
    } else {
        docs.iter().map(String::len).sum::<usize>() / docs.len()
    };
    eprintln!(
        "[profile] corpus generated in {:.2}s ({} docs, ~{} bytes/doc avg)",
        t0.elapsed().as_secs_f64(),
        docs.len(),
        avg_bytes,
    );

    let mut builder = FtsBuilder::new(default_tokenizer());
    builder
        .register_column("title".to_string())
        .expect("register column");

    let t_add = Instant::now();
    for (i, text) in docs.iter().enumerate() {
        builder.add_doc(0, i as u32, text).expect("add doc");
    }
    let add_elapsed = t_add.elapsed();
    eprintln!(
        "[profile] add_doc total: {:.3}s ({:.0}ns/doc)",
        add_elapsed.as_secs_f64(),
        add_elapsed.as_nanos() as f64 / N_DOCS as f64,
    );

    let t_finish = Instant::now();
    let blob = builder.finish().expect("finish");
    let finish_elapsed = t_finish.elapsed();
    eprintln!(
        "[profile] finish: {:.3}s   blob_len={} ({:.1} MiB)",
        finish_elapsed.as_secs_f64(),
        blob.len(),
        blob.len() as f64 / (1024.0 * 1024.0),
    );

    let total = add_elapsed + finish_elapsed;
    eprintln!(
        "[profile] total build: {:.3}s   (add_doc {:.0}% + finish {:.0}%)",
        total.as_secs_f64(),
        100.0 * add_elapsed.as_secs_f64() / total.as_secs_f64(),
        100.0 * finish_elapsed.as_secs_f64() / total.as_secs_f64(),
    );
}
