//! Scale bench bundle: at-scale pinned-recall assertion runners that
//! need release-profile compilation to finish in seconds rather than
//! minutes. Each `run()` prints single-line summaries per phase to
//! stdout.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench scale                     # every scale bench
//! cargo bench --bench scale -- vector_recall    # IVF + RaBitQ + rerank recall battery (10K × 384)
//! ```

#[path = "vector_recall.rs"]
mod vector_recall;

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let run_all = filter.is_empty();
    let want = |needle: &str| run_all || filter.contains(needle);

    if want("vector_recall") {
        eprintln!("[scale] --- vector_recall ---");
        vector_recall::run();
    }
}
