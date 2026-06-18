"""Local sentence-embedding helper shared across the RAG examples.

Uses a small local model (`all-MiniLM-L6-v2`, 384-dim) so the examples run
with no API key, no network at query time, and identical results everywhere.

To swap in a hosted embedder (e.g. OpenAI), replace `embed`/`embed_query` with
calls to that API and set `DIM` / `METRIC` to match the model — nothing else in
the examples needs to change, because everything downstream keys off these
two constants and the Arrow column built by `as_vector_column`.
"""

import os
import threading

import pyarrow as pa

# Keep example output clean and reproducible: no download progress bars, no
# tokenizer fork warnings, no HF rate-limit notices for these public models.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Embedding model + the two facts the rest of the pipeline derives from it.
MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384          # output dimension; must match the Infino vector index `dim`
METRIC = "cosine"  # all-MiniLM vectors are normalized → cosine is the right metric

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Load the model once, lazily (it is a few hundred MB the first time)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                from transformers.utils import logging as hf_logging

                hf_logging.set_verbosity_error()  # silence the weight-loading banner
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents into normalized 384-dim vectors."""
    model = _get_model()
    return model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string into one normalized 384-dim vector."""
    return embed([text])[0]


def as_vector_column(embeddings: list[list[float]]) -> pa.Array:
    """Pack embeddings into the `fixed_size_list<float32, DIM>` Infino expects."""
    return pa.array(embeddings, type=pa.list_(pa.float32(), DIM))
