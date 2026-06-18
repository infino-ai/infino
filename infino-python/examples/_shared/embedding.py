"""Local sentence embeddings for the examples: `all-MiniLM-L6-v2`, 384-dim, cosine.

Swap in a hosted embedder by editing `embed` / `embed_query` and `DIM` / `METRIC`.
"""

import os
import threading

import pyarrow as pa

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384          # must match the vector index dim
METRIC = "cosine"  # MiniLM outputs are normalized

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Load the model once, lazily."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                from transformers.utils import logging as hf_logging

                hf_logging.set_verbosity_error()
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
