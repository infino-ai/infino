"""Minimal fixed-size, overlapping text chunker.

The canonical RAG pre-processing step: split a document into windows small
enough to embed precisely, with a little overlap so a sentence spanning a
boundary is not lost. Word-based (not character-based) so chunks never split
mid-word. No external dependency on purpose — this is all the core path needs.
"""

DEFAULT_CHUNK_WORDS = 120
DEFAULT_OVERLAP_WORDS = 20


def chunk_text(
    text: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """Split `text` into overlapping word windows.

    Returns at least one chunk for any non-empty input; empty/whitespace input
    yields an empty list.
    """
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    words = text.split()
    if not words:
        return []

    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks
