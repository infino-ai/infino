"""Fixed-size, overlapping word chunker.

Splits text into overlapping word windows so a sentence spanning a boundary
isn't lost. Word-based, so chunks never split mid-word.
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
        chunks.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break  # stop at the end; avoids a redundant trailing chunk
    return chunks
