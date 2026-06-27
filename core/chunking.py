"""Splits documents into overlapping chunks small enough to embed and feed to an LLM."""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Splits text into overlapping windows of chunk_size characters.

    Character-based rather than token-based — simple, dependency-free, and close
    enough for the chunk sizes we use here.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    index = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, chunk_index=index))
            index += 1
        if end >= len(text):
            break
        start = end - overlap

    return chunks
