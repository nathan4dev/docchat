"""Wraps the embedding model. Runs locally via sentence-transformers — no API key needed."""

import os

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingClient:
    def __init__(self, model_name: str | None = None):
        self.model = SentenceTransformer(model_name or os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()
