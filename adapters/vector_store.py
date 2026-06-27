"""Wraps Chroma so the rest of the app only deals with plain dicts, not Chroma's API."""

import chromadb

DEFAULT_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "docchat"


class VectorStore:
    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    def add(self, ids: list[str], embeddings: list[list[float]], texts: list[str], metadatas: list[dict]) -> None:
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        # Chroma nests everything one level for batch queries; we only ever send one query.
        return [
            {"text": doc, "source": meta["source"], "distance": dist}
            for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
