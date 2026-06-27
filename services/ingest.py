"""Builds the vector index from documents in ./data. Run as `python -m services.ingest`."""

from pathlib import Path

from pypdf import PdfReader

from adapters.embeddings import EmbeddingClient
from adapters.vector_store import VectorStore
from core.chunking import chunk_text

DATA_DIR = Path("data")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents(data_dir: Path = DATA_DIR) -> list[tuple[str, str]]:
    """Returns (filename, text) pairs for every supported file in data_dir."""
    documents = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        if path.suffix == ".pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            text = path.read_text(encoding="utf-8")
        documents.append((path.name, text))
    return documents


def run() -> None:
    documents = load_documents()
    if not documents:
        print(f"No documents found in {DATA_DIR}/")
        return

    embedder = EmbeddingClient()
    store = VectorStore()
    store.reset()

    for filename, text in documents:
        chunks = chunk_text(text, source=filename)
        if not chunks:
            continue
        embeddings = embedder.embed([c.text for c in chunks])
        ids = [f"{filename}-{c.chunk_index}" for c in chunks]
        metadatas = [{"source": filename, "chunk_index": c.chunk_index} for c in chunks]
        store.add(ids=ids, embeddings=embeddings, texts=[c.text for c in chunks], metadatas=metadatas)
        print(f"Indexed {filename}: {len(chunks)} chunks")

    print(f"Done. {store.count()} chunks in the index.")


if __name__ == "__main__":
    run()
