"""Answers a question end-to-end: embed, retrieve, build prompt, generate, cite."""

from adapters.embeddings import EmbeddingClient
from adapters.llm import LLMClient
from adapters.vector_store import VectorStore
from core.prompt import SYSTEM_PROMPT, build_user_prompt
from core.ranking import rank_results

TOP_K = 4


def answer_question(
    question: str,
    embedder: EmbeddingClient | None = None,
    store: VectorStore | None = None,
    llm: LLMClient | None = None,
    top_k: int = TOP_K,
) -> dict:
    """Returns {"answer": str, "sources": list[dict]} for the given question.

    Accepts pre-built adapters so callers (the UI, the eval harness) can reuse
    one embedding model / store / client across many questions instead of
    reloading them per call.
    """
    embedder = embedder or EmbeddingClient()
    store = store or VectorStore()
    llm = llm or LLMClient()

    [question_embedding] = embedder.embed([question])
    raw_results = store.query(question_embedding, top_k=top_k * 2)
    chunks = rank_results(raw_results, top_k=top_k)

    if not chunks:
        return {"answer": "I don't have any indexed documents to answer from.", "sources": []}

    user_prompt = build_user_prompt(question, chunks)
    answer = llm.generate(SYSTEM_PROMPT, user_prompt)

    return {"answer": answer, "sources": chunks}
