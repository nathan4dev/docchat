"""Builds the grounded prompt sent to the LLM from a question and retrieved chunks."""

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided context. "
    "If the context doesn't contain the answer, say you don't know — don't make one up. "
    "Keep answers concise and cite sources by their bracketed number, e.g. [1]."
)


def build_context_block(chunks: list[dict]) -> str:
    """Formats retrieved chunks into a numbered context block the model can cite by number.

    Expects each chunk dict to have "text" and "source" keys.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"
