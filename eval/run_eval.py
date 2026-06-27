"""Runs the test set through the full pipeline and prints a quality scorecard.

Retrieval hit rate is checked directly (did the expected source show up in the
retrieved chunks). Faithfulness and relevance are judged by the LLM itself,
since there's no cheap automatic way to check those without a reference model.
"""

import json
from pathlib import Path

from adapters.embeddings import EmbeddingClient
from adapters.llm import LLMClient
from adapters.vector_store import VectorStore
from services.query import answer_question

TESTSET_PATH = Path(__file__).parent / "testset.json"

JUDGE_PROMPT = """You are grading a RAG system's answer. Respond with only "yes" or "no".

Question: {question}
Retrieved context: {context}
Generated answer: {answer}

{criterion}"""

FAITHFULNESS_CRITERION = (
    "Is the generated answer fully supported by the retrieved context, with no claims "
    "that aren't backed by it? Answer yes or no."
)
RELEVANCE_CRITERION = (
    "Does the generated answer actually address what was asked in the question? "
    "Answer yes or no."
)


def judge(llm: LLMClient, question: str, context: str, answer: str, criterion: str) -> bool:
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer, criterion=criterion)
    verdict = llm.generate(system_prompt="You are a strict, concise grader.", user_prompt=prompt, max_tokens=10)
    return verdict.strip().lower().startswith("yes")


def run() -> None:
    testset = json.loads(TESTSET_PATH.read_text())

    embedder = EmbeddingClient()
    store = VectorStore()
    llm = LLMClient()

    if store.count() == 0:
        print("Index is empty — run `python -m services.ingest` first.")
        return

    hits = 0
    faithful = 0
    relevant = 0

    for case in testset:
        result = answer_question(case["question"], embedder=embedder, store=store, llm=llm)
        sources_hit = any(s["source"] == case["expected_source"] for s in result["sources"])
        hits += sources_hit

        context = "\n\n".join(s["text"] for s in result["sources"])
        is_faithful = judge(llm, case["question"], context, result["answer"], FAITHFULNESS_CRITERION)
        is_relevant = judge(llm, case["question"], context, result["answer"], RELEVANCE_CRITERION)
        faithful += is_faithful
        relevant += is_relevant

        status = "✓" if sources_hit and is_faithful and is_relevant else "✗"
        print(f"{status} {case['question']}")

    n = len(testset)
    print("\nScorecard")
    print(f"  Retrieval hit rate:   {hits}/{n} ({hits / n:.0%})")
    print(f"  Answer faithfulness:  {faithful}/{n} ({faithful / n:.0%})")
    print(f"  Answer relevance:     {relevant}/{n} ({relevant / n:.0%})")


if __name__ == "__main__":
    run()
