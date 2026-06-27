# DocChat — A Production-Minded RAG Chatbot

Ask natural-language questions about your own documents and get answers grounded in the source text — with citations and a built-in evaluation harness that measures whether the answers are actually correct.

> Most RAG demos stop at "it works on my one test question." This project goes further: it's structured with clean architecture, it cites its sources, and it ships with an evaluation suite so quality is *measured*, not assumed.

---

## Why this project exists

Retrieval-Augmented Generation (RAG) is the backbone of most real-world LLM applications — internal knowledge bases, support assistants, document search. This repo is a compact, readable reference implementation that demonstrates the full lifecycle:

**ingest → chunk → embed → retrieve → generate → evaluate**

It's deliberately scoped to be *understandable in one sitting* rather than feature-maximal. Clarity over cleverness.

---

## What it does

- **Ingests** mixed document sources (PDF, plain text, Markdown) into a vector store.
- **Answers questions** about those documents using semantic retrieval + an LLM.
- **Cites sources** so every answer is traceable back to the original text.
- **Evaluates itself** against a question/answer test set, reporting retrieval and answer-quality scores.
- **Runs locally** with a simple web UI — no cloud account required to try it.

---

## Architecture

The codebase follows a clean, layered design. Each layer has one responsibility and depends only on the layer beneath it, so any piece (the LLM, the vector store, the chunking strategy) can be swapped without touching the rest.

```
┌──────────────────────────────────────────────┐
│                   UI Layer                     │   Streamlit app — chat + source display
├──────────────────────────────────────────────┤
│                Service Layer                   │   Orchestrates the RAG flow
│   ┌──────────────┐      ┌──────────────────┐   │
│   │  Ingestion   │      │   Query / Answer  │  │
│   │  pipeline    │      │   pipeline        │  │
│   └──────────────┘      └──────────────────┘   │
├──────────────────────────────────────────────┤
│                 Core Layer                     │   Pure logic, no I/O
│   chunking · prompt building · ranking         │
├──────────────────────────────────────────────┤
│              Adapter Layer (ports)             │   Swappable integrations
│   LLM client · embeddings · vector store       │
└──────────────────────────────────────────────┘
```

### Request flow

**Ingestion (one-time, per document set)**
1. Load raw documents from a folder.
2. Split them into overlapping chunks.
3. Generate an embedding for each chunk.
4. Store chunks + embeddings + metadata in the vector store.

**Query (per question)**
1. Embed the user's question.
2. Retrieve the top-_k_ most similar chunks.
3. Build a grounded prompt (question + retrieved context).
4. Call the LLM to generate an answer.
5. Return the answer with its source citations.

---

## Project structure

```
docchat/
├── app.py                  # Streamlit UI entry point
├── core/                   # Pure logic — no external calls, fully unit-testable
│   ├── chunking.py         # Document splitting strategy
│   ├── prompt.py           # Prompt construction
│   └── ranking.py          # Retrieval result ordering
├── adapters/               # Swappable integrations (the "ports")
│   ├── llm.py              # LLM client wrapper
│   ├── embeddings.py       # Embedding model wrapper
│   └── vector_store.py     # Vector DB wrapper (Chroma)
├── services/               # Orchestration — wires core + adapters together
│   ├── ingest.py           # Build the index from documents
│   └── query.py            # Answer a question end-to-end
├── eval/                   # Quality measurement
│   ├── testset.json        # Question / expected-answer pairs
│   └── run_eval.py         # Scores retrieval + answer faithfulness
├── data/                   # Sample documents to index
├── tests/                  # Unit tests for the core layer
├── requirements.txt
└── README.md
```

---

## Evaluation

This is the part most portfolio projects skip — and the part that signals engineering maturity.

`eval/run_eval.py` runs a fixed set of questions through the pipeline and reports:

| Metric | What it measures |
|---|---|
| **Retrieval hit rate** | Did the correct source chunk make it into the retrieved context? |
| **Answer faithfulness** | Is the generated answer actually supported by the retrieved text (not hallucinated)? |
| **Answer relevance** | Does the answer address the question that was asked? |

Run it with:

```bash
python eval/run_eval.py
```

Output is a simple scorecard so you can tell whether a change (different chunk size, different _k_, different prompt) made the system better or worse — instead of guessing.

---

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python | Standard for AI engineering |
| LLM | Configurable (Claude / OpenAI) | Swapped via the adapter layer |
| Embeddings | Configurable | Same — isolated behind a port |
| Vector store | Chroma | Lightweight, runs locally, zero infra |
| UI | Streamlit | Fast to build, easy to demo |
| Evaluation | Custom + LLM-as-judge | Transparent and easy to read |

No framework lock-in: the RAG loop is written explicitly so the mechanics are visible, not hidden inside a library.

---

## Getting started

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/docchat.git
cd docchat
pip install -r requirements.txt

# 2. Set your API key
export LLM_API_KEY="your-key-here"

# 3. Add documents to ./data, then build the index
python -m services.ingest

# 4. Launch the chat UI
streamlit run app.py

# 5. (Optional) Measure quality
python eval/run_eval.py
```

---

## Design principles

- **Separation of concerns** — pure logic (`core/`) never touches the network, so it's trivial to test.
- **Dependency inversion** — services depend on adapter *interfaces*, not concrete vendors. Switching from one LLM to another is a one-file change.
- **Explicit over magic** — the retrieval loop is hand-written so a reader can follow exactly what happens to a question.
- **Measured, not assumed** — quality is verified with an eval suite, not vibes.

---

## Possible extensions

These are intentionally left out to keep the core focused, but each is a natural next step:

- Re-ranking retrieved chunks with a cross-encoder.
- Hybrid search (keyword + semantic).
- Conversation memory for multi-turn follow-ups.
- Swapping the local vector store for a hosted one (Pinecone, pgvector).

---

## License

MIT
