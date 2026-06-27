"""Streamlit chat UI for DocChat. Run with `streamlit run app.py`."""

import streamlit as st

from adapters.embeddings import EmbeddingClient
from adapters.llm import LLMClient
from adapters.vector_store import VectorStore
from services.query import answer_question

st.set_page_config(page_title="DocChat", page_icon="📄")
st.title("DocChat")
st.caption("Ask questions about the documents in ./data")


@st.cache_resource
def load_adapters():
    return EmbeddingClient(), VectorStore(), LLMClient()


embedder, store, llm = load_adapters()

if store.count() == 0:
    st.warning("No documents indexed yet. Run `python -m services.ingest` first.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**[{i}] {source['source']}**\n\n{source['text']}")

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(question, embedder=embedder, store=store, llm=llm)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for i, source in enumerate(result["sources"], start=1):
                    st.markdown(f"**[{i}] {source['source']}**\n\n{source['text']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
