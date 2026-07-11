import os
import tempfile

import streamlit as st

from src.loader import load_document
from src.chunker import chunk_documents
from src.embedder import get_embedding_model
from src.vectorstore import build_vector_store
from src.retriever import retrieve_relevant_chunks
from src.generator import get_llm_pipeline, generate_answer


st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")

st.title("Document Question Answering System")
st.write("Upload a PDF and ask questions about its content.")

# Sidebar lets you control how many chunks get retrieved without editing code.
# Fewer chunks = more focused context (better for precise facts).
# More chunks = broader context (better for summary-style questions).
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve (Top-K)", min_value=1, max_value=8, value=3)


@st.cache_resource
def load_embedding_model():
    return get_embedding_model()


@st.cache_resource
def load_llm():
    return get_llm_pipeline()


# NOTE: embedding_model and llm_pipeline are no longer loaded here at the top.
# They're only loaded inside the blocks below, right before they're actually needed,
# so the page (title, uploader) renders instantly instead of blocking on model load.

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:

    if st.session_state.get("uploaded_file_name") != uploaded_file.name:

        with st.spinner("Loading embedding model..."):
            embedding_model = load_embedding_model()

        with st.spinner("Reading and indexing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            documents = load_document(tmp_file_path)
            chunks = chunk_documents(documents)
            vector_store = build_vector_store(chunks, embedding_model)

            os.remove(tmp_file_path)

        st.session_state["vector_store"] = vector_store
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.session_state["total_chunks"] = len(chunks)

    st.success(f"Document indexed: {st.session_state['total_chunks']} chunks created.")

    user_query = st.text_input("Ask a question about the document:")

    if st.button("Get Answer") and user_query:
        with st.spinner("Loading language model (first question only)..."):
            llm_pipeline = load_llm()

        with st.spinner("Retrieving relevant context and generating answer..."):
            retrieved_results = retrieve_relevant_chunks(
                st.session_state["vector_store"], user_query, top_k=top_k
            )
            answer = generate_answer(llm_pipeline, user_query, retrieved_results)

        st.subheader("Answer")
        st.write(answer)