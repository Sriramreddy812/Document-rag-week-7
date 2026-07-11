from langchain_community.vectorstores import FAISS


def build_vector_store(chunks, embedding_model):
    """
    Embed all chunks and store them in a FAISS vector database.
    """

    vector_store = FAISS.from_documents(chunks, embedding_model)

    return vector_store


def save_vector_store(vector_store, save_path: str = "faiss_index"):
    """
    Save the FAISS index to disk so it doesn't need to be rebuilt every run.
    """

    vector_store.save_local(save_path)


def load_vector_store(embedding_model, save_path: str = "faiss_index"):
    """
    Load a previously saved FAISS index from disk.
    """

    vector_store = FAISS.load_local(
        save_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    return vector_store