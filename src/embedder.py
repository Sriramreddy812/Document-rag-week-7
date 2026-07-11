from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load and return a HuggingFace sentence-transformer embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    return embedding_model