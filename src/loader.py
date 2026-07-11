from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)


def load_document(file_path: str):
    """
    Load a PDF or TXT document and return a list of LangChain Document objects.
    """

    file_extension = Path(file_path).suffix.lower()

    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    documents = loader.load()

    return documents