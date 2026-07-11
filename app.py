from src.loader import load_document
from src.chunker import chunk_documents
from src.embedder import get_embedding_model
from src.vectorstore import build_vector_store
from src.retriever import retrieve_relevant_chunks
from src.generator import get_llm_pipeline, generate_answer

def main():

    file_path = "data/book review.pdf"

    documents = load_document(file_path)

    print(f"Total Documents Loaded: {len(documents)}")

    print("\nMetadata:")
    print(documents[0].metadata)

    print("\nContent Preview:\n")
    print(documents[0].page_content[:500])
    chunks = chunk_documents(documents)

    print(f"\nTotal Chunks Created: {len(chunks)}")
    print("\nFirst Chunk Preview:\n")
    print(chunks[0].page_content[:500])

    embedding_model = get_embedding_model()

    vector_store = build_vector_store(chunks, embedding_model)

    print(f"\nVector Store Created")
    print(f"Total Vectors Indexed: {vector_store.index.ntotal}")
    user_query = "What is the book review about?"

    retrieved_results = retrieve_relevant_chunks(vector_store, user_query, top_k=5)

    print(f"\nUser Query: {user_query}\n")
    for i, (chunk, score) in enumerate(retrieved_results):
        print(f"Result {i+1} (score: {score:.4f}):")
        print(chunk.page_content[:300])
        print("-" * 60)

    
    llm_pipeline = get_llm_pipeline()

    answer = generate_answer(llm_pipeline, user_query, retrieved_results)

    print(f"\nGenerated Answer:\n{answer}")

if __name__ == "__main__":
    main()