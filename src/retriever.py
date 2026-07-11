def retrieve_relevant_chunks(vector_store, query: str, top_k: int = 3):
    """
    Retrieve the top_k most relevant chunks for a given query,
    along with their similarity scores.
    """

    results = vector_store.similarity_search_with_score(query, k=top_k)

    return results