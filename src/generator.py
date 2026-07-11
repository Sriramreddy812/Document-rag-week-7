from transformers import pipeline


def get_llm_pipeline(model_name: str = "google/flan-t5-base"):
    """
    Load and return a text2text-generation pipeline for answer generation.
    """

    llm_pipeline = pipeline("text2text-generation", model=model_name)

    return llm_pipeline


def generate_answer(llm_pipeline, query: str, retrieved_results):
    """
    Build a context-grounded prompt from retrieved chunks and generate an answer.
    """

    context_text = "\n\n".join([chunk.page_content for chunk, score in retrieved_results])

    prompt = (
        f"Read the context carefully and answer the question with a short, direct, factual answer. "
        f"Extract the exact answer from the context if possible. "
        f"If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    result = llm_pipeline(
        prompt,
        max_new_tokens=150,
        truncation=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    answer = result[0]["generated_text"].strip()

    return answer