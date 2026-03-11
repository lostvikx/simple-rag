from langchain_ollama import ChatOllama

def prompt_llm(query: str, matches: zip, model_name: str = "qwen3.5:2b") -> str:
    """Prompts an Ollama model to answer a query using provided context matches."""

    llm = ChatOllama(
        model=model_name,
        temperature=0.8,
    )

    context_chunks = []
    for id, text, metadata, distance in matches:
        header = f"ID: {id}"
        context_chunks.append(f"{header}\n{text}")

    context = "\n\n".join(context_chunks)

    prompt = f"""
    You are a helpful assistant. Use the context to answer the question. If the answer is not in the context, say you do not know.

    Question: { query }

    Context:
    { context }

    Answer:
    """

    response = llm.invoke(prompt)
    return str(response.content).strip()
