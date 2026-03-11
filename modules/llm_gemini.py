from langchain_google_genai import ChatGoogleGenerativeAI

def prompt_llm(query: str, matches: zip, model_name: str = "gemini-3-flash-preview") -> str:
    """Prompts a Google Generative AI model to answer a query using provided context matches."""

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=1.0
    )

    context_chunks = []
    for id, text, metadata, distance in matches:
        header = f"ID: {id}"
        context_chunks.append(f"{header}\n{text}")

    context = "\n\n".join(context_chunks)

    prompt = f"""
    You are a helpful assistant. Use the context to answer the question.
    If the answer is not in the context, say you do not know.

    Question: { query }

    Context:
    { context }

    Answer:
    """

    response = llm.invoke(prompt)
    response_content = response.content[0]

    if isinstance(response_content, str):
        ai_response = response_content.strip()
    elif isinstance(response_content, dict):
        ai_response = str(response_content.get("text", "")).strip()
    else:
        ai_response = "Sorry, I encountered an error while processing the response."

    return ai_response
