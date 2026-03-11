import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(
    corpus: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 75
) -> list:
    """Splits the extracted text into smaller chunks suitable for embedding and retrieval."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,                                  # Maximum size of each chunk
        chunk_overlap=chunk_overlap,                            # Number of tokens to overlap (10-20% of chunk_size)
        separators=["\\n\\n", "\\n", " ", ""],                  # Hierarchy of separators
        length_function=len,
    )

    text = [page.get("text", "").strip() for page in corpus]
    metas = [{ "source": page.get("source"), "page": page.get("page") } for page in corpus]

    chunks = text_splitter.create_documents(text, metadatas=metas)

    # store the chunks in json
    with open("data/chunks.json", "w") as f:
        json.dump([{ "page_content": chunk.page_content, "metadata": chunk.metadata } for chunk in chunks], f, indent=4)

    return chunks
