import json
import chromadb
from langchain_ollama import OllamaEmbeddings

def create_embeddings(
    chunks: list,
    model_name: str = "nomic-embed-text"
) -> list[list[float]]:
    """Creates embeddings for the given document chunks using the Ollama API."""

    model = OllamaEmbeddings(model=model_name)
    embeddings = model.embed_documents([chunk.page_content for chunk in chunks])

    with open("data/embeddings.json", "w") as f:
        json.dump(embeddings, f, indent=4)

    return embeddings

def store_embeddings(embeddings: list, chunks: list, data_dir: str) -> int:
    """Stores the embeddings and associated metadata in a ChromaDB collection."""

    client = chromadb.PersistentClient(path=data_dir)
    collection = client.get_or_create_collection(name="document_chunks")

    ids = [
        f"{chunk.metadata['source']}_page{chunk.metadata['page']}_{i}"
        for i, chunk in enumerate(chunks)
    ]

    # insert or update the embeddings with metadata
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
        documents=[chunk.page_content for chunk in chunks]
    )

    return len(ids)
