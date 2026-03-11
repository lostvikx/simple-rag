import chromadb
from pathlib import Path
from langchain_ollama import OllamaEmbeddings

def retrieve_chunks(
    query: str,
    data_dir: str,
    pdf_path: str | Path,
    embedding_model_name: str = "nomic-embed-text",
    k: int = 5
) -> zip:
    """Retrieves the most relevant document chunks for a given query using the Ollama API and ChromaDB."""

    model = OllamaEmbeddings(model=embedding_model_name)
    embedded_query = model.embed_query(query)

    client = chromadb.PersistentClient(path=data_dir)
    collection = client.get_or_create_collection(name="document_chunks")    # getting the collection

    results = collection.query(
        query_embeddings=embedded_query, n_results=k,
        include=["documents", "metadatas", "distances"],
        where={"source": str(pdf_path)}
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    return zip(ids, documents[0], metadatas[0], distances[0])
