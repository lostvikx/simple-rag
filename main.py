import sys
from pathlib import Path

from modules.scrape import extract_text
from modules.chunks import create_chunks
from modules.embedding import create_embeddings, store_embeddings
from modules.retrieve import retrieve_chunks
from modules.llm_gemini import prompt_llm

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <file.pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    pdf_path = Path(pdf_file).expanduser()
    chroma_dir = "data/chroma"

    Path("data").mkdir(parents=True, exist_ok=True)
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)

    pages = extract_text(pdf_path)
    chunks = create_chunks(pages)
    embeddings = create_embeddings(chunks)

    count = store_embeddings(embeddings, chunks, data_dir=chroma_dir)
    print(f"Complete: Stored {count} embeddings in {chroma_dir} directory.")

    print("Note: Use the terms 'exit', 'quit', or 'bye' to end the conversation.\n")
    print("AI: Hi! Ask me anything about the document you uploaded.")

    while True:
        query = input("User: ").strip()

        if query.lower() in {"exit", "quit", "bye"}:
            print("AI: Goodbye!")
            break

        matched_chunks = retrieve_chunks(query, data_dir=chroma_dir, pdf_path=pdf_path, k=5)

        response = prompt_llm(query, matched_chunks)
        print(f"AI: {response}")


if __name__ == "__main__":
    main()
