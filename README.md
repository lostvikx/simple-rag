# Naive RAG

This directory contains code for a CLI application I build to read technical research papers with the help of a LLM. This takes advantage of a AI framework called as Retrieval Augmented Generation which improves LLM accuracy.

## Setup

```bash
cd simple-rag
uv sync
uv run python main.py paper.pdf
```

Make sure you have a Google Gemini API key.

```bash
export GEMINI_API_KEY="AIxxx"
```

## Situation

Let's assume we have documents that are not available on the internet. Meaning a LLM is not aware about the fine details about the information in these documents. How can we still use a chat-like interface to access knowledge from these documents?

Using RAG technique we can make the LLM understand the context of such documents and access knowledge through a chat-like interface.

## RAG Process

![Simple RAG Workflow](assets/imgs/simple_rag_workflow.png)

1. We create chucks (e.g. 1,000 chars) of these documents.
2. Create embeddings of these chunks using an embedding model.
3. Store these embeddings in a vector database.
4. Create an embedding of the user prompt.
5. Using a similarity metric find the most similar chunk.
6. Include that text chunk in the LLM context window with the prompt.
7. Generate a response from the LLM.

A few examples of similarity metrics:

1. Cosine similarity: angle between two vectors.
2. Euclidean distance: distance between the two vectors.

### Note on Embedding Models

We may use more specialized embedding models instead of general ones for searching in a specialized text corpus.

![Text Embedding](assets/imgs/word_embeddings.png)

## Formal Step-by-Step RAG Process

1. Ingestion & Indexing: External documents are cleaned, split into chunks, converted into vector embeddings, and stored in a vector database.
2. User Query: A user asks a question or submits a prompt.
3. Retrieval: The system searches the vector database to find the top-most relevant text chunks related to the query.
4. Augmentation: The retrieved context is combined with the user query, creating a prompt that provides the necessary context.
5. Generation: The LLM uses this augmented prompt to generate a response that is grounded in the retrieved data.

## Benefits of RAGs

1. Reduced Hallucinations: Answers are based on retrieved facts rather than just training data or web searches.
2. Real-time Data: Enables access to up-to-date information without retraining the entire LLM.
3. Cost-Effective: Lower expense compared to full fine-tuning of the LLM.

## Chatbot Prompt Example

```
You are a helpful assistant. Use the context to answer the question.
If the answer is not in the context, say you do not know.

Context:
{ context }

Question: { query }

Answer: 
```

## Useful Links

- [Ollama Text Embedding](https://docs.langchain.com/oss/python/integrations/text_embedding/ollama)
- [Document Splitters](https://docs.langchain.com/oss/python/integrations/splitters)
- [Chroma Vector Store](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [Google Generative AI Chat](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)

## Test Questions

* Explain the main idea behind this research paper.
* How difficult is it to implement this using Python?
* Explain {topic_name} in detail.
* What implications can {topic} have on the future of technology?
