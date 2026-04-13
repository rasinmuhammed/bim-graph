from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

QUERY = "List all equipment and mechanical assets located specifically on level 2"

def run_baseline_rag():
    client = chromadb.PersistentClient(path="./data/chroma_db")
    collection = client.get_or_create_collection(name="bim_baseline")

    embedder = OllamaEmbeddings(model="nomic-embed-text")
    query_vector = embedder.embed_query(QUERY)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5
    )

    retrieved_docs = results["documents"][0]

    context = "\n\n".join(retrieved_docs)

    prompt = f"""
    You are a BIM analyst.
    Use the following context to answer the query.

    Context: {context}

    Query: {QUERY}

    Answer:"""

    llm = ChatGroq(model="qwen/qwen3-32b", api_key=os.getenv("GROQ_API_KEY"))
    response = llm.invoke(prompt)
    print("=== RETRIEVED CHUNKS ===")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nChunk {i}: \n{doc}")

    print("\n=== LLM RESPOSNE ===")
    print(response.content)

if __name__ == "__main__":
    run_baseline_rag()
