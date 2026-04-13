"""
baseline_runner.py
──────────────────
Runs a single query through the naive ChromaDB RAG path (no agentic loop).
This is Phase 2 — the system we are trying to beat.
"""
import os
import re
import time
import pathlib
import chromadb
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_CHROMA_PATH  = str(_PROJECT_ROOT / "data" / "chroma_db")
_N_RESULTS    = 5


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def run_baseline(query: str) -> dict:
    """
    Run one query through the naive RAG pipeline.

    Returns
    -------
    {
        "query": str,
        "chunks_retrieved": int,
        "generation": str,
        "retrieved_docs": list[str],
    }
    """
    client     = chromadb.PersistentClient(path=_CHROMA_PATH)
    collection = client.get_or_create_collection(name="bim_baseline")
    embedder   = OllamaEmbeddings(model="nomic-embed-text")

    query_vector = embedder.embed_query(query)
    results      = collection.query(query_embeddings=[query_vector], n_results=_N_RESULTS)
    docs         = results["documents"][0]

    context = "\n\n".join(docs)
    prompt  = f"""You are a BIM analyst.
Use ONLY the following context to answer the query.
If the context does not contain enough spatial information, say so explicitly.

Context:
{context}

Query: {query}

Answer:"""

    # Small pacing delay for Groq free-tier rate limits
    time.sleep(5)
    llm      = ChatGroq(model="qwen/qwen3-32b", api_key=os.getenv("GROQ_API_KEY"))
    response = llm.invoke(prompt)
    answer   = _strip_thinking(response.content)

    return {
        "query":            query,
        "chunks_retrieved": len(docs),
        "generation":       answer,
        "retrieved_docs":   docs,
    }
