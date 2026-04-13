"""
Builds a BM25 lexical index from all chunks stored in ChromaDB.
Saves the index to data/bm25_index.pkl for use by Node 1.

Run once after indexing IFC data:
python src/indexer/bm25_indexer.py
"""

import pickle
import pathlib
import chromadb
from rank_bm25 import BM25Okapi

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_CHROMA_PATH = str(_PROJECT_ROOT / "data" / "chroma_db")
_BM25_PATH = _PROJECT_ROOT / "data" / "bm25_index.pkl"

def build_bm25_index(db_path: str = _CHROMA_PATH) -> None:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("bim_baseline")

    all_docs = collection.get(include=["documents"])
    corpus = all_docs["documents"]

    if not corpus:
        print("ERROR: ChromaDB collection is empty. Run the IFC indexer first.")
        return

    # Tokenise
    tokenised = [doc.lower().split() for doc in corpus]

    bm25 = BM25Okapi(tokenised)
    payload = {"bm25": bm25, "corpus": corpus}

    with open(_BM25_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"✅ Built BM25 index from {len(corpus)} documents.")
    print(f"Saved to {_BM25_PATH}")

if __name__ == "__main__":
    build_bm25_index()
    