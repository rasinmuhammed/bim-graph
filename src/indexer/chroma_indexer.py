import pathlib
import chromadb
from langchain_ollama import OllamaEmbeddings

_PROJECT_ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB_PATH = str(_PROJECT_ROOT / "data" / "chroma_db")

def index_chunks(chunks: list[str], db_path: str = _DEFAULT_DB_PATH) -> None:

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="bim_baseline")
    embedder = OllamaEmbeddings(model="nomic-embed-text")

    ids, embeddings, documents = [], [], []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        embeddings.append(embedder.embed_query(chunk))
        documents.append(chunk)

    collection.add(ids=ids, embeddings=embeddings, documents=documents)
    print(f"Indexed {len(chunks)} chunks into ChromaDB.")

if __name__ == "__main__":
    from naive_chunker import chunk_ifc_to_text
    chunks = chunk_ifc_to_text()
    index_chunks(chunks)

        

