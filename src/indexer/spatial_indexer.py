import pathlib
import ifcopenshell
import chromadb
from langchain_ollama import OllamaEmbeddings
import logging
from typing import List, Dict, Any
import pickle
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_CHROMA_PATH = _PROJECT_ROOT / "data" / "chroma_db"
_BM25_PATH = _PROJECT_ROOT / "data" / "bm25_index.pkl"

def index_ifc_file(ifc_path: pathlib.Path, client: chromadb.PersistentClient) -> None:
    logger.info(f"Indexing IFC file: {ifc_path.name}")
    
    try:
        model = ifcopenshell.open(str(ifc_path))
    except Exception as e:
        logger.error(f"Failed to open {ifc_path}: {e}")
        return

    collection = client.get_or_create_collection(name="bim_baseline")
    embedder = OllamaEmbeddings(model="nomic-embed-text")

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    # Iterate by building storey to inject spatial context
    storeys = model.by_type("IfcBuildingStorey")
    
    if not storeys:
        logger.warning(f"No storeys found in {ifc_path.name}. Extracting elements globally.")
        # Fallback to naive chunking if no spatial hierarchy
        elements = model.by_type("IfcProduct")
        for i, element in enumerate(elements):
             doc_str = f"Entity: {element.is_a()} | Name: {element.Name} | GUID: {element.GlobalId}"
             docs = [doc_str[i:i+500] for i in range(0, len(doc_str), 500)]
             for j, d in enumerate(docs):
                 ids.append(f"{ifc_path.name}_global_{i}_{j}")
                 documents.append(d)
                 embeddings.append(embedder.embed_query(d))
                 metadatas.append({"file_name": ifc_path.name, "entity_type": element.is_a()})
    else:
        for storey in storeys:
           logger.info(f"  Indexing storey: {storey.Name}")
           # Get all elements related to this storey
           # Elements are typically contained via IfcRelContainedInSpatialStructure
           elements = []
           for rel in getattr(storey, 'ContainsElements', []):
               for element in rel.RelatedElements:
                   elements.append(element)
           
           for i, element in enumerate(elements):
               # Spatial wrapper format
               doc_str = f"[Storey: {storey.Name}] Entity: {element.is_a()} | Name: {element.Name} | GUID: {element.GlobalId}"
               
               # Chunk if necessary (usually elements are small enough, but just in case)
               chunks = [doc_str[i:i+500] for i in range(0, len(doc_str), 500)]
               
               for j, chunk in enumerate(chunks):
                   ids.append(f"{ifc_path.name}_{storey.Name}_{i}_{j}")
                   documents.append(chunk)
                   embeddings.append(embedder.embed_query(chunk))
                   metadatas.append({
                       "file_name": ifc_path.name,
                       "floor": storey.Name,
                       "entity_type": element.is_a()
                   })

    if documents:
        # ChromaDB batch limits
        batch_size = 5461 # Chroma limit
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        logger.info(f"Successfully indexed {len(documents)} chunks from {ifc_path.name} into Chroma.")
    else:
        logger.warning(f"No documents extracted from {ifc_path.name}")
        
    return documents

def build_bm25_from_chroma():
    """Rebuilds the BM25 index using all documents currently in Chroma."""
    logger.info("Rebuilding BM25 index from ChromaDB...")
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    try:
        collection = client.get_collection(name="bim_baseline")
    except Exception as e:
        logger.error(f"Could not get collection: {e}")
        return

    result = collection.get(include=["documents", "metadatas"])
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    
    if not docs:
         logger.warning("No documents in ChromaDB to build BM25 index.")
         return

    tokenized_corpus = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(_BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": docs, "metas": metas}, f)
        
    logger.info(f"Successfully saved BM25 index with {len(docs)} documents.")

if __name__ == "__main__":
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    
    data_dir = _PROJECT_ROOT / "data"
    
    # Optional: Clear existing collection
    try:
         client.delete_collection("bim_baseline")
         logger.info("Cleared existing bim_baseline collection.")
    except Exception:
         pass
         
    # Index all IFC files in the data directory
    all_chunks = []
    for ifc_file in data_dir.glob("*.ifc"):
       index_ifc_file(ifc_file, client)
       
    # Build a unified BM25 index
    build_bm25_from_chroma()
