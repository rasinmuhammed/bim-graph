"""
spatial_indexer.py
──────────────────
Indexes IFC files into ChromaDB using storey-grouped, type-grouped chunks.

Previous design: 1 document per element → O(n_elements) chunks.
  • A building with 5,000 elements on one floor = 5,000 ChromaDB docs.
  • Dense top-10 retrieval = 0.2% floor coverage. Statistically useless.

New design: 1 document per (storey × IFC type) group, max 50 elements/chunk.
  • Same 5,000 elements across 20 types = ~100 ChromaDB docs.
  • Dense top-10 = 10 full type-groups = potentially thousands of GUIDs in context.
  • Semantic search now distinguishes query intent (walls vs ducts vs doors)
    rather than finding the most similar individual element name.

Each chunk includes key Pset properties (fire rating, load bearing, etc.) so
semantic search can resolve property queries ("load-bearing walls", "fire-rated
doors") that were previously impossible with name+GUID-only indexing.
"""
import logging
import pathlib
import pickle

import chromadb
import ifcopenshell
import ifcopenshell.util.element
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

logger = logging.getLogger("bim_graph.indexer")

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_CHROMA_PATH  = _PROJECT_ROOT / "data" / "chroma_db"
_BM25_PATH    = _PROJECT_ROOT / "data" / "bm25_index.pkl"

# Properties worth extracting per IFC type — chosen for semantic usefulness
_PSET_PROPS: dict[str, list[str]] = {
    "IfcWall":             ["IsExternal", "LoadBearing", "FireRating", "AcousticRating"],
    "IfcWallStandardCase": ["IsExternal", "LoadBearing", "FireRating", "AcousticRating"],
    "IfcDoor":             ["FireRating", "SecurityRating", "IsExternal", "AcousticRating"],
    "IfcWindow":           ["IsExternal", "FireRating", "AcousticRating", "ThermalTransmittance"],
    "IfcSlab":             ["LoadBearing", "IsExternal", "FireRating"],
    "IfcBeam":             ["LoadBearing", "FireRating", "Reference"],
    "IfcColumn":           ["LoadBearing", "FireRating", "Reference"],
    "IfcStair":            ["FireRating", "NumberOfRiser"],
    "IfcRamp":             ["FireRating"],
    "IfcCovering":         ["IsExternal"],
    "IfcRoof":             ["IsExternal"],
}


def _get_elements_on_storey(model: ifcopenshell.file, storey) -> list:
    """Return all elements directly contained in the given storey via IfcRelContainedInSpatialStructure."""
    elements = []
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        if rel.RelatingStructure == storey:
            elements.extend(rel.RelatedElements)
    return elements


def _extract_props(element) -> dict[str, str]:
    """
    Extract selected Pset property values for this element type.
    Uses ifcopenshell.util.element.get_psets() — handles all Pset traversal internally.
    Returns empty dict if no relevant properties found or type not in _PSET_PROPS.
    """
    target_keys = _PSET_PROPS.get(element.is_a(), [])
    if not target_keys:
        return {}

    try:
        all_psets = ifcopenshell.util.element.get_psets(element)
    except Exception:
        return {}

    result: dict[str, str] = {}
    for pset_props in all_psets.values():
        for key in target_keys:
            if key in pset_props and key not in result:
                val = pset_props[key]
                if val is not None:
                    result[key] = str(val)
    return result


def _build_type_group_chunks(
    ifc_path: pathlib.Path,
    storey_name: str,
    elements: list,
    group_size: int,
) -> tuple[list[str], list[str], list[dict]]:
    """
    Group elements by IFC type and produce one ChromaDB document per
    (storey × type) group of up to `group_size` elements.

    Each document text:
      [Storey: Level 2 | Type: IfcWall | Count: 23]
      Name: W-01 | GUID: 3AxGUt8yz4AQ4P0r12AB12 | IsExternal: True | LoadBearing: True
      Name: W-02 | GUID: 5BzKLm9wv3BR5Q1s34CD34 | FireRating: 60
      ...

    This collapses thousands of individual element docs into O(types) chunks,
    making dense retrieval meaningful at scale.
    """
    by_type: dict[str, list] = {}
    for el in elements:
        by_type.setdefault(el.is_a(), []).append(el)

    ids: list[str]   = []
    docs: list[str]  = []
    metas: list[dict] = []

    for ifc_type, type_els in sorted(by_type.items()):
        total = len(type_els)
        for chunk_idx in range(0, total, group_size):
            chunk_els  = type_els[chunk_idx : chunk_idx + group_size]
            chunk_num  = chunk_idx // group_size
            total_chunks = (total + group_size - 1) // group_size

            header = (
                f"[Storey: {storey_name} | Type: {ifc_type} | "
                f"Count: {total} | Part: {chunk_num + 1}/{total_chunks}]"
            )
            lines = [header]
            for el in chunk_els:
                line = f"Name: {el.Name or '—'} | GUID: {el.GlobalId}"
                props = _extract_props(el)
                if props:
                    prop_str = " | ".join(f"{k}: {v}" for k, v in props.items())
                    line += f" | {prop_str}"
                lines.append(line)

            doc    = "\n".join(lines)
            doc_id = f"{ifc_path.name}__{storey_name}__{ifc_type}__{chunk_num}"

            ids.append(doc_id)
            docs.append(doc)
            metas.append({
                "file_name":   ifc_path.name,
                "floor":       storey_name,
                "entity_type": ifc_type,
                "chunk_idx":   chunk_num,
                "total_count": total,
            })

    return ids, docs, metas


def index_ifc_file(ifc_path: pathlib.Path, client: chromadb.PersistentClient, group_size: int = 50) -> int:
    """
    Parse one IFC file and upsert grouped chunks into ChromaDB.
    Returns the number of chunks indexed.
    """
    logger.info("Indexing IFC file: %s", ifc_path.name)

    try:
        model = ifcopenshell.open(str(ifc_path))
    except Exception as exc:
        logger.error("Failed to open %s: %s", ifc_path, exc)
        return 0

    collection = client.get_or_create_collection(name="bim_baseline")
    embedder   = OllamaEmbeddings(model="nomic-embed-text")

    all_ids: list[str]   = []
    all_docs: list[str]  = []
    all_metas: list[dict] = []

    storeys = model.by_type("IfcBuildingStorey")
    if not storeys:
        # No storeys — index individual elements globally (fallback for non-building IFCs)
        logger.warning("No storeys in %s — indexing all IfcProduct globally.", ifc_path.name)
        elements = list(model.by_type("IfcProduct"))
        ids, docs, metas = _build_type_group_chunks(ifc_path, "Global", elements, group_size)
        all_ids.extend(ids)
        all_docs.extend(docs)
        all_metas.extend(metas)
    else:
        for storey in storeys:
            elements = _get_elements_on_storey(model, storey)
            logger.info("  Storey %r — %d elements → %d chunks",
                        storey.Name, len(elements),
                        (len(set(e.is_a() for e in elements)) if elements else 0))
            if not elements:
                continue
            ids, docs, metas = _build_type_group_chunks(ifc_path, storey.Name, elements, group_size)
            all_ids.extend(ids)
            all_docs.extend(docs)
            all_metas.extend(metas)

    if not all_docs:
        logger.warning("No documents extracted from %s.", ifc_path.name)
        return 0

    # Embed all chunks
    logger.info("  Embedding %d chunks...", len(all_docs))
    all_embeddings = embedder.embed_documents(all_docs)

    # Upsert in batches (ChromaDB hard limit per call)
    _BATCH = 5461
    for start in range(0, len(all_ids), _BATCH):
        sl = slice(start, start + _BATCH)
        collection.upsert(
            ids        = all_ids[sl],
            embeddings = all_embeddings[sl],
            documents  = all_docs[sl],
            metadatas  = all_metas[sl],
        )

    logger.info("Indexed %d chunks from %s.", len(all_docs), ifc_path.name)
    return len(all_docs)


def index_single_file(ifc_path_str: str) -> int:
    """Convenience wrapper for the upload endpoint — creates its own ChromaDB client."""
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    return index_ifc_file(pathlib.Path(ifc_path_str), client)


def build_bm25_from_chroma() -> None:
    """Rebuild the BM25 index from whatever is currently in ChromaDB."""
    logger.info("Rebuilding BM25 index from ChromaDB...")
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    try:
        collection = client.get_collection(name="bim_baseline")
    except Exception as exc:
        logger.error("Could not get ChromaDB collection: %s", exc)
        return

    result = collection.get(include=["documents", "metadatas"])
    docs   = result.get("documents", [])
    metas  = result.get("metadatas", [])

    if not docs:
        logger.warning("ChromaDB collection is empty — nothing to build BM25 from.")
        return

    tokenized = [doc.lower().split() for doc in docs]
    bm25      = BM25Okapi(tokenized)

    with open(_BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": docs, "metas": metas}, f)

    logger.info("BM25 index saved: %d documents → %s", len(docs), _BM25_PATH)


if __name__ == "__main__":
    import sys
    from config import settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    client   = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    data_dir = _PROJECT_ROOT / "data"

    try:
        client.delete_collection("bim_baseline")
        logger.info("Cleared existing bim_baseline collection.")
    except Exception:
        pass

    total = 0
    for ifc_file in sorted(data_dir.glob("*.ifc")):
        total += index_ifc_file(ifc_file, client, group_size=settings.chroma_group_size)

    if total == 0:
        logger.error("No IFC files found in %s — drop .ifc files into data/ first.", data_dir)
        sys.exit(1)

    build_bm25_from_chroma()
    logger.info("Indexing complete. Total chunks: %d", total)
