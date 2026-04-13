import os
import re
import time
import logging
import pathlib
import chromadb
import ifcopenshell
import pickle
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from state import BIMGraphState

load_dotenv()

# ── Paths (absolute, works regardless of CWD) ─────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
IFC_PATH      = str(_PROJECT_ROOT / "data" / "Duplex_A_20110907.ifc")
_CHROMA_PATH  = str(_PROJECT_ROOT / "data" / "chroma_db")
_LOGS_PATH    = str(_PROJECT_ROOT / "logs")
_BM25_PATH    = _PROJECT_ROOT / "data" / "bm25_index.pkl"

# Fast model for constraint extraction (no thinking tokens, near-instant)
_LLM_MODEL     = "llama-3.1-8b-instant"
# Full model for generation and evaluation
_LLM_MODEL_BIG = "qwen/qwen3-32b"

# ── IFC domain knowledge ───────────────────────────────────────────────────────
# Entity types that represent MEP / mechanical / equipment assets in IFC schema
_MEP_TYPES: set[str] = {
    "IfcFlowTerminal", "IfcFlowFitting", "IfcFlowSegment",
    "IfcFlowController", "IfcDistributionFlowElement",
    "IfcEnergyConversionDevice", "IfcFlowMovingDevice",
    "IfcFlowStorageDevice", "IfcAirTerminal", "IfcFlowInstrument",
    "IfcValve", "IfcDuctFitting", "IfcDuctSegment",
    "IfcPipeFitting", "IfcPipeSegment",
    "IfcPump", "IfcFan", "IfcCompressor",
    "IfcBoiler", "IfcChiller", "IfcHeatExchanger", "IfcUnitaryEquipment",
    "IfcDistributionElement", "IfcDistributionControlElement",
    "IfcSanitaryTerminal", "IfcElectricAppliance",
    "IfcLightFixture", "IfcOutlet",
    "IfcSensor", "IfcActuator", "IfcController",
}

# Human-readable guide for the LLM explaining what IFC types mean
_IFC_TYPE_GUIDE = """\
IFC entity type → domain meaning (use this to identify asset categories):
  • IfcFlowTerminal, IfcAirTerminal          → HVAC terminals / diffusers / grilles
  • IfcFlowFitting, IfcDuctFitting           → duct connectors / elbows / tees
  • IfcFlowSegment, IfcDuctSegment           → straight duct runs
  • IfcPipeFitting, IfcPipeSegment           → plumbing pipe connectors / runs
  • IfcFlowController, IfcValve             → control valves / dampers
  • IfcFlowMovingDevice, IfcPump, IfcFan    → pumps, fans, air handlers
  • IfcEnergyConversionDevice, IfcBoiler,
    IfcChiller, IfcHeatExchanger,
    IfcUnitaryEquipment                      → energy plant / major HVAC equipment
  • IfcSanitaryTerminal                      → plumbing fixtures (sinks, toilets)
  • IfcElectricAppliance, IfcLightFixture,
    IfcOutlet                                → electrical equipment
  • IfcSensor, IfcActuator, IfcController   → BMS / controls
  • IfcDistributionElement                   → generic distribution system element
"""

# Keywords that indicate the user is asking about MEP/equipment rather than architecture
_EQUIPMENT_KEYWORDS: set[str] = {
    "equipment", "mechanical", "hvac", "plumbing", "electrical",
    "duct", "pipe", "pump", "fan", "valve", "terminal", "fixture",
    "asset", "system", "mep", "air", "water", "sensor", "actuator",
}

# Token-budget guards — Groq free tier TPM limit for qwen3-32b is 6000 tokens
_MAX_AST_ELEMENTS   = 40    # Max elements passed from AST traversal to the LLM
_MAX_CONTEXT_CHARS  = 3500  # Hard character cap on the context string

# ── Audit Logger Setup ─────────────────────────────────────────────────────────
os.makedirs(_LOGS_PATH, exist_ok=True)
_log_file = str(
    pathlib.Path(_LOGS_PATH)
    / f"bim_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("bim_graph.audit")
logger.info("Audit log initialised → %s", _log_file)


# ── Structured Output Schemas ──────────────────────────────────────────────────
class ConstraintOutput(BaseModel):
    spatial_constraints: str = Field(
        description=(
            "The floor, level, or zone reference extracted verbatim from the query. "
            "Examples: 'Level 2', 'Floor 3', 'Ground Floor', 'Basement'. "
            "Return an empty string if the query mentions no specific floor or level."
        )
    )


class EvaluatorOutput(BaseModel):
    spatial_match: bool = Field(
        description=(
            "True  → the answer lists specific, named assets confirmed to be on the requested floor. "
            "False → the answer is vague, says 'cannot determine', admits missing floor data, "
            "        or names assets without confirming their floor."
        )
    )
    reason: str = Field(
        description="One concise sentence explaining the verdict."
    )


# ── Helpers ────────────────────────────────────────────────────────────────────
def _is_equipment_query(query: str) -> bool:
    """Returns True when the query is about MEP / mechanical / equipment assets."""
    q = query.lower()
    return any(kw in q for kw in _EQUIPMENT_KEYWORDS)


def _is_mep_entity(element) -> bool:
    """Returns True when an IFC element belongs to the MEP/equipment domain."""
    if element.is_a() in _MEP_TYPES:
        return True
    return (
        element.is_a("IfcDistributionElement")
        or element.is_a("IfcEnergyConversionDevice")
    )


# ── Node 0 ─────────────────────────────────────────────────────────────────────
def extract_spatial_constraints(state: BIMGraphState) -> dict:
    """
    Extract the floor / level / zone from a natural-language query.
    Uses structured output so JSON parsing errors are impossible.
    """
    logger.info(
        "▶ [Node 0] extract_spatial_constraints  |  query: %r", state["query"]
    )

    prompt = (
        f'Extract the floor, level, or zone reference from this BIM query.\n'
        f'Query: "{state["query"]}"\n\n'
        f'Return the exact label as written in the query (e.g. "Level 2", "Floor 3", '
        f'"Ground Floor"). Return an empty string if no floor is mentioned.'
    )

    llm    = ChatGroq(model=_LLM_MODEL, api_key=os.getenv("GROQ_API_KEY"))
    result = llm.with_structured_output(ConstraintOutput).invoke(prompt)

    logger.info("  ✓ Extracted spatial constraint: %r", result.spatial_constraints)
    return {"spatial_constraints": result.spatial_constraints}


# ── Node 1 ─────────────────────────────────────────────────────────────────────
def retrieve_hybrid(state: BIMGraphState) -> dict:
    """
    Hybrid retriever: BM25 (lexical) + ChromaDB (vector) merged via
    Reciprocal Rank Fusion (RRF).

    RRF score = Σ  1 / (rank_i + 60)

    Top-5 fused documents are returned as context for Node 2.
    Falls back to dense-only if the BM25 index is missing.
    """
    query = state["query"]
    logger.info("▶ [Node 1] retrieve_hybrid  |  query: %r", query[:80])

    # ── Dense retrieval (ChromaDB) ─────────────────────────────────────────
    logger.info("  Embedding query via Ollama nomic-embed-text …")
    client       = chromadb.PersistentClient(path=_CHROMA_PATH)
    collection   = client.get_or_create_collection(name="bim_baseline")
    embedder     = OllamaEmbeddings(model="nomic-embed-text")
    query_vector = embedder.embed_query(query)

    dense_results = collection.query(
        query_embeddings=[query_vector],
        n_results=10,
        include=["documents"],
    )
    dense_docs = dense_results["documents"][0]

    # ── BM25 retrieval ─────────────────────────────────────────────────────
    bm25_docs: list[str] = []
    if _BM25_PATH.exists():
        with open(_BM25_PATH, "rb") as f:
            payload = pickle.load(f)
        bm25   = payload["bm25"]
        corpus = payload["corpus"]

        tokenised_query = query.lower().split()
        scores          = bm25.get_scores(tokenised_query)
        top_indices     = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        bm25_docs       = [corpus[i] for i in top_indices]
        logger.info("  BM25 top-10 retrieved (%d docs in index).", len(corpus))
    else:
        logger.warning("  BM25 index not found at %s — using dense-only.", _BM25_PATH)

    # ── Reciprocal Rank Fusion ─────────────────────────────────────────────
    K        = 60
    rrf_map: dict[str, float] = {}

    for rank, doc in enumerate(dense_docs):
        rrf_map[doc] = rrf_map.get(doc, 0.0) + 1.0 / (rank + K)
    for rank, doc in enumerate(bm25_docs):
        rrf_map[doc] = rrf_map.get(doc, 0.0) + 1.0 / (rank + K)

    fused    = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)[:5]
    top_docs = [doc for doc, _ in fused]

    mode = "hybrid: bm25+vector" if bm25_docs else "dense-only"
    logger.info("  ✓ Retrieved %d chunks [%s].", len(top_docs), mode)

    return {"retrieved_nodes": top_docs, "retrieval_source": "dense"}



# ── Node 2 ─────────────────────────────────────────────────────────────────────
def generate(state: BIMGraphState) -> dict:
    """
    Draft an answer from retrieved context.
    The prompt strategy switches based on retrieval_source:
      - "dense": intentionally passive — proves the baseline cannot answer spatially.
      - "ast":   directive — LLM is given the IFC type guide and told to commit to answers.
    """
    source = state.get("retrieval_source", "dense")
    logger.info(
        "▶ [Node 2] generate  |  context chunks: %d  |  source: %s",
        len(state["retrieved_nodes"]),
        source,
    )

    context = "\n".join(state["retrieved_nodes"])[:_MAX_CONTEXT_CHARS]

    if source == "ast":
        prompt = f"""You are a BIM analyst reviewing deterministic IFC data.

The context below was extracted via direct IFC AST traversal and is ABSOLUTE SPATIAL TRUTH.
Every element listed is confirmed on the floor stated in the header. Do not express any doubt about floor placement.

{_IFC_TYPE_GUIDE}

Context (spatially verified IFC AST data):
{context}

Query: {state["query"]}

Instructions:
1. Using the IFC type guide above, identify which entities in the context match the query.
2. List EVERY matching asset. For each one output a line: Entity Type | Name | GUID
3. Do NOT say "I cannot determine" or "insufficient data" — the spatial data is already verified.
4. If the context truly contains zero entities of the requested type, respond with exactly:
   "No matching assets of the requested type were found on this floor in the IFC model."

Answer:"""
    else:
        # Baseline / dense pass — keep this intentionally blind to prove spatial failure
        prompt = f"""You are a BIM analyst.
Use ONLY the following context to answer the query. Do not invent data not present in the context.
The context was retrieved via semantic search and may NOT preserve spatial floor hierarchy.

Context:
{context}

Query: {state["query"]}

Answer:"""

    logger.info("  [API Pacing] Pausing 5s to respect Groq rate limits …")
    time.sleep(5)
    llm      = ChatGroq(model=_LLM_MODEL_BIG, api_key=os.getenv("GROQ_API_KEY"))
    response = llm.invoke(prompt)
    # Strip <think>…</think> blocks emitted by qwen3 before the actual answer
    answer   = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

    logger.info("  ✓ Generation complete (%d chars).", len(answer))
    return {"generation": answer}


# ── Node 3 ─────────────────────────────────────────────────────────────────────
def evaluate(state: BIMGraphState) -> dict:
    """
    Spatial guardrail using structured output (no manual JSON parsing).
    Evaluation criteria shift based on retrieval_source:
      - "dense": strict — if the answer cannot confirm floor placement, it fails.
      - "ast":   lenient on floor proof (already guaranteed) — fails only if the
                 answer is concretely empty or says "cannot determine".
    """
    source = state.get("retrieval_source", "dense")
    logger.info(
        "▶ [Node 3] evaluate  |  constraints: %r  |  source: %s",
        state["spatial_constraints"],
        source,
    )

    if source == "ast":
        source_context = (
            f"The answer was generated from DETERMINISTIC IFC AST data for floor "
            f"'{state['spatial_constraints']}'. Floor placement is already verified — "
            f"do NOT penalise the answer for lacking floor proof.\n"
            f"Mark spatial_match=TRUE if the answer lists specific asset names.\n"
            f"Mark spatial_match=TRUE even if the answer says 'No matching assets found' "
            f"(that is a valid IFC result, not a retrieval failure).\n"
            f"Mark spatial_match=FALSE ONLY if the answer says 'I cannot determine', "
            f"'insufficient data', or equivalent uncertainty language."
        )
    else:
        source_context = (
            f"The answer was generated from chunked semantic search. "
            f"Be strict: if the answer cannot confidently confirm which floor each asset "
            f"is on, or if it hedges with 'may be' / 'possibly', mark spatial_match=FALSE."
        )

    prompt = f"""You are a strict BIM spatial auditor.

Answer under review:
\"\"\"{state["generation"]}\"\"\"

User requested assets specifically on: {state["spatial_constraints"]}

{source_context}

Evaluate the answer and return your verdict."""

    logger.info("  [API Pacing] Pausing 5s to respect Groq rate limits …")
    time.sleep(5)
    llm      = ChatGroq(model=_LLM_MODEL_BIG, api_key=os.getenv("GROQ_API_KEY"))
    result   = llm.with_structured_output(EvaluatorOutput).invoke(prompt)
    feedback = result.model_dump()

    match = feedback["spatial_match"]
    logger.info("  Evaluator verdict → spatial_match=%s | reason: %s", match, feedback["reason"])

    if not match:
        if source == "ast":
            logger.warning(
                "  ⚠  AST generation failed to produce concrete answer: %s", feedback["reason"]
            )
            logger.warning("  ↳  Routing to END (AST is deterministic — no further retrieval possible).")
        else:
            logger.warning("  ⚠  Spatial mismatch on dense retrieval: %s", feedback["reason"])
            logger.warning("  ↳  Self-healing will trigger (spatial_ast_retrieval).")

    return {"evaluator_feedback": feedback}


# ── Node 4 ─────────────────────────────────────────────────────────────────────
def spatial_ast_retrieval(state: BIMGraphState) -> dict:
    """
    IDE-Cursor self-healing node. Abandons semantic search entirely.

    1. Opens the IFC model via IfcOpenShell.
    2. Finds the exact IfcBuildingStorey matching the spatial constraint.
    3. Traverses IfcRelContainedInSpatialStructure to get only elements on that floor.
    4. Filters to MEP/equipment types when the query is about mechanical assets —
       this prevents the LLM from being confused by walls, doors, and columns.
    5. Prepends a deterministic spatial-proof header so the generate node knows
       the data is ground truth.
    """
    target   = state["spatial_constraints"]
    query    = state["query"]
    is_equip = _is_equipment_query(query)

    logger.info(
        "▶ [Node 4] spatial_ast_retrieval  |  target storey: %r  |  equipment_filter: %s",
        target,
        is_equip,
    )

    ifc_model     = ifcopenshell.open(IFC_PATH)
    target_storey = None

    for storey in ifc_model.by_type("IfcBuildingStorey"):
        if target.lower() in storey.Name.lower():
            target_storey = storey
            logger.info(
                "  Matched IFC storey: %r  (GUID: %s)", storey.Name, storey.GlobalId
            )
            break

    if target_storey is None:
        logger.error("  No storey matching %r found in IFC model.", target)
        context_lines = [f"ERROR: No storey matching '{target}' found in IFC model."]
    else:
        # Spatial-proof header — tells the generate node this is verified data
        context_lines = [
            f"--- [SOURCE: DETERMINISTIC IFC AST TRAVERSAL | "
            f"CONFIRMED FLOOR: {target_storey.Name} | "
            f"GUID: {target_storey.GlobalId}] ---"
        ]

        all_elements:  list[str] = []
        mep_elements:  list[str] = []
        arch_elements: list[str] = []

        for rel in ifc_model.by_type("IfcRelContainedInSpatialStructure"):
            if rel.RelatingStructure == target_storey:
                for element in rel.RelatedElements:
                    entry = (
                        f"Entity: {element.is_a()} | "
                        f"Name: {element.Name} | "
                        f"GUID: {element.GlobalId}"
                    )
                    all_elements.append(entry)
                    if _is_mep_entity(element):
                        mep_elements.append(entry)
                    else:
                        arch_elements.append(entry)

        # Select the right bucket based on query intent
        if is_equip:
            if mep_elements:
                selected     = mep_elements
                filter_label = "MEP/Equipment only"
            else:
                # No MEP found — pass everything so generate can report honestly
                selected     = all_elements
                filter_label = "ALL (no MEP types found on this floor)"
        else:
            selected     = all_elements
            filter_label = "ALL elements"

        # Cap elements to stay within Groq's token budget
        if len(selected) > _MAX_AST_ELEMENTS:
            logger.info(
                "  Capping context from %d → %d elements (TPM budget).",
                len(selected), _MAX_AST_ELEMENTS,
            )
            selected = selected[:_MAX_AST_ELEMENTS]

        context_lines.extend(selected)

        logger.info(
            "  ✓ Extracted %d elements from storey %r "
            "(filter: %s | total on floor: %d | MEP: %d | arch: %d).",
            len(selected),
            target_storey.Name,
            filter_label,
            len(all_elements),
            len(mep_elements),
            len(arch_elements),
        )

    new_loop_count = state.get("loop_count", 0) + 1
    correction_entry = {
        "attempt":         new_loop_count,
        "search_strategy": "spatial_ast",
        "failure_reason":  state["evaluator_feedback"].get("reason", "Spatial mismatch"),
        "action_taken": (
            f"Traversed IFC AST; extracted "
            f"{'MEP/equipment' if is_equip else 'all'} elements "
            f"from storey '{target}' only"
        ),
    }
    logger.info(
        "  Correction log entry #%d appended. Loop count now: %d",
        correction_entry["attempt"],
        new_loop_count,
    )

    return {
        "retrieved_nodes": context_lines,
        "retrieval_source": "ast",
        "correction_log":  state["correction_log"] + [correction_entry],
        "loop_count":      new_loop_count,
    }
