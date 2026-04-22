import ifcopenshell
import re
import time
import logging
import pathlib
import chromadb
import pickle
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from agent.state import BIMGraphState
from config import settings
from graph_db import queries as gq
from observability.logging import set_request_id
from agent.token_stream import get_token_queue

load_dotenv()


_logger = logging.getLogger("bim_graph.nodes")

# starts on Groq, switches to Cerebras on daily quota exhaustion
_provider: str = "groq"


def _is_daily_quota(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "tokens per day" in msg or "tpd" in msg


def _is_retryable(exc: BaseException) -> bool:
    global _provider
    if _is_daily_quota(exc):
        if _provider == "groq":
            print("\n[provider] Groq daily quota exhausted — switching to Cerebras permanently.")
            _provider = "cerebras"
            _get_llm_fast.cache_clear()
            _get_llm_big.cache_clear()
        return True
    msg = str(exc).lower()
    return (
        "rate_limit_exceeded" in msg
        or "429" in msg
        or "rate limit" in msg
        or "queue_exceeded" in msg
        or "high traffic" in msg
    )


_llm_retry = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=1, max=32),
    stop=stop_after_attempt(8),
    before_sleep=before_sleep_log(_logger, logging.WARNING),
    reraise=True,
)

_CHROMA_PATH   = settings.chroma_path
_BM25_PATH     = pathlib.Path(settings.bm25_path)
_LLM_MODEL     = settings.llm_model
_LLM_MODEL_BIG = settings.llm_model_big

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

# Token-budget guards — Groq Llama-3.3-70b-versatile has a large context window
_MAX_AST_ELEMENTS   = settings.max_ast_elements
_MAX_CONTEXT_CHARS  = settings.max_context_chars   # fixed: was max_content_chars (typo)

# ── Lazy singletons — initialized on first use, not at import time ────────────
# This prevents expensive I/O (Ollama load, ChromaDB open, HTTP connections)
# from running during test collection when nodes.py is merely imported.
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text")

@lru_cache(maxsize=1)
def _get_chroma_collection():
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    return client.get_or_create_collection(name="bim_baseline")

@lru_cache(maxsize=1)
def _get_llm_fast():
    if _provider == "cerebras":
        _logger.info("LLM fast: Cerebras llama3.1-8b")
        return ChatOpenAI(
            model="llama3.1-8b",
            api_key=settings.cerebras_api_key,
            base_url=settings.cerebras_base_url,
        )
    _logger.info("LLM fast: Groq %s", settings.llm_model)
    return ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key)


@lru_cache(maxsize=1)
def _get_llm_big():
    if _provider == "cerebras":
        _logger.info("LLM big: Cerebras qwen-3-235b")
        return ChatOpenAI(
            model="qwen-3-235b-a22b-instruct-2507",
            api_key=settings.cerebras_api_key,
            base_url=settings.cerebras_base_url,
        )
    _logger.info("LLM big: Groq %s", settings.llm_model_big)
    return ChatGroq(model=settings.llm_model_big, api_key=settings.groq_api_key)

@lru_cache(maxsize=1)
def _get_bm25_payload() -> dict | None:
    if not _BM25_PATH.exists():
        logger.warning("BM25 index not found at %s", _BM25_PATH)
        return None
    with open(_BM25_PATH, "rb") as f:
        payload = pickle.load(f)
    logger.info("BM25 index loaded: %d docs", len(payload["corpus"]))
    return payload

# IFC model cache - avoids re-parsing the same file on every AST call
_ifc_cache: dict[str, ifcopenshell.file] = {}

def _get_ifc(filename: str) -> ifcopenshell.file:
    """Load IFC file from disk or cache, returning a parsed IfcOpenShell file object."""
    if filename not in _ifc_cache:
        path = pathlib.Path(settings.ifc_data_dir) / filename
        logger.info("Parsing IFC file: %s (first access)", filename)
        _ifc_cache[filename] = ifcopenshell.open(str(path))
    return _ifc_cache[filename]

logger = logging.getLogger("bim_graph.nodes")


# ── Structured Output Schemas ──────────────────────────────────────────────────
class ConstraintOutput(BaseModel):
    spatial_constraints: str = Field(
        description=(
            "The floor, level, or zone reference extracted verbatim from the query. "
            "Examples: 'Level 2', 'Floor 3', 'Ground Floor', 'Basement'. "
            "Return an empty string if the query mentions no specific floor or level."
        )
    )
    is_inventory_query: bool = Field(
        default=False,
        description=(
            "Set to True ONLY if the query asks for an exhaustive list, inventory, or 'what is present'. "
            "Set to False for specific targeted questions like 'Where is boiler X?' or 'Does Level 2 have any doors?'."
        )
    )

    @field_validator("is_inventory_query", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return v


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

    @field_validator("spatial_match", mode="before")
    @classmethod
    def _coerce_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return v


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
    _t0 = time.perf_counter()
    set_request_id(state.get("request_id", "-"))
    logger.info(
        "▶ [Node 0] extract_spatial_constraints  |  query: %r", state["query"]
    )

    prompt = (
        f'Analyze this BIM query and respond ONLY with a JSON object — no markdown, no explanation.\n\n'
        f'Query: "{state["query"]}"\n\n'
        f'Required JSON format:\n'
        f'{{"spatial_constraints": "<floor name or empty string>", "is_inventory_query": true or false}}\n\n'
        f'Rules:\n'
        f'- spatial_constraints: the exact floor/level/zone (e.g. "Level 2", "Ground Floor"). '
        f'Empty string if no floor is mentioned.\n'
        f'- is_inventory_query: true if asking for exhaustive list/inventory ("what is present", '
        f'"list all", "every element"). false for targeted questions.'
    )

    llm = _get_llm_fast()
    # Use json_mode to avoid Groq server-side tool-call schema validation, which
    # rejects boolean-as-string outputs from some Llama models before we can coerce them.
    result = _llm_retry(
        llm.with_structured_output(ConstraintOutput, method="json_mode").invoke
    )(prompt)

    elapsed = time.perf_counter() - _t0
    logger.info("  ✓ Extracted spatial constraint: %r (Inventory: %s) [%.2fs]",
                result.spatial_constraints, result.is_inventory_query, elapsed)
    return {
        "spatial_constraints": result.spatial_constraints,
        "is_inventory_query":  result.is_inventory_query,
        "node_timings": {**state.get("node_timings", {}), "extract_spatial_constraints": elapsed},
    }


# ── Node 1 ─────────────────────────────────────────────────────────────────────
def retrieve_hybrid(state: BIMGraphState) -> dict:
    """
    Hybrid retriever: BM25 (lexical) + ChromaDB (vector) merged via
    Reciprocal Rank Fusion (RRF).

    RRF score = Σ  1 / (rank_i + 60)

    Top-5 fused documents are returned as context for Node 2.
    Falls back to dense-only if the BM25 index is missing.
    """
    def _expand_query(query: str) -> str:
        """Append IFC entity type names when query is about MEP/equipment."""
        if not _is_equipment_query(query):
            return query
        
        #MAP human terms to IFC tpye names the index actually contains
        expansions = {
            "hvac":       ["IfcAirTerminal", "IfcUnitaryEquipment", "IfcFan"],
            "pump":       ["IfcPump"],
            "duct":       ["IfcDuctSegment", "IfcDuctFitting"],
            "pipe":       ["IfcPipeSegment", "IfcPipeFitting"],
            "valve":      ["IfcValve"],
            "sensor":     ["IfcSensor"],
            "electrical": ["IfcElectricAppliance", "IfcOutlet", "IfcLightFixture"],
        }
        q_lower = query.lower()
        extra_terms: list[str] = []
        for keyword, ifc_types in expansions.items():
            if keyword in q_lower:
                extra_terms.extend(ifc_types)
        
        if extra_terms:
            return f"{query} {' '.join(extra_terms)}"
        return query

            
    _t0   = time.perf_counter()
    query = state["query"]
    set_request_id(state.get("request_id", "-"))
    logger.info("▶ [Node 1] retrieve_hybrid  |  query: %r", query[:80])

    # ── Dense retrieval (ChromaDB) ─────────────────────────────────────────
    logger.info("  Embedding query via Ollama nomic-embed-text …")
    expanded = _expand_query(query)
    query_vector = _get_embedder().embed_query(expanded)
    logger.info("  Query expanded: %r → %r", query[:60], expanded[:80])

    ifc_filename = state.get("ifc_filename")
    floor        = state.get("spatial_constraints", "")

    # Build the tightest valid ChromaDB $where filter we can construct.
    # Each grouped chunk now has entity_type metadata, enabling type-level filtering.
    and_clauses: list[dict] = []
    if ifc_filename:
        and_clauses.append({"file_name": {"$eq": ifc_filename}})
    if floor:
        and_clauses.append({"floor": {"$eq": floor}})

    if len(and_clauses) == 0:
        where_filter = None
    elif len(and_clauses) == 1:
        where_filter = and_clauses[0]
    else:
        where_filter = {"$and": and_clauses}

    # With grouped chunks, each result represents up to `chroma_group_size` elements.
    # Fetch more candidates (15) so RRF has richer signal to rank.
    _DENSE_TOP_K = 15
    dense_results = _get_chroma_collection().query(
        query_embeddings=[query_vector],
        n_results=_DENSE_TOP_K,
        where=where_filter,
        include=["documents"],
    )
    dense_docs = dense_results["documents"][0] if dense_results["documents"] else []

    # ── BM25 retrieval ─────────────────────────────────────────────────────
    bm25_docs: list[str] = []
    bm25_payload = _get_bm25_payload()
    if bm25_payload is not None:
        bm25   = bm25_payload["bm25"]
        corpus = bm25_payload["corpus"]
        metas  = bm25_payload.get("metas", [])

        tokenised_query = query.lower().split()
        scores          = bm25.get_scores(tokenised_query)

        # Filter by file_name and optionally floor to match ChromaDB behaviour
        filtered_indices = [
            i for i in range(len(scores))
            if (not ifc_filename or not metas
                or (i < len(metas) and metas[i].get("file_name") == ifc_filename))
            and (not floor or not metas
                 or (i < len(metas) and metas[i].get("floor") == floor))
        ]
        top_indices = sorted(filtered_indices, key=lambda i: scores[i], reverse=True)[:_DENSE_TOP_K]
        bm25_docs   = [corpus[i] for i in top_indices]
        logger.info("  BM25 top-%d retrieved (%d docs in index).", _DENSE_TOP_K, len(corpus))
    else:
        logger.warning("  BM25 index not found at %s — using dense-only.", _BM25_PATH)

    # ── Reciprocal Rank Fusion ─────────────────────────────────────────────
    K        = settings.rrf_k
    rrf_map: dict[str, float] = {}

    for rank, doc in enumerate(dense_docs):
        rrf_map[doc] = rrf_map.get(doc, 0.0) + 1.0 / (rank + K)
    for rank, doc in enumerate(bm25_docs):
        rrf_map[doc] = rrf_map.get(doc, 0.0) + 1.0 / (rank + K)

    fused    = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)[:settings.retrieval_top_k]
    top_docs = [doc for doc, _ in fused]

    mode    = "hybrid: bm25+vector" if bm25_docs else "dense-only"
    elapsed = time.perf_counter() - _t0
    logger.info("  ✓ Retrieved %d chunks [%s] [%.2fs].", len(top_docs), mode, elapsed)

    return {
        "retrieved_nodes":  top_docs,
        "retrieval_source": "dense",
        "node_timings": {**state.get("node_timings", {}), "retrieve_hybrid": elapsed},
    }



# ── Node 2 ─────────────────────────────────────────────────────────────────────
def generate(state: BIMGraphState) -> dict:
    """
    Draft an answer from retrieved context.
    The prompt strategy switches based on retrieval_source:
      - "dense": intentionally passive — proves the baseline cannot answer spatially.
      - "ast":   directive — LLM is given the IFC type guide and told to commit to answers.
    """
    _t0    = time.perf_counter()
    source = state.get("retrieval_source", "dense")
    set_request_id(state.get("request_id", "-"))
    logger.info(
        "▶ [Node 2] generate  |  context chunks: %d  |  source: %s",
        len(state["retrieved_nodes"]),
        source,
    )

    # Slice at document boundaries, not character boundaries
    context_parts: list[str] = []
    char_count = 0
    for doc in state["retrieved_nodes"]:
        if char_count + len(doc) > _MAX_CONTEXT_CHARS:
            break
        context_parts.append(doc)
        char_count += len(doc)
    context = "\n".join(context_parts)

    if source in ("ast", "graph"):
        # Both AST and graph results are spatially verified ground truth.
        # AST = deterministic IFC traversal. Graph = Cypher query against Neo4j hierarchy.
        # Neither needs hedging language — the data is correct by construction.
        source_label = (
            "DETERMINISTIC IFC AST TRAVERSAL" if source == "ast"
            else "NEO4J GRAPH DATABASE (Cypher query against IFC hierarchy)"
        )
        prompt = f"""You are a BIM analyst reviewing verified BIM data.

The context below was extracted via {source_label} and is SPATIALLY CONFIRMED.
Every element listed is confirmed on the floor stated in the header. Do not express doubt about floor placement.

{_IFC_TYPE_GUIDE}

Context (spatially verified data):
{context}

Query: {state["query"]}

Instructions:
1. If the query asks for specific equipment, use the IFC type guide to identify matches.
   If the query asks 'What is present?', 'List all', or implies a general inventory, list EVERY entity in the context.
2. For each matching asset, output a line: [Entity Type] | [Name] | [GUID]
3. Do NOT say "I cannot determine" or "insufficient data" — the spatial data is already verified.
4. If the context truly contains zero matching entities, respond with:
   "No matching assets of the requested type were found on this floor in the IFC model."

Answer:"""
    else:
        # Dense / hybrid pass — intentionally neutral to prove spatial blindness in the baseline
        prompt = f"""You are a BIM analyst.
Use ONLY the following context to answer the query. Do not invent data not present in the context.
The context was retrieved via semantic search and may NOT preserve spatial floor hierarchy.

Context:
{context}

Query: {state["query"]}

Answer:"""

    llm     = _get_llm_big()
    token_q = get_token_queue()   # None when called outside HTTP (benchmark, tests)

    def _stream_generate() -> list[str]:
        buf: list[str] = []
        for chunk in llm.stream(prompt):
            text = chunk.content
            if text:
                buf.append(text)
                if token_q is not None:
                    token_q.put(text)
        return buf

    chunks = _llm_retry(_stream_generate)()

    # Signal the SSE drain loop that generation is done
    if token_q is not None:
        token_q.put(None)

    raw_answer = "".join(chunks)
    answer     = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

    # Extract IFC GUIDs (22-char base64-like strings) from the generated answer.
    # Stored explicitly in state so the benchmark oracle and evaluator can use
    # the exact set without re-running regex on free text downstream.
    extracted_guids = re.findall(r"[A-Za-z0-9$_]{22}", answer)

    elapsed     = time.perf_counter() - _t0
    token_count = len(context.split())
    logger.info(
        "  ✓ Generation complete (%d chars, %d context-tokens, %d GUIDs extracted) [%.2fs].",
        len(answer), token_count, len(extracted_guids), elapsed,
    )
    return {
        "generation":          answer,
        "extracted_guids":     extracted_guids,
        "context_token_count": token_count,
        "node_timings": {**state.get("node_timings", {}), "generate": elapsed},
    }


# ── Node 3 ─────────────────────────────────────────────────────────────────────
def evaluate(state: BIMGraphState) -> dict:
    """
    Spatial guardrail using structured output (no manual JSON parsing).
    Evaluation criteria shift based on retrieval_source:
      - "dense": strict — if the answer cannot confirm floor placement, it fails.
      - "ast":   lenient on floor proof (already guaranteed) — fails only if the
                 answer is concretely empty or says "cannot determine".
    """
    _t0    = time.perf_counter()
    source = state.get("retrieval_source", "dense")
    set_request_id(state.get("request_id", "-"))
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
            f"The answer was generated from chunked semantic search.\n"
            f"EXPECTATIONS FOR DENSE SEARCH:\n"
            f"1. EXHAUSTIVENESS FAILURE: If the user query asks for an inventory (e.g. 'What is present in...', 'List all...', 'Show me every...'), dense search is mathematically guaranteed to miss elements.\n"
            f"   → In this case, you MUST mark spatial_match=FALSE and state 'Dense search cannot guarantee an exhaustive list' in the reason.\n"
            f"2. SPATIAL BLINDNESS: If the answer cannot confidently confirm which floor each asset is on, or if it hedges with 'may be' / 'possibly'.\n"
            f"   → Mark spatial_match=FALSE."
        )

    prompt = f"""You are a strict BIM spatial auditor.

Answer under review:
\"\"\"{state["generation"]}\"\"\"


User requested assets specifically on: {state["spatial_constraints"]}

{source_context}

Evaluate the answer and return your verdict as a JSON object with keys "spatial_match" (boolean) and "reason" (string). No markdown."""

    llm      = _get_llm_big()
    result   = _llm_retry(
        llm.with_structured_output(EvaluatorOutput, method="json_mode").invoke
    )(prompt)
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

    elapsed = time.perf_counter() - _t0
    logger.info("  ✓ Evaluation complete [%.2fs].", elapsed)
    return {
        "evaluator_feedback": feedback,
        "node_timings": {**state.get("node_timings", {}), "evaluate": elapsed},
    }


_FOUNDATION_KW = {"foundation", "fdn", "basement", "cellar", "underground", "sub-grade"}
_GROUND_KW     = {"ground floor", "ground level", "first floor", "lower floor",
                   "lower level", "parterre", "erd", "00 ground"}
_UPPER_KW      = {"upper floor", "upper level", "second floor", "top floor",
                   "obergeschoss", "dachgeschoss", "attic", "penthouse"}
# "roof" absent so exact-match still catches an actual "Roof" storey
_ROOF_KW       = {"roof", "rooftop"}


def _resolve_storey(ifc_model, target: str):
    """Resolve informal floor names to IfcBuildingStorey via exact → substring → elevation rules → fuzzy."""
    import difflib

    storeys = [s for s in ifc_model.by_type("IfcBuildingStorey") if s.Name]
    if not storeys:
        return None

    t = target.lower().strip()

    for s in storeys:
        if s.Name.lower().strip() == t:
            return s

    for s in storeys:
        n = s.Name.lower()
        if t in n or n in t:
            return s

    def _elev(s):
        try:
            return float(s.Elevation or 0)
        except (TypeError, ValueError):
            return 0.0

    by_elev = sorted(storeys, key=_elev)

    if any(k in t for k in _FOUNDATION_KW):
        for s in by_elev:
            if any(k in s.Name.lower() for k in _FOUNDATION_KW):
                return s
        return by_elev[0]

    if any(k in t for k in _GROUND_KW) or t in {"ground floor", "ground level", "first floor"}:
        non_fdn = [s for s in by_elev if not any(k in s.Name.lower() for k in _FOUNDATION_KW)]
        return (non_fdn or by_elev)[0]

    if any(k in t for k in _UPPER_KW) or "upper" in t or "second" in t:
        non_roof = [s for s in by_elev if not any(k in s.Name.lower() for k in _ROOF_KW)]
        return (non_roof or by_elev)[-1]

    names_lower = [s.Name.lower() for s in storeys]
    close = difflib.get_close_matches(t, names_lower, n=1, cutoff=0.5)
    if close:
        for s in storeys:
            if s.Name.lower() == close[0]:
                return s

    return None


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
    _t0      = time.perf_counter()
    target   = state["spatial_constraints"]
    query    = state["query"]
    is_equip = _is_equipment_query(query)
    set_request_id(state.get("request_id", "-"))
    logger.info(
        "▶ [Node 4] spatial_ast_retrieval  |  target storey: %r  |  equipment_filter: %s",
        target,
        is_equip,
    )

    ifc_filename = state.get("ifc_filename")
    if not ifc_filename:
         logger.error("No IFC filename provided in state.")
         return {"retrieved_nodes": [], "retrieval_source": "ast", "correction_log": state.get("correction_log", [])}
         
    ifc_model     = _get_ifc(ifc_filename)
    target_storey = _resolve_storey(ifc_model, target)

    if target_storey is None:
        logger.error("  No storey matching %r found in IFC model.", target)
        context_lines = [f"ERROR: No storey matching '{target}' found in IFC model."]
    else:
        logger.info("Storey resolved: %r → %r (GUID: %s)",
                    target, target_storey.Name, target_storey.GlobalId)
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

        total_selected = len(selected)

        # Priority-ordered truncation: sort by IFC type so related elements stay
        # together and the LLM sees a representative sample rather than a random cut.
        # For equipment queries, MEP types are already isolated so no resorting needed.
        if not is_equip:
            selected = sorted(selected, key=lambda s: s.split("|")[0])  # sort by Entity type prefix

        if total_selected > _MAX_AST_ELEMENTS:
            shown   = _MAX_AST_ELEMENTS
            omitted = total_selected - shown
            logger.info(
                "  Floor has %d elements — showing %d, omitting %d (token budget).",
                total_selected, shown, omitted,
            )
            selected = selected[:shown]
            context_lines.append(
                f"[NOTE: Floor has {total_selected} elements. "
                f"Showing {shown} — {omitted} omitted. "
                f"Narrow your query to a specific type for complete results.]"
            )

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
    
    # Determine why we are running AST Proof (Self-Heal vs Early Route)
    fb_reason = state["evaluator_feedback"].get("reason", "Spatial mismatch") if state.get("evaluator_feedback") else "Early Routing logic intercepted inventory query."
    
    correction_entry = {
        "attempt":         new_loop_count,
        "search_strategy": "deterministic_ifc_ast",
        "failure_reason":  fb_reason,
        "action_taken":    "Crawled strictly the required IfcBuildingStorey to guarantee spatial match."
    }
    logger.info(
        "  Correction log entry #%d appended. Loop count now: %d",
        correction_entry["attempt"],
        new_loop_count,
    )

    elapsed      = time.perf_counter() - _t0
    token_count  = sum(len(line.split()) for line in context_lines)
    logger.info("  ✓ AST retrieval complete (%d tokens) [%.2fs].", token_count, elapsed)
    return {
        "retrieved_nodes":     context_lines,
        "retrieval_source":    "ast",
        "correction_log":      state["correction_log"] + [correction_entry],
        "loop_count":          new_loop_count,
        "context_token_count": token_count,
        "node_timings": {**state.get("node_timings", {}), "spatial_ast_retrieval": elapsed},
    }


def _resolve_graph_storey(target: str, available_names: list[str]) -> str | None:
    """Fuzzy match informal floor names against strict names in Neo4j."""
    import difflib
    if not available_names:
        return None
        
    t = target.lower().strip()
    
    # 1. Exact match
    for n in available_names:
        if n.lower().strip() == t:
            return n
            
    # 2. Substring match
    for n in available_names:
        nl = n.lower()
        if t in nl or nl in t:
            return n
            
    # 3. Known keyword mappings (Foundation/Ground/Upper)
    if any(k in t for k in _FOUNDATION_KW):
        for n in available_names:
            if any(k in n.lower() for k in _FOUNDATION_KW):
                return n
                
    if any(k in t for k in _GROUND_KW):
        for n in available_names:
            if any(k in n.lower() for k in _GROUND_KW):
                return n
                
    if any(k in t for k in _UPPER_KW):
        for n in available_names:
            if any(k in n.lower() for k in _UPPER_KW):
                return n
                
    # 4. Fuzzy difflib match
    close = difflib.get_close_matches(t, [n.lower() for n in available_names], n=1, cutoff=0.5)
    if close:
        # Find original name
        for n in available_names:
            if n.lower() == close[0]:
                return n
                
    return None



# ── Node 5 ─────────────────────────────────────────────────────────────────────
def graph_query(state: BIMGraphState) -> dict:
    """
    Neo4j graph retrieval node — the primary spatial retrieval path.

    WHY THIS IS BETTER THAN DENSE RETRIEVAL:
    Dense retrieval embeds the query and finds semantically similar text chunks.
    It cannot guarantee completeness or spatial accuracy.

    Graph retrieval executes a Cypher query against a structured graph of the
    IFC hierarchy. It returns every element on the requested floor by definition —
    no approximation, no hallucination risk, no evaluator loop needed.

    ROUTING LOGIC (set in graph.py):
    - If spatial constraints + Neo4j available  → this node (graph_query)
    - Otherwise                                  → retrieve_hybrid

    FALLBACK:
    If Neo4j is down or the file isn't loaded yet, this node returns
    retrieval_source="graph_unavailable" so the evaluator can route to AST.
    """
    _t0   = time.perf_counter()
    floor = state.get("spatial_constraints", "")
    ifc_f = state.get("ifc_filename", "")
    set_request_id(state.get("request_id", "-"))
    logger.info("▶ [Node 5] graph_query  |  floor: %r  |  file: %r", floor, ifc_f)

    # ── Guard: Neo4j must be available and the file must be loaded ─────────────
    if not gq.is_graph_available():
        logger.warning("  Neo4j unavailable — falling back to dense retrieval route.")
        return {
            "retrieved_nodes":    [],
            "retrieval_source":   "graph_unavailable",
            "graph_result_count": 0,
            "node_timings": {**state.get("node_timings", {}), "graph_query": time.perf_counter() - _t0},
        }

    if not gq.is_file_loaded(ifc_f):
        logger.warning("  IFC file %r not in Neo4j — run loader first. Falling back.", ifc_f)
        return {
            "retrieved_nodes":    [],
            "retrieval_source":   "graph_unavailable",
            "graph_result_count": 0,
            "node_timings": {**state.get("node_timings", {}), "graph_query": time.perf_counter() - _t0},
        }

    # ── Fuzzy Floor Resolution ────────────────────────────────────────────────
    # Extracted floor names (e.g. "ground floor") often mismatch IFC names (e.g. "Level 1")
    available_names = gq.get_all_storey_names(ifc_f)
    resolved_floor  = _resolve_graph_storey(floor, available_names)
    
    if resolved_floor:
        if resolved_floor != floor:
            logger.info("  Floor resolved: %r → %r", floor, resolved_floor)
        floor = resolved_floor
    else:
        logger.warning("  Could not resolve floor %r in Neo4j. Available: %r", floor, available_names)

    # ── Select Cypher query based on query intent ──────────────────────────────
    is_equip = _is_equipment_query(state.get("query", ""))
    is_inv   = state.get("is_inventory_query", False)

    if is_equip and not is_inv:
        records = gq.get_mep_elements_on_floor(floor, ifc_f)
        strategy = "graph_mep_filter"
    else:
        records = gq.get_all_elements_on_floor(floor, ifc_f)
        strategy = "graph_full_inventory"


    context_lines = gq.format_results_as_context(records, floor)
    token_count   = sum(len(line.split()) for line in context_lines)
    elapsed       = time.perf_counter() - _t0

    logger.info(
        "  ✓ Graph query complete: %d records, strategy=%s [%.2fs]",
        len(records), strategy, elapsed,
    )

    return {
        "retrieved_nodes":     context_lines,
        "retrieval_source":    "graph",
        "graph_result_count":  len(records),
        "context_token_count": token_count,
        "node_timings": {**state.get("node_timings", {}), "graph_query": elapsed},
    }
