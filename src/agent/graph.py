from langgraph.graph import StateGraph, END
from agent.state import BIMGraphState
from agent.nodes import (
    extract_spatial_constraints,
    retrieve_hybrid,
    generate,
    evaluate,
    spatial_ast_retrieval,
    graph_query,
)


def should_self_heal(state: BIMGraphState) -> str:
    """
    Conditional edge after evaluate.

    Source hierarchy (best → worst):
      graph  → if evaluation fails, fall back to AST (ultimate ground truth)
      dense  → if evaluation fails, try graph first, then AST
      ast    → already ground truth, do not loop again

    graph_unavailable → treated same as "dense" (Neo4j was down)
    """
    spatial_match = state["evaluator_feedback"].get("spatial_match", False)
    source        = state.get("retrieval_source", "dense")
    loop_count    = state.get("loop_count", 0)

    if spatial_match:
        return END

    # AST is the final fallback — never re-run it
    if source == "ast":
        return END

    # After graph or dense failure, escalate to deterministic AST
    if loop_count < 3:
        return "spatial_ast_retrieval"

    return END


def route_after_generate(state: BIMGraphState) -> str:
    """
    Skip the LLM evaluator for spatially verified sources.

    Both graph and AST results are ground truth by construction:
      - graph: Cypher query against the IFC hierarchy in Neo4j — exact membership.
      - ast:   deterministic IfcOpenShell traversal of the IFC file — same source of truth.

    An LLM judge adds latency (~800ms) and occasional wrong verdicts for data
    that doesn't need judging. should_self_heal already short-circuits on
    source == "ast", so the evaluate call for AST was always ignored anyway.

    Dense results still go through evaluate — semantic search has spatial blindness
    and the evaluator is the mechanism that triggers the self-healing fallback.
    """
    if state.get("retrieval_source") in ("graph", "ast"):
        return END
    return "evaluate"


def route_after_extraction(state: BIMGraphState) -> str:
    """
    Smart router — decides which retrieval strategy to try first.

    Priority:
      1. graph_query  — if spatial constraint exists (graph is exact + fast)
      2. retrieve_hybrid — if no spatial constraint (semantic search makes sense)

    The graph_query node handles its own fallback internally:
    if Neo4j is down it sets retrieval_source="graph_unavailable",
    the evaluator will fail, and should_self_heal escalates to AST.
    """
    has_floor = bool(state.get("spatial_constraints"))

    if has_floor:

        return "graph_query"   # graph-first when we know the floor
    return "retrieve_hybrid"


def route_after_graph_query(state: BIMGraphState) -> str:
    """
    Confidence router — jumps straight to AST proof if Neo4j is empty.
    
    If Neo4j returns 0 results for a known floor, the dense search fallback (hybrid)
    is likely to fail too. Skipping 'generate' and 'evaluate' for the empty set
    saves 2-3 seconds of useless LLM latency.
    """
    source = state.get("retrieval_source")
    count  = state.get("graph_result_count", 0)
    
    if source == "graph_unavailable" or count == 0:
        return "spatial_ast_retrieval"
        
    return "generate"



builder = StateGraph(BIMGraphState)

# Register all nodes
builder.add_node("extract_spatial_constraints", extract_spatial_constraints)
builder.add_node("retrieve_hybrid",             retrieve_hybrid)
builder.add_node("graph_query",                 graph_query)
builder.add_node("generate",                    generate)
builder.add_node("evaluate",                    evaluate)
builder.add_node("spatial_ast_retrieval",       spatial_ast_retrieval)

# Wire the edges
builder.set_entry_point("extract_spatial_constraints")
builder.add_conditional_edges("extract_spatial_constraints", route_after_extraction)
builder.add_edge("retrieve_hybrid",             "generate")

# Graph query now has its own confidence router
builder.add_conditional_edges("graph_query",    route_after_graph_query)

builder.add_conditional_edges("generate",       route_after_generate)   # graph/ast → END, others → evaluate
builder.add_conditional_edges("evaluate",       should_self_heal)
builder.add_edge("spatial_ast_retrieval",       "generate")


graph = builder.compile()
