from langgraph.graph import StateGraph, END
from agent.state import BIMGraphState
from agent.nodes import (
    extract_spatial_constraints,
    retrieve_hybrid,
    generate,
    evaluate,
    spatial_ast_retrieval,
)


def should_self_heal(state: BIMGraphState) -> str:
    """
    Conditional edge: trigger self-healing ONLY when ALL three conditions are true:
      1. Evaluator flagged a spatial mismatch.
      2. We have NOT already used the deterministic AST — once AST runs, it is
         ground truth and re-running it produces identical output, so looping is pointless.
      3. Loop guard not exceeded (safety net).
    """
    spatial_match    = state["evaluator_feedback"].get("spatial_match", False)
    retrieval_source = state.get("retrieval_source", "dense")
    loop_count       = state.get("loop_count", 0)

    if not spatial_match and retrieval_source != "ast" and loop_count < 3:
        return "spatial_ast_retrieval"
    return END


def route_after_extraction(state: BIMGraphState) -> str:
    """
    Early Router: If the query demands an exhaustive inventory, bypass dense retrieval
    entirely and route straight to deterministic AST. Saves tokens and latency.
    """
    if state.get("is_inventory_query") and state.get("spatial_constraints"):
        return "spatial_ast_retrieval"
    return "retrieve_hybrid"


builder = StateGraph(BIMGraphState)

# Register all nodes
builder.add_node("extract_spatial_constraints", extract_spatial_constraints)
builder.add_node("retrieve_hybrid",             retrieve_hybrid)
builder.add_node("generate",                    generate)
builder.add_node("evaluate",                    evaluate)
builder.add_node("spatial_ast_retrieval",       spatial_ast_retrieval)

# Wire the edges
builder.set_entry_point("extract_spatial_constraints")
builder.add_conditional_edges("extract_spatial_constraints", route_after_extraction)
builder.add_edge("retrieve_hybrid",             "generate")
builder.add_edge("generate",                    "evaluate")
builder.add_conditional_edges("evaluate",       should_self_heal)
builder.add_edge("spatial_ast_retrieval",       "generate")

graph = builder.compile()
