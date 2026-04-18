from unittest.mock import patch, MagicMock
from langgraph.graph import END

def test_should_self_heal_triggers_on_mismatch():
    """When dense retrieval fails spatial chekc, graph must route to AST."""
    from agent.graph import should_self_heal

    state = {
        "evaluator_feedback": {"spatial_match": False, "reason": "Floor not confirmed"},
        "retrieval_source": "dense",
        "loop_count": 0,
    }
    assert should_self_heal(state) == "spatial_ast_retrieval"

def test_should_self_heal_stops_after_ast():
    """Once AST has run, never loop again — AST is deterministic ground truth."""
    from agent.graph import should_self_heal

    state = {
        "evaluator_feedback": {"spatial_match": False, "reason": "something"},
        "retrieval_source":   "ast",
        "loop_count":         1,
    }
    assert should_self_heal(state) == END


def test_route_after_extraction_inventory_query():
    """Inventory queries must skip dense retrieval and go straight to AST."""
    from agent.graph import route_after_extraction

    state = {
        "is_inventory_query":  True,
        "spatial_constraints": "Level 2",
    }
    assert route_after_extraction(state) == "spatial_ast_retrieval"


def test_route_after_extraction_normal_query():
    """Non-inventory queries must go through hybrid retrieval first."""
    from agent.graph import route_after_extraction

    state = {
        "is_inventory_query":  False,
        "spatial_constraints": "Level 2",
    }
    assert route_after_extraction(state) == "retrieve_hybrid"