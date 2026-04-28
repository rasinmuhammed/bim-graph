"""
Tests for the Neo4j graph layer.

These tests use unittest.mock to patch the Neo4j driver so they run
without a real Neo4j instance — same pattern as test_graph_routing.py
mocking the LLM.

WHAT YOU LEARN HERE:
  - Patching a module's imported name (not the original definition)
  - MagicMock chaining for fluent APIs (driver.session().__enter__.return_value)
  - Testing graph routing logic independently of infrastructure
"""
from unittest.mock import MagicMock, patch


# ── format_results_as_context ───────────────────────────────────────────────────
def test_format_results_as_context_populated():
    """Results should produce a header line + one line per element."""
    from graph_db.queries import format_results_as_context

    records = [
        {"ifc_type": "IfcPump", "name": "Main Pump", "guid": "3AxGUt8yz4AQ4P0r12AB12"},
        {"ifc_type": "IfcFan",  "name": "AHU-01",    "guid": "5BzKLm9wv3BR5Q1s34CD34"},
    ]
    lines = format_results_as_context(records, "Level 2")

    assert lines[0].startswith("--- [SOURCE: NEO4J GRAPH DB")
    assert "Level 2" in lines[0]
    assert any("IfcPump" in l for l in lines)
    assert any("3AxGUt8yz4AQ4P0r12AB12" in l for l in lines)


def test_format_results_as_context_empty():
    """Empty results should still produce a header + a 'not found' message."""
    from graph_db.queries import format_results_as_context

    lines = format_results_as_context([], "Level 3")

    assert len(lines) == 2
    assert "Level 3" in lines[0]
    assert "No elements" in lines[1]


# ── graph_query node ────────────────────────────────────────────────────────────
def test_graph_query_returns_graph_unavailable_when_neo4j_down():
    """
    When Neo4j is unreachable, the node must return retrieval_source='graph_unavailable'
    so should_self_heal can escalate to the AST fallback.
    """
    with patch("graph_db.queries.is_graph_available", return_value=False):
        from agent.nodes import graph_query

        state = {
            "query":               "What HVAC equipment is on Level 2?",
            "spatial_constraints": "Level 2",
            "is_inventory_query":  False,
            "ifc_filename":        "Duplex_A_20110907.ifc",
            "node_timings":        {},
        }
        result = graph_query(state)

        assert result["retrieval_source"] == "graph_unavailable"
        assert result["retrieved_nodes"] == []
        assert result["graph_result_count"] == 0


def test_graph_query_returns_mep_elements_for_equipment_query():
    """
    Equipment queries should trigger get_mep_elements_on_floor,
    not get_all_elements_on_floor.
    """
    mock_records = [
        {"ifc_type": "IfcPump", "name": "Main Pump", "guid": "3AxGUt8yz4AQ4P0r12AB12"},
    ]

    with patch("graph_db.queries.is_graph_available", return_value=True), \
         patch("graph_db.queries.is_file_loaded", return_value=True), \
         patch("graph_db.queries.get_all_storey_names", return_value=["Level 2"]), \
         patch("graph_db.queries.get_mep_elements_on_floor", return_value=mock_records) as mock_mep, \
         patch("graph_db.queries.get_all_elements_on_floor", return_value=[]) as mock_all:

        from agent.nodes import graph_query

        state = {
            "query":               "What HVAC equipment is on Level 2?",
            "spatial_constraints": "Level 2",
            "is_inventory_query":  False,
            "ifc_filename":        "Duplex_A_20110907.ifc",
            "node_timings":        {},
        }
        result = graph_query(state)

        mock_mep.assert_called_once_with("Level 2", "Duplex_A_20110907.ifc")
        mock_all.assert_not_called()
        assert result["retrieval_source"] == "graph"
        assert result["graph_result_count"] == 1


def test_graph_query_returns_all_elements_for_inventory_query():
    """
    Inventory queries (is_inventory_query=True) must return ALL elements,
    not just MEP — the user asked for everything.
    """
    mock_records = [
        {"ifc_type": "IfcWall",  "name": "Wall-01",   "guid": "1AAaaa0000000000000001"},
        {"ifc_type": "IfcPump",  "name": "Main Pump", "guid": "2BBbbb0000000000000002"},
    ]

    with patch("graph_db.queries.is_graph_available", return_value=True), \
         patch("graph_db.queries.is_file_loaded", return_value=True), \
         patch("graph_db.queries.get_all_storey_names", return_value=["Level 2"]), \
         patch("graph_db.queries.get_all_elements_on_floor", return_value=mock_records) as mock_all, \
         patch("graph_db.queries.get_mep_elements_on_floor", return_value=[]) as mock_mep:

        from agent.nodes import graph_query

        state = {
            "query":               "What is present on Level 2?",
            "spatial_constraints": "Level 2",
            "is_inventory_query":  True,
            "ifc_filename":        "Duplex_A_20110907.ifc",
            "node_timings":        {},
        }
        result = graph_query(state)

        mock_all.assert_called_once_with("Level 2", "Duplex_A_20110907.ifc")
        mock_mep.assert_not_called()
        assert result["graph_result_count"] == 2


# ── routing ─────────────────────────────────────────────────────────────────────
def test_route_after_extraction_goes_to_graph_when_floor_known():
    """
    When spatial constraints are present, routing must go to graph_query —
    not retrieve_hybrid. Graph is now the primary path.
    """
    from agent.graph import route_after_extraction

    state = {
        "spatial_constraints": "Level 2",
        "is_inventory_query":  False,
    }
    assert route_after_extraction(state) == "graph_query"


def test_route_after_extraction_goes_to_dense_when_no_floor():
    """
    Without a floor constraint, graph queries make no sense — use dense retrieval.
    """
    from agent.graph import route_after_extraction

    state = {
        "spatial_constraints": "",
        "is_inventory_query":  False,
    }
    assert route_after_extraction(state) == "retrieve_hybrid"


def test_should_self_heal_escalates_to_ast_after_graph_failure():
    """
    If graph_query ran but the evaluator rejected the answer,
    should_self_heal must route to spatial_ast_retrieval as the fallback.
    """
    from agent.graph import should_self_heal

    state = {
        "evaluator_feedback": {"spatial_match": False, "reason": "incomplete answer"},
        "retrieval_source":   "graph",
        "loop_count":         0,
    }
    assert should_self_heal(state) == "spatial_ast_retrieval"


def test_should_self_heal_ends_after_graph_unavailable_and_ast():
    """
    After AST has run, never loop again — even if evaluator still fails.
    AST is deterministic ground truth; re-running it gives identical output.
    """
    from langgraph.graph import END
    from agent.graph import should_self_heal

    state = {
        "evaluator_feedback": {"spatial_match": False, "reason": "no floor proof"},
        "retrieval_source":   "ast",
        "loop_count":         1,
    }
    assert should_self_heal(state) == END
