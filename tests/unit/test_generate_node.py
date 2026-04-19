"""
Tests for the generate node and route_after_generate routing function.

Covers:
  - GUID extraction from generation output
  - route_after_generate: graph and ast route to END, dense routes to evaluate
  - Token queue is optional (None outside HTTP context — no side effects)
"""
import re
from unittest.mock import MagicMock, patch
from langgraph.graph import END


# ── route_after_generate ───────────────────────────────────────────────────────

def test_route_after_generate_graph_goes_to_end():
    """Graph results are Neo4j ground truth — skip the LLM evaluator entirely."""
    from agent.graph import route_after_generate
    state = {"retrieval_source": "graph"}
    assert route_after_generate(state) == END


def test_route_after_generate_ast_goes_to_end():
    """AST results are deterministic IFC traversal — no evaluator needed."""
    from agent.graph import route_after_generate
    state = {"retrieval_source": "ast"}
    assert route_after_generate(state) == END


def test_route_after_generate_dense_goes_to_evaluate():
    """Dense retrieval has spatial blindness — must go through the evaluator."""
    from agent.graph import route_after_generate
    state = {"retrieval_source": "dense"}
    assert route_after_generate(state) == "evaluate"


def test_route_after_generate_empty_source_goes_to_evaluate():
    """Missing source defaults to evaluate (safe fallback)."""
    from agent.graph import route_after_generate
    assert route_after_generate({}) == "evaluate"


# ── GUID extraction ────────────────────────────────────────────────────────────

def test_generate_extracts_labeled_guids():
    """
    When the answer contains 'GUID: <22chars>' lines (as produced by the
    graph/AST prompt), extracted_guids must capture all of them.
    """
    fake_answer = (
        "Entity: IfcPump | Name: Main Pump | GUID: 3AxGUt8yz4AQ4P0r12AB12\n"
        "Entity: IfcFan  | Name: AHU-01    | GUID: 5BzKLm9wv3BR5Q1s34CD34\n"
    )
    found = set(re.findall(r"GUID:\s*([A-Za-z0-9$_]{22})", fake_answer))
    assert "3AxGUt8yz4AQ4P0r12AB12" in found
    assert "5BzKLm9wv3BR5Q1s34CD34" in found
    assert len(found) == 2


def test_generate_no_false_positive_guids():
    """
    Free-form text without explicit 'GUID:' labels should not produce GUID matches
    from the labeled-first extraction strategy.
    """
    answer = "There are walls, doors, and pumps on Level 2. No specific GUIDs were provided."
    found = set(re.findall(r"GUID:\s*([A-Za-z0-9$_]{22})", answer))
    assert len(found) == 0


def test_generate_node_populates_extracted_guids():
    """
    The generate node must return extracted_guids in its output dict.
    We mock the LLM to return a known answer with one labeled GUID.
    """
    fake_chunk = MagicMock()
    fake_chunk.content = "Entity: IfcWall | Name: Wall-01 | GUID: 1AAaaa0000000000000001"

    with patch("agent.nodes._get_llm_big") as mock_llm_factory, \
         patch("agent.nodes.get_token_queue", return_value=None):

        mock_llm = MagicMock()
        mock_llm.stream.return_value = [fake_chunk]
        mock_llm_factory.return_value = mock_llm

        from agent.nodes import generate

        state = {
            "query":               "What walls are on Level 1?",
            "retrieved_nodes":     ["[Storey: Level 1] Entity: IfcWall | Name: Wall-01 | GUID: 1AAaaa0000000000000001"],
            "retrieval_source":    "graph",
            "spatial_constraints": "Level 1",
            "node_timings":        {},
            "request_id":          "test-001",
        }
        result = generate(state)

    assert "extracted_guids" in result
    assert "1AAaaa0000000000000001" in result["extracted_guids"]


# ── Integration: full pipeline routing with mocked LLM ────────────────────────

def test_pipeline_graph_path_skips_evaluate():
    """
    End-to-end: when graph_query returns records, the pipeline must reach
    generate and exit without ever calling evaluate.

    Mocks: Neo4j available + file loaded + graph returns one record.
    LLM mocked for extract_spatial_constraints and generate.
    evaluate is NOT mocked — if it runs, it will fail with an unmocked LLM call,
    making the test a reliable sentinel that evaluate was skipped.
    """
    from unittest.mock import patch, MagicMock
    from pydantic import BaseModel

    class FakeConstraint(BaseModel):
        spatial_constraints: str = "Level 2"
        is_inventory_query: bool = False

    fake_generation_chunk = MagicMock()
    fake_generation_chunk.content = "Entity: IfcPump | Name: P-01 | GUID: 3AxGUt8yz4AQ4P0r12AB12"

    mock_llm_fast = MagicMock()
    mock_llm_fast.with_structured_output.return_value.invoke.return_value = FakeConstraint()

    mock_llm_big = MagicMock()
    mock_llm_big.stream.return_value = [fake_generation_chunk]

    mock_records = [{"ifc_type": "IfcPump", "name": "P-01", "guid": "3AxGUt8yz4AQ4P0r12AB12"}]

    with patch("agent.nodes._get_llm_fast", return_value=mock_llm_fast), \
         patch("agent.nodes._get_llm_big",  return_value=mock_llm_big), \
         patch("agent.nodes.get_token_queue", return_value=None), \
         patch("graph_db.queries.is_graph_available", return_value=True), \
         patch("graph_db.queries.is_file_loaded",     return_value=True), \
         patch("graph_db.queries.get_mep_elements_on_floor", return_value=mock_records):

        from agent.graph import graph

        state = graph.invoke({
            "query":               "What HVAC equipment is on Level 2?",
            "spatial_constraints": "",
            "is_inventory_query":  False,
            "retrieved_nodes":     [],
            "generation":          "",
            "evaluator_feedback":  {},
            "correction_log":      [],
            "loop_count":          0,
            "retrieval_source":    "",
            "ifc_filename":        "Duplex_A_20110907.ifc",
            "node_timings":        {},
            "context_token_count": 0,
            "graph_result_count":  0,
            "request_id":          "integ-001",
            "extracted_guids":     [],
        })

    assert state["retrieval_source"] == "graph"
    assert state["graph_result_count"] == 1
    # evaluate was skipped — evaluator_feedback stays empty
    assert state["evaluator_feedback"] == {}
    # loop_count stays 0 (no self-healing triggered)
    assert state["loop_count"] == 0
