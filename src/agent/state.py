from typing import TypedDict, Optional, NotRequired

class CorrectionEntry(TypedDict):
    attempt: int
    search_strategy: str
    failure_reason: str
    action_taken: str

class Evidence(TypedDict, total=False):
    source: str
    ifc_file: str
    resolved_floor: str
    matched_storey_guid: str
    element_guids: list[str]
    cypher_summary: str
    confidence_label: str

class BIMGraphState(TypedDict):
    query:               str
    spatial_constraints: str
    is_inventory_query:  bool
    retrieved_nodes:     list[str]
    generation:          str
    evaluator_feedback:  dict
    correction_log:      list[CorrectionEntry]
    loop_count:          int
    retrieval_source:    str          # "dense" | "graph" | "ast" | ""
    ifc_filename:        str
    node_timings:        dict[str, float]
    context_token_count: int
    graph_result_count:  int          # how many records Neo4j returned
    request_id:          str          # UUID per HTTP request, threads through all nodes
    extracted_guids:     list[str]    # GUIDs parsed from generation output, used by oracle scorer
    query_kind:          NotRequired[str]
    floors:              NotRequired[list[str]]
    ifc_types:           NotRequired[list[str]]
    needs_exhaustive_answer: NotRequired[bool]
    resolved_floor:      NotRequired[str]
    evidence:            NotRequired[Evidence]
    confidence_label:    NotRequired[str]
