from typing import TypedDict, Optional

class CorrectionEntry(TypedDict):
    attempt: int
    search_strategy: str
    failure_reason: str
    action_taken: str

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