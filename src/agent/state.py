from typing import TypedDict, Optional

class CorrectionEntry(TypedDict):
    attempt: int
    search_strategy: str
    failure_reason: str
    action_taken: str

class BIMGraphState(TypedDict):
    query: str
    spatial_constraints: str
    retrieved_nodes: list[str]
    generation: str
    evaluator_feedback: dict
    correction_log: list[CorrectionEntry]
    loop_count: int
    retrieval_source: str   # "dense" | "ast" | ""
    