import json
import logging
from graph import graph

logger = logging.getLogger("bim_graph.audit")

# User just types a natural language query — Node 0 extracts the spatial constraint automatically
QUERY = "List all doors and walls located specifically on level 2"


def main() -> None:
    logger.info("=" * 60)
    logger.info("BIM-Graph Pipeline START")
    logger.info("Query: %s", QUERY)
    logger.info("=" * 60)

    result = graph.invoke({
        "query":               QUERY,
        "spatial_constraints": "",     # Node 0 (extract_spatial_constraints) fills this
        "retrieved_nodes":     [],
        "generation":          "",
        "evaluator_feedback":  {},
        "correction_log":      [],
        "loop_count":          0,
        "retrieval_source":    "",     # set to "dense" or "ast" by retrieval nodes
    })

    logger.info("=" * 60)
    logger.info("BIM-Graph Pipeline COMPLETE")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("  SPATIAL CONSTRAINT DETECTED")
    print("=" * 60)
    print(f"  {result['spatial_constraints']}")

    print("\n" + "=" * 60)
    print("  FINAL ANSWER")
    print("=" * 60)
    print(result["generation"])

    print("\n" + "=" * 60)
    print("  PIPELINE DIAGNOSTICS")
    print("=" * 60)
    print(f"  Retrieval source      : {result.get('retrieval_source', 'unknown')}")
    print(f"  Self-healing loops    : {result.get('loop_count', 0)}")
    print(f"  Spatial match verdict : {result['evaluator_feedback'].get('spatial_match')}")
    print(f"  Evaluator reason      : {result['evaluator_feedback'].get('reason', 'N/A')}")

    print("\n" + "=" * 60)
    print("  CORRECTION LOG  (self-healing audit trail)")
    print("=" * 60)
    if result["correction_log"]:
        print(json.dumps(result["correction_log"], indent=2))
    else:
        print("  (empty — pipeline succeeded on first pass, no self-healing needed)")
    print()


if __name__ == "__main__":
    main()