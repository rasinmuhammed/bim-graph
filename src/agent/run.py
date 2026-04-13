import sys
import json
import pathlib
import logging

# ── Path bootstrap: must happen BEFORE any local module imports ────────────────
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from graph import graph                            # noqa: E402
from cache.redis_cache import cache_get, cache_set  # noqa: E402

logger = logging.getLogger("bim_graph.audit")

# User just types a natural language query — Node 0 extracts the spatial constraint automatically
QUERY = "List all doors and walls located specifically on level 2"


def main() -> None:
    logger.info("=" * 60)
    logger.info("BIM-Graph Pipeline START")
    logger.info("Query: %s", QUERY)
    logger.info("=" * 60)

    # Cache is keyed on query alone at lookup time (floor not yet extracted)
    cached = cache_get(QUERY, "")
    if cached:
        print("\n⚡ CACHE HIT — skipping pipeline")
        print(cached["answer"])
        return

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

    cache_set(
        QUERY,
        result.get("spatial_constraints", ""),
        result.get("generation", ""),
        result.get("correction_log", []),
    )

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