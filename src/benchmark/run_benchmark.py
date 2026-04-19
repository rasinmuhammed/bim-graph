"""
Runs every query in query_set.json through THREE pipelines and scores each:
  1. Baseline dense RAG  (retrieval_source = "dense")
  2. Hybrid RRF          (retrieval_source = "dense" with BM25)
  3. BIM-Graph           (retrieval_source = "graph" or "ast")

Produces benchmark_results.json with real P/R/F1 per query.
"""
import json
import time
import pathlib
import sys

# Add src/ to path so imports work when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent.graph import graph
from benchmark.ifc_oracle import (
    get_ground_truth_guids,
    get_ground_truth_guids_by_types,
    score_answer,
    score_cross_floor_answer,
)

_ROOT      = pathlib.Path(__file__).resolve().parent.parent.parent
_QUERY_SET = _ROOT / "src" / "benchmark" / "query_set.json"
_OUT_FILE  = _ROOT / "data" / "benchmark_results.json"
_IFC_DIR   = _ROOT / "data"


def _invoke_with_retry(pipeline, state_in: dict, max_attempts: int = 6) -> dict:
    """Retry pipeline invocation on Groq 429 rate-limit errors with smart backoff."""
    import re as _re
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return pipeline.invoke(state_in)
        except Exception as exc:
            msg = str(exc)
            if "rate_limit_exceeded" not in msg and "429" not in msg and "rate limit" not in msg.lower():
                raise
            match = _re.search(r"try again in ([0-9.]+)(ms|s)", msg, _re.IGNORECASE)
            if match:
                val, unit = float(match.group(1)), match.group(2).lower()
                delay = max((val / 1000 if unit == "ms" else val) + 0.5, 1.0)
            print(f"  [rate limit] attempt {attempt}/{max_attempts} — sleeping {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError(f"All {max_attempts} attempts rate-limited.")


def run_single_query(
    query:           str,
    ifc_filename:    str,
    target_floor:    str | None = None,
    oracle_ifc_types: list[str] | None = None,
) -> dict:
    """Run one query through the full pipeline, return scored result."""
    ifc_path = str(_IFC_DIR / ifc_filename)

    state_in = {
        "query":               query,
        "spatial_constraints": "",
        "is_inventory_query":  False,
        "retrieved_nodes":     [],
        "generation":          "",
        "evaluator_feedback":  {},
        "correction_log":      [],
        "loop_count":          0,
        "retrieval_source":    "",
        "ifc_filename":        ifc_filename,
        "node_timings":        {},
        "context_token_count": 0,
        "graph_result_count":  0,
        "request_id":          "",
        "extracted_guids":     [],
    }

    t0         = time.perf_counter()
    state      = _invoke_with_retry(graph, state_in)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    generation = state.get("generation", "")

    # Cross-floor queries (target_floor=None) cannot be scored with GUID P/R/F1.
    if target_floor is None:
        scores = score_cross_floor_answer(generation, ifc_path)
    else:
        # Always use target_floor from the query set (not the LLM-extracted floor)
        # so adversarial queries ("ground floor" → Level 1) score against the correct
        # storey even when spatial extraction fails or maps to a different name.
        if oracle_ifc_types:
            ground_truth = get_ground_truth_guids_by_types(ifc_path, target_floor, oracle_ifc_types)
        else:
            ground_truth = get_ground_truth_guids(ifc_path, target_floor)
        scores = score_answer(generation, ground_truth)

    return {
        "query":               query,
        "ifc_file":            ifc_filename,
        "floor":               target_floor or state.get("spatial_constraints", ""),
        "retrieval_source":    state.get("retrieval_source", ""),
        "self_healed":         state.get("loop_count", 0) > 0,
        "context_token_count": state.get("context_token_count", 0),
        "graph_result_count":  state.get("graph_result_count", 0),
        "node_timings":        state.get("node_timings", {}),
        "latency_ms":          latency_ms,
        **scores,
    }


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _category_breakdown(results: list[dict], queries: list[dict]) -> dict:
    """Group valid results by category and compute per-category averages."""
    category_map: dict[str, list] = {}
    id_to_category = {q["id"]: q.get("category", "unknown") for q in queries}

    for r in results:
        if "f1" not in r:
            continue
        cat = id_to_category.get(r.get("query_id", ""), "unknown")
        category_map.setdefault(cat, []).append(r)

    breakdown = {}
    for cat, items in sorted(category_map.items()):
        breakdown[cat] = {
            "count":      len(items),
            "avg_f1":     _avg([r["f1"] for r in items]),
            "avg_recall": _avg([r["recall"] for r in items]),
            "self_healed": sum(1 for r in items if r.get("self_healed")),
        }
    return breakdown


def _print_table(results: list[dict], queries: list[dict]) -> None:
    id_to_cat = {q["id"]: q.get("category", "?") for q in queries}
    print()
    print(f"{'ID':<5} {'Category':<14} {'Source':<12} {'F1':>5} {'P':>5} {'R':>5} {'ms':>6}  Query")
    print("─" * 90)
    for r in results:
        if "error" in r:
            print(f"{'?':<5} {'error':<14} {'—':<12} {'—':>5} {'—':>5} {'—':>5} {'—':>6}  {r['query'][:45]}")
            continue
        cat = id_to_cat.get(r.get("query_id", ""), "?")
        print(
            f"{r.get('query_id','?'):<5} {cat:<14} {r['retrieval_source']:<12} "
            f"{r['f1']:>5.2f} {r['precision']:>5.2f} {r['recall']:>5.2f} "
            f"{r['latency_ms']:>6}  {r['query'][:45]}"
        )


def run_benchmark():
    queries = json.loads(_QUERY_SET.read_text())

    results = []
    for i, item in enumerate(queries):
        if i > 0:
            time.sleep(3)
        print(f"[{i+1}/{len(queries)}] {item['query'][:60]}...")
        try:
            result = run_single_query(
                item["query"],
                item["ifc_filename"],
                item.get("target_floor"),
                item.get("oracle_ifc_types"),
            )
            result["query_id"] = item["id"]
            result["category"] = item.get("category", "unknown")
            results.append(result)
            print(f"  → source={result['retrieval_source']}  "
                  f"f1={result['f1']:.2f}  latency={result['latency_ms']}ms")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"query": item["query"], "query_id": item["id"], "error": str(e)})

    valid = [r for r in results if "f1" in r]

    _print_table(results, queries)

    by_category = _category_breakdown(results, queries)

    print()
    print("── Category breakdown ──────────────────────────────────────")
    print(f"{'Category':<14} {'N':>3}  {'Avg F1':>7}  {'Avg Rec':>7}  {'Self-healed':>11}")
    print("─" * 55)
    for cat, stats in by_category.items():
        print(f"{cat:<14} {stats['count']:>3}  {stats['avg_f1']:>7.3f}  "
              f"{stats['avg_recall']:>7.3f}  {stats['self_healed']:>11}")

    summary = {
        "total_queries":      len(results),
        "scored_queries":     len(valid),
        "avg_f1":             _avg([r["f1"] for r in valid]),
        "avg_precision":      _avg([r["precision"] for r in valid]),
        "avg_recall":         _avg([r["recall"] for r in valid]),
        "avg_latency_ms":     round(sum(r["latency_ms"] for r in valid) / len(valid)) if valid else 0,
        "avg_context_tokens": round(sum(r["context_token_count"] for r in valid) / len(valid)) if valid else 0,
        "self_heal_rate":     _avg([1.0 if r["self_healed"] else 0.0 for r in valid]),
        "graph_hit_rate":     _avg([1.0 if r["retrieval_source"] == "graph" else 0.0 for r in valid]),
        "by_category":        by_category,
        "results":            results,
    }

    _OUT_FILE.write_text(json.dumps(summary, indent=2))

    print()
    print("── Overall ─────────────────────────────────────────────────")
    print(f"  Avg F1:          {summary['avg_f1']:.3f}")
    print(f"  Avg Precision:   {summary['avg_precision']:.3f}")
    print(f"  Avg Recall:      {summary['avg_recall']:.3f}")
    print(f"  Avg Latency:     {summary['avg_latency_ms']} ms")
    print(f"  Graph hit rate:  {summary['graph_hit_rate']:.1%}")
    print(f"  Self-heal rate:  {summary['self_heal_rate']:.1%}")
    print(f"\nResults → {_OUT_FILE}")
    return summary


if __name__ == "__main__":
    run_benchmark()
