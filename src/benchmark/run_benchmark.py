"""
run_benchmark.py
────────────────
BIM-Graph Benchmark Runner

Compares the naive RAG baseline against the agentic self-healing pipeline
across the test query set. Outputs:
  - A formatted terminal table
  - data/benchmark_results.csv
  - data/benchmark_results.json

Scoring uses two signals:
1. Evaluator Spatial Match (ESM)  — did the LLM evaluator accept the answer?
2. IFC Ground Truth Hit (GTH)     — did the generation mention ≥1 element
                                    that is actually on the target floor?
"""
import sys
import os
import re
import csv
import json
import time
import pathlib
import logging
from datetime import datetime

# ── Path bootstrap so imports work when run from any CWD ──────────────────────
_BENCHMARK_DIR = pathlib.Path(__file__).resolve().parent
_SRC_DIR       = _BENCHMARK_DIR.parent
_PROJECT_ROOT  = _SRC_DIR.parent
sys.path.insert(0, str(_SRC_DIR / "agent"))   # nodes, graph, state
sys.path.insert(0, str(_BENCHMARK_DIR))        # ifc_oracle, baseline_runner

from ifc_oracle     import get_floor_elements, list_all_floors
from baseline_runner import run_baseline
from dotenv import load_dotenv
load_dotenv()

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("bim_graph.benchmark")

# ── Paths ─────────────────────────────────────────────────────────────────────
_QUERY_SET   = _BENCHMARK_DIR / "query_set.json"
_RESULTS_DIR = _PROJECT_ROOT / "data"
_CSV_OUT     = _RESULTS_DIR / "benchmark_results.csv"
_JSON_OUT    = _RESULTS_DIR / "benchmark_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _evaluate_spatial_match(generation: str, target_floor: str) -> dict:
    """
    Use qwen3-32b to judge if the generation answers the floor-specific query.
    Returns {"spatial_match": bool, "reason": str}
    """
    import os
    from langchain_groq import ChatGroq

    if not target_floor:
        return {"spatial_match": None, "reason": "No single target floor — skipped"}

    prompt = f"""You are a strict BIM spatial auditor.

Answer under review:
\"\"\"{generation}\"\"\"

The user asked specifically for assets on: {target_floor}

- spatial_match = true  → answer confidently lists specific named assets on the correct floor
- spatial_match = false → answer is vague, admits ignorance, or mixes floors

Respond ONLY with valid JSON: {{"spatial_match": false, "reason": "explain"}}"""

    time.sleep(5)
    llm   = ChatGroq(model="qwen/qwen3-32b", api_key=os.getenv("GROQ_API_KEY"))
    resp  = llm.invoke(prompt)
    clean = _strip_thinking(resp.content)
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.IGNORECASE).strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"spatial_match": False, "reason": f"parse error: {clean[:80]}"}


def _ground_truth_hit(generation: str, target_floor: str | None) -> bool:
    """
    Check if the generation mentions ≥1 element GUID or Name that
    actually belongs to the target floor according to the IFC oracle.
    """
    if not target_floor:
        return False
    oracle = get_floor_elements(target_floor)
    for el in oracle["elements"]:
        if el["name"] and el["name"] in generation:
            return True
        if el["guid"] and el["guid"] in generation:
            return True
    return False


def _run_agentic(query: str, target_floor: str) -> dict:
    """
    Run the full LangGraph pipeline for a single query.
    Returns the final state dict.
    """
    # Lazy import so the benchmark can be run standalone
    from graph import graph

    state = graph.invoke({
        "query":               query,
        "spatial_constraints": "",
        "retrieved_nodes":     [],
        "generation":          "",
        "evaluator_feedback":  {},
        "correction_log":      [],
        "loop_count":          0,
        "retrieval_source":    "",
    })
    return state


# ── Main ──────────────────────────────────────────────────────────────────────
def run_benchmark():
    logger.info("=" * 70)
    logger.info("BIM-Graph Benchmark START  —  %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 70)

    with open(_QUERY_SET) as f:
        queries = json.load(f)

    # Print IFC floor summary first
    logger.info("\n=== IFC Ground Truth : Floor Summary ===")
    for floor in list_all_floors():
        logger.info("  %-12s | elev: %7.2fm | elements: %d",
                    floor["name"], floor["elevation_m"], floor["element_count"])

    results = []

    for q in queries:
        qid          = q["id"]
        query        = q["query"]
        target_floor = q.get("target_floor")

        logger.info("\n%s  [%s]  %s", "─" * 60, qid, query[:70])

        # ── Step A: Baseline ──────────────────────────────────────────────────
        logger.info("  Running BASELINE …")
        t0       = time.time()
        baseline = run_baseline(query)
        b_time   = time.time() - t0

        logger.info("  Evaluating BASELINE response …")
        b_eval   = _evaluate_spatial_match(baseline["generation"], target_floor)
        b_gth    = _ground_truth_hit(baseline["generation"], target_floor)

        b_match  = b_eval.get("spatial_match")
        logger.info("  Baseline → ESM=%s | GTH=%s | %.1fs", b_match, b_gth, b_time)

        # ── Step B: Agentic ───────────────────────────────────────────────────
        logger.info("  Running AGENTIC pipeline …")
        t0    = time.time()
        state = _run_agentic(query, target_floor or "")
        a_time = time.time() - t0

        a_gen        = state.get("generation", "")
        a_eval       = state.get("evaluator_feedback", {})
        a_match      = a_eval.get("spatial_match")
        a_loops      = state.get("loop_count", 0)
        a_source     = state.get("retrieval_source", "?")
        a_gth        = _ground_truth_hit(a_gen, target_floor)

        logger.info("  Agentic  → ESM=%s | GTH=%s | loops=%d | %.1fs",
                    a_match, a_gth, a_loops, a_time)

        results.append({
            "id":              qid,
            "query":           query,
            "target_floor":    target_floor or "N/A",
            # Baseline columns
            "b_chunks":        baseline["chunks_retrieved"],
            "b_esm":           b_match,
            "b_gth":           b_gth,
            "b_time_s":        round(b_time, 1),
            # Agentic columns
            "a_esm":           a_match,
            "a_gth":           a_gth,
            "a_self_healed":   a_loops > 0,
            "a_loops":         a_loops,
            "a_source":        a_source,
            "a_time_s":        round(a_time, 1),
            # Texts
            "b_reason":        b_eval.get("reason", ""),
            "a_reason":        a_eval.get("reason", ""),
            "b_generation":    baseline["generation"][:400],
            "a_generation":    a_gen[:400],
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    n            = len(results)
    n_floor_q    = sum(1 for r in results if r["target_floor"] != "N/A")
    b_esm_rate   = sum(1 for r in results if r["b_esm"] is True) / max(n_floor_q, 1)
    a_esm_rate   = sum(1 for r in results if r["a_esm"] is True) / max(n_floor_q, 1)
    b_gth_rate   = sum(1 for r in results if r["b_gth"] is True) / max(n_floor_q, 1)
    a_gth_rate   = sum(1 for r in results if r["a_gth"] is True) / max(n_floor_q, 1)
    heal_rate    = sum(1 for r in results if r["a_self_healed"]) / max(n, 1)

    # Terminal table
    col = lambda s, w: str(s).ljust(w)[:w]
    header = (f"{'ID':<5} {'Floor':<12} {'B_ESM':<6} {'B_GTH':<6} "
              f"{'A_ESM':<6} {'A_GTH':<6} {'Healed':<7} {'Loops':<6} {'Src':<6}")
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(header)
    print("─" * 70)
    for r in results:
        print(f"{col(r['id'],5)} {col(r['target_floor'],12)} "
              f"{col(r['b_esm'],6)} {col(r['b_gth'],6)} "
              f"{col(r['a_esm'],6)} {col(r['a_gth'],6)} "
              f"{col(r['a_self_healed'],7)} {col(r['a_loops'],6)} {col(r['a_source'],6)}")

    print("─" * 70)
    print(f"\n  Queries run          : {n}")
    print(f"  Floor-specific       : {n_floor_q}")
    print(f"  Baseline  ESM rate   : {b_esm_rate:.0%}   ← naive RAG spatial accuracy")
    print(f"  Agentic   ESM rate   : {a_esm_rate:.0%}   ← self-healing pipeline accuracy")
    print(f"  Baseline  GTH rate   : {b_gth_rate:.0%}   ← ground-truth element hits")
    print(f"  Agentic   GTH rate   : {a_gth_rate:.0%}   ← ground-truth element hits")
    print(f"  Self-heal trigger %  : {heal_rate:.0%}   ← how often AST was needed")
    print("=" * 70)

    # ── Save outputs ──────────────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(exist_ok=True)

    # CSV
    with open(_CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # JSON
    with open(_JSON_OUT, "w") as f:
        json.dump({
            "run_at":        datetime.now().isoformat(),
            "ifc_file":      "Duplex_A_20110907.ifc",
            "total_queries": n,
            "metrics": {
                "baseline_esm_rate":  b_esm_rate,
                "agentic_esm_rate":   a_esm_rate,
                "baseline_gth_rate":  b_gth_rate,
                "agentic_gth_rate":   a_gth_rate,
                "self_heal_rate":     heal_rate,
            },
            "results": results,
        }, f, indent=2)

    logger.info("\nResults saved → %s", _CSV_OUT)
    logger.info("Results saved → %s", _JSON_OUT)
    logger.info("Benchmark COMPLETE.")


if __name__ == "__main__":
    run_benchmark()
