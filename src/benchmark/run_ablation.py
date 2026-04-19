"""
run_ablation.py
───────────────
Progressive-enablement ablation study for the research paper.

Conditions (each row adds exactly one capability over the previous):
  D          — dense-only: BM25 + ChromaDB RRF, no routing, no evaluator, no AST
  D+E        — dense + LLM evaluator gate (catches spatial failures, cannot fix them)
  G          — graph-only (Cypher over Neo4j), no fallback, no AST
  G+D        — graph primary, dense fallback when graph unavailable
  D+AST      — dense + AST self-heal (no graph) — isolates what AST alone buys
  Full-AST   — full pipeline with AST disabled but evaluator kept (headline ablation)
  Full       — complete pipeline: graph → dense → evaluator → AST self-heal → generate

Reported metrics (all required by the research paper):
  Avg F1, Avg Precision, Avg Recall (macro over queries)
  95% bootstrap CI on F1 (10k resamples)
  Graph Hit % — fraction served by Cypher without fallback
  Self-Heal % — fraction where AST fired
  Evaluator Recall — fraction of spatially-wrong answers the evaluator caught
  Per-category F1 breakdown (architectural, inventory, mep, cross_floor, adversarial)
  Latency by path (median / p95)
  Oracle-set size distribution (min / median / max GUIDs per query)
  Correctly-empty F1=1.0 rate
"""
import json
import random
import re
import time
import pathlib
import sys
import statistics

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, END
from agent.state import BIMGraphState
from agent.nodes import (
    extract_spatial_constraints,
    retrieve_hybrid,
    generate,
    evaluate,
    graph_query,
    spatial_ast_retrieval,
)
from benchmark.ifc_oracle import (
    get_ground_truth_guids,
    get_ground_truth_guids_by_types,
    score_answer,
    score_cross_floor_answer,
)

_ROOT      = pathlib.Path(__file__).resolve().parent.parent.parent
_QUERY_SET = _ROOT / "src" / "benchmark" / "query_set.json"
_OUT_FILE  = _ROOT / "data" / "ablation_results.json"
_IFC_DIR   = _ROOT / "data"


# ── Pipeline builders ──────────────────────────────────────────────────────────

def _build_dense_only():
    """D — Baseline: semantic search only. No graph, no evaluator, no AST."""
    b = StateGraph(BIMGraphState)
    b.add_node("extract_spatial_constraints", extract_spatial_constraints)
    b.add_node("retrieve_hybrid",             retrieve_hybrid)
    b.add_node("generate",                    generate)
    b.set_entry_point("extract_spatial_constraints")
    b.add_edge("extract_spatial_constraints", "retrieve_hybrid")
    b.add_edge("retrieve_hybrid",             "generate")
    b.add_edge("generate",                    END)
    return b.compile()


def _build_dense_plus_evaluator():
    """
    D+E — Dense + LLM evaluator gate.
    The evaluator catches spatial failures but has no repair path — it routes to END
    regardless of verdict. This isolates the evaluator's detection capability alone.
    """
    b = StateGraph(BIMGraphState)
    b.add_node("extract_spatial_constraints", extract_spatial_constraints)
    b.add_node("retrieve_hybrid",             retrieve_hybrid)
    b.add_node("generate",                    generate)
    b.add_node("evaluate",                    evaluate)
    b.set_entry_point("extract_spatial_constraints")
    b.add_edge("extract_spatial_constraints", "retrieve_hybrid")
    b.add_edge("retrieve_hybrid",             "generate")
    b.add_edge("generate",                    "evaluate")
    b.add_edge("evaluate",                    END)   # caught but not repaired
    return b.compile()


def _build_graph_only():
    """G — Graph retrieval only. Mirrors BIMConverse-style setups."""
    def _route(state):
        return "graph_query" if bool(state.get("spatial_constraints")) else "retrieve_hybrid"

    b = StateGraph(BIMGraphState)
    b.add_node("extract_spatial_constraints", extract_spatial_constraints)
    b.add_node("graph_query",                 graph_query)
    b.add_node("retrieve_hybrid",             retrieve_hybrid)
    b.add_node("generate",                    generate)
    b.set_entry_point("extract_spatial_constraints")
    b.add_conditional_edges("extract_spatial_constraints", _route)
    b.add_edge("graph_query",     "generate")
    b.add_edge("retrieve_hybrid", "generate")
    b.add_edge("generate",        END)
    return b.compile()


def _build_graph_plus_dense():
    """G+D — Graph primary, dense fallback when graph unavailable. No self-healing."""
    def _route(state):
        return "graph_query" if bool(state.get("spatial_constraints")) else "retrieve_hybrid"

    def _after_gen(_state):
        return END  # no evaluator, no AST

    b = StateGraph(BIMGraphState)
    b.add_node("extract_spatial_constraints", extract_spatial_constraints)
    b.add_node("graph_query",                 graph_query)
    b.add_node("retrieve_hybrid",             retrieve_hybrid)
    b.add_node("generate",                    generate)
    b.set_entry_point("extract_spatial_constraints")
    b.add_conditional_edges("extract_spatial_constraints", _route)
    b.add_edge("graph_query",     "generate")
    b.add_edge("retrieve_hybrid", "generate")
    b.add_conditional_edges("generate", _after_gen)
    return b.compile()


def _build_dense_plus_ast():
    """D+AST — Dense + AST self-heal, no graph. Isolates AST contribution alone."""
    def _after_gen(state):
        if state.get("retrieval_source") == "ast":
            return END
        return "evaluate"

    def _self_heal(state):
        feedback     = state.get("evaluator_feedback", {})
        spatial_match = feedback.get("spatial_match", False)
        source       = state.get("retrieval_source", "dense")
        loop_count   = state.get("loop_count", 0)
        if spatial_match or source == "ast":
            return END
        if loop_count < 3:
            return "spatial_ast_retrieval"
        return END

    b = StateGraph(BIMGraphState)
    b.add_node("extract_spatial_constraints", extract_spatial_constraints)
    b.add_node("retrieve_hybrid",             retrieve_hybrid)
    b.add_node("generate",                    generate)
    b.add_node("evaluate",                    evaluate)
    b.add_node("spatial_ast_retrieval",       spatial_ast_retrieval)
    b.set_entry_point("extract_spatial_constraints")
    b.add_edge("extract_spatial_constraints", "retrieve_hybrid")
    b.add_edge("retrieve_hybrid",             "generate")
    b.add_conditional_edges("generate",              _after_gen)
    b.add_conditional_edges("evaluate",              _self_heal)
    b.add_edge("spatial_ast_retrieval",       "generate")
    return b.compile()


def _build_full_minus_ast():
    """
    Full−AST — Full pipeline with AST disabled but evaluator kept.
    The headline ablation: shows how much self-healing contributes.
    """
    def _route(state):
        return "graph_query" if bool(state.get("spatial_constraints")) else "retrieve_hybrid"

    def _after_gen(state):
        if state.get("retrieval_source") in ("graph", "ast"):
            return END
        return "evaluate"

    b = StateGraph(BIMGraphState)
    b.add_node("extract_spatial_constraints", extract_spatial_constraints)
    b.add_node("graph_query",                 graph_query)
    b.add_node("retrieve_hybrid",             retrieve_hybrid)
    b.add_node("generate",                    generate)
    b.add_node("evaluate",                    evaluate)
    b.set_entry_point("extract_spatial_constraints")
    b.add_conditional_edges("extract_spatial_constraints", _route)
    b.add_edge("graph_query",     "generate")
    b.add_edge("retrieve_hybrid", "generate")
    b.add_conditional_edges("generate", _after_gen)
    b.add_edge("evaluate",        END)   # evaluator fires but AST is disabled
    return b.compile()


def _build_full_pipeline():
    """Full — Complete production pipeline."""
    from agent.graph import graph
    return graph


# ── Rate-limit safe invoke ─────────────────────────────────────────────────────

def _invoke_with_retry(pipeline, item: dict, max_attempts: int = 6) -> dict:
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return pipeline.invoke(_blank_state(item))
        except Exception as exc:
            msg = str(exc)
            if "rate_limit_exceeded" not in msg and "429" not in msg and "rate limit" not in msg.lower():
                raise
            match = re.search(r"try again in ([0-9.]+)(ms|s)", msg, re.IGNORECASE)
            if match:
                val, unit = float(match.group(1)), match.group(2).lower()
                delay = max((val / 1000 if unit == "ms" else val) + 0.5, 1.0)
            print(f"         [rate limit] attempt {attempt}/{max_attempts} — sleeping {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError(f"All {max_attempts} attempts rate-limited for: {item['query'][:60]}")


# ── State and scoring ──────────────────────────────────────────────────────────

def _blank_state(item: dict) -> dict:
    return {
        "query":               item["query"],
        "spatial_constraints": "",
        "is_inventory_query":  False,
        "retrieved_nodes":     [],
        "generation":          "",
        "evaluator_feedback":  {},
        "correction_log":      [],
        "loop_count":          0,
        "retrieval_source":    "",
        "ifc_filename":        item["ifc_filename"],
        "node_timings":        {},
        "context_token_count": 0,
        "graph_result_count":  0,
        "request_id":          "",
        "extracted_guids":     [],
    }


def _score(state: dict, item: dict) -> dict:
    ifc_path     = str(_IFC_DIR / item["ifc_filename"])
    generation   = state.get("generation", "")
    target_floor = item.get("target_floor")
    oracle_types = item.get("oracle_ifc_types")

    if target_floor is None:
        return score_cross_floor_answer(generation, ifc_path)

    if oracle_types:
        gt = get_ground_truth_guids_by_types(ifc_path, target_floor, oracle_types)
    else:
        gt = get_ground_truth_guids(ifc_path, target_floor)

    scores = score_answer(generation, gt)
    scores["oracle_size"] = len(gt)
    return scores


# ── Run one condition ──────────────────────────────────────────────────────────

def run_condition(name: str, pipeline, queries: list[dict]) -> list[dict]:
    results = []
    print(f"\n{'━' * 72}")
    print(f"  Condition: {name}")
    print(f"{'━' * 72}")

    for i, item in enumerate(queries):
        if i > 0:
            time.sleep(3)
        print(f"  [{i+1:02d}/{len(queries)}] {item['query'][:56]}...")
        try:
            t0    = time.perf_counter()
            state = _invoke_with_retry(pipeline, item)
            ms    = int((time.perf_counter() - t0) * 1000)
            scores = _score(state, item)
            row = {
                "query_id":            item["id"],
                "category":            item.get("category", ""),
                "query":               item["query"],
                "retrieval_source":    state.get("retrieval_source", ""),
                "self_healed":         state.get("loop_count", 0) > 0,
                "evaluator_verdict":   state.get("evaluator_feedback", {}).get("spatial_match"),
                "latency_ms":          ms,
                "context_token_count": state.get("context_token_count", 0),
                **scores,
            }
            results.append(row)
            print(f"         source={row['retrieval_source']:<10} f1={row['f1']:.3f}  oracle={row.get('oracle_size','?')}  {ms}ms")
        except Exception as e:
            print(f"         ERROR: {e}")
            results.append({
                "query_id": item["id"], "category": item.get("category", ""),
                "query": item["query"], "error": str(e),
                "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "latency_ms": 0, "retrieval_source": "error",
                "self_healed": False, "evaluator_verdict": None, "oracle_size": 0,
            })
    return results


# ── Statistics ────────────────────────────────────────────────────────────────

def _bootstrap_ci(values: list[float], n: int = 10_000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap 95% confidence interval on the mean."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v
    rng  = random.Random(42)
    means = [
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(n)
    ]
    means.sort()
    lo = int((1 - ci) / 2 * n)
    hi = int((1 + ci) / 2 * n)
    return means[lo], means[hi]


def _pct(vals: list[float]) -> tuple[float, float]:
    """Median and 95th percentile."""
    if not vals:
        return 0.0, 0.0
    s = sorted(vals)
    p95_idx = int(0.95 * len(s))
    return statistics.median(s), s[min(p95_idx, len(s) - 1)]


def _summarise(results: list[dict]) -> dict:
    valid = [r for r in results if "f1" in r and "error" not in r]
    if not valid:
        return {"avg_f1": 0.0, "scored": 0}

    f1s        = [r["f1"] for r in valid]
    ci_lo, ci_hi = _bootstrap_ci(f1s)

    cats: dict[str, list] = {}
    for r in valid:
        cats.setdefault(r.get("category", "?"), []).append(r["f1"])

    # Evaluator recall — fraction of low-F1 answers where evaluator said False
    low_f1 = [r for r in valid if r["f1"] < 0.5 and r.get("evaluator_verdict") is not None]
    eval_recall = (
        sum(1 for r in low_f1 if r["evaluator_verdict"] is False) / len(low_f1)
        if low_f1 else None
    )

    # Latency by retrieval path
    lat_by_source: dict[str, list] = {}
    for r in valid:
        lat_by_source.setdefault(r.get("retrieval_source", "?"), []).append(r["latency_ms"])
    lat_stats = {
        src: {"median_ms": _pct(lats)[0], "p95_ms": _pct(lats)[1]}
        for src, lats in lat_by_source.items()
    }

    # Oracle-set sizes
    oracle_sizes = [r["oracle_size"] for r in valid if r.get("oracle_size") is not None]

    # Correctly-empty rate
    correctly_empty = [r for r in valid if r.get("scoring_method") == "correctly_empty"]

    return {
        "avg_f1":           round(statistics.mean(f1s), 3),
        "f1_ci_lo":         round(ci_lo, 3),
        "f1_ci_hi":         round(ci_hi, 3),
        "avg_precision":    round(statistics.mean(r["precision"] for r in valid), 3),
        "avg_recall":       round(statistics.mean(r["recall"] for r in valid), 3),
        "avg_latency_ms":   round(statistics.mean(r["latency_ms"] for r in valid)),
        "graph_hit_rate":   round(sum(1 for r in valid if r.get("retrieval_source") == "graph") / len(valid), 3),
        "self_heal_rate":   round(sum(1 for r in valid if r.get("self_healed")) / len(valid), 3),
        "evaluator_recall": round(eval_recall, 3) if eval_recall is not None else None,
        "correctly_empty_count": len(correctly_empty),
        "correctly_empty_f1_rate": round(
            sum(1 for r in correctly_empty if r["f1"] == 1.0) / len(correctly_empty), 3
        ) if correctly_empty else None,
        "oracle_size_min":    min(oracle_sizes) if oracle_sizes else None,
        "oracle_size_median": round(statistics.median(oracle_sizes), 1) if oracle_sizes else None,
        "oracle_size_max":    max(oracle_sizes) if oracle_sizes else None,
        "latency_by_path":    lat_stats,
        "by_category":        {c: round(statistics.mean(f), 3) for c, f in sorted(cats.items())},
        "scored":             len(valid),
        "errors":             len(results) - len(valid),
    }


# ── Print tables ──────────────────────────────────────────────────────────────

def _print_tables(summaries: dict[str, dict]) -> None:
    cats = ["architectural", "inventory", "mep", "cross_floor", "adversarial"]
    W    = 14

    print(f"\n{'═' * 110}")
    print("  TABLE 1 — Overall Performance and Routing")
    print(f"{'═' * 110}")
    hdr = f"{'Condition':<{W}} {'F1':>6} {'95% CI':>14} {'Prec':>6} {'Rec':>6} {'Graph%':>7} {'Heal%':>6} {'EvalRec':>8} {'ms':>6}"
    print(hdr)
    print("─" * 110)
    for cond, s in summaries.items():
        ci   = f"[{s.get('f1_ci_lo','?'):.3f},{s.get('f1_ci_hi','?'):.3f}]"
        er   = f"{s['evaluator_recall']:.2f}" if s.get("evaluator_recall") is not None else "  —  "
        row  = (f"{cond:<{W}} {s['avg_f1']:>6.3f} {ci:>14} "
                f"{s['avg_precision']:>6.3f} {s['avg_recall']:>6.3f} "
                f"{s['graph_hit_rate']:>7.1%} {s['self_heal_rate']:>6.1%} "
                f"{er:>8} {s['avg_latency_ms']:>6}")
        print(row)

    print(f"\n{'═' * 110}")
    print("  TABLE 2 — Per-Category F1 Breakdown")
    print(f"{'═' * 110}")
    hdr2 = f"{'Condition':<{W}}" + "".join(f"  {c[:10]:>10}" for c in cats)
    print(hdr2)
    print("─" * 110)
    for cond, s in summaries.items():
        row2 = f"{cond:<{W}}" + "".join(
            f"  {s['by_category'].get(c, 0.0):>10.3f}" for c in cats
        )
        print(row2)

    print(f"\n{'═' * 110}")
    print("  TABLE 3 — Oracle and Correctly-Empty Statistics")
    print(f"{'═' * 110}")
    print(f"{'Condition':<{W}} {'N':>4} {'Errors':>6} {'CE Count':>9} {'CE F1=1 %':>10} {'Oracle min/med/max'}")
    print("─" * 110)
    for cond, s in summaries.items():
        ce_rate = f"{s['correctly_empty_f1_rate']:.0%}" if s.get("correctly_empty_f1_rate") is not None else "—"
        oracle  = (f"{s.get('oracle_size_min','?')} / {s.get('oracle_size_median','?')} / {s.get('oracle_size_max','?')}"
                   if s.get("oracle_size_min") is not None else "—")
        print(f"{cond:<{W}} {s['scored']:>4} {s.get('errors',0):>6} "
              f"{s.get('correctly_empty_count',0):>9} {ce_rate:>10}   {oracle}")

    print(f"\n{'═' * 110}")
    print("  TABLE 4 — Latency by Retrieval Path (median / p95 ms)")
    print(f"{'═' * 110}")
    for cond, s in summaries.items():
        paths = s.get("latency_by_path", {})
        parts = ", ".join(f"{p}: {v['median_ms']:.0f}/{v['p95_ms']:.0f}" for p, v in paths.items())
        print(f"  {cond:<{W}}: {parts}")
    print(f"{'═' * 110}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_ablation():
    queries = json.loads(_QUERY_SET.read_text())

    conditions = {
        "D (dense)":       _build_dense_only(),
        "D+E":             _build_dense_plus_evaluator(),
        "G (graph)":       _build_graph_only(),
        "G+D":             _build_graph_plus_dense(),
        "D+AST":           _build_dense_plus_ast(),
        "Full-AST":        _build_full_minus_ast(),
        "Full":            _build_full_pipeline(),
    }

    all_results: dict[str, list] = {}
    summaries:   dict[str, dict] = {}

    for name, pipeline in conditions.items():
        results          = run_condition(name, pipeline, queries)
        all_results[name] = results
        summaries[name]   = _summarise(results)

    _print_tables(summaries)

    _OUT_FILE.write_text(json.dumps({"conditions": summaries, "raw": all_results}, indent=2))
    print(f"\nResults → {_OUT_FILE}")
    return {"conditions": summaries, "raw": all_results}


if __name__ == "__main__":
    run_ablation()
