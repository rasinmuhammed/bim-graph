"""
BIM-Graph FastAPI Server
────────────────────────
Exposes the full LangGraph self-healing pipeline as a REST API with
real-time Server-Sent Events (SSE) streaming so the Next.js UI can
watch each node fire as it happens.

Endpoints:
  GET  /health           → liveness probe
  GET  /floors           → IFC floor list from oracle
  GET  /benchmark        → serve data/benchmark_results.json
  POST /query            → blocking JSON response (CLI-friendly)
  GET  /query/stream     → SSE stream of node-by-node events

Start with:
  cd /Users/muhammedrasin/bim-graph
  uvicorn src.api.main:app --reload --port 8000
"""

import sys
import json
import time
import queue
import pathlib
import logging
import threading
import asyncio
from datetime import datetime
from typing import AsyncGenerator

# ── Path bootstrap (must come before local imports) ────────────────────────────
_API_DIR      = pathlib.Path(__file__).resolve().parent
_SRC_DIR      = _API_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent

from fastapi import FastAPI, Query as QParam, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.graph import graph
from cache.redis_cache import cache_get, cache_set
from benchmark.ifc_oracle import list_all_floors

logger = logging.getLogger("bim_graph.api")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "BIM-Graph API",
    description = "Self-healing agentic RAG pipeline for BIM spatial queries.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Schema ─────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    ifc_filename: str

# ── Helpers ────────────────────────────────────────────────────────────────────
def _sse(event_type: str, data: dict) -> str:
    """Format a single Server-Sent Event string."""
    return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"


def _blank_state(query: str, ifc_filename: str) -> dict:
    return {
        "query":               query,
        "spatial_constraints": "",
        "retrieved_nodes":     [],
        "generation":          "",
        "evaluator_feedback":  {},
        "correction_log":      [],
        "loop_count":          0,
        "retrieval_source":    "",
        "ifc_filename":        ifc_filename,
    }


def _node_to_event(node_name: str, node_output: dict) -> dict:
    """
    Map a LangGraph node output to a structured frontend event.
    Returns {"type": str, "node": str, "data": dict}
    """
    base = {"node": node_name}

    if node_name == "extract_spatial_constraints":
        return {"type": "node_end", **base,
                "data": {"floor": node_output.get("spatial_constraints", ""),
                         "is_inventory_query": node_output.get("is_inventory_query", False)}}

    if node_name == "retrieve_hybrid":
        docs   = node_output.get("retrieved_nodes", [])
        source = node_output.get("retrieval_source", "dense")
        return {"type": "node_end", **base,
                "data": {"chunks": len(docs), "source": source}}

    if node_name == "generate":
        answer = node_output.get("generation", "")
        return {"type": "node_end", **base,
                "data": {"chars": len(answer), "preview": answer[:120]}}

    if node_name == "evaluate":
        fb = node_output.get("evaluator_feedback", {})
        return {"type": "node_end", **base,
                "data": {"spatial_match": fb.get("spatial_match"),
                         "reason": fb.get("reason", "")}}

    if node_name == "spatial_ast_retrieval":
        log   = node_output.get("correction_log", [])
        loops = node_output.get("loop_count", 1)
        reason = log[-1].get("failure_reason", "") if log else ""
        
        if "Early Routing" in reason:
            return {"type": "node_end", **base, "data": {"reason": reason}}
        else:
            return {"type": "self_heal", **base,
                    "data": {"loop": loops,
                             "reason": reason}}

    # Fallback for any future nodes
    return {"type": "node_end", **base, "data": {}}


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/floors", tags=["IFC"])
async def get_floors(f: str = QParam("Duplex_A_20110907.ifc", description="IFC Filename")):
    """Return every IfcBuildingStorey with name, elevation and element count."""
    ifc_path = str(_PROJECT_ROOT / "data" / f)
    if not pathlib.Path(ifc_path).exists():
        raise HTTPException(status_code=404, detail=f"IFC file {f} not found.")
    floors = list_all_floors(ifc_path)
    return {"floors": floors}


@app.get("/benchmark", tags=["Research"])
async def get_benchmark():
    """Serve the pre-computed benchmark_results.json for the research dashboard."""
    path = _PROJECT_ROOT / "data" / "benchmark_results.json"
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail="Benchmark not run yet. Execute src/benchmark/run_benchmark.py first.")
    with open(path) as f:
        return json.load(f)


@app.post("/query", tags=["Pipeline"])
async def query_pipeline(req: QueryRequest):
    """
    Run the full BIM-Graph pipeline synchronously.
    Returns a single JSON response (use /query/stream for real-time updates).
    """
    t0 = time.time()

    cached = cache_get(req.query)
    if cached:
        return {
            "answer":              cached["answer"],
            "spatial_constraints": "",
            "retrieval_source":    "cache",
            "cache_hit":           True,
            "self_healed":         False,
            "correction_log":      cached.get("correction_log", []),
            "latency_ms":          int((time.time() - t0) * 1000),
        }

    state = graph.invoke(_blank_state(req.query, req.ifc_filename))

    cache_set(
        req.query,
        state.get("spatial_constraints", ""),
        state.get("generation", ""),
        state.get("correction_log", []),
    )

    return {
        "answer":              state.get("generation", ""),
        "spatial_constraints": state.get("spatial_constraints", ""),
        "retrieval_source":    state.get("retrieval_source", ""),
        "cache_hit":           False,
        "self_healed":         state.get("loop_count", 0) > 0,
        "correction_log":      state.get("correction_log", []),
        "latency_ms":          int((time.time() - t0) * 1000),
    }


@app.get("/query/stream", tags=["Pipeline"])
async def query_stream(
    q: str = QParam(..., description="Natural language BIM query"),
    f: str = QParam("Duplex_A_20110907.ifc", description="IFC Filename")
):
    """
    Stream the pipeline execution as Server-Sent Events.
    Each event has shape: { "type": str, "node": str, "data": {} }

    Event types:
      cache_hit          → result served from Redis, pipeline skipped
      node_end           → a LangGraph node completed
      self_heal          → spatial_ast_retrieval was triggered
      final              → pipeline done, full result attached
      error              → unhandled exception
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        t0 = time.time()

        # ── Cache check ────────────────────────────────────────────────────
        cached = cache_get(q)   
        if cached:
            yield _sse("cache_hit", {
                "answer":       cached["answer"],
                "latency_ms":   int((time.time() - t0) * 1000),
            })
            yield _sse("final", {
                "answer":           cached["answer"],
                "correction_log":   cached.get("correction_log", []),
                "cache_hit":        True,
                "latency_ms":       int((time.time() - t0) * 1000),
            })
            return

        # ── Run LangGraph in a background thread, push events via queue ────
        event_queue: queue.Queue = queue.Queue()
        final_state: dict = {}

        def _run_graph():
            try:
                state = _blank_state(q, f)
                for node_event in graph.stream(state, stream_mode="updates"):
                    event_queue.put(("node", node_event))
                    # Track the last full output for the final event
                    node_name   = next(iter(node_event))
                    node_output = node_event[node_name]
                    final_state.update(node_output)
            except Exception as exc:
                event_queue.put(("error", str(exc)))
            finally:
                event_queue.put(("done", None))

        thread = threading.Thread(target=_run_graph, daemon=True)
        thread.start()

        # ── Drain queue and yield SSE events ───────────────────────────────
        while True:
            try:
                kind, payload = event_queue.get(timeout=0.05)
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            if kind == "error":
                yield _sse("error", {"message": payload})
                break

            if kind == "done":
                # Write to cache
                cache_set(
                    q,
                    final_state.get("spatial_constraints", ""),
                    final_state.get("generation", ""),
                    final_state.get("correction_log", []),
                )
                yield _sse("final", {
                    "answer":              final_state.get("generation", ""),
                    "spatial_constraints": final_state.get("spatial_constraints", ""),
                    "retrieval_source":    final_state.get("retrieval_source", ""),
                    "self_healed":         final_state.get("loop_count", 0) > 0,
                    "cache_hit":           False,
                    "correction_log":      final_state.get("correction_log", []),
                    "latency_ms":          int((time.time() - t0) * 1000),
                })
                break

            if kind == "node":
                node_name   = next(iter(payload))
                node_output = payload[node_name]
                evt         = _node_to_event(node_name, node_output)
                yield _sse(evt["type"], {"node": evt["node"], **evt["data"]})

            await asyncio.sleep(0)  # yield to FastAPI event loop

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering in production
            "Connection":       "keep-alive",
        },
    )
