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

import json
import time
import uuid
import queue
import pathlib
import logging
import threading
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

# ── Path bootstrap (must come before local imports) ────────────────────────────
_API_DIR      = pathlib.Path(__file__).resolve().parent
_SRC_DIR      = _API_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent

from fastapi import FastAPI, Query as QParam, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from agent.graph import graph
from agent.token_stream import set_token_queue
from cache.redis_cache import cache_get, cache_set
from benchmark.ifc_oracle import list_all_floors
from config import settings
from observability.logging import setup_logging, set_request_id

logger = logging.getLogger("bim_graph.api")


# ── App lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging(settings.logs_dir)
    logger.info("BIM-Graph API starting up")
    yield
    logger.info("BIM-Graph API shutting down")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "BIM-Graph API",
    description = "Self-healing agentic RAG pipeline for BIM spatial queries.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
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
_MAX_QUERY_LEN = 2000
_ALLOWED_IFC_SUFFIXES = {".ifc"}

def _validate_request(query: str, filename: str) -> None:
    """
    Guard against two trivial but real attack surfaces:
      1. Query length — prevents runaway LLM token spend.
      2. Path traversal — filename comes from user input and is joined to a
         filesystem path. pathlib.Path(f).name strips directory components,
         so '../../etc/passwd' becomes 'passwd' which won't match an IFC file.
    """
    if len(query) > _MAX_QUERY_LEN:
        raise HTTPException(400, f"Query too long (max {_MAX_QUERY_LEN} chars).")
    safe = pathlib.Path(filename).name
    if safe != filename or pathlib.Path(filename).suffix.lower() not in _ALLOWED_IFC_SUFFIXES:
        raise HTTPException(400, "Invalid IFC filename.")


def _sse(event_type: str, data: dict) -> str:
    """Format a single Server-Sent Event string."""
    return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"


def _blank_state(query: str, ifc_filename: str, request_id: str = "") -> dict:
    return {
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
        "request_id":          request_id,
        "extracted_guids":     [],
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

    if node_name == "graph_query":
        source  = node_output.get("retrieval_source", "graph")
        records = node_output.get("graph_result_count", 0)
        return {"type": "node_end", **base,
                "data": {"records": records, "source": source}}

    if node_name == "spatial_ast_retrieval":
        log    = node_output.get("correction_log", [])
        loops  = node_output.get("loop_count", 1)
        reason = log[-1].get("failure_reason", "") if log else ""
        return {"type": "self_heal", **base,
                "data": {"loop": loops, "reason": reason}}

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


@app.get("/models", tags=["IFC"])
async def list_models():
    """Return all IFC files available in the data directory."""
    files = sorted(p.name for p in (_PROJECT_ROOT / "data").glob("*.ifc"))
    return {"models": files}


@app.get("/ifc/{filename}", tags=["IFC"])
async def serve_ifc(filename: str):
    """Serve raw IFC file for in-browser 3D loading."""
    safe = pathlib.Path(filename).name
    if safe != filename or not filename.endswith(".ifc"):
        raise HTTPException(400, "Invalid filename.")
    path = _PROJECT_ROOT / "data" / safe
    if not path.exists():
        raise HTTPException(404, f"{filename} not found.")
    return FileResponse(path, media_type="application/octet-stream",
                        headers={"Content-Disposition": f'inline; filename="{safe}"'})


# upload job registry — maps job_id → status dict
_upload_jobs: dict[str, dict] = {}


@app.post("/upload", tags=["IFC"])
async def upload_ifc(file: UploadFile = File(...)):
    """Accept an IFC upload, save it, and re-index in the background."""
    if not file.filename or not file.filename.endswith(".ifc"):
        raise HTTPException(400, "Only .ifc files accepted.")
    safe = pathlib.Path(file.filename).name
    dest = _PROJECT_ROOT / "data" / safe
    job_id = uuid.uuid4().hex[:8]
    _upload_jobs[job_id] = {"status": "saving", "filename": safe}

    content = await file.read()
    dest.write_bytes(content)
    _upload_jobs[job_id]["status"] = "indexing"
    logger.info("upload saved %s (%d bytes) — indexing job %s", safe, len(content), job_id)

    def _index():
        try:
            from indexer.spatial_indexer import index_single_file, build_bm25_from_chroma
            index_single_file(str(dest))
            build_bm25_from_chroma()
            _upload_jobs[job_id]["status"] = "ready"
            logger.info("indexing complete for job %s", job_id)
        except Exception as exc:
            _upload_jobs[job_id]["status"] = "error"
            _upload_jobs[job_id]["error"] = str(exc)
            logger.error("indexing failed for job %s: %s", job_id, exc)

    threading.Thread(target=_index, daemon=True).start()
    return {"job_id": job_id, "filename": safe, "status": "indexing"}


@app.get("/upload/{job_id}", tags=["IFC"])
async def upload_status(job_id: str):
    """Poll indexing status for an uploaded IFC file."""
    job = _upload_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


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
    _validate_request(req.query, req.ifc_filename)
    rid = uuid.uuid4().hex[:8]
    set_request_id(rid)
    t0  = time.time()
    logger.info("request_start query=%r ifc=%s", req.query, req.ifc_filename)

    cached = cache_get(req.query)
    if cached:
        logger.info("cache_hit latency_ms=%d", int((time.time() - t0) * 1000))
        return {
            "answer":              cached["answer"],
            "spatial_constraints": "",
            "retrieval_source":    "cache",
            "cache_hit":           True,
            "self_healed":         False,
            "correction_log":      cached.get("correction_log", []),
            "latency_ms":          int((time.time() - t0) * 1000),
            "request_id":          rid,
        }

    state = graph.invoke(_blank_state(req.query, req.ifc_filename, rid))

    cache_set(req.query, state.get("generation", ""), state.get("correction_log", []), state.get("extracted_guids", []))

    latency_ms = int((time.time() - t0) * 1000)
    logger.info(
        "request_end source=%s self_healed=%s latency_ms=%d",
        state.get("retrieval_source"), state.get("loop_count", 0) > 0, latency_ms,
    )
    return {
        "answer":              state.get("generation", ""),
        "spatial_constraints": state.get("spatial_constraints", ""),
        "retrieval_source":    state.get("retrieval_source", ""),
        "cache_hit":           False,
        "self_healed":         state.get("loop_count", 0) > 0,
        "correction_log":      state.get("correction_log", []),
        "node_timings":        state.get("node_timings", {}),
        "context_token_count": state.get("context_token_count", 0),
        "graph_result_count":  state.get("graph_result_count", 0),
        "latency_ms":          latency_ms,
        "request_id":          rid,
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
    _validate_request(q, f)
    rid = uuid.uuid4().hex[:8]
    set_request_id(rid)
    logger.info("stream_start query=%r ifc=%s", q, f)

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
                "answer":              cached["answer"],
                "correction_log":      cached.get("correction_log", []),
                "cache_hit":           True,
                "latency_ms":          int((time.time() - t0) * 1000),
                "retrieval_source":    "cache",
                "self_healed":         False,
                "node_timings":        {},
                "graph_result_count":  0,
                "extracted_guids":     cached.get("extracted_guids", []),
                "spatial_constraints": "",
            })
            return

        # ── Run LangGraph in a background thread, push events via queue ────
        event_queue: queue.Queue = queue.Queue()
        token_queue: queue.Queue = queue.Queue()
        final_state: dict = {}

        def _run_graph():
            set_request_id(rid)
            set_token_queue(token_queue)   # lets generate node write tokens here
            try:
                state = _blank_state(q, f, rid)
                for node_event in graph.stream(state, stream_mode="updates"):
                    event_queue.put(("node", node_event))
                    node_name   = next(iter(node_event))
                    node_output = node_event[node_name]
                    final_state.update(node_output)
            except Exception as exc:
                event_queue.put(("error", str(exc)))
            finally:
                set_token_queue(None)
                event_queue.put(("done", None))

        thread = threading.Thread(target=_run_graph, daemon=True)
        thread.start()

        # ── Drain queues and yield SSE events ──────────────────────────────
        while True:
            # Flush any pending tokens first — they arrive faster than node events
            while True:
                try:
                    token = token_queue.get_nowait()
                    if token is not None:   # None is the sentinel that generation ended
                        yield _sse("token", {"text": token})
                except queue.Empty:
                    break

            try:
                kind, payload = event_queue.get(timeout=0.05)
            except queue.Empty:
                await asyncio.sleep(0)
                continue

            if kind == "error":
                yield _sse("error", {"message": payload})
                break

            if kind == "done":
                # Write to cache
                cache_set(q, final_state.get("generation", ""), final_state.get("correction_log", []), final_state.get("extracted_guids", []))
                latency_ms = int((time.time() - t0) * 1000)
                logger.info(
                    "stream_end source=%s self_healed=%s latency_ms=%d",
                    final_state.get("retrieval_source"),
                    final_state.get("loop_count", 0) > 0,
                    latency_ms,
                )
                yield _sse("final", {
                    "answer":              final_state.get("generation", ""),
                    "spatial_constraints": final_state.get("spatial_constraints", ""),
                    "retrieval_source":    final_state.get("retrieval_source", ""),
                    "self_healed":         final_state.get("loop_count", 0) > 0,
                    "cache_hit":           False,
                    "correction_log":      final_state.get("correction_log", []),
                    "node_timings":        final_state.get("node_timings", {}),
                    "context_token_count": final_state.get("context_token_count", 0),
                    "graph_result_count":  final_state.get("graph_result_count", 0),
                    "extracted_guids":     final_state.get("extracted_guids", []),
                    "latency_ms":          latency_ms,
                    "request_id":          rid,
                })
                break

            if kind == "node":
                node_name   = next(iter(payload))
                node_output = payload[node_name]
                evt         = _node_to_event(node_name, node_output)
                yield _sse(evt["type"], {"node": evt["node"], **evt["data"]})
                if node_name == "generate":
                    yield _sse("generation_complete", {
                        "answer": node_output.get("generation", ""),
                        "node":   "generate",
                    })

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
