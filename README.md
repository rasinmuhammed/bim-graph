# BIM-Graph

A RAG pipeline for querying Building Information Modeling (BIM) data — specifically IFC files. It started as an experiment to prove that standard chunked-text RAG is the wrong approach for spatial data, and turned into a full agentic system with a graph database at its core.

The short version: ask it "What HVAC equipment is on Level 2?" and it will go find the exact answer using Neo4j, cross-check it, and fall back to deterministic IFC AST traversal if anything looks wrong.

---

## The problem this solves

IFC files encode buildings as a strict spatial hierarchy: Project → Site → Building → Floor → Element. When you chunk that file and embed it into a vector database the way standard RAG does, you destroy that hierarchy. The LLM ends up retrieving chunks that mention "Level 2" somewhere in the text, not chunks that are *actually* from Level 2. In practice this means the model confidently describes elements from the wrong floor.

I call this **spatial blindness** — the retriever finds semantically similar text, not spatially correct elements.

The fix isn't to write better prompts. The fix is to use the right data structure for the job. A floor containment relationship in a building is a graph edge, not something you should infer from cosine similarity.

---

## How it works

The pipeline is a LangGraph state machine with five nodes:

```
query → extract_spatial_constraints
          │
          ├── (floor known) → graph_query → generate → evaluate
          │                                               │
          └── (no floor)   → retrieve_hybrid             ├── (pass) → done
                                                          │
                                                          └── (fail) → spatial_ast_retrieval → generate
```

**Node 1 — extract_spatial_constraints**
The LLM extracts the floor name and query type (inventory vs. specific element). This is a fast, cheap call — it just classifies the intent and sets routing flags.

**Node 2a — graph_query (primary path)**
When a floor is known, we go to Neo4j first. The IFC hierarchy is stored as a proper graph (`Storey -[:CONTAINS]-> Element`), so retrieving everything on Level 2 is a single Cypher query, not a semantic similarity search. For MEP/equipment queries it narrows to equipment-type nodes only. For inventory queries it returns everything.

**Node 2b — retrieve_hybrid (no-floor fallback)**
When there's no spatial constraint, Neo4j doesn't help. So we run BM25 (lexical) + ChromaDB (vector) retrieval and fuse the results with Reciprocal Rank Fusion. This handles cross-floor questions like "how many walls does the building have total?"

**Node 3 — generate**
Standard generation node. The context it receives is already spatially filtered before it gets here — we're not asking the LLM to filter spatial data, just to synthesize a readable answer from it.

**Node 4 — evaluate**
An LLM-as-judge call that checks one specific thing: does the answer respect the floor constraint? Not quality, not completeness — just spatial correctness. If the answer mentions elements from a different floor, it fails.

**Node 5 — spatial_ast_retrieval (self-healing fallback)**
If evaluation fails, we throw out the retrieval results entirely and use IfcOpenShell to parse the IFC file directly. This traverses the actual AST, finds the exact `IfcBuildingStorey` node, and extracts only the elements directly contained in it. This is deterministic — it gives the same answer every time regardless of how the embeddings are set up. Once AST has run, we never loop again.

---

## Retrieval source hierarchy

The system tracks which retrieval path produced the final answer via `retrieval_source` in the state:

| Source | Meaning |
|---|---|
| `graph` | Neo4j answered it — fastest, most precise |
| `dense` | Hybrid BM25 + vector — used when no floor constraint |
| `ast` | IFC AST traversal — self-healing fallback, always correct |
| `graph_unavailable` | Neo4j was down, fell back to hybrid then AST |

---

## Stack

- **Orchestration:** LangGraph + LangChain
- **Graph database:** Neo4j 5 (Community Edition) — stores IFC hierarchy as a property graph
- **Vector search:** ChromaDB (local persistent) — nomic-embed-text via Ollama for embeddings
- **Lexical search:** Rank-BM25 — serialized index, loaded once at startup
- **LLM:** Groq API (llama-3.3-70b-versatile) — fast inference, free tier sufficient for development
- **Semantic cache:** Redis — SHA-256 keyed on normalized query text, 1 hour TTL
- **API:** FastAPI with Server-Sent Events — streams node completions to the UI in real time
- **BIM parsing:** IfcOpenShell — used for both indexing and AST fallback
- **Frontend:** Next.js 14 (App Router)

---

## Setup

**Prerequisites:** Docker, Python 3.11+, Node 18+, [Ollama](https://ollama.com) running locally, a free [Groq API key](https://console.groq.com).

```bash
# 1. Clone and install
git clone https://github.com/yourusername/bim-graph.git
cd bim-graph
pip install -e ".[dev]"

# 2. Pull the embedding model
ollama pull nomic-embed-text

# 3. Start infrastructure
docker compose up redis neo4j -d

# 4. Set your API key
cp .env.example .env
# edit .env and set GROQ_API_KEY=your_key_here

# 5. Drop your IFC files into data/
# Free sample files: https://github.com/buildingSMART/Sample-Test-Files

# 6. Index the IFC files (builds ChromaDB + BM25 index + loads Neo4j)
python -m indexer.chroma_indexer
python -m indexer.bm25_index
python -m graph_db.loader

# 7. Start the API
uvicorn api.main:app --reload --port 8000

# 8. Start the UI
cd ui && npm install && npm run dev
```

Open `http://localhost:3000`.

---

## Running the benchmark

The benchmark runs all 25 queries from `src/benchmark/query_set.json` through the pipeline and scores each answer against the IFC oracle (GUID-level precision/recall/F1):

```bash
python -m benchmark.run_benchmark
```

Results are written to `data/benchmark_results.json`. The oracle is deterministic — it uses IfcOpenShell to get the exact set of element GUIDs on each floor, then checks how many appear in the generated answer.

---

## Tests

```bash
pytest tests/ -v
```

17 tests, all mocked — no running Neo4j, Redis, or Ollama required. The unit tests cover the routing logic, graph query dispatch, cache roundtrip, and context formatting. The test suite runs in under 2 seconds.

---

## Known limitations

**Negation queries don't work.** "Show me everything that is NOT on Level 1" — the floor extractor pulls "Level 1" as the constraint and retrieves Level 1 elements. This would need a separate query intent classifier.

**Multi-floor single queries are partially handled.** "What's on floors 1 and 2?" extracts one floor, queries that, and misses the other. The benchmark includes this as an adversarial test case.

**The evaluate node is LLM-judged.** It's good but not perfect — occasionally passes answers that have minor floor confusion or fails correct answers that don't explicitly mention the floor. A GUID-based evaluator (checking retrieved GUIDs against the oracle) would be more reliable but requires the generate node to always output GUIDs, which makes the answers less readable.

**Embeddings are cold-started.** The Ollama nomic-embed-text model loads on first request. If you query immediately after startup there's a 3-5 second delay before the first real response.

---

## Project structure

```
src/
  agent/
    graph.py      — LangGraph state machine and routing logic
    nodes.py      — all five node functions
    state.py      — TypedDict state schema
  api/
    main.py       — FastAPI endpoints + SSE streaming
  benchmark/
    ifc_oracle.py      — ground truth GUID extraction from IFC AST
    run_benchmark.py   — runs all 25 queries, scores with P/R/F1
    query_set.json     — 25 test queries across 5 categories
  cache/
    redis_cache.py     — get/set with SHA-256 key + fakeredis fallback
  config.py            — pydantic-settings, all config from .env
  graph_db/
    loader.py     — IFC → Neo4j ingestion (MERGE idempotent)
    queries.py    — Cypher query library
  indexer/
    chroma_indexer.py  — spatial chunking → ChromaDB
    bm25_index.py      — BM25 index builder
    spatial_indexer.py — IFC → spatial chunk extraction
  parser/
    ifc_parser.py      — raw IfcOpenShell traversal
ui/                    — Next.js frontend
tests/
  unit/                — 17 mocked unit tests
```

---

## What I'd do differently

The biggest design decision I'd revisit is the evaluate node. Right now it's an LLM call on every query, which adds ~800ms of latency and occasionally produces wrong verdicts. A better design would skip LLM evaluation for graph-sourced results entirely — if Neo4j says an element is on Level 2, it is on Level 2, you don't need an LLM to second-guess it. The evaluator should only run when the retrieval source is `dense` or after AST (where spatial proof is embedded in the context, not guaranteed).

The other thing: the AST fallback is currently triggered by evaluation failure, not by confidence. A more robust approach would be to run a lightweight confidence check on the graph result count — if Neo4j returns 0 elements, skip generation and go straight to AST rather than generating a "no elements found" answer and then evaluating it.

---

## Background

This came out of work on digital twins for large construction projects, where querying "what assets are on this floor" from a chunked IFC file was producing consistently wrong answers. The spatial hierarchy in IFC is explicit and machine-readable — it seemed wrong to throw it away and reconstruct it from semantic similarity. BIM-Graph is an attempt to keep that hierarchy intact through the entire query pipeline.
