"use client";

import { useState, useEffect, useRef, useCallback } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────
type NodeStatus = "idle" | "active" | "done" | "heal" | "hit";
type NodeStates = Record<string, NodeStatus>;

interface Floor {
  name: string;
  elevation_m: number;
  element_count: number;
}

interface CorrectionEntry {
  attempt: number;
  search_strategy: string;
  failure_reason: string;
  action_taken: string;
}

interface FinalResult {
  answer: string;
  spatial_constraints: string;
  retrieval_source: string;
  self_healed: boolean;
  cache_hit: boolean;
  correction_log: CorrectionEntry[];
  latency_ms: number;
}

interface LogLine {
  id: number;
  text: string;
  type: "info" | "success" | "warn" | "heal";
}

// ── Constants ──────────────────────────────────────────────────────────────────
const API = "http://localhost:8000";

const PIPELINE_NODES = [
  "extract_spatial_constraints",
  "retrieve_hybrid",
  "generate",
  "evaluate",
  "spatial_ast_retrieval",
] as const;

const NODE_META: Record<string, { label: string }> = {
  extract_spatial_constraints: { label: "Extract Context" },
  retrieve_hybrid:             { label: "Hybrid Retrieve" },
  generate:                    { label: "Generate" },
  evaluate:                    { label: "Evaluate" },
  spatial_ast_retrieval:       { label: "AST Proof" },
};

const BLANK_NODES: NodeStates = {
  extract_spatial_constraints: "idle",
  retrieve_hybrid:             "idle",
  generate:                    "idle",
  evaluate:                    "idle",
  spatial_ast_retrieval:       "idle",
};

const FILES = [
  "Duplex_A_20110907.ifc",
  "AC20-FZK-Haus.ifc",
  "AC20-Institute-Var-2.ifc"
];

// ── Helpers ────────────────────────────────────────────────────────────────────
let _logId = 0;
const mkLog = (text: string, type: LogLine["type"] = "info"): LogLine => ({
  id: _logId++,
  text,
  type,
});

// ── Component ──────────────────────────────────────────────────────────────────
export default function BIMGraphUI() {
  const [query,      setQuery]      = useState("");
  const [floors,     setFloors]     = useState<Floor[]>([]);
  const [running,    setRunning]    = useState(false);
  const [nodeStates, setNodeStates] = useState<NodeStates>(BLANK_NODES);
  const [logs,       setLogs]       = useState<LogLine[]>([]);
  const [result,     setResult]     = useState<FinalResult | null>(null);
  const [nodeData,   setNodeData]   = useState<Record<string, Record<string, unknown>>>({});
  const [selectedFile, setSelectedFile] = useState(FILES[0]);

  const esRef    = useRef<EventSource | null>(null);
  const logEnd   = useRef<HTMLDivElement>(null);

  // Load floors on file selection
  useEffect(() => {
    fetch(`${API}/floors?f=${selectedFile}`)
      .then((r) => r.json())
      .then((d) => setFloors(d.floors ?? []))
      .catch(() => {});
  }, [selectedFile]);

  // Auto-scroll logs
  useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Pipeline execution
  const runQuery = useCallback(() => {
    if (!query.trim() || running) return;

    esRef.current?.close();
    setRunning(true);
    setResult(null);
    setLogs([mkLog(`Initialized query execution`, "info")]);
    setNodeStates({ ...BLANK_NODES, extract_spatial_constraints: "active" });
    setNodeData({});

    const es = new EventSource(
      `${API}/query/stream?q=${encodeURIComponent(query.trim())}&f=${selectedFile}`
    );
    esRef.current = es;

    es.onmessage = (e: MessageEvent) => {
      const { type, data } = JSON.parse(e.data) as {
        type: string;
        data: Record<string, unknown>;
      };

      if (type === "cache_hit") {
        setNodeStates((p) => ({ ...p, extract_spatial_constraints: "hit" }));
        setLogs((p) => [...p, mkLog("Cache hit verified, skipping pipeline", "warn")]);
        return;
      }

      if (type === "node_end") {
        const node = data.node as string;

        let shouldSkipDense = false;
        if (node === "extract_spatial_constraints" && data.is_inventory_query) {
            shouldSkipDense = true;
            setLogs((p) => [...p, mkLog(`Inventory query detected: bypassing semantic layer and routing to deterministic AST...`, "heal")]);
        }

        setNodeStates((prev) => {
          const next = { ...prev, [node]: "done" as NodeStatus };

          if (shouldSkipDense) {
              next["retrieve_hybrid"] = "hit";
              next["generate"] = "hit";
              next["evaluate"] = "hit";
              next["spatial_ast_retrieval"] = "active";
          } else {
              const idx = PIPELINE_NODES.indexOf(node as typeof PIPELINE_NODES[number]);
              const nextNode = PIPELINE_NODES[idx + 1];
              if (nextNode && next[nextNode] === "idle") next[nextNode] = "active";
          }
          return next;
        });
        setNodeData((p) => ({ ...p, [node]: data }));

        const labels: Record<string, string> = {
          extract_spatial_constraints: `Spatial constraints resolved`,
          retrieve_hybrid:             `Vector/BM25 retrieval complete`,
          generate:                    `Base generation step finished`,
          evaluate:                    `Verification layer processed`,
          spatial_ast_retrieval:       `AST self-repair successful`,
        };
        setLogs((p) => [...p, mkLog(labels[node] ?? node, "success")]);
      }

      if (type === "self_heal") {
        setNodeStates((p) => ({
          ...p,
          evaluate:              "done",
          spatial_ast_retrieval: "heal",
          generate:              "active",
        }));
        setLogs((p) => [
          ...p,
          mkLog(`Triggering AST repair: ${data.reason}`, "heal"),
        ]);
      }

      if (type === "error") {
        setLogs((p) => [...p, mkLog(`Exception: ${data.message}`, "warn")]);
        setRunning(false);
        es.close();
      }

      if (type === "final") {
        const r = data as unknown as FinalResult;
        setResult(r);
        setNodeStates((p) => {
          const next = { ...p };
          for (const n of PIPELINE_NODES) {
            if (next[n] === "active") next[n] = "done";
          }
          return next;
        });
        setLogs((p) => [
          ...p,
          mkLog(`Execution completed in ${r.latency_ms}ms`, "success"),
        ]);
        setRunning(false);
        es.close();
      }
    };

    es.onerror = () => {
      setLogs((p) => [...p, mkLog("Lost connection to inference engine", "warn")]);
      setRunning(false);
      es.close();
    };
  }, [query, running, selectedFile]);


  // ── Layout Render ────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen w-screen bg-black text-slate-200 font-sans overflow-hidden">
      
      {/* ── LEFT SIDEBAR ── */}
      <aside className="w-[280px] bg-[#050505] border-r border-white/10 flex flex-col shrink-0 z-10 h-full">
        {/* Brand Block */}
        <div className="p-6 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-3.5 h-3.5 bg-white rounded-sm" />
            <span className="text-white font-semibold text-base tracking-tight">BIM-Graph</span>
          </div>
          <p className="mt-2 text-xs text-slate-500">Agentic Spatial Retrieval</p>
        </div>

        {/* File Config */}
        <div className="p-6 flex flex-col gap-6 overflow-y-auto">
          <div>
            <label className="block text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Active Model
            </label>
            <select
              className="w-full bg-[#111] border border-white/10 text-slate-200 text-xs p-2.5 rounded-md outline-none appearance-none hover:border-white/20 transition-colors"
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
            >
              {FILES.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>

          {/* Floors list minimal */}
          {floors.length > 0 && (
            <div>
              <label className="block text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-3">
                Spatial Index
              </label>
              <div className="flex flex-col gap-1.5">
                {floors.map((f) => (
                  <div key={f.name} className="flex justify-between items-baseline px-2 py-1.5 rounded bg-white/5 hover:bg-white/10 transition-colors">
                    <span className="text-xs text-slate-300">{f.name}</span>
                    <span className="text-[10px] text-slate-500 font-mono">{f.element_count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="flex-1" />
        <div className="p-6 border-t border-white/5">
           <a href="http://localhost:8000/docs" target="_blank" className="text-xs text-slate-500 hover:text-white transition-colors flex items-center gap-1">
             API Reference <span className="text-[10px]">↗</span>
           </a>
        </div>
      </aside>

      {/* ── RIGHT MAIN CANVAS ── */}
      <main className="flex-1 bg-[#0a0a0a] overflow-y-auto custom-scroll relative">
        <div className="w-full max-w-3xl mx-auto pt-24 pb-32 px-8 flex flex-col items-center">
          
          {/* Command Input */}
          <div className="w-full bg-[#111] border border-white/10 rounded-xl shadow-2xl overflow-hidden focus-within:border-white/20 transition-colors">
            <textarea
              className="w-full bg-transparent text-white text-lg p-6 outline-none resize-none"
              placeholder="Query the active BIM model..."
              rows={2}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  runQuery();
                }
              }}
            />
            <div className="flex items-center justify-between px-6 py-3 border-t border-white/5 bg-[#0a0a0a]">
              <span className="text-xs text-slate-500 font-mono">
                Press <strong className="text-slate-300 font-medium">Enter</strong> to execute analysis
              </span>
              <button 
                className="bg-white text-black px-4 py-1.5 rounded-md text-xs font-semibold hover:bg-slate-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={runQuery} 
                disabled={running || !query.trim()}
              >
                {running ? (
                  <span className="flex items-center gap-2">
                    <div className="w-3 h-3 border-2 border-black/20 border-t-black rounded-full animate-spin" />
                    Processing
                  </span>
                ) : "Submit"}
              </button>
            </div>
          </div>

          {/* Stepper Toolbar (Only visible when running or result exists) */}
          {(running || result) && (
            <div className="w-full mt-12 px-6 flex items-center gap-3 animate-in fade-in duration-500">
              {PIPELINE_NODES.map((id, index) => {
                const isLast = index === PIPELINE_NODES.length - 1;
                const status = nodeStates[id];
                const label = NODE_META[id].label;
                
                // Colors based on status
                let dotColor = "border-slate-700 bg-transparent";
                if (status === 'active') dotColor = "border-blue-500 bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.5)]";
                if (status === 'done' || status === 'hit') dotColor = "border-emerald-500 bg-emerald-500";
                if (status === 'heal') dotColor = "border-amber-500 bg-amber-500 animate-pulse";

                return (
                  <div key={id} className={`flex items-center ${isLast ? '' : 'flex-1'} ${status === 'idle' ? 'opacity-30' : 'opacity-100'} transition-opacity duration-300`}>
                    <div className="flex items-center gap-2">
                      <div className={`w-2.5 h-2.5 rounded-full border ${dotColor} transition-all duration-300`} />
                      <span className={`text-[10px] uppercase font-semibold tracking-wider ${status === 'idle' ? 'hidden sm:block text-slate-600' : 'text-slate-300'}`}>
                        {label}
                      </span>
                    </div>
                    {!isLast && <div className="flex-1 h-px bg-white/10 ml-3 hidden sm:block min-w-[20px]" />}
                  </div>
                );
              })}
            </div>
          )}

          {/* Answer Report */}
          {result && (
            <div className="w-full mt-12 border-t border-white/10 pt-10 animate-in slide-in-from-bottom-4 duration-500">
              
              {/* Badges */}
              <div className="flex flex-wrap gap-2 mb-6">
                {result.cache_hit && (
                  <span className="px-2 py-1 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-mono rounded tracking-wide">
                    ⚡ CACHE HIT
                  </span>
                )}
                {result.self_healed && (
                  <span className="px-2 py-1 bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[10px] font-mono rounded tracking-wide">
                    ⟳ HEALED ROUTE
                  </span>
                )}
                <span className="px-2 py-1 bg-white/5 border border-white/10 text-slate-400 text-[10px] font-mono rounded tracking-wide">
                  {result.retrieval_source === "ast" ? "SPATIAL TRUTH" : "SEMANTIC SEARCH"}
                </span>
                <span className="px-2 py-1 bg-white/5 border border-white/10 text-slate-400 text-[10px] font-mono rounded tracking-wide">
                  {result.latency_ms}MS
                </span>
              </div>

              {/* Text Answer */}
              <div className="text-slate-200 text-base leading-relaxed whitespace-pre-wrap font-sans">
                {result.answer}
              </div>

              {/* Event Logs Trace */}
              <div className="mt-16 border-t border-white/5 pt-8">
                <p className="text-xs font-semibold text-slate-400 mb-4">Execution Trace</p>
                <div className="flex flex-col gap-1.5">
                  {logs.map(l => {
                    let textColor = "text-slate-500";
                    if (l.type === "success") textColor = "text-emerald-500/80";
                    if (l.type === "warn" || l.type === "heal") textColor = "text-amber-500/80";

                    return (
                      <div key={l.id} className={`text-[11px] font-mono ${textColor}`}>
                        <span className="opacity-50 mr-2">[+]</span>
                        {l.text}
                      </div>
                    );
                  })}
                  <div ref={logEnd} />
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
