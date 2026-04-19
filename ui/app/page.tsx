"use client";

import { useState, useEffect, useRef, useCallback } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────
type NodeStatus = "idle" | "active" | "done" | "heal" | "hit" | "skipped";
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
  node_timings: Record<string, number>;
  graph_result_count: number;
  request_id: string;
}

interface LogLine {
  id: number;
  text: string;
  type: "info" | "success" | "warn" | "heal";
}

// ── Constants ──────────────────────────────────────────────────────────────────
const API = "http://localhost:8000";

// graph_query is now included — it's the primary retrieval path when a floor is known
const PIPELINE_NODES = [
  "extract_spatial_constraints",
  "graph_query",
  "retrieve_hybrid",
  "generate",
  "evaluate",
  "spatial_ast_retrieval",
] as const;

const NODE_META: Record<string, { label: string }> = {
  extract_spatial_constraints: { label: "Extract Context" },
  graph_query:                 { label: "Graph Query" },
  retrieve_hybrid:             { label: "Hybrid Retrieve" },
  generate:                    { label: "Generate" },
  evaluate:                    { label: "Evaluate" },
  spatial_ast_retrieval:       { label: "AST Proof" },
};

const BLANK_NODES: NodeStates = {
  extract_spatial_constraints: "idle",
  graph_query:                 "idle",
  retrieve_hybrid:             "idle",
  generate:                    "idle",
  evaluate:                    "idle",
  spatial_ast_retrieval:       "idle",
};

const FILES = [
  "Duplex_A_20110907.ifc",
  "AC20-FZK-Haus.ifc",
  "AC20-Institute-Var-2.ifc",
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
  const [query,           setQuery]           = useState("");
  const [floors,          setFloors]          = useState<Floor[]>([]);
  const [running,         setRunning]         = useState(false);
  const [nodeStates,      setNodeStates]      = useState<NodeStates>(BLANK_NODES);
  const [logs,            setLogs]            = useState<LogLine[]>([]);
  const [result,          setResult]          = useState<FinalResult | null>(null);
  const [selectedFile,    setSelectedFile]    = useState(FILES[0]);
  const [streamingAnswer, setStreamingAnswer] = useState("");

  const esRef  = useRef<EventSource | null>(null);
  const logEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${API}/floors?f=${selectedFile}`)
      .then((r) => r.json())
      .then((d) => setFloors(d.floors ?? []))
      .catch(() => {});
  }, [selectedFile]);

  useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const runQuery = useCallback(() => {
    if (!query.trim() || running) return;

    esRef.current?.close();
    setRunning(true);
    setResult(null);
    setStreamingAnswer("");
    setLogs([mkLog("Initialized query execution", "info")]);
    setNodeStates({ ...BLANK_NODES, extract_spatial_constraints: "active" });

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
        setLogs((p) => [...p, mkLog("Cache hit — skipping pipeline", "warn")]);
        return;
      }

      if (type === "token") {
        // data.text is the field name — backend sends {"type":"token","data":{"text":"..."}}
        setStreamingAnswer((prev) => prev + (data.text as string ?? ""));
        return;
      }

      if (type === "generation_complete") {
        // Streaming finished — the full answer is now in streamingAnswer.
        // Keep it displayed until the final event arrives and sets result.
        return;
      }

      if (type === "node_end") {
        const node = data.node as string;

        setNodeStates((prev) => {
          const next = { ...prev, [node]: "done" as NodeStatus };

          // After extract, mark which retrieval path is active and which is skipped.
          // graph_query fires when a floor is known; retrieve_hybrid fires otherwise.
          if (node === "extract_spatial_constraints") {
            const hasFloor = Boolean(data.floor);
            if (hasFloor) {
              next["graph_query"]     = "active";
              next["retrieve_hybrid"] = "skipped";
            } else {
              next["graph_query"]     = "skipped";
              next["retrieve_hybrid"] = "active";
            }
          } else if (node === "graph_query" || node === "retrieve_hybrid") {
            // Both feed into generate
            next["generate"] = "active";
          } else if (node === "generate") {
            // evaluate is only active for dense results — graph/ast route to END.
            // We don't know retrieval_source here, so leave evaluate idle;
            // it will flip to active if it actually fires.
          } else if (node === "evaluate") {
            // evaluate ran — keep evaluate as done, spatial_ast_retrieval may follow
          }

          return next;
        });



        const labels: Record<string, string> = {
          extract_spatial_constraints: `Constraints extracted — floor: ${data.floor || "none"}`,
          graph_query:                 `Graph query complete — ${data.records ?? 0} records`,
          retrieve_hybrid:             `Hybrid retrieval complete`,
          generate:                    `Generation finished`,
          evaluate:                    `Spatial evaluation passed`,
          spatial_ast_retrieval:       `AST self-repair complete`,
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
          mkLog(`Self-healing triggered: ${data.reason}`, "heal"),
        ]);
      }

      if (type === "error") {
        setLogs((p) => [...p, mkLog(`Error: ${data.message}`, "warn")]);
        setRunning(false);
        es.close();
      }

      if (type === "final") {
        const r = data as unknown as FinalResult;
        setStreamingAnswer("");   // clear streaming display — result.answer takes over
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
          mkLog(
            `Done in ${r.latency_ms}ms — source: ${r.retrieval_source}${r.self_healed ? " (self-healed)" : ""}`,
            "success"
          ),
        ]);
        setRunning(false);
        es.close();
      }
    };

    es.onerror = () => {
      setLogs((p) => [...p, mkLog("Connection lost", "warn")]);
      setRunning(false);
      es.close();
    };
  }, [query, running, selectedFile]);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen w-screen bg-black text-slate-200 font-sans overflow-hidden">

      {/* ── LEFT SIDEBAR ── */}
      <aside className="w-[280px] bg-[#050505] border-r border-white/10 flex flex-col shrink-0 z-10 h-full">
        <div className="p-6 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-3.5 h-3.5 bg-white rounded-sm" />
            <span className="text-white font-semibold text-base tracking-tight">BIM-Graph</span>
          </div>
          <p className="mt-2 text-xs text-slate-500">Agentic Spatial Retrieval</p>
        </div>

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

          {floors.length > 0 && (
            <div>
              <label className="block text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-3">
                Spatial Index
              </label>
              <div className="flex flex-col gap-1.5">
                {floors.map((f) => (
                  <div
                    key={f.name}
                    className="flex justify-between items-baseline px-2 py-1.5 rounded bg-white/5 hover:bg-white/10 transition-colors"
                  >
                    <span className="text-xs text-slate-300">{f.name}</span>
                    <span className="text-[10px] text-slate-500 font-mono">{f.element_count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="flex-1" />
        <div className="p-6 border-t border-white/5 flex flex-col gap-3">
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            className="text-xs text-slate-500 hover:text-white transition-colors flex items-center gap-1"
          >
            API Reference <span className="text-[10px]">↗</span>
          </a>
          <a
            href="/benchmark"
            className="text-xs text-slate-500 hover:text-white transition-colors flex items-center gap-1"
          >
            Benchmark Dashboard <span className="text-[10px]">↗</span>
          </a>
        </div>
      </aside>

      {/* ── MAIN CANVAS ── */}
      <main className="flex-1 bg-[#0a0a0a] overflow-y-auto relative">
        <div className="w-full max-w-3xl mx-auto pt-24 pb-32 px-8 flex flex-col items-center">

          {/* Query input */}
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
                Press <strong className="text-slate-300 font-medium">Enter</strong> to execute
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

          {/* Pipeline stepper */}
          {(running || result) && (
            <div className="w-full mt-12 px-2 flex items-center gap-2 animate-in fade-in duration-500 flex-wrap">
              {PIPELINE_NODES.map((id, index) => {
                const isLast  = index === PIPELINE_NODES.length - 1;
                const status  = nodeStates[id];
                const label   = NODE_META[id].label;

                let dotColor = "border-slate-700 bg-transparent";
                if (status === "active")  dotColor = "border-blue-500 bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.5)]";
                if (status === "done" || status === "hit") dotColor = "border-emerald-500 bg-emerald-500";
                if (status === "heal")    dotColor = "border-amber-500 bg-amber-500 animate-pulse";
                if (status === "skipped") dotColor = "border-slate-700 bg-slate-700 opacity-30";

                return (
                  <div
                    key={id}
                    className={`flex items-center ${isLast ? "" : "flex-1"} ${
                      status === "idle" ? "opacity-30" : "opacity-100"
                    } transition-opacity duration-300`}
                  >
                    <div className="flex items-center gap-2">
                      <div className={`w-2.5 h-2.5 rounded-full border ${dotColor} transition-all duration-300`} />
                      <span className={`text-[10px] uppercase font-semibold tracking-wider whitespace-nowrap ${
                        status === "idle" || status === "skipped" ? "text-slate-600" : "text-slate-300"
                      }`}>
                        {label}
                      </span>
                    </div>
                    {!isLast && <div className="flex-1 h-px bg-white/10 ml-3 min-w-[12px]" />}
                  </div>
                );
              })}
            </div>
          )}

          {/* Streaming answer — visible while generate node is running */}
          {streamingAnswer && !result && (
            <div className="w-full mt-12 border border-white/10 rounded-xl p-6 bg-white/[0.02] animate-in fade-in duration-300">
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse inline-block" />
                Generating
              </p>
              <p className="text-slate-200 text-base leading-relaxed whitespace-pre-wrap font-sans">
                {streamingAnswer}
                <span className="inline-block w-0.5 h-4 bg-slate-400 animate-pulse ml-0.5 align-middle" />
              </p>
            </div>
          )}

          {/* Final answer */}
          {result && (
            <div className="w-full mt-12 border-t border-white/10 pt-10 animate-in slide-in-from-bottom-4 duration-500">

              {/* Badges */}
              <div className="flex flex-wrap gap-2 mb-6">
                {result.cache_hit && (
                  <span className="px-2 py-1 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-mono rounded tracking-wide">
                    CACHE HIT
                  </span>
                )}
                {result.self_healed && (
                  <span className="px-2 py-1 bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[10px] font-mono rounded tracking-wide">
                    SELF-HEALED
                  </span>
                )}
                <span className="px-2 py-1 bg-white/5 border border-white/10 text-slate-400 text-[10px] font-mono rounded tracking-wide uppercase">
                  {result.retrieval_source}
                </span>
                {result.graph_result_count > 0 && (
                  <span className="px-2 py-1 bg-purple-500/10 border border-purple-500/20 text-purple-400 text-[10px] font-mono rounded tracking-wide">
                    {result.graph_result_count} GRAPH RECORDS
                  </span>
                )}
                <span className="px-2 py-1 bg-white/5 border border-white/10 text-slate-400 text-[10px] font-mono rounded tracking-wide">
                  {result.latency_ms}MS
                </span>
              </div>

              {/* Answer text */}
              <div className="text-slate-200 text-base leading-relaxed whitespace-pre-wrap font-sans">
                {result.answer}
              </div>

              {/* Node timings */}
              {result.node_timings && Object.keys(result.node_timings).length > 0 && (
                <div className="mt-8 border border-white/5 rounded-lg p-4">
                  <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-3">Node Timings</p>
                  <div className="flex flex-col gap-1.5">
                    {Object.entries(result.node_timings).map(([node, ms]) => (
                      <div key={node} className="flex justify-between items-center">
                        <span className="text-[11px] text-slate-400 font-mono">{node}</span>
                        <span className="text-[11px] text-slate-500 font-mono">{(ms * 1000).toFixed(0)}ms</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Execution trace */}
              <div className="mt-8 border-t border-white/5 pt-8">
                <p className="text-xs font-semibold text-slate-400 mb-4">Execution Trace</p>
                <div className="flex flex-col gap-1.5">
                  {logs.map((l) => {
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
