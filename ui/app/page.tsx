"use client";

import dynamic from "next/dynamic";
import { useState, useEffect, useRef, useCallback } from "react";

const IfcViewer = dynamic(() => import("../components/IfcViewer"), { ssr: false });

// ── Types ──────────────────────────────────────────────────────────────────────
type NodeStatus = "idle" | "active" | "done" | "heal" | "hit" | "skipped";
type NodeStates = Record<string, NodeStatus>;

interface Floor {
  name: string;
  elevation_m: number;
  element_count: number;
}

interface FinalResult {
  answer: string;
  spatial_constraints: string;
  retrieval_source: string;
  self_healed: boolean;
  cache_hit: boolean;
  latency_ms: number;
  node_timings: Record<string, number>;
  graph_result_count: number;
  extracted_guids: string[];
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
  "graph_query",
  "retrieve_hybrid",
  "generate",
  "evaluate",
  "spatial_ast_retrieval",
] as const;

const NODE_LABELS: Record<string, string> = {
  extract_spatial_constraints: "Extract",
  graph_query:                 "Graph",
  retrieve_hybrid:             "Hybrid",
  generate:                    "Generate",
  evaluate:                    "Evaluate",
  spatial_ast_retrieval:       "AST Proof",
};

const BLANK_NODES: NodeStates = Object.fromEntries(
  PIPELINE_NODES.map((n) => [n, "idle"])
);

let _logId = 0;
const mkLog = (text: string, type: LogLine["type"] = "info"): LogLine => ({
  id: _logId++, text, type,
});

// ── Component ──────────────────────────────────────────────────────────────────
export default function BIMGraphUI() {
  const [query,           setQuery]           = useState("");
  const [floors,          setFloors]          = useState<Floor[]>([]);
  const [models,          setModels]          = useState<string[]>([]);
  const [running,         setRunning]         = useState(false);
  const [nodeStates,      setNodeStates]      = useState<NodeStates>(BLANK_NODES);
  const [logs,            setLogs]            = useState<LogLine[]>([]);
  const [result,          setResult]          = useState<FinalResult | null>(null);
  const [selectedFile,    setSelectedFile]    = useState("Duplex_A_20110907.ifc");
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [highlightGuids,  setHighlightGuids]  = useState<string[]>([]);
  const [uploading,       setUploading]       = useState(false);
  const [uploadMsg,       setUploadMsg]       = useState("");

  const esRef      = useRef<EventSource | null>(null);
  const chatEnd    = useRef<HTMLDivElement>(null);
  const fileInput  = useRef<HTMLInputElement>(null);

  // load model list and floors
  useEffect(() => {
    fetch(`${API}/models`).then(r => r.json()).then(d => setModels(d.models ?? [])).catch(() => {});
  }, []);

  useEffect(() => {
    fetch(`${API}/floors?f=${selectedFile}`)
      .then(r => r.json())
      .then(d => setFloors(d.floors ?? []))
      .catch(() => {});
  }, [selectedFile]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs, streamingAnswer]);

  // ── Upload ─────────────────────────────────────────────────────────────────
  const handleUpload = useCallback(async (file: File) => {
    if (!file.name.endsWith(".ifc")) { setUploadMsg("Only .ifc files accepted."); return; }
    setUploading(true);
    setUploadMsg("Uploading…");
    const form = new FormData();
    form.append("file", file);
    const resp = await fetch(`${API}/upload`, { method: "POST", body: form });
    if (!resp.ok) { setUploading(false); setUploadMsg("Upload failed."); return; }
    const { job_id, filename } = await resp.json();
    setUploadMsg("Indexing…");
    // poll until ready
    const poll = setInterval(async () => {
      const s = await fetch(`${API}/upload/${job_id}`).then(r => r.json());
      if (s.status === "ready") {
        clearInterval(poll);
        setUploading(false);
        setUploadMsg(`${filename} ready.`);
        setModels(prev => prev.includes(filename) ? prev : [...prev, filename]);
        setSelectedFile(filename);
        setTimeout(() => setUploadMsg(""), 3000);
      } else if (s.status === "error") {
        clearInterval(poll);
        setUploading(false);
        setUploadMsg("Indexing failed: " + s.error);
      }
    }, 1500);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  }, [handleUpload]);

  // ── Query ──────────────────────────────────────────────────────────────────
  const runQuery = useCallback(() => {
    if (!query.trim() || running) return;
    esRef.current?.close();
    setRunning(true);
    setResult(null);
    setStreamingAnswer("");
    setHighlightGuids([]);
    setLogs([mkLog("Initialized", "info")]);
    setNodeStates({ ...BLANK_NODES, extract_spatial_constraints: "active" });

    const es = new EventSource(
      `${API}/query/stream?q=${encodeURIComponent(query.trim())}&f=${selectedFile}`
    );
    esRef.current = es;

    es.onmessage = (e: MessageEvent) => {
      const { type, data } = JSON.parse(e.data);

      if (type === "token") {
        setStreamingAnswer(prev => prev + (data.text ?? ""));
        return;
      }
      if (type === "node_end") {
        const node = data.node as string;
        setNodeStates(prev => {
          const next = { ...prev, [node]: "done" as NodeStatus };
          if (node === "extract_spatial_constraints") {
            const hasFloor = Boolean(data.floor);
            next["graph_query"]     = hasFloor ? "active" : "skipped";
            next["retrieve_hybrid"] = hasFloor ? "skipped" : "active";
          } else if (node === "graph_query" || node === "retrieve_hybrid") {
            next["generate"] = "active";
          }
          return next;
        });
        const labels: Record<string, string> = {
          extract_spatial_constraints: `Floor: ${data.floor || "none"}`,
          graph_query:                 `${data.records ?? 0} graph records`,
          retrieve_hybrid:             "Hybrid retrieval done",
          generate:                    "Answer generated",
          evaluate:                    "Evaluation passed",
          spatial_ast_retrieval:       "AST proof complete",
        };
        setLogs(p => [...p, mkLog(labels[node] ?? node, "success")]);
      }
      if (type === "self_heal") {
        setNodeStates(p => ({
          ...p, evaluate: "done", spatial_ast_retrieval: "heal", generate: "active",
        }));
        setLogs(p => [...p, mkLog(`Self-heal: ${data.reason}`, "heal")]);
      }
      if (type === "error") {
        setLogs(p => [...p, mkLog(`Error: ${data.message}`, "warn")]);
        setRunning(false);
        es.close();
      }
      if (type === "final") {
        const r = data as FinalResult;
        setStreamingAnswer("");
        setResult(r);
        if (r.extracted_guids?.length) setHighlightGuids(r.extracted_guids);
        setNodeStates(p => {
          const next = { ...p };
          for (const n of PIPELINE_NODES) if (next[n] === "active") next[n] = "done";
          return next;
        });
        setLogs(p => [...p, mkLog(
          `Done ${r.latency_ms}ms · ${r.retrieval_source}${r.self_healed ? " · self-healed" : ""}`,
          "success"
        )]);
        setRunning(false);
        es.close();
      }
    };

    es.onerror = () => {
      setLogs(p => [...p, mkLog("Connection lost", "warn")]);
      setRunning(false);
      es.close();
    };
  }, [query, running, selectedFile]);

  const ifcUrl = `${API}/ifc/${selectedFile}`;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen w-screen bg-black text-slate-200 font-sans overflow-hidden">

      {/* LEFT SIDEBAR */}
      <aside className="w-[220px] bg-[#050505] border-r border-white/10 flex flex-col shrink-0">
        <div className="p-5 border-b border-white/5">
          <div className="flex items-center gap-2.5">
            <div className="w-3 h-3 bg-white rounded-sm" />
            <span className="text-white font-semibold text-sm tracking-tight">BIM-Graph</span>
          </div>
          <p className="mt-1 text-[10px] text-slate-500">Agentic Spatial Retrieval</p>
        </div>

        <div className="p-4 flex flex-col gap-4 overflow-y-auto flex-1">
          {/* Model selector */}
          <div>
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">Active Model</p>
            <select
              className="w-full bg-[#111] border border-white/10 text-slate-200 text-xs p-2 rounded-md outline-none appearance-none"
              value={selectedFile}
              onChange={e => { setSelectedFile(e.target.value); setResult(null); setHighlightGuids([]); }}
            >
              {models.map(f => <option key={f} value={f}>{f.replace(".ifc", "")}</option>)}
            </select>
          </div>

          {/* Drag-and-drop upload */}
          <div>
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">Upload IFC</p>
            <div
              className={`border border-dashed rounded-lg p-3 text-center cursor-pointer transition-colors ${
                uploading ? "border-blue-500/50 bg-blue-500/5" : "border-white/10 hover:border-white/30"
              }`}
              onDrop={onDrop}
              onDragOver={e => e.preventDefault()}
              onClick={() => fileInput.current?.click()}
            >
              <input ref={fileInput} type="file" accept=".ifc" className="hidden"
                onChange={e => { const f = e.target.files?.[0]; if (f) handleUpload(f); }} />
              {uploading
                ? <div className="w-4 h-4 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto" />
                : <p className="text-[10px] text-slate-500">Drop .ifc or click</p>
              }
              {uploadMsg && <p className="text-[10px] mt-1 text-slate-400">{uploadMsg}</p>}
            </div>
          </div>

          {/* Floor index */}
          {floors.length > 0 && (
            <div>
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">Floors</p>
              <div className="flex flex-col gap-1">
                {floors.map(f => (
                  <div key={f.name} className="flex justify-between px-2 py-1 rounded bg-white/5 text-xs">
                    <span className="text-slate-300 truncate">{f.name}</span>
                    <span className="text-slate-500 font-mono ml-2 shrink-0">{f.element_count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Highlighted element count */}
          {highlightGuids.length > 0 && (
            <div className="p-2 rounded-lg bg-orange-500/10 border border-orange-500/20">
              <p className="text-[10px] text-orange-400 font-semibold">
                {highlightGuids.length} elements highlighted
              </p>
              <button
                className="text-[10px] text-slate-500 hover:text-white mt-0.5"
                onClick={() => setHighlightGuids([])}
              >
                Clear
              </button>
            </div>
          )}
        </div>

        <div className="p-4 border-t border-white/5 flex flex-col gap-2">
          <a href="/benchmark" className="text-[10px] text-slate-500 hover:text-white transition-colors">
            Benchmark →
          </a>
          <a href="http://localhost:8000/docs" target="_blank" className="text-[10px] text-slate-500 hover:text-white transition-colors">
            API Docs →
          </a>
        </div>
      </aside>

      {/* 3D VIEWER */}
      <div className="flex-1 relative border-r border-white/10 min-w-0">
        <IfcViewer ifcUrl={ifcUrl} highlightGuids={highlightGuids} />

        {/* Viewer overlay — model name + element count */}
        <div className="absolute top-3 left-3 flex items-center gap-2 pointer-events-none">
          <span className="text-[10px] font-mono text-slate-500 bg-black/60 px-2 py-1 rounded">
            {selectedFile.replace(".ifc", "")}
          </span>
          {highlightGuids.length > 0 && (
            <span className="text-[10px] font-mono text-orange-400 bg-black/60 px-2 py-1 rounded animate-pulse">
              {highlightGuids.length} selected
            </span>
          )}
        </div>
      </div>

      {/* CHAT PANEL */}
      <div className="w-[420px] shrink-0 flex flex-col bg-[#0a0a0a]">

        {/* Pipeline stepper */}
        <div className="px-5 py-3 border-b border-white/5 flex items-center gap-1.5 flex-wrap">
          {PIPELINE_NODES.map((id, i) => {
            const status = nodeStates[id];
            let dot = "bg-slate-700";
            if (status === "active")  dot = "bg-blue-500 shadow-[0_0_6px_rgba(59,130,246,0.6)]";
            if (status === "done" || status === "hit") dot = "bg-emerald-500";
            if (status === "heal")    dot = "bg-amber-500 animate-pulse";
            if (status === "skipped") dot = "bg-slate-800";
            return (
              <div key={id} className={`flex items-center gap-1 ${status === "idle" ? "opacity-25" : ""}`}>
                <div className={`w-1.5 h-1.5 rounded-full ${dot} transition-all duration-300`} />
                <span className="text-[9px] uppercase tracking-wider text-slate-500 font-semibold">
                  {NODE_LABELS[id]}
                </span>
                {i < PIPELINE_NODES.length - 1 && <div className="w-3 h-px bg-white/10 ml-1" />}
              </div>
            );
          })}
        </div>

        {/* Chat / logs scroll area */}
        <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-3">
          {/* Streaming answer */}
          {streamingAnswer && !result && (
            <div className="border border-white/10 rounded-xl p-4 bg-white/[0.02]">
              <p className="text-[9px] font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse inline-block" />
                Generating
              </p>
              <p className="text-slate-200 text-sm leading-relaxed whitespace-pre-wrap">
                {streamingAnswer}
                <span className="inline-block w-0.5 h-3.5 bg-slate-400 animate-pulse ml-0.5 align-middle" />
              </p>
            </div>
          )}

          {/* Final answer */}
          {result && (
            <div className="flex flex-col gap-3 animate-in slide-in-from-bottom-2 duration-300">
              <div className="flex flex-wrap gap-1.5">
                {result.cache_hit   && <Badge color="emerald">CACHED</Badge>}
                {result.self_healed && <Badge color="amber">SELF-HEALED</Badge>}
                <Badge color="slate">{result.retrieval_source.toUpperCase()}</Badge>
                {result.extracted_guids?.length > 0 && (
                  <Badge color="orange">{result.extracted_guids.length} ELEMENTS</Badge>
                )}
                <Badge color="slate">{result.latency_ms}MS</Badge>
              </div>

              <div className="text-slate-200 text-sm leading-relaxed whitespace-pre-wrap border border-white/10 rounded-xl p-4 bg-white/[0.02]">
                {result.answer}
              </div>

              {result.extracted_guids?.length > 0 && (
                <button
                  className="text-xs text-orange-400 border border-orange-500/30 rounded-lg px-3 py-1.5 hover:bg-orange-500/10 transition-colors self-start"
                  onClick={() => setHighlightGuids(
                    highlightGuids.length ? [] : (result.extracted_guids ?? [])
                  )}
                >
                  {highlightGuids.length ? "Clear 3D highlight" : "Show in 3D viewer ↗"}
                </button>
              )}

              {/* Node timings */}
              {Object.keys(result.node_timings ?? {}).length > 0 && (
                <div className="border border-white/5 rounded-lg p-3">
                  <p className="text-[9px] font-semibold text-slate-500 uppercase tracking-wider mb-2">Timings</p>
                  {Object.entries(result.node_timings).map(([n, s]) => (
                    <div key={n} className="flex justify-between text-[10px] font-mono">
                      <span className="text-slate-500">{n}</span>
                      <span className="text-slate-400">{(s * 1000).toFixed(0)}ms</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Execution log */}
          {logs.length > 0 && (
            <div className="flex flex-col gap-1 border-t border-white/5 pt-3 mt-1">
              {logs.map(l => (
                <div key={l.id} className={`text-[10px] font-mono ${
                  l.type === "success" ? "text-emerald-500/70" :
                  l.type === "heal"    ? "text-amber-500/70"   :
                  l.type === "warn"    ? "text-red-400/70"     : "text-slate-600"
                }`}>
                  <span className="opacity-40 mr-1.5">[+]</span>{l.text}
                </div>
              ))}
              <div ref={chatEnd} />
            </div>
          )}
        </div>

        {/* Query input */}
        <div className="p-4 border-t border-white/10">
          <div className="bg-[#111] border border-white/10 rounded-xl overflow-hidden focus-within:border-white/20 transition-colors">
            <textarea
              className="w-full bg-transparent text-white text-sm p-4 outline-none resize-none"
              placeholder="Query the BIM model…"
              rows={2}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runQuery(); } }}
            />
            <div className="flex items-center justify-between px-4 py-2.5 border-t border-white/5 bg-[#0a0a0a]">
              <span className="text-[10px] text-slate-600">Enter to run · Shift+Enter newline</span>
              <button
                className="bg-white text-black px-3 py-1 rounded-md text-xs font-semibold hover:bg-slate-200 transition-colors disabled:opacity-40"
                onClick={runQuery}
                disabled={running || !query.trim()}
              >
                {running
                  ? <span className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 border-2 border-black/20 border-t-black rounded-full animate-spin" />
                      Running
                    </span>
                  : "Run"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Badge({ children, color }: { children: React.ReactNode; color: string }) {
  const map: Record<string, string> = {
    emerald: "bg-emerald-500/10 border-emerald-500/20 text-emerald-400",
    amber:   "bg-amber-500/10 border-amber-500/20 text-amber-400",
    orange:  "bg-orange-500/10 border-orange-500/20 text-orange-400",
    slate:   "bg-white/5 border-white/10 text-slate-400",
  };
  return (
    <span className={`px-2 py-0.5 border rounded text-[9px] font-mono tracking-wide ${map[color] ?? map.slate}`}>
      {children}
    </span>
  );
}
