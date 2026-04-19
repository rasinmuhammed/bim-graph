"use client";
import { useEffect, useState } from "react";

interface CategoryStats {
  count:       number;
  avg_f1:      number;
  avg_recall:  number;
  self_healed: number;
}

interface BenchmarkSummary {
  total_queries:      number;
  scored_queries:     number;
  avg_f1:             number;
  avg_precision:      number;
  avg_recall:         number;
  avg_latency_ms:     number;
  avg_context_tokens: number;
  self_heal_rate:     number;
  graph_hit_rate:     number;
  by_category:        Record<string, CategoryStats>;
  results:            QueryResult[];
}

interface QueryResult {
  query_id?:           string;
  category?:           string;
  query:               string;
  retrieval_source:    string;
  scoring_method?:     string;
  f1:                  number;
  precision:           number;
  recall:              number;
  latency_ms:          number;
  context_token_count: number;
  self_healed:         boolean;
  error?:              string;
}

const API = "http://localhost:8000";

const CATEGORY_COLORS: Record<string, string> = {
  architectural: "text-blue-400   bg-blue-500/10   border-blue-500/20",
  inventory:     "text-purple-400 bg-purple-500/10 border-purple-500/20",
  mep:           "text-cyan-400   bg-cyan-500/10   border-cyan-500/20",
  cross_floor:   "text-amber-400  bg-amber-500/10  border-amber-500/20",
  adversarial:   "text-red-400    bg-red-500/10    border-red-500/20",
};

const SOURCE_COLORS: Record<string, string> = {
  graph: "bg-purple-500/20 text-purple-400",
  ast:   "bg-amber-500/20  text-amber-400",
  dense: "bg-slate-500/20  text-slate-400",
};

function F1Badge({ f1 }: { f1: number }) {
  const color = f1 >= 0.8 ? "text-emerald-400" : f1 >= 0.5 ? "text-amber-400" : "text-red-400";
  return <span className={`font-mono ${color}`}>{f1.toFixed(3)}</span>;
}

export default function BenchmarkPage() {
  const [data, setData] = useState<BenchmarkSummary | null>(null);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    fetch(`${API}/benchmark`)
      .then((r) => r.json())
      .then(setData)
      .catch(console.error);
  }, []);

  if (!data) return (
    <div className="flex h-screen items-center justify-center bg-black text-slate-400 text-sm flex-col gap-3">
      <p>No benchmark results found.</p>
      <code className="text-white bg-white/5 px-3 py-1.5 rounded text-xs font-mono">
        python -m benchmark.run_benchmark
      </code>
    </div>
  );

  const overallCards = [
    { label: "Avg F1",          value: data.avg_f1.toFixed(3),                          color: "text-emerald-400" },
    { label: "Avg Precision",   value: data.avg_precision.toFixed(3),                   color: "text-blue-400"    },
    { label: "Avg Recall",      value: data.avg_recall.toFixed(3),                      color: "text-cyan-400"    },
    { label: "Avg Latency",     value: `${data.avg_latency_ms}ms`,                      color: "text-slate-300"   },
    { label: "Graph Hit Rate",  value: `${(data.graph_hit_rate  * 100).toFixed(0)}%`,   color: "text-purple-400"  },
    { label: "Self-Heal Rate",  value: `${(data.self_heal_rate  * 100).toFixed(0)}%`,   color: "text-amber-400"   },
    { label: "Queries Scored",  value: `${data.scored_queries}/${data.total_queries}`,  color: "text-slate-400"   },
    { label: "Avg Ctx Tokens",  value: `${data.avg_context_tokens}`,                    color: "text-slate-400"   },
  ];

  const visible = data.results.filter(
    (r) => filter === "all" || r.category === filter
  );

  return (
    <div className="min-h-screen bg-black text-slate-200 p-8 max-w-7xl mx-auto">

      {/* Header */}
      <div className="mb-8">
        <a href="/" className="text-xs text-slate-500 hover:text-white transition-colors mb-4 inline-block">
          ← Back to pipeline
        </a>
        <h1 className="text-xl font-semibold">Benchmark Results</h1>
        <p className="text-xs text-slate-500 mt-1">
          {data.total_queries} queries · GUID-level P/R/F1 vs IFC oracle · cross-floor: coverage scoring
        </p>
      </div>

      {/* Overall stat cards */}
      <div className="grid grid-cols-4 gap-3 mb-10">
        {overallCards.map((c) => (
          <div key={c.label} className="bg-white/5 border border-white/10 rounded-xl p-5">
            <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">{c.label}</p>
            <p className={`text-2xl font-bold font-mono ${c.color}`}>{c.value}</p>
          </div>
        ))}
      </div>

      {/* Category breakdown */}
      {data.by_category && Object.keys(data.by_category).length > 0 && (
        <div className="mb-10">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Performance by Category
          </h2>
          <div className="grid grid-cols-5 gap-3">
            {Object.entries(data.by_category).map(([cat, stats]) => {
              const colorClass = CATEGORY_COLORS[cat] ?? "text-slate-400 bg-white/5 border-white/10";
              const [textC] = colorClass.split(" ");
              return (
                <button
                  key={cat}
                  onClick={() => setFilter(filter === cat ? "all" : cat)}
                  className={`border rounded-xl p-4 text-left transition-all ${colorClass} ${
                    filter === cat ? "ring-1 ring-current" : "hover:opacity-80"
                  }`}
                >
                  <p className={`text-[10px] uppercase font-semibold tracking-wider mb-2 ${textC}`}>{cat}</p>
                  <p className={`text-xl font-bold font-mono ${textC}`}>{stats.avg_f1.toFixed(3)}</p>
                  <p className="text-[10px] text-current opacity-60 mt-1">
                    {stats.count} queries · {stats.self_healed} healed
                  </p>
                </button>
              );
            })}
          </div>
          {filter !== "all" && (
            <button
              onClick={() => setFilter("all")}
              className="mt-3 text-xs text-slate-500 hover:text-white transition-colors"
            >
              Clear filter ×
            </button>
          )}
        </div>
      )}

      {/* Per-query table */}
      <div>
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
          Per-Query Results {filter !== "all" && <span className="text-slate-600 normal-case">— {filter}</span>}
        </h2>
        <table className="w-full text-xs font-mono border-collapse">
          <thead>
            <tr className="text-slate-500 border-b border-white/10 text-left">
              <th className="pb-2 pr-3 font-medium">ID</th>
              <th className="pb-2 pr-3 font-medium">Category</th>
              <th className="pb-2 pr-4 font-medium">Query</th>
              <th className="pb-2 pr-3 font-medium">Source</th>
              <th className="pb-2 pr-3 font-medium">F1</th>
              <th className="pb-2 pr-3 font-medium">P</th>
              <th className="pb-2 pr-3 font-medium">R</th>
              <th className="pb-2 pr-3 font-medium">Latency</th>
              <th className="pb-2 font-medium">Method</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r, i) => {
              if (r.error) return (
                <tr key={i} className="border-b border-white/5">
                  <td className="py-2 pr-3 text-slate-600">{r.query_id ?? "?"}</td>
                  <td colSpan={7} className="py-2 text-red-400/70">{r.query.slice(0, 60)} — ERROR</td>
                </tr>
              );
              const catColor = CATEGORY_COLORS[r.category ?? ""] ?? "";
              const [catText] = catColor.split(" ");
              return (
                <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                  <td className="py-2 pr-3 text-slate-500">{r.query_id ?? i}</td>
                  <td className="py-2 pr-3">
                    <span className={`text-[10px] ${catText}`}>{r.category ?? "—"}</span>
                  </td>
                  <td className="py-2 pr-4 text-slate-300 max-w-xs truncate">{r.query}</td>
                  <td className="py-2 pr-3">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] ${SOURCE_COLORS[r.retrieval_source] ?? "bg-slate-500/20 text-slate-400"}`}>
                      {r.retrieval_source || "—"}
                    </span>
                  </td>
                  <td className="py-2 pr-3"><F1Badge f1={r.f1} /></td>
                  <td className="py-2 pr-3 text-slate-500">{r.precision?.toFixed(2) ?? "—"}</td>
                  <td className="py-2 pr-3 text-slate-500">{r.recall?.toFixed(2) ?? "—"}</td>
                  <td className="py-2 pr-3 text-slate-500">{r.latency_ms}ms</td>
                  <td className="py-2 text-slate-600 text-[10px]">{r.scoring_method ?? "guid"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
