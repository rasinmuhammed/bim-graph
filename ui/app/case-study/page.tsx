import Link from "next/link";

const points = [
  ["Pain", "IFC files know exactly where assets live, but chunked RAG breaks that spatial hierarchy."],
  ["Wedge", "BIM-Graph answers location and asset questions through graph containment first, then visualizes the proof."],
  ["Proof", "Verified answers carry source, resolved floor, storey GUID, element GUIDs, and a concise query summary."],
  ["Expansion", "The same graph can grow into handover QA, asset registers, model audit, and digital twin search."],
];

export default function CaseStudyPage() {
  return (
    <main className="min-h-screen bg-black text-slate-200">
      <section className="mx-auto flex min-h-screen max-w-5xl flex-col justify-center px-6 py-16">
        <Link href="/" className="mb-8 text-xs text-slate-500 transition-colors hover:text-white">
          ← Open demo
        </Link>
        <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-emerald-400">
          Spatial QA Copilot for IFC
        </p>
        <h1 className="max-w-3xl text-4xl font-semibold leading-tight text-white">
          Chat with a building model, with answers grounded in spatial proof.
        </h1>
        <p className="mt-5 max-w-2xl text-sm leading-7 text-slate-400">
          BIM-Graph turns IFC spatial containment into a queryable graph, so floor and asset
          questions are resolved by building structure before language generation. The demo
          highlights answer GUIDs directly in the 3D model.
        </p>

        <div className="mt-12 grid gap-3 md:grid-cols-4">
          {points.map(([title, body]) => (
            <div key={title} className="border border-white/10 bg-white/[0.03] p-4">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                {title}
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-300">{body}</p>
            </div>
          ))}
        </div>

        <div className="mt-10 flex flex-wrap gap-3">
          <Link href="/" className="bg-white px-4 py-2 text-sm font-semibold text-black transition-colors hover:bg-slate-200">
            Try the copilot
          </Link>
          <Link href="/benchmark" className="border border-white/15 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-white/30 hover:text-white">
            View benchmark
          </Link>
        </div>
      </section>
    </main>
  );
}
