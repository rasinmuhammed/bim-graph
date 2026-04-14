import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BIM-Graph — Agentic Spatial RAG",
  description:
    "Self-healing agentic RAG pipeline that solves Spatial Blindness in BIM / Digital Twin data using deterministic IFC AST traversal.",
  keywords: ["BIM", "RAG", "LangGraph", "IFC", "Digital Twin", "Agentic AI"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
