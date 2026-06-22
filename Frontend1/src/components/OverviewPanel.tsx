import React from "react";
import {
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  LineChart,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { FeaturePill, SectionLabel } from "@/components/layout/AppShell";

interface OverviewPanelProps {
  selectedTicker: string;
  onTickerChange: (ticker: string) => void;
  onNavigate: (tab: string) => void;
}

const POPULAR_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "AMZN"];

const DEMO_STEPS = [
  {
    step: "01",
    title: "Ingest market data",
    detail: "Load Yahoo Finance fundamentals and price history into the Chroma vector store.",
  },
  {
    step: "02",
    title: "Ask the RAG agent",
    detail: "Query with natural language — answers are grounded in retrieved documents.",
  },
  {
    step: "03",
    title: "Run comprehensive analysis",
    detail: "Generate fundamental, technical, and AI insight reports in one view.",
  },
  {
    step: "04",
    title: "Optimize portfolio",
    detail: "Adjust risk preference and receive AI-assisted allocation recommendations.",
  },
];

const CAPABILITIES = [
  {
    icon: BrainCircuit,
    title: "Retrieval-Augmented Generation",
    body: "Semantic search over ingested financial documents powers accurate, context-aware responses.",
  },
  {
    icon: LineChart,
    title: "Multi-Layer Analysis",
    body: "Combines live market data, technical indicators, and LLM reasoning in a unified report.",
  },
  {
    icon: TrendingUp,
    title: "Portfolio Intelligence",
    body: "Risk-aware weight optimization across multiple holdings with explainable AI output.",
  },
];

export const OverviewPanel: React.FC<OverviewPanelProps> = ({
  selectedTicker,
  onTickerChange,
  onNavigate,
}) => {
  return (
    <div className="space-y-8">
      <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/90 via-slate-900/60 to-emerald-950/40 p-8 sm:p-10">
        <div className="absolute right-0 top-0 h-64 w-64 rounded-full bg-emerald-500/10 blur-3xl" />
        <div className="relative max-w-3xl">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-300">
            <Sparkles className="h-3.5 w-3.5" />
            AI-Powered Investment Research Platform
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Institutional-grade stock intelligence, powered by RAG
          </h2>
          <p className="mt-4 text-base leading-relaxed text-slate-300 sm:text-lg">
            StockIntel combines vector search, large language models, and live market data to
            deliver research workflows that are fast, explainable, and interview-ready.
          </p>

          <div className="mt-8 grid gap-3 sm:grid-cols-3">
            <FeaturePill icon={<BarChart3 className="h-4 w-4" />} label="Data Sources" value="Yahoo Finance" />
            <FeaturePill icon={<BrainCircuit className="h-4 w-4" />} label="LLM Provider" value="Groq API" />
            <FeaturePill icon={<TrendingUp className="h-4 w-4" />} label="Vector Store" value="ChromaDB" />
          </div>
        </div>
      </section>

      <section>
        <SectionLabel>Quick Start — Select a Symbol</SectionLabel>
        <div className="flex flex-wrap gap-2">
          {POPULAR_TICKERS.map((symbol) => (
            <button
              key={symbol}
              onClick={() => onTickerChange(symbol)}
              className={`rounded-full px-4 py-2 text-sm font-semibold transition-all ${
                selectedTicker === symbol
                  ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/25"
                  : "bg-white/5 text-slate-300 ring-1 ring-white/10 hover:bg-white/10 hover:text-white"
              }`}
            >
              {symbol}
            </button>
          ))}
        </div>
        <div className="mt-6 flex flex-wrap gap-3">
          <Button
            onClick={() => onNavigate("agent")}
            className="gap-2 bg-emerald-600 hover:bg-emerald-500"
          >
            Open AI Agent <ArrowRight className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            onClick={() => onNavigate("analysis")}
            className="border-white/10 bg-transparent text-slate-200 hover:bg-white/10"
          >
            Run Analysis on {selectedTicker || "Ticker"}
          </Button>
          <Button
            variant="outline"
            onClick={() => onNavigate("portfolio")}
            className="border-white/10 bg-transparent text-slate-200 hover:bg-white/10"
          >
            Optimize Portfolio
          </Button>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        {CAPABILITIES.map(({ icon: Icon, title, body }) => (
          <Card key={title} className="border-white/10 bg-slate-900/50 backdrop-blur-sm">
            <CardContent className="p-6">
              <div className="mb-4 inline-flex rounded-2xl bg-emerald-500/10 p-3 text-emerald-400">
                <Icon className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-semibold text-white">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{body}</p>
            </CardContent>
          </Card>
        ))}
      </section>

      <section>
        <SectionLabel>Demo Walkthrough</SectionLabel>
        <div className="grid gap-4 md:grid-cols-2">
          {DEMO_STEPS.map(({ step, title, detail }) => (
            <div
              key={step}
              className="flex gap-4 rounded-2xl border border-white/10 bg-white/[0.03] p-5"
            >
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-500/15 text-sm font-bold text-emerald-400">
                {step}
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                  <h4 className="font-semibold text-white">{title}</h4>
                </div>
                <p className="mt-1 text-sm leading-relaxed text-slate-400">{detail}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};
