import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Layers,
  TrendingUp,
  Zap,
} from "lucide-react";
import { API_BASE_URL } from "@/lib/config";
import { Badge } from "@/components/ui/badge";

interface AppShellProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
  selectedTicker: string;
  children: React.ReactNode;
}

const NAV_ITEMS = [
  {
    id: "overview",
    label: "Overview",
    description: "Platform capabilities & demo guide",
    icon: Layers,
  },
  {
    id: "agent",
    label: "AI Agent",
    description: "Conversational RAG assistant",
    icon: BrainCircuit,
  },
  {
    id: "analysis",
    label: "Analysis",
    description: "Fundamental + technical reports",
    icon: BarChart3,
  },
  {
    id: "portfolio",
    label: "Portfolio",
    description: "Risk-aware allocation engine",
    icon: TrendingUp,
  },
];

const TECH_STACK = ["FastAPI", "Groq LLM", "ChromaDB", "LangChain", "Yahoo Finance"];

export const AppShell: React.FC<AppShellProps> = ({
  activeTab,
  onTabChange,
  selectedTicker,
  children,
}) => {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        await axios.get(`${API_BASE_URL}/health`, { timeout: 5000 });
        setBackendOnline(true);
      } catch {
        setBackendOnline(false);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 30000);
    return () => clearInterval(interval);
  }, []);

  const activeNav = NAV_ITEMS.find((item) => item.id === activeTab);

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950 text-foreground">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:4rem_4rem]" />
      <div className="pointer-events-none absolute -left-32 top-0 h-96 w-96 rounded-full bg-emerald-500/10 blur-3xl" />
      <div className="pointer-events-none absolute -right-32 top-40 h-96 w-96 rounded-full bg-cyan-500/10 blur-3xl" />

      <div className="relative mx-auto flex min-h-screen max-w-[1400px]">
        <aside className="hidden w-72 shrink-0 flex-col border-r border-white/10 bg-slate-950/70 p-6 backdrop-blur-xl lg:flex">
          <div className="mb-8 flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-400 to-cyan-500 shadow-lg shadow-emerald-500/20">
              <Activity className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white">StockIntel</h1>
              <p className="text-xs text-slate-400">RAG Investment Platform</p>
            </div>
          </div>

          <nav className="space-y-2">
            {NAV_ITEMS.map(({ id, label, description, icon: Icon }) => (
              <button
                key={id}
                onClick={() => onTabChange(id)}
                className={`w-full rounded-2xl border px-4 py-3 text-left transition-all ${
                  activeTab === id
                    ? "border-emerald-500/40 bg-emerald-500/10 shadow-lg shadow-emerald-500/10"
                    : "border-transparent bg-white/5 hover:border-white/10 hover:bg-white/10"
                }`}
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`rounded-xl p-2 ${
                      activeTab === id ? "bg-emerald-500 text-white" : "bg-white/10 text-slate-300"
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-white">{label}</p>
                    <p className="text-xs text-slate-400">{description}</p>
                  </div>
                </div>
              </button>
            ))}
          </nav>

          <div className="mt-auto space-y-4 pt-8">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
                Active Symbol
              </p>
              <p className="mt-1 text-2xl font-bold text-emerald-400">
                {selectedTicker || "—"}
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">
                Architecture
              </p>
              <div className="flex flex-wrap gap-2">
                {TECH_STACK.map((tech) => (
                  <span
                    key={tech}
                    className="rounded-full bg-slate-900 px-2.5 py-1 text-[10px] font-medium text-slate-300 ring-1 ring-white/10"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-40 border-b border-white/10 bg-slate-950/80 backdrop-blur-xl">
            <div className="flex items-center justify-between gap-4 px-4 py-4 sm:px-6">
              <div className="lg:hidden">
                <h1 className="text-lg font-bold text-white">StockIntel RAG</h1>
                <p className="text-xs text-slate-400">{activeNav?.description}</p>
              </div>
              <div className="hidden lg:block">
                <p className="text-xs font-medium uppercase tracking-[0.2em] text-emerald-400">
                  {activeNav?.label}
                </p>
                <h2 className="text-xl font-semibold text-white">{activeNav?.description}</h2>
              </div>

              <div className="flex items-center gap-2">
                <Badge
                  variant="outline"
                  className="hidden border-white/10 text-slate-300 sm:inline-flex"
                >
                  <Zap className="mr-1 h-3 w-3 text-amber-400" />
                  Real-time RAG
                </Badge>
                <Badge
                  variant="outline"
                  className={`gap-2 border-white/10 ${
                    backendOnline === true
                      ? "text-emerald-400"
                      : backendOnline === false
                        ? "text-red-400"
                        : "text-slate-400"
                  }`}
                >
                  <span
                    className={`h-2 w-2 rounded-full ${
                      backendOnline === true
                        ? "animate-pulse bg-emerald-400"
                        : backendOnline === false
                          ? "bg-red-400"
                          : "bg-slate-500"
                    }`}
                  />
                  {backendOnline === true
                    ? "Live"
                    : backendOnline === false
                      ? "Offline"
                      : "Connecting"}
                </Badge>
              </div>
            </div>

            <nav className="flex gap-2 overflow-x-auto px-4 pb-4 lg:hidden">
              {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => onTabChange(id)}
                  className={`inline-flex shrink-0 items-center gap-2 rounded-full px-4 py-2 text-sm font-medium ${
                    activeTab === id
                      ? "bg-emerald-500 text-white"
                      : "bg-white/5 text-slate-300"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </button>
              ))}
            </nav>
          </header>

          <main className="flex-1 px-4 py-6 sm:px-6 sm:py-8">{children}</main>
        </div>
      </div>
    </div>
  );
};

export const FeaturePill: React.FC<{
  icon: React.ReactNode;
  label: string;
  value: string;
}> = ({ icon, label, value }) => (
  <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
    <div className="rounded-xl bg-emerald-500/15 p-2 text-emerald-400">{icon}</div>
    <div>
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-sm font-semibold text-white">{value}</p>
    </div>
  </div>
);

export const SectionLabel: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
    {children}
  </p>
);

export const EmptyState: React.FC<{
  icon: React.ReactNode;
  title: string;
  description: string;
}> = ({ icon, title, description }) => (
  <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 bg-white/[0.02] px-6 py-16 text-center">
    <div className="mb-4 rounded-2xl bg-white/5 p-4 text-slate-400">{icon}</div>
    <h3 className="text-lg font-semibold text-white">{title}</h3>
    <p className="mt-2 max-w-md text-sm leading-relaxed text-slate-400">{description}</p>
  </div>
);

export const LoadingSkeleton: React.FC = () => (
  <div className="space-y-4 animate-pulse">
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {[...Array(4)].map((_, i) => (
        <div key={i} className="h-24 rounded-2xl bg-white/5" />
      ))}
    </div>
    <div className="h-64 rounded-2xl bg-white/5" />
    <div className="h-40 rounded-2xl bg-white/5" />
  </div>
);
