import React, { useState, useRef, useEffect, useContext } from "react";
import axios from "axios";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Table, TableHead, TableRow, TableHeader, TableBody, TableCell } from "@/components/ui/table";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Loader2,
  Send,
  Database,
  Sparkles,
  AlertTriangle,
  Bot,
  User,
  Copy,
  Check,
  BarChart3,
  Lightbulb,
  TrendingUp,
} from "lucide-react";
import {
  AppShell,
  EmptyState,
  LoadingSkeleton,
  SectionLabel,
} from "@/components/layout/AppShell";
import { OverviewPanel } from "@/components/OverviewPanel";
import { API_BASE_URL } from "@/lib/config";

const POPULAR_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"];

const SUGGESTED_PROMPTS = [
  "What are the key risks for this stock?",
  "Summarize recent financial performance.",
  "Compare growth vs value characteristics.",
  "What portfolio allocation would you suggest?",
];

const Tabs: React.FC<{
  defaultValue: string;
  className?: string;
  children: React.ReactNode;
}> = ({ defaultValue, className, children }) => {
  const [value, setValue] = useState(defaultValue);
  return (
    <TabsContext.Provider value={{ value, setValue }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  );
};

const TabsContext = React.createContext<{
  value: string;
  setValue: React.Dispatch<React.SetStateAction<string>>;
}>({ value: "", setValue: () => {} });

const TabsList: React.FC<{ className?: string; children: React.ReactNode }> = ({
  className,
  children,
}) => (
  <div
    className={`inline-flex h-11 items-center justify-center rounded-xl bg-white/5 p-1 text-slate-400 ring-1 ring-white/10 ${className || ""}`}
  >
    {children}
  </div>
);

const TabsTrigger: React.FC<{
  value: string;
  className?: string;
  children: React.ReactNode;
}> = ({ value, className, children }) => {
  const { value: activeTab, setValue } = useContext(TabsContext);
  return (
    <button
      className={`inline-flex items-center justify-center whitespace-nowrap rounded-lg px-4 py-2 text-sm font-medium transition-all ${
        activeTab === value
          ? "bg-emerald-500 text-white shadow-md shadow-emerald-500/20"
          : "text-slate-400 hover:text-white"
      } ${className || ""}`}
      onClick={() => setValue(value)}
    >
      {children}
    </button>
  );
};

const TabsContent: React.FC<{
  value: string;
  className?: string;
  children: React.ReactNode;
}> = ({ value, className, children }) => {
  const { value: activeTab } = useContext(TabsContext);
  if (activeTab !== value) return null;
  return <div className={`mt-4 ${className || ""}`}>{children}</div>;
};

const PanelCard: React.FC<{
  title: string;
  description?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}> = ({ title, description, action, children }) => (
  <Card className="border-white/10 bg-slate-900/60 shadow-2xl shadow-black/20 backdrop-blur-sm">
    <CardHeader className="flex flex-row items-start justify-between border-b border-white/5 pb-4">
      <div>
        <CardTitle className="flex items-center gap-2 text-xl text-white">
          <Sparkles className="h-5 w-5 text-emerald-400" />
          {title}
        </CardTitle>
        {description && <p className="mt-1 text-sm text-slate-400">{description}</p>}
      </div>
      {action}
    </CardHeader>
    <CardContent className="pt-6">{children}</CardContent>
  </Card>
);

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface IngestionStatus {
  loading: boolean;
  success: boolean | null;
}

interface StockInfo {
  name: string;
  symbol: string;
  sector: string;
  industry: string;
  market_cap: number;
  price: number;
  currency: string;
}

interface FundamentalAnalysis {
  metrics: Record<string, number | string | null>;
  analysis: string;
}

interface TechnicalAnalysis {
  indicators: Record<string, number | string | null>;
  analysis: string;
}

interface RAGInsights {
  answer: string;
  sources: Array<{
    source: string;
    ticker?: string;
    publish_date?: number;
    [key: string]: unknown;
  }>;
}

interface ComprehensiveAnalysisData {
  stock_info: StockInfo;
  fundamental_analysis: FundamentalAnalysis;
  technical_analysis: TechnicalAnalysis;
  rag_insights: RAGInsights;
  error?: string;
}

interface PortfolioStock {
  ticker: string;
  allocation: number;
}

const extractErrorMessage = (error: unknown, fallback: string): string => {
  if (axios.isAxiosError(error)) {
    const data = error.response?.data as { error?: string; detail?: string } | undefined;
    return data?.error || data?.detail || fallback;
  }
  return fallback;
};

const formatTime = (date: Date) =>
  date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

const QuickTickerChips: React.FC<{
  selected: string;
  onSelect: (ticker: string) => void;
}> = ({ selected, onSelect }) => (
  <div className="flex flex-wrap gap-2">
    {POPULAR_TICKERS.map((symbol) => (
      <button
        key={symbol}
        onClick={() => onSelect(symbol)}
        className={`rounded-full px-3 py-1 text-xs font-semibold transition-all ${
          selected === symbol
            ? "bg-emerald-500 text-white"
            : "bg-white/5 text-slate-400 ring-1 ring-white/10 hover:text-white"
        }`}
      >
        {symbol}
      </button>
    ))}
  </div>
);

const CopyButton: React.FC<{ text: string }> = ({ text }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={handleCopy}
      className="gap-2 border-white/10 bg-transparent text-slate-300 hover:bg-white/10"
    >
      {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? "Copied" : "Copy Report"}
    </Button>
  );
};

interface StockAgentChatProps {
  ticker: string;
  onTickerChange: (ticker: string) => void;
}

const StockAgentChat: React.FC<StockAgentChatProps> = ({ ticker, onTickerChange }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Welcome to StockIntel. Ingest a ticker to load market data into the vector store, then ask me about fundamentals, technicals, or portfolio strategy.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatMode, setChatMode] = useState<"agent" | "rag">("agent");
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus>({
    loading: false,
    success: null,
  });
  const messageEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const appendMessage = (role: Message["role"], content: string | undefined | null) => {
    setMessages((prev) => [
      ...prev,
      { role, content: content ?? "No response received from the server.", timestamp: new Date() },
    ]);
  };

  const handleSendMessage = async (queryText?: string) => {
    const text = (queryText ?? input).trim();
    if (!text) return;
    appendMessage("user", text);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/rag/agent_query`, { query: text });
      if (response.data?.error) {
        appendMessage("assistant", response.data.error);
      } else {
        appendMessage("assistant", response.data?.result ?? "No response received from the agent.");
      }
    } catch (error) {
      appendMessage(
        "assistant",
        extractErrorMessage(error, "Sorry, I encountered an error. Please try again.")
      );
    } finally {
      setLoading(false);
    }
  };

  const handleRAGQuery = async (queryText?: string) => {
    const text = (queryText ?? input).trim();
    if (!text) return;
    appendMessage("user", text);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/rag/query`, {
        question: text,
        ticker: ticker.trim() || null,
      });

      const sources = (response.data?.sources ?? [])
        .map((source: { source: string; ticker?: string }) =>
          `[${source.source}${source.ticker ? ` - ${source.ticker}` : ""}]`
        )
        .join(", ");

      const answer = response.data?.answer ?? response.data?.error ?? "No answer returned.";
      appendMessage("assistant", sources ? `${answer}\n\nSources: ${sources}` : answer);
    } catch (error) {
      appendMessage(
        "assistant",
        extractErrorMessage(error, "Sorry, I encountered an error. Please try again.")
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = () => {
    if (chatMode === "agent") handleSendMessage();
    else handleRAGQuery();
  };

  const handleIngestData = async () => {
    if (!ticker.trim()) return;
    setIngestionStatus({ loading: true, success: null });

    try {
      const response = await axios.post(`${API_BASE_URL}/rag/ingest_stock_data`, {
        ticker: ticker.trim(),
      });

      if (response.data.success === false) {
        throw new Error(response.data.error || "Ingestion failed");
      }

      setIngestionStatus({ loading: false, success: true });
      appendMessage(
        "assistant",
        `Data pipeline complete for ${ticker.trim()}. Vector embeddings are ready — you can now run RAG queries with document-grounded answers.`
      );
    } catch (error) {
      setIngestionStatus({ loading: false, success: false });
      appendMessage(
        "assistant",
        extractErrorMessage(
          error,
          `Ingestion failed for ${ticker.trim()}. Yahoo Finance may be rate-limiting — wait a few minutes and retry.`
        )
      );
    }
  };

  return (
    <PanelCard
      title="Intelligent Research Assistant"
      description="Document-grounded conversations powered by retrieval-augmented generation."
    >
      <SectionLabel>Select Symbol & Ingest Data</SectionLabel>
      <div className="mb-3">
        <QuickTickerChips selected={ticker} onSelect={onTickerChange} />
      </div>

      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center">
        <Input
          placeholder="Ticker symbol"
          value={ticker}
          onChange={(e) => onTickerChange(e.target.value.toUpperCase())}
          className="max-w-xs border-white/10 bg-white/5 text-white placeholder:text-slate-500"
        />
        <Button
          onClick={handleIngestData}
          disabled={ingestionStatus.loading || !ticker.trim()}
          className="gap-2 bg-emerald-600 hover:bg-emerald-500"
        >
          {ingestionStatus.loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Ingesting
            </>
          ) : (
            <>
              <Database className="h-4 w-4" /> Ingest into Vector DB
            </>
          )}
        </Button>
        {ingestionStatus.success !== null && (
          <Badge
            className={
              ingestionStatus.success
                ? "bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/20"
                : ""
            }
            variant={ingestionStatus.success ? "default" : "destructive"}
          >
            {ingestionStatus.success ? `${ticker} indexed` : "Ingestion failed"}
          </Badge>
        )}
      </div>

      <Tabs defaultValue="agent" className="mb-4">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="agent">Agent Chat</TabsTrigger>
          <TabsTrigger value="rag">RAG Query</TabsTrigger>
        </TabsList>

        <TabsContent value="agent">
          <div
            onFocus={() => setChatMode("agent")}
            className="space-y-3"
          >
            <p className="text-sm text-slate-400">
              Open-ended reasoning with Groq LLM and retrieved financial context.
            </p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTED_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => handleSendMessage(prompt)}
                  disabled={loading}
                  className="rounded-full bg-white/5 px-3 py-1.5 text-xs text-slate-300 ring-1 ring-white/10 transition hover:bg-emerald-500/10 hover:text-emerald-300"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="rag">
          <div onFocus={() => setChatMode("rag")}>
            <p className="text-sm text-slate-400">
              Precise answers grounded strictly in ingested vector store documents.
            </p>
          </div>
        </TabsContent>
      </Tabs>

      <div className="mb-4 flex gap-2">
        <Input
          placeholder={
            chatMode === "agent"
              ? "Ask about valuation, risks, or strategy..."
              : "Search ingested documents..."
          }
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          className="flex-1 border-white/10 bg-white/5 text-white placeholder:text-slate-500"
        />
        <Button
          onClick={handleSubmit}
          disabled={loading || !input.trim()}
          className="bg-emerald-600 hover:bg-emerald-500"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </Button>
      </div>

      <div className="h-[26rem] overflow-y-auto rounded-2xl border border-white/10 bg-slate-950/50 p-4 scrollbar-thin">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`mb-5 flex gap-3 ${message.role === "user" ? "flex-row-reverse" : "flex-row"}`}
          >
            <div
              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full ${
                message.role === "user"
                  ? "bg-emerald-500/20 text-emerald-400"
                  : "bg-white/10 text-slate-300"
              }`}
            >
              {message.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
            </div>
            <div
              className={`max-w-[85%] ${message.role === "user" ? "items-end" : "items-start"} flex flex-col`}
            >
              <div
                className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  message.role === "user"
                    ? "bg-emerald-600 text-white"
                    : "bg-white/5 text-slate-200 ring-1 ring-white/10"
                }`}
              >
                {(message.content ?? "").split("\n").map((line, i, lines) => (
                  <React.Fragment key={i}>
                    {line}
                    {i < lines.length - 1 && <br />}
                  </React.Fragment>
                ))}
              </div>
              <span className="mt-1 px-1 text-[10px] text-slate-500">
                {formatTime(message.timestamp)}
              </span>
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <Loader2 className="h-4 w-4 animate-spin text-emerald-400" />
            Generating response...
          </div>
        )}
        <div ref={messageEndRef} />
      </div>
    </PanelCard>
  );
};

const MetricCard: React.FC<{ label: string; value: string; accent?: string }> = ({
  label,
  value,
  accent,
}) => (
  <div className="rounded-2xl border border-white/10 bg-gradient-to-br from-white/5 to-transparent p-4">
    <p className="text-xs font-medium uppercase tracking-wider text-slate-500">{label}</p>
    <p className={`mt-1 text-lg font-semibold ${accent || "text-white"}`}>{value}</p>
  </div>
);

const getRsiSignal = (rsi: number | string | null | undefined): { label: string; color: string } => {
  const val = typeof rsi === "number" ? rsi : parseFloat(String(rsi));
  if (isNaN(val)) return { label: "N/A", color: "text-slate-400" };
  if (val >= 70) return { label: "Overbought", color: "text-red-400" };
  if (val <= 30) return { label: "Oversold", color: "text-emerald-400" };
  return { label: "Neutral", color: "text-amber-400" };
};

interface ComprehensiveAnalysisProps {
  ticker: string;
  onTickerChange: (ticker: string) => void;
}

const ComprehensiveAnalysis: React.FC<ComprehensiveAnalysisProps> = ({
  ticker,
  onTickerChange,
}) => {
  const [analysis, setAnalysis] = useState<ComprehensiveAnalysisData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalysis = async () => {
    if (!ticker.trim()) return;
    setLoading(true);
    setError(null);
    setAnalysis(null);

    try {
      const response = await axios.get<ComprehensiveAnalysisData>(
        `${API_BASE_URL}/rag/comprehensive_analysis/${ticker.trim()}`
      );

      if (response.data.error) {
        setError(response.data.error);
        return;
      }

      if (!response.data.stock_info || !response.data.fundamental_analysis || !response.data.technical_analysis) {
        setError("Incomplete analysis data received from the server.");
        return;
      }

      setAnalysis(response.data);
    } catch (error) {
      setError(
        extractErrorMessage(
          error,
          "Failed to fetch analysis. Verify the ticker or wait if Yahoo Finance is rate-limiting."
        )
      );
    } finally {
      setLoading(false);
    }
  };

  const reportText = analysis
    ? [
        `# ${analysis.stock_info.name} (${analysis.stock_info.symbol})`,
        `Price: ${analysis.stock_info.price}`,
        `Sector: ${analysis.stock_info.sector}`,
        "",
        "## Fundamental Analysis",
        analysis.fundamental_analysis.analysis,
        "",
        "## Technical Analysis",
        analysis.technical_analysis.analysis,
        "",
        "## RAG Insights",
        analysis.rag_insights?.answer ?? "N/A",
      ].join("\n")
    : "";

  const rsi = analysis?.technical_analysis.indicators?.RSI;
  const rsiSignal = getRsiSignal(rsi);

  return (
    <PanelCard
      title="Comprehensive Equity Report"
      description="Unified fundamental, technical, and AI-generated investment thesis."
      action={
        analysis ? <CopyButton text={reportText} /> : undefined
      }
    >
      <SectionLabel>Target Symbol</SectionLabel>
      <div className="mb-3">
        <QuickTickerChips selected={ticker} onSelect={onTickerChange} />
      </div>

      <div className="mb-6 flex flex-col gap-3 sm:flex-row">
        <Input
          placeholder="Enter ticker"
          value={ticker}
          onChange={(e) => onTickerChange(e.target.value.toUpperCase())}
          className="max-w-xs border-white/10 bg-white/5 text-white placeholder:text-slate-500"
        />
        <Button
          onClick={fetchAnalysis}
          disabled={loading || !ticker.trim()}
          className="gap-2 bg-emerald-600 hover:bg-emerald-500"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Generating Report
            </>
          ) : (
            <>
              <BarChart3 className="h-4 w-4" /> Generate Report
            </>
          )}
        </Button>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-4 border-red-500/30 bg-red-500/10">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {loading && <LoadingSkeleton />}

      {!loading && !analysis && !error && (
        <EmptyState
          icon={<BarChart3 className="h-8 w-8" />}
          title="No report generated yet"
          description="Select a ticker and run analysis to view fundamental metrics, technical indicators, and AI-powered insights."
        />
      )}

      {analysis && !loading && (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="text-2xl font-bold text-white">
              {analysis.stock_info.name}
            </h3>
            <Badge className="bg-emerald-500/20 text-emerald-300">{analysis.stock_info.symbol}</Badge>
            {analysis.stock_info.sector && (
              <Badge variant="outline" className="border-white/10 text-slate-300">
                {analysis.stock_info.sector}
              </Badge>
            )}
            <Badge variant="outline" className={`border-white/10 ${rsiSignal.color}`}>
              RSI: {rsiSignal.label}
            </Badge>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard label="Company" value={analysis.stock_info.name || "N/A"} />
            <MetricCard label="Symbol" value={analysis.stock_info.symbol || "N/A"} />
            <MetricCard
              label="Price"
              value={
                analysis.stock_info.price != null
                  ? `${analysis.stock_info.price.toFixed(2)} ${analysis.stock_info.currency || ""}`
                  : "N/A"
              }
              accent="text-emerald-400"
            />
            <MetricCard
              label="Market Cap"
              value={
                analysis.stock_info.market_cap
                  ? `$${(analysis.stock_info.market_cap / 1e9).toFixed(2)}B`
                  : "N/A"
              }
            />
          </div>

          <Tabs defaultValue="fundamental">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="fundamental">Fundamental</TabsTrigger>
              <TabsTrigger value="technical">Technical</TabsTrigger>
              <TabsTrigger value="rag">AI Insights</TabsTrigger>
            </TabsList>

            <TabsContent value="fundamental">
              <Card className="border-white/10 bg-white/5">
                <CardContent className="pt-6">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableHeader>Metric</TableHeader>
                        <TableHeader>Value</TableHeader>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(analysis.fundamental_analysis?.metrics ?? {}).map(([key, value]) => (
                        <TableRow key={key}>
                          <TableCell className="text-slate-300">{key}</TableCell>
                          <TableCell className="font-medium text-white">
                            {value !== null
                              ? typeof value === "number"
                                ? value.toFixed(2)
                                : String(value)
                              : "N/A"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  <div className="mt-6 rounded-2xl border border-white/5 bg-slate-950/50 p-5">
                    <div className="mb-2 flex items-center gap-2 text-sm font-medium text-white">
                      <Lightbulb className="h-4 w-4 text-amber-400" />
                      Analyst Summary
                    </div>
                    <p className="text-sm leading-relaxed text-slate-300">
                      {analysis.fundamental_analysis?.analysis ?? "No fundamental analysis available."}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="technical">
              <Card className="border-white/10 bg-white/5">
                <CardContent className="pt-6">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableHeader>Indicator</TableHeader>
                        <TableHeader>Value</TableHeader>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(analysis.technical_analysis?.indicators ?? {}).map(([key, value]) => (
                        <TableRow key={key}>
                          <TableCell className="text-slate-300">{key}</TableCell>
                          <TableCell className="font-medium text-white">
                            {value !== null
                              ? typeof value === "number"
                                ? value.toFixed(2)
                                : String(value)
                              : "N/A"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  <div className="mt-6 rounded-2xl border border-white/5 bg-slate-950/50 p-5">
                    <div className="mb-2 flex items-center gap-2 text-sm font-medium text-white">
                      <Lightbulb className="h-4 w-4 text-amber-400" />
                      Technical Outlook
                    </div>
                    <p className="text-sm leading-relaxed text-slate-300">
                      {analysis.technical_analysis?.analysis ?? "No technical analysis available."}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="rag">
              <Card className="border-white/10 bg-white/5">
                <CardContent className="pt-6">
                  <p className="whitespace-pre-line text-sm leading-relaxed text-slate-300">
                    {analysis.rag_insights?.answer ?? "No RAG insights available."}
                  </p>
                  <h4 className="mt-6 text-sm font-semibold text-white">Document Sources</h4>
                  <ul className="mt-3 space-y-2">
                    {(analysis.rag_insights?.sources ?? []).map((source, index) => (
                      <li
                        key={index}
                        className="rounded-lg bg-slate-950/50 px-3 py-2 text-sm text-slate-400"
                      >
                        {source.source}
                        {source.ticker && ` · ${source.ticker}`}
                        {source.publish_date &&
                          ` · ${new Date(source.publish_date * 1000).toLocaleDateString()}`}
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      )}
    </PanelCard>
  );
};

const AllocationBar: React.FC<{ label: string; value: number; color?: string }> = ({
  label,
  value,
  color = "bg-emerald-500",
}) => (
  <div>
    <div className="mb-1.5 flex justify-between text-sm">
      <span className="font-medium text-slate-200">{label}</span>
      <span className="font-semibold text-white">{value.toFixed(1)}%</span>
    </div>
    <div className="h-2.5 overflow-hidden rounded-full bg-white/10">
      <div
        className={`h-full rounded-full transition-all duration-700 ${color}`}
        style={{ width: `${Math.min(value, 100)}%` }}
      />
    </div>
  </div>
);

const PortfolioOptimizerRAG: React.FC = () => {
  const [portfolio, setPortfolio] = useState<PortfolioStock[]>([
    { ticker: "AAPL", allocation: 25 },
    { ticker: "MSFT", allocation: 25 },
    { ticker: "NVDA", allocation: 25 },
    { ticker: "GOOGL", allocation: 25 },
  ]);
  const [riskPreference, setRiskPreference] = useState<string>("medium");
  const [optimizedPortfolio, setOptimizedPortfolio] = useState<Record<string, number> | null>(
    null
  );
  const [analysis, setAnalysis] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const totalAllocation = portfolio.reduce((sum, s) => sum + (s.allocation || 0), 0);
  const allocationValid = Math.abs(totalAllocation - 100) < 0.01;

  const addStock = () => setPortfolio([...portfolio, { ticker: "", allocation: 0 }]);

  const removeStock = (index: number) => {
    const newPortfolio = [...portfolio];
    newPortfolio.splice(index, 1);
    setPortfolio(newPortfolio);
  };

  const updatePortfolio = (index: number, field: string, value: string) => {
    const newPortfolio = [...portfolio];
    if (field === "ticker") {
      newPortfolio[index].ticker = value.toUpperCase();
    } else if (field === "allocation") {
      newPortfolio[index].allocation = parseFloat(value) || 0;
    }
    setPortfolio(newPortfolio);
  };

  const optimizePortfolio = async () => {
    if (portfolio.some((stock) => !stock.ticker.trim())) {
      setError("Please fill in all ticker symbols");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const tickers = portfolio.map((stock) => stock.ticker);
      const allocations = portfolio.reduce<Record<string, number>>((acc, stock) => {
        acc[stock.ticker] = stock.allocation;
        return acc;
      }, {});

      const response = await axios.post(`${API_BASE_URL}/rag/optimize_portfolio`, {
        tickers,
        allocations,
        risk_preference: riskPreference,
      });

      if (response.data?.error) {
        setError(response.data.error);
        setOptimizedPortfolio(null);
        setAnalysis("");
        return;
      }

      setOptimizedPortfolio(response.data.optimized_portfolio ?? null);
      setAnalysis(response.data.analysis ?? "");

      if (!response.data.optimized_portfolio) {
        setError("Optimization completed but no allocation weights were returned.");
      }
    } catch (error) {
      setError(extractErrorMessage(error, "Failed to optimize portfolio. Please try again."));
    } finally {
      setLoading(false);
    }
  };

  const barColors = [
    "bg-emerald-500",
    "bg-cyan-500",
    "bg-violet-500",
    "bg-amber-500",
    "bg-rose-500",
    "bg-blue-500",
  ];

  const riskLabels: Record<string, string> = {
    low: "Conservative — capital preservation focus",
    medium: "Balanced — growth with moderate risk",
    high: "Aggressive — maximum growth exposure",
  };

  return (
    <PanelCard
      title="Portfolio Optimization Engine"
      description="AI-assisted rebalancing with configurable risk tolerance."
    >
      <div className="mb-6 grid gap-4 lg:grid-cols-[1fr_auto]">
        <div className="space-y-3">
          <SectionLabel>Holdings</SectionLabel>
          {portfolio.map((stock, index) => (
            <div key={index} className="flex flex-wrap items-center gap-2">
              <Input
                placeholder="Ticker"
                value={stock.ticker}
                onChange={(e) => updatePortfolio(index, "ticker", e.target.value)}
                className="w-28 border-white/10 bg-white/5 text-white"
              />
              <Input
                type="number"
                placeholder="Weight %"
                value={stock.allocation}
                onChange={(e) => updatePortfolio(index, "allocation", e.target.value)}
                className="w-28 border-white/10 bg-white/5 text-white"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={() => removeStock(index)}
                disabled={portfolio.length <= 1}
                className="border-white/10 bg-transparent text-slate-300 hover:bg-white/10"
              >
                Remove
              </Button>
            </div>
          ))}
          <Button
            variant="outline"
            onClick={addStock}
            className="border-white/10 bg-transparent text-slate-300 hover:bg-white/10"
          >
            + Add Holding
          </Button>
        </div>

        <div className="flex flex-col items-center justify-center rounded-2xl border border-white/10 bg-white/5 p-6">
          <p className="text-xs uppercase tracking-wider text-slate-500">Total Weight</p>
          <p
            className={`mt-1 text-4xl font-bold ${
              allocationValid ? "text-emerald-400" : "text-amber-400"
            }`}
          >
            {totalAllocation.toFixed(0)}%
          </p>
          <p className="mt-2 text-xs text-slate-400">
            {allocationValid ? "Balanced portfolio" : "Does not equal 100%"}
          </p>
        </div>
      </div>

      <div className="mb-6">
        <SectionLabel>Risk Profile</SectionLabel>
        <div className="flex flex-wrap gap-2">
          {(["low", "medium", "high"] as const).map((risk) => (
            <button
              key={risk}
              onClick={() => setRiskPreference(risk)}
              className={`rounded-2xl border px-4 py-3 text-left transition-all ${
                riskPreference === risk
                  ? "border-emerald-500/40 bg-emerald-500/10"
                  : "border-white/10 bg-white/5 hover:border-white/20"
              }`}
            >
              <p className="text-sm font-semibold capitalize text-white">{risk} Risk</p>
              <p className="text-xs text-slate-400">{riskLabels[risk]}</p>
            </button>
          ))}
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-4 border-red-500/30 bg-red-500/10">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Button
        onClick={optimizePortfolio}
        disabled={loading}
        className="w-full gap-2 bg-emerald-600 hover:bg-emerald-500"
      >
        {loading ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" /> Running Optimization
          </>
        ) : (
          "Generate Optimal Allocation"
        )}
      </Button>

      {optimizedPortfolio && (
        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <Card className="border-white/10 bg-white/5">
            <CardHeader className="py-4">
              <CardTitle className="text-lg text-white">Recommended Weights</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              {Object.entries(optimizedPortfolio).map(([ticker, allocation], index) => (
                <AllocationBar
                  key={ticker}
                  label={ticker}
                  value={typeof allocation === "number" ? allocation : parseFloat(String(allocation))}
                  color={barColors[index % barColors.length]}
                />
              ))}
            </CardContent>
          </Card>

          {analysis && (
            <Card className="border-white/10 bg-white/5">
              <CardHeader className="flex flex-row items-center justify-between py-4">
                <CardTitle className="text-lg text-white">Strategy Rationale</CardTitle>
                <CopyButton text={analysis} />
              </CardHeader>
              <CardContent>
                <p className="whitespace-pre-line text-sm leading-relaxed text-slate-300">
                  {analysis}
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {!optimizedPortfolio && !loading && (
        <div className="mt-6">
          <EmptyState
            icon={<TrendingUp className="h-8 w-8" />}
            title="Ready to optimize"
            description="Configure your holdings and risk profile, then generate an AI-assisted allocation strategy."
          />
        </div>
      )}
    </PanelCard>
  );
};

const StockAnalysisRAG: React.FC = () => {
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedTicker, setSelectedTicker] = useState("AAPL");

  return (
    <AppShell
      activeTab={activeTab}
      onTabChange={setActiveTab}
      selectedTicker={selectedTicker}
    >
      {activeTab === "overview" && (
        <OverviewPanel
          selectedTicker={selectedTicker}
          onTickerChange={setSelectedTicker}
          onNavigate={setActiveTab}
        />
      )}
      {activeTab === "agent" && (
        <StockAgentChat ticker={selectedTicker} onTickerChange={setSelectedTicker} />
      )}
      {activeTab === "analysis" && (
        <ComprehensiveAnalysis ticker={selectedTicker} onTickerChange={setSelectedTicker} />
      )}
      {activeTab === "portfolio" && <PortfolioOptimizerRAG />}
    </AppShell>
  );
};

export default StockAnalysisRAG;
