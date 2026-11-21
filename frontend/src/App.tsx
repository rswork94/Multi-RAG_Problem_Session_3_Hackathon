import { lazy, Suspense, useEffect, useState } from "react";
import { getJSON } from "./api/client";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { PracticeAreaSwitcher } from "./components/PracticeAreaSwitcher";
import { setState, useApp } from "./store/useAppStore";
import type { Health } from "./types";
import T1IngestChat from "./tabs/T1IngestChat";

const T2 = lazy(() => import("./tabs/T2KnowledgeSphere"));
const T3 = lazy(() => import("./tabs/T3WorkflowCanvas"));
const T4 = lazy(() => import("./tabs/T4HyperparamsTelemetry"));
const T5 = lazy(() => import("./tabs/T5Benchmark"));

const TABS = [
  { id: "chat", label: "1 · Ingestion & Multi-RAG Chat" },
  { id: "sphere", label: "2 · Knowledge Sphere" },
  { id: "workflow", label: "3 · LangGraph Workflow" },
  { id: "telemetry", label: "4 · Hyperparameters & KPIs" },
  { id: "bench", label: "5 · Comparative Benchmark" },
];

function tabFromHash(): string {
  const h = window.location.hash.replace("#", "");
  return TABS.some((t) => t.id === h) ? h : "chat";
}

export default function App() {
  const [tab, setTab] = useState(tabFromHash());
  const health = useApp((s) => s.health);
  const graphVersion = useApp((s) => s.graphVersion);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    let alive = true;
    const load = () =>
      getJSON<Health>("/api/health")
        .then((h) => alive && setState({ health: h }))
        .catch(() => alive && setState({ health: null }));
    load();
    const t = setInterval(load, 8000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [graphVersion]);

  const go = (id: string) => {
    window.location.hash = id;
    setTab(id);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <b>Multi-RAG Benchmark</b>
          <span>Standard vs HyDE vs Graph RAG · PwC India advisory corpus (synthetic)</span>
        </div>
        <PracticeAreaSwitcher />
        <div className="spacer" />
        {health ? (
          <>
            <span className={`pill ${health.llm_mode === "online" ? "good" : "warn"}`}
              title={health.llm_mode === "online" ? `${health.synth_model} answers · ${health.utility_model} utility` : "No API key: extractive answers, lexical judge (set a key in Tab 1 → Model settings)"}>
              {health.llm_mode === "online" ? `Claude online · ${health.synth_model.replace("anthropic/", "")}` : "Offline mode"}
            </span>
            <span className="pill" title={`${health.embed_model} · ${health.rerank_model}`}>
              {health.device.toUpperCase()} · {health.warm.status === "ready" ? "models warm" : `models ${health.warm.status}`}
            </span>
            <span className="pill">{health.documents} docs · {health.chunks} passages · {health.graph_nodes} graph nodes</span>
            {health.corpus_version && <span className="pill mono" title="Gold corpus version (manifest hash)">corpus {health.corpus_version}</span>}
          </>
        ) : (
          <span className="pill bad">Backend unreachable on :8000</span>
        )}
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => go(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
      <main className="main">
        <ErrorBoundary key={tab}>
          <Suspense fallback={<div className="empty">Loading…</div>}>
            {tab === "chat" && <T1IngestChat />}
            {tab === "sphere" && <T2 />}
            {tab === "workflow" && <T3 />}
            {tab === "telemetry" && <T4 />}
            {tab === "bench" && <T5 />}
          </Suspense>
        </ErrorBoundary>
      </main>
    </div>
  );
}
