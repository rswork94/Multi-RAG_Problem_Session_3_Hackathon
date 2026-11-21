import { useEffect, useMemo, useState } from "react";
import { Background, Controls, Handle, MarkerType, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "@dagrejs/dagre";
import { getJSON, postSSE } from "../api/client";
import { fmt, ms } from "../components/GuardBadges";
import { ENGINE_COLORS, ENGINE_LABELS } from "../lib/colors";
import { getState, setState, useApp } from "../store/useAppStore";
import type { Engine, QueryResult, TraceEntry } from "../types";

type Pipe = Engine | "ingest";
interface Topo {
  engine: string;
  label: string;
  nodes: { id: string; label: string; optional?: boolean; locked?: boolean }[];
  edges: { id: string; source: string; target: string; conditional: boolean; optional: boolean }[];
  last?: { trace?: TraceEntry[] } | null;
}

const DESC: Record<string, string> = {
  input_guard: "PII masking, scope and integrity classification; routes to refuse",
  embed_query: "bge-large-en-v1.5 query vector (instruction prefix), MPS",
  dense: "HNSW over ≤510-token child windows, collapsed to 600-token parents",
  hyde_generate: "N hypothetical passages from distinct regulatory angles (Claude Haiku 5.5)",
  multi_dense: "One dense search per hypothesis + the original query",
  query_expansion: "Entity linking via identifiers, alias index and entity-name vectors",
  bm25: "Sparse BM25 (legal tokenizer) on the expanded query",
  graph_traverse: "k-hop BFS over entity→relation→entity adjacency; PageRank fan-out cap",
  rrf: "Reciprocal Rank Fusion Σ 1/(α + rank)",
  rerank: "bge-reranker-base cross-encoder on the matched child window",
  sufficiency: "Deterministic check: seeds covered and asked-for value type reached; else loop k+1 once",
  pack_context: "Same for every engine: K=6 passages, ~4k-token budget, evidenced triples only",
  synthesize: "Claude Opus 5.5, CO-STAR + practice-area guardrails (same for all engines)",
  output_guard: "Citation validation, area-section checks, PII re-mask, disclaimer",
  refuse: "Guardrail refusal (scope / integrity G6)",
  kpi: "RCR, faithfulness (Haiku judge), ER accuracy, latency breakdown",
  s1_load: "UnstructuredPDFLoader (fast) / Docx2txt / row-atomic JSON & CSV, provenance metadata",
  s2_chunk: "Legal-separator recursive split (600/60 bge tokens) + per-clause entity-relation extraction",
  s3_tokenize: "bge WordPiece counts, ≤510-token child windows, entity normalisation",
  s4_embed: "1024-d bge vectors, Node2Vec structural vectors, entity resolution (ER F1)",
  s5_index: "Dual index: HNSW vectors + graph adjacency; Louvain, PageRank, BM25",
  s6_store: "ChromaDB persistent collections + NetworkX graph engine",
};

type NData = { label: string; status?: string; duration?: number; optional: boolean; off: boolean; selected: boolean; terminal: boolean };

function WfNode({ data: d }: NodeProps<Node<NData>>) {
  return (
    <div className={`wfnode ${d.status ?? ""} ${d.off ? "off" : ""} ${d.selected ? "selected" : ""}`}
      style={d.terminal ? { minWidth: 80, textAlign: "center", borderRadius: 20 } : undefined}>
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <b>{d.off ? "⏸ " : d.optional ? "◐ " : ""}{d.label}</b>
      {!d.terminal && (
        <span className="muted">
          {d.off ? "disabled (passthrough)" : d.status === "bypassed" ? "bypassed last run"
            : d.duration != null ? `${d.status} · ${d.duration < 1000 ? `${Math.round(d.duration)} ms` : `${(d.duration / 1000).toFixed(2)} s`}` : "not run yet"}
        </span>
      )}
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </div>
  );
}
const nodeTypes = { wf: WfNode };

function layout(nodes: Node<NData>[], edges: Edge[]) {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "TB", nodesep: 40, ranksep: 50 });
  g.setDefaultEdgeLabel(() => ({}));
  nodes.forEach((n) => g.setNode(n.id, { width: n.data.terminal ? 100 : 210, height: n.data.terminal ? 34 : 56 }));
  edges.filter((e) => !(e.source === "sufficiency" && e.target === "graph_traverse")).forEach((e) => g.setEdge(e.source, e.target));
  dagre.layout(g);
  return nodes.map((n) => {
    const p = g.node(n.id);
    return { ...n, position: { x: p.x - p.width / 2, y: p.y - p.height / 2 } };
  });
}

export default function T3WorkflowCanvas() {
  const area = useApp((s) => s.area);
  const toggles = useApp((s) => s.toggles);
  const lastResult = useApp((s) => s.lastResult);
  const lastQuestion = useApp((s) => s.lastQuestion);
  const [pipe, setPipe] = useState<Pipe>("graph");
  const [topo, setTopo] = useState<Topo | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [compare, setCompare] = useState<{ before?: QueryResult; after?: QueryResult }>({});

  useEffect(() => {
    getJSON<Topo>(`/api/workflow?engine=${pipe}`).then(setTopo).catch((e) => setErr(String(e.message ?? e)));
    setSel(null);
  }, [pipe, lastResult]);

  const engine = pipe === "ingest" ? null : (pipe as Engine);
  const tg = engine ? toggles[engine] : { disabled_nodes: [], disabled_edges: [] };
  const trace: TraceEntry[] = (engine && lastResult[engine]?.trace) || topo?.last?.trace || [];
  const byStep = useMemo(() => {
    const m = new Map<string, TraceEntry>();
    trace.forEach((t) => {
      const prev = m.get(t.step);
      m.set(t.step, prev ? { ...t, duration_ms: (prev.duration_ms ?? 0) + (t.duration_ms ?? 0) } : t);
    });
    return m;
  }, [trace]);

  const { nodes, edges } = useMemo(() => {
    if (!topo) return { nodes: [], edges: [] };
    const ns: Node<NData>[] = topo.nodes.map((n) => {
      const t = byStep.get(n.id);
      return {
        id: n.id, type: "wf", position: { x: 0, y: 0 },
        data: { label: n.id.replace(/__/g, ""), status: t?.status, duration: t?.duration_ms, optional: !!n.optional,
          off: tg.disabled_nodes.includes(n.id), selected: sel === n.id, terminal: n.id.startsWith("__") },
      };
    });
    const ran = (id: string) => id.startsWith("__") || (byStep.has(id) && byStep.get(id)!.status === "done");
    const es: Edge[] = topo.edges.map((e) => {
      const off = tg.disabled_edges.includes(e.id);
      const active = ran(e.source) && ran(e.target) && !off;
      const loop = e.source === "sufficiency" && e.target === "graph_traverse";
      return {
        id: e.id, source: e.source, target: e.target, animated: active && !e.conditional,
        label: loop ? "loop (k+1)" : e.conditional && e.target === "refuse" ? "refuse" : e.optional ? (off ? "off" : "") : undefined,
        style: { stroke: off ? "var(--muted)" : active ? (engine ? ENGINE_COLORS[engine] : "var(--good)") : "var(--border)",
          strokeDasharray: e.conditional || off ? "6 4" : undefined, strokeWidth: e.optional ? 2.2 : 1.4, opacity: off ? 0.5 : 1 },
        markerEnd: { type: MarkerType.ArrowClosed },
        data: { optional: e.optional },
      };
    });
    return { nodes: layout(ns, es), edges: es };
  }, [topo, byStep, tg, sel, engine]);

  const optionalNodes = topo?.nodes.filter((n) => n.optional) ?? [];
  const optionalEdges = topo?.edges.filter((e) => e.optional) ?? [];

  function flip(kind: "disabled_nodes" | "disabled_edges", id: string) {
    if (!engine) return;
    setState((s) => {
      const cur = s.toggles[engine][kind];
      const next = cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id];
      return { toggles: { ...s.toggles, [engine]: { ...s.toggles[engine], [kind]: next } } };
    });
  }

  async function runAblation() {
    if (!engine || !lastQuestion) return;
    setBusy(true);
    setErr("");
    const before = getState().lastResult[engine];
    try {
      const fin = (await postSSE("/api/chat", { question: lastQuestion, engine, area, pipeline_config: getState().toggles[engine] }, () => undefined)) as QueryResult;
      setState((s) => ({ lastResult: { ...s.lastResult, [engine]: fin }, kpiVersion: s.kpiVersion + 1 }));
      setCompare({ before, after: fin });
    } catch (e: any) {
      setErr(String(e.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  const selTrace = sel ? byStep.get(sel) : undefined;

  return (
    <div className="split">
      <div className="col">
        <div className="row">
          <div className="seg">
            {(["standard", "hyde", "graph", "ingest"] as Pipe[]).map((p) => (
              <button key={p} className={pipe === p ? "on" : ""} onClick={() => setPipe(p)}>
                {p === "ingest" ? "Ingestion (6 steps)" : ENGINE_LABELS[p]}
              </button>
            ))}
          </div>
          <span className="small muted">Click a node for its last-run stats. Double-click an optional node (◐), or click an optional edge, to switch it off.</span>
        </div>
        <div className="canvas">
          {topo && (
            <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView proOptions={{ hideAttribution: true }}
              nodesDraggable={false} nodesConnectable={false}
              onNodeClick={(_, n) => setSel(n.id)}
              onNodeDoubleClick={(_, n) => (n.data as NData).optional && flip("disabled_nodes", n.id)}
              onEdgeClick={(_, e) => (e.data as any)?.optional && flip("disabled_edges", e.id)}>
              <Background />
              <Controls showInteractive={false} />
            </ReactFlow>
          )}
        </div>
      </div>
      <div className="col">
        {engine ? (
          <div className="card col">
            <h3>Pipeline variation · {ENGINE_LABELS[engine]}</h3>
            {optionalNodes.length === 0 && <div className="small muted">Standard RAG has no optional nodes: it is the baseline.</div>}
            {optionalNodes.map((n) => (
              <label key={n.id} className="check small">
                <input type="checkbox" checked={!tg.disabled_nodes.includes(n.id)} onChange={() => flip("disabled_nodes", n.id)} />
                <span className="mono">{n.id}</span>
              </label>
            ))}
            {optionalEdges.length > 0 && <h4 style={{ margin: "4px 0 0" }}>Edges</h4>}
            {optionalEdges.map((e) => (
              <label key={e.id} className="check small">
                <input type="checkbox" checked={!tg.disabled_edges.includes(e.id)} onChange={() => flip("disabled_edges", e.id)} />
                <span className="mono">{e.id}</span>
              </label>
            ))}
            <div className="row">
              <button className="btn primary" disabled={busy || !lastQuestion} onClick={runAblation}>
                {busy ? "Running…" : "Run ablation on last question"}
              </button>
              <button className="btn" onClick={() => engine && setState((s) => ({ toggles: { ...s.toggles, [engine]: { disabled_nodes: [], disabled_edges: [] } } }))}>Reset</button>
            </div>
            <div className="small muted">{lastQuestion ? `Last question: “${lastQuestion.slice(0, 90)}”` : "Ask a question on Tab 1 first."}</div>
            <div className="small muted">These toggles also apply to Tab 1 chat for this engine.</div>
            {err && <div className="small" style={{ color: "var(--bad)" }}>{err}</div>}
            {compare.after && (
              <table className="data">
                <thead><tr><th>KPI</th><th className="num">Before</th><th className="num">After</th></tr></thead>
                <tbody>
                  <tr><td>RCR</td><td className="num">{fmt(compare.before?.kpis.rcr)}</td><td className="num">{fmt(compare.after.kpis.rcr)}</td></tr>
                  <tr><td>Faithfulness</td><td className="num">{fmt(compare.before?.kpis.faithfulness)}</td><td className="num">{fmt(compare.after.kpis.faithfulness)}</td></tr>
                  <tr><td>Retrieval</td><td className="num">{ms(compare.before?.kpis.latency?.retrieval_ms)}</td><td className="num">{ms(compare.after.kpis.latency?.retrieval_ms)}</td></tr>
                  <tr><td>End-to-end</td><td className="num">{ms(compare.before?.kpis.latency_ms)}</td><td className="num">{ms(compare.after.kpis.latency_ms)}</td></tr>
                  <tr><td>Disabled</td><td className="num small">{compare.before?.pipeline?.effective.join(", ") || "—"}</td><td className="num small">{compare.after.pipeline?.effective.join(", ") || "—"}</td></tr>
                </tbody>
              </table>
            )}
          </div>
        ) : (
          <div className="card small">The ingestion graph is linear and always runs all six steps; open a step for its statistics from the last ingest.</div>
        )}
        <div className="card col small">
          <h3>{sel ?? "Node details"}</h3>
          {sel ? (
            <>
              <div className="muted">{DESC[sel] ?? ""}</div>
              {selTrace ? (
                <div className="kv">
                  <div>status</div><div>{selTrace.status}</div>
                  <div>duration</div><div>{ms(selTrace.duration_ms)}</div>
                  {Object.entries(selTrace.stats ?? {}).map(([k, v]) => (
                    <FragmentKV key={k} k={k} v={v} />
                  ))}
                </div>
              ) : <div className="muted">Not run yet.</div>}
            </>
          ) : <div className="muted">Select a node.</div>}
        </div>
      </div>
    </div>
  );
}

function FragmentKV({ k, v }: { k: string; v: any }) {
  return (
    <>
      <div>{k.replace(/_/g, " ")}</div>
      <div className="mono" style={{ overflowWrap: "anywhere" }}>{typeof v === "object" ? JSON.stringify(v) : String(v)}</div>
    </>
  );
}
