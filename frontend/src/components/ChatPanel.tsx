import { useEffect, useMemo, useRef, useState } from "react";
import { getJSON, postSSE } from "../api/client";
import { ENGINE_COLORS, ENGINE_LABELS } from "../lib/colors";
import { getState, setState, useApp } from "../store/useAppStore";
import type { Engine, GoldQuestion, Passage, QueryResult, SSEEvent } from "../types";
import { Answer } from "./Answer";
import { GuardBadges } from "./GuardBadges";
import { SourceDrawer } from "./SourceDrawer";

interface Stream {
  text: string;
  status: "thinking" | "streaming" | "done" | "error";
  result?: QueryResult;
  step?: string;
  error?: string;
}
interface Turn {
  id: number;
  question: string;
  mode: Engine | "compare";
  streams: Partial<Record<Engine, Stream>>;
}

const ENGINES: Engine[] = ["standard", "hyde", "graph"];
const CAT: Record<string, string> = { keyword: "keyword", multihop: "multi-hop", vocab_alias: "alias", vocab_paraphrase: "paraphrase" };

export function ChatPanel() {
  const area = useApp((s) => s.area);
  const mode = useApp((s) => s.engine);
  const selected = useApp((s) => s.selectedNodeIds);
  const scopeOn = useApp((s) => s.useSelectionScope);
  const blocked = useApp((s) => s.blockedEdgeIds);
  const [q, setQ] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [gold, setGold] = useState<GoldQuestion[]>([]);
  const [drawer, setDrawer] = useState<Passage | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getJSON<GoldQuestion[]>("/api/gold").then(setGold).catch(() => setGold([]));
  }, []);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); // returns a Promise in new Chromium: never return it
  }, [turns]);
  const goldForArea = useMemo(() => gold.filter((g) => g.area === area), [gold, area]);

  const patch = (id: number, engine: Engine, p: Partial<Stream> | ((s: Stream) => Partial<Stream>)) =>
    setTurns((ts) =>
      ts.map((t) => {
        if (t.id !== id) return t;
        const cur = t.streams[engine] ?? { text: "", status: "thinking" };
        const upd = typeof p === "function" ? p(cur) : p;
        return { ...t, streams: { ...t.streams, [engine]: { ...cur, ...upd } } };
      }),
    );

  const onEvent = (id: number, fallback: Engine) => (e: SSEEvent) => {
    const eng = ((e as any).engine as Engine) || fallback;
    if (e.type === "token") patch(id, eng, (s) => ({ text: s.text + e.text, status: "streaming" }));
    else if (e.type === "replace") patch(id, eng, { text: e.text });
    else if (e.type === "step" && e.status === "running") patch(id, eng, { step: e.label });
  };

  async function ask(question?: string) {
    const text = (question ?? q).trim();
    if (!text || busy) return;
    const id = Date.now();
    const engines = mode === "compare" ? ENGINES : [mode];
    setTurns((ts) => [...ts, { id, question: text, mode, streams: Object.fromEntries(engines.map((e) => [e, { text: "", status: "thinking" }])) }]);
    setQ("");
    setBusy(true);
    const st = getState();
    const body = { question: text, area, selected_node_ids: scopeOn ? selected : [], blocked_edge_ids: scopeOn ? blocked : [] };
    try {
      if (mode === "compare") {
        const fin = await postSSE("/api/compare", { ...body, engine: "graph" }, onEvent(id, "graph"));
        const results: Record<Engine, QueryResult> = fin?.results ?? {};
        for (const e of ENGINES) if (results[e]) patch(id, e, { status: "done", result: results[e], text: results[e].answer });
        setState((s) => ({ lastResult: { ...s.lastResult, ...results }, lastQuestion: text, kpiVersion: s.kpiVersion + 1 }));
      } else {
        const fin = await postSSE("/api/chat", { ...body, engine: mode, pipeline_config: st.toggles[mode] }, onEvent(id, mode));
        const res = fin as QueryResult;
        patch(id, mode, { status: "done", result: res, text: res.answer });
        setState((s) => ({ lastResult: { ...s.lastResult, [mode]: res }, lastQuestion: text, kpiVersion: s.kpiVersion + 1 }));
      }
    } catch (err: any) {
      for (const e of engines) patch(id, e, { status: "error", error: String(err.message ?? err) });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card col">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }}>Multi-RAG chat</h3>
        <div className="seg" role="radiogroup" aria-label="RAG engine">
          {[...ENGINES, "compare" as const].map((e) => (
            <button key={e} className={mode === e ? "on" : ""} onClick={() => setState({ engine: e })}>
              {e === "compare" ? "Compare all 3" : ENGINE_LABELS[e]}
            </button>
          ))}
        </div>
      </div>
      {scopeOn && (selected.length > 0 || blocked.length > 0) && (
        <div className="banner info small">Retrieval is scoped by the Knowledge Sphere selection: {selected.length || "all"} nodes, {blocked.length} edges switched off.</div>
      )}
      <div className="chat" style={{ minHeight: 200 }}>
        {turns.length === 0 && (
          <div className="empty">
            Ask a question about the indexed corpus, or pick one of the labelled gold questions below. Gold questions are
            scored against ground truth (RCR uses gold anchors).
          </div>
        )}
        {turns.map((t) => (
          <div key={t.id} className="col">
            <div className="msg user"><b>You:</b> {t.question}</div>
            <div className={t.mode === "compare" ? "compare" : ""}>
              {(t.mode === "compare" ? ENGINES : [t.mode as Engine]).map((e) => (
                <EngineAnswer key={e} engine={e} s={t.streams[e]} onCite={(n) => {
                  const p = t.streams[e]?.result?.context.find((c) => c.n === n);
                  if (p) setDrawer(p);
                }} />
              ))}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <div className="col">
        <select value="" onChange={(e) => e.target.value && setQ(e.target.value)} aria-label="Gold questions">
          <option value="">Gold questions for this practice area ({goldForArea.length})…</option>
          {goldForArea.map((g) => (
            <option key={g.id} value={g.question} title={g.question}>
              [{g.id} · {CAT[g.category] ?? g.category}{g.hops ? ` · ${g.hops} hops` : ""}] {g.question.length > 90 ? g.question.slice(0, 88) + "…" : g.question}
            </option>
          ))}
        </select>
        <textarea rows={3} value={q} placeholder="e.g. What is the due date for the statement under the rule amended by Notification No. 41/2026?"
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask(); }} />
        <div className="row">
          <button className="btn primary" disabled={busy || !q.trim()} onClick={() => ask()}>
            {busy ? "Running…" : mode === "compare" ? "Ask all three engines" : `Ask ${ENGINE_LABELS[mode]}`}
          </button>
          <span className="small muted">⌘/Ctrl + Enter to send</span>
        </div>
      </div>
      <SourceDrawer passage={drawer} onClose={() => setDrawer(null)} />
    </div>
  );
}

function EngineAnswer({ engine, s, onCite }: { engine: Engine; s?: Stream; onCite: (n: number) => void }) {
  if (!s) return null;
  const r = s.result;
  return (
    <div className="msg" style={{ borderTop: `3px solid ${ENGINE_COLORS[engine]}` }}>
      <div className="meta">
        <b style={{ color: ENGINE_COLORS[engine] }}>{ENGINE_LABELS[engine]}</b>
        {r && <span className="pill small">{r.model_used}</span>}
        {r?.gold_id && <span className="pill small accent">gold {r.gold_id}</span>}
        {s.status !== "done" && s.status !== "error" && <span className="small muted">{s.step ?? "starting"}…</span>}
      </div>
      {s.status === "error" && <div style={{ color: "var(--bad)" }}>{s.error}</div>}
      {!s.text && s.status === "thinking" && <div className="thinking">Retrieving and reasoning before the first token…</div>}
      {s.text && <Answer text={s.text} onCite={onCite} />}
      {r && <GuardBadges r={r} />}
      {r && <RetrievalDetails r={r} onCite={onCite} />}
    </div>
  );
}

function RetrievalDetails({ r, onCite }: { r: QueryResult; onCite: (n: number) => void }) {
  return (
    <details style={{ marginTop: 8 }}>
      <summary className="small muted">Retrieval details ({r.context.length} passages{r.triples.length ? `, ${r.triples.length} relations` : ""})</summary>
      <div className="col small" style={{ marginTop: 6 }}>
        {r.linked_entities.length > 0 && (
          <div><b>Linked entities:</b> {r.linked_entities.map((l) => `${l.label} (${l.how})`).join(" · ")}</div>
        )}
        {r.hypotheses.length > 0 && (
          <div><b>HyDE passages:</b><ol style={{ margin: "4px 0" }}>{r.hypotheses.map((h, i) => <li key={i}>{h}</li>)}</ol></div>
        )}
        {r.sufficiency && (
          <div><b>Sufficiency:</b> {r.sufficiency.sufficient ? "sufficient" : "insufficient"} · wanted {String(r.sufficiency.wanted_types ?? [])} · found {String(r.sufficiency.found_types ?? [])}{r.loops ? ` · looped to k=${r.hop_k_used}` : ""}</div>
        )}
        {r.triples.length > 0 && (
          <div><b>Relations sent to the model:</b><ul style={{ margin: "4px 0" }}>{r.triples.slice(0, 10).map((t, i) => (
            <li key={i}>{t.src_label} —{t.rel}→ {t.dst_label} <span className="muted">(hop {t.hop}, evidence {t.refs?.join(", ")})</span></li>))}</ul></div>
        )}
        <div className="table-wrap"><table className="data">
          <thead><tr><th>#</th><th>Source · section</th><th>Via</th><th className="num">Tokens</th></tr></thead>
          <tbody>
            {r.context.map((c) => (
              <tr key={c.chunk_id}>
                <td><button className="cite" onClick={() => onCite(c.n)}>{c.n}</button></td>
                <td>{c.source} · {c.section?.slice(0, 70)}{c.injection_flag ? " ⚠" : ""}</td>
                <td className="mono">{Object.entries(c.ranks ?? {}).map(([k, v]) => `${k}#${v}`).join(" ")}{c.hop != null ? ` hop${c.hop}` : ""}</td>
                <td className="num">{c.n_tokens}</td>
              </tr>
            ))}
          </tbody>
        </table></div>
      </div>
    </details>
  );
}
