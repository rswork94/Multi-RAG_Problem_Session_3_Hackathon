import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getJSON, sendJSON } from "../api/client";
import { fmt, ms } from "../components/GuardBadges";
import { ENGINE_COLORS, ENGINE_LABELS, SEGMENTS, kpiBand } from "../lib/colors";
import { setState, useApp } from "../store/useAppStore";
import type { Engine, HpMeta, Hyperparams, RunRow } from "../types";

const ENGINES: Engine[] = ["standard", "hyde", "graph"];
const KPI_KEYS = [
  { key: "rcr", label: "Retrieval Context Recall", green: 0.8, amber: 0.5, up: true, fmt: (v: any) => fmt(v) },
  { key: "faithfulness", label: "Faithfulness", green: 0.9, amber: 0.75, up: true, fmt: (v: any) => fmt(v) },
  { key: "er_accuracy", label: "Entity-Relation Resolution (F1)", green: 0.8, amber: 0.6, up: true, fmt: (v: any) => fmt(v) },
  { key: "latency_ms", label: "End-to-end latency", green: 15000, amber: 30000, up: false, fmt: (v: any) => ms(v) },
] as const;
const BAND_LABEL: Record<string, string> = { good: "✓ on target", warn: "! watch", bad: "✗ below target", neutral: "no target" };

export default function T4HyperparamsTelemetry() {
  const kpiVersion = useApp((s) => s.kpiVersion);
  const [hp, setHp] = useState<Hyperparams | null>(null);
  const [draft, setDraft] = useState<Hyperparams | null>(null);
  const [meta, setMeta] = useState<Record<string, HpMeta>>({});
  const [reingest, setReingest] = useState(false);
  const [err, setErr] = useState("");
  const [rows, setRows] = useState<RunRow[]>([]);
  const [engine, setEngine] = useState<Engine>("graph");
  const [metric, setMetric] = useState<"rcr" | "faithfulness" | "latency_ms">("rcr");

  const loadHp = () =>
    getJSON<any>("/api/hyperparams").then((r) => {
      setHp(r.values);
      setDraft(r.values);
      setMeta(r.meta);
      setReingest(r.reingest_required);
    });
  useEffect(() => { loadHp().catch((e) => setErr(String(e.message ?? e))); }, []);
  useEffect(() => { getJSON<any>("/api/kpis?limit=500").then((r) => setRows(r.history)).catch(() => setRows([])); }, [kpiVersion]);

  async function apply() {
    if (!draft) return;
    setErr("");
    try {
      const r = await sendJSON<any>("/api/hyperparams", "PUT", draft);
      setHp(r.values);
      setDraft(r.values);
      setReingest(r.reingest_required);
      setState((s) => ({ graphVersion: s.graphVersion + 1 }));
    } catch (e: any) {
      setErr(String(e.message ?? e));
    }
  }

  const live = rows.filter((r) => !r.refused);
  const byEngine = (e: Engine) => live.filter((r) => r.engine === e);
  const latest = byEngine(engine).at(-1);
  const prev = byEngine(engine).at(-2);

  const history = useMemo(() => {
    const out: Record<string, any>[] = [];
    live.forEach((r, i) => {
      const v = metric === "latency_ms" ? r.kpis.latency_ms : (r.kpis as any)[metric];
      out.push({ run: i + 1, [r.engine]: v, q: r.question.slice(0, 60), hp: r.hyperparams, disabled: r.disabled });
    });
    return out;
  }, [live, metric]);

  const latencyRows = ENGINES.map((e) => {
    const r = byEngine(e).at(-1);
    const l = r?.kpis.latency;
    return { engine: ENGINE_LABELS[e], ...(l ? Object.fromEntries(SEGMENTS.map((s) => [s.key, Math.round((l as any)[s.key] ?? 0)])) : {}) };
  }).filter((r) => Object.keys(r).length > 1);

  const groups = useMemo(() => {
    const g = new Map<string, string[]>();
    Object.entries(meta).forEach(([k, m]) => g.set(m.group, [...(g.get(m.group) ?? []), k]));
    return [...g.entries()];
  }, [meta]);

  function exportCsv() {
    const head = ["run", "engine", "area", "question", "gold_id", "rcr", "faithfulness", "er_accuracy", "latency_ms", "retrieval_ms",
      "chunk_size", "chunk_overlap", "hyde_n", "hop_k", "rrf_alpha", "disabled"];
    const lines = live.map((r, i) => [i + 1, r.engine, r.area, JSON.stringify(r.question), r.gold_id ?? "", r.kpis.rcr, r.kpis.faithfulness,
      r.kpis.er_accuracy, r.kpis.latency_ms, r.kpis.latency?.retrieval_ms, r.hyperparams?.chunk_size, r.hyperparams?.chunk_overlap,
      r.hyperparams?.hyde_n, r.hyperparams?.hop_k, r.hyperparams?.rrf_alpha, (r.disabled ?? []).join("|")].join(","));
    const blob = new Blob([[head.join(","), ...lines].join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "kpi-history.csv";
    a.click();
  }

  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="card col">
        <h3>4 configurable hyperparameters</h3>
        {reingest && <div className="banner">Chunk size / overlap differ from the indexed corpus: re-ingest on Tab 1 to apply them.</div>}
        <div className="grid4">
          {draft && groups.map(([group, keys]) => (
            <div key={group} className="col">
              <b className="small">{group}</b>
              {keys.map((k) => {
                const m = meta[k];
                const v = (draft as any)[k];
                return (
                  <div key={k} className="col" style={{ gap: 2 }} title={m.help}>
                    <div className="row small" style={{ justifyContent: "space-between" }}>
                      <span className="mono">{k}</span>
                      <input type="number" value={v} min={m.min} max={m.max} step={m.step} style={{ width: 80 }}
                        onChange={(e) => setDraft({ ...draft, [k]: Number(e.target.value) })} />
                    </div>
                    <input type="range" min={m.min} max={m.max} step={m.step} value={v} aria-label={k}
                      onChange={(e) => setDraft({ ...draft, [k]: Number(e.target.value) })} />
                    <span className="small muted">{m.help}{m.reingest ? " (needs re-ingest)" : " (applies to the next query)"}</span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
        <div className="row">
          <button className="btn primary" disabled={!draft || JSON.stringify(draft) === JSON.stringify(hp)} onClick={apply}>Apply</button>
          <button className="btn" onClick={() => setDraft(hp)}>Revert</button>
          <button className="btn" onClick={() => setDraft({ chunk_size: 600, chunk_overlap: 60, hyde_n: 3, hop_k: 2, rrf_alpha: 60 })}>Brief defaults (600/60, N=3, k=2, α=60)</button>
          {err && <span className="small" style={{ color: "var(--bad)" }}>{err}</span>}
        </div>
      </div>

      <div className="card col">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3 style={{ margin: 0 }}>4 core KPIs · latest run</h3>
          <div className="seg">
            {ENGINES.map((e) => <button key={e} className={engine === e ? "on" : ""} onClick={() => setEngine(e)}>{ENGINE_LABELS[e]}</button>)}
          </div>
        </div>
        {!latest ? <div className="empty">No {ENGINE_LABELS[engine]} runs yet: ask a question on Tab 1.</div> : (
          <div className="grid4">
            {KPI_KEYS.map((k) => {
              const v = (latest.kpis as any)[k.key];
              const pv = prev ? (prev.kpis as any)[k.key] : null;
              const band = kpiBand(v, k.green, k.amber, k.up);
              const na = v === null || v === undefined;
              return (
                <div key={k.key} className={`kpi ${band}`} title={latest.kpis.rcr_method && k.key === "rcr" ? latest.kpis.rcr_method : ""}>
                  <div className="small muted">{k.label}</div>
                  <div className="v">{na ? "N/A" : k.fmt(v)}</div>
                  <div className={`tile-status ${band}`}>{na ? (k.key === "er_accuracy" ? "Graph RAG only" : "n/a") : BAND_LABEL[band]}</div>
                  {!na && pv !== null && pv !== undefined && (
                    <div className="small muted">vs previous: {k.key === "latency_ms" ? ms(v - pv) : (v - pv >= 0 ? "+" : "") + (v - pv).toFixed(2)}</div>
                  )}
                  {k.key === "rcr" && latest.kpis.rcr_method && <div className="small muted">{latest.kpis.rcr_method}</div>}
                  {k.key === "latency_ms" && <div className="small muted">retrieval {ms(latest.kpis.latency?.retrieval_ms)} · graph {ms(latest.kpis.latency?.graph_ms)} · LLM {ms(latest.kpis.latency?.llm_inference_ms)}</div>}
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="grid2" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3 style={{ margin: 0 }}>KPI history by engine</h3>
            <div className="seg">
              {(["rcr", "faithfulness", "latency_ms"] as const).map((m) => (
                <button key={m} className={metric === m ? "on" : ""} onClick={() => setMetric(m)}>{m === "latency_ms" ? "latency" : m}</button>
              ))}
            </div>
          </div>
          {history.length === 0 ? <div className="empty">No runs yet.</div> : (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={history} margin={{ top: 12, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis dataKey="run" tickLine={false} axisLine={{ stroke: "var(--border)" }} />
                <YAxis tickLine={false} axisLine={false} width={metric === "latency_ms" ? 56 : 36}
                  domain={metric === "latency_ms" ? [0, "auto"] : [0, 1]}
                  tickFormatter={(v) => (metric === "latency_ms" ? `${(v / 1000).toFixed(0)}s` : v)} />
                <Tooltip formatter={(v: any) => (metric === "latency_ms" ? ms(v) : fmt(v))}
                  labelFormatter={(l, p) => `Run ${l}: ${p?.[0]?.payload?.q ?? ""}`} />
                <Legend />
                {ENGINES.map((e) => (
                  <Line key={e} type="monotone" dataKey={e} name={ENGINE_LABELS[e]} stroke={ENGINE_COLORS[e]} strokeWidth={2}
                    dot={{ r: 4, strokeWidth: 2, stroke: "var(--panel)", fill: ENGINE_COLORS[e] }} connectNulls isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
        <div className="card">
          <h3>Latency breakdown · latest run per engine</h3>
          {latencyRows.length === 0 ? <div className="empty">No runs yet.</div> : (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={latencyRows} layout="vertical" margin={{ top: 8, right: 16, bottom: 4, left: 8 }} barCategoryGap={18}>
                <CartesianGrid stroke="var(--grid)" horizontal={false} />
                <XAxis type="number" tickLine={false} axisLine={{ stroke: "var(--border)" }} tickFormatter={(v) => `${(v / 1000).toFixed(1)}s`} />
                <YAxis type="category" dataKey="engine" tickLine={false} axisLine={false} width={96} />
                <Tooltip formatter={(v: any) => ms(v)} />
                <Legend />
                {SEGMENTS.map((s, i) => (
                  <Bar key={s.key} dataKey={s.key} name={s.label} stackId="lat" fill={s.color} stroke="var(--panel)" strokeWidth={2}
                    radius={i === SEGMENTS.length - 1 ? [0, 4, 4, 0] : 0} isAnimationActive={false} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          )}
          <div className="small muted">The brief's ~450 / 1,200 / 850 ms targets are compared with retrieval-stage latency on Tab 5; LLM synthesis is reported separately.</div>
        </div>
      </div>

      <div className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3 style={{ margin: 0 }}>Run history ({live.length})</h3>
          <button className="btn small" onClick={exportCsv} disabled={!live.length}>Export CSV</button>
        </div>
        <div className="scroll">
          <table className="data">
            <thead>
              <tr><th>#</th><th>Engine</th><th>Question</th><th className="num">RCR</th><th className="num">Faith.</th><th className="num">ER F1</th>
                <th className="num">Retrieval</th><th className="num">E2E</th><th>chunk · N · k · α</th><th>Disabled nodes</th></tr>
            </thead>
            <tbody>
              {[...live].reverse().slice(0, 60).map((r, i) => (
                <tr key={i}>
                  <td>{live.length - i}</td>
                  <td style={{ borderLeft: `3px solid ${ENGINE_COLORS[r.engine]}` }}>{ENGINE_LABELS[r.engine]}</td>
                  <td title={r.question}>{r.gold_id ? <span className="pill">{r.gold_id}</span> : null} {r.question.slice(0, 70)}</td>
                  <td className="num">{fmt(r.kpis.rcr)}</td>
                  <td className="num">{fmt(r.kpis.faithfulness)}</td>
                  <td className="num">{fmt(r.kpis.er_accuracy)}</td>
                  <td className="num">{ms(r.kpis.latency?.retrieval_ms)}</td>
                  <td className="num">{ms(r.kpis.latency_ms)}</td>
                  <td className="mono small">{r.hyperparams ? `${r.hyperparams.chunk_size}/${r.hyperparams.chunk_overlap} · ${r.hyperparams.hyde_n} · ${r.hyperparams.hop_k} · ${r.hyperparams.rrf_alpha}` : ""}</td>
                  <td className="mono small">{(r.disabled ?? []).join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
