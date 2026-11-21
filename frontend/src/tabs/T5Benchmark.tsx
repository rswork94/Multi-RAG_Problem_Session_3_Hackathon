import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getJSON, postSSE } from "../api/client";
import { Answer } from "../components/Answer";
import { fmt, ms } from "../components/GuardBadges";
import { ENGINE_COLORS, ENGINE_LABELS } from "../lib/colors";
import { setState } from "../store/useAppStore";

const AREAS = [["tax", "Tax"], ["grc", "GRC"], ["deals", "Deals"]];
const CATS = [["keyword", "Simple keyword"], ["multihop", "Multi-hop relational"], ["vocab_alias", "Vocabulary mismatch: alias"], ["vocab_paraphrase", "Vocabulary mismatch: paraphrase"]];
const VARIANTS = ["standard", "hyde", "graph", "graph_no_graph", "graph_no_rerank"];
const CORE = ["standard", "hyde", "graph"];
const SHORT: Record<string, string> = { keyword: "Keyword", multihop: "Multi-hop", vocab_alias: "Alias", vocab_paraphrase: "Paraphrase" };

interface Run { id: string; created: number; areas: string[]; quick: boolean; llm_mode: string; n_questions: number; cost_usd: number; corpus_version: string }

export default function T5Benchmark() {
  const [areas, setAreas] = useState<string[]>(["tax", "grc", "deals"]);
  const [cats, setCats] = useState<string[]>(CATS.map((c) => c[0]));
  const [variants, setVariants] = useState<string[]>(CORE);
  const [quick, setQuick] = useState(true);
  const [est, setEst] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [prog, setProg] = useState<{ phase: string; done: number; total: number } | null>(null);
  const [err, setErr] = useState("");
  const [runs, setRuns] = useState<Run[]>([]);
  const [res, setRes] = useState<any>(null);
  const [qid, setQid] = useState<string>("");
  const [metric, setMetric] = useState<"rcr" | "faithfulness" | "retrieval_p50">("rcr");

  const body = { areas, categories: cats.length === CATS.length ? null : cats, variants, quick };
  useEffect(() => {
    fetch("/api/benchmark/estimate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then((r) => r.json()).then(setEst).catch(() => setEst(null));
  }, [JSON.stringify(body)]);
  const loadRuns = () => getJSON<Run[]>("/api/benchmark/runs").then((r) => {
    setRuns(r);
    if (!res && r[0]) getJSON(`/api/benchmark/${r[0].id}`).then(setRes);
  }).catch(() => setRuns([]));
  useEffect(() => { loadRuns(); }, []);

  async function run() {
    if (est?.online && est.est_cost_usd > 0 && !confirm(`This run makes about ${est.runs} Claude syntheses plus judge calls. Estimated cost: $${est.est_cost_usd} (~${est.est_minutes} min). Continue?`)) return;
    setBusy(true);
    setErr("");
    setProg({ phase: "starting", done: 0, total: est?.runs ?? 0 });
    try {
      const fin = await postSSE("/api/benchmark", { ...body, confirm_cost: true }, (e) => {
        if (e.type === "bench") setProg({ phase: e.phase, done: e.done, total: e.total });
      });
      setRes(fin.result);
      setQid("");
      setState((s) => ({ kpiVersion: s.kpiVersion + 1 }));
      loadRuns();
    } catch (e: any) {
      setErr(String(e.message ?? e));
    } finally {
      setBusy(false);
      setProg(null);
    }
  }

  const toggle = (list: string[], v: string, set: (l: string[]) => void) => set(list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  const agg = res?.aggregate;
  const vs: string[] = res ? Object.keys(res.variants) : [];

  const chartData = useMemo(() => {
    if (!agg) return [];
    return agg.categories.map((c: string) => ({
      category: SHORT[c] ?? c,
      ...Object.fromEntries(vs.map((v) => [v, agg.cells[`${v}|${c}`]?.[metric] ?? null])),
    }));
  }, [agg, metric, vs]);

  const questions = useMemo(() => {
    if (!res) return [];
    const m = new Map<string, any>();
    res.rows.forEach((r: any) => m.set(r.question_id, r));
    return [...m.values()];
  }, [res]);
  const qRows = res?.rows.filter((r: any) => r.question_id === qid) ?? [];

  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="card col">
        <h3>Comparative benchmark · gold question set</h3>
        <div className="row">
          <span className="small muted">Areas</span>
          {AREAS.map(([a, l]) => <label key={a} className="check small"><input type="checkbox" checked={areas.includes(a)} onChange={() => toggle(areas, a, setAreas)} />{l}</label>)}
          <span className="small muted" style={{ marginLeft: 12 }}>Categories</span>
          {CATS.map(([c, l]) => <label key={c} className="check small"><input type="checkbox" checked={cats.includes(c)} onChange={() => toggle(cats, c, setCats)} />{l}</label>)}
        </div>
        <div className="row">
          <span className="small muted">Engines</span>
          {VARIANTS.map((v) => (
            <label key={v} className="check small">
              <input type="checkbox" checked={variants.includes(v)} onChange={() => toggle(variants, v, setVariants)} />
              <i style={{ width: 10, height: 10, borderRadius: 5, display: "inline-block", background: ENGINE_COLORS[v] }} />{ENGINE_LABELS[v]}
            </label>
          ))}
          <label className="check small" style={{ marginLeft: 12 }}><input type="checkbox" checked={quick} onChange={(e) => setQuick(e.target.checked)} />Quick (one question per category per area)</label>
        </div>
        <div className="row">
          <button className="btn primary" disabled={busy || !areas.length || !variants.length} onClick={run}>{busy ? "Running…" : "Run benchmark"}</button>
          {est && <span className="small muted">{est.questions} questions × {est.variants} engines = {est.runs} runs · {est.online ? `≈ $${est.est_cost_usd} · ~${est.est_minutes} min (Claude online)` : "offline: $0 (extractive answers, lexical judge)"}</span>}
        </div>
        {prog && (
          <div className="col small">
            <span>{prog.phase === "retrieval" ? "Phase 1: retrieval, serial and timed" : prog.phase === "synthesis" ? "Phase 2: synthesis + judge (concurrent)" : prog.phase} · {prog.done}/{prog.total}</span>
            <div className="bar"><div style={{ width: `${(100 * prog.done) / Math.max(1, prog.total)}%` }} /></div>
          </div>
        )}
        {err && <div className="small" style={{ color: "var(--bad)" }}>{err}</div>}
        <div className="small muted">
          Protocol: identical K=6 passages, ~4k-token context, prompt, model and effort for every engine. Engine order is randomised per question after a discarded warm-up. Winner rule (declared before running): {res?.winner_rule ?? "highest RCR; within 0.05, higher faithfulness; then lower retrieval latency"}.
        </div>
        {runs.length > 0 && (
          <div className="row small">
            <span className="muted">Saved runs</span>
            <select value={res?.id ?? ""} onChange={(e) => getJSON(`/api/benchmark/${e.target.value}`).then(setRes)}>
              {runs.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.llm_mode} · {r.n_questions} q · ${r.cost_usd}</option>)}
            </select>
            {res && <a href={`/api/benchmark/${res.id}/export.csv`}>Export CSV</a>}
            {res && <a href={`/api/benchmark/${res.id}`} target="_blank" rel="noreferrer">JSON</a>}
          </div>
        )}
      </div>

      {res && agg && (
        <>
          <div className="card">
            <h3>Brief's expectations vs measured ({res.llm_mode}, {res.synth_model}, corpus {res.corpus_version})</h3>
            <div className="scroll">
              <table className="data">
                <thead><tr><th>Evaluation metric</th><th>Standard (brief)</th><th>HyDE (brief)</th><th>Graph (brief)</th><th>Measured</th><th>Verdict</th></tr></thead>
                <tbody>
                  {res.expected_vs_measured.map((r: any) => (
                    <tr key={r.metric}>
                      <td><b>{r.metric}</b></td>
                      {CORE.map((e) => <td key={e} className="small">{r.brief[e]}</td>)}
                      <td className="small">{measuredText(r)}</td>
                      <td>{r.verdict ? <span className={`pill ${r.verdict === "confirmed" ? "good" : "warn"}`}>{r.verdict === "confirmed" ? "✓ confirmed" : "✗ not confirmed"}</span> : <span className="muted small">{r.note}</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="grid2" style={{ gridTemplateColumns: "1fr 1fr" }}>
            <div className="card">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h3 style={{ margin: 0 }}>By question category</h3>
                <div className="seg">
                  {(["rcr", "faithfulness", "retrieval_p50"] as const).map((m) => (
                    <button key={m} className={metric === m ? "on" : ""} onClick={() => setMetric(m)}>{m === "retrieval_p50" ? "retrieval p50" : m}</button>
                  ))}
                </div>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData} margin={{ top: 12, right: 12, bottom: 4, left: 0 }} barGap={2} barCategoryGap={16}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="category" tickLine={false} axisLine={{ stroke: "var(--border)" }} interval={0} tick={{ fontSize: 11 }} />
                  <YAxis tickLine={false} axisLine={false} width={44} domain={metric === "retrieval_p50" ? [0, "auto"] : [0, 1]}
                    tickFormatter={(v) => (metric === "retrieval_p50" ? `${v}ms` : v)} />
                  <Tooltip formatter={(v: any) => (metric === "retrieval_p50" ? ms(v) : fmt(v))} />
                  <Legend />
                  {vs.map((v) => (
                    <Bar key={v} dataKey={v} name={ENGINE_LABELS[v]} fill={ENGINE_COLORS[v]} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="card">
              <h3>Results matrix</h3>
              <div className="scroll">
                <table className="data">
                  <thead>
                    <tr><th>Category</th><th>Engine</th><th className="num">RCR (95% CI)</th><th className="num">Faith.</th><th className="num">Halluc.</th>
                      <th className="num">Retr. p50 / p90</th><th className="num">E2E p50</th><th className="num">Answer pass</th></tr>
                  </thead>
                  <tbody>
                    {[...agg.categories, "all"].flatMap((c: string) => vs.map((v) => {
                      const cell = agg.cells[`${v}|${c}`];
                      if (!cell) return null;
                      const win = agg.winners[c] === v;
                      return (
                        <tr key={`${v}|${c}`}>
                          <td>{v === vs[0] ? <b>{c === "all" ? "All" : CATS.find((x) => x[0] === c)?.[1]}</b> : ""}</td>
                          <td className={win ? "win" : ""} style={{ borderLeft: `3px solid ${ENGINE_COLORS[v]}` }}>{ENGINE_LABELS[v]}{win ? " ★" : ""}</td>
                          <td className="num">{fmt(cell.rcr)}{cell.rcr_ci ? <span className="muted small"> [{cell.rcr_ci[0]}–{cell.rcr_ci[1]}]</span> : ""}</td>
                          <td className="num">{fmt(cell.faithfulness)}</td>
                          <td className="num">{fmt(cell.hallucination_rate)}</td>
                          <td className="num">{ms(cell.retrieval_p50)} / {ms(cell.retrieval_p90)}</td>
                          <td className="num">{ms(cell.e2e_p50)}</td>
                          <td className="num">{fmt(cell.answer_pass_rate)}</td>
                        </tr>
                      );
                    }))}
                  </tbody>
                </table>
              </div>
              <div className="small muted">★ = winner by the declared rule (core engines only). n = {res.n_questions} questions; intervals are bootstrap 95% CIs.</div>
            </div>
          </div>

          <div className="card col">
            <div className="row">
              <h3 style={{ margin: 0 }}>Per-question drill-down</h3>
              <select value={qid} onChange={(e) => setQid(e.target.value)} style={{ maxWidth: "100%" }}>
                <option value="">Choose a question…</option>
                {questions.map((q: any) => <option key={q.question_id} value={q.question_id}>[{q.question_id} · {q.category}] {q.question.slice(0, 110)}</option>)}
              </select>
            </div>
            {qRows.length > 0 && (
              <>
                <div className="small"><b>{qRows[0].question}</b></div>
                <div className="compare" style={{ gridTemplateColumns: `repeat(${Math.min(qRows.length, 3)}, minmax(0, 1fr))` }}>
                  {qRows.map((r: any) => (
                    <div key={r.variant} className="msg" style={{ borderTop: `3px solid ${ENGINE_COLORS[r.variant]}` }}>
                      <div className="meta">
                        <b>{ENGINE_LABELS[r.variant]}</b>
                        <span className="pill">RCR {fmt(r.kpis.rcr)}</span>
                        <span className="pill">Faith. {fmt(r.kpis.faithfulness)}</span>
                        <span className="pill">{ms(r.kpis.latency?.retrieval_ms)} retrieval</span>
                        {r.kpis.answer_check && <span className={`pill ${r.kpis.answer_check.pass ? "good" : "warn"}`}>{r.kpis.answer_check.pass ? "✓ must-include met" : `✗ missing ${r.kpis.answer_check.missing.join(", ")}`}</span>}
                      </div>
                      <Answer text={r.answer} onCite={() => undefined} />
                      <details className="small"><summary className="muted">Gold facts & passages</summary>
                        <ul>{(r.kpis.rcr_facts ?? []).map((f: any, i: number) => <li key={i}>{f.present ? "✓" : "✗"} {f.fact}</li>)}</ul>
                        <ol>{r.context_sources.map((s: string, i: number) => <li key={i}>{s}</li>)}</ol>
                        {r.linked?.length > 0 && <div>Linked: {r.linked.join(" · ")}</div>}
                      </details>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </>
      )}
      {!res && <div className="card empty">No benchmark runs yet. Load the gold corpus on Tab 1, then run a quick benchmark (offline runs cost nothing).</div>}
    </div>
  );
}

function measuredText(r: any): string {
  if (r.measured_order) {
    return Object.entries(r.measured).map(([e, v]: any) => `${ENGINE_LABELS[e]?.split(" ")[0]} ${ms(v.retrieval_p50_ms)} (E2E ${ms(v.e2e_p50_ms)})`).join(" · ");
  }
  if (r.measured_best !== undefined) {
    return Object.entries(r.measured).map(([e, v]: any) => `${ENGINE_LABELS[e]?.split(" ")[0]} RCR ${fmt(v.rcr)}`).join(" · ") + ` → best ${ENGINE_LABELS[r.measured_best] ?? "—"}`;
  }
  return Object.entries(r.measured ?? {}).map(([e, v]: any) => `${ENGINE_LABELS[e]?.split(" ")[0]} ${fmt(v.hallucination_rate)}`).join(" · ");
}
