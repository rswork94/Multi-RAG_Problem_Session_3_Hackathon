import { useEffect, useState } from "react";
import { getJSON, postSSE, sendJSON } from "../api/client";
import { ChatPanel } from "../components/ChatPanel";
import { IngestStepper, type StepState } from "../components/IngestStepper";
import { ModelSettings } from "../components/ModelSettings";
import { UploadDropzone } from "../components/UploadDropzone";
import { AREA_LABELS } from "../lib/colors";
import { setState, useApp } from "../store/useAppStore";

interface Doc {
  doc_id: string;
  source: string;
  title: string;
  doc_type: string;
  issued_date: string;
  practice_area: string;
  format: string;
  loader_used: string;
  chunks: number;
}

const STEPS = [
  ["s1_load", "1 · Data format & ingestion"],
  ["s2_chunk", "2 · Semantic & entity chunking"],
  ["s3_tokenize", "3 · Sub-word tokenization & entity normalization"],
  ["s4_embed", "4 · Multi-modal embedding"],
  ["s5_index", "5 · Dual indexing"],
  ["s6_store", "6 · Hybrid vector DB & graph engine"],
];
const fresh = (): StepState[] => STEPS.map(([step, label]) => ({ step, label, status: "pending" }));

export default function T1IngestChat() {
  const area = useApp((s) => s.area);
  const health = useApp((s) => s.health);
  const graphVersion = useApp((s) => s.graphVersion);
  const [steps, setSteps] = useState<StepState[]>(fresh());
  const [busy, setBusy] = useState(false);
  const [useLlm, setUseLlm] = useState(true);
  const [msg, setMsg] = useState<string[]>([]);
  const [docs, setDocs] = useState<Doc[]>([]);

  useEffect(() => {
    getJSON<Doc[]>("/api/documents").then(setDocs).catch(() => setDocs([]));
    getJSON<any>("/api/workflow?engine=ingest").then((w) => {
      const tr = w?.last?.trace as StepState[] | undefined;
      if (tr?.length) setSteps(fresh().map((s) => ({ ...s, ...(tr.find((t) => t.step === s.step) ?? {}) })));
    }).catch(() => undefined);
  }, [graphVersion]);

  async function run(url: string, body?: FormData) {
    setBusy(true);
    setMsg([]);
    setSteps(fresh());
    try {
      const fin = await postSSE(url, body, (e) => {
        if (e.type === "step") setSteps((ss) => ss.map((s) => (s.step === e.step ? { ...s, ...e, progress: s.progress } : s)));
        if (e.type === "progress") setSteps((ss) => ss.map((s) => (s.step === e.step ? { ...s, progress: { done: e.done, total: e.total } } : s)));
      });
      setMsg([...(fin?.warnings ?? []), ...(fin?.errors ?? [])]);
    } catch (e: any) {
      setMsg([String(e.message ?? e)]);
    } finally {
      setBusy(false);
      setState((s) => ({ graphVersion: s.graphVersion + 1 }));
    }
  }

  const upload = (files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    fd.append("area", area);
    fd.append("use_llm", String(useLlm));
    run("/api/ingest", fd);
  };

  async function reset() {
    if (!confirm("Delete all indexed documents, vectors, the knowledge graph and run history? This cannot be undone.")) return;
    await sendJSON("/api/data", "DELETE");
    setSteps(fresh());
    setState((s) => ({ graphVersion: s.graphVersion + 1, selectedNodeIds: [] }));
  }

  return (
    <div className="grid2">
      <div className="col">
        <ModelSettings />
        <div className="card col">
          <h3>Data ingestion</h3>
          <div className="small muted">
            New uploads are tagged <b>{AREA_LABELS[area]}</b> (switch the practice area in the header to change it).
          </div>
          <UploadDropzone onFiles={upload} disabled={busy} />
          <label className="check small">
            <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
            LLM entity-relation extraction (utility model, cached per clause){health?.llm_mode === "offline" ? ": offline, so regex is used" : ""}
          </label>
          <div className="row">
            <button className="btn primary" disabled={busy} onClick={() => run(`/api/ingest/gold?use_llm=${useLlm}`)}>
              Load synthetic gold corpus (22 docs)
            </button>
            <button className="btn" disabled={busy} onClick={() => run(`/api/reingest?use_llm=${useLlm}`)}>Re-ingest</button>
            <button className="btn" disabled={busy} onClick={reset}>Reset</button>
          </div>
          {health?.reingest_required && <div className="banner small">Chunk size/overlap changed on Tab 4: re-ingest to apply.</div>}
          <IngestStepper steps={steps} />
          {msg.map((m, i) => <div key={i} className="small" style={{ color: "var(--warn)" }}>{m}</div>)}
        </div>
        <div className="card">
          <h3>Indexed documents ({docs.length})</h3>
          {docs.length === 0 ? (
            <div className="muted small">Nothing indexed yet.</div>
          ) : (
            <div className="scroll" style={{ maxHeight: 320 }}>
              <table className="data">
                <thead><tr><th>Document</th><th>Area</th><th>Type</th><th className="num">Passages</th></tr></thead>
                <tbody>
                  {docs.map((d) => (
                    <tr key={d.doc_id} title={`${d.title}\nloader: ${d.loader_used}\nissued: ${d.issued_date}`}>
                      <td>{d.source}</td><td>{d.practice_area}</td><td>{d.doc_type}</td><td className="num">{d.chunks}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
      <ChatPanel />
    </div>
  );
}
