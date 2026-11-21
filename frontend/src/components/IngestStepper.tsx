import { useState } from "react";

export interface StepState {
  step: string;
  label: string;
  status: string;
  duration_ms?: number;
  stats?: Record<string, any>;
  error?: string;
  progress?: { done: number; total: number };
}

function Value({ v }: { v: any }) {
  if (v === null || v === undefined) return <span className="muted">—</span>;
  if (typeof v === "object") {
    return (
      <span className="mono small">
        {Object.entries(v)
          .map(([k, x]) => `${k}: ${typeof x === "object" ? JSON.stringify(x) : x}`)
          .join(" · ")}
      </span>
    );
  }
  return <span>{String(v)}</span>;
}

/** The 6-step Graph RAG ingestion stepper: lights up live from SSE events, expandable per-step stats. */
export function IngestStepper({ steps }: { steps: StepState[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="steps">
      {steps.map((s) => (
        <div className="step" key={s.step}>
          <div className="head" onClick={() => setOpen(open === s.step ? null : s.step)}>
            <span className={`dot ${s.status}`} />
            <b style={{ flex: 1 }}>{s.label}</b>
            {s.duration_ms !== undefined && <span className="small muted">{(s.duration_ms / 1000).toFixed(1)} s</span>}
          </div>
          {s.progress && s.status === "running" && (
            <div className="bar" style={{ marginTop: 6 }}>
              <div style={{ width: `${(100 * s.progress.done) / Math.max(1, s.progress.total)}%` }} />
            </div>
          )}
          {s.error && <div className="small" style={{ color: "var(--bad)" }}>{s.error}</div>}
          {open === s.step && s.stats && (
            <div className="kv">
              {Object.entries(s.stats).map(([k, v]) => (
                <Fragment2 key={k} k={k} v={v} />
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function Fragment2({ k, v }: { k: string; v: any }) {
  return (
    <>
      <div>{k.replace(/_/g, " ")}</div>
      <div><Value v={v} /></div>
    </>
  );
}
