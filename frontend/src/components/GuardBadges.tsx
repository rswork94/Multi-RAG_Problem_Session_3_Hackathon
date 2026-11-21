import type { QueryResult } from "../types";

export function fmt(v: number | null | undefined, digits = 2): string {
  return v === null || v === undefined ? "n/a" : v.toFixed(digits);
}

export function ms(v: number | null | undefined): string {
  if (v === null || v === undefined) return "n/a";
  return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`;
}

/** Guardrail outcome + KPI chips under an answer. */
export function GuardBadges({ r }: { r: QueryResult }) {
  const g = r.guard_report || {};
  const k = r.kpis || ({} as QueryResult["kpis"]);
  const area = g.area_checks || {};
  return (
    <div className="chips" style={{ marginTop: 8 }}>
      {g.refused && <span className="pill bad">Refused: {g.refusal_reason}</span>}
      {(g.pii_masked_input ?? 0) > 0 && <span className="pill warn">PII masked ×{g.pii_masked_input}</span>}
      {!g.refused && (
        <span className={`pill ${area.compliant ? "good" : "warn"}`}
          title={area.missing_sections?.length ? `Missing: ${area.missing_sections.join(", ")}` : "All practice-area sections present"}>
          {area.compliant ? "Area guardrails met" : `Area guardrails: ${area.missing_sections?.length ?? 0} missing`}
          {area.auto_filled?.length ? " · auto-filled" : ""}
        </span>
      )}
      {(g.invalid_citations?.length ?? 0) > 0 && <span className="pill bad">Invalid citations {g.invalid_citations.join(", ")}</span>}
      {(g.injection_passages ?? 0) > 0 && <span className="pill warn">Injection passage disregarded</span>}
      {r.selected_nodes > 0 && <span className="pill accent">Scoped to {r.selected_nodes} selected nodes</span>}
      {!g.refused && (
        <>
          <span className="pill" title={k.rcr_method}>RCR {fmt(k.rcr)}{k.rcr_method?.startsWith("estimated") ? " (est.)" : ""}</span>
          <span className="pill" title={k.faithfulness_method}>Faithfulness {k.abstained ? "abstained" : fmt(k.faithfulness)}</span>
          {r.engine === "graph" && <span className="pill">ER F1 {fmt(k.er_accuracy)}</span>}
          <span className="pill" title={`retrieval ${ms(k.latency?.retrieval_ms)} · graph ${ms(k.latency?.graph_ms)} · LLM ${ms(k.latency?.llm_inference_ms)}`}>
            {ms(k.latency_ms)} (retrieval {ms(k.latency?.retrieval_ms)})
          </span>
          {r.loops > 0 && <span className="pill accent">Cyclic loop: k → {r.hop_k_used}</span>}
          {r.fallback_used && <span className="pill warn">Server-side fallback model used</span>}
          {(r.usage?.cost_usd ?? 0) > 0 && <span className="pill">${r.usage.cost_usd.toFixed(4)}</span>}
        </>
      )}
    </div>
  );
}
