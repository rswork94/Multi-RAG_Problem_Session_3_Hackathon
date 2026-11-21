import { setState, useApp } from "../store/useAppStore";
import type { Area } from "../types";

const AREAS: { id: Area; label: string; rules: string }[] = [
  { id: "tax", label: "Tax & Regulatory", rules: "Cite exact Section/Rule numbers · prospective vs retrospective" },
  { id: "grc", label: "Risk & GRC", rules: "Flag penalties immediately · map controls to obligations" },
  { id: "deals", label: "Deal Advisory", rules: "Highlight indemnities, MAC clauses, exposure thresholds" },
];

/** Global practice-area switch: activates that area's guardrails and scopes retrieval to its documents. */
export function PracticeAreaSwitcher() {
  const area = useApp((s) => s.area);
  const current = AREAS.find((a) => a.id === area)!;
  return (
    <div className="row" title={`Active guardrails: ${current.rules}`}>
      <span className="small muted">Practice area</span>
      <div className="seg">
        {AREAS.map((a) => (
          <button key={a.id} className={area === a.id ? "on" : ""} onClick={() => setState({ area: a.id })}>
            {a.label}
          </button>
        ))}
      </div>
    </div>
  );
}
