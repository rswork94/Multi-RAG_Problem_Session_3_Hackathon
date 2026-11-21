import { useEffect, useState } from "react";
import { getJSON, sendJSON } from "../api/client";
import { setState } from "../store/useAppStore";

interface LlmSettings {
  provider: "anthropic" | "openrouter";
  synth_model: string;
  utility_model: string;
  synth_effort: "low" | "medium" | "high";
  remember: boolean;
  has_key: boolean;
  key_hint: string;
  mode: "online" | "offline";
  env_credentials: boolean;
}
interface ModelInfo { id: string; name: string; price_in: number; price_out: number }

const PROVIDERS = [
  { id: "openrouter", label: "OpenRouter" },
  { id: "anthropic", label: "Anthropic API" },
] as const;
const DEFAULTS = {
  anthropic: { synth: "claude-opus-5-5", util: "claude-haiku-5-5" },
  openrouter: { synth: "anthropic/claude-opus-5.5", util: "anthropic/claude-haiku-5.5" },
};

/** Tab 1: choose the provider, paste a key, pick the Claude models. The key is sent once and never shown again. */
export function ModelSettings() {
  const [s, setS] = useState<LlmSettings | null>(null);
  const [provider, setProvider] = useState<"anthropic" | "openrouter">("openrouter");
  const [key, setKey] = useState("");
  const [synth, setSynth] = useState("");
  const [util, setUtil] = useState("");
  const [effort, setEffort] = useState<"low" | "medium" | "high">("medium");
  const [remember, setRemember] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [msg, setMsg] = useState<{ kind: "good" | "bad" | "info"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(true);

  useEffect(() => {
    getJSON<LlmSettings>("/api/llm/settings").then((r) => {
      setS(r);
      setProvider(r.provider);
      setSynth(r.synth_model);
      setUtil(r.utility_model);
      setEffort(r.synth_effort);
      setRemember(r.remember);
      setOpen(r.mode === "offline");
    }).catch(() => setS(null));
  }, []);

  useEffect(() => {
    setModels([]);
    getJSON<{ models: ModelInfo[] }>(`/api/llm/models?provider=${provider}`)
      .then((r) => setModels(r.models))
      .catch((e) => setMsg({ kind: "bad", text: String(e.message ?? e) }));
  }, [provider]);

  const switchProvider = (p: "anthropic" | "openrouter") => {
    setProvider(p);
    setSynth(DEFAULTS[p].synth);
    setUtil(DEFAULTS[p].util);
    setKey("");
  };

  async function save(test: boolean) {
    setBusy(true);
    setMsg(null);
    try {
      const body: Record<string, unknown> = { provider, synth_model: synth, utility_model: util, synth_effort: effort, remember };
      if (key.trim()) body.api_key = key.trim();
      else if (s && s.provider !== provider) body.api_key = "";
      const r = await sendJSON<LlmSettings>("/api/llm/settings", "PUT", body);
      setS(r);
      setKey("");
      if (test) {
        const t = await sendJSON<any>("/api/llm/test", "POST");
        setMsg({ kind: "good", text: `Connected · ${t.model} replied in ${t.ms} ms ($${(t.cost_usd ?? 0).toFixed(5)})` });
      } else {
        setMsg({ kind: "info", text: r.mode === "online" ? "Saved. Answers now use Claude." : "Saved. No key yet, so the app stays offline." });
      }
      const h = await getJSON<any>("/api/health");
      setState({ health: h });
    } catch (e: any) {
      setMsg({ kind: "bad", text: String(e.message ?? e) });
    } finally {
      setBusy(false);
    }
  }

  const price = (id: string) => {
    const m = models.find((x) => x.id === id);
    return m ? ` · $${m.price_in}/$${m.price_out} per MTok` : "";
  };

  return (
    <div className="card col">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }}>Model settings</h3>
        <div className="row">
          <span className={`pill ${s?.mode === "online" ? "good" : "warn"}`}>
            {s?.mode === "online" ? `Online · ${s.provider}${s.key_hint ? " " + s.key_hint : ""}` : "Offline"}
          </span>
          <button className="btn small" onClick={() => setOpen(!open)}>{open ? "Hide" : "Edit"}</button>
        </div>
      </div>
      {open && (
        <>
          <div className="seg" role="radiogroup" aria-label="Provider">
            {PROVIDERS.map((p) => (
              <button key={p.id} className={provider === p.id ? "on" : ""} onClick={() => switchProvider(p.id)}>{p.label}</button>
            ))}
          </div>
          <label className="col small" style={{ gap: 4 }}>
            <span>{provider === "openrouter" ? "OpenRouter API key" : "Anthropic API key"}</span>
            <input type="password" autoComplete="off" spellCheck={false} value={key} onChange={(e) => setKey(e.target.value)}
              placeholder={s?.has_key && s.provider === provider ? `saved key ${s.key_hint} (leave blank to keep)` : provider === "openrouter" ? "sk-or-v1-…" : "sk-ant-…"} />
          </label>
          <label className="col small" style={{ gap: 4 }}>
            <span>Answer model (all three engines use it, so the comparison stays fair)</span>
            <select value={synth} onChange={(e) => setSynth(e.target.value)}>
              {!models.some((m) => m.id === synth) && synth && <option value={synth}>{synth}</option>}
              {models.map((m) => <option key={m.id} value={m.id}>{m.name.replace("Anthropic: ", "")}{price(m.id)}</option>)}
            </select>
          </label>
          <label className="col small" style={{ gap: 4 }}>
            <span>Utility model (HyDE passages, graph extraction, faithfulness judge)</span>
            <select value={util} onChange={(e) => setUtil(e.target.value)}>
              {!models.some((m) => m.id === util) && util && <option value={util}>{util}</option>}
              {models.map((m) => <option key={m.id} value={m.id}>{m.name.replace("Anthropic: ", "")}{price(m.id)}</option>)}
            </select>
          </label>
          <div className="row small">
            <span>Answer effort</span>
            <div className="seg">
              {(["low", "medium", "high"] as const).map((x) => (
                <button key={x} className={effort === x ? "on" : ""} onClick={() => setEffort(x)}>{x}</button>
              ))}
            </div>
            <label className="check"><input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />Remember key on this machine</label>
          </div>
          <div className="row">
            <button className="btn primary" disabled={busy} onClick={() => save(true)}>Save & test</button>
            <button className="btn" disabled={busy} onClick={() => save(false)}>Save</button>
          </div>
          <div className="small muted">
            The key goes only to the local backend, which uses it to call {provider === "openrouter" ? "openrouter.ai" : "api.anthropic.com"}.
            It is never shown again{remember ? " and is stored in backend/data/llm_settings.json (owner-only, git-ignored)" : " and lives in memory until the server restarts"}.
          </div>
        </>
      )}
      {msg && <div className="small" style={{ color: msg.kind === "bad" ? "var(--bad)" : msg.kind === "good" ? "var(--good)" : "var(--muted)" }}>{msg.text}</div>}
    </div>
  );
}
