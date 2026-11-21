import type { Passage } from "../types";

/** Provenance drawer for a cited passage. */
export function SourceDrawer({ passage, onClose }: { passage: Passage | null; onClose: () => void }) {
  if (!passage) return null;
  const ranks = Object.entries(passage.ranks ?? {}).map(([k, v]) => `${k} #${v}`).join(" · ");
  return (
    <aside className="drawer" role="dialog" aria-label="Source passage">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }}>[{passage.n}] {passage.source}</h3>
        <button className="btn small" onClick={onClose}>Close</button>
      </div>
      <div className="kv">
        <div>Title</div><div>{passage.title}</div>
        <div>Type / issued</div><div>{passage.doc_type} · {passage.issued_date || "date not detected"}</div>
        <div>Section</div><div>{passage.section || "—"}</div>
        <div>Page</div><div>{passage.page}</div>
        <div>Practice area</div><div>{passage.practice_area}</div>
        <div>Chunk id</div><div className="mono">{passage.chunk_id}</div>
        <div>Tokens</div><div>{passage.n_tokens}</div>
        <div>Retrieved via</div><div>{ranks || "—"}{passage.hop != null ? ` · graph hop ${passage.hop}` : ""}</div>
        {passage.rerank != null && (<><div>Rerank score</div><div>{passage.rerank}</div></>)}
      </div>
      {passage.injection_flag && (
        <p className="banner" style={{ marginTop: 10 }}>
          This passage contains instruction-like text. It was down-weighted, shown to the model as data, and disregarded.
        </p>
      )}
      <pre>{passage.text}</pre>
    </aside>
  );
}
