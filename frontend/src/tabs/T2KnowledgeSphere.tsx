import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import SpriteText from "three-spritetext";
import { getJSON } from "../api/client";
import { paletteColor, typeColor } from "../lib/colors";
import { fibonacciSphere } from "../lib/fibonacciSphere";
import { setState, useApp } from "../store/useAppStore";
import type { GraphLink, GraphNode } from "../types";

type ColorBy = "type" | "community" | "n2v_cluster";
const STRUCT = new Set(["MENTIONED_IN", "PART_OF"]);
const endId = (e: string | GraphNode) => (typeof e === "string" ? e : e.id);
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

export default function T2KnowledgeSphere() {
  const area = useApp((s) => s.area);
  const graphVersion = useApp((s) => s.graphVersion);
  const scopeOn = useApp((s) => s.useSelectionScope);
  const [allAreas, setAllAreas] = useState(false);
  const [chunks, setChunks] = useState(false);
  const [data, setData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({ nodes: [], links: [] });
  const [er, setEr] = useState<any>(null);
  const [off, setOff] = useState<Set<string>>(new Set());
  const [offLinks, setOffLinks] = useState<Set<string>>(new Set());
  const [colorBy, setColorBy] = useState<ColorBy>("type");
  const [mode, setMode] = useState<"dim" | "hide">("dim");
  const [rotate, setRotate] = useState(true);
  const [search, setSearch] = useState("");
  const [focus, setFocus] = useState<GraphNode | null>(null);
  const [err, setErr] = useState("");
  const fgRef = useRef<any>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 640 });
  const dark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    setErr("");
    getJSON<{ nodes: GraphNode[]; links: GraphLink[]; er: any }>(`/api/graph?area=${allAreas ? "all" : area}&include_chunks=${chunks}`)
      .then((g) => {
        const nodes = [...g.nodes].sort((a, b) => (a.community - b.community) || (b.pagerank - a.pagerank));
        const R = 60 + Math.sqrt(nodes.length) * 9;
        const pts = fibonacciSphere(nodes.length, R);
        nodes.forEach((n, i) => { n.fx = n.x = pts[i].x; n.fy = n.y = pts[i].y; n.fz = n.z = pts[i].z; });
        setData({ nodes, links: g.links });
        setEr(g.er);
        setOff(new Set());
        setOffLinks(new Set());
        setState({ selectedNodeIds: [], blockedEdgeIds: [], useSelectionScope: false });
      })
      .catch((e) => setErr(String(e.message ?? e)));
  }, [area, allAreas, chunks, graphVersion]);

  const radius = useMemo(() => 60 + Math.sqrt(data.nodes.length) * 9, [data.nodes.length]);
  const nodeById = useMemo(() => new Map(data.nodes.map((n) => [n.id, n])), [data.nodes]);
  const nodeOn = useCallback((n: GraphNode) => !off.has(n.id), [off]);
  const linkOn = useCallback((l: GraphLink) => !offLinks.has(l.id) && !off.has(endId(l.source)) && !off.has(endId(l.target)), [off, offLinks]);

  // publish the selection to the global store (an all-on selection means "no scope")
  useEffect(() => {
    const ids = off.size === 0 ? [] : data.nodes.filter((n) => !off.has(n.id)).map((n) => n.id);
    const edges = data.links.filter((l) => offLinks.has(l.id)).map((l) => l.id);
    setState({ selectedNodeIds: ids, blockedEdgeIds: edges });
  }, [off, offLinks, data]);

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg || !data.nodes.length) return;
    const scene = fg.scene();
    scene.getObjectByName("sphere-shell") && scene.remove(scene.getObjectByName("sphere-shell"));
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.985, 36, 24),
      new THREE.MeshBasicMaterial({ color: dark ? 0xf07a3a : 0xd04a02, wireframe: true, transparent: true, opacity: 0.05 }));
    mesh.name = "sphere-shell";
    scene.add(mesh);
    fg.cameraPosition({ x: 0, y: radius * 0.4, z: radius * 2.6 });
  }, [radius, data, dark]);

  useEffect(() => {
    const c = fgRef.current?.controls?.();
    if (c) { c.autoRotate = rotate; c.autoRotateSpeed = 0.5; }
  }, [rotate, data]);

  const neighbours = useCallback((id: string) => {
    const out = new Set<string>();
    data.links.forEach((l) => {
      if (endId(l.source) === id) out.add(endId(l.target));
      if (endId(l.target) === id) out.add(endId(l.source));
    });
    return out;
  }, [data.links]);

  const toggle = (set: Set<string>, ids: string[], setter: (s: Set<string>) => void, forceOff?: boolean) => {
    const n = new Set(set);
    const turnOff = forceOff ?? !ids.every((i) => n.has(i));
    ids.forEach((i) => (turnOff ? n.add(i) : n.delete(i)));
    setter(n);
  };

  const onNodeClick = (n: GraphNode, ev: MouseEvent) => {
    setFocus(n);
    if (ev.shiftKey) toggle(off, [n.id, ...neighbours(n.id)], setOff);
    else toggle(off, [n.id], setOff);
  };

  const color = (n: GraphNode) => {
    if (!nodeOn(n)) return dark ? "#3b3936" : "#d8d4ce";
    if (colorBy === "community") return paletteColor(n.community);
    if (colorBy === "n2v_cluster") return paletteColor(n.n2v_cluster ?? 0);
    return typeColor(n.type);
  };

  const topLabels = useMemo(() => new Set([...data.nodes].sort((a, b) => b.pagerank - a.pagerank).slice(0, 40).map((n) => n.id)), [data.nodes]);
  const nodeObject = useCallback((n: GraphNode) => {
    if (!(topLabels.has(n.id) || focus?.id === n.id) || (mode === "hide" && !nodeOn(n))) return null as any;
    const s = new SpriteText(n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label);
    s.color = nodeOn(n) ? (dark ? "#ece9e4" : "#1f1d1a") : "#8a857f";
    s.textHeight = 3;
    s.backgroundColor = dark ? "rgba(29,28,26,0.7)" : "rgba(255,255,255,0.75)";
    s.padding = 1;
    (s as any).position.set(0, 6 + Math.sqrt(n.pagerank * 3000), 0);
    return s;
  }, [topLabels, focus, mode, nodeOn, dark]);

  const types = useMemo(() => {
    const m = new Map<string, number>();
    data.nodes.forEach((n) => m.set(n.type, (m.get(n.type) ?? 0) + 1));
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, [data.nodes]);
  const rels = useMemo(() => {
    const m = new Map<string, number>();
    data.links.forEach((l) => m.set(l.rel, (m.get(l.rel) ?? 0) + 1));
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, [data.links]);
  const list = useMemo(() => {
    const s = search.toLowerCase();
    return data.nodes.filter((n) => !s || n.label.toLowerCase().includes(s) || n.aliases?.toLowerCase().includes(s))
      .sort((a, b) => b.pagerank - a.pagerank).slice(0, 200);
  }, [data.nodes, search]);
  const onCount = data.nodes.length - off.size;
  const onLinks = data.links.filter(linkOn).length;

  return (
    <div className="split">
      <div className="col">
        <div className="row">
          <span className="pill">{onCount}/{data.nodes.length} nodes · {onLinks}/{data.links.length} edges active</span>
          <label className="check small"><input type="checkbox" checked={allAreas} onChange={(e) => setAllAreas(e.target.checked)} /> all practice areas</label>
          <label className="check small"><input type="checkbox" checked={chunks} onChange={(e) => setChunks(e.target.checked)} /> show passage & document nodes</label>
          <span className="small muted">Colour by</span>
          <div className="seg">
            {(["type", "community", "n2v_cluster"] as ColorBy[]).map((c) => (
              <button key={c} className={colorBy === c ? "on" : ""} onClick={() => setColorBy(c)}>
                {c === "n2v_cluster" ? "Node2Vec cluster" : c}
              </button>
            ))}
          </div>
          <div className="seg">
            <button className={mode === "dim" ? "on" : ""} onClick={() => setMode("dim")}>Dim unselected</button>
            <button className={mode === "hide" ? "on" : ""} onClick={() => setMode("hide")}>Hide unselected</button>
          </div>
          <button className="btn small" onClick={() => setRotate(!rotate)}>{rotate ? "Pause rotation" : "Rotate"}</button>
        </div>
        <div className="canvas" ref={wrapRef}>
          {err && <div className="empty">{err}</div>}
          {!err && data.nodes.length === 0 && <div className="empty">No knowledge graph yet. Load the gold corpus on Tab 1.</div>}
          {data.nodes.length > 0 && (
            <ForceGraph3D
              ref={fgRef}
              width={size.w}
              height={size.h}
              graphData={data as any}
              backgroundColor={dark ? "#1d1c1a" : "#ffffff"}
              showNavInfo={false}
              cooldownTicks={0}
              warmupTicks={0}
              enableNodeDrag={false}
              nodeRelSize={3}
              nodeVal={(n: any) => (n.type === "Document" ? 6 : n.type === "Chunk" ? 0.6 : 1 + n.pagerank * 500)}
              nodeColor={(n: any) => color(n)}
              nodeOpacity={0.95}
              nodeResolution={10}
              nodeVisibility={(n: any) => mode === "dim" || nodeOn(n)}
              nodeLabel={(n: any) => `<div style="padding:4px 6px;background:rgba(0,0,0,.75);color:#fff;border-radius:6px;font-size:12px"><b>${esc(n.label)}</b><br/>${n.type} · degree ${n.degree} · community ${n.community}${n.aliases ? `<br/>aliases: ${esc(n.aliases.slice(0, 120))}` : ""}<br/><i>click: toggle · shift-click: toggle neighbours</i></div>`}
              nodeThreeObject={nodeObject as any}
              nodeThreeObjectExtend={true}
              linkVisibility={(l: any) => (mode === "dim" ? true : linkOn(l))}
              linkColor={(l: any) => (!linkOn(l) ? "rgba(140,140,140,0.07)"
                : focus && (endId(l.source) === focus.id || endId(l.target) === focus.id) ? "#d04a02"
                : STRUCT.has(l.rel) ? "rgba(140,140,140,0.25)" : dark ? "rgba(240,180,140,0.55)" : "rgba(120,80,40,0.45)")}
              linkWidth={(l: any) => (linkOn(l) ? (focus && (endId(l.source) === focus.id || endId(l.target) === focus.id) ? 1.6 : 0.5) : 0.1)}
              linkCurvature={0.25}
              linkDirectionalArrowLength={(l: any) => (linkOn(l) && !STRUCT.has(l.rel) ? 2.5 : 0)}
              linkDirectionalArrowRelPos={0.9}
              linkLabel={(l: any) => `<div style="padding:4px 6px;background:rgba(0,0,0,.75);color:#fff;border-radius:6px;font-size:12px">${esc(nodeById.get(endId(l.source))?.label ?? "")} <b>${l.rel}</b> ${esc(nodeById.get(endId(l.target))?.label ?? "")}<br/><i>click: toggle edge</i></div>`}
              onNodeClick={onNodeClick as any}
              onLinkClick={(l: any) => toggle(offLinks, [l.id], setOffLinks)}
              onBackgroundClick={() => setFocus(null)}
            />
          )}
        </div>
        <div className="legend">
          {colorBy === "type" && types.map(([t]) => <span key={t}><i style={{ background: typeColor(t) }} />{t}</span>)}
          {colorBy !== "type" && <span className="muted">Colours are {colorBy === "community" ? "Louvain communities" : "k-means clusters of the 64-d Node2Vec embeddings"}; nodes are placed on the sphere grouped by community.</span>}
        </div>
      </div>

      <div className="col">
        <div className="card col">
          <h3>Selection → retrieval scope</h3>
          <label className="check">
            <input type="checkbox" checked={scopeOn} disabled={off.size === 0 && offLinks.size === 0}
              onChange={(e) => setState({ useSelectionScope: e.target.checked })} />
            Use selection as the active retrieval context
          </label>
          <div className="small muted">
            When on, chat retrieval only uses passages linked to the selected nodes, and Graph RAG only traverses selected
            nodes and active edges.
          </div>
          <div className="row">
            <button className="btn small" onClick={() => setOff(new Set())}>All</button>
            <button className="btn small" onClick={() => setOff(new Set(data.nodes.map((n) => n.id)))}>None</button>
            <button className="btn small" onClick={() => setOff(new Set(data.nodes.filter((n) => !off.has(n.id)).map((n) => n.id)))}>Invert</button>
            <button className="btn small" onClick={() => setOffLinks(new Set())}>All edges on</button>
            {focus && <button className="btn small" onClick={() => setOff(new Set(data.nodes.filter((n) => n.id !== focus.id && !neighbours(focus.id).has(n.id)).map((n) => n.id)))}>
              Only {focus.label.slice(0, 18)} + neighbours</button>}
          </div>
          {er?.final && (
            <div className="small">
              <b>Entity resolution F1</b> vs gold alias clusters:{" "}
              {Object.entries(er.variants ?? {}).map(([k, v]: any) => `${k}: ${v.f1}`).join(" · ")}
            </div>
          )}
        </div>
        <div className="card col">
          <h4 style={{ margin: 0 }}>Entity types</h4>
          <div className="chips">
            {types.map(([t, n]) => {
              const ids = data.nodes.filter((x) => x.type === t).map((x) => x.id);
              const allOff = ids.every((i) => off.has(i));
              return <button key={t} className={`chip ${allOff ? "" : "on"}`} onClick={() => toggle(off, ids, setOff, !allOff)}>{t} {n}</button>;
            })}
          </div>
          <h4 style={{ margin: 0 }}>Relation types</h4>
          <div className="chips">
            {rels.map(([r, n]) => {
              const ids = data.links.filter((l) => l.rel === r).map((l) => l.id);
              const allOff = ids.every((i) => offLinks.has(i));
              return <button key={r} className={`chip ${allOff ? "" : "on"}`} onClick={() => toggle(offLinks, ids, setOffLinks, !allOff)}>{r} {n}</button>;
            })}
          </div>
        </div>
        <div className="card col">
          <input type="text" placeholder="Search entities or aliases…" value={search} onChange={(e) => setSearch(e.target.value)} />
          <div className="scroll" style={{ maxHeight: 300 }}>
            {list.map((n) => (
              <label key={n.id} className="check small" style={{ display: "flex" }}>
                <input type="checkbox" checked={nodeOn(n)} onChange={() => toggle(off, [n.id], setOff)} />
                <i style={{ display: "inline-block", width: 8, height: 8, borderRadius: 4, background: typeColor(n.type) }} />
                <span>{n.label}</span><span className="muted">· {n.type}</span>
              </label>
            ))}
          </div>
        </div>
        {focus && (
          <div className="card small">
            <b>{focus.label}</b> <span className="muted">({focus.type})</span>
            <div className="kv">
              <div>Aliases</div><div>{focus.aliases || "—"}</div>
              <div>Mentions</div><div>{focus.mentions}</div>
              <div>PageRank</div><div>{focus.pagerank}</div>
              <div>Community</div><div>{focus.community}</div>
              <div>Neighbours</div><div>{neighbours(focus.id).size}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
