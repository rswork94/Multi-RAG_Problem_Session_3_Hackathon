"""Steps 4-6 helpers: cached embedding, dual index build (HNSW vectors + graph adjacency + BM25) and persistence."""
from __future__ import annotations

import hashlib
import re
import json
import pickle

import networkx as nx
import numpy as np

from .. import runtime
from ..config import settings
from ..retrieval.stores import KB, atomic_write, legal_tokens
from .resolve import SEED_ALIASES, norm


# ---------------------------------------------------------------- embedding with a content-hash cache
def _cache_file():
    return settings.path(f"emb_cache_{settings.embed_model.replace('/', '_')}{'_fake' if settings.fake_embeddings else ''}.npz")


def embed_cached(texts: list[str], progress=None) -> tuple[np.ndarray, int]:
    """Returns (vectors, n_newly_embedded). Re-ingesting unchanged text costs nothing."""
    f = _cache_file()
    cache: dict[str, np.ndarray] = {}
    if f.exists():
        z = np.load(f, allow_pickle=False)
        cache = dict(zip(json.loads(str(z["keys"])), z["vecs"]))
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    for s in range(0, len(todo), 32):
        idx = todo[s:s + 32]
        vecs = runtime.embed_passages([texts[i] for i in idx])
        for i, v in zip(idx, vecs):
            cache[keys[i]] = v
        if progress:
            progress(min(s + 32, len(todo)), len(todo))
    if todo:
        ks = list(cache)
        np.savez(f, keys=json.dumps(ks), vecs=np.stack([cache[k] for k in ks]))
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32), 0
    return np.stack([cache[k] for k in keys]), len(todo)


# ---------------------------------------------------------------- index build
def build_graph(documents: list[dict], chunks: list[dict], entities: list[dict], relations: list[dict]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for d in documents:
        g.add_node(f"doc:{d['doc_id']}", label=d["title"][:70], type="Document", area=d.get("practice_area", ""),
                   source=d["source"], doc_type=d.get("doc_type", ""))
    for c in chunks:
        cid = f"chunk:{c['chunk_id']}"
        g.add_node(cid, label=(c.get("section") or c["text"][:50])[:70], type="Chunk", area=c["practice_area"],
                   chunk_id=c["chunk_id"], source=c["source"])
        g.add_edge(cid, f"doc:{c['doc_id']}", key="PART_OF", rel="PART_OF")
    for e in entities:
        g.add_node(e["id"], label=e["label"], type=e["type"], aliases=" | ".join(e["aliases"][:12]),
                   area=",".join(e["areas"]), mentions=e["mentions"])
        for ch in e["chunks"]:
            g.add_edge(e["id"], f"chunk:{ch}", key="MENTIONED_IN", rel="MENTIONED_IN")
    for r in relations:
        g.add_edge(r["src"], r["dst"], key=r["rel"], rel=r["rel"], weight=r["weight"], evidence=",".join(r["evidence"][:8]))
    annotate(g)
    return g


def annotate(g: nx.MultiDiGraph) -> None:
    if g.number_of_nodes() == 0:
        return
    ug = nx.Graph()
    ug.add_nodes_from(g.nodes)
    for u, v, d in g.edges(data=True):
        w = (2 if d.get("rel") not in ("MENTIONED_IN", "PART_OF") else 1)
        ug.add_edge(u, v, weight=ug[u][v]["weight"] + w if ug.has_edge(u, v) else w)
    try:
        comms = nx.community.louvain_communities(ug, weight="weight", seed=42)
    except Exception:
        comms = [set(c) for c in nx.connected_components(ug)]
    for i, cset in enumerate(sorted(comms, key=len, reverse=True)):
        for n in cset:
            g.nodes[n]["community"] = i
    try:
        pr = nx.pagerank(ug, weight="weight")
    except Exception:
        pr = {n: 1 / g.number_of_nodes() for n in g.nodes}
    for n in g.nodes:
        g.nodes[n]["pagerank"] = round(pr.get(n, 0.0), 6)
        g.nodes[n]["degree"] = ug.degree(n)


def structural_clusters(n2v: dict[str, np.ndarray], k: int = 8) -> dict[str, int]:
    ids = sorted(n2v)
    if len(ids) < k:
        return {i: 0 for i in ids}
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=k, n_init=4, random_state=42).fit_predict(np.stack([n2v[i] for i in ids]))
    return dict(zip(ids, map(int, labels)))


def build_alias_index(entities: list[dict]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for e in sorted(entities, key=lambda e: -e["mentions"]):
        names = [e["label"]] + e["aliases"]
        if e["type"] == "Form":  # 'Form FC-TRS' is also written 'FC-TRS'
            names += [re.sub(r"(?i)^form\s+", "", n) for n in names]
        for n in names:
            if len(norm(n)) >= 2:
                idx.setdefault(norm(n), e["id"])
    for group in SEED_ALIASES:
        hit = next((idx[norm(m)] for m in group if norm(m) in idx), None)
        if hit:
            for m in group:
                idx.setdefault(norm(m), hit)
    return idx


def persist_all(documents, chunks, child_ids, child_vecs, child_meta, child_docs, entities, ent_vecs, graph, n2v,
                er_report, stats) -> dict:
    """Step 6: write the vector DB (Chroma) and graph engine files atomically, then reload the KB."""
    KB.children.reset()
    KB.entity_names.reset()
    KB.children.upsert(child_ids, child_vecs, child_meta, child_docs)
    if entities:
        KB.entity_names.upsert([e["id"] for e in entities], ent_vecs,
                               [{"type": e["type"], "label": e["label"], "areas": ",".join(e["areas"])} for e in entities],
                               [e["label"] for e in entities])
    bm25: dict[str, tuple] = {}
    from rank_bm25 import BM25Okapi
    for area in sorted({c["practice_area"] for c in chunks}) + ["all"]:
        sel = [i for i, m in enumerate(child_meta) if area == "all" or m["area"] == area]
        if sel:
            bm25[area] = (BM25Okapi([legal_tokens(f"{child_meta[i].get('header', '')} {child_docs[i]}") for i in sel]),
                          [child_meta[i]["parent_id"] for i in sel], [child_docs[i] for i in sel])
    clusters = structural_clusters(n2v)
    for nid, cl in clusters.items():
        if nid in graph:
            graph.nodes[nid]["n2v_cluster"] = cl
    parents = {c["chunk_id"]: c for c in chunks}
    atomic_write(settings.path("parents.json"), json.dumps(parents))
    atomic_write(settings.path("documents.json"), json.dumps([{k: v for k, v in d.items() if k not in ("text", "rows", "page_spans")}
                                                              for d in documents]))
    atomic_write(settings.path("bm25.pkl"), pickle.dumps(bm25))
    KB.graph_store.save(graph)
    ids = sorted(n2v)
    if ids:
        np.savez(settings.path("n2v.npz"), ids=json.dumps(ids), vecs=np.stack([n2v[i] for i in ids]))
    atomic_write(settings.path("alias_index.json"), json.dumps(build_alias_index(entities)))
    atomic_write(settings.path("er_report.json"), json.dumps(er_report))
    atomic_write(settings.path("ingest_stats.json"), json.dumps(stats, default=str))
    KB.load()
    size = sum(p.stat().st_size for p in settings.data_dir.rglob("*") if p.is_file())
    return {"vectors": KB.children.count(), "entity_vectors": KB.entity_names.count(), "parents": len(parents),
            "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
            "bm25_indexes": sorted(bm25), "disk_mb": round(size / 1e6, 1)}
