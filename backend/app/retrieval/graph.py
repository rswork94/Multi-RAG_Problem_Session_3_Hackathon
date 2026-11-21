"""Graph side of Graph RAG: query entity linking, k-hop traversal over the entity-relation adjacency index,
and the deterministic sufficiency check that drives the cyclic loop."""
from __future__ import annotations

import re
import time
from collections import deque

import numpy as np

from ..config import Const
from ..ingest.extract import regex_extract
from ..ingest.resolve import canonical, norm
from .stores import KB

STRUCTURAL = {"MENTIONED_IN", "PART_OF"}
ANSWER_RELS = {"HAS_DEADLINE", "PRESCRIBES_RATE", "IMPOSES_PENALTY", "CAPPED_BY", "PRESCRIBES_FORM", "SUPERSEDES",
               "AMENDS", "TRIGGERS", "REQUIRES", "MAPS_TO", "APPLIES_TO", "DEFINES"}
ANSWER_TYPES = [
    ("Deadline", r"\bwhen\b|due date|deadline|by when|time limit|within (how|what)|long-?stop|interval|how often|periodicity|date"),
    ("Rate", r"\brate\b|per ?cent|percentage|%"),
    ("Penalty", r"penalt|\bfine\b|consequence|sanction|punish"),
    ("Threshold", r"\bcap\b|capped|limit|threshold|how much|ceiling|exceed|turnover|escrow|basket|de minimis|amount|minimum price|floor"),
    ("Form", r"\bform\b|which (return|statement)"),
]
NAME_LINK_MIN = 0.70
HUB_DEGREE = 30


def graph_trust(linked: list[dict]) -> float:
    """How much the fused ranking should trust the graph list: full when the question itself names an entity
    (identifier or alias), lower when the only seeds come from entity-name vectors or from retrieved passages."""
    hows = {l["how"] for l in linked}
    if hows & {"identifier", "identifier-alias", "alias"}:
        return 1.0
    if "name-vector" in hows:
        return 0.6
    return 0.4 if linked else 0.0


def answer_types(question: str) -> set[str]:
    q = question.lower()
    return {t for t, rx in ANSWER_TYPES if re.search(rx, q)}


def link_entities(question: str, qvec: np.ndarray | None, area: str | None) -> list[dict]:
    g = KB.graph
    linked: dict[str, dict] = {}

    weight = {"identifier": 1.0, "identifier-alias": 1.0, "alias": 0.9, "name-vector": 0.6, "passage": 0.3}

    def add(eid: str, how: str, score: float = 1.0) -> None:
        if eid in g and eid not in linked and (not area or area == "all" or area in (g.nodes[eid].get("area") or "")):
            linked[eid] = {"id": eid, "label": g.nodes[eid].get("label", eid), "type": g.nodes[eid].get("type"),
                           "how": how, "score": round(score, 3), "weight": round(weight[how] * score, 3)}

    # 1) identifiers in the question (regex + canonical form) -> exact entity ids or alias index
    for e in regex_extract(question)["entities"]:
        c = canonical(e["name"], e["type"])
        add(f"ent:{e['type']}:{c.lower()}", "identifier")
        if norm(c) in KB.alias_index:
            add(KB.alias_index[norm(c)], "identifier-alias")
    # 2) alias dictionary n-grams (planted aliases resolved at ingest land here)
    words = re.findall(r"[\w₹./()-]+", question)
    for n in range(6, 0, -1):
        for i in range(len(words) - n + 1):
            gram = norm(" ".join(words[i:i + n]))
            if len(gram) >= 4 and gram in KB.alias_index:
                add(KB.alias_index[gram], "alias")
    # 3) entity-name vectors (vocabulary mismatch), scoped to non-value types
    if qvec is not None and KB.entity_names.count():
        for eid, sim, meta, _ in KB.entity_names.query(qvec, 8):
            if sim >= NAME_LINK_MIN and meta.get("type") not in ("Rate", "Threshold", "Deadline", "Penalty"):
                add(eid, "name-vector", float(sim))
            if sum(1 for v in linked.values() if v["how"] == "name-vector") >= 3:
                break
    # 4) passage seeds: specific entities in the two best-matching passages (low weight). Gives the graph a foothold
    #    for descriptive questions that name no entity at all.
    if qvec is not None:
        from .search import dense_search
        top, _ = dense_search(qvec, area, None, n=2)
        for h in top:
            ents = [u for u, _, d in g.in_edges(f"chunk:{h['chunk_id']}", data=True) if d.get("rel") == "MENTIONED_IN"]
            ents = [e for e in ents if g.nodes[e].get("degree", 0) <= HUB_DEGREE
                    and g.nodes[e].get("type") not in ("Authority", "Act", "Rate", "Threshold", "Deadline", "Penalty")]
            for e in sorted(ents, key=lambda e: g.nodes[e].get("degree", 0))[:4]:
                add(e, "passage", 1.0)
    return list(linked.values())[:14]


def _n2v_sim(a: str, seeds: list[str]) -> float:
    va = KB.n2v.get(a)
    if va is None:
        return 0.0
    sims = [float(va @ KB.n2v[s]) for s in seeds if s in KB.n2v]
    return max(sims, default=0.0)


def traverse(seeds: list[str], k: int, area: str | None, allowed_nodes: set[str] | None,
             chunk_scope: set[str] | None, blocked: set[str] | None = None) -> tuple[list[dict], list[dict], dict, float]:
    """BFS to depth k over entity-entity relations (both directions), fan-out capped by PageRank.
    Returns (ranked chunk hits, triples, reached {entity: hop}, ms)."""
    t0 = time.perf_counter()
    g = KB.graph
    seeds = [s for s in seeds if s in g and (allowed_nodes is None or s in allowed_nodes)]
    hop: dict[str, int] = {s: 0 for s in seeds}
    triples: list[dict] = []
    edge_evidence: dict[str, int] = {}
    dq = deque(seeds)
    while dq:
        n = dq.popleft()
        if hop[n] >= k:
            continue
        if hop[n] > 0 and g.nodes[n].get("degree", 0) > HUB_DEGREE:
            continue  # do not expand through hubs (e.g. an authority linked to every instrument)
        nbrs = []
        for _, v, d in g.out_edges(n, data=True):
            if d.get("rel") not in STRUCTURAL:
                nbrs.append((v, d, n, v))
        for u, _, d in g.in_edges(n, data=True):
            if d.get("rel") not in STRUCTURAL:
                nbrs.append((u, d, u, n))
        if allowed_nodes is not None:
            nbrs = [x for x in nbrs if x[0] in allowed_nodes]
        if blocked:
            nbrs = [x for x in nbrs if f"{x[2]}|{x[1].get('rel')}|{x[3]}" not in blocked]
        # answer-bearing relations first, then specific (low-degree) neighbours before hubs
        nbrs.sort(key=lambda x: (x[1].get("rel") not in ANSWER_RELS, g.nodes[x[0]].get("degree", 0)))
        for v, d, a, b in nbrs[:Const.GRAPH_FANOUT]:
            if v not in hop:
                hop[v] = hop[n] + 1
                dq.append(v)
            ev = [e for e in (d.get("evidence") or "").split(",") if e]
            triples.append({"src": a, "dst": b, "rel": d.get("rel"), "hop": hop[n] + 1,
                            "src_label": g.nodes[a].get("label", a), "dst_label": g.nodes[b].get("label", b),
                            "evidence": ev})
            for e in ev:
                if e not in edge_evidence or hop[n] + 1 < edge_evidence[e][0]:
                    edge_evidence[e] = (hop[n] + 1, b)
    cand: dict[str, dict] = {}

    def consider(cid: str, h: int, why: str, ent: str | None) -> None:
        if chunk_scope is not None and cid not in chunk_scope:
            return
        if not KB.area_ok(cid, area) or cid not in KB.parents:
            return
        prio = (1.0 / (1 + h)) * (1.2 if why == "edge-evidence" else 1.0) * (0.7 + 0.3 * (_n2v_sim(ent, seeds) if ent else 1))
        if cid not in cand or prio > cand[cid]["score"]:
            cand[cid] = {"chunk_id": cid, "score": round(prio, 4), "hop": h, "why": why, "entity": ent, "list": "graph",
                         "entity_type": g.nodes[ent].get("type") if ent and ent in g else None}

    for cid, (h, tgt) in edge_evidence.items():
        consider(cid, h, "edge-evidence", tgt)
    for e, h in sorted(hop.items(), key=lambda x: (x[1], -g.nodes[x[0]].get("pagerank", 0))):
        got = 0
        for _, v, d in g.out_edges(e, data=True):
            if d.get("rel") == "MENTIONED_IN" and got < Const.CHUNKS_PER_ENTITY:
                cid = g.nodes[v].get("chunk_id")
                if cid:
                    consider(cid, h, "mention", e)
                    got += 1
    hits = sorted(cand.values(), key=lambda c: (-c["score"], c["hop"]))[:Const.CANDIDATES]
    for r, h in enumerate(hits, 1):
        h["rank"] = r
    uniq = {(t["src"], t["rel"], t["dst"]): t for t in triples}
    return hits, list(uniq.values()), hop, (time.perf_counter() - t0) * 1000


def sufficiency(question: str, seeds: list[str], reached: dict, context_ids: list[str], k: int) -> dict:
    """Deterministic, no LLM: (1) every linked seed has an evidence passage; (2) if the question asks for a value
    type (date, rate, penalty, cap, form), an entity of that type was reached and is mentioned in the context."""
    g = KB.graph
    ctx = set(context_ids)

    def mentioned(e: str) -> bool:
        return any(g.nodes[v].get("chunk_id") in ctx for _, v, d in g.out_edges(e, data=True) if d.get("rel") == "MENTIONED_IN")

    seeds_ok = all(mentioned(s) for s in seeds if s in g)
    want = answer_types(question)
    found = sorted({g.nodes[e].get("type") for e, h in reached.items() if h > 0 and g.nodes[e].get("type") in want
                    and mentioned(e)})
    frontier = [e for e, h in reached.items() if h == k and any(
        d.get("rel") in ANSWER_RELS and v not in reached for _, v, d in g.out_edges(e, data=True))]
    answer_ok = not want or bool(found) or not seeds
    ok = (seeds_ok and answer_ok) or not frontier
    return {"sufficient": ok, "seeds_covered": seeds_ok, "wanted_types": sorted(want), "found_types": found,
            "frontier": len(frontier), "k": k}


# ---------------------------------------------------------------- Personalized PageRank (HippoRAG-style ranking)
PPR_ALPHA = 0.5     # continuation probability; restart 0.5 keeps the walk local to the query's entities


def _hub_damp(n: str) -> float:
    import math
    return 1.0 / math.log2(2 + KB.graph.nodes[n].get("degree", 0))


def ppr_rank(seeds: dict[str, float], k: int, area: str | None, allowed_nodes: set[str] | None,
             chunk_scope: set[str] | None, blocked: set[str] | None = None) -> tuple[list[dict], float]:
    """Rank passages by Personalized PageRank over the k-hop entity subgraph around the seeds.

    Nodes: entities within k relation-hops of a seed, plus the passages that mention them. Edges: entity-relation-
    entity and entity-MENTIONED_IN-passage. Restart mass sits on the seeds (weighted by link confidence and damped for
    hubs), so a passage scores high when several query-related entities - including bridge entities k hops away -
    point to it. That is exactly the multi-hop signal a flat vector search lacks."""
    import networkx as nx

    t0 = time.perf_counter()
    g = KB.graph
    seeds = {s: w for s, w in seeds.items() if s in g and (allowed_nodes is None or s in allowed_nodes)}
    if not seeds or k < 0:
        return [], 0.0
    hop: dict[str, int] = {s: 0 for s in seeds}
    frontier = list(seeds)
    for h in range(1, k + 1):
        nxt = []
        for n in frontier:
            if h > 1 and g.nodes[n].get("degree", 0) > HUB_DEGREE:
                continue
            for a, b, d in list(g.out_edges(n, data=True)) + list(g.in_edges(n, data=True)):
                if d.get("rel") in STRUCTURAL:
                    continue
                v = b if a == n else a
                if v in hop or (allowed_nodes is not None and v not in allowed_nodes):
                    continue
                if blocked and f"{a}|{d.get('rel')}|{b}" in blocked:
                    continue
                hop[v] = h
                nxt.append(v)
        frontier = nxt
    sub = nx.Graph()
    for e in hop:
        sub.add_node(e)
        for _, v, d in g.out_edges(e, data=True):
            if d.get("rel") == "MENTIONED_IN":
                cid = g.nodes[v].get("chunk_id")
                if cid and KB.area_ok(cid, area) and (chunk_scope is None or cid in chunk_scope):
                    sub.add_edge(e, v, weight=1.0)
            elif d.get("rel") not in STRUCTURAL and v in hop:
                w = sub[e][v]["weight"] + 1.0 if sub.has_edge(e, v) else 1.0
                sub.add_edge(e, v, weight=w)
    if sub.number_of_edges() == 0:
        return [], (time.perf_counter() - t0) * 1000
    pers = {n: 0.0 for n in sub.nodes}
    for s, w in seeds.items():
        pers[s] = w * _hub_damp(s)
    if sum(pers.values()) <= 0:
        return [], (time.perf_counter() - t0) * 1000
    pr = nx.pagerank(sub, alpha=PPR_ALPHA, personalization=pers, weight="weight", max_iter=200, tol=1e-8)
    hits = []
    for n, score in pr.items():
        cid = g.nodes[n].get("chunk_id") if n in g and g.nodes[n].get("type") == "Chunk" else None
        if not cid:
            continue
        ents = [e for e in sub.neighbors(n) if e in hop]
        hits.append({"chunk_id": cid, "score": round(score * 1000, 4), "hop": min((hop[e] for e in ents), default=0),
                     "why": "ppr", "list": "graph",
                     "entity": max(ents, key=lambda e: pr.get(e, 0)) if ents else None})
    hits.sort(key=lambda h: -h["score"])
    hits = hits[:Const.CANDIDATES]
    for r, h in enumerate(hits, 1):
        h["rank"] = r
        h["entity_type"] = g.nodes[h["entity"]].get("type") if h.get("entity") else None
    return hits, (time.perf_counter() - t0) * 1000
