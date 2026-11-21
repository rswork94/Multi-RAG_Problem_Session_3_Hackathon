"""Builds the three engine StateGraphs and runs queries through them.

Standard:  input_guard -> embed_query -> dense -> pack_context -> synthesize -> output_guard -> kpi
HyDE:      input_guard -> embed_query -> hyde_generate -> multi_dense -> rrf -> pack_context -> ...
Graph RAG: input_guard -> embed_query -> query_expansion -> {bm25 | dense | graph_traverse} -> rrf -> rerank
           -> sufficiency -(insufficient, once)-> graph_traverse (k+1) ... -> pack_context -> ...
All:       input_guard -(out of scope | integrity risk)-> refuse -> kpi
"""
from __future__ import annotations

import time
from typing import Callable

from langgraph.graph import END, START, StateGraph

from ..config import Const, hyperparams
from ..eval import gold as gold_mod
from ..eval import history
from ..guardrails import output_checks
from ..llm import client
from ..retrieval.stores import KB
from . import nodes as N
from .state import QueryState, merge_dict
from .toggles import OPTIONAL, normalise

ENGINE_INFO = {
    "standard": {"label": "Standard RAG", "blurb": "Dense vector top-K over bge-large child windows."},
    "hyde": {"label": "HyDE RAG", "blurb": "N hypothetical passages from distinct regulatory angles, fused by RRF."},
    "graph": {"label": "Graph RAG", "blurb": "Entity linking, BM25 + dense + k-hop graph traversal, RRF, rerank, "
                                             "cyclic sufficiency loop."},
}
LAST_TRACE: dict[str, dict] = {}


def _tail(sg: StateGraph) -> None:
    for name, fn in (("pack_context", N.pack_context), ("synthesize", N.synthesize), ("output_guard", N.output_guard),
                     ("refuse", N.refuse), ("kpi", N.kpi)):
        sg.add_node(name, fn)
    sg.add_edge("pack_context", "synthesize")
    sg.add_edge("synthesize", "output_guard")
    sg.add_edge("output_guard", "kpi")
    sg.add_edge("refuse", "kpi")
    sg.add_edge("kpi", END)


def _head(sg: StateGraph) -> None:
    sg.add_node("input_guard", N.input_guard)
    sg.add_node("embed_query", N.embed_query)
    sg.add_edge(START, "input_guard")
    sg.add_conditional_edges("input_guard", N.route_after_guard, {"retrieve": "embed_query", "refuse": "refuse"})


def build_standard():
    sg = StateGraph(QueryState)
    _head(sg)
    sg.add_node("dense", N.dense)
    _tail(sg)
    sg.add_edge("embed_query", "dense")
    sg.add_edge("dense", "pack_context")
    return sg.compile()


def build_hyde():
    sg = StateGraph(QueryState)
    _head(sg)
    sg.add_node("hyde_generate", N.hyde_generate)
    sg.add_node("multi_dense", N.multi_dense)
    sg.add_node("rrf", N.rrf)
    _tail(sg)
    sg.add_edge("embed_query", "hyde_generate")
    sg.add_edge("hyde_generate", "multi_dense")
    sg.add_edge("multi_dense", "rrf")
    sg.add_edge("rrf", "pack_context")
    return sg.compile()


def build_graph():
    sg = StateGraph(QueryState)
    _head(sg)
    for name, fn in (("query_expansion", N.query_expansion), ("bm25", N.bm25), ("dense", N.dense),
                     ("graph_traverse", N.graph_traverse), ("rrf", N.rrf), ("rerank", N.rerank),
                     ("sufficiency", N.sufficiency)):
        sg.add_node(name, fn)
    _tail(sg)
    sg.add_edge("embed_query", "query_expansion")
    for b in ("bm25", "dense", "graph_traverse"):
        sg.add_edge("query_expansion", b)
        sg.add_edge(b, "rrf")  # plain edges (not a barrier join) so a bypassed branch can never hang the join
    sg.add_edge("rrf", "rerank")
    sg.add_edge("rerank", "sufficiency")
    sg.add_conditional_edges("sufficiency", N.route_after_sufficiency,
                             {"graph_traverse": "graph_traverse", "pack_context": "pack_context"})
    return sg.compile()


APPS = {"standard": build_standard(), "hyde": build_hyde(), "graph": build_graph()}


# ---------------------------------------------------------------- Tab-2 selection -> retrieval scope
def scope_from_selection(selected: list[str] | None) -> tuple[list[str] | None, list[str] | None]:
    if not selected:
        return None, None
    g = KB.ensure().graph
    chunks: set[str] = set()
    allowed: set[str] = set()
    for nid in selected:
        if nid not in g:
            continue
        allowed.add(nid)
        t = g.nodes[nid].get("type")
        if t == "Chunk":
            chunks.add(g.nodes[nid]["chunk_id"])
        elif t == "Document":
            chunks.update(g.nodes[u]["chunk_id"] for u, _, d in g.in_edges(nid, data=True) if d.get("rel") == "PART_OF")
        else:
            chunks.update(g.nodes[v]["chunk_id"] for _, v, d in g.out_edges(nid, data=True) if d.get("rel") == "MENTIONED_IN")
    return sorted(chunks), sorted(allowed)


# ---------------------------------------------------------------- run
def _initial_state(engine, question, area, selected, gold, blocked=None) -> dict:
    scope_chunks, allowed = scope_from_selection(selected)
    return {"question": question, "engine": engine, "area": area, "selected_node_ids": selected or [],
            "hyperparams": hyperparams.get().model_dump(), "gold": gold, "scope_chunks": scope_chunks,
            "allowed_nodes": allowed, "blocked_edges": blocked or [], "loops": 0, "trace": [], "errors": [], "usage": {}, "timings": {}}


def run_query(engine: str, question: str, area: str, selected: list[str] | None = None,
              pipeline_config: dict | None = None, emit_fn: Callable | None = None, gold: dict | None = None,
              retrieval_only: bool = False, skip_judge: bool = False, all_areas: bool = False,
              record: bool = True, blocked_edges: list[str] | None = None) -> dict:
    if engine not in APPS:
        raise ValueError(f"unknown engine {engine}")
    if area not in Const.PRACTICE_AREAS:
        raise ValueError(f"unknown practice area {area}")
    KB.ensure()
    pc = normalise(engine, pipeline_config)
    gold = gold or gold_mod.by_question(question)
    cfg = {"configurable": {"emit": emit_fn, "pipeline": pc, "retrieval_only": retrieval_only,
                            "skip_judge": skip_judge, "all_areas": all_areas}, "recursion_limit": 60}
    out = APPS[engine].invoke(_initial_state(engine, question, area, selected, gold, blocked_edges), config=cfg)
    out["_pipeline"] = pc
    out["_all_areas"] = all_areas
    if retrieval_only:
        return out
    res = summarize(out)
    LAST_TRACE[engine] = {"trace": res["trace"], "pipeline": pc, "question": question, "at": time.time()}
    if record:
        history.record(res)
    return res


def finish(state: dict, emit_fn: Callable | None = None, skip_judge: bool = False, record: bool = True) -> dict:
    """Phase 2 of a benchmark run: synthesis + output guard + KPIs on a retrieval-only state."""
    cfg = {"configurable": {"emit": emit_fn, "pipeline": state.get("_pipeline"), "skip_judge": skip_judge,
                            "all_areas": state.get("_all_areas", False)}}
    st = dict(state)
    tail = (N.kpi,) if st.get("route") == "refuse" else (N.synthesize, N.output_guard, N.kpi)
    for fn in tail:
        upd = fn(st, cfg)
        for k, v in upd.items():
            if k == "trace":
                st["trace"] = [t for t in st.get("trace", []) if not (t["step"] == v[0]["step"] and t["stats"].get("deferred"))] + v
            elif k in ("usage",):
                st[k] = client.merge_usage(st.get(k) or {}, v)
            elif k == "timings":
                st[k] = merge_dict(st.get(k), v)
            elif k == "errors":
                st[k] = (st.get(k) or []) + v
            else:
                st[k] = v
    res = summarize(st)
    if record:
        history.record(res)
    return res


def summarize(out: dict) -> dict:
    ctx = [{**c, "text": c["text"]} for c in out.get("context") or []]
    # The footer is added after the kpi node, so the judge, citation check and answer_pass all see the body only.
    answer, out_tokens_est = output_checks.append_token_estimate(out.get("answer", ""))
    return {
        "engine": out.get("engine"), "engine_label": ENGINE_INFO[out.get("engine", "standard")]["label"],
        "area": out.get("area"), "question": out.get("question"), "masked_question": out.get("masked_question"),
        "answer": answer, "output_tokens_est": out_tokens_est, "citations": out.get("citations") or [], "context": ctx,
        "triples": out.get("triples") or [], "linked_entities": out.get("linked_entities") or [],
        "hypotheses": out.get("hypotheses") or [], "sufficiency": out.get("sufficiency"),
        "hop_k_used": out.get("hop_k_used"), "loops": out.get("loops", 0),
        "guard_report": out.get("guard_report") or {}, "kpis": out.get("kpis") or {}, "trace": [
            {k: v for k, v in t.items() if k not in ("t_start", "t_end")} for t in out.get("trace") or []],
        "usage": out.get("usage") or {}, "timings": out.get("timings") or {}, "mode": out.get("mode"),
        "model_used": out.get("model_used"), "fallback_used": out.get("fallback_used", False),
        "pipeline": out.get("_pipeline"), "hyperparams": out.get("hyperparams"),
        "gold_id": (out.get("gold") or {}).get("id"), "gold_category": (out.get("gold") or {}).get("category"),
        "selected_nodes": len(out.get("selected_node_ids") or []), "errors": out.get("errors") or [],
        "llm_mode": client.mode(), "at": time.time(),
    }


# ---------------------------------------------------------------- topology for Tab 3
def topology(engine: str) -> dict:
    g = APPS[engine].get_graph()
    opt = OPTIONAL[engine]
    nodes = [{"id": n, "label": n, "optional": n in opt["nodes"],
              "locked": n not in opt["nodes"] and n not in ("__start__", "__end__")} for n in g.nodes]
    edges = []
    for e in g.edges:
        eid = f"{e.source}->{e.target}"
        edges.append({"id": eid, "source": e.source, "target": e.target, "conditional": bool(e.conditional),
                      "optional": eid in opt["edges"]})
    return {"engine": engine, "label": ENGINE_INFO[engine]["label"], "nodes": nodes, "edges": edges,
            "last": LAST_TRACE.get(engine)}
