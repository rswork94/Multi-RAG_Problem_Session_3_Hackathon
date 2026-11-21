"""Tab-3 toggles: optional nodes become defined passthroughs (the compiled graph topology never changes).

pipeline_config = {"disabled_nodes": [...], "disabled_edges": ["bm25->rrf", ...]}
Edge semantics: a retriever->rrf edge off drops that list from fusion (the retriever is bypassed);
the sufficiency->graph_traverse edge off disables the cyclic loop.
"""
from __future__ import annotations

import functools
import hashlib
import json
from typing import Callable

from .tracing import cfg, emit

OPTIONAL = {
    "standard": {"nodes": [], "edges": []},
    "hyde": {"nodes": ["hyde_generate", "rrf"], "edges": []},
    "graph": {"nodes": ["query_expansion", "bm25", "dense", "graph_traverse", "rrf", "rerank", "sufficiency"],
              "edges": ["bm25->rrf", "dense->rrf", "graph_traverse->rrf", "sufficiency->graph_traverse"]},
}
EDGE_TO_NODE = {"bm25->rrf": "bm25", "dense->rrf": "dense", "graph_traverse->rrf": "graph_traverse"}
RETRIEVERS = {"bm25", "dense", "graph_traverse"}


def normalise(engine: str, pc: dict | None) -> dict:
    pc = pc or {}
    opt = OPTIONAL.get(engine, {"nodes": [], "edges": []})
    nodes = sorted(set(pc.get("disabled_nodes") or []) & set(opt["nodes"]))
    edges = sorted(set(pc.get("disabled_edges") or []) & set(opt["edges"]))
    effective = set(nodes) | {EDGE_TO_NODE[e] for e in edges if e in EDGE_TO_NODE}
    if engine == "graph" and RETRIEVERS <= effective:
        raise ValueError("At least one retriever (bm25, dense, graph_traverse) must stay enabled.")
    return {"disabled_nodes": nodes, "disabled_edges": edges, "effective": sorted(effective),
            "loop_enabled": "sufficiency->graph_traverse" not in edges and "sufficiency" not in nodes,
            "hash": config_hash(nodes, edges)}


def config_hash(nodes: list[str], edges: list[str]) -> str:
    if not nodes and not edges:
        return "default"
    return hashlib.sha1(json.dumps([nodes, edges]).encode()).hexdigest()[:8]


def is_disabled(config, name: str) -> bool:
    pc = cfg(config, "pipeline") or {}
    return name in (pc.get("effective") or [])


def toggleable(name: str, passthrough: Callable[[dict], dict], why: str = "toggled off in the workflow canvas"):
    def deco(traced_fn):
        @functools.wraps(traced_fn)
        def wrapper(state, config=None):
            if not is_disabled(config, name):
                return traced_fn(state, config)
            out = passthrough(state) or {}
            entry = {"step": name, "label": getattr(traced_fn, "label", name), "status": "bypassed",
                     "duration_ms": 0.0, "stats": {"reason": why}}
            emit(config, {"type": "step", **entry})
            out["trace"] = [entry]
            return out
        wrapper.step = name
        return wrapper
    return deco
