"""LangGraph state definitions. Keys written by parallel branches carry reducers (no InvalidUpdateError)."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dict(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class IngestState(TypedDict, total=False):
    files: list[dict]                 # {path, filename, area}
    reingest: bool
    use_llm: bool
    confirm_cost: bool
    hyperparams: dict
    documents: list[dict]
    chunks: list[dict]
    units: list[dict]
    unit_chunks: dict
    extractions: dict
    children: dict
    entities: list[dict]
    relations: list[dict]
    er: dict
    n2v: Any
    ent_vecs: Any
    graph: Any
    stats: Annotated[dict, merge_dict]
    usage: Annotated[dict, merge_dict]
    trace: Annotated[list[dict], operator.add]
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


class QueryState(TypedDict, total=False):
    # request
    question: str
    engine: str
    area: str
    selected_node_ids: list[str]
    hyperparams: dict
    gold: dict | None
    scope_chunks: list[str] | None    # Tab-2 selection -> allowed parent chunks (None = unrestricted)
    allowed_nodes: list[str] | None   # Tab-2 selection -> allowed graph nodes for traversal
    blocked_edges: list[str]          # Tab-2 edges switched off ('src|REL|dst'): never traversed
    # guard
    masked_question: str
    pii_found: list[dict]
    scope: dict
    route: str
    # retrieval
    qvec: Any
    linked_entities: list[dict]
    expanded_query: str
    hypotheses: list[str]
    bm25_hits: list[dict]
    dense_hits: list[dict]
    graph_hits: list[dict]
    hyde_lists: dict
    fused: list[dict]
    reranked: list[dict]
    triples: list[dict]
    reached: dict
    hop_k_used: int
    loops: int
    sufficiency: dict
    context: list[dict]
    # generation
    answer: str
    stop_reason: str
    model_used: str
    fallback_used: bool
    mode: str
    citations: list[dict]
    guard_report: dict
    kpis: dict
    # bookkeeping
    usage: Annotated[dict, merge_dict]
    timings: Annotated[dict, merge_dict]
    trace: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
