"""Node functions shared by the Standard, HyDE and Graph RAG StateGraphs.

Fairness: every engine ends in the same pack_context -> synthesize -> output_guard -> kpi tail, with the same
FINAL_K, context budget, prompt, model and effort. Only the retrieval nodes differ.
"""
from __future__ import annotations

import time
from datetime import date

from .. import runtime
from ..config import Const, settings
from ..eval import metrics
from ..guardrails import output_checks, pii
from ..guardrails import scope as scope_mod
from ..llm import calls, client, offline
from ..prompts.areas import AREAS, system_blocks
from ..prompts.costar import REFUSAL_INTEGRITY, REFUSAL_OUT_OF_SCOPE
from ..retrieval import graph as rgraph
from ..retrieval import hyde as rhyde
from ..retrieval import search
from ..retrieval.stores import KB
from .toggles import is_disabled, toggleable
from .tracing import cfg, emit, traced


def _area(state) -> str:
    return state.get("area") or "tax"


def _search_area(state, config) -> str:
    return "all" if cfg(config, "all_areas") else _area(state)


def _scope(state) -> set[str] | None:
    s = state.get("scope_chunks")
    return set(s) if s is not None else None


def _emit_text(config, text: str, size: int = 28) -> None:
    for i in range(0, len(text), size):
        emit(config, {"type": "token", "text": text[i:i + size]})


def _top(hits: list[dict], n: int = 5) -> list[dict]:
    return [{"chunk_id": h["chunk_id"], "score": h.get("score", h.get("rrf", h.get("rerank"))), "hop": h.get("hop"),
             "ranks": h.get("ranks")} for h in hits[:n]]


# ---------------------------------------------------------------- guard + query embedding
@traced("input_guard", "Input guard (PII, scope, integrity)")
def input_guard(state, config=None):
    masked, found = pii.mask_pii(state["question"])
    sc, usage = scope_mod.classify(masked)
    route = "refuse" if (not sc["in_scope"] or sc["integrity_risk"] == "high") else "retrieve"
    return {"masked_question": masked, "pii_found": found, "scope": sc, "route": route, "usage": usage,
            "_stats": {"pii_masked": len(found), "pii_types": sorted({f["type"] for f in found}),
                       "in_scope": sc["in_scope"], "integrity_risk": sc["integrity_risk"], "method": sc.get("method"),
                       "route": route, "practice_area": AREAS[_area(state)]["label"]}}


def route_after_guard(state) -> str:
    return state.get("route", "retrieve")


@traced("embed_query", "Embed query (bge-large-en-v1.5)")
def embed_query(state, config=None):
    v, ms = runtime.embed_query(state["masked_question"])
    return {"qvec": v, "timings": {"embed_ms": round(ms, 1)},
            "_stats": {"dim": int(v.shape[0]), "model_ms": round(ms, 1), "device": runtime.device(),
                       "query_instruction": True}}


# ---------------------------------------------------------------- Standard / shared dense
@toggleable("dense", lambda s: {"dense_hits": []})
@traced("dense", "Dense vector search (HNSW)", fatal=False)
def dense(state, config=None):
    hits, ms = search.dense_search(state["qvec"], _search_area(state, config), _scope(state))
    return {"dense_hits": hits, "timings": {"dense_ms": round(ms, 1)},
            "_stats": {"hits": len(hits), "area": _search_area(state, config), "scoped": state.get("scope_chunks") is not None,
                       "top": _top(hits)}}


# ---------------------------------------------------------------- HyDE
@toggleable("hyde_generate", lambda s: {"hypotheses": []}, "HyDE generation off: only the original query vector is searched")
@traced("hyde_generate", "Generate N hypothetical passages (Haiku)")
def hyde_generate(state, config=None):
    n = state["hyperparams"]["hyde_n"]
    r = rhyde.generate(state["masked_question"], _area(state), n)
    return {"hypotheses": r["hypotheses"], "usage": r["usage"], "timings": {"llm_pre_ms": r["ms"]},
            "_stats": {"n": n, "mode": r["mode"], "llm_ms": r["ms"], "hypotheses": [h[:220] for h in r["hypotheses"]]}}


@traced("multi_dense", "Multi-vector dense search (query + N hypotheses)")
def multi_dense(state, config=None):
    area, scope = _search_area(state, config), _scope(state)
    lists = {}
    qhits, _ = search.dense_search(state["qvec"], area, scope, list_name="query")
    lists["query"] = qhits
    hyps = state.get("hypotheses") or []
    emb_ms = 0.0
    if hyps:
        t0 = time.perf_counter()
        vecs = runtime.embed_passages(hyps)  # hypotheses are passages: no query instruction prefix
        emb_ms = (time.perf_counter() - t0) * 1000
        for i, v in enumerate(vecs, 1):
            lists[f"hyde_{i}"], _ = search.dense_search(v, area, scope, list_name=f"hyde_{i}")
    return {"hyde_lists": lists, "dense_hits": qhits, "timings": {"hyde_embed_ms": round(emb_ms, 1)},
            "_stats": {"vectors": len(lists), "hits_per_list": {k: len(v) for k, v in lists.items()},
                       "top_per_list": {k: [h["chunk_id"] for h in v[:3]] for k, v in lists.items()}}}


# ---------------------------------------------------------------- Graph RAG
@toggleable("query_expansion", lambda s: {"linked_entities": [], "expanded_query": s.get("masked_question", "")},
            "Query expansion off: no entity seeds, so graph traversal has nothing to expand")
@traced("query_expansion", "Query expansion (entity linking)")
def query_expansion(state, config=None):
    t0 = time.perf_counter()
    linked = rgraph.link_entities(state["masked_question"], state.get("qvec"), _search_area(state, config))
    if state.get("allowed_nodes") is not None:
        allowed = set(state["allowed_nodes"])
        linked = [l for l in linked if l["id"] in allowed]
    g = KB.graph
    extra = []
    for l in [x for x in linked if x["how"] != "passage"][:6]:  # only entities the question itself names
        extra.append(l["label"])
        extra += [a for a in (g.nodes[l["id"]].get("aliases") or "").split(" | ") if a][:3]
    expanded = state["masked_question"] + (" " + " ".join(dict.fromkeys(extra)) if extra else "")
    return {"linked_entities": linked, "expanded_query": expanded,
            "timings": {"expansion_ms": round((time.perf_counter() - t0) * 1000, 1)},
            "_stats": {"linked": [f'{l["label"]} · {l["type"]} · {l["how"]}' for l in linked],
                       "expanded_terms": list(dict.fromkeys(extra))[:12]}}


@toggleable("bm25", lambda s: {"bm25_hits": []})
@traced("bm25", "Sparse BM25 search", fatal=False)
def bm25(state, config=None):
    q = state.get("expanded_query") or state["masked_question"]
    hits, ms = search.bm25_search(q, _search_area(state, config), _scope(state))
    return {"bm25_hits": hits, "timings": {"bm25_ms": round(ms, 1)}, "_stats": {"hits": len(hits), "top": _top(hits)}}


@toggleable("graph_traverse", lambda s: {"graph_hits": [], "triples": [], "reached": {}})
@traced("graph_traverse", "k-hop graph traversal", fatal=False)
def graph_traverse(state, config=None):
    k = state.get("hop_k_used") or state["hyperparams"]["hop_k"]
    linked = state.get("linked_entities") or []
    seeds = [l["id"] for l in linked]
    allowed = set(state["allowed_nodes"]) if state.get("allowed_nodes") is not None else None
    blocked = set(state.get("blocked_edges") or [])
    area = _search_area(state, config)
    # BFS: relation triples (sent to the model) + reached entities (sufficiency check)
    _, triples, reached, ms_bfs = rgraph.traverse(seeds, k, area, allowed, _scope(state), blocked)
    # Personalized PageRank over the k-hop subgraph: how graph passages are ranked
    weights = {l["id"]: l.get("weight", 1.0) for l in linked}
    hits, ms_ppr = rgraph.ppr_rank(weights, k, area, allowed, _scope(state), blocked)
    prev = (state.get("timings") or {}).get("graph_ms", 0.0)
    return {"graph_hits": hits, "triples": triples, "reached": reached, "hop_k_used": k,
            "timings": {"graph_ms": round(prev + ms_bfs + ms_ppr, 1)},
            "_stats": {"k": k, "seeds": len(seeds), "reached": len(reached), "ranking": "personalized PageRank",
                       "by_hop": {str(h): sum(1 for v in reached.values() if v == h) for h in range(k + 1)},
                       "triples": len(triples), "chunks": len(hits), "loop": state.get("loops", 0), "top": _top(hits)}}


def _lists(state) -> dict:
    if state.get("engine") == "hyde":
        return state.get("hyde_lists") or {"query": state.get("dense_hits") or []}
    out = {}
    for name, key in (("bm25", "bm25_hits"), ("dense", "dense_hits"), ("graph", "graph_hits")):
        if state.get(key):
            out[name] = state[key]
    return out


@toggleable("rrf", lambda s: {"fused": search.round_robin(_lists(s))}, "RRF off: ranked lists interleaved round-robin")
@traced("rrf", "Reciprocal Rank Fusion")
def rrf(state, config=None):
    alpha = state["hyperparams"]["rrf_alpha"]
    lists = _lists(state)
    weights = {"graph": rgraph.graph_trust(state.get("linked_entities") or [])} if "graph" in lists else None
    fused = search.rrf(lists, alpha, weights)
    return {"fused": fused, "_stats": {"alpha": alpha, "lists": {k: len(v) for k, v in lists.items()},
                                       "weights": weights or {}, "fused": len(fused), "top": _top(fused)}}


@toggleable("rerank", lambda s: {"reranked": s.get("fused") or []}, "Rerank off: fused order kept")
@traced("rerank", "Cross-encoder rerank (bge-reranker-base)")
def rerank(state, config=None):
    out, ms = search.rerank(state["masked_question"], [dict(f) for f in state.get("fused") or []])
    prev = (state.get("timings") or {}).get("rerank_ms", 0.0)
    return {"reranked": out, "timings": {"rerank_ms": round(prev + ms, 1)},
            "_stats": {"scored": min(len(out), Const.RERANK_CANDIDATES), "model_ms": round(ms, 1), "top": _top(out)}}


@toggleable("sufficiency", lambda s: {"sufficiency": {"sufficient": True, "bypassed": True, "loop": False}},
            "Sufficiency check off: no cyclic re-traversal")
@traced("sufficiency", "Sufficiency check (cyclic loop)")
def sufficiency(state, config=None):
    k = state.get("hop_k_used") or state["hyperparams"]["hop_k"]
    top_ids = [r["chunk_id"] for r in graph_order(state)[:Const.FINAL_K]]
    seeds = [l["id"] for l in state.get("linked_entities") or []]
    s = rgraph.sufficiency(state["masked_question"], seeds, state.get("reached") or {}, top_ids, k)
    loops = state.get("loops", 0)
    pc = cfg(config, "pipeline") or {}
    can_loop = pc.get("loop_enabled", True) and not is_disabled(config, "graph_traverse") \
        and loops < Const.MAX_LOOPS and k < Const.MAX_HOPS
    loop = (not s["sufficient"]) and can_loop
    out = {"sufficiency": {**s, "loop": loop}, "_stats": {**s, "loop": loop, "loops_so_far": loops}}
    if loop:
        out.update({"loops": loops + 1, "hop_k_used": k + 1})
    return out


def route_after_sufficiency(state) -> str:
    return "graph_traverse" if (state.get("sufficiency") or {}).get("loop") else "pack_context"


# ---------------------------------------------------------------- shared tail
def graph_order(state) -> list[dict]:
    """Graph RAG context order: the fused + reranked list, except that up to GRAPH_RESERVED of the K slots go to the
    top Personalized-PageRank passages from hop >= 1 that are not already selected - but only when the question
    itself names an entity (identifier or alias), i.e. when the graph walk starts from a trusted seed.
    Bridge passages in a multi-hop chain rank high under PPR yet low under BM25/dense/cross-encoder, because their
    wording does not match the question; fusion alone would bury them. The '- graph branch' ablation measures this."""
    ranked = state.get("reranked") or state.get("fused") or []
    if rgraph.graph_trust(state.get("linked_entities") or []) < 1.0:
        return ranked
    head_n = Const.FINAL_K - Const.GRAPH_RESERVED
    head = ranked[:head_n]
    taken = {h["chunk_id"] for h in head}
    reserved = []
    for g in state.get("graph_hits") or []:  # already in PPR order
        if g.get("hop", 0) >= 1 and g["chunk_id"] not in taken:
            reserved.append({**next((r for r in ranked if r["chunk_id"] == g["chunk_id"]), {}), **g, "reserved": True})
            taken.add(g["chunk_id"])
        if len(reserved) >= Const.GRAPH_RESERVED:
            break
    rest = [r for r in ranked[head_n:] if r["chunk_id"] not in taken]
    return head + reserved + rest


def _ranked_for_engine(state) -> list[dict]:
    eng = state.get("engine")
    if eng == "standard":
        return state.get("dense_hits") or []
    if eng == "hyde":
        return state.get("fused") or state.get("dense_hits") or []
    return graph_order(state)


@traced("pack_context", "Pack context (K passages, token budget)")
def pack_context(state, config=None):
    ctx, used = [], 0
    for h in _ranked_for_engine(state):
        p = KB.parents.get(h["chunk_id"])
        if not p or any(c["chunk_id"] == h["chunk_id"] for c in ctx):
            continue
        if ctx and used + p["n_tokens"] > Const.CONTEXT_BUDGET:
            continue
        ctx.append({"chunk_id": p["chunk_id"], "text": p["text"], "source": p["source"], "title": p.get("title", ""),
                    "doc_type": p.get("doc_type", ""), "issued_date": p.get("issued_date", ""),
                    "section": p.get("section", ""), "page": p.get("page", 1), "practice_area": p.get("practice_area"),
                    "n_tokens": p["n_tokens"], "injection_flag": p.get("injection_flag", False),
                    "ranks": h.get("ranks") or {h.get("list", "dense"): h.get("rank")}, "hop": h.get("hop"),
                    "reserved": bool(h.get("reserved")),
                    "rerank": h.get("rerank"), "rrf": h.get("rrf"), "cosine": h.get("cosine")})
        used += p["n_tokens"]
        if len(ctx) >= Const.FINAL_K:
            break
    for i, c in enumerate(ctx, 1):
        c["n"] = i
    by_chunk = {c["chunk_id"]: c["n"] for c in ctx}
    kept = []
    if state.get("engine") == "graph":
        for t in sorted(state.get("triples") or [], key=lambda t: t["hop"]):
            refs = sorted({by_chunk[e] for e in t["evidence"] if e in by_chunk})
            if not refs:
                continue  # no orphan triples: evidence must be in the context
            line_tokens = 12 + len(t["src_label"].split()) + len(t["dst_label"].split())
            if used + line_tokens > Const.CONTEXT_BUDGET or len(kept) >= 25:
                break
            kept.append({**t, "refs": refs})
            used += line_tokens
    return {"context": ctx, "triples": kept if state.get("engine") == "graph" else [],
            "_stats": {"passages": len(ctx), "context_tokens": used, "budget": Const.CONTEXT_BUDGET,
                       "final_k": Const.FINAL_K, "relation_triples": len(kept),
                       "injection_passages": sum(c["injection_flag"] for c in ctx),
                       "passages_list": [f'[{c["n"]}] {c["source"]} · {c["section"][:60]}' for c in ctx]}}


def _user_turn(state) -> str:
    parts = []
    for c in state.get("context") or []:
        flag = ' injection_warning="true"' if c["injection_flag"] else ""
        parts.append(f'[{c["n"]}] <passage source="{c["source"]}" doc_type="{c["doc_type"]}" issued="{c["issued_date"] or "unknown"}" '
                     f'section="{c["section"] or "-"}" page="{c["page"]}"{flag}>\n{c["text"]}\n</passage>')
    rel = "\n".join(f'- {t["src_label"]} --{t["rel"]}--> {t["dst_label"]} [evidence: {", ".join(map(str, t["refs"]))}]'
                    for t in state.get("triples") or [])
    return (f"<today>{date.today().isoformat()}</today>\n<practice_area>{AREAS[_area(state)]['label']}</practice_area>\n\n"
            f"<context>\n{chr(10).join(parts) or '(no passages retrieved)'}\n</context>\n\n"
            + (f"<relations>\n{rel}\n</relations>\n\n" if rel else "")
            + f"<question>\n{state['masked_question']}\n</question>")


@traced("synthesize", "Synthesis (Claude Opus 5.5)")
def synthesize(state, config=None):
    if cfg(config, "retrieval_only"):
        return {"_stats": {"deferred": True}}
    ctx = state.get("context") or []
    usage, stop, model, fallback, ttft = {}, "end_turn", None, False, None
    if ctx and client.available():
        try:
            r = calls.stream_synthesis(system_blocks(_area(state)), _user_turn(state),
                                       on_token=lambda t: emit(config, {"type": "token", "text": t}))
            answer, stop, usage, model, fallback, ttft = r["text"], r["stop_reason"], r["usage"], r["model"], \
                r["fallback_used"], r["ttft_ms"]
            mode = "claude"
        except client.LLMError as e:
            answer = f"_Claude unavailable ({e}); extractive fallback shown._\n\n" + \
                offline.extractive_answer(state["masked_question"], _area(state), ctx, state.get("triples") or [])
            _emit_text(config, answer)
            mode = "fallback"
    else:
        answer = offline.extractive_answer(state["masked_question"], _area(state), ctx, state.get("triples") or [])
        _emit_text(config, answer)
        mode = "offline" if ctx else "no-context"
    return {"answer": answer, "stop_reason": stop, "mode": mode, "model_used": model or "extractive",
            "fallback_used": fallback, "usage": usage, "timings": {"ttft_ms": ttft},
            "_stats": {"mode": mode, "model": model or "extractive (offline)", "effort": client.current().synth_effort,
                       "stop_reason": stop, "passages": len(ctx), "fallback_used": fallback, "ttft_ms": ttft,
                       "output_tokens": usage.get("output_tokens"), "cache_read_tokens": usage.get("cache_read_input_tokens"),
                       "cost_usd": usage.get("cost_usd")}}


@traced("output_guard", "Output guard (citations, area checks, PII)")
def output_guard(state, config=None):
    if cfg(config, "retrieval_only"):
        return {"_stats": {"deferred": True}}
    ctx = state.get("context") or []
    raw = state.get("answer", "")
    ans, found = pii.mask_pii(raw)
    ans, area_rep = output_checks.area_checks(_area(state), ans, ctx)
    ans, added = output_checks.enforce_disclaimer(ans)
    if ans != raw:
        emit(config, {"type": "replace", "text": ans})
    cv = output_checks.validate_citations(ans, len(ctx))
    by_n = {c["n"]: c for c in ctx}
    cites = [{k: by_n[n][k] for k in ("n", "chunk_id", "source", "section", "page", "doc_type", "issued_date", "ranks",
                                       "hop")} for n in cv["valid"]]
    report = {"pii_masked_input": len(state.get("pii_found") or []), "pii_masked_output": len(found),
              "invalid_citations": cv["invalid"], "uncited_paragraphs": cv["uncited_paragraphs"],
              "citation_coverage": cv["coverage"], "disclaimer_added": added, "area_checks": area_rep,
              "injection_passages": sum(1 for c in ctx if c["injection_flag"]), "refused": False,
              "scope": state.get("scope"), "stop_reason": state.get("stop_reason")}
    return {"answer": ans, "citations": cites, "guard_report": report,
            "_stats": {"citations": len(cites), "invalid": len(cv["invalid"]), "coverage": cv["coverage"],
                       "area_compliant": area_rep["compliant"], "missing": area_rep["missing_sections"],
                       "auto_filled": area_rep["auto_filled"], "pii_masked_output": len(found)}}


@traced("refuse", "Refuse (guardrail G6 / scope)")
def refuse(state, config=None):
    sc = state.get("scope") or {}
    integrity = sc.get("integrity_risk") == "high"
    answer = REFUSAL_INTEGRITY if integrity else REFUSAL_OUT_OF_SCOPE
    _emit_text(config, answer)
    reason = "integrity (G6)" if integrity else "out of scope"
    return {"answer": answer, "citations": [], "context": [], "mode": "refused", "model_used": "none",
            "guard_report": {"refused": True, "refusal_reason": reason, "scope": sc,
                             "pii_masked_input": len(state.get("pii_found") or [])},
            "_stats": {"reason": reason}}


@traced("kpi", "KPIs")
def kpi(state, config=None):
    if cfg(config, "retrieval_only"):
        return {"_stats": {"deferred": True}}
    k, usage = metrics.compute(state, judge=not cfg(config, "skip_judge", False))
    return {"kpis": k, "usage": usage,
            "_stats": {"rcr": k["rcr"], "faithfulness": k["faithfulness"], "er_accuracy": k["er_accuracy"],
                       "latency_ms": k["latency_ms"], "rcr_method": k.get("rcr_method")}}
