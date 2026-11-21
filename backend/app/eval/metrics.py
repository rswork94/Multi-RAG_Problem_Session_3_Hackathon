"""The 4 KPIs (+ secondary metrics), computed per query.

1. Retrieval Context Recall (RCR) - share of gold facts whose anchor phrase is present in the packed context.
   Without gold labels (live chat on uploads): a Haiku estimate (online) or a lexical estimate (offline), labelled.
2. Faithfulness - supported claims / total claims (Haiku judge, blind to the engine; lexical fallback offline).
3. Entity-Relation Resolution Accuracy - corpus ER F1 vs gold alias clusters + per-question entity-link hit (Graph only).
4. End-to-end latency with a breakdown: retrieval, graph traversal, LLM inference (HyDE + synthesis), guards.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading

from ..config import settings
from ..ingest.resolve import norm
from ..llm import calls, client, offline
from ..prompts.tasks import JUDGE_SCHEMA, JUDGE_SYSTEM, RCR_SCHEMA, RCR_SYSTEM
from ..retrieval.stores import KB

KPI_META = {
    "rcr": {"label": "Retrieval Context Recall", "unit": "", "green": 0.8, "amber": 0.5, "higher_better": True,
            "help": "Share of the ground-truth regulatory facts present in the passages handed to synthesis."},
    "faithfulness": {"label": "Faithfulness", "unit": "", "green": 0.9, "amber": 0.75, "higher_better": True,
                     "help": "Share of the answer's claims supported by the cited passages (0-1). 1 - this = hallucination rate."},
    "er_accuracy": {"label": "Entity-Relation Resolution", "unit": "", "green": 0.8, "amber": 0.6, "higher_better": True,
                    "help": "Pairwise F1 of entity resolution against gold alias clusters (Graph RAG only)."},
    "latency_ms": {"label": "End-to-end latency", "unit": "ms", "green": 15000, "amber": 30000, "higher_better": False,
                   "help": "Request to final token, broken into retrieval, graph traversal and LLM inference."},
}
RETRIEVAL_NODES = {"embed_query", "dense", "multi_dense", "hyde_generate", "query_expansion", "bm25", "graph_traverse",
                   "rrf", "rerank", "sufficiency", "pack_context"}
GRAPH_NODES = {"query_expansion", "graph_traverse", "sufficiency"}
_judge_lock = threading.Lock()


def _contains(hay: str, needle: str) -> bool:
    n = re.sub(r"\s+", " ", needle.strip().lower())
    return bool(n) and re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", re.sub(r"\s+", " ", hay.lower())) is not None


def rcr_anchor(context: list[dict], gold_facts: list[dict]) -> tuple[float, list[dict]]:
    text = "\n".join(c["text"] for c in context)
    per = []
    for gf in gold_facts:
        hit = any(_contains(text, a) for a in gf.get("anchors") or [])
        per.append({"fact": gf["fact"], "present": hit, "evidence": gf.get("evidence")})
    return (round(sum(p["present"] for p in per) / len(per), 3) if per else 0.0), per


def _judge_cache() -> dict:
    p = settings.path("judge_cache.json")
    return json.loads(p.read_text()) if p.exists() else {}


def _judge_cache_put(key: str, val: dict) -> None:
    with _judge_lock:
        c = _judge_cache()
        c[key] = val
        settings.path("judge_cache.json").write_text(json.dumps(c))


def _passages(ctx: list[dict]) -> str:
    return "\n\n".join(f"[{c['n']}] {c['text']}" for c in ctx)


def faithfulness(answer: str, ctx: list[dict], use_llm: bool) -> dict:
    if use_llm and client.available() and ctx:
        key = hashlib.sha1((answer + "\x00" + _passages(ctx)).encode()).hexdigest()
        cached = _judge_cache().get(key)
        if cached:
            return {**cached, "cached": True}
        try:
            data, usage = calls.structured(JUDGE_SYSTEM, f"<passages>\n{_passages(ctx)}\n</passages>\n\n<answer>\n{answer}\n</answer>",
                                           JUDGE_SCHEMA, max_tokens=4000)
            claims = data.get("claims") or []
            w = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
            score = None if data.get("abstained") or not claims else round(sum(w[c["verdict"]] for c in claims) / len(claims), 3)
            out = {"score": score, "abstained": bool(data.get("abstained")), "n_claims": len(claims),
                   "unsupported": [c["claim"] for c in claims if c["verdict"] == "unsupported"][:5],
                   "method": "claude-haiku judge", "usage": usage}
            _judge_cache_put(key, {k: v for k, v in out.items() if k != "usage"})
            return out
        except client.LLMError as e:
            fallback = f"lexical (judge error: {e})"
    else:
        fallback = "lexical (offline)"
    score, abstained, unsupported = offline.heuristic_support(answer, ctx)
    return {"score": score, "abstained": abstained, "n_claims": None, "unsupported": unsupported, "method": fallback}


def rcr_estimate(question: str, ctx: list[dict], use_llm: bool) -> dict:
    if use_llm and client.available() and ctx:
        try:
            data, usage = calls.structured(RCR_SYSTEM, f"<question>\n{question}\n</question>\n\n<passages>\n{_passages(ctx)}\n</passages>",
                                           RCR_SCHEMA, max_tokens=2000)
            facts = data.get("facts") or []
            return {"score": round(sum(f["present"] for f in facts) / len(facts), 3) if facts else 0.0,
                    "facts": facts, "method": "estimated (claude-haiku, no gold)", "usage": usage}
        except client.LLMError:
            pass
    qw = offline._words(question)
    text = " ".join(c["text"] for c in ctx)
    cw = offline._words(text)
    return {"score": round(len(qw & cw) / len(qw), 3) if qw and ctx else 0.0, "facts": [],
            "method": "estimated (lexical, no gold)"}


def entity_link_hit(state: dict) -> float | None:
    gold = state.get("gold") or {}
    if state.get("engine") != "graph" or not gold.get("gold_entities"):
        return None
    reached = set((state.get("reached") or {}).keys()) | {l["id"] for l in state.get("linked_entities") or []}
    ids = [KB.alias_index.get(norm(n)) for n in gold["gold_entities"]]
    return round(sum(1 for i in ids if i and i in reached) / len(ids), 3)


def latency(trace: list[dict], timings: dict) -> dict:
    ents = [t for t in trace if t.get("t_start") is not None]
    by = {}
    for t in ents:
        by.setdefault(t["step"], []).append(t)
    retr = [t for t in ents if t["step"] in RETRIEVAL_NODES and t.get("status") != "bypassed"]
    retrieval_ms = (max(t["t_end"] for t in retr) - min(t["t_start"] for t in retr)) * 1000 if retr else 0.0
    graph_ms = sum(t["duration_ms"] for t in ents if t["step"] in GRAPH_NODES)
    llm_pre = sum(t["duration_ms"] for t in ents if t["step"] == "hyde_generate")
    synth = sum(t["duration_ms"] for t in by.get("synthesize", []) if not t["stats"].get("deferred"))
    guards = sum(t["duration_ms"] for t in ents if t["step"] in ("input_guard", "output_guard"))
    e2e = guards + retrieval_ms + synth
    return {"e2e_ms": round(e2e, 1), "retrieval_ms": round(retrieval_ms, 1), "graph_ms": round(graph_ms, 1),
            "llm_pre_ms": round(llm_pre, 1), "synthesis_ms": round(synth, 1), "guards_ms": round(guards, 1),
            "retrieval_other_ms": round(max(0.0, retrieval_ms - graph_ms - llm_pre), 1),
            "llm_inference_ms": round(llm_pre + synth, 1), "embed_ms": timings.get("embed_ms"),
            "rerank_ms": timings.get("rerank_ms"), "ttft_ms": timings.get("ttft_ms")}


def answer_checks(answer: str, gold: dict | None) -> dict | None:
    if not gold:
        return None
    inc = [s for s in gold.get("must_include") or [] if s.lower() not in answer.lower()]
    exc = [s for s in gold.get("must_not") or [] if s.lower() in answer.lower()]
    return {"missing": inc, "forbidden_present": exc, "pass": not inc and not exc}


def compute(state: dict, judge: bool = True) -> tuple[dict, dict]:
    ctx = state.get("context") or []
    report = state.get("guard_report") or {}
    gold = state.get("gold")
    lat = latency(state.get("trace") or [], state.get("timings") or {})
    if report.get("refused"):
        return {"rcr": None, "faithfulness": None, "er_accuracy": None, "latency_ms": lat["e2e_ms"], "latency": lat,
                "refused": True, "rcr_method": None, "faithfulness_method": None}, {}
    usage: dict = {}
    if gold:
        rcr, per = rcr_anchor(ctx, gold.get("gold_facts") or [])
        rcr_info = {"score": rcr, "facts": per, "method": "gold anchors"}
    else:
        rcr_info = rcr_estimate(state.get("masked_question", ""), ctx, judge)
        usage = client.merge_usage(usage, rcr_info.pop("usage", {}))
    f = faithfulness(state.get("answer", ""), ctx, judge)
    usage = client.merge_usage(usage, f.pop("usage", {}))
    er = KB.er_report.get("final", {}).get("f1") if state.get("engine") == "graph" else None
    return {
        "rcr": rcr_info["score"], "rcr_method": rcr_info["method"], "rcr_facts": rcr_info.get("facts", []),
        "faithfulness": f["score"], "faithfulness_method": f["method"], "abstained": f["abstained"],
        "hallucination_rate": round(1 - f["score"], 3) if f["score"] is not None else None,
        "unsupported_claims": f.get("unsupported", []),
        "er_accuracy": er, "entity_link_hit": entity_link_hit(state),
        "latency_ms": lat["e2e_ms"], "latency": lat,
        "guard_compliant": (report.get("area_checks") or {}).get("compliant"),
        "citation_coverage": report.get("citation_coverage"),
        "answer_check": answer_checks(state.get("answer", ""), gold),
        "context_tokens": sum(c.get("n_tokens", 0) for c in ctx), "refused": False,
    }, usage
