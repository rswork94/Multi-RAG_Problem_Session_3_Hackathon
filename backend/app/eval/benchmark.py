"""Tab 5 benchmark: gold questions x engines (+ ablations), with the fairness protocol from the plan.

Phase 1 (serial, timed): retrieval for every (question, variant) in randomised order after a discarded warm-up.
Phase 2 (concurrent, max 4): synthesis + output guard + KPIs (judge time excluded from latency).
Winner rule (declared before running): highest RCR; within 0.05, higher faithfulness; then lower retrieval latency.
"""
from __future__ import annotations

import json
import random
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ..config import hyperparams, settings
from ..engines import registry
from ..llm import client
from . import gold as gold_mod

VARIANTS = {
    "standard": {"engine": "standard", "label": "Standard RAG", "pipeline": {}},
    "hyde": {"engine": "hyde", "label": "HyDE RAG", "pipeline": {}},
    "graph": {"engine": "graph", "label": "Graph RAG", "pipeline": {}},
    "graph_no_graph": {"engine": "graph", "label": "Graph − graph branch (ablation)",
                       "pipeline": {"disabled_edges": ["graph_traverse->rrf"]}},
    "graph_no_rerank": {"engine": "graph", "label": "Graph − rerank (ablation)", "pipeline": {"disabled_nodes": ["rerank"]}},
}
CATEGORY_LABEL = {"keyword": "Simple keyword", "multihop": "Multi-hop relational", "vocab_alias": "Vocabulary mismatch (alias)",
                  "vocab_paraphrase": "Vocabulary mismatch (paraphrase)"}
WINNER_RULE = "Highest RCR; if within 0.05, higher faithfulness; then lower retrieval latency."
EXPECTED = [  # the brief's Section 5 table, used as hypotheses
    {"metric": "Simple Keyword Queries", "categories": ["keyword"], "expect_best": "standard",
     "brief": {"standard": "High Precision / Low Latency", "hyde": "Moderate (Overhead of HyDE)", "graph": "Moderate (Graph Overhead)"}},
    {"metric": "Multi-Hop Relational Queries", "categories": ["multihop"], "expect_best": "graph",
     "brief": {"standard": "Poor (Misses indirect relationships)", "hyde": "Moderate (Semantic expansion)", "graph": "Superior (Explores k-hop edges)"}},
    {"metric": "Domain Vocabulary Mismatch", "categories": ["vocab_alias", "vocab_paraphrase"], "expect_best": "hyde",
     "brief": {"standard": "Poor (Relies on exact vector distance)", "hyde": "Superior (Synthesizes domain terms)", "graph": "High (Entity resolution maps synonyms)"}},
    {"metric": "Average Latency (ms)", "categories": None, "expect_order": ["standard", "graph", "hyde"],
     "brief": {"standard": "~450 ms", "hyde": "~1,200 ms", "graph": "~850 ms"}},
    {"metric": "Hallucination Risk", "categories": None, "expect_lowest": "graph",
     "brief": {"standard": "Moderate (Out-of-context chunks)", "hyde": "Moderate (Synthetic passage drift)", "graph": "Lowest (Strict entity-relation grounding)"}},
]
SYNTH_TOKENS = (5500, 1000)    # per answer: input (passages + cached system prompt), output (incl. thinking)
UTILITY_TOKENS = (2500, 400)    # per HyDE / judge call


def _call_cost(model: str, tokens: tuple[int, int]) -> float:
    p_in, p_out, _ = client.price(model)
    return (tokens[0] * p_in + tokens[1] * p_out) / 1e6


def select_questions(areas: list[str], categories: list[str] | None, quick: bool) -> list[dict]:
    qs = [q for q in gold_mod.questions() if q["area"] in areas and (not categories or q["category"] in categories)]
    if quick:
        picked, seen = [], set()
        for q in qs:
            key = (q["area"], "vocab" if q["category"].startswith("vocab") else q["category"])
            if key not in seen:
                seen.add(key)
                picked.append(q)
        qs = picked
    return qs


def estimate(n_questions: int, variants: list[str]) -> dict:
    online = client.available()
    runs = n_questions * len(variants)
    hyde_calls = n_questions * hyperparams.get().hyde_n * sum(1 for v in variants if VARIANTS[v]["engine"] == "hyde")
    llm = client.current()
    synth, util = _call_cost(llm.synth_model, SYNTH_TOKENS), _call_cost(llm.utility_model, UTILITY_TOKENS)
    cost = runs * (synth + util) + hyde_calls * util if online else 0.0
    return {"questions": n_questions, "variants": len(variants), "runs": runs, "online": online,
            "est_cost_usd": round(cost, 2), "est_minutes": round(runs * (12 if online else 0.6) / 4 / 60 + 0.2, 1)}


def _pct(xs: list[float], p: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 3) if xs else None


def _ci(xs, seed: int = 7, n: int = 1000) -> list[float] | None:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n) - 1], 3)]


def aggregate(rows: list[dict], variants: list[str]) -> dict:
    cats = sorted({r["category"] for r in rows}, key=lambda c: list(CATEGORY_LABEL).index(c) if c in CATEGORY_LABEL else 9)
    cells: dict = {}
    for v in variants:
        for c in cats + ["all"]:
            rs = [r for r in rows if r["variant"] == v and (c == "all" or r["category"] == c)]
            if not rs:
                continue
            k = [r["kpis"] for r in rs]
            cells[f"{v}|{c}"] = {
                "n": len(rs), "rcr": _mean([x.get("rcr") for x in k]), "rcr_ci": _ci([x.get("rcr") for x in k]),
                "faithfulness": _mean([x.get("faithfulness") for x in k]),
                "faithfulness_ci": _ci([x.get("faithfulness") for x in k]),
                "hallucination_rate": _mean([x.get("hallucination_rate") for x in k]),
                "abstention_rate": _mean([1.0 if x.get("abstained") else 0.0 for x in k]),
                "answer_pass_rate": _mean([1.0 if (x.get("answer_check") or {}).get("pass") else 0.0 for x in k]),
                "guard_compliance": _mean([1.0 if x.get("guard_compliant") else 0.0 for x in k]),
                "entity_link_hit": _mean([x.get("entity_link_hit") for x in k]),
                "retrieval_p50": _pct([x["latency"]["retrieval_ms"] for x in k], 0.5),
                "retrieval_p90": _pct([x["latency"]["retrieval_ms"] for x in k], 0.9),
                "e2e_p50": _pct([x["latency"]["e2e_ms"] for x in k], 0.5),
                "e2e_p90": _pct([x["latency"]["e2e_ms"] for x in k], 0.9),
                "graph_p50": _pct([x["latency"]["graph_ms"] for x in k], 0.5),
                "llm_p50": _pct([x["latency"]["llm_inference_ms"] for x in k], 0.5),
                "not_comparable": sum(1 for r in rs if r.get("fallback_used")),
            }
    winners = {}
    for c in cats + ["all"]:
        cand = [(v, cells[f"{v}|{c}"]) for v in variants if f"{v}|{c}" in cells and v in ("standard", "hyde", "graph")]
        winners[c] = _winner(cand)
    return {"categories": cats, "cells": cells, "winners": winners}


def _winner(cand: list[tuple[str, dict]]) -> str | None:
    cand = [(v, c) for v, c in cand if c.get("rcr") is not None]
    if not cand:
        return None
    best_rcr = max(c["rcr"] for _, c in cand)
    tier = [(v, c) for v, c in cand if c["rcr"] >= best_rcr - 0.05]
    best_f = max((c["faithfulness"] or 0) for _, c in tier)
    tier = [(v, c) for v, c in tier if (c["faithfulness"] or 0) >= best_f - 1e-9]
    return min(tier, key=lambda vc: vc[1]["retrieval_p50"] or 1e12)[0]


def expected_vs_measured(agg: dict) -> list[dict]:
    cells, out = agg["cells"], []
    for row in EXPECTED:
        r = {"metric": row["metric"], "brief": row["brief"], "measured": {}, "verdict": None, "note": ""}
        if row.get("categories"):
            pool = [c for c in row["categories"] if c in agg["categories"]]
            if not pool:
                r["note"] = "category not in this run"
                out.append(r)
                continue
            score = {}
            for v in ("standard", "hyde", "graph"):
                cs = [cells.get(f"{v}|{c}") for c in pool if cells.get(f"{v}|{c}")]
                if cs:
                    score[v] = {"rcr": _mean([c["rcr"] for c in cs]), "faithfulness": _mean([c["faithfulness"] for c in cs]),
                                "retrieval_p50": _mean([c["retrieval_p50"] for c in cs])}
            r["measured"] = score
            best = _winner(list(score.items()))
            r["measured_best"] = best
            r["verdict"] = "confirmed" if best == row["expect_best"] else "not confirmed"
            r["note"] = f"expected best: {row['expect_best']}; measured best by the declared rule: {best}"
        elif row.get("expect_order"):
            lat = {v: cells.get(f"{v}|all", {}).get("retrieval_p50") for v in ("standard", "hyde", "graph")}
            r["measured"] = {v: {"retrieval_p50_ms": x, "e2e_p50_ms": cells.get(f"{v}|all", {}).get("e2e_p50")} for v, x in lat.items()}
            order = [v for v, _ in sorted(((v, x) for v, x in lat.items() if x is not None), key=lambda t: t[1])]
            r["measured_order"] = order
            r["verdict"] = "confirmed" if order == row["expect_order"] else "not confirmed"
            r["note"] = "Retrieval-stage latency (before synthesis) compared with the brief; end-to-end shown alongside."
        else:
            hal = {v: cells.get(f"{v}|all", {}).get("hallucination_rate") for v in ("standard", "hyde", "graph")}
            r["measured"] = {v: {"hallucination_rate": x} for v, x in hal.items()}
            valid = {v: x for v, x in hal.items() if x is not None}
            lowest = min(valid, key=valid.get) if valid else None
            r["verdict"] = "confirmed" if lowest == row["expect_lowest"] else "not confirmed"
            r["note"] = f"lowest measured hallucination rate: {lowest}"
        out.append(r)
    return out


def run(areas: list[str], categories: list[str] | None, variants: list[str], quick: bool,
        emit: Callable[[dict], None] | None = None, seed: int = 42) -> dict:
    emit = emit or (lambda e: None)
    variants = [v for v in variants if v in VARIANTS] or ["standard", "hyde", "graph"]
    qs = select_questions(areas, categories, quick)
    if not qs:
        raise ValueError("No gold questions match. Build the corpus (scripts/build_corpus.py) and load it first.")
    bid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    rng = random.Random(seed)
    total = len(qs) * len(variants)
    emit({"type": "bench", "phase": "warmup", "done": 0, "total": total})
    for v in variants:  # discarded warm-up pass (model kernels, caches)
        spec = VARIANTS[v]
        registry.run_query(spec["engine"], qs[0]["question"], qs[0]["area"], pipeline_config=spec["pipeline"],
                           gold=qs[0], retrieval_only=True, record=False)
    states = []
    done = 0
    for q in qs:
        order = list(variants)
        rng.shuffle(order)
        for v in order:
            spec = VARIANTS[v]
            st = registry.run_query(spec["engine"], q["question"], q["area"], pipeline_config=spec["pipeline"], gold=q,
                                    retrieval_only=True, record=False)
            states.append((q, v, st))
            done += 1
            emit({"type": "bench", "phase": "retrieval", "done": done, "total": total, "question": q["id"], "variant": v})
    rows = []
    done = 0

    def phase2(item):
        q, v, st = item
        res = registry.finish(st, record=False)
        return q, v, res

    with ThreadPoolExecutor(4 if client.available() else 2) as ex:
        for q, v, res in ex.map(phase2, states):
            done += 1
            rows.append({"question_id": q["id"], "area": q["area"], "category": q["category"], "hops": q.get("hops"),
                         "question": q["question"], "variant": v, "engine": VARIANTS[v]["engine"],
                         "answer": res["answer"], "citations": res["citations"], "kpis": res["kpis"],
                         "context_ids": [c["chunk_id"] for c in res["context"]],
                         "context_sources": [f'{c["source"]} · {c["section"][:50]}' for c in res["context"]],
                         "linked": [l["label"] for l in res["linked_entities"]], "loops": res["loops"],
                         "model": res["model_used"], "fallback_used": res["fallback_used"], "mode": res["mode"],
                         "cost_usd": res["usage"].get("cost_usd", 0)})
            emit({"type": "bench", "phase": "synthesis", "done": done, "total": total, "question": q["id"], "variant": v})
    agg = aggregate(rows, variants)
    result = {"id": bid, "created": time.time(), "areas": areas, "categories": categories, "quick": quick,
              "variants": {v: VARIANTS[v]["label"] for v in variants}, "winner_rule": WINNER_RULE,
              "llm_mode": client.mode(), "provider": client.current().provider,
              "synth_model": client.current().synth_model if client.available() else "extractive (offline)",
              "utility_model": client.current().utility_model, "hyperparams": hyperparams.get().model_dump(),
              "corpus_version": gold_mod.manifest().get("version"), "n_questions": len(qs), "rows": rows,
              "aggregate": agg, "expected_vs_measured": expected_vs_measured(agg),
              "cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 4)}
    (settings.benchmarks_dir / f"{bid}.json").write_text(json.dumps(result, default=str))
    return result


def list_runs() -> list[dict]:
    out = []
    for p in sorted(settings.benchmarks_dir.glob("*.json"), reverse=True)[:30]:
        try:
            d = json.loads(p.read_text())
            out.append({k: d.get(k) for k in ("id", "created", "areas", "quick", "llm_mode", "n_questions", "cost_usd",
                                              "corpus_version")})
        except Exception:
            pass
    return out


def load(bid: str) -> dict | None:
    p = settings.benchmarks_dir / f"{bid}.json"
    return json.loads(p.read_text()) if p.exists() and p.parent == settings.benchmarks_dir else None


def to_csv(result: dict) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["question_id", "area", "category", "variant", "rcr", "faithfulness", "hallucination_rate", "abstained",
                "answer_pass", "retrieval_ms", "graph_ms", "llm_ms", "e2e_ms", "model", "fallback_used", "question"])
    for r in result["rows"]:
        k = r["kpis"]
        lat = k.get("latency") or {}
        w.writerow([r["question_id"], r["area"], r["category"], r["variant"], k.get("rcr"), k.get("faithfulness"),
                    k.get("hallucination_rate"), k.get("abstained"), (k.get("answer_check") or {}).get("pass"),
                    lat.get("retrieval_ms"), lat.get("graph_ms"), lat.get("llm_inference_ms"), lat.get("e2e_ms"),
                    r["model"], r["fallback_used"], r["question"]])
    return buf.getvalue()
