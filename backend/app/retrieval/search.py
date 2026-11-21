"""Dense (HNSW over child windows, collapsed to parents), BM25, RRF fusion and cross-encoder rerank."""
from __future__ import annotations

import time

import numpy as np

from .. import runtime
from ..config import Const
from ..guardrails.injection import DOWNWEIGHT
from .stores import KB, legal_tokens


def _where(area: str | None, scope: set[str] | None) -> dict | None:
    conds = []
    if area and area != "all":
        conds.append({"area": area})
    if scope is not None:
        conds.append({"parent_id": {"$in": sorted(scope) or ["__none__"]}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def dense_search(qvec: np.ndarray, area: str | None, scope: set[str] | None, n: int = Const.CANDIDATES,
                 list_name: str = "dense") -> tuple[list[dict], float]:
    """Children are <=510-token windows; a parent's score is the max over its windows (parent-child retrieval)."""
    t0 = time.perf_counter()
    rows = KB.children.query(qvec, Const.CHILD_POOL, _where(area, scope))
    best: dict[str, dict] = {}
    for cid, sim, meta, doc in rows:
        pid = meta.get("parent_id")
        score = sim * (DOWNWEIGHT if meta.get("injection") else 1.0)
        if pid not in best or score > best[pid]["score"]:
            best[pid] = {"chunk_id": pid, "score": round(float(score), 4), "cosine": round(float(sim), 4),
                         "child_text": doc, "window": meta.get("window"), "list": list_name}
    hits = sorted(best.values(), key=lambda h: -h["score"])[:n]
    for r, h in enumerate(hits, 1):
        h["rank"] = r
    return hits, (time.perf_counter() - t0) * 1000


def bm25_search(query: str, area: str | None, scope: set[str] | None, n: int = Const.CANDIDATES) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    entry = KB.bm25.get(area if area and area != "all" else "all") or KB.bm25.get("all")
    if not entry:
        return [], 0.0
    bm, parent_ids, child_texts = entry
    scores = bm.get_scores(legal_tokens(query))
    best: dict[str, dict] = {}
    for i in np.argsort(-scores)[:Const.CHILD_POOL]:  # child-level BM25, collapsed to the best child per parent
        if scores[i] <= 0:
            break
        cid = parent_ids[i]
        if scope is not None and cid not in scope:
            continue
        p = KB.parents.get(cid, {})
        s = float(scores[i]) * (DOWNWEIGHT if p.get("injection_flag") else 1.0)
        if cid not in best or s > best[cid]["score"]:
            best[cid] = {"chunk_id": cid, "score": round(s, 3), "list": "bm25", "child_text": child_texts[i]}
    hits = sorted(best.values(), key=lambda h: -h["score"])[:n]
    for r, h in enumerate(hits, 1):
        h["rank"] = r
    return hits, (time.perf_counter() - t0) * 1000


def rrf(lists: dict[str, list[dict]], alpha: int, weights: dict[str, float] | None = None) -> list[dict]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of w_list / (alpha + rank_list(d)) (w = 1 by default)."""
    fused: dict[str, dict] = {}
    for name, hits in lists.items():
        w = (weights or {}).get(name, 1.0)
        for h in hits:
            f = fused.setdefault(h["chunk_id"], {"chunk_id": h["chunk_id"], "rrf": 0.0, "ranks": {},
                                                 "child_text": h.get("child_text", ""), "hop": h.get("hop")})
            f["rrf"] += w / (alpha + h["rank"])
            f["ranks"][name] = h["rank"]
            if h.get("list") == "dense" or not f["child_text"]:
                f["child_text"] = h.get("child_text") or f["child_text"]
            if h.get("hop") is not None:
                f["hop"] = h["hop"] if f.get("hop") is None else min(f["hop"], h["hop"])
    out = sorted(fused.values(), key=lambda f: -f["rrf"])
    for r, f in enumerate(out, 1):
        f["rrf"] = round(f["rrf"], 5)
        f["rank"] = r
    return out


def round_robin(lists: dict[str, list[dict]]) -> list[dict]:
    """Passthrough when RRF is toggled off: interleave the lists in rank order, first occurrence wins."""
    seen: dict[str, dict] = {}
    for i in range(max((len(v) for v in lists.values()), default=0)):
        for name, hits in lists.items():
            if i < len(hits) and hits[i]["chunk_id"] not in seen:
                h = hits[i]
                seen[h["chunk_id"]] = {"chunk_id": h["chunk_id"], "rrf": None, "ranks": {name: h["rank"]},
                                       "child_text": h.get("child_text", ""), "hop": h.get("hop")}
    out = list(seen.values())
    for r, f in enumerate(out, 1):
        f["rank"] = r
    return out


def rerank(question: str, fused: list[dict]) -> tuple[list[dict], float]:
    """Cross-encoder (bge-reranker-base) on the matched child window of the top RERANK_CANDIDATES."""
    head = fused[:Const.RERANK_CANDIDATES]
    if not head:
        return [], 0.0
    texts = []
    for f in head:
        p = KB.parents.get(f["chunk_id"], {})
        body = f.get("child_text") or p.get("text", "")
        texts.append(f'{p.get("section", "")}\n{body}' if p.get("section") else body)
    scores, ms = runtime.rerank_scores(question, texts)
    for f, s in zip(head, scores):
        f["rerank"] = round(float(s), 4)
    # The cross-encoder is one voter, not the final word: its order is fused (RRF) with the retrieval order, so a
    # passage that every retriever ranks highly is not discarded because its wording differs from the question
    # (lay paraphrases, bridge passages in multi-hop chains).
    by_rerank = sorted(head, key=lambda f: -f["rerank"])
    voted = {f["chunk_id"]: 1.0 / (Const.RERANK_FUSION_K + i) for i, f in enumerate(by_rerank, 1)}
    for i, f in enumerate(head, 1):
        voted[f["chunk_id"]] += 1.0 / (Const.RERANK_FUSION_K + i)
    out = sorted(head, key=lambda f: -voted[f["chunk_id"]]) + fused[Const.RERANK_CANDIDATES:]
    for r, f in enumerate(out, 1):
        f["rank"] = r
    return out, ms
