"""Retrieval failure analysis (offline, no API spend).

For every gold fact: which passage holds its anchor, and at what rank it appears in each retriever list
(dense / bm25 / graph / HyDE vectors / fused / reranked) and whether it reached the final context.
Also: entity-resolution coverage (which gold alias members never occur in the corpus text at all).

Usage:  .venv/bin/python scripts/diagnose.py [--engine graph] [--area tax] [--misses]
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ["MULTIRAG_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engines.registry import run_query  # noqa: E402
from app.eval import gold  # noqa: E402
from app.ingest.resolve import norm  # noqa: E402
from app.retrieval.stores import KB  # noqa: E402


def _has(text: str, anchor: str) -> bool:
    return re.search(r"(?<![\w])" + re.escape(re.sub(r"\s+", " ", anchor.lower())) + r"(?![\w])",
                     re.sub(r"\s+", " ", text.lower())) is not None


def evidence_chunks(gf: dict) -> list[str]:
    return [cid for cid, p in KB.parents.items() if any(_has(p["text"], a) for a in gf["anchors"])]


def rank_in(hits: list[dict] | None, ids: list[str]) -> int | None:
    for i, h in enumerate(hits or [], 1):
        if h["chunk_id"] in ids:
            return i
    return None


def main() -> None:
    args = sys.argv[1:]
    engines = [args[args.index("--engine") + 1]] if "--engine" in args else ["standard", "hyde", "graph"]
    area = args[args.index("--area") + 1] if "--area" in args else None
    only_misses = "--misses" in args
    KB.ensure()
    miss_reasons: Counter = Counter()
    total = Counter()
    for q in gold.questions():
        if area and q["area"] != area:
            continue
        for eng in engines:
            st = run_query(eng, q["question"], q["area"], gold=q, retrieval_only=True, record=False)
            ctx = [c["chunk_id"] for c in st.get("context") or []]
            lines = []
            for i, gf in enumerate(q["gold_facts"]):
                ev = evidence_chunks(gf)
                total[eng] += 1
                in_ctx = any(c in ctx for c in ev)
                ranks = {"dense": rank_in(st.get("dense_hits"), ev), "bm25": rank_in(st.get("bm25_hits"), ev),
                         "graph": rank_in(st.get("graph_hits"), ev), "fused": rank_in(st.get("fused"), ev),
                         "rerank": rank_in(st.get("reranked"), ev)}
                for name, hits in (st.get("hyde_lists") or {}).items():
                    if name != "query":
                        ranks[name] = rank_in(hits, ev)
                if not in_ctx:
                    if not ev:
                        reason = "anchor not in any passage"
                    elif all(v is None for v in ranks.values()):
                        reason = "never retrieved"
                    elif ranks.get("fused") and ranks.get("rerank") and ranks["rerank"] > 6 >= ranks["fused"]:
                        reason = "demoted by rerank"
                    else:
                        reason = "retrieved but ranked below K"
                    miss_reasons[(eng, q["category"], reason)] += 1
                if only_misses and in_ctx:
                    continue
                r = " ".join(f"{k}:{v}" for k, v in ranks.items() if v is not None) or "-"
                lines.append(f"    fact{i} {'OK ' if in_ctx else 'MISS'} ev={[e[-4:] for e in ev]} {r}  | {gf['fact'][:70]}")
            if lines:
                linked = [l["label"] for l in st.get("linked_entities") or []][:5]
                extra = f" linked={linked} loops={st.get('loops')}" if eng == "graph" else ""
                print(f"[{q['id']} {q['category']} {eng}]{extra}")
                print("\n".join(lines))
    print("\nMisses by (engine, category, reason):")
    for k, v in sorted(miss_reasons.items()):
        print(f"  {k}: {v}")
    print("facts per engine:", dict(total))

    text = "\n".join(p["text"] for p in KB.parents.values()).lower()
    import yaml
    clusters = yaml.safe_load((Path(__file__).resolve().parent.parent / "corpus/gold/aliases.yaml").read_text())["clusters"]
    members = [(c["canonical"], m) for c in clusters for m in c["members"]]
    absent = [(c, m) for c, m in members if m.lower() not in text]
    unresolved = [(c, m) for c, m in members if m.lower() in text and norm(m) not in KB.alias_index]
    print(f"\nER coverage: {len(members)} gold alias members; {len(absent)} never occur in the corpus text; "
          f"{len(unresolved)} occur but are not in the alias index")
    for c, m in unresolved[:40]:
        print(f"   unresolved: {m!r}  (cluster {c!r})")


if __name__ == "__main__":
    main()
