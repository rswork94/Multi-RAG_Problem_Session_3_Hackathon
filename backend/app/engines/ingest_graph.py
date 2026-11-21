"""The 6-step Graph RAG ingestion pipeline as a LangGraph StateGraph (live SSE stepper on Tab 1).

START -> s1_load -> s2_chunk -> s3_tokenize -> s4_embed -> s5_index -> s6_store -> END
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from langgraph.graph import END, START, StateGraph

from .. import runtime
from ..config import Const, hyperparams, settings
from ..ingest import extract as ex
from ..ingest.index import build_graph, embed_cached, persist_all
from ..ingest.loaders import load_file
from ..ingest.resolve import resolve
from ..ingest.splitter import child_windows, split_document, units_in_chunk
from ..llm import client
from ..retrieval.stores import atomic_write
from .state import IngestState
from .tracing import emit, traced

STEPS = [("s1_load", "1 · Data format & ingestion"), ("s2_chunk", "2 · Semantic & entity chunking"),
         ("s3_tokenize", "3 · Sub-word tokenization & entity normalization"), ("s4_embed", "4 · Multi-modal embedding"),
         ("s5_index", "5 · Dual indexing"), ("s6_store", "6 · Hybrid vector DB & graph engine")]


def _docstore_path() -> Path:
    return settings.path("docstore.json")


def _load_docstore() -> list[dict]:
    p = _docstore_path()
    return json.loads(p.read_text()) if p.exists() else []


def _gold_clusters(areas: set[str]) -> list[dict]:
    p = settings.corpus_dir / "gold" / "aliases.yaml"
    if not p.exists():
        return []
    return [c for c in (yaml.safe_load(p.read_text()) or {}).get("clusters", []) if c.get("area") in areas]


@traced("s1_load", STEPS[0][1])
def s1_load(state: IngestState, config=None):
    docs = {d["doc_id"]: d for d in _load_docstore()}
    added, dupes, replaced, loaders, failed = [], 0, 0, Counter(), []
    for f in state.get("files") or []:
        try:
            d = load_file(Path(f["path"]), f.get("area") or "tax", f.get("filename"))
        except Exception as e:  # noqa: BLE001
            failed.append(f"{f.get('filename')}: {e}")
            continue
        if d["doc_id"] in docs:
            dupes += 1
            continue
        stale = [k for k, v in docs.items() if v["source"] == d["source"] and v["practice_area"] == d["practice_area"]]
        for k in stale:  # same file name + area, new content: the new version replaces the old one
            del docs[k]
        replaced += len(stale)
        docs[d["doc_id"]] = d
        added.append(d["source"])
        loaders[d["loader_used"]] += 1
    documents = list(docs.values())
    if not documents:
        raise ValueError("No documents to ingest. " + "; ".join(failed))
    atomic_write(_docstore_path(), json.dumps(documents))
    by_area = Counter(d["practice_area"] for d in documents)
    return {"documents": documents, "warnings": failed,
            "_stats": {"files_added": len(added), "duplicates_skipped": dupes, "replaced_versions": replaced, "failed": len(failed),
                       "documents_total": len(documents), "by_area": dict(by_area),
                       "by_format": dict(Counter(d["format"] for d in documents)), "loaders": dict(loaders),
                       "pages": sum(int(d.get("pages", 1)) for d in documents),
                       "chars": sum(int(d.get("chars", 0)) for d in documents)}}


@traced("s2_chunk", STEPS[1][1])
def s2_chunk(state: IngestState, config=None):
    hp = state["hyperparams"]
    chunks, units, unit_chunks, unit_area = [], [], {}, {}
    for d in state["documents"]:
        cs, us = split_document(d, hp["chunk_size"], hp["chunk_overlap"])
        for c in cs:
            for uid in units_in_chunk(c, us):
                unit_chunks.setdefault(uid, []).append(c["chunk_id"])
        for u in us:
            unit_area[u["unit_id"]] = d["practice_area"]
        chunks += cs
        units += us
    use_llm = bool(state.get("use_llm")) and client.available()
    est = ex.estimate_cost(units, ex.load_cache())
    pre_warn = []
    if use_llm and est["est_cost_usd"] > settings.extraction_cost_limit_usd and not state.get("confirm_cost"):
        use_llm = False
        pre_warn.append(f"LLM extraction skipped: estimated ${est['est_cost_usd']} exceeds the "
                        f"${settings.extraction_cost_limit_usd} limit (confirm to proceed). Regex extraction used.")
    progress = lambda done, total: emit(config, {"type": "progress", "step": "s2_chunk", "done": done, "total": total})
    extractions, usage, warns = ex.extract_units(units, use_llm, progress)
    warns = pre_warn + warns
    by_doc: dict[str, list[dict]] = {}
    for u in units:
        by_doc.setdefault(u["unit_id"].split(":")[0], []).append(u)
    for d in state["documents"]:  # document's own instrument + the title the text gives it
        doc_units = by_doc.get(d["doc_id"], [])
        for e in ex.instrument_titles(doc_units, d.get("title", "")):
            if doc_units:
                extractions[doc_units[0]["unit_id"]]["entities"].append(e)
    methods = Counter(v["method"] for v in extractions.values())
    sizes = [c["n_tokens"] for c in chunks]
    for u in units:
        u["area"] = unit_area.get(u["unit_id"], "")
    return {"chunks": chunks, "units": units, "unit_chunks": unit_chunks, "extractions": extractions,
            "usage": usage, "warnings": warns,
            "_stats": {"chunks": len(chunks), "clause_units": len(units), "chunk_size": hp["chunk_size"],
                       "chunk_overlap": hp["chunk_overlap"], "avg_tokens": round(float(np.mean(sizes)), 1) if sizes else 0,
                       "max_tokens": max(sizes, default=0), "table_row_chunks": sum(c["kind"] == "rows" for c in chunks),
                       "injection_flagged": sum(c["injection_flag"] for c in chunks),
                       "extraction": dict(methods), "extraction_estimate": est,
                       "llm_cost_usd": usage.get("cost_usd", 0), "llm_available": client.available()}}


@traced("s3_tokenize", STEPS[2][1])
def s3_tokenize(state: IngestState, config=None):
    tok = runtime.tokenizer()
    ids, metas, texts, embed_texts = [], [], [], []
    hist = Counter()
    for c in state["chunks"]:
        wins = child_windows(c["text"], tok)
        hist[min(c["n_tokens"] // 100 * 100, 1000)] += 1
        header = f'{c.get("title", "")[:120]} | {c.get("section", "")[:100]}'.strip(" |")
        for w, t in wins:
            ids.append(f'{c["chunk_id"]}#{w}')
            texts.append(t)
            embed_texts.append(f"{header}\n{t}" if header else t)  # contextual header: which document/section
            metas.append({"parent_id": c["chunk_id"], "area": c["practice_area"], "doc_id": c["doc_id"],
                          "source": c["source"], "doc_type": c.get("doc_type", ""), "window": w,
                          "header": header, "injection": int(c["injection_flag"])})
    # entity normalisation is computed here; resolution is refined with embeddings in step 4
    n_mentions = sum(len(v["entities"]) for v in state["extractions"].values())
    return {"children": {"ids": ids, "metas": metas, "texts": texts, "embed_texts": embed_texts},
            "_stats": {"tokenizer": settings.embed_model, "max_seq_len": 512, "child_tokens": Const.CHILD_TOKENS,
                       "child_stride": Const.CHILD_STRIDE, "parents": len(state["chunks"]), "children": len(ids),
                       "children_per_parent": round(len(ids) / max(1, len(state["chunks"])), 1),
                       "token_histogram": {str(k): v for k, v in sorted(hist.items())},
                       "total_tokens": sum(c["n_tokens"] for c in state["chunks"]),
                       "entity_mentions": n_mentions}}


@traced("s4_embed", STEPS[3][1])
def s4_embed(state: IngestState, config=None):
    ch = state["children"]
    progress = lambda done, total: emit(config, {"type": "progress", "step": "s4_embed", "done": done, "total": total})
    import time
    t0 = time.perf_counter()
    vecs, fresh = embed_cached(ch["embed_texts"], progress)
    secs = time.perf_counter() - t0
    unit_area = {u["unit_id"]: u.get("area", "") for u in state["units"]}
    gold = _gold_clusters({d["practice_area"] for d in state["documents"]})
    res = resolve(state["extractions"], state["unit_chunks"], unit_area,
                  lambda names: embed_cached(names)[0], gold_clusters=gold,
                  corpus_text="\n".join(c["text"] for c in state["chunks"]))
    ent_vecs, _ = embed_cached([e["label"] for e in res["entities"]])
    by_type = Counter(e["type"] for e in res["entities"])
    er = res["er"]
    return {"children": {**ch, "vecs": vecs}, "entities": res["entities"], "relations": res["relations"], "er": er,
            "n2v": res["n2v"], "ent_vecs": ent_vecs,
            "_stats": {"model": settings.embed_model, "device": runtime.device(), "dim": int(vecs.shape[1]) if len(vecs) else 0,
                       "vectors": len(vecs), "newly_embedded": fresh, "cached": len(vecs) - fresh,
                       "vectors_per_sec": round(fresh / secs, 1) if fresh and secs else None,
                       "node2vec_dim": 64, "node2vec_nodes": len(res["n2v"]), "entities": len(res["entities"]),
                       "entities_by_type": dict(by_type.most_common()), "relations": len(res["relations"]),
                       "er_f1": {k: v["f1"] for k, v in er.get("variants", {}).items()}}}


@traced("s5_index", STEPS[4][1])
def s5_index(state: IngestState, config=None):
    g = build_graph(state["documents"], state["chunks"], state["entities"], state["relations"])
    ent_nodes = [n for n, d in g.nodes(data=True) if d.get("type") not in ("Chunk", "Document")]
    rels = Counter(d.get("rel") for _, _, d in g.edges(data=True))
    return {"graph": g,
            "_stats": {"hnsw": "Chroma HNSW, cosine", "children_to_index": len(state["children"]["ids"]),
                       "graph_nodes": g.number_of_nodes(), "entity_nodes": len(ent_nodes),
                       "graph_edges": g.number_of_edges(), "relations_by_type": dict(rels.most_common()),
                       "communities": len({d.get("community") for _, d in g.nodes(data=True)}),
                       "bm25": "per practice area + all"}}


@traced("s6_store", STEPS[5][1])
def s6_store(state: IngestState, config=None):
    ch = state["children"]
    s = persist_all(state["documents"], state["chunks"], ch["ids"], ch["vecs"], ch["metas"], ch["texts"],
                    state["entities"], state["ent_vecs"], state["graph"], state["n2v"], state["er"],
                    {"stats": state.get("stats", {}), "hyperparams": state["hyperparams"]})
    hyperparams.mark_ingested()
    return {"_stats": {"vector_db": "ChromaDB (persistent)", "graph_engine": "NetworkX MultiDiGraph", **s}}


def build():
    sg = StateGraph(IngestState)
    fns = [s1_load, s2_chunk, s3_tokenize, s4_embed, s5_index, s6_store]
    for (name, _), fn in zip(STEPS, fns):
        sg.add_node(name, fn)
    sg.add_edge(START, STEPS[0][0])
    for (a, _), (b, _) in zip(STEPS, STEPS[1:]):
        sg.add_edge(a, b)
    sg.add_edge(STEPS[-1][0], END)
    return sg.compile()


ingest_app = build()


def run_ingest(files: list[dict], use_llm: bool, confirm_cost: bool, emit_fn=None, reingest: bool = False) -> dict:
    hp = hyperparams.get().model_dump()
    out = ingest_app.invoke({"files": files, "use_llm": use_llm, "confirm_cost": confirm_cost, "hyperparams": hp,
                             "reingest": reingest, "trace": [], "warnings": [], "errors": []},
                            config={"configurable": {"emit": emit_fn}})
    trace = [{k: v for k, v in t.items() if k not in ("t_start", "t_end")} for t in out.get("trace", [])]
    steps = {t["step"]: t for t in trace}
    import time as _t
    from ..retrieval.stores import KB
    summary = {"trace": trace, "steps": {k: v["stats"] for k, v in steps.items()}, "at": _t.time(),
               "warnings": out.get("warnings", []), "er": out.get("er", {})}
    atomic_write(settings.path("ingest_stats.json"), json.dumps(summary, default=str))
    KB.ingest_stats = summary
    return {"trace": trace, "warnings": out.get("warnings", []), "errors": out.get("errors", []),
            "er": out.get("er", {}), "usage": out.get("usage", {}),
            "documents": len(out.get("documents", [])), "chunks": len(out.get("chunks", [])),
            "entities": len(out.get("entities", [])), "steps": {k: v["stats"] for k, v in steps.items()}}
