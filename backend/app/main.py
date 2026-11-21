"""FastAPI app: PwC India Multi-RAG benchmark (Standard vs HyDE vs Graph RAG)."""
from __future__ import annotations

import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from . import runtime
from .api.sse import stream
from .config import BACKEND_DIR, HYPERPARAM_META, Const, Hyperparams, hyperparams, settings
from .engines import registry
from .engines.ingest_graph import STEPS, run_ingest
from .engines.toggles import OPTIONAL
from .eval import benchmark, gold, history
from .eval.metrics import KPI_META
from .llm import calls, client
from .prompts.areas import AREAS
from .retrieval.stores import KB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("multirag")

app = FastAPI(title="PwC India Multi-RAG Benchmark", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
_ingest_lock = threading.Lock()
_bench_lock = threading.Lock()
_warm = {"status": "pending"}


@app.on_event("startup")
def _startup() -> None:
    KB.ensure()

    def warm() -> None:
        _warm["status"] = "warming"
        try:
            _warm.update(runtime.warm_up(), status="ready")
        except Exception as e:  # noqa: BLE001
            _warm.update(status=f"failed: {e}")
    threading.Thread(target=warm, daemon=True).start()


# ---------------------------------------------------------------- meta
@app.get("/api/health")
def health():
    kb = KB.ensure()
    llm = client.current()
    return {"llm_mode": client.mode(), "provider": llm.provider, "synth_model": llm.synth_model,
            "utility_model": llm.utility_model, "synth_effort": llm.synth_effort,
            "embed_model": settings.embed_model, "rerank_model": settings.rerank_model, "device": runtime.device(),
            "warm": _warm, "documents": len(kb.documents), "chunks": len(kb.parents),
            "graph_nodes": kb.graph.number_of_nodes(), "corpus_version": gold.manifest().get("version"),
            "gold_questions": len(gold.questions()), "reingest_required": hyperparams.reingest_required()}


@app.get("/api/meta")
def meta():
    return {"areas": AREAS, "engines": registry.ENGINE_INFO, "hyperparams": HYPERPARAM_META, "kpis": KPI_META,
            "optional": OPTIONAL, "ingest_steps": [{"step": s, "label": l} for s, l in STEPS],
            "constants": {k: getattr(Const, k) for k in ("FINAL_K", "CONTEXT_BUDGET", "CANDIDATES", "RERANK_CANDIDATES",
                                                          "CHILD_WINDOW", "MAX_LOOPS", "MAX_HOPS")},
            "variants": {k: v["label"] for k, v in benchmark.VARIANTS.items()}, "winner_rule": benchmark.WINNER_RULE,
            "expected": benchmark.EXPECTED, "categories": benchmark.CATEGORY_LABEL}


# ---------------------------------------------------------------- LLM provider / model settings (Tab 1)
class LLMSettingsBody(BaseModel):
    provider: str = "anthropic"
    api_key: str | None = None  # None keeps the current key; "" clears it
    synth_model: str = ""
    utility_model: str = ""
    synth_effort: str = "medium"


@app.get("/api/llm/settings")
def get_llm_settings():
    import os
    return {**client.current().public(), "mode": client.mode(),
            "env_credentials": bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))}


@app.put("/api/llm/settings")
def put_llm_settings(body: LLMSettingsBody):
    try:
        client.update(body.provider, body.api_key, body.synth_model, body.utility_model, body.synth_effort)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return get_llm_settings()


@app.get("/api/llm/models")
def llm_models(provider: str = "anthropic"):
    try:
        return {"provider": provider, "models": client.list_models(provider)}
    except Exception as e:  # noqa: BLE001 - catalogue fetch failed (offline machine)
        raise HTTPException(502, f"Could not fetch the {provider} model list: {e}")


@app.post("/api/llm/test")
def llm_test():
    if not client.available():
        raise HTTPException(400, "No API key configured for the selected provider.")
    try:
        return calls.test_connection()
    except client.LLMError as e:
        raise HTTPException(502, str(e))


# ---------------------------------------------------------------- ingest
def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\-]", "_", Path(name).name)[:120]


@app.post("/api/ingest")
async def ingest(files: list[UploadFile] = File(...), area: str = Form("tax"), use_llm: bool = Form(True),
                 confirm_cost: bool = Form(False)):
    if area not in Const.PRACTICE_AREAS:
        raise HTTPException(400, f"area must be one of {Const.PRACTICE_AREAS}")
    saved = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in settings.allowed_ext:
            raise HTTPException(400, f"{f.filename}: unsupported type {ext}")
        data = await f.read()
        if len(data) > settings.max_file_mb * 1024 * 1024:
            raise HTTPException(400, f"{f.filename}: larger than {settings.max_file_mb} MB")
        p = settings.uploads_dir / f"{uuid.uuid4().hex[:8]}_{_safe_name(f.filename)}"
        p.write_bytes(data)
        saved.append({"path": str(p), "filename": _safe_name(f.filename), "area": area})
    return _ingest_stream(saved, use_llm, confirm_cost)


@app.post("/api/ingest/gold")
def ingest_gold(use_llm: bool = True, confirm_cost: bool = False):
    root = settings.corpus_dir / "generated"
    files = [{"path": str(p), "filename": p.name, "area": p.parent.name}
             for p in sorted(root.glob("*/*")) if p.parent.name in Const.PRACTICE_AREAS]
    if not files:
        raise HTTPException(404, "No generated corpus. Run: .venv/bin/python scripts/build_corpus.py")
    gold.reload()
    return _ingest_stream(files, use_llm, confirm_cost)


@app.post("/api/reingest")
def reingest(use_llm: bool = True, confirm_cost: bool = False):
    return _ingest_stream([], use_llm, confirm_cost, reingest=True)


def _ingest_stream(files, use_llm, confirm_cost, reingest=False):
    def target(emit):
        if not _ingest_lock.acquire(blocking=False):
            raise RuntimeError("Another ingest is running")
        try:
            return run_ingest(files, use_llm, confirm_cost, emit, reingest=reingest)
        finally:
            _ingest_lock.release()
    return stream(target)


@app.get("/api/documents")
def documents():
    kb = KB.ensure()
    counts: dict = {}
    for c in kb.parents.values():
        counts[c["doc_id"]] = counts.get(c["doc_id"], 0) + 1
    return [{**d, "chunks": counts.get(d["doc_id"], 0)} for d in kb.documents]


@app.delete("/api/data")
def reset_data():
    with _ingest_lock:
        KB.reset()
        hyperparams.ingested_with = None
    return {"ok": True}


@app.get("/api/chunks/{chunk_id}")
def chunk(chunk_id: str):
    c = KB.ensure().parents.get(chunk_id)
    if not c:
        raise HTTPException(404, "chunk not found")
    return c


# ---------------------------------------------------------------- chat / compare
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    engine: str = "graph"
    area: str = "tax"
    selected_node_ids: list[str] = []
    blocked_edge_ids: list[str] = []
    pipeline_config: dict = {}
    all_areas: bool = False


@app.post("/api/chat")
def chat(req: ChatRequest):
    if req.engine not in registry.APPS or req.area not in Const.PRACTICE_AREAS:
        raise HTTPException(400, "unknown engine or practice area")

    def target(emit):
        return registry.run_query(req.engine, req.question, req.area, req.selected_node_ids, req.pipeline_config, emit,
                                  all_areas=req.all_areas, blocked_edges=req.blocked_edge_ids)
    return stream(target)


@app.post("/api/compare")
def compare(req: ChatRequest):
    """One question through all three engines in parallel; events are tagged with the engine."""
    def target(emit):
        def one(engine):
            tag = lambda e: emit({**e, "engine": engine})
            return engine, registry.run_query(engine, req.question, req.area, req.selected_node_ids, {}, tag,
                                              all_areas=req.all_areas, blocked_edges=req.blocked_edge_ids)
        with ThreadPoolExecutor(3) as ex:
            results = dict(ex.map(one, ("standard", "hyde", "graph")))
        return {"results": results}
    return stream(target)


# ---------------------------------------------------------------- graph / workflow
@app.get("/api/graph")
def graph(area: str = "all", include_chunks: bool = False, limit: int = 2500):
    g = KB.ensure().graph
    keep = []
    for n, d in g.nodes(data=True):
        t = d.get("type")
        if t in ("Chunk", "Document") and not include_chunks:
            continue
        if area != "all" and area not in (d.get("area") or ""):
            continue
        keep.append(n)
    keep = sorted(keep, key=lambda n: -g.nodes[n].get("pagerank", 0))[:limit]
    ks = set(keep)
    nodes = [{"id": n, "label": g.nodes[n].get("label", n), "type": g.nodes[n].get("type"),
              "community": g.nodes[n].get("community", 0), "n2v_cluster": g.nodes[n].get("n2v_cluster"),
              "pagerank": g.nodes[n].get("pagerank", 0), "degree": g.nodes[n].get("degree", 0),
              "area": g.nodes[n].get("area", ""), "aliases": g.nodes[n].get("aliases", ""),
              "mentions": g.nodes[n].get("mentions", 0)} for n in keep]
    links = [{"id": f"{u}|{k}|{v}", "source": u, "target": v, "rel": d.get("rel"), "weight": d.get("weight", 1),
              "evidence": d.get("evidence", "")}
             for u, v, k, d in g.edges(keys=True, data=True) if u in ks and v in ks]
    return {"nodes": nodes, "links": links, "er": KB.er_report}


@app.get("/api/workflow")
def workflow(engine: str = "graph"):
    if engine == "ingest":
        nodes = [{"id": "__start__", "label": "__start__"}] + [{"id": s, "label": l, "locked": True} for s, l in STEPS] + \
            [{"id": "__end__", "label": "__end__"}]
        ids = [n["id"] for n in nodes]
        return {"engine": "ingest", "label": "Ingestion (6 steps)", "nodes": nodes,
                "edges": [{"id": f"{a}->{b}", "source": a, "target": b, "conditional": False, "optional": False}
                          for a, b in zip(ids, ids[1:])],
                "last": {"trace": (KB.ingest_stats or {}).get("trace")}}
    if engine not in registry.APPS:
        raise HTTPException(400, "unknown engine")
    return registry.topology(engine)


# ---------------------------------------------------------------- hyperparameters & KPIs
@app.get("/api/hyperparams")
def get_hp():
    return {"values": hyperparams.get().model_dump(), "meta": HYPERPARAM_META,
            "reingest_required": hyperparams.reingest_required(), "ingested_with": hyperparams.ingested_with}


@app.put("/api/hyperparams")
def put_hp(values: dict):
    try:
        hp = Hyperparams(**{**hyperparams.get().model_dump(), **values})
    except ValidationError as e:
        raise HTTPException(422, e.errors()[0]["msg"])
    hyperparams.set(hp)
    return get_hp()


@app.get("/api/kpis")
def kpis(limit: int = Query(200, le=2000)):
    h = history.read(limit)
    return {"latest": h[-1] if h else None, "history": h, "meta": KPI_META, "er": KB.ensure().er_report}


# ---------------------------------------------------------------- gold + benchmark
@app.get("/api/gold")
def gold_questions():
    return [{k: q.get(k) for k in ("id", "area", "category", "question", "hops")} for q in gold.questions()]


class BenchRequest(BaseModel):
    areas: list[str] = list(Const.PRACTICE_AREAS)
    categories: list[str] | None = None
    variants: list[str] = ["standard", "hyde", "graph"]
    quick: bool = False
    confirm_cost: bool = False


@app.post("/api/benchmark/estimate")
def bench_estimate(req: BenchRequest):
    qs = benchmark.select_questions(req.areas, req.categories, req.quick)
    return benchmark.estimate(len(qs), req.variants)


@app.post("/api/benchmark")
def bench_run(req: BenchRequest):
    qs = benchmark.select_questions(req.areas, req.categories, req.quick)
    est = benchmark.estimate(len(qs), req.variants)
    if est["online"] and est["est_cost_usd"] > 0 and not req.confirm_cost:
        raise HTTPException(409, f"Estimated cost ${est['est_cost_usd']}: confirm to run")
    if KB.ensure().empty:
        raise HTTPException(409, "Knowledge base is empty: load the gold corpus first")

    def target(emit):
        if not _bench_lock.acquire(blocking=False):
            raise RuntimeError("A benchmark is already running")
        try:
            return {"result": benchmark.run(req.areas, req.categories, req.variants, req.quick, emit)}
        finally:
            _bench_lock.release()
    return stream(target)


@app.get("/api/benchmark/runs")
def bench_runs():
    return benchmark.list_runs()


@app.get("/api/benchmark/{bid}")
def bench_get(bid: str):
    if not re.fullmatch(r"[\w-]+", bid):
        raise HTTPException(400, "bad id")
    r = benchmark.load(bid)
    if not r:
        raise HTTPException(404, "benchmark not found")
    return r


@app.get("/api/benchmark/{bid}/export.csv", response_class=PlainTextResponse)
def bench_csv(bid: str):
    r = bench_get(bid)
    return PlainTextResponse(benchmark.to_csv(r), headers={"Content-Disposition": f'attachment; filename="benchmark-{bid}.csv"'})


# ---------------------------------------------------------------- static frontend (single-port mode)
DIST = BACKEND_DIR.parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = DIST / path
        if path and f.is_file() and DIST in f.resolve().parents:
            return FileResponse(f)
        # never cache the HTML shell: it names the hashed bundle of the current build
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})
