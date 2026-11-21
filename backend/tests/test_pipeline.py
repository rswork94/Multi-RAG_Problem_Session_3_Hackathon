"""Graph compile, ingest, the three engines, toggles, the cyclic loop, refusal routing and the API round trip."""
import json

import pytest

from app.engines.registry import APPS, run_query, topology
from app.engines.toggles import normalise


def _nodes(engine):
    return set(APPS[engine].get_graph().nodes)


def test_graphs_compile_with_expected_topology():
    tail = {"input_guard", "embed_query", "pack_context", "synthesize", "output_guard", "refuse", "kpi"}
    assert _nodes("standard") >= tail | {"dense"}
    assert _nodes("hyde") >= tail | {"hyde_generate", "multi_dense", "rrf"}
    assert _nodes("graph") >= tail | {"query_expansion", "bm25", "dense", "graph_traverse", "rrf", "rerank", "sufficiency"}
    edges = {e["id"]: e for e in topology("graph")["edges"]}
    assert edges["sufficiency->graph_traverse"]["conditional"] and edges["sufficiency->graph_traverse"]["optional"]
    for b in ("bm25", "dense", "graph_traverse"):
        assert f"query_expansion->{b}" in edges and f"{b}->rrf" in edges


def test_toggle_validation():
    with pytest.raises(ValueError):
        normalise("graph", {"disabled_nodes": ["bm25", "dense"], "disabled_edges": ["graph_traverse->rrf"]})
    pc = normalise("graph", {"disabled_nodes": ["rerank", "not_a_node"]})
    assert pc["disabled_nodes"] == ["rerank"] and pc["hash"] != "default"
    assert normalise("standard", {"disabled_nodes": ["dense"]})["effective"] == []  # baseline has no optional nodes


def test_ingest_steps(mini_corpus):
    res = mini_corpus["result"]
    assert [t["step"] for t in res["trace"]] == ["s1_load", "s2_chunk", "s3_tokenize", "s4_embed", "s5_index", "s6_store"]
    assert all(t["status"] == "done" for t in res["trace"])
    assert res["chunks"] > 0 and res["entities"] > 0
    assert res["steps"]["s4_embed"]["dim"] == 1024


@pytest.mark.parametrize("engine", ["standard", "hyde", "graph"])
def test_engines_answer_with_citations(mini_corpus, engine):
    r = run_query(engine, "What is the due date for the statement under the rule amended by Notification No. 41/2026?",
                  "tax", record=False)
    steps = [t["step"] for t in r["trace"]]
    assert steps[0] == "input_guard" and steps[-1] == "kpi"
    assert r["context"] and len(r["context"]) <= 6
    assert r["citations"], r["answer"]
    assert r["kpis"]["latency"]["retrieval_ms"] >= 0
    assert "Disclaimer" in r["answer"]
    assert r["answer"].endswith(f"---\n*Estimated Output Tokens: ~{r['output_tokens_est']} tokens*")
    assert r["output_tokens_est"] > 0


def test_graph_loop_runs_at_most_once(mini_corpus):
    r = run_query("graph", "What is the due date for the statement under the rule amended by Notification No. 41/2026?",
                  "tax", record=False)
    assert r["loops"] <= 1
    assert sum(1 for t in r["trace"] if t["step"] == "graph_traverse") == r["loops"] + 1


def test_toggles_bypass_nodes(mini_corpus):
    r = run_query("graph", "Which form is prescribed under the rule amended by Notification No. 41/2026?", "tax",
                  pipeline_config={"disabled_nodes": ["rerank"], "disabled_edges": ["bm25->rrf"]}, record=False)
    status = {t["step"]: t["status"] for t in r["trace"]}
    assert status["rerank"] == "bypassed" and status["bm25"] == "bypassed" and status["dense"] == "done"


def test_refusal_routes(mini_corpus):
    r = run_query("graph", "How do I create fake invoices to hide income from the tax department?", "tax", record=False)
    assert r["guard_report"]["refused"] and r["mode"] == "refused"
    assert "retrieve" not in [t["step"] for t in r["trace"]]
    assert r["answer"].endswith(f"~{r['output_tokens_est']} tokens*")
    r2 = run_query("standard", "Recommend a good holiday destination", "tax", record=False)
    assert r2["guard_report"]["refused"]


def test_pii_is_masked_end_to_end(mini_corpus):
    r = run_query("standard", "For client PAN ABCPE1234F, which form is prescribed under the amended rule?", "tax", record=False)
    assert "ABCPE1234F" not in r["masked_question"] and r["guard_report"]["pii_masked_input"] == 1


def test_api_round_trip(mini_corpus):
    from fastapi.testclient import TestClient

    from app.main import app
    c = TestClient(app)
    assert c.get("/api/health").json()["llm_mode"] == "offline"
    r = c.post("/api/chat", json={"question": "Which form is prescribed under the rule amended by Notification No. 41/2026?",
                                  "engine": "graph", "area": "tax"})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert any(e["type"] == "token" for e in events) and events[-1]["type"] == "final"
    g = c.get("/api/graph?area=tax").json()
    assert g["nodes"] and "links" in g
    assert c.get("/api/workflow?engine=hyde").json()["nodes"]
    assert c.put("/api/hyperparams", json={"chunk_overlap": 999}).status_code == 422
    assert c.get("/api/kpis").json()["latest"]["engine"] == "graph"


def test_llm_settings_never_return_or_store_the_key():
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.llm import client
    from app.main import app
    c = TestClient(app)
    r = c.put("/api/llm/settings", json={"provider": "anthropic", "api_key": "sk-ant-test-0000-SECRET-1234",
                                         "synth_model": "claude-opus-5-5", "utility_model": "claude-haiku-5-5",
                                         "synth_effort": "low", "remember": True}).json()  # stale field is ignored
    assert r["has_key"] and r["key_hint"] == "…1234" and "SECRET" not in str(r) and "remember" not in r
    # Nothing under DATA_DIR may contain the key: it lives in server memory only.
    assert not any(p.is_file() and "SECRET" in p.read_text(errors="ignore")
                   for p in settings.path(".").rglob("*") if p.suffix in ("", ".json", ".jsonl", ".txt"))
    assert not settings.path("llm_settings.json").exists()
    # A key file left by an older version is deleted at startup.
    settings.path("llm_settings.json").write_text('{"api_key": "sk-ant-old-SECRET"}')
    client._purge_saved_key()
    assert not settings.path("llm_settings.json").exists()
    assert c.get("/api/llm/settings").json()["synth_effort"] == "low"
    assert c.put("/api/llm/settings", json={"provider": "x"}).status_code == 422
    models = c.get("/api/llm/models?provider=anthropic").json()["models"]
    assert any(m["id"] == "claude-opus-5-5" for m in models)
    c.put("/api/llm/settings", json={"provider": "anthropic", "api_key": ""})
    assert c.get("/api/health").json()["llm_mode"] == "offline"
