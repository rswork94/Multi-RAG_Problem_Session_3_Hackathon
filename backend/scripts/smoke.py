"""End-to-end smoke run: ingest the generated gold corpus, then an offline benchmark. Never calls the Claude API.

Usage:  MULTIRAG_OFFLINE=1 .venv/bin/python scripts/smoke.py [--quick] [--no-ingest]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["MULTIRAG_OFFLINE"] = "1"  # guarantee no API spend, whatever credentials exist
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.engines.ingest_graph import run_ingest  # noqa: E402
from app.eval import benchmark  # noqa: E402


def _step(e: dict) -> None:
    if e.get("type") == "step" and e.get("status") != "running":
        print(f'  {e["step"]:12s} {e["status"]:8s} {e.get("duration_ms", 0):9.0f} ms', flush=True)


def main() -> None:
    if "--no-ingest" not in sys.argv:
        files = [{"path": str(p), "filename": p.name, "area": p.parent.name}
                 for p in sorted((settings.corpus_dir / "generated").glob("*/*"))]
        t0 = time.time()
        r = run_ingest(files, use_llm=False, confirm_cost=False, emit_fn=_step)
        print(f"ingest {time.time() - t0:.0f}s: {r['documents']} docs, {r['chunks']} chunks, {r['entities']} entities")
        print("ER F1:", json.dumps({k: v.get("f1") for k, v in r["er"].get("variants", {}).items()}))
        for w in r["warnings"][:5]:
            print("  warning:", w)
    res = benchmark.run(["tax", "grc", "deals"], None, ["standard", "hyde", "graph", "graph_no_graph", "graph_no_rerank"],
                        quick="--quick" in sys.argv)
    agg = res["aggregate"]
    print(f"\nbenchmark {res['id']} ({res['llm_mode']}), {res['n_questions']} questions")
    print(f"{'variant':18s} {'category':18s} {'RCR':>6s} {'faith':>6s} {'retr p50':>9s} {'pass':>5s}")
    for key, c in agg["cells"].items():
        v, cat = key.split("|")
        print(f"{v:18s} {cat:18s} {c['rcr'] or 0:6.2f} {c['faithfulness'] or 0:6.2f} {c['retrieval_p50'] or 0:9.0f} "
              f"{c['answer_pass_rate'] or 0:5.2f}")
    print("winners:", agg["winners"])
    for row in res["expected_vs_measured"]:
        print(f"  {row['metric']:30s} {row['verdict']}: {row['note']}")


if __name__ == "__main__":
    main()
