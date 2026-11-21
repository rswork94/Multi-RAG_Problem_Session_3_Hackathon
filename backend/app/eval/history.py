"""Append-only run history (Tab 4 telemetry) and audit log. Stores the masked question, never raw PII."""
from __future__ import annotations

import hashlib
import json
import threading

from ..config import settings

_lock = threading.Lock()


def record(res: dict) -> None:
    row = {
        "at": res.get("at"), "engine": res.get("engine"), "area": res.get("area"),
        "question": (res.get("masked_question") or "")[:300],
        "question_sha": hashlib.sha256((res.get("question") or "").encode()).hexdigest()[:16],
        "gold_id": res.get("gold_id"), "category": res.get("gold_category"), "mode": res.get("mode"),
        "model": res.get("model_used"), "fallback_used": res.get("fallback_used"),
        "pipeline_hash": (res.get("pipeline") or {}).get("hash", "default"),
        "disabled": (res.get("pipeline") or {}).get("effective", []),
        "hyperparams": res.get("hyperparams"), "kpis": {k: v for k, v in (res.get("kpis") or {}).items()
                                                         if k not in ("rcr_facts", "unsupported_claims")},
        "citations": [c["chunk_id"] for c in res.get("citations") or []],
        "refused": (res.get("guard_report") or {}).get("refused", False),
        "cost_usd": (res.get("usage") or {}).get("cost_usd", 0),
    }
    with _lock, settings.path("runs.jsonl").open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def read(limit: int = 200) -> list[dict]:
    p = settings.path("runs.jsonl")
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines()[-limit:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out
