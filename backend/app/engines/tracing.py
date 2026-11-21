"""`traced`: wraps a LangGraph node, records {step, status, duration_ms, stats} into state['trace'] and emits
live SSE events through config['configurable']['emit'] (Python 3.10: sync nodes + a worker-thread queue)."""
from __future__ import annotations

import functools
import logging
import time
import traceback
from typing import Callable

log = logging.getLogger("multirag.trace")


def emit(config, event: dict) -> None:
    fn = ((config or {}).get("configurable") or {}).get("emit")
    if fn:
        try:
            fn(event)
        except Exception:
            pass


def cfg(config, key: str, default=None):
    return ((config or {}).get("configurable") or {}).get(key, default)


class StepFailed(RuntimeError):
    def __init__(self, msg: str, partial: dict):
        super().__init__(msg)
        self.partial = partial


def traced(step: str, label: str, fatal: bool = True):
    """fatal=False (retriever branches): an exception becomes an error entry and an empty result."""

    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapper(state, config=None):
            emit(config, {"type": "step", "step": step, "label": label, "status": "running"})
            t0 = time.perf_counter()
            try:
                out = fn(state, config) or {}
                status, err = "done", None
            except StepFailed:
                raise
            except Exception as e:  # surface failures to the UI instead of a 500
                traceback.print_exc()
                out, status, err = {}, "error", f"{type(e).__name__}: {e}"
            t1 = time.perf_counter()
            ms = round((t1 - t0) * 1000, 1)
            stats = out.pop("_stats", {})
            entry = {"step": step, "label": label, "status": status, "duration_ms": ms, "stats": stats,
                     "t_start": t0, "t_end": t1}
            if err:
                entry["error"] = err
                out["errors"] = [f"{label}: {err}"]
            out["trace"] = [entry]
            out.setdefault("timings", {})
            out["timings"] = {**out["timings"], f"node:{step}": ms}
            emit(config, {"type": "step", **entry})
            if err and fatal:
                raise StepFailed(err, out)
            return out
        wrapper.step = step
        wrapper.label = label
        return wrapper
    return deco
