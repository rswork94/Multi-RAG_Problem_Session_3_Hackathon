"""Run a sync pipeline in a worker thread and stream its events as Server-Sent Events."""
from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Callable

from fastapi.responses import StreamingResponse

log = logging.getLogger("multirag.sse")


def sse_line(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def stream(target: Callable[[Callable[[dict], None]], dict]) -> StreamingResponse:
    q: queue.Queue = queue.Queue()
    done = object()

    def worker() -> None:
        try:
            final = target(q.put)
            q.put({"type": "final", **(final or {})})
        except Exception as e:  # noqa: BLE001
            log.exception("pipeline failed")
            q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            q.put(done)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            item = q.get()
            if item is done:
                break
            yield sse_line(item)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
