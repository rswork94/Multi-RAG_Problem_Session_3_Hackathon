"""The three kinds of Claude call the app makes, provider-aware.

- stream_synthesis: answer model (Opus 5.5 by default), streamed. Identical for all three engines.
- structured:       utility model (Haiku 5.5), JSON output.
- short_text:       utility model, plain text (HyDE passages), lowest latency settings.

Each call has an ordered list of request variants, from the full Anthropic-native request down to the plainest one.
On a 400 (a parameter the provider does not accept, e.g. via OpenRouter) the next variant is tried, and the variant
that worked is remembered per (provider, model, call kind). Never sent: temperature/top_p/top_k, assistant prefill,
forced tool_choice, or thinking disabled on Opus 5.5.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable

import anthropic

from . import client as C
from .client import LLMError


def _err(e: Exception) -> LLMError:
    if isinstance(e, anthropic.AuthenticationError):
        return LLMError("Authentication failed: check the API key in Tab 1 → Model settings.")
    if isinstance(e, anthropic.NotFoundError):
        return LLMError(f"Model not found for this provider: {getattr(e, 'message', e)}")
    if isinstance(e, anthropic.RateLimitError):
        return LLMError("Rate limit reached; retry shortly.")
    if isinstance(e, anthropic.APIStatusError):
        return LLMError(f"API error {e.status_code}: {e.message}")
    if isinstance(e, anthropic.APIConnectionError):
        return LLMError(f"Could not reach the API: {e}")
    return LLMError(str(e))


def _run_variants(kind: str, model: str, variants: list[Callable[[], object]]):
    """Try request variants in order; remember the first that the provider accepts."""
    key = (C.current().provider, model, kind)
    start = C.CAPS.get(key, 0)
    last: Exception | None = None
    for i in range(start, len(variants)):
        try:
            out = variants[i]()
            C.CAPS[key] = i
            return out, i
        except anthropic.BadRequestError as e:  # unsupported parameter: degrade to the next variant
            last = e
            continue
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            raise _err(e) from e
    raise _err(last) if last else LLMError("no request variant succeeded")


def stream_synthesis(system_blocks: list[dict], user: str, on_token: Callable[[str], None]) -> dict:
    """Returns {text, stop_reason, usage, model, fallback_used, ttft_ms, ms, variant}."""
    s = C.current()
    model, cli = s.synth_model, C.client()
    messages = [{"role": "user", "content": user}]
    plain_system = "\n\n".join(b["text"] for b in system_blocks)
    state = {"parts": [], "ttft": None}
    t0 = time.perf_counter()

    def consume(stream):
        for text in stream.text_stream:
            if state["ttft"] is None:
                state["ttft"] = (time.perf_counter() - t0) * 1000
            state["parts"].append(text)
            on_token(text)
        return stream.get_final_message()

    def native():  # Anthropic API: effort + server-side refusal fallbacks (beta) + cached system blocks
        with cli.beta.messages.stream(model=model, max_tokens=16000, system=system_blocks, messages=messages,
                                      output_config={"effort": s.synth_effort},
                                      betas=["server-side-fallback-2026-07-01"], fallbacks="default") as st:
            return consume(st)

    def effort_only():
        with cli.messages.stream(model=model, max_tokens=16000, system=system_blocks, messages=messages,
                                 output_config={"effort": s.synth_effort}) as st:
            return consume(st)

    def plain():
        with cli.messages.stream(model=model, max_tokens=16000, system=plain_system, messages=messages) as st:
            return consume(st)

    variants = [native, effort_only, plain] if s.provider == "anthropic" else [effort_only, plain]
    try:
        final, vi = _run_variants("synthesis", model, variants)
    except LLMError:
        raise
    text = "".join(state["parts"])
    stop = getattr(final, "stop_reason", None) or "end_turn"
    served_by = getattr(final, "model", model) or model
    iters = getattr(getattr(final, "usage", None), "iterations", None) or []
    fallback_used = any(getattr(it, "type", "") == "fallback_message" for it in iters)
    if stop == "refusal":
        cat = getattr(getattr(final, "stop_details", None), "category", None)
        note = f"\n\n_The model declined to answer (safety category: {cat or 'unspecified'})._"
        text += note
        on_token(note)
    elif stop == "max_tokens":
        note = "\n\n_(Answer truncated at the token limit.)_"
        text += note
        on_token(note)
    return {"text": text, "stop_reason": stop, "usage": C.usage_dict(final, model), "model": served_by,
            "fallback_used": fallback_used, "ttft_ms": round(state["ttft"] or 0, 1),
            "ms": round((time.perf_counter() - t0) * 1000, 1), "variant": variants[vi].__name__}


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text or "")
        if not m:
            raise
        return json.loads(m.group(0))


def structured(system: str, user: str, schema: dict, model: str | None = None, effort: str = "low",
               max_tokens: int = 4000) -> tuple[dict, dict]:
    """One JSON-output call on the utility model. Returns (parsed_json, usage)."""
    model = model or C.current().utility_model
    cli = C.client()
    messages = [{"role": "user", "content": user}]
    instructed = (system + "\n\nReturn ONLY a JSON object (no prose, no code fences) that validates against this JSON "
                  "schema:\n" + json.dumps(schema))

    def call(**kw):
        resp = cli.messages.create(model=model, max_tokens=max_tokens, messages=messages, **kw)
        if resp.stop_reason == "refusal":
            raise LLMError("Model declined the request (refusal).")
        return resp

    variants = [
        lambda: call(system=system, output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}}),
        lambda: call(system=system, output_config={"format": {"type": "json_schema", "schema": schema}}),
        lambda: call(system=instructed),
    ]
    for attempt in range(3):
        try:
            resp, _ = _run_variants("structured", model, variants)
            text = next((b.text for b in resp.content if b.type == "text"), "")
            return _parse_json(text), C.usage_dict(resp, model)
        except LLMError as e:
            if "Rate limit" in str(e) and attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            raise
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON from model: {e}") from e
    raise LLMError("Rate limited after retries.")


def short_text(system: str, user: str, model: str | None = None, max_tokens: int = 600) -> tuple[str, dict]:
    """Fast plain-text call (HyDE passages). Haiku 5.5 accepts thinking disabled at low effort."""
    model = model or C.current().utility_model
    cli = C.client()
    messages = [{"role": "user", "content": user}]

    def call(**kw):
        return cli.messages.create(model=model, max_tokens=max_tokens, system=system, messages=messages, **kw)

    variants = [lambda: call(thinking={"type": "disabled"}, output_config={"effort": "low"}),
                lambda: call(output_config={"effort": "low"}),
                lambda: call()]
    resp, _ = _run_variants("short_text", model, variants)
    if resp.stop_reason == "refusal":
        raise LLMError("Model declined the request (refusal).")
    return "".join(b.text for b in resp.content if b.type == "text").strip(), C.usage_dict(resp, model)


def test_connection() -> dict:
    """One tiny call to the utility model: proves the key, provider and model id work."""
    s = C.current()
    t0 = time.perf_counter()
    text, usage = short_text("Reply with exactly: OK", "Connection test.", max_tokens=20)
    return {"ok": True, "provider": s.provider, "model": s.utility_model, "reply": text[:40],
            "ms": round((time.perf_counter() - t0) * 1000), "cost_usd": usage.get("cost_usd", 0),
            "variant": C.CAPS.get((s.provider, s.utility_model, "short_text"))}
