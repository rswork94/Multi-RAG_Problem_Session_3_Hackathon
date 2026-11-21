"""LLM provider settings (switchable at runtime from Tab 1), credential handling and usage/cost accounting.

Providers (both through the official `anthropic` SDK, Messages API):
  anthropic   - api.anthropic.com. Key from the UI, ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN, or an `ant auth` profile.
  openrouter  - OpenRouter's Anthropic-compatible endpoint (base_url https://openrouter.ai/api, bearer key).
                Model ids look like "anthropic/claude-opus-5.5".
The key is never returned to the UI (only its last 4 characters), never logged and never written to disk: it lives in
server memory only and is gone when the server stops. A data/llm_settings.json left by older versions (which could
save the key) is deleted at startup.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic

from ..config import settings

OPENROUTER_BASE = "https://openrouter.ai/api"
DEFAULTS = {
    "anthropic": {"synth_model": "claude-opus-5-5", "utility_model": "claude-haiku-5-5"},
    "openrouter": {"synth_model": "anthropic/claude-opus-5.5", "utility_model": "anthropic/claude-haiku-5.5"},
}
# Anthropic first-party list (claude-api skill model table, cached 2026-11-03). USD per 1M tokens: in, out, cache read.
ANTHROPIC_MODELS = [
    ("claude-opus-5-5", "Claude Opus 5.5", 4.0, 20.0, 0.20),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5", 2.0, 10.0, 0.10),
    ("claude-haiku-5-5", "Claude Haiku 5.5", 0.10, 0.50, 0.01),
    ("claude-fable-5-2", "Claude Fable 5.2", 8.0, 40.0, 0.20),
    ("claude-opus-5", "Claude Opus 5", 5.0, 25.0, 0.50),
    ("claude-sonnet-5", "Claude Sonnet 5", 2.0, 10.0, 0.20),
    ("claude-opus-4-8", "Claude Opus 4.8", 5.0, 25.0, 0.50),
]
_PRICES: dict[str, tuple[float, float, float]] = {m[0]: m[2:] for m in ANTHROPIC_MODELS}
_lock = threading.RLock()
log = logging.getLogger("multirag.llm")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMSettings:
    provider: str = "anthropic"
    api_key: str = field(default="", repr=False)
    synth_model: str = ""
    utility_model: str = ""
    synth_effort: str = "medium"

    def public(self) -> dict:
        d = asdict(self)
        d.pop("api_key")
        d["has_key"] = bool(self.api_key)
        d["key_hint"] = f"…{self.api_key[-4:]}" if len(self.api_key) >= 8 else ""
        return d


def _purge_saved_key() -> None:
    """Older versions could save the key to data/llm_settings.json. Keys are never kept on disk, so delete it."""
    p = settings.path("llm_settings.json")
    if p.exists():
        p.unlink()
        log.warning("Deleted %s: API keys are kept in memory only, never on disk.", p)


def _initial() -> LLMSettings:
    _purge_saved_key()
    provider = os.getenv("LLM_PROVIDER", "openrouter" if os.getenv("OPENROUTER_API_KEY") else "anthropic")
    key = os.getenv("OPENROUTER_API_KEY", "") if provider == "openrouter" else ""
    return LLMSettings(provider=provider, api_key=key,
                       synth_model=os.getenv("SYNTH_MODEL", "") or DEFAULTS[provider]["synth_model"],
                       utility_model=os.getenv("UTILITY_MODEL", "") or DEFAULTS[provider]["utility_model"],
                       synth_effort=os.getenv("SYNTH_EFFORT", "medium"))


_settings = _initial()
_client: anthropic.Anthropic | None = None
CAPS: dict[tuple[str, str, str], int] = {}  # (provider, model, call kind) -> index of the request variant that worked


def current() -> LLMSettings:
    return _settings


def update(provider: str, api_key: str | None, synth_model: str, utility_model: str, synth_effort: str) -> LLMSettings:
    """Applies settings in memory only. Nothing, the key least of all, is written to disk."""
    global _settings, _client
    if provider not in DEFAULTS:
        raise ValueError("provider must be 'anthropic' or 'openrouter'")
    if synth_effort not in ("low", "medium", "high"):
        raise ValueError("effort must be low, medium or high")
    with _lock:
        key = _settings.api_key if api_key is None else api_key.strip()
        if provider != _settings.provider and api_key is None:
            key = ""  # a key belongs to one provider
        _settings = LLMSettings(provider=provider, api_key=key,
                                synth_model=synth_model or DEFAULTS[provider]["synth_model"],
                                utility_model=utility_model or DEFAULTS[provider]["utility_model"],
                                synth_effort=synth_effort)
        _client = None
        CAPS.clear()
    return _settings


def available() -> bool:
    if settings.force_offline:
        return False
    s = _settings
    if s.api_key:
        return True
    if s.provider == "anthropic":
        if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
            return True
        return (Path.home() / ".config" / "anthropic").exists()  # `ant auth login` profile
    return False


def mode() -> str:
    return "online" if available() else "offline"


def client() -> anthropic.Anthropic:
    global _client
    with _lock:
        if _client is None:
            s = _settings
            if s.provider == "openrouter":
                _client = anthropic.Anthropic(api_key=None, auth_token=s.api_key, base_url=OPENROUTER_BASE,
                                              max_retries=3, default_headers={"X-Title": "Multi-RAG Benchmark"})
            elif s.api_key:
                _client = anthropic.Anthropic(api_key=s.api_key, max_retries=3)
            else:
                _client = anthropic.Anthropic(max_retries=3)
        return _client


# ---------------------------------------------------------------- models + prices
_or_cache: dict = {"at": 0.0, "models": []}


def openrouter_models() -> list[dict]:
    """Public OpenRouter catalogue filtered to Anthropic models (no key needed). Cached for an hour."""
    if time.time() - _or_cache["at"] < 3600 and _or_cache["models"]:
        return _or_cache["models"]
    import httpx
    r = httpx.get("https://openrouter.ai/api/v1/models", timeout=20)
    r.raise_for_status()
    out = []
    for m in r.json().get("data", []):
        mid = m.get("id", "")
        if not mid.startswith("anthropic/") or mid.endswith(":batch"):
            continue
        p = m.get("pricing") or {}
        p_in, p_out = float(p.get("prompt") or 0) * 1e6, float(p.get("completion") or 0) * 1e6
        p_cache = float(p.get("input_cache_read") or 0) * 1e6 or p_in * 0.1
        _PRICES[mid] = (p_in, p_out, p_cache)
        out.append({"id": mid, "name": m.get("name", mid), "price_in": round(p_in, 4), "price_out": round(p_out, 4),
                    "created": m.get("created", 0)})
    out.sort(key=lambda m: -m["created"])
    _or_cache.update(at=time.time(), models=out)
    return out


def list_models(provider: str) -> list[dict]:
    if provider == "openrouter":
        return openrouter_models()
    return [{"id": m[0], "name": m[1], "price_in": m[2], "price_out": m[3]} for m in ANTHROPIC_MODELS]


def price(model: str) -> tuple[float, float, float]:
    if model not in _PRICES and _settings.provider == "openrouter":
        try:
            openrouter_models()
        except Exception:
            pass
    return _PRICES.get(model) or _PRICES.get(model.replace("anthropic/", "").replace(".", "-")) or (4.0, 20.0, 0.2)


def usage_dict(msg, model: str) -> dict:
    u = getattr(msg, "usage", None)
    if not u:
        return {}
    d = {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }
    p_in, p_out, p_cache = price(model)
    d["cost_usd"] = round((d["input_tokens"] + d["cache_creation_input_tokens"] * 1.25) * p_in / 1e6
                          + d["cache_read_input_tokens"] * p_cache / 1e6 + d["output_tokens"] * p_out / 1e6, 6)
    return d


def merge_usage(*usages: dict) -> dict:
    out: dict = {}
    for u in usages:
        for k, v in (u or {}).items():
            if isinstance(v, (int, float)):
                out[k] = round(out.get(k, 0) + v, 6)
    return out
