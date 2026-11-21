"""Process-wide ML singletons: bge-large embedder (LangChain HuggingFaceEmbeddings), tokenizer, reranker.

All model calls go through `compute_lock`: LangGraph runs parallel branches in a thread pool and concurrent
MPS calls are unsafe. Latency measured inside the lock excludes time spent waiting for it.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from typing import Sequence

import numpy as np
from langchain_core.embeddings import Embeddings

from .config import Const, settings

log = logging.getLogger("multirag.runtime")
compute_lock = threading.RLock()
_init_lock = threading.Lock()
_embedder: Embeddings | None = None
_tokenizer = None
_reranker = None
_device: str | None = None


def device() -> str:
    global _device
    if _device is None:
        if settings.device in ("cpu", "mps"):
            _device = settings.device
        else:
            try:
                import torch
                _device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                _device = "cpu"
    return _device


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embedder (1024-d). Used by tests so no model download is needed."""

    dim = 1024

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0 if (h >> 9) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def embedder() -> Embeddings:
    global _embedder
    with _init_lock:
        if _embedder is None:
            if settings.fake_embeddings:
                _embedder = HashingEmbeddings()
            else:
                from langchain_huggingface import HuggingFaceEmbeddings
                t0 = time.time()
                _embedder = HuggingFaceEmbeddings(
                    model_name=settings.embed_model,
                    model_kwargs={"device": device()},
                    encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
                    query_encode_kwargs={"normalize_embeddings": True, "prompt": Const.QUERY_PREFIX},
                )
                log.info("Loaded %s on %s in %.1fs", settings.embed_model, device(), time.time() - t0)
        return _embedder


def tokenizer():
    """bge WordPiece tokenizer (loaded on its own so chunking does not need the 1.3 GB model)."""
    global _tokenizer
    with _init_lock:
        if _tokenizer is None:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(settings.embed_model)
        return _tokenizer


def count_tokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])


def embed_passages(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    with compute_lock:
        return np.asarray(embedder().embed_documents(list(texts)), dtype=np.float32)


def embed_query(text: str) -> tuple[np.ndarray, float]:
    """Returns (vector, ms measured inside the lock)."""
    with compute_lock:
        t0 = time.perf_counter()
        v = np.asarray(embedder().embed_query(text), dtype=np.float32)
        return v, (time.perf_counter() - t0) * 1000


def reranker():
    global _reranker
    with _init_lock:
        if _reranker is None and not settings.fake_embeddings:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(settings.rerank_model, device=device(), max_length=Const.RERANK_MAX_LEN)
        return _reranker


def rerank_scores(query: str, passages: Sequence[str]) -> tuple[list[float], float]:
    """Cross-encoder relevance scores (or lexical overlap in fake mode). Returns (scores, ms inside lock)."""
    with compute_lock:
        t0 = time.perf_counter()
        ce = reranker()
        if ce is None:
            q = set(re.findall(r"[a-z0-9]+", query.lower()))
            scores = [len(q & set(re.findall(r"[a-z0-9]+", p.lower()))) / (len(q) or 1) for p in passages]
        else:
            scores = [float(s) for s in ce.predict([(query, p) for p in passages], batch_size=len(passages) or 1)]
        return scores, (time.perf_counter() - t0) * 1000


def warm_up() -> dict:
    """Load models and compile common MPS kernels so the first user query is not 1-3 s slower."""
    t0 = time.time()
    embed_query("warm up query about tax deduction at source")
    embed_query("what penalty applies for failure to update KYC records of high risk customers within the period")
    embed_passages(["warm up passage " * 40])
    rerank_scores("warm up", ["warm up passage " * 40] * 2)
    return {"device": device(), "seconds": round(time.time() - t0, 1)}
