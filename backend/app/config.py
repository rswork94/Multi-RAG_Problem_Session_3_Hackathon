"""Settings, fixed benchmark constants and the 4 tunable hyperparameters. Defaults live ONLY here."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

BACKEND_DIR = Path(__file__).resolve().parent.parent
# No .env file is read. The API key comes only from the UI (see llm/client.py). The environment variables below are
# non-secret runtime switches (data dir, local models, test/offline flags).

# Quiet, deterministic, Apple-silicon-friendly ML runtime (must be set before torch/tokenizers import).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class Settings(BaseModel):
    data_dir: Path = Field(default_factory=lambda: (BACKEND_DIR / os.getenv("DATA_DIR", "./data")).resolve())
    corpus_dir: Path = BACKEND_DIR / "corpus"
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    device: str = os.getenv("DEVICE", "auto")  # auto | mps | cpu
    extraction_cost_limit_usd: float = float(os.getenv("EXTRACTION_COST_LIMIT_USD", "1.0"))
    use_llm_scope: bool = os.getenv("USE_LLM_SCOPE", "0") == "1"
    force_offline: bool = os.getenv("MULTIRAG_OFFLINE", "0") == "1"
    fake_embeddings: bool = os.getenv("MULTIRAG_FAKE_EMBED", "0") == "1"  # tests: hashing embedder, no model download
    max_file_mb: int = 25
    allowed_ext: tuple[str, ...] = (".pdf", ".docx", ".json", ".csv", ".txt", ".md")


    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def benchmarks_dir(self) -> Path:
        return self.data_dir / "benchmarks"

    def path(self, name: str) -> Path:
        return self.data_dir / name


settings = Settings()
for _d in (settings.data_dir, settings.uploads_dir, settings.chroma_dir, settings.benchmarks_dir):
    _d.mkdir(parents=True, exist_ok=True)


# Fixed benchmark constants: identical for every engine so only retrieval differs (fairness rules).
class Const:
    FINAL_K = 6                 # parent passages handed to synthesis
    CONTEXT_BUDGET = 4000       # bge tokens of passages + relation triples
    CANDIDATES = 20             # per ranked list (dense / bm25 / graph / each HyDE vector)
    RERANK_CANDIDATES = 12      # fused candidates scored by the cross-encoder
    RERANK_MAX_LEN = 384        # cross-encoder max tokens (latency budget on M1)
    RERANK_FUSION_K = 10        # cross-encoder order fused with retrieval order: 1/(K + rank) each
    CHILD_WINDOW = 510          # bge-large max 512 incl. [CLS]/[SEP] (hard ceiling for any embedded text)
    CHILD_TOKENS = 128          # small-to-big: match on ~128-token windows, return the 600-token parent
    CHILD_STRIDE = 96           # window stride (32-token overlap)
    CHILD_POOL = 160            # child hits gathered per dense / BM25 query before collapsing to parents
    GRAPH_FANOUT = 12           # neighbours expanded per entity per hop
    CHUNKS_PER_ENTITY = 3       # passages pulled per reached entity
    MAX_LOOPS = 1               # Graph RAG sufficiency loop: at most one extra hop
    GRAPH_RESERVED = 0          # evaluated: reserving 1-2 of K slots for PPR bridge passages was zero-sum at K=6 (it
                                # rescued as many facts as it displaced), so graph evidence enters only via weighted RRF
    MAX_HOPS = 3
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    PRACTICE_AREAS = ("tax", "grc", "deals")
    ENGINES = ("standard", "hyde", "graph")


class Hyperparams(BaseModel):
    """The 4 configurable hyperparameters from the brief (chunk size & overlap form one pair)."""

    chunk_size: int = Field(600, ge=200, le=1000, description="Parent chunk size in bge tokens")
    chunk_overlap: int = Field(60, ge=0, le=150, description="Overlap between consecutive chunks (bge tokens)")
    hyde_n: int = Field(3, ge=1, le=5, description="Multi-HyDE synthetic passages per query")
    hop_k: int = Field(2, ge=0, le=3, description="Graph traversal hop radius (0 = no graph expansion)")
    rrf_alpha: int = Field(60, ge=1, le=100, description="RRF constant: score = sum 1/(alpha + rank)")

    @model_validator(mode="after")
    def _overlap_lt_half(self) -> "Hyperparams":
        if self.chunk_overlap >= self.chunk_size // 2:
            raise ValueError("chunk_overlap must be less than chunk_size / 2")
        return self


HYPERPARAM_META = {
    "chunk_size": {"group": "Chunk size & overlap", "min": 200, "max": 1000, "step": 20, "reingest": True,
                   "help": "Parent passage size in bge tokens (what the LLM reads). 600 keeps statutory clauses intact; retrieval matches ~128-token child windows inside it."},
    "chunk_overlap": {"group": "Chunk size & overlap", "min": 0, "max": 150, "step": 10, "reingest": True,
                      "help": "Tokens shared by neighbouring passages so a clause split at a boundary is not lost."},
    "hyde_n": {"group": "Multi-HyDE passages (N)", "min": 1, "max": 5, "step": 1, "reingest": False,
               "help": "Synthetic passages generated per query, each from a different regulatory angle (HyDE engine)."},
    "hop_k": {"group": "Graph hop radius (k)", "min": 0, "max": 3, "step": 1, "reingest": False,
              "help": "Entity-relation hops explored from the linked query entities (Graph RAG). 0 disables traversal."},
    "rrf_alpha": {"group": "RRF constant (alpha)", "min": 1, "max": 100, "step": 1, "reingest": False,
                  "help": "Reciprocal Rank Fusion constant. Larger alpha flattens the advantage of top ranks across BM25, vector and graph lists."},
}


class _HyperparamStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ingested_with: dict | None = None
        self._hp = self._load()

    def _load(self) -> Hyperparams:
        p = settings.path("hyperparams.json")
        if p.exists():
            try:
                data = json.loads(p.read_text())
                self.ingested_with = data.pop("_ingested_with", None)
                return Hyperparams(**data)
            except Exception:
                pass
        return Hyperparams()

    def get(self) -> Hyperparams:
        with self._lock:
            return self._hp.model_copy()

    def set(self, hp: Hyperparams) -> None:
        with self._lock:
            self._hp = hp
            self._save()

    def mark_ingested(self) -> None:
        with self._lock:
            self.ingested_with = {"chunk_size": self._hp.chunk_size, "chunk_overlap": self._hp.chunk_overlap}
            self._save()

    def reingest_required(self) -> bool:
        if not self.ingested_with:
            return False
        return (self.ingested_with.get("chunk_size"), self.ingested_with.get("chunk_overlap")) != \
            (self._hp.chunk_size, self._hp.chunk_overlap)

    def _save(self) -> None:
        data = self._hp.model_dump()
        data["_ingested_with"] = self.ingested_with
        settings.path("hyperparams.json").write_text(json.dumps(data, indent=2))


hyperparams = _HyperparamStore()
