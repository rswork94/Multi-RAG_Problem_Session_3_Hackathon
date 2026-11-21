"""Storage seams + the in-process knowledge base.

VectorIndex  -> ChromaVectorIndex (HNSW, cosine). QdrantVectorIndex is a documented stub.
GraphStore   -> NetworkXGraphStore (MultiDiGraph adjacency). Neo4jGraphStore is a documented stub.
BM25         -> rank_bm25 per practice area with a legal-aware tokenizer.
KnowledgeBase holds everything the engines read; it is reloaded after each ingest.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import threading
from pathlib import Path
from typing import Protocol

import networkx as nx
import numpy as np

from ..config import settings

STOP = set("the a an of to in for on and or is are be by with as at from that this which it its under any such shall "
           "may not no if than other been has have was were will".split())
TOKEN = re.compile(r"[a-z0-9₹]+(?:[./-][a-z0-9]+)*(?:\([a-z0-9]{1,4}\))*")


def legal_tokens(text: str) -> list[str]:
    """Keeps provision identifiers intact: '194j', '16(4)', '41/2026', 'c-kyc-07', '9.4(c)'."""
    return [t for t in TOKEN.findall((text or "").lower()) if t not in STOP]


def atomic_write(path: Path, data: str | bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data if isinstance(data, bytes) else data.encode())
    os.replace(tmp, path)


# ---------------------------------------------------------------- protocols
class VectorIndex(Protocol):
    def upsert(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict], documents: list[str]) -> None: ...
    def query(self, vector: np.ndarray, n: int, where: dict | None = None) -> list[tuple[str, float, dict, str]]: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...


class GraphStore(Protocol):
    def load(self) -> nx.MultiDiGraph: ...
    def save(self, g: nx.MultiDiGraph) -> None: ...


_chroma_lock = threading.Lock()
_chroma_client = None


def _chroma():
    global _chroma_client
    with _chroma_lock:
        if _chroma_client is None:
            import chromadb
            from chromadb.config import Settings as CSettings
            _chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir),
                                                       settings=CSettings(anonymized_telemetry=False))
        return _chroma_client


class ChromaVectorIndex:
    """Embeddings are always supplied by us (never Chroma's default ONNX function); metadata is scalar-only."""

    def __init__(self, name: str) -> None:
        self.name = name

    def _coll(self):
        return _chroma().get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def upsert(self, ids, embeddings, metadatas, documents) -> None:
        coll = self._coll()
        for i in range(0, len(ids), 500):
            coll.upsert(ids=ids[i:i + 500], embeddings=np.asarray(embeddings[i:i + 500]).tolist(),
                        metadatas=metadatas[i:i + 500], documents=documents[i:i + 500])

    def query(self, vector, n, where=None):
        coll = self._coll()
        total = coll.count()
        if total == 0:
            return []
        res = coll.query(query_embeddings=[np.asarray(vector).tolist()], n_results=min(n, total), where=where or None,
                         include=["metadatas", "distances", "documents"])
        return [(i, 1 - d, m, doc) for i, d, m, doc in
                zip(res["ids"][0], res["distances"][0], res["metadatas"][0], res["documents"][0])]

    def count(self) -> int:
        return self._coll().count()

    def reset(self) -> None:
        try:
            _chroma().delete_collection(self.name)
        except Exception:
            pass


class QdrantVectorIndex:  # pragma: no cover - seam only
    """Drop-in seam for Qdrant (HNSW, cosine). Not built: the hackathon build is embedded Chroma by decision."""

    def __init__(self, *_, **__):
        raise NotImplementedError("Qdrant adapter not built; set VECTOR_BACKEND=chroma")


class NetworkXGraphStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> nx.MultiDiGraph:
        if not self.path.exists():
            return nx.MultiDiGraph()
        return nx.node_link_graph(json.loads(self.path.read_text()), directed=True, multigraph=True, edges="links")

    def save(self, g: nx.MultiDiGraph) -> None:
        atomic_write(self.path, json.dumps(nx.node_link_data(g, edges="links")))


class Neo4jGraphStore:  # pragma: no cover - seam only
    """Drop-in seam for Neo4j (Cypher MERGE of Entity-[REL]->Entity). Not built: NetworkX by decision."""

    def __init__(self, *_, **__):
        raise NotImplementedError("Neo4j adapter not built; set GRAPH_BACKEND=networkx")


# ---------------------------------------------------------------- knowledge base
class KnowledgeBase:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.children = ChromaVectorIndex("children")
        self.entity_names = ChromaVectorIndex("entity_names")
        self.graph_store = NetworkXGraphStore(settings.path("graph.json"))
        self._loaded = False
        self._clear()

    def _clear(self) -> None:
        self.parents: dict[str, dict] = {}
        self.documents: list[dict] = []
        self.bm25: dict[str, tuple] = {}
        self.graph = nx.MultiDiGraph()
        self.n2v: dict[str, np.ndarray] = {}
        self.alias_index: dict[str, str] = {}
        self.er_report: dict = {}
        self.ingest_stats: dict = {}

    def ensure(self) -> "KnowledgeBase":
        if not self._loaded:
            self.load()
        return self

    def load(self) -> None:
        with self.lock:
            self._clear()
            p = settings.path("parents.json")
            if p.exists():
                self.parents = json.loads(p.read_text())
            d = settings.path("documents.json")
            if d.exists():
                self.documents = json.loads(d.read_text())
            b = settings.path("bm25.pkl")
            if b.exists():
                self.bm25 = pickle.loads(b.read_bytes())
            self.graph = self.graph_store.load()
            n = settings.path("n2v.npz")
            if n.exists():
                z = np.load(n, allow_pickle=False)
                self.n2v = dict(zip(json.loads(str(z["ids"])), z["vecs"]))
            a = settings.path("alias_index.json")
            if a.exists():
                self.alias_index = json.loads(a.read_text())
            e = settings.path("er_report.json")
            if e.exists():
                self.er_report = json.loads(e.read_text())
            s = settings.path("ingest_stats.json")
            if s.exists():
                self.ingest_stats = json.loads(s.read_text())
            self._loaded = True

    @property
    def empty(self) -> bool:
        return not self.parents

    def area_ok(self, chunk_id: str, area: str | None) -> bool:
        if not area or area == "all":
            return True
        return self.parents.get(chunk_id, {}).get("practice_area") == area

    def reset(self) -> None:
        with self.lock:
            self.children.reset()
            self.entity_names.reset()
            for name in ("parents.json", "documents.json", "bm25.pkl", "graph.json", "n2v.npz", "alias_index.json",
                         "er_report.json", "ingest_stats.json", "docstore.json", "runs.jsonl"):
                p = settings.path(name)
                if p.exists():
                    p.unlink()
            for f in settings.uploads_dir.glob("*"):
                f.unlink()
            self._clear()
            self._loaded = True


KB = KnowledgeBase()
