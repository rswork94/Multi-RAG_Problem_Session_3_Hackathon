"""Node2Vec structural embeddings: p/q-biased random walks + gensim skip-gram Word2Vec (seeded, single worker).

Run on the entity-entity graph only (no chunk/document nodes), so hub entities do not blow up the walks.
"""
from __future__ import annotations

import random

import networkx as nx
import numpy as np


def biased_walks(g: nx.Graph, walk_len: int, walks_per_node: int, p: float, q: float, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    nodes = sorted(g.nodes)
    nbrs = {n: sorted(g.neighbors(n)) for n in nodes}
    walks = []
    for _ in range(walks_per_node):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < walk_len:
                cur = walk[-1]
                cand = nbrs[cur]
                if not cand:
                    break
                if len(walk) == 1:
                    ws = [g[cur][x].get("weight", 1.0) for x in cand]
                else:
                    prev = walk[-2]
                    prev_n = set(nbrs[prev])
                    ws = [g[cur][x].get("weight", 1.0) * (1 / p if x == prev else 1.0 if x in prev_n else 1 / q)
                          for x in cand]
                walk.append(rng.choices(cand, weights=ws, k=1)[0])
            walks.append([str(n) for n in walk])
    return walks


def node2vec(g: nx.Graph, dim: int = 64, walk_len: int = 20, walks_per_node: int = 10, p: float = 1.0,
             q: float = 0.5, seed: int = 42) -> dict[str, np.ndarray]:
    if g.number_of_nodes() == 0:
        return {}
    from gensim.models import Word2Vec

    walks = biased_walks(g, walk_len, walks_per_node, p, q, seed)
    model = Word2Vec(sentences=walks, vector_size=dim, window=5, min_count=0, sg=1, workers=1, seed=seed, epochs=3)
    out = {}
    for n in g.nodes:
        v = model.wv[str(n)] if str(n) in model.wv else np.zeros(dim, dtype=np.float32)
        nv = np.linalg.norm(v)
        out[n] = (v / nv if nv else v).astype(np.float32)
    return out
