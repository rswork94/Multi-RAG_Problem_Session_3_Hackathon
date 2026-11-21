"""Step 3/4 - Entity normalisation and resolution ("CBDT" = "Central Board of Direct Taxes" = "Revenue Board").

Union-find over (type, canonical name) keys, in four stages. ER F1 is computed after stages a, c and d:
  a) dictionary    - canonical forms + a seed dictionary of GENERIC abbreviations only (planted aliases held out)
  b) text aliases  - aliases the documents define ("hereinafter referred to as ...", LLM-extracted aliases)
  c) name vectors  - bge cosine between entity names (non-identifier types only)
  d) node2vec      - name similarity combined with structural similarity in the entity graph
Identifier types (Section 194J, Form 26QF, Clause 9.4(c), 7.5% ...) only ever merge on exact canonical form.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict

import networkx as nx
import numpy as np

from .node2vec import node2vec

SEED_ALIASES: list[list[str]] = [
    ["CBDT", "Central Board of Direct Taxes"],
    ["CBIC", "Central Board of Indirect Taxes and Customs"],
    ["RBI", "Reserve Bank of India", "Reserve Bank"],
    ["SEBI", "Securities and Exchange Board of India"],
    ["FIU-IND", "Financial Intelligence Unit-India", "Financial Intelligence Unit - India", "FIU-India"],
    ["PMLA", "Prevention of Money-laundering Act, 2002", "Prevention of Money Laundering Act, 2002"],
    ["FEMA", "Foreign Exchange Management Act, 1999"],
    ["FCRA", "Foreign Contribution (Regulation) Act, 2010"],
    ["GST", "Goods and Services Tax"],
    ["TDS", "tax deducted at source"],
    ["ISD", "Input Service Distributor"],
    ["MAC", "Material Adverse Change", "Material Adverse Effect"],
    ["SPA", "Share Purchase Agreement"],
    ["KYC", "Know Your Customer"],
    ["CDD", "Customer Due Diligence"],
    ["BRSR", "Business Responsibility and Sustainability Report"],
    ["IRP", "Invoice Registration Portal"],
    ["IRN", "Invoice Reference Number"],
    ["NCLT", "National Company Law Tribunal"],
    ["ROC", "Registrar of Companies"],
]
IDENTIFIER_TYPES = {"Section", "Rule", "Notification", "Circular", "MasterDirection", "Form", "Clause", "Control", "Rate",
                    "Threshold", "Deadline", "Penalty"}
FLEX_TYPES = {"Authority", "Party", "Concept", "Act", "Standard", "Agreement", "Obligation", "Regulation"}
NAME_COS = 0.90
COMBINED = 0.86
N2V_WEIGHT = 0.25


LEGAL_SUFFIX = re.compile(r",?\s+(?:private\s+limited|pvt\.?\s+ltd\.?|limited|ltd\.?|llp)$")


def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def norm(s: str) -> str:
    """Matching key: lower case, no quotes/leading 'the', last word singular, company suffix and standard edition
    dropped ('Meridian Small Finance Bank Limited' = 'Meridian Small Finance Bank', 'ISO 31000:2018' = 'ISO 31000',
    'politically exposed persons' = 'politically exposed person')."""
    s = re.sub(r"[\"“”'’]", "", s or "").strip().strip(".,;:").lower()
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    s = LEGAL_SUFFIX.sub("", s)
    s = re.sub(r"^(iso\s?\d{4,5}):\d{4}$", r"\1", s)
    parts = s.split(" ")
    parts[-1] = _singular(parts[-1])
    return " ".join(parts)


def canonical(name: str, etype: str) -> str:
    n = re.sub(r"\s+", " ", (name or "").strip().strip(".,;:"))
    n = re.sub(r"^(?i:the)\s+", "", n)
    if etype == "Section":
        m = re.search(r"(?i)(?:section|sec\.?|u/s)\s*(\d+[A-Z]{0,3}(?:\(\w+\))*)", n)
        if m:
            return f"Section {m.group(1)}"
    if etype == "Rule":
        m = re.search(r"(?i)rule\s*(\d+[A-Z]{0,3}(?:\(\w+\))*)", n)
        if m:
            return f"Rule {m.group(1)}"
    if etype in ("Notification", "Circular"):
        m = re.search(r"(?i)\bno\.?\s*(\d[\w/.()-]*)", n)
        if m:
            return f"{etype} No. {m.group(1).rstrip('.,')}"
    if etype == "Form":
        m = re.search(r"(?i)form\s*(?:no\.?\s*)?([\w-]+)", n)
        if m:
            return f"Form {m.group(1).upper()}"
    if etype == "Clause":
        n = re.sub(r"(?i)^para\.?\s", "Paragraph ", n)
    if etype == "Rate":
        m = re.search(r"(\d+(?:\.\d+)?)\s*(%|per\s*cent)", n, re.I)
        if m:
            return f"{float(m.group(1)):g}%"
    if etype in ("Threshold", "Penalty"):
        n = re.sub(r"(?i)\b(?:rs\.?|inr)\s?", "₹", n).replace("₹ ", "₹")
    return n[:100]


class _UF:
    def __init__(self) -> None:
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def snapshot(self) -> dict:
        return {k: self.find(k) for k in list(self.p)}


def ent_id(key: tuple[str, str]) -> str:
    return f"ent:{key[0]}:{key[1].lower()}"


def resolve(extractions: dict[str, dict], unit_chunks: dict[str, list[str]], unit_area: dict[str, str],
            embed_fn, gold_clusters: list[dict] | None = None, corpus_text: str = "") -> dict:
    """Returns {entities, relations, er: {variants, ...}, n2v}."""
    keys: dict[tuple, dict] = {}
    unit_keys: dict[str, list[tuple]] = defaultdict(list)
    name_to_keys: dict[str, set] = defaultdict(set)
    raw_rel: list[tuple[str, str, str, str]] = []
    for uid, ex in extractions.items():
        local: dict[str, tuple] = {}
        for e in ex.get("entities", []):
            et = e.get("type") or "Concept"
            c = canonical(e.get("name", ""), et)
            if not c:
                continue
            k = (et, c.lower())
            rec = keys.setdefault(k, {"type": et, "names": Counter(), "aliases": set(), "units": set(),
                                      "chunks": set(), "areas": set(), "mentions": 0})
            rec["names"][c] += 1
            rec["aliases"].update(a for a in (e.get("aliases") or []) if a)
            rec["units"].add(uid)
            rec["chunks"].update(unit_chunks.get(uid, []))
            rec["areas"].add(unit_area.get(uid, ""))
            rec["mentions"] += 1
            unit_keys[uid].append(k)
            for n in (e.get("name", ""), c):
                local[norm(n)] = k
                name_to_keys[norm(n)].add(k)
        for r in ex.get("relations", []):
            raw_rel.append((uid, r.get("src", ""), r.get("rel", "REFERS_TO"), r.get("dst", "")))
            for side in ("src", "dst"):
                nk = norm(r.get(side, ""))
                if nk and nk not in local and nk in name_to_keys:
                    local[nk] = next(iter(name_to_keys[nk]))
        extractions[uid]["_local"] = local

    uf = _UF()
    for k in keys:
        uf.find(k)
    # (a) dictionary: same normalised canonical name across flexible types, plus generic seed abbreviations
    by_norm: dict[str, list] = defaultdict(list)
    for k in keys:
        by_norm[norm(k[1])].append(k)
    for ks in by_norm.values():
        flex = [k for k in ks if k[0] in FLEX_TYPES]
        for a, b in zip(flex, flex[1:]):
            uf.union(a, b)
    for group in SEED_ALIASES:
        members = [k for g in group for k in by_norm.get(norm(g), []) if k[0] in FLEX_TYPES]
        for a, b in zip(members, members[1:]):
            uf.union(a, b)
    snap_a = uf.snapshot()
    # (b) aliases the documents define
    for k, rec in keys.items():
        for a in rec["aliases"]:
            for k2 in by_norm.get(norm(a), []):
                if k2 != k:  # the text explicitly defines this alias: merge whatever type the other mention has
                    uf.union(k, k2)
    # (c) name embeddings among flexible types
    # names with digits are identifiers ("TS-07" vs "TS-08"): never merged by vector similarity
    flex_keys = [k for k in keys if k[0] in FLEX_TYPES and not re.search(r"\d", k[1])]
    labels = [keys[k]["names"].most_common(1)[0][0] for k in flex_keys]
    vecs = np.asarray(embed_fn(labels), dtype=np.float32) if labels else np.zeros((0, 1))
    sims = vecs @ vecs.T if len(labels) else np.zeros((0, 0))
    for i, j in itertools.combinations(range(len(flex_keys)), 2):
        if sims[i, j] >= NAME_COS and _type_compatible(flex_keys[i][0], flex_keys[j][0]):
            uf.union(flex_keys[i], flex_keys[j])
    snap_c = uf.snapshot()
    # (d) node2vec on the provisional entity graph, combined with name similarity
    prov = _entity_graph(uf, unit_keys, raw_rel, extractions)
    n2v = node2vec(prov)
    for i, j in itertools.combinations(range(len(flex_keys)), 2):
        if sims[i, j] < 0.80 or not _type_compatible(flex_keys[i][0], flex_keys[j][0]):
            continue
        ri, rj = uf.find(flex_keys[i]), uf.find(flex_keys[j])
        if ri == rj or ri not in n2v or rj not in n2v:
            continue
        s = (1 - N2V_WEIGHT) * sims[i, j] + N2V_WEIGHT * float(n2v[ri] @ n2v[rj])
        if s >= COMBINED:
            uf.union(ri, rj)
    snap_d = uf.snapshot()

    entities, key_to_id = _build_entities(keys, uf)
    relations = _build_relations(raw_rel, extractions, uf, key_to_id, unit_chunks)
    final_graph = nx.Graph()
    final_graph.add_nodes_from(e["id"] for e in entities)
    for r in relations:
        w = final_graph[r["src"]][r["dst"]]["weight"] + 2 if final_graph.has_edge(r["src"], r["dst"]) else 2
        final_graph.add_edge(r["src"], r["dst"], weight=w)
    for uid, ks in unit_keys.items():
        ids = sorted({key_to_id[uf.find(k)] for k in ks})
        for a, b in itertools.combinations(ids[:25], 2):
            w = final_graph[a][b]["weight"] + 1 if final_graph.has_edge(a, b) else 1
            final_graph.add_edge(a, b, weight=w)
    n2v_final = node2vec(final_graph)

    er = {"variants": {}}
    if gold_clusters:
        # Score only gold alias members that are actually mentioned in the indexed documents: a name that never
        # occurs cannot be linked across documents by any system. The strict all-members score is kept too.
        flat = re.sub(r"\s+", " ", corpus_text.lower())

        def occurs(m: str) -> bool:
            return re.search(r"(?<![\w])" + re.escape(re.sub(r"\s+", " ", m.lower())) + r"(?![\w])", flat) is not None
        mentioned = [{**c, "members": [m for m in c["members"] if occurs(m)]} for c in gold_clusters] \
            if corpus_text else gold_clusters
        for label, snap in (("dictionary", snap_a), ("+ aliases & name embeddings", snap_c), ("+ node2vec", snap_d)):
            er["variants"][label] = er_f1(mentioned, _string_index(keys, snap))
        er["final"] = er["variants"]["+ node2vec"]
        er["strict_all_members"] = er_f1(gold_clusters, _string_index(keys, snap_d))
        n_all = sum(len(c["members"]) for c in gold_clusters)
        n_men = sum(len(c["members"]) for c in mentioned)
        er["coverage"] = {"gold_members": n_all, "mentioned_in_corpus": n_men}
    return {"entities": entities, "relations": relations, "er": er, "n2v": n2v_final}


def _type_compatible(a: str, b: str) -> bool:
    return a == b or "Concept" in (a, b)


def _entity_graph(uf: _UF, unit_keys, raw_rel, extractions) -> nx.Graph:
    g = nx.Graph()
    for uid, ks in unit_keys.items():
        roots = sorted({uf.find(k) for k in ks})
        g.add_nodes_from(roots)
        for a, b in itertools.combinations(roots[:25], 2):
            g.add_edge(a, b, weight=g[a][b]["weight"] + 1 if g.has_edge(a, b) else 1)
    for uid, s, _, t in raw_rel:
        local = extractions[uid].get("_local", {})
        ks, kt = local.get(norm(s)), local.get(norm(t))
        if ks and kt:
            a, b = uf.find(ks), uf.find(kt)
            if a != b:
                g.add_edge(a, b, weight=g[a][b]["weight"] + 2 if g.has_edge(a, b) else 2)
    return g


def _build_entities(keys: dict, uf: _UF) -> tuple[list[dict], dict]:
    clusters: dict = defaultdict(list)
    for k in keys:
        clusters[uf.find(k)].append(k)
    seed_abbr = {norm(g[0]) for g in SEED_ALIASES}
    entities, key_to_id = [], {}
    for root, members in clusters.items():
        rep = max(members, key=lambda k: (norm(k[1]) in seed_abbr, keys[k]["mentions"], -len(k[1])))
        label = keys[rep]["names"].most_common(1)[0][0]
        names = set()
        for k in members:
            names.update(keys[k]["names"])
            names.update(keys[k]["aliases"])
        eid = ent_id(rep)
        key_to_id[root] = eid
        entities.append({
            "id": eid, "label": label, "type": rep[0],
            "aliases": sorted({n for n in names if norm(n) != norm(label)}),
            "chunks": sorted(set().union(*[keys[k]["chunks"] for k in members])),
            "areas": sorted(a for a in set().union(*[keys[k]["areas"] for k in members]) if a),
            "mentions": sum(keys[k]["mentions"] for k in members),
        })
    return entities, key_to_id


def _build_relations(raw_rel, extractions, uf: _UF, key_to_id: dict, unit_chunks: dict) -> list[dict]:
    agg: dict[tuple, dict] = {}
    for uid, s, rel, t in raw_rel:
        local = extractions[uid].get("_local", {})
        ks, kt = local.get(norm(s)), local.get(norm(t))
        if not ks or not kt:
            continue
        a, b = key_to_id[uf.find(ks)], key_to_id[uf.find(kt)]
        if a == b:
            continue
        r = agg.setdefault((a, rel, b), {"src": a, "rel": rel, "dst": b, "weight": 0, "evidence": set()})
        r["weight"] += 1
        r["evidence"].update(unit_chunks.get(uid, []))
    out = []
    for r in agg.values():
        r["evidence"] = sorted(r["evidence"])
        out.append(r)
    return out


def _string_index(keys: dict, snap: dict) -> dict[str, object]:
    idx: dict[str, object] = {}
    for k, rec in keys.items():
        root = snap.get(k, k)
        names = list(rec["names"]) + list(rec["aliases"]) + [k[1]]
        if k[0] == "Form":
            names += [re.sub(r"(?i)^form\s+", "", n) for n in names]
        for n in names:
            idx.setdefault(norm(n), root)
    # seed dictionary members that never appear in text still resolve through the dictionary
    for group in SEED_ALIASES:
        roots = [idx[norm(g)] for g in group if norm(g) in idx]
        if roots:
            for g in group:
                idx.setdefault(norm(g), roots[0])
    return idx


def er_f1(gold_clusters: list[dict], index: dict[str, object]) -> dict:
    """Pairwise precision/recall/F1 of predicted co-reference over gold alias-cluster members (per area)."""
    tp = fp = fn = 0
    by_area: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for ci, c in enumerate(gold_clusters):
        for m in c["members"]:
            by_area[c.get("area", "")].append((norm(m), ci))
    resolved = total = 0
    for members in by_area.values():
        for (m1, c1), (m2, c2) in itertools.combinations(members, 2):
            p1, p2 = index.get(m1, ("_unresolved", m1)), index.get(m2, ("_unresolved", m2))
            same_pred, same_gold = p1 == p2, c1 == c2
            tp += same_pred and same_gold
            fp += same_pred and not same_gold
            fn += same_gold and not same_pred
        for m, _ in members:
            total += 1
            resolved += m in index
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "pairs_tp": tp, "pairs_fp": fp,
            "pairs_fn": fn, "members_found": resolved, "members_total": total}
