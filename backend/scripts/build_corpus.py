"""Render corpus_spec/<area>.yaml into documents + gold labels.

Usage:  .venv/bin/python scripts/build_corpus.py [area ...]
Writes:
  corpus/generated/<area>/<doc_id>.<pdf|docx|json|csv>
  corpus/gold/questions.yaml   (all areas; evidence also resolved to file names)
  corpus/gold/aliases.yaml     (gold alias clusters for ER accuracy)
  corpus/gold/relations.yaml   (gold fact-graph edges)
  corpus/gold/manifest.json    (sha256 of every file + spec -> the corpus version recorded with benchmark runs)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_lib import (GOLD_DIR, OUT_DIR, RENDERERS, SPEC_DIR, assembled_documents, available_areas,  # noqa: E402
                        doc_filename, load_spec)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(areas: list[str]) -> dict:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    questions, clusters, relations = [], [], []
    manifest: dict = {"files": {}, "specs": {}}
    for area in areas:
        spec = load_spec(area)
        out = OUT_DIR / area
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        docs = assembled_documents(area, spec)
        fname = {}
        for d in docs:
            p = out / doc_filename(d)
            RENDERERS[d["format"]](d, p)
            fname[d["id"]] = p.name
            manifest["files"][f"{area}/{p.name}"] = _sha(p)
        manifest["specs"][area] = _sha(SPEC_DIR / f"{area}.yaml")
        for q in spec.get("questions") or []:
            q = dict(q, area=area)
            q["gold_facts"] = [dict(gf, source=fname.get(gf["evidence"].split(":")[0])) for gf in q["gold_facts"]]
            questions.append(q)
        for e in spec.get("entities") or []:
            if e.get("aliases"):
                clusters.append({"area": area, "canonical": e["name"], "type": e["type"],
                                 "members": [e["name"]] + [a["text"] for a in e["aliases"]],
                                 "planted": [a["text"] for a in e["aliases"] if a.get("kind") == "planted"]})
        for r in spec.get("relations") or []:
            relations.append({"area": area, "src": r[0], "rel": r[1], "dst": r[2], "evidence": r[3]})
        print(f"[{area}] {len(docs)} documents -> {out}")
    manifest["version"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    (GOLD_DIR / "questions.yaml").write_text(yaml.safe_dump({"questions": questions}, sort_keys=False, allow_unicode=True))
    (GOLD_DIR / "aliases.yaml").write_text(yaml.safe_dump({"clusters": clusters}, sort_keys=False, allow_unicode=True))
    (GOLD_DIR / "relations.yaml").write_text(yaml.safe_dump({"relations": relations}, sort_keys=False, allow_unicode=True))
    (GOLD_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(questions)} questions, {len(clusters)} alias clusters, corpus version {manifest['version']}")
    return manifest


if __name__ == "__main__":
    build([a for a in sys.argv[1:]] or available_areas())
