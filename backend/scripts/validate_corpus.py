"""Validate corpus_spec/<area>.yaml files against the design rules in corpus_spec/SCHEMA.md.

Usage:  .venv/bin/python scripts/validate_corpus.py [area ...] [--chunks]
  --chunks    also checks that no anchor is split across a parent-chunk boundary (needs the app splitter + bge tokenizer)
  --rendered  also re-loads every generated file with the app loaders and checks every anchor survived
Exit code 1 if any ERROR is found. WARN lines are advisory.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_lib import AREAS, assembled_documents, available_areas, load_spec, unit_texts  # noqa: E402

DOC_TYPES = {"Act", "Rules", "Notification", "Circular", "MasterDirection", "Guideline", "SOP", "Agreement", "Schedule",
             "Table", "Memo", "Glossary"}
FORMATS = {"pdf", "docx", "json", "csv"}
ENTITY_TYPES = {"Act", "Section", "Rule", "Notification", "Circular", "MasterDirection", "Regulation", "Standard", "Form",
                "Authority", "Rate", "Threshold", "Deadline", "Penalty", "Control", "Obligation", "Clause", "Party",
                "Agreement", "Concept"}
REL_TYPES = {"AMENDS", "SUPERSEDES", "REFERS_TO", "PRESCRIBES_RATE", "PRESCRIBES_FORM", "HAS_DEADLINE", "IMPOSES_PENALTY",
             "MAPS_TO", "DEFINES", "ISSUED_BY", "APPLIES_TO", "CAPPED_BY", "TRIGGERS", "REQUIRES", "PART_OF"}
CATS = {"keyword": 4, "multihop": 4, "vocab_alias": 2, "vocab_paraphrase": 2}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _contains(hay: str, needle: str) -> bool:
    n = _norm(needle)
    if not n:
        return False
    # word-ish boundaries so "Rule 31A" does not match inside "Rule 31AB"
    return re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", _norm(hay)) is not None


def validate(areas: list[str], check_chunks: bool = False, check_rendered: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []
    all_units: dict[str, str] = {}          # "doc:unit" -> text, across all areas
    per_area: dict[str, dict] = {}
    doc_ids: Counter = Counter()

    for area in areas:
        spec = load_spec(area)
        if spec.get("area") != area:
            errors.append(f"[{area}] 'area' field is {spec.get('area')!r}")
        docs = assembled_documents(area, spec)
        per_area[area] = {"spec": spec, "docs": docs}
        fmt = Counter(d.get("format") for d in docs)
        if not 6 <= len(docs) <= 8:
            errors.append(f"[{area}] needs 6-8 documents, has {len(docs)}")
        if fmt["pdf"] < 3 or fmt["docx"] < 2 or (fmt["json"] + fmt["csv"]) < 1:
            errors.append(f"[{area}] format mix must be >=3 pdf, >=2 docx, >=1 json/csv; got {dict(fmt)}")
        injection = False
        for d in docs:
            doc_ids[d["id"]] += 1
            if not d["id"].startswith(area + "_"):
                errors.append(f"[{area}] doc id {d['id']} must start with '{area}_'")
            if d.get("doc_type") not in DOC_TYPES:
                errors.append(f"[{d['id']}] invalid doc_type {d.get('doc_type')}")
            if d.get("format") not in FORMATS:
                errors.append(f"[{d['id']}] invalid format {d.get('format')}")
            if d.get("format") in ("pdf", "docx") and not [c for c in d["clauses"] if not c.get("filler")]:
                errors.append(f"[{d['id']}] pdf/docx documents need clauses")
            if d.get("format") in ("json", "csv") and not d.get("rows"):
                errors.append(f"[{d['id']}] json/csv documents need rows")
            ids = [c["id"] for c in d["clauses"]]
            if len(ids) != len(set(ids)):
                errors.append(f"[{d['id']}] duplicate clause ids")
            for c in d["clauses"]:
                if "annotation" in c.get("heading", "").lower():
                    injection = True
                if not c.get("filler"):
                    wc = len(c["text"].split())
                    if wc < 40 or wc > 260:
                        warns.append(f"[{d['id']}:{c['id']}] clause has {wc} words (aim 60-220)")
            for uid, txt in unit_texts(d).items():
                all_units[f"{d['id']}:{uid}"] = txt
        if not injection:
            errors.append(f"[{area}] needs one prompt-injection clause with 'Annotation' in its heading")

    for did, n in doc_ids.items():
        if n > 1:
            errors.append(f"duplicate document id {did}")

    for area, info in per_area.items():
        spec = info["spec"]
        ents = {e["name"]: e for e in spec.get("entities") or []}
        for e in ents.values():
            if e.get("type") not in ENTITY_TYPES:
                errors.append(f"[{area}] entity {e['name']} has invalid type {e.get('type')}")
            for a in e.get("aliases") or []:
                if a.get("kind") not in ("generic", "planted"):
                    errors.append(f"[{area}] alias {a} of {e['name']} needs kind generic|planted")
        for r in spec.get("relations") or []:
            if len(r) != 4:
                errors.append(f"[{area}] relation must be [src, REL, dst, evidence]: {r}")
                continue
            s, rel, t, ev = r
            if rel not in REL_TYPES:
                errors.append(f"[{area}] relation type {rel} invalid")
            for n in (s, t):
                if n not in ents:
                    errors.append(f"[{area}] relation {r} references unknown entity {n!r}")
            if ev not in all_units:
                errors.append(f"[{area}] relation {r} evidence {ev} not found")

        qs = spec.get("questions") or []
        cats = Counter(q.get("category") for q in qs)
        for c, n in CATS.items():
            if cats[c] != n:
                errors.append(f"[{area}] needs {n} '{c}' questions, has {cats[c]}")
        if not any(q.get("category") == "multihop" and q.get("hops") == 3 for q in qs):
            errors.append(f"[{area}] needs at least one multihop question with hops: 3")
        qids = Counter(q["id"] for q in qs)
        for qid, n in qids.items():
            if n > 1:
                errors.append(f"[{area}] duplicate question id {qid}")

        for q in qs:
            qid = q["id"]
            ev_docs = set()
            for gf in q.get("gold_facts") or []:
                ev = gf.get("evidence")
                if ev not in all_units:
                    errors.append(f"[{qid}] evidence {ev} not found")
                    continue
                ev_docs.add(ev.split(":")[0])
                for a in gf.get("anchors") or []:
                    if not _contains(all_units[ev], a):
                        errors.append(f"[{qid}] anchor {a!r} not in its evidence {ev}")
                    others = [u for u, txt in all_units.items() if u != ev and _contains(txt, a)]
                    if others:
                        errors.append(f"[{qid}] anchor {a!r} is not unique; also in {others[:4]}")
            if not q.get("gold_facts"):
                errors.append(f"[{qid}] has no gold_facts")
            for ge in q.get("gold_entities") or []:
                if ge not in ents:
                    errors.append(f"[{qid}] gold_entity {ge!r} not declared in entities")
            if q.get("category") == "multihop":
                if len(ev_docs) < 2:
                    errors.append(f"[{qid}] multihop evidence must span >=2 documents, got {sorted(ev_docs)}")
                if not q.get("bridge_entities"):
                    errors.append(f"[{qid}] multihop needs bridge_entities")
                for b in q.get("bridge_entities") or []:
                    if b not in ents:
                        errors.append(f"[{qid}] bridge entity {b!r} not declared")
                        continue
                    names = [b] + [a["text"] for a in ents[b].get("aliases") or []]
                    hit = [n for n in names if _contains(q["question"], n)]
                    if hit:
                        errors.append(f"[{qid}] question mentions bridge entity/alias {hit}")
            if q.get("category") == "vocab_alias":
                planted = [(e["name"], a["text"]) for e in ents.values() for a in e.get("aliases") or []
                           if a.get("kind") == "planted" and _contains(q["question"], a["text"])]
                if not planted:
                    errors.append(f"[{qid}] vocab_alias question must use a planted alias")
                for name, alias in planted:
                    defined_in = {u.split(":")[0] for u, txt in all_units.items() if _contains(txt, alias)}
                    if not defined_in - ev_docs:
                        errors.append(f"[{qid}] planted alias {alias!r} must be defined in a different document "
                                      f"than the evidence ({sorted(ev_docs)})")
                    if _contains(q["question"], name):
                        errors.append(f"[{qid}] vocab_alias question also uses the canonical name {name!r}")

    if check_chunks:
        errors += _check_chunk_boundaries(per_area)
    if check_rendered:
        errors += _check_rendered(per_area)
    return errors, warns


def _check_rendered(per_area: dict) -> list[str]:
    """Round trip: load every generated file with the app's own loaders and confirm each gold anchor survives
    (catches font/glyph loss such as a missing ₹ in PDFs, or loader quirks)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.ingest.loaders import load_file  # noqa: E402
    from corpus_lib import OUT_DIR, doc_filename  # noqa: E402

    errs = []
    for area, info in per_area.items():
        loaded = {}
        for d in info["docs"]:
            p = OUT_DIR / area / doc_filename(d)
            if not p.exists():
                errs.append(f"[{d['id']}] not rendered; run build_corpus.py first")
                continue
            rec = load_file(p, area)
            loaded[d["id"]] = rec["text"] + "\n" + "\n".join(rec["rows"])
        for q in info["spec"].get("questions") or []:
            for gf in q.get("gold_facts") or []:
                did = gf["evidence"].split(":")[0]
                for a in gf.get("anchors") or []:
                    if did in loaded and not _contains(loaded[did], a):
                        errs.append(f"[{q['id']}] anchor {a!r} lost after rendering + loading {did}")
    return errs


def _check_chunk_boundaries(per_area: dict) -> list[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.ingest.splitter import split_document_text  # noqa: E402

    errs = []
    for area, info in per_area.items():
        doc_text = {}
        for d in info["docs"]:
            doc_text[d["id"]] = "\n\n".join(unit_texts(d).values())
        for q in info["spec"].get("questions") or []:
            for gf in q.get("gold_facts") or []:
                did = gf["evidence"].split(":")[0]
                chunks = split_document_text(doc_text.get(did, ""))
                for a in gf.get("anchors") or []:
                    if not any(_contains(c, a) for c in chunks):
                        errs.append(f"[{q['id']}] anchor {a!r} is split across a chunk boundary in {did}")
    return errs


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    areas = args or available_areas()
    unknown = [a for a in areas if a not in AREAS]
    if unknown:
        sys.exit(f"unknown areas {unknown}")
    errors, warns = validate(areas, check_chunks="--chunks" in sys.argv, check_rendered="--rendered" in sys.argv)
    for w in warns:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"\n{len(areas)} area(s) checked: {len(errors)} error(s), {len(warns)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
