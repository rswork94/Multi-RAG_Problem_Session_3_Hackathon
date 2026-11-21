"""Step 2b - LLM-guided entity-relation extraction per clause unit (claude-haiku-5-5), with a regex extractor
that always runs (anchors identifiers the LLM may miss) and is the whole extractor in offline mode.

Extraction is cached by clause-text hash, so changing chunk size never re-runs the LLM.
"""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

from ..config import settings
from ..llm import calls, client
from ..prompts.tasks import EXTRACTION_SCHEMA, EXTRACTION_SYSTEM

BATCH = 5
PARALLEL = 4

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
RX: dict[str, re.Pattern] = {
    "Section": re.compile(r"\b(?:Section|Sec\.|u/s)\s*\d+[A-Z]{0,3}(?:\(\w{1,4}\))*"),
    "Rule": re.compile(r"\bRule\s+\d+[A-Z]{0,3}(?:\(\w{1,4}\))*"),
    "Notification": re.compile(r"\bNotification\s+No\.?\s*[\w./()-]*\d[\w/.()-]*"),
    "Circular": re.compile(r"\bCircular\s+(?:No\.?\s*)?(?=[\w/.()-]*\d)[\w/.()-]+"),
    "MasterDirection": re.compile(r"\b[A-Z]{2,6}(?:\.[A-Z]{2,6}){1,4}\.?\s?(?:No\.\s?)?\d+(?:/[\d.]+)*/\d{4}-\d{2,4}\b"),
    "Form": re.compile(r"\bForm\s+(?:No\.?\s*)?(?:[A-Z]{2,}(?:-[A-Z0-9]+)+|(?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{1,10})\b"),
    "Clause": re.compile(r"\b(?:Clause|Paragraph|Para|Item|Schedule|Annexure|Article)\s+\d+[A-Z]?(?:\.\d+)*(?:\([a-z0-9ivx]{1,4}\))*"),
    "Control": re.compile(r"\b[A-Z]{1,3}-[A-Z]{2,6}-\d{2,3}\b"),
    "Rate": re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|per\s*cent)"),
    "Threshold": re.compile(r"(?:₹|Rs\.?|INR)\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:crore|lakh|million))?"),
    "Deadline": re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})\s*,?\s+\d{{4}}\b|\b\d{{1,2}}(?:st|nd|rd|th)\s+day\s+"
                           r"(?:of|following|after|from)\s+[^.,;]{3,50}|\bevery\s+\d+\s+(?:months|years)\b"),
    "Authority": re.compile(r"\b(?:CBDT|CBIC|GST Council|Reserve Bank of India|Reserve Bank|RBI|SEBI|FIU-IND|"
                            r"Securities and Exchange Board of India|Financial Intelligence Unit-India|"
                            r"Central Board of Direct Taxes|Central Board of Indirect Taxes and Customs|"
                            r"Registrar of Companies|Assessing Officer|National Company Law Tribunal|GSTN)\b"),
    "Act": re.compile(r"\b[A-Z][\w()'-]*(?:\s+(?:(?:and|of|for|the|on|to|in|&)\s+){0,2}[A-Z(][\w()'-]*){0,8}\s+Act,?\s*\d{4}"),
    # named instruments: "Income-tax (Seventeenth Amendment) Rules, 2026", "... (Amendment) Directions, 2026"
    "Regulation": re.compile(r"\b[A-Z][\w()'-]*(?:\s+(?:(?:and|of|for|the|on|to|in|&|-|–)\s+){0,2}[A-Z(][\w()'-]*){0,9}"
                             r"\s+(?:Rules|Regulations|Directions?|Guidelines)(?:,\s*|\s+)\d{4}"),
    "Standard": re.compile(r"\bISO\s?\d{4,5}(?::\d{4})?\b|\bBRSR(?: Core)?\b|\bInd AS\s?\d+\b"),
    "Agreement": re.compile(r"\b(?:Share Purchase Agreement|Amendment Agreement(?: No\.\s*\d+)?|Disclosure Letter|"
                            r"Escrow Agreement|Shareholders'? Agreement|Term Sheet)\b"),
    "Party": re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}(?:\s+\((?:Mauritius|Singapore|India)\))?\s+"
                        r"(?:Private Limited|Pvt\.? Ltd\.?|Limited|LLP|Ltd\.?)"),
    "Penalty": re.compile(r"(?i)(?:monetary\s+)?penalty\s+(?:of|not exceeding|up to|which may extend to)\s+"
                          r"(?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d+)?(?:\s*(?:crore|lakh))?(?:\s+(?:per|for each)\s+\w+)?"),
}
ALIAS_DEF = re.compile(
    r"\((?:hereinafter|herein|hereafter|also)?\s*(?:(?:referred to|called|known)\s+(?:internally\s+)?"
    r"(?:in this [\w -]{2,25}\s+)?as\s+)?(?:the\s+)?[\"“](?P<alias>[^\"”]{2,60})[\"”]"
    r"(?:\s+or\s+(?:the\s+)?[\"“](?P<alias2>[^\"”]{2,60})[\"”])?\)", re.I)
# 'X is referred to as the "Alias"', 'X, the transferee, is hereinafter referred to as the "Alias"'
ALIAS_INLINE = re.compile(
    r"(?:\b(?:is|are|shall be|will be)\s+)?(?:also\s+|hereinafter\s+|internally\s+)?(?:referred to|called|known)"
    r"(?:\s+internally)?(?:\s+in this [\w -]{2,25})?\s+as\s+(?:the\s+)?[\"“](?P<alias>[^\"”]{2,60})[\"”]"
    r"(?:\s+or\s+(?:the\s+)?[\"“](?P<alias2>[^\"”]{2,60})[\"”])?", re.I)
# 'Know Your Customer (KYC)', 'Input Service Distributor ("ISD")'
ABBR_DEF = re.compile(r"(?P<long>[A-Z][\w'-]*(?:\s+(?:(?:of|and|for|the|on|to|in|&)\s+){0,2}[A-Z][\w'-]*){1,8})\s*"
                      r"\((?:the\s+)?[\"“]?(?P<abbr>[A-Z][A-Z0-9&-]{1,9})[\"”]?\)")
CODE = re.compile(r"(?<![\w-])[A-Z]{1,4}-\d{1,3}[A-Z]?\b")
# '(the KYC Direction)': unquoted short-form definition right after a name
UNQUOTED_DEF = re.compile(r"\(the\s+(?P<alias>[A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,4})\)")
# 'C-KYC-07 (Re-KYC Due-Date Sentinel)': an identifier followed by its Title Case name in brackets
CODE_NAME = re.compile(r"(?P<code>\b[A-Z]{1,3}-[A-Z]{2,6}-\d{2,3}\b|(?<![\w-])[A-Z]{1,4}-\d{1,3}[A-Z]?\b)\s*"
                       r"\((?P<name>[A-Z][\w-]*(?:\s+[A-Z&][\w-]*){0,6})\)")
# table rows: 'control_id: C-KYC-02; control_name: Risk-Grade Review Scheduler'
ROW_ID_NAME = re.compile(r"(?P<f>\w+)_id:\s*(?P<code>[^;]+?);\s*(?P=f)_name:\s*(?P<name>[^;(]+)")
# legal drafting: 'These rules may be called the Income-tax (Seventeenth Amendment) Rules, 2026'
MAY_BE_CALLED = re.compile(r"\b(?:These|This)\s+(?:rules|regulations|directions|circular|notification|agreement|policy|"
                           r"SOP|guidelines|deed|letter|memorandum)\s+may\s+be\s+(?:called|cited\s+as)\s+(?:the\s+)?"
                           r"(?P<name>[^.;]{5,140}?\d{4})", re.I)
# "service code TS-07", "Enforcement Head E-16", "Customer Risk Grade CRG-4": a descriptor phrase naming a code
DESCRIBED_CODE = re.compile(r"\b(?P<desc>(?:[A-Za-z]+\s+){0,2}(?:code|head|item|category|grade|flag|cohort|tier|level|"
                            r"control|annex|annexure|schedule|note|finding))\s+(?P<code>[A-Z]{1,4}-\d{1,3}[A-Z]?)\b", re.I)
FUNCTION_WORDS = {"under", "of", "the", "in", "for", "a", "an", "to", "by", "with", "and", "or", "per", "each", "any",
                  "as", "at", "from", "on", "that", "this", "such", "every", "its", "their", "is", "are", "was"}
CAP_PHRASE = re.compile(r"((?:[A-Z][\w&'()-]*)(?:\s+(?:(?:of|and|for|the|on|to|in|&)\s+){0,2}[A-Z][\w&'()-]*){0,9})[\s,]*$")


def _known_acronyms() -> re.Pattern:
    from .resolve import SEED_ALIASES
    abbrs = sorted({g[0] for g in SEED_ALIASES}, key=len, reverse=True)
    return re.compile(r"(?<![\w-])(?:" + "|".join(map(re.escape, abbrs)) + r")(?![\w-])")


ACRONYM = _known_acronyms()

# Lookbehinds include the period (they are evaluated after it): do not split after "No.", "Sec.", "Rs.", "Pvt.",
# "Ltd.", "viz.", "Co." or after an identifier such as "194J." / "31AB.".
SENT = re.compile(r"(?<!\bNo\.)(?<!\bSec\.)(?<!\bRs\.)(?<!\bPvt\.)(?<!\bLtd\.)(?<!\bviz\.)(?<!\bCo\.)"
                  r"(?<!\d[A-Z]\.)(?<!\d[A-Z]{2}\.)(?<!\d[A-Z]{3}\.)(?<=[.;])\s+(?=[A-Z(])")
VALUE_TYPES = {"Rate", "Threshold", "Deadline", "Penalty"}
PROVISION_TYPES = {"Section", "Rule", "Clause", "Control", "Form", "Notification", "Circular", "MasterDirection",
                   "Agreement", "Act", "Standard"}


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip(".,;:"))


def _heading_subjects(heading: str) -> list[tuple[int, str, str]]:
    out = []
    for etype in ("Section", "Rule", "Clause", "Control", "Notification", "Circular", "MasterDirection", "Form"):
        for m in RX[etype].finditer(heading or ""):
            out.append((-1, etype, _norm_ws(m.group(0))))
    return out[:1]


def regex_extract(text: str, heading: str = "") -> dict:
    """Regex entities + heuristic relations. A clause heading's provision (e.g. "Rule 31AB") is the subject of
    body sentences that name no provision of their own."""
    ents: dict[tuple, dict] = {}
    rels: list[dict] = []
    subject = _heading_subjects(heading)
    for _, et, nm in subject:
        ents.setdefault((et, nm.lower()), {"name": nm, "type": et, "aliases": []})
    for sent in SENT.split(text):
        found: list[tuple[int, str, str]] = []
        for etype, pat in RX.items():
            for m in pat.finditer(sent):
                name = _norm_ws(m.group(0))
                if etype == "Party" and len(name.split()) < 2:
                    continue
                ents.setdefault((etype, name.lower()), {"name": name, "type": etype, "aliases": []})
                found.append((m.start(), etype, name))
        for m in ACRONYM.finditer(sent):
            nm = m.group(0)
            if not any(f[2] == nm for f in found):
                ents.setdefault(("Concept", nm.lower()), {"name": nm, "type": "Concept", "aliases": []})
                found.append((m.start(), "Concept", nm))
        for m in CODE.finditer(sent):
            nm = m.group(0)
            if not any(nm in f[2] for f in found):
                ents.setdefault(("Concept", nm.lower()), {"name": nm, "type": "Concept", "aliases": []})
                found.append((m.start(), "Concept", nm))
        for m in DESCRIBED_CODE.finditer(sent):
            words = m.group("desc").split()
            while words and words[0].lower() in FUNCTION_WORDS:
                words = words[1:]
            code, phrase = m.group("code"), _norm_ws(" ".join(words + [m.group("code")]))
            rec = next((e for (t, k), e in ents.items() if e["name"] == code), None)
            if rec is not None and phrase.lower() != code.lower() and phrase not in rec["aliases"]:
                rec["aliases"].append(phrase)
        for m in ABBR_DEF.finditer(sent):
            long, abbr = _norm_ws(re.sub(r"^(?:The|This|That|Such|Each|Every|Any)\s+", "", m.group("long"))), m.group("abbr")
            if len(long.split()) >= 2 and "".join(w[0] for w in long.split() if w[0].isupper()).startswith(abbr[0]):
                rec = ents.setdefault(("Concept", long.lower()), {"name": long, "type": "Concept", "aliases": []})
                if abbr not in rec["aliases"]:
                    rec["aliases"].append(abbr)
                found.append((m.start(), "Concept", long))
        for m in CODE_NAME.finditer(sent):
            code, name = m.group("code"), _norm_ws(m.group("name"))
            rec = next((e for e in ents.values() if e["name"] == code), None)
            if rec is not None and name not in rec["aliases"] and len(name.split()) >= 2:
                rec["aliases"].append(name)
        for m in ROW_ID_NAME.finditer(sent):
            code, name = _norm_ws(m.group("code")), _norm_ws(m.group("name"))
            rec = next((e for e in ents.values() if e["name"] == code), None)
            if rec is None:
                rec = ents.setdefault(("Concept", code.lower()), {"name": code, "type": "Concept", "aliases": []})
            if name not in rec["aliases"]:
                rec["aliases"].append(name)
        for m in UNQUOTED_DEF.finditer(sent):
            before = [f for f in found if f[0] + len(f[2]) <= m.start() and m.start() - (f[0] + len(f[2])) <= 3]
            if before:
                _, et, nm = max(before, key=lambda f: f[0])
                al = ents[(et, nm.lower())]["aliases"]
                if m.group("alias") not in al:
                    al.append(m.group("alias"))
        # explicit alias definitions attach to the nearest entity that ends before the definition
        for rx in (ALIAS_DEF, ALIAS_INLINE):
            for m in rx.finditer(sent):
                before = [f for f in found if f[0] < m.start() and m.start() - (f[0] + len(f[2])) <= 90
                          and f[1] not in VALUE_TYPES]
                best = max(before, key=lambda f: f[0] + len(f[2])) if before else None
                # contracts: 'among X, a partnership registered under the ... Act, 2008 (the "Seller"); Y, ...' -
                # the defined term belongs to the party that opens this ';'-separated segment, not the nearest name
                seg_start = sent.rfind(";", 0, m.start()) + 1
                parties = [f for f in found if seg_start <= f[0] < m.start() and f[1] == "Party"]
                if parties:
                    best = min(parties, key=lambda f: f[0])
                cap = CAP_PHRASE.search(sent[:m.start()].rstrip())
                cap_ok = cap is not None and len(cap.group(1)) >= 3 and not parties
                if cap_ok and (best is None or cap.start(1) >= best[0] + len(best[2])):
                    nm, et = _norm_ws(re.sub(r"^the\s+", "", cap.group(1))), "Concept"
                    ents.setdefault((et, nm.lower()), {"name": nm, "type": et, "aliases": []})
                elif best:
                    _, et, nm = best
                else:
                    continue
                for a in (m.groupdict().get("alias"), m.groupdict().get("alias2")):
                    al = ents[(et, nm.lower())]["aliases"]
                    if a and a.lower() != nm.lower() and _norm_ws(a) not in al:
                        al.append(_norm_ws(a))
        found.sort()
        lower = sent.lower()
        provs = [f for f in found if f[1] in PROVISION_TYPES]
        if subject and not any(f[1] in ("Section", "Rule", "Clause", "Control") for f in provs):
            provs = subject + provs
        for spos, st, sn in provs:
            for tpos, tt, tn in found:
                if tn == sn:
                    continue
                rel = None
                if st in ("Notification", "Circular", "MasterDirection", "Agreement") and tt in PROVISION_TYPES:
                    rel = "SUPERSEDES" if re.search(r"supersed|in substitution|replac", lower) else \
                        "AMENDS" if "amend" in lower else "REFERS_TO"
                elif tt == "Rate":
                    rel = "PRESCRIBES_RATE"
                elif tt == "Form" and st != "Form":
                    rel = "PRESCRIBES_FORM"
                elif tt == "Deadline":
                    rel = "HAS_DEADLINE"
                elif tt == "Penalty":
                    rel = "IMPOSES_PENALTY"
                elif tt == "Threshold":
                    rel = "CAPPED_BY" if re.search(r"\bcap|aggregate liability|shall not exceed", lower) else "APPLIES_TO"
                elif st == "Control" and tt in ("Clause", "Section", "Rule", "MasterDirection"):
                    rel = "MAPS_TO"
                elif tt == "Authority" and st in ("Notification", "Circular", "MasterDirection"):
                    rel = "ISSUED_BY"
                elif tt in PROVISION_TYPES and tpos > spos:
                    rel = "REFERS_TO"
                elif tt in ("Concept", "Party", "Obligation"):
                    rel = "APPLIES_TO" if re.search(r"appl(y|ies|icable)", lower) else "REFERS_TO"
                if rel:
                    rels.append({"src": sn, "rel": rel, "dst": tn})
        if re.search(r"material adverse", lower) and re.search(r"terminat", lower):
            macs = [f for f in found if f[1] in ("Clause", "Agreement")]
            for _, _, n in macs[:1]:
                ents.setdefault(("Concept", "material adverse change"), {"name": "Material Adverse Change",
                                                                       "type": "Concept", "aliases": ["MAC"]})
                rels.append({"src": "Material Adverse Change", "rel": "TRIGGERS", "dst": n})
    uniq = {(r["src"], r["rel"], r["dst"]): r for r in rels}
    return {"entities": list(ents.values()), "relations": list(uniq.values())}


# ---------------------------------------------------------------- LLM extraction with cache
def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _cache_path():
    return settings.path("extraction_cache.json")


def load_cache() -> dict:
    # A cache shipped with the corpus lets the offline demo use the LLM-built graph.
    p = _cache_path()
    if p.exists():
        return json.loads(p.read_text())
    shipped = settings.corpus_dir / "extraction_cache.json"
    return json.loads(shipped.read_text()) if shipped.exists() else {}


def save_cache(cache: dict) -> None:
    _cache_path().write_text(json.dumps(cache))


def estimate_cost(units: list[dict], cache: dict) -> dict:
    todo = [u for u in units if _hash(u["text"]) not in cache]
    chars = sum(len(u["text"]) for u in todo)
    in_tok = chars / 3.8 + len(todo) / BATCH * 900
    out_tok = len(todo) * 160
    p_in, p_out, _ = client.price(client.current().utility_model)
    return {"units_to_extract": len(todo), "cached": len(units) - len(todo),
            "est_cost_usd": round(in_tok * p_in / 1e6 + out_tok * p_out / 1e6, 4)}


def _merge(llm: dict, rx: dict) -> dict:
    """Union of LLM entities and regex anchors (regex guarantees identifiers like Form 26QF exist)."""
    names = {(e["type"], e["name"].lower()) for e in llm["entities"]}
    ents = list(llm["entities"]) + [e for e in rx["entities"] if (e["type"], e["name"].lower()) not in names]
    by_name = {e["name"].lower(): e for e in ents}
    for e in rx["entities"]:  # carry regex-found alias definitions onto LLM entities
        tgt = by_name.get(e["name"].lower())
        if tgt is not None and e.get("aliases"):
            tgt["aliases"] = sorted(set((tgt.get("aliases") or []) + e["aliases"]))
    rels = {(r["src"], r["rel"], r["dst"]): r for r in rx["relations"] + llm["relations"]}
    return {"entities": ents, "relations": list(rels.values())}


def instrument_titles(units: list[dict], title: str) -> list[dict]:
    """Entities to add to a document's units: the instrument named in the document title (e.g. 'Notification No.
    41/2026'), carrying as alias the name the text gives it ('These rules may be called ...')."""
    primary = None
    for etype in ("Notification", "Circular", "MasterDirection", "Agreement"):
        m = RX[etype].search(title or "")
        if m:
            primary = (etype, _norm_ws(m.group(0)))
            break
    if not primary:
        return []
    names = sorted({_norm_ws(m.group("name")) for u in units for m in MAY_BE_CALLED.finditer(u["text"])})
    return [{"name": primary[1], "type": primary[0], "aliases": names}] if names else []


def extract_units(units: list[dict], use_llm: bool, progress=None) -> tuple[dict[str, dict], dict, list[str]]:
    """Returns ({unit_id: {entities, relations, method}}, usage, warnings)."""
    out: dict[str, dict] = {}
    usage: dict = {}
    warnings: list[str] = []
    cache = load_cache()
    todo = []
    for u in units:
        rx = regex_extract(u["text"], u.get("heading", ""))
        h = _hash(u["text"])
        if h in cache:
            out[u["unit_id"]] = {**_merge(cache[h], rx), "method": "llm (cached)"}
        elif use_llm and client.available():
            todo.append(u)
            out[u["unit_id"]] = {**rx, "method": "regex"}  # replaced below on success
        else:
            out[u["unit_id"]] = {**rx, "method": "regex"}
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]

    def work(batch):
        body = "\n\n".join(f'<unit id="{u["unit_id"]}">\n{u["text"]}\n</unit>' for u in batch)
        try:
            data, us = calls.structured(EXTRACTION_SYSTEM, f"Extract from these units.\n\n{body}", EXTRACTION_SCHEMA,
                                        max_tokens=8000)
            return batch, data, us, None
        except Exception as e:  # noqa: BLE001 - fall back to regex for this batch
            return batch, None, {}, str(e)

    done = 0
    with ThreadPoolExecutor(PARALLEL) as ex:
        for batch, data, us, err in ex.map(work, batches):
            usage = client.merge_usage(usage, us)
            done += len(batch)
            if progress:
                progress(done, len(todo))
            if err or data is None:
                warnings.append(f"LLM extraction failed for {len(batch)} units, regex used: {err}")
                continue
            got = {u["unit_id"]: u for u in data.get("units", [])}
            for u in batch:
                res = got.get(u["unit_id"]) or {"entities": [], "relations": []}
                res = {"entities": res.get("entities", []), "relations": res.get("relations", [])}
                cache[_hash(u["text"])] = res
                out[u["unit_id"]] = {**_merge(res, regex_extract(u["text"], u.get("heading", ""))), "method": "llm"}
    if todo:
        save_cache(cache)
    return out, usage, warnings
