"""Step 1 - Data format & ingestion. LangChain loaders -> one normalised record per file with strict provenance.

PDF   -> UnstructuredPDFLoader(mode="elements", strategy="fast"); runtime fallback PyPDFLoader.
DOCX  -> Docx2txtLoader.
JSON / CSV -> custom loader (no jq): notes become text, every row is an atomic unit that is never split.
TXT / MD -> TextLoader.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from pathlib import Path

log = logging.getLogger("multirag.loaders")

META_LINE = re.compile(r"^(\w+) \| Issued by: (.+?) \| Date: (\d{4}-\d{2}-\d{2})", re.M)
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RXS = [re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
            re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})\s*,?\s+(\d{{4}})\b")]
DOC_TYPES = [("MasterDirection", r"master direction"), ("Notification", r"\bnotification\b"),
             ("Circular", r"\bcircular\b"), ("Agreement", r"\bagreement\b|\bdeed\b"),
             ("Memo", r"\bmemo(randum)?\b|\bnote\b|\bletter\b|\breport\b"), ("SOP", r"standard operating procedure|\bsop\b"),
             ("Rules", r"\brules\b"), ("Act", r"\bact,\s*\d{4}"), ("Guideline", r"guideline|policy|framework"),
             ("Table", r"\btable\b|schedule|matrix")]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _detect(text: str, title: str) -> dict:
    head = text[:1500]
    m = META_LINE.search(head)
    if m:
        return {"doc_type": m.group(1), "issuer": m.group(2), "issued_date": m.group(3)}
    out = {"doc_type": "Document", "issuer": "", "issued_date": ""}
    probe = f"{title}\n{head}".lower()
    for dt, rx in DOC_TYPES:
        if re.search(rx, probe):
            out["doc_type"] = dt
            break
    for rx in DATE_RXS:
        d = rx.search(head)
        if d:
            if d.re is DATE_RXS[0]:
                out["issued_date"] = d.group(0)
            else:
                mi = MONTHS.split("|").index(d.group(2)) + 1
                out["issued_date"] = f"{d.group(3)}-{mi:02d}-{int(d.group(1)):02d}"
            break
    return out


def _title_from(lines: list[str]) -> str:
    for ln in lines:
        s = ln.strip()
        if s and not s.upper().startswith("SYNTHETIC") and not s.startswith("#") and len(s) > 3:
            return s[:160]
    return "Untitled"


def _flatten(meta: dict) -> dict:
    """Chroma accepts scalars only; drop None, join lists."""
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out[k] = ", ".join(map(str, v))
        elif isinstance(v, dict):
            continue
        else:
            out[k] = v
    return out


def _load_pdf(path: Path) -> tuple[str, list[tuple[int, int]], str, str]:
    """Returns (text, page_spans[(char_start, page)], title, loader_used)."""
    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader
        els = UnstructuredPDFLoader(str(path), mode="elements", strategy="fast").load()
        if not els:
            raise ValueError("unstructured returned no elements")
        parts, spans, pos, title = [], [], 0, ""
        for e in els:
            t = e.page_content.strip()
            if not t:
                continue
            if not title and e.metadata.get("category") == "Title" and not t.upper().startswith("SYNTHETIC"):
                title = t
            spans.append((pos, int(e.metadata.get("page_number") or 1)))
            parts.append(t)
            pos += len(t) + 2
        return "\n\n".join(parts), spans, title, "UnstructuredPDFLoader(fast)"
    except Exception as e:  # runtime fallback: keeps ingest working if unstructured breaks
        log.warning("UnstructuredPDFLoader failed for %s (%s); falling back to PyPDFLoader", path.name, e)
        from langchain_community.document_loaders import PyPDFLoader
        pages = PyPDFLoader(str(path)).load()
        parts, spans, pos = [], [], 0
        for i, p in enumerate(pages):
            t = p.page_content.strip()
            spans.append((pos, i + 1))
            parts.append(t)
            pos += len(t) + 2
        return "\n\n".join(parts), spans, "", "PyPDFLoader (fallback)"


def _rows_to_units(rows: list[dict]) -> list[str]:
    return ["; ".join(f"{k}: {v}" for k, v in r.items() if v not in (None, "")) for r in rows]


def _load_json(path: Path) -> tuple[str, list[str], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta: dict = {}
    notes: list[str] = []
    rows: list[dict] = []
    if isinstance(data, list):
        rows = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict):
        for k in ("title", "doc_type", "issuer", "issued_date"):
            if isinstance(data.get(k), str):
                meta[k] = data[k]
        if isinstance(data.get("banner"), str):
            notes.append(data["banner"])
        for n in data.get("notes") or []:
            if isinstance(n, dict):
                notes.append(f'{n.get("heading", "")}\n{n.get("text", "")}'.strip())
        for k, v in data.items():
            if isinstance(v, list) and v and all(isinstance(r, dict) for r in v) and k != "notes":
                rows.extend(v)
            elif k not in ("title", "doc_type", "issuer", "issued_date", "banner", "notes") and isinstance(v, (str, int, float)):
                notes.append(f"{k}: {v}")
    return "\n\n".join(notes), _rows_to_units(rows), meta


def _load_csv(path: Path) -> tuple[str, list[str], dict]:
    raw = path.read_text(encoding="utf-8")
    notes = [ln[1:].strip() for ln in raw.splitlines() if ln.startswith("#")]
    body = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("#"))
    rows = list(csv.DictReader(io.StringIO(body))) if body.strip() else []
    meta = {}
    if len(notes) > 1 and "|" in notes[1]:  # "# title | DocType | Issued by: X | Date: YYYY-MM-DD"
        parts = [p.strip() for p in notes[1].split("|")]
        meta["title"] = parts[0]
        for p in parts[1:]:
            if p.startswith("Issued by:"):
                meta["issuer"] = p.split(":", 1)[1].strip()
            elif p.startswith("Date:"):
                meta["issued_date"] = p.split(":", 1)[1].strip()
            elif re.fullmatch(r"[A-Za-z]+", p):
                meta["doc_type"] = p
    return "\n\n".join(notes), _rows_to_units(rows), meta


def load_file(path: Path, practice_area: str, source_name: str | None = None) -> dict:
    """Load one file into a normalised document record."""
    data = path.read_bytes()
    ext = path.suffix.lower()
    source = source_name or path.name
    rows: list[str] = []
    title = ""
    spans: list[tuple[int, int]] = [(0, 1)]
    extra_meta: dict = {}
    if ext == ".pdf":
        text, spans, title, loader = _load_pdf(path)
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        text = "\n\n".join(re.sub(r"\n{2,}", "\n\n", d.page_content).strip() for d in Docx2txtLoader(str(path)).load())
        loader = "Docx2txtLoader"
    elif ext == ".json":
        text, rows, extra_meta = _load_json(path)
        loader = "JSONLoader (row-atomic)"
    elif ext == ".csv":
        text, rows, extra_meta = _load_csv(path)
        loader = "CSVLoader (row-atomic)"
    else:
        from langchain_community.document_loaders import TextLoader
        text = "\n\n".join(d.page_content for d in TextLoader(str(path), encoding="utf-8").load())
        loader = "TextLoader"
    if not text.strip() and not rows:
        raise ValueError(f"{source}: no extractable text (scanned PDFs need OCR, which is not supported)")
    title = extra_meta.get("title") or title or _title_from(text.splitlines())
    det = _detect(text, title)
    for k in ("doc_type", "issuer", "issued_date"):
        if extra_meta.get(k):
            det[k] = str(extra_meta[k])
    return _flatten({
        "doc_id": _sha(data), "source": source, "title": title, "format": ext.lstrip("."),
        "practice_area": practice_area, "loader_used": loader, "chars": len(text) + sum(map(len, rows)),
        "pages": max((p for _, p in spans), default=1), **det,
    }) | {"text": text, "rows": rows, "page_spans": spans}


def page_at(doc: dict, offset: int) -> int:
    page = 1
    for start, p in doc.get("page_spans") or [(0, 1)]:
        if start <= offset:
            page = p
        else:
            break
    return page
