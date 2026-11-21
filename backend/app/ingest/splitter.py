"""Step 2 - Semantic chunking that does not split legal clauses.

RecursiveCharacterTextSplitter measured in bge-large WordPiece tokens, with legal separators tried first:
provision headers -> numbered paragraphs -> blank lines -> sub-clauses -> lines -> sentences.
Table rows are atomic: they are grouped into chunks but never split.
Also provides the clause-unit pre-parse used for per-clause entity extraction.
"""
from __future__ import annotations

import re
from functools import lru_cache

from ..config import Const
from ..guardrails.injection import injection_flags
from .loaders import page_at

HEADING_WORDS = r"(?:Section|Rule|Clause|Article|Paragraph|Para|Regulation|Schedule|Annexure|Item|Part|Chapter|Note)"
LEGAL_SEPARATORS = [
    rf"\n\n(?={HEADING_WORDS}\s+[\dA-Z])",
    r"\n\n(?=\d+(?:\.\d+)*[.)]?\s+[A-Z])",
    r"\n\n",
    r"\n(?=\([a-z0-9ivx]{1,4}\)\s)",
    r"\n",
    r"(?<=[.;:])\s+",
    r"\s+",
    "",
]
HEADING_LINE = re.compile(rf"^\s*(?:{HEADING_WORDS}\s+[\w().-]+|\d+(?:\.\d+)*[.)]?\s+[A-Z]|[A-Z][\w ,&()'’/-]{{2,90}}$)")


@lru_cache(maxsize=8)
def make_splitter(chunk_size: int = 600, chunk_overlap: int = 60):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from ..runtime import tokenizer
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer(), chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=LEGAL_SEPARATORS,
        is_separator_regex=True, keep_separator="start", add_start_index=True)


def split_document_text(text: str, chunk_size: int = 600, chunk_overlap: int = 60) -> list[str]:
    return make_splitter(chunk_size, chunk_overlap).split_text(text)


def clause_units(text: str) -> list[dict]:
    """Regex structural pre-parse into clause units {start, end, heading, text} (blocks separated by blank lines;
    a short heading-like block attaches to the paragraph that follows it)."""
    blocks = [(m.start(), m.group(0)) for m in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text)]
    units: list[dict] = []
    pending: tuple[int, str] | None = None
    for start, blk in blocks:
        is_heading = len(blk) < 140 and "\n" not in blk.strip() and bool(HEADING_LINE.match(blk)) \
            and not blk.rstrip().endswith((".", ";"))
        if is_heading:
            if pending:
                units.append({"start": pending[0], "end": pending[0] + len(pending[1]), "heading": pending[1].strip(),
                              "text": pending[1]})
            pending = (start, blk)
            continue
        if pending:
            units.append({"start": pending[0], "end": start + len(blk), "heading": pending[1].strip(),
                          "text": text[pending[0]: start + len(blk)]})
            pending = None
        else:
            units.append({"start": start, "end": start + len(blk), "heading": "", "text": blk})
    if pending:
        units.append({"start": pending[0], "end": pending[0] + len(pending[1]), "heading": pending[1].strip(),
                      "text": pending[1]})
    return units


def _section_at(units: list[dict], offset: int) -> str:
    sec = ""
    for u in units:
        if u["start"] <= offset and u["heading"]:
            sec = u["heading"]
        if u["start"] > offset:
            break
    return sec[:120]


def split_document(doc: dict, chunk_size: int, chunk_overlap: int) -> tuple[list[dict], list[dict]]:
    """Returns (chunks, units). Chunk = parent passage the LLM reads."""
    from ..runtime import count_tokens

    text = doc.get("text") or ""
    units = clause_units(text)
    for i, u in enumerate(units):
        u["unit_id"] = f'{doc["doc_id"]}:u{i:03d}'
    chunks: list[dict] = []
    base = {k: doc.get(k, "") for k in ("doc_id", "source", "title", "doc_type", "issued_date", "practice_area",
                                         "issuer", "format")}
    if text.strip():
        for d in make_splitter(chunk_size, chunk_overlap).create_documents([text]):
            start = int(d.metadata.get("start_index", 0))
            chunks.append({**base, "text": d.page_content, "start": start, "end": start + len(d.page_content),
                           "page": page_at(doc, start), "section": _section_at(units, start), "kind": "text"})
    # table rows: atomic, grouped up to chunk_size tokens
    rows = doc.get("rows") or []
    group: list[str] = []
    gtok = 0
    row_units = []
    for i, r in enumerate(rows):
        row_units.append({"unit_id": f'{doc["doc_id"]}:row{i}', "start": -1, "end": -1, "heading": f"Row {i + 1}",
                          "text": r, "row": i})
        n = count_tokens(r)
        if group and gtok + n > chunk_size:
            chunks.append({**base, "text": "\n".join(group), "start": -1, "end": -1, "page": 1,
                           "section": "Table rows", "kind": "rows"})
            group, gtok = [], 0
        group.append(r)
        gtok += n
    if group:
        chunks.append({**base, "text": "\n".join(group), "start": -1, "end": -1, "page": 1, "section": "Table rows",
                       "kind": "rows"})
    for i, c in enumerate(chunks):
        c["chunk_id"] = f'{doc["doc_id"]}:{i:04d}'
        c["n_tokens"] = count_tokens(c["text"])
        flags = injection_flags(c["text"])
        c["injection_flag"] = bool(flags)
        c["injection_phrases"] = flags
    return chunks, units + row_units


def units_in_chunk(chunk: dict, units: list[dict]) -> list[str]:
    """Clause units overlapping a chunk's character span (row chunks: rows contained in the chunk text)."""
    if chunk["kind"] == "rows":
        return [u["unit_id"] for u in units if u.get("row") is not None and u["text"] in chunk["text"]]
    return [u["unit_id"] for u in units if u.get("row") is None and u["start"] < chunk["end"] and u["end"] > chunk["start"]]


def child_windows(text: str, tokenizer, size: int = Const.CHILD_TOKENS,
                  stride: int = Const.CHILD_STRIDE) -> list[tuple[str, str]]:
    """Small-to-big children: sliding windows of `size` tokens (stride `stride`), sliced from the original text via
    tokenizer offsets so casing and punctuation are preserved. The last window is aligned to the end of the parent.
    A 600-token parent mixes several clauses; matching on ~128-token windows stops the one relevant clause from
    being diluted by its neighbours, while the LLM still reads the whole parent."""
    size = min(size, Const.CHILD_WINDOW)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    if len(offs) <= size:
        return [("w0", text)]
    starts = list(range(0, len(offs) - size + 1, stride))
    if starts[-1] != len(offs) - size:
        starts.append(len(offs) - size)
    return [(f"w{i}", text[offs[s][0]: offs[s + size - 1][1]]) for i, s in enumerate(starts)]
