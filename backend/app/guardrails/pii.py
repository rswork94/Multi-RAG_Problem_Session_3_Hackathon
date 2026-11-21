"""PII masking (input, ingest and output). Masked values become typed placeholders such as <PAN_1>."""
from __future__ import annotations

import re

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("GSTIN", re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("TAN", re.compile(r"\b[A-Z]{4}\d{5}[A-Z]\b")),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    ("BANK_ACCOUNT", re.compile(r"(?i)\b(?:a/c|account)(?:\s+(?:no\.?|number))?\s*[:#]?\s*(\d{9,18})\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")),
]


def mask_pii(text: str) -> tuple[str, list[dict]]:
    found: list[dict] = []
    counters: dict[str, int] = {}
    out = text or ""
    for kind, pat in PATTERNS:
        def repl(m: re.Match, kind=kind) -> str:
            counters[kind] = counters.get(kind, 0) + 1
            found.append({"type": kind})
            token = f"<{kind}_{counters[kind]}>"
            if kind == "BANK_ACCOUNT" and m.lastindex:
                return m.group(0).replace(m.group(1), token)
            return token
        out = pat.sub(repl, out)
    return out, found
