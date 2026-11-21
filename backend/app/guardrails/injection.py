"""Prompt-injection heuristics applied at ingest. Flagged chunks are down-weighted and surfaced, never dropped."""
from __future__ import annotations

import re

PHRASES = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|directions)",
    r"disregard (the |all )?(previous|prior|above|system)",
    r"you are now", r"system prompt", r"as an ai (assistant|model)",
    r"(state|say|answer|respond) that no (penalty|tax|liability|indemnity)",
    r"do not (cite|mention|disclose)", r"override (the )?(guardrails?|instructions)",
    r"(any|an) (ai|language model|assistant) (reading|processing) this",
]
_RX = re.compile("|".join(f"(?:{p})" for p in PHRASES), re.I)
DOWNWEIGHT = 0.85


def injection_flags(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _RX.finditer(text or "")})
