"""Scope / integrity classification of the incoming question.

Heuristic by default (no API spend, deterministic for the benchmark). Set USE_LLM_SCOPE=1 to use claude-haiku-5-5.
"""
from __future__ import annotations

import re

from ..config import settings
from ..llm import calls, client
from ..prompts.tasks import SCOPE_SCHEMA, SCOPE_SYSTEM

DOMAIN = re.compile(
    r"(?i)\b(tax|tds|tcs|gst|cgst|igst|isd|input service|e-?invoic|irn|cbdt|cbic|notification|circular|section|rule|form|"
    r"income|deduct|assess|rate|threshold|kyc|aml|pmla|rbi|reserve bank|sebi|brsr|iso ?31000|risk|audit|control|"
    r"penalt|complian|obligation|customer|due diligence|fiu|suspicious|spa|share purchase|indemnit|mac\b|material adverse|"
    r"escrow|warrant|valu|fema|fcra|fc-trs|companies act|acqui|seller|purchaser|deal|clause|agreement|disclos|"
    r"regulat|governance|board|policy|filing|return|statement|due date|deadline|amend|supersed|crore|lakh|₹)")
EVASION = re.compile(
    r"(?i)(evad|evasion|hide|conceal|avoid (detection|reporting|the (tax )?(department|authorit|regulator))|"
    r"fake invoice|bogus|backdat|round[- ]?trip|launder|structur\w* .{0,30}(below|under|avoid) .{0,20}(threshold|limit|report)|"
    r"split .{0,30}(avoid|below) .{0,20}(threshold|report)|without (the )?(regulator|rbi|sebi|tax).{0,20}(know|find)|"
    r"mislead|misstate|cook the books|off the books|bypass (kyc|aml|reporting))")


def heuristic_scope(question: str) -> dict:
    if EVASION.search(question):
        return {"in_scope": True, "integrity_risk": "high", "reason": "matches evasion / concealment pattern",
                "method": "heuristic"}
    if not DOMAIN.search(question):
        return {"in_scope": False, "integrity_risk": "none", "reason": "no advisory-domain vocabulary found",
                "method": "heuristic"}
    return {"in_scope": True, "integrity_risk": "none", "reason": "domain vocabulary present", "method": "heuristic"}


def classify(question: str) -> tuple[dict, dict]:
    """Returns (scope, usage)."""
    h = heuristic_scope(question)
    if not (settings.use_llm_scope and client.available()) or h["integrity_risk"] == "high":
        return h, {}
    try:
        data, usage = calls.structured(SCOPE_SYSTEM, f"<question>\n{question}\n</question>", SCOPE_SCHEMA,
                                       max_tokens=600)
        data["method"] = "claude-haiku"
        return data, usage
    except client.LLMError as e:
        h["method"] = f"heuristic (LLM error: {e})"
        return h, {}
