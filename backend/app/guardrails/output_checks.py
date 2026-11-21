"""Output guard: citation validation, practice-area section checks, disclaimer, PII re-masking.

Code-level enforcement, so the prompt is not relied on alone. Missing GRC penalty blocks are filled with a
labelled auto-extracted block (never a second Opus call, which would distort the benchmark).
"""
from __future__ import annotations

import json
import re

from ..prompts.areas import REQUIRED_SECTIONS
from ..prompts.costar import DISCLAIMER

CITE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
PROVISION = re.compile(r"(?i)\b(section|sec\.|rule|notification|circular|clause|paragraph|para|regulation|form)\s+"
                       r"(no\.\s*)?[\w./()-]*\d")
PENALTY_SENT = re.compile(r"(?i)[^.\n]*\b(penalt\w*|fine|monetary penalty|enforcement action)\b[^.\n]*\.")


def _expand(group: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"\s*,\s*", group):
        if re.search(r"[–-]", part):
            a, b = [int(x) for x in re.split(r"\s*[–-]\s*", part)[:2]]
            out.extend(range(a, b + 1))
        elif part.strip().isdigit():
            out.append(int(part))
    return out


def validate_citations(answer: str, n_passages: int) -> dict:
    nums = [n for g in CITE.findall(answer) for n in _expand(g)]
    valid = sorted({n for n in nums if 1 <= n <= n_passages})
    invalid = sorted({n for n in nums if not 1 <= n <= n_passages})
    body = re.split(r"(?i)\*\*\s*(gaps|disclaimer)", answer)[0]
    paras = [p for p in re.split(r"\n\s*\n|\n(?=\s*[-*] )", body)
             if len(p.split()) >= 8 and not re.match(r"\s*\*\*[^*]+\*\*\s*$", p) and "|" not in p[:3]]
    uncited = [p.strip()[:120] for p in paras if not CITE.search(p) and "not found" not in p.lower()]
    coverage = round(1 - len(uncited) / len(paras), 3) if paras else 1.0
    return {"valid": valid, "invalid": invalid, "uncited_paragraphs": uncited, "coverage": coverage}


def enforce_disclaimer(answer: str) -> tuple[str, bool]:
    if DISCLAIMER.lower()[:40] in answer.lower():
        return answer, False
    return answer.rstrip() + f"\n\n**Disclaimer**\n{DISCLAIMER}", True


TOKEN_FOOTER = re.compile(r"\s*---\n\*Estimated Output Tokens: ~\d+ tokens\*\s*$")


def _is_json(text: str) -> bool:
    if text[0] not in "{[":
        return False
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def estimate_tokens(text: str) -> int:
    """Heuristic output-token estimate: characters / 4, plus 12.5% when the text has code blocks, is JSON,
    or is at least 5% non-English letters. Deterministic, so it is identical online and offline."""
    text = text.strip()
    if not text:
        return 0
    n = len(text) / 4
    letters = [c for c in text if c.isalpha()]
    non_english = sum(1 for c in letters if not c.isascii()) / len(letters) if letters else 0.0
    if "```" in text or _is_json(text) or non_english >= 0.05:
        n *= 1.125
    return max(1, round(n))


def append_token_estimate(answer: str) -> tuple[str, int]:
    """Appends the "Estimated Output Tokens" footer at the very bottom of the answer (idempotent)."""
    body = TOKEN_FOOTER.sub("", answer).rstrip()
    if not body:
        return answer, 0
    n = estimate_tokens(body)
    return f"{body}\n\n---\n*Estimated Output Tokens: ~{n} tokens*", n


def area_checks(area: str, answer: str, context: list[dict]) -> tuple[str, dict]:
    """Returns (possibly repaired answer, report)."""
    missing = [name for name, rx in REQUIRED_SECTIONS.get(area, {}).items() if not re.search(rx, answer, re.I)]
    report: dict = {"missing_sections": missing, "auto_filled": [], "provision_citations": None}
    if area == "tax":
        body = re.split(r"(?i)\*\*\s*(gaps|disclaimer)", answer)[0]
        report["provision_citations"] = len(PROVISION.findall(body))
        if report["provision_citations"] == 0 and "not found" not in body.lower():
            report["missing_sections"].append("Section/Rule/Notification number")
    if area == "grc" and "Penalty exposure block" in missing:
        lines = []
        for c in context:
            for m in PENALTY_SENT.finditer(c["text"]):
                s = " ".join(m.group(0).split())
                if 20 < len(s) < 400:
                    lines.append(f"- {s} [{c['n']}]")
        block = "**⚠ Penalty exposure** _(auto-extracted by the output guard from the cited passages)_\n" + \
            ("\n".join(dict.fromkeys(lines[:5])) if lines else "- No penalty identified in the indexed sources.")
        answer = block + "\n\n" + answer
        report["auto_filled"].append("Penalty exposure block")
        report["missing_sections"] = [m for m in missing if m != "Penalty exposure block"]
    report["compliant"] = not report["missing_sections"]
    return answer, report
