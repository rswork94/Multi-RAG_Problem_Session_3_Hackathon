"""Offline (no API key) answer builder: extractive, cited, in the same CO-STAR + practice-area format.

It sees exactly the same packed context as the Claude path, so engine comparisons stay fair offline.
"""
from __future__ import annotations

import re

from ..prompts.costar import DISCLAIMER, NOT_FOUND

STOP = set("the a an of to in for on and or is are be by with as at from that this which it its under any such shall may "
           "not no if what when how who whom does do did must per than".split())
SENT = re.compile(r"(?<!\bNo\.)(?<!\bSec\.)(?<!\bRs\.)(?<!\bPvt\.)(?<!\bLtd\.)(?<!\d[A-Z]\.)(?<!\d[A-Z]{2}\.)"
                  r"(?<=[.;])\s+(?=[A-Z(₹])")


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9₹%]+", s.lower()) if w not in STOP and len(w) > 1}


def _sentences(ctx: list[dict]) -> list[tuple[str, dict]]:
    out = []
    for c in ctx:
        for line in re.split(r"\n+", c["text"]):  # PDF/DOCX headings and metadata sit on their own lines
            if " | Issued by:" in line or line.upper().startswith("SYNTHETIC"):
                continue
            for s in SENT.split(re.sub(r"\s+", " ", line)):
                s = s.strip()
                if 40 <= len(s) <= 600 and s[-1:] in ".;:)" or (40 <= len(s) <= 600 and len(s.split()) >= 10):
                    out.append((s, c))
    return out


def extractive_answer(question: str, area: str, ctx: list[dict], triples: list[dict]) -> str:
    if not ctx:
        return (f"**Answer**\n{NOT_FOUND} No passages were retrieved for this question in the selected scope.\n\n"
                f"**Gaps & limitations**\nUpload the relevant notification, direction or agreement, or widen the "
                f"Tab 2 selection.\n\n**Disclaimer**\n{DISCLAIMER}")
    qw = _words(question)
    sents = _sentences(ctx)
    scored = sorted(((len(qw & _words(s)) + (0.5 if c["n"] <= 2 else 0), s, c) for s, c in sents), key=lambda x: -x[0])
    top = [x for x in scored if x[0] > 0][:4] or scored[:2]
    lines = ["**Answer**", "_Offline mode (no Claude API key): the most relevant source sentences, quoted verbatim._", ""]
    lines += [f"{s} [{c['n']}]" for _, s, c in top]
    lines.append("")
    if area == "tax":
        pro = next(((s, c) for s, c in sents if re.search(r"prospective|with effect from|w\.e\.f\.", s, re.I)), None)
        ret = next(((s, c) for s, c in sents if re.search(r"retrospective|since inception|clarif", s, re.I)), None)
        nature = "Retrospective clarification" if ret and (not pro or ret[1]["n"] <= pro[1]["n"]) else \
            "Prospective" if pro else "Not stated in the retrieved passages"
        ref = ret or pro
        lines.append(f"**Amendment nature:** {nature}" + (f" [{ref[1]['n']}]" if ref else ""))
    elif area == "grc":
        pens = [(s, c) for s, c in sents if re.search(r"penalt|fine|enforcement action", s, re.I)][:4]
        lines.append("**⚠ Penalty exposure**")
        lines += [f"- {s} [{c['n']}]" for s, c in pens] or ["- No penalty identified in the indexed sources."]
        lines.append("\n**Control → obligation mapping**")
        ctrls = [(m.group(0), s, c) for s, c in sents for m in re.finditer(r"\b[A-Z]{1,3}-[A-Z]{2,6}-\d{2,3}\b", s)][:4]
        if ctrls:
            lines += ["| Control | Regulatory obligation | Source |", "|---|---|---|"]
            lines += [f"| {cid} | {s[:140]} | [{c['n']}] |" for cid, s, c in ctrls]
        else:
            lines.append("No control mapping in the indexed sources.")
    elif area == "deals":
        lines.append("**Deal exposure**")
        for label, rx in (("Indemnities", r"indemn|basket|de minimis|survival"), ("MAC clauses", r"material adverse|\bMAC\b"),
                          ("Financial exposure thresholds", r"cap\b|₹|per cent|escrow|threshold")):
            hit = next(((s, c) for s, c in sents if re.search(rx, s, re.I)), None)
            lines.append(f"- {label}: " + (f"{hit[0]} [{hit[1]['n']}]" if hit else "Not addressed in the indexed sources."))
    lines.append("\n**Basis**")
    seen = set()
    for _, s, c in top:
        if c["n"] not in seen:
            seen.add(c["n"])
            lines.append(f"- {c.get('section') or c['title']} ({c['doc_type']}, {c['source']}, p.{c['page']}) [{c['n']}]")
    lines.append("\n**Gaps & limitations**\nExtractive summary only; no legal reasoning was applied. Configure "
                 "ANTHROPIC_API_KEY for a reasoned answer.")
    if any(c.get("injection_flag") for c in ctx):
        lines.append("A retrieved passage contained instruction-like text; it was treated as data and disregarded.")
    lines.append(f"\n**Disclaimer**\n{DISCLAIMER}")
    return "\n".join(lines)


def heuristic_support(answer: str, ctx: list[dict]) -> tuple[float | None, bool, list[str]]:
    """(faithfulness, abstained, unsupported sentences) by lexical overlap; used when no API key."""
    body = re.split(r"(?i)\*\*\s*(gaps|disclaimer)", answer)[0]
    abstained = NOT_FOUND.lower() in body.lower() and len(body) < 400
    sents = [s for s in re.split(r"(?<=[.;])\s+|\n", re.sub(r"\[\d+(?:[,–-]\s*\d+)*\]", "", body))
             if len(_words(s)) >= 5 and not s.strip().startswith(("_", "|---", "**"))]
    if abstained or not sents:
        return (None if abstained else 1.0), abstained, []
    ctx_words = set().union(*[_words(c["text"]) for c in ctx]) if ctx else set()
    unsupported = [s.strip()[:160] for s in sents if len(_words(s) & ctx_words) / len(_words(s)) < 0.6]
    return round(1 - len(unsupported) / len(sents), 3), False, unsupported[:5]
