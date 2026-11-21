"""Practice-area guardrails (UI option switcher). Each block is static so it is cached per area.

`REQUIRED_SECTIONS` is what guardrails/output_checks.py verifies in every answer.
"""
from __future__ import annotations

from .costar import SYSTEM_PROMPT

AREAS = {
    "tax": {
        "label": "Tax & Regulatory Services",
        "focus": "Direct tax (CBDT notifications, Income-tax Act and Rules) and indirect tax (CBIC circulars, GST Council "
                 "recommendations, ISD rules, e-invoicing).",
        "rules": ["Cite exact Section / Rule / Notification / Circular numbers for every legal claim.",
                  "Distinguish prospective amendments from retrospective clarifications."],
    },
    "grc": {
        "label": "Risk Advisory & Governance (GRC)",
        "focus": "RBI Master Directions (KYC/PMLA), SEBI BRSR guidelines, ISO 31000 risk management frameworks and "
                 "internal audit controls.",
        "rules": ["Flag non-compliance penalties immediately.",
                  "Strictly map controls to specific regulatory obligations."],
    },
    "deals": {
        "label": "Deal Advisory & Corporate Finance",
        "focus": "M&A Share Purchase Agreements (SPA), Companies Act 2013 valuation standards and FEMA/FCRA foreign "
                 "investment rules.",
        "rules": ["Highlight indemnities, material adverse change (MAC) clauses and financial exposure thresholds."],
    },
}

AREA_BLOCKS = {
    "tax": """Active practice area: 1. Tax & Regulatory Services.
Focus: Direct tax (CBDT notifications, Income-tax Act and Rules) and indirect tax (CBIC circulars, GST Council recommendations, ISD rules, e-invoicing).
Enforced guardrails:
T1 Cite the exact Section, Rule, Notification or Circular number for every legal claim, exactly as it appears in the passage.
T2 Distinguish prospective amendments from retrospective clarifications. Immediately after the Answer, add a line that starts with "**Amendment nature:**" followed by "Prospective (effective <date>)", "Retrospective clarification (applies from <date or event>)", "Mixed: ..." or "Not applicable", citing the passage.
T3 If a rate or threshold has been superseded, give the current figure and name the superseded one.""",
    "grc": """Active practice area: 2. Risk Advisory & Governance (GRC).
Focus: RBI Master Directions (KYC/PMLA), SEBI BRSR guidelines, ISO 31000 risk management frameworks and internal audit controls.
Enforced guardrails:
R1 Flag non-compliance penalties immediately: the response must begin with a block headed "**⚠ Penalty exposure**" that lists each penalty or enforcement consequence in the passages (amount, basis, provision) with citations, or states "No penalty identified in the indexed sources."
R2 Strictly map controls to specific regulatory obligations: include a section headed "**Control → obligation mapping**" with a Markdown table | Control | Regulatory obligation | Source |. Use only controls and obligations named in the passages; write "No control mapping in the indexed sources." if there are none.""",
    "deals": """Active practice area: 3. Deal Advisory & Corporate Finance.
Focus: M&A Share Purchase Agreements (SPA), Companies Act 2013 valuation standards and FEMA/FCRA foreign investment rules.
Enforced guardrails:
D1 Immediately after the Answer, include a section headed "**Deal exposure**" with bullets for Indemnities (caps, baskets, de minimis, survival), MAC clauses (definition and triggers) and Financial exposure thresholds (amounts, percentages, escrow), each with clause references and citations. Write "Not addressed in the indexed sources." for any item the passages do not cover.
D2 Where an amendment agreement or later document changes a figure, state the operative figure and the superseded one.""",
}

# Regexes (case-insensitive) that every answer for the area must contain.
REQUIRED_SECTIONS = {
    "tax": {"Amendment nature line": r"\*\*\s*amendment nature\s*:?\s*\*\*"},
    "grc": {"Penalty exposure block": r"penalty exposure", "Control → obligation mapping": r"control\s*(→|->|to)\s*obligation"},
    "deals": {"Deal exposure section": r"\*\*\s*deal exposure\s*\*\*"},
}


def system_blocks(area: str) -> list[dict]:
    """Two cache breakpoints: the shared CO-STAR prompt, then the area block."""
    return [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": AREA_BLOCKS[area], "cache_control": {"type": "ephemeral"}}]
