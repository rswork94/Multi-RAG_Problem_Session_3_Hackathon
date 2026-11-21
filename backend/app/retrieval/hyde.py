"""Multi-HyDE: N hypothetical passages written from distinct regulatory angles (claude-haiku-5-5, in parallel).
Offline: deterministic lay-term -> regulatory-vocabulary expansion, so the engine still behaves like HyDE."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

from ..llm import calls, client
from ..prompts.areas import AREAS
from ..prompts.tasks import HYDE_ANGLES, HYDE_SYSTEM, HYDE_USER

LAY_TO_FORMAL = {
    "tax": {
        r"digital (invoice|bill|billing)|electronic (bill|invoice)|e-?bill": "e-invoicing Invoice Reference Number IRN Invoice Registration Portal",
        r"big (companies|business)|large (companies|business|taxpayers)": "registered person whose aggregate turnover exceeds the threshold",
        r"tax (department|authority|office)|income tax dept": "CBDT Central Board of Direct Taxes",
        r"indirect tax (department|board)|gst (department|office)": "CBIC Central Board of Indirect Taxes and Customs",
        r"withhold|deduct(ing)? tax": "tax deducted at source TDS deductor deduction",
        r"professional fees|consultan|advis(er|or) fees": "fees for professional services",
        r"head office credit|share (the )?credit|credit distribution": "Input Service Distributor ISD distribution of input tax credit",
        r"deadline|due date|last date|file by": "shall be furnished on or before the due date statement",
        r"from the (start|beginning)|backdated|retrospective": "retrospective clarification with effect from inception",
        r"going forward|future|prospective": "prospective amendment with effect from",
    },
    "grc": {
        r"customer (checks|verification)|know your customer|identity checks": "KYC customer due diligence CDD",
        r"update (records|details)|refresh": "periodic updation re-KYC",
        r"money laundering|dirty money": "PMLA anti-money laundering AML",
        r"suspicious|report(ing)? to (the )?(authorit|government)": "suspicious transaction report STR FIU-IND",
        r"central bank|banking regulator": "Reserve Bank of India RBI master direction",
        r"market regulator|securities regulator|stock market regulator": "SEBI circular",
        r"sustainability|esg|climate report": "BRSR Business Responsibility and Sustainability Report BRSR Core assurance",
        r"\bfine|punish|sanction": "monetary penalty enforcement action",
        r"risky customers|high risk": "high risk customers enhanced due diligence",
        r"check(s)? (and|&) balance|safeguard": "internal control control owner frequency",
    },
    "deals": {
        r"buyer|acquirer|investor": "Purchaser",
        r"seller|founder|promoter": "Seller",
        r"compensat\w+ for loss|make good|reimburse": "indemnify indemnification claims losses",
        r"maximum (liability|payout)|limit on claims|most .{0,20} (pay|liable)": "aggregate liability cap of the Seller",
        r"(drop|fall|decline) in (profit|earnings)|business (deteriorat|collapse)": "Material Adverse Change EBITDA decline",
        r"walk away|exit the deal|cancel the deal|pull out": "termination right long-stop date",
        r"fair price|price per share|worth": "fair value per share registered valuer Section 247",
        r"foreign (buyer|investor)|overseas|non-?resident": "FEMA pricing guidelines non-resident transfer Form FC-TRS",
        r"held back|retention|kept aside": "escrow holdback amount release date",
        r"foreign (donation|money|funding)": "FCRA foreign contribution",
    },
}


def offline_hypotheses(question: str, area: str, n: int) -> list[str]:
    """Offline stand-in for the LLM: rewrite lay phrases into regulatory vocabulary (query rewriting).
    Variants: (1) lay terms substituted in place, (2) the question plus the formal terms, (3) formal terms only.
    Angle descriptions are not embedded, because they are style instructions, not content."""
    table = LAY_TO_FORMAL.get(area, {})
    adds = [formal for rx, formal in table.items() if re.search(rx, question, re.I)]
    rewritten = question
    for rx, formal in table.items():
        rewritten = re.sub(rx, formal, rewritten, flags=re.I)
    variants = [rewritten, f"{question} {' '.join(adds)}".strip(), " ".join(adds) or rewritten]
    return [variants[i % len(variants)] for i in range(n)]


def generate(question: str, area: str, n: int) -> dict:
    """Returns {hypotheses, usage, ms, mode}. All N calls run in parallel."""
    t0 = time.perf_counter()
    if not client.available():
        return {"hypotheses": offline_hypotheses(question, area, n), "usage": {}, "ms": 0.0, "mode": "offline-expansion"}
    angles = HYDE_ANGLES.get(area, HYDE_ANGLES["tax"])

    def one(i: int):
        user = HYDE_USER.format(area_label=AREAS[area]["label"], question=question, angle=angles[i % len(angles)])
        try:
            return calls.short_text(HYDE_SYSTEM, user)
        except client.LLMError as e:
            return None, {"error": str(e)}

    with ThreadPoolExecutor(max(1, n)) as ex:
        res = list(ex.map(one, range(n)))
    hyps = [t for t, _ in res if t]
    usage = client.merge_usage(*[u for _, u in res if u and "error" not in u])
    mode = "claude-haiku"
    if not hyps:
        hyps, mode = offline_hypotheses(question, area, n), "offline-expansion (LLM failed)"
    return {"hypotheses": hyps, "usage": usage, "ms": round((time.perf_counter() - t0) * 1000, 1), "mode": mode}
