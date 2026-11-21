"""Shared corpus helpers: load area specs, add deterministic filler clauses, render documents.

Used by build_corpus.py (renders files + gold labels) and validate_corpus.py (checks the design rules).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import yaml

BACKEND = Path(__file__).resolve().parent.parent
SPEC_DIR = BACKEND / "corpus_spec"
OUT_DIR = BACKEND / "corpus" / "generated"
GOLD_DIR = BACKEND / "corpus" / "gold"
AREAS = ("tax", "grc", "deals")
BANNER = "SYNTHETIC: benchmark only, not actual law. Fictional instruments, parties and figures."

# ---------------------------------------------------------------- filler templates
# Generic procedural provisions that bulk documents out to realistic length. They avoid specific
# numbers so they never collide with gold anchors (the validator double-checks this).
FILLER = {
    "tax": [
        ("Maintenance of records", "Every {actor} shall maintain {record} in respect of {subject} and shall preserve such records for the period specified in the Act, so that they can be produced before the {officer} when called upon to do so. The records may be kept in electronic form, provided that they remain accessible, legible and capable of being reproduced in a paper copy where required."),
        ("Manner of furnishing", "The {document} referred to in this {instrument} shall be furnished electronically under a digital signature or through an electronic verification code, in the format and manner specified by the {system_officer}. A {document} furnished otherwise than in the specified manner shall be treated as defective, and the {actor} shall be intimated of the defect and given an opportunity to rectify it."),
        ("Rectification of mistakes", "Where any mistake apparent from the record is noticed in a {document} furnished under this {instrument}, the {actor} may furnish a correction statement. The {officer} may also, on his own motion, rectify such a mistake after giving the {actor} a reasonable opportunity of being heard, and shall record the reasons for the rectification in writing."),
        ("Interpretation", "Words and expressions used in this {instrument} and not defined herein but defined in the {act} shall have the meanings respectively assigned to them in that Act. Any reference in this {instrument} to a provision of the {act} shall be construed as a reference to that provision as amended from time to time."),
        ("Saving", "Nothing contained in this {instrument} shall affect any proceeding initiated, any liability incurred or any {document} furnished before the date of its commencement, and every such proceeding, liability or {document} shall continue to be governed by the provisions in force immediately before that date."),
        ("Electronic communication", "Any notice, intimation or order under this {instrument} may be served on the {actor} by electronic mail to the address registered on the portal or by placing it in the registered account of the {actor}. Service so effected shall be deemed to be valid service for all purposes."),
        ("Removal of difficulties", "If any difficulty arises in giving effect to the provisions of this {instrument}, the {authority} may issue guidelines, not inconsistent with the provisions of the {act}, for the purpose of removing the difficulty. Every guideline so issued shall be laid before each House of Parliament."),
        ("Verification", "Every {document} furnished under this {instrument} shall be verified by the person responsible for {subject}, or by a person duly authorised by him in this behalf, and the verification shall state that the particulars furnished are true, correct and complete to the best of his knowledge and belief."),
        ("Processing of statements", "The {document} furnished by the {actor} shall be processed by the {system_officer} in a centralised manner. Any sum found payable, or any refund due, shall be computed after making the adjustments specified in the {act}, and an intimation of the computation shall be sent to the {actor}."),
        ("Clarification on scope", "This {instrument} applies only to the matters expressly referred to herein. It does not extend the scope of any charging provision, and it shall not be relied upon to determine the taxability of any transaction outside the scope of {subject}."),
    ],
    "grc": [
        ("Board oversight", "The Board of the {entity} shall approve the {policy} and review its implementation at least once in every financial year. The Board may delegate the periodic monitoring of the {policy} to a committee of the Board, which shall report its observations and the corrective action taken to the Board."),
        ("Training and awareness", "The {entity} shall put in place an ongoing training programme for its employees covering {subject}. The training shall be differentiated for front-line staff, compliance staff and staff dealing with new customers, and records of attendance and assessment shall be maintained."),
        ("Record retention", "The {entity} shall maintain records relating to {subject} in a manner that allows the reconstruction of individual transactions, and shall make such records available to the {regulator} and other competent authorities upon request, within the time permitted by law."),
        ("Management information", "The {function} shall present a management information report on {subject} to the {committee} at each scheduled meeting. The report shall highlight exceptions, overdue actions, emerging risks and the status of remediation plans agreed in earlier meetings."),
        ("Escalation", "Any exception relating to {subject} that remains unresolved beyond the time permitted under the {policy} shall be escalated by the {function} to the {committee}. Repeated exceptions shall be considered in the annual performance assessment of the responsible business unit."),
        ("Outsourcing", "Where any activity relating to {subject} is outsourced, the {entity} shall remain fully responsible for compliance. The outsourcing agreement shall provide for audit and inspection rights for the {entity} and the {regulator}, and for the confidentiality of customer information."),
        ("Independent review", "The internal audit function shall carry out an independent review of the adequacy and effectiveness of the controls relating to {subject}. Its findings shall be reported to the {committee} together with management's response and the agreed timelines for closure."),
        ("Risk assessment", "The {entity} shall carry out a documented risk assessment of {subject}, considering customers, products, delivery channels and geographies, consistent with the principles of the {standard}. The assessment shall be refreshed whenever there is a material change in the business."),
        ("Policy exceptions", "Any deviation from the {policy} shall require the prior approval of the {function}, recorded in writing with reasons. A register of approved deviations shall be maintained and placed before the {committee} for information."),
        ("Data protection", "Information collected for {subject} shall be used only for the purposes for which it was collected, shall be stored securely, and shall be shared with third parties only where required by law or with the consent of the customer."),
    ],
    "deals": [
        ("Notices", "Any notice under this {agreement} shall be in writing in the English language and shall be delivered by hand, by courier or by electronic mail to the address of the {party} set out in this {agreement}. A notice shall be deemed received on delivery or, in the case of electronic mail, at the time of transmission if sent during business hours."),
        ("Further assurances", "Each {party} shall, at its own cost, execute such further documents and do such further acts as may reasonably be required to give full effect to this {agreement} and the transactions contemplated by it, including any filings required under the {law}."),
        ("Confidentiality", "Each {party} shall keep confidential all information received from the other {party} in connection with this {agreement}, except to the extent disclosure is required by law, by a stock exchange or by a {regulator}, or is made to professional advisers bound by duties of confidentiality."),
        ("Entire agreement", "This {agreement}, together with the documents referred to in it, constitutes the entire agreement between the parties relating to its subject matter and supersedes any prior draft, agreement, undertaking or understanding, whether written or oral, relating to that subject matter."),
        ("Assignment", "No {party} may assign, transfer or create any trust in respect of the benefit of any of its rights under this {agreement} without the prior written consent of the other {party}, save for an assignment to an affiliate that remains an affiliate for the duration of its holding."),
        ("Costs", "Except as otherwise provided in this {agreement}, each {party} shall bear its own costs and expenses incurred in connection with the negotiation, preparation and execution of this {agreement}, including the fees of its legal, financial and tax advisers."),
        ("Severability", "If any provision of this {agreement} is held to be invalid or unenforceable, that provision shall be deemed deleted to the minimum extent necessary, and the validity and enforceability of the remaining provisions shall not be affected."),
        ("Counterparts", "This {agreement} may be executed in any number of counterparts, each of which when executed shall constitute a duplicate original, and all the counterparts together shall constitute one agreement. Delivery of an executed counterpart by electronic means shall be effective."),
        ("Governing law and disputes", "This {agreement} shall be governed by the laws of India. Any dispute arising out of or in connection with this {agreement} shall be referred to arbitration seated in Mumbai under the Arbitration and Conciliation Act, 1996, before a tribunal of three arbitrators."),
        ("Conduct of business pending completion", "Pending completion, the {party} shall procure that the business of the {company} is carried on in the ordinary course, consistent with past practice, and that no material contract is entered into or terminated without the prior written consent of the {counterparty}."),
    ],
}

SLOTS = {
    "tax": {
        "actor": ["deductor", "collector", "specified person", "registered person", "assessee"],
        "record": ["books of account and supporting documents", "a register of payments", "copies of invoices and debit notes"],
        "subject": ["deduction of tax at source", "collection of tax at source", "input tax credit distribution", "electronic invoicing", "advance tax payments"],
        "officer": ["Assessing Officer", "proper officer", "jurisdictional Commissioner"],
        "document": ["statement", "return", "certificate", "correction statement"],
        "instrument": ["notification", "rule", "circular"],
        "system_officer": ["Principal Director General of Income-tax (Systems)", "Director General of Systems", "Commissioner (GST Network)"],
        "act": ["Income-tax Act", "Central Goods and Services Tax Act, 2017"],
        "authority": ["Board", "Central Government"],
    },
    "grc": {
        "entity": ["regulated entity", "bank", "non-banking financial company", "listed entity"],
        "policy": ["KYC policy", "risk management policy", "compliance policy", "outsourcing policy"],
        "subject": ["customer due diligence", "transaction monitoring", "sustainability disclosures", "operational risk", "suspicious transaction reporting"],
        "regulator": ["Reserve Bank", "Securities and Exchange Board of India", "Financial Intelligence Unit-India"],
        "function": ["Chief Compliance Officer", "Principal Officer", "Chief Risk Officer"],
        "committee": ["Audit Committee", "Risk Management Committee", "Board"],
        "standard": ["risk management framework", "applicable international standard"],
    },
    "deals": {
        "agreement": ["Agreement", "Share Purchase Agreement", "Deed"],
        "party": ["Party", "Seller", "Purchaser"],
        "counterparty": ["Purchaser", "Seller"],
        "company": ["Company", "Target Company"],
        "law": ["Companies Act, 2013", "Foreign Exchange Management Act, 1999"],
        "regulator": ["regulator", "governmental authority"],
    },
}


REPEAT_QUALIFIER = {
    "tax": "This paragraph shall be read together with the other provisions of this instrument and the rules made under the Act.",
    "grc": "This requirement applies in addition to, and not in derogation of, the other requirements of this direction.",
    "deals": "This clause shall survive termination of this agreement to the extent necessary to give it effect.",
}


def _fill(template: str, area: str, rng: random.Random) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: rng.choice(SLOTS[area][m.group(1)]), template)


def filler_clauses(area: str, doc: dict) -> list[dict]:
    n = int(doc.get("filler", 0) or 0)
    if n <= 0:
        return []
    seed = int(hashlib.sha1(doc["id"].encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    pool = list(FILLER[area])
    rng.shuffle(pool)
    out = []
    for i in range(n):
        title, tmpl = pool[i % len(pool)]
        text = _fill(tmpl, area, rng)
        if i >= len(pool):  # second pass: add a qualifying sentence so repeats are not identical
            text += " " + REPEAT_QUALIFIER[area]
        out.append({"id": f"f{i + 1}", "heading": title, "text": text, "filler": True})
    return out


# ---------------------------------------------------------------- loading
def load_spec(area: str) -> dict:
    return yaml.safe_load((SPEC_DIR / f"{area}.yaml").read_text())


def available_areas() -> list[str]:
    return [a for a in AREAS if (SPEC_DIR / f"{a}.yaml").exists()]


def assembled_documents(area: str, spec: dict | None = None) -> list[dict]:
    """Documents with filler interleaved deterministically (after every 1-2 planted clauses)."""
    spec = spec or load_spec(area)
    docs = []
    for d in spec["documents"]:
        planted = [dict(c, filler=False) for c in (d.get("clauses") or [])]
        fill = filler_clauses(area, d)
        merged: list[dict] = []
        rng = random.Random(d["id"])
        fi = 0
        for c in planted:
            merged.append(c)
            for _ in range(rng.choice((0, 1, 1, 2))):
                if fi < len(fill):
                    merged.append(fill[fi])
                    fi += 1
        merged.extend(fill[fi:])
        docs.append({**d, "area": area, "clauses": merged})
    return docs


def unit_texts(doc: dict) -> dict[str, str]:
    """clause_id / row<i> -> text, used for anchor checks."""
    out = {c["id"]: f'{c.get("heading", "")}\n{c["text"]}' for c in doc.get("clauses") or []}
    for i, r in enumerate(doc.get("rows") or []):
        out[f"row{i}"] = "; ".join(f"{k}: {v}" for k, v in r.items())
    return out


# ---------------------------------------------------------------- rendering
def _meta_line(doc: dict) -> str:
    return f'{doc["doc_type"]} | Issued by: {doc.get("issuer", "-")} | Date: {doc["issued_date"]}'


def _pdf_fonts() -> tuple[str, str]:
    """reportlab's built-in Helvetica has no ₹ glyph (U+20B9), so every ₹ anchor would vanish from the PDF text.
    Use DejaVuSans (shipped with matplotlib, which has the glyph) when available."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    try:
        import matplotlib
        base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("DejaVu", str(base / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(base / "DejaVuSans-Bold.ttf")))
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "", ""


def render_pdf(doc: dict, path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    regular, bold = _pdf_fonts()
    st = getSampleStyleSheet()
    if regular:
        for name in ("Normal", "Italic", "Title", "Heading4"):
            st[name].fontName = bold if name in ("Title", "Heading4") else regular
    else:  # no Unicode font: keep anchors searchable by spelling the currency out
        doc = {**doc, "clauses": [{**c, "text": c["text"].replace("₹", "INR "), "heading": c.get("heading", "").replace("₹", "INR ")}
                                  for c in doc.get("clauses") or []]}
    banner = ParagraphStyle("banner", parent=st["Normal"], textColor="#B00020", fontSize=8)
    body = ParagraphStyle("body", parent=st["Normal"], fontSize=10, leading=14, spaceAfter=6)
    head = ParagraphStyle("head", parent=st["Heading4"], spaceBefore=8, spaceAfter=2)
    flow = [Paragraph(escape(BANNER), banner), Spacer(1, 6), Paragraph(escape(doc["title"]), st["Title"]),
            Paragraph(escape(_meta_line(doc)), st["Italic"]), Spacer(1, 10)]
    for c in doc.get("clauses") or []:
        flow.append(Paragraph(escape(c.get("heading", "")), head))
        flow.append(Paragraph(escape(c["text"].strip()), body))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      title=doc["title"], author=doc.get("issuer", "")).build(flow)


def render_docx(doc: dict, path: Path) -> None:
    import docx

    d = docx.Document()
    d.add_paragraph(BANNER)
    d.add_heading(doc["title"], level=1)
    d.add_paragraph(_meta_line(doc))
    for c in doc.get("clauses") or []:
        d.add_heading(c.get("heading", ""), level=3)
        d.add_paragraph(c["text"].strip())
    d.core_properties.title = doc["title"]
    d.save(str(path))


def render_json(doc: dict, path: Path) -> None:
    payload = {"banner": BANNER, "title": doc["title"], "doc_type": doc["doc_type"], "issuer": doc.get("issuer"),
               "issued_date": str(doc["issued_date"]),
               "notes": [{"heading": c.get("heading", ""), "text": c["text"].strip()} for c in doc.get("clauses") or []],
               "rows": doc.get("rows") or []}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def render_csv(doc: dict, path: Path) -> None:
    import csv

    rows = doc.get("rows") or []
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# {BANNER}\n# {doc['title']} | {_meta_line(doc)}\n")
        for c in doc.get("clauses") or []:
            f.write(f"# {c.get('heading', '')}: {' '.join(c['text'].split())}\n")
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


RENDERERS = {"pdf": render_pdf, "docx": render_docx, "json": render_json, "csv": render_csv}


def doc_filename(doc: dict) -> str:
    return f'{doc["id"]}.{doc["format"]}'
