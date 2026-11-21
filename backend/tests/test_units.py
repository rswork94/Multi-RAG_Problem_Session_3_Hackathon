"""Unit tests: splitter, tokenizer windows, BM25 tokens, RRF, entity resolution, guardrails, sufficiency rules."""
import re

from app.config import Const
from app.guardrails.output_checks import append_token_estimate, area_checks, estimate_tokens, validate_citations
from app.guardrails.pii import mask_pii
from app.guardrails.scope import heuristic_scope
from app.ingest.extract import regex_extract
from app.ingest.resolve import canonical, er_f1, norm
from app.ingest.splitter import child_windows, clause_units, split_document_text
from app.retrieval.graph import answer_types
from app.retrieval.search import round_robin, rrf
from app.retrieval.stores import legal_tokens
from app.runtime import tokenizer


def test_splitter_keeps_clauses_and_size():
    clause = "Rule 31AB. Statement of deduction\n\n" + ("The deductor shall furnish the statement in the prescribed form. " * 30)
    text = "\n\n".join([clause.replace("31AB", f"31A{c}") for c in "BCDEF"])
    chunks = split_document_text(text, 300, 30)
    tok = tokenizer()
    assert all(len(tok(c, add_special_tokens=False)["input_ids"]) <= 300 for c in chunks)
    # chunks start on a provision header when one is available inside the window
    assert sum(1 for c in chunks if c.lstrip().startswith("Rule ")) >= 3


def test_child_windows_small_to_big():
    long = "Section 194J tax deducted at source on fees for professional services. " * 60
    wins = child_windows(long, tokenizer())
    assert len(wins) > 3
    for _, t in wins:
        assert len(tokenizer()(t, add_special_tokens=False)["input_ids"]) <= Const.CHILD_TOKENS <= Const.CHILD_WINDOW
    assert long.rstrip().endswith(wins[-1][1].rstrip()[-40:])  # last window is aligned to the end of the parent
    assert child_windows("short text", tokenizer()) == [("w0", "short text")]


def test_clause_units_attach_headings():
    units = clause_units("Rule 31AB. Statement of deduction\n\nThe statement shall be furnished in Form 26QF.\n\nOther text.")
    assert units[0]["heading"].startswith("Rule 31AB")
    assert "Form 26QF" in units[0]["text"]


def test_legal_tokens_keep_identifiers():
    toks = legal_tokens("Under Section 194J and Rule 16(4) per Notification No. 41/2026, control C-KYC-07 (Clause 9.4(c))")
    for t in ("194j", "16(4)", "41/2026", "c-kyc-07", "9.4(c)"):
        assert t in toks


def test_rrf_math_and_round_robin():
    a = [{"chunk_id": "x", "rank": 1}, {"chunk_id": "y", "rank": 2}]
    b = [{"chunk_id": "y", "rank": 1}, {"chunk_id": "z", "rank": 2}]
    fused = rrf({"a": a, "b": b}, alpha=60)
    assert fused[0]["chunk_id"] == "y"
    assert abs(fused[0]["rrf"] - (1 / 62 + 1 / 61)) < 1e-5
    assert [f["chunk_id"] for f in round_robin({"a": a, "b": b})] == ["x", "y", "z"]


def test_regex_extraction_relations_and_aliases():
    r = regex_extract("The statement under this rule shall be furnished in Form 26QF on or before the 19th day following "
                      "the end of each quarter.", heading="Rule 31AB. Statement")
    rels = {(x["src"], x["rel"], x["dst"]) for x in r["relations"]}
    assert ("Rule 31AB", "PRESCRIBES_FORM", "Form 26QF") in rels
    r2 = regex_extract('In this SOP, the Reserve Bank of India is referred to as the "Central Regulator".')
    assert any("Central Regulator" in e["aliases"] for e in r2["entities"])


def test_canonical_and_er_f1():
    assert canonical("Sec. 194J", "Section") == "Section 194J"
    assert canonical("7.50 per cent", "Rate") == "7.5%"
    gold = [{"area": "tax", "members": ["CBDT", "Revenue Board"]}, {"area": "tax", "members": ["CBIC"]}]
    perfect = {norm("CBDT"): 1, norm("Revenue Board"): 1, norm("CBIC"): 2}
    assert er_f1(gold, perfect)["f1"] == 1.0
    merged_wrong = {norm("CBDT"): 1, norm("Revenue Board"): 1, norm("CBIC"): 1}
    assert er_f1(gold, merged_wrong)["precision"] < 1.0


def test_pii_masking():
    masked, found = mask_pii("PAN ABCPE1234F, GSTIN 27ABCPE1234F1Z5, call +91 9876543210")
    assert "ABCPE1234F" not in masked and "9876543210" not in masked
    assert {f["type"] for f in found} >= {"PAN", "GSTIN", "PHONE"}


def test_scope_heuristic():
    assert heuristic_scope("How can I create fake invoices to avoid the tax department?")["integrity_risk"] == "high"
    assert heuristic_scope("What is the best pizza in Mumbai?")["in_scope"] is False
    assert heuristic_scope("What is the TDS rate on professional fees?")["in_scope"] is True


def test_citation_validator_and_area_checks():
    ans = "**Answer**\nThe rate is 7.5 per cent under Section 194J [1]. Another claim [7].\n\n**Disclaimer**\nx"
    cv = validate_citations(ans, 3)
    assert cv["valid"] == [1] and cv["invalid"] == [7]
    _, rep = area_checks("tax", ans, [])
    assert "Amendment nature line" in rep["missing_sections"]
    ctx = [{"n": 1, "text": "A monetary penalty of ₹2,40,000 per instance applies for each failure."}]
    fixed, rep = area_checks("grc", "**Answer**\nx [1]\n\n**Control → obligation mapping**\nnone", ctx)
    assert "Penalty exposure" in fixed and rep["auto_filled"] == ["Penalty exposure block"]


def test_output_token_estimate_footer():
    assert estimate_tokens("a" * 400) == 100                                   # characters / 4
    # 800 characters -> 200, +12.5% -> 225
    assert estimate_tokens("```py\n" + "x" * 790 + "\n```") == 225             # code blocks
    assert estimate_tokens('{"rate": "' + "y" * 788 + '"}') == 225             # JSON
    assert estimate_tokens("कर " * 267) == 225                                 # non-English script (trailing space stripped)
    assert estimate_tokens("Rate ₹2,40,000 → 7.5% ⚠ " * 4) == round(len(("Rate ₹2,40,000 → 7.5% ⚠ " * 4).strip()) / 4)
    ans, n = append_token_estimate("**Answer**\nThe rate is 7.5 per cent [1].\n\n**Disclaimer**\nx")
    assert ans.endswith(f"\n\n---\n*Estimated Output Tokens: ~{n} tokens*") and n == estimate_tokens(ans.split("\n\n---")[0])
    assert append_token_estimate(ans) == (ans, n)                              # idempotent
    assert append_token_estimate("") == ("", 0)


def test_answer_types():
    assert "Deadline" in answer_types("What is the due date for the statement?")
    assert "Penalty" in answer_types("Which penalty applies if the control fails?")
    assert re.search("Threshold", " ".join(answer_types("What cap applies to claims?")))


def test_drafting_convention_aliases():
    from app.ingest.extract import instrument_titles
    t = ('This Share Purchase Agreement is executed among Halden Holdings LLP, a limited liability partnership registered '
         'under the Limited Liability Partnership Act, 2008 (the "Seller"); Orion Capital Partners (Mauritius) Ltd, a '
         'company holding a Global Business Licence (the "Purchaser"). The Risk Management Committee of the Board (RMCB) '
         'meets quarterly. control_id: C-KYC-02; control_name: Risk-Grade Review Scheduler (weekly)')
    al = {e["name"]: e["aliases"] for e in regex_extract(t)["entities"]}
    assert al["Halden Holdings LLP"] == ["Seller"] and al["Orion Capital Partners (Mauritius) Ltd"] == ["Purchaser"]
    assert "RMCB" in al["Risk Management Committee of the Board"]
    assert al["C-KYC-02"] == ["Risk-Grade Review Scheduler"]
    got = instrument_titles([{"text": "(1) These rules may be called the Income-tax (Seventeenth Amendment) Rules, 2026."}],
                            "Notification No. 41/2026: amendment")
    assert got[0]["name"] == "Notification No. 41/2026" and got[0]["aliases"] == ["Income-tax (Seventeenth Amendment) Rules, 2026"]


def test_norm_matching_keys():
    assert norm("Meridian Small Finance Bank Limited") == norm("Meridian Small Finance Bank")
    assert norm("ISO 31000:2018") == norm("ISO 31000")
    assert norm("politically exposed persons") == norm("Politically exposed person")


def test_graph_trust_levels():
    from app.retrieval.graph import graph_trust
    assert graph_trust([{"how": "identifier"}, {"how": "passage"}]) == 1.0
    assert graph_trust([{"how": "name-vector"}]) == 0.6
    assert graph_trust([{"how": "passage"}]) == 0.4
    assert graph_trust([]) == 0.0


def test_weighted_rrf():
    a = [{"chunk_id": "x", "rank": 1}]
    g = [{"chunk_id": "y", "rank": 1}]
    fused = rrf({"dense": a, "graph": g}, alpha=60, weights={"graph": 0.4})
    assert fused[0]["chunk_id"] == "x" and fused[1]["rrf"] < fused[0]["rrf"]
