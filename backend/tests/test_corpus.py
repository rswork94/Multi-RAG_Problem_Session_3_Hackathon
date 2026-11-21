"""The shipped corpus specs satisfy every design rule (counts, unique anchors, bridge entities, planted aliases)."""
from validate_corpus import validate


def test_corpus_specs_validate():
    errors, _ = validate(["tax", "grc", "deals"])
    assert errors == []


def test_each_area_has_the_question_mix():
    from corpus_lib import load_spec

    for area in ("tax", "grc", "deals"):
        cats = [q["category"] for q in load_spec(area)["questions"]]
        assert cats.count("keyword") == 4 and cats.count("multihop") == 4
        assert cats.count("vocab_alias") == 2 and cats.count("vocab_paraphrase") == 2
