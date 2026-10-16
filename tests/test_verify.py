from app.state import Claim, build_refs, normalize_label
from app.tools.verify import citation_accuracy, quote_score

SOURCE = "Acme’s Pro plan costs $12 per user per month.  It includes unlimited projects and priority support."


def test_exact_quote_with_whitespace_and_curly_quotes():
    assert quote_score("Acme's Pro plan costs $12 per user per month.", SOURCE) == 1.0


def test_small_wording_difference_still_matches():
    assert quote_score("Acme's Pro plan costs $12 per user each month.", SOURCE) >= 0.9


def test_hallucinated_quote_is_rejected():
    assert quote_score("Acme offers a lifetime license for a single payment of $99.", SOURCE) < 0.9


def test_short_quote_is_rejected():
    assert quote_score("costs $12", SOURCE) == 0.0


def test_citation_accuracy():
    cites = [
        {"url": "u", "quote": "Acme's Pro plan costs $12 per user per month."},
        {"url": "u", "quote": "Acme offers a lifetime license for a single payment of $99."},
    ]
    assert citation_accuracy(cites, {"u": SOURCE}) == 0.5
    assert citation_accuracy([], {}) == 0.0


def test_normalize_label():
    assert normalize_label("alpha", ["Alpha", "Beta"]) == "Alpha"
    assert normalize_label("Alpha Inc.", ["Alpha", "Beta"]) == "Alpha"
    assert normalize_label("Gamma", ["Alpha", "Beta"]) == "other"
    assert normalize_label("", ["Alpha"]) == "other"


def test_build_refs_only_numbers_verified_claims():
    ok = Claim(id="a", task_id="t", text="x", source_url="u", supporting_quote="q", status="verified")
    bad = Claim(id="b", task_id="t", text="y", source_url="u", supporting_quote="q", status="rejected")
    assert build_refs([ok, bad]) == {1: "a"}
