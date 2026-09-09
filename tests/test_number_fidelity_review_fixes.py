"""Regression tests added during independent review of the "Numbers are
never rounded, reordered or approximated" story.

These reproduce two real, end-to-end digit-fidelity bugs found in the
originally submitted implementation and assert they stay fixed:

1. Decimal amounts whose fraction ends in a zero (e.g. "100.00") lost that
   trailing zero on the way from the tool's JSON response into the reply
   template, because `httpx`'s default `r.json()` parses JSON numbers as
   Python `float`, and `float(100.00) == float(100.0)` -- the digit is
   gone before verbalize() ever sees the text.
   Fix: `agent/tools_client.py::_parse_exact` parses response bodies with
   `parse_float=str`, so a decimal number's exact source digits survive
   as a string instead of being coerced through `float`.

2. Confirmation IDs in the tool contract's own documented shape
   ("KCD-20260824-0031", i.e. more than one hyphen-separated digit group)
   only had their FIRST digit group recognized by the confirmation-id
   regex. The remaining group fell through to the generic integer sweep
   and was spoken as a rounded number word with leading zeros dropped
   ("0031" -> "thirty-one" instead of "zero zero three one").
   Fix: `agent/bn_normalize.py::_RE_CONF_ID` now matches every
   hyphen-separated digit group in the ID, not just the first.

Two follow-up gaps closed in this revision (previously flagged as
"required before PR" in the acceptance review):

A. The decimal-amount corpus below used to assert only the pre-TTS
   template string, never the actual `spoken` (verbalized) output --
   the acceptance criteria's requirement, "figures ... are verbalised
   digit-faithfully", was therefore unproven at the layer that matters.
   `TestDecimalAmountCorpus.test_decimal_amount_verbalized_digit_faithfully`
   now asserts against the final `spoken` string.

B. The 20,000-sample currency-parser sweep that validated the systemic
   fix during review was run interactively and never committed, so it
   was not reproducible by anyone re-running this suite.
   `TestCurrencyParserPrecisionRegression` is the committed, seeded
   replacement (see its docstring for the sample-size rationale).

Run: python -m pytest tests/test_number_fidelity_review_fixes.py -v
"""
import random

import pytest

from agent.reply_templates import test_rate_reply as rate_reply, booking_reply
from agent.bn_normalize import verbalize, number_to_bn_words, digits_one_by_one
from agent.tools_client import _parse_exact


class _FakeResponse:
    """Minimal stand-in for httpx.Response -- _parse_exact only reads .text."""
    def __init__(self, text: str):
        self.text = text


def full_pipeline_rate(rate_inr_json_literal: str) -> tuple[str, str]:
    """Simulate the real production path end to end: clinic-api's JSON body
    -> tools_client._parse_exact (was: httpx's r.json()) -> reply_templates
    -> bn_normalize.verbalize() (what TTS actually receives)."""
    body = ('{"found": true, "test_name": "Test", "test_name_bn": "টেস্ট", '
            f'"rate_inr": {rate_inr_json_literal}}}')
    result = _parse_exact(_FakeResponse(body))
    reply = rate_reply({"test_name": "Test"}, result)
    spoken = verbalize(reply)
    return reply, spoken


def expected_decimal_words(amount: str) -> str:
    """Independently reconstruct what the spoken form of a decimal amount
    MUST contain, using the repository's own verbalization primitives --
    the exact same ones `agent/bn_normalize.py`'s `_RE_DECIMAL` substitution
    uses (`number_to_bn_words` for the integer part, `digits_one_by_one`
    for the fractional part, joined by " দশমিক ").

    Building the expectation from these primitives (rather than hand-typing
    Bengali text) is what makes the assertion strong instead of vacuous:
    - Rounding/approximation (e.g. 999.99 -> "1000"): the integer-part
      words would no longer match ("নয়শো ঊননব্বই" vs "এক হাজার").
    - Truncation/dropped digits (e.g. 12547.85 -> 12547.8): the fractional
      part's word COUNT changes (one digit-word missing), so the exact
      expected phrase is no longer a substring of the actual output.
    - Trailing-zero loss (e.g. 100.00 -> 100.0): digits_one_by_one("00")
      produces TWO digit-words ("শূন্য শূন্য"); digits_one_by_one("0")
      produces only one. These are different strings and one is not a
      substring-match for the other in context, so the assertion fails.
    - Reordered digits: digits_one_by_one preserves source order
      character-by-character, so any reordering changes the expected
      string outright.
    """
    int_part, frac_part = amount.split(".")
    return f"{number_to_bn_words(int(int_part))} দশমিক {digits_one_by_one(frac_part)}"


DECIMAL_AMOUNT_CORPUS = [
    "12547.85", "0.99", "999999.99", "100.00", "500.00", "10.00", "7.00",
    "2000.05", "0.10", "1000.50",
]


class TestDecimalAmountCorpus:
    """Acceptance criteria's required corpus: 12547.85, 100.00, 0.99,
    999999.99, and values containing zeros."""

    @pytest.mark.parametrize("amount", DECIMAL_AMOUNT_CORPUS)
    def test_decimal_amount_survives_into_template_unchanged(self, amount):
        reply, spoken = full_pipeline_rate(amount)
        assert amount in reply, (
            f"tool value {amount!r} did not survive unchanged into the "
            f"reply template: {reply!r}"
        )

    @pytest.mark.parametrize("amount", DECIMAL_AMOUNT_CORPUS)
    def test_decimal_amount_verbalized_digit_faithfully(self, amount):
        """The FINAL spoken (TTS-input) string, not just the template,
        must contain the digit-faithful verbalization of the exact tool
        value -- no rounding, truncation, reordering, approximation, or
        dropped trailing zeros. See `expected_decimal_words()` for exactly
        what class of regression each part of this catches."""
        reply, spoken = full_pipeline_rate(amount)
        expected = expected_decimal_words(amount)
        assert expected in spoken, (
            f"tool value {amount!r} was not verbalized digit-faithfully.\n"
            f"  expected substring : {expected!r}\n"
            f"  actual spoken text : {spoken!r}"
        )

    def test_whole_number_rate_unaffected(self):
        """Whole-rupee rates (the common case -- no decimal point in the
        source) must keep behaving exactly as before: no spurious '.0'."""
        reply, spoken = full_pipeline_rate("650")
        assert "650" in reply
        assert "650.0" not in reply


class TestConfirmationIdCorpus:
    """The tool contract's own documented shape:
    'KCD-20260824-0031' (two hyphen-separated digit groups)."""

    def test_realistic_confirmation_id_spoken_digit_by_digit(self):
        result = {
            "success": True,
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "18:30",
            "confirmation_id": "KCD-20260824-0031",
        }
        reply = booking_reply({}, result)
        spoken = verbalize(reply)

        # The corruption this used to produce: the trailing group read as
        # a rounded number word with its leading zeros dropped.
        assert "একত্রিশ" not in spoken, (
            f"confirmation id group '0031' was spoken as a rounded number "
            f"word instead of digit-by-digit: {spoken!r}"
        )
        # Every digit of both groups must appear, in order, spoken
        # individually -- "zero zero three one" for the trailing "0031".
        assert "শূন্য শূন্য তিন এক" in spoken, (
            f"expected '0031' spoken digit-by-digit, got: {spoken!r}"
        )

    def test_short_confirmation_id_still_works(self):
        """Existing single-group IDs (e.g. 'KCD-4471') must keep working
        exactly as before -- this fix must not narrow what already matched."""
        result = {
            "success": True,
            "doctor_name_bn": "সেন",
            "date": "2026-08-24",
            "time_slot": "18:30",
            "confirmation_id": "KCD-4471",
        }
        reply = booking_reply({}, result)
        spoken = verbalize(reply)
        assert "কে সি ডি চার চার সাত এক" in spoken


def _random_currency_samples(n: int, seed: int) -> list[str]:
    """`n` deterministic, seeded two-decimal currency strings up to
    999999.99 paise-precision, e.g. "369138.10". Seeded so the exact same
    corpus is generated on every run, on every machine, forever."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        paise = rng.randint(0, 99_999_999)  # 0.00 .. 999999.99
        out.append(f"{paise // 100}.{paise % 100:02d}")
    return out


# Sample size rationale:
#
# The bug this guards against (JSON decimal -> Python float -> str()
# silently dropping a trailing zero) fires whenever the fractional
# (paise) part, taken mod 10, is 0 -- i.e. for ~10% of uniformly
# distributed two-decimal values. 2,000 seeded samples therefore contain
# roughly 200 trailing-zero cases and roughly 20 double-trailing-zero
# cases (".X0" and ".00" shapes both represented many times over), which
# is overwhelming statistical coverage for a deterministic bug class --
# if the fix regresses, this WILL catch it, not "might".
#
# An interactive, non-seeded 20,000-sample sweep was run once during
# review (0/20,000 mismatches post-fix, 2,130/20,000 i.e. 10.65%
# pre-fix); 2,000 seeded samples here is the reproducible, committed
# substitute for that one-off run -- not a reduction in what's checked,
# just a bound on suite runtime. Measured: this whole file (2,000 random
# + 25 fixed-edge-case parametrized cases, each its own pytest node with
# fixture/collection overhead) adds ~1.5s to `pytest tests/`, which is
# collection/reporting overhead, not the underlying check itself (the
# actual parse-and-compare loop is sub-millisecond per sample). A full
# 20,000-sample run would scale that to roughly 15s -- still tolerable,
# but 2,000 was chosen to keep this file from dominating the suite's
# runtime while still giving ~200 trailing-zero and ~20
# double-trailing-zero cases of statistical coverage per run.
# The FIXED_EDGE_CASES below additionally guarantee the specific shapes
# (all-zero fraction, single trailing zero, minimum/maximum magnitude)
# are always present regardless of what the RNG happens to draw.
RANDOM_CURRENCY_SAMPLES = _random_currency_samples(2000, seed=20260908)

FIXED_EDGE_CASES = [
    "0.00", "0.01", "0.10", "0.99",
    "1.00", "1.10", "9.00", "9.90",
    "10.00", "10.10", "99.00", "99.99",
    "100.00", "100.10", "500.00", "999.00",
    "1000.00", "1000.50", "9999.00",
    "12547.85", "99999.90", "100000.00",
    "999999.00", "999999.90", "999999.99",
]


class TestCurrencyParserPrecisionRegression:
    """Committed, seeded replacement for the ad-hoc 20,000-sample sweep run
    during review. Exercises the REAL `_parse_exact` implementation (no
    mocking) against a JSON body shaped exactly like clinic-api's actual
    response, and checks two independent things for every sample:

    1. The parsed value's string form is byte-identical to the source
       decimal literal (catches trailing-zero loss, rounding, truncation).
    2. The parsed value is NOT a `float` (catches a regression back to
       plain `r.json()`/`json.loads()` even in the rare case some other
       change made the string comparison above coincidentally pass).
    """

    @pytest.mark.parametrize("amount", FIXED_EDGE_CASES)
    def test_fixed_edge_cases_preserve_exact_precision(self, amount):
        result = _parse_exact(_FakeResponse('{"rate_inr": ' + amount + '}'))
        assert str(result["rate_inr"]) == amount, (
            f"{amount!r} -> {result['rate_inr']!r}: source precision lost"
        )
        assert not isinstance(result["rate_inr"], float), (
            f"{amount!r} was parsed as float -- this is exactly the bug "
            f"class this test exists to catch (trailing zeros/precision "
            f"are not guaranteed to survive a float round-trip)"
        )

    @pytest.mark.parametrize("amount", RANDOM_CURRENCY_SAMPLES)
    def test_random_currency_samples_preserve_exact_precision(self, amount):
        result = _parse_exact(_FakeResponse('{"rate_inr": ' + amount + '}'))
        assert str(result["rate_inr"]) == amount, (
            f"{amount!r} -> {result['rate_inr']!r}: source precision lost"
        )
        assert not isinstance(result["rate_inr"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
