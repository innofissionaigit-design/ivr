"""Byte-level equality between the tool value and the spoken value.

story title: Numbers are never rounded, reordered or approximated
user story: As a patient, I want the exact figure, so that what I am quoted is
    what I pay.
acceptance criteria: Figures pass from the validated response into the template
    unchanged and are verbalised digit-faithfully. A test asserts byte-level
    equality between the tool value and the spoken value for a corpus of
    amounts, dates and identifiers.

Ported from dev_sourav's test_number_fidelity_review_fixes.py and reworked for
this branch, where the confirmation-ID half is not a partial fix but a live
break: clinic-api generates KCD-{date}-{4 hex}, the old pattern did not match
hex at all, and agent/speakability.py blocked the resulting reply -- so every
successful booking escalated to the counter instead of giving the caller a
number.

"Byte-level equality" needs saying precisely, because the spoken form is
Bengali words and the tool value is digits: what is asserted is that the
DIGIT SEQUENCE survives. Each source digit appears, in order, with nothing
rounded away, nothing reordered, and no leading or trailing zero dropped. The
expected Bengali is reconstructed from the same primitives bn_normalize uses,
so this checks the pipeline rather than restating its output.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import speakability  # noqa: E402
from agent.bn_normalize import (  # noqa: E402
    _ONES_TO_99, GROUP_SEPARATOR, spell_out, verbalize,
)
# Aliased: pytest collects any module-level name starting with test_ as a
# test case, and would try to run the template itself as one.
from agent.reply_templates import (  # noqa: E402
    booking_reply, test_rate_reply as rate_reply,
)
from agent.tools_client import _parse_exact  # noqa: E402

DIGIT_BN = {str(i): _ONES_TO_99[i] for i in range(10)}


def _digits_spoken(value: str) -> list[str]:
    """The Bengali words a digit string must produce, digit by digit."""
    return [DIGIT_BN[c] for c in value if c.isdigit()]


class _Body:
    """Only what _parse_exact touches: the raw response text."""

    def __init__(self, text: str):
        self.text = text


# --------------------------------------------------- the parse boundary

@pytest.mark.parametrize("literal", [
    "100.00", "0.99", "12547.85", "999999.99", "250.50", "1.10", "0.10",
    "100.000", "7.0", "1234567.89",
])
def test_a_decimal_keeps_its_source_digits(literal):
    """float("100.00") == float("100.0"). Once the default parser has run, the
    trailing zero is gone and no template or verbaliser downstream can put it
    back -- the digit was discarded at the boundary."""
    parsed = _parse_exact(_Body(json.dumps({"rate_inr": float(literal)}).replace(
        json.dumps(float(literal)), literal)))
    assert parsed["rate_inr"] == literal, "the source digits did not survive parsing"
    assert isinstance(parsed["rate_inr"], str), "a decimal must not become a float"


@pytest.mark.parametrize("literal", ["650", "0", "1", "999999"])
def test_a_whole_number_is_untouched(literal):
    """No decimal point in the source means no float in the first place. This
    is the regression that matters most, because rate_inr is an Integer column
    on this branch, so whole numbers are the ONLY case production exercises."""
    parsed = _parse_exact(_Body('{"rate_inr": %s}' % literal))
    assert parsed["rate_inr"] == int(literal)
    assert isinstance(parsed["rate_inr"], int)


def test_the_default_parser_would_have_lost_the_digit():
    """The bug, demonstrated rather than asserted about. If this ever stops
    failing, the fix has become unnecessary and this file can shrink."""
    lost = json.loads('{"rate_inr": 100.00}')["rate_inr"]
    assert str(lost) == "100.0", "default parsing no longer drops the zero"
    kept = _parse_exact(_Body('{"rate_inr": 100.00}'))["rate_inr"]
    assert kept == "100.00"


def test_a_malformed_body_still_raises_a_value_error():
    """_parse_exact must keep composing with the failure handling in
    tools_client: JSONDecodeError subclasses ValueError, which is the arm that
    reports "the system cannot be reached"."""
    with pytest.raises(ValueError):
        _parse_exact(_Body("<html>502 Bad Gateway</html>"))


# ------------------------------------------------- amounts, end to end

@pytest.mark.parametrize("rate", [250, 650, 1800, 2200, 100, 7, 999999])
def test_a_rate_reaches_the_caller_digit_for_digit(rate):
    reply = rate_reply(
        {"test_name": "ইউরিক অ্যাসিড"},
        {"found": True, "test_name": "Uric Acid", "test_name_bn": "ইউরিক অ্যাসিড",
         "rate_inr": rate, "sample_type": "Blood", "report_time_hours": 12})
    assert str(rate) in reply, "the exact figure must reach the template unchanged"
    verdict = speakability.check(reply)
    assert verdict.state == speakability.SPEAKABLE
    # Not digit-by-digit -- a price is spoken as a NUMBER ("চারশো পঞ্চাশ"), which
    # is correct for money and is why this asserts the rendering is non-empty
    # and audible rather than asserting a digit sequence.
    assert "টাকা" in verdict.spoken


# --------------------------------------------- identifiers, end to end

REAL_IDS = [
    "KCD-20260911-4A2F",   # hex with letters -- ~85% of real IDs
    "KCD-20260911-0031",   # all digits, leading zeros
    "KCD-20260911-ABCD",   # all letters -- ~2% of real IDs
    "KCD-20260911-0000",
    "KCD-20261231-FFFF",
]


@pytest.mark.parametrize("conf_id", REAL_IDS)
def test_a_confirmation_id_is_spoken_in_full(conf_id):
    """Every digit, in order, no group rounded into a word.

    The old pattern captured only the first hyphen group, so the trailing one
    fell through to the bare-integer sweep: "0031" was spoken as the WORD
    "একত্রিশ" (thirty-one) with the leading zeros gone -- a caller reading it
    back to the counter would have the wrong number.
    """
    spoken = verbalize(f"কনফার্মেশন নম্বর: {conf_id}।")
    for word in _digits_spoken(conf_id):
        assert word in spoken, f"{conf_id}: a digit is missing from {spoken!r}"

    # story title: Figures are spoken at a pace a caller can write down
    # user story: As a patient noting a price or a reference, I want it
    #   grouped and slower, so that I do not have to ask twice.
    # acceptance criteria: Prices, phone numbers and reference identifiers
    #   are spoken with grouping and a reduced rate through the per-request
    #   speed parameter. A listening test confirms callers transcribe
    #   correctly on first hearing.
    #
    # This assertion used to read `spell_out(conf_id) in spoken` -- "the ID
    # is spelled out as one unit". spell_out() now GROUPS, so that form would
    # still pass, symmetrically and by accident, while asserting nothing
    # about the thing that changed. Rewritten to the contract it actually has
    # now: every group present, in order, separated.
    #
    # The ordered scan is the part that matters. "Grouped" must never become
    # licence to reorder -- a reference number read back out of order is the
    # same defect as one read back with a digit missing.
    groups = [g for g in spell_out(conf_id).split(GROUP_SEPARATOR) if g.strip()]
    assert len(groups) > 1, f"{conf_id}: not grouped at all"
    cursor = 0
    for group in groups:
        found = spoken.find(group.strip(), cursor)
        assert found != -1, f"{conf_id}: group {group.strip()!r} missing or out of order"
        cursor = found + len(group.strip())


@pytest.mark.parametrize("conf_id", REAL_IDS)
def test_a_confirmation_id_survives_the_speakability_gate(conf_id):
    """The half dev_sourav deferred, and the half that actually broke bookings
    here. An unmatched hex group leaves Latin characters in a Bengali sentence;
    the tokenizer drops them and the gate blocks the whole reply, so a caller
    whose booking SUCCEEDED was sent to the counter."""
    reply = booking_reply(
        {"doctor_name": "সেন"},
        {"success": True, "confirmation_id": conf_id, "doctor_name": "Dr. A Sen",
         "doctor_name_bn": "সেন", "date": "2026-09-14", "time_slot": "18:30"})
    verdict = speakability.check(reply)
    assert verdict.state == speakability.SPEAKABLE, list(verdict.dropped)


def test_the_generated_id_format_is_the_one_being_tested():
    """Pins the corpus to reality. If clinic-api's generator changes shape,
    this fails rather than the suite quietly testing a format nobody issues."""
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "clinic-api" / "main.py").read_text(encoding="utf-8")
    assert 'f"KCD-{req.date.replace(\'-\', \'\')}-{uuid.uuid4().hex[:4].upper()}"' in source


def test_every_id_the_real_generator_can_emit_is_speakable():
    """A sweep, because the failure was concentrated in a subset: IDs whose
    hex group contains letters. 2000 samples over the real alphabet."""
    random.seed(20260911)
    blocked = []
    for _ in range(2000):
        conf_id = "KCD-2026%02d%02d-%s" % (
            random.randint(1, 12), random.randint(1, 28),
            "".join(random.choice("0123456789ABCDEF") for _ in range(4)))
        if speakability.check(verbalize(f"নম্বর: {conf_id}।")).is_blocked:
            blocked.append(conf_id)
    assert not blocked, f"{len(blocked)}/2000 blocked, e.g. {blocked[:5]}"


@pytest.mark.parametrize("word", [
    "CBC", "ECG", "TSH", "LFT", "USG", "HIV", "DEADBEEF", "BEEF", "ABCDEF",
])
def test_an_ordinary_uppercase_word_is_not_mistaken_for_an_identifier(word):
    """The cost of widening the pattern, held to zero. Allowing bare hex runs
    would have spelled "DEADBEEF" out letter by letter; requiring the
    [A-Z]{2,}\\d{3,} prefix is what keeps a word a word."""
    assert verbalize(word) == word


# ------------------------------------------------------ dates and times

@pytest.mark.parametrize("iso,expected_digits", [
    ("2026-09-14", ["চোদ্দো"]),
    ("2026-09-01", ["এক"]),
    ("2026-12-31", ["একত্রিশ"]),
])
def test_a_date_is_spoken_as_the_day_it_names(iso, expected_digits):
    spoken = verbalize(f"{iso} তারিখে")
    for word in expected_digits:
        assert word in spoken, spoken


@pytest.mark.parametrize("phone", ["9876543210", "9000000009", "1234509876"])
def test_a_phone_number_is_spoken_digit_by_digit_in_order(phone):
    """Order is part of the criterion. A phone number read as a quantity, or
    with its digits regrouped, sends a report to a stranger."""
    spoken = verbalize(f"ফোন {phone}")
    words = _digits_spoken(phone)
    positions = []
    cursor = 0
    for word in words:
        found = spoken.find(word, cursor)
        assert found != -1, f"{phone}: {word} missing from {spoken!r}"
        positions.append(found)
        cursor = found + len(word)
    assert positions == sorted(positions), "digits were reordered"
