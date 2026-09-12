"""The write happened; the agent cannot say what it did.

story title: The agent says it cannot confirm rather than guessing
user story: As a caller, I want to be told plainly when the system cannot
    verify something, so that I am not given a confident guess.
acceptance criteria: The insufficient-verified-information outcome has its own
    template per language, its own metric and its own escalation path, distinct
    from not-found and from an infrastructure apology. Its rate is reported per
    intent because a rise means a data or integration problem.

Ported from dev_sourav's test_insufficient_verified_information.py and
test_finish_booking_insufficient_information.py, Bengali only, and rebuilt
around this branch's three existing outcomes rather than a parallel set.

The distinctness tests are the load-bearing ones. A fourth outcome that says
something almost like one of the other three is worse than no fourth outcome,
because it adds a code path without adding an answer -- and the three
instructions genuinely differ: stop asking, call back, do not rebook.
"""
from __future__ import annotations

import asyncio
import json

import pathlib
import unicodedata
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from agent import outcomes, speakability, tool_contract, tool_outcome, turn_log  # noqa: E402
from agent.bn_normalize import verbalize  # noqa: E402
from agent.reply_templates import (  # noqa: E402
    INSUFFICIENT_VERIFIED_INFORMATION_BN, booking_reply,
)

COMPLETE = {"success": True, "confirmation_id": "KCD-20260914-4A2F",
            "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
            "date": "2026-09-14", "time_slot": "18:30"}
SLOTS = {"doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন", "date": "2026-09-14",
         "time_slot": "18:30", "patient_name": "রিয়া দাস", "phone": "9876543210"}


def _says(phrase: str, sentence: str) -> bool:
    """Substring test that survives Unicode normalisation.

    Bengali য় exists as one codepoint (U+09DF) and as two (U+09AF U+09BC), and
    this repository contains both -- visually identical, and `in` says False.
    A test that compares Bengali without normalising is testing byte forms
    rather than words, and will fail on a sentence that is perfectly correct.
    """
    return unicodedata.normalize("NFC", phrase) in unicodedata.normalize("NFC", sentence)


# ------------------------------------------------------- the presence check

def test_a_complete_response_has_nothing_missing():
    assert outcomes.missing_booking_write_fields(COMPLETE) == []


@pytest.mark.parametrize("field", outcomes.REQUIRED_BOOKING_WRITE_FIELDS)
def test_each_write_field_is_checked_individually(field):
    assert outcomes.missing_booking_write_fields(
        {k: v for k, v in COMPLETE.items() if k != field}) == [field]


@pytest.mark.parametrize("field", outcomes.REQUIRED_BOOKING_WRITE_FIELDS)
def test_an_empty_value_counts_as_missing(field):
    """The half a presence check cannot do. confirmation_id="" satisfies every
    "is the key there" test and would then be read aloud as a confirmation
    number consisting of nothing."""
    assert outcomes.missing_booking_write_fields(dict(COMPLETE, **{field: ""})) == [field]


def test_several_missing_fields_are_all_named():
    """The list is the diagnostic. Which part of the write failed to come back
    is what tells a data problem from an integration one."""
    assert outcomes.missing_booking_write_fields({"success": True}) == list(
        outcomes.REQUIRED_BOOKING_WRITE_FIELDS)


# -------------------------------------------- distinct from the other three

def test_the_sentence_differs_from_both_neighbours():
    """Three outcomes, three different instructions to the caller:
        not-found    -> the thing does not exist. Stop asking.
        unreachable  -> could not check. Call back.
        this         -> it IS done. Do not rebook."""
    not_found = booking_reply({"doctor_name": "ঘোষ"},
                              {"success": False, "reason": "doctor_not_found"})
    assert INSUFFICIENT_VERIFIED_INFORMATION_BN != main.SYSTEM_UNREACHABLE_BN
    assert INSUFFICIENT_VERIFIED_INFORMATION_BN != not_found


def test_it_never_tells_the_caller_to_call_back():
    """The specific harm this outcome exists to prevent. The appointment was
    created; "call back" invites a duplicate of a booking that already exists,
    which is why borrowing the unreachable framing here would be worse than
    saying nothing."""
    assert not _says("কাউন্টারে যোগাযোগ করুন", INSUFFICIENT_VERIFIED_INFORMATION_BN)


def test_it_says_the_booking_happened():
    """It must not read as a failure. The row exists; the caller should not
    walk away believing nothing was booked."""
    assert _says("হয়ে গেছে", INSUFFICIENT_VERIFIED_INFORMATION_BN)
    assert _says("দরকার নেই", INSUFFICIENT_VERIFIED_INFORMATION_BN), (
        "the reply must tell the caller there is no need to rebook")


def test_the_sentence_is_sayable():
    verdict = speakability.check(INSUFFICIENT_VERIFIED_INFORMATION_BN)
    assert verdict.state == speakability.SPEAKABLE
    assert ":" not in verbalize(INSUFFICIENT_VERIFIED_INFORMATION_BN)


# ------------------------------------------------------ the contract split

def test_the_write_fields_left_the_contract():
    """REGRESSION. While confirmation_id/date/time_slot sat in the booking
    contract, an absent one raised ToolContractError -> ToolCallError -> the
    infrastructure apology: a caller whose booking SUCCEEDED was told to call
    back. Putting them back collapses this outcome into that one."""
    _, when_true, _ = tool_contract._CONTRACTS["book_appointment"]
    for field in outcomes.REQUIRED_BOOKING_WRITE_FIELDS:
        assert field not in when_true, (
            f"{field} is back in the booking contract -- an absent one would be "
            f"reported as the clinic being unreachable"
        )


def test_the_contract_still_guards_what_the_template_needs():
    """The line between the two modules: the contract keeps the fields a reply
    needs to RENDER, this story owns the ones that confirm the write."""
    _, when_true, _ = tool_contract._CONTRACTS["book_appointment"]
    assert "doctor_name_bn" in when_true


def test_a_response_missing_only_write_fields_passes_the_contract():
    """It must reach _finish_booking's check rather than being refused at the
    boundary -- otherwise the fourth outcome is unreachable."""
    tool_contract.validate("book_appointment", {
        "success": True, "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন"})


# ------------------------------------------------- the metric and the record

@pytest.fixture
def wired(monkeypatch, tmp_path):
    said: list[str] = []

    async def _say(session, text, fallback_reason=None):
        said.append(text)

    class _Tools:
        def __init__(self):
            self.outcomes = tool_outcome.OutcomeCounter()
            self.result = COMPLETE

        async def book_appointment(self, *a):
            return self.result

    tools = _Tools()
    monkeypatch.setattr(main, "_speak", _say)
    monkeypatch.setattr(main, "_tools", tools)
    monkeypatch.setattr(turn_log, "TURN_LOG_PATH", str(tmp_path / "turn.jsonl"))
    return tools, said, tmp_path / "turn.jsonl"


class _Session:
    call_id = "iv01"
    utt_seq = 3

    def __init__(self):
        self.call_state = main.call_state_mod.build()
        self.pending = None


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_a_complete_write_is_read_back_exactly_as_before(wired):
    """Regression-safety: the guard must be invisible on every healthy call."""
    tools, said, log = wired
    asyncio.run(main._finish_booking(_Session(), SLOTS, confirmed=True))

    assert said == [booking_reply(SLOTS, COMPLETE)]
    assert tools.outcomes.snapshot()["book_appointment"][tool_outcome.INSUFFICIENT] == 0
    assert _rows(log) == []


@pytest.mark.parametrize("field", outcomes.REQUIRED_BOOKING_WRITE_FIELDS)
def test_an_unverifiable_write_says_so_instead_of_guessing(wired, field):
    tools, said, log = wired
    tools.result = {k: v for k, v in COMPLETE.items() if k != field}

    asyncio.run(main._finish_booking(_Session(), SLOTS, confirmed=True))

    assert said == [INSUFFICIENT_VERIFIED_INFORMATION_BN]
    assert main.SYSTEM_UNREACHABLE_BN not in said, "not the infrastructure apology"
    counts = tools.outcomes.snapshot()["book_appointment"]
    assert counts[tool_outcome.INSUFFICIENT] == 1
    assert counts[tool_outcome.UNREACHABLE] == 0, "must not be counted as an outage"

    (row,) = [r for r in _rows(log) if r["event"] == turn_log.EVENT_INSUFFICIENT]
    assert row["missing"] == [field]
    assert row["intent"] == "book_appointment"
    assert (row["call_id"], row["turn"]) == ("iv01", 3)


def test_no_confirmation_number_is_ever_fabricated(wired):
    """The behaviour the story is named for. An empty confirmation_id used to
    be read aloud as if it were one."""
    tools, said, _ = wired
    tools.result = dict(COMPLETE, confirmation_id="")

    asyncio.run(main._finish_booking(_Session(), SLOTS, confirmed=True))

    assert said == [INSUFFICIENT_VERIFIED_INFORMATION_BN]
    assert not _says("কনফার্মেশন নম্বর", said[0]), "no confirmation number is claimed"


def test_the_escalation_record_carries_no_caller_data(wired):
    """`missing` holds field NAMES, never field values, so turn_log's PHI rule
    holds here without an exception."""
    tools, _, log = wired
    tools.result = dict(COMPLETE, confirmation_id="", time_slot="")

    asyncio.run(main._finish_booking(_Session(), SLOTS, confirmed=True))

    (row,) = [r for r in _rows(log) if r["event"] == turn_log.EVENT_INSUFFICIENT]
    assert set(row["missing"]) == {"confirmation_id", "time_slot"}
    blob = json.dumps(row, ensure_ascii=False)
    for value in (SLOTS["patient_name"], SLOTS["phone"]):
        assert value not in blob


def test_the_fourth_outcome_stays_out_of_the_not_found_rate():
    """Folding it into the denominator would let a data-integrity incident
    look like the catalogue improving."""
    counter = tool_outcome.OutcomeCounter()
    counter.record("book_appointment", tool_outcome.NOT_FOUND)
    counter.record("book_appointment", tool_outcome.ANSWERED)
    for _ in range(8):
        counter.record("book_appointment", tool_outcome.INSUFFICIENT)
    assert counter.snapshot()["book_appointment"]["not_found_rate"] == 0.5
