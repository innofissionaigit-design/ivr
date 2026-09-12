"""The readback, and the correction path a rejection opens.

story title: Every critical value is read back before it is used
user story: As a patient giving a phone number, I want it read back, so that a
    misheard digit does not send my report to a stranger.
acceptance criteria: Phone numbers, dates, times and names are confirmed aloud
    before any write, and a rejection opens a correction path rather than
    repeating the prompt. Readback is mandatory regardless of confidence for
    values that affect a write.

Ported from dev_sourav's test_booking_readback.py, Bengali only, and narrowed
to what this branch did not already have. The readback itself arrived with the
confidence-gate story and is already pinned by tests/test_fact_provenance.py
(all five values present, exactly one write site, and it passes confirmed=True).

What was MISSING here, and what most of this file is about, is the second
clause. A rejection used to re-ask "just say yes or no" twice and then abandon
the booking -- which is the literal behaviour the criterion names as wrong, and
a poor exchange besides: the caller reports an error and the agent responds by
repeating itself, then hanging up on the booking.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from agent.reply_templates import (  # noqa: E402
    booking_confirm_prompt, booking_correction_prompt, missing_slot_prompt,
)
from agent.slot_parse import parse_correction_field  # noqa: E402

FULL = {"doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
        "date": "2026-09-14", "time_slot": "18:30",
        "patient_name": "রিয়া দাস", "phone": "9876543210"}


class _Session:
    call_id = "rb01"
    utt_seq = 1

    def __init__(self, awaiting="confirm_booking", slots=None):
        self.call_state = main.call_state_mod.build()
        self.said: list[str] = []
        self.pending = {"awaiting": awaiting, "slots": dict(slots or FULL),
                        "candidates": None, "offered_date": None, "retries": 0}


async def _say(session, text, fallback_reason=None):
    session.said.append(text)


@pytest.fixture
def wired(monkeypatch):
    """No TTS, no clinic API. _finish_booking is replaced by a spy so a write
    is observable without one ever happening."""
    written: list[dict] = []

    async def _finish(session, slots, *, confirmed=False):
        written.append({"slots": dict(slots), "confirmed": confirmed})
        session.pending = None

    monkeypatch.setattr(main, "_speak", _say)
    monkeypatch.setattr(main, "_finish_booking", _finish)
    return written


def _turn(session, text):
    return asyncio.run(main._continue_pending(session, text))


# ------------------------------------------------------- the readback

def test_all_five_values_are_read_back_before_any_write():
    """Phone, date, time and name, aloud, in one sentence. The doctor too."""
    prompt = booking_confirm_prompt(FULL)
    for value in ("রিয়া দাস", "সেন", "2026-09-14", "18:30", "9876543210"):
        assert value in prompt, f"{value!r} is not read back"


def test_an_affirmative_is_what_writes(wired):
    session = _Session()
    assert _turn(session, "হ্যাঁ")
    assert len(wired) == 1 and wired[0]["confirmed"] is True
    assert wired[0]["slots"]["phone"] == "9876543210"


@pytest.mark.parametrize("reply", ["", "ইয়ে", "একটু দাঁড়ান", "ডাক্তার সেন", "1815"])
def test_an_ambiguous_reply_is_not_consent(wired, reply):
    """Silence, a restatement, a half-heard grunt. None of these are a yes, and
    treating one as a yes is the guess-becomes-a-booking failure the whole
    state exists to prevent."""
    session = _Session()
    _turn(session, reply)
    assert wired == [], f"{reply!r} was treated as consent"


# ------------------------------------------------- the correction path

def test_a_rejection_opens_a_correction_path_instead_of_repeating(wired):
    """The clause this branch failed. "না" must ask WHICH value is wrong --
    a different question -- not re-ask the same yes/no."""
    session = _Session()
    assert _turn(session, "না")

    assert wired == [], "a rejection must never write"
    assert session.said == [booking_correction_prompt()]
    assert session.said[0] != booking_confirm_prompt(FULL), "the prompt was repeated"
    assert session.pending["awaiting"] == "confirm_correction"


def test_a_rejection_does_not_discard_the_four_correct_values(wired):
    """A correction is not a restart. The caller said one value was wrong, so
    four are right and re-collecting them would be its own small insult."""
    session = _Session()
    _turn(session, "না")
    assert session.pending["slots"] == FULL


@pytest.mark.parametrize("said,field", [
    ("ডাক্তারের নাম ভুল", "doctor_name"),
    ("তারিখটা ভুল", "date"),
    ("সময় ঠিক নেই", "time_slot"),
    ("ফোন নম্বরটা", "phone"),
    ("রোগীর নাম", "patient_name"),
])
def test_the_named_field_is_the_one_re_collected(wired, said, field):
    session = _Session(awaiting="confirm_correction")
    assert _turn(session, said)
    assert session.pending["awaiting"] == field
    assert session.said == [missing_slot_prompt("book_appointment", field)]
    assert wired == []


def test_a_bare_name_means_the_patient_not_the_doctor():
    """Order inside parse_correction_field, pinned. "নাম" is a substring of
    how a caller says "the doctor's name", so doctor_name is matched first on
    its own word and a bare "নাম" can only fall through to patient_name."""
    assert parse_correction_field("নাম") == "patient_name"
    assert parse_correction_field("ডাক্তারের নাম") == "doctor_name"


def test_an_unnameable_field_re_asks_rather_than_guessing(wired):
    """Guessing here would re-collect the wrong value and then read the SAME
    wrong one back -- worse than asking twice."""
    session = _Session(awaiting="confirm_correction")
    _turn(session, "জানি না")
    assert session.pending["awaiting"] == "confirm_correction"
    assert session.said == [booking_correction_prompt()]


def test_the_correction_loop_is_capped(wired):
    session = _Session(awaiting="confirm_correction")
    for _ in range(4):
        _turn(session, "জানি না")
    assert session.pending is None, "the correction loop never ended"
    assert session.said[-1] == main.BOOKING_NOT_CONFIRMED_BN
    assert wired == [], "nothing may be written after giving up"


def test_saying_no_to_the_correction_question_abandons(wired):
    """The considered divergence from dev_sourav, which excludes this state
    from the abandon hatch too. Once the agent has listed the five options, a
    caller saying "না" is not naming a field -- they have given up."""
    session = _Session(awaiting="confirm_correction")
    _turn(session, "না")
    assert session.pending is None
    assert wired == []


# ------------------------------------------------ the round trip

def test_a_corrected_booking_is_read_back_in_full_again(wired):
    """A correction must never shorten the path to the write. After the one
    field is re-collected, all five are read back and an affirmative is still
    required."""
    session = _Session()

    _turn(session, "না")                       # readback rejected
    _turn(session, "ফোন নম্বরটা ভুল")           # names the field
    _turn(session, "৯১২৩৪৫৬৭৮৯")               # gives the new value

    assert wired == [], "nothing written yet"
    assert session.pending["awaiting"] == "confirm_booking"
    assert session.pending["slots"]["phone"] == "9123456789", "the new value"
    assert session.pending["slots"]["date"] == FULL["date"], "the others are intact"
    assert session.said[-1] == booking_confirm_prompt(session.pending["slots"])

    _turn(session, "হ্যাঁ")
    assert len(wired) == 1 and wired[0]["slots"]["phone"] == "9123456789"


def test_the_correction_prompt_is_sayable():
    """It is spoken on a phone line, so it has to survive the tokenizer."""
    from agent import speakability
    assert speakability.check(booking_correction_prompt()).state == speakability.SPEAKABLE
