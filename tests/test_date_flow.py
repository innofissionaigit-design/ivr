"""The whole date path, end to end, as a conversation.

story title: The model never originates a fact
user story: As a clinical lead, I want every price, date and identifier to come
    from a verified system response, so that a wrong answer is a data bug rather
    than a model bug.
acceptance criteria: Every factual sentence is a template substitution from a
    validated tool response and the model is never shown a figure it could
    restate. An automated assertion on every commit proves no model-composed
    span reaches synthesis on a factual intent.

The unit tests in test_fact_provenance.py check each component in isolation.
This file checks the thing that actually matters: that a caller saying "আগামী
সপ্তাহে" is asked about the right seven days and then answered about the right
one, with every digit on that journey produced by code.

Nothing here touches a model, a GPU or the network -- the LLM's contribution is
supplied directly as the expression it would have returned, which is the point:
its entire contribution to a date is now one string from a closed vocabulary.
"""
from __future__ import annotations

import asyncio
import datetime
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from agent import date_calc  # noqa: E402
from agent.bn_normalize import verbalize  # noqa: E402

THURSDAY = datetime.date(2026, 9, 10)


class _SpyTools:
    """Records what the clinic API was asked, and answers plausibly."""

    def __init__(self):
        self.availability_calls: list[tuple[str, str]] = []

    async def get_doctor_availability(self, doctor_name, date):
        self.availability_calls.append((doctor_name, date))
        return {"found": True, "doctor_name": "Dr. A Sen", "doctor_name_bn": "সেন",
                "date": date, "available": True, "chamber_hours": "18:00-20:00",
                "next_available_date": None}


class _Session:
    call_id = "d4te"
    utt_seq = 1

    def __init__(self):
        self.call_state = main.call_state_mod.build()
        self.pending = None
        self.said: list[str] = []
        # story title: The same question gets the same answer within one call
        # user story: As a caller who asks twice, I want the same answer, so
        #   that I know which one to believe.
        # acceptance criteria: Repeating a question in one call produces an
        #   identical factual answer unless the underlying data changed, in
        #   which case the change is stated. A test asserts consistency
        #   across three repeats with an unchanged backend.
        #
        # A real CallSession carries one, and every factual reply now passes
        # through it. Given to the double rather than defended against in
        # main.py: a session without a ledger is not a state that exists, and
        # a getattr fallback would make forgetting to create one survivable.
        self.answer_ledger = main.answer_ledger.AnswerLedger()


async def _say(session, text, fallback_reason=None):
    session.said.append(text)


@pytest.fixture
def wired(monkeypatch):
    tools = _SpyTools()
    monkeypatch.setattr(main, "_tools", tools)
    monkeypatch.setattr(main, "_speak", _say)
    return tools


def test_next_week_is_confirmed_then_answered(wired, monkeypatch):
    """The scenario the architecture was designed around.

    Caller: "আগামী সপ্তাহে ডাক্তার সেন আছেন?"
      model  -> "next_week"                     (meaning only)
      code   -> 2026-09-14 .. 2026-09-20        (every digit)
      agent  -> "did you mean the 14th to the 20th?"
    Caller: "হ্যাঁ"
      agent  -> asks clinic-api about 2026-09-14
    """
    span = date_calc.resolve("আগামী সপ্তাহে ডাক্তার সেন আছেন?",
                             date_expr="next_week", today=THURSDAY)
    assert (span.start, span.end) == ("2026-09-14", "2026-09-20")
    assert span.needs_confirmation

    session = _Session()
    session.pending = {
        "awaiting": "confirm_date",
        "slots": {"doctor_name": "Dr. A Sen"},
        "candidates": None, "offered_date": span.start, "retries": 0,
        "resume_intent": "doctor_availability", "span_end": span.end,
    }

    handled = asyncio.run(main._continue_pending(session, "হ্যাঁ"))

    assert handled
    assert wired.availability_calls == [("Dr. A Sen", "2026-09-14")], (
        "the confirmed range start is what reaches the clinic API"
    )
    assert session.said, "the caller got an answer"


def test_saying_no_to_the_range_asks_for_one_day(wired):
    """A rejected range must not re-ask the same question -- the caller already
    answered it. It narrows to "which day?" instead."""
    session = _Session()
    session.pending = {
        "awaiting": "confirm_date",
        "slots": {"doctor_name": "Dr. A Sen"},
        "candidates": None, "offered_date": "2026-09-14", "retries": 0,
        "resume_intent": "doctor_availability", "span_end": "2026-09-20",
    }

    asyncio.run(main._continue_pending(session, "অন্য দিন"))

    assert wired.availability_calls == [], "nothing was looked up on a rejection"
    assert session.pending["awaiting"] == "availability_date"
    assert session.said == ["কোন দিনের কথা বলছেন, একটু বলবেন?"]


def test_the_confirmation_question_speaks_real_dates(wired):
    """The dates in the question are audible Bengali, not Latin digits the
    tokenizer would drop -- so the caller can actually check them."""
    from agent import speakability
    from agent.reply_templates import date_range_confirm_prompt

    prompt = date_range_confirm_prompt("2026-09-14", "2026-09-20")
    assert speakability.check(prompt).state == speakability.SPEAKABLE
    assert "চোদ্দো" in verbalize(prompt)


def test_a_single_day_skips_the_confirmation_entirely(wired):
    """"আগামীকাল" is one day. Asking about it would be noise, and a
    confirmation that fires on everything stops being a confirmation."""
    span = date_calc.resolve("আগামীকাল ডাক্তার সেন আছেন?",
                             date_expr="tomorrow", today=THURSDAY)
    assert not span.needs_confirmation
    assert span.start == span.end == "2026-09-11"
    assert span.source == date_calc.SOURCE_PARSED, (
        "the caller's own word outranks the model's expression"
    )
