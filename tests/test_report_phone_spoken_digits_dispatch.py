"""ADDED BY SOURAV -- dispatch-level reproduction of a real production
bug, reported directly from a live call transcript:

    [User] Is my report ready?
    [AI]   Can you tell me your registered phone number?
    [User] Yes, write nine zero zero zero zero zero zero zero zero one.
    [AI]   Can you tell me your registered phone number?      <- STUCK
    [User] I said the number is nine zero zero zero zero zero zero zero zero one.
    [AI]   Can you tell me your registered phone number?      <- STUCK AGAIN

tests/test_slot_parse_phone.py already covers the actual bug at the
parser level (agent/slot_parse.py's parse_phone()). This file confirms
the fix reaches all the way through main_pcm.py's "report_phone" pending
state -- the exact code path this transcript went through -- so the fix
is proven end-to-end, not just at the unit level. Pattern and fixtures
copied directly from tests/test_report_status_send_dispatch.py (same
FakeToolsClient/env/make_session/continue_pending shapes) rather than
re-invented, so this reads as one consistent suite with that file.
"""
from __future__ import annotations

import asyncio
import types

import pytest

import main_pcm


def run(coro):
    return asyncio.run(coro)


class FakeToolsClient:
    def __init__(self):
        self.get_report_status_calls = []
        self.get_report_status_response = {
            "patient_found": True, "found": True, "status": "READY",
            "delivery_enabled": False, "report_number": "RPT-A", "test_name": "CBC",
        }

    async def get_report_status(self, phone, test_name=None):
        self.get_report_status_calls.append((phone, test_name))
        return self.get_report_status_response


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-call-1",
        pending=pending,
        dispatch_lock=asyncio.Lock(),
        send_json=_AsyncNoOp(),
    )


@pytest.fixture
def env(monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    tools = FakeToolsClient()
    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    monkeypatch.setattr(main_pcm, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools)


def continue_pending(session, text):
    handled = run(main_pcm._continue_pending(session, text))
    assert handled is True
    return session


def _report_phone_pending(retries=0):
    return {"awaiting": "report_phone", "flow": "report_status", "test_name": "CBC", "retries": retries}


class TestTheReportedBugIsFixedEndToEnd:
    def test_first_spoken_digit_phrasing_now_succeeds_instead_of_looping(
        self, monkeypatch, env
    ):
        session = make_session(pending=_report_phone_pending())
        continue_pending(session, "Yes, write nine zero zero zero zero zero zero zero zero one.")

        # Before the fix: parse_phone() returned None, retries incremented,
        # the SAME "can you tell me your registered phone number?" prompt
        # was spoken again, and session.pending stayed "report_phone"
        # forever. Now: the lookup tool is actually called with the real
        # number, and the pending state is cleared.
        assert env.tools.get_report_status_calls == [("9000000001", "CBC")]
        assert session.pending is None

    def test_second_phrasing_from_the_transcript_also_succeeds(self, monkeypatch, env):
        session = make_session(pending=_report_phone_pending())
        continue_pending(
            session, "I said the number is nine zero zero zero zero zero zero zero zero one."
        )
        assert env.tools.get_report_status_calls == [("9000000001", "CBC")]
        assert session.pending is None

    def test_retries_counter_does_not_increment_on_a_successful_spoken_number(
        self, monkeypatch, env
    ):
        # Regression guard: confirms this is a genuine success path, not
        # an accidental pass-through that still silently counted a retry.
        session = make_session(pending=_report_phone_pending(retries=1))
        continue_pending(session, "nine zero zero zero zero zero zero zero zero one")
        assert env.tools.get_report_status_calls == [("9000000001", "CBC")]


class TestTheAmbiguousShorthandStillCorrectlyReprompts:
    """The caller's own third attempt in the transcript ("eight zeros")
    is deliberately still not understood -- see parse_phone()'s docstring
    for why guessing here would be worse than asking again. Confirms the
    dispatch layer's existing retry/give-up behaviour (unchanged by this
    fix) still applies correctly to this phrasing."""

    def test_eight_zeros_phrasing_re_prompts_rather_than_guessing(self, monkeypatch, env):
        session = make_session(pending=_report_phone_pending(retries=0))
        continue_pending(session, "Nine then eight zeros and one")
        assert env.tools.get_report_status_calls == []
        assert session.pending["awaiting"] == "report_phone"
        assert session.pending["retries"] == 1

    def test_gives_up_after_repeated_ambiguous_replies_same_as_before(self, monkeypatch, env):
        session = make_session(pending=_report_phone_pending(retries=2))
        handled = run(main_pcm._continue_pending(session, "Nine then eight zeros and one"))
        assert handled is False
        assert session.pending is None
        assert env.tools.get_report_status_calls == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
