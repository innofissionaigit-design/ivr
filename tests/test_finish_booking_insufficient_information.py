"""Integration tests for the ONE real trigger "The agent says it cannot
confirm rather than guessing" wires up: main_pcm._finish_booking()
checking confirmation_id/date/time_slot are non-empty before reading a
booking confirmation aloud.

Exercises the actual production code path (main_pcm._finish_booking),
not agent/outcomes.py in isolation -- that's tests/test_insufficient_
verified_information.py. Same deterministic-stub style as
tests/test_booking_readback.py: main_pcm._speak and main_pcm._tools are
stubbed, nothing here needs the real ASR/TTS/NeMo stack, no GPU, no
network.
"""
from __future__ import annotations

import asyncio
import json
import types

import pytest

import agent.outcomes as outcomes_module
from agent.outcomes import insufficient_verified_information_counts, _reset_for_testing

import main_pcm


def run(coro):
    return asyncio.run(coro)


SLOTS = {
    "doctor_name": "Dr. A. Sen",
    "date": "2026-09-15",
    "time_slot": "18:30",
    "patient_name": "Rina Das",
    "phone": "9831234567",
}


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session():
    return types.SimpleNamespace(
        call_id="test-call-finish-booking",
        pending={"awaiting": "confirm_booking", "slots": dict(SLOTS),
                  "candidates": None, "offered_date": None, "retries": 0},
        dispatch_lock=asyncio.Lock(),
        send_json=_AsyncNoOp(),
    )


class ConfigurableToolsClient:
    """Returns whatever book_appointment response the test hands it --
    unlike test_booking_readback.py's FakeToolsClient, which always
    returns a complete response, this one exists specifically to return
    incomplete ones."""

    def __init__(self, response: dict):
        self._response = response
        self.calls = []

    async def book_appointment(self, doctor_name, date, time_slot, patient_name, phone):
        self.calls.append({"doctor_name": doctor_name, "date": date, "time_slot": time_slot,
                            "patient_name": patient_name, "phone": phone})
        return self._response


@pytest.fixture(autouse=True)
def stub_speak(monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append((text, fallback_reason))

    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    return spoken


@pytest.fixture(autouse=True)
def isolated_outcomes_state(monkeypatch, tmp_path):
    monkeypatch.setattr(outcomes_module, "ESCALATION_LOG_PATH", str(tmp_path / "escalations.jsonl"))
    _reset_for_testing()
    yield tmp_path / "escalations.jsonl"
    _reset_for_testing()


def _stub_tools(monkeypatch, response: dict) -> ConfigurableToolsClient:
    tools = ConfigurableToolsClient(response)
    monkeypatch.setattr(main_pcm, "_tools", tools)
    return tools


class TestCompleteBookingIsUnaffected:
    """Regression: a normal, complete success response must still be read
    aloud exactly as booking_reply() would have produced before this
    story existed."""

    def test_complete_response_is_spoken_via_booking_reply(self, monkeypatch, stub_speak):
        response = {"success": True, "confirmation_id": "KCD-20260915-0099",
                    "doctor_name": "Dr. A. Sen", "date": "2026-09-15", "time_slot": "18:30"}
        _stub_tools(monkeypatch, response)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS)))

        assert len(stub_speak) == 1
        text, fallback_reason = stub_speak[0]
        assert "KCD-20260915-0099" in text
        assert fallback_reason is None

    def test_complete_response_does_not_record_anything(self, monkeypatch, stub_speak):
        response = {"success": True, "confirmation_id": "KCD-20260915-0099",
                    "doctor_name": "Dr. A. Sen", "date": "2026-09-15", "time_slot": "18:30"}
        _stub_tools(monkeypatch, response)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS)))

        assert insufficient_verified_information_counts("book_appointment") == 0

    def test_a_failed_booking_is_still_handled_by_booking_reply_not_this_outcome(self, monkeypatch, stub_speak):
        # success=False (e.g. slot_taken) is a DIFFERENT, pre-existing
        # outcome -- this story must not intercept it.
        response = {"success": False, "reason": "slot_taken", "alternative_slots": ["18:45"]}
        _stub_tools(monkeypatch, response)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS)))

        assert len(stub_speak) == 1
        assert insufficient_verified_information_counts("book_appointment") == 0


class TestIncompleteWriteTriggersTheNewOutcome:
    """The real trigger: success=True but one of confirmation_id/date/
    time_slot came back empty."""

    @pytest.mark.parametrize("missing_field", ["confirmation_id", "date", "time_slot"])
    def test_missing_field_withholds_the_confirmation(self, monkeypatch, stub_speak, missing_field):
        response = {"success": True, "confirmation_id": "KCD-20260915-0099",
                    "doctor_name": "Dr. A. Sen", "date": "2026-09-15", "time_slot": "18:30"}
        response[missing_field] = ""
        _stub_tools(monkeypatch, response)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS)))

        assert len(stub_speak) == 1
        text, fallback_reason = stub_speak[0]
        # The caller must NOT hear a confirmation number it cannot verify.
        assert "KCD-20260915-0099" not in text
        assert fallback_reason == "insufficient_verified_information"

    def test_missing_field_speaks_the_dedicated_template_not_booking_reply(self, monkeypatch, stub_speak):
        from agent.outcomes import insufficient_verified_information_reply
        response = {"success": True, "confirmation_id": "", "doctor_name": "Dr. A. Sen",
                    "date": "2026-09-15", "time_slot": "18:30"}
        _stub_tools(monkeypatch, response)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS)))

        assert stub_speak[0][0] == insufficient_verified_information_reply()

    def test_missing_field_increments_the_book_appointment_count(self, monkeypatch, stub_speak):
        response = {"success": True, "confirmation_id": "", "doctor_name": "Dr. A. Sen",
                    "date": "2026-09-15", "time_slot": "18:30"}
        _stub_tools(monkeypatch, response)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS)))

        assert insufficient_verified_information_counts("book_appointment") == 1

    def test_missing_field_writes_an_escalation_record_with_the_call_id(self, monkeypatch, stub_speak, isolated_outcomes_state):
        response = {"success": True, "confirmation_id": "", "doctor_name": "Dr. A. Sen",
                    "date": "2026-09-15", "time_slot": "18:30"}
        _stub_tools(monkeypatch, response)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS)))

        raw = isolated_outcomes_state.read_text(encoding="utf-8").strip()
        entry = json.loads(raw)
        assert entry["intent"] == "book_appointment"
        assert entry["field"] == "confirmation_id"
        assert entry["call_id"] == session.call_id

    def test_multiple_missing_fields_are_all_named_in_the_escalation(self, monkeypatch, stub_speak, isolated_outcomes_state):
        response = {"success": True, "confirmation_id": "", "doctor_name": "Dr. A. Sen",
                    "date": "", "time_slot": "18:30"}
        _stub_tools(monkeypatch, response)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS)))

        entry = json.loads(isolated_outcomes_state.read_text(encoding="utf-8").strip())
        assert entry["field"] == "confirmation_id,date"

    def test_book_appointment_is_still_only_called_once(self, monkeypatch, stub_speak):
        # The outcome fires AFTER the write, not instead of it -- this
        # story is about withholding the READBACK, not preventing the
        # attempt. Confirm no retry/second call happens on its own.
        response = {"success": True, "confirmation_id": "", "doctor_name": "Dr. A. Sen",
                    "date": "2026-09-15", "time_slot": "18:30"}
        tools = _stub_tools(monkeypatch, response)
        session = make_session()

        run(main_pcm._finish_booking(session, dict(SLOTS)))

        assert len(tools.calls) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
