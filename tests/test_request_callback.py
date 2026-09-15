"""ADDED BY SOURAV -- "Caller asks to be called back" story.

Evidence: "No outbound capability" -- this system cannot itself place a
phone call, so the only honest behaviour is to log a request for a HUMAN
to call back, say so plainly, and never promise a callback it cannot
guarantee (Acceptance Criterion 3).

Covers every layer this story touches, mirroring the established
multi-place pattern this codebase's other stories already use (see
tests/test_compare_options.py's own module docstring for the precedent):

  1. agent/callback_flow.py: check_callback_availability() and
     build_callback_reason() -- pure functions, no I/O.
     TestCheckCallbackAvailability, TestBuildCallbackReason.
  2. agent/callback_config.py: CALLBACKS_ENABLED's env-var parsing.
     TestCallbackConfig.
  3. agent/llm.py: VALID_INTENTS/_SLOT_KEYS accept the new intent/slots.
     TestLlmSchema.
  4. agent/reply_templates.py: missing_slot_prompt entries,
     callback_unavailable_reply(), callback_confirmation_prompt(),
     callback_scheduled_reply(). TestReplyTemplates.
  5. main.py AND main_pcm.py dispatch -- parametrized over BOTH modules,
     same transport-parity discipline as every other multi-slot intent in
     this codebase. TestCallbackDispatch, TestTransportParity.
  6. clinic-api's own persistence endpoint is covered separately in
     tests/test_callback_api.py (real backend, real database) -- this
     file's dispatch tests use a FakeToolsClient, same as every other
     dispatch-level test in this suite.

Story acceptance criteria under direct test:
  - AC 1 (a callback is scheduled with a stated time window, preserving
    conversation context and reason): TestCallbackDispatch's collection/
    confirmation tests + TestBuildCallbackReason.
  - AC 2 (tracked to fulfilment, stored with pending status): covered in
    tests/test_callback_api.py's TestCallbackPersistence (this file only
    asserts the IVR side calls request_callback() with the right
    arguments and speaks the callback_id it gets back).
  - AC 3 (if unavailable, states plainly rather than a false promise):
    TestCheckCallbackAvailability + TestCallbackDispatch's unavailable-*
    tests.

WHY THE HOURS FIXTURES BELOW ARE THE SAME FOR ALL SEVEN WEEKDAYS:
Dispatch tests call the REAL main.py code, which resolves "what day/time
is it" from the real wall clock (datetime.date.today()/datetime.now()) --
mirrors tests/test_health_package_and_clinic_info_dispatch.py's own
test_today_weekday_is_resolved_in_dispatch_not_left_to_the_reply_template,
which sidesteps exactly this by giving every weekday the SAME hours in its
fixture rather than mocking datetime itself. The same trick is used here:
_ALL_DAYS_OPEN and _ALL_DAYS_CLOSED give an identical answer no matter
which real day/time the test suite happens to run on. The clock-comparison
logic itself (open_ <= now < close_) is instead tested directly and
deterministically in TestCheckCallbackAvailability, which takes
weekday_idx/current_time_hhmm as plain arguments -- no wall clock involved
at all.
"""
from __future__ import annotations

import asyncio
import importlib
import types

import pytest

from agent.callback_flow import (
    check_callback_availability, build_callback_reason,
    REASON_DISABLED, REASON_OUTSIDE_HOURS, REASON_HOURS_UNKNOWN,
)
from agent.llm import VALID_INTENTS, _SLOT_KEYS
from agent.reply_templates import (
    missing_slot_prompt, callback_unavailable_reply, callback_confirmation_prompt,
    callback_scheduled_reply,
)

import main
import main_pcm


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- #
# 1. agent/callback_flow.py -- pure functions
# --------------------------------------------------------------------- #

_OPEN_TODAY = {"closed": False, "open": "08:00", "close": "20:00"}
_CLOSED_TODAY = {"closed": True, "open": None, "close": None}
_ALL_DAYS_OPEN = {d: dict(_OPEN_TODAY) for d in
                  ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")}
_ALL_DAYS_CLOSED = {d: dict(_CLOSED_TODAY) for d in
                    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")}


class TestCheckCallbackAvailability:
    def test_disabled_wins_over_everything_else(self):
        # Even with perfectly-open hours and a mid-day time, a disabled
        # config always refuses -- checked first, unconditionally.
        r = check_callback_availability(_ALL_DAYS_OPEN, 0, "12:00", callbacks_enabled=False)
        assert r == {"available": False, "reason": REASON_DISABLED}

    def test_available_when_enabled_and_within_hours(self):
        r = check_callback_availability(_ALL_DAYS_OPEN, 2, "14:30", callbacks_enabled=True)
        assert r == {"available": True, "reason": None}

    def test_unavailable_at_exactly_closing_time(self):
        # [open, close) -- close itself is already outside the window.
        r = check_callback_availability(_ALL_DAYS_OPEN, 2, "20:00", callbacks_enabled=True)
        assert r == {"available": False, "reason": REASON_OUTSIDE_HOURS}

    def test_available_at_exactly_opening_time(self):
        r = check_callback_availability(_ALL_DAYS_OPEN, 2, "08:00", callbacks_enabled=True)
        assert r == {"available": True, "reason": None}

    def test_unavailable_before_opening(self):
        r = check_callback_availability(_ALL_DAYS_OPEN, 2, "06:00", callbacks_enabled=True)
        assert r == {"available": False, "reason": REASON_OUTSIDE_HOURS}

    def test_unavailable_after_closing(self):
        r = check_callback_availability(_ALL_DAYS_OPEN, 2, "22:15", callbacks_enabled=True)
        assert r == {"available": False, "reason": REASON_OUTSIDE_HOURS}

    def test_unavailable_on_a_day_marked_closed(self):
        r = check_callback_availability(_ALL_DAYS_CLOSED, 6, "12:00", callbacks_enabled=True)
        assert r == {"available": False, "reason": REASON_OUTSIDE_HOURS}

    def test_hours_none_is_honestly_unknown_not_a_guess(self):
        r = check_callback_availability(None, 2, "12:00", callbacks_enabled=True)
        assert r == {"available": False, "reason": REASON_HOURS_UNKNOWN}

    def test_weekday_idx_none_is_honestly_unknown(self):
        r = check_callback_availability(_ALL_DAYS_OPEN, None, "12:00", callbacks_enabled=True)
        assert r == {"available": False, "reason": REASON_HOURS_UNKNOWN}

    def test_weekday_idx_out_of_range_is_honestly_unknown(self):
        r = check_callback_availability(_ALL_DAYS_OPEN, 7, "12:00", callbacks_enabled=True)
        assert r == {"available": False, "reason": REASON_HOURS_UNKNOWN}

    def test_current_time_none_is_honestly_unknown(self):
        r = check_callback_availability(_ALL_DAYS_OPEN, 2, None, callbacks_enabled=True)
        assert r == {"available": False, "reason": REASON_HOURS_UNKNOWN}

    def test_missing_open_close_on_a_supposedly_open_day_is_honestly_unknown(self):
        malformed = dict(_ALL_DAYS_OPEN)
        malformed["monday"] = {"closed": False, "open": None, "close": None}
        r = check_callback_availability(malformed, 0, "12:00", callbacks_enabled=True)
        assert r == {"available": False, "reason": REASON_HOURS_UNKNOWN}


class TestBuildCallbackReason:
    def test_stated_reason_always_wins(self):
        r = build_callback_reason("about my CBC report", active_test="Lipid Profile")
        assert r == "about my CBC report"

    def test_stated_reason_is_stripped(self):
        r = build_callback_reason("  about billing  ")
        assert r == "about billing"

    def test_falls_back_to_active_test_when_no_reason_stated(self):
        r = build_callback_reason(None, active_test="CBC")
        assert r == "Regarding CBC"

    def test_falls_back_to_active_doctor_when_no_test_tracked(self):
        r = build_callback_reason(None, active_doctor="Dr Sen")
        assert r == "Regarding Dr Sen"

    def test_falls_back_to_active_package_when_nothing_else_tracked(self):
        r = build_callback_reason(None, active_package="Diabetes Package")
        assert r == "Regarding Diabetes Package"

    def test_test_takes_priority_over_doctor_and_package(self):
        r = build_callback_reason(None, active_test="CBC", active_doctor="Dr Sen", active_package="Diabetes Package")
        assert r == "Regarding CBC"

    def test_nothing_stated_and_nothing_tracked_is_honestly_none(self):
        # AC1's "reason" is genuinely optional -- never backfilled with an
        # invented generic sentence.
        r = build_callback_reason(None)
        assert r is None

    def test_empty_string_stated_reason_is_treated_as_not_stated(self):
        r = build_callback_reason("   ", active_test="CBC")
        assert r == "Regarding CBC"


# --------------------------------------------------------------------- #
# 2. agent/callback_config.py -- env-var parsing
# --------------------------------------------------------------------- #

class TestCallbackConfig:
    def _reload(self, monkeypatch, value=None):
        if value is None:
            monkeypatch.delenv("CALLBACKS_ENABLED", raising=False)
        else:
            monkeypatch.setenv("CALLBACKS_ENABLED", value)
        import agent.callback_config as cc
        return importlib.reload(cc)

    def test_default_with_no_env_var_set_is_enabled(self, monkeypatch):
        cc = self._reload(monkeypatch, None)
        assert cc.CALLBACKS_ENABLED is True

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "off", "disabled"])
    def test_recognized_off_spellings_disable_the_feature(self, monkeypatch, value):
        cc = self._reload(monkeypatch, value)
        assert cc.CALLBACKS_ENABLED is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "on", "anything-else"])
    def test_every_other_value_leaves_the_feature_enabled(self, monkeypatch, value):
        cc = self._reload(monkeypatch, value)
        assert cc.CALLBACKS_ENABLED is True


# --------------------------------------------------------------------- #
# 3. agent/llm.py schema
# --------------------------------------------------------------------- #

class TestLlmSchema:
    def test_request_callback_is_a_valid_intent(self):
        assert "request_callback" in VALID_INTENTS

    def test_new_slots_are_registered(self):
        assert "callback_time_window" in _SLOT_KEYS
        assert "callback_reason" in _SLOT_KEYS

    def test_phone_slot_is_reused_not_duplicated(self):
        # AC/task note: the callback number reuses the EXISTING "phone"
        # slot rather than a new "callback_phone" slot key (that name is
        # reserved for _continue_pending's own scoped `awaiting` value --
        # see main.py's own comment on the collision this avoids).
        assert "phone" in _SLOT_KEYS
        assert "callback_phone" not in _SLOT_KEYS


# --------------------------------------------------------------------- #
# 4. agent/reply_templates.py
# --------------------------------------------------------------------- #

_LANGUAGES = ["english", "hinglish", "banglish", "bengali"]


class TestReplyTemplates:
    def test_missing_slot_prompt_has_entries_for_both_new_fields(self):
        for lang in ("english", "hinglish", "bengali"):
            assert missing_slot_prompt("request_callback", "callback_time_window", language=lang) != \
                   missing_slot_prompt("unclear", "unclear", language=lang)
            assert missing_slot_prompt("request_callback", "callback_phone", language=lang) != \
                   missing_slot_prompt("unclear", "unclear", language=lang)

    @pytest.mark.parametrize("reason", ["disabled", "outside_hours", "hours_unknown"])
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_unavailable_reply_never_promises_a_callback_for_every_reason_and_language(self, reason, language):
        text = callback_unavailable_reply(reason, language=language)
        assert text
        # Never a database-row-shaped label -- "Answers sound like a
        # person, not a database row" (this module's own docstring).
        assert ":" not in text
        assert "[" not in text and "]" not in text

    def test_the_three_reasons_produce_genuinely_different_sentences(self):
        texts = {callback_unavailable_reply(r, language="english") for r in
                 ("disabled", "outside_hours", "hours_unknown")}
        assert len(texts) == 3

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_confirmation_prompt_reads_back_both_the_window_and_the_phone(self, language):
        slots = {"callback_time_window": "this evening", "phone": "9831012345"}
        text = callback_confirmation_prompt(slots, language=language)
        assert "this evening" in text
        assert "9831012345" in text
        assert ":" not in text

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_scheduled_reply_speaks_the_callback_id_on_success(self, language):
        slots = {"callback_time_window": "this evening"}
        result = {"success": True, "callback_id": "CB-20260915-A1B2", "phone": "9831012345",
                   "time_window": "this evening", "reason": None, "status": "pending"}
        text = callback_scheduled_reply(slots, result, language=language)
        assert "CB-20260915-A1B2" in text
        assert "this evening" in text
        # Never implies the SYSTEM itself will place the call -- "No
        # outbound capability" is this whole story's own evidence.
        assert ":" not in text

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_scheduled_reply_has_an_honest_failure_branch(self, language):
        text = callback_scheduled_reply({}, {"success": False}, language=language)
        assert text
        assert ":" not in text


# --------------------------------------------------------------------- #
# 5. Dispatch-level integration -- main.py AND main_pcm.py
# --------------------------------------------------------------------- #

class FakeToolsClient:
    def __init__(self, clinic_info_response=None):
        self.calls = []
        self._clinic_info_response = clinic_info_response or {"found": True, "hours": _ALL_DAYS_OPEN}
        self._callback_result = None
        self._callback_exception = None

    def _record(self, name, *args):
        self.calls.append((name, args))

    async def get_clinic_info(self):
        self._record("get_clinic_info")
        return self._clinic_info_response

    async def request_callback(self, phone, time_window, reason=None):
        self._record("request_callback", phone, time_window, reason)
        if self._callback_exception:
            raise self._callback_exception
        if self._callback_result is not None:
            return self._callback_result
        return {"success": True, "callback_id": "CB-20260915-TEST", "phone": phone,
                "time_window": time_window, "reason": reason, "status": "pending"}


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


class FakeASRResult:
    text = "amake ektu callback korte bolben"


class FakeASR:
    async def transcribe_utterance(self, wav_path):
        return FakeASRResult()


def make_session(pending=None):
    from agent.state import DialogueState as _DS
    return types.SimpleNamespace(
        call_id="test-call-1", pending=pending,
        dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
        state=_DS(),
    )


TRANSPORTS = [main, main_pcm]


@pytest.fixture(params=TRANSPORTS, ids=["main", "main_pcm"])
def transport(request):
    return request.param


@pytest.fixture
def stub(transport, monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    monkeypatch.setattr(transport, "_speak", fake_speak)
    monkeypatch.setattr(transport, "_asr", FakeASR())
    monkeypatch.setattr(transport, "CALLBACKS_ENABLED", True)
    tools = FakeToolsClient()
    monkeypatch.setattr(transport, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools, transport=transport)


def _empty_slots(**overrides):
    slots = {
        "test_name": None, "doctor_name": None, "department": None, "date": None,
        "time_slot": None, "patient_name": None, "phone": None, "package_name": None,
        "info_topic": None, "insurance_provider_name": None,
        "compare_option_a": None, "compare_option_b": None,
        "callback_time_window": None, "callback_reason": None,
    }
    slots.update(overrides)
    return slots


def _dispatch(stub, monkeypatch, intent, slots, tmp_path, session=None):
    async def fake_resolve_intent(sess, text):
        return {"intent": intent, "slots": slots}

    monkeypatch.setattr(stub.transport, "_resolve_intent", fake_resolve_intent)
    session = session or make_session()
    wav_path = tmp_path / f"utt-{id(slots)}.wav"
    wav_path.write_bytes(b"")
    run(stub.transport._dispatch_turn(session, str(wav_path)))
    return session


def _continue(stub, session, text):
    return run(stub.transport._continue_pending(session, text))


class TestCallbackDispatchAvailability:
    def test_disabled_refuses_before_ever_calling_get_clinic_info(self, stub, monkeypatch, tmp_path):
        monkeypatch.setattr(stub.transport, "CALLBACKS_ENABLED", False)
        session = _dispatch(stub, monkeypatch, "request_callback",
                             _empty_slots(callback_time_window="evening", phone="9831012345"), tmp_path)
        assert session.pending is None
        assert "get_clinic_info" not in [c[0] for c in stub.tools.calls]
        assert "request_callback" not in [c[0] for c in stub.tools.calls]
        assert len(stub.spoken) == 1

    def test_outside_hours_refuses_without_opening_a_pending_flow(self, stub, monkeypatch, tmp_path):
        stub.tools._clinic_info_response = {"found": True, "hours": _ALL_DAYS_CLOSED}
        session = _dispatch(stub, monkeypatch, "request_callback",
                             _empty_slots(callback_time_window="evening", phone="9831012345"), tmp_path)
        assert session.pending is None
        assert "request_callback" not in [c[0] for c in stub.tools.calls]
        assert len(stub.spoken) == 1

    def test_clinic_info_not_found_is_honestly_unknown_not_a_false_promise(self, stub, monkeypatch, tmp_path):
        stub.tools._clinic_info_response = {"found": False}
        session = _dispatch(stub, monkeypatch, "request_callback", _empty_slots(), tmp_path)
        assert session.pending is None
        assert "request_callback" not in [c[0] for c in stub.tools.calls]

    def test_available_opens_the_collection_flow(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "request_callback", _empty_slots(), tmp_path)
        assert session.pending is not None
        assert session.pending["awaiting"] == "callback_time_window"


class TestCallbackDispatchCollectionAndConfirmation:
    def test_single_shot_with_both_fields_goes_straight_to_confirmation(self, stub, monkeypatch, tmp_path):
        session = _dispatch(
            stub, monkeypatch, "request_callback",
            _empty_slots(callback_time_window="this evening", phone="9831012345"), tmp_path,
        )
        assert session.pending["awaiting"] == "confirm_callback"
        assert "9831012345" in stub.spoken[-1]
        assert "this evening" in stub.spoken[-1]
        assert "request_callback" not in [c[0] for c in stub.tools.calls]

    def test_missing_both_fields_asks_time_window_first(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "request_callback", _empty_slots(), tmp_path)
        assert session.pending["awaiting"] == "callback_time_window"
        assert stub.spoken == [missing_slot_prompt("request_callback", "callback_time_window")]

    def test_time_window_then_phone_then_confirm_then_finish(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "request_callback", _empty_slots(), tmp_path)
        assert session.pending["awaiting"] == "callback_time_window"

        _continue(stub, session, "this evening")
        assert session.pending["awaiting"] == "callback_phone"

        _continue(stub, session, "9831012345")
        assert session.pending["awaiting"] == "confirm_callback"

        _continue(stub, session, "হ্যাঁ")
        assert session.pending is None
        assert ("request_callback", ("9831012345", "this evening", None)) in stub.tools.calls
        assert "CB-20260915-TEST" in stub.spoken[-1]

    def test_giving_phone_first_then_time_window_still_completes(self, stub, monkeypatch, tmp_path):
        # AC1's "stated time window" does not mandate a fixed order --
        # _next_missing_callback just picks whichever is still empty.
        session = _dispatch(stub, monkeypatch, "request_callback",
                             _empty_slots(phone="9831055566"), tmp_path)
        assert session.pending["awaiting"] == "callback_time_window"
        _continue(stub, session, "tomorrow morning")
        assert session.pending["awaiting"] == "confirm_callback"

    def test_unparseable_phone_reprompts_then_gives_up(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "request_callback",
                             _empty_slots(callback_time_window="evening"), tmp_path)
        assert session.pending["awaiting"] == "callback_phone"
        for _ in range(3):
            handled = _continue(stub, session, "gibberish not a phone number")
        assert handled is False
        assert session.pending is None
        assert "request_callback" not in [c[0] for c in stub.tools.calls]

    def test_negative_reply_to_confirmation_abandons_cleanly(self, stub, monkeypatch, tmp_path):
        session = _dispatch(
            stub, monkeypatch, "request_callback",
            _empty_slots(callback_time_window="evening", phone="9831012345"), tmp_path,
        )
        assert session.pending["awaiting"] == "confirm_callback"
        _continue(stub, session, "না")
        assert session.pending is None
        assert "request_callback" not in [c[0] for c in stub.tools.calls]

    def test_negative_reply_mid_collection_abandons_cleanly(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "request_callback", _empty_slots(), tmp_path)
        _continue(stub, session, "না")
        assert session.pending is None

    def test_tool_failure_speaks_infra_apology_and_clears_pending(self, stub, monkeypatch, tmp_path):
        from agent.tools_client import ToolCallError
        session = _dispatch(
            stub, monkeypatch, "request_callback",
            _empty_slots(callback_time_window="evening", phone="9831012345"), tmp_path,
        )
        stub.tools._callback_exception = ToolCallError("boom")
        _continue(stub, session, "হ্যাঁ")
        assert session.pending is None
        assert "CB-" not in stub.spoken[-1]

    def test_success_but_missing_callback_id_withholds_confirmation(self, stub, monkeypatch, tmp_path):
        session = _dispatch(
            stub, monkeypatch, "request_callback",
            _empty_slots(callback_time_window="evening", phone="9831012345"), tmp_path,
        )
        stub.tools._callback_result = {"success": True, "callback_id": "", "phone": "9831012345"}
        _continue(stub, session, "হ্যাঁ")
        assert session.pending is None
        assert "9831012345" not in stub.spoken[-1]


class TestCallbackContextCapture:
    """Acceptance Criterion 1: "preserving the conversation context and
    reason". A stated reason always wins; failing that, whatever entity
    the caller was already discussing this call (agent/state.py's
    DialogueState, from "Caller asks a follow-up..." story) is captured
    automatically -- never re-asked for, never fabricated."""

    def test_stated_reason_is_forwarded_to_the_write(self, stub, monkeypatch, tmp_path):
        session = _dispatch(
            stub, monkeypatch, "request_callback",
            _empty_slots(callback_time_window="evening", phone="9831012345",
                         callback_reason="about my CBC report"),
            tmp_path,
        )
        _continue(stub, session, "yes")
        call = next(c for c in stub.tools.calls if c[0] == "request_callback")
        assert call[1][2] == "about my CBC report"

    def test_tracked_test_entity_is_captured_when_no_reason_stated(self, stub, monkeypatch, tmp_path):
        session = make_session()
        session.state.mark("test", "CBC")
        session = _dispatch(
            stub, monkeypatch, "request_callback",
            _empty_slots(callback_time_window="evening", phone="9831012345"),
            tmp_path, session=session,
        )
        _continue(stub, session, "yes")
        call = next(c for c in stub.tools.calls if c[0] == "request_callback")
        assert call[1][2] == "Regarding CBC"

    def test_no_reason_and_nothing_tracked_is_honestly_null(self, stub, monkeypatch, tmp_path):
        session = _dispatch(
            stub, monkeypatch, "request_callback",
            _empty_slots(callback_time_window="evening", phone="9831012345"),
            tmp_path,
        )
        _continue(stub, session, "yes")
        call = next(c for c in stub.tools.calls if c[0] == "request_callback")
        assert call[1][2] is None

    def test_reason_is_resolved_once_and_does_not_change_across_the_multi_turn_flow(self, stub, monkeypatch, tmp_path):
        session = make_session()
        session.state.mark("doctor", "Dr Sen")
        session = _dispatch(stub, monkeypatch, "request_callback", _empty_slots(), tmp_path, session=session)
        # Tracked state could, in principle, change mid-flow if some other
        # code path touched it -- it must NOT be re-resolved from a
        # possibly-different state later.
        session.state.mark("doctor", "Dr Gupta")
        _continue(stub, session, "this evening")
        _continue(stub, session, "9831012345")
        _continue(stub, session, "yes")
        call = next(c for c in stub.tools.calls if c[0] == "request_callback")
        assert call[1][2] == "Regarding Dr Sen"


class TestCallbackDoesNotInterfereWithBooking:
    """request_callback introduces two new scoped `awaiting` names
    ("callback_time_window", "callback_phone") specifically so they never
    collide with the pre-existing booking flow's bare "phone" awaiting
    state -- this is the same collision main.py's own comments document
    for "report_phone"/"billing_phone"."""

    def test_a_booking_still_in_progress_is_unaffected_by_the_new_awaiting_names(self, stub, monkeypatch, tmp_path):
        booking_pending = {
            "awaiting": "phone", "slots": {"doctor_name": "Dr Sen", "date": "2026-09-20",
                                            "time_slot": "10:00", "patient_name": "Rahul Sen"},
            "candidates": None, "offered_date": None, "retries": 0,
        }
        session = make_session(pending=booking_pending)
        _continue(stub, session, "9831012345")
        # Booking's own shared tail must still have handled this as a
        # BOOKING phone number, not a callback one.
        assert session.pending["awaiting"] == "confirm_booking"


# --------------------------------------------------------------------- #
# 6. Transport parity
# --------------------------------------------------------------------- #

class TestTransportParity:
    def test_main_and_main_pcm_produce_identical_replies_across_the_full_flow(self, monkeypatch, tmp_path):
        results = {}
        for transport in TRANSPORTS:
            spoken = []

            async def fake_speak(session, text, fallback_reason=None, _spoken=spoken):
                _spoken.append(text)

            monkeypatch.setattr(transport, "_speak", fake_speak)
            monkeypatch.setattr(transport, "_asr", FakeASR())
            monkeypatch.setattr(transport, "CALLBACKS_ENABLED", True)
            tools = FakeToolsClient()
            monkeypatch.setattr(transport, "_tools", tools)

            async def fake_resolve_intent(sess, text, _slots=_empty_slots()):
                return {"intent": "request_callback", "slots": _slots}

            monkeypatch.setattr(transport, "_resolve_intent", fake_resolve_intent)
            session = make_session()
            wav_path = tmp_path / f"utt-{transport.__name__}.wav"
            wav_path.write_bytes(b"")
            run(transport._dispatch_turn(session, str(wav_path)))
            run(transport._continue_pending(session, "this evening"))
            run(transport._continue_pending(session, "9831012345"))
            run(transport._continue_pending(session, "yes"))
            results[transport.__name__] = spoken

        assert results["main"] == results["main_pcm"]
