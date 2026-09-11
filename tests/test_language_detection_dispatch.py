"""ADDED BY SOURAV -- fixes a real production bug, reported directly by
the client:

    "why voice is giving response only in bengali, not in english or
    hinglish or hindi, when the user asks in hindi aur hinglish or
    english"

Root cause (see agent/bn_normalize.py::detect_language()'s own
UPDATED BY SOURAV comment for the full writeup, and main.py's own
"ADDED BY SOURAV" comment just above `language = detect_language(text)`
in _dispatch_turn): every reply-generating function in this codebase
(agent/reply_templates.py, agent/report_flow.py, agent/outcomes.py) has
ALWAYS accepted a `language=` argument and already had all 4 languages'
text written and unit-tested -- but main.py's dispatch never called
detect_language() at all and never passed `language=` to any of the ~24
reply call sites across `_dispatch_turn` and `_continue_pending`, so
every single reply defaulted to Bengali regardless of what the caller
actually said. A second, independent bug was found and fixed inside
detect_language() itself while wiring it up -- see
tests/test_hinglish_fidelity.py's own module-level tests for that half.

This file is the OTHER half: proving, at the full dispatch level (driving
main_pcm._dispatch_turn()/_continue_pending() exactly the way a real call
does), that the detected language from the CALLER'S OWN utterance text
actually reaches the spoken reply -- not just that detect_language()
returns the right label in isolation.

Scope note (explicitly NOT covered here, flagged as a separate,
pre-existing gap -- see TEST_REPORT_language_detection_threading.md):
the several hardcoded Bengali-only canned strings still sitting directly
in main.py (asr_empty apology, llm_failure apology, unclear-intent
apology, tool_failure apology, the idle-timeout farewell, and the very
first greeting spoken before any caller utterance exists) were never
built multilingual at all -- no translated text exists anywhere for them,
unlike every reply_templates.py function this fix threads language
through. Also out of scope: the LLM's own `direct_reply_bn` smalltalk
reply (hardcoded Bengali, a separate prompt-engineering concern).

Fixture pattern (FakeToolsClient / make_session / FakeASR / dispatch
helper) mirrors tests/test_doctor_schedule_dispatch.py's own
TestDispatchWithFakeTools exactly, with one addition: FakeASRResult.text
is made a per-call instance attribute (not a fixed class constant) since
this fix's whole point is that dispatch must react to what THAT specific
utterance's ASR text actually says.
"""
from __future__ import annotations

import asyncio
import types

import pytest

import main_pcm
from agent.bn_normalize import detect_language, weekday_to_words
# Aliased -- these names start with "test_", and pytest would otherwise
# try to collect the imported functions themselves as test items (see
# tests/test_test_duration_reply.py's own import of test_duration_reply
# for the same precedent).
from agent.reply_templates import (
    test_rate_reply as rate_reply,
    doctor_schedule_reply,
    doctors_by_department_reply,
    doctor_availability_reply,
    missing_slot_prompt,
    booking_confirmation_prompt,
)


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- #
# Fixtures -- mirrors tests/test_doctor_schedule_dispatch.py's own
# --------------------------------------------------------------------- #

class FakeToolsClient:
    def __init__(self):
        self.get_test_rate_response = None
        self.get_doctor_schedule_response = None
        self.get_doctors_by_department_response = None
        self.get_doctor_availability_response = None

    async def get_test_rate(self, test_name):
        return self.get_test_rate_response

    async def get_doctor_schedule(self, doctor_name):
        return self.get_doctor_schedule_response

    async def get_doctors_by_department(self, department, date_iso):
        return self.get_doctors_by_department_response

    async def get_doctor_availability(self, doctor_name, date_iso):
        return self.get_doctor_availability_response


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-language-detection-call",
        pending=pending,
        dispatch_lock=asyncio.Lock(),
        send_json=_AsyncNoOp(),
    )


class FakeASRResult:
    def __init__(self, text):
        self.text = text


class FakeASR:
    def __init__(self, text):
        self._text = text

    async def transcribe_utterance(self, wav_path):
        return FakeASRResult(self._text)


@pytest.fixture
def env(monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    tools = FakeToolsClient()
    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    monkeypatch.setattr(main_pcm, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools)


def dispatch(monkeypatch, env, utterance_text, intent, slots, tmp_path, pending=None):
    async def fake_resolve_intent(session, text):
        return {"intent": intent, "slots": slots}

    monkeypatch.setattr(main_pcm, "_asr", FakeASR(utterance_text))
    monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)

    session = make_session(pending=pending)
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return session


# Utterances deliberately built from detect_language()'s own documented
# marker vocabulary (agent/bn_normalize.py's _HINGLISH_MARKERS /
# _BANGLISH_MARKERS), so this test is pinned to a real, working
# classification rather than to a guess about what the detector accepts.
_ENGLISH_UTTERANCE = "What is the rate of the urine test?"
_HINGLISH_UTTERANCE = "Urine test ka rate kya hai?"
_BANGLISH_UTTERANCE = "Ei test ta korte koto taka lagbe"
_BENGALI_UTTERANCE = "ইউরিন টেস্টের রেট কত?"


class TestDetectedLanguageActuallyReachesTheSpokenReply:
    """The core of the fix: the SAME intent/slots/tool-result, spoken back
    in 4 different languages depending ONLY on what the caller's own
    utterance said -- proving language is threaded end to end, not just
    detected and discarded."""

    @pytest.mark.parametrize("utterance,expected_language", [
        (_ENGLISH_UTTERANCE, "english"),
        (_HINGLISH_UTTERANCE, "hinglish"),
        (_BANGLISH_UTTERANCE, "banglish"),
        (_BENGALI_UTTERANCE, "bengali"),
    ])
    def test_test_rate_reply_matches_the_detected_language(
        self, monkeypatch, env, tmp_path, utterance, expected_language
    ):
        # Sanity check on the fixture itself -- if detect_language() ever
        # changes and stops agreeing with these utterances, this test
        # must fail loudly here, not silently pass for the wrong reason.
        assert detect_language(utterance) == expected_language

        slots = {"test_name": "Urine Routine Examination"}
        result = {
            "found": True, "test_name": "Urine Routine Examination",
            "test_name_bn": "ইউরিন রুটিন পরীক্ষা", "rate_inr": "200",
            "sample_type": "Urine", "report_time_hours": 24,
        }
        env.tools.get_test_rate_response = result
        dispatch(monkeypatch, env, utterance, "test_rate", slots, tmp_path)

        expected = rate_reply(slots, result, language=expected_language)
        assert env.spoken == [expected]
        # Regression guard: never silently falls back to Bengali for a
        # non-Bengali utterance (the exact bug reported).
        if expected_language != "bengali":
            bengali_reply = rate_reply(slots, result, language="bengali")
            assert env.spoken[0] != bengali_reply

    @pytest.mark.parametrize("utterance,expected_language", [
        (_ENGLISH_UTTERANCE, "english"),
        (_HINGLISH_UTTERANCE, "hinglish"),
        (_BANGLISH_UTTERANCE, "banglish"),
    ])
    def test_missing_slot_prompt_matches_the_detected_language(
        self, monkeypatch, env, tmp_path, utterance, expected_language
    ):
        dispatch(
            monkeypatch, env, utterance, "test_duration", {"test_name": None}, tmp_path,
        )
        expected = missing_slot_prompt("test_duration", "test_name", language=expected_language)
        assert env.spoken == [expected]

    def test_doctor_schedule_reply_in_hinglish(self, monkeypatch, env, tmp_path):
        result = {
            "found": True, "doctor_name": "Dr A. Sen", "doctor_name_bn": "ডক্টর সেন",
            "schedule": [{"weekday": 0, "start_time": "10:00", "end_time": "12:00"}],
        }
        env.tools.get_doctor_schedule_response = result
        slots = {"doctor_name": "Dr A. Sen"}
        dispatch(
            monkeypatch, env, "Dr Sen kab baithte hain?", "doctor_schedule", slots, tmp_path,
        )
        expected = doctor_schedule_reply(slots, result, language="hinglish")
        assert env.spoken == [expected]
        assert weekday_to_words(0, "hinglish") in env.spoken[0]

    def test_doctors_by_department_reply_in_banglish(self, monkeypatch, env, tmp_path):
        result = {"found": True, "doctors": []}
        env.tools.get_doctors_by_department_response = result
        slots = {"department": "Cardiology"}
        dispatch(
            monkeypatch, env, "Cardiology te ke ache aajke", "doctors_by_department", slots, tmp_path,
        )
        expected = doctors_by_department_reply(slots, result, language="banglish")
        assert env.spoken == [expected]


class TestPerTurnCodeSwitchingNotCarriedAcrossTurns:
    """Language is detected fresh from EACH turn's own text, never stored
    on session.pending -- a caller who opens a flow in one language and
    replies in another must be answered in the language of the reply they
    JUST gave, not the one the flow started in."""

    def test_doctor_availability_opens_in_bengali_then_continuation_replies_in_english(
        self, monkeypatch, env, tmp_path
    ):
        avail_result = {
            "found": True, "doctor_name": "Dr A. Sen", "available": False,
            "next_available_date": "2026-09-14",
        }
        env.tools.get_doctor_availability_response = avail_result
        slots = {"doctor_name": "Dr A. Sen"}
        session = dispatch(
            monkeypatch, env, _BENGALI_UTTERANCE, "doctor_availability", slots, tmp_path,
        )
        first_expected = doctor_availability_reply(slots, avail_result, language="bengali")
        assert env.spoken == [first_expected]
        assert session.pending is not None
        assert session.pending["awaiting"] == "date"

        # Second turn: caller replies in English to "want that next date
        # instead?" -- _continue_pending must detect THIS turn's language
        # independently, not inherit the Bengali the flow opened in.
        env.spoken.clear()
        async def fake_speak(s, text, fallback_reason=None):
            env.spoken.append(text)
        monkeypatch.setattr(main_pcm, "_speak", fake_speak)
        monkeypatch.setattr(main_pcm, "_asr", FakeASR("no, some other day please"))

        wav_path = tmp_path / "utt2.wav"
        wav_path.write_bytes(b"")
        run(main_pcm._dispatch_turn(session, str(wav_path)))

        expected_second = missing_slot_prompt("book_appointment", "date", language="english")
        assert env.spoken == [expected_second]


class TestBookAppointmentSingleShotThreadsLanguage:
    """book_appointment's own branch inside _dispatch_turn (not routed
    through _continue_pending at all on a caller's first mention) --
    covers the two calls in main.py's `if missing is None: ... else: ...`
    split."""

    def test_all_fields_known_confirmation_readback_in_english(self, monkeypatch, env, tmp_path):
        slots = {
            "doctor_name": "Dr A. Sen", "date": "2026-09-14", "time_slot": "10:00",
            "patient_name": "Rahim", "phone": "9000000001",
        }
        dispatch(
            monkeypatch, env,
            "Please book Dr Sen on 14th September at 10 am for Rahim, phone 9000000001",
            "book_appointment", slots, tmp_path,
        )
        expected = booking_confirmation_prompt(slots, language="english")
        assert env.spoken == [expected]

    def test_a_missing_field_prompt_is_in_hinglish(self, monkeypatch, env, tmp_path):
        slots = {"doctor_name": "Dr A. Sen", "date": "2026-09-14"}
        session = dispatch(
            monkeypatch, env, "Dr Sen ke saath 14 tarikh ko appointment chahiye",
            "book_appointment", slots, tmp_path,
        )
        assert len(env.spoken) == 1
        # Whichever field main.py's _next_missing() names next, it must be
        # asked for in the detected language, not defaulted to Bengali.
        missing_field = session.pending["awaiting"]
        expected = missing_slot_prompt("book_appointment", missing_field, language="hinglish")
        assert env.spoken == [expected]


class TestTransportParity:
    """main.py and main_pcm.py must dispatch identically -- main_pcm.py is
    a GENERATED file (tools/make_pcm_variant.py); this guards against hand-
    edit drift, same precedent as every other story's own TransportParity
    class in this test suite."""

    def test_both_files_call_detect_language_in_dispatch_turn(self):
        for path in ("main.py", "main_pcm.py"):
            src = open(path, encoding="utf-8").read()
            assert "language = detect_language(text)" in src, path

    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
