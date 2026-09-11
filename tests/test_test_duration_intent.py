"""ADDED BY SOURAV -- fixes a real production bug, reported directly from
a live call transcript:

    [User] How long does it take to get the urine test report?
    [AI]   Urine test rate is 200 taka.
    [User] How long will it take to get the urine test report?
    [AI]   Urine test rate is 200 taka.

Mirrors tests/test_test_sample_intent.py's structure -- that story added
"test_sample" the same way this one adds "test_duration", touching the
same three non-reply-template places:

  1. agent/llm.py: "test_duration" is a valid classification target,
     distinct from "test_rate" (price only, since "Caller asks the price
     of a test" narrowed it) and from "test_sample" (sample only). Root
     cause of the reported bug: test_rate's own intent description used
     to claim it also covered "how long results take" -- a claim that
     became false the moment test_rate_reply() was narrowed to speak only
     the price, and was never corrected until this fix.
  2. agent/semantic_cache.py: "test_duration" is registered in
     _REQUIRED_ENTITY_FOR_INTENT, exactly like "test_rate"/"test_sample"
     already are -- an extraction missing test_name must never enter the
     L2 semantic-cache index.
  3. main_pcm.py: the new `elif intent == "test_duration":` dispatch
     branch -- same tool call as test_rate/test_sample (clinic-api's test
     lookup already returns report_time_hours on every call), rendered
     through test_duration_reply() instead, gated on test_name being
     present exactly like the other two.

Reply-text-level coverage (all 4 languages, not-found delegation, honest
missing-hours fallback, spoken-punctuation cleanliness) lives in
tests/test_test_duration_reply.py instead, mirroring where test_sample's
own reply coverage lives (test_reply_templates_fidelity.py).

Dispatch-level tests follow tests/test_test_sample_intent.py's own
established pattern: no pytest-asyncio, ASR and intent extraction stubbed
via monkeypatch, driven with asyncio.run() inside ordinary sync test
functions.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from agent.llm import VALID_INTENTS, _validate
from agent.semantic_cache import _REQUIRED_ENTITY_FOR_INTENT, SemanticCache

import main_pcm


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- #
# agent/llm.py
# --------------------------------------------------------------------- #

class TestTestDurationIsAValidIntent:
    def test_registered_in_valid_intents(self):
        assert "test_duration" in VALID_INTENTS

    def test_distinct_from_test_rate_and_test_sample(self):
        # The whole point of the new intent: a caller asking about
        # turnaround time must not be classified the same way as one
        # asking about price (test_rate) or the sample (test_sample).
        assert "test_rate" in VALID_INTENTS
        assert "test_sample" in VALID_INTENTS
        assert "test_duration" not in ("test_rate", "test_sample")

    def test_validate_accepts_a_well_formed_test_duration_payload(self):
        data = {
            "intent": "test_duration",
            "slots": {
                "test_name": "Urine Routine Examination", "doctor_name": None,
                "department": None, "date": None, "time_slot": None,
                "patient_name": None, "phone": None,
            },
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_rejects_an_intent_not_in_the_enum(self):
        # Regression guard: adding "test_duration" must not have loosened
        # _validate() into accepting arbitrary strings.
        data = {
            "intent": "test_speed",
            "slots": {
                "test_name": None, "doctor_name": None, "department": None,
                "date": None, "time_slot": None, "patient_name": None, "phone": None,
            },
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is False
        assert any("invalid intent" in e for e in errors)


# --------------------------------------------------------------------- #
# agent/semantic_cache.py
# --------------------------------------------------------------------- #

class TestTestDurationRequiredEntity:
    """_REQUIRED_ENTITY_FOR_INTENT / _is_l2_eligible() -- registered here
    for "test_duration" the same way it already is for "test_rate" and
    "test_sample", proactively (not requested) -- a new single-entity
    intent left OUT of this dict would silently reopen the cache-
    poisoning bug this dict exists to close."""

    def test_registered_with_test_name_as_the_required_entity(self):
        assert _REQUIRED_ENTITY_FOR_INTENT.get("test_duration") == "test_name"

    def test_extraction_with_test_name_is_l2_eligible(self):
        value = {"intent": "test_duration", "slots": {"test_name": "Urine Routine Examination"}}
        assert SemanticCache._is_l2_eligible(value) is True

    def test_extraction_missing_test_name_is_not_l2_eligible(self):
        value = {"intent": "test_duration", "slots": {"test_name": None}}
        assert SemanticCache._is_l2_eligible(value) is False

    def test_behaves_identically_to_test_rate_and_test_sample_for_the_same_slots(self):
        for intent in ("test_rate", "test_sample", "test_duration"):
            assert SemanticCache._is_l2_eligible(
                {"intent": intent, "slots": {"test_name": "Urine Routine Examination"}}
            ) is True
            assert SemanticCache._is_l2_eligible(
                {"intent": intent, "slots": {}}
            ) is False


# --------------------------------------------------------------------- #
# main_pcm.py dispatch (mirrors tests/test_test_sample_intent.py's pattern)
# --------------------------------------------------------------------- #

class FakeToolsClient:
    """Records every get_test_rate() call -- test_duration reuses the
    SAME clinic-api call test_rate/test_sample already make (the response
    has always carried report_time_hours); only the reply-rendering
    function differs."""

    def __init__(self, response=None):
        self.get_test_rate_calls = []
        self._response = response or {
            "found": True, "test_name": "Urine Routine Examination",
            "test_name_bn": "ইউরিন রুটিন পরীক্ষা",
            "rate_inr": "200", "sample_type": "Urine", "report_time_hours": 24,
        }

    async def get_test_rate(self, test_name):
        self.get_test_rate_calls.append(test_name)
        return self._response


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


class FakeASRResult:
    # UPDATED BY SOURAV -- must be real (Bengali) text, not the old English
    # placeholder ("ignored -- ..."), now that main_pcm.py's dispatch
    # actually calls detect_language() on it (see main.py's own "ADDED BY
    # SOURAV" comment on `language = detect_language(text)`, threaded
    # through by the language-detection production fix). This suite's own
    # assertions below compare against the Bengali-default reply text, so
    # the fake utterance is kept in Bengali to match -- language variation
    # itself is covered separately in tests/test_language_detection_dispatch.py.
    text = "টেস্টের সময় জানতে চাই"


class FakeASR:
    async def transcribe_utterance(self, wav_path):
        return FakeASRResult()


@pytest.fixture
def stub_speak_and_tools(monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    monkeypatch.setattr(main_pcm, "_asr", FakeASR())
    fake_tools = FakeToolsClient()
    monkeypatch.setattr(main_pcm, "_tools", fake_tools)
    return types.SimpleNamespace(spoken=spoken, tools=fake_tools)


def _dispatch(monkeypatch, slots, tmp_path):
    async def fake_resolve_intent(session, text):
        return {"intent": "test_duration", "slots": slots}

    monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
    session = make_session(pending=None)
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")  # _dispatch_turn just os.remove()s this
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return session


class TestDispatchTestDuration:
    def test_calls_get_test_rate_and_speaks_test_duration_reply(
        self, monkeypatch, stub_speak_and_tools, tmp_path
    ):
        from agent.reply_templates import test_duration_reply

        slots = {"test_name": "Urine Routine Examination"}
        _dispatch(monkeypatch, slots, tmp_path)

        assert stub_speak_and_tools.tools.get_test_rate_calls == ["Urine Routine Examination"]
        assert len(stub_speak_and_tools.spoken) == 1
        expected = test_duration_reply(slots, stub_speak_and_tools.tools._response)
        assert stub_speak_and_tools.spoken[0] == expected

    def test_reproduces_and_fixes_the_reported_bug_reply_has_no_price_in_it(
        self, monkeypatch, stub_speak_and_tools, tmp_path
    ):
        # The exact bug reported: "How long does it take to get the urine
        # test report?" must no longer be answered with the price.
        slots = {"test_name": "Urine Routine Examination"}
        _dispatch(monkeypatch, slots, tmp_path)
        assert "200" not in stub_speak_and_tools.spoken[0]

    def test_missing_test_name_asks_for_it_without_calling_the_tool(
        self, monkeypatch, stub_speak_and_tools, tmp_path
    ):
        from agent.reply_templates import missing_slot_prompt

        _dispatch(monkeypatch, {"test_name": None}, tmp_path)

        assert stub_speak_and_tools.tools.get_test_rate_calls == []
        assert stub_speak_and_tools.spoken == [missing_slot_prompt("test_duration", "test_name")]

    def test_tool_failure_gets_the_same_infrastructure_apology_as_test_rate(
        self, monkeypatch, stub_speak_and_tools, tmp_path
    ):
        # The new branch sits inside the same try/except ToolCallError
        # block as every other intent -- confirms it inherits that
        # handling automatically rather than needing its own.
        from agent.tools_client import ToolCallError

        class FailingTools(FakeToolsClient):
            async def get_test_rate(self, test_name):
                raise ToolCallError("clinic-api unreachable")

        monkeypatch.setattr(main_pcm, "_tools", FailingTools())
        _dispatch(monkeypatch, {"test_name": "Urine Routine Examination"}, tmp_path)
        assert len(stub_speak_and_tools.spoken) == 1
        assert "কাউন্টারে" in stub_speak_and_tools.spoken[0]


class TestTransportParity:
    """main.py and main_pcm.py must dispatch test_duration identically --
    main_pcm.py is a GENERATED file (tools/make_pcm_variant.py); this
    guards against the same kind of hand-edit drift test_test_sample_
    intent.py's own module docstring documents having happened once
    already for test_sample. Reads both files directly off disk rather
    than `import main` -- mirrors tests/test_doctor_schedule_dispatch.py's
    own TestTransportParity, which uses this exact pattern deliberately:
    a bare `import main` risks colliding in sys.modules with clinic-api's
    own main.py (also importable as top-level "main" under some test
    orderings/sys.path states), a real, confirmed collision hit while
    building this fix."""

    def test_main_dot_py_has_the_test_duration_branch(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "test_duration"' in src
        assert "test_duration_reply" in src

    def test_main_pcm_dot_py_has_the_test_duration_branch(self):
        src = open("main_pcm.py", encoding="utf-8").read()
        assert 'intent == "test_duration"' in src
        assert "test_duration_reply" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
