"""Tests for story "Caller asks what sample is needed" (Epic: Conversation
-- Information and Enquiry, owner: Saurav) covering the parts of this
story that are NOT reply-template text -- those are in
test_reply_templates_fidelity.py alongside the other reply_templates.py
tests. This file covers the three other places this story touched:

  1. agent/llm.py: the new "test_sample" intent is a valid classification
     target, distinct from "test_rate" (which still bundles price+sample+
     duration together).
  2. agent/semantic_cache.py: "test_sample" is registered in
     _REQUIRED_ENTITY_FOR_INTENT, exactly like "test_rate" already was --
     an extraction missing test_name must never enter the L2 semantic-
     cache index, or a later, differently-worded caller could be served a
     stale, entity-less answer (see that dict's own comment for the bug
     this prevents).
  3. main_pcm.py: the new `elif intent == "test_sample":` dispatch branch
     -- same tool call as test_rate (clinic-api's test lookup already
     returns sample_type on every call), but rendered through
     sample_type_reply() instead of test_rate_reply(), and gated on
     test_name being present exactly like test_rate is.

Dispatch-level tests follow tests/test_booking_readback.py's established
pattern: no pytest-asyncio (the project only pins plain pytest), ASR and
intent extraction stubbed via monkeypatch, driven with asyncio.run()
inside ordinary sync test functions. See tests/conftest.py for why
main_pcm.py imports cleanly here without the real ASR/VAD/LLM stack.
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

class TestTestSampleIsAValidIntent:
    def test_registered_in_valid_intents(self):
        assert "test_sample" in VALID_INTENTS

    def test_distinct_from_test_rate(self):
        # The whole point of the new intent: a caller asking ONLY about
        # the sample must not be classified the same way as one asking
        # about price (which still bundles rate+sample+duration).
        assert "test_rate" in VALID_INTENTS
        assert "test_sample" != "test_rate"

    def test_validate_accepts_a_well_formed_test_sample_payload(self):
        data = {
            "intent": "test_sample",
            "slots": {
                "test_name": "CBC", "doctor_name": None, "department": None,
                "date": None, "time_slot": None, "patient_name": None, "phone": None,
            },
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_rejects_an_intent_not_in_the_enum(self):
        # Regression guard: adding "test_sample" must not have loosened
        # _validate() into accepting arbitrary strings.
        data = {
            "intent": "test_smell",
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

class TestTestSampleRequiredEntity:
    """_REQUIRED_ENTITY_FOR_INTENT / _is_l2_eligible() -- the safety net
    that stops an extraction missing its defining slot from ever being
    reused for a later, differently-worded utterance. Registered here for
    "test_sample" the same way it was already registered for "test_rate",
    "doctor_availability" and "doctors_by_department" -- proactively
    identified as necessary (not requested), since a new single-entity
    intent left OUT of this dict would silently reopen exactly the
    caching bug this dict exists to close."""

    def test_registered_with_test_name_as_the_required_entity(self):
        assert _REQUIRED_ENTITY_FOR_INTENT.get("test_sample") == "test_name"

    def test_extraction_with_test_name_is_l2_eligible(self):
        value = {"intent": "test_sample", "slots": {"test_name": "CBC"}}
        assert SemanticCache._is_l2_eligible(value) is True

    def test_extraction_missing_test_name_is_not_l2_eligible(self):
        value = {"intent": "test_sample", "slots": {"test_name": None}}
        assert SemanticCache._is_l2_eligible(value) is False

    def test_behaves_identically_to_test_rate_for_the_same_slots(self):
        # test_sample reuses test_rate's exact clinic-api call, so it
        # should be governed by the exact same cache-safety rule.
        for intent in ("test_rate", "test_sample"):
            assert SemanticCache._is_l2_eligible(
                {"intent": intent, "slots": {"test_name": "CBC"}}
            ) is True
            assert SemanticCache._is_l2_eligible(
                {"intent": intent, "slots": {}}
            ) is False


# --------------------------------------------------------------------- #
# main_pcm.py dispatch (mirrors tests/test_booking_readback.py's pattern)
# --------------------------------------------------------------------- #

class FakeToolsClient:
    """Records every get_test_rate() call -- the whole point of this
    branch is that test_sample and test_rate both call this SAME tool
    (clinic-api already returns sample_type on every test lookup; nothing
    new was added to the API for this story), only the reply-rendering
    function differs."""

    def __init__(self, response=None):
        self.get_test_rate_calls = []
        self._response = response or {
            "found": True, "test_name": "CBC", "test_name_bn": "সিবিসি",
            "rate_inr": 850, "sample_type": "Blood", "report_time_hours": 24,
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
    text = "ignored -- _resolve_intent is stubbed directly below"


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
        return {"intent": "test_sample", "slots": slots}

    monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
    session = make_session(pending=None)
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")  # _dispatch_turn just os.remove()s this
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return session


class TestDispatchTestSample:
    def test_calls_get_test_rate_and_speaks_sample_type_reply(
        self, monkeypatch, stub_speak_and_tools, tmp_path
    ):
        from agent.reply_templates import sample_type_reply

        slots = {"test_name": "CBC"}
        _dispatch(monkeypatch, slots, tmp_path)

        assert stub_speak_and_tools.tools.get_test_rate_calls == ["CBC"]
        assert len(stub_speak_and_tools.spoken) == 1
        expected = sample_type_reply(slots, stub_speak_and_tools.tools._response)
        assert stub_speak_and_tools.spoken[0] == expected

    def test_reply_does_not_contain_the_rate(self, monkeypatch, stub_speak_and_tools, tmp_path):
        # The whole point of this story: a caller who asked ONLY about
        # the sample must not hear the bundled rate+sample+duration
        # answer test_rate gives.
        slots = {"test_name": "CBC"}
        _dispatch(monkeypatch, slots, tmp_path)
        assert "850" not in stub_speak_and_tools.spoken[0]

    def test_missing_test_name_asks_for_it_without_calling_the_tool(
        self, monkeypatch, stub_speak_and_tools, tmp_path
    ):
        from agent.reply_templates import missing_slot_prompt

        _dispatch(monkeypatch, {"test_name": None}, tmp_path)

        assert stub_speak_and_tools.tools.get_test_rate_calls == []
        assert stub_speak_and_tools.spoken == [missing_slot_prompt("test_sample", "test_name")]

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
        _dispatch(monkeypatch, {"test_name": "CBC"}, tmp_path)
        assert len(stub_speak_and_tools.spoken) == 1
        assert "কাউন্টারে" in stub_speak_and_tools.spoken[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
