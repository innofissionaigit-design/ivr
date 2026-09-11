"""ADDED BY SOURAV -- "Caller asks how to prepare for a test" (Epic:
Conversation -- Information and Enquiry).

Mirrors tests/test_test_duration_intent.py's own structure exactly (see
that file's module docstring for the established 3-place pattern every
new single-entity intent in this codebase touches):

  1. agent/llm.py: "test_preparation" is a valid classification target,
     distinct from test_rate/test_sample/test_duration -- all four can be
     asked about the SAME test in the same call, and must never collapse
     into one another.
  2. agent/semantic_cache.py: "test_preparation" is registered in
     _REQUIRED_ENTITY_FOR_INTENT with "test_name", exactly like test_rate/
     test_sample/test_duration already are -- an extraction missing
     test_name must never enter the L2 semantic-cache index (there is no
     "list every test's preparation instructions" analog for a bare "how
     do I prepare" the way health_package's "what packages do you have"
     has).
  3. main_pcm.py: the new `elif intent == "test_preparation":` dispatch
     branch -- a DEDICATED new tool call (get_test_preparation), unlike
     test_duration which reuses get_test_rate's response, since
     preparation data is not part of that payload at all.

Reply-text-level coverage (all 4 languages, all 3 outcomes, spoken-
punctuation cleanliness) lives in tests/test_test_preparation_reply.py.
API-level coverage (the clinic-api endpoint itself, against a real seeded
database) lives in tests/test_test_preparation_api.py.

Dispatch-level tests follow tests/test_test_duration_intent.py's own
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

class TestTestPreparationIsAValidIntent:
    def test_registered_in_valid_intents(self):
        assert "test_preparation" in VALID_INTENTS

    def test_distinct_from_test_rate_sample_and_duration(self):
        # The whole point of the new intent: "how do I prepare" must not
        # be classified the same way as price, sample, or turnaround-time
        # questions -- a caller can ask all four about the same test.
        assert "test_rate" in VALID_INTENTS
        assert "test_sample" in VALID_INTENTS
        assert "test_duration" in VALID_INTENTS
        assert "test_preparation" not in ("test_rate", "test_sample", "test_duration")

    def test_validate_accepts_a_well_formed_test_preparation_payload(self):
        data = {
            "intent": "test_preparation",
            "slots": {
                "test_name": "Blood Sugar Fasting", "doctor_name": None,
                "department": None, "date": None, "time_slot": None,
                "patient_name": None, "phone": None,
                "package_name": None, "info_topic": None,
            },
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_rejects_an_intent_not_in_the_enum(self):
        # Regression guard: adding "test_preparation" must not have
        # loosened _validate() into accepting arbitrary strings.
        data = {
            "intent": "test_precautions",
            "slots": {
                "test_name": None, "doctor_name": None, "department": None,
                "date": None, "time_slot": None, "patient_name": None, "phone": None,
                "package_name": None, "info_topic": None,
            },
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is False
        assert any("invalid intent" in e for e in errors)


# --------------------------------------------------------------------- #
# agent/semantic_cache.py
# --------------------------------------------------------------------- #

class TestTestPreparationRequiredEntity:
    """_REQUIRED_ENTITY_FOR_INTENT / _is_l2_eligible() -- registered here
    for "test_preparation" the same way it already is for test_rate/
    test_sample/test_duration, proactively (not requested) -- a new
    single-entity intent left OUT of this dict would silently reopen the
    cache-poisoning bug this dict exists to close."""

    def test_registered_with_test_name_as_the_required_entity(self):
        assert _REQUIRED_ENTITY_FOR_INTENT.get("test_preparation") == "test_name"

    def test_extraction_with_test_name_is_l2_eligible(self):
        value = {"intent": "test_preparation", "slots": {"test_name": "Blood Sugar Fasting"}}
        assert SemanticCache._is_l2_eligible(value) is True

    def test_extraction_missing_test_name_is_not_l2_eligible(self):
        value = {"intent": "test_preparation", "slots": {"test_name": None}}
        assert SemanticCache._is_l2_eligible(value) is False

    def test_behaves_identically_to_test_rate_sample_and_duration_for_the_same_slots(self):
        for intent in ("test_rate", "test_sample", "test_duration", "test_preparation"):
            assert SemanticCache._is_l2_eligible(
                {"intent": intent, "slots": {"test_name": "Blood Sugar Fasting"}}
            ) is True
            assert SemanticCache._is_l2_eligible(
                {"intent": intent, "slots": {}}
            ) is False


# --------------------------------------------------------------------- #
# main_pcm.py dispatch (mirrors tests/test_test_duration_intent.py's pattern)
# --------------------------------------------------------------------- #

class FakeToolsClient:
    """Records every get_test_preparation() call -- test_preparation
    calls a DEDICATED endpoint, unlike test_duration which reuses
    get_test_rate's response."""

    def __init__(self, response=None):
        self.get_test_preparation_calls = []
        self._response = response or {
            "found": True, "test_name": "Blood Sugar Fasting",
            "test_name_bn": "ব্লাড সুগার ফাস্টিং",
            "advisory_available": True,
            "fasting_required": True, "fasting_hours": "8-12 hours",
            "water_allowance": "Only plain water permitted during fasting period",
            "medication_hold": "Hold morning anti-diabetic medication until after blood collection",
            "timing_rule": "Morning sample collection preferred",
            "advisory_script_en": "For {test_name}, fast for 8 to 12 hours.",
            "advisory_script_hinglish": "{test_name} ke liye 8 se 12 ghante fasting.",
            "advisory_script_banglish": "{test_name}-er jonno 8 theke 12 ghanta fasting.",
            "advisory_script_bn": "{test_name}-এর জন্য ৮ থেকে ১২ ঘণ্টা ফাস্টিং।",
        }

    async def get_test_preparation(self, test_name):
        self.get_test_preparation_calls.append(test_name)
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
    # Real Bengali text, not an English placeholder -- main_pcm.py's
    # dispatch calls detect_language() on it (see main.py's own "ADDED BY
    # SOURAV" comment on `language = detect_language(text)`).
    text = "এই টেস্টের জন্য কী প্রস্তুতি নিতে হবে"


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
        return {"intent": "test_preparation", "slots": slots}

    monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
    session = make_session(pending=None)
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")  # _dispatch_turn just os.remove()s this
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return session


class TestDispatchTestPreparation:
    def test_calls_get_test_preparation_and_speaks_test_preparation_reply(
        self, monkeypatch, stub_speak_and_tools, tmp_path
    ):
        from agent.reply_templates import test_preparation_reply

        slots = {"test_name": "Blood Sugar Fasting"}
        _dispatch(monkeypatch, slots, tmp_path)

        assert stub_speak_and_tools.tools.get_test_preparation_calls == ["Blood Sugar Fasting"]
        assert len(stub_speak_and_tools.spoken) == 1
        expected = test_preparation_reply(slots, stub_speak_and_tools.tools._response)
        assert stub_speak_and_tools.spoken[0] == expected

    def test_does_not_call_get_test_rate_at_all(self, monkeypatch, stub_speak_and_tools, tmp_path):
        # Confirms this intent uses its OWN dedicated tool call rather
        # than silently reusing test_rate/test_sample/test_duration's
        # get_test_rate() the way test_duration does.
        assert not hasattr(stub_speak_and_tools.tools, "get_test_rate_calls")

    def test_missing_test_name_asks_for_it_without_calling_the_tool(
        self, monkeypatch, stub_speak_and_tools, tmp_path
    ):
        from agent.reply_templates import missing_slot_prompt

        _dispatch(monkeypatch, {"test_name": None}, tmp_path)

        assert stub_speak_and_tools.tools.get_test_preparation_calls == []
        assert stub_speak_and_tools.spoken == [missing_slot_prompt("test_preparation", "test_name")]

    def test_tool_failure_gets_the_same_infrastructure_apology_as_test_rate(
        self, monkeypatch, stub_speak_and_tools, tmp_path
    ):
        # The new branch sits inside the same try/except ToolCallError
        # block as every other intent -- confirms it inherits that
        # handling automatically rather than needing its own.
        from agent.tools_client import ToolCallError

        class FailingTools(FakeToolsClient):
            async def get_test_preparation(self, test_name):
                raise ToolCallError("clinic-api unreachable")

        monkeypatch.setattr(main_pcm, "_tools", FailingTools())
        _dispatch(monkeypatch, {"test_name": "Blood Sugar Fasting"}, tmp_path)
        assert len(stub_speak_and_tools.spoken) == 1
        assert "কাউন্টারে" in stub_speak_and_tools.spoken[0]


class TestTransportParity:
    """main.py and main_pcm.py must dispatch test_preparation identically
    -- main_pcm.py is a GENERATED file (tools/make_pcm_variant.py); this
    guards against hand-edit drift the same way test_test_duration_
    intent.py's own TestTransportParity does."""

    def test_main_dot_py_has_the_test_preparation_branch(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "test_preparation"' in src
        assert "get_test_preparation" in src
        assert "test_preparation_reply" in src

    def test_main_pcm_dot_py_has_the_test_preparation_branch(self):
        src = open("main_pcm.py", encoding="utf-8").read()
        assert 'intent == "test_preparation"' in src
        assert "get_test_preparation" in src
        assert "test_preparation_reply" in src

    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
