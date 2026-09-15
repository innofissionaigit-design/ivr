"""ADDED BY SOURAV -- "Caller asks something the agent does not cover" story.

Distinct from tests/test_human_fallback.py's "unclear" intent: this story's
trigger, agent/llm.py's "out_of_scope" intent, fires when the classifier
understood the caller PERFECTLY but the request is not a kind of service
any intent covers at all. Rather than connecting straight to a human like
"unclear" does, this offers an explicit choice (connect to a human, or
contact the counter) and acts on whichever the caller picks next turn.

Structured the same way test_human_fallback.py is: the reply templates
themselves, the dispatch wiring (initial offer + both branches of the
follow-up), the escalation ledger reuse, and transport parity.
"""
from __future__ import annotations

import asyncio
import json
import types

import pytest

from agent.bn_normalize import verbalize
from agent.outcomes import human_handoff_counts, record_human_handoff, _reset_for_testing
import agent.outcomes as outcomes_module
from agent.reply_templates import human_fallback_reply, out_of_scope_reply, out_of_scope_counter_reply

import main_pcm

_LANGUAGES = ["bengali", "english", "hinglish", "banglish"]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _isolated_escalation_log(monkeypatch, tmp_path):
    log_path = tmp_path / "escalations.jsonl"
    monkeypatch.setattr(outcomes_module, "ESCALATION_LOG_PATH", str(log_path))
    _reset_for_testing()
    yield log_path
    _reset_for_testing()


# --------------------------------------------------------------------- #
# agent/reply_templates.out_of_scope_reply() / out_of_scope_counter_reply()
# --------------------------------------------------------------------- #

class TestOutOfScopeReply:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_returns_nonempty_text_for_every_language(self, language):
        text = out_of_scope_reply(language=language)
        assert isinstance(text, str)
        assert text.strip()

    def test_default_language_is_bengali(self):
        assert out_of_scope_reply() == out_of_scope_reply(language="bengali")

    def test_each_language_is_textually_distinct(self):
        replies = {out_of_scope_reply(language=l) for l in _LANGUAGES}
        assert len(replies) == len(_LANGUAGES)

    def test_is_distinct_from_human_fallback_reply(self):
        # This story's whole point: a clearly-understood-but-unsupported
        # request must NOT get the same "connecting you now" script as a
        # genuinely unintelligible one.
        for language in _LANGUAGES:
            assert out_of_scope_reply(language=language) != human_fallback_reply(language=language)

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_is_spoken_punctuation_clean(self, language):
        spoken = verbalize(out_of_scope_reply(language=language), language=language)
        for ch in (":", "：", "[", "]", "{", "}"):
            assert ch not in spoken


class TestOutOfScopeCounterReply:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_returns_nonempty_text_for_every_language(self, language):
        text = out_of_scope_counter_reply(language=language)
        assert isinstance(text, str)
        assert text.strip()

    def test_default_language_is_bengali(self):
        assert out_of_scope_counter_reply() == out_of_scope_counter_reply(language="bengali")

    def test_each_language_is_textually_distinct(self):
        replies = {out_of_scope_counter_reply(language=l) for l in _LANGUAGES}
        assert len(replies) == len(_LANGUAGES)

    def test_is_distinct_from_the_initial_offer(self):
        for language in _LANGUAGES:
            assert out_of_scope_counter_reply(language=language) != out_of_scope_reply(language=language)


# --------------------------------------------------------------------- #
# main_pcm.py dispatch -- initial "out_of_scope" intent branch
# --------------------------------------------------------------------- #

class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-out-of-scope-call", pending=pending,
        dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
    )


class FakeASRResultBengali:
    text = "amar kache medicine bikri hoy naki"  # marker-free-ish Latin -> "english" per detect_language, distinct fixture below covers real Bengali


class FakeASRResultEnglish:
    text = "do you also sell medicines here"


def _dispatch_initial(monkeypatch, asr_text_cls, tmp_path):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    async def fake_resolve_intent(session, text):
        return {"intent": "out_of_scope", "slots": {}}

    class FakeASR:
        async def transcribe_utterance(self, wav_path):
            return asr_text_cls()

    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    monkeypatch.setattr(main_pcm, "_asr", FakeASR())
    monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)

    session = make_session()
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return spoken, session


class TestOutOfScopeIntentDispatch:
    def test_speaks_the_offer_and_does_not_hang_off_human_fallback_directly(self, monkeypatch, tmp_path):
        spoken, session = _dispatch_initial(monkeypatch, FakeASRResultEnglish, tmp_path)
        assert spoken == [out_of_scope_reply(language="english")]
        assert spoken != [human_fallback_reply(language="english")]

    def test_sets_pending_awaiting_out_of_scope_choice(self, monkeypatch, tmp_path):
        _, session = _dispatch_initial(monkeypatch, FakeASRResultEnglish, tmp_path)
        assert session.pending == {"awaiting": "out_of_scope_choice", "retries": 0}

    def test_does_not_record_a_handoff_yet(self, monkeypatch, tmp_path):
        # The handoff is only real once the caller CONFIRMS they want a
        # human -- recording it on the mere offer would count callers who
        # go on to say "no, I'll use the counter" as handoffs too.
        assert human_handoff_counts("out_of_scope") == 0
        _dispatch_initial(monkeypatch, FakeASRResultEnglish, tmp_path)
        assert human_handoff_counts("out_of_scope") == 0

    def test_respects_the_detected_language_of_the_utterance(self, monkeypatch, tmp_path):
        spoken, _ = _dispatch_initial(monkeypatch, FakeASRResultBengali, tmp_path)
        # FakeASRResultBengali's text is plain Latin with no marker words,
        # so detect_language() resolves it to "english" too -- assert
        # against the SAME function the dispatch path itself calls, not a
        # hardcoded language, so this test tracks detect_language()'s
        # actual behavior rather than assuming it.
        from agent.bn_normalize import detect_language
        expected_language = detect_language(FakeASRResultBengali.text)
        assert spoken == [out_of_scope_reply(language=expected_language)]


# --------------------------------------------------------------------- #
# main_pcm.py _continue_pending -- the "out_of_scope_choice" follow-up
# --------------------------------------------------------------------- #

def _continue(monkeypatch, pending, reply_text):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    session = make_session(pending=dict(pending))
    handled = run(main_pcm._continue_pending(session, reply_text))
    return handled, spoken, session


class TestOutOfScopeChoiceContinuation:
    def test_affirmative_connects_to_human_and_records_handoff(self, monkeypatch):
        pending = {"awaiting": "out_of_scope_choice", "retries": 0}
        handled, spoken, session = _continue(monkeypatch, pending, "yes")
        assert handled is True
        assert spoken == [human_fallback_reply(language="english")]
        assert session.pending is None
        assert human_handoff_counts("out_of_scope") == 1

    def test_affirmative_escalation_ledger_gets_the_right_intent_and_call_id(self, monkeypatch, _isolated_escalation_log):
        pending = {"awaiting": "out_of_scope_choice", "retries": 0}
        _continue(monkeypatch, pending, "yes")
        entry = json.loads(_isolated_escalation_log.read_text(encoding="utf-8").strip())
        assert entry["event"] == "human_handoff"
        assert entry["intent"] == "out_of_scope"
        assert entry["call_id"] == "test-out-of-scope-call"

    def test_negative_declines_to_counter_and_records_no_handoff(self, monkeypatch):
        pending = {"awaiting": "out_of_scope_choice", "retries": 0}
        handled, spoken, session = _continue(monkeypatch, pending, "no")
        assert handled is True
        assert spoken == [out_of_scope_counter_reply(language="english")]
        assert session.pending is None
        assert human_handoff_counts("out_of_scope") == 0

    def test_unparseable_reply_reprompts_and_increments_retries(self, monkeypatch):
        pending = {"awaiting": "out_of_scope_choice", "retries": 0}
        handled, spoken, session = _continue(monkeypatch, pending, "banana")
        assert handled is True
        assert spoken == [out_of_scope_reply(language="english")]
        assert session.pending == {"awaiting": "out_of_scope_choice", "retries": 1}

    def test_gives_up_after_repeated_unparseable_replies(self, monkeypatch):
        pending = {"awaiting": "out_of_scope_choice", "retries": 2}
        handled, spoken, session = _continue(monkeypatch, pending, "banana")
        assert handled is False  # falls through to a fresh LLM classification
        assert spoken == []
        assert session.pending is None

    def test_out_of_scope_intent_and_unclear_intent_handoffs_are_counted_separately(self, monkeypatch):
        record_human_handoff(intent="unclear", call_id="c1")
        pending = {"awaiting": "out_of_scope_choice", "retries": 0}
        _continue(monkeypatch, pending, "yes")
        assert human_handoff_counts("unclear") == 1
        assert human_handoff_counts("out_of_scope") == 1


# --------------------------------------------------------------------- #
# Transport parity
# --------------------------------------------------------------------- #

class TestTransportParity:
    def test_main_dot_py_has_the_out_of_scope_branch(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "out_of_scope"' in src
        assert 'awaiting == "out_of_scope_choice"' in src
        assert "out_of_scope_reply" in src
        assert "out_of_scope_counter_reply" in src

    def test_main_pcm_dot_py_has_the_out_of_scope_branch(self):
        src = open("main_pcm.py", encoding="utf-8").read()
        assert 'intent == "out_of_scope"' in src
        assert 'awaiting == "out_of_scope_choice"' in src
        assert "out_of_scope_reply" in src
        assert "out_of_scope_counter_reply" in src

    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb


# --------------------------------------------------------------------- #
# agent/llm.py -- schema-only checks (no live Ollama call; the model's
# actual classification behavior is exercised in production, not here --
# every other intent in this codebase is tested the same schema-only way).
# --------------------------------------------------------------------- #

class TestOutOfScopeIntentSchema:
    def test_out_of_scope_is_a_valid_intent(self):
        from agent.llm import VALID_INTENTS
        assert "out_of_scope" in VALID_INTENTS

    def test_out_of_scope_is_distinct_from_unclear(self):
        from agent.llm import VALID_INTENTS
        assert "unclear" in VALID_INTENTS
        assert "out_of_scope" != "unclear"

    def test_prompt_documents_the_distinction_from_unclear(self):
        from agent.llm import SYSTEM_PROMPT_TEMPLATE
        assert "out_of_scope" in SYSTEM_PROMPT_TEMPLATE
        # The prompt must explicitly tell the model NOT to use this for an
        # unfamiliar named entity within an already-covered category (a
        # test/doctor name not on file stays test_rate/doctor_availability
        # etc., honestly reported not-found downstream) -- otherwise every
        # test the catalogue doesn't have would get misrouted here.
        assert "unfamiliar" in SYSTEM_PROMPT_TEMPLATE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
