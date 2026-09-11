"""ADDED BY SOURAV -- "Caller asks how to prepare for a test" story's
bundled human_fallback config (lab_tests_with_fallback_config sample
file's voice_agent_config.human_fallback block).

The config: trigger_condition "query_unresolved_or_low_confidence",
action "transfer_to_human_agent", fallback_responses per language. This
codebase has no telephony transfer capability of any kind (see
agent/reply_templates.human_fallback_reply()'s and agent/outcomes.
record_human_handoff()'s own module comments for the full reasoning), so
the action is honestly split into what IS buildable:

  1. agent/reply_templates.human_fallback_reply() -- the business's own
     "connecting you to an expert" script, spoken verbatim per language.
  2. agent/outcomes.record_human_handoff() -- an escalation-ledger entry
     (the SAME JSONL ledger record_insufficient_verified_information()
     already writes to, distinguished by "event": "human_handoff"), a
     FOURTH outcome alongside the three agent/outcomes.py's own module
     docstring already lists.
  3. main.py/main_pcm.py's existing "unclear" intent branch (the one
     real, already-existing signal for "query unresolved") now calls
     both of the above instead of a fixed Bengali-only "please repeat"
     line with no language selection.

Structured as four sections: the reply template itself, the ledger
(mirrors tests/test_insufficient_verified_information.py's own
TestRecordAndCounts/TestEscalationLedger structure), the dispatch wiring,
and transport parity.
"""
from __future__ import annotations

import asyncio
import json
import types

import pytest

from agent.bn_normalize import verbalize
from agent.outcomes import (
    human_handoff_counts,
    insufficient_verified_information_counts,
    record_human_handoff,
    record_insufficient_verified_information,
    _reset_for_testing,
)
from agent.reply_templates import human_fallback_reply
import agent.outcomes as outcomes_module

import main_pcm

_LANGUAGES = ["bengali", "english", "hinglish", "banglish"]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _isolated_escalation_log(monkeypatch, tmp_path):
    """Same isolation discipline as tests/test_insufficient_verified_
    information.py's own fixture -- agent.outcomes state is module-level
    and must not leak between tests."""
    log_path = tmp_path / "escalations.jsonl"
    monkeypatch.setattr(outcomes_module, "ESCALATION_LOG_PATH", str(log_path))
    _reset_for_testing()
    yield log_path
    _reset_for_testing()


# --------------------------------------------------------------------- #
# agent/reply_templates.human_fallback_reply()
# --------------------------------------------------------------------- #

class TestHumanFallbackReply:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_returns_nonempty_text_for_every_language(self, language):
        text = human_fallback_reply(language=language)
        assert isinstance(text, str)
        assert text.strip()

    def test_matches_the_business_supplied_source_text_exactly(self):
        # Sourced verbatim from lab_tests_with_fallback_config's
        # voice_agent_config.human_fallback.fallback_responses -- stored
        # as given, never recomposed (same discipline as clinic-api/
        # seed.py's LAB_TEST_ADVISORIES scripts).
        assert human_fallback_reply(language="english") == \
            "I understand. Let me connect you with one of our experts right away."
        assert human_fallback_reply(language="hinglish") == \
            "Acha samjh gaya! Mai aapko humare expert ke saath connect kar deta hoon."
        assert human_fallback_reply(language="banglish") == \
            "Acha, bujhte perechi! Ami apnake amader ekjon expert-er sathe connect kore dichi."
        assert human_fallback_reply(language="bengali") == \
            "আচ্ছা, বুঝতে পেরেছি! আমি আপনাকে আমাদের একজন এক্সপার্টের সাথে কানেক্ট করে দিচ্ছি।"

    def test_default_language_is_bengali(self):
        assert human_fallback_reply() == human_fallback_reply(language="bengali")

    def test_each_language_is_textually_distinct(self):
        replies = {human_fallback_reply(language=l) for l in _LANGUAGES}
        assert len(replies) == len(_LANGUAGES)

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_is_spoken_punctuation_clean(self, language):
        spoken = verbalize(human_fallback_reply(language=language), language=language)
        for ch in (":", "：", "[", "]", "{", "}"):
            assert ch not in spoken


# --------------------------------------------------------------------- #
# agent/outcomes.record_human_handoff() / human_handoff_counts()
# --------------------------------------------------------------------- #

class TestRecordAndCounts:
    def test_count_starts_at_zero(self):
        assert human_handoff_counts("unclear") == 0

    def test_recording_increments_the_named_intent_only(self):
        record_human_handoff(intent="unclear", call_id="call-1")
        assert human_handoff_counts("unclear") == 1
        assert human_handoff_counts("test_rate") == 0

    def test_recording_multiple_times_accumulates(self):
        for _ in range(3):
            record_human_handoff(intent="unclear")
        assert human_handoff_counts("unclear") == 3

    def test_counts_with_no_argument_returns_full_snapshot(self):
        record_human_handoff(intent="unclear")
        snapshot = human_handoff_counts()
        assert snapshot == {"unclear": 1}

    def test_snapshot_is_a_copy_not_the_live_dict(self):
        record_human_handoff(intent="unclear")
        snapshot = human_handoff_counts()
        snapshot["unclear"] = 999
        assert human_handoff_counts("unclear") == 1

    def test_call_id_is_optional(self):
        record_human_handoff(intent="unclear")  # must not raise

    def test_kept_entirely_separate_from_insufficient_verified_information_counts(self):
        # The whole reason _handoff_counts is a SEPARATE dict from
        # _counts: these are two different outcomes, and merging them
        # would make both numbers meaningless.
        record_human_handoff(intent="unclear")
        record_human_handoff(intent="unclear")
        record_insufficient_verified_information(intent="unclear", field="x", reason="y")
        assert human_handoff_counts("unclear") == 2
        assert insufficient_verified_information_counts("unclear") == 1


class TestEscalationLedger:
    """Same shared JSONL ledger record_insufficient_verified_information()
    already writes to -- distinguished by "event", not a separate file."""

    def test_writes_one_json_line(self, _isolated_escalation_log):
        record_human_handoff(intent="unclear", call_id="call-42")
        lines = _isolated_escalation_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "human_handoff"
        assert entry["intent"] == "unclear"
        assert entry["reason"] == "query_unresolved_or_low_confidence"
        assert entry["call_id"] == "call-42"
        assert "timestamp" in entry and entry["timestamp"]

    def test_shares_the_ledger_with_insufficient_verified_information_distinguished_by_event(
        self, _isolated_escalation_log
    ):
        record_human_handoff(intent="unclear", call_id="call-1")
        record_insufficient_verified_information(
            intent="book_appointment", field="confirmation_id", reason="missing_after_success",
            call_id="call-2",
        )
        lines = _isolated_escalation_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        entries = [json.loads(l) for l in lines]
        assert entries[0]["event"] == "human_handoff"
        assert "event" not in entries[1]  # the pre-existing record shape is untouched
        assert entries[1]["intent"] == "book_appointment"

    def test_appends_rather_than_overwrites(self, _isolated_escalation_log):
        record_human_handoff(intent="unclear", call_id="call-1")
        record_human_handoff(intent="unclear", call_id="call-2")
        lines = _isolated_escalation_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_timestamp_is_iso8601_and_utc(self, _isolated_escalation_log):
        import datetime
        record_human_handoff(intent="unclear")
        entry = json.loads(_isolated_escalation_log.read_text(encoding="utf-8").strip())
        parsed = datetime.datetime.fromisoformat(entry["timestamp"])
        assert parsed.tzinfo is not None


# --------------------------------------------------------------------- #
# main_pcm.py dispatch -- the "unclear" intent branch
# --------------------------------------------------------------------- #

class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session():
    return types.SimpleNamespace(
        call_id="test-handoff-call",
        pending=None,
        dispatch_lock=asyncio.Lock(),
        send_json=_AsyncNoOp(),
    )


class FakeASRResultBengali:
    text = "কিছু একটা বুঝতে পারছি না অদ্ভুত কথা"


class FakeASRResultEnglish:
    text = "some garbled unintelligible noise"


def _dispatch(monkeypatch, asr_text_cls, tmp_path):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    async def fake_resolve_intent(session, text):
        return {"intent": "unclear", "slots": {}}

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


class TestUnclearIntentDispatch:
    def test_speaks_the_human_fallback_reply_not_the_old_fixed_line(
        self, monkeypatch, tmp_path
    ):
        spoken, session = _dispatch(monkeypatch, FakeASRResultBengali, tmp_path)
        assert spoken == [human_fallback_reply(language="bengali")]
        # The OLD behaviour this replaces -- must not still be reachable.
        assert spoken != ["দুঃখিত, বুঝতে পারিনি। আবার একটু বলবেন?"]
        assert session.pending is None

    def test_records_a_human_handoff_for_the_unclear_intent(self, monkeypatch, tmp_path):
        assert human_handoff_counts("unclear") == 0
        _dispatch(monkeypatch, FakeASRResultBengali, tmp_path)
        assert human_handoff_counts("unclear") == 1

    def test_escalation_ledger_gets_the_real_call_id(self, monkeypatch, tmp_path, _isolated_escalation_log):
        _dispatch(monkeypatch, FakeASRResultBengali, tmp_path)
        entry = json.loads(_isolated_escalation_log.read_text(encoding="utf-8").strip())
        assert entry["call_id"] == "test-handoff-call"
        assert entry["event"] == "human_handoff"

    def test_respects_the_detected_language_of_the_utterance(self, monkeypatch, tmp_path):
        # UPDATED BY SOURAV -- this is the actual bug fixed alongside
        # wiring in human_fallback: the OLD line was fixed Bengali
        # regardless of what language the caller was speaking. An
        # English-sounding garbled utterance must now get the English
        # fallback script, not a Bengali one.
        spoken, _ = _dispatch(monkeypatch, FakeASRResultEnglish, tmp_path)
        assert spoken == [human_fallback_reply(language="english")]


class TestTransportParity:
    def test_main_dot_py_has_the_updated_unclear_branch(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "unclear"' in src
        assert "human_fallback_reply" in src
        assert "record_human_handoff" in src

    def test_main_pcm_dot_py_has_the_updated_unclear_branch(self):
        src = open("main_pcm.py", encoding="utf-8").read()
        assert 'intent == "unclear"' in src
        assert "human_fallback_reply" in src
        assert "record_human_handoff" in src

    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
