"""_speak()'s blocked path: what the caller hears, and what the record says.

These are the only tests here that import main.py, which is why
tests/conftest.py exists. Nothing in them touches a model, a socket or the
network -- the TTS client and the session are both stubs, and what is being
asserted is purely which branch ran and in what order.

The ordering assertion is the point of the file. Synthesis was moved ahead of
the transcript send in _speak() specifically so that a blocked reply cannot
leave the transcript claiming the agent said something the caller never heard;
a test that only checked the audio would pass just as happily with the bug
back in place.
"""
# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from agent import turn_log  # noqa: E402
from agent.reply_templates import UNSPEAKABLE_ESCALATION  # noqa: E402
from agent.tts import UnspeakableReply  # noqa: E402

BLOCKED_TEXT = "কার্ডিওলজি বিভাগে ডাঃ Radiology আছেন।"
DROPPED = ("Radiology",)


class _StubTTS:
    """Raises for BLOCKED_TEXT, returns bytes for anything else."""

    def __init__(self, *, enforce: bool):
        self.enforce = enforce
        self.synthesized: list[str] = []

    async def synthesize(self, text, speech_rate="default", language=None):
        self.synthesized.append(text)
        if text == BLOCKED_TEXT and self.enforce:
            raise UnspeakableReply(DROPPED, text)
        return b"WAV:" + text.encode("utf-8")

    @staticmethod
    def fallback_audio(reason):
        return b"FALLBACK:" + reason.encode("utf-8")


class _StubSession:
    call_id = "t3st"
    utt_seq = 7

    def __init__(self):
        self.call_state = main.call_state_mod.build()
        self.transcript: list[tuple[str, str]] = []
        self.audio: list[bytes] = []
        self.gate_holds: list[float] = []

    async def send_json(self, sender, text):
        self.transcript.append((sender, text))

    async def send_audio(self, wav):
        self.audio.append(wav)

    def hold_gate_for(self, seconds):
        self.gate_holds.append(seconds)


@pytest.fixture
def logfile(tmp_path, monkeypatch):
    path = tmp_path / "turn_signal.jsonl"
    monkeypatch.setattr(turn_log, "TURN_LOG_PATH", str(path))
    return path


def _rows(logfile):
    if not logfile.exists():
        return []
    return [json.loads(line) for line in logfile.read_text(encoding="utf-8").splitlines()]


def test_blocked_reply_escalates_and_the_transcript_agrees(monkeypatch, logfile):
    tts = _StubTTS(enforce=True)
    monkeypatch.setattr(main, "_tts", tts)
    monkeypatch.setattr(main, "SPEAKABILITY_ENFORCE", True)
    session = _StubSession()

    asyncio.run(main._speak(session, BLOCKED_TEXT))

    # The caller heard the escalation, not the reply with the hole in it.
    assert tts.synthesized == [BLOCKED_TEXT, UNSPEAKABLE_ESCALATION]
    assert session.audio == [b"WAV:" + UNSPEAKABLE_ESCALATION.encode("utf-8")]

    # And the transcript says the same thing. If synthesis ever moves back
    # after this send, the pane will show BLOCKED_TEXT and this fails.
    assert session.transcript == [("AI", UNSPEAKABLE_ESCALATION)]

    (row,) = [r for r in _rows(logfile) if r["event"] == turn_log.EVENT_UNSPEAKABLE]
    assert row["dropped"] == list(DROPPED)
    assert row["enforced"] is True
    assert (row["call_id"], row["turn"]) == (session.call_id, session.utt_seq)


def test_shadow_mode_speaks_the_hole_but_records_it(monkeypatch, logfile):
    """Shadow mode is today's behaviour plus a measurement -- the caller still
    hears the hole. That is the cost of finding out how often this fires
    before diverting anyone, and it is why the enforced flag is exported."""
    tts = _StubTTS(enforce=False)
    monkeypatch.setattr(main, "_tts", tts)
    monkeypatch.setattr(main, "SPEAKABILITY_ENFORCE", False)
    session = _StubSession()

    asyncio.run(main._speak(session, BLOCKED_TEXT))

    assert tts.synthesized == [BLOCKED_TEXT]
    assert session.transcript == [("AI", BLOCKED_TEXT)]

    (row,) = [r for r in _rows(logfile) if r["event"] == turn_log.EVENT_UNSPEAKABLE]
    assert row["enforced"] is False


def test_a_speakable_reply_is_untouched(monkeypatch, logfile):
    """The gate must be invisible on the overwhelming majority of turns."""
    tts = _StubTTS(enforce=True)
    monkeypatch.setattr(main, "_tts", tts)
    monkeypatch.setattr(main, "SPEAKABILITY_ENFORCE", True)
    session = _StubSession()

    asyncio.run(main._speak(session, "রেট চারশো টাকা।"))

    assert session.transcript == [("AI", "রেট চারশো টাকা।")]
    assert session.audio == [b"WAV:" + "রেট চারশো টাকা।".encode("utf-8")]
    assert _rows(logfile) == []


def test_a_vocoder_outage_still_gets_the_busy_clip(monkeypatch, logfile):
    """The generic handler must survive the new one in front of it: a transport
    failure is an outage and still answers with the pre-recorded apology, not
    with the counter referral meant for our own content defects."""
    class _DeadTTS(_StubTTS):
        async def synthesize(self, text, speech_rate="default", language=None):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(main, "_tts", _DeadTTS(enforce=True))
    monkeypatch.setattr(main, "SPEAKABILITY_ENFORCE", True)
    session = _StubSession()

    asyncio.run(main._speak(session, "রেট চারশো টাকা।", fallback_reason="llm_failure"))

    assert session.audio == [b"FALLBACK:llm_failure"]
    assert session.transcript == [("AI", "রেট চারশো টাকা।")]
