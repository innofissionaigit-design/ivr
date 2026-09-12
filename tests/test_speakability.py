"""E12-S2: a reply with a known hole in it never reaches a caller.

THE TWO TESTS THAT MATTER MOST ARE THE DATA-DRIVEN ONES.
test_every_seeded_sample_type_is_speakable and its department twin do not
assert against strings written here; they walk the REAL catalogue constants in
clinic-api/seed.py, compose replies with the REAL templates, and assert nothing
survives that the tokenizer would drop. So adding a lab test whose sample type
has no spoken Bengali form fails the build, rather than reaching a caller as
silence. That is what "the measured failure cannot recur" has to mean in a
repository -- a rule that holds for data nobody has written yet.

Everything here is pure: no GPU, no model, no network. tests/conftest.py stubs
the ASR stack only for the two tests that import main.py.
"""
# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
from __future__ import annotations

import ast
import asyncio
import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import speakability, turn_log  # noqa: E402
from agent.bn_normalize import unspeakable_spans, verbalize  # noqa: E402
# test_rate_reply is aliased: pytest collects any module-level name starting
# with test_ as a test case, and would try to run the template as one.
from agent.reply_templates import (  # noqa: E402
    UNSPEAKABLE_ESCALATION, doctors_by_department_reply,
    test_rate_reply as rate_reply,
)
from agent.tts import PREWARM_LINES, TTSClient, UnspeakableReply  # noqa: E402

_SEED = pathlib.Path(__file__).resolve().parents[1] / "clinic-api" / "seed.py"


def _seed_literal(name: str):
    """Read one constant out of seed.py without importing it.

    seed.py imports SQLAlchemy models, which would make this whole suite need
    a database driver to read two list literals. ast.literal_eval instead:
    no import, no exec, and it still reads the single source of truth rather
    than a copy that could drift away from it.
    """
    tree = ast.parse(_SEED.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {_SEED}")


LAB_TESTS = _seed_literal("LAB_TESTS")
DEPARTMENT_ALIASES = _seed_literal("DEPARTMENT_ALIASES")

# A Latin word with no spoken form, used wherever a test needs the BLOCKED
# path. Not "Blood" or "Imaging": those are in _LATIN_SPOKEN_BN now, so a test
# written against them would quietly stop testing the block the day their entry
# landed -- which is exactly what happened on the first run of this file.
UNCOVERED_LATIN = "Radiology"


# --------------------------------------------------------------- criterion 3

@pytest.mark.parametrize("name,aliases,rate,sample,hours", LAB_TESTS)
def test_every_seeded_sample_type_is_speakable(name, aliases, rate, sample, hours):
    result = {"found": True, "test_name": name, "test_name_bn": aliases[0],
              "rate_inr": rate, "sample_type": sample, "report_time_hours": hours}
    verdict = speakability.check(rate_reply({"test_name": aliases[0]}, result))
    assert verdict.state == speakability.SPEAKABLE, (
        f"{name!r} (sample_type={sample!r}) would reach the caller with "
        f"{list(verdict.dropped)} missing. Add a spoken form to "
        f"agent/bn_normalize.py's _LATIN_SPOKEN_BN, or fix the seed value."
    )


@pytest.mark.parametrize("department", sorted(DEPARTMENT_ALIASES))
def test_every_seeded_department_is_speakable(department):
    """The department name is the SUBJECT of the listing sentence, so dropping
    it left the caller with '...বিভাগে ডাঃ সেন আছেন' -- a sentence missing the
    thing they asked about. This held for all eight departments."""
    result = {"found": True, "department": department,
              "department_bn": DEPARTMENT_ALIASES[department][0], "date": None,
              "doctors": [{"name": "Dr. A Sen", "doctor_name_bn": "সেন"}]}
    verdict = speakability.check(doctors_by_department_reply({}, result))
    assert verdict.state == speakability.SPEAKABLE, list(verdict.dropped)


def test_the_measured_colon_failure_is_blocked_not_synthesised():
    """The failure this story is named after, asserted directly.

    'Sample: Blood' now renders (রক্ত is in the table), so the literal example
    from the docs would pass and prove nothing. The assertion that carries the
    criterion is about an UNCOVERED value: it must be BLOCKED, not spoken with
    a hole after the colon.
    """
    covered = speakability.check("রেট চারশো টাকা। স্যাম্পল: Blood।")
    assert covered.state == speakability.SPEAKABLE
    assert "রক্ত" in covered.spoken

    uncovered = speakability.check("রেট চারশো টাকা। স্যাম্পল: Radiology।")
    assert uncovered.state == speakability.BLOCKED
    assert uncovered.dropped == ("Radiology",)


# ------------------------------------------------------------ the detector

def test_single_latin_letter_is_reported():
    """A lone 'T' is dropped by the tokenizer exactly as completely as 'TSH'.
    The old len>1 filter reported only the latter, so a gate built on it would
    have passed a sentence that still loses two characters."""
    spoken = verbalize("থাইরয়েড প্রোফাইল (T3 T4 TSH)")
    assert unspeakable_spans(spoken) == ["T", "T", "TSH"]


def test_phrase_entries_win_over_shorter_keys():
    """Longest-key-first is load-bearing, not tidiness: 'swab' matching inside
    'cervical smear'-style phrases would rewrite half and strand the rest as a
    fresh unspeakable span -- a rewrite that manufactures the defect."""
    assert speakability.check("স্যাম্পল Cervical Smear।").state == speakability.SPEAKABLE


# ------------------------------------------------------------- the verdict

def test_unchecked_is_not_a_pass():
    """A non-Bengali matrix reports UNCHECKED, never SPEAKABLE. Same discipline
    as ASRResult.decoder_agreement=None: when nothing was examined, say so."""
    verdict = speakability.check("The rate is four hundred rupees.", language="en-IN")
    assert verdict.state == speakability.UNCHECKED
    assert not verdict.is_blocked
    assert verdict.dropped == ()

    # ...and the same text under the language this detector DOES model.
    assert speakability.check("The rate is four hundred rupees.").is_blocked


def test_bengali_variants_are_still_checked():
    for lang in ("bn", "bn-IN", "BN-in"):
        assert speakability.check(UNCOVERED_LATIN, language=lang).is_blocked


def test_verdict_is_frozen():
    verdict = speakability.check(UNCOVERED_LATIN)
    with pytest.raises(Exception):
        verdict.state = speakability.SPEAKABLE


# ------------------------------------------------------------- canned lines

def test_canned_lines_are_speakable():
    """Includes UNSPEAKABLE_ESCALATION, which is the recursion guard: _speak()
    falls back to it when a reply is blocked, so it must never be blockable."""
    TTSClient.assert_canned_lines_speakable()
    assert UNSPEAKABLE_ESCALATION in PREWARM_LINES


# ------------------------------------------- the gate, at the choke point

class _SpyResponse:
    content = b"RIFF....WAVEfake"

    def raise_for_status(self):
        return None


class _SpyHTTP:
    def __init__(self):
        self.posts = []

    async def post(self, url, json=None):
        self.posts.append(json)
        return _SpyResponse()


def _client_with_spy(monkeypatch, *, enforce: bool):
    monkeypatch.setattr("agent.tts.SPEAKABILITY_ENFORCE", enforce)
    client = TTSClient()
    spy = _SpyHTTP()
    client._client = spy
    return client, spy


def test_blocked_reply_never_reaches_the_synthesizer(monkeypatch):
    client, spy = _client_with_spy(monkeypatch, enforce=True)
    with pytest.raises(UnspeakableReply) as excinfo:
        asyncio.run(client.synthesize(f"স্যাম্পল: {UNCOVERED_LATIN}।"))
    assert excinfo.value.dropped == (UNCOVERED_LATIN,)
    assert spy.posts == [], "a blocked reply was sent for synthesis anyway"
    assert client.stats["unspeakable_blocked"] == 1


def test_shadow_mode_counts_without_blocking(monkeypatch):
    """Shadow mode must still MEASURE. A mode that neither blocks nor counts
    is just the old warning with extra ceremony."""
    client, spy = _client_with_spy(monkeypatch, enforce=False)
    asyncio.run(client.synthesize(f"স্যাম্পল: {UNCOVERED_LATIN}।"))
    assert len(spy.posts) == 1
    assert client.stats["unspeakable_blocked"] == 1


def test_speakable_reply_is_synthesized_verbalized(monkeypatch):
    client, spy = _client_with_spy(monkeypatch, enforce=True)
    asyncio.run(client.synthesize("রেট 450 টাকা।"))
    assert client.stats["unspeakable_blocked"] == 0
    assert "চারশো পঞ্চাশ" in spy.posts[0]["text"], "numbers must reach TTS as words"


# ------------------------------------------------------------------ export

def test_unspeakable_event_is_recorded_with_its_spans(tmp_path, monkeypatch):
    path = tmp_path / "turn_signal.jsonl"
    monkeypatch.setattr(turn_log, "TURN_LOG_PATH", str(path))
    turn_log.record_unspeakable("abc123", 4, ("Imaging",), enforced=False)

    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["event"] == turn_log.EVENT_UNSPEAKABLE
    assert row["dropped"] == ["Imaging"]
    # enforced=False is what separates "the caller heard the hole" from
    # "the caller was escalated" in the export.
    assert row["enforced"] is False
    assert (row["call_id"], row["turn"]) == ("abc123", 4)


def test_export_failure_never_raises(monkeypatch):
    """Telemetry must not break a call: a read-only mount is a reason to lose
    a log line, not to drop a caller."""
    monkeypatch.setattr(turn_log, "TURN_LOG_PATH", os.path.join("\x00bad", "x.jsonl"))
    turn_log.record_unspeakable("abc123", 1, ("Imaging",), enforced=True)
