"""A caller who asked two things gets two answers.

story title: A multi-part question is answered in full
user story: As a caller who asked two things, I want both answered, so that
    I do not have to ask again.
acceptance criteria: Every answerable part of a turn is answered in the order
    asked, and any part that cannot be answered is explicitly addressed
    rather than dropped. Completeness is scored on a labelled multi-part set.

WHAT THE LABELLED SET SCORES, AND WHAT IT DOES NOT
---------------------------------------------------
It scores the DISPATCHER, with the extraction supplied. Given the parts a
turn contains, does every one of them get addressed, in the order asked?

It does NOT score whether the model finds both parts in the first place.
That needs Ollama, which this suite cannot run, and it is a different
measurement with a different owner -- E9-S4, intent and slot accuracy, which
is ABSENT. Claiming otherwise would be the more comfortable lie: completeness
would look solved while the half that actually reads the caller's words went
unmeasured.

The split is the same one agent/turn_log.py already makes between a signal
and its ground truth, and for the same reason: each half is measurable
somewhere, and pretending one file measures both is how a metric stops
meaning what its name says.

tests/data/multipart_bn.jsonl is HAND-BUILT, not transcribed from calls.
Forty labelled utterances written against the seeded catalogue. E9-S2 (a
golden set of real consented calls) replaces it; until then this is a
fixture set that is honest about being one.

Everything here is pure -- no GPU, no model, no network, no database.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import unicodedata

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from agent import answer_ledger, turn_parts  # noqa: E402
from agent.reply_templates import (  # noqa: E402
    DEFERRED_PART_BN, RESUMING_PART_BN, UNANSWERED_PART_BN,
    unanswered_part_prompt,
)

LABELLED = ROOT / "tests" / "data" / "multipart_bn.jsonl"

# What a healthy dispatcher must reach on the labelled set.
#
# Not 1.0, and the reason is in the set rather than in the code. Four of the
# 78 labelled parts are dropped ON PURPOSE -- a trailing thank-you, trailing
# ASR noise, the same question said twice, and a fourth part against a
# MAX_PARTS of three. So the ceiling on this file is 74/78 = 0.949, and a
# threshold of 1.0 could only be met by removing a rule that exists to stop
# the agent answering questions nobody asked.
#
# The floor sits just under the ceiling: a single real regression -- one part
# silently skipped -- takes it to 73/78 = 0.936 and fails. The four
# deductions are pinned individually by
# test_the_only_parts_the_set_loses_are_the_ones_we_chose_to_lose, so this
# number cannot be quietly lowered later to accommodate a fifth.
COMPLETENESS_FLOOR = 0.94
ORDER_FLOOR = 1.0


def _cases():
    for line in LABELLED.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


CASES = list(_cases())


def _says(phrase: str, sentence: str) -> bool:
    return unicodedata.normalize("NFC", phrase) in unicodedata.normalize("NFC", sentence)


# --------------------------------------------------------- normalisation

def test_the_set_loads_and_is_not_trivial():
    assert len(CASES) >= 40
    assert sum(1 for c in CASES if len(c["parts"]) > 1) >= 30, (
        "a multi-part set made mostly of single-part turns scores nothing")


def test_a_turn_with_no_parts_field_is_the_old_single_part_turn():
    """The fallback that lets this ship before the model is retrained. An
    extraction with no `parts` must produce exactly the agent that existed
    before this story, through the same code path."""
    data = {"intent": "test_rate", "slots": {"test_name": "সিবিসি"},
            "direct_reply_bn": None}
    parts = turn_parts.normalise(data)
    assert parts == [{"intent": "test_rate", "slots": {"test_name": "সিবিসি"},
                      "direct_reply_bn": None}]
    assert not turn_parts.is_multi(parts)


@pytest.mark.parametrize("broken", [None, "two", 7, {}, []])
def test_a_malformed_parts_field_degrades_instead_of_failing(broken):
    """A degraded extraction must answer one question, not zero. On a phone
    line that trade is not close."""
    data = {"intent": "test_rate", "slots": {"test_name": "সিবিসি"},
            "parts": broken}
    parts = turn_parts.normalise(data)
    assert len(parts) == 1 and parts[0]["intent"] == "test_rate"


def test_the_first_part_is_always_the_top_level():
    """The invariant every other consumer depends on. semantic_cache,
    fast_path, tool_outcome and answer_ledger all read data["intent"]; if
    parts[0] could disagree with it they would answer a different question
    than the dispatcher."""
    data = {"intent": "test_rate", "slots": {"test_name": "সিবিসি"},
            "parts": [{"intent": "doctor_availability", "slots": {"doctor_name": "সেন"}},
                      {"intent": "doctors_by_department", "slots": {"department": "অর্থো"}}]}
    parts = turn_parts.normalise(data)
    assert parts[0]["intent"] == "test_rate"
    assert parts[0]["slots"] == {"test_name": "সিবিসি"}


def test_a_trailing_thanks_is_not_a_second_question():
    """The failure mode this story could easily invent: the agent answering
    a greeting it was never asked, or apologising for noise at the end of a
    sentence."""
    for noise in ("smalltalk", "unclear"):
        data = {"intent": "test_rate", "slots": {"test_name": "সিবিসি"},
                "parts": [{"intent": "test_rate", "slots": {"test_name": "সিবিসি"}},
                          {"intent": noise, "slots": {}}]}
        assert len(turn_parts.normalise(data)) == 1, noise


def test_a_turn_that_is_only_smalltalk_still_works():
    """The rule above drops non-substantive parts -- unless that is all
    there is."""
    data = {"intent": "smalltalk", "slots": {}, "direct_reply_bn": "নমস্কার"}
    parts = turn_parts.normalise(data)
    assert len(parts) == 1 and parts[0]["intent"] == "smalltalk"


def test_the_same_question_twice_is_one_part():
    """Answering it twice reads the same price out in two consecutive
    sentences, which is worse than merging."""
    data = {"intent": "test_rate", "slots": {"test_name": "সিবিসি"},
            "parts": [{"intent": "test_rate", "slots": {"test_name": "সিবিসি"}},
                      {"intent": "test_rate", "slots": {"test_name": "সিবিসি"}}]}
    assert len(turn_parts.normalise(data)) == 1


def test_parts_are_capped():
    data = {"intent": "test_rate", "slots": {"test_name": "ক"},
            "parts": [{"intent": "test_rate", "slots": {"test_name": c}}
                      for c in "কখগঘঙ"]}
    assert len(turn_parts.normalise(data)) == turn_parts.MAX_PARTS == 3


def test_the_subject_is_the_callers_own_words():
    part = {"intent": "test_rate", "slots": {"test_name": "ইউরিক অ্যাসিড"}}
    assert turn_parts.subject_of(part) == "ইউরিক অ্যাসিড"
    assert turn_parts.subject_of({"intent": "unclear", "slots": {}}) is None


# --------------------------------------------------------- the dispatcher

class _Session:
    call_id = "mp01"
    utt_seq = 1

    def __init__(self):
        self.answer_ledger = answer_ledger.AnswerLedger()
        self.pending = None
        self.deferred = None
        self.call_state = main.call_state_mod.build()


@pytest.fixture
def wired(monkeypatch):
    """Replaces _answer_part with a scriptable double.

    The real one is 300 lines over four intents and three tool calls; what
    this story added is the LOOP around it, and testing the loop through the
    chain would be testing the chain. The double answers to a script keyed by
    intent, so every outcome ordering can be produced deliberately.
    """
    said: list[str] = []
    seen: list[dict] = []
    script: dict = {}

    async def _say(session, text, fallback_reason=None):
        said.append(text)

    async def _answer(session, text, part):
        seen.append(part)
        outcome = script.get(part["intent"], turn_parts.ANSWERED)
        said.append(f"<{part['intent']}>")
        if outcome == turn_parts.INTERACTIVE:
            session.pending = {"awaiting": "date", "slots": {},
                               "candidates": None, "offered_date": None,
                               "retries": 0}
        return outcome

    monkeypatch.setattr(main, "_speak", _say)
    monkeypatch.setattr(main, "_answer_part", _answer)
    return _Session(), said, seen, script


P_RATE = {"intent": "test_rate", "slots": {"test_name": "সিবিসি"}}
P_AVAIL = {"intent": "doctor_availability", "slots": {"doctor_name": "সেন"}}
P_DEPT = {"intent": "doctors_by_department", "slots": {"department": "অর্থো"}}


def test_both_parts_are_answered_in_the_order_asked(wired):
    session, said, seen, _script = wired
    deferred = asyncio.run(main._run_parts(session, "t", [P_RATE, P_AVAIL]))

    assert deferred is False
    assert [p["intent"] for p in seen] == ["test_rate", "doctor_availability"]
    assert said == ["<test_rate>", "<doctor_availability>"]


def test_three_parts_are_all_answered(wired):
    session, said, seen, _script = wired
    asyncio.run(main._run_parts(session, "t", [P_RATE, P_AVAIL, P_DEPT]))
    assert [p["intent"] for p in seen] == [
        "test_rate", "doctor_availability", "doctors_by_department"]


def test_an_unanswerable_part_does_not_stop_the_rest(wired):
    """It has already said something about itself -- that is why it is a
    third outcome and not a silent skip."""
    session, said, seen, script = wired
    script["test_rate"] = turn_parts.UNANSWERABLE
    asyncio.run(main._run_parts(session, "t", [P_RATE, P_AVAIL]))
    assert [p["intent"] for p in seen] == ["test_rate", "doctor_availability"]


def test_a_question_back_stops_the_turn_and_defers_the_rest(wired):
    """Not "skip it and do the rest": answering the second question while
    the first is waiting on a clarification reorders the conversation from
    the caller's side."""
    session, said, seen, script = wired
    script["test_rate"] = turn_parts.INTERACTIVE

    deferred = asyncio.run(main._run_parts(session, "utterance", [P_RATE, P_AVAIL]))

    assert deferred is True
    assert [p["intent"] for p in seen] == ["test_rate"]
    assert said[-1] == DEFERRED_PART_BN
    assert session.deferred["parts"] == [P_AVAIL]
    assert session.deferred["text"] == "utterance"


def test_a_question_back_on_the_last_part_defers_nothing(wired):
    """No queue, and no promise made that nothing would keep."""
    session, said, _seen, script = wired
    script["doctor_availability"] = turn_parts.INTERACTIVE
    asyncio.run(main._run_parts(session, "t", [P_RATE, P_AVAIL]))
    assert session.deferred is None
    assert DEFERRED_PART_BN not in said


def test_a_soft_continuation_is_dropped_when_another_part_follows(wired, monkeypatch):
    """doctor_availability ends by asking "today or another day?" and stays
    in the flow. That convenience belongs to the LAST thing said -- left
    open, it would catch the caller's reply to a question they have already
    stopped thinking about."""
    session, _said, _seen, _script = wired

    async def _answer(session, text, part):
        session.pending = {"awaiting": "date", "slots": {}, "candidates": None,
                           "offered_date": "2026-09-14", "retries": 0}
        return turn_parts.ANSWERED

    monkeypatch.setattr(main, "_answer_part", _answer)
    asyncio.run(main._run_parts(session, "t", [P_AVAIL, P_RATE]))
    assert session.pending is not None, "the LAST part keeps its continuation"


def test_the_last_parts_continuation_survives(wired, monkeypatch):
    session, _said, _seen, _script = wired

    calls = {"n": 0}

    async def _answer(session, text, part):
        calls["n"] += 1
        if calls["n"] == 1:
            session.pending = {"awaiting": "date", "slots": {"first": True},
                               "candidates": None, "offered_date": None, "retries": 0}
        else:
            session.pending = {"awaiting": "date", "slots": {"second": True},
                               "candidates": None, "offered_date": None, "retries": 0}
        return turn_parts.ANSWERED

    monkeypatch.setattr(main, "_answer_part", _answer)
    asyncio.run(main._run_parts(session, "t", [P_AVAIL, P_RATE]))
    assert session.pending["slots"] == {"second": True}


# ----------------------------------------------------------- the deferral

def test_a_deferred_part_is_answered_once_the_flow_clears(wired):
    """The half that makes the promise true. A queue that is only ever
    created is a drop with better manners."""
    session, said, seen, _script = wired
    session.deferred = {"parts": [P_AVAIL], "text": "original", "age": 0}
    session.pending = None

    asyncio.run(main._drain_deferred(session, "this turn's words"))

    assert RESUMING_PART_BN in said
    assert [p["intent"] for p in seen] == ["doctor_availability"]
    assert session.deferred is None


def test_the_deferred_part_is_resolved_against_the_original_utterance(wired, monkeypatch):
    """date_calc and slot_parse read the raw words. "কাল" said once for two
    questions means it for both, and re-resolving the second part against
    the caller's "হ্যাঁ" would lose the day entirely."""
    session, _said, _seen, _script = wired
    texts: list[str] = []

    async def _answer(session, text, part):
        texts.append(text)
        return turn_parts.ANSWERED

    monkeypatch.setattr(main, "_answer_part", _answer)
    session.deferred = {"parts": [P_AVAIL], "text": "কাল ডাক্তার সেন", "age": 0}
    asyncio.run(main._drain_deferred(session, "হ্যাঁ"))
    assert texts == ["কাল ডাক্তার সেন"]


def test_a_queue_waits_while_a_flow_is_still_open(wired):
    session, said, seen, _script = wired
    session.deferred = {"parts": [P_AVAIL], "text": "t", "age": 0}
    session.pending = {"awaiting": "time_slot", "slots": {}, "candidates": None,
                       "offered_date": None, "retries": 0}

    asyncio.run(main._drain_deferred(session, "t"))

    assert seen == [], "nothing answered while the caller owes an answer"
    assert session.deferred["age"] == 1


def test_a_queue_that_waits_too_long_is_said_out_loud_not_binned(wired):
    """The caller asked. They are owed the information that it went
    unanswered, even when the answer is that too much has happened since."""
    session, said, seen, _script = wired
    session.deferred = {"parts": [P_AVAIL], "text": "t", "age": 0}
    session.pending = {"awaiting": "time_slot", "slots": {}, "candidates": None,
                       "offered_date": None, "retries": 0}

    for _ in range(main.MAX_DEFERRED_TURNS + 1):
        asyncio.run(main._drain_deferred(session, "t"))

    assert session.deferred is None
    assert seen == [], "it was never answered"
    assert said[-1] == unanswered_part_prompt("সেন")


def test_draining_an_empty_queue_does_nothing(wired):
    session, said, _seen, _script = wired
    asyncio.run(main._drain_deferred(session, "t"))
    assert said == []


def test_a_deferred_part_that_asks_back_defers_what_is_behind_it(wired):
    """Two parts behind a question, the first of which asks another. The
    queue must shrink by one, not vanish."""
    session, _said, _seen, script = wired
    script["doctor_availability"] = turn_parts.INTERACTIVE
    session.deferred = {"parts": [P_AVAIL, P_DEPT], "text": "t", "age": 0}
    session.pending = None

    asyncio.run(main._drain_deferred(session, "t"))

    assert session.deferred is not None
    assert [p["intent"] for p in session.deferred["parts"]] == ["doctors_by_department"]


# ------------------------------------------- the score on the labelled set

def _expected_parts(case):
    """What the dispatcher should address, after normalisation.

    Normalisation is part of the system under test, not a fixture -- the
    labelled file records what the CALLER said, and dropping a trailing
    thank-you is a decision this code makes rather than one the label should
    pre-bake.
    """
    data = {"intent": case["parts"][0]["intent"],
            "slots": case["parts"][0].get("slots") or {},
            "parts": case["parts"], "direct_reply_bn": None}
    return turn_parts.normalise(data)


def test_completeness_is_scored_on_the_labelled_set(monkeypatch):
    """The criterion's third clause.

    Every part the dispatcher is given must be addressed -- answered, or
    said something about -- and in order. Interactive parts are scripted OFF
    here: a turn that stops to ask a question is measured by
    test_a_deferred_part_is_answered_once_the_flow_clears, and folding the
    two together would let a deferral that never drains score as complete.
    """
    addressed_total = expected_total = 0
    out_of_order = []

    for case in CASES:
        expected = _expected_parts(case)
        seen: list[dict] = []

        async def _answer(session, text, part, _seen=seen):
            _seen.append(part)
            return turn_parts.ANSWERED

        async def _say(session, text, fallback_reason=None):
            pass

        monkeypatch.setattr(main, "_answer_part", _answer)
        monkeypatch.setattr(main, "_speak", _say)

        session = _Session()
        asyncio.run(main._run_parts(session, case["transcript"], expected))

        expected_total += len(case["parts"])
        addressed_total += len(seen)
        if [p["intent"] for p in seen] != [p["intent"] for p in expected]:
            out_of_order.append(case["id"])

    completeness = addressed_total / expected_total
    order = 1.0 - len(out_of_order) / len(CASES)

    assert order >= ORDER_FLOOR, f"out of order: {out_of_order}"
    assert completeness >= COMPLETENESS_FLOOR, (
        f"completeness {completeness:.3f} < {COMPLETENESS_FLOOR} "
        f"({addressed_total}/{expected_total} parts addressed)"
    )


def test_the_only_parts_the_set_loses_are_the_ones_we_chose_to_lose():
    """Pins WHY completeness is not 1.0, so the floor cannot be quietly
    lowered later to accommodate a real regression."""
    lost = {}
    for case in CASES:
        dropped = len(case["parts"]) - len(_expected_parts(case))
        if dropped:
            lost[case["id"]] = dropped

    assert lost == {
        "rate-with-trailing-thanks": 1,   # smalltalk is not a question
        "rate-with-trailing-noise": 1,    # neither is ASR noise
        "duplicate-part": 1,              # the same question said twice
        "four-parts-capped": 1,           # MAX_PARTS
    }, lost
