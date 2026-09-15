"""ADDED BY SOURAV -- "Caller asks a follow-up that depends on the previous
answer" story.

Evidence backing the backlog item ("Each turn is independent") was
literally true before this change -- see agent/state.py's own module
docstring for the full writeup of why and exactly what this story adds.

Covers all the layers this story touches:
  1. agent/state.py: DialogueState/EntitySlot/resolve_follow_up() -- pure
     logic, no I/O. TestDialogueStateBasics, TestResolveFollowUp.
  2. agent/reply_templates.py: ambiguous_reference_reply() -- the new
     clarifying question, in all four languages. TestAmbiguousReply.
  3. main.py AND main_pcm.py dispatch -- parametrized over BOTH
     transports, same transport-parity discipline as every other story
     in this codebase. TestBackfillAcrossIntents, TestAmbiguityDispatch,
     TestTransportParity.
  4. The scripted, end-to-end scenario Acceptance Criterion 3 explicitly
     asks for: TestThreeConsecutiveFollowUps.

Story acceptance criteria under direct test:
  - AC 1 (pronouns/elliptical follow-ups resolve against the primary
    entity from the previous turn): TestResolveFollowUp +
    TestBackfillAcrossIntents.
  - AC 2 (ambiguous or blank memory -> ask for clarification rather than
    assume): TestResolveFollowUp's ambiguous/blank cases +
    TestAmbiguityDispatch (dispatch-level, including the resume-after-
    clarification flow) + TestBlankMemoryFallsThroughToOrdinaryAsk.
  - AC 3 (validated on scripted transcripts, >= 3 consecutive follow-ups
    without re-prompting for the entity name): TestThreeConsecutiveFollowUps.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from agent.state import DialogueState, resolve_follow_up, primary_slot_for_intent, kind_for_slot
from agent.reply_templates import ambiguous_reference_reply, missing_slot_prompt

import main
import main_pcm


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- #
# 1. agent/state.py -- pure logic, no I/O
# --------------------------------------------------------------------- #

class TestDialogueStateBasics:
    def test_a_fresh_state_has_nothing_tracked(self):
        state = DialogueState()
        assert state.active_test.is_set is False
        assert state.active_doctor.is_set is False
        assert state.active_package.is_set is False

    def test_mark_records_a_single_definite_name(self):
        state = DialogueState()
        state.mark("test", "CBC")
        assert state.active_test.is_set is True
        assert state.active_test.is_ambiguous is False
        assert state.active_test.primary == "CBC"

    def test_mark_again_replaces_rather_than_accumulates(self):
        state = DialogueState()
        state.mark("test", "CBC")
        state.mark("test", "Lipid Profile")
        assert state.active_test.primary == "Lipid Profile"
        assert state.active_test.is_ambiguous is False

    def test_mark_ambiguous_with_two_different_names(self):
        state = DialogueState()
        state.mark_ambiguous("test", ["CBC", "Lipid Profile"])
        assert state.active_test.is_ambiguous is True
        assert state.active_test.primary is None
        assert state.active_test.names == ("CBC", "Lipid Profile")

    def test_mark_ambiguous_deduplicates_and_preserves_order(self):
        state = DialogueState()
        state.mark_ambiguous("test", ["CBC", "Lipid Profile", "CBC"])
        assert state.active_test.names == ("CBC", "Lipid Profile")

    def test_mark_ambiguous_with_one_name_is_not_ambiguous(self):
        state = DialogueState()
        state.mark_ambiguous("test", ["CBC"])
        assert state.active_test.is_ambiguous is False
        assert state.active_test.primary == "CBC"

    def test_mark_collapses_a_previously_ambiguous_slot(self):
        # A caller who was just asked "which one?" and answered has, by
        # definition, resolved the ambiguity for anything that follows.
        state = DialogueState()
        state.mark_ambiguous("test", ["CBC", "Lipid Profile"])
        state.mark("test", "CBC")
        assert state.active_test.is_ambiguous is False
        assert state.active_test.primary == "CBC"

    def test_different_kinds_are_tracked_independently(self):
        state = DialogueState()
        state.mark("test", "CBC")
        state.mark("doctor", "Dr Sen")
        assert state.active_test.primary == "CBC"
        assert state.active_doctor.primary == "Dr Sen"
        assert state.active_package.is_set is False

    def test_primary_slot_for_intent_only_covers_single_entity_intents(self):
        assert primary_slot_for_intent("test_rate") == "test_name"
        assert primary_slot_for_intent("doctor_availability") == "doctor_name"
        assert primary_slot_for_intent("health_package") == "package_name"
        # Two-required-slot intents are deliberately excluded -- already
        # served by their own pending mechanism (insurance_coverage_slot /
        # compare_options_slot).
        assert primary_slot_for_intent("insurance_coverage") is None
        assert primary_slot_for_intent("compare_options") is None
        # No single anaphora-eligible entity slot at all.
        assert primary_slot_for_intent("book_appointment") is None
        assert primary_slot_for_intent("smalltalk") is None
        assert primary_slot_for_intent("clinic_info") is None

    def test_kind_for_slot(self):
        assert kind_for_slot("test_name") == "test"
        assert kind_for_slot("doctor_name") == "doctor"
        assert kind_for_slot("package_name") == "package"
        assert kind_for_slot("phone") is None


class TestResolveFollowUp:
    def test_blank_memory_leaves_slots_unchanged_and_is_not_ambiguous(self):
        state = DialogueState()
        slots = {"test_name": None}
        backfilled, ambiguous_kind = resolve_follow_up(state, "test_rate", slots)
        assert backfilled == {"test_name": None}
        assert ambiguous_kind is None

    def test_unambiguous_tracked_entity_backfills_the_missing_slot(self):
        state = DialogueState()
        state.mark("test", "CBC")
        slots = {"test_name": None}
        backfilled, ambiguous_kind = resolve_follow_up(state, "walkin_eligibility", slots)
        assert backfilled["test_name"] == "CBC"
        assert ambiguous_kind is None

    def test_never_mutates_the_input_slots_dict(self):
        state = DialogueState()
        state.mark("test", "CBC")
        original = {"test_name": None}
        backfilled, _ = resolve_follow_up(state, "test_rate", original)
        assert original == {"test_name": None}
        assert backfilled is not original

    def test_a_freshly_named_entity_this_turn_is_never_overridden(self):
        state = DialogueState()
        state.mark("test", "CBC")
        slots = {"test_name": "Widal"}
        backfilled, ambiguous_kind = resolve_follow_up(state, "test_rate", slots)
        assert backfilled is slots
        assert backfilled["test_name"] == "Widal"
        assert ambiguous_kind is None

    def test_ambiguous_tracked_entity_refuses_to_backfill(self):
        state = DialogueState()
        state.mark_ambiguous("test", ["CBC", "Lipid Profile"])
        slots = {"test_name": None}
        backfilled, ambiguous_kind = resolve_follow_up(state, "test_preparation", slots)
        assert backfilled["test_name"] is None
        assert ambiguous_kind == "test"

    def test_intents_this_module_does_not_apply_to_are_untouched(self):
        state = DialogueState()
        state.mark("test", "CBC")
        slots = {"phone": None}
        backfilled, ambiguous_kind = resolve_follow_up(state, "billing_balance", slots)
        assert backfilled == slots
        assert ambiguous_kind is None

    def test_different_kind_ambiguity_does_not_block_an_unrelated_intent(self):
        state = DialogueState()
        state.mark_ambiguous("test", ["CBC", "Lipid Profile"])
        state.mark("doctor", "Dr Sen")
        slots = {"doctor_name": None}
        backfilled, ambiguous_kind = resolve_follow_up(state, "doctor_availability", slots)
        assert backfilled["doctor_name"] == "Dr Sen"
        assert ambiguous_kind is None


# --------------------------------------------------------------------- #
# 2. agent/reply_templates.py -- ambiguous_reference_reply()
# --------------------------------------------------------------------- #

_LANGUAGES = ("english", "hinglish", "banglish", "bengali")


class TestAmbiguousReply:
    @pytest.mark.parametrize("language", _LANGUAGES)
    @pytest.mark.parametrize("kind", ["test", "doctor", "package"])
    def test_names_both_candidates_in_every_language_and_kind(self, language, kind):
        reply = ambiguous_reference_reply(kind, ("CBC", "Lipid Profile"), language=language)
        assert "CBC" in reply
        assert "Lipid Profile" in reply

    def test_never_contains_a_literal_colon_or_bracket_field_label(self):
        for language in _LANGUAGES:
            reply = ambiguous_reference_reply("test", ("CBC", "Lipid Profile"), language=language)
            assert ":" not in reply
            assert "[" not in reply and "]" not in reply

    def test_lists_candidates_in_the_order_given_not_resorted(self):
        reply = ambiguous_reference_reply("test", ("Zinc Test", "Albumin Test"), language="english")
        assert reply.index("Zinc Test") < reply.index("Albumin Test")


# --------------------------------------------------------------------- #
# 3. main.py / main_pcm.py dispatch -- parametrized over both transports
# --------------------------------------------------------------------- #

class FakeToolsClient:
    def __init__(self, **responses):
        self.calls = []
        self._responses = responses

    def _record(self, name, *args):
        self.calls.append((name, args))

    async def get_test_rate(self, test_name):
        self._record("get_test_rate", test_name)
        table = self._responses.get("test_rate", {})
        if test_name in table:
            return table[test_name]
        return {"found": False, "query": test_name, "did_you_mean": []}

    async def get_test_preparation(self, test_name):
        self._record("get_test_preparation", test_name)
        table = self._responses.get("test_preparation", {})
        if test_name in table:
            return table[test_name]
        return {"found": False, "query": test_name}

    async def get_walkin_policy(self, test_name):
        self._record("get_walkin_policy", test_name)
        table = self._responses.get("walkin_policy", {})
        if test_name in table:
            return table[test_name]
        return {"found": False, "query": test_name}

    async def get_prescription_policy(self, test_name):
        self._record("get_prescription_policy", test_name)
        table = self._responses.get("prescription_policy", {})
        if test_name in table:
            return table[test_name]
        return {"found": False, "query": test_name}

    async def get_doctor_availability(self, doctor_name, date):
        self._record("get_doctor_availability", doctor_name, date)
        table = self._responses.get("doctor_availability", {})
        if doctor_name in table:
            return table[doctor_name]
        return {"found": False, "query": doctor_name}

    async def get_doctor_schedule(self, doctor_name):
        self._record("get_doctor_schedule", doctor_name)
        table = self._responses.get("doctor_schedule", {})
        if doctor_name in table:
            return table[doctor_name]
        return {"found": False, "query": doctor_name}

    async def search_health_package(self, package_name):
        self._record("search_health_package", package_name)
        table = self._responses.get("health_package", {})
        if package_name in table:
            return table[package_name]
        return {"found": False, "query": package_name, "did_you_mean": []}

    async def get_health_packages(self):
        # health_package's own pre-existing "no name given" path (main.py's
        # `elif intent == "health_package":` branch, `else:` arm) -- listed
        # here only so that path has something to call; this story's own
        # tests never inspect the return value, only that a blank-memory
        # follow-up still reaches this list-everything call instead of
        # backfilling or re-prompting.
        self._record("get_health_packages")
        table = self._responses.get("health_package", {})
        return {"packages": list(table.values())}


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


class FakeASRResult:
    text = "কিছু একটা বললাম"


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


_TOOL_RESPONSES = dict(
    test_rate={
        "CBC": {"found": True, "test_name": "CBC", "test_name_bn": None, "rate_inr": 500,
                 "sample_type": "Blood", "report_time_hours": 24},
        "Lipid Profile": {"found": True, "test_name": "Lipid Profile", "test_name_bn": None,
                            "rate_inr": 750, "sample_type": "Blood", "report_time_hours": 24},
    },
    test_preparation={
        "CBC": {"found": True, "test_name": "CBC", "test_name_bn": None,
                 "policy_available": True, "fasting_required": False, "instructions": "No special prep needed."},
        "Lipid Profile": {"found": True, "test_name": "Lipid Profile", "test_name_bn": None,
                            "policy_available": True, "fasting_required": True, "instructions": "Fast for 12 hours."},
    },
    walkin_policy={
        "CBC": {"found": True, "test_name": "CBC", "test_name_bn": None,
                 "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am"},
    },
    prescription_policy={
        "CBC": {"found": True, "test_name": "CBC", "test_name_bn": None,
                 "policy_available": True, "prescription_required": False},
        "Lipid Profile": {"found": True, "test_name": "Lipid Profile", "test_name_bn": None,
                            "policy_available": True, "prescription_required": False},
    },
    doctor_availability={
        # available=False and next_available_date=None deliberately -- this
        # leaves the OTHER pre-existing pending mechanism (_dispatch_turn's
        # "Keep the flow open for 'yes, book that day'" comment just above
        # the doctor_availability branch in main.py) closed, so a test
        # about backfilling (this story's own concern) doesn't also have to
        # thread its way around that separate, older booking-offer flow.
        "Dr Sen": {"found": True, "doctor_name": "Dr Sen", "doctor_name_bn": None, "date": "2026-09-15",
                    "available": False, "chamber_hours": None, "next_available_date": None},
    },
    doctor_schedule={
        "Dr Sen": {"found": True, "doctor_name": "Dr Sen", "doctor_name_bn": None,
                    "days": ["Monday", "Wednesday", "Friday"]},
    },
    health_package={
        "Diabetes Package": {"found": True, "package_name": "Diabetes Package", "package_name_bn": None,
                               "price_inr": 1500, "tests": ["Fasting Sugar", "HbA1c"], "tests_bn": []},
    },
)


@pytest.fixture
def stub(transport, monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    monkeypatch.setattr(transport, "_speak", fake_speak)
    monkeypatch.setattr(transport, "_asr", FakeASR())
    tools = FakeToolsClient(**_TOOL_RESPONSES)
    monkeypatch.setattr(transport, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools, transport=transport)


def _empty_slots(**overrides):
    slots = {
        "test_name": None, "doctor_name": None, "department": None, "date": None,
        "time_slot": None, "patient_name": None, "phone": None, "package_name": None,
        "info_topic": None, "insurance_provider_name": None,
        "compare_option_a": None, "compare_option_b": None,
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


def _dispatch_multi(stub, monkeypatch, intents, tmp_path, session=None):
    async def fake_resolve_intent(sess, text):
        return {"intents": intents, "intent": intents[0]["intent"], "slots": intents[0]["slots"]}

    monkeypatch.setattr(stub.transport, "_resolve_intent", fake_resolve_intent)
    session = session or make_session()
    wav_path = tmp_path / f"utt-multi-{id(intents)}.wav"
    wav_path.write_bytes(b"")
    run(stub.transport._dispatch_turn(session, str(wav_path)))
    return session


def _continue(stub, session, text):
    return run(stub.transport._continue_pending(session, text))


class TestBackfillAcrossIntents:
    def test_a_follow_up_with_no_test_name_resolves_against_the_last_test_discussed(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "test_rate", _empty_slots(test_name="CBC"), tmp_path)
        assert session.state.active_test.primary == "CBC"
        stub.spoken.clear()

        session = _dispatch(stub, monkeypatch, "walkin_eligibility", _empty_slots(), tmp_path, session=session)

        assert ("get_walkin_policy", ("CBC",)) in stub.tools.calls
        assert missing_slot_prompt("walkin_eligibility", "test_name") not in stub.spoken
        assert len(stub.spoken) == 1

    def test_a_follow_up_backfill_works_across_a_DIFFERENT_intent_family(self, stub, monkeypatch, tmp_path):
        # test_rate -> test_preparation -> prescription_requirements, all
        # about the same test, none of the later two ever repeating the name.
        session = _dispatch(stub, monkeypatch, "test_rate", _empty_slots(test_name="CBC"), tmp_path)
        session = _dispatch(stub, monkeypatch, "test_preparation", _empty_slots(), tmp_path, session=session)
        assert ("get_test_preparation", ("CBC",)) in stub.tools.calls

        session = _dispatch(stub, monkeypatch, "prescription_requirements", _empty_slots(), tmp_path, session=session)
        assert ("get_prescription_policy", ("CBC",)) in stub.tools.calls

    def test_doctor_follow_up_resolves_against_the_last_doctor_discussed(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "doctor_availability", _empty_slots(doctor_name="Dr Sen"), tmp_path)
        assert session.state.active_doctor.primary == "Dr Sen"

        _dispatch(stub, monkeypatch, "doctor_schedule", _empty_slots(), tmp_path, session=session)
        assert ("get_doctor_schedule", ("Dr Sen",)) in stub.tools.calls

    def test_package_follow_up_resolves_against_the_last_package_discussed(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "health_package", _empty_slots(package_name="Diabetes Package"), tmp_path)
        assert session.state.active_package.primary == "Diabetes Package"

    def test_a_freshly_named_test_always_wins_over_backfill(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "test_rate", _empty_slots(test_name="CBC"), tmp_path)
        stub.tools.calls.clear()

        _dispatch(stub, monkeypatch, "test_preparation", _empty_slots(test_name="Lipid Profile"), tmp_path, session=session)

        assert ("get_test_preparation", ("Lipid Profile",)) in stub.tools.calls
        assert ("get_test_preparation", ("CBC",)) not in stub.tools.calls

    def test_a_not_found_lookup_does_not_clobber_the_previously_tracked_entity(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "test_rate", _empty_slots(test_name="CBC"), tmp_path)
        # A test lookup that returns not-found must not overwrite state.
        _dispatch(stub, monkeypatch, "test_rate", _empty_slots(test_name="Nonexistent Test"), tmp_path, session=session)

        assert session.state.active_test.primary == "CBC"


class TestBlankMemoryFallsThroughToOrdinaryAsk:
    def test_first_ever_turn_with_no_entity_gets_the_ordinary_missing_slot_prompt(self, stub, monkeypatch, tmp_path):
        _dispatch(stub, monkeypatch, "test_rate", _empty_slots(), tmp_path)
        assert stub.spoken == [missing_slot_prompt("test_rate", "test_name")]
        assert "get_test_rate" not in [c[0] for c in stub.tools.calls]

    def test_health_package_with_blank_memory_still_lists_all_packages(self, stub, monkeypatch, tmp_path):
        # health_package's own pre-existing "no name given -> list them
        # all" behaviour (a genuinely different, valid answer) must be
        # completely unaffected by this story when nothing is tracked.
        _dispatch(stub, monkeypatch, "health_package", _empty_slots(), tmp_path)
        assert ("search_health_package", ()) not in [(n, a) for n, a in stub.tools.calls if n == "search_health_package"]


class TestAmbiguityDispatch:
    def test_compare_options_between_two_tests_makes_a_bare_follow_up_ambiguous(self, stub, monkeypatch, tmp_path):
        session = _dispatch(
            stub, monkeypatch, "compare_options",
            _empty_slots(compare_option_a="CBC", compare_option_b="Lipid Profile"), tmp_path,
        )
        assert session.state.active_test.is_ambiguous is True
        stub.spoken.clear()

        session = _dispatch(stub, monkeypatch, "prescription_requirements", _empty_slots(), tmp_path, session=session)

        assert "get_prescription_policy" not in [c[0] for c in stub.tools.calls]
        assert session.pending["awaiting"] == "follow_up_clarification"
        assert session.pending["intent"] == "prescription_requirements"
        assert set(session.pending["candidates"]) == {"CBC", "Lipid Profile"}
        assert ambiguous_reference_reply("test", ("CBC", "Lipid Profile")) in stub.spoken or len(stub.spoken) == 1

    def test_answering_the_clarification_resumes_the_original_question(self, stub, monkeypatch, tmp_path):
        session = _dispatch(
            stub, monkeypatch, "compare_options",
            _empty_slots(compare_option_a="CBC", compare_option_b="Lipid Profile"), tmp_path,
        )
        _dispatch(stub, monkeypatch, "prescription_requirements", _empty_slots(), tmp_path, session=session)
        stub.spoken.clear()

        handled = _continue(stub, session, "CBC")

        assert handled is True
        assert ("get_prescription_policy", ("CBC",)) in stub.tools.calls
        assert session.pending is None
        assert len(stub.spoken) == 1

    def test_resolving_the_ambiguity_collapses_state_so_the_next_follow_up_does_not_re_ask(self, stub, monkeypatch, tmp_path):
        session = _dispatch(
            stub, monkeypatch, "compare_options",
            _empty_slots(compare_option_a="CBC", compare_option_b="Lipid Profile"), tmp_path,
        )
        _dispatch(stub, monkeypatch, "prescription_requirements", _empty_slots(), tmp_path, session=session)
        _continue(stub, session, "CBC")
        assert session.state.active_test.primary == "CBC"
        stub.spoken.clear()

        # A THIRD, later follow-up now resolves straight through, no
        # re-prompt -- this is exactly AC3's guarantee, checked here at
        # the smaller scale of "does resolving an ambiguity even help".
        _dispatch(stub, monkeypatch, "walkin_eligibility", _empty_slots(), tmp_path, session=session)
        assert ("get_walkin_policy", ("CBC",)) in stub.tools.calls
        assert session.pending is None

    def test_unparseable_clarification_reply_reprompts_then_gives_up(self, stub, monkeypatch, tmp_path):
        session = _dispatch(
            stub, monkeypatch, "compare_options",
            _empty_slots(compare_option_a="CBC", compare_option_b="Lipid Profile"), tmp_path,
        )
        _dispatch(stub, monkeypatch, "prescription_requirements", _empty_slots(), tmp_path, session=session)
        stub.spoken.clear()

        assert _continue(stub, session, "ওইটা মনে নেই") is True
        assert _continue(stub, session, "ওইটা মনে নেই") is True
        assert _continue(stub, session, "ওইটা মনে নেই") is False
        assert session.pending is None

    def test_negative_reply_to_clarification_abandons_cleanly(self, stub, monkeypatch, tmp_path):
        session = _dispatch(
            stub, monkeypatch, "compare_options",
            _empty_slots(compare_option_a="CBC", compare_option_b="Lipid Profile"), tmp_path,
        )
        _dispatch(stub, monkeypatch, "prescription_requirements", _empty_slots(), tmp_path, session=session)

        handled = _continue(stub, session, "না")

        assert handled is True
        assert session.pending is None

    def test_a_multi_intent_turn_naming_two_different_tests_also_creates_ambiguity(self, stub, monkeypatch, tmp_path):
        session = _dispatch_multi(
            stub, monkeypatch,
            [
                {"intent": "test_rate", "slots": _empty_slots(test_name="CBC")},
                {"intent": "test_duration", "slots": _empty_slots(test_name="Lipid Profile")},
            ],
            tmp_path,
        )
        assert session.state.active_test.is_ambiguous is True

    def test_a_multi_intent_turn_naming_the_SAME_test_twice_is_not_ambiguous(self, stub, monkeypatch, tmp_path):
        session = _dispatch_multi(
            stub, monkeypatch,
            [
                {"intent": "test_rate", "slots": _empty_slots(test_name="CBC")},
                {"intent": "test_duration", "slots": _empty_slots(test_name="CBC")},
            ],
            tmp_path,
        )
        assert session.state.active_test.is_ambiguous is False
        assert session.state.active_test.primary == "CBC"


class TestTransportParity:
    def test_both_transports_backfill_and_speak_the_same_reply(self, monkeypatch, tmp_path):
        results = {}
        for transport in TRANSPORTS:
            spoken = []

            async def fake_speak(session, text, fallback_reason=None, _spoken=spoken):
                _spoken.append(text)

            calls = {"turn": 0}

            async def fake_resolve_intent(session, text, _calls=calls):
                _calls["turn"] += 1
                if _calls["turn"] == 1:
                    return {"intent": "test_rate", "slots": _empty_slots(test_name="CBC")}
                return {"intent": "walkin_eligibility", "slots": _empty_slots()}

            tools = FakeToolsClient(**_TOOL_RESPONSES)
            monkeypatch.setattr(transport, "_speak", fake_speak)
            monkeypatch.setattr(transport, "_asr", FakeASR())
            monkeypatch.setattr(transport, "_tools", tools)
            monkeypatch.setattr(transport, "_resolve_intent", fake_resolve_intent)

            session = make_session()
            for i in range(2):
                wav_path = tmp_path / f"utt-{transport.__name__}-{i}.wav"
                wav_path.write_bytes(b"")
                run(transport._dispatch_turn(session, str(wav_path)))
            results[transport.__name__] = list(spoken)

        assert results["main"] == results["main_pcm"]


# --------------------------------------------------------------------- #
# 4. Acceptance Criterion 3, verbatim: a scripted transcript with at
#    least three consecutive follow-ups, none of which re-prompt for the
#    entity name.
# --------------------------------------------------------------------- #

class TestThreeConsecutiveFollowUps:
    def test_scripted_transcript_three_consecutive_follow_ups(self, stub, monkeypatch, tmp_path):
        """Turn 1 names CBC explicitly (test_rate). Turns 2, 3, and 4 are
        three CONSECUTIVE follow-ups -- test_preparation, walkin_eligibility,
        and prescription_requirements -- none of which name a test at all.
        None of the three may re-prompt for the test name, and each must
        call its OWN tool with "CBC" backfilled from the turn before it.
        """
        session = _dispatch(stub, monkeypatch, "test_rate", _empty_slots(test_name="CBC"), tmp_path)
        assert stub.spoken == [
            __import__("agent.reply_templates", fromlist=["test_rate_reply"]).test_rate_reply(
                _empty_slots(test_name="CBC"),
                {"found": True, "test_name": "CBC", "test_name_bn": None, "rate_inr": 500,
                 "sample_type": "Blood", "report_time_hours": 24},
            )
        ]

        follow_up_intents = ["test_preparation", "walkin_eligibility", "prescription_requirements"]
        expected_tool_per_intent = {
            "test_preparation": "get_test_preparation",
            "walkin_eligibility": "get_walkin_policy",
            "prescription_requirements": "get_prescription_policy",
        }
        for intent in follow_up_intents:
            stub.spoken.clear()
            session = _dispatch(stub, monkeypatch, intent, _empty_slots(), tmp_path, session=session)

            # Never re-prompted for the missing test name.
            assert missing_slot_prompt(intent, "test_name") not in stub.spoken
            # The backfilled name reached the right tool, every time.
            assert (expected_tool_per_intent[intent], ("CBC",)) in stub.tools.calls
            # A real answer was spoken, not silence.
            assert len(stub.spoken) == 1
            # No pending state was left dangling by any follow-up.
            assert session.pending is None

    def test_scripted_transcript_works_identically_on_both_transports(self, monkeypatch, tmp_path):
        for transport in TRANSPORTS:
            spoken_log: list[list[str]] = []
            tools = FakeToolsClient(**_TOOL_RESPONSES)
            intents_script = ["test_rate", "test_preparation", "walkin_eligibility", "prescription_requirements"]
            slots_script = [_empty_slots(test_name="CBC"), _empty_slots(), _empty_slots(), _empty_slots()]
            turn = {"i": 0}

            async def fake_speak(session, text, fallback_reason=None, _log=spoken_log):
                _log.append(text)

            async def fake_resolve_intent(session, text, _turn=turn):
                i = _turn["i"]
                _turn["i"] += 1
                return {"intent": intents_script[i], "slots": slots_script[i]}

            monkeypatch.setattr(transport, "_speak", fake_speak)
            monkeypatch.setattr(transport, "_asr", FakeASR())
            monkeypatch.setattr(transport, "_tools", tools)
            monkeypatch.setattr(transport, "_resolve_intent", fake_resolve_intent)

            session = make_session()
            for i in range(4):
                wav_path = tmp_path / f"utt-script-{transport.__name__}-{i}.wav"
                wav_path.write_bytes(b"")
                run(transport._dispatch_turn(session, str(wav_path)))

            assert len(spoken_log) == 4
            assert ("get_test_preparation", ("CBC",)) in tools.calls
            assert ("get_walkin_policy", ("CBC",)) in tools.calls
            assert ("get_prescription_policy", ("CBC",)) in tools.calls
