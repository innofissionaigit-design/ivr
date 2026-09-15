"""ADDED BY SOURAV -- "Caller asks two questions in one breath" story.

Covers all three layers this story touches, mirroring
tests/test_phase1_intents_and_dispatch.py's own established 3-place
pattern (see that file's module docstring):

  1. agent/llm.py: _validate() now accepts the new "intents" array shape
     (TestMultiIntentSchema), with the exact same fatal-vs-tolerated
     semantics the legacy single-intent shape always had, plus
     _apply_backward_compat_mirror()'s own contract (TestBackwardCompatMirror).
  2. agent/semantic_cache.py: a multi-intent value is never L2 (fuzzy)
     eligible, while an ordinary single-intent value's eligibility is
     completely unaffected by this story (TestSemanticCacheMultiIntentGuard).
  3. main.py AND main_pcm.py dispatch -- parametrized over BOTH modules
     wherever the assertion is transport-independent, so a gap between the
     two (the exact "test_sample" bug this codebase has already hit once
     for real, per main_pcm.py's own module docstring) cannot slip through
     unnoticed (TestMultiIntentDispatch, TestTransportParity).

Story acceptance criteria under direct test:
  - Criterion 1 (Order Preservation): TestMultiIntentDispatch's
    "..._in_requested_order" tests, run with BOTH orderings of the same
    two intents.
  - Criterion 2 (Honest Partial Handling): TestMultiIntentDispatch's
    "..._explicitly_addressed"/"..._missing_slot"/"..._out_of_scope"/
    "..._needs_separate_flow" tests -- nothing is silently dropped.
  - Tool-failure isolation: test_one_tool_failure_does_not_block_the_other.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from agent.llm import _validate, _apply_backward_compat_mirror
from agent.semantic_cache import SemanticCache
from agent.tools_client import ToolCallError
from agent.reply_templates import (
    # Aliased -- a module-level name starting with "test_" gets collected
    # by pytest as a (fixture-less) test function otherwise.
    test_rate_reply as _test_rate_reply,
    walkin_eligibility_reply,
    multi_intent_missing_info_reply, multi_intent_out_of_scope_reply,
    multi_intent_needs_separate_flow_reply, human_fallback_reply,
)

import main
import main_pcm


def run(coro):
    return asyncio.run(coro)


_ALL_SLOT_KEYS = ("test_name", "doctor_name", "department", "date", "time_slot",
                  "patient_name", "phone", "package_name", "info_topic",
                  "insurance_provider_name")


def _empty_slots(**overrides):
    slots = {k: None for k in _ALL_SLOT_KEYS}
    slots.update(overrides)
    return slots


# --------------------------------------------------------------------- #
# agent/llm.py -- new "intents" array schema
# --------------------------------------------------------------------- #

class TestMultiIntentSchema:
    def test_validate_accepts_a_two_item_intents_array(self):
        data = {
            "intents": [
                {"intent": "test_rate", "slots": _empty_slots(test_name="CBC")},
                {"intent": "walkin_eligibility", "slots": _empty_slots(test_name="CBC")},
            ],
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_accepts_a_one_item_intents_array(self):
        # The common case wrapped in the new shape -- length-1 is valid too.
        data = {"intents": [{"intent": "test_rate", "slots": _empty_slots(test_name="CBC")}],
                "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_still_accepts_the_legacy_single_intent_shape(self):
        # No "intents" key at all -- every pre-this-story caller/test.
        data = {"intent": "test_rate", "slots": _empty_slots(test_name="CBC"), "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_rejects_empty_intents_array(self):
        data = {"intents": [], "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is False

    def test_validate_rejects_non_list_intents(self):
        data = {"intents": "test_rate", "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is False

    def test_validate_rejects_invalid_intent_inside_the_array(self):
        data = {"intents": [{"intent": "made_up_intent", "slots": _empty_slots()}],
                "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is False
        assert any("invalid intent" in e for e in errors)

    def test_validate_tolerates_a_missing_slot_key_inside_one_array_item(self):
        # RULE (unchanged from the legacy shape): an individual missing
        # slot dict key is never fatal by itself.
        slots = _empty_slots(test_name="CBC")
        del slots["insurance_provider_name"]
        data = {"intents": [{"intent": "test_rate", "slots": slots}], "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is True, errors
        assert any("insurance_provider_name" in e and "missing" in e for e in errors)

    def test_validate_one_bad_item_fails_the_whole_batch(self):
        # Not a partial-success shape -- either the whole extraction is
        # schema-valid or extract_intent() retries the whole turn.
        data = {
            "intents": [
                {"intent": "test_rate", "slots": _empty_slots(test_name="CBC")},
                {"intent": "not_a_real_intent", "slots": _empty_slots()},
            ],
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is False

    def test_validate_rejects_non_dict_array_item(self):
        data = {"intents": ["test_rate"], "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is False

    def test_direct_reply_bn_stripped_when_intents_has_more_than_one_entry(self):
        # Even if the intent WERE smalltalk, direct_reply_bn only survives
        # for an intents array of length exactly 1.
        data = {
            "intents": [
                {"intent": "smalltalk", "slots": _empty_slots()},
                {"intent": "test_rate", "slots": _empty_slots(test_name="CBC")},
            ],
            "direct_reply_bn": "hi there",
        }
        _validate(data)
        assert data["direct_reply_bn"] is None

    def test_direct_reply_bn_kept_for_a_single_smalltalk_entry(self):
        data = {"intents": [{"intent": "smalltalk", "slots": _empty_slots()}],
                "direct_reply_bn": "hi there"}
        ok, errors = _validate(data)
        assert ok is True, errors
        assert data["direct_reply_bn"] == "hi there"


class TestBackwardCompatMirror:
    def test_mirrors_first_intent_onto_top_level_keys(self):
        data = {
            "intents": [
                {"intent": "test_rate", "slots": _empty_slots(test_name="CBC")},
                {"intent": "walkin_eligibility", "slots": _empty_slots(test_name="CBC")},
            ],
            "direct_reply_bn": None,
        }
        _apply_backward_compat_mirror(data)
        assert data["intent"] == "test_rate"
        assert data["slots"] == _empty_slots(test_name="CBC")

    def test_noop_when_intents_key_absent(self):
        data = {"intent": "test_rate", "slots": _empty_slots(test_name="CBC"), "direct_reply_bn": None}
        _apply_backward_compat_mirror(data)
        assert data["intent"] == "test_rate"


# --------------------------------------------------------------------- #
# agent/semantic_cache.py
# --------------------------------------------------------------------- #

class TestSemanticCacheMultiIntentGuard:
    def test_multi_intent_value_never_l2_eligible(self):
        value = {
            "intent": "test_rate", "slots": {"test_name": "CBC"},
            "intents": [
                {"intent": "test_rate", "slots": {"test_name": "CBC"}},
                {"intent": "walkin_eligibility", "slots": {"test_name": "CBC"}},
            ],
        }
        assert SemanticCache._is_l2_eligible(value) is False

    def test_single_item_intents_array_unaffected(self):
        value = {
            "intent": "test_rate", "slots": {"test_name": "CBC"},
            "intents": [{"intent": "test_rate", "slots": {"test_name": "CBC"}}],
        }
        assert SemanticCache._is_l2_eligible(value) is True

    def test_ordinary_value_with_no_intents_key_unaffected(self):
        # Every value before this story, and every ordinary one-question
        # turn after it.
        value = {"intent": "test_rate", "slots": {"test_name": "CBC"}}
        assert SemanticCache._is_l2_eligible(value) is True


# --------------------------------------------------------------------- #
# main.py / main_pcm.py dispatch -- parametrized over both transports
# --------------------------------------------------------------------- #

class FakeToolsClient:
    def __init__(self, **responses):
        self.calls = {}
        self._responses = responses

    def _record(self, name, *args):
        self.calls.setdefault(name, []).append(args)

    async def get_test_rate(self, test_name):
        self._record("get_test_rate", test_name)
        resp = self._responses.get("test_rate")
        if isinstance(resp, Exception):
            raise resp
        return resp or {
            "found": True, "test_name": test_name, "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }

    async def get_walkin_policy(self, test_name):
        self._record("get_walkin_policy", test_name)
        resp = self._responses.get("walkin_policy")
        if isinstance(resp, Exception):
            raise resp
        return resp or {
            "found": True, "test_name": test_name, "test_name_bn": None,
            "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am",
        }

    async def get_doctor_availability(self, doctor_name, date):
        self._record("get_doctor_availability", doctor_name, date)
        resp = self._responses.get("doctor_availability")
        if isinstance(resp, Exception):
            raise resp
        return resp or {
            "found": True, "doctor_name": doctor_name, "doctor_name_bn": None,
            "available": True, "date": date, "next_available_date": None,
        }


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


class FakeASRResult:
    text = "কিছু একটা বললাম"


class FakeASR:
    async def transcribe_utterance(self, wav_path):
        return FakeASRResult()


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-call-1", pending=pending,
        dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
    )


TRANSPORTS = [main, main_pcm]


@pytest.fixture(params=TRANSPORTS, ids=["main", "main_pcm"])
def transport(request):
    return request.param


@pytest.fixture
def stub(transport, monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    monkeypatch.setattr(transport, "_speak", fake_speak)
    monkeypatch.setattr(transport, "_asr", FakeASR())
    tools = FakeToolsClient()
    monkeypatch.setattr(transport, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools, transport=transport)


def _dispatch(stub, monkeypatch, intents, tmp_path, pending=None):
    async def fake_resolve_intent(session, text):
        return {
            "intents": intents,
            "intent": intents[0]["intent"], "slots": intents[0]["slots"],
        }

    monkeypatch.setattr(stub.transport, "_resolve_intent", fake_resolve_intent)
    session = make_session(pending=pending)
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(stub.transport._dispatch_turn(session, str(wav_path)))
    return session


_CBC_RATE = {"intent": "test_rate", "slots": _empty_slots(test_name="CBC")}
_CBC_WALKIN = {"intent": "walkin_eligibility", "slots": _empty_slots(test_name="CBC")}


class TestMultiIntentDispatch:
    def test_both_questions_answered_in_requested_order(self, stub, monkeypatch, tmp_path):
        _dispatch(stub, monkeypatch, [_CBC_RATE, _CBC_WALKIN], tmp_path)

        assert stub.tools.calls["get_test_rate"] == [("CBC",)]
        assert stub.tools.calls["get_walkin_policy"] == [("CBC",)]
        assert len(stub.spoken) == 1

        rate_result = run(stub.tools.get_test_rate("CBC")) if False else {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }
        walkin_result = {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am",
        }
        expected = (
            _test_rate_reply(_CBC_RATE["slots"], rate_result, language="bengali") + " " +
            walkin_eligibility_reply(_CBC_WALKIN["slots"], walkin_result, language="bengali")
        )
        assert stub.spoken[0] == expected

    def test_order_preservation_reversed(self, stub, monkeypatch, tmp_path):
        # Criterion 1, other direction: swapping the order the caller asked
        # in must swap the order of the fragments in the reply.
        _dispatch(stub, monkeypatch, [_CBC_WALKIN, _CBC_RATE], tmp_path)

        rate_result = {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }
        walkin_result = {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am",
        }
        expected = (
            walkin_eligibility_reply(_CBC_WALKIN["slots"], walkin_result, language="bengali") + " " +
            _test_rate_reply(_CBC_RATE["slots"], rate_result, language="bengali")
        )
        assert stub.spoken[0] == expected

    def test_one_answered_one_missing_slot_explicitly_addressed(self, stub, monkeypatch, tmp_path):
        incomplete_walkin = {"intent": "walkin_eligibility", "slots": _empty_slots()}
        _dispatch(stub, monkeypatch, [_CBC_RATE, incomplete_walkin], tmp_path)

        assert "get_walkin_policy" not in stub.tools.calls
        assert stub.tools.calls["get_test_rate"] == [("CBC",)]
        assert len(stub.spoken) == 1
        assert multi_intent_missing_info_reply(language="bengali") in stub.spoken[0]
        # Not silently dropped -- the first question's real answer is
        # still there too.
        assert stub.spoken[0].startswith(
            _test_rate_reply(_CBC_RATE["slots"], {
                "found": True, "test_name": "CBC", "test_name_bn": None,
                "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
            }, language="bengali")
        )

    def test_one_answered_one_unreviewed_policy_explicitly_addressed(self, stub, monkeypatch, tmp_path):
        # Criterion 2's own "unreviewed" example: the walk-in policy ROW
        # exists but has never been reviewed for this test -- gets the
        # SAME honest sentence walkin_eligibility_reply() already gives a
        # solo turn, with zero new code for this case.
        stub.tools._responses["walkin_policy"] = {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "policy_available": False, "walkin_eligible": None, "walkin_hours": None,
        }
        _dispatch(stub, monkeypatch, [_CBC_RATE, _CBC_WALKIN], tmp_path)

        unreviewed_fragment = walkin_eligibility_reply(_CBC_WALKIN["slots"], stub.tools._responses["walkin_policy"],
                                                        language="bengali")
        assert unreviewed_fragment in stub.spoken[0]
        assert "policy_available" not in unreviewed_fragment  # sanity: a real sentence, not a dict dump

    def test_one_answered_one_out_of_scope_explicitly_addressed(self, stub, monkeypatch, tmp_path):
        out_of_scope_item = {"intent": "out_of_scope", "slots": _empty_slots()}
        session = _dispatch(stub, monkeypatch, [_CBC_RATE, out_of_scope_item], tmp_path)

        assert len(stub.spoken) == 1
        assert multi_intent_out_of_scope_reply(language="bengali") in stub.spoken[0]
        # Combined-turn out_of_scope is non-interactive -- no pending
        # choice state opened, unlike the solo "out_of_scope" dispatch.
        assert session.pending is None

    def test_one_answered_one_book_appointment_needs_separate_flow(self, stub, monkeypatch, tmp_path):
        # Even with every booking field already present, book_appointment
        # is NEVER answered inline in a combined turn.
        full_booking = {
            "intent": "book_appointment",
            "slots": _empty_slots(doctor_name="Dr Sen", date="2026-09-20", time_slot="10am",
                                   patient_name="Rahul Sen", phone="9000000001"),
        }
        session = _dispatch(stub, monkeypatch, [_CBC_RATE, full_booking], tmp_path)

        assert multi_intent_needs_separate_flow_reply(language="bengali") in stub.spoken[0]
        assert session.pending is None

    def test_smalltalk_contributes_no_fragment(self, stub, monkeypatch, tmp_path):
        smalltalk_item = {"intent": "smalltalk", "slots": _empty_slots()}
        _dispatch(stub, monkeypatch, [_CBC_RATE, smalltalk_item], tmp_path)

        expected = _test_rate_reply(_CBC_RATE["slots"], {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }, language="bengali")
        assert stub.spoken == [expected]

    def test_unclear_contributes_no_fragment_and_order_still_holds(self, stub, monkeypatch, tmp_path):
        unclear_item = {"intent": "unclear", "slots": _empty_slots()}
        _dispatch(stub, monkeypatch, [unclear_item, _CBC_RATE], tmp_path)

        expected = _test_rate_reply(_CBC_RATE["slots"], {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }, language="bengali")
        assert stub.spoken == [expected]

    def test_all_entries_no_fragment_falls_back_to_human_handoff(self, stub, monkeypatch, tmp_path):
        smalltalk_item = {"intent": "smalltalk", "slots": _empty_slots()}
        unclear_item = {"intent": "unclear", "slots": _empty_slots()}
        _dispatch(stub, monkeypatch, [smalltalk_item, unclear_item], tmp_path)

        assert stub.spoken == [human_fallback_reply(language="bengali")]

    def test_one_tool_failure_does_not_block_the_other(self, stub, monkeypatch, tmp_path):
        stub.tools._responses["test_rate"] = ToolCallError("clinic-api unreachable")
        _dispatch(stub, monkeypatch, [_CBC_RATE, _CBC_WALKIN], tmp_path)

        assert len(stub.spoken) == 1
        # First fragment is the apology; second is the real, unaffected answer.
        assert "কাউন্টারে যোগাযোগ করুন" in stub.spoken[0]
        walkin_result = {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am",
        }
        assert stub.spoken[0].endswith(
            walkin_eligibility_reply(_CBC_WALKIN["slots"], walkin_result, language="bengali")
        )
        # The SECOND tool call still ran -- not short-circuited by the first's failure.
        assert stub.tools.calls["get_walkin_policy"] == [("CBC",)]

    def test_single_intent_turn_is_completely_unaffected(self, stub, monkeypatch, tmp_path):
        # The fast_path/solo-extraction backward-compatibility guarantee:
        # a length-1 "intents" array takes the ORIGINAL if/elif chain, not
        # _dispatch_multi_intent_turn at all.
        session = _dispatch(stub, monkeypatch, [_CBC_RATE], tmp_path)

        expected = _test_rate_reply(_CBC_RATE["slots"], {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "rate_inr": "500", "sample_type": "blood", "report_time_hours": 24,
        }, language="bengali")
        assert stub.spoken == [expected]
        assert session.pending is None


# --------------------------------------------------------------------- #
# Transport parity: the exact "test_sample" gap this codebase already hit
# once (main_pcm.py's own module docstring) -- confirm every new symbol
# this story adds exists verbatim in BOTH files, not hand-duplicated and
# drifted.
# --------------------------------------------------------------------- #

class TestTransportParity:
    @pytest.mark.parametrize("marker", [
        "async def _dispatch_multi_intent_turn(",
        "async def _resolve_combinable_intent_fragment(",
        'intents_list = data.get("intents")',
        "_MULTI_INTENT_NEEDS_SEPARATE_FLOW = ",
        "_MULTI_INTENT_NO_FRAGMENT = ",
    ])
    def test_symbol_present_in_both_transports(self, marker):
        import inspect
        main_src = inspect.getsource(main)
        main_pcm_src = inspect.getsource(main_pcm)
        assert marker in main_src, f"{marker!r} missing from main.py"
        assert marker in main_pcm_src, f"{marker!r} missing from main_pcm.py"

    def test_both_transports_agree_on_a_two_intent_turn(self, monkeypatch, tmp_path):
        # Same scenario run through BOTH transports independently -- the
        # actual reply text produced must match, not just the source code.
        results = {}
        for transport in TRANSPORTS:
            spoken = []

            async def fake_speak(session, text, fallback_reason=None, _spoken=spoken):
                _spoken.append(text)

            monkeypatch.setattr(transport, "_speak", fake_speak)
            monkeypatch.setattr(transport, "_asr", FakeASR())
            tools = FakeToolsClient()
            monkeypatch.setattr(transport, "_tools", tools)

            async def fake_resolve_intent(session, text):
                return {
                    "intents": [_CBC_RATE, _CBC_WALKIN],
                    "intent": _CBC_RATE["intent"], "slots": _CBC_RATE["slots"],
                }

            monkeypatch.setattr(transport, "_resolve_intent", fake_resolve_intent)
            session = make_session()
            wav_path = tmp_path / f"utt-{transport.__name__}.wav"
            wav_path.write_bytes(b"")
            run(transport._dispatch_turn(session, str(wav_path)))
            results[transport.__name__] = spoken

        assert results["main"] == results["main_pcm"]
        assert len(results["main"]) == 1
