"""ADDED BY SOURAV -- "Caller asks the agent to compare two options" story.

Covers all the layers this story touches, mirroring the established
3(+)-place pattern this codebase's other Phase-1/2 stories already use
(see tests/test_phase1_intents_and_dispatch.py's own module docstring):

  1. agent/compare_flow.py: build_comparison() -- pure arithmetic, no I/O.
     TestBuildComparisonPriceMath, TestBuildComparisonPackageTests,
     TestBuildComparisonNotFound. Includes an explicit float-vs-Decimal
     regression test proving no binary-rounding artifact ever reaches a
     price the caller hears (Acceptance Criterion 1).
  2. agent/reply_templates.py: compare_options_reply() -- renders that
     dict into one sentence per language, adding no new facts of its own
     (Acceptance Criterion 2 -- ONLY facts). TestCompareOptionsReply.
  3. agent/llm.py: VALID_INTENTS/_SLOT_KEYS accept the new intent/slots,
     and the CLINICAL SAFETY NOTE guardrail text actually exists in
     SYSTEM_PROMPT_TEMPLATE (Acceptance Criterion 3).
     TestLlmSchemaAndGuardrail.
  4. agent/semantic_cache.py: compare_options is excluded from L2 (fuzzy)
     eligibility, same reason book_appointment/insurance_coverage are.
     TestSemanticCacheCompareOptionsGuard.
  5. main.py AND main_pcm.py dispatch -- parametrized over BOTH modules,
     same transport-parity discipline as every other multi-slot intent in
     this codebase. TestCompareOptionsDispatch, TestTransportParity.
  6. Clinical safety boundary, end to end: TestClinicalSafetyBoundary
     scans every reply this story can ever produce, across every
     language and every found/not-found/package permutation, for
     recommendation-shaped language, and confirms direct_reply_bn can
     never survive a compare_options extraction.

Story acceptance criteria under direct test:
  - AC 1 (computed strictly in Python, from live values, one concise
    sentence): TestBuildComparisonPriceMath (the arithmetic itself, via
    decimal.Decimal, never float) + TestCompareOptionsReply (the sentence).
  - AC 2 (ONLY facts -- price difference, test count, component
    differences): TestBuildComparisonPackageTests + TestCompareOptionsReply.
  - AC 3 (Strict Clinical Advice Boundary): TestLlmSchemaAndGuardrail +
    TestClinicalSafetyBoundary.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from agent.compare_flow import build_comparison
from agent.llm import VALID_INTENTS, _SLOT_KEYS, SYSTEM_PROMPT_TEMPLATE, _validate
from agent.reply_templates import compare_options_reply, missing_slot_prompt
from agent.semantic_cache import SemanticCache

import main
import main_pcm


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- #
# 1. agent/compare_flow.py -- pure arithmetic, no I/O
# --------------------------------------------------------------------- #

class TestBuildComparisonPriceMath:
    def test_test_vs_test_price_delta_and_cheaper_side(self):
        a = {"kind": "test", "rate_inr": 650}
        b = {"kind": "test", "rate_inr": "750.50"}
        result = build_comparison(a, b)
        assert result["price_delta"] == "100.50"
        assert result["cheaper"] == "a"

    def test_reversed_sides_flip_cheaper_but_keep_delta_magnitude(self):
        a = {"kind": "test", "rate_inr": "750.50"}
        b = {"kind": "test", "rate_inr": 650}
        result = build_comparison(a, b)
        assert result["price_delta"] == "100.50"
        assert result["cheaper"] == "b"

    def test_equal_price_reports_no_cheaper_side_but_a_zero_delta_fact(self):
        a = {"kind": "test", "rate_inr": 500}
        b = {"kind": "package", "rate_inr": None, "price_inr": 500}
        result = build_comparison(a, b)
        assert result["cheaper"] is None
        assert result["price_delta"] == "0"

    def test_decimal_arithmetic_has_no_binary_float_rounding_artifact(self):
        # THE regression this story's whole "Decimal, never float" design
        # decision exists to prevent -- see agent/compare_flow.py's own
        # module docstring. float(100.10) - float(100.00) in real Python
        # is 0.09999999999999432, not 0.10 -- a wrong number that would
        # then be spoken to a caller with total confidence. Proves the
        # actual code path here never produces that artifact.
        a = {"kind": "package", "price_inr": "100.10"}
        b = {"kind": "package", "price_inr": "100.00"}
        # Sanity-check the bug this guards against is real in plain float:
        assert float("100.10") - float("100.00") != 0.10
        result = build_comparison(a, b)
        assert result["price_delta"] == "0.10"

    def test_whole_number_int_and_decimal_string_compare_correctly(self):
        # rate_inr as a bare `int` (per agent/tools_client.py's
        # _parse_exact() -- whole numbers stay int, decimals stay str).
        a = {"kind": "test", "rate_inr": 400}
        b = {"kind": "test", "rate_inr": "399.99"}
        result = build_comparison(a, b)
        assert result["price_delta"] == "0.01"
        assert result["cheaper"] == "b"

    def test_missing_price_on_one_side_yields_no_price_comparison(self):
        a = {"kind": "test", "rate_inr": None}
        b = {"kind": "test", "rate_inr": 500}
        result = build_comparison(a, b)
        assert result["price_delta"] is None
        assert result["cheaper"] is None

    def test_mixed_test_and_package_kinds_still_compare_by_price(self):
        a = {"kind": "test", "rate_inr": 300}
        b = {"kind": "package", "price_inr": "999.00"}
        result = build_comparison(a, b)
        assert result["cheaper"] == "a"
        assert result["both_packages"] is False


class TestBuildComparisonPackageTests:
    def test_identical_test_sets_report_no_differences(self):
        a = {"kind": "package", "price_inr": 1000, "tests": ["CBC", "Lipid"]}
        b = {"kind": "package", "price_inr": 1000, "tests": ["Lipid", "CBC"]}
        result = build_comparison(a, b)
        assert result["both_packages"] is True
        assert result["identical_tests"] is True
        assert result["test_count_delta"] == 0
        assert result["more_tests_side"] is None
        assert result["extra_tests_a"] == []
        assert result["extra_tests_b"] == []

    def test_one_side_has_more_tests(self):
        a = {"kind": "package", "price_inr": 1500,
             "tests": ["Fasting Sugar", "HbA1c", "Lipid"], "tests_bn": ["ফাস্টিং সুগার", "এইচবিএ১সি", "লিপিড"]}
        b = {"kind": "package", "price_inr": 1200, "tests": ["Fasting Sugar"]}
        result = build_comparison(a, b)
        assert result["test_count_delta"] == 2
        assert result["more_tests_side"] == "a"
        assert {e["name"] for e in result["extra_tests_a"]} == {"HbA1c", "Lipid"}
        assert result["extra_tests_b"] == []
        # Bengali alias carried along positionally, not re-looked-up.
        names_bn = {e["name"]: e["name_bn"] for e in result["extra_tests_a"]}
        assert names_bn["HbA1c"] == "এইচবিএ১সি"

    def test_same_count_but_different_actual_tests(self):
        a = {"kind": "package", "price_inr": 1000, "tests": ["CBC", "Sugar"]}
        b = {"kind": "package", "price_inr": 1000, "tests": ["CBC", "Lipid"]}
        result = build_comparison(a, b)
        assert result["test_count_delta"] == 0
        assert result["more_tests_side"] is None
        assert result["identical_tests"] is False
        assert {e["name"] for e in result["extra_tests_a"]} == {"Sugar"}
        assert {e["name"] for e in result["extra_tests_b"]} == {"Lipid"}

    def test_test_vs_package_never_sets_both_packages(self):
        a = {"kind": "test", "rate_inr": 500}
        b = {"kind": "package", "price_inr": 1000, "tests": ["CBC"]}
        result = build_comparison(a, b)
        assert result["both_packages"] is False
        assert result["identical_tests"] is None
        assert result["test_count_delta"] is None


class TestBuildComparisonNotFound:
    def test_both_not_found_short_circuits_with_no_arithmetic(self):
        a = {"kind": "not_found"}
        b = {"kind": "not_found"}
        result = build_comparison(a, b)
        assert result["a_found"] is False and result["b_found"] is False
        assert result["price_delta"] is None
        assert result["cheaper"] is None

    def test_one_not_found_short_circuits_with_no_arithmetic(self):
        a = {"kind": "not_found"}
        b = {"kind": "test", "rate_inr": 500}
        result = build_comparison(a, b)
        assert result["a_found"] is False
        assert result["b_found"] is True
        assert result["price_delta"] is None


# --------------------------------------------------------------------- #
# 2. agent/reply_templates.py -- compare_options_reply()
# --------------------------------------------------------------------- #

_LANGUAGES = ("english", "hinglish", "banglish", "bengali")


class TestCompareOptionsReply:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_both_found_price_only_mentions_the_cheaper_name_and_exact_delta(self, language):
        a = {"kind": "test", "rate_inr": 650, "test_name": "CBC"}
        b = {"kind": "test", "rate_inr": "750.50", "test_name": "Lipid Profile"}
        comparison = build_comparison(a, b)
        reply = compare_options_reply("CBC", "Lipid Profile", a, b, comparison, language=language)
        assert "100.50" in reply
        assert "CBC" in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_both_not_found_names_both_terms_without_fabricating_a_comparison(self, language):
        a = {"kind": "not_found"}
        b = {"kind": "not_found"}
        comparison = build_comparison(a, b)
        reply = compare_options_reply("Foo Test", "Bar Package", a, b, comparison, language=language)
        assert "Foo Test" in reply
        assert "Bar Package" in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_one_not_found_names_the_missing_one_and_does_not_compare(self, language):
        a = {"kind": "not_found"}
        b = {"kind": "test", "rate_inr": 500, "test_name": "CBC"}
        comparison = build_comparison(a, b)
        reply = compare_options_reply("XYZ Test", "CBC", a, b, comparison, language=language)
        assert "XYZ Test" in reply
        # No fabricated price fact for a side that was never found.
        assert "500" not in reply

    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_package_component_difference_names_specific_extra_tests(self, language):
        a = {"kind": "package", "price_inr": "1500", "package_name": "Diabetes Package",
             "tests": ["Fasting Sugar", "HbA1c"]}
        b = {"kind": "package", "price_inr": "1200", "package_name": "Basic Sugar Package",
             "tests": ["Fasting Sugar"]}
        comparison = build_comparison(a, b)
        reply = compare_options_reply("Diabetes Package", "Basic Sugar Package", a, b, comparison, language=language)
        assert "HbA1c" in reply

    def test_equal_price_states_the_fact_without_naming_a_cheaper_side(self):
        a = {"kind": "test", "rate_inr": 500, "test_name": "CBC"}
        b = {"kind": "test", "rate_inr": 500, "test_name": "Lipid"}
        comparison = build_comparison(a, b)
        reply = compare_options_reply("CBC", "Lipid", a, b, comparison, language="english")
        assert "cost the same" in reply.lower()

    def test_bengali_prefers_seeded_alias_when_available(self):
        a = {"kind": "test", "rate_inr": 500, "test_name": "CBC", "test_name_bn": "সিবিসি"}
        b = {"kind": "test", "rate_inr": 600, "test_name": "Lipid Profile"}
        comparison = build_comparison(a, b)
        reply = compare_options_reply("CBC", "Lipid Profile", a, b, comparison, language="bengali")
        assert "সিবিসি" in reply

    def test_non_bengali_language_never_speaks_a_bengali_alias(self):
        a = {"kind": "test", "rate_inr": 500, "test_name": "CBC", "test_name_bn": "সিবিসি"}
        b = {"kind": "test", "rate_inr": 600, "test_name": "Lipid Profile"}
        comparison = build_comparison(a, b)
        for language in ("english", "hinglish", "banglish"):
            reply = compare_options_reply("CBC", "Lipid Profile", a, b, comparison, language=language)
            assert "সিবিসি" not in reply

    def test_reply_never_contains_a_literal_colon_or_bracket_field_label(self):
        # Same spoken-punctuation discipline this whole file enforces
        # everywhere else (see module docstring's "SPOKEN PUNCTUATION").
        a = {"kind": "package", "price_inr": "1500", "tests": ["A", "B"]}
        b = {"kind": "package", "price_inr": "1200", "tests": ["A"]}
        comparison = build_comparison(a, b)
        for language in _LANGUAGES:
            reply = compare_options_reply("Pkg A", "Pkg B", a, b, comparison, language=language)
            assert ":" not in reply
            assert "[" not in reply and "]" not in reply


# --------------------------------------------------------------------- #
# 3. agent/llm.py -- schema + CLINICAL SAFETY NOTE guardrail
# --------------------------------------------------------------------- #

class TestLlmSchemaAndGuardrail:
    def test_compare_options_is_a_valid_intent(self):
        assert "compare_options" in VALID_INTENTS

    def test_compare_option_slots_are_recognised(self):
        assert "compare_option_a" in _SLOT_KEYS
        assert "compare_option_b" in _SLOT_KEYS

    def test_system_prompt_contains_a_clinical_safety_guardrail_for_compare_options(self):
        prompt = SYSTEM_PROMPT_TEMPLATE
        assert "CLINICAL SAFETY NOTE" in prompt
        assert "compare_options" in prompt.split("CLINICAL SAFETY NOTE", 1)[1][:400]

    def test_validate_accepts_compare_options_payload_with_both_slots(self):
        slots = {k: None for k in _SLOT_KEYS}
        slots.update(compare_option_a="CBC", compare_option_b="Lipid Profile")
        data = {"intents": [{"intent": "compare_options", "slots": slots}], "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_accepts_compare_options_payload_with_one_slot_missing(self):
        # Same tolerance as insurance_coverage's own two-required-slots
        # shape -- an individual missing slot is never fatal by itself.
        slots = {k: None for k in _SLOT_KEYS}
        slots.update(compare_option_a="CBC")
        data = {"intents": [{"intent": "compare_options", "slots": slots}], "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_direct_reply_bn_is_always_stripped_for_compare_options(self):
        # Structural half of AC 3: compare_options is never smalltalk, so
        # _validate() must strip any direct_reply_bn the model tried to
        # inject -- see this function's own is_single_smalltalk check.
        slots = {k: None for k in _SLOT_KEYS}
        slots.update(compare_option_a="CBC", compare_option_b="Lipid Profile")
        data = {
            "intents": [{"intent": "compare_options", "slots": slots}],
            "direct_reply_bn": "Lipid Profile ta beshi bhalo hobe apnar jonno.",
        }
        ok, errors = _validate(data)
        assert ok is True, errors
        assert data["direct_reply_bn"] is None


# --------------------------------------------------------------------- #
# 4. agent/semantic_cache.py -- L2 exclusion
# --------------------------------------------------------------------- #

class TestSemanticCacheCompareOptionsGuard:
    def test_compare_options_never_l2_eligible_even_with_both_slots(self):
        value = {"intent": "compare_options",
                 "slots": {"compare_option_a": "CBC", "compare_option_b": "Lipid Profile"}}
        assert SemanticCache._is_l2_eligible(value) is False

    def test_compare_options_excluded_even_with_only_one_slot(self):
        value = {"intent": "compare_options", "slots": {"compare_option_a": "CBC"}}
        assert SemanticCache._is_l2_eligible(value) is False

    def test_ordinary_single_entity_intent_eligibility_is_unaffected(self):
        # Confirms this story's change is purely additive -- test_rate's
        # own eligibility (an existing, unrelated intent) is untouched.
        value = {"intent": "test_rate", "slots": {"test_name": "CBC"}}
        assert SemanticCache._is_l2_eligible(value) is True


# --------------------------------------------------------------------- #
# 5. main.py / main_pcm.py dispatch -- parametrized over both transports
# --------------------------------------------------------------------- #

class FakeToolsClient:
    def __init__(self, **responses):
        self.calls = {}
        self._responses = responses

    def _record(self, name, *args):
        self.calls.setdefault(name, []).append(args)

    async def get_test_rate(self, test_name):
        self._record("get_test_rate", test_name)
        resp = self._responses.get("test_rate", {}).get(test_name)
        if isinstance(resp, Exception):
            raise resp
        if resp is not None:
            return resp
        return {"found": False, "query": test_name, "did_you_mean": []}

    async def search_health_package(self, package_name):
        self._record("search_health_package", package_name)
        resp = self._responses.get("health_package", {}).get(package_name)
        if isinstance(resp, Exception):
            raise resp
        if resp is not None:
            return resp
        return {"found": False, "query": package_name, "did_you_mean": []}


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
    tools = FakeToolsClient(
        test_rate={
            "CBC": {"found": True, "test_name": "CBC", "test_name_bn": None, "rate_inr": 650,
                     "sample_type": "Blood", "report_time_hours": 24},
            "Lipid Profile": {"found": True, "test_name": "Lipid Profile", "test_name_bn": None,
                                "rate_inr": "750.50", "sample_type": "Blood", "report_time_hours": 24},
        },
        health_package={
            "Diabetes Package": {"found": True, "package_name": "Diabetes Package", "package_name_bn": None,
                                   "price_inr": "1500.00", "tests": ["Fasting Sugar", "HbA1c"], "tests_bn": []},
            "Basic Sugar Package": {"found": True, "package_name": "Basic Sugar Package", "package_name_bn": None,
                                      "price_inr": "1200.00", "tests": ["Fasting Sugar"], "tests_bn": []},
        },
    )
    monkeypatch.setattr(transport, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools, transport=transport)


def _empty_compare_slots(**overrides):
    slots = {
        "test_name": None, "doctor_name": None, "department": None, "date": None,
        "time_slot": None, "patient_name": None, "phone": None, "package_name": None,
        "info_topic": None, "insurance_provider_name": None,
        "compare_option_a": None, "compare_option_b": None,
    }
    slots.update(overrides)
    return slots


def _dispatch(stub, monkeypatch, intent, slots, tmp_path, pending=None):
    async def fake_resolve_intent(session, text):
        return {"intent": intent, "slots": slots}

    monkeypatch.setattr(stub.transport, "_resolve_intent", fake_resolve_intent)
    session = make_session(pending=pending)
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(stub.transport._dispatch_turn(session, str(wav_path)))
    return session


def _continue(stub, session, text):
    return run(stub.transport._continue_pending(session, text))


class TestCompareOptionsDispatch:
    def test_both_slots_present_resolves_both_entities_and_speaks_one_reply(self, stub, monkeypatch, tmp_path):
        slots = _empty_compare_slots(compare_option_a="CBC", compare_option_b="Lipid Profile")
        session = _dispatch(stub, monkeypatch, "compare_options", slots, tmp_path)

        assert stub.tools.calls["get_test_rate"] == [("CBC",), ("Lipid Profile",)]
        assert "search_health_package" not in stub.tools.calls
        assert len(stub.spoken) == 1
        assert "100.50" in stub.spoken[0]
        assert session.pending is None

    def test_missing_second_option_asks_for_it_and_keeps_the_first(self, stub, monkeypatch, tmp_path):
        slots = _empty_compare_slots(compare_option_a="CBC")
        session = _dispatch(stub, monkeypatch, "compare_options", slots, tmp_path)

        assert "get_test_rate" not in stub.tools.calls
        assert stub.spoken == [missing_slot_prompt("compare_options", "compare_option_b")]
        assert session.pending["awaiting"] == "compare_options_slot"
        assert session.pending["slots"]["compare_option_a"] == "CBC"
        assert session.pending["missing_field"] == "compare_option_b"

    def test_missing_first_option_asks_for_it_and_keeps_the_second(self, stub, monkeypatch, tmp_path):
        slots = _empty_compare_slots(compare_option_b="Lipid Profile")
        session = _dispatch(stub, monkeypatch, "compare_options", slots, tmp_path)

        assert session.pending["missing_field"] == "compare_option_a"
        assert session.pending["slots"]["compare_option_b"] == "Lipid Profile"

    def test_continuation_fills_missing_second_option_then_resolves_both(self, stub, monkeypatch, tmp_path):
        slots = _empty_compare_slots(compare_option_a="CBC")
        session = _dispatch(stub, monkeypatch, "compare_options", slots, tmp_path)
        stub.spoken.clear()

        handled = _continue(stub, session, "Lipid Profile")

        assert handled is True
        # Neither side is resolved until BOTH names are known -- the
        # first dispatch turn (missing compare_option_b) returns before
        # calling any tool at all, same as insurance_coverage's own
        # merge-onto-pending pattern never calls get_insurance_coverage
        # until its second required slot is filled.
        assert stub.tools.calls["get_test_rate"] == [("CBC",), ("Lipid Profile",)]
        assert session.pending is None
        assert len(stub.spoken) == 1

    def test_a_test_compared_against_a_package_resolves_each_independently(self, stub, monkeypatch, tmp_path):
        slots = _empty_compare_slots(compare_option_a="CBC", compare_option_b="Diabetes Package")
        _dispatch(stub, monkeypatch, "compare_options", slots, tmp_path)

        assert stub.tools.calls["get_test_rate"] == [("CBC",), ("Diabetes Package",)]
        assert stub.tools.calls["search_health_package"] == [("Diabetes Package",)]
        assert len(stub.spoken) == 1

    def test_two_packages_compares_price_and_components_in_one_reply(self, stub, monkeypatch, tmp_path):
        slots = _empty_compare_slots(compare_option_a="Diabetes Package", compare_option_b="Basic Sugar Package")
        _dispatch(stub, monkeypatch, "compare_options", slots, tmp_path)

        reply = stub.spoken[0]
        assert "300" in reply  # price delta: 1500.00 - 1200.00
        assert "HbA1c" in reply  # the extra component

    def test_unknown_first_option_is_honestly_reported_not_fabricated(self, stub, monkeypatch, tmp_path):
        slots = _empty_compare_slots(compare_option_a="Nonexistent Thing", compare_option_b="CBC")
        session = _dispatch(stub, monkeypatch, "compare_options", slots, tmp_path)

        assert "Nonexistent Thing" in stub.spoken[0]
        assert session.pending is None

    def test_pending_string_is_not_shared_with_insurance_coverage_slot(self, stub, monkeypatch, tmp_path):
        slots = _empty_compare_slots(compare_option_a="CBC")
        session = _dispatch(stub, monkeypatch, "compare_options", slots, tmp_path)
        assert session.pending["awaiting"] != "insurance_coverage_slot"

    def test_continuation_negative_reply_abandons_the_comparison(self, stub, monkeypatch, tmp_path):
        slots = _empty_compare_slots(compare_option_a="CBC")
        session = _dispatch(stub, monkeypatch, "compare_options", slots, tmp_path)
        stub.spoken.clear()

        handled = _continue(stub, session, "না")

        assert handled is True
        assert session.pending is None
        assert "get_test_rate" not in stub.tools.calls or len(stub.tools.calls["get_test_rate"]) == 1


class TestTransportParity:
    def test_both_transports_produce_the_same_reply_text(self, monkeypatch, tmp_path):
        slots = _empty_compare_slots(compare_option_a="CBC", compare_option_b="Lipid Profile")
        results = {}
        for transport in TRANSPORTS:
            spoken = []

            async def fake_speak(session, text, fallback_reason=None, _spoken=spoken):
                _spoken.append(text)

            async def fake_resolve_intent(session, text):
                return {"intent": "compare_options", "slots": slots}

            tools = FakeToolsClient(test_rate={
                "CBC": {"found": True, "test_name": "CBC", "test_name_bn": None, "rate_inr": 650,
                         "sample_type": "Blood", "report_time_hours": 24},
                "Lipid Profile": {"found": True, "test_name": "Lipid Profile", "test_name_bn": None,
                                    "rate_inr": "750.50", "sample_type": "Blood", "report_time_hours": 24},
            })
            monkeypatch.setattr(transport, "_speak", fake_speak)
            monkeypatch.setattr(transport, "_asr", FakeASR())
            monkeypatch.setattr(transport, "_tools", tools)
            monkeypatch.setattr(transport, "_resolve_intent", fake_resolve_intent)

            session = make_session()
            wav_path = tmp_path / f"utt-{transport.__name__}.wav"
            wav_path.write_bytes(b"")
            run(transport._dispatch_turn(session, str(wav_path)))
            results[transport.__name__] = spoken[0]

        assert results["main"] == results["main_pcm"]


# --------------------------------------------------------------------- #
# 6. Clinical safety boundary, end to end (AC 3)
# --------------------------------------------------------------------- #

# Words/phrases that would signal a recommendation on medical/clinical
# grounds -- a reply containing any of these (in a comparison context)
# would mean this story's central safety guarantee failed. Checked
# case-insensitively against every reply this module can ever produce.
_FORBIDDEN_RECOMMENDATION_MARKERS = (
    "better for you", "recommend", "you should choose", "you should get",
    "healthier", "more suitable", "best option for", "clinically better",
    "I suggest you", "advise you to",
)


class TestClinicalSafetyBoundary:
    @pytest.mark.parametrize("language", _LANGUAGES)
    def test_no_reply_ever_contains_recommendation_language(self, language):
        cases = [
            build_comparison({"kind": "test", "rate_inr": 650, "test_name": "CBC"},
                              {"kind": "test", "rate_inr": "750.50", "test_name": "Lipid Profile"}),
            build_comparison({"kind": "package", "price_inr": "1500", "package_name": "Diabetes Package",
                               "tests": ["Fasting Sugar", "HbA1c"]},
                              {"kind": "package", "price_inr": "1200", "package_name": "Basic Sugar Package",
                               "tests": ["Fasting Sugar"]}),
            build_comparison({"kind": "not_found"}, {"kind": "test", "rate_inr": 500, "test_name": "CBC"}),
            build_comparison({"kind": "not_found"}, {"kind": "not_found"}),
        ]
        entities = [
            ({"kind": "test", "rate_inr": 650, "test_name": "CBC"},
             {"kind": "test", "rate_inr": "750.50", "test_name": "Lipid Profile"}),
            ({"kind": "package", "price_inr": "1500", "package_name": "Diabetes Package",
              "tests": ["Fasting Sugar", "HbA1c"]},
             {"kind": "package", "price_inr": "1200", "package_name": "Basic Sugar Package",
              "tests": ["Fasting Sugar"]}),
            ({"kind": "not_found"}, {"kind": "test", "rate_inr": 500, "test_name": "CBC"}),
            ({"kind": "not_found"}, {"kind": "not_found"}),
        ]
        for comparison, (entity_a, entity_b) in zip(cases, entities):
            reply = compare_options_reply("Option A", "Option B", entity_a, entity_b, comparison, language=language)
            lowered = reply.lower()
            for marker in _FORBIDDEN_RECOMMENDATION_MARKERS:
                assert marker not in lowered, f"forbidden marker {marker!r} found in: {reply!r}"

    def test_build_comparison_output_has_no_recommendation_shaped_key(self):
        # Structural guarantee, not just a wording check -- there is no
        # field this dict could ever carry a recommendation in, so no
        # future change to compare_options_reply() could surface one that
        # doesn't already exist in the data.
        result = build_comparison({"kind": "test", "rate_inr": 500}, {"kind": "test", "rate_inr": 600})
        forbidden_keys = {"better", "best", "recommended", "recommendation", "suggestion", "winner"}
        assert forbidden_keys.isdisjoint(result.keys())

    def test_llm_prompt_explicitly_forbids_choosing_a_better_option(self):
        section = SYSTEM_PROMPT_TEMPLATE.split("CLINICAL SAFETY NOTE", 1)[1][:800].lower()
        assert "better" in section or "recommend" in section

    def test_direct_reply_bn_cannot_carry_a_recommendation_through_validation(self):
        # Even if the model tried, _validate() strips it -- confirmed
        # already by TestLlmSchemaAndGuardrail above; repeated here with
        # explicitly recommendation-shaped text for this section's own
        # end-to-end clarity.
        slots = {k: None for k in _SLOT_KEYS}
        slots.update(compare_option_a="CBC", compare_option_b="Lipid Profile")
        data = {
            "intents": [{"intent": "compare_options", "slots": slots}],
            "direct_reply_bn": "Lipid Profile apnar jonno beshi bhalo hobe, oita nin.",
        }
        ok, _ = _validate(data)
        assert ok is True
        assert data["direct_reply_bn"] is None
