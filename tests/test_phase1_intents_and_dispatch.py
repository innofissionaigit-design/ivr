"""ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables (Walk-in
Eligibility, Prescription Requirements, Insurance Coverage Policy,
Outstanding Balance / Billing stories).

Mirrors tests/test_test_preparation_intent.py's own established 3-place
pattern (see that file's module docstring) across all four new intents:

  1. agent/llm.py: each is a valid classification target, and
     "insurance_provider_name" is a real slot _validate() checks for.
  2. agent/semantic_cache.py: walkin_eligibility/prescription_requirements
     are single-required-slot intents (test_name), same as test_rate/
     test_preparation; insurance_coverage/billing_balance are excluded
     from L2 entirely (same treatment as book_appointment/report_status).
  3. main.py AND main_pcm.py dispatch -- parametrized over BOTH modules
     wherever the assertion is transport-independent, so a gap between
     the two (the exact "test_sample" bug this codebase has already hit
     once for real, per main_pcm.py's own module docstring) cannot slip
     through unnoticed.

Reply-text-level coverage lives in tests/test_phase1_reply_templates.py.
API-level coverage lives in tests/test_walkin_eligibility_api.py,
tests/test_prescription_requirements_api.py,
tests/test_insurance_coverage_api.py, tests/test_billing_balance_api.py.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from agent.llm import VALID_INTENTS, _validate
from agent.semantic_cache import _REQUIRED_ENTITY_FOR_INTENT, _ENTITY_SLOTS, SemanticCache

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
# agent/llm.py
# --------------------------------------------------------------------- #

class TestPhase1IntentsAreValid:
    @pytest.mark.parametrize("intent", [
        "walkin_eligibility", "prescription_requirements",
        "insurance_coverage", "billing_balance",
    ])
    def test_registered_in_valid_intents(self, intent):
        assert intent in VALID_INTENTS

    def test_distinct_from_each_other_and_from_test_rate(self):
        # No accidental aliasing -- four genuinely distinct intents.
        phase1 = {"walkin_eligibility", "prescription_requirements",
                  "insurance_coverage", "billing_balance"}
        assert len(phase1) == 4
        assert "test_rate" not in phase1

    def test_validate_accepts_walkin_eligibility_payload(self):
        data = {"intent": "walkin_eligibility", "slots": _empty_slots(test_name="CBC"),
                "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_accepts_prescription_requirements_payload(self):
        data = {"intent": "prescription_requirements", "slots": _empty_slots(test_name="CBC"),
                "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_accepts_insurance_coverage_payload_with_both_slots(self):
        data = {
            "intent": "insurance_coverage",
            "slots": _empty_slots(test_name="CBC", insurance_provider_name="Star Health"),
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_accepts_insurance_coverage_payload_with_one_slot_missing(self):
        # agent/llm.py's own intent description: "if the caller names
        # only one, still classify this intent and fill whichever slot
        # they gave" -- a partial extraction is still schema-valid.
        data = {
            "intent": "insurance_coverage",
            "slots": _empty_slots(test_name="CBC", insurance_provider_name=None),
            "direct_reply_bn": None,
        }
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_accepts_billing_balance_payload(self):
        data = {"intent": "billing_balance", "slots": _empty_slots(phone="9000000001"),
                "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is True, errors

    def test_validate_still_rejects_unknown_intent(self):
        data = {"intent": "billing_dues", "slots": _empty_slots(), "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert ok is False
        assert any("invalid intent" in e for e in errors)

    def test_validate_flags_insurance_provider_name_key_when_absent(self):
        # A slots dict missing the NEW key entirely (e.g. an older cached
        # extraction) is flagged in `errors`, the same non-fatal-missing-
        # key diagnostic _validate() already gives for any other absent
        # slot key (see its own return-line comment: a "missing" key
        # error alone does not flip `ok` to False -- only an invalid
        # intent or a non-dict `slots` does).
        slots = _empty_slots()
        del slots["insurance_provider_name"]
        data = {"intent": "insurance_coverage", "slots": slots, "direct_reply_bn": None}
        ok, errors = _validate(data)
        assert any("insurance_provider_name" in e and "missing" in e for e in errors)


# --------------------------------------------------------------------- #
# agent/semantic_cache.py
# --------------------------------------------------------------------- #

class TestPhase1SemanticCacheRegistration:
    def test_insurance_provider_name_is_a_registered_entity_slot(self):
        assert "insurance_provider_name" in _ENTITY_SLOTS

    @pytest.mark.parametrize("intent", ["walkin_eligibility", "prescription_requirements"])
    def test_single_slot_intents_registered_with_test_name(self, intent):
        assert _REQUIRED_ENTITY_FOR_INTENT.get(intent) == "test_name"

    @pytest.mark.parametrize("intent", ["walkin_eligibility", "prescription_requirements"])
    def test_single_slot_intents_l2_eligible_with_test_name_present(self, intent):
        assert SemanticCache._is_l2_eligible({"intent": intent, "slots": {"test_name": "CBC"}}) is True

    @pytest.mark.parametrize("intent", ["walkin_eligibility", "prescription_requirements"])
    def test_single_slot_intents_not_l2_eligible_missing_test_name(self, intent):
        assert SemanticCache._is_l2_eligible({"intent": intent, "slots": {"test_name": None}}) is False

    def test_insurance_coverage_never_l2_eligible_even_with_both_slots(self):
        # Same treatment as book_appointment -- two entity-shaped slots,
        # either of which can be missing on a given turn, so there is no
        # ONE required field the way test_rate/etc. have.
        assert "insurance_coverage" not in _REQUIRED_ENTITY_FOR_INTENT
        value = {"intent": "insurance_coverage",
                 "slots": {"test_name": "CBC", "insurance_provider_name": "Star Health"}}
        assert SemanticCache._is_l2_eligible(value) is False

    def test_billing_balance_never_l2_eligible_even_with_phone(self):
        # Same treatment as report_status/report_send -- caller-specific
        # identity-bound data, not a fact that's the same for every caller.
        assert "billing_balance" not in _REQUIRED_ENTITY_FOR_INTENT
        value = {"intent": "billing_balance", "slots": {"phone": "9000000001"}}
        assert SemanticCache._is_l2_eligible(value) is False

    def test_billing_balance_excluded_even_with_no_phone_stated(self):
        value = {"intent": "billing_balance", "slots": {}}
        assert SemanticCache._is_l2_eligible(value) is False


# --------------------------------------------------------------------- #
# main.py / main_pcm.py dispatch -- parametrized over both transports
# --------------------------------------------------------------------- #

class FakeToolsClient:
    def __init__(self, **responses):
        self.calls = {}
        self._responses = responses

    def _record(self, name, *args):
        self.calls.setdefault(name, []).append(args)

    async def get_walkin_policy(self, test_name):
        self._record("get_walkin_policy", test_name)
        return self._responses.get("walkin_policy", {
            "found": True, "test_name": test_name, "test_name_bn": None,
            "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am",
        })

    async def get_prescription_policy(self, test_name):
        self._record("get_prescription_policy", test_name)
        return self._responses.get("prescription_policy", {
            "found": True, "test_name": test_name, "test_name_bn": None,
            "policy_available": True, "prescription_required": True,
            "prescription_channels": ["whatsapp_photo"],
        })

    async def get_insurance_coverage(self, test_name, provider_name):
        self._record("get_insurance_coverage", test_name, provider_name)
        return self._responses.get("insurance_coverage", {
            "test_found": True, "test_name": test_name,
            "provider_found": True, "provider_name": provider_name,
            "policy_available": True, "coverage_status": "COVERED", "pre_auth_required": False,
        })

    async def get_patient_billing(self, phone):
        self._record("get_patient_billing", phone)
        return self._responses.get("patient_billing", {
            "patient_found": True, "found": True,
            "outstanding_amount": "1250.50", "due_date": None,
        })


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


class TestWalkinEligibilityDispatch:
    def test_calls_get_walkin_policy_and_speaks_reply(self, stub, monkeypatch, tmp_path):
        from agent.reply_templates import walkin_eligibility_reply

        slots = _empty_slots(test_name="CBC")
        _dispatch(stub, monkeypatch, "walkin_eligibility", slots, tmp_path)

        assert stub.tools.calls["get_walkin_policy"] == [("CBC",)]
        assert len(stub.spoken) == 1
        expected = walkin_eligibility_reply(slots, stub.tools._responses.get("walkin_policy") or {
            "found": True, "test_name": "CBC", "test_name_bn": None,
            "policy_available": True, "walkin_eligible": True, "walkin_hours": "Mon-Sat 7am-11am",
        })
        assert stub.spoken[0] == expected

    def test_missing_test_name_asks_without_calling_the_tool(self, stub, monkeypatch, tmp_path):
        from agent.reply_templates import missing_slot_prompt

        _dispatch(stub, monkeypatch, "walkin_eligibility", _empty_slots(), tmp_path)

        assert "get_walkin_policy" not in stub.tools.calls
        assert stub.spoken == [missing_slot_prompt("walkin_eligibility", "test_name")]


class TestPrescriptionRequirementsDispatch:
    def test_calls_get_prescription_policy_and_speaks_reply(self, stub, monkeypatch, tmp_path):
        slots = _empty_slots(test_name="CBC")
        _dispatch(stub, monkeypatch, "prescription_requirements", slots, tmp_path)

        assert stub.tools.calls["get_prescription_policy"] == [("CBC",)]
        assert len(stub.spoken) == 1

    def test_missing_test_name_asks_without_calling_the_tool(self, stub, monkeypatch, tmp_path):
        from agent.reply_templates import missing_slot_prompt

        _dispatch(stub, monkeypatch, "prescription_requirements", _empty_slots(), tmp_path)

        assert "get_prescription_policy" not in stub.tools.calls
        assert stub.spoken == [missing_slot_prompt("prescription_requirements", "test_name")]


class TestBillingBalanceDispatch:
    def test_phone_present_calls_tool_directly(self, stub, monkeypatch, tmp_path):
        slots = _empty_slots(phone="9000000001")
        session = _dispatch(stub, monkeypatch, "billing_balance", slots, tmp_path)

        assert stub.tools.calls["get_patient_billing"] == [("9000000001",)]
        assert len(stub.spoken) == 1
        assert session.pending is None

    def test_missing_phone_sets_billing_phone_pending_and_asks(self, stub, monkeypatch, tmp_path):
        from agent.reply_templates import missing_slot_prompt

        session = _dispatch(stub, monkeypatch, "billing_balance", _empty_slots(), tmp_path)

        assert "get_patient_billing" not in stub.tools.calls
        assert stub.spoken == [missing_slot_prompt("billing_balance", "phone")]
        assert session.pending == {"awaiting": "billing_phone", "retries": 0}

    def test_continuation_with_a_spoken_phone_completes_the_lookup(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "billing_balance", _empty_slots(), tmp_path)
        stub.spoken.clear()

        handled = _continue(stub, session, "amar number 9000000001")

        assert handled is True
        assert stub.tools.calls["get_patient_billing"] == [("9000000001",)]
        assert len(stub.spoken) == 1

    def test_continuation_with_a_spoken_phone_clears_pending(self, stub, monkeypatch, tmp_path):
        # REGRESSION -- found by Phase 2's real end-to-end integration
        # test (tests/test_phase1_end_to_end_integration.py): this branch
        # used to speak the real answer but never reset session.pending,
        # so the call stayed stuck "awaiting a phone number" and the
        # caller's next utterance, whatever it actually was, would have
        # been misinterpreted as another phone attempt. The FakeToolsClient
        # here always answers `found: True`, so this test alone could not
        # have caught it -- it only asserted `spoken`/`tools.calls`, never
        # `session.pending`, which is exactly why it slipped through
        # until real seeded content made the gap observable end-to-end.
        session = _dispatch(stub, monkeypatch, "billing_balance", _empty_slots(), tmp_path)

        handled = _continue(stub, session, "amar number 9000000001")

        assert handled is True
        assert session.pending is None

    def test_continuation_with_unparseable_phone_reprompts_then_gives_up(self, stub, monkeypatch, tmp_path):
        session = _dispatch(stub, monkeypatch, "billing_balance", _empty_slots(), tmp_path)
        stub.spoken.clear()

        assert _continue(stub, session, "ওইটা মনে নেই") is True
        assert _continue(stub, session, "ওইটা মনে নেই") is True
        assert _continue(stub, session, "ওইটা মনে নেই") is False  # 3rd failure gives up
        assert session.pending is None
        assert "get_patient_billing" not in stub.tools.calls

    def test_pending_string_is_not_the_shared_report_phone_or_bare_phone_string(self, stub, monkeypatch, tmp_path):
        # RULE: own distinct string, same collision reasoning as
        # "report_phone" vs the booking flow's bare "phone" (see
        # _continue_pending's own docstring).
        session = _dispatch(stub, monkeypatch, "billing_balance", _empty_slots(), tmp_path)
        assert session.pending["awaiting"] not in ("phone", "report_phone")


class TestInsuranceCoverageDispatch:
    def test_both_slots_present_calls_tool_directly(self, stub, monkeypatch, tmp_path):
        slots = _empty_slots(test_name="CBC", insurance_provider_name="Star Health")
        session = _dispatch(stub, monkeypatch, "insurance_coverage", slots, tmp_path)

        assert stub.tools.calls["get_insurance_coverage"] == [("CBC", "Star Health")]
        assert len(stub.spoken) == 1
        assert session.pending is None

    def test_missing_provider_asks_for_it_and_keeps_test_name(self, stub, monkeypatch, tmp_path):
        from agent.reply_templates import missing_slot_prompt

        slots = _empty_slots(test_name="CBC")
        session = _dispatch(stub, monkeypatch, "insurance_coverage", slots, tmp_path)

        assert "get_insurance_coverage" not in stub.tools.calls
        assert stub.spoken == [missing_slot_prompt("insurance_coverage", "insurance_provider_name")]
        assert session.pending["awaiting"] == "insurance_coverage_slot"
        assert session.pending["slots"]["test_name"] == "CBC"
        assert session.pending["missing_field"] == "insurance_provider_name"

    def test_missing_test_name_asks_for_it_and_keeps_provider(self, stub, monkeypatch, tmp_path):
        slots = _empty_slots(insurance_provider_name="Star Health")
        session = _dispatch(stub, monkeypatch, "insurance_coverage", slots, tmp_path)

        assert session.pending["missing_field"] == "test_name"
        assert session.pending["slots"]["insurance_provider_name"] == "Star Health"

    def test_continuation_fills_missing_provider_then_calls_tool(self, stub, monkeypatch, tmp_path):
        slots = _empty_slots(test_name="CBC")
        session = _dispatch(stub, monkeypatch, "insurance_coverage", slots, tmp_path)
        stub.spoken.clear()

        handled = _continue(stub, session, "Star Health")

        assert handled is True
        assert stub.tools.calls["get_insurance_coverage"] == [("CBC", "Star Health")]
        assert session.pending is None

    def test_continuation_fills_missing_test_name_then_calls_tool(self, stub, monkeypatch, tmp_path):
        slots = _empty_slots(insurance_provider_name="Star Health")
        session = _dispatch(stub, monkeypatch, "insurance_coverage", slots, tmp_path)
        stub.spoken.clear()

        handled = _continue(stub, session, "CBC")

        assert handled is True
        assert stub.tools.calls["get_insurance_coverage"] == [("CBC", "Star Health")]

    def test_negative_reply_during_slot_fill_abandons_cleanly(self, stub, monkeypatch, tmp_path):
        slots = _empty_slots(test_name="CBC")
        session = _dispatch(stub, monkeypatch, "insurance_coverage", slots, tmp_path)

        handled = _continue(stub, session, "না থাক")
        assert handled is True
        assert session.pending is None
        assert "get_insurance_coverage" not in stub.tools.calls


# --------------------------------------------------------------------- #
# Transport parity: the exact "test_sample" gap this codebase already hit
# once (main_pcm.py's own module docstring) -- confirm all four new
# branches exist verbatim in BOTH files, not hand-duplicated and drifted.
# --------------------------------------------------------------------- #

class TestTransportParity:
    @pytest.mark.parametrize("marker", [
        'elif intent == "walkin_eligibility":',
        'elif intent == "prescription_requirements":',
        'elif intent == "insurance_coverage":',
        'elif intent == "billing_balance":',
        'if awaiting == "billing_phone":',
        'if awaiting == "insurance_coverage_slot":',
    ])
    def test_branch_present_in_both_transports(self, marker):
        import inspect
        main_src = inspect.getsource(main)
        main_pcm_src = inspect.getsource(main_pcm)
        assert marker in main_src, f"{marker!r} missing from main.py"
        assert marker in main_pcm_src, f"{marker!r} missing from main_pcm.py"
