"""ADDED BY SOURAV -- Phase 2: true end-to-end integration tests for the
4 Phase 1 intents (Walk-in Eligibility, Prescription Requirements,
Insurance Coverage Policy, Outstanding Balance / Billing), wiring every
real layer together at once: main.py/main_pcm.py dispatch -> a REAL
ClinicToolsClient -> a REAL clinic-api FastAPI app (bound via httpx's
ASGI transport, no real socket or server process) -> the REAL Phase 2
seeded SQLite database -> agent/reply_templates.py.

Distinct from the other Phase 1/2 test files:
  - tests/test_phase1_intents_and_dispatch.py stubs ToolsClient entirely,
    to isolate dispatch/pending-slot logic from any network or DB call.
  - tests/test_walkin_eligibility_api.py & friends hit clinic-api
    directly over HTTP, but never go through main.py's own dispatch/
    session/pending-slot machinery.
This file is the one place all of those layers run together against
real seeded data -- closest to what an actual call exercises end-to-end,
short of real ASR/TTS/LLM extraction (those stories are tested
elsewhere; `_resolve_intent` and `_asr` are faked here, same as the
other dispatch test file, to inject a fixed intent/slots per test).

Parametrized over both main.py and main_pcm.py (the `transport` fixture)
per this codebase's own transport-parity discipline, PLUS an explicit
same-input-same-output cross-transport check at the end.

Language note: main.py/main_pcm.py detect the reply language fresh from
each turn's own utterance text (see _continue_pending's own "Detected
fresh from THIS turn's own utterance" comment). Tests below deliberately
use plain Latin text with no Hinglish/Banglish marker words (test names,
provider names, bare digit strings) so `detect_language()` resolves to
"english" predictably -- see agent/bn_normalize.py's detect_language()
docstring for why marker-free Latin text falls through to "english".
"""
from __future__ import annotations

import asyncio
import os
import sys
import types

import httpx
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CLINIC_API_DIR = os.path.join(TESTS_DIR, "..", "clinic-api")


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def real_clinic_api(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("clinic_test_e2e_live") / "clinic.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    sys.path.insert(0, CLINIC_API_DIR)
    for mod in ("db", "models", "seed", "main"):
        sys.modules.pop(mod, None)
    import db as clinic_db
    import models as clinic_models
    clinic_models.Base.metadata.create_all(clinic_db.engine)
    import seed as clinic_seed
    clinic_seed.seed()

    import main as clinic_main

    yield types.SimpleNamespace(app=clinic_main.app, db=clinic_db, models=clinic_models, seed=clinic_seed)

    sys.path.remove(CLINIC_API_DIR)


@pytest.fixture
def live_tools_client(real_clinic_api):
    """A REAL ClinicToolsClient, wired to the REAL (Phase 2 seeded)
    clinic-api app via httpx's ASGI transport -- exercises the exact same
    HTTP/JSON round-trip (including _parse_exact's digit-fidelity
    parsing) production traffic would, without a real socket or server
    process."""
    from agent.tools_client import ClinicToolsClient

    client = ClinicToolsClient(base_url="http://testserver")
    client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=real_clinic_api.app),
        base_url="http://testserver",
    )
    yield client
    run(client.aclose())


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


class FakeASRResult:
    text = "some utterance"  # marker-free Latin text -> detect_language() = "english"


class FakeASR:
    async def transcribe_utterance(self, wav_path):
        return FakeASRResult()


def make_session(pending=None):
    return types.SimpleNamespace(
        call_id="test-call-e2e", pending=pending,
        dispatch_lock=asyncio.Lock(), send_json=_AsyncNoOp(),
    )


_ALL_SLOT_KEYS = ("test_name", "doctor_name", "department", "date", "time_slot",
                  "patient_name", "phone", "package_name", "info_topic",
                  "insurance_provider_name")


def _empty_slots(**overrides):
    slots = {k: None for k in _ALL_SLOT_KEYS}
    slots.update(overrides)
    return slots


import main
import main_pcm

TRANSPORTS = [main, main_pcm]


@pytest.fixture(params=TRANSPORTS, ids=["main", "main_pcm"])
def transport(request):
    return request.param


@pytest.fixture
def live(transport, live_tools_client, monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    monkeypatch.setattr(transport, "_speak", fake_speak)
    monkeypatch.setattr(transport, "_asr", FakeASR())
    monkeypatch.setattr(transport, "_tools", live_tools_client)
    return types.SimpleNamespace(spoken=spoken, transport=transport)


def _dispatch(live, monkeypatch, intent, slots, tmp_path):
    async def fake_resolve_intent(session, text):
        return {"intent": intent, "slots": slots}

    monkeypatch.setattr(live.transport, "_resolve_intent", fake_resolve_intent)
    session = make_session()
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(live.transport._dispatch_turn(session, str(wav_path)))
    return session


def _continue(live, session, text):
    return run(live.transport._continue_pending(session, text))


class TestWalkinEligibilityEndToEnd:
    def test_reviewed_test_speaks_real_hours(self, live, monkeypatch, tmp_path):
        session = _dispatch(live, monkeypatch, "walkin_eligibility",
                             _empty_slots(test_name="Complete Blood Count (CBC)"), tmp_path)
        assert len(live.spoken) == 1
        assert "Mon-Sat 7:00 AM - 7:00 PM" in live.spoken[0]
        assert session.pending is None

    def test_not_eligible_test_speaks_real_not_eligible_answer(self, live, monkeypatch, tmp_path):
        _dispatch(live, monkeypatch, "walkin_eligibility",
                  _empty_slots(test_name="USG Whole Abdomen"), tmp_path)
        assert "not available on a walk-in basis" in live.spoken[0]

    def test_unreviewed_test_gives_an_honest_reply_not_a_guess(self, live, monkeypatch, tmp_path):
        _dispatch(live, monkeypatch, "walkin_eligibility", _empty_slots(test_name="ECG"), tmp_path)
        text = live.spoken[0]
        assert text  # a real reply was produced
        assert "Mon-Sat" not in text  # never a fabricated hours string


class TestPrescriptionRequirementsEndToEnd:
    def test_reviewed_test_speaks_real_channels(self, live, monkeypatch, tmp_path):
        _dispatch(live, monkeypatch, "prescription_requirements",
                  _empty_slots(test_name="HIV Test (ELISA)"), tmp_path)
        text = live.spoken[0]
        assert "whatsapp photo" in text
        assert "counter in person" in text

    def test_reviewed_test_not_required(self, live, monkeypatch, tmp_path):
        _dispatch(live, monkeypatch, "prescription_requirements",
                  _empty_slots(test_name="Complete Blood Count (CBC)"), tmp_path)
        assert "does not require a doctor's prescription" in live.spoken[0]


class TestInsuranceCoverageEndToEndMultiTurn:
    """Exercises the real multi-turn missing-slot follow-up (per the
    user's explicit ask to verify this) against the real seeded provider/
    policy catalogue, not a mocked tool response."""

    def test_missing_provider_then_continuation_resolves_real_coverage(self, live, monkeypatch, tmp_path):
        session = _dispatch(live, monkeypatch, "insurance_coverage",
                             _empty_slots(test_name="Complete Blood Count (CBC)"), tmp_path)
        assert session.pending["awaiting"] == "insurance_coverage_slot"
        assert session.pending["missing_field"] == "insurance_provider_name"
        live.spoken.clear()

        handled = _continue(live, session, "Star Health")

        assert handled is True
        assert session.pending is None
        text = live.spoken[0]
        assert "covered" in text.lower()
        assert "Star Health and Allied Insurance" in text

    def test_missing_test_name_then_continuation_resolves_real_coverage(self, live, monkeypatch, tmp_path):
        session = _dispatch(live, monkeypatch, "insurance_coverage",
                             _empty_slots(insurance_provider_name="Star Health"), tmp_path)
        assert session.pending["missing_field"] == "test_name"
        live.spoken.clear()

        handled = _continue(live, session, "Lipid Profile")

        assert handled is True
        assert "covered" in live.spoken[0].lower()

    def test_missing_provider_then_bengali_alias_continuation_still_resolves(self, live, monkeypatch, tmp_path):
        # The caller's free-text continuation is taken verbatim (see
        # main.py's own _continue_pending docstring on why -- no fuzzy
        # matching happens at the slot-extraction level) and resolved by
        # clinic-api's own alias ladder, so a Bengali-script provider name
        # must work here exactly as it does calling the endpoint directly.
        session = _dispatch(live, monkeypatch, "insurance_coverage",
                             _empty_slots(test_name="Complete Blood Count (CBC)"), tmp_path)
        live.spoken.clear()

        handled = _continue(live, session, "স্টার হেলথ")

        assert handled is True
        assert "Star Health and Allied Insurance" in live.spoken[0]

    def test_unrecognized_provider_after_continuation_is_honest_not_a_guess(self, live, monkeypatch, tmp_path):
        session = _dispatch(live, monkeypatch, "insurance_coverage",
                             _empty_slots(test_name="Complete Blood Count (CBC)"), tmp_path)
        live.spoken.clear()

        handled = _continue(live, session, "Acko General Insurance")

        assert handled is True
        assert "Acko General Insurance" in live.spoken[0]
        assert "covered" not in live.spoken[0].lower()


class TestBillingBalanceEndToEndMultiTurn:
    """Exercises the real missing-phone follow-up against real seeded
    PatientBilling rows -- including the digit-fidelity guarantees
    (whole number vs. fractional vs. real zero) end-to-end through the
    actual HTTP/JSON round-trip, not a synthetic string literal."""

    def test_missing_phone_then_continuation_speaks_real_fractional_balance(self, live, monkeypatch, tmp_path):
        session = _dispatch(live, monkeypatch, "billing_balance", _empty_slots(), tmp_path)
        assert session.pending == {"awaiting": "billing_phone", "retries": 0}
        live.spoken.clear()

        handled = _continue(live, session, "9000000002")

        assert handled is True
        assert session.pending is None
        text = live.spoken[0]
        assert "1250.75" in text
        assert "2026-10-15" in text

    def test_missing_phone_then_continuation_speaks_whole_number_without_trailing_point_zero(self, live, monkeypatch, tmp_path):
        session = _dispatch(live, monkeypatch, "billing_balance", _empty_slots(), tmp_path)
        live.spoken.clear()

        _continue(live, session, "9000000003")

        text = live.spoken[0]
        assert "500" in text
        assert "500.0" not in text

    def test_missing_phone_then_continuation_speaks_real_zero_balance_as_a_fact(self, live, monkeypatch, tmp_path):
        session = _dispatch(live, monkeypatch, "billing_balance", _empty_slots(), tmp_path)
        live.spoken.clear()

        _continue(live, session, "9000000001")

        assert "0 rupees" in live.spoken[0]

    def test_missing_phone_then_continuation_for_unbilled_patient_is_honest(self, live, monkeypatch, tmp_path):
        # Priyanka Sengupta (Patient G) has no PATIENT_BILLING row.
        session = _dispatch(live, monkeypatch, "billing_balance", _empty_slots(), tmp_path)
        live.spoken.clear()

        _continue(live, session, "9000000008")

        text = live.spoken[0]
        assert "0" not in text  # never a guessed/defaulted zero balance


class TestTransportParityEndToEnd:
    """Confirms the full stack -- not just dispatch structure -- produces
    byte-identical spoken replies whichever transport module handles the
    call, against the real seeded database."""

    def test_walkin_reply_identical_across_transports(self, real_clinic_api, live_tools_client, monkeypatch, tmp_path):
        results = {}
        for name, module in (("main", main), ("main_pcm", main_pcm)):
            spoken = []

            async def fake_speak(session, text, fallback_reason=None, _spoken=spoken):
                _spoken.append(text)

            async def fake_resolve_intent(session, text):
                return {"intent": "walkin_eligibility",
                        "slots": _empty_slots(test_name="Complete Blood Count (CBC)")}

            monkeypatch.setattr(module, "_speak", fake_speak)
            monkeypatch.setattr(module, "_asr", FakeASR())
            monkeypatch.setattr(module, "_tools", live_tools_client)
            monkeypatch.setattr(module, "_resolve_intent", fake_resolve_intent)

            session = make_session()
            wav_path = tmp_path / f"utt_{name}.wav"
            wav_path.write_bytes(b"")
            run(module._dispatch_turn(session, str(wav_path)))
            results[name] = spoken[0]

        assert results["main"] == results["main_pcm"]

    def test_billing_reply_identical_across_transports(self, real_clinic_api, live_tools_client, monkeypatch, tmp_path):
        results = {}
        for name, module in (("main", main), ("main_pcm", main_pcm)):
            spoken = []

            async def fake_speak(session, text, fallback_reason=None, _spoken=spoken):
                _spoken.append(text)

            async def fake_resolve_intent(session, text):
                return {"intent": "billing_balance", "slots": _empty_slots()}

            monkeypatch.setattr(module, "_speak", fake_speak)
            monkeypatch.setattr(module, "_asr", FakeASR())
            monkeypatch.setattr(module, "_tools", live_tools_client)
            monkeypatch.setattr(module, "_resolve_intent", fake_resolve_intent)

            session = make_session()
            wav_path = tmp_path / f"utt_{name}.wav"
            wav_path.write_bytes(b"")
            run(module._dispatch_turn(session, str(wav_path)))
            run(module._continue_pending(session, "9000000002"))
            results[name] = spoken[-1]

        assert results["main"] == results["main_pcm"]
