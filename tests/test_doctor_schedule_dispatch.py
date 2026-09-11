"""ADDED BY SOURAV -- dispatch-level and real-backend tests for "Caller
asks when a doctor sits" (Epic: Conversation -- Information and Enquiry).

Two halves, mirroring this codebase's established split (see
tests/test_report_status_send_dispatch.py's module docstring and
tests/test_live_test_price_lookup.py's):

  1. TestDispatchWithFakeTools -- drives main_pcm._dispatch_turn() with a
     stubbed ClinicToolsClient, to isolate and verify main.py/main_pcm.py's
     own dispatch-branch logic (the phone-name gate, which reply function
     gets called, what gets passed to the tool) without needing a real
     database for every case.

  2. TestAgainstRealClinicApi -- boots the REAL clinic-api FastAPI app on
     a real temporary SQLite database seeded with the REAL seed.py data,
     and drives the SAME main_pcm._dispatch_turn() end to end, proving the
     new /api/v1/doctors/schedule endpoint and the reply function agree
     with each other against genuine seeded rows, not just hand-typed
     fixtures.

WHY main_pcm.py AND NOT main.py: both are produced from the same source
(main.py is hand-edited; main_pcm.py is regenerated from it by
tools/make_pcm_variant.py -- see that script's own docstring). Testing
main_pcm.py exercises the identical logic in both by construction;
TestTransportParity below additionally asserts that identity directly.
"""
from __future__ import annotations

import asyncio
import os
import sys
import types

import pytest

import main_pcm
from agent.bn_normalize import weekday_to_words

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CLINIC_API_DIR = os.path.join(TESTS_DIR, "..", "clinic-api")


def run(coro):
    return asyncio.run(coro)


class FakeToolsClient:
    def __init__(self):
        self.get_doctor_schedule_calls = []
        self.get_doctor_schedule_response = None

    async def get_doctor_schedule(self, doctor_name):
        self.get_doctor_schedule_calls.append(doctor_name)
        return self.get_doctor_schedule_response


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session():
    return types.SimpleNamespace(
        call_id="test-doctor-schedule-call",
        pending=None,
        dispatch_lock=asyncio.Lock(),
        send_json=_AsyncNoOp(),
    )


class FakeASRResult:
    # UPDATED BY SOURAV -- must be real (Bengali) text, not the old English
    # placeholder ("ignored -- ..."), now that main_pcm.py's dispatch
    # actually calls detect_language() on it (see main.py's own "ADDED BY
    # SOURAV" comment on `language = detect_language(text)`, threaded
    # through by the language-detection production fix). This suite's own
    # assertions below compare against the Bengali-default reply text, so
    # the fake utterance is kept in Bengali to match -- language variation
    # itself is covered separately in tests/test_language_detection_dispatch.py.
    text = "ডাক্তারের সময়সূচী জানতে চাই"


class FakeASR:
    async def transcribe_utterance(self, wav_path):
        return FakeASRResult()


@pytest.fixture
def env(monkeypatch):
    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    tools = FakeToolsClient()
    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    monkeypatch.setattr(main_pcm, "_asr", FakeASR())
    monkeypatch.setattr(main_pcm, "_tools", tools)
    return types.SimpleNamespace(spoken=spoken, tools=tools)


def dispatch_with_intent(monkeypatch, intent, slots, tmp_path):
    async def fake_resolve_intent(session, text):
        return {"intent": intent, "slots": slots}

    monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
    session = make_session()
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return session


class TestDispatchWithFakeTools:
    def test_missing_doctor_name_asks_for_it_and_never_calls_the_tool(self, monkeypatch, env, tmp_path):
        session = dispatch_with_intent(
            monkeypatch, "doctor_schedule", {"doctor_name": None}, tmp_path,
        )
        assert env.tools.get_doctor_schedule_calls == []
        assert len(env.spoken) == 1
        # Known, documented limitation shared with doctor_availability
        # (see main.py's comment on this branch): no pending state is
        # opened, unlike report_status/report_send's phone gate.
        assert session.pending is None

    def test_a_date_slot_extracted_alongside_is_never_passed_to_the_tool(self, monkeypatch, env, tmp_path):
        # Even if the LLM extraction happened to also pull a "date" out of
        # the same utterance (e.g. an ambiguous phrasing), this intent must
        # stay date-free -- passing it through would silently turn this
        # back into the doctor_availability question it exists to be
        # distinct from.
        env.tools.get_doctor_schedule_response = {
            "found": True, "doctor_name": "Dr A. Sen", "doctor_name_bn": "ডক্টর সেন",
            "schedule": [{"weekday": 0, "start_time": "10:00", "end_time": "12:00"}],
        }
        dispatch_with_intent(
            monkeypatch, "doctor_schedule",
            {"doctor_name": "Dr A. Sen", "date": "2026-09-14"}, tmp_path,
        )
        assert env.tools.get_doctor_schedule_calls == ["Dr A. Sen"]

    def test_found_with_schedule_speaks_the_reply_and_opens_no_pending_state(
        self, monkeypatch, env, tmp_path
    ):
        env.tools.get_doctor_schedule_response = {
            "found": True, "doctor_name": "Dr A. Sen", "doctor_name_bn": "ডক্টর সেন",
            "schedule": [
                {"weekday": 0, "start_time": "10:00", "end_time": "12:00"},
                {"weekday": 2, "start_time": "10:00", "end_time": "12:00"},
            ],
        }
        session = dispatch_with_intent(
            monkeypatch, "doctor_schedule", {"doctor_name": "Dr A. Sen"}, tmp_path,
        )
        assert env.tools.get_doctor_schedule_calls == ["Dr A. Sen"]
        assert len(env.spoken) == 1
        assert weekday_to_words(0, "bengali") in env.spoken[0]
        # Purely informational -- unlike doctor_availability, never opens
        # a follow-up "book today or another day?" pending state.
        assert session.pending is None

    def test_doctor_not_found_speaks_the_not_found_reply(self, monkeypatch, env, tmp_path):
        env.tools.get_doctor_schedule_response = {"found": False, "query": "Dr Ghost"}
        session = dispatch_with_intent(
            monkeypatch, "doctor_schedule", {"doctor_name": "Dr Ghost"}, tmp_path,
        )
        assert len(env.spoken) == 1
        assert "Dr Ghost" in env.spoken[0]
        assert session.pending is None

    def test_tool_failure_gives_the_shared_infrastructure_apology(self, monkeypatch, env, tmp_path):
        from agent.tools_client import ToolCallError

        async def failing_get_doctor_schedule(doctor_name):
            raise ToolCallError("boom")

        env.tools.get_doctor_schedule = failing_get_doctor_schedule
        session = dispatch_with_intent(
            monkeypatch, "doctor_schedule", {"doctor_name": "Dr A. Sen"}, tmp_path,
        )
        assert len(env.spoken) == 1
        assert session.pending is None


@pytest.fixture(scope="module")
def real_clinic_api(tmp_path_factory):
    """Boots the REAL clinic-api FastAPI app against a throwaway SQLite
    file and seeds it with the REAL seed.py data. Duplicated fixture
    pattern (not shared across files) -- see
    tests/test_live_test_price_lookup.py's own fixture docstring for why
    this codebase keeps one self-contained fixture per test file."""
    db_path = tmp_path_factory.mktemp("clinic_doctor_schedule_live") / "clinic.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    sys.path.insert(0, CLINIC_API_DIR)
    for mod in ("db", "models", "seed", "main"):
        sys.modules.pop(mod, None)
    import db as clinic_db
    import models as clinic_models
    clinic_models.Base.metadata.create_all(clinic_db.engine)
    from seed import seed
    seed()

    import main as clinic_main
    from fastapi.testclient import TestClient
    client = TestClient(clinic_main.app)

    yield types.SimpleNamespace(client=client, db=clinic_db, models=clinic_models)

    sys.path.remove(CLINIC_API_DIR)


class RealClinicToolsClient:
    """Stands in for agent.tools_client.ClinicToolsClient, exactly like
    test_live_test_price_lookup.py's own RealClinicToolsClient -- calls
    the REAL clinic-api app in-process via the FastAPI TestClient, no
    stubbing of clinic-api's own logic at all."""

    def __init__(self, client):
        self._client = client
        self.raw_responses = []

    async def get_doctor_schedule(self, doctor_name: str) -> dict:
        r = self._client.get("/api/v1/doctors/schedule", params={"name": doctor_name})
        result = r.json()
        self.raw_responses.append(dict(result))
        return result


def _dispatch_doctor_schedule(monkeypatch, tools, doctor_name, tmp_path):
    async def fake_resolve_intent(session, text):
        return {"intent": "doctor_schedule", "slots": {"doctor_name": doctor_name}}

    spoken = []

    async def fake_speak(session, text, fallback_reason=None):
        spoken.append(text)

    monkeypatch.setattr(main_pcm, "_asr", FakeASR())
    monkeypatch.setattr(main_pcm, "_resolve_intent", fake_resolve_intent)
    monkeypatch.setattr(main_pcm, "_speak", fake_speak)
    monkeypatch.setattr(main_pcm, "_tools", tools)

    session = make_session()
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"")
    run(main_pcm._dispatch_turn(session, str(wav_path)))
    return spoken


class TestAgainstRealClinicApi:
    def test_a_real_seeded_doctor_speaks_their_real_sitting_days(
        self, monkeypatch, real_clinic_api, tmp_path
    ):
        db = real_clinic_api.db.SessionLocal()
        try:
            doctor = db.query(real_clinic_api.models.Doctor).first()
            assert doctor is not None, "seed.py must still seed at least one doctor -- fixture assumption"
            rows = (
                db.query(real_clinic_api.models.DoctorSchedule)
                .filter_by(doctor_id=doctor.id)
                .order_by(real_clinic_api.models.DoctorSchedule.weekday.asc())
                .all()
            )
            assert rows, "seed.py must still give every doctor a schedule -- fixture assumption"
            expected_weekdays = [r.weekday for r in rows]
        finally:
            db.close()

        tools = RealClinicToolsClient(real_clinic_api.client)
        spoken = _dispatch_doctor_schedule(monkeypatch, tools, doctor.name, tmp_path)

        assert len(spoken) == 1
        assert tools.raw_responses[0]["found"] is True
        for w in expected_weekdays:
            assert weekday_to_words(w, "bengali") in spoken[0]

    def test_real_bengali_alias_resolves_the_same_doctor(
        self, monkeypatch, real_clinic_api, tmp_path
    ):
        db = real_clinic_api.db.SessionLocal()
        try:
            doctor = db.query(real_clinic_api.models.Doctor).filter(
                real_clinic_api.models.Doctor.aliases_bn != ""
            ).first()
            alias = [a for a in doctor.aliases_bn.split("|") if a][0]
        finally:
            db.close()

        tools = RealClinicToolsClient(real_clinic_api.client)
        _dispatch_doctor_schedule(monkeypatch, tools, alias, tmp_path)

        assert tools.raw_responses[0]["found"] is True
        assert tools.raw_responses[0]["doctor_name"] == doctor.name

    def test_real_misspelled_or_unknown_doctor_gets_the_not_found_path(
        self, monkeypatch, real_clinic_api, tmp_path
    ):
        tools = RealClinicToolsClient(real_clinic_api.client)
        spoken = _dispatch_doctor_schedule(monkeypatch, tools, "Dr Zzznonexistent", tmp_path)

        assert tools.raw_responses[0]["found"] is False
        assert "Dr Zzznonexistent" in spoken[0]

    def test_real_endpoint_response_shape_matches_what_the_reply_function_expects(
        self, real_clinic_api,
    ):
        # Direct endpoint check (no dispatch): every schedule row's keys
        # match exactly what agent/reply_templates.py::doctor_schedule_reply()
        # and _group_schedule_by_hours() read -- "weekday", "start_time",
        # "end_time" -- so a clinic-api response shape drift would be
        # caught here even before it ever reached a live dispatch turn.
        db = real_clinic_api.db.SessionLocal()
        try:
            doctor = db.query(real_clinic_api.models.Doctor).first()
        finally:
            db.close()

        r = real_clinic_api.client.get("/api/v1/doctors/schedule", params={"name": doctor.name})
        body = r.json()
        assert body["found"] is True
        assert "schedule" in body
        for entry in body["schedule"]:
            assert set(entry.keys()) == {"weekday", "start_time", "end_time"}
            assert 0 <= entry["weekday"] <= 6


class TestTransportParity:
    def test_reasoning_half_is_byte_identical_between_main_and_main_pcm(self):
        a = open("main.py", encoding="utf-8").read()
        b = open("main_pcm.py", encoding="utf-8").read()
        sa = a[a.index("async def _resolve_intent("):a.index("async def _resync_after_playback(")]
        sb = b[b.index("async def _resolve_intent("):b.index("async def _resync_after_playback(")]
        assert sa == sb

    def test_main_dot_py_has_the_doctor_schedule_branch(self):
        src = open("main.py", encoding="utf-8").read()
        assert 'intent == "doctor_schedule"' in src
        assert "get_doctor_schedule" in src
        assert "doctor_schedule_reply" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
