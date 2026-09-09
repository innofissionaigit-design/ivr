"""The GENUINE LIVE DEMONSTRATION the user asked for: "The agent says it
cannot confirm rather than guessing", exercised against the REAL
clinic-api FastAPI app, backed by a REAL SQLite database, seeded with the
REAL clinic-api/seed.py data -- not a hand-written stub of the backend.

What is real here: the FastAPI app object from clinic-api/main.py, the
SQLAlchemy models, a real (temporary) SQLite file, seed.py's actual 8
departments / 32 doctors / 34 tests / weekly schedules, and a real
POST /api/v1/appointments call that really inserts a row and returns a
real, uuid4-derived confirmation_id.

What is deliberately synthetic, and clearly marked as such: a healthy
clinic-api will never itself return a booking response with an empty
confirmation_id/date/time_slot (see clinic-api/main.py's book_appointment
-- every success branch fills all three). So to actually WATCH the guard
in agent/outcomes.py fire against something more than a hand-typed dict,
this file takes one genuinely real, freshly-written booking response and
removes exactly one field from it before handing it to main_pcm.
_finish_booking() -- simulating the kind of thing that would actually
justify this story existing (a proxy, a serialization layer, or a future
schema change silently dropping a field between a working backend and the
agent). Every value that DOES reach the agent in this test came from the
real database.

Run directly for a human-readable walkthrough:
    python3 -m pytest tests/test_live_insufficient_information_demo.py -q -s
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
import sys
import types

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CLINIC_API_DIR = os.path.join(TESTS_DIR, "..", "clinic-api")


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def real_clinic_api(tmp_path_factory):
    """Boots the REAL clinic-api FastAPI app against a throwaway SQLite
    file and seeds it with the REAL seed.py data. Module-scoped: one
    seeded database serves every test in this file, the same way one
    `deploy/start_all.sh` run would serve a whole test session on a real
    machine."""
    db_path = tmp_path_factory.mktemp("clinic_live") / "clinic.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    sys.path.insert(0, CLINIC_API_DIR)
    # Fresh imports every test session -- clinic-api's db.py reads
    # DATABASE_URL at import time, so a prior test run's module-level
    # engine must not be reused.
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


_slot_week_counter = iter(range(1, 1000))


@pytest.fixture
def real_seeded_booking_slot(real_clinic_api):
    """Picks a REAL doctor and a REAL open slot from the seeded schedule
    -- not hand-typed data. Returns (doctor_name, date_iso, time_slot).

    Each call lands on the doctor's usual weekday but a different week
    (module-scoped counter), so the several tests in this file -- which
    all share ONE seeded database, same as one real deployment would --
    never collide on the doctor_id/date/time_slot unique constraint from
    a previous test's real booking."""
    db = real_clinic_api.db.SessionLocal()
    try:
        Doctor = real_clinic_api.models.Doctor
        DoctorSchedule = real_clinic_api.models.DoctorSchedule
        doctor = db.query(Doctor).first()
        sched = db.query(DoctorSchedule).filter_by(doctor_id=doctor.id).first()

        weeks_ahead = next(_slot_week_counter)
        today = datetime.date.today()
        days_ahead = (sched.weekday - today.weekday()) % 7
        days_ahead = days_ahead or 7  # today's own weekday -> next occurrence, avoid past-time edge cases
        target_date = today + datetime.timedelta(days=days_ahead, weeks=weeks_ahead - 1)

        start_h, start_m = (int(x) for x in sched.start_time.split(":"))
        slot_time = datetime.time(start_h, start_m)
        return doctor.name, target_date.isoformat(), slot_time.strftime("%H:%M")
    finally:
        db.close()


class RealBackendFieldDroppingClient:
    """Stands in for agent.tools_client.ClinicToolsClient. Its
    book_appointment() calls the REAL clinic-api app (via the FastAPI
    TestClient, in-process HTTP) exactly like the real
    ClinicToolsClient.book_appointment() does over the network, gets back
    a REAL response with a REAL confirmation_id that really exists in the
    REAL database now -- and then, ONLY for this demonstration, deletes
    `drop_field` from the dict before returning it, to simulate a field
    lost between a working backend and the agent."""

    def __init__(self, client, drop_field: str | None = None):
        self._client = client
        self._drop_field = drop_field
        self.raw_backend_responses = []

    async def book_appointment(self, doctor_name, date, time_slot, patient_name, phone):
        r = self._client.post("/api/v1/appointments", json={
            "doctor_name": doctor_name, "date": date, "time_slot": time_slot,
            "patient_name": patient_name, "phone": phone,
        })
        result = r.json()
        self.raw_backend_responses.append(dict(result))
        if self._drop_field and self._drop_field in result:
            result[self._drop_field] = ""
        return result


class _AsyncNoOp:
    async def __call__(self, *args, **kwargs):
        return None


def make_session():
    return types.SimpleNamespace(
        call_id="live-demo-call",
        pending=None,
        dispatch_lock=asyncio.Lock(),
        send_json=_AsyncNoOp(),
    )


@pytest.fixture(autouse=True)
def isolated_outcomes_state(monkeypatch, tmp_path):
    import agent.outcomes as outcomes_module
    from agent.outcomes import _reset_for_testing
    monkeypatch.setattr(outcomes_module, "ESCALATION_LOG_PATH", str(tmp_path / "escalations.jsonl"))
    _reset_for_testing()
    yield tmp_path / "escalations.jsonl"
    _reset_for_testing()


class TestLiveAgainstRealClinicApi:
    """Real backend, real seed data, real HTTP round-trip through the
    real FastAPI app."""

    def test_a_healthy_real_booking_is_read_back_normally(
        self, monkeypatch, real_clinic_api, real_seeded_booking_slot, capsys,
    ):
        """The regression check: against a REAL, healthy backend response,
        this story changes NOTHING -- the caller hears their real
        confirmation number, same as before this story existed."""
        import main_pcm
        doctor_name, date, time_slot = real_seeded_booking_slot

        tools = RealBackendFieldDroppingClient(real_clinic_api.client, drop_field=None)
        monkeypatch.setattr(main_pcm, "_tools", tools)
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append((text, fallback_reason))

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)

        slots = {"doctor_name": doctor_name, "date": date, "time_slot": time_slot,
                  "patient_name": "Live Demo Patient", "phone": "9800000001"}
        session = make_session()

        run(main_pcm._finish_booking(session, slots))

        real_response = tools.raw_backend_responses[0]
        assert real_response["success"] is True
        print(f"\n[LIVE] real clinic-api response: {real_response}")
        print(f"[LIVE] spoken to caller: {spoken[0][0]!r}")

        assert real_response["confirmation_id"] in spoken[0][0]
        assert spoken[0][1] is None

        from agent.outcomes import insufficient_verified_information_counts
        assert insufficient_verified_information_counts("book_appointment") == 0

    @pytest.mark.parametrize("drop_field", ["confirmation_id", "date", "time_slot"])
    def test_a_field_dropped_after_a_real_write_is_caught_live(
        self, monkeypatch, real_clinic_api, real_seeded_booking_slot, isolated_outcomes_state,
        drop_field, capsys,
    ):
        """THE DEMONSTRATION: a real booking really lands in the real
        database (the row exists, the confirmation_id was really
        generated by clinic-api's uuid4 call) -- and then one field is
        dropped in transit, simulating exactly the kind of integration bug
        this story exists to catch. Watch the agent refuse to guess."""
        import main_pcm
        from agent.outcomes import insufficient_verified_information_reply, insufficient_verified_information_counts

        doctor_name, date, time_slot = real_seeded_booking_slot
        tools = RealBackendFieldDroppingClient(real_clinic_api.client, drop_field=drop_field)
        monkeypatch.setattr(main_pcm, "_tools", tools)
        spoken = []

        async def fake_speak(session, text, fallback_reason=None):
            spoken.append((text, fallback_reason))

        monkeypatch.setattr(main_pcm, "_speak", fake_speak)

        slots = {"doctor_name": doctor_name, "date": date, "time_slot": time_slot,
                  "patient_name": "Live Demo Patient", "phone": "9800000002"}
        session = make_session()

        run(main_pcm._finish_booking(session, slots))

        real_response = tools.raw_backend_responses[0]
        print(f"\n[LIVE] real clinic-api wrote: {real_response} "
              f"(this row now really exists in the seeded database)")
        print(f"[LIVE] '{drop_field}' dropped in transit -- agent spoke: {spoken[0][0]!r}")

        # The real confirmation_id genuinely exists in the DB now, but the
        # caller must NOT hear it read back with false confidence.
        assert real_response["success"] is True
        assert spoken[0][0] == insufficient_verified_information_reply()
        assert spoken[0][1] == "insufficient_verified_information"
        assert insufficient_verified_information_counts("book_appointment") == 1

        escalation = json.loads(isolated_outcomes_state.read_text(encoding="utf-8").strip())
        print(f"[LIVE] escalation ledger entry: {escalation}")
        assert escalation["field"] == drop_field
        assert escalation["call_id"] == "live-demo-call"

        # And the row this test dropped a field FROM really is in the real
        # database -- proving this was a live write, not a mock.
        db = real_clinic_api.db.SessionLocal()
        try:
            Appointment = real_clinic_api.models.Appointment
            row = db.query(Appointment).filter_by(
                confirmation_id=real_response["confirmation_id"],
            ).first()
            assert row is not None
            print(f"[LIVE] confirmed in the real DB: appointment id={row.id}, "
                  f"confirmation_id={row.confirmation_id}")
        finally:
            db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
