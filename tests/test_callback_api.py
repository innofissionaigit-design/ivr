"""ADDED BY SOURAV -- tests for the new clinic-api endpoint backing
"Caller asks to be called back" (Evidence: "No outbound capability").

Against the REAL clinic-api FastAPI app and a REAL SQLite database --
mirrors tests/test_clinic_api_health_and_info.py's own established "real
backend, real HTTP round-trip" pattern, including its one-fixture-per-file
convention (the `real_clinic_api` fixture below is duplicated from that
file, not imported).

Endpoint under test: POST /api/v1/callbacks

Acceptance Criterion 2 ("the callback record is tracked to fulfilment --
stored in a DB table/queue with pending status") is exercised directly
against the database in TestCallbackPersistence below, not just against
the HTTP response -- a response that merely echoes what it was given back
would pass every assertion in TestCallbackEndpoint on its own even if
nothing had actually been written to the CallbackRequest table.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CLINIC_API_DIR = os.path.join(TESTS_DIR, "..", "clinic-api")


@pytest.fixture(scope="module")
def real_clinic_api(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("clinic_callback_live") / "clinic.db"
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


def request_callback(client, phone, time_window, reason=None):
    body = {"phone": phone, "time_window": time_window, "reason": reason}
    return client.post("/api/v1/callbacks", json=body).json()


# --------------------------------------------------------------------- #
# POST /api/v1/callbacks -- response shape
# --------------------------------------------------------------------- #

class TestCallbackEndpoint:
    def test_success_response_echoes_every_field_given(self, real_clinic_api):
        r = request_callback(real_clinic_api.client, "9831012345", "this evening", "about my CBC report")
        assert r["success"] is True
        assert r["phone"] == "9831012345"
        assert r["time_window"] == "this evening"
        assert r["reason"] == "about my CBC report"
        assert r["status"] == "pending"

    def test_reason_is_genuinely_optional_and_comes_back_null_not_a_placeholder(self, real_clinic_api):
        r = request_callback(real_clinic_api.client, "9831099999", "tomorrow morning")
        assert r["success"] is True
        assert r["reason"] is None

    def test_callback_id_is_present_and_unique_across_two_requests(self, real_clinic_api):
        r1 = request_callback(real_clinic_api.client, "9831011111", "morning")
        r2 = request_callback(real_clinic_api.client, "9831022222", "morning")
        assert r1["callback_id"]
        assert r2["callback_id"]
        assert r1["callback_id"] != r2["callback_id"]

    def test_missing_required_field_is_a_422_not_a_silent_success(self, real_clinic_api):
        resp = real_clinic_api.client.post("/api/v1/callbacks", json={"phone": "9831000000"})
        assert resp.status_code == 422


# --------------------------------------------------------------------- #
# Acceptance Criterion 2: tracked to fulfilment -- a real, persisted row
# --------------------------------------------------------------------- #

class TestCallbackPersistence:
    def test_a_row_is_actually_written_to_the_database_not_just_echoed_in_the_response(self, real_clinic_api):
        r = request_callback(real_clinic_api.client, "9830011122", "afternoon", "billing question")

        db = real_clinic_api.db.SessionLocal()
        try:
            row = (
                db.query(real_clinic_api.models.CallbackRequest)
                .filter_by(callback_id=r["callback_id"])
                .first()
            )
            assert row is not None
            assert row.phone == "9830011122"
            assert row.time_window == "afternoon"
            assert row.reason == "billing question"
            assert row.status == "pending"
            assert row.created_at is not None
        finally:
            db.close()

    def test_every_new_callback_request_starts_pending_never_pre_fulfilled(self, real_clinic_api):
        r = request_callback(real_clinic_api.client, "9830033344", "evening")
        db = real_clinic_api.db.SessionLocal()
        try:
            row = (
                db.query(real_clinic_api.models.CallbackRequest)
                .filter_by(callback_id=r["callback_id"])
                .first()
            )
            assert row.status == "pending"
        finally:
            db.close()

    def test_two_requests_from_the_same_phone_number_are_tracked_as_two_separate_rows(self, real_clinic_api):
        r1 = request_callback(real_clinic_api.client, "9830055566", "morning", "first call")
        r2 = request_callback(real_clinic_api.client, "9830055566", "evening", "second call")
        assert r1["callback_id"] != r2["callback_id"]

        db = real_clinic_api.db.SessionLocal()
        try:
            rows = (
                db.query(real_clinic_api.models.CallbackRequest)
                .filter_by(phone="9830055566")
                .all()
            )
            reasons = {row.reason for row in rows}
            assert "first call" in reasons
            assert "second call" in reasons
        finally:
            db.close()
