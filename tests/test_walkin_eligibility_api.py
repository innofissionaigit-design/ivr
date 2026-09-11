"""ADDED BY SOURAV -- tests for the new GET /api/v1/tests/walkin-policy
endpoint backing "Walk-in Eligibility" (Phase 1: Database Schema & Policy
Tables). Against the REAL clinic-api FastAPI app, a REAL SQLite database
-- same "real backend, real seed data, real HTTP round-trip" pattern as
tests/test_clinic_api_health_and_info.py / tests/test_test_preparation_api.py
(the `real_clinic_api` fixture below is duplicated from those files, not
imported, per this codebase's own established one-fixture-per-file
convention).

UPDATED BY SOURAV -- Phase 2: clinic-api/seed.py's WALKIN_POLICY now gives
SOME real seeded tests a reviewed walk-in policy (sample/demo values, not
verified business content -- see that dict's own comment). This file now
asserts two things against the real seeded catalogue: every test listed in
WALKIN_POLICY speaks its real reviewed value (checked against WALKIN_POLICY
itself, imported from seed.py, so this test can never silently drift from
what is actually seeded), and every test NOT listed still honestly reports
`policy_available: False`, never a guessed default. The synthetic-fixture
class below now targets a test deliberately left OUT of WALKIN_POLICY, so
it exercises the response shape independently of Phase 2's own sample data.
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
    db_path = tmp_path_factory.mktemp("clinic_test_walkin_live") / "clinic.db"
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
    from fastapi.testclient import TestClient
    client = TestClient(clinic_main.app)

    yield types.SimpleNamespace(client=client, db=clinic_db, models=clinic_models, seed=clinic_seed)

    sys.path.remove(CLINIC_API_DIR)


def walkin(client, name):
    return client.get("/api/v1/tests/walkin-policy", params={"name": name}).json()


class TestUnknownTestIsHonestlyNotFound:
    def test_unknown_test_name_returns_found_false_with_suggestions(self, real_clinic_api):
        result = walkin(real_clinic_api.client, "Complete Blod Count")
        assert result["found"] is False
        assert result["query"] == "Complete Blod Count"
        assert "did_you_mean" in result


class TestReviewedTestsSpeakRealPolicy:
    """ADDED BY SOURAV -- Phase 2: every test listed in seed.py's
    WALKIN_POLICY must speak its real reviewed value. Asserted against
    WALKIN_POLICY itself (imported via the fixture, not retyped here), so
    this test can never silently drift from what is actually seeded."""

    def test_every_policy_seeded_test_speaks_its_reviewed_value(self, real_clinic_api):
        walkin_policy = real_clinic_api.seed.WALKIN_POLICY
        assert len(walkin_policy) > 0

        for test_name, expected in walkin_policy.items():
            result = walkin(real_clinic_api.client, test_name)
            assert result["found"] is True, test_name
            assert result["policy_available"] is True, test_name
            assert result["walkin_eligible"] == expected["walkin_eligible"], test_name
            assert result["walkin_hours"] == expected["walkin_hours"], test_name


class TestUnreviewedTestsStayHonestlyUnavailable:
    """Every seeded test NOT listed in WALKIN_POLICY must still say
    `policy_available: False` -- confirms Phase 2's sample content did not
    leak a default onto every row (see models.py's own comment on why
    walkin_eligible must never default)."""

    def test_every_non_policy_test_reports_policy_unavailable(self, real_clinic_api):
        db = real_clinic_api.db.SessionLocal()
        try:
            all_tests = db.query(real_clinic_api.models.LabTest).all()
        finally:
            db.close()

        reviewed_names = set(real_clinic_api.seed.WALKIN_POLICY)
        unreviewed = [t for t in all_tests if t.name not in reviewed_names]
        assert len(unreviewed) > 0

        for t in unreviewed:
            result = walkin(real_clinic_api.client, t.name)
            assert result["found"] is True, t.name
            assert result["policy_available"] is False, t.name
            assert "walkin_eligible" not in result
            assert "walkin_hours" not in result


class TestSyntheticFixtureFormatsCorrectly:
    """Inserts a fabricated (never real) walk-in policy row directly, on a
    test deliberately left OUT of seed.py's WALKIN_POLICY ("ECG",
    "Widal Test"), to verify the `policy_available: True` response shape
    and the honest "not eligible" branch independently of Phase 2's own
    sample content above."""

    def test_eligible_test_with_hours(self, real_clinic_api):
        db = real_clinic_api.db.SessionLocal()
        try:
            assert "ECG" not in real_clinic_api.seed.WALKIN_POLICY
            t = db.query(real_clinic_api.models.LabTest).filter_by(name="ECG").first()
            t.walkin_eligible = True
            t.walkin_hours = "Mon-Sat 7am-11am, no walk-ins Sunday"
            db.commit()
        finally:
            db.close()

        result = walkin(real_clinic_api.client, "ECG")
        assert result == {
            "found": True, "test_name": "ECG",
            "test_name_bn": result["test_name_bn"],
            "policy_available": True,
            "walkin_eligible": True,
            "walkin_hours": "Mon-Sat 7am-11am, no walk-ins Sunday",
        }

    def test_not_eligible_test(self, real_clinic_api):
        db = real_clinic_api.db.SessionLocal()
        try:
            assert "Widal Test" not in real_clinic_api.seed.WALKIN_POLICY
            t = db.query(real_clinic_api.models.LabTest).filter_by(name="Widal Test").first()
            t.walkin_eligible = False
            t.walkin_hours = None
            db.commit()
        finally:
            db.close()

        result = walkin(real_clinic_api.client, "Widal Test")
        assert result["policy_available"] is True
        assert result["walkin_eligible"] is False
        assert result["walkin_hours"] is None
