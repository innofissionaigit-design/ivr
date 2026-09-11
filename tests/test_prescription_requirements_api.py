"""ADDED BY SOURAV -- tests for the new GET /api/v1/tests/prescription-policy
endpoint backing "Prescription Requirements" (Phase 1: Database Schema &
Policy Tables). Same "real backend, real seed data, real HTTP round-trip"
pattern as tests/test_walkin_eligibility_api.py (fixture duplicated, not
imported, per this codebase's convention).

UPDATED BY SOURAV -- Phase 2: clinic-api/seed.py's PRESCRIPTION_POLICY now
gives SOME real seeded tests a reviewed prescription requirement (sample/
demo values, not verified business content -- see that dict's own
comment). See tests/test_walkin_eligibility_api.py's own docstring for the
identical reasoning applied here: every test listed in PRESCRIPTION_POLICY
is checked against that dict directly (imported via the fixture, never
retyped), every test left out still honestly reports
`policy_available: False`, and the synthetic-fixture class targets tests
deliberately left OUT of PRESCRIPTION_POLICY.
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
    db_path = tmp_path_factory.mktemp("clinic_test_prescription_live") / "clinic.db"
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


def prescription(client, name):
    return client.get("/api/v1/tests/prescription-policy", params={"name": name}).json()


class TestUnknownTestIsHonestlyNotFound:
    def test_unknown_test_name_returns_found_false_with_suggestions(self, real_clinic_api):
        result = prescription(real_clinic_api.client, "Complete Blod Count")
        assert result["found"] is False
        assert "did_you_mean" in result


class TestReviewedTestsSpeakRealPolicy:
    """ADDED BY SOURAV -- Phase 2: every test listed in seed.py's
    PRESCRIPTION_POLICY must speak its real reviewed value. Asserted
    against PRESCRIPTION_POLICY itself (imported via the fixture, not
    retyped here), so this test can never silently drift from what is
    actually seeded."""

    def test_every_policy_seeded_test_speaks_its_reviewed_value(self, real_clinic_api):
        prescription_policy = real_clinic_api.seed.PRESCRIPTION_POLICY
        assert len(prescription_policy) > 0

        for test_name, expected in prescription_policy.items():
            result = prescription(real_clinic_api.client, test_name)
            expected_channels = [c for c in (expected["prescription_channels"] or "").split("|") if c]
            assert result["found"] is True, test_name
            assert result["policy_available"] is True, test_name
            assert result["prescription_required"] == expected["prescription_required"], test_name
            assert result["prescription_channels"] == expected_channels, test_name


class TestUnreviewedTestsStayHonestlyUnavailable:
    """Every seeded test NOT listed in PRESCRIPTION_POLICY must still say
    `policy_available: False` -- confirms Phase 2's sample content did not
    leak a default onto every row."""

    def test_every_non_policy_test_reports_policy_unavailable(self, real_clinic_api):
        db = real_clinic_api.db.SessionLocal()
        try:
            all_tests = db.query(real_clinic_api.models.LabTest).all()
        finally:
            db.close()

        reviewed_names = set(real_clinic_api.seed.PRESCRIPTION_POLICY)
        unreviewed = [t for t in all_tests if t.name not in reviewed_names]
        assert len(unreviewed) > 0

        for t in unreviewed:
            result = prescription(real_clinic_api.client, t.name)
            assert result["found"] is True, t.name
            assert result["policy_available"] is False, t.name
            assert "prescription_required" not in result
            assert "prescription_channels" not in result


class TestSyntheticFixtureFormatsCorrectly:
    """Inserts fabricated (never real) prescription policy rows directly,
    on tests deliberately left OUT of seed.py's PRESCRIPTION_POLICY
    ("HbA1c", "Vitamin B12", "Widal Test"), to verify the response
    formatting shapes independently of Phase 2's own sample content
    above."""

    def test_required_with_channels(self, real_clinic_api):
        db = real_clinic_api.db.SessionLocal()
        try:
            assert "HbA1c" not in real_clinic_api.seed.PRESCRIPTION_POLICY
            t = db.query(real_clinic_api.models.LabTest).filter_by(name="HbA1c").first()
            t.prescription_required = True
            t.prescription_channels = "whatsapp_photo|counter_in_person"
            db.commit()
            name = t.name
        finally:
            db.close()

        result = prescription(real_clinic_api.client, name)
        assert result["found"] is True
        assert result["policy_available"] is True
        assert result["prescription_required"] is True
        assert result["prescription_channels"] == ["whatsapp_photo", "counter_in_person"]

    def test_not_required_has_empty_channel_list_not_none(self, real_clinic_api):
        db = real_clinic_api.db.SessionLocal()
        try:
            assert "Vitamin B12" not in real_clinic_api.seed.PRESCRIPTION_POLICY
            t = db.query(real_clinic_api.models.LabTest).filter_by(name="Vitamin B12").first()
            t.prescription_required = False
            t.prescription_channels = None
            db.commit()
        finally:
            db.close()

        result = prescription(real_clinic_api.client, "Vitamin B12")
        assert result["prescription_required"] is False
        # A None DB column must never crash the split() -- it degrades to
        # an empty list, never null and never a fabricated channel.
        assert result["prescription_channels"] == []

    def test_pipe_delimited_split_ignores_empty_segments(self, real_clinic_api):
        db = real_clinic_api.db.SessionLocal()
        try:
            assert "Widal Test" not in real_clinic_api.seed.PRESCRIPTION_POLICY
            t = db.query(real_clinic_api.models.LabTest).filter_by(name="Widal Test").first()
            t.prescription_required = True
            t.prescription_channels = "email||counter_in_person|"
            db.commit()
        finally:
            db.close()

        result = prescription(real_clinic_api.client, "Widal Test")
        assert result["prescription_channels"] == ["email", "counter_in_person"]
