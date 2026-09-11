"""ADDED BY SOURAV -- tests for the new GET /api/v1/patient/billing
endpoint backing "Outstanding Balance / Billing" (Phase 1: Database Schema
& Policy Tables). Same "real backend, real seed data, real HTTP
round-trip" pattern as tests/test_walkin_eligibility_api.py.

UPDATED BY SOURAV -- Phase 2: clinic-api/seed.py's PATIENT_BILLING now
gives 3 real seeded patients (by phone) a reviewed billing row -- sample/
demo values, not verified business content, see that dict's own comment.
This file now asserts the billed patients speak their real reviewed
balance (checked against PATIENT_BILLING itself, imported via the
fixture, never retyped), and every patient NOT in PATIENT_BILLING still
honestly reports `found: False` (never a guessed/defaulted zero balance).
The synthetic-fixture class below picks an unbilled patient dynamically
(same defensive pattern its own zero-balance/due-date tests already
used) rather than assuming the first patient row is free -- Patient A
(Arjun Sen) now has a real PATIENT_BILLING row, and PatientBilling.
patient_id is unique, so inserting a second row for an already-billed
patient would violate that constraint. Identity is resolved by PHONE
(RULE 14/15), same as tests/test_clinic_api_reports.py's own
report_status coverage -- reuses the SAME real patient rows, not a
separate fixture set.
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
    db_path = tmp_path_factory.mktemp("clinic_test_billing_live") / "clinic.db"
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


def billing(client, phone):
    return client.get("/api/v1/patient/billing", params={"phone": phone}).json()


class TestUnknownPhoneIsPatientNotFound:
    def test_unregistered_phone(self, real_clinic_api):
        result = billing(real_clinic_api.client, "9999999999")
        assert result == {"patient_found": False}


class TestBilledPatientsSpeakRealBalance:
    """ADDED BY SOURAV -- Phase 2: every phone listed in seed.py's
    PATIENT_BILLING must speak its real reviewed balance. Asserted
    against PATIENT_BILLING itself (imported via the fixture, not
    retyped here), so this test can never silently drift from what is
    actually seeded."""

    def test_every_billed_phone_speaks_its_reviewed_balance(self, real_clinic_api):
        patient_billing = real_clinic_api.seed.PATIENT_BILLING
        assert len(patient_billing) > 0

        for phone, expected in patient_billing.items():
            result = billing(real_clinic_api.client, phone)
            expected_due = expected["due_date"].isoformat() if expected["due_date"] else None
            assert result["patient_found"] is True, phone
            assert result["found"] is True, phone
            assert result["outstanding_amount"] == expected["outstanding_amount"], phone
            assert result["due_date"] == expected_due, phone


class TestUnbilledPatientsStayHonestlyUnbilled:
    """Every seeded patient whose phone is NOT in PATIENT_BILLING must
    still say `found: False, reason: NOT_FOUND` -- confirms Phase 2's
    sample content did not leak a default onto every patient."""

    def test_every_unbilled_patient_reports_no_billing_record(self, real_clinic_api):
        db = real_clinic_api.db.SessionLocal()
        try:
            all_patients = db.query(real_clinic_api.models.Patient).all()
        finally:
            db.close()

        billed_phones = set(real_clinic_api.seed.PATIENT_BILLING)
        unbilled = [p for p in all_patients if p.phone not in billed_phones]
        assert len(unbilled) > 0

        for p in unbilled:
            result = billing(real_clinic_api.client, p.phone)
            assert result == {"patient_found": True, "found": False, "reason": "NOT_FOUND"}, p.phone


class TestSyntheticFixtureFormatsCorrectly:
    def test_real_patient_with_billing_row(self, real_clinic_api):
        db = real_clinic_api.db.SessionLocal()
        try:
            # Pick a patient with no existing PatientBilling row -- Phase
            # 2 already billed 3 real patients (PATIENT_BILLING), and
            # PatientBilling.patient_id is unique, so blindly using the
            # first patient row would violate that constraint.
            all_patients = db.query(real_clinic_api.models.Patient).all()
            billed_ids = {b.patient_id for b in db.query(real_clinic_api.models.PatientBilling).all()}
            patient = next(p for p in all_patients if p.id not in billed_ids)
            db.add(real_clinic_api.models.PatientBilling(
                patient_id=patient.id, outstanding_amount=1250.50, due_date=None,
            ))
            db.commit()
            phone = patient.phone
        finally:
            db.close()

        result = billing(real_clinic_api.client, phone)
        assert result == {
            "patient_found": True, "found": True,
            "outstanding_amount": 1250.50, "due_date": None,
        }

    def test_zero_balance_is_a_real_distinct_outcome_from_no_record(self, real_clinic_api):
        """0.0 (a real, reviewed zero balance) must NOT be conflated with
        "no billing record at all" -- both are honest, but different,
        outcomes (see clinic-api/models.py's PatientBilling docstring)."""
        db = real_clinic_api.db.SessionLocal()
        try:
            all_patients = db.query(real_clinic_api.models.Patient).all()
            patient = next(p for p in all_patients if p.id not in
                            {b.patient_id for b in db.query(real_clinic_api.models.PatientBilling).all()})
            db.add(real_clinic_api.models.PatientBilling(
                patient_id=patient.id, outstanding_amount=0.0, due_date=None,
            ))
            db.commit()
            phone = patient.phone
        finally:
            db.close()

        result = billing(real_clinic_api.client, phone)
        assert result["found"] is True
        assert result["outstanding_amount"] == 0.0

    def test_due_date_is_iso_formatted_when_present(self, real_clinic_api):
        import datetime
        db = real_clinic_api.db.SessionLocal()
        try:
            all_patients = db.query(real_clinic_api.models.Patient).all()
            billed_ids = {b.patient_id for b in db.query(real_clinic_api.models.PatientBilling).all()}
            patient = next(p for p in all_patients if p.id not in billed_ids)
            due = datetime.datetime(2026, 10, 15, 0, 0, 0)
            db.add(real_clinic_api.models.PatientBilling(
                patient_id=patient.id, outstanding_amount=500.0, due_date=due,
            ))
            db.commit()
            phone = patient.phone
        finally:
            db.close()

        result = billing(real_clinic_api.client, phone)
        assert result["due_date"] == "2026-10-15T00:00:00"
