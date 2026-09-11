"""ADDED BY SOURAV -- tests for the new GET /api/v1/insurance/coverage
endpoint backing "Insurance Coverage Policy" (Phase 1: Database Schema &
Policy Tables). Same "real backend, real seed data, real HTTP round-trip"
pattern as tests/test_walkin_eligibility_api.py.

UPDATED BY SOURAV -- Phase 2: clinic-api/seed.py now seeds 4 real
InsuranceProvider rows (INSURANCE_PROVIDERS) with real INsurancePolicy
rows against some tests (INSURANCE_POLICIES) -- sample/demo values, not
verified business content, see those dicts' own comments. This file now
verifies the real seeded providers/policies directly against seed.py's
own dicts (imported via the fixture, never retyped), covers a real
provider + real test with NO reviewed policy row, and a genuinely unknown
provider name. The synthetic-fixture class below now uses a fabricated
provider name/aliases ("TATA AIG General Insurance") that deliberately
does not overlap with any real seeded provider's name or aliases -- an
overlapping name would make _find_insurance_provider's substring-then-
alias ladder ambiguous between the real and synthetic rows, which is
exactly the kind of collision Phase 2 needed to check for before seeding.
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
    db_path = tmp_path_factory.mktemp("clinic_test_insurance_live") / "clinic.db"
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


def coverage(client, test_name, provider_name):
    return client.get(
        "/api/v1/insurance/coverage",
        params={"test_name": test_name, "provider_name": provider_name},
    ).json()


class TestUnknownTestIsHonestlyNotFound:
    def test_unknown_test_name(self, real_clinic_api):
        result = coverage(real_clinic_api.client, "Complete Blod Count", "Star Health")
        assert result["test_found"] is False
        assert "did_you_mean" in result


class TestRealSeededProvidersAndPolicies:
    """ADDED BY SOURAV -- Phase 2: every (provider, test) pair listed in
    seed.py's INSURANCE_POLICIES must speak its real reviewed coverage.
    Asserted against INSURANCE_POLICIES/INSURANCE_PROVIDERS themselves
    (imported via the fixture, not retyped here), so this test can never
    silently drift from what is actually seeded."""

    def test_every_seeded_policy_speaks_its_reviewed_coverage(self, real_clinic_api):
        policies = real_clinic_api.seed.INSURANCE_POLICIES
        assert len(policies) > 0

        for provider_name, test_name, coverage_status, pre_auth_required in policies:
            result = coverage(real_clinic_api.client, test_name, provider_name)
            assert result["test_found"] is True, (provider_name, test_name)
            assert result["provider_found"] is True, (provider_name, test_name)
            assert result["provider_name"] == provider_name, (provider_name, test_name)
            assert result["policy_available"] is True, (provider_name, test_name)
            assert result["coverage_status"] == coverage_status, (provider_name, test_name)
            assert result["pre_auth_required"] == pre_auth_required, (provider_name, test_name)

    def test_bengali_alias_resolves_to_real_seeded_provider(self, real_clinic_api):
        # "স্টার হেলথ" is Star Health and Allied Insurance's seeded Bengali
        # alias (INSURANCE_PROVIDERS) -- proves the alias ladder resolves
        # a real seeded provider, not just a synthetic fixture one.
        result = coverage(real_clinic_api.client, "Complete Blood Count (CBC)", "স্টার হেলথ")
        assert result["provider_found"] is True
        assert result["provider_name"] == "Star Health and Allied Insurance"
        assert result["coverage_status"] == "COVERED"

    def test_real_provider_and_test_both_found_but_no_reviewed_policy_row(self, real_clinic_api):
        # National Insurance Company has no INSURANCE_POLICIES row for
        # ESR -- must be honestly unreviewed, never defaulted to
        # NOT_COVERED just because no row exists.
        assert ("National Insurance Company", "ESR") not in {
            (p, t) for p, t, *_ in real_clinic_api.seed.INSURANCE_POLICIES
        }
        result = coverage(real_clinic_api.client, "ESR", "National Insurance Company")
        assert result["test_found"] is True
        assert result["provider_found"] is True
        assert result["policy_available"] is False
        assert "coverage_status" not in result

    def test_unrecognized_provider_name_still_honest(self, real_clinic_api):
        result = coverage(real_clinic_api.client, "Complete Blood Count (CBC)", "Acko General Insurance")
        assert result["provider_found"] is False
        assert result["query_provider"] == "Acko General Insurance"


class TestSyntheticProviderAndPolicy:
    """Inserts a fabricated (never real) InsuranceProvider + InsurancePolicy
    row pair directly, to verify provider alias matching and every
    coverage_status/pre_auth_required formatting path. Uses a name/alias
    set ("TATA AIG General Insurance") that deliberately does not overlap
    with any of Phase 2's real seeded providers -- see this file's own
    top-of-file docstring for why an overlapping name would make provider
    resolution ambiguous."""

    @classmethod
    @pytest.fixture(autouse=True, scope="class")
    def _seed_provider(cls, real_clinic_api):
        db = real_clinic_api.db.SessionLocal()
        try:
            provider = real_clinic_api.models.InsuranceProvider(
                name="TATA AIG General Insurance", aliases="tata aig|tata|টাটা এআইজি",
            )
            db.add(provider)
            db.commit()

            cbc = db.query(real_clinic_api.models.LabTest).filter_by(name="Complete Blood Count (CBC)").first()
            lipid = db.query(real_clinic_api.models.LabTest).filter_by(name="Lipid Profile").first()

            db.add(real_clinic_api.models.InsurancePolicy(
                test_id=cbc.id, provider_id=provider.id,
                coverage_status="COVERED", pre_auth_required=False,
            ))
            db.add(real_clinic_api.models.InsurancePolicy(
                test_id=lipid.id, provider_id=provider.id,
                coverage_status="NOT_COVERED", pre_auth_required=None,
            ))
            db.commit()
        finally:
            db.close()

    def test_exact_provider_name_match(self, real_clinic_api):
        result = coverage(real_clinic_api.client, "Complete Blood Count (CBC)", "TATA AIG General Insurance")
        assert result["provider_found"] is True
        assert result["provider_name"] == "TATA AIG General Insurance"
        assert result["policy_available"] is True
        assert result["coverage_status"] == "COVERED"
        assert result["pre_auth_required"] is False

    def test_english_alias_match_case_insensitive_substring(self, real_clinic_api):
        result = coverage(real_clinic_api.client, "Complete Blood Count (CBC)", "tata")
        assert result["provider_found"] is True
        assert result["provider_name"] == "TATA AIG General Insurance"

    def test_bengali_alias_match(self, real_clinic_api):
        result = coverage(real_clinic_api.client, "Complete Blood Count (CBC)", "টাটা এআইজি")
        assert result["provider_found"] is True
        assert result["provider_name"] == "TATA AIG General Insurance"

    def test_not_covered_status(self, real_clinic_api):
        result = coverage(real_clinic_api.client, "Lipid Profile", "TATA AIG General Insurance")
        assert result["policy_available"] is True
        assert result["coverage_status"] == "NOT_COVERED"

    def test_test_and_provider_both_found_but_no_reviewed_policy_row(self, real_clinic_api):
        # A different real, seeded test that has NO InsurancePolicy row
        # against this synthetic provider at all -- must be honestly
        # unreviewed, never defaulted to NOT_COVERED just because no row
        # exists. (USG Whole Abdomen does have a real policy row under
        # the REAL "Star Health and Allied Insurance" provider -- this
        # queries the unrelated synthetic TATA AIG provider instead, so
        # the two don't interfere.)
        result = coverage(real_clinic_api.client, "USG Whole Abdomen", "TATA AIG General Insurance")
        assert result["test_found"] is True
        assert result["provider_found"] is True
        assert result["policy_available"] is False
        assert "coverage_status" not in result

    def test_unrecognized_provider_name_still_honest(self, real_clinic_api):
        result = coverage(real_clinic_api.client, "Complete Blood Count (CBC)", "Acko General Insurance")
        assert result["provider_found"] is False
        assert result["query_provider"] == "Acko General Insurance"
