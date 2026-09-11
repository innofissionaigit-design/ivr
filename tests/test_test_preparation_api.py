"""ADDED BY SOURAV -- tests for the new GET /api/v1/tests/preparation
endpoint backing "Caller asks how to prepare for a test" (Epic:
Conversation -- Information and Enquiry). Against the REAL clinic-api
FastAPI app, a REAL SQLite database, seeded with the REAL clinic-api/
seed.py data -- mirrors tests/test_clinic_api_health_and_info.py's own
established "real backend, real seed data, real HTTP round-trip"
pattern, including its one-fixture-per-file convention (the
`real_clinic_api` fixture below is duplicated from that file, not
imported).

Also covers the two decisions this story's DB work turned on, both
sourced from the business's own lab_tests_with_fallback_config sample
file:

  1. PRICE CONFLICT RESOLUTION -- the user's own explicit instruction:
     "if prices are not same keep the price from seed.py as it is". The
     sample file's Lipid Profile price_inr (650) disagreed with the
     already-seeded value (670.17) -- TestPriceConflictIsResolvedInFavor
     OfSeedPy below locks in that the seeded price won, not the file's.

  2. HONEST "NOT SEEDED YET" -- only 8 of the ~27 seeded LabTest rows
     have real advisory content (the ones the sample file actually
     covered). Every other test must say `advisory_available: false`,
     never guess "no restrictions" -- see models.py's own comment on why
     every advisory column is nullable with no default.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CLINIC_API_DIR = os.path.join(TESTS_DIR, "..", "clinic-api")

# The 8 tests the sample file actually supplied advisory content for.
_ADVISORY_COVERED_TESTS = [
    "Complete Blood Count (CBC)",
    "Blood Sugar Fasting",
    "Blood Sugar PP",
    "Lipid Profile",
    "Thyroid Profile (T3 T4 TSH)",
    "Urine Routine Examination",
    "Chest X-Ray (PA view)",
    "USG Whole Abdomen",
]

_ADVISORY_SCRIPT_KEYS = (
    "advisory_script_en", "advisory_script_hinglish",
    "advisory_script_banglish", "advisory_script_bn",
)


@pytest.fixture(scope="module")
def real_clinic_api(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("clinic_test_preparation_live") / "clinic.db"
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


def preparation(client, name):
    return client.get("/api/v1/tests/preparation", params={"name": name}).json()


def search(client, name):
    return client.get("/api/v1/tests/search", params={"name": name}).json()


class TestEveryAdvisoryCoveredTestReturnsFullData:
    @pytest.mark.parametrize("test_name", _ADVISORY_COVERED_TESTS)
    def test_found_with_advisory_available(self, real_clinic_api, test_name):
        r = preparation(real_clinic_api.client, test_name)
        assert r["found"] is True
        assert r["advisory_available"] is True
        assert r["test_name"] == test_name

    @pytest.mark.parametrize("test_name", _ADVISORY_COVERED_TESTS)
    def test_fasting_required_is_a_real_bool_not_none(self, real_clinic_api, test_name):
        r = preparation(real_clinic_api.client, test_name)
        assert r["fasting_required"] in (True, False)

    @pytest.mark.parametrize("test_name", _ADVISORY_COVERED_TESTS)
    def test_every_advisory_script_language_is_present_and_non_empty(self, real_clinic_api, test_name):
        r = preparation(real_clinic_api.client, test_name)
        for key in _ADVISORY_SCRIPT_KEYS:
            assert r[key], f"{key} missing or empty for {test_name!r}"
            assert "{test_name}" in r[key], (
                f"{key} for {test_name!r} must still contain the literal "
                f"placeholder -- substitution happens at speak time, not here"
            )

    def test_a_fasting_required_test_says_so_with_real_hours(self, real_clinic_api):
        r = preparation(real_clinic_api.client, "Blood Sugar Fasting")
        assert r["fasting_required"] is True
        assert r["fasting_hours"] == "8-12 hours"
        assert "plain water" in r["water_allowance"].lower()

    def test_a_no_fasting_test_says_so_explicitly(self, real_clinic_api):
        r = preparation(real_clinic_api.client, "Complete Blood Count (CBC)")
        assert r["fasting_required"] is False


class TestAliasBasedLookupWorksForPreparationToo:
    """Same matching ladder /api/v1/tests/search already uses (see
    clinic-api/main.py's _find_lab_test(), factored out specifically so
    this endpoint could reuse it) -- these are spot checks, not a full
    re-test of that ladder (test_clinic_api_health_and_info.py and
    friends already cover it thoroughly elsewhere)."""

    def test_bengali_script_alias_resolves(self, real_clinic_api):
        r = preparation(real_clinic_api.client, "খালি পেটে সুগার")
        assert r["found"] is True
        assert r["test_name"] == "Blood Sugar Fasting"
        assert r["advisory_available"] is True

    def test_newly_merged_alias_from_the_sample_file_resolves(self, real_clinic_api):
        # "FBS" -- merged into seed.py's aliases specifically for this
        # story (see seed.py's own "ADDED BY SOURAV" comment on this
        # LAB_TESTS entry).
        r = preparation(real_clinic_api.client, "fbs")
        assert r["found"] is True
        assert r["test_name"] == "Blood Sugar Fasting"

    def test_unknown_query_returns_not_found_with_suggestions(self, real_clinic_api):
        r = preparation(real_clinic_api.client, "Zzznonexistent Test")
        assert r == {
            "found": False, "query": "Zzznonexistent Test",
            "did_you_mean": r["did_you_mean"],
        }
        # Not asserting exact suggestions (difflib's own fuzzy scoring is
        # covered elsewhere) -- just that this shape, not a crash, comes
        # back for a genuinely unknown query.


class TestTestsWithNoAdvisoryDataAreHonestNotGuessed:
    """The other ~19 seeded tests never got real advisory content from
    the business -- this must never be papered over with a guessed
    "no special preparation needed" default (see models.py's own comment
    on LabTest's advisory columns)."""

    # NOTE: deliberately NOT using "TSH" here. seed.py has both a
    # standalone "TSH" test AND "Thyroid Profile (T3 T4 TSH)" (which DOES
    # have advisory data) -- _find_lab_test()'s substring "contains"
    # matching resolves a bare "TSH" query to the Thyroid Profile panel
    # first (pre-existing ambiguity in the shared matching ladder, not
    # something introduced by this story -- see TEST_REPORT_test_
    # preparation.md for the flag raised to the user about it). Using it
    # here would make this test assert something that isn't actually true
    # of the running system.
    @pytest.mark.parametrize("test_name", ["ESR", "HbA1c", "Vitamin D (25-OH)", "Vitamin B12"])
    def test_found_but_advisory_not_available(self, real_clinic_api, test_name):
        r = preparation(real_clinic_api.client, test_name)
        assert r["found"] is True
        assert r["advisory_available"] is False
        # No structured or scripted advisory fields leak through when
        # there is genuinely nothing to say.
        assert "fasting_required" not in r
        for key in _ADVISORY_SCRIPT_KEYS:
            assert key not in r


class TestPriceConflictIsResolvedInFavorOfSeedPy:
    """The one real price disagreement between the sample file and the
    already-seeded catalogue: Lipid Profile (file said 650, seed.py
    already had 670.17). The user's own explicit instruction was to keep
    seed.py's price -- checked here directly against the DB, and via the
    UNRELATED /api/v1/tests/search endpoint, to prove the preparation
    story's own DB changes never touched pricing at all."""

    def test_seeded_price_is_unchanged_in_the_database(self, real_clinic_api):
        with real_clinic_api.db.SessionLocal() as db:
            t = db.query(real_clinic_api.models.LabTest).filter_by(
                name="Lipid Profile").first()
            assert t.rate_inr == 670.17

    def test_search_endpoint_still_returns_the_seeded_price_not_the_file_price(self, real_clinic_api):
        r = search(real_clinic_api.client, "Lipid Profile")
        assert r["rate_inr"] == 670.17
        assert r["rate_inr"] != 650

    def test_preparation_endpoint_does_not_return_pricing_at_all(self, real_clinic_api):
        # Pricing and preparation are deliberately separate concerns/
        # endpoints (see _test_reply_dict() vs _test_preparation_reply_
        # dict() in clinic-api/main.py) -- this endpoint never echoes a
        # price either way, so there is no second place price drift
        # could hide.
        r = preparation(real_clinic_api.client, "Lipid Profile")
        assert "rate_inr" not in r
        assert "price_inr" not in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
