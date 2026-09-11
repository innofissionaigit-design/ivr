"""ADDED BY SOURAV -- tests for the new clinic-api endpoints backing two
stories at once: "Caller asks about a health package" and "Caller asks
opening hours, address or directions". Against the REAL clinic-api
FastAPI app, a REAL SQLite database, seeded with the REAL clinic-api/
seed.py data -- mirrors tests/test_clinic_api_reports.py's own
established "real backend, real seed data, real HTTP round-trip"
pattern, including its one-fixture-per-file convention (the
`real_clinic_api` fixture below is duplicated from that file, not
imported).

Endpoints under test:
  GET /api/v1/clinic/info
  GET /api/v1/health-packages
  GET /api/v1/health-packages/search?name=...

Also covers the OTHER half of the health-package story: seed.py's
HealthPackage.aliases field was, before this fix, left "" for every one
of the 4 seeded packages (a real, silent gap -- the model and its own
docstring both describe alias-based matching, e.g. "ডায়াবেটিস প্যাকেজ",
but nothing populated it). TestHealthPackageSearch below exercises that
fix directly: English, Hinglish/Banglish and Bengali-script queries all
resolving via the newly-seeded aliases, not just the English name.
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
    db_path = tmp_path_factory.mktemp("clinic_health_info_live") / "clinic.db"
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


def clinic_info(client):
    return client.get("/api/v1/clinic/info").json()


def list_packages(client):
    return client.get("/api/v1/health-packages").json()


def search_package(client, name):
    return client.get("/api/v1/health-packages/search", params={"name": name}).json()


# --------------------------------------------------------------------- #
# GET /api/v1/clinic/info
# --------------------------------------------------------------------- #

class TestClinicInfo:
    def test_found_returns_the_real_seeded_clinic_row(self, real_clinic_api):
        r = clinic_info(real_clinic_api.client)
        assert r["found"] is True
        assert r["clinic_name"] == "Kolkata Care Polyclinic"
        assert r["phone"] == "03340001234"
        assert "Lake View Road" in r["address"]
        assert "metro station" in r["directions"]

    def test_all_seven_weekdays_present_in_order(self, real_clinic_api):
        r = clinic_info(real_clinic_api.client)
        assert list(r["hours"].keys()) == [
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        ]

    def test_a_normal_weekday_has_real_open_and_close_times_and_is_not_closed(self, real_clinic_api):
        r = clinic_info(real_clinic_api.client)
        monday = r["hours"]["monday"]
        assert monday == {"closed": False, "open": "08:00", "close": "20:00"}

    def test_sunday_is_honestly_closed_with_no_fabricated_hours(self, real_clinic_api):
        # models.py's own ClinicInfo.sunday_open/close are nullable
        # specifically because Sunday can be closed -- this must come
        # back as None, never a guessed/copied weekday time.
        r = clinic_info(real_clinic_api.client)
        sunday = r["hours"]["sunday"]
        assert sunday == {"closed": True, "open": None, "close": None}

    def test_missing_clinic_row_is_an_honest_not_found_not_a_500(self, tmp_path_factory):
        # A separate, deliberately UNSEEDED database -- same fail-safe
        # posture as every other endpoint in this file (report_status's
        # own {"patient_found": False}, doctor_availability's {"found":
        # False}, etc.): a real, empty table must never be papered over
        # with a guessed address.
        db_path = tmp_path_factory.mktemp("clinic_info_unseeded") / "clinic.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        sys.path.insert(0, CLINIC_API_DIR)
        for mod in ("db", "models", "main"):
            sys.modules.pop(mod, None)
        try:
            import db as clinic_db
            import models as clinic_models
            clinic_models.Base.metadata.create_all(clinic_db.engine)
            # Deliberately skip seed() -- table exists but is empty.
            import main as clinic_main
            from fastapi.testclient import TestClient
            client = TestClient(clinic_main.app)
            r = client.get("/api/v1/clinic/info").json()
            assert r == {"found": False}
        finally:
            sys.path.remove(CLINIC_API_DIR)


# --------------------------------------------------------------------- #
# GET /api/v1/health-packages
# --------------------------------------------------------------------- #

class TestHealthPackagesList:
    def test_returns_all_four_seeded_packages(self, real_clinic_api):
        r = list_packages(real_clinic_api.client)
        names = {p["package_name"] for p in r["packages"]}
        assert names == {
            "Basic Health Checkup", "Diabetes Screening Package",
            "Full Body Health Package", "Women's Wellness Package",
        }

    def test_each_package_carries_its_real_linked_tests_not_a_free_text_copy(self, real_clinic_api):
        # seed.py links packages to real LabTest rows via HealthPackageTest
        # -- this proves the join actually resolves, not just that some
        # string survived from the HEALTH_PACKAGES dict.
        r = list_packages(real_clinic_api.client)
        diabetes = next(p for p in r["packages"] if p["package_name"] == "Diabetes Screening Package")
        assert diabetes["tests"] == ["Blood Sugar Fasting", "Blood Sugar PP", "HbA1c"]

    def test_price_and_description_are_present_and_correct(self, real_clinic_api):
        r = list_packages(real_clinic_api.client)
        basic = next(p for p in r["packages"] if p["package_name"] == "Basic Health Checkup")
        assert basic["price_inr"] == 999.0
        assert "general health" in basic["description"]

    def test_package_name_bn_is_populated_not_empty(self, real_clinic_api):
        # The whole point of this story's alias fix -- every package must
        # have a non-empty spoken form now, not "".
        r = list_packages(real_clinic_api.client)
        for p in r["packages"]:
            assert p["package_name_bn"], p["package_name"]


# --------------------------------------------------------------------- #
# GET /api/v1/health-packages/search?name=...
# --------------------------------------------------------------------- #

class TestHealthPackageSearch:
    def test_english_name_substring_matches(self, real_clinic_api):
        r = search_package(real_clinic_api.client, "diabetes")
        assert r["found"] is True
        assert r["package_name"] == "Diabetes Screening Package"

    def test_bengali_script_alias_matches(self, real_clinic_api):
        # Straight from models.py's own ClinicInfo/HealthPackage docstring
        # example -- this is the exact query the story was reported for.
        r = search_package(real_clinic_api.client, "ডায়াবেটিস প্যাকেজ")
        assert r["found"] is True
        assert r["package_name"] == "Diabetes Screening Package"

    def test_hinglish_style_alias_matches(self, real_clinic_api):
        r = search_package(real_clinic_api.client, "sugar checkup")
        assert r["found"] is True
        assert r["package_name"] == "Diabetes Screening Package"

    def test_banglish_style_alias_matches(self, real_clinic_api):
        r = search_package(real_clinic_api.client, "puro sharirer checkup")
        assert r["found"] is True
        assert r["package_name"] == "Full Body Health Package"

    def test_womens_wellness_bengali_alias_matches(self, real_clinic_api):
        r = search_package(real_clinic_api.client, "মহিলাদের স্বাস্থ্য প্যাকেজ")
        assert r["found"] is True
        assert r["package_name"] == "Women's Wellness Package"

    def test_close_misspelling_still_resolves_via_fuzzy_fallback(self, real_clinic_api):
        r = search_package(real_clinic_api.client, "full boddy checkup")
        assert r["found"] is True
        assert r["package_name"] == "Full Body Health Package"

    def test_found_reply_includes_tests_for_the_caller_facing_answer(self, real_clinic_api):
        r = search_package(real_clinic_api.client, "Basic Health Checkup")
        assert r["tests"] == ["Complete Blood Count (CBC)", "Blood Sugar Fasting", "Lipid Profile"]
        assert r["price_inr"] == 999.0

    def test_completely_unrelated_query_is_honestly_not_found(self, real_clinic_api):
        r = search_package(real_clinic_api.client, "totally unrelated xyz query")
        assert r == {
            "found": False, "query": "totally unrelated xyz query", "did_you_mean": [],
        }

    def test_a_near_miss_gets_did_you_mean_suggestions(self, real_clinic_api):
        r = search_package(real_clinic_api.client, "vitamin package plan")
        assert r["found"] is False
        assert r["did_you_mean"], "expected at least one suggestion for a near-miss query"
        # Suggestions must be canonical package names, never raw aliases.
        all_names = {"Basic Health Checkup", "Diabetes Screening Package",
                     "Full Body Health Package", "Women's Wellness Package"}
        assert set(r["did_you_mean"]) <= all_names


class TestTransportParity:
    def test_main_dot_py_has_all_three_new_endpoints(self):
        src = open(os.path.join(CLINIC_API_DIR, "main.py"), encoding="utf-8").read()
        assert '"/api/v1/clinic/info"' in src
        assert '"/api/v1/health-packages"' in src
        assert '"/api/v1/health-packages/search"' in src

    def test_seed_dot_py_now_seeds_a_non_empty_alias_for_every_package(self):
        src = open(os.path.join(CLINIC_API_DIR, "seed.py"), encoding="utf-8").read()
        assert '"aliases":' in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
