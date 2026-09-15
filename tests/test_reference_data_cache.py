"""ADDED BY SOURAV -- "fetch from cache instead of DB directly" request.

Tests agent/reference_data_cache.py's TTLCache directly, and its wiring
into agent/tools_client.py's ClinicToolsClient: every RARELY-CHANGING,
admin-set endpoint must serve a repeat query from cache within its TTL
(skipping the HTTP round trip entirely), while every caller-specific or
fast-changing endpoint (billing, report status, appointment availability,
every write) must hit the HTTP layer on EVERY call, unconditionally --
exactly as before this cache was introduced. See
agent/reference_data_cache.py's own docstring for the full list and the
reasoning behind each side of that split.

httpx.MockTransport is used instead of the real clinic-api app (already
exercised end-to-end elsewhere, in test_phase1_end_to_end_integration.py)
specifically so these tests can count HTTP calls precisely -- the thing
under test here is "did the network get hit", not "is the JSON correct".
"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from agent.reference_data_cache import TTLCache
from agent.tools_client import ClinicToolsClient


def run(coro):
    return asyncio.run(coro)


class _CountingHandler:
    """Records every request URL it sees and returns a canned JSON body
    keyed by path, so tests can assert exact HTTP hit counts."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(str(request.url))
        body = self.responses.get(request.url.path, {"found": False})
        return httpx.Response(200, json=body)


def make_client(handler: _CountingHandler, **kwargs) -> ClinicToolsClient:
    client = ClinicToolsClient(base_url="http://testserver", **kwargs)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://testserver",
    )
    return client


# ---------------------------------------------------------------------------
# TTLCache itself, in isolation
# ---------------------------------------------------------------------------

class TestTTLCacheDirect:
    def test_miss_then_hit(self):
        cache = TTLCache(ttl_s=60)
        assert cache.get(("k",)) is None
        cache.set(("k",), {"v": 1})
        assert cache.get(("k",)) == {"v": 1}

    def test_expires_after_ttl(self):
        cache = TTLCache(ttl_s=0.05)
        cache.set(("k",), {"v": 1})
        assert cache.get(("k",)) == {"v": 1}
        time.sleep(0.1)
        assert cache.get(("k",)) is None

    def test_empty_value_is_never_stored(self):
        # An empty dict/None is never a real answer worth caching --
        # mirrors _is_l2_eligible-style defensiveness in semantic_cache.py.
        cache = TTLCache(ttl_s=60)
        cache.set(("k",), {})
        assert cache.get(("k",)) is None

    def test_evicts_oldest_when_over_capacity(self):
        cache = TTLCache(ttl_s=60, max_entries=2)
        cache.set(("a",), {"v": "a"})
        time.sleep(0.01)
        cache.set(("b",), {"v": "b"})
        time.sleep(0.01)
        cache.set(("c",), {"v": "c"})  # evicts "a", the oldest
        assert cache.get(("a",)) is None
        assert cache.get(("b",)) == {"v": "b"}
        assert cache.get(("c",)) == {"v": "c"}

    def test_snapshot_tracks_hits_and_misses(self):
        cache = TTLCache(ttl_s=60)
        cache.get(("k",))          # miss
        cache.set(("k",), {"v": 1})
        cache.get(("k",))          # hit
        cache.get(("k",))          # hit
        snap = cache.snapshot()
        assert snap["hits"] == 2
        assert snap["misses"] == 1
        assert snap["entries"] == 1
        assert snap["hit_rate"] == pytest.approx(2 / 3, rel=1e-3)


# ---------------------------------------------------------------------------
# Reference (rarely-changing) endpoints: must be served from cache
# ---------------------------------------------------------------------------

class TestReferenceEndpointsAreCached:
    def test_repeated_test_rate_query_hits_http_once(self):
        handler = _CountingHandler({
            "/api/v1/tests/search": {"found": True, "test_name": "CBC", "rate_inr": 650,
                                      "sample_type": "Blood", "report_time_hours": 24},
        })
        client = make_client(handler)
        first = run(client.get_test_rate("CBC"))
        second = run(client.get_test_rate("CBC"))
        assert first == second
        assert len(handler.calls) == 1

    def test_case_and_whitespace_variants_share_one_cache_entry(self):
        handler = _CountingHandler({
            "/api/v1/tests/search": {"found": True, "test_name": "CBC", "rate_inr": 650,
                                      "sample_type": "Blood", "report_time_hours": 24},
        })
        client = make_client(handler)
        run(client.get_test_rate("CBC"))
        run(client.get_test_rate(" cbc "))
        run(client.get_test_rate("Cbc"))
        assert len(handler.calls) == 1

    def test_different_test_names_are_separate_cache_entries(self):
        handler = _CountingHandler({
            "/api/v1/tests/search": {"found": True, "test_name": "x", "rate_inr": 1,
                                      "sample_type": "Blood", "report_time_hours": 1},
        })
        client = make_client(handler)
        run(client.get_test_rate("CBC"))
        run(client.get_test_rate("ESR"))
        assert len(handler.calls) == 2

    def test_cache_entry_re_fetches_after_ttl_expiry(self):
        handler = _CountingHandler({
            "/api/v1/tests/search": {"found": True, "test_name": "CBC", "rate_inr": 650,
                                      "sample_type": "Blood", "report_time_hours": 24},
        })
        client = make_client(handler, cache_ttl_s=0.05)
        run(client.get_test_rate("CBC"))
        time.sleep(0.1)
        run(client.get_test_rate("CBC"))
        assert len(handler.calls) == 2

    def test_test_preparation_is_cached(self):
        handler = _CountingHandler({
            "/api/v1/tests/preparation": {"found": True, "test_name": "HbA1c",
                                           "advisory_available": True, "fasting_required": False},
        })
        client = make_client(handler)
        run(client.get_test_preparation("HbA1c"))
        run(client.get_test_preparation("HbA1c"))
        assert len(handler.calls) == 1

    def test_walkin_policy_is_cached(self):
        handler = _CountingHandler({
            "/api/v1/tests/walkin-policy": {"found": True, "test_name": "CBC",
                                             "policy_available": True, "walkin_eligible": True},
        })
        client = make_client(handler)
        run(client.get_walkin_policy("CBC"))
        run(client.get_walkin_policy("CBC"))
        assert len(handler.calls) == 1

    def test_prescription_policy_is_cached(self):
        handler = _CountingHandler({
            "/api/v1/tests/prescription-policy": {"found": True, "test_name": "HIV Test",
                                                   "policy_available": True, "prescription_required": True},
        })
        client = make_client(handler)
        run(client.get_prescription_policy("HIV Test"))
        run(client.get_prescription_policy("HIV Test"))
        assert len(handler.calls) == 1

    def test_insurance_coverage_is_cached(self):
        handler = _CountingHandler({
            "/api/v1/insurance/coverage": {"test_found": True, "test_name": "CBC",
                                            "provider_found": True, "provider_name": "Star Health",
                                            "policy_available": True, "coverage_status": "COVERED"},
        })
        client = make_client(handler)
        run(client.get_insurance_coverage("CBC", "Star Health"))
        run(client.get_insurance_coverage("CBC", "Star Health"))
        assert len(handler.calls) == 1

    def test_doctor_schedule_is_cached(self):
        handler = _CountingHandler({
            "/api/v1/doctors/schedule": {"found": True, "doctor_name": "Dr. Roy", "days": ["Mon", "Wed"]},
        })
        client = make_client(handler)
        run(client.get_doctor_schedule("Dr. Roy"))
        run(client.get_doctor_schedule("Dr. Roy"))
        assert len(handler.calls) == 1

    def test_clinic_info_is_cached(self):
        handler = _CountingHandler({
            "/api/v1/clinic/info": {"found": True, "clinic_name": "Kolkata Care Diagnostics"},
        })
        client = make_client(handler)
        run(client.get_clinic_info())
        run(client.get_clinic_info())
        assert len(handler.calls) == 1

    def test_health_packages_and_search_are_cached(self):
        handler = _CountingHandler({
            "/api/v1/health-packages": {"packages": []},
            "/api/v1/health-packages/search": {"found": True, "package_name": "Full Body"},
        })
        client = make_client(handler)
        run(client.get_health_packages())
        run(client.get_health_packages())
        run(client.search_health_package("Full Body"))
        run(client.search_health_package("Full Body"))
        assert len(handler.calls) == 2  # one per distinct endpoint, each cached on its 2nd call


# ---------------------------------------------------------------------------
# Dynamic / caller-specific endpoints: must NEVER be served from cache
# ---------------------------------------------------------------------------

class TestDynamicEndpointsAlwaysHitHttp:
    def test_doctor_availability_always_hits_http(self):
        handler = _CountingHandler({
            "/api/v1/doctors/availability": {"found": True, "doctor_name": "Dr. Roy", "available": True},
        })
        client = make_client(handler)
        run(client.get_doctor_availability("Dr. Roy", "2026-09-12"))
        run(client.get_doctor_availability("Dr. Roy", "2026-09-12"))
        assert len(handler.calls) == 2

    def test_book_appointment_always_hits_http(self):
        handler = _CountingHandler({
            "/api/v1/appointments": {"success": True, "confirmation_id": "KCD-1"},
        })
        client = make_client(handler)
        run(client.book_appointment("Dr. Roy", "2026-09-12", "18:00", "Arjun Sen", "9000000001"))
        run(client.book_appointment("Dr. Roy", "2026-09-12", "18:00", "Arjun Sen", "9000000001"))
        assert len(handler.calls) == 2

    def test_report_status_always_hits_http(self):
        handler = _CountingHandler({
            "/api/v1/reports/status": {"patient_found": True, "found": False, "reason": "NOT_FOUND"},
        })
        client = make_client(handler)
        run(client.get_report_status("9000000001"))
        run(client.get_report_status("9000000001"))
        assert len(handler.calls) == 2

    def test_request_report_delivery_always_hits_http(self):
        handler = _CountingHandler({
            "/api/v1/reports/delivery/request": {"success": True, "reason": "OTP_REQUIRED"},
        })
        client = make_client(handler)
        run(client.request_report_delivery("9000000001", "R-1"))
        run(client.request_report_delivery("9000000001", "R-1"))
        assert len(handler.calls) == 2

    def test_verify_report_otp_always_hits_http(self):
        handler = _CountingHandler({
            "/api/v1/reports/otp/verify": {"success": False, "reason": "OTP_INVALID"},
        })
        client = make_client(handler)
        run(client.verify_report_otp("9000000001", "R-1", "1234"))
        run(client.verify_report_otp("9000000001", "R-1", "1234"))
        assert len(handler.calls) == 2

    def test_patient_billing_always_hits_http(self):
        handler = _CountingHandler({
            "/api/v1/patient/billing": {"patient_found": True, "found": True, "outstanding_amount": 0.0},
        })
        client = make_client(handler)
        run(client.get_patient_billing("9000000001"))
        run(client.get_patient_billing("9000000001"))
        assert len(handler.calls) == 2

    def test_doctors_by_department_with_date_always_hits_http(self):
        handler = _CountingHandler({
            "/api/v1/doctors/by-department": {"found": True, "department": "Cardiology", "doctors": []},
        })
        client = make_client(handler)
        run(client.get_doctors_by_department("Cardiology", date="2026-09-12"))
        run(client.get_doctors_by_department("Cardiology", date="2026-09-12"))
        assert len(handler.calls) == 2

    def test_doctors_by_department_without_date_is_cached(self):
        # The date-free roster call IS reference data -- see
        # reference_data_cache.py's exclusion note distinguishing it from
        # the date-filtered call just above.
        handler = _CountingHandler({
            "/api/v1/doctors/by-department": {"found": True, "department": "Cardiology", "doctors": []},
        })
        client = make_client(handler)
        run(client.get_doctors_by_department("Cardiology"))
        run(client.get_doctors_by_department("Cardiology"))
        assert len(handler.calls) == 1


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

class TestReferenceCacheSnapshot:
    def test_snapshot_reflects_real_hits_and_misses(self):
        handler = _CountingHandler({
            "/api/v1/tests/search": {"found": True, "test_name": "CBC", "rate_inr": 650,
                                      "sample_type": "Blood", "report_time_hours": 24},
        })
        client = make_client(handler)
        run(client.get_test_rate("CBC"))   # miss
        run(client.get_test_rate("CBC"))   # hit
        run(client.get_test_rate("ESR"))   # miss (different key; handler answers generically)
        snap = client.reference_cache_snapshot()
        assert snap["hits"] == 1
        assert snap["misses"] == 2
        assert snap["entries"] == 2
