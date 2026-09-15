"""TTL cache for RARELY-CHANGING reference data fetched from clinic-api.

WHY THIS IS A SEPARATE CACHE FROM agent/semantic_cache.py
-----------------------------------------------------------
semantic_cache.py caches a different step entirely: turning the caller's
WORDS into {intent, slots}. Its own docstring is explicit that a cache hit
there still does the live clinic-api lookup afterward -- "the actual number
in the caller's ear always comes from the ... response" was treated as a
guarantee worth keeping, at the cost of a few milliseconds per call.

This module is what changes that trade for ONE narrow class of data:
reference facts that a clinic administrator sets and rarely touches again
(a test's price, whether it needs a prescription, a doctor's recurring
weekly days) -- as opposed to facts that change because of what callers
themselves do on the phone (a slot getting booked, a bill getting paid, a
report finishing processing). Caching the first kind for a short TTL saves
a real ~5ms-plus-network Postgres round trip on repeat questions without
meaningfully risking a wrong answer; caching the second kind is exactly
the bug class this codebase has repeatedly gone out of its way to prevent
(see semantic_cache.py's PII rule, and models.py's nullable-not-defaulted
columns for "not reviewed yet" vs a guessed value).

WHAT IS CACHED (ClinicToolsClient wraps these with `self._ref_cache`):
  - get_test_rate            (price / sample type / report time -- admin-set)
  - get_test_preparation     (advisory content -- admin-set)
  - get_walkin_policy        (walk-in eligibility -- admin-set)
  - get_prescription_policy  (prescription requirement -- admin-set)
  - get_doctor_schedule      (RECURRING weekly days -- admin-set, distinct
                               from get_doctor_availability below)
  - get_doctors_by_department, but ONLY when called with date=None (the
    plain department roster). See the exclusion note below for why a date
    argument routes this call around the cache instead.
  - get_clinic_info          (singleton row, changes essentially never)
  - get_health_packages / search_health_package (admin-set catalogue)
  - get_insurance_coverage   (admin-set (provider, test) policy row)

WHAT IS DELIBERATELY NEVER CACHED, AND WHY
  - get_doctor_availability(name, date): a SPECIFIC day's slot state --
    another caller can book or cancel between two questions about the same
    doctor. This is the exact distinction get_doctor_schedule's own
    docstring in tools_client.py draws between "recurring schedule" and
    "is he in today" -- only the former is reference data.
  - get_doctors_by_department(dept, date=...): passing a date turns this
    into the same "who is actually sitting today" question as
    get_doctor_availability, for the same reason -- see clinic-api's own
    filtering by that day's schedule. Only the date-free roster call is
    reference data.
  - book_appointment: a write, not a fetch. Never a cache candidate.
  - get_report_status, request_report_delivery, verify_report_otp: these
    resolve a specific caller's specific report/OTP state -- caller-
    identity-bound and constantly changing (see semantic_cache.py's own
    exclusion of report_status/report_send from L2 for the identical
    reason). A cache hit here could hand one caller a fact that only ever
    belonged to a different call.
  - get_patient_billing: caller-identity-bound financial data that changes
    the moment a bill is paid. Same reasoning as report status, above.

TTL, NOT "FOREVER"
-------------------
This is reference data, not immutable data -- a clinic admin can still
correct a price or flip a walk-in flag. DEFAULT_TTL_S bounds how long a
correction can take to reach a caller's ear through this cache. Tuned
short enough that a same-shift correction lands within one TTL window,
long enough that repeat questions in a single busy period are still served
from memory instead of hitting Postgres again.
"""
from __future__ import annotations

import threading
import time

# 10 minutes: short enough that an admin fixing a wrong price during a
# shift is corrected within the same shift, long enough to absorb the
# bursty repeat-question pattern a single busy hour actually produces.
DEFAULT_TTL_S = 600.0

# Small on purpose -- this is a handful of tests/doctors/departments/
# packages/providers, not an open-ended keyspace like caller utterances.
DEFAULT_MAX_ENTRIES = 500


class TTLCache:
    """Small, in-process, thread-safe. Exact-key only -- unlike
    semantic_cache.py there is no fuzzy matching here: the caller-facing
    risk that module's PII rule and entity guard exist to prevent doesn't
    apply to admin-set reference facts, so a plain exact-match cache is
    both simpler and sufficient."""

    def __init__(self, ttl_s: float = DEFAULT_TTL_S, max_entries: int = DEFAULT_MAX_ENTRIES):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._store: dict[tuple, tuple[float, dict]] = {}
        self.stats = {"hits": 0, "misses": 0}

    def get(self, key: tuple) -> dict | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.stats["misses"] += 1
                return None
            stored_at, value = entry
            if (time.time() - stored_at) > self.ttl_s:
                del self._store[key]
                self.stats["misses"] += 1
                return None
            self.stats["hits"] += 1
            return value

    def set(self, key: tuple, value: dict) -> None:
        if not value:
            return
        with self._lock:
            self._store[key] = (time.time(), value)
            while len(self._store) > self.max_entries:
                oldest = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest]

    def snapshot(self) -> dict:
        with self._lock:
            total = self.stats["hits"] + self.stats["misses"]
            return {
                **self.stats,
                "entries": len(self._store),
                "hit_rate": round(self.stats["hits"] / total, 3) if total else 0.0,
            }
