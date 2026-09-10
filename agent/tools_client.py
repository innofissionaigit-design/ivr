"""REST client for the clinic's Java (Spring Boot) + PostgreSQL service.

This service owns the actual facts -- prices, schedules, slot availability
-- and is the only thing allowed to state them. The contract below is what
that service needs to implement; nothing here assumes it exists yet.

Every method returns a plain dict and NEVER raises for a normal "not
found" / "unavailable" outcome -- those are valid, expected answers a
caller can be told. It only raises ToolCallError for actual infrastructure
failure (timeout, connection refused, 5xx), which main.py maps to a
distinct "I couldn't check that right now" reply instead of a false
"not found".
"""
from __future__ import annotations

import httpx

DEFAULT_TIMEOUT_S = 4.0  # a phone caller will not wait much longer than this per lookup

# httpx defaults to max_connections=100. That is far more than clinic-api can
# actually work on: its endpoints are sync `def`, so FastAPI runs them in a
# thread pool, and each one needs a SQLAlchemy connection from a pool of
# pool_size=5 + max_overflow=10 = 15. Opening 100 sockets against a service
# that can only process ~15 at once does not make anything faster -- it just
# moves the queue from here (where it is visible and bounded) into clinic-api
# (where it is neither), and every socket waiting there is still burning this
# caller's 4-second timeout.
#
# 16 sits just above that 15-connection ceiling, so the transport stops being
# the thing that oversubscribes the database. Keepalive is set to the same
# number because this is a hot localhost path -- several lookups per turn,
# every turn -- and there is nothing to gain from tearing connections down
# between them.
DEFAULT_LIMITS = httpx.Limits(max_connections=16, max_keepalive_connections=16)


class ToolCallError(Exception):
    """The backing service itself failed -- distinct from a normal
    not-found/unavailable result, which is not an error."""


class ClinicToolsClient:
    def __init__(self, base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout_s, limits=DEFAULT_LIMITS,
        )

    async def aclose(self):
        await self._client.aclose()

    # ---- Tool 1: GET /api/v1/tests/search?name=... ----
    # Expected response shape:
    #   found=true:  {"found": true, "test_name": "...", "rate_inr": 650,
    #                 "sample_type": "Blood", "report_time_hours": 24}
    #   found=false: {"found": false, "query": "...", "did_you_mean": ["..."]}
    async def get_test_rate(self, test_name: str) -> dict:
        try:
            r = await self._client.get("/api/v1/tests/search", params={"name": test_name})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"get_test_rate({test_name!r}): {e}") from e

    # ---- Tool 2: GET /api/v1/doctors/availability?name=...&date=YYYY-MM-DD ----
    # date is OPTIONAL -- omit it to ask "when is this doctor next available".
    # Expected response shape:
    #   found=true:  {"found": true, "doctor_name": "...", "date": "...",
    #                 "available": true, "chamber_hours": "18:00-20:00",
    #                 "next_available_date": null}
    #                or, if not available that date:
    #                {"found": true, ..., "available": false,
    #                 "next_available_date": "2026-08-27"}
    #   found=false: {"found": false, "query": "..."}
    async def get_doctor_availability(self, doctor_name: str, date: str | None) -> dict:
        params = {"name": doctor_name}
        if date:
            params["date"] = date
        try:
            r = await self._client.get("/api/v1/doctors/availability", params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"get_doctor_availability({doctor_name!r}, {date!r}): {e}") from e

    # ---- Tool 3: POST /api/v1/appointments ----
    # Body: {"doctor_name", "date", "time_slot", "patient_name", "phone"}
    # Expected response shape:
    #   success=true:  {"success": true, "confirmation_id": "KCD-20260824-0031",
    #                    "doctor_name": "...", "date": "...", "time_slot": "...",
    #                    "notification": {"id": 41, "event": "booked",
    #                                     "status": "queued", "error_code": null}}
    #   success=false: {"success": false, "reason": "slot_taken" | "missing_field" | "doctor_not_found",
    #                    "alternative_slots": ["17:30", "18:15"]}
    #
    # `notification` is the written confirmation the patient is owed. It is
    # NOT a delivery result -- at the moment this response is built the
    # message has not been attempted yet, and "queued" is the normal,
    # healthy value. reply_templates.booking_reply() reads it only to
    # decide whether it may PROMISE the caller a message; a status of
    # "skipped" or "failed" means it must not, because the caller would
    # then hang up waiting for an SMS that is never coming.
    async def book_appointment(self, doctor_name: str, date: str, time_slot: str,
                                patient_name: str, phone: str) -> dict:
        body = {
            "doctor_name": doctor_name, "date": date, "time_slot": time_slot,
            "patient_name": patient_name, "phone": phone,
        }
        try:
            r = await self._client.post("/api/v1/appointments", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"book_appointment({body!r}): {e}") from e

    # ---- POST /api/v1/appointments/{confirmation_id}/reschedule ----
    # Body: {"date", "time_slot"}
    # Expected response shape:
    #   success=true:  {"success": true, "confirmation_id": "<UNCHANGED>",
    #                    "doctor_name": "...", "doctor_name_bn": "...",
    #                    "date": "...", "time_slot": "...",
    #                    "previous_date": "...", "previous_time_slot": "...",
    #                    "notification": {...}}
    #   success=false: {"success": false,
    #                    "reason": "appointment_not_found" | "appointment_cancelled"
    #                              | "slot_taken" | "missing_field"
    #                              | "doctor_not_available_that_day",
    #                    "alternative_slots": [...]}
    #
    # The confirmation_id is deliberately NOT reissued -- see that
    # endpoint's docstring. A caller who kept the first message still holds
    # a valid reference.
    async def reschedule_appointment(self, confirmation_id: str, date: str,
                                      time_slot: str) -> dict:
        body = {"date": date, "time_slot": time_slot}
        try:
            r = await self._client.post(
                f"/api/v1/appointments/{confirmation_id}/reschedule", json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(
                f"reschedule_appointment({confirmation_id!r}, {body!r}): {e}") from e

    # ---- POST /api/v1/appointments/{confirmation_id}/cancel ----
    # Body: {"reason": "..." | null}   -- audit only, never sent to the patient
    # Expected response shape:
    #   success=true:  {"success": true, "already_cancelled": false,
    #                    "confirmation_id": "...", "doctor_name": "...",
    #                    "date": "...", "time_slot": "...", "notification": {...}}
    #   success=false: {"success": false, "reason": "appointment_not_found"
    #                                               | "cancel_failed"}
    #
    # already_cancelled=true is a SUCCESS, and no second message is sent.
    # Repeat cancellations are normal (a retry, a double-tap, a patient
    # ringing twice) and none of them is a reason to message somebody about
    # a cancellation they were already told about.
    async def cancel_appointment(self, confirmation_id: str,
                                  reason: str | None = None) -> dict:
        try:
            r = await self._client.post(
                f"/api/v1/appointments/{confirmation_id}/cancel", json={"reason": reason})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"cancel_appointment({confirmation_id!r}): {e}") from e

    # ---- Tool 4: GET /api/v1/doctors/by-department?department=...&date=YYYY-MM-DD ----
    # date is OPTIONAL -- omit it to list every doctor in the department
    # regardless of schedule. Pass it (main.py does, whenever the caller
    # named a date, e.g. "today") to filter down to doctors who actually
    # sit that day, each with their chamber hours.
    # Expected response shape:
    #   found=true:  {"found": true, "department": "...", "date": "..." | null,
    #                 "doctors": [{"name", "doctor_name_bn", "qualifications",
    #                              "chamber_hours"?}, ...]}
    #   found=false: {"found": false, "query": "..."}
    async def get_doctors_by_department(self, department: str, date: str | None = None) -> dict:
        params = {"department": department}
        if date:
            params["date"] = date
        try:
            r = await self._client.get("/api/v1/doctors/by-department", params=params)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise ToolCallError(f"get_doctors_by_department({department!r}, {date!r}): {e}") from e
