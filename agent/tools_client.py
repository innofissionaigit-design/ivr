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

import json

import httpx

DEFAULT_TIMEOUT_S = 4.0  # a phone caller will not wait much longer than this per lookup


def _parse_exact(response: httpx.Response) -> dict:
    """Parse a JSON body the way `httpx.Response.json()` does, except every
    JSON number with a decimal point is kept as the literal source text
    instead of being coerced to `float`.

    Bug this fixes: `float(100.00) == float(100.0)`, so once a value like a
    rate goes through the ordinary `float`, its trailing zero is gone for
    good -- "100.00" becomes "100.0" and stays that way through every
    template and TTS stage downstream, no matter how carefully they quote
    it. Money is exactly what this loses digits on, silently, on every
    figure whose decimals happen to end in zero. Parsing decimals as `str`
    keeps the exact digits the API sent, which is what "figures pass from
    the validated response into the template unchanged" requires.

    Whole numbers (no decimal point in the source, e.g. `650`) are
    unaffected -- they stay `int`, exactly as before this change.
    """
    return json.loads(response.text, parse_float=str)


class ToolCallError(Exception):
    """The backing service itself failed -- distinct from a normal
    not-found/unavailable result, which is not an error."""


class ClinicToolsClient:
    def __init__(self, base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)

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
            return _parse_exact(r)
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
            return _parse_exact(r)
        except httpx.HTTPError as e:
            raise ToolCallError(f"get_doctor_availability({doctor_name!r}, {date!r}): {e}") from e

    # ---- Tool 3: POST /api/v1/appointments ----
    # Body: {"doctor_name", "date", "time_slot", "patient_name", "phone"}
    # Expected response shape:
    #   success=true:  {"success": true, "confirmation_id": "KCD-20260824-0031",
    #                    "doctor_name": "...", "date": "...", "time_slot": "..."}
    #   success=false: {"success": false, "reason": "slot_taken" | "missing_field" | "doctor_not_found",
    #                    "alternative_slots": ["17:30", "18:15"]}
    async def book_appointment(self, doctor_name: str, date: str, time_slot: str,
                                patient_name: str, phone: str) -> dict:
        body = {
            "doctor_name": doctor_name, "date": date, "time_slot": time_slot,
            "patient_name": patient_name, "phone": phone,
        }
        try:
            r = await self._client.post("/api/v1/appointments", json=body)
            r.raise_for_status()
            return _parse_exact(r)
        except httpx.HTTPError as e:
            raise ToolCallError(f"book_appointment({body!r}): {e}") from e

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
            return _parse_exact(r)
        except httpx.HTTPError as e:
            raise ToolCallError(f"get_doctors_by_department({department!r}, {date!r}): {e}") from e

    # =========================================================================
    # ADDED BY SOURAV -- "Lab Report Status & Secure Delivery" combined story.
    # Three new tool calls, mirroring the exact shape of the four above:
    # every method still returns a plain dict, still never raises for a
    # normal outcome (NOT_READY, OTP_INVALID, DELIVERY_DISABLED are all
    # valid answers a caller can be told, not errors), and only raises
    # ToolCallError for real infrastructure failure -- clinic-api/main.py's
    # new endpoints follow that same "named reason, not an exception"
    # convention for exactly this reason.
    # =========================================================================

    # ---- Tool 5: GET /api/v1/reports/status?phone=...&test_name=... ----
    # Expected response shape:
    #   patient not found:  {"patient_found": false}
    #   no/no-matching report: {"patient_found": true, "found": false, "reason": "NOT_FOUND"}
    #   multiple candidates: {"patient_found": true, "found": false, "reason": "AMBIGUOUS",
    #                         "candidates": [{"report_number", "test_name", "status", ...}]}
    #   single match: {"patient_found": true, "found": true, "report_number", "test_name",
    #                  "status", "delivery_enabled", "expected_ready_at", "ready_at"}
    async def get_report_status(self, phone: str, test_name: str | None = None) -> dict:
        params = {"phone": phone}
        if test_name:
            params["test_name"] = test_name
        try:
            r = await self._client.get("/api/v1/reports/status", params=params)
            r.raise_for_status()
            return _parse_exact(r)
        except httpx.HTTPError as e:
            raise ToolCallError(f"get_report_status({phone!r}, {test_name!r}): {e}") from e

    # ---- Tool 6: POST /api/v1/reports/delivery/request ----
    # Body: {"phone", "report_number"}
    # Expected response shape:
    #   success=true:  {"success": true, "reason": "OTP_REQUIRED", "masked_phone": "...1234"}
    #   success=false: {"success": false, "reason": "PATIENT_NOT_FOUND" | "NOT_FOUND" |
    #                    "NOT_READY" | "PROCESSING" | "CANCELLED" | "DELIVERY_DISABLED"}
    async def request_report_delivery(self, phone: str, report_number: str) -> dict:
        body = {"phone": phone, "report_number": report_number}
        try:
            r = await self._client.post("/api/v1/reports/delivery/request", json=body)
            r.raise_for_status()
            return _parse_exact(r)
        except httpx.HTTPError as e:
            raise ToolCallError(f"request_report_delivery({body!r}): {e}") from e

    # ---- Tool 7: POST /api/v1/reports/otp/verify ----
    # Body: {"phone", "report_number", "otp_code"}
    # Expected response shape:
    #   success=true:  {"success": true, "reason": "DELIVERY_SENT", "masked_phone": "...1234",
    #                    "signed_link_expires_minutes": 15}
    #   success=false: {"success": false, "reason": "PATIENT_NOT_FOUND" | "NOT_FOUND" |
    #                    "NOT_READY" | "PROCESSING" | "CANCELLED" | "DELIVERY_DISABLED" |
    #                    "OTP_NOT_REQUESTED" | "OTP_ALREADY_USED" | "OTP_EXPIRED" |
    #                    "OTP_MAX_ATTEMPTS" | "OTP_INVALID" | "DELIVERY_FAILED"}
    #
    # RULE 9 (never reveal the OTP): note there is no code path anywhere in
    # this method, clinic-api's endpoint, or reply_templates.py's reply
    # function that reads `otp_code` back out of a response -- the caller
    # only ever finds out whether their guess was accepted, never what the
    # right value was.
    async def verify_report_otp(self, phone: str, report_number: str, otp_code: str) -> dict:
        body = {"phone": phone, "report_number": report_number, "otp_code": otp_code}
        try:
            r = await self._client.post("/api/v1/reports/otp/verify", json=body)
            r.raise_for_status()
            return _parse_exact(r)
        except httpx.HTTPError as e:
            # otp_code is deliberately NOT interpolated into this message --
            # DoD gate 6 ("no secret-shaped literal ... in a log line"), and
            # this exception's text is exactly what logger.error() below
            # (main_pcm.py / main.py) writes to the log on a tool failure.
            raise ToolCallError(
                f"verify_report_otp(phone={phone!r}, report_number={report_number!r}): {e}"
            ) from e
