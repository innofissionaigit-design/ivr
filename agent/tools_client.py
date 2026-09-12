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

# story title: The model never originates a fact
# user story: As a clinical lead, I want every price, date and identifier
#   to come from a verified system response, so that a wrong answer is a
#   data bug rather than a model bug.
# acceptance criteria: Every factual sentence is a template substitution
#   from a validated tool response and the model is never shown a figure
#   it could restate. An automated assertion on every commit proves no
#   model-composed span reaches synthesis on a factual intent.
#
# The "Expected response shape" comments above each method below used to be
# the ONLY statement of the contract. They are now checked: every response is
# validated at this boundary before any template is allowed to read a fact out
# of it. A violation is re-raised as ToolCallError so main.py's existing
# failure path handles it unchanged -- but the log line says "contract", not
# "connection", so a clinic-api schema regression stops looking like an outage.
from agent.tool_contract import ToolContractError, validate as _validate

# story title: A thing not existing is never confused with a system being down
# user story: As a caller, I want to know whether my test does not exist or the
#   system cannot be reached, so that I know whether to call back.
# acceptance criteria: The two produce different spoken sentences and different
#   metrics, and the distinction survives every refactor. This behaviour exists
#   today and gains a permanent regression case.
#
# THE CHOKE POINT. Every clinic call returns through this module and every
# failure is raised from it, so this is the one place that can tell "the thing
# does not exist" from "the clinic did not answer" without anyone remembering
# to say so. See agent/tool_outcome.py for why not main.py.
from agent import tool_outcome

# story title: Numbers are never rounded, reordered or approximated
# user story: As a patient, I want the exact figure, so that what I am quoted
#   is what I pay.
# acceptance criteria: Figures pass from the validated response into the
#   template unchanged and are verbalised digit-faithfully. A test asserts
#   byte-level equality between the tool value and the spoken value for a
#   corpus of amounts, dates and identifiers.
#
# Ported from dev_sourav. The whole fix is the parse_float argument, and it
# earns its place: httpx's r.json() uses json.loads' default float parsing, and
# float("100.00") == float("100.0") -- so a trailing zero is gone before any
# template, any verbaliser or any speech stage could possibly preserve it. No
# amount of care downstream can put a digit back that was discarded at the
# boundary. Money is precisely the kind of figure this loses digits on, on
# every value whose decimals happen to end in zero, silently, every time.
#
# Whole numbers are untouched: no decimal point in the source means no float in
# the first place, so 650 stays int 650 exactly as before.
#
# It also composes with this module's own failure handling rather than fighting
# it -- json.loads raises JSONDecodeError, which subclasses ValueError, so a
# malformed body still lands in the ValueError arm each method already has and
# is still reported as "the system cannot be reached".
#
# HONEST STATUS ON THIS BRANCH: dormant. clinic-api/models.py declares
# rate_inr as Column(Integer), so the live schema cannot emit a decimal rate
# today and nothing in production exercises this path. Correct and defensive,
# and it costs one keyword argument, so it goes in now rather than being
# remembered later when a rate first gains paise.
def _parse_exact(response: httpx.Response) -> dict:
    """Parse a JSON body exactly as httpx.Response.json() does, except that a
    number written with a decimal point keeps its literal source digits."""
    return json.loads(response.text, parse_float=str)


DEFAULT_TIMEOUT_S = 4.0  # a phone caller will not wait much longer than this per lookup


class ToolCallError(Exception):
    """The backing service itself failed -- distinct from a normal
    not-found/unavailable result, which is not an error."""


class ClinicToolsClient:
    def __init__(self, base_url: str, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)
        self.outcomes = tool_outcome.OutcomeCounter()

    async def aclose(self):
        await self._client.aclose()

    def snapshot(self) -> dict:
        return self.outcomes.snapshot()

    def _answered(self, tool: str, payload: dict) -> dict:
        self.outcomes.record(tool, tool_outcome.classify(tool, payload))
        return payload

    def _unreachable(self, tool: str) -> None:
        self.outcomes.record(tool, tool_outcome.UNREACHABLE)

    # ---- Tool 1: GET /api/v1/tests/search?name=... ----
    # Expected response shape:
    #   found=true:  {"found": true, "test_name": "...", "rate_inr": 650,
    #                 "sample_type": "Blood", "report_time_hours": 24}
    #   found=false: {"found": false, "query": "...", "did_you_mean": ["..."]}
    async def get_test_rate(self, test_name: str) -> dict:
        try:
            r = await self._client.get("/api/v1/tests/search", params={"name": test_name})
            r.raise_for_status()
            return self._answered("test_rate", _validate("test_rate", _parse_exact(r)))
        except httpx.HTTPError as e:
            self._unreachable("test_rate")
            raise ToolCallError(f"get_test_rate({test_name!r}): {e}") from e
        except ToolContractError as e:
            self._unreachable("test_rate")
            raise ToolCallError(f"get_test_rate: clinic-api contract violation: {e}") from e
        # story title: A thing not existing is never confused with a system being down
        # user story: As a caller, I want to know whether my test does not exist
        #   or the system cannot be reached, so that I know whether to call back.
        # acceptance criteria: The two produce different spoken sentences and
        #   different metrics, and the distinction survives every refactor. This
        #   behaviour exists today and gains a permanent regression case.
        #
        # json.JSONDecodeError subclasses ValueError, NOT httpx.HTTPError -- so a
        # proxy or gateway answering 200 with an HTML error page escaped every
        # handler above and propagated as an unhandled exception, which on this
        # deployment is the single MOST LIKELY real shape of "the system is
        # down". The caller then got neither sentence: the turn died and the
        # line went quiet. Caught here so it says "cannot be reached", which is
        # exactly what happened.
        except ValueError as e:
            self._unreachable("test_rate")
            raise ToolCallError(f"get_test_rate: malformed response body: {e}") from e

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
            return self._answered("doctor_availability", _validate("doctor_availability", _parse_exact(r)))
        except httpx.HTTPError as e:
            self._unreachable("doctor_availability")
            raise ToolCallError(f"get_doctor_availability({doctor_name!r}, {date!r}): {e}") from e
        except ToolContractError as e:
            self._unreachable("doctor_availability")
            raise ToolCallError(f"get_doctor_availability: clinic-api contract violation: {e}") from e
        # story title: A thing not existing is never confused with a system being down
        # user story: As a caller, I want to know whether my test does not exist
        #   or the system cannot be reached, so that I know whether to call back.
        # acceptance criteria: The two produce different spoken sentences and
        #   different metrics, and the distinction survives every refactor. This
        #   behaviour exists today and gains a permanent regression case.
        #
        # json.JSONDecodeError subclasses ValueError, NOT httpx.HTTPError -- so a
        # proxy or gateway answering 200 with an HTML error page escaped every
        # handler above and propagated as an unhandled exception, which on this
        # deployment is the single MOST LIKELY real shape of "the system is
        # down". The caller then got neither sentence: the turn died and the
        # line went quiet. Caught here so it says "cannot be reached", which is
        # exactly what happened.
        except ValueError as e:
            self._unreachable("doctor_availability")
            raise ToolCallError(f"get_doctor_availability: malformed response body: {e}") from e

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
            return self._answered("book_appointment", _validate("book_appointment", _parse_exact(r)))
        except httpx.HTTPError as e:
            self._unreachable("book_appointment")
            raise ToolCallError(f"book_appointment({body!r}): {e}") from e
        except ToolContractError as e:
            self._unreachable("book_appointment")
            raise ToolCallError(f"book_appointment: clinic-api contract violation: {e}") from e
        # story title: A thing not existing is never confused with a system being down
        # user story: As a caller, I want to know whether my test does not exist
        #   or the system cannot be reached, so that I know whether to call back.
        # acceptance criteria: The two produce different spoken sentences and
        #   different metrics, and the distinction survives every refactor. This
        #   behaviour exists today and gains a permanent regression case.
        #
        # json.JSONDecodeError subclasses ValueError, NOT httpx.HTTPError -- so a
        # proxy or gateway answering 200 with an HTML error page escaped every
        # handler above and propagated as an unhandled exception, which on this
        # deployment is the single MOST LIKELY real shape of "the system is
        # down". The caller then got neither sentence: the turn died and the
        # line went quiet. Caught here so it says "cannot be reached", which is
        # exactly what happened.
        except ValueError as e:
            self._unreachable("book_appointment")
            raise ToolCallError(f"book_appointment: malformed response body: {e}") from e

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
            return self._answered("doctors_by_department", _validate("doctors_by_department", _parse_exact(r)))
        except httpx.HTTPError as e:
            self._unreachable("doctors_by_department")
            raise ToolCallError(f"get_doctors_by_department({department!r}, {date!r}): {e}") from e
        except ToolContractError as e:
            self._unreachable("doctors_by_department")
            raise ToolCallError(f"get_doctors_by_department: clinic-api contract violation: {e}") from e
        # story title: A thing not existing is never confused with a system being down
        # user story: As a caller, I want to know whether my test does not exist
        #   or the system cannot be reached, so that I know whether to call back.
        # acceptance criteria: The two produce different spoken sentences and
        #   different metrics, and the distinction survives every refactor. This
        #   behaviour exists today and gains a permanent regression case.
        #
        # json.JSONDecodeError subclasses ValueError, NOT httpx.HTTPError -- so a
        # proxy or gateway answering 200 with an HTML error page escaped every
        # handler above and propagated as an unhandled exception, which on this
        # deployment is the single MOST LIKELY real shape of "the system is
        # down". The caller then got neither sentence: the turn died and the
        # line went quiet. Caught here so it says "cannot be reached", which is
        # exactly what happened.
        except ValueError as e:
            self._unreachable("doctors_by_department")
            raise ToolCallError(f"get_doctors_by_department: malformed response body: {e}") from e
