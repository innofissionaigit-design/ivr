"""Pure decision logic for "Caller asks to be called back".

Evidence backing this story: "No outbound capability" -- this system has
no way to actually place a phone call itself (no telephony egress, no
dialer, nothing). The only honest thing it CAN do when a caller asks for a
callback is: (1) write down who to call, when they'd like it, and why, so
a HUMAN staff member can place the call later, and (2) say so plainly --
never imply the system itself will ring back, and never promise a callback
it cannot actually guarantee will happen (e.g. outside hours, or with the
feature turned off for this deployment).

Mirrors agent/report_flow.py's, agent/compare_flow.py's and agent/state.py's
own established convention: every DECISION here is a pure function with no
dependency on CallSession, WebSocket, tools_client, or asyncio, and no
datetime/os import of its own -- main.py resolves "what time is it" /
"what day is it" / "is the feature enabled" (see agent/callback_config.py)
and hands already-resolved values in, exactly the way it already resolves
`date_iso`/`today_weekday` for doctor_availability/clinic_info before
calling into agent/reply_templates.py.

WHY "OPERATING HOURS" MEANS THE CLINIC'S OWN HOURS
===================================================
There is no separate "callback hours" concept anywhere in this codebase's
data model, and inventing one out of thin air would violate the same
"never invent a fact" discipline agent/reply_templates.py's own module
docstring holds itself to. A callback can only ever be FULFILLED by a
clinic staff member who is physically there to make the call -- so the
clinic's own opening hours (clinic-api's ClinicInfo table, already served
by get_clinic_info() and already used by clinic_info_reply()) is the one
real, existing fact this check is grounded in, not a fabricated new one.
"""
from __future__ import annotations

_CALLBACK_WEEKDAY_KEYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)

# Every reason check_callback_availability() can return, alongside
# `available: False`. `None` (paired with `available: True`) is the only
# value meaning "go ahead and collect the rest of the details".
REASON_DISABLED = "disabled"
REASON_OUTSIDE_HOURS = "outside_hours"
REASON_HOURS_UNKNOWN = "hours_unknown"


def check_callback_availability(hours: dict | None, weekday_idx: int | None,
                                 current_time_hhmm: str | None,
                                 callbacks_enabled: bool) -> dict:
    """-> {"available": bool, "reason": str | None}.

    Checked in this order, each one a real, checkable fact rather than a
    guess:

      1. `callbacks_enabled` -- a deploying clinic's own config switch
         (agent/callback_config.py's CALLBACKS_ENABLED). False turns the
         feature off entirely, regardless of what time it is.
      2. `hours`/`weekday_idx`/`current_time_hhmm` unresolved or malformed
         -- honestly reported as REASON_HOURS_UNKNOWN rather than silently
         defaulting to either "available" (a false promise) or
         "unavailable" (a false refusal); see callback_unavailable_reply()
         for how this is spoken.
      3. Today closed entirely (clinic-api's own `closed` flag for the
         weekday, e.g. Sunday in the seeded data) -- REASON_OUTSIDE_HOURS.
      4. `current_time_hhmm` outside today's [open, close) window -- also
         REASON_OUTSIDE_HOURS. Zero-padded "HH:MM" strings compare
         correctly with plain string comparison, so no time-parsing
         library is needed here, matching this module's "no imports
         beyond what's already used" discipline.

    `hours` is exactly the dict get_clinic_info() already returns under
    its own "hours" key (clinic_info_reply()'s own docstring): keyed by
    the seven lower-case English weekday names, each value
    {"closed": bool, "open": "HH:MM" | None, "close": "HH:MM" | None}.
    """
    if not callbacks_enabled:
        return {"available": False, "reason": REASON_DISABLED}

    if not hours or weekday_idx is None or not (0 <= weekday_idx <= 6) or not current_time_hhmm:
        return {"available": False, "reason": REASON_HOURS_UNKNOWN}

    today = hours.get(_CALLBACK_WEEKDAY_KEYS[weekday_idx])
    if not today or today.get("closed"):
        return {"available": False, "reason": REASON_OUTSIDE_HOURS}

    open_, close_ = today.get("open"), today.get("close")
    if not open_ or not close_:
        return {"available": False, "reason": REASON_HOURS_UNKNOWN}

    if open_ <= current_time_hhmm < close_:
        return {"available": True, "reason": None}
    return {"available": False, "reason": REASON_OUTSIDE_HOURS}


def build_callback_reason(stated_reason: str | None, active_test: str | None = None,
                           active_doctor: str | None = None, active_package: str | None = None) -> str | None:
    """-> the context/reason string to persist alongside a callback
    request (Acceptance Criterion 1: "preserving the conversation context
    and reason"), or None when there is genuinely nothing to preserve.

    Priority, each one a real fact, never a guess:
      1. `stated_reason` -- the caller's OWN words (agent/llm.py's
         "callback_reason" slot), when they actually said why. Always
         wins when present; nothing below ever overrides it.
      2. Whichever of `active_test`/`active_doctor`/`active_package` is
         set -- main.py's own agent/state.py DialogueState, i.e. the last
         entity actually discussed and CONFIRMED to exist earlier in this
         same call (see agent/state.py's own docstring on why `.primary`
         is only ever a lookup-confirmed name, never a guess). Checked in
         this fixed order only because a call can track at most one of
         the three meaningfully most of the time; it is not a claim that
         a test is somehow more relevant than a doctor.
      3. None -- no stated reason and nothing tracked. Deliberately never
         falls back to a generic invented sentence like "general
         enquiry": an empty, honest "no reason given" is preserved as
         such (main.py stores this as a null `reason` field), not
         papered over with a fabricated one.
    """
    if stated_reason and stated_reason.strip():
        return stated_reason.strip()
    if active_test:
        return f"Regarding {active_test}"
    if active_doctor:
        return f"Regarding {active_doctor}"
    if active_package:
        return f"Regarding {active_package}"
    return None
