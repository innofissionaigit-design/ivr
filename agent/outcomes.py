"""The write succeeded, but the agent cannot verify what it says it did.

story title: The agent says it cannot confirm rather than guessing
user story: As a caller, I want to be told plainly when the system cannot
    verify something, so that I am not given a confident guess.
acceptance criteria: The insufficient-verified-information outcome has its own
    template per language, its own metric and its own escalation path, distinct
    from not-found and from an infrastructure apology. Its rate is reported per
    intent because a rise means a data or integration problem.

Ported from dev_sourav's agent/outcomes.py and reshaped for this branch.

THE SCENARIO, WHICH IS NOT THEORETICAL
--------------------------------------
A caller books Dr A Sen for Tuesday at 6:30. The appointment row IS written --
the clinic's database really does have it -- but the response coming back is
missing the confirmation_id, or the date, or the time slot. A serialisation
bug, a truncated body, a proxy that mangled the payload.

The agent then reads back a confirmation number it never received. The caller
writes down nothing, or something wrong, and believes they are booked. In a
diagnostics context that is not a cosmetic bug: they may not follow up, may
arrive on the wrong day, or may hold a number the counter cannot find.

WHY THIS IS A FOURTH OUTCOME AND NOT ONE OF THE EXISTING THREE
--------------------------------------------------------------
agent/tool_outcome.py already separates three cases, and this is none of them:

  answered     the clinic replied and the reply was usable
  not_found    the clinic replied "that thing does not exist"
  unreachable  the clinic did not reply

Here the clinic replied, the write happened, and the reply is unusable. Calling
that "unreachable" would tell the caller to call back about a booking that
already exists, and would put a real data-integrity event in the same bucket as
a network blip -- so the number that should page someone gets averaged into the
number that fires whenever wifi hiccups.

WHAT MOVED OUT OF tool_contract.py TO MAKE ROOM
-----------------------------------------------
confirmation_id, date and time_slot used to sit in that module's
required-when-success list, so an absent one raised ToolContractError ->
ToolCallError -> the infrastructure apology. That is precisely the collapse the
criterion forbids, so those three keys moved here. tool_contract keeps the
fields a template structurally needs in order to render AT ALL (doctor_name,
doctor_name_bn); this module owns the three that constitute the confirmation
OF the write, and checks them for content rather than presence -- an empty
string is exactly as unusable as a missing key and used to pass.

DELIBERATELY NOT IN SCOPE
-------------------------
Presence and non-emptiness. Nothing else. Not schema or type validation, not
business-rule checks, not freshness bounds, not re-querying the database to
confirm the row, not the doctor-name fuzzy-match ambiguity. Each of those is
its own story and naming them here is what keeps this one from quietly growing
into all of them.
"""
from __future__ import annotations

# The three values that ARE the confirmation of a write. A booking reply that
# cannot produce all three has not confirmed anything, whatever its success
# flag says.
REQUIRED_BOOKING_WRITE_FIELDS = ("confirmation_id", "date", "time_slot")


def missing_booking_write_fields(result: dict) -> list[str]:
    """-> the fields that came back missing OR empty on a success=True booking.

    Empty counts as missing, and that is the half a presence check cannot do:
    confirmation_id="" would satisfy any "is the key there" test and would then
    be read aloud as a confirmation number consisting of nothing.

    An empty list is the overwhelmingly common case against a healthy backend,
    so this is a cheap guard on a hot path, not a routine branch.
    """
    return [field for field in REQUIRED_BOOKING_WRITE_FIELDS if not result.get(field)]
