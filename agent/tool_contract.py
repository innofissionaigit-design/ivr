"""What a clinic-api response must contain before a template may read it.

# story title: The model never originates a fact
# user story: As a clinical lead, I want every price, date and identifier
#   to come from a verified system response, so that a wrong answer is a
#   data bug rather than a model bug.
# acceptance criteria: Every factual sentence is a template substitution
#   from a validated tool response and the model is never shown a figure
#   it could restate. An automated assertion on every commit proves no
#   model-composed span reaches synthesis on a factual intent.

WHY THIS EXISTS
---------------
The criterion says every factual sentence is a substitution from a VALIDATED
tool response. Nothing was validating anything. agent/tools_client.py returned
r.json() verbatim and the expected shape was written down as a COMMENT above
each method -- accurate, and enforced by nobody.

What happened instead was that reply_templates.py did the checking by accident:

    rate = result["rate_inr"]                    # test_rate_reply
    result['confirmation_id']                    # booking_reply

A response with found=true and no rate_inr raised KeyError inside the template,
which main.py's broad `except` turned into "এই মুহূর্তে দেখতে পারছি না".

That fails SAFE, which is why nobody noticed, and it is worth being precise
about what was wrong with it. Three things:

  * The failure was reported as a tool-call failure, so a clinic-api schema
    regression would have looked exactly like the API being down.
  * It only covers the fields a template happens to subscript with [] rather
    than .get(). `sample_type`, `chamber_hours` and `available` are all read
    with .get() and would silently render as "absent" instead of "wrong".
  * A guard that lives in whichever template happens to be executing is not a
    contract. It cannot be pointed at, and it changes whenever someone edits a
    sentence.

WHAT THIS DOES NOT DO
---------------------
It does not validate TYPES or VALUES -- it will not catch a rate of -5 or a
date of "yesterday". Required keys, checked once, at the boundary. That is the
line between "the response has the shape we agreed on" and "the clinic's data
is correct", and the second one is the clinic's job, not the agent's. Widening
this into a schema library would be more dependency than rule.

Keys are checked for PRESENCE, not truthiness. `chamber_hours: null` is a
correct answer to "is the doctor in on Tuesday" and must not be an error.
"""
from __future__ import annotations


class ToolContractError(Exception):
    """A clinic-api response is missing a field the templates depend on.

    Deliberately NOT a subclass of ToolCallError, and not raised past
    tools_client.py: that module re-raises it as ToolCallError so main.py's
    existing failure path handles it unchanged, while the log line still
    names this as a contract violation rather than a network problem.
    """


# tool -> (success_flag, required_when_true, required_when_false)
#
# Derived from clinic-api/main.py's actual return statements, not from the
# docstrings above them -- the two had already drifted once, which is how
# department_bn came to be missing from a reply that needed it.
_CONTRACTS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "test_rate": (
        "found",
        ("test_name", "test_name_bn", "rate_inr", "sample_type", "report_time_hours"),
        ("query",),
    ),
    "doctor_availability": (
        "found",
        ("doctor_name", "doctor_name_bn", "date", "available",
         "chamber_hours", "next_available_date"),
        ("query",),
    ),
    "doctors_by_department": (
        "found",
        ("department", "department_bn", "date", "doctors"),
        ("query",),
    ),
    # story title: The agent says it cannot confirm rather than guessing
    # user story: As a caller, I want to be told plainly when the system cannot
    #   verify something, so that I am not given a confident guess.
    # acceptance criteria: The insufficient-verified-information outcome has its
    #   own template per language, its own metric and its own escalation path,
    #   distinct from not-found and from an infrastructure apology. Its rate is
    #   reported per intent because a rise means a data or integration problem.
    #
    # confirmation_id, date and time_slot USED TO BE in this list. An absent one
    # therefore raised ToolContractError -> ToolCallError -> the infrastructure
    # apology, which told a caller whose booking had actually SUCCEEDED to call
    # back -- inviting a duplicate of an appointment that already existed, and
    # filing a data-integrity event in the same bucket as a network blip.
    #
    # They moved to agent/outcomes.py. The line between the two modules: this
    # one keeps the fields a template needs in order to RENDER AT ALL, and
    # checks them for presence; that one owns the three fields that constitute
    # the confirmation OF the write, and checks them for content, because
    # confirmation_id="" is exactly as unusable as a missing key and would have
    # passed every presence test here.
    "book_appointment": (
        "success",
        ("doctor_name", "doctor_name_bn"),
        ("reason",),
    ),
}


def validate(tool: str, payload: object) -> dict:
    """-> the payload, unchanged, or raise ToolContractError.

    Returns the payload so call sites read as `return validate(...)` and the
    check cannot be added in a way that forgets to use the result.
    """
    contract = _CONTRACTS.get(tool)
    if contract is None:
        raise ToolContractError(f"no contract registered for tool {tool!r}")

    if not isinstance(payload, dict):
        raise ToolContractError(f"{tool}: expected a JSON object, got {type(payload).__name__}")

    flag, when_true, when_false = contract

    # story title: Near matches are offered rather than guessed or refused
    # user story: As a caller naming something loosely, I want the close
    #   matches offered, so that I am not told my test does not exist when it
    #   does.
    # acceptance criteria: When several catalogue rows fall within the match
    #   band the agent offers up to three by name and asks which. Candidates
    #   are generated across every supported language and romanised spelling.
    #   The did-you-mean path covers the ambiguous case and not only total
    #   failure.
    #
    # A THIRD SHAPE, checked before the flag. An ambiguous response carries
    # found=false and must not be held to the not-found contract, which would
    # pass it through on `query` alone and let a reply be composed with no
    # candidates in it -- the caller would hear a question offering nothing.
    #
    # `candidates` is required to be PRESENT, not non-empty: an empty list is
    # a legitimate answer meaning "several rows matched and at least one of
    # them cannot be said aloud" (see clinic-api's _candidate_dicts), and the
    # template turns that into a re-ask. Requiring content here would report
    # that as the clinic being unreachable.
    if payload.get("ambiguous"):
        missing = [key for key in ("query", "candidates") if key not in payload]
        if missing:
            raise ToolContractError(
                f"{tool}: ambiguous response is missing {missing} -- the agent "
                f"would have asked the caller to choose between nothing"
            )
        return payload

    if flag not in payload:
        raise ToolContractError(f"{tool}: response has no {flag!r} field")

    required = when_true if payload[flag] else when_false
    missing = [key for key in required if key not in payload]
    if missing:
        raise ToolContractError(
            f"{tool}: {flag}={payload[flag]!r} response is missing {missing} "
            f"-- a template would have read this and spoken a wrong or empty fact"
        )
    return payload
