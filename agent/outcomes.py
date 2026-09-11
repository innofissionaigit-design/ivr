"""The insufficient-verified-information outcome.

Story: "The agent says it cannot confirm rather than guessing" (Epic:
Answer Quality and Grounding; near-duplicate acceptance criteria also sit
under the Truth Validator epic's "The insufficient-verified-information
outcome as a first-class path" -- same owner, not yet picked up, flagged
rather than silently merged).

This is a THIRD outcome, distinct from the two that already exist in this
codebase:

  * "not-found"              -- the tool answered cleanly and the answer
                                 is "no" (agent/reply_templates.py's
                                 found=False / success=False branches).
  * "infrastructure apology" -- the tool call itself failed
                                 (agent/tools_client.py's ToolCallError,
                                 caught in four places in main_pcm.py,
                                 all four speaking the same hardcoded
                                 "can't check this right now" line).
  * "insufficient-verified-information" (this file) -- the tool call
    SUCCEEDED and returned a value, but the value cannot be trusted
    enough to speak. The agent says so plainly instead of guessing.

SCOPE -- deliberately narrow, per the three constraints agreed before this
was built:

  1. REAL TRIGGER, and only this one: main_pcm.py's _finish_booking()
     calls missing_booking_write_fields() after book_appointment() reports
     success, and checks only that confirmation_id, date and time_slot
     came back non-empty before any of them is read aloud. That is the
     bare minimum any caller of a write API should do, and it is
     deliberately NOT:
       - shape/type validation of the response       ("Schema drift
         rejected at the boundary" -- a different, unpicked story; that
         story's own acceptance criteria route a malformed response to
         the INFRASTRUCTURE APOLOGY, not to this outcome)
       - a plausibility/business-rule check on the values ("Business rule
         validation before verbalisation" -- different, unpicked story)
       - a freshness/staleness check                 ("Freshness bounds
         per field with withholding" -- different, unpicked story)
       - a re-verification of who is allowed to hear the value
         ("Authorisation re-verified at the validator" -- different,
         unpicked story)
       - a re-query of the database to confirm the row really persisted
         ("Write confirmation validated before it is spoken" -- different,
         unpicked story)
       - a fix for the doctor-name fuzzy-matcher silently picking its best
         guess with no confirmation ("Near matches are offered rather than
         guessed or refused" -- different, unpicked story; that one is
         about OFFERING candidates, a different UX from this outcome)
     Building any of those is the rest of the "Truth Validator" epic, not
     this story.

  2. NO FAKE RATE. The acceptance criterion says "its rate is reported per
     intent". A rate needs a denominator -- attempts per intent -- and
     nothing in this codebase counts attempts per intent today (there is
     no metrics system here at all; see below). This file tracks the
     COUNT per intent only, via insufficient_verified_information_counts().
     Dividing by a future attempts-counter, wherever that ends up living,
     is a one-line change against this function -- it is not something
     this file fabricates a denominator for now.

  3. ESCALATION PATH is the JSONL ledger this file appends to
     (ESCALATION_LOG_PATH). This codebase has no metrics or audit
     infrastructure of any kind today -- every existing signal is a plain
     `logging.getLogger("main")` call with no aggregation. A JSONL ledger
     is the smallest thing that is still durable, greppable by a human,
     and diffable, without inventing a dependency this repo doesn't have.
     Wiring it to a real paging/alerting system (PagerDuty, a Slack
     webhook, whatever the clinic actually uses) is a deployment
     integration decision that this file cannot honestly make up against
     a backend that has none today -- record_insufficient_verified_
     information() is exactly the function such an integration would call
     into.
"""
from __future__ import annotations

import datetime
import json
import os
import threading

# One JSON object per line: {"timestamp", "intent", "field", "reason",
# "call_id"}. Overridable via environment variable, the same convention
# agent/tools_client.py's base URL and clinic-api/db.py's DATABASE_URL
# already use, so tests and deployments can point this elsewhere without
# editing this file.
ESCALATION_LOG_PATH = os.environ.get("ESCALATION_LOG_PATH", "logs/escalations.jsonl")

_lock = threading.Lock()
_counts: dict[str, int] = {}

# The bare-minimum check this story wires up. Presence only -- see the
# module docstring for exactly why shape, plausibility, freshness,
# authorization and DB re-confirmation are all deliberately NOT here.
_REQUIRED_BOOKING_WRITE_FIELDS = ("confirmation_id", "date", "time_slot")


def missing_booking_write_fields(result: dict) -> list[str]:
    """Which of confirmation_id/date/time_slot came back missing or empty
    on a booking response that otherwise reported success=True.

    Called from main_pcm.py's _finish_booking() -- the ONE real trigger
    this story wires up. Returns an empty list when the write looks
    complete (the overwhelmingly common case against a healthy backend);
    a non-empty list means _finish_booking() must not read any of these
    values aloud with false confidence.
    """
    return [f for f in _REQUIRED_BOOKING_WRITE_FIELDS if not result.get(f)]


def insufficient_verified_information_reply(language: str = "bengali") -> str:
    """The dedicated spoken template for this outcome -- deliberately
    worded differently from both of the other two outcomes it must stay
    distinct from:

      * a not-found reply ("...নামে কোনো ডাক্তার নেই") says the thing does
        not exist -- this outcome makes no such claim, the value might
        well be correct, it just was not confirmed.
      * the infrastructure apology ("এই মুহূর্তে দেখতে পারছি না। কাউন্টারে
        যোগাযোগ করুন, দয়া করে।") says the system could not be reached at
        all -- this outcome is the opposite: the call to the backend
        worked fine.

    Instead it tells the caller plainly that something came back but
    could not be confirmed, and that a human will follow up -- matching
    "an escalation path", not a dead end or a silent retry. Same 4
    languages as the rest of this codebase's booking-flow templates
    (bengali default, english, hinglish, banglish), and phrased as a
    natural clause with no field label, colon or bracket, consistent with
    "Answers sound like a person, not a database row".
    """
    if language == "english":
        return ("I'm not able to confirm that with certainty right now. "
                "We'll check it directly and get back to you.")
    elif language == "hinglish":
        return ("Abhi iski pakki confirmation nahi kar pa raha hoon. "
                "Hum ise check karke aapko batayenge.")
    elif language == "banglish":
        return ("Ekhon eta thik moto confirm korte parchi na. "
                "Amra eta check kore apnake janiye debo.")
    else:  # bengali
        return ("এই মুহূর্তে এটা নিশ্চিতভাবে বলতে পারছি না। "
                "আমরা এটা যাচাই করে আপনাকে জানাব।")


def record_insufficient_verified_information(intent: str, field: str, reason: str,
                                              call_id: str | None = None) -> None:
    """Called at the ONE trigger point this story wires up. Does two
    things:

      1. Increments an in-process count for `intent` -- see
         insufficient_verified_information_counts(). Deliberately a count,
         not a rate; see the module docstring, constraint 2.
      2. Appends one JSON line to ESCALATION_LOG_PATH -- the escalation
         path the acceptance criteria requires. A human, or a real
         alerting integration once one exists, can tail or grep this
         file.

    A single process-wide lock is enough for this workload (one call
    center's worth of concurrent calls on one process, the same scale
    every other piece of in-process state in this codebase -- session
    dicts, the semantic cache -- already assumes); this is not meant to
    survive a multi-process deployment without becoming a real metrics
    store, which is exactly the upgrade path constraint 2 and 3 above are
    written to not block.
    """
    with _lock:
        _counts[intent] = _counts.get(intent, 0) + 1
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "intent": intent,
            "field": field,
            "reason": reason,
            "call_id": call_id,
        }
        log_dir = os.path.dirname(ESCALATION_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(ESCALATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def insufficient_verified_information_counts(intent: str | None = None) -> dict[str, int] | int:
    """Per-intent occurrence COUNTS, not a rate -- deliberately (see the
    module docstring, constraint 2). Pass `intent` for just that intent's
    count (0 if it has never been recorded); omit it for a snapshot dict
    of every intent's count so far.
    """
    with _lock:
        if intent is not None:
            return _counts.get(intent, 0)
        return dict(_counts)


# =============================================================================
# ADDED BY SOURAV -- "Caller asks how to prepare for a test" story's bundled
# human_fallback config (lab_tests_with_fallback_config sample file's
# voice_agent_config.human_fallback block). A FOURTH outcome, distinct from
# the three the module docstring above already lists: the caller's query
# could not be resolved at all (agent/llm.py's "unclear" intent).
#
# The config's own action is "transfer_to_human_agent", but there is no
# telephony transfer capability anywhere in this codebase (no SIP/PSTN
# library, no call-control API of any kind) -- an actual call transfer is
# not something this file can honestly build. What IS real and buildable,
# following this file's own constraint 3 above ("ESCALATION PATH is the
# JSONL ledger this file appends to... wiring it to a real paging/alerting
# system is a deployment integration decision this file cannot honestly
# make up against a backend that has none today"): the same ledger gets a
# new record, so a human follow-up process has something durable and
# greppable to act on. agent/reply_templates.human_fallback_reply() is the
# spoken half of this same story; this is only the logging half.
#
# Kept as a SEPARATE counter (_handoff_counts) from _counts/
# insufficient_verified_information_counts() above -- these are two
# different outcomes, and merging their counts under one dict would make
# both numbers meaningless.
# =============================================================================

_handoff_counts: dict[str, int] = {}


def record_human_handoff(intent: str, call_id: str | None = None) -> None:
    """Records one "query unresolved -> hand off to a human" event: bumps
    the per-intent count and appends a JSON line to the same
    ESCALATION_LOG_PATH ledger record_insufficient_verified_information()
    above writes to (distinguished by "event": "human_handoff", so the two
    outcomes remain greppable apart in one shared, durable file)."""
    with _lock:
        _handoff_counts[intent] = _handoff_counts.get(intent, 0) + 1
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": "human_handoff",
            "intent": intent,
            "reason": "query_unresolved_or_low_confidence",
            "call_id": call_id,
        }
        log_dir = os.path.dirname(ESCALATION_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(ESCALATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def human_handoff_counts(intent: str | None = None) -> dict[str, int] | int:
    """Per-intent human-handoff occurrence COUNTS -- mirrors
    insufficient_verified_information_counts() above exactly, but reads
    the separate _handoff_counts dict, never _counts."""
    with _lock:
        if intent is not None:
            return _handoff_counts.get(intent, 0)
        return dict(_handoff_counts)


def _reset_for_testing() -> None:
    """Test-only: clear the in-process counters between test cases. Never
    called from production code -- tests import and call this explicitly
    in a fixture, the same pattern as clearing any other module-level
    cache in a test suite."""
    with _lock:
        _counts.clear()
        _handoff_counts.clear()
