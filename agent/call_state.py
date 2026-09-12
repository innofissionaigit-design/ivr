"""The single structure every downstream layer reads for caller signals.

Blueprint 4.5 specifies exactly nine fields. They are not nine of the same
thing, and reading them as one list is what makes this look more blocked than
it is:

  DETECTED (3) -- observations about the caller. Each needs an acoustic or
      lexical detector. None of those detectors exists yet, so these stay
      unknown/None on every call today.
          caller_state, senior, language

  DERIVED (6) -- deterministic outputs of Appendix C's policy table, computed
      from the three above. No detector, no audio, no model. These are real
      and testable now.
          speech_rate, response_length, confirmation,
          one_question_at_a_time, interruption_tolerance, escalation_threshold

WHY THE DETECTED FIELDS DEFAULT TO unknown/None AND NOT TO PLAUSIBLE VALUES
--------------------------------------------------------------------------
`senior: false` is not a neutral default. It is a claim that this caller is
not elderly, which nothing has verified. Once it is inside a structure other
layers trust, a fabricated value is indistinguishable from an observed one.

That is exactly the sentinel bug already fixed one layer down in agent/asr.py,
where decoder_agreement reported 1.0 -- perfect agreement -- for turns where
nothing had been compared. Same failure, higher up. `None` forces every
consumer to decide what to do about absence instead of silently trusting a
number nobody measured.

WRITE-ONCE
----------
frozen=True makes "written only by call intelligence" enforceable rather than
merely documented: policy/ and responses/ physically cannot mutate it. The
only way to produce one is build() below, which the Call Intelligence
detectors call (Blueprint 4.5: "agent/distress.py, agent/senior_mode.py,
agent/emergency.py - detectors + Call State writer").

NOTE ON FILE LOCATION
---------------------
Blueprint Part 6 names the detector modules and the policy/ and responses/
packages, but never says where the Call State object itself lives. This
location is a choice, not a doc instruction -- made so the three detector
modules share one definition rather than each importing from another
detector.
"""
from __future__ import annotations

import dataclasses

from responses import speech_policy
from responses.speech_policy import EmergencyDetected  # re-exported for callers

SCHEMA_VERSION = "1.0"


@dataclasses.dataclass(frozen=True)
class CallState:
    """The nine Blueprint 4.5 fields, plus the metadata the story requires.

    schema_version and provenance sit ALONGSIDE the nine rather than among
    them, so the payload stays literally the specified shape while
    "versioned" and the false-coverage problem are both still answered.
    """

    # --- metadata ---------------------------------------------------------
    schema_version: str = SCHEMA_VERSION

    # --- detected: written only by Call Intelligence detectors -------------
    caller_state: str = "unknown"        # normal|distressed|angry|confused|emergency|unknown
    senior: bool | None = None
    language: str | None = None          # BCP-47, e.g. "bn-IN"

    # --- derived: Appendix C, via responses/speech_policy.py ---------------
    speech_rate: str = "default"                 # default|slow_normal|slow
    response_length: str = "normal"              # normal|short|very_short
    confirmation: str = "implicit"               # implicit|explicit
    one_question_at_a_time: bool = False         # False on the normal row: "up to 2"
    interruption_tolerance: str = "normal"       # normal|high|very_high
    escalation_threshold: str = "normal"         # normal|lowered|low

    # Which detector set each detected field, and how sure it was. Blueprint
    # 4.5 says detectors report "each with a confidence"; without recording it
    # a default is indistinguishable from an observation. Empty tuple is the
    # honest state today -- no detector has contributed anything.
    provenance: tuple = ()

    def as_dict(self) -> dict:
        """For the per-turn log. Every value is an enum, a bool, a BCP-47 tag
        or None -- none of them can carry a name, a number or a phrase, so
        there is nothing to redact. That is a stronger guarantee than
        filtering a free-text field and hoping the pattern list is complete."""
        return dataclasses.asdict(self)


def build(caller_state: str = "unknown", senior: bool | None = None,
          language: str | None = None, provenance: tuple = ()) -> CallState:
    """The ONLY place a CallState is constructed.

    Called by the Call Intelligence detectors as each is built; nothing in
    policy/ or responses/ may call it. Until a detector exists the three
    detected fields stay unknown/None and the six derived ones come out as
    Appendix C's normal row -- which is the agent's current behaviour, so
    wiring this in changes nothing until something actually detects.

    Raises EmergencyDetected rather than returning a styled CallState when
    caller_state == "emergency": the correct response is to stop the
    conversation, not to hold it more carefully.
    """
    derived = speech_policy.derive(caller_state, senior)
    if derived is None:
        raise EmergencyDetected(provenance)
    rate, length, confirm, one_q, interrupt, escalate = derived
    return CallState(
        caller_state=caller_state, senior=senior, language=language,
        speech_rate=rate, response_length=length, confirmation=confirm,
        one_question_at_a_time=one_q, interruption_tolerance=interrupt,
        escalation_threshold=escalate, provenance=tuple(provenance),
    )
