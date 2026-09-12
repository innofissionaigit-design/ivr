"""Blueprint Appendix C: caller state -> conversation parameters.

This is the deterministic half of "empathy is a policy, not a prompt". The
model never decides how slowly to speak or how often to confirm; this table
does, and the model only picks words inside the constraints it hands down.

The table is transcribed VERBATIM from Appendix C, including values that look
like they could be tidied away:

  * "slow_normal" is a real third speech_rate that only the anger row uses.
    Collapsing it into "default" would silently lose a state the Blueprint
    distinguishes.

  * one_question_at_a_time is FALSE on the normal row, because Appendix C
    gives q/turn as "up to 2" there. That reads wrong at a glance. The field
    states what is PERMITTED, not what the agent happens to do:
    missing_slot_prompt() only ever asks one question, which is inside
    "up to 2" and therefore compliant. Setting it True would over-constrain
    the normal case and contradict the table.

  * The emergency row carries no speech parameters at all. See derive().
"""
from __future__ import annotations

# (speech_rate, response_length, confirmation, one_question_at_a_time,
#  interruption_tolerance, escalation_threshold)
#
# Order of evaluation matters: distress outranks senior, because a distressed
# elderly caller needs the distress row, not a merge of the two.
_ROWS = {
    "distressed": ("slow",        "short",      "explicit", True, "very_high", "low"),
    "angry":      ("slow_normal", "short",      "explicit", True, "high",      "low"),
    "confused":   ("slow",        "very_short", "explicit", True, "high",      "lowered"),
}

_SENIOR = ("slow",    "short",  "explicit", True,  "high",   "lowered")
_NORMAL = ("default", "normal", "implicit", False, "normal", "normal")


class EmergencyDetected(Exception):
    """caller_state == 'emergency' is not a speech-policy outcome.

    Raised rather than returned so it cannot be ignored by a caller that only
    unpacks the happy path. Appendix C's emergency row says "stop normal flow,
    escalate immediately" -- it is Appendix D's state machine, not a set of
    speech settings, and policy/emergency.py owns it.
    """


def derive(caller_state: str, senior: bool | None) -> tuple | None:
    """-> the six response parameters, or None for an emergency. Pure; no I/O.

    Returns None when caller_state == "emergency" ON PURPOSE. Appendix C's last
    row carries no speech parameters, because the correct response to an
    emergency is to stop the conversation, not to hold it more carefully.
    Inventing six plausible values would turn a hard stop into a
    slightly-more-considerate chat, which is the failure that row exists to
    prevent.

    "unknown" falls to the NORMAL row deliberately. That is NOT a claim the
    caller is calm -- it is the behaviour the agent already has, so wiring this
    in changes nothing until a detector exists. The absence stays recorded in
    caller_state itself, which remains "unknown".
    """
    if caller_state == "emergency":
        return None
    if caller_state in _ROWS:
        return _ROWS[caller_state]
    if senior is True:
        return _SENIOR
    return _NORMAL
