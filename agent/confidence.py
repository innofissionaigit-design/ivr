"""Turn-level ASR confidence, and the two floors that decide what to do with it.

WHAT THE SIGNAL IS
------------------
agent/asr.py decodes every utterance TWICE -- once through IndicConformer's
CTC head, once through RNNT -- and records the word-set overlap between the two
transcripts as `decoder_agreement`. Two independently-decoded transcripts
agreeing is real evidence the words are right; disagreeing is real evidence
they are not.

That number was already being computed on every turn and thrown away: main.py
took `asr_result.text` and dropped the rest, so the only ASR gate in the whole
pipeline was empty-vs-non-empty. This module is what reads it.

WHY THREE ZONES AND NOT A THRESHOLD
-----------------------------------
A single threshold forces a choice between acting on a doubtful transcript and
rejecting a usable one. A middle band lets the agent do what a person at a
counter does with a half-heard sentence -- repeat it back and wait for a nod.

    < REJECT_FLOOR            re-prompt, take no action
    REJECT .. CONFIRM_FLOOR   read back what was heard, require an affirmative
    >= CONFIRM_FLOOR          proceed (writes still confirm -- see below)

Independently of the scale: a WRITE always requires an explicit affirmative, at
any confidence. The backlog story only demands that below the threshold, but a
booking is cheap to confirm and expensive to get wrong, and a rule with no
exceptions is far easier to test than one that fires conditionally. The
enforcement for that lives in main.py's _finish_booking, not here.

BOTH FLOORS ARE REASONED, NOT MEASURED
--------------------------------------
They are placeholders, and they are labelled that way on purpose -- the same
way clinic-api's FUZZY_SURNAME_FLOOR says so about itself. Nothing here has
been swept against real call audio yet. Until it has:

  * do not quote these numbers as if they were calibrated;
  * do not tune them to reduce how often the agent asks for confirmation.

The confirmation rate becomes visible the moment this ships and will look like
friction. Widening the proceed zone to make it go away trades a cheap, measured
error (an extra turn) for an expensive, unmeasured one (a wrong booking). The
floors move when calibration data moves them.

Calibration procedure, and the false-accept / false-reject figures it must
publish, are in the implementation plan for this story.
"""
from __future__ import annotations

# REASONED, not measured. See module docstring.
AGREEMENT_CONFIRM_FLOOR = 0.60
AGREEMENT_REJECT_FLOOR = 0.30

PROCEED = "proceed"
CONFIRM = "confirm"
REJECT = "reject"


def zone(asr_result) -> str:
    """-> PROCEED | CONFIRM | REJECT for one transcribed turn.

    The `ctc_fallback` branch is the subtle one. asr.py sets
    decoder_agreement=0.0 when RNNT returned nothing and CTC's text was used
    instead. That zero means "there was no second opinion to compare against",
    NOT "the two decoders disagreed completely" -- but numerically the two are
    indistinguishable, and scoring it as total disagreement would send every
    such turn to REJECT. That is exactly backwards: CTC fallback happens on the
    audio ASR is already finding hard, so it would make the agent refuse to
    listen precisely when a caller is having the most trouble being understood.

    One decoder that produced text is worth reading back. It is not worth
    acting on unchecked, and it is not worth refusing.
    """
    if getattr(asr_result, "decoder_used", None) == "ctc_fallback":
        return CONFIRM

    agreement = getattr(asr_result, "decoder_agreement", None)
    if agreement is None:
        # No comparison was made, so there is no evidence either way. Treat an
        # absent signal as doubt rather than as confidence: the alternative is
        # to proceed on a turn nothing has vouched for. (asr.py used to put a
        # sentinel here, which made this branch unreachable and hid the case.)
        return CONFIRM
    if agreement < AGREEMENT_REJECT_FLOOR:
        return REJECT
    if agreement < AGREEMENT_CONFIRM_FLOOR:
        return CONFIRM
    return PROCEED
