"""Per-turn export of the decoder-agreement signal, as JSONL.

WHY A FILE AND NOT JUST THE LOG LINE
------------------------------------
The acceptance criterion asks for the signal to be "exported per turn and
correlated against human labels on the locked set to establish that it is
meaningful". A formatted log line is exportable in the sense that grep exists,
but the correlation study would then begin with log-scraping -- parsing a
human-readable string whose format nobody promised to keep stable. One JSON
object per turn is the same information with a contract attached.

WHAT IS DELIBERATELY NOT IN IT
------------------------------
The transcript. Not an oversight, and not something to "improve" later.

Every log line in this codebase carries call_id plus metadata and never the
caller's words; that property is one of the few PHI controls the prototype
actually has today, and Blueprint 4.10 requires PII redaction in logs. Writing
transcripts to a rolling file on the pod would quietly undo it.

The correlation the story wants does not need this file to hold the words. The
locked set (Blueprint 4.2) is built from consented, de-identified recordings on
a separate governed pipeline, and it carries the audio, the human transcript
and the human label. This file carries the SIGNAL and a join key. The study
joins the two on (call_id, turn) -- signal from here, ground truth from there,
and the words never leave the governed side.

WHAT IS IN IT, AND WHY EACH FIELD EARNS ITS PLACE
-------------------------------------------------
  ts             UTC ISO-8601. Lets a run be sliced by deployment or by model
                 version once a registry exists.
  call_id        Join key, half of it.
  turn           Join key, other half. Monotonic within a call.
  decoder_used   rnnt | ctc_fallback | none. The agreement score means
                 different things per decoder, so a study that ignores this
                 column will mix three populations.
  agreement      The signal. null when the decoders were not compared -- see
                 agent/asr.py's ASRResult docstring for why that is null rather
                 than a sentinel.
  ctc_words      Size of the two word sets the Jaccard was computed over.
  rnnt_words     A score of 0.0 at n=1 is a coin flip; at n=8 it is evidence.
                 Without these the exported number cannot be weighted.
  zone           What the agent DECIDED (proceed | confirm | reject). Recording
                 the decision alongside the signal is what makes false-accept
                 and false-reject countable later, which is the other half of
                 the calibration story.
  text_len       Character count only. A coarse duration proxy for bucketing;
                 no content.

FAILURE POLICY
--------------
Writing this must never affect a call. Every failure is swallowed after one
warning: a full disk or a read-only mount is a reason to lose telemetry, not a
reason to drop a caller.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import threading

logger = logging.getLogger("turn_log")

# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
# Every row carries one of these in "event". Added when the speakability gate
# needed somewhere to record a blocked reply.
#
# WHY THE GATE COULD NOT JUST ADD A FIELD
# ---------------------------------------
# record() is called immediately after ASR, before the reply has been composed
# -- so by the time a reply is found unspeakable, that turn's row is already on
# disk. The choice was to defer the ASR row until the end of the turn (which
# would lose it entirely whenever a turn failed before replying) or to write a
# second row. Second row.
#
# The cost is that (call_id, turn) is no longer unique in this file, so a naive
# join now multiplies rows. Hence the discriminator on BOTH kinds: the
# correlation study filters event == "asr_turn" and is unaffected. This is a
# breaking change to the file's shape, and it is safe only because nothing
# consumes it yet -- worth saying out loud rather than changing quietly.
EVENT_ASR_TURN = "asr_turn"
EVENT_UNSPEAKABLE = "unspeakable_reply"
# story title: The agent says it cannot confirm rather than guessing
# user story: As a caller, I want to be told plainly when the system
#   cannot verify something, so that I am not given a confident guess.
# acceptance criteria: The insufficient-verified-information outcome has
#   its own template per language, its own metric and its own escalation
#   path, distinct from not-found and from an infrastructure apology. Its
#   rate is reported per intent because a rise means a data or
#   integration problem.
#
# ITS OWN ESCALATION PATH. dev_sourav writes these to a separate
# logs/escalations.jsonl; here they are a third event type in the export that
# already exists, because that file already has the discriminator, the failure
# policy and the PHI rule this needs, and a second append-only file nobody
# reads is not an escalation path -- it is a second thing to forget.
EVENT_INSUFFICIENT = "insufficient_verified_information"

# Default sits on the network volume, which is the only thing that survives a
# pod restart on this deployment. Set to "" to disable the export entirely.
TURN_LOG_PATH = os.environ.get("TURN_LOG_PATH", "/workspace/logs/turn_signal.jsonl")

_lock = threading.Lock()
_warned = False


def record(call_id: str, turn: int, asr_result, zone: str,
           call_state=None) -> None:
    """Append one turn's confidence signal and call state. Never raises.

    call_state is redacted BY CONSTRUCTION, not by a filter: every one of
    its nine fields is a small enum, a boolean or a BCP-47 language tag,
    so none of them can carry a name, a number or a phrase. That is a
    stronger guarantee than stripping a free-text field and hoping the
    pattern list is complete -- and it is why the transcript still is not
    written here (see the module docstring).
    """
    if not TURN_LOG_PATH:
        return

    try:
        row = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "event": EVENT_ASR_TURN,
            "call_id": call_id,
            "turn": turn,
            "decoder_used": getattr(asr_result, "decoder_used", None),
            "agreement": getattr(asr_result, "decoder_agreement", None),
            "ctc_words": getattr(asr_result, "ctc_words", 0),
            "rnnt_words": getattr(asr_result, "rnnt_words", 0),
            "zone": zone,
            "text_len": len(getattr(asr_result, "text", "") or ""),
        }
        if call_state is not None:
            row["call_state"] = call_state.as_dict()
        _append(row)
    except Exception as e:  # noqa: BLE001 - telemetry must not break a call
        _swallow(e)


# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
def record_unspeakable(call_id: str, turn: int, dropped, enforced: bool) -> None:
    """Append one blocked-reply event. Never raises.

    `dropped` is the list of spans the tokenizer would have discarded. It is
    the ONE place in this file where text from a reply is written down, and it
    is safe under the module's PHI rule for a specific reason, not by
    exception: a dropped span is by definition a token the spoken-form tables
    do not cover -- a catalogue label like "Imaging" or a department name --
    and never the caller's words, which are Bengali and never reach this path.
    Recording them is also the point: the list IS the work queue for the next
    table entry, and without it an alert would say only that something broke.
    """
    if not TURN_LOG_PATH:
        return
    try:
        _append({
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "event": EVENT_UNSPEAKABLE,
            "call_id": call_id,
            "turn": turn,
            "dropped": list(dropped),
            # False means the caller heard the reply WITH the hole in it --
            # shadow mode. Without this column the two are indistinguishable
            # in the export, and "how often would we have blocked" is exactly
            # the question shadow mode exists to answer.
            "enforced": bool(enforced),
        })
    except Exception as e:  # noqa: BLE001 - telemetry must not break a call
        _swallow(e)


# story title: The agent says it cannot confirm rather than guessing
# user story: As a caller, I want to be told plainly when the system
#   cannot verify something, so that I am not given a confident guess.
# acceptance criteria: The insufficient-verified-information outcome has
#   its own template per language, its own metric and its own escalation
#   path, distinct from not-found and from an infrastructure apology. Its
#   rate is reported per intent because a rise means a data or
#   integration problem.
#
# `missing` names the fields that came back empty -- a field NAME, never a
# field value, so the PHI rule this module states in its docstring holds
# without an exception. That list is the whole diagnostic: which part of the
# write failed to come back is what tells a data problem from an integration
# one, and without it an alert can only say that something broke.
def record_insufficient(call_id: str, turn: int, intent: str, missing) -> None:
    """Append one unverifiable-write event. Never raises."""
    if not TURN_LOG_PATH:
        return
    try:
        _append({
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "event": EVENT_INSUFFICIENT,
            "call_id": call_id,
            "turn": turn,
            "intent": intent,
            "missing": list(missing),
        })
    except Exception as e:  # noqa: BLE001 - telemetry must not break a call
        _swallow(e)


def _append(row: dict) -> None:
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _lock:
        os.makedirs(os.path.dirname(TURN_LOG_PATH), exist_ok=True)
        with open(TURN_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line)


def _swallow(e: Exception) -> None:
    global _warned
    if not _warned:
        logger.warning("turn signal export disabled after error: %s", e)
        _warned = True
