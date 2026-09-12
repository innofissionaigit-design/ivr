"""Kolkata Care Diagnostics -- Bengali voice agent, WebSocket orchestrator.

Turn loop, once a caller's utterance is judged complete (agent/vad_stream.py):

  utterance WAV -> ASR (agent/asr.py, IndicConformer)
                -> intent+slots (agent/llm.py, Ollama JSON-mode,
                   fronted by agent/semantic_cache.py)
                -> Spring Boot lookup (agent/tools_client.py) -- ALWAYS
                   live, never cached; see semantic_cache.py's docstring
                -> reply text, TEMPLATED from the API response, never
                   restated by the model (agent/reply_templates.py)
                -> TTS (agent/tts.py) -> WAV bytes back over the socket

Every stage has a named failure path (see _dispatch_turn) so a caller never
gets dead air: ASR-empty, LLM-failure, tool-failure and TTS-failure each
speak a distinct, pre-recorded apology rather than the process hanging or
the socket just going quiet. See README.md "Error handling" for the full
table and the reasoning behind each choice.

HALF-DUPLEX GATE
----------------
The mic is open for the entire call, and the agent's replies play out of
the caller's speaker. With no gate, the agent hears itself: its own
greeting lands in the same buffer the turn detector is watching, so VAD
fires a "the caller finished talking" on the agent's own voice, ASR
transcribes the agent, and processed_until_s advances past audio the
caller never produced. That is a self-sustaining loop, and it is what
made real calls cut the caller off in the first second and then run a
turn behind for the rest of the call.

Browser echoCancellation does not save this. It is built to cancel a
remote WebRTC peer's rendered stream; here the audio is synthesized
locally and played through Web Audio, which the canceller never sees as a
far-end reference.

So the pipeline is explicitly half-duplex, gated from BOTH ends:
  * client mutes the mic track while agent audio is playing (static/
    index.html) -- the track stays live and keeps emitting, so the WebM
    timeline never breaks, it just carries silence;
  * server refuses to run turn detection while `agent_speaking`, then
    resynchronizes processed_until_s past the muted region once playback
    is confirmed finished.

The cost is no barge-in: a caller cannot interrupt the agent mid-sentence.
That is a real limitation, chosen deliberately over the alternative, which
was a system that interrupted ITSELF. Supporting barge-in properly needs
an acoustic echo canceller with the played audio as a reference signal
(WebRTC APM or speex AEC), which is a much larger change.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime
import difflib
import io
import json
import logging
import os
import tempfile
import time
import uuid
import wave

import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from agent import answer_ledger
from agent import turn_parts
from agent.turn_parts import ANSWERED, INTERACTIVE, UNANSWERABLE
from agent import call_state as call_state_mod
from agent import confidence, outcomes, speakability, tool_outcome, turn_log
from agent.asr import TurnASR
from agent.llm import extract_intent, ExtractionError
from agent.reply_templates import (
    missing_slot_prompt, test_rate_reply, doctor_availability_reply, booking_reply,
    booking_confirm_prompt, heard_confirm_prompt, doctors_by_department_reply,
    booking_correction_prompt, INSUFFICIENT_VERIFIED_INFORMATION_BN,
    date_range_confirm_prompt, UNSPEAKABLE_ESCALATION, with_change_notice,
    near_match_prompt, NEAR_MATCH_UNCLEAR_BN,
    DEFERRED_PART_BN, RESUMING_PART_BN, unanswered_part_prompt,
)
from agent.fast_path import Catalogue, FastPath, COMMIT_MARGIN
from agent.semantic_cache import SemanticCache, embed as _embed_probe
# story title: The model never originates a fact
# user story: As a clinical lead, I want every price, date and identifier
#   to come from a verified system response, so that a wrong answer is a
#   data bug rather than a model bug.
# acceptance criteria: Every factual sentence is a template substitution
#   from a validated tool response and the model is never shown a figure
#   it could restate. An automated assertion on every commit proves no
#   model-composed span reaches synthesis on a factual intent.
#
# resolve_date is the story's core: the model's date becomes a candidate
# and slot_parse.py becomes the record. See its docstring.
from agent.slot_parse import (
    parse_date, parse_time, parse_phone, is_affirmative, is_negative,
    parse_correction_field,
)
# story title: The model never originates a fact
# user story: As a clinical lead, I want every price, date and identifier to
#   come from a verified system response, so that a wrong answer is a data bug
#   rather than a model bug.
# acceptance criteria: Every factual sentence is a template substitution from a
#   validated tool response and the model is never shown a figure it could
#   restate. An automated assertion on every commit proves no model-composed
#   span reaches synthesis on a factual intent.
#
# date_calc owns the calendar. The model names what the caller MEANT
# ("next_week"); every digit is computed here. See that module's docstring for
# the division of authority and for why an expression, unlike an ISO date,
# cannot go stale in the semantic cache.
from agent import date_calc
from agent.date_calc import SOURCE_INTERPRETED, SOURCE_PARSED
from agent.tools_client import ClinicToolsClient, ToolCallError
# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
from agent import tts as tts_mod
from agent.tts import TTSClient, UnspeakableReply, SPEAKABILITY_ENFORCE
from agent.vad_stream import TurnDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

POLL_INTERVAL_S = 0.5
IDLE_TIMEOUT_S = 90.0
UTTERANCE_PAD_S = 0.15  # small trailing pad so ASR doesn't clip the last phoneme

# A live call was observed closing itself ~26s after the last exchange --
# far short of IDLE_TIMEOUT_S, which only fires at 90s. That gap points to
# an intermediate proxy (RunPod's or an nginx in front of it) closing
# WebSocket connections that go quiet for a while, independent of this
# app's own idle logic. A small periodic heartbeat keeps real traffic
# flowing on the socket so no proxy in between decides it's abandoned.
HEARTBEAT_INTERVAL_S = 15.0

# Backstop for the half-duplex gate. Normally the client reports playback
# finished and the gate lifts immediately; this only fires when that
# message never arrives (JS error, stale cached page, a client that
# predates the control channel). Generous on purpose -- lifting the gate
# early puts the agent back to hearing itself, which is the bug.
PLAYBACK_GUARD_S = 3.0

# On resync, rewind slightly before the buffer's decoded end. The region
# being skipped is muted silence, so rewinding into it costs nothing,
# while NOT rewinding risks clipping the caller's first syllable if the
# WebM decode is running a beat behind real time.
RESYNC_REWIND_S = 0.25

CLINIC_API_BASE = os.environ.get("CLINIC_API_BASE", "http://localhost:8080")

# Order also doubles as PRIORITY: the field _next_missing() asks for next
# when several are still empty. doctor_name first because it is almost
# always already known by the time booking starts (named directly, or
# carried over from session.pending after a doctors_by_department /
# doctor_availability turn -- see _continue_pending below).
_BOOKING_FIELDS = ("doctor_name", "date", "time_slot", "patient_name", "phone")

# story title: A thing not existing is never confused with a system being down
# user story: As a caller, I want to know whether my test does not exist or the
#   system cannot be reached, so that I know whether to call back.
# acceptance criteria: The two produce different spoken sentences and different
#   metrics, and the distinction survives every refactor. This behaviour exists
#   today and gains a permanent regression case.
#
# THE SYSTEM-IS-DOWN SENTENCE, and the only one. It was written out at all
# seven ToolCallError handlers, which is seven chances for one of them to drift
# into a different wording -- or, worse, into a not-found wording -- during a
# refactor nobody reviewed carefully. A caller must be able to tell "your test
# does not exist" from "I could not reach the system" without knowing which
# code path they happened to hit, because the two carry opposite instructions:
# one means stop asking, the other means call back.
#
# The not-found sentences deliberately stay in agent/reply_templates.py, where
# every other sentence that NAMES something lives. The separation is the point:
# this file says what the system could not do, that file says what the clinic
# said. tests/test_outcome_distinction.py asserts the two sets never overlap.
SYSTEM_UNREACHABLE_BN = "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।"

# story title: Every critical value is read back before it is used
# user story: As a patient giving a phone number, I want it read back, so that
#   a misheard digit does not send my report to a stranger.
# acceptance criteria: Phone numbers, dates, times and names are confirmed
#   aloud before any write, and a rejection opens a correction path rather than
#   repeating the prompt. Readback is mandatory regardless of confidence for
#   values that affect a write.
#
# Said when the repair ladder runs out -- from confirm_booking, and now also
# from confirm_correction. Named once for the same reason SYSTEM_UNREACHABLE_BN
# is: two copies of a sentence are one refactor away from two different
# sentences, and this one carries a promise -- that nothing was written -- which
# must not drift.
BOOKING_NOT_CONFIRMED_BN = "এখনো কনফার্ম করতে পারলাম না। অ্যাপয়েন্টমেন্টটা করা হয়নি — কাউন্টারে একবার কথা বলে নেবেন।"

# Turns that died on an exception nobody expected. Counted because a metric
# that only tallies TIDY failures reads healthy during exactly the incident it
# exists for. Surfaced at /api/stats beside the per-tool outcomes.
_turn_crashes = 0

# story title: The same question gets the same answer within one call
# user story: As a caller who asks twice, I want the same answer, so that I
#   know which one to believe.
# acceptance criteria: Repeating a question in one call produces an identical
#   factual answer unless the underlying data changed, in which case the
#   change is stated. A test asserts consistency across three repeats with an
#   unchanged backend.
#
# Each AnswerLedger lives and dies with its call, which is the right scope for
# the behaviour and the wrong one for a metric -- nothing would ever read it.
# This is the process-level roll-up, accumulated as each call's verdicts come
# in, on the same argument as _turn_crashes above.
_consistency = {"repeats": 0, answer_ledger.SAME: 0, answer_ledger.CHANGED: 0}

app = FastAPI()

# ---- process-wide singletons: loaded once, shared by every call ----
_asr: TurnASR | None = None
_turn_detector: TurnDetector | None = None
_tools: ClinicToolsClient | None = None
_tts: TTSClient | None = None
_intent_cache: SemanticCache | None = None
_fast_path: FastPath | None = None


@app.on_event("startup")
async def _startup():
    global _asr, _turn_detector, _tools, _tts, _intent_cache, _fast_path
    logger.info("loading IndicConformer...")
    _asr = await asyncio.to_thread(TurnASR)
    logger.info("loading Silero VAD...")
    _turn_detector = await asyncio.to_thread(TurnDetector)
    _tools = ClinicToolsClient(CLINIC_API_BASE)
    _tts = TTSClient()
    _intent_cache = SemanticCache()

    # Pull bge-m3 into VRAM before the first caller needs it. Cold-loading
    # it inside a live turn measured past the client's patience AND past
    # the embed timeout, which silently degraded the cache to exact-match
    # only for the opening minutes of the process -- healthy-looking logs,
    # zero semantic hits. OLLAMA_KEEP_ALIVE=-1 keeps it resident after.
    try:
        await asyncio.to_thread(_embed_probe, "warmup")
        logger.info("embedding model warm")
    except Exception as e:  # noqa: BLE001 - cache is optional, the call is not
        logger.warning("embedding warmup failed, cache starts L1-only: %s", e)

    # Load the 74-row catalogue once so the fast path can identify a test
    # or doctor locally. Optional: if the clinic API is not up yet, every
    # turn simply goes to the LLM, which is the behaviour that existed
    # before this path did.
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=10) as c:
            payload = (await c.get(f"{CLINIC_API_BASE}/api/v1/catalogue")).json()
        _fast_path = FastPath(Catalogue(payload))
        # story title: A thing not existing is never confused with a system
        #   being down
        # user story: As a caller, I want to know whether my test does not
        #   exist or the system cannot be reached, so that I know whether to
        #   call back.
        # acceptance criteria: The two produce different spoken sentences and
        #   different metrics, and the distinction survives every refactor.
        #   This behaviour exists today and gains a permanent regression case.
        #
        # An EMPTY catalogue used to log this same line with a 0 in it and
        # carry on. It is not a quiet condition: the clinic API is up and
        # answering, so nothing is "unreachable", and every single caller is
        # about to be told in a well-formed sentence that their test does not
        # exist. That is the confusion this story is named after, arriving from
        # the data side rather than the code side.
        #
        # clinic-api's own /api/health reports these counts. Nobody was looking.
        if not len(_fast_path.catalogue):
            logger.error("CLINIC CATALOGUE IS EMPTY -- the API is up and has no "
                         "rows, so every caller will be told their test does not "
                         "exist. Check the clinic database before taking calls.")
        else:
            logger.info("fast path ready over %d catalogue rows", len(_fast_path.catalogue))
    except Exception as e:  # noqa: BLE001 - degrade to LLM-only, never fail startup
        logger.warning("catalogue unavailable, fast path disabled: %s", e)
        _fast_path = None

    # STORY [Answer Quality and Grounding]
    # As a patient, I want to hear the whole sentence, so that I am
    # not left guessing what the agent tried to say.
    # Before a single caller connects. Deliberately NOT wrapped in a try:
    # a canned line the synthesizer would mangle is a defect in this
    # repository, and the escalation line being sayable is what stops
    # _speak()'s blocked path recursing. Refusing to start is the correct
    # response to either -- and it is checked whether or not enforcement is
    # on, because a literal in the source is not the unknown data shadow
    # mode exists to measure.
    _tts.assert_canned_lines_speakable()
    logger.info("canned lines verified speakable (%d)", len(tts_mod.PREWARM_LINES))

    logger.info("prewarming TTS...")
    await _tts.prewarm()
    logger.info("startup complete -- ready for calls (speakability enforce=%s)",
                SPEAKABILITY_ENFORCE)


@app.on_event("shutdown")
async def _shutdown():
    if _tools:
        await _tools.aclose()
    if _tts:
        await _tts.aclose()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "asr_loaded": _asr is not None,
        "clinic_api_base": CLINIC_API_BASE,
    }


@app.get("/api/stats")
async def stats():
    """Cache effectiveness, for tuning the similarity threshold against
    real traffic rather than against my assumptions about it.

    `speakability` is not a cache statistic and is here anyway, because this
    is the only endpoint anything scrapes. unspeakable_blocked is the number
    an alert rule would watch; while enforce is false it reads as "replies
    that WOULD have been blocked", which is the whole question shadow mode
    exists to answer. Any non-zero value is a missing spoken-form entry --
    agent/turn_log.py's unspeakable_reply rows name which one.
    """
    tts_snapshot = _tts.snapshot() if _tts else None
    return {
        "fast_path": _fast_path.snapshot() if _fast_path else None,
        "intent_cache": _intent_cache.snapshot() if _intent_cache else None,
        "tts_cache": tts_snapshot,
        # STORY [Answer Quality and Grounding]
        # As a patient, I want to hear the whole sentence, so that I am
        # not left guessing what the agent tried to say.
        "speakability": {
            "enforced": SPEAKABILITY_ENFORCE,
            "blocked": tts_snapshot["unspeakable_blocked"] if tts_snapshot else None,
        },
        # story title: A thing not existing is never confused with a system
        #   being down
        # user story: As a caller, I want to know whether my test does not
        #   exist or the system cannot be reached, so that I know whether to
        #   call back.
        # acceptance criteria: The two produce different spoken sentences and
        #   different metrics, and the distinction survives every refactor.
        #   This behaviour exists today and gains a permanent regression case.
        #
        # The "different metrics" half of the criterion. Per tool: answered,
        # not_found, unreachable, and the derived rate. Alertable in BOTH
        # directions -- unreachable rising is the API or the network;
        # not_found_rate rising is either callers asking for things the clinic
        # does not stock, or the catalogue emptying itself, which today tells
        # every caller their test does not exist in a perfectly well-formed
        # sentence while nothing anywhere notices.
        #
        # turn_crashes sits beside them because a turn that died in our own
        # code is the system being down from the caller's seat, and a metric
        # that counts only tidy failures reads healthy during an incident.
        "clinic": {
            "tools": _tools.snapshot() if _tools else None,
            "turn_crashes": _turn_crashes,
        },
        # story title: The same question gets the same answer within one call
        # user story: As a caller who asks twice, I want the same answer, so
        #   that I know which one to believe.
        # acceptance criteria: Repeating a question in one call produces an
        #   identical factual answer unless the underlying data changed, in
        #   which case the change is stated. A test asserts consistency across
        #   three repeats with an unchanged backend.
        #
        # `changed` is the alertable one. On a catalogue nobody is editing it
        # should sit at zero, so a rising rate is either real churn in the
        # clinic's data or entity resolution landing on a different row for
        # the same words -- and the second of those is precisely the wrong-hit
        # auditing E9-S12 asks for and nothing in this repository has ever
        # been able to see. `repeats` is the denominator: a `changed` count
        # means nothing without knowing how many repeats there were at all.
        "consistency": dict(_consistency),
    }


def _wav_duration_s(wav_bytes: bytes) -> float:
    try:
        with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001 - a fallback clip may not be canonical WAV
        return 5.0


async def _decode_to_wav(raw_path: str, wav_path: str) -> bool:
    """Re-decode the WHOLE growing webm buffer every poll -- same choice
    server.py makes for the consultation recorder, and for the same reason:
    MediaRecorder only puts the container header on the FIRST chunk, so
    later chunks are not independently decodable, and re-decoding a few
    seconds of audio is cheap next to ASR+LLM+TTS.

    NOTE the cost profile: this is O(call length) on every poll, so the
    work per poll grows for the whole duration of a call. Fine for a
    handful of concurrent bench calls; it is the first thing that has to
    change for real concurrency (see README.md "Scaling")."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    return proc.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 44


class CallSession:
    """One raw/decoded buffer for the ENTIRE call, not one per turn.

    An earlier version deleted the buffer and started a fresh file after
    every completed turn. That broke on real testing: MediaRecorder only
    puts the WebM container's header in the very first chunk of the whole
    recording session -- every chunk after a mid-call "reset" was being
    appended to a file that could never be decoded, because its one valid
    header lived in the turn-1 file that had already been deleted. Every
    turn after the first silently failed to decode, forever, for the rest
    of the call.

    Fixed the same way voice-to-rx-repo/server.py already had to: keep
    ONE continuous file for the whole call and track `processed_until_s`
    -- a marker for how much of it a prior turn has already consumed.
    Each poll only ever looks at the UNPROCESSED TAIL past that marker.
    """

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.call_id = uuid.uuid4().hex[:8]
        self.tmpdir = tempfile.mkdtemp(prefix=f"kcd_call_{self.call_id}_")
        self.last_activity = time.time()
        self.dispatch_lock = asyncio.Lock()
        self.processed_until_s = 0.0
        self.utt_seq = 0
        self.last_heartbeat = time.time()
        self.raw_path = os.path.join(self.tmpdir, "call.webm")
        self.wav_path = self.raw_path + ".wav"
        open(self.raw_path, "wb").close()

        # Starts True: the greeting goes out before the caller has said
        # anything, so the gate must already be closed when the first poll
        # tick runs, not opened a moment later by _speak().
        self.agent_speaking = True
        self.speak_deadline = time.time() + PLAYBACK_GUARD_S
        self.resync_pending = False

        # Cross-turn booking state. None outside a booking flow. See
        # _continue_pending's docstring for the shape and why this exists --
        # in short, it is the only thing that survives between turns, since
        # every _resolve_intent call otherwise starts from zero context.
        self.pending: dict | None = None

        # The single structure every downstream layer reads for caller signals
        # (Blueprint 4.5). Replaced wholesale each turn by Call Intelligence,
        # never mutated -- CallSession stays the transport bookkeeper it is,
        # and the caller picture lives in its own frozen object.
        #
        # Today no detector exists, so this is the Appendix C "normal" row on
        # every call with caller_state="unknown". That is deliberate: it is
        # the behaviour the agent already had, so the object changes nothing
        # until something actually detects.
        self.call_state = call_state_mod.build()

        # story title: The same question gets the same answer within one call
        # user story: As a caller who asks twice, I want the same answer, so
        #   that I know which one to believe.
        # acceptance criteria: Repeating a question in one call produces an
        #   identical factual answer unless the underlying data changed, in
        #   which case the change is stated. A test asserts consistency across
        #   three repeats with an unchanged backend.
        #
        # What this caller has already been told. Per-call by construction --
        # it is created here and nothing outlives the session, which is the
        # scope the story asks for ("within one call") and also the only scope
        # that is safe: a process-wide version would compare one caller's
        # answer against another's. Never holds a reply, only the facts a
        # reply was rendered from. See agent/answer_ledger.py.
        self.answer_ledger = answer_ledger.AnswerLedger()

        # story title: A multi-part question is answered in full
        # user story: As a caller who asked two things, I want both answered,
        #   so that I do not have to ask again.
        # acceptance criteria: Every answerable part of a turn is answered in
        #   the order asked, and any part that cannot be answered is
        #   explicitly addressed rather than dropped. Completeness is scored
        #   on a labelled multi-part set.
        #
        # Parts of an earlier turn that a question got in front of, the
        # utterance they came from, and how many turns they have waited.
        # Deliberately NOT stored inside session.pending: that dict is
        # cleared on a dozen paths, every one of which would silently bin a
        # question the caller actually asked.
        self.deferred: dict | None = None

        # Consecutive turns that were echoed back for confirmation because the
        # decoders disagreed. Reset by any turn that proceeds normally. Capped
        # so a caller on a bad line is offered a human instead of being asked
        # "did I hear you right?" indefinitely -- a repair ladder that never
        # ends is a D-grade outcome even though every individual turn is safe.
        self.confirm_attempts = 0

    def hold_gate_for(self, audio_duration_s: float):
        """Called before each reply goes out. Extends rather than replaces
        the deadline: replies queue on the client, so a second clip starts
        playing only after the first finishes."""
        base = max(self.speak_deadline, time.time()) if self.agent_speaking else time.time()
        self.agent_speaking = True
        self.speak_deadline = base + audio_duration_s + PLAYBACK_GUARD_S

    def release_gate(self):
        """Playback is over. Don't touch processed_until_s here -- the poll
        loop owns the decoded buffer and does the resync on its next tick."""
        self.agent_speaking = False
        self.resync_pending = True

    async def append(self, chunk: bytes):
        self.last_activity = time.time()
        with open(self.raw_path, "ab") as f:
            f.write(chunk)

    async def send_json(self, sender: str, text: str):
        await self.ws.send_text(json.dumps({"sender": sender, "text": text}, ensure_ascii=False))

    async def send_audio(self, wav_bytes: bytes):
        if wav_bytes:
            await self.ws.send_bytes(wav_bytes)

    def cleanup(self):
        import shutil
        with contextlib.suppress(OSError):
            shutil.rmtree(self.tmpdir, ignore_errors=True)


# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
async def _speak(session: CallSession, text_bn: str, fallback_reason: str | None = None):
    """Say one line to the caller, or say why it could not be said.

    SYNTHESIZE FIRST, THEN SEND THE TRANSCRIPT.
    This used to push text_bn to the browser before synthesis, which read as
    snappier -- the line appeared while the vocoder was still working. It also
    meant the pane was a record of what the agent INTENDED to say. Once a reply
    can be blocked and replaced, that stops being a harmless discrepancy: the
    transcript would show the answer while the caller heard a referral to the
    counter, and the one artefact anyone would check afterwards would disagree
    with the call. The cost is that the text now appears with the audio instead
    of a beat before it; correctness of the record wins.
    """
    wav = None
    try:
        # speech_rate is read from the call state rather than hardcoded. This
        # is the one place the object currently changes what a caller HEARS,
        # and it resolves to "default" until a detector sets caller_state or
        # senior -- so it is a no-op today by design, not by accident.
        # language likewise: always None until a detector sets it, and threaded
        # through because the speakability rule is Bengali-only.
        wav = await _tts.synthesize(text_bn,
                                    speech_rate=session.call_state.speech_rate,
                                    language=session.call_state.language)
    # STORY [Answer Quality and Grounding]
    # As a patient, I want to hear the whole sentence, so that I am
    # not left guessing what the agent tried to say.
    except UnspeakableReply as e:
        # A hole was found in a reply we composed. Caught BEFORE the generic
        # handler below on purpose: that one answers with the "system busy"
        # clip, which is right when the vocoder is down and wrong here. This
        # is a defect on our side, not an outage, and the honest response is
        # to send the caller somewhere that can actually answer them.
        logger.error("[%s] reply blocked, dropped=%s -- escalating to counter",
                     session.call_id, list(e.dropped))
        turn_log.record_unspeakable(session.call_id, session.utt_seq,
                                    e.dropped, enforced=True)
        text_bn = UNSPEAKABLE_ESCALATION
        try:
            # Cannot recurse: this line is in PREWARM_LINES and startup asserts
            # every one of them is speakable, so it can never be blocked itself.
            # The broad catch is the belt to that braces -- if it somehow were,
            # the caller still gets the pre-recorded clip rather than silence.
            wav = await _tts.synthesize(text_bn,
                                        speech_rate=session.call_state.speech_rate)
        except Exception:  # noqa: BLE001 - last resort, never raise past here
            logger.exception("[%s] escalation line failed to synthesize", session.call_id)
            wav = _tts.fallback_audio("tool_failure")
    except Exception as e:  # noqa: BLE001 - TTS is the last mile, must not raise past here
        logger.warning("[%s] TTS failed (%s) -- using fallback audio", session.call_id, e)
        wav = _tts.fallback_audio(fallback_reason or "tts_failure")

    # STORY [Answer Quality and Grounding]
    # As a patient, I want to hear the whole sentence, so that I am
    # not left guessing what the agent tried to say.
    if not SPEAKABILITY_ENFORCE:
        # SHADOW MODE. synthesize() counted and logged this, but only the
        # orchestrator knows the call and turn, so the exported row is written
        # here. Guarded so it cannot double up with the enforced branch above:
        # exactly one of the two runs.
        #
        # This whole block is temporary. It exists to answer one question --
        # how often would real traffic have been blocked -- and comes out when
        # SPEAKABILITY_ENFORCE becomes the default.
        _verdict = speakability.check(text_bn, language=session.call_state.language)
        if _verdict.is_blocked:
            turn_log.record_unspeakable(session.call_id, session.utt_seq,
                                        _verdict.dropped, enforced=False)

    await session.send_json("AI", text_bn)

    # Close the gate BEFORE the bytes leave, never after: the client can
    # start playing the moment they land, and a poll tick that slips in
    # between send and gate is exactly the echo this prevents.
    session.hold_gate_for(_wav_duration_s(wav))
    await session.send_audio(wav)


# story title: The same question gets the same answer within one call
# user story: As a caller who asks twice, I want the same answer, so that I
#   know which one to believe.
# acceptance criteria: Repeating a question in one call produces an identical
#   factual answer unless the underlying data changed, in which case the
#   change is stated. A test asserts consistency across three repeats with an
#   unchanged backend.
#
# WHY A WRAPPER AND NOT A LINE INSIDE _speak
# ------------------------------------------
# _speak knows the sentence and nothing else -- not which intent produced it,
# not which clinic response it was rendered from -- and both are needed to
# decide whether this answer contradicts an earlier one. Passing them into
# _speak would push tool responses into the transport layer for the sake of
# one caller in eight.
#
# WHY A WRAPPER AND NOT A LINE AT EACH CALL SITE
# ----------------------------------------------
# There are EIGHT places that speak a factual reply: three in
# _dispatch_turn_inner and five in _continue_pending. Eight places to remember
# is eight places to eventually forget, and the guarantee would then hold on
# the routes someone happened to think about. Same argument tool_outcome.py
# makes about its twelve, and tests/test_answer_consistency.py enforces it
# statically: a reply template handed straight to _speak fails the build.
# story title: Near matches are offered rather than guessed or refused
# user story: As a caller naming something loosely, I want the close matches
#   offered, so that I am not told my test does not exist when it does.
# acceptance criteria: When several catalogue rows fall within the match band
#   the agent offers up to three by name and asks which. Candidates are
#   generated across every supported language and romanised spelling. The
#   did-you-mean path covers the ambiguous case and not only total failure.
async def _offer_near_matches(session: CallSession, intent: str, result: dict,
                              offered_date: str | None) -> bool:
    """-> True if this response was an ambiguity and the turn is now finished.

    An ambiguous response is a QUESTION the clinic asked back, so nothing
    factual is spoken and nothing is recorded as an answer. The caller's
    reply lands in _continue_pending's entity_choice state, which re-runs the
    lookup against the canonical name they chose.

    `offered_date` is carried through so the second lookup asks about the
    same day as the first. Without it, "is Dr Sen in on Tuesday" answered
    with a choice, then resolved, would silently become a question about
    today.

    An EMPTY candidate list is not a bug: clinic-api sends one when several
    rows matched and at least one of them has no Bengali alias, because
    offering a partial list would be a guess wearing a question mark. There
    is nothing to choose from, so no choice state is opened -- the caller is
    asked to name it again and the next turn starts clean.
    """
    if not isinstance(result, dict) or not result.get("ambiguous"):
        return False

    candidates = result.get("candidates") or []
    logger.info("[%s] %s ambiguous (%d candidate(s)) -- offering instead of guessing",
                session.call_id, intent, len(candidates))

    session.pending = {
        "awaiting": "entity_choice", "intent": intent, "slots": {},
        "candidates": candidates, "offered_date": offered_date, "retries": 0,
    } if candidates else None

    await _speak(session, near_match_prompt(candidates))
    return True


async def _speak_fact(session: CallSession, intent: str, slots: dict,
                      result: dict, reply: str,
                      offered_date: str | None = None) -> bool:
    """Speak a factual reply, saying so if it contradicts an earlier one.

    `reply` is always rendered FRESH by the caller from a live clinic
    response. Nothing here substitutes a remembered answer -- the ledger only
    compares, and the most it can do is prepend one sentence. The
    Architecture Plan is explicit on this point ("do not optimise by caching
    replies"), and a reply cache is also the one change that would reintroduce
    the stale price this whole design avoids.

    story title: Near matches are offered rather than guessed or refused
    user story: As a caller naming something loosely, I want the close
        matches offered, so that I am not told my test does not exist when
        it does.
    acceptance criteria: When several catalogue rows fall within the match
        band the agent offers up to three by name and asks which. Candidates
        are generated across every supported language and romanised
        spelling. The did-you-mean path covers the ambiguous case and not
        only total failure.

    -> True when the response was ambiguous and an offer was spoken instead
    of the reply; every call site must then end the turn. The guard lives
    here rather than at the eight call sites for the same reason the
    consistency check does, and the return value exists because the callers
    set session.pending AFTER speaking -- an offer that set the choice state
    from in here would be overwritten a line later by the caller's own
    bookkeeping. tests/test_near_match_offers.py fails the build if a call
    site drops the guard.
    """
    if await _offer_near_matches(session, intent, result, offered_date):
        return True

    verdict, previous = session.answer_ledger.check(intent, slots, result)

    if verdict != answer_ledger.FIRST:
        _consistency["repeats"] += 1
        _consistency[verdict] += 1

    if verdict == answer_ledger.CHANGED:
        # Logged at WARNING, not INFO. On a catalogue nobody is editing this
        # should not happen, and when it does the two candidate causes -- the
        # clinic's data moved, or entity resolution landed on a different row
        # for the same words -- are told apart by whether the leading identity
        # in the two tuples matches. Facts only; no caller data reaches here.
        logger.warning("[%s] %s answer changed within the call: %s -> %s",
                       session.call_id, intent, previous,
                       answer_ledger.facts(intent, result))
        reply = with_change_notice(reply)

    await _speak(session, reply)
    return False


async def _slice_utterance(session: CallSession, start_s: float, end_s: float, seq: int) -> str:
    """Cuts [start_s, end_s+pad] -- both ABSOLUTE call-time offsets -- out
    of the call's decoded WAV into its own small file for ASR."""
    wav, sr = await asyncio.to_thread(torchaudio.load, session.wav_path)
    a = max(0, int(start_s * sr))
    b = min(int((end_s + UTTERANCE_PAD_S) * sr), wav.shape[-1])
    clip_path = f"{session.wav_path}.utt{seq}.wav"
    await asyncio.to_thread(torchaudio.save, clip_path, wav[:, a:b], sr)
    return clip_path


async def _resolve_intent(session: CallSession, text: str) -> dict:
    """Semantic cache in front of the LLM. A hit skips Ollama entirely --
    the slowest hop in the turn -- but the clinic lookup that follows still
    runs live, so a cached intent can never serve a stale price."""
    # Tier 1: decide it locally if we can. For a fixed catalogue the
    # entity is a string-matching problem with a 0.32 confidence margin,
    # where the embedding route had 0.03 -- see agent/fast_path.py. This
    # returns None whenever it is not sure, which is the common case for
    # anything except a routine price or availability question.
    if _fast_path is not None:
        hit = await asyncio.to_thread(_fast_path.resolve, text)
        if hit is not None:
            logger.info("[%s] fast path resolved %s (%.2f) -- no LLM call",
                        session.call_id, hit.intent, hit.confidence)
            return hit.as_llm_shape()

    cached, how = await asyncio.to_thread(_intent_cache.get, text)
    if cached is not None:
        logger.info("[%s] intent cache %s hit", session.call_id, how)
        return cached

    data, diag = await asyncio.to_thread(extract_intent, text)
    logger.info("[%s] intent extracted in %.2fs (%d attempt(s))",
                session.call_id, diag["total_time_s"], diag["attempts"])
    await asyncio.to_thread(_intent_cache.put, text, data)
    return data


def _next_missing(slots: dict) -> str | None:
    """-> the first still-empty field in _BOOKING_FIELDS order, or None
    once every field a booking needs is filled."""
    for field in _BOOKING_FIELDS:
        if not slots.get(field):
            return field
    return None


def _match_offered(text: str, candidates: list[dict]) -> dict | None:
    """-> the canonical `name` (the form book_appointment/get_doctor_
    availability need) of the doctor the caller just named out of a list
    session.pending offered a moment ago, or None if the utterance is not
    confidently one of them.

    Same trust model as fast_path.Catalogue.match: score every candidate
    against every spoken form (English surname AND the seeded Bengali
    alias, since the caller may answer in either script), and only commit
    above a floor rather than always taking the best of a bad field. 0.55
    is fast_path.ENTITY_MATCH_FLOOR -- reused here because the situation is
    the same shape (matching a short spoken name against a small local
    list), just with the candidate list narrowed to what was JUST spoken
    to the caller instead of the whole 74-row catalogue.
    """
    if not candidates:
        return None
    norm_text = text.strip().lower()
    if not norm_text:
        return None
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
    # The runner-up is tracked now, and the margin applies here too. This is
    # the turn AFTER an offer -- the caller has just been read two names and
    # answered -- so an answer that fits both of them equally is the one
    # place where guessing would be least forgivable: the whole point of the
    # preceding turn was that the agent had stopped guessing.
    best, best_score, runner_up = None, 0.0, 0.0
    for c in candidates:
        forms = [c["name"], c["name"].split()[-1]]
        if c.get("name_bn"):
            forms.append(c["name_bn"])
        row_best = 0.0
        for form in forms:
            if not form:
                continue
            form_l = form.lower()
            score = difflib.SequenceMatcher(None, form_l, norm_text).ratio()
            if form_l in norm_text or norm_text in form_l:
                score = max(score, 0.85)
            row_best = max(row_best, score)
        if row_best > best_score:
            best, best_score, runner_up = c, row_best, best_score
        elif row_best > runner_up:
            runner_up = row_best
    # Returns the whole candidate, not just the canonical name. The API needs
    # the English `name`; the booking readback needs `name_bn`, because it is
    # SPOKEN. Returning one and looking the other up later is what put an
    # English name into a Bengali sentence -- see booking_confirm_prompt.
    if best_score < 0.55:
        return None
    if (best_score - runner_up) < COMMIT_MARGIN:
        # Two of the offered names fit what the caller just said equally
        # well. Re-asking is the only honest move; picking one would undo
        # the turn that produced the offer.
        return None
    return best


# Bare "নাম বলছি" prefixes a caller sometimes leads a name with. Stripped
# rather than relied upon -- most callers just say the name on its own.
_NAME_PREFIXES = ("আমার নাম ", "নাম ", "আমি ")


def _clean_patient_name(text: str) -> str | None:
    t = text.strip().strip("।!?., ")
    if not t:
        return None
    for prefix in _NAME_PREFIXES:
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    return t or None


async def _finish_booking(session: CallSession, slots: dict, *, confirmed: bool = False):
    """All 5 fields are filled -- place the booking and clear pending
    regardless of outcome. Failure here is reported the same way the old
    single-shot book_appointment branch reported it (tool_failure
    fallback audio), just reachable now from either that branch OR from
    the tail of a multi-turn _continue_pending flow.

    `confirmed` must be True. This is the one irreversible thing the agent
    does, and the guard lives HERE rather than at the call sites on purpose:
    there are already two ways in (the book_appointment intent branch and the
    tail of _continue_pending), a third will eventually be added, and a guard
    that has to be remembered at each entry point is a guard that will
    eventually be forgotten at one of them. Refusing inside the function makes
    the write unreachable by omission rather than by discipline.
    """
    if not confirmed:
        # Not an error the caller caused -- most likely a new code path that
        # skipped the readback. Log it loudly, then do the safe thing rather
        # than the convenient one: ask, and write only if they say yes.
        logger.error("[%s] booking reached _finish_booking unconfirmed -- "
                     "refusing the write and asking the caller", session.call_id)
        session.pending = {
            "awaiting": "confirm_booking", "slots": slots,
            "candidates": None, "offered_date": slots.get("date"), "retries": 0,
        }
        await _speak(session, booking_confirm_prompt(slots))
        return

    session.pending = None
    try:
        result = await _tools.book_appointment(
            slots["doctor_name"], slots["date"], slots["time_slot"],
            slots["patient_name"], slots["phone"],
        )
    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        await _speak(session, SYSTEM_UNREACHABLE_BN,
                     fallback_reason="tool_failure")
        return
    # story title: The agent says it cannot confirm rather than guessing
    # user story: As a caller, I want to be told plainly when the system
    #   cannot verify something, so that I am not given a confident guess.
    # acceptance criteria: The insufficient-verified-information outcome has
    #   its own template per language, its own metric and its own escalation
    #   path, distinct from not-found and from an infrastructure apology. Its
    #   rate is reported per intent because a rise means a data or
    #   integration problem.
    #
    # THE WRITE HAPPENED. Whether the agent can say what it did is a separate
    # question, and this is where it is asked -- after the POST returned
    # success and before a single one of its values is read aloud.
    #
    # Routed to its own outcome rather than to either neighbour, because the
    # instruction to the caller is different from both: not "that does not
    # exist", not "call back", but "it IS booked, we will follow up, do not
    # rebook". Telling them to call back here is what produces the duplicate
    # appointment.
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
    # NO WRITE UNDER AMBIGUITY. This is the worst version of the bug the story
    # names: booking the higher-scoring of two plausible doctors leaves the
    # caller believing they have an appointment, and they do -- with someone
    # else. clinic-api refuses the write and hands back the candidates, so
    # this asks instead of reporting "no such doctor" for a doctor who exists
    # twice over.
    #
    # The booking state is NOT resumed automatically after the choice. The
    # caller picks a doctor, hears their availability, and walks the booking
    # again with the readback intact -- longer, and the only version where
    # every field is re-confirmed against the doctor they actually chose.
    if result.get("reason") == "doctor_ambiguous":
        candidates = result.get("candidates") or []
        logger.info("[%s] booking refused: %r matches %d doctors",
                    session.call_id, slots.get("doctor_name"), len(candidates))
        session.pending = {
            "awaiting": "entity_choice", "intent": "doctor_availability",
            "slots": {}, "candidates": candidates,
            "offered_date": slots.get("date"), "retries": 0,
        } if candidates else None
        await _speak(session, near_match_prompt(candidates))
        return

    unverified = outcomes.missing_booking_write_fields(result)
    if unverified:
        logger.error("[%s] booking write is unverifiable -- missing %s. The "
                     "appointment WAS created; the response did not carry it back.",
                     session.call_id, unverified)
        if _tools is not None:
            _tools.outcomes.record("book_appointment", tool_outcome.INSUFFICIENT)
        turn_log.record_insufficient(session.call_id, session.utt_seq,
                                     "book_appointment", unverified)
        await _speak(session, INSUFFICIENT_VERIFIED_INFORMATION_BN)
        return
    await _speak(session, booking_reply(slots, result))


async def _continue_pending(session: CallSession, text: str) -> bool:
    """The fix for "appointment pipeline breaking": every turn used to be
    classified from a bare transcript with ZERO memory of the turn before
    it (see _resolve_intent / agent/llm.py's docstring -- one utterance
    in, one classification out, nothing carried over). A caller who had
    just been asked "কোন দিন চান?" and replied "আজ" produced a fresh,
    context-free classification of the single word "আজ", which the model
    has no way to recognise as a date answer -- it almost always came
    back "unclear", and the booking that was three-quarters filled a
    moment ago silently died with no record it had ever started.

    This function is the session's memory. While session.pending is set,
    EVERY turn is routed here first (see _dispatch_turn), and is
    interpreted against exactly the one field pending["awaiting"] says was
    just asked for -- using agent/slot_parse.py's local parsers, not
    another LLM call (see that module's docstring for why a fresh
    classification is the wrong tool for a reply this short). The LLM is
    not consulted again until the flow ends, one way or another.

    pending shape: {
        "awaiting": "doctor_choice" | "department_date" | "date" | "time_slot"
                    | "patient_name" | "phone",
        "slots": {<whatever of the 5 booking fields is already known>},
        "candidates": [{"name", "name_bn"}, ...] | None,  # only for "doctor_choice"
        "offered_date": "<iso>" | None,  # the date main.py already SPOKE to
                                          # the caller ("today", or a
                                          # next-available date) -- lets a
                                          # bare "হ্যাঁ" confirm THAT date
                                          # instead of literally "today"
        "retries": int,
    }

    Returns True when the turn was fully handled here (caller must not
    also run intent extraction on top of it); False to fall through to
    the normal pipeline -- either because there was no pending flow, or
    because this one gave up on it after repeated unparseable replies.
    """
    pending = session.pending
    if pending is None:
        return False

    awaiting = pending["awaiting"]

    # Universal escape hatch, checked before any field-specific parsing:
    # a caller mid-flow who says "না" / "থাক" is abandoning the booking,
    # not answering whichever question was pending.
    # story title: Every critical value is read back before it is used
    # user story: As a patient giving a phone number, I want it read back, so
    #   that a misheard digit does not send my report to a stranger.
    # acceptance criteria: Phone numbers, dates, times and names are confirmed
    #   aloud before any write, and a rejection opens a correction path rather
    #   than repeating the prompt. Readback is mandatory regardless of
    #   confidence for values that affect a write.
    #
    # The hatch now SKIPS confirm_booking, and that exclusion is the story.
    # A "no" answering "did I get this right?" does not mean "cancel my
    # appointment" -- it means one of the five values is wrong. Letting the
    # universal hatch see it first threw the whole booking away at the exact
    # moment the caller was trying to repair it.
    #
    # confirm_correction is deliberately NOT excluded, which is a considered
    # divergence from dev_sourav, where both states skip the hatch. By the time
    # the agent has asked "which one should I fix -- doctor, date, time, name,
    # or phone?", a caller answering "no" is not naming a field; the likeliest
    # reading is that they have given up, and that is what the hatch does.
    if is_negative(text) and awaiting != "confirm_booking":
        session.pending = None
        await _speak(session, "ঠিক আছে, অ্যাপয়েন্টমেন্ট বাদ থাক। আর কিছু জানতে চান?")
        return True

    if awaiting == "confirm_booking":
        # The whole booking has been read back to the caller and this turn is
        # their answer. ONLY an explicit affirmative writes.
        #
        # The negative case never reaches here -- is_negative() above already
        # cancels the flow, which is the correct outcome for "না" / "থাক".
        # Everything that is neither yes nor no falls through to a re-ask:
        # silence, a restatement of the details, a half-heard grunt. None of
        # those are consent, and treating an ambiguous reply as one would give
        # back exactly the guess-becomes-a-booking failure this state exists to
        # prevent.
        if is_affirmative(text):
            slots = pending["slots"]
            session.pending = None
            logger.info("[%s] booking confirmed by caller", session.call_id)
            await _finish_booking(session, slots, confirmed=True)
            return True

        if is_negative(text):
            # THE CORRECTION PATH. Before this, a rejection re-asked "just say
            # yes or no" twice and then abandoned the booking: the caller said
            # something was wrong and the agent's reply was to ask the same
            # question again, then hang up on it. The criterion names that
            # exact behaviour as the thing not to do.
            #
            # Nothing is discarded -- the four correct values stay in
            # pending["slots"], and only the named one is re-collected.
            pending["awaiting"] = "confirm_correction"
            pending["retries"] = 0
            logger.info("[%s] readback rejected -- opening the correction path",
                        session.call_id)
            await _speak(session, booking_correction_prompt())
            return True

        pending["retries"] += 1
        if pending["retries"] > 2:
            # Three unclear answers to a yes/no question is a handoff, not a
            # fourth attempt. Nothing has been written, and saying so plainly
            # is a B-grade outcome; looping again would trend towards D.
            session.pending = None
            logger.info("[%s] booking abandoned -- no clear confirmation", session.call_id)
            await _speak(session, BOOKING_NOT_CONFIRMED_BN)
            return True
        await _speak(session, "শুধু বলুন — হ্যাঁ, নাকি না?")
        return True

    # story title: Every critical value is read back before it is used
    # user story: As a patient giving a phone number, I want it read back, so
    #   that a misheard digit does not send my report to a stranger.
    # acceptance criteria: Phone numbers, dates, times and names are confirmed
    #   aloud before any write, and a rejection opens a correction path rather
    #   than repeating the prompt. Readback is mandatory regardless of
    #   confidence for values that affect a write.
    #
    # The caller rejected the readback and has been asked which single value
    # is wrong. This turn is that answer.
    #
    # Re-collecting ONE field and returning to the readback is what makes this
    # a correction rather than a restart: the tail of this function fills the
    # named field, finds nothing missing, and routes straight back to
    # confirm_booking -- so the corrected booking is read back IN FULL and
    # still needs an explicit affirmative. A correction never shortens the
    # path to the write.
    if awaiting == "confirm_correction":
        field = parse_correction_field(text)
        if field is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                logger.info("[%s] correction abandoned -- no field named",
                            session.call_id)
                await _speak(session, BOOKING_NOT_CONFIRMED_BN)
                return True
            await _speak(session, booking_correction_prompt())
            return True

        logger.info("[%s] correcting %s", session.call_id, field)
        pending["awaiting"] = field
        pending["retries"] = 0
        await _speak(session, missing_slot_prompt("book_appointment", field))
        return True

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
    # The turn after an offer. The caller has been read up to three names and
    # has said one of them; this resolves which, then re-runs the SAME lookup
    # against the canonical name rather than against their words -- so the
    # second attempt cannot be ambiguous for the same reason the first was.
    #
    # Separate from doctor_choice, which looks nearly identical and is not:
    # that state follows a department LISTING, where every candidate is a
    # correct answer and the caller is choosing who to see. Here the
    # candidates are competing readings of one thing the caller already said,
    # and only one of them is what they meant.
    if awaiting == "entity_choice":
        intent = pending.get("intent")
        chosen = _match_offered(text, pending.get("candidates") or [])
        if chosen is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                # Two failed attempts at the same choice. Drop the state and
                # let a fresh classification try, rather than asking a third
                # time -- a repair ladder with no end is its own bad outcome,
                # the same cap doctor_choice uses.
                session.pending = None
                return False
            await _speak(session, NEAR_MATCH_UNCLEAR_BN)
            return True

        date_iso = pending.get("offered_date")
        session.pending = None
        logger.info("[%s] %s disambiguated to %r", session.call_id, intent,
                    chosen.get("name"))

        try:
            if intent == "test_rate":
                result = await _tools.get_test_rate(chosen["name"])
            elif intent == "doctor_availability":
                result = await _tools.get_doctor_availability(
                    chosen["name"], date_iso or datetime.date.today().isoformat())
            else:
                result = await _tools.get_doctors_by_department(chosen["name"], date_iso)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            await _speak(session, SYSTEM_UNREACHABLE_BN,
                         fallback_reason="tool_failure")
            return True

        # Three near-identical calls rather than one over a `reply` local,
        # and deliberately so: tests/test_fact_provenance.py requires the
        # sentence reaching _speak_fact to be a reply_templates CALL at the
        # call site, not a name that a local could have been reassigned to.
        # Collapsing these three would pass a variable and blind that gate --
        # which is the regression it exists to catch, so the duplication is
        # the cheaper half of the trade.
        #
        # A canonical name resolving to another ambiguity would mean two rows
        # share a name outright, a catalogue defect rather than a caller one.
        # _speak_fact would offer again and return True; session.pending was
        # cleared above, so it asks once and stops rather than looping.
        spoken_name = chosen.get("name_bn") or chosen["name"]
        if intent == "test_rate":
            asked = {"test_name": spoken_name}
            if await _speak_fact(session, intent, asked, result,
                                 test_rate_reply(asked, result),
                                 offered_date=date_iso):
                return True
        elif intent == "doctor_availability":
            asked = {"doctor_name": spoken_name}
            if await _speak_fact(session, intent, asked, result,
                                 doctor_availability_reply(asked, result),
                                 offered_date=date_iso):
                return True
        else:
            asked = {"department": spoken_name}
            if await _speak_fact(session, intent, asked, result,
                                 doctors_by_department_reply(asked, result),
                                 offered_date=date_iso):
                return True
        return True

    if awaiting == "doctor_choice":
        match = _match_offered(text, pending.get("candidates") or [])
        if match is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False  # give a fresh LLM classification a chance instead
            await _speak(session, "দুঃখিত, ডাক্তারের নামটা একটু স্পষ্ট করে বলবেন?")
            return True

        date_iso = pending.get("offered_date") or datetime.date.today().isoformat()
        try:
            result = await _tools.get_doctor_availability(match["name"], date_iso)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, SYSTEM_UNREACHABLE_BN,
                         fallback_reason="tool_failure")
            return True

        # story title: The same question gets the same answer within one call
        # user story: As a caller who asks twice, I want the same answer, so that
        #   I know which one to believe.
        # acceptance criteria: Repeating a question in one call produces an
        #   identical factual answer unless the underlying data changed, in which
        #   case the change is stated. A test asserts consistency across three
        #   repeats with an unchanged backend.
        #
        # Every factual reply goes through _speak_fact rather than _speak, so the
        # consistency check cannot be forgotten on one route. The reply itself is
        # still rendered here, fresh, from this turn's live clinic response.
        asked = {"doctor_name": match.get("name_bn") or match["name"]}
        if await _speak_fact(session, "doctor_availability", asked, result,
                             doctor_availability_reply(asked, result),
                             offered_date=date_iso):
            return True

        offered = None
        if result.get("found"):
            offered = result.get("date") if result.get("available") else result.get("next_available_date")
        if offered:
            # doctor_availability_reply() just asked "today or another
            # day" (or, if not available today, "want that next date
            # instead?") -- stay in the flow so the caller's answer to
            # THAT question is picked up as the "date" field next.
            session.pending = {
                "awaiting": "date",
                # doctor_name is the canonical English label the booking API
                # needs; doctor_name_bn is the one that gets SPOKEN in the
                # readback. Both are carried from here on -- see
                # booking_confirm_prompt for what happens when they are not.
                "slots": {
                    "doctor_name": result.get("doctor_name") or match["name"],
                    "doctor_name_bn": result.get("doctor_name_bn") or match.get("name_bn"),
                },
                "candidates": None, "offered_date": offered, "retries": 0,
            }
        else:
            session.pending = None
        return True

    if awaiting == "confirm_date":
        # story title: The model never originates a fact
        # user story: As a clinical lead, I want every price, date and identifier
        #   to come from a verified system response, so that a wrong answer is a
        #   data bug rather than a model bug.
        # acceptance criteria: Every factual sentence is a template substitution
        #   from a validated tool response and the model is never shown a figure
        #   it could restate. An automated assertion on every commit proves no
        #   model-composed span reaches synthesis on a factual intent.
        #
        # The caller said something covering several days ("আগামী সপ্তাহে"),
        # date_calc worked out which days those are, and the agent read the
        # range back. This turn is the answer.
        #
        # হ্যাঁ  -> look up the FIRST day of the range. clinic-api answers
        #          "in that day?" and, when not, "next available day" computed
        #          from its own schedule table -- which is the honest answer to
        #          "is Dr Sen in next week?" and needs no new API surface.
        # anything else -> the range was wrong, so ask for one specific day.
        #          Deliberately NOT a re-ask of the same question: the caller
        #          already said no to it once.
        start = pending.get("offered_date")
        resume = pending.get("resume_intent")
        if is_affirmative(text) and start and resume:
            if resume == "doctor_availability":
                doctor_name = pending["slots"]["doctor_name"]
                try:
                    result = await _tools.get_doctor_availability(doctor_name, start)
                except ToolCallError as e:
                    logger.error("[%s] clinic API call failed: %s", session.call_id, e)
                    session.pending = None
                    await _speak(session, SYSTEM_UNREACHABLE_BN,
                                 fallback_reason="tool_failure")
                    return True
                asked = {"doctor_name": doctor_name}
                if await _speak_fact(session, "doctor_availability", asked, result,
                                     doctor_availability_reply(asked, result),
                                     offered_date=start):
                    return True
                offered = None
                if result.get("found"):
                    offered = result.get("date") if result.get("available") else result.get("next_available_date")
                session.pending = {
                    "awaiting": "date",
                    "slots": {
                        "doctor_name": result.get("doctor_name") or doctor_name,
                        "doctor_name_bn": result.get("doctor_name_bn"),
                    },
                    "candidates": None, "offered_date": offered, "retries": 0,
                } if offered else None
                return True

            department = pending["slots"]["department"]
            try:
                result = await _tools.get_doctors_by_department(department, start)
            except ToolCallError as e:
                logger.error("[%s] clinic API call failed: %s", session.call_id, e)
                session.pending = None
                await _speak(session, SYSTEM_UNREACHABLE_BN,
                             fallback_reason="tool_failure")
                return True
            asked = {"department": department}
            if await _speak_fact(session, "doctors_by_department", asked, result,
                                 doctors_by_department_reply(asked, result),
                                 offered_date=start):
                return True
            if result.get("found") and result.get("doctors"):
                session.pending = {
                    "awaiting": "doctor_choice", "slots": {},
                    "candidates": [
                        {"name": d["name"], "name_bn": d.get("doctor_name_bn")}
                        for d in result["doctors"]
                    ],
                    "offered_date": start, "retries": 0,
                }
            else:
                session.pending = None
            return True

        ask_state = ("availability_date" if resume == "doctor_availability"
                     else "department_date")
        pending["awaiting"] = ask_state
        pending["offered_date"] = None
        pending["retries"] = 0
        await _speak(session, missing_slot_prompt(resume or "book_appointment", "date"))
        return True

    if awaiting == "availability_date":
        # story title: The model never originates a fact
        # user story: As a clinical lead, I want every price, date and identifier
        #   to come from a verified system response, so that a wrong answer is a
        #   data bug rather than a model bug.
        # acceptance criteria: Every factual sentence is a template substitution
        #   from a validated tool response and the model is never shown a figure
        #   it could restate. An automated assertion on every commit proves no
        #   model-composed span reaches synthesis on a factual intent.
        #
        # The caller said something date-shaped the local parser could not
        # resolve, so rather than state the model's guess as fact the agent
        # asked which day they meant. This is that answer. Mirrors
        # "department_date" below exactly, one intent over: parse it locally,
        # re-run the same lookup, and never fall back to the model's original
        # guess -- an unparseable answer re-asks and then gives up to a fresh
        # classification, which is the same trust model every other state here
        # uses.
        value = parse_date(text, offered_date=pending.get("offered_date"))
        if value is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("doctor_availability", "date"))
            return True

        doctor_name = pending["slots"]["doctor_name"]
        try:
            result = await _tools.get_doctor_availability(doctor_name, value)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, SYSTEM_UNREACHABLE_BN,
                         fallback_reason="tool_failure")
            return True

        asked = {"doctor_name": doctor_name}
        if await _speak_fact(session, "doctor_availability", asked, result,
                             doctor_availability_reply(asked, result),
                             offered_date=value):
            return True

        offered = None
        if result.get("found"):
            offered = result.get("date") if result.get("available") else result.get("next_available_date")
        session.pending = {
            "awaiting": "date",
            "slots": {
                "doctor_name": result.get("doctor_name") or doctor_name,
                "doctor_name_bn": result.get("doctor_name_bn"),
            },
            "candidates": None, "offered_date": offered, "retries": 0,
        } if offered else None
        return True

    if awaiting == "department_date":
        # Mirrors "doctor_choice" above, one level up: the caller was just
        # told nobody in this department sits TODAY and asked for another
        # day. Parse that reply as a date and re-run the same department
        # lookup with it, rather than dropping back to a cold LLM
        # classification of a bare date phrase (see this function's
        # docstring for why that silently loses context).
        value = parse_date(text, offered_date=pending.get("offered_date"))
        if value is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("doctors_by_department", "date"))
            return True

        department = pending["slots"]["department"]
        try:
            result = await _tools.get_doctors_by_department(department, value)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, SYSTEM_UNREACHABLE_BN,
                         fallback_reason="tool_failure")
            return True

        asked = {"department": department}
        if await _speak_fact(session, "doctors_by_department", asked, result,
                             doctors_by_department_reply(asked, result),
                             offered_date=value):
            return True

        if result.get("found") and result.get("doctors"):
            session.pending = {
                "awaiting": "doctor_choice",
                "slots": {},
                "candidates": [
                    {"name": d["name"], "name_bn": d.get("doctor_name_bn")}
                    for d in result["doctors"]
                ],
                "offered_date": value, "retries": 0,
            }
        elif result.get("found"):
            # Still nobody that day either -- stay in the same state and
            # let the caller name yet another day, capped by the shared
            # retries counter above so this cannot loop forever.
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
            else:
                pending["awaiting"] = "department_date"
        else:
            session.pending = None
        return True

    # Remaining states (date / time_slot / patient_name / phone) all share
    # the same shape: parse the ONE field awaited, fill it in, ask for the
    # next missing one or finish the booking.
    value = None
    if awaiting == "date":
        value = parse_date(text, offered_date=pending.get("offered_date"))
    elif awaiting == "time_slot":
        value = parse_time(text)
    elif awaiting == "phone":
        value = parse_phone(text)
    elif awaiting == "patient_name":
        value = _clean_patient_name(text)

    if value is None:
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None
            return False
        await _speak(session, missing_slot_prompt("book_appointment", awaiting))
        return True

    pending["slots"][awaiting] = value
    pending["retries"] = 0
    missing = _next_missing(pending["slots"])
    if missing is None:
        # Every field is filled, but nothing is written yet. Read the whole
        # thing back and wait for a yes -- see the "confirm_booking" state
        # above for why an affirmative is required rather than assumed.
        pending["awaiting"] = "confirm_booking"
        await _speak(session, booking_confirm_prompt(pending["slots"]))
        return True
    pending["awaiting"] = missing
    await _speak(session, missing_slot_prompt("book_appointment", missing))
    return True


# story title: A thing not existing is never confused with a system being down
# user story: As a caller, I want to know whether my test does not exist or the
#   system cannot be reached, so that I know whether to call back.
# acceptance criteria: The two produce different spoken sentences and different
#   metrics, and the distinction survives every refactor. This behaviour exists
#   today and gains a permanent regression case.
#
# THE THIRD OUTCOME, WHICH SHOULD NOT EXIST.
# _dispatch_turn used to be the whole turn with no outer handler, fired by
# asyncio.create_task() with nothing attached to it. Anything uncaught -- a
# KeyError in a template, a torchaudio failure slicing the clip, a bug in code
# not yet written -- became an unretrieved task exception and the caller heard
# NOTHING. Neither sentence: dead air, which this module's own docstring
# promises never happens.
#
# An unexpected exception is the SYSTEM failing, so it maps to the
# system-unreachable side of the distinction this story is about. It is never
# "your test does not exist": we do not know that, and telling a caller to stop
# asking because our code raised would be the exact confusion the story names.
# story title: A multi-part question is answered in full
# user story: As a caller who asked two things, I want both answered, so that
#   I do not have to ask again.
# acceptance criteria: Every answerable part of a turn is answered in the
#   order asked, and any part that cannot be answered is explicitly addressed
#   rather than dropped. Completeness is scored on a labelled multi-part set.
#
# How long a deferred part may wait. Two turns is one clarification plus its
# answer; past that the caller has moved on, and reviving a question they
# asked four turns ago reads as the agent losing the thread rather than
# keeping it. The queue is never silently binned -- see _drain_deferred.
MAX_DEFERRED_TURNS = 2


# story title: A multi-part question is answered in full
# user story: As a caller who asked two things, I want both answered, so that
#   I do not have to ask again.
# acceptance criteria: Every answerable part of a turn is answered in the
#   order asked, and any part that cannot be answered is explicitly addressed
#   rather than dropped. Completeness is scored on a labelled multi-part set.
async def _run_parts(session: CallSession, text: str, parts: list[dict]) -> bool:
    """Answer the parts in the order the caller asked them.

    -> True if a queue was created on THIS turn, which tells the caller not
    to immediately try to drain it.

    TWO RULES CARRY THIS FUNCTION.

    STOP AT THE FIRST INTERACTIVE PART. Not "skip it and do the rest": the
    criterion says in the order asked, and answering the second question
    while the first is still waiting on a clarification reorders the
    conversation from the caller's side. Whatever follows is deferred, which
    is a promise -- _drain_deferred keeps it.

    DISCARD A SOFT CONTINUATION WHEN ANOTHER PART FOLLOWS. Several answers
    leave session.pending set opportunistically: doctor_availability ends by
    asking "today or another day?" and stays in the flow so a bare date is
    understood next turn. That convenience belongs to the LAST thing said. If
    part two is about to speak, part one's open flow would catch the caller's
    reply to a question they have already stopped thinking about -- so it is
    dropped, deliberately, rather than left to misread the next utterance.
    """
    for index, part in enumerate(parts):
        remaining = parts[index + 1:]
        outcome = await _answer_part(session, text, part)

        if outcome == INTERACTIVE:
            if remaining:
                session.deferred = {"parts": remaining, "text": text, "age": 0}
                logger.info("[%s] deferring %d part(s) behind a question",
                            session.call_id, len(remaining))
                await _speak(session, DEFERRED_PART_BN)
                return True
            return False

        # ANSWERED or UNANSWERABLE: carry on. UNANSWERABLE has already said
        # something about itself inside _answer_part -- that is the whole
        # point of it being a third outcome rather than a silent skip.
        if remaining and session.pending is not None:
            session.pending = None

    return False


# story title: A multi-part question is answered in full
# user story: As a caller who asked two things, I want both answered, so that
#   I do not have to ask again.
# acceptance criteria: Every answerable part of a turn is answered in the
#   order asked, and any part that cannot be answered is explicitly addressed
#   rather than dropped. Completeness is scored on a labelled multi-part set.
async def _drain_deferred(session: CallSession, text: str) -> None:
    """Answer what was put aside, once the question in front of it is done.

    A queue that is only ever created is not a deferral, it is a drop with
    better manners. This is the half that makes DEFERRED_PART_BN a true
    sentence.
    """
    queue = session.deferred
    if not queue:
        return

    if session.pending is not None:
        # Still mid-flow. Wait, but not forever.
        queue["age"] += 1
        if queue["age"] <= MAX_DEFERRED_TURNS:
            return
        session.deferred = None
        logger.info("[%s] dropping %d deferred part(s) after %d turns",
                    session.call_id, len(queue["parts"]), queue["age"])
        # Said, not silently binned. The caller asked; they are owed the
        # information that it went unanswered even when the answer is that
        # too much has happened since.
        for part in queue["parts"]:
            await _speak(session, unanswered_part_prompt(turn_parts.subject_of(part)))
        return

    session.deferred = None
    logger.info("[%s] resuming %d deferred part(s)", session.call_id, len(queue["parts"]))
    await _speak(session, RESUMING_PART_BN)
    # The ORIGINAL utterance, not this turn's. date_calc and slot_parse read
    # the raw words, and "আগামীকাল" said once for two questions means it for
    # both -- re-resolving the second part against the caller's answer to the
    # first ("হ্যাঁ") would lose the day entirely.
    await _run_parts(session, queue["text"], queue["parts"])


# story title: A multi-part question is answered in full
# user story: As a caller who asked two things, I want both answered, so that
#   I do not have to ask again.
# acceptance criteria: Every answerable part of a turn is answered in the
#   order asked, and any part that cannot be answered is explicitly addressed
#   rather than dropped. Completeness is scored on a labelled multi-part set.
#
# LIFTED VERBATIM out of _dispatch_turn_inner, which is why it reads like a
# chain rather than like a function: every branch, comment and ordering
# decision below predates this story and none of them changed. What changed
# is that each of the thirteen `return` statements -- each of which used to
# mean "the turn is over" -- now names WHICH KIND of over it was, so a caller
# with a second question can be told apart from a caller who is owed an
# answer to the first.
#
# `text` is still passed in whole, not just the part: date_calc.resolve and
# slot_parse's parsers read the raw utterance, and a caller who says "আগামীকাল"
# once for two questions means it for both.
async def _answer_part(session: CallSession, text: str, part: dict) -> str:
    """Answer one part of a turn. -> ANSWERED | INTERACTIVE | UNANSWERABLE."""
    intent = part["intent"]
    slots = part["slots"]

    if intent == "smalltalk":
        await _speak(session, part.get("direct_reply_bn") or "নমস্কার, কী সাহায্য করতে পারি?")
        return ANSWERED

    if intent == "unclear":
        await _speak(session, "দুঃখিত, বুঝতে পারিনি। আবার একটু বলবেন?")
        return UNANSWERABLE

    try:
        if intent == "test_rate":
            if not slots.get("test_name"):
                await _speak(session, missing_slot_prompt(intent, "test_name"))
                return INTERACTIVE
            result = await _tools.get_test_rate(slots["test_name"])
            # story title: The same question gets the same answer within one call
            # user story: As a caller who asks twice, I want the same answer, so that
            #   I know which one to believe.
            # acceptance criteria: Repeating a question in one call produces an
            #   identical factual answer unless the underlying data changed, in which
            #   case the change is stated. A test asserts consistency across three
            #   repeats with an unchanged backend.
            #
            # Every factual reply goes through _speak_fact rather than _speak, so the
            # consistency check cannot be forgotten on one route. The reply itself is
            # still rendered here, fresh, from this turn's live clinic response.
            if await _speak_fact(session, intent, slots, result,
                                 test_rate_reply(slots, result)):
                return INTERACTIVE

        elif intent == "doctor_availability":
            if not slots.get("doctor_name"):
                await _speak(session, missing_slot_prompt(intent, "doctor_name"))
                return INTERACTIVE
            # Default to TODAY, not "whenever next available": a bare
            # "ডাক্তার সেন আছেন?" with no date mentioned is a caller
            # asking about right now, and the reply text below already
            # said " আজ" (today) for exactly this case -- the old code
            # passed date=None through to the API, which answers a
            # different question ("when next"), so a doctor who simply
            # wasn't in today got reported by their NEXT sitting date
            # instead of "not today, but they're on Tuesdays" etc.
            # story title: The model never originates a fact
            # user story: As a clinical lead, I want every price, date and
            #   identifier to come from a verified system response, so that
            #   a wrong answer is a data bug rather than a model bug.
            # acceptance criteria: Every factual sentence is a template
            #   substitution from a validated tool response and the model is
            #   never shown a figure it could restate. An automated
            #   assertion on every commit proves no model-composed span
            #   reaches synthesis on a factual intent.
            #
            # The model said what the caller MEANT; date_calc did the
            # calendar. Four outcomes, and they are genuinely different:
            #
            #   range      -> say which days it computed and ask. Safe to
            #                 read the dates out loud precisely because
            #                 code produced them.
            #   unmapped   -> the caller named a day nothing could express.
            #                 Ask which one. Never today: that was the old
            #                 silent wrong answer.
            #   single day -> answer it.
            #   absent     -> no day was mentioned; today is the question
            #                 the caller actually asked.
            span = date_calc.resolve(text, slots.get("date_expr"))
            if span.needs_confirmation:
                logger.info("[%s] %s -> %s..%s, confirming the range",
                            session.call_id, span.expression, span.start, span.end)
                session.pending = {
                    "awaiting": "confirm_date",
                    "slots": {"doctor_name": slots["doctor_name"]},
                    "candidates": None, "offered_date": span.start, "retries": 0,
                    "resume_intent": "doctor_availability", "span_end": span.end,
                }
                await _speak(session, date_range_confirm_prompt(span.start, span.end))
                return INTERACTIVE
            if span.source == date_calc.SOURCE_UNMAPPED:
                logger.info("[%s] caller named a day the vocabulary cannot express "
                            "-- asking instead of assuming", session.call_id)
                session.pending = {
                    "awaiting": "availability_date",
                    "slots": {"doctor_name": slots["doctor_name"]},
                    "candidates": None, "offered_date": None, "retries": 0,
                }
                await _speak(session, missing_slot_prompt(intent, "date"))
                return INTERACTIVE
            if span.source == SOURCE_INTERPRETED:
                logger.info("[%s] date interpreted: %s -> %s",
                            session.call_id, span.expression, span.start)
            date_iso = span.start or datetime.date.today().isoformat()
            result = await _tools.get_doctor_availability(slots["doctor_name"], date_iso)
            if await _speak_fact(session, intent, slots, result,
                                 doctor_availability_reply(slots, result),
                                 offered_date=date_iso):
                return INTERACTIVE

            # Keep the flow open for "yes, book that day" / "another
            # day" -- doctor_availability_reply() just asked exactly
            # that question. See _continue_pending's "date" state.
            offered = None
            if result.get("found"):
                offered = result.get("date") if result.get("available") else result.get("next_available_date")
            session.pending = {
                "awaiting": "date",
                "slots": {
                    "doctor_name": result.get("doctor_name") or slots["doctor_name"],
                    "doctor_name_bn": result.get("doctor_name_bn"),
                },
                "candidates": None, "offered_date": offered, "retries": 0,
            } if offered else None

        elif intent == "doctors_by_department":
            if not slots.get("department"):
                await _speak(session, missing_slot_prompt(intent, "department"))
                return INTERACTIVE
            # Default to TODAY when the caller didn't name a date, same
            # reasoning as doctor_availability above: "অর্থোতে কারা
            # আছেন" (who's in ortho) is almost always asking who is
            # actually in the chamber right now, not for a roster of
            # every doctor the department has ever employed regardless
            # of whether they sit this week. Only an EXPLICIT date
            # bypasses this (used as-is below).
            # story title: The model never originates a fact
            # user story: As a clinical lead, I want every price, date and
            #   identifier to come from a verified system response, so that
            #   a wrong answer is a data bug rather than a model bug.
            # acceptance criteria: Every factual sentence is a template
            #   substitution from a validated tool response and the model is
            #   never shown a figure it could restate. An automated
            #   assertion on every commit proves no model-composed span
            #   reaches synthesis on a factual intent.
            #
            # The model said what the caller MEANT; date_calc did the
            # calendar. Four outcomes, and they are genuinely different:
            #
            #   range      -> say which days it computed and ask. Safe to
            #                 read the dates out loud precisely because
            #                 code produced them.
            #   unmapped   -> the caller named a day nothing could express.
            #                 Ask which one. Never today: that was the old
            #                 silent wrong answer.
            #   single day -> answer it.
            #   absent     -> no day was mentioned; today is the question
            #                 the caller actually asked.
            span = date_calc.resolve(text, slots.get("date_expr"))
            if span.needs_confirmation:
                logger.info("[%s] %s -> %s..%s, confirming the range",
                            session.call_id, span.expression, span.start, span.end)
                session.pending = {
                    "awaiting": "confirm_date",
                    "slots": {"department": slots["department"]},
                    "candidates": None, "offered_date": span.start, "retries": 0,
                    "resume_intent": "doctors_by_department", "span_end": span.end,
                }
                await _speak(session, date_range_confirm_prompt(span.start, span.end))
                return INTERACTIVE
            if span.source == date_calc.SOURCE_UNMAPPED:
                logger.info("[%s] caller named a day the vocabulary cannot express "
                            "-- asking instead of assuming", session.call_id)
                session.pending = {
                    "awaiting": "department_date",
                    "slots": {"department": slots["department"]},
                    "candidates": None, "offered_date": None, "retries": 0,
                }
                await _speak(session, missing_slot_prompt(intent, "date"))
                return INTERACTIVE
            if span.source == SOURCE_INTERPRETED:
                logger.info("[%s] date interpreted: %s -> %s",
                            session.call_id, span.expression, span.start)
            date_iso = span.start or datetime.date.today().isoformat()
            result = await _tools.get_doctors_by_department(slots["department"], date_iso)
            if await _speak_fact(session, intent, slots, result,
                                 doctors_by_department_reply(slots, result),
                                 offered_date=date_iso):
                return INTERACTIVE

            # Continue straight into booking: offer the doctors just
            # listed as candidates, so the caller's very next utterance
            # -- which may be nothing but a bare doctor name -- is
            # matched against THIS list rather than sent to the LLM with
            # no context to interpret it against. See _continue_pending's
            # "doctor_choice" state.
            if result.get("found") and result.get("doctors"):
                session.pending = {
                    "awaiting": "doctor_choice",
                    "slots": {},
                    "candidates": [
                        {"name": d["name"], "name_bn": d.get("doctor_name_bn")}
                        for d in result["doctors"]
                    ],
                    "offered_date": date_iso,
                    "retries": 0,
                }
            elif result.get("found"):
                # Department exists but nobody sits that day --
                # doctors_by_department_reply() just told the caller
                # exactly that and invited another day ("অন্য কোনো
                # দিনের কথা জিজ্ঞেস করতে পারেন"). Stay in the flow so the
                # caller's next utterance is interpreted as THAT date
                # instead of needing to restate the whole department
                # question from scratch -- see _continue_pending's
                # "department_date" state.
                session.pending = {
                    "awaiting": "department_date",
                    "slots": {"department": slots["department"]},
                    "candidates": None, "offered_date": None, "retries": 0,
                }
            else:
                session.pending = None

        elif intent == "book_appointment":
            # Merge onto whatever session.pending already knows (e.g. a
            # doctor_name carried over from a doctor_availability or
            # doctors_by_department turn moments ago) rather than
            # requiring every field in one utterance -- that all-or-
            # nothing check was the other half of "pipeline breaking":
            # a caller who gave the doctor and date in one sentence and
            # the time in the next used to have the doctor/date silently
            # discarded the moment ANY field was still missing.
            merged = dict(session.pending["slots"]) if session.pending else {}
            for field in _BOOKING_FIELDS:
                if slots.get(field):
                    merged[field] = slots[field]

            # story title: The model never originates a fact
            # user story: As a clinical lead, I want every price, date and identifier
            #   to come from a verified system response, so that a wrong answer is a
            #   data bug rather than a model bug.
            # acceptance criteria: Every factual sentence is a template substitution
            #   from a validated tool response and the model is never shown a figure
            #   it could restate. An automated assertion on every commit proves no
            #   model-composed span reaches synthesis on a factual intent.
            #
            # The booking path is the one place that ALREADY had a
            # verifier: every field below is read back in full and an
            # explicit হ্যাঁ is required before _finish_booking writes
            # anything, so a mis-resolved date here is caught by the one
            # party who knows what "কাল" meant. That readback is not
            # touched by this story and must not be weakened by it.
            #
            # What this adds is removing the model from the loop wherever
            # a deterministic parser can do the same job on the same
            # words -- a date, a time and a phone number are all things
            # slot_parse.py resolves in code. Only fields the model
            # claimed from THIS utterance are corrected; a value carried
            # over from an earlier turn was already parsed locally by
            # _continue_pending and must not be re-derived from a
            # transcript that no longer mentions it.
            for field, parser in (("time_slot", parse_time), ("phone", parse_phone)):
                if not slots.get(field):
                    continue
                parsed = parser(text)
                if parsed and parsed != merged.get(field):
                    logger.warning("[%s] %s disagreement: model=%s parsed=%s -- using parsed",
                                   session.call_id, field, merged.get(field), parsed)
                    merged[field] = parsed

            # The date is not merged from `slots` at all any more, because
            # `slots["date"]` now holds the caller's WORDS ("১৫ তারিখ"),
            # not a calendar date -- llm.py stopped producing those. Only a
            # value date_calc computed may be stored, or the API would be
            # handed a Bengali phrase and _next_missing() would report the
            # field as filled while holding something unusable.
            #
            # A RANGE is dropped rather than confirmed here: a booking is
            # one slot on one day, so "আগামী সপ্তাহে অ্যাপয়েন্টমেন্ট চাই"
            # has to become a specific day, and leaving the field empty
            # makes _next_missing() ask for exactly that. The range
            # confirmation belongs to the two read-only intents, which can
            # actually answer about a span.
            if slots.get("date") or slots.get("date_expr"):
                span = date_calc.resolve(text, slots.get("date_expr"))
                if span.start and not span.is_range:
                    merged["date"] = span.start
                else:
                    merged.pop("date", None)

            missing = _next_missing(merged)
            if missing is None:
                # Everything arrived in one utterance. That is the case
                # MOST in need of a readback, not least: five fields pulled
                # from a single sentence of phone audio is where a
                # mishearing is likeliest and least visible. Route it
                # through the same confirmation state as the slow path.
                session.pending = {
                    "awaiting": "confirm_booking", "slots": merged,
                    "candidates": None,
                    "offered_date": merged.get("date"), "retries": 0,
                }
                await _speak(session, booking_confirm_prompt(merged))
                return INTERACTIVE

            session.pending = {
                "awaiting": missing, "slots": merged, "candidates": None,
                "offered_date": (session.pending or {}).get("offered_date"), "retries": 0,
            }
            await _speak(session, missing_slot_prompt(intent, missing))
            return INTERACTIVE

    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        await _speak(session, SYSTEM_UNREACHABLE_BN,
                     fallback_reason="tool_failure")
        return UNANSWERABLE

    return ANSWERED


async def _dispatch_turn(session: CallSession, utterance_wav: str):
    global _turn_crashes
    try:
        await _dispatch_turn_inner(session, utterance_wav)
    except Exception:  # noqa: BLE001 - a turn must not die silently
        _turn_crashes += 1
        logger.exception("[%s] turn crashed -- answering as unreachable", session.call_id)
        with contextlib.suppress(Exception):
            await _speak(session, SYSTEM_UNREACHABLE_BN, fallback_reason="tool_failure")


async def _dispatch_turn_inner(session: CallSession, utterance_wav: str):
    """One full turn: ASR -> intent -> tool -> templated reply -> TTS.
    Serialized per-call via session.dispatch_lock so replies never
    interleave, even if the caller starts talking again immediately."""
    async with session.dispatch_lock:
        try:
            asr_result = await _asr.transcribe_utterance(utterance_wav)
        finally:
            with contextlib.suppress(OSError):
                os.remove(utterance_wav)

        text = asr_result.text.strip()
        if not text:
            logger.info("[%s] ASR returned empty text", session.call_id)
            await _speak(session, "দুঃখিত, শুনতে পাইনি। আবার বলবেন?", fallback_reason="asr_empty")
            return
        await session.send_json("User", text)

        # How much did the two decoders agree about what was said? Logged on
        # every turn -- not for debugging, but because the floors in
        # agent/confidence.py are REASONED and cannot become measured until
        # there is a body of these lines to sweep a threshold against. No
        # transcript is logged; only the score, the decoder and the zone.
        turn_zone = confidence.zone(asr_result)
        # "n/a" rather than a number when the decoders were not compared --
        # %.2f would raise on None, and printing 0.00 there would be the same
        # lie the sentinel used to tell.
        _agree = asr_result.decoder_agreement
        logger.info("[%s] asr agreement=%s decoder=%s words=%d/%d zone=%s",
                    session.call_id,
                    "n/a" if _agree is None else f"{_agree:.2f}",
                    asr_result.decoder_used,
                    asr_result.ctc_words, asr_result.rnnt_words, turn_zone)
        # Structured export for the correlation study. Signal and join key
        # only -- never the transcript. See agent/turn_log.py.
        turn_log.record(session.call_id, session.utt_seq, asr_result, turn_zone,
                        call_state=session.call_state)

        if turn_zone == confidence.REJECT:
            # Both decoders produced text and disagreed about nearly all of it.
            # Acting on either version is a guess, so the turn buys nothing and
            # is not worth reading back either -- ask again instead. Any booking
            # in progress is left intact: the caller has not withdrawn it, this
            # one utterance was simply not understood.
            logger.info("[%s] turn rejected on low decoder agreement", session.call_id)
            await _speak(session, "দুঃখিত, ভালো করে শুনতে পাইনি। আরেকবার বলবেন?")
            return

        # ---- answer to a "did I hear you right?" echo -----------------------
        # Handled HERE rather than in _continue_pending, and before its
        # universal is_negative() escape, because "না" means two different
        # things in the two places. In a booking flow it abandons the booking;
        # here it means "you misheard me", and must leave any booking in
        # progress exactly as it was.
        resumed_from_confirm = False
        if session.pending and session.pending.get("awaiting") == "confirm_transcript":
            echoed = session.pending
            session.pending = echoed.get("resume")   # put the real flow back
            if is_affirmative(text):
                # Confirmed. Continue this turn with what was originally heard,
                # not with the word "yes".
                logger.info("[%s] caller confirmed the transcript", session.call_id)
                text = echoed["heard"]
                resumed_from_confirm = True
            elif is_negative(text):
                logger.info("[%s] caller rejected the transcript", session.call_id)
                await _speak(session, "ঠিক আছে, আরেকবার বলবেন?")
                return
            else:
                # Neither yes nor no -- callers usually just say the thing
                # again rather than answering. Treat this utterance as a fresh
                # turn: it carries its own confidence score and gets judged on
                # its own merits below.
                logger.info("[%s] transcript echo answered with a restatement",
                            session.call_id)

        # ---- criterion 1: below the floor, check before acting -------------
        # The decoders disagreed enough that acting on this transcript would be
        # a guess. Reads are included, not just writes: a wrong price spoken
        # confidently is the same class of failure as a wrong booking, and
        # inside a booking flow this is the ONLY place a misheard field is
        # caught -- the final readback faithfully reads back whatever was
        # captured, so a name misheard three turns earlier is confirmed by a
        # caller who hears their own answer echoed correctly.
        skip_zones = ("confirm_transcript", "confirm_booking")
        already_confirming = bool(session.pending) and \
            session.pending.get("awaiting") in skip_zones
        if turn_zone == confidence.CONFIRM and not resumed_from_confirm \
                and not already_confirming:
            session.confirm_attempts += 1
            if session.confirm_attempts > 2:
                # Three in a row means the line, not the utterance, is the
                # problem. Offer a human rather than ask a fourth time.
                logger.info("[%s] repeated low-agreement turns -- offering handoff",
                            session.call_id)
                session.confirm_attempts = 0
                await _speak(session, "লাইনটা পরিষ্কার শোনা যাচ্ছে না। "
                                      "কাউন্টারে একবার কথা বলে নিলে ভালো হয়।")
                return
            logger.info("[%s] echoing transcript for confirmation (attempt %d)",
                        session.call_id, session.confirm_attempts)
            session.pending = {
                "awaiting": "confirm_transcript", "slots": {}, "candidates": None,
                "offered_date": None, "retries": 0,
                "heard": text,             # replayed verbatim once confirmed
                "resume": session.pending,  # the flow this interrupted
            }
            await _speak(session, heard_confirm_prompt(text))
            return

        # The turn is trusted from here on.
        session.confirm_attempts = 0

        # A booking (or the doctor-choice / date-confirm step just before
        # one) already in progress owns this turn -- see _continue_pending's
        # docstring for why intent extraction must NOT also run on top of it.
        if await _continue_pending(session, text):
            # story title: A multi-part question is answered in full
            # The flow that was holding the turn may have just finished, and
            # something the caller asked before it is still waiting.
            await _drain_deferred(session, text)
            return

        try:
            data = await _resolve_intent(session, text)
        except ExtractionError as e:
            logger.error("[%s] intent extraction failed: %s", session.call_id, e)
            await _speak(session, "একটু সমস্যা হচ্ছে, একটু ধরুন।", fallback_reason="llm_failure")
            return

        # story title: A multi-part question is answered in full
        # user story: As a caller who asked two things, I want both answered,
        #   so that I do not have to ask again.
        # acceptance criteria: Every answerable part of a turn is answered in
        #   the order asked, and any part that cannot be answered is
        #   explicitly addressed rather than dropped. Completeness is scored
        #   on a labelled multi-part set.
        #
        # normalise() always returns at least one part and guarantees that
        # parts[0] is the top-level intent and slots -- so a model that never
        # emits `parts` produces exactly the single-part turn this agent had
        # before the story, through the same code path.
        parts = turn_parts.normalise(data)
        if turn_parts.is_multi(parts):
            logger.info("[%s] %d-part turn: %s", session.call_id, len(parts),
                        [p["intent"] for p in parts])

        if not await _run_parts(session, text, parts):
            await _drain_deferred(session, text)


async def _resync_after_playback(session: CallSession) -> bool:
    """Drop everything captured while the agent was talking, by moving
    processed_until_s to the current end of the decoded buffer. That region
    is muted silence from the client's side; skipping it keeps the turn
    detector from ever analysing it, and -- more importantly -- keeps
    processed_until_s anchored to real time instead of drifting a full
    reply behind, which is what made later turns surface late."""
    if not await _decode_to_wav(session.raw_path, session.wav_path):
        return False
    wav, sr = await asyncio.to_thread(torchaudio.load, session.wav_path)
    buffer_end_s = wav.shape[-1] / sr
    session.processed_until_s = max(session.processed_until_s,
                                    buffer_end_s - RESYNC_REWIND_S)
    session.resync_pending = False
    logger.info("[%s] resynced to %.2fs after playback", session.call_id, session.processed_until_s)
    return True


async def _turn_poll_loop(session: CallSession):
    """Runs for the lifetime of the call. Every POLL_INTERVAL_S, re-decodes
    the growing buffer -- ALWAYS from byte 0, since that's the only way
    the WebM container stays valid -- then asks the turn detector "is the
    caller done talking yet?" using only the slice of audio past
    session.processed_until_s (a prior turn's already-consumed audio).
    On yes: slice that utterance out for ASR, hand it to _dispatch_turn as
    a background task (so ingestion of the NEXT turn's audio is never
    blocked by this turn's ASR/LLM/TTS work), and advance the marker.
    """
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)

        if time.time() - session.last_activity > IDLE_TIMEOUT_S:
            logger.info("[%s] idle timeout, closing", session.call_id)
            await _speak(session, "লাইনে কোনো সাড়া পাচ্ছি না, কল শেষ করছি। ধন্যবাদ।")
            with contextlib.suppress(Exception):
                await session.ws.close()
            return

        if time.time() - session.last_heartbeat > HEARTBEAT_INTERVAL_S:
            session.last_heartbeat = time.time()
            with contextlib.suppress(Exception):
                await session.ws.send_text('{"sender":"_ping","text":""}')

        # --- half-duplex gate: never run turn detection on our own voice ---
        if session.agent_speaking:
            if time.time() < session.speak_deadline:
                continue
            logger.warning("[%s] no playback-done from client, releasing gate on deadline",
                           session.call_id)
            session.release_gate()

        if session.resync_pending:
            await _resync_after_playback(session)
            continue

        if not await _decode_to_wav(session.raw_path, session.wav_path):
            continue  # too little data yet to form a valid container -- not an error

        wav, sr = await asyncio.to_thread(torchaudio.load, session.wav_path)
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)

        tail_start_sample = min(int(session.processed_until_s * sr), wav.shape[-1])
        tail = wav[tail_start_sample:]

        result = await asyncio.to_thread(_turn_detector.poll, tail, sr)
        if result.utterance_end_s is None:
            continue

        absolute_end_s = session.processed_until_s + result.utterance_end_s
        session.utt_seq += 1
        utterance_wav = await _slice_utterance(
            session, session.processed_until_s, absolute_end_s, session.utt_seq,
        )
        session.processed_until_s = absolute_end_s
        # Second layer: the wrapper above handles everything it can while the
        # session is alive, but a failure in the wrapper itself -- or a
        # cancellation -- would still be swallowed by asyncio. Retrieving the
        # exception is what turns "silently discarded" into "in the log".
        task = asyncio.create_task(_dispatch_turn(session, utterance_wav))
        task.add_done_callback(_log_task_failure)


def _log_task_failure(task: asyncio.Task) -> None:
    """Retrieve a background turn's exception so asyncio cannot discard it."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("dispatch task failed after its own handler: %r", exc)


async def _handle_control(session: CallSession, raw: str):
    """Client -> server control channel. Only one message today, but it is
    the load-bearing half of the echo gate: the server cannot otherwise
    know when the caller's speaker actually stopped."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[%s] unparseable control frame: %r", session.call_id, raw[:80])
        return
    if msg.get("type") == "playback_done":
        session.release_gate()


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    session = CallSession(ws)
    logger.info("[%s] call started", session.call_id)
    poll_task = asyncio.create_task(_turn_poll_loop(session))

    try:
        await _speak(session, "নমস্কার, কলকাতা কেয়ার ডায়াগনস্টিকসে স্বাগতম। কীভাবে সাহায্য করতে পারি?")
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await session.append(message["bytes"])
            elif message.get("text"):
                await _handle_control(session, message["text"])
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[%s] session crashed", session.call_id)
    finally:
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        session.cleanup()
        logger.info("[%s] call ended", session.call_id)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
