"""Kolkata Care Diagnostics -- voice agent on the RAW PCM transport.

GENERATED FILE -- do not edit directly. Produced by
tools/make_pcm_variant.py from main.py; edit main.py and re-run that.

This variant changes only the TRANSPORT (how audio arrives and how the
turn detector reads it):
  * No ffmpeg, no WebM, no temp audio files, no subprocess per poll.
  * Appending is O(chunk) and reading the tail is O(tail), where the WebM
    path was O(call length) EVERY poll -- O(T^2) per call.
  * The sample index IS the timeline, exactly, so processed_until_s
    cannot drift away from real time.
See agent/pcm_buffer.py for the argument and the arithmetic.

----------------------------------------------------------------------


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

ECHO GATE AND BARGE-IN
----------------------
The mic is open for the entire call, and the agent's replies play out of
the caller's speaker. With no gate, the agent hears itself: its own
greeting lands in the same buffer the turn detector is watching, so VAD
fires a "the caller finished talking" on the agent's own voice, ASR
transcribes the agent, and processed_until_s advances past audio the
caller never produced. That is a self-sustaining loop, and it is what
made real calls cut the caller off in the first second and then run a
turn behind for the rest of the call.

This USED to be solved by going half-duplex: the client muted the mic
track while agent audio played, and the server refused to run turn
detection at all while `agent_speaking`. It worked, and it cost barge-in
entirely -- a caller could not interrupt the agent mid-sentence. For a
caller on a speakerphone, who is not holding a handset but talking across
a room at it, that is the single thing they most need.

The mute is gone. What replaces it is arbitration, in agent/echo_guard.py:
the server keeps the audio it just played as a REFERENCE signal, and every
window of microphone audio captured during playback is correlated against
it. Our own voice returning is a delayed, attenuated copy of something we
still hold; the caller's voice is not. So:

  * echo            -> stays gated, exactly as before;
  * real speech     -> stops playback (`_stop_audio`) and opens the gate;
  * anything unsure -> treated as echo, because a false barge-in truncates
                       a reply the caller then never hears.

Note the original objection to relying on the browser alone still stands:
its echoCancellation is built around a remote WebRTC peer's rendered
stream, and this audio is synthesized locally and played through Web
Audio, so how much of it the canceller sees as a far-end reference varies
by browser and platform. That is precisely why the arbitration above is
server-side and reference-based: it does not depend on the browser's AEC
having worked.

`barge_in()` deliberately does NOT go through `release_gate()`, because
the resync that follows release_gate discards everything captured during
playback -- which during a barge-in is the interruption itself.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime
import difflib
import hmac
import io
import json
import logging
import os
import tempfile
import time
import uuid
import wave

import torchaudio
from agent.pcm_buffer import PcmCallBuffer, SAMPLE_RATE
from fastapi import FastAPI, Header, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from agent.asr import TurnASR
from agent.audio_quality import condition_wav_file
from agent.echo_guard import (
    CONFIG as ECHO_CFG, EchoGuard, pcm_from_wav_bytes, suppress_echo,
)
from agent.executors import asr_gate, run_http, shutdown as _shutdown_http_pool
from agent.llm import extract_intent, ExtractionError
from agent.reply_templates import (
    missing_slot_prompt, test_rate_reply, doctor_availability_reply, booking_reply,
    doctors_by_department_reply, payment_reply, report_collection_reply,
    counter_fallback, language_switch_reply, language_unavailable_reply,
)
from agent import language as lang_mod
from agent import asr as asr_mod
from agent.i18n import t as _t
from agent import privacy
from agent.reply_templates import (
    verification_prompt, verification_failed_reply, verification_locked_reply,
    disclosure_blocked_reply, history_reply,
)
from agent.fast_path import Catalogue, FastPath
from agent.quality_metrics import ACTION_KEYPAD, METRICS, TurnFailureTracker
from agent.semantic_cache import SemanticCache, embed as _embed_probe
from agent.slot_parse import parse_date, parse_time, parse_phone, is_negative
from agent.tools_client import ClinicToolsClient, ToolCallError
from agent.tts import BUSY_LINE, TTSClient
from agent.vad_stream import TurnDetector
from agent import call_audit

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

# Is reading the recent microphone tail cheap on THIS transport?
#
# False here, and the generator flips it to True in main_pcm.py. It decides
# whether barge-in may poll at ECHO_CFG.barge_in_poll_s (0.10s) or has to stay
# on the ordinary POLL_INTERVAL_S (0.5s).
#
# On the WebM transport every tail read goes through _decode_to_wav, which
# re-decodes the WHOLE growing buffer from byte 0 and spawns an ffmpeg process
# to do it -- the O(T^2) behaviour agent/pcm_buffer.py exists to remove.
# Polling that ten times a second during every reply would mean ~10 ffmpeg
# spawns per second per speaking call, and at MAX_CONCURRENT_CALLS it
# saturates CPU long before the GPU is busy. On raw PCM the same read is a
# slice of a bytearray, so the fast cadence costs essentially nothing.
#
# CONSEQUENCE, stated plainly: barge-in can only meet barge_in_target_s on
# the PCM transport. On WebM it still works, but detection takes up to
# POLL_INTERVAL_S. WebM is the legacy/bench client; PCM is what production
# serves.
TAIL_READ_IS_CHEAP = True   # raw PCM: reading the tail is a slice, not a decode

# Which process a call record came from. main.py and main_pcm.py share one
# audit database (agent/call_audit.py), and crash recovery at startup must
# only ever close the records of ITS OWN transport. The generator flips it.
AUDIT_TRANSPORT = "pcm"

# THE RETRY LADDER, in words.
#
# Two rungs, because a caller who cannot be heard needs a different answer
# the second time than the first. Asking the same question twice in a row
# in the same market is not a retry, it is a loop -- the room has not got
# quieter between the two attempts, so nothing about repeating the request
# makes the next clip any better.
#
# Rung 1 names the actual problem (noise) and asks for the one thing the
# caller CAN change (volume, distance to the phone). Rung 2 stops asking
# for speech at all and moves to a channel the room cannot corrupt.
CLARIFY_PROMPT_BN = "দুঃখিত, আশেপাশে খুব আওয়াজ হচ্ছে। আর একটু জোরে, ফোনের কাছে এসে বলবেন?"
KEYPAD_PROMPT_BN = (
    "এখনও পরিষ্কার শোনা যাচ্ছে না। কী-প্যাড ব্যবহার করুন — "
    "পরীক্ষার রেটের জন্য ১, ডাক্তারের সময়ের জন্য ২, "
    "অ্যাপয়েন্টমেন্টের জন্য ৩ টিপুন।"
)

# What each key means, as the Bengali the caller would have spoken. Mapping
# to TEXT rather than to an intent id is deliberate: the digit then enters
# the SAME reasoning path as speech -- fast path, cache, LLM, slot filling,
# _continue_pending -- instead of needing a second, parallel dispatcher that
# would drift out of step with the spoken one.
KEYPAD_MENU_BN = {
    "1": "পরীক্ষার রেট জানতে চাই",
    "2": "ডাক্তারের সময় জানতে চাই",
    "3": "অ্যাপয়েন্টমেন্ট বুক করতে চাই",
}

CLINIC_API_BASE = os.environ.get("CLINIC_API_BASE", "http://localhost:8080")

# ADMISSION CONTROL -- the ceiling on simultaneous calls in this process.
#
# Every other limit added for peak hour (asr_gate, the TTS gate, the HTTP
# pool, Ollama's own queue) bounds ONE stage. None of them bounds how many
# callers are admitted in the first place, so without this the system's
# answer to overload is to accept everybody and let every caller degrade
# together -- longer ASR queues, longer Ollama queues, longer TTS queues, for
# all of them at once.
#
# That is the exact outcome the peak-hour requirement rules out. Quality
# stops being a function of when you ring only if the system is willing to
# say "not right now" to the caller who would push it past what it can serve
# well. A caller told plainly that the lines are busy can ring back in a
# minute; a caller silently placed in a queue that degrades everyone gets a
# worse experience AND makes it worse for the people already on the line.
#
# 12 is a starting point, not a measurement -- it is deliberately env-tunable
# so it can be set from a real load test (tools/bench_transport.py) rather
# than from this guess. Raise it once the stages behind it are known to keep
# up; lower it the moment they do not.
MAX_CONCURRENT_CALLS = int(os.environ.get("VOICE_AGENT_MAX_CALLS", "12"))

# Plain int, no lock: FastAPI runs one event loop, and every read/modify pair
# below is free of awaits, so it cannot interleave.
_active_calls = 0

# Order also doubles as PRIORITY: the field _next_missing() asks for next
# when several are still empty. doctor_name first because it is almost
# always already known by the time booking starts (named directly, or
# carried over from session.pending after a doctors_by_department /
# doctor_availability turn -- see _continue_pending below).
_BOOKING_FIELDS = ("doctor_name", "date", "time_slot", "patient_name", "phone")

app = FastAPI()

# ---- process-wide singletons: loaded once, shared by every call ----
_asr: TurnASR | None = None
_turn_detector: TurnDetector | None = None
_tools: ClinicToolsClient | None = None
_tts: TTSClient | None = None
_intent_cache: SemanticCache | None = None
_fast_path: FastPath | None = None
_audit_store: call_audit.AuditStore | None = None

# Stands in for a session with no CallAudit of its own (a test double built
# without CallSession). Writes nowhere. Exists so every capture point below
# can record unconditionally instead of each one guarding against it.
_NULL_AUDIT = call_audit.CallAudit(None, "no-call")


def _audit(session) -> call_audit.CallAudit:
    return getattr(session, "audit", None) or _NULL_AUDIT


@app.on_event("startup")
async def _startup():
    global _asr, _turn_detector, _tools, _tts, _intent_cache, _fast_path, _audit_store
    # FIRST, before any model loads: a caller accepted the moment startup
    # finishes must already have somewhere to be recorded. The store never
    # raises -- a database that cannot be opened is logged and reported under
    # /api/health's `audit` block, and calls are still served.
    _audit_store = call_audit.AuditStore()
    recovered = _audit_store.recover_unfinished(AUDIT_TRANSPORT)
    if recovered:
        logger.warning("audit: finalised %d call record(s) a previous run left open",
                       recovered)
    if not os.environ.get("VOICE_AGENT_AUDIT_TOKEN"):
        logger.warning("audit: VOICE_AGENT_AUDIT_TOKEN is unset -- /api/audit/* is "
                       "readable without a token. Set it on any non-bench deployment.")

    logger.info("loading IndicConformer...")
    _asr = await asyncio.to_thread(TurnASR)
    # Hand the singleton to the language registry under the pod's default
    # language, so a later per-language lookup reuses THIS instance instead
    # of loading a second copy of the same checkpoint onto the same card.
    asr_mod.register(lang_mod.default_lang(), _asr)
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
        await run_http(_embed_probe, "warmup")
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
        logger.info("fast path ready over %d catalogue rows", len(_fast_path.catalogue))
    except Exception as e:  # noqa: BLE001 - degrade to LLM-only, never fail startup
        logger.warning("catalogue unavailable, fast path disabled: %s", e)
        _fast_path = None

    logger.info("prewarming TTS...")
    await _tts.prewarm()
    logger.info("startup complete -- ready for calls")


@app.on_event("shutdown")
async def _shutdown():
    if _tools:
        await _tools.aclose()
    if _tts:
        await _tts.aclose()
    # wait=False: a worker parked on a socket read to Ollama must not hold
    # the process open past shutdown.
    _shutdown_http_pool()
    # Last, so everything recorded above is committed. Anything a call
    # records after this is counted as dropped, and the call's record is
    # finalised by recover_unfinished() on the next start.
    if _audit_store:
        await asyncio.to_thread(_audit_store.close)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "asr_loaded": _asr is not None,
        "clinic_api_base": CLINIC_API_BASE,
        # Surfaced so deploy/status.sh shows headroom at a glance -- "are we
        # near the ceiling" is the first question at peak, and it should not
        # require reading logs to answer.
        "active_calls": _active_calls,
        "max_calls": MAX_CONCURRENT_CALLS,
        # Non-zero write_failures or dropped means some call records are
        # incomplete -- see /api/audit/calls/{id}'s `integrity` block.
        "audit": _audit_store.health() if _audit_store else {"available": False},
    }


@app.get("/api/stats")
async def stats():
    """Cache effectiveness, for tuning the similarity threshold against
    real traffic rather than against my assumptions about it."""
    return {
        "fast_path": _fast_path.snapshot() if _fast_path else None,
        "intent_cache": _intent_cache.snapshot() if _intent_cache else None,
        "tts_cache": _tts.snapshot() if _tts else None,
        "audio_quality": METRICS.snapshot(),
    }


@app.get("/api/quality")
async def quality_stats():
    """Turn outcomes split by audio-quality bucket.

    Its own endpoint as well as a key in /api/stats, because this is the
    number the noisy-environment work is answerable to and it should be
    fetchable without pulling cache internals along with it.

    Read `noisy_bucket.accuracy` on its own. A blended figure is dominated
    by whichever bucket is larger -- in practice the quiet one -- so it can
    improve purely because quiet traffic grew, with nothing having got
    better for the callers this work exists for. `overall_accuracy` is
    published beside the split, never instead of it."""
    return METRICS.snapshot()


# ===========================================================================
# CALL RECORDS, FOR STAFF -- Author: Chakravardhan
# ===========================================================================
def _audit_read_denied(token: str | None):
    """The same conditional pattern clinic-api uses for delivery receipts:
    once VOICE_AGENT_AUDIT_TOKEN is set, a matching X-Audit-Token header is
    required; unset (a bench pod), the records are readable as /api/stats
    is, and startup logs a warning saying so. Read at request time so a
    token can be rotated without a restart."""
    expected = os.environ.get("VOICE_AGENT_AUDIT_TOKEN", "")
    if expected and not hmac.compare_digest((token or "").encode(), expected.encode()):
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    return None


@app.get("/api/audit/calls")
async def audit_calls(limit: int = Query(50, ge=1, le=500),
                      status: str | None = Query(None),
                      x_audit_token: str | None = Header(default=None)):
    """Most recent calls first, one row each. Filter by final_status to find
    the ones that went wrong: ?status=failed, ?status=error."""
    denied = _audit_read_denied(x_audit_token)
    if denied is not None:
        return denied
    if _audit_store is None:
        return JSONResponse(status_code=503, content={"error": "audit store not initialised"})
    calls = await asyncio.to_thread(_audit_store.list_calls, limit, status)
    return {"count": len(calls), "calls": calls}


@app.get("/api/audit/calls/{call_id}")
async def audit_call(call_id: str, x_audit_token: str | None = Header(default=None)):
    """One call, every event in order, and whether the record is whole."""
    denied = _audit_read_denied(x_audit_token)
    if denied is not None:
        return denied
    if _audit_store is None:
        return JSONResponse(status_code=503, content={"error": "audit store not initialised"})
    record = await asyncio.to_thread(_audit_store.get_call, call_id)
    if record is None:
        return JSONResponse(status_code=404, content={"error": "no such call"})
    return record


def _wav_duration_s(wav_bytes: bytes) -> float:
    try:
        with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001 - a fallback clip may not be canonical WAV
        return 5.0


class CallSession:
    """One PCM buffer for the ENTIRE call, and a marker for how much of it
    has been consumed.

    The continuous-buffer design is inherited from the WebM version, where
    it was forced: MediaRecorder puts the container header only in the
    first chunk, so resetting the buffer mid-call produced audio that
    could never be decoded again. That constraint is GONE here -- raw PCM
    has no header and any byte range is independently valid.

    It is kept anyway, because the second reason for it was always the
    better one: `processed_until_s` gives every turn an absolute,
    monotonic position on one call-long timeline. Silero's segment
    boundaries move as more audio arrives, so a turn detector run against
    a buffer that keeps restarting drops any utterance straddling the
    seam. Each poll therefore looks only at the UNPROCESSED TAIL.

    What PCM changes is the cost and the accuracy of that lookup: the tail
    is a slice rather than a full re-decode, and sample index maps to
    wall-clock exactly, so the timeline cannot drift.
    """

    def __init__(self, ws: WebSocket):
        self.ws = ws
        # The full 128-bit uuid4, not the 8-hex-char prefix this used to be.
        # It is now the permanent key of a call's audit record, and 32 bits
        # is a coin-flip chance of two calls colliding within ~77,000 calls.
        self.call_id = uuid.uuid4().hex
        self.tmpdir = tempfile.mkdtemp(prefix=f"kcd_call_{self.call_id}_")
        self.last_activity = time.time()
        self.dispatch_lock = asyncio.Lock()
        self.processed_until_s = 0.0
        # The language this caller is being served in. Set once from the
        # pod default and only ever changed by the caller asking, so a
        # single mis-transcribed word cannot flip a call into a language
        # the caller does not speak. See agent/language.py.
        self.lang = lang_mod.default_lang()

        # EVERY CALL LEAVES A COMPLETE RECORD -- Author: Chakravardhan
        #
        # Opened here, the moment the socket is accepted, so a call that
        # dies in its first second still has a record. See agent/call_audit.py.
        self.audit = call_audit.CallAudit(_audit_store, self.call_id,
                                          transport=AUDIT_TRANSPORT, language=self.lang)
        # Set by code that decides to END the call itself (the idle timeout).
        # ws_audio falls back to what it observed when this is None.
        self.end_reason: str | None = None

        # HISTORY VERIFICATION STATE -- Author: Chakravardhan
        #
        # Scoped to ONE CALL and never persisted. The handset is shared:
        # the person who verified may have handed the phone to somebody
        # else before the next call, so a token that outlived the
        # conversation would be the exact hole this story closes.
        # _cleanup() revokes it when the socket drops.
        self.history_token: str | None = None
        self.history_phone: str | None = None
        self.utt_seq = 0
        self.last_heartbeat = time.time()
        self.audio = PcmCallBuffer()
        self.declared_rate: int | None = None

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

        # The retry ladder for THIS caller. Per-session, not global: the
        # person in the market who has now failed twice needs the keypad,
        # and the person on the next line who failed once does not. See
        # agent/quality_metrics.py.
        self.failures = TurnFailureTracker()

        # Wall-clock origin for this call. The echo reference is timestamped
        # against it rather than against the decoded-buffer length, because
        # capture runs continuously at real time (see the client's
        # setMicMuted comment) and reading the buffer length would cost a
        # full WebM decode inside _speak. Any residual skew between the two
        # clocks is absorbed by the lag search in best_lag_correlation.
        self.started_at = time.time()

        # Holds what we played, decides echo vs barge-in, and classifies the
        # path. See agent/echo_guard.py.
        self.echo = EchoGuard()
        self._pending_echo_ref = None

    def call_time_s(self) -> float:
        return time.time() - self.started_at

    def playback_start_s(self) -> float:
        """Call-time at which the NEXT reply handed to the client will
        actually begin playing.

        Now, unless audio is already in flight -- in which case the client
        queues this clip behind it, and it starts when the current deadline
        (minus the guard that deadline carries) is reached."""
        now_s = self.call_time_s()
        if not self.agent_speaking:
            return now_s
        queued_s = (self.speak_deadline - PLAYBACK_GUARD_S) - self.started_at
        return max(now_s, queued_s)

    def take_echo_reference(self) -> object:
        """Hand over the reference for the window a barge-in was detected
        in, exactly once. Cleared on read so an ordinary turn that follows
        never has stale playback subtracted out of it."""
        ref, self._pending_echo_ref = self._pending_echo_ref, None
        return ref

    def barge_in(self):
        """The caller talked over the agent. Deliberately NOT release_gate().

        release_gate() sets resync_pending, and _resync_after_playback then
        jumps processed_until_s to the end of the buffer to throw away
        everything captured while the agent spoke. During a barge-in that
        region is precisely the caller's interruption -- discarding it would
        stop the agent and then ignore what stopped it."""
        # Stash what we were playing across the barge-in window. The clip
        # that follows contains the caller talking OVER this, so it is the
        # one turn in the call where subtracting our own audio is both
        # possible and worth doing.
        now_s = self.call_time_s()
        self._pending_echo_ref = self.echo.reference.slice(
            now_s - ECHO_CFG.barge_in_window_s, now_s)

        # Playback is about to be STOPPED, so the rest of this reply will
        # never leave the speaker. Forget it, or the level test keeps judging
        # later windows against sound that was never made -- the expected echo
        # ceiling stays high and the caller cannot interrupt a second time.
        self.echo.reference.truncate_after(now_s)

        # Skip forward to the barge-in window, but NO further. Everything
        # before it is the agent's own reply, and leaving processed_until_s
        # behind it would hand the turn detector a tail containing our echo --
        # it would then place utterance_start_s at the echo's onset and send
        # ASR a clip of the agent talking, which is exactly the self-answering
        # loop the old half-duplex gate existed to prevent.
        #
        # Not the full resync release_gate() would do: that jumps to the end
        # of the buffer and would discard the interruption itself. The
        # utterance-boundary fix then refines the true onset inside this
        # window, as it does for any other turn.
        self.processed_until_s = max(
            self.processed_until_s,
            now_s - ECHO_CFG.barge_in_window_s - UTTERANCE_PAD_S,
        )

        self.agent_speaking = False
        self.resync_pending = False
        self.speak_deadline = 0.0
        self.echo.barge_in_count += 1

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
        self.audio.append(chunk)

    async def send_json(self, sender: str, text: str):
        await self.ws.send_text(json.dumps({"sender": sender, "text": text}, ensure_ascii=False))

    async def send_audio(self, wav_bytes: bytes):
        if wav_bytes:
            await self.ws.send_bytes(wav_bytes)

    def cleanup(self):
        import shutil
        # The verification token dies with the call, deliberately. The
        # handset is shared -- the person who verified may hand the phone to
        # somebody else before the next call, and a token that outlived the
        # conversation would be the exact hole this story closes. Dropping
        # the local reference also stops it reaching any later log line.
        self.history_token = None
        self.history_phone = None
        with contextlib.suppress(OSError):
            shutil.rmtree(self.tmpdir, ignore_errors=True)


async def _speak(session: CallSession, text_bn: str, fallback_reason: str | None = None,
                 audit_redact: str | None = None):
    # AUDIT: what the caller was told is recorded in `finally`, so it is
    # recorded whether or not it reached them -- `delivered` stays False if
    # the socket was already gone -- and records what they HEARD: when TTS
    # fails the caller hears a pre-recorded apology, not these words.
    # `audit_redact` withholds the words themselves (a patient's history)
    # while keeping the fact that something was said.
    audio, tts_error, delivered = "none", None, False
    try:
        await session.send_json("AI", text_bn)
        try:
            wav = await _tts.synthesize(text_bn, session.lang)
            audio = "synthesized" if wav else "none"
        except Exception as e:  # noqa: BLE001 - TTS is the last mile, must not raise past here
            logger.warning("[%s] TTS failed (%s) -- using fallback audio", session.call_id, e)
            tts_error = f"{type(e).__name__}: {e}"
            wav = _tts.fallback_audio(fallback_reason or "tts_failure")
            audio = "fallback_clip" if wav else "none"

        # Close the gate BEFORE the bytes leave, never after: the client can
        # start playing the moment they land, and a poll tick that slips in
        # between send and gate is exactly the echo this prevents.
        # Record what we are about to play as the echo reference BEFORE the
        # bytes leave, for the same reason the gate closes first: the client can
        # start playing the moment they land, and a barge-in check that runs
        # before the reference exists would find "no reference" and treat our own
        # voice as the caller.
        #
        # Timestamped at the point this clip will actually START playing, which
        # is NOT now when a reply is already in flight -- replies queue on the
        # client (see hold_gate_for). Using send time for a queued clip puts its
        # reference earlier than the sound it describes, so the lookup during the
        # real playback returns silence, "no_reference" fires, and our own echo
        # is read as the caller interrupting.
        session.echo.note_playback(session.playback_start_s(),
                                   pcm_from_wav_bytes(wav, session.echo.sample_rate))

        session.hold_gate_for(_wav_duration_s(wav))
        await session.send_audio(wav)
        delivered = True
    finally:
        _audit(session).agent_response(
            text_bn, lang=getattr(session, "lang", None), audio=audio, delivered=delivered,
            fallback_reason=fallback_reason, tts_error=tts_error, redact=audit_redact)


async def _slice_utterance(session: CallSession, start_s: float, end_s: float, seq: int) -> str:
    """Cuts [start_s, end_s+pad] -- both ABSOLUTE call-time offsets -- out
    of the call's decoded WAV into its own small file for ASR."""
    sr = session.audio.sample_rate
    clip = session.audio.slice_tensor(start_s, end_s + UTTERANCE_PAD_S)
    clip_path = os.path.join(session.tmpdir, f"utt{seq}.wav")

    def _write():
        wav = clip.unsqueeze(0)
        out_sr = sr
        if sr != SAMPLE_RATE:
            # Only reachable when the browser refused a 16kHz AudioContext.
            # ASR expects 16k, so convert here rather than letting it
            # silently transcribe pitch-shifted audio.
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            out_sr = SAMPLE_RATE
        torchaudio.save(clip_path, wav, out_sr)

    await asyncio.to_thread(_write)
    return clip_path


def _record_intent(session, data: dict, source: str, **detail) -> None:
    """AUDIT: INTENT_DETECTED and SLOTS_EXTRACTED, from the very dict that
    is about to drive the turn -- and which of the three tiers produced it,
    because "the LLM decided" and "a string match decided" are different
    claims about how the system understood the caller."""
    _audit(session).intent(data.get("intent"), source, slots=data.get("slots") or {},
                           direct_reply_bn=data.get("direct_reply_bn"), **detail)


async def _resolve_intent(session: CallSession, text: str) -> dict:
    """Semantic cache in front of the LLM. A hit skips Ollama entirely --
    the slowest hop in the turn -- but the clinic lookup that follows still
    runs live, so a cached intent can never serve a stale price."""
    # Tier 1: decide it locally if we can. For a fixed catalogue the
    # entity is a string-matching problem with a 0.32 confidence margin,
    # where the embedding route had 0.03 -- see agent/fast_path.py. This
    # returns None whenever it is not sure, which is the common case for
    # anything except a routine price or availability question.
    # to_thread (default pool) for this one: it is local CPU work over the
    # 74-row catalogue with no network hop, so it belongs with the audio
    # path, not behind the blocking-HTTP pool. See agent/executors.py.
    if _fast_path is not None:
        hit = await asyncio.to_thread(_fast_path.resolve, text)
        if hit is not None:
            logger.info("[%s] fast path resolved %s (%.2f) -- no LLM call",
                        session.call_id, hit.intent, hit.confidence)
            data = hit.as_llm_shape()
            _record_intent(session, data, "fast_path", confidence=round(hit.confidence, 3),
                           matched_form=hit.matched_form)
            return data

    # The three calls below all make BLOCKING urllib requests to Ollama --
    # cache.get/put embed via bge-m3, extract_intent generates via Qwen --
    # so they run on the dedicated HTTP pool. On the shared default pool a
    # burst of concurrent callers parks every worker on a socket read and
    # the audio path (VAD polls, torchaudio, ASR) stops running for EVERY
    # call, not just the slow ones. agent/executors.py has the full write-up.
    cached, how = await run_http(_intent_cache.get, text)
    if cached is not None:
        logger.info("[%s] intent cache %s hit", session.call_id, how)
        _record_intent(session, cached, f"intent_cache_{how}")
        return cached

    data, diag = await run_http(extract_intent, text)
    logger.info("[%s] intent extracted in %.2fs (%d attempt(s))",
                session.call_id, diag["total_time_s"], diag["attempts"])
    # Recorded BEFORE the cache write below, so a cache failure cannot cost
    # the record of what the model actually returned.
    _record_intent(session, data, "llm", attempts=diag["attempts"],
                   latency_s=round(diag["total_time_s"], 3), retry_errors=diag["errors"])
    await run_http(_intent_cache.put, text, data)
    return data


def _next_missing(slots: dict) -> str | None:
    """-> the first still-empty field in _BOOKING_FIELDS order, or None
    once every field a booking needs is filled."""
    for field in _BOOKING_FIELDS:
        if not slots.get(field):
            return field
    return None


def _match_candidate_doctor(text: str, candidates: list[dict]) -> str | None:
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
    best_name, best_score = None, 0.0
    for c in candidates:
        forms = [c["name"], c["name"].split()[-1]]
        if c.get("name_bn"):
            forms.append(c["name_bn"])
        for form in forms:
            if not form:
                continue
            form_l = form.lower()
            score = difflib.SequenceMatcher(None, form_l, norm_text).ratio()
            if form_l in norm_text or norm_text in form_l:
                score = max(score, 0.85)
            if score > best_score:
                best_name, best_score = c["name"], score
    return best_name if best_score >= 0.55 else None


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


async def _finish_booking(session: CallSession, slots: dict):
    """All 5 fields are filled -- place the booking and clear pending
    regardless of outcome. Failure here is reported the same way the old
    single-shot book_appointment branch reported it (tool_failure
    fallback audio), just reachable now from either that branch OR from
    the tail of a multi-turn _continue_pending flow."""
    session.pending = None
    try:
        result = await _tools.book_appointment(
            slots["doctor_name"], slots["date"], slots["time_slot"],
            slots["patient_name"], slots["phone"],
        )
    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        await _speak(session, _t(session.lang, "generic.tool_failure"),
                     fallback_reason="tool_failure")
        return
    await _speak(session, booking_reply(slots, result, session.lang))


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

    # Verification owns its turn completely, and is checked BEFORE the
    # negative escape hatch below. A caller answering a PIN challenge with a
    # digit string that happens to parse as "na" must not silently abandon
    # the flow, and more importantly a wrong answer must burn an attempt
    # rather than being reinterpreted as a polite refusal.
    if awaiting == "history_verify":
        return await _continue_history_verification(session, text)

    # Universal escape hatch, checked before any field-specific parsing:
    # a caller mid-flow who says "না" / "থাক" is abandoning the booking,
    # not answering whichever question was pending.
    if is_negative(text):
        _audit(session).intent("abandon_flow", "slot_parse", flow=awaiting)
        session.pending = None
        await _speak(session, "ঠিক আছে, অ্যাপয়েন্টমেন্ট বাদ থাক। আর কিছু জানতে চান?")
        return True

    if awaiting == "doctor_choice":
        match = _match_candidate_doctor(text, pending.get("candidates") or [])
        _audit(session).slots("slot_parse", {"doctor_name": match}, awaiting="doctor_choice")
        if match is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False  # give a fresh LLM classification a chance instead
            await _speak(session, "দুঃখিত, ডাক্তারের নামটা একটু স্পষ্ট করে বলবেন?")
            return True

        date_iso = pending.get("offered_date") or datetime.date.today().isoformat()
        try:
            result = await _tools.get_doctor_availability(match, date_iso)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                         fallback_reason="tool_failure")
            return True

        await _speak(session, doctor_availability_reply({"doctor_name": match}, result))

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
                "slots": {"doctor_name": result.get("doctor_name") or match},
                "candidates": None, "offered_date": offered, "retries": 0,
            }
        else:
            session.pending = None
        return True

    if awaiting == "department_date":
        # Mirrors "doctor_choice" above, one level up: the caller was just
        # told nobody in this department sits TODAY and asked for another
        # day. Parse that reply as a date and re-run the same department
        # lookup with it, rather than dropping back to a cold LLM
        # classification of a bare date phrase (see this function's
        # docstring for why that silently loses context).
        value = parse_date(text, offered_date=pending.get("offered_date"))
        _audit(session).slots("slot_parse", {"date": value}, awaiting="department_date")
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
            await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                         fallback_reason="tool_failure")
            return True

        await _speak(session, doctors_by_department_reply({"department": department}, result))

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
    _audit(session).slots("slot_parse", {awaiting: value}, awaiting=awaiting)

    if value is None:
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None
            return False
        await _speak(session, missing_slot_prompt("book_appointment", awaiting, session.lang))
        return True

    pending["slots"][awaiting] = value
    pending["retries"] = 0
    missing = _next_missing(pending["slots"])
    if missing is None:
        await _finish_booking(session, pending["slots"])
        return True
    pending["awaiting"] = missing
    await _speak(session, missing_slot_prompt("book_appointment", missing))
    return True


def _suppress_echo_in_place(clip_path: str, reference) -> None:
    """Subtract the agent's own playback out of a barge-in clip, in place.

    Blocking; callers run it on a worker thread. Swallows its own errors on
    purpose -- echo suppression is an improvement to a clip that is already
    usable enough to have triggered a barge-in, so a failure here must leave
    the turn alone rather than lose it."""
    try:
        import soundfile as sf

        samples, sr = sf.read(clip_path, dtype="float32", always_2d=False)
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        cleaned = suppress_echo(samples, reference, sr)
        sf.write(clip_path, cleaned, sr, subtype="PCM_16")
    except Exception as e:  # noqa: BLE001
        logger.warning("echo suppression skipped for %s: %s", clip_path, e)


async def _clarify_or_offer_keypad(session: CallSession, quality=None,
                                   reason: str = "low_quality"):
    """The turn could not be acted on. Decide WHICH way to say so.

    Everything that means "we did not understand this caller" funnels
    through here -- a clip below the quality floor, and ASR returning
    nothing -- so the ladder counts real consecutive failures rather than
    one particular failure mode. Two different silent failures in a row
    are still two failures to the caller.

    The rung is chosen by session.failures (agent/quality_metrics.py), not
    by anything about this turn: the caller's recent history is what says
    whether another question is worth asking."""
    action = session.failures.record_failure()

    if action == ACTION_KEYPAD:
        METRICS.record_keypad_offer()
        logger.info("[%s] %d consecutive failed turns (%s) -- offering keypad",
                    session.call_id, session.failures.consecutive_failures, reason)
        # Control frame first so the keys are on screen before the caller
        # hears why. Carries no display text of its own -- the spoken line
        # below is the one the caller reads in the log, and sending both
        # would print it twice.
        await session.send_json("_keypad", "on")
        await _speak(session, KEYPAD_PROMPT_BN, fallback_reason="keypad_offer")
        return

    METRICS.record_clarification()
    logger.info("[%s] turn unusable (%s) -- asking again (failure %d/%d)",
                session.call_id, reason, session.failures.consecutive_failures,
                session.failures.max_retries)
    await _speak(session, CLARIFY_PROMPT_BN, fallback_reason=reason)


async def _handle_keypad_digit(session: CallSession, digit: str):
    """A keypad press. Translated to the Bengali the caller would have said
    and pushed through the ordinary text path -- see KEYPAD_MENU_BN for why
    it is mapped to text rather than to an intent id.

    Counts as a success for the ladder: the fallback did its job, the
    caller got through, and the next isolated misheard turn deserves an
    ordinary clarification rather than the keypad again."""
    text = KEYPAD_MENU_BN.get(digit.strip())
    if text is None:
        logger.info("[%s] keypad: ignoring unmapped key %r", session.call_id, digit)
        return

    METRICS.record_keypad_entry()
    session.failures.record_success()
    logger.info("[%s] keypad: %r -> %r", session.call_id, digit, text)
    await _dispatch_turn(session, "", text_override=text)



# ===========================================================================
# PATIENT HISTORY -- disclosed only after verification
# Author: Chakravardhan
# ===========================================================================
async def _history_guard(session: CallSession, phone: str) -> bool:
    """-> True if it is safe to speak private information on THIS call.

    Checked SEPARATELY from verification and BEFORE any history is fetched.
    The acceptance criterion says "so that whoever else uses this handset
    cannot HEAR" -- a correctly verified patient with the phone on
    loudspeaker is entitled to their history and must still not have it read
    out, because the criterion is about who ends up hearing it, not about
    entitlement.

    agent/privacy.py explains why this uses EchoGuard.classify() rather than
    reporting_path(), and why an unclassified path counts as unsafe.
    """
    safe, reason = privacy.audio_path_is_private(session.echo)
    if safe:
        return True

    logger.info("[%s] history disclosure refused: %s", session.call_id, reason)
    # Audited on the clinic side, not only in this log: without a row,
    # "it would not tell me my history" has no explanation at the clinic and
    # the likeliest support response is to switch the check off.
    try:
        await _tools.record_disclosure_refusal(phone, reason, session.call_id)
    except Exception:                                  # noqa: BLE001
        pass
    await _speak(session, disclosure_blocked_reply(reason, session.lang))
    return False


async def _start_history_verification(session: CallSession, phone: str | None):
    """Begin the challenge. Never says whether the number is known."""
    if not phone:
        # No number to look up. Answered with the same sentence as a failed
        # verification -- see verification_failed_reply()'s docstring.
        await _speak(session, verification_failed_reply(True, session.lang))
        return

    if not await _history_guard(session, phone):
        return

    try:
        challenge = await _tools.begin_verification(phone, session.call_id)
    except ToolCallError as e:
        logger.error("[%s] verification start failed: %s", session.call_id, e)
        await _speak(session, _t(session.lang, "generic.tool_failure"),
                     fallback_reason="tool_failure")
        return

    if challenge.get("locked"):
        await _speak(session, verification_locked_reply(session.lang))
        return

    session.history_phone = phone
    session.pending = {
        "awaiting": "history_verify",
        "factor": challenge.get("factor") or "dob",
        "slots": {},
        "candidates": None,
        "offered_date": None,
        # Counted here for the WORDING only ("try again" vs "go to the
        # counter"). The real lockout is counted per PATIENT on the clinic
        # side, so hanging up and redialling does not reset it.
        "retries": 0,
    }
    await _speak(session, verification_prompt(session.pending["factor"], session.lang))


async def _speak_history(session: CallSession):
    """Fetch and speak, re-checking the room immediately beforehand.

    The audio path is checked AGAIN here, not just at the start of
    verification. A caller can put the phone on speaker between answering
    the challenge and hearing the answer -- and that is the exact moment
    the private information is about to be spoken.
    """
    if not await _history_guard(session, session.history_phone or ""):
        return
    try:
        result = await _tools.read_history(session.history_token, session.call_id)
    except ToolCallError as e:
        logger.error("[%s] history read failed: %s", session.call_id, e)
        await _speak(session, _t(session.lang, "generic.tool_failure"),
                     fallback_reason="tool_failure")
        return

    if not result.get("found"):
        # Token expired or revoked mid-call. Treated as "not verified",
        # which is what it is.
        session.history_token = None
        await _speak(session, verification_failed_reply(True, session.lang))
        return

    # Redacted in the audit: the fact of disclosure is recorded (here, in the
    # read_history API_RESPONSE, and in clinic-api's disclosure_audit), the
    # medical history itself is not copied into a second store.
    await _speak(session, history_reply(result, session.lang), audit_redact="patient_history")


async def _continue_history_verification(session: CallSession, text: str) -> bool:
    """The caller just answered the challenge. Always returns True -- this
    turn belongs to verification either way."""
    pending = session.pending
    factor = pending.get("factor") or "dob"

    # Folded to ASCII first: a Bengali or Devanagari numeral from the
    # matching ASR checkpoint is the same PIN as its ASCII form, and a
    # caller must not fail verification over which script their digits
    # arrived in.
    answer = lang_mod.to_ascii_digits(text)

    try:
        outcome = await _tools.verify_caller(
            session.history_phone or "", factor, answer, session.call_id)
    except ToolCallError as e:
        logger.error("[%s] verification failed to run: %s", session.call_id, e)
        session.pending = None
        await _speak(session, _t(session.lang, "generic.tool_failure"),
                     fallback_reason="tool_failure")
        return True

    reply = outcome.get("reply")
    if reply == "verified":
        session.pending = None
        session.history_token = outcome.get("token")
        logger.info("[%s] caller verified (factor=%s)", session.call_id, factor)
        await _speak_history(session)
        return True

    if reply == "locked":
        session.pending = None
        await _speak(session, verification_locked_reply(session.lang))
        return True

    # Failed. The caller is told the same sentence whatever the reason --
    # wrong answer, unknown number, no factor on file. Only whether ANOTHER
    # ATTEMPT IS POSSIBLE changes the wording, and that is not a hint.
    pending["retries"] += 1
    exhausted = pending["retries"] >= 2
    if exhausted:
        session.pending = None
    await _speak(session, verification_failed_reply(exhausted, session.lang))
    return True

async def _dispatch_turn(session: CallSession, utterance_wav: str,
                         text_override: str | None = None):
    """_run_turn, with any exception it raises put on the call's record.

    A turn runs as a fire-and-forget task (see _turn_poll_loop), so an
    exception escaping it used to reach nothing but asyncio's "Task
    exception was never retrieved" at garbage collection -- the caller got
    dead air and there was no trace of which call it happened on. It is
    recorded here and then RE-RAISED unchanged: auditing observes the
    failure, it does not change how the turn fails."""
    try:
        await _run_turn(session, utterance_wav, text_override)
    except Exception as e:
        _audit(session).error("turn", e)
        raise


async def _run_turn(session: CallSession, utterance_wav: str,
                    text_override: str | None = None):
    """One full turn: ASR -> intent -> tool -> templated reply -> TTS.
    Serialized per-call via session.dispatch_lock so replies never
    interleave, even if the caller starts talking again immediately.

    text_override skips audio entirely. It is how a keypad digit enters
    this function: the alternative -- a second dispatcher for DTMF -- would
    have to re-implement the fast path, the cache, slot filling and
    _continue_pending, and would drift out of step with the spoken path the
    first time either was touched."""
    async with session.dispatch_lock:
        quality = None
        audit = _audit(session)
        audit.begin_turn()
        # The caller is answering a verification challenge, so what they say
        # IS the secret -- a PIN or a date of birth. Withheld from the record
        # exactly as clinic-api's disclosure_audit withholds it; the length is
        # kept so the record still shows an answer was given.
        secret = ("verification_answer"
                  if (session.pending or {}).get("awaiting") == "history_verify" else None)

        if text_override is not None:
            text = text_override.strip()
            if not text:
                return
            audit.transcript(text, source="keypad", redacted=secret)
        else:
            try:
                # CONDITION BEFORE ASR, and gate before the GPU is asked for
                # anything. Two reasons, in order of importance:
                #
                #  1. A clip below the floor must not produce a confident
                #     answer. Downstream cannot tell a transcript of speech
                #     from a transcript of a bus, so the decision has to be
                #     made here, on the audio, while that distinction still
                #     exists.
                #  2. A rejected clip then costs no inference at all, which
                #     is the stage under most pressure at peak.
                #
                # to_thread because it is numpy over the whole clip -- tens
                # of milliseconds of CPU that would otherwise block every
                # other call's socket on this event loop.
                # A barge-in clip is the one turn where the caller's speech
                # is genuinely mixed with our own playback, and the only turn
                # where we hold the exact signal mixed into it. Subtract it
                # before anything else looks at the audio; every other turn
                # reads None here and is untouched.
                #
                # OUTSIDE the try below on purpose. That try fails OPEN --
                # it drops the quality floor and sends the raw clip -- so a
                # fault raised inside it would silently disarm an unrelated
                # feature rather than surfacing.
                echo_ref = getattr(session, "take_echo_reference", lambda: None)()
                if echo_ref is not None and ECHO_CFG.echo_suppression_enabled:
                    await asyncio.to_thread(
                        _suppress_echo_in_place, utterance_wav, echo_ref)

                try:
                    conditioned = await asyncio.to_thread(condition_wav_file, utterance_wav)
                    quality = conditioned.quality
                    logger.info(
                        "[%s] clip: %.2fs snr=%.1fdB speech=%.0f%% gain=%+.1fdB %s%s",
                        session.call_id, quality.duration_s, quality.snr_db,
                        100 * quality.speech_ratio, conditioned.gain_db,
                        quality.bucket(),
                        "" if quality.usable else f" REJECT{list(quality.reasons)}",
                    )
                except Exception as e:  # noqa: BLE001
                    # Fail OPEN. A bug in the conditioner must not take the
                    # whole service down to "sorry, say again" on every turn;
                    # an unconditioned clip still transcribes, which is the
                    # behaviour that shipped before this stage existed.
                    logger.warning("[%s] conditioning failed (%s) -- sending raw clip",
                                   session.call_id, e)
                    audit.error("audio_conditioning", e, handled=True, fail_open=True)

                if quality is not None and not quality.usable:
                    audit.transcript(None, source="speech", status="rejected_low_quality",
                                     audio=call_audit.describe_quality(quality))
                    METRICS.record_turn(quality, success=False,
                                        path=session.echo.reporting_path())
                    await _clarify_or_offer_keypad(
                        session, quality, reason=quality.reasons[0])
                    return

                # asr_gate bounds how many turns may occupy a thread-pool worker
                # waiting on the GPU. dispatch_lock above is per-CALL ordering;
                # this is process-wide admission control. See agent/executors.py.
                async with asr_gate:
                    # The node for the caller's language, falling back to
                    # the default one. Falling back rather than failing is
                    # deliberate: a caller whose language this pod cannot
                    # hear is still a caller, and a Bengali transcript we
                    # can act on beats a dropped turn.
                    node = asr_mod.for_language(session.lang) or _asr
                    asr_result = await node.transcribe_utterance(utterance_wav)
            finally:
                with contextlib.suppress(OSError):
                    os.remove(utterance_wav)

            text = asr_result.text.strip()
            if not text:
                # Empty text from a clip that PASSED the floor. Counted as a
                # failed turn like any other: the caller was not understood,
                # and which stage failed to understand them is our problem,
                # not theirs.
                logger.info("[%s] ASR returned empty text", session.call_id)
                audit.transcript("", source="speech", status="empty",
                                 decoder_used=getattr(asr_result, "decoder_used", None),
                                 audio=call_audit.describe_quality(quality))
                if quality is not None:
                    METRICS.record_turn(quality, success=False,
                                        path=session.echo.reporting_path())
                await _clarify_or_offer_keypad(session, quality, reason="asr_empty")
                return

            if quality is not None:
                METRICS.record_turn(quality, success=True,
                                    path=session.echo.reporting_path())
            session.failures.record_success()
            audit.transcript(text, source="speech", redacted=secret,
                             decoder_used=getattr(asr_result, "decoder_used", None),
                             decoder_agreement=getattr(asr_result, "decoder_agreement", None),
                             audio=call_audit.describe_quality(quality))

        await session.send_json("User", text)

        # ---- an explicit language request owns the turn -------------------
        #
        # Checked BEFORE _continue_pending and before intent extraction, and
        # deliberately so. A caller who says "can you speak English" halfway
        # through a booking is not answering the question they were just
        # asked; running that utterance through the date parser produces
        # either a wrong slot or a re-prompt, and either way the request is
        # ignored, which reads as the system not having heard them.
        #
        # The booking flow is NOT abandoned -- session.pending is untouched,
        # so the very next turn resumes exactly where it was, now in the new
        # language.
        switched = lang_mod.requested_switch(text)
        if switched and switched != session.lang:
            logger.info("[%s] caller switched language: %s -> %s",
                        session.call_id, session.lang, switched)
            audit.intent("language_switch", "keyword", language_from=session.lang,
                         language_to=switched)
            session.lang = switched
            await _speak(session, language_switch_reply(session.lang))
            return
        unavailable = lang_mod.requested_switch_unavailable(text)
        if unavailable:
            # Told the truth rather than ignored. A caller asking for a
            # language this pod has no ASR checkpoint for will otherwise ask
            # again, and again, burning turns on a line that can never say
            # yes -- see agent/language.py's enabled().
            logger.info("[%s] caller asked for unavailable language %s",
                        session.call_id, unavailable)
            audit.intent("language_unavailable", "keyword", requested=unavailable)
            await _speak(session, language_unavailable_reply(session.lang))
            return

        # A booking (or the doctor-choice / date-confirm step just before
        # one) already in progress owns this turn -- see _continue_pending's
        # docstring for why intent extraction must NOT also run on top of it.
        if await _continue_pending(session, text):
            return

        try:
            data = await _resolve_intent(session, text)
        except ExtractionError as e:
            logger.error("[%s] intent extraction failed: %s", session.call_id, e)
            audit.error("intent_extraction", e, handled=True)
            await _speak(session, _t(session.lang, "fallback.llm_failure"), fallback_reason="llm_failure")
            return

        intent = data["intent"]
        slots = data["slots"]

        if intent == "smalltalk":
            await _speak(session, data.get("direct_reply_bn") or _t(session.lang, "fallback.greeting"))
            return

        if intent == "unclear":
            await _speak(session, _t(session.lang, "fallback.unclear"))
            return

        try:
            if intent == "test_rate":
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", session.lang))
                    return
                result = await _tools.get_test_rate(slots["test_name"])
                await _speak(session, test_rate_reply(slots, result, session.lang))

            elif intent == "doctor_availability":
                if not slots.get("doctor_name"):
                    await _speak(session, missing_slot_prompt(intent, "doctor_name", session.lang))
                    return
                # Default to TODAY, not "whenever next available": a bare
                # "ডাক্তার সেন আছেন?" with no date mentioned is a caller
                # asking about right now, and the reply text below already
                # said " আজ" (today) for exactly this case -- the old code
                # passed date=None through to the API, which answers a
                # different question ("when next"), so a doctor who simply
                # wasn't in today got reported by their NEXT sitting date
                # instead of "not today, but they're on Tuesdays" etc.
                date_iso = slots.get("date") or datetime.date.today().isoformat()
                result = await _tools.get_doctor_availability(slots["doctor_name"], date_iso)
                await _speak(session, doctor_availability_reply(slots, result, session.lang))

                # Keep the flow open for "yes, book that day" / "another
                # day" -- doctor_availability_reply() just asked exactly
                # that question. See _continue_pending's "date" state.
                offered = None
                if result.get("found"):
                    offered = result.get("date") if result.get("available") else result.get("next_available_date")
                session.pending = {
                    "awaiting": "date",
                    "slots": {"doctor_name": result.get("doctor_name") or slots["doctor_name"]},
                    "candidates": None, "offered_date": offered, "retries": 0,
                } if offered else None

            elif intent == "doctors_by_department":
                if not slots.get("department"):
                    await _speak(session, missing_slot_prompt(intent, "department", session.lang))
                    return
                # Default to TODAY when the caller didn't name a date, same
                # reasoning as doctor_availability above: "অর্থোতে কারা
                # আছেন" (who's in ortho) is almost always asking who is
                # actually in the chamber right now, not for a roster of
                # every doctor the department has ever employed regardless
                # of whether they sit this week. Only an EXPLICIT date
                # bypasses this (used as-is below).
                date_iso = slots.get("date") or datetime.date.today().isoformat()
                result = await _tools.get_doctors_by_department(slots["department"], date_iso)
                await _speak(session, doctors_by_department_reply(slots, result, session.lang))

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

            elif intent == "payment":
                # THIS BRANCH MUST NOT BE ABLE TO FAIL.
                #
                # The story it serves is "every flow completes without a
                # smartphone", and a flow that answers "I couldn't check
                # that right now" has dead-ended just as surely as one that
                # sends a payment link -- the caller is left holding
                # nothing either way. So the rate lookup is decoration: if
                # the caller named a test we quote the amount, and if
                # clinic-api is down we still tell them HOW to pay, which
                # is what they actually asked and which never depended on
                # the database.
                result = {}
                if slots.get("test_name"):
                    try:
                        result = await _tools.get_test_rate(slots["test_name"])
                    except ToolCallError as e:
                        logger.warning("[%s] rate lookup failed during payment reply: %s",
                                       session.call_id, e)
                await _speak(session, payment_reply(slots, result, session.lang))

            elif intent == "report_collection":
                # Same rule as payment above: the collection path is clinic
                # policy, not a database row, so it survives clinic-api
                # being unreachable. Only the "ready in N hours" part needs
                # the lookup, and its absence is handled by the template.
                result = {}
                if slots.get("test_name"):
                    try:
                        result = await _tools.get_test_rate(slots["test_name"])
                    except ToolCallError as e:
                        logger.warning("[%s] rate lookup failed during report reply: %s",
                                       session.call_id, e)
                await _speak(session, report_collection_reply(slots, result, session.lang))

            elif intent == "patient_history":
                # NOTHING IS FETCHED UNTIL VERIFICATION PASSES. The lookup
                # is not performed and then withheld -- it is not performed
                # at all, so there is nothing in this process to leak.
                if session.history_token:
                    await _speak_history(session)
                else:
                    await _start_history_verification(session, slots.get("phone"))

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

                missing = _next_missing(merged)
                if missing is None:
                    await _finish_booking(session, merged)
                    return

                session.pending = {
                    "awaiting": missing, "slots": merged, "candidates": None,
                    "offered_date": (session.pending or {}).get("offered_date"), "retries": 0,
                }
                await _speak(session, missing_slot_prompt(intent, missing))

        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                         fallback_reason="tool_failure")


async def _resync_after_playback(session: CallSession) -> bool:
    """Drop everything captured while the agent was talking, by moving
    processed_until_s to the current end of the decoded buffer. That region
    is muted silence from the client's side; skipping it keeps the turn
    detector from ever analysing it, and -- more importantly -- keeps
    processed_until_s anchored to real time instead of drifting a full
    reply behind, which is what made later turns surface late."""
    buffer_end_s = session.audio.duration_s
    session.processed_until_s = max(session.processed_until_s,
                                    buffer_end_s - RESYNC_REWIND_S)
    session.resync_pending = False
    logger.info("[%s] resynced to %.2fs after playback", session.call_id, session.processed_until_s)
    return True


async def _recent_mic_tail(session: CallSession, seconds: float):
    """The last `seconds` of captured microphone audio, as (samples, sr).

    TRANSPORT-SPECIFIC -- tools/make_pcm_variant.py swaps this body. Returns
    None when there is not yet enough audio to judge."""
    sr = session.audio.sample_rate
    n = int(seconds * sr)
    total = len(session.audio)
    if total < n:
        return None
    tail = session.audio.tail_tensor((total - n) / sr)
    if tail.numel() == 0:
        return None
    return tail.numpy(), sr


async def _check_barge_in(session: CallSession) -> bool:
    """Is the caller talking over the agent right now?

    This is what replaces the half-duplex mute. It runs only while
    `agent_speaking`, on the most recent window of microphone audio, and
    asks agent/echo_guard.py to separate our own echo from a real
    interruption -- using the audio we just played as the reference.

    Returns True when playback was stopped and the caller's turn should be
    detected normally from here.

    Deliberately conservative: EchoGuard.assess answers "echo" whenever it
    is unsure, so a doubtful case leaves the agent speaking. A false
    barge-in truncates a reply the caller then never hears, which is worse
    than a missed one -- they can always speak again."""
    if not ECHO_CFG.barge_in_enabled:
        return False

    tail = await _recent_mic_tail(session, ECHO_CFG.barge_in_window_s)
    if tail is None:
        return False
    samples, sr = tail

    # The reference is looked up over the window ENDING now, on the same
    # wall-clock the reply was timestamped with. Any skew between that clock
    # and the capture buffer is absorbed by the lag search inside assess().
    now_s = session.call_time_s()
    verdict = await asyncio.to_thread(
        session.echo.assess, samples, now_s - ECHO_CFG.barge_in_window_s, sr)

    if not verdict.is_barge_in:
        return False

    session.barge_in()
    METRICS.record_barge_in()
    logger.info("[%s] barge-in: %s", session.call_id, verdict.as_dict())
    # The reply in flight was cut off: the AGENT_RESPONSE before this event
    # was NOT heard in full, and the record must not imply it was.
    _audit(session).record(call_audit.AGENT_INTERRUPTED,
                           {"at_call_s": round(now_s, 3), "verdict": verdict.as_dict()})

    # Tell the client to stop playing immediately. Without this the agent
    # keeps talking into the caller's interruption -- the gate would be open
    # on the server while the speaker is still going, which is both rude and
    # a fresh source of echo.
    with contextlib.suppress(Exception):
        await session.send_json("_stop_audio", "on")
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
        # Faster cadence WHILE the agent is speaking: barge-in cannot be
        # detected sooner than the poll rate, so the interval has to sit well
        # under barge_in_target_s. Outside playback the original cadence is
        # unchanged -- this must not make idle calls busier.
        #
        # Gated on TAIL_READ_IS_CHEAP: on the WebM transport each tail read
        # re-decodes the entire call, so the fast cadence would trade barge-in
        # latency for the O(T^2) CPU blowup the PCM transport was built to
        # avoid. See that constant.
        fast = session.agent_speaking and TAIL_READ_IS_CHEAP
        await asyncio.sleep(ECHO_CFG.barge_in_poll_s if fast else POLL_INTERVAL_S)

        if time.time() - session.last_activity > IDLE_TIMEOUT_S:
            logger.info("[%s] idle timeout, closing", session.call_id)
            session.end_reason = call_audit.END_IDLE_TIMEOUT
            await _speak(session, "লাইনে কোনো সাড়া পাচ্ছি না, কল শেষ করছি। ধন্যবাদ।")
            with contextlib.suppress(Exception):
                await session.ws.close()
            return

        if time.time() - session.last_heartbeat > HEARTBEAT_INTERVAL_S:
            session.last_heartbeat = time.time()
            with contextlib.suppress(Exception):
                await session.ws.send_text('{"sender":"_ping","text":""}')

        # --- full-duplex gate: tell our own echo apart from the caller ---
        #
        # This used to be an unconditional `continue`: while the agent spoke,
        # turn detection did not run at all and the client muted the mic, so
        # nothing the caller said during a reply could ever be heard. That is
        # what made barge-in impossible.
        #
        # Now the window is examined and arbitrated. Only a real interruption
        # opens the gate early; our own echo still does not.
        if session.agent_speaking:
            if await _check_barge_in(session):
                pass          # gate opened by barge_in(); fall through and
                              # detect the caller's turn from the same audio
            elif time.time() < session.speak_deadline:
                continue
            else:
                logger.warning("[%s] no playback-done from client, releasing gate on deadline",
                               session.call_id)
                session.release_gate()

        if session.resync_pending:
            await _resync_after_playback(session)
            continue

        sr = session.audio.sample_rate
        tail = session.audio.tail_tensor(session.processed_until_s)
        if tail.numel() < int(0.2 * sr):
            continue  # not enough new audio to judge yet -- not an error

        result = await asyncio.to_thread(_turn_detector.poll, tail, sr)
        if result.utterance_end_s is None:
            continue

        absolute_end_s = session.processed_until_s + result.utterance_end_s

        # Cut the clip at the caller's first syllable, NOT at the end of the
        # previous turn. Those are the same thing only when the caller replies
        # instantly; every second they spend thinking sits between the two, and
        # in a noisy room that gap is not silence, it is traffic or a crowd.
        # Sending it to ASR turns a 3-second question into a mostly-noise clip
        # and gets longer the longer the caller hesitates -- which is exactly
        # when they are least able to be understood.
        #
        # UTTERANCE_PAD_S of lead-in for the same reason it is already added to
        # the far end: a hard cut at the detected boundary clips the first
        # phoneme. Clamped so the clip can never start before audio this call
        # has already consumed.
        absolute_start_s = max(
            session.processed_until_s,
            session.processed_until_s + result.utterance_start_s - UTTERANCE_PAD_S,
        )
        session.utt_seq += 1
        utterance_wav = await _slice_utterance(
            session, absolute_start_s, absolute_end_s, session.utt_seq,
        )
        # Still advances to the END, not the start: the skipped lead-in is
        # consumed, not left behind for the next poll to re-examine.
        session.processed_until_s = absolute_end_s
        asyncio.create_task(_dispatch_turn(session, utterance_wav))


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
    elif msg.get("type") == "hello":
        # The browser may refuse the 16kHz AudioContext we ask for. Trust
        # what the client reports over what we requested: a wrong assumed
        # rate would not fail loudly, it would just make every timestamp
        # and every transcript quietly wrong.
        rate = int(msg.get("sampleRate") or SAMPLE_RATE)
        session.declared_rate = rate
        session.audio.sample_rate = rate
        if rate != SAMPLE_RATE:
            logger.warning("[%s] client capturing at %dHz, not %dHz -- resampling per utterance",
                           session.call_id, rate, SAMPLE_RATE)
        logger.info("[%s] transport: %s @ %dHz", session.call_id,
                    msg.get("format", "pcm_s16le"), rate)
    elif msg.get("type") == "audio_mode":
        # A hint only. EchoGuard treats it as a starting point and lets the
        # measured Echo Return Loss override it, because a hint can be absent
        # or simply wrong and the measurement cannot.
        session.echo.declare_path(msg.get("mode"))
        logger.info("[%s] client declares audio path: %r", session.call_id, msg.get("mode"))
    elif msg.get("type") == "dtmf":
        # Dispatched as a task, not awaited: _handle_keypad_digit runs a full
        # turn (LLM, clinic API, TTS) and this coroutine is the socket's
        # receive path. Awaiting it here would stop reading audio -- and the
        # caller may well keep talking while the keypad turn is in flight.
        asyncio.create_task(_handle_keypad_digit(session, str(msg.get("digit", ""))))


def _note_task_crash(session: CallSession, stage: str, task: asyncio.Task) -> None:
    """Done-callback for a call's background task. Records a crash at the
    moment it happens, with its own traceback, rather than whenever -- if
    ever -- somebody awaits the task. Retrieving the exception here also
    retires asyncio's "never retrieved" warning; the crash is logged below
    instead, with the call id on it."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("[%s] %s crashed: %r", session.call_id, stage, exc, exc_info=exc)
        _audit(session).error(stage, exc)


async def _reject_at_capacity(ws: WebSocket):
    """Turn a caller away in words, not by dropping the socket.

    A bare close looks to the caller like the line is broken. Saying it --
    and saying it fast, from the prewarmed TTS cache -- is the difference
    between "this service is down" and "call back in a minute".

    A refused caller is still a caller, and the busiest minutes are exactly
    when a hospital will want to know how many were turned away -- so the
    refusal gets a call record of its own (final_status "rejected")."""
    audit = call_audit.CallAudit(_audit_store, transport=AUDIT_TRANSPORT,
                                 language=lang_mod.default_lang())
    logger.warning("[%s] at capacity (%d/%d active) -- refusing call",
                   audit.call_id, _active_calls, MAX_CONCURRENT_CALLS)
    delivered, audio = False, "none"
    with contextlib.suppress(Exception):
        await ws.send_text(json.dumps({"sender": "AI", "text": BUSY_LINE},
                                      ensure_ascii=False))
        delivered = True
    with contextlib.suppress(Exception):
        # Cache hit in the normal case (BUSY_LINE is prewarmed), so this does
        # not queue behind the TTS gate it is protecting. If TTS is down
        # entirely the text above already went out; audio is a bonus.
        await ws.send_bytes(await _tts.synthesize(BUSY_LINE))
        audio = "synthesized"
    audit.agent_response(BUSY_LINE, lang=lang_mod.default_lang(), audio=audio,
                         delivered=delivered)
    # Give the client a moment to receive both frames before the close lands.
    await asyncio.sleep(0.25)
    with contextlib.suppress(Exception):
        await ws.close()
    audit.end(call_audit.END_AT_CAPACITY, active_calls=_active_calls,
              max_calls=MAX_CONCURRENT_CALLS)


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    global _active_calls
    await ws.accept()

    # No await between this check and the increment below, so the count
    # cannot be overshot by a concurrently-arriving call.
    if _active_calls >= MAX_CONCURRENT_CALLS:
        await _reject_at_capacity(ws)
        return
    _active_calls += 1

    session = CallSession(ws)
    # Every task this call creates from here on (the poll loop, each turn,
    # each keypad press) inherits this binding. It is how the process-wide
    # ClinicToolsClient knows which call an API event belongs to, without two
    # concurrent calls ever seeing each other's. See agent/call_audit.py.
    call_audit.bind(session.audit)
    logger.info("[%s] call started (%d/%d active)",
                session.call_id, _active_calls, MAX_CONCURRENT_CALLS)
    poll_task = asyncio.create_task(_turn_poll_loop(session))
    poll_task.add_done_callback(lambda t: _note_task_crash(session, "turn_poll_loop", t))
    # How the call ended, as observed here. The idle timeout overrides it via
    # session.end_reason, because that path ends the call from the inside and
    # then arrives here looking like an ordinary disconnect.
    ending = call_audit.END_CLIENT_DISCONNECT

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
    except asyncio.CancelledError:
        # The server is stopping underneath the call.
        ending = call_audit.END_AGENT_SHUTDOWN
        raise
    except Exception as e:
        logger.exception("[%s] session crashed", session.call_id)
        session.audit.error("session", e)
        ending = call_audit.END_EXCEPTION
    finally:
        poll_task.cancel()
        # Exception as well as CancelledError: a poll loop that had already
        # crashed re-raises here, and used to skip everything below it -- the
        # temp-dir cleanup, the capacity slot, and now the call's final
        # record. The crash itself was logged and recorded by _note_task_crash.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await poll_task
        session.cleanup()
        # Must be in finally, and must pair with the increment above: a slot
        # leaked on a crash path is a permanent reduction in capacity that
        # only a restart clears.
        _active_calls -= 1
        # FINALISE THE RECORD. In finally, so every exit path reaches it:
        # a hang-up, the idle timeout, a crash, a shutdown.
        status = session.audit.end(
            session.end_reason or ending,
            pending_flow=(session.pending or {}).get("awaiting"),
            language=session.lang)
        logger.info("[%s] call ended (%d/%d active) -- %s",
                    session.call_id, _active_calls, MAX_CONCURRENT_CALLS, status)


app.mount("/", StaticFiles(directory="static/pcm", html=True), name="static")
