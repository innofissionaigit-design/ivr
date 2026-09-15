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
from agent.pcm_buffer import PcmCallBuffer, SAMPLE_RATE
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from agent.asr import TurnASR
# ADDED BY SOURAV -- real production bug, reported directly by the
# caller: "why voice is giving response only in bengali... when the user
# asks in hindi aur hinglish or english." detect_language() already
# existed but was NEVER CALLED anywhere in this repo -- every reply
# function below already accepts a `language` argument (all default to
# "bengali"), but nothing ever passed anything else. See detect_language()
# below wherever it's called for the full writeup, and its own updated
# docstring in agent/bn_normalize.py for a second, real bug found and
# fixed in the function itself while wiring this in.
from agent.bn_normalize import detect_language
from agent.llm import extract_intent, ExtractionError
from agent.reply_templates import (
    missing_slot_prompt, test_rate_reply, sample_type_reply, test_duration_reply, doctor_availability_reply,
    # ADDED BY SOURAV -- "Caller asks when a doctor sits" story.
    doctor_schedule_reply,
    booking_reply, doctors_by_department_reply, booking_confirmation_prompt,
    booking_correction_prompt,
    # ADDED BY SOURAV -- "Lab Report Status & Secure Delivery" combined story.
    # delivery_declined_reply / otp_disclosure_refusal_reply are the two new
    # reply functions _dispatch_turn/_continue_pending speak directly
    # (every other new reply function is only ever reached indirectly,
    # through agent/report_flow.py's interpret_*() functions -- see that
    # module for why the decision logic itself lives there and not here).
    delivery_declined_reply, otp_disclosure_refusal_reply,
    # ADDED BY SOURAV -- "Caller asks about a health package" combined
    # with "Caller asks opening hours, address or directions".
    health_package_reply, health_packages_list_reply, clinic_info_reply,
    # ADDED BY SOURAV -- "Caller asks how to prepare for a test" story,
    # plus its bundled human_fallback config (see human_fallback_reply's
    # own module-level comment in agent/reply_templates.py for why that
    # part lives here rather than as an actual call transfer).
    test_preparation_reply, human_fallback_reply,
    # ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables (Walk-in
    # Eligibility, Prescription Requirements, Insurance Coverage Policy,
    # Outstanding Balance / Billing stories).
    walkin_eligibility_reply, prescription_requirements_reply,
    insurance_coverage_reply, billing_balance_reply,
    # ADDED BY SOURAV -- "Caller asks something the agent does not cover"
    # story. human_fallback_reply above is reused verbatim for the
    # "connect me to a human" branch -- these two are only the new
    # initial-offer and declined-offer replies.
    out_of_scope_reply, out_of_scope_counter_reply,
    # ADDED BY SOURAV -- "Caller asks two questions in one breath" story.
    # These three back _resolve_combinable_intent_fragment()'s three
    # non-fabricating fallback fragments below -- every OTHER fragment in
    # a combined reply reuses an existing single-question reply function
    # from this same import block verbatim.
    multi_intent_missing_info_reply, multi_intent_out_of_scope_reply,
    multi_intent_needs_separate_flow_reply,
    # ADDED BY SOURAV -- "Caller asks the agent to compare two options"
    # story. Renders agent/compare_flow.py's build_comparison() output --
    # see that module and this function's own docstring for the
    # arithmetic/clinical-safety design.
    compare_options_reply,
    # ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
    # previous answer" story. See agent/state.py's own module docstring
    # and this function's docstring for when this is spoken.
    ambiguous_reference_reply,
    # ADDED BY SOURAV -- "Caller asks to be called back" story. See
    # agent/callback_flow.py's module docstring and each function's own
    # docstring for when these are spoken.
    callback_unavailable_reply, callback_confirmation_prompt, callback_scheduled_reply,
)
from agent.compare_flow import build_comparison
# ADDED BY SOURAV -- "Caller asks a follow-up that depends on the previous
# answer" story. Cross-turn entity memory (pronoun/elliptical follow-up
# resolution) -- see agent/state.py's own module docstring for the full
# design and exactly which intents this applies to.
from agent.state import DialogueState, resolve_follow_up, primary_slot_for_intent, kind_for_slot
from agent.fast_path import Catalogue, FastPath
from agent.outcomes import (
    missing_booking_write_fields, insufficient_verified_information_reply,
    record_insufficient_verified_information,
    # ADDED BY SOURAV -- "Caller asks how to prepare for a test" story's
    # bundled human_fallback config -- see record_human_handoff()'s own
    # docstring in agent/outcomes.py.
    record_human_handoff,
    # ADDED BY SOURAV -- "Caller asks to be called back" story. Mirrors
    # missing_booking_write_fields() exactly -- see that function's own
    # docstring in agent/outcomes.py.
    missing_callback_write_fields,
)
# ADDED BY SOURAV -- "Caller asks to be called back" story. Pure
# availability-check/context-building logic -- see that module's own
# docstring for why "operating hours" means the clinic's own hours and why
# nothing here reads os.environ or datetime directly.
from agent.callback_flow import check_callback_availability, build_callback_reason
from agent.callback_config import CALLBACKS_ENABLED
from agent.semantic_cache import SemanticCache, embed as _embed_probe
from agent.slot_parse import (
    parse_date, parse_time, parse_phone, is_negative, is_affirmative, parse_correction_field,
    # ADDED BY SOURAV -- report_status/report_send combined story: OTP entry
    # is parsed deterministically here, never sent to the LLM or the
    # semantic cache (see agent/slot_parse.py's parse_otp() docstring and
    # RULE 9 -- the OTP must never appear in an Ollama prompt or a cache key).
    parse_otp, looks_like_otp_disclosure_request,
)
from agent.tools_client import ClinicToolsClient, ToolCallError
# ADDED BY SOURAV -- new shared module holding the actual report-flow
# DECISIONS as pure functions, so main.py and main_pcm.py (regenerated from
# this file by tools/make_pcm_variant.py) both get identical business logic
# for report_status/report_send without hand-duplicating the branching a
# second time. See agent/report_flow.py's module docstring for the full
# reasoning (it also explains the pre-existing test_sample drift this same
# story restores parity on, just below).
from agent.report_flow import (
    interpret_report_status_result, interpret_delivery_request_result,
    interpret_otp_verify_result, match_candidate_report,
)
from agent.tts import TTSClient
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

# ADDED BY SOURAV -- "Caller asks to be called back" story. Deliberately
# its OWN tuple/helper, not a repurposing of _BOOKING_FIELDS/_next_missing
# above -- those two are hard-wired to each other (see _next_missing's own
# docstring) and to book_appointment's specific 5-field shape; a request-
# callback flow needs only two fields, and giving it a separate helper
# keeps this story's blast radius off the booking flow's already-tested
# behaviour entirely. Values here are the SLOT keys ("phone" -- the same
# slot key booking/report flows already use), not the scoped `awaiting`
# strings _continue_pending uses for its pending state (those are
# "callback_time_window" and "callback_phone" -- see that function's own
# comment on why "phone" alone would collide with the booking flow's
# existing "phone" awaiting state).
_CALLBACK_FIELDS = ("callback_time_window", "phone")

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
    real traffic rather than against my assumptions about it."""
    return {
        "fast_path": _fast_path.snapshot() if _fast_path else None,
        "intent_cache": _intent_cache.snapshot() if _intent_cache else None,
        "tts_cache": _tts.snapshot() if _tts else None,
        # ADDED BY SOURAV -- reference-data TTL cache (agent/
        # reference_data_cache.py), for tuning cache_ttl_s against real
        # traffic the same way intent_cache's threshold was tuned above.
        "reference_cache": _tools.reference_cache_snapshot() if _tools else None,
    }


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
        self.call_id = uuid.uuid4().hex[:8]
        self.tmpdir = tempfile.mkdtemp(prefix=f"kcd_call_{self.call_id}_")
        self.last_activity = time.time()
        self.dispatch_lock = asyncio.Lock()
        self.processed_until_s = 0.0
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

        # ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
        # previous answer" story. A SEPARATE kind of cross-turn memory
        # from `pending` just above: `pending` tracks one IN-PROGRESS flow
        # waiting on one specific missing field; `state` tracks the last
        # entity (test/doctor/package) each already-FINISHED question was
        # actually about, so a brand-new question that refers back with a
        # pronoun ("eta-r jonno ki prescription lagbe?") can resolve
        # without making the caller repeat the name. See agent/state.py's
        # own module docstring for the full design.
        self.state = DialogueState()

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
        with contextlib.suppress(OSError):
            shutil.rmtree(self.tmpdir, ignore_errors=True)


async def _speak(session: CallSession, text_bn: str, fallback_reason: str | None = None):
    await session.send_json("AI", text_bn)
    try:
        wav = await _tts.synthesize(text_bn)
    except Exception as e:  # noqa: BLE001 - TTS is the last mile, must not raise past here
        logger.warning("[%s] TTS failed (%s) -- using fallback audio", session.call_id, e)
        wav = _tts.fallback_audio(fallback_reason or "tts_failure")

    # Close the gate BEFORE the bytes leave, never after: the client can
    # start playing the moment they land, and a poll tick that slips in
    # between send and gate is exactly the echo this prevents.
    session.hold_gate_for(_wav_duration_s(wav))
    await session.send_audio(wav)


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


def _next_missing_callback(slots: dict) -> str | None:
    """-> the scoped `awaiting` value (NOT the bare slot key -- see
    _CALLBACK_FIELDS' own comment) for the first still-empty field this
    story needs, or None once both are filled. Checks slots["phone"] (the
    real slot key) but returns "callback_phone" (the scoped awaiting
    name) so _continue_pending routes it through this story's own branch
    rather than the pre-existing booking flow's bare "phone" tail."""
    if not slots.get("callback_time_window"):
        return "callback_time_window"
    if not slots.get("phone"):
        return "callback_phone"
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


def _match_candidate_name(text: str, candidates: list[str]) -> str | None:
    """ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
    previous answer" story. Same trust model and score floor as
    _match_candidate_doctor()/agent/report_flow.py's match_candidate_report()
    just above/elsewhere -- matching a short spoken reply against a SMALL
    list just offered to the caller (here: the 2+ ambiguous names
    ambiguous_reference_reply() just spoke, from agent/state.py's
    EntitySlot.names) -- kept as its own function rather than reused
    because those two match against `{"name":..., "name_bn":...}` dicts
    while this one's candidates are already plain strings (agent/state.py
    never tracks a Bengali alias, only the catalogue's own canonical
    name -- see that module's own docstring)."""
    if not candidates:
        return None
    norm_text = text.strip().lower()
    if not norm_text:
        return None
    best_name, best_score = None, 0.0
    for candidate in candidates:
        form_l = candidate.lower()
        score = difflib.SequenceMatcher(None, form_l, norm_text).ratio()
        if form_l in norm_text or norm_text in form_l:
            score = max(score, 0.85)
        if score > best_score:
            best_name, best_score = candidate, score
    return best_name if best_score >= 0.55 else None


# Bare "নাম বলছি" prefixes a caller sometimes leads a name with. Stripped
# rather than relied upon -- most callers just say the name on its own.
_NAME_PREFIXES = ("আমার নাম ", "নাম ", "আমি ")


def _clean_patient_name(text: str) -> str | None:
    """Strip at most ONE leading filler phrase off a caller's spoken
    patient name, e.g. "আমার নাম রাহুল সেন" -> "রাহুল সেন".

    Bug fixed here: this used to re-check ALL of _NAME_PREFIXES in a
    plain `for` loop with no `break`, testing each prefix against the
    ALREADY-stripped text from the previous iteration. Real disfluent
    speech (or ASR output) that happens to start with more than one
    filler phrase in a row -- e.g. "নাম আমি সেন" ("name -- I'm Sen") --
    walked through BOTH matching prefixes one after another
    ("নাম আমি সেন" -> strip "নাম " -> "আমি সেন" -> strip "আমি " -> "সেন"),
    silently eating the caller's first name along with the filler words
    and leaving only the surname. Stopping after the first match means
    at most one filler phrase is ever removed -- the rest of whatever
    the caller said, first name included, is left alone."""
    t = text.strip().strip("।!?., ")
    if not t:
        return None
    for prefix in _NAME_PREFIXES:
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
            break
    return t or None


async def _finish_booking(session: CallSession, slots: dict, language: str = "bengali"):
    """Place the booking and clear pending regardless of outcome.

    UPDATED BY SOURAV -- `language` is new (default "bengali" so every
    pre-existing caller of this function keeps working unchanged): see
    the module-level detect_language import comment above for why this
    is threaded through now. Passed straight to booking_reply() and
    insufficient_verified_information_reply() below, which already
    accepted it and already had full 4-language bodies -- nothing in
    either function needed to change, only this call site.

    The ONLY caller of this function is the "confirm_booking" branch of
    _continue_pending, below -- every path that fills the 5th booking
    field (multi-turn collection, the single-shot book_appointment
    branch in _dispatch_turn, doctor/department flows that feed into
    it) now routes through a spoken readback and an explicit
    affirmative first (Answer Quality and Grounding: "every critical
    value is read back before it is used"). Failure here is reported the
    same way the old single-shot book_appointment branch reported it
    (tool_failure fallback audio).

    "The agent says it cannot confirm rather than guessing": a
    success=True response is trusted only after confirming
    confirmation_id/date/time_slot actually came back non-empty. This is
    the ONE real trigger that story wires up -- deliberately just a
    presence check, not shape validation, not a business-rule check, not
    a database re-query (see agent/outcomes.py's module docstring for
    exactly which sibling stories those belong to instead). A caller is
    never read a confirmation number the code cannot itself verify it
    received."""
    session.pending = None
    try:
        result = await _tools.book_appointment(
            slots["doctor_name"], slots["date"], slots["time_slot"],
            slots["patient_name"], slots["phone"],
        )
    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                     fallback_reason="tool_failure")
        return

    if result.get("success"):
        missing = missing_booking_write_fields(result)
        if missing:
            logger.error("[%s] booking reported success but missing %s -- withholding confirmation",
                         session.call_id, missing)
            record_insufficient_verified_information(
                intent="book_appointment", field=",".join(missing),
                reason="missing_after_success", call_id=session.call_id,
            )
            await _speak(session, insufficient_verified_information_reply(language=language),
                         fallback_reason="insufficient_verified_information")
            return

    await _speak(session, booking_reply(slots, result, language=language))


# ADDED BY SOURAV -- "Caller asks to be called back" story. Same shape as
# _finish_booking() just above: place the write and clear pending
# regardless of outcome, catch ToolCallError for the infra-apology path,
# and withhold the confirmation via missing_callback_write_fields() rather
# than ever speaking a callback_id the response didn't actually confirm
# (mirrors _finish_booking()'s own missing_booking_write_fields() check --
# see agent/outcomes.py for both). The ONLY caller is the "confirm_callback"
# branch of _continue_pending, below.
async def _finish_callback(session: CallSession, slots: dict, language: str = "bengali"):
    session.pending = None
    try:
        result = await _tools.request_callback(
            slots["phone"], slots["callback_time_window"], slots.get("callback_reason"),
        )
    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                     fallback_reason="tool_failure")
        return

    if result.get("success"):
        missing = missing_callback_write_fields(result)
        if missing:
            logger.error("[%s] callback request reported success but missing %s -- withholding confirmation",
                         session.call_id, missing)
            record_insufficient_verified_information(
                intent="request_callback", field=",".join(missing),
                reason="missing_after_success", call_id=session.call_id,
            )
            await _speak(session, insufficient_verified_information_reply(language=language),
                         fallback_reason="insufficient_verified_information")
            return

    await _speak(session, callback_scheduled_reply(slots, result, language=language))


# ADDED BY SOURAV -- "Lab Report Status & Secure Delivery" combined story
# (previously two separate stories, "is my report ready" / "send my
# report"). These two helpers are the only NEW glue _dispatch_turn and
# _continue_pending need: every actual decision (what to say, what pending
# state comes next) lives in agent/report_flow.py's pure interpret_*()
# functions -- see that module's docstring for why. These two functions
# exist only to do the I/O those pure functions cannot do themselves:
# await the tools client, then hand the response to the right interpret_*()
# call and speak/store whatever it returns.
async def _finish_report_flow(session: CallSession, phone: str, result: dict, flow: str,
                               language: str = "bengali"):
    """Common tail for BOTH a fresh report_status/report_send lookup and a
    caller resolving a "which report?" disambiguation (see the
    "which_report" pending state below, which reconstructs a `result`
    locally from the remembered candidate list rather than re-querying).

    UPDATED BY SOURAV -- `language` is new (default "bengali", so every
    pre-existing caller of this function keeps working unchanged); see
    the module-level detect_language import comment above. Passed
    straight to agent/report_flow.py's interpret_*() functions below,
    which already accepted it -- only this call site needed updating."""
    text, pending = interpret_report_status_result(result, flow, language=language)
    if pending and pending.get("awaiting") == "__request_delivery_now__":
        # flow == "report_send" on a READY + delivery-enabled report: the
        # caller already asked for delivery, so go straight to requesting
        # an OTP rather than asking "shall I send it?" first (that offer
        # question is only for flow == "report_status").
        report_number = pending["report_number"]
        try:
            delivery_result = await _tools.request_report_delivery(phone, report_number)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                         fallback_reason="tool_failure")
            return
        text, pending = interpret_delivery_request_result(delivery_result, report_number, language=language)
    if pending is not None:
        # phone is never something the caller re-supplies mid-flow (RULE 15
        # -- identity was already resolved) -- carry it forward on every
        # pending dict this story introduces so later states never need to
        # re-ask for it.
        pending["phone"] = phone
    session.pending = pending
    if text:
        await _speak(session, text)


async def _handle_report_lookup(session: CallSession, phone: str, test_name: str | None, flow: str,
                                 language: str = "bengali"):
    """Entry point for BOTH the report_status and report_send intents (see
    _dispatch_turn below) once a phone number is in hand, and for the
    "phone" pending state once a caller who was first asked for one gives
    it. `flow` tells report_status and report_send apart -- same lookup,
    different thing to do once a READY+enabled report is found (see
    agent/report_flow.py's interpret_report_status_result).

    UPDATED BY SOURAV -- `language` is new (default "bengali", so every
    pre-existing caller keeps working unchanged); threaded straight
    through to _finish_report_flow. See the module-level detect_language
    import comment above."""
    try:
        result = await _tools.get_report_status(phone, test_name)
    except ToolCallError as e:
        logger.error("[%s] clinic API call failed: %s", session.call_id, e)
        session.pending = None
        await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                     fallback_reason="tool_failure")
        return
    await _finish_report_flow(session, phone, result, flow, language=language)


async def _resolve_comparable_entity(name: str) -> dict:
    """ADDED BY SOURAV -- "Caller asks the agent to compare two options"
    story. agent/llm.py deliberately never classifies whether a caller-
    named term is a TEST or a PACKAGE (see its own CLINICAL SAFETY NOTE
    and the "compare_option_a"/"compare_option_b" slot rule) -- it only
    ever copies the literal span the caller said, same discipline as
    "test_name"/"package_name" elsewhere in this file. Resolving WHICH
    catalogue a name belongs to is this function's only job, done the
    same way a human clinic-desk operator would: try the test catalogue
    first (get_test_rate), and only if that comes back not-found, try the
    package catalogue (search_health_package) -- tests significantly
    outnumber packages in this clinic's catalogue, so this order resolves
    the common case (comparing two tests) in a single tool call.

    Returns the underlying tool response dict, unmodified, plus one added
    key `"kind"`: "test" | "package" | "not_found" -- read by
    agent/compare_flow.py's build_comparison() (which fields it reads
    depends on this tag) and agent/reply_templates.py's
    compare_options_reply() (which alias field to prefer). Never raises
    ToolCallError itself -- a caller of this function (the compare_options
    dispatch branch below) awaits it for BOTH sides before deciding how to
    handle a tool failure, same as every other two-tool-call intent in
    this file.
    """
    test_result = await _tools.get_test_rate(name)
    if test_result.get("found"):
        return {**test_result, "kind": "test"}
    package_result = await _tools.search_health_package(name)
    if package_result.get("found"):
        return {**package_result, "kind": "package"}
    # Neither catalogue has it -- prefer the package lookup's own
    # did_you_mean suggestions (already computed by clinic-api) since a
    # caller who names something unfamiliar in a comparison is at least as
    # likely to mean a package as a plain test; the test lookup's own
    # did_you_mean is not lost, just not the one used here, since this
    # dict only needs to say "not found", not carry both catalogues' near
    # matches.
    return {**package_result, "kind": "not_found"}


# ADDED BY SOURAV -- "Caller asks a follow-up that depends on the previous
# answer" story. Which result-dict field carries the CATALOGUE's own
# canonical name for each trackable slot -- see _remember_primary_entity()
# below for why the canonical name, not the caller's raw words, is what
# gets remembered.
_CANONICAL_NAME_FIELD = {"test_name": "test_name", "doctor_name": "doctor_name", "package_name": "package_name"}


def _session_state(session: CallSession) -> DialogueState | None:
    """ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
    previous answer" story. Every REAL CallSession always has `.state`
    (see its own __init__), but a large number of EXISTING tests written
    before this story build a lightweight `types.SimpleNamespace` fake
    session instead -- with only the specific attributes that story
    needed at the time, never `.state`. Rather than retrofit `.state=...`
    into every one of those pre-existing fakes (an unrelated change to
    18+ test files this story has no reason to touch), every call site
    below goes through this helper and treats "no `.state` attribute at
    all" the same as "follow-up resolution is simply not available this
    turn" -- the exact behaviour those tests already expect and pass
    with today, completely unaffected by this story. A real caller,
    which always has `.state`, is never affected by this fallback.
    """
    return getattr(session, "state", None)


def _remember_primary_entity(session: CallSession, intent: str, slots: dict, result: dict | None) -> None:
    """Called right after a single-primary-entity intent's lookup
    returns (see agent/state.py's primary_slot_for_intent() for exactly
    which intents this applies to), successful or not.

    Only a `result.get("found")` lookup updates agent/state.py's
    DialogueState -- and even then with the CATALOGUE's own canonical
    name (e.g. clinic-api's own `test_name`), never the caller's raw
    spoken words: a later follow-up backfills THIS value straight into
    the next tool call's argument (see main.py's dispatch, right after
    resolve_follow_up()), and the canonical name is guaranteed to still
    match on that next lookup the way an ASR-mangled or partial spoken
    form is not. A not-found result intentionally changes nothing --
    nothing new was actually confirmed to exist this turn, so clobbering
    a still-valid, previously-tracked entity with a miss would make the
    NEXT follow-up resolve to nothing instead of the last real one.
    """
    state = _session_state(session)
    if state is None:
        return
    slot_key = primary_slot_for_intent(intent)
    if slot_key is None or not result or not result.get("found"):
        return
    kind = kind_for_slot(slot_key)
    if kind is None:
        return
    canonical = result.get(_CANONICAL_NAME_FIELD[slot_key]) or slots.get(slot_key)
    if canonical:
        state.mark(kind, canonical)


def _remember_compared_entities(session: CallSession, entity_a: dict, entity_b: dict) -> None:
    """Called after compare_options resolves both sides (dispatch branch
    and its "compare_options_slot" continuation below both call this).

    If both sides turned out to be the SAME kind (two tests, or two
    packages) with DIFFERENT canonical names, that kind becomes AMBIGUOUS
    for the next turn's follow-up (see agent/state.py's mark_ambiguous())
    -- a caller who just asked to compare CBC and Lipid Profile, then
    says "does IT need a prescription?", cannot honestly have either one
    guessed. If both sides are the same kind with the SAME name (a caller
    comparing a test to itself), or only one side resolved to that kind
    at all, there is only one real candidate -- mark() as usual. A
    not_found side never contributes anything to remember, same
    not-found-changes-nothing rule as _remember_primary_entity() above.
    """
    state = _session_state(session)
    if state is None:
        return
    by_kind: dict[str, list[str]] = {}
    for entity in (entity_a, entity_b):
        kind = entity.get("kind")
        if kind not in ("test", "package"):
            continue
        # "test_name" or "package_name" -- the exact field
        # _resolve_comparable_entity() tags each result with.
        name = entity.get(f"{kind}_name")
        if name:
            by_kind.setdefault(kind, []).append(name)
    for kind, names in by_kind.items():
        distinct = list(dict.fromkeys(names))  # de-dupe, order-preserved
        if len(distinct) > 1:
            state.mark_ambiguous(kind, distinct)
        else:
            state.mark(kind, distinct[0])


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
                    | "patient_name" | "phone" | "confirm_booking" | "confirm_correction",
        "slots": {<whatever of the 5 booking fields is already known>},
        "candidates": [{"name", "name_bn"}, ...] | None,  # only for "doctor_choice"
        "offered_date": "<iso>" | None,  # the date main.py already SPOKE to
                                          # the caller ("today", or a
                                          # next-available date) -- lets a
                                          # bare "হ্যাঁ" confirm THAT date
                                          # instead of literally "today"
        "retries": int,
    }

    ADDED BY SOURAV -- "Lab Report Status & Secure Delivery" combined story
    adds FOUR more "awaiting" values, handled in their own block below
    (checked BEFORE the universal "না" escape hatch, same reason
    "confirm_booking"/"confirm_correction" already are: that hatch's
    booking-specific wording would be wrong mid-report-flow, and RULE 4/9
    need their own "না means decline delivery, not abandon the call"
    wording and their own OTP-disclosure-attempt handling instead):
        "report_phone"     -- report_status/report_send asked for a phone
                               number (RULE 15, identity-by-phone-first);
                               pending also carries "flow" and "test_name".
                               NOT plain "phone" -- that string is already
                               used by the booking flow below (a caller
                               correcting a booking's phone number
                               re-enters awaiting="phone"); a shared name
                               made a correcting-a-booking's-phone-number
                               caller get routed into the report flow
                               instead (caught by test_booking_readback.py).
        "which_report"     -- RULE 13, caller has more than one report and
                               was asked which; pending carries "flow" and
                               the remembered "candidates" list.
        "confirm_delivery" -- the "shall I send it to your phone?" offer
                               after a report_status lookup found a
                               READY + delivery-enabled report (RULE 4);
                               pending carries "report_number" and "phone".
        "otp_code"         -- RULE 4-9, waiting for the caller to speak
                               back the OTP just sent; pending carries
                               "report_number" and "phone". A caller who
                               asks to be TOLD the otp instead of speaking
                               it back is refused (RULE 9 / ATTACK 8) via
                               looks_like_otp_disclosure_request(), not
                               treated as an ordinary unparseable reply.
    All four pending dicts also carry "phone" (see _finish_report_flow /
    _handle_report_lookup above) so none of these states ever needs to
    re-ask for a phone number it already resolved identity with.

    ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables adds TWO
    more "awaiting" values:
        "billing_phone"          -- billing_balance asked for a phone
                                     number (RULE 14/15, same as
                                     "report_phone" above). Own distinct
                                     string for the same reason
                                     "report_phone" isn't just "phone".
        "insurance_coverage_slot" -- insurance_coverage is missing
                                     test_name and/or insurance_provider_
                                     name; pending also carries "slots"
                                     (whichever of the two is already
                                     known) and "missing_field" (which one
                                     this turn's reply fills). Unlike
                                     phone/date/time_slot, both fields are
                                     free-text named entities with no
                                     local grammar -- the caller's
                                     utterance is accepted verbatim for
                                     whichever field is missing, same as
                                     agent/llm.py's own extraction rule
                                     for these two slots ("copy the term
                                     as said, do not normalize").

    ADDED BY SOURAV -- "Caller asks something the agent does not cover"
    story adds ONE more "awaiting" value:
        "out_of_scope_choice" -- the caller was offered a choice (connect
                                  to a human, or contact the counter
                                  themselves) after an "out_of_scope"
                                  intent; carries no lookup state at all
                                  (unlike confirm_delivery above, there is
                                  no report_number/phone to remember), just
                                  "retries", same yes/no/unparseable shape
                                  as confirm_delivery.

    ADDED BY SOURAV -- "Caller asks the agent to compare two options"
    story adds ONE more "awaiting" value:
        "compare_options_slot" -- compare_options is missing
                                   compare_option_a and/or compare_option_b;
                                   pending also carries "slots" (whichever
                                   of the two is already known) and
                                   "missing_field" (which one this turn's
                                   reply fills). Same free-text, no-local-
                                   grammar, accept-verbatim shape as
                                   "insurance_coverage_slot" above, for the
                                   identical reason -- neither slot has a
                                   local grammar to parse against, and
                                   agent/llm.py never classifies which one
                                   is a test versus a package anyway (that
                                   happens downstream, in
                                   _resolve_comparable_entity(), only once
                                   both names are in hand).

    ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
    previous answer" story adds ONE more "awaiting" value:
        "follow_up_clarification" -- a brand-new question's primary
                                       entity slot (test/doctor/package)
                                       was left null AND agent/state.py's
                                       tracked state for that kind was
                                       AMBIGUOUS (2+ different entities
                                       discussed a moment ago -- see that
                                       module's own docstring); pending
                                       carries "intent" and "slots" (the
                                       question to RESUME, exactly as
                                       extracted, once the ambiguity is
                                       resolved), "kind", and "candidates"
                                       (the names ambiguous_reference_
                                       reply() just spoke). Unlike every
                                       other pending state above, this one
                                       does not ask for a brand-new piece
                                       of information the caller never
                                       gave -- it asks them to pick which
                                       of two things they ALREADY said a
                                       moment ago they meant, so the reply
                                       is matched against `candidates`
                                       (via _match_candidate_name(), same
                                       trust model as _match_candidate_
                                       doctor()/agent/report_flow.py's
                                       match_candidate_report()) rather
                                       than accepted verbatim the way
                                       insurance_coverage_slot/compare_
                                       options_slot's free-text fields
                                       are -- a stray word or two around
                                       the real name should not silently
                                       fail to match.

    ADDED BY SOURAV -- "Caller asks to be called back" story adds THREE
    more "awaiting" values:
        "callback_time_window" -- request_callback is missing a time
                                   window; free-text, accepted verbatim
                                   (same as insurance_coverage_slot/
                                   compare_options_slot above -- a time
                                   window like "this evening" has no local
                                   grammar to parse against).
        "callback_phone"       -- request_callback is missing a phone
                                   number. NOT plain "phone" -- same
                                   collision reasoning as "report_phone"/
                                   "billing_phone" above, since the
                                   booking flow's own shared tail further
                                   down already owns bare "phone".
        "confirm_callback"     -- both fields are known; the caller is
                                   read back the number and time window
                                   and asked to confirm before the write
                                   (Answer Quality and Grounding, same
                                   shape as "confirm_booking" above).
    All three carry "slots" (whatever of "callback_time_window"/"phone"/
    "callback_reason" is already known) -- see agent/callback_flow.py's
    own module docstring for the availability check that runs BEFORE any
    of these three states is ever entered, and _next_missing_callback()
    just above _continue_pending's own definition for the field order.

    Returns True when the turn was fully handled here (caller must not
    also run intent extraction on top of it); False to fall through to
    the normal pipeline -- either because there was no pending flow, or
    because this one gave up on it after repeated unparseable replies.
    """
    pending = session.pending
    if pending is None:
        return False

    # ADDED BY SOURAV -- see the module-level detect_language import
    # comment above for the real bug this fixes. Detected fresh from
    # THIS turn's own utterance (not carried over from an earlier turn,
    # and not stored on `pending`) -- a caller can code-switch mid-call,
    # and every reply below should reflect what they just said, not what
    # they said several turns ago when the flow started.
    language = detect_language(text)

    awaiting = pending["awaiting"]

    # Answer Quality and Grounding: "every critical value is read back
    # before it is used." Handled BEFORE the universal "না" == abandon-
    # the-whole-booking hatch just below, on purpose -- a caller who says
    # "না" here is rejecting one misheard value, not hanging up on the
    # appointment, and folding the two together would make a correction
    # indistinguishable from an abandonment.
    if awaiting == "confirm_booking":
        if is_affirmative(text):
            await _finish_booking(session, pending["slots"], language=language)
            return True
        if is_negative(text):
            # AC: "opens a correction path rather than repeating the
            # prompt" -- ask a DIFFERENT question (which field?) instead
            # of reading the same five values back again.
            session.pending = {
                "awaiting": "confirm_correction", "slots": dict(pending["slots"]),
                "candidates": None, "offered_date": pending.get("offered_date"), "retries": 0,
            }
            await _speak(session, booking_correction_prompt(language=language))
            return True
        # Neither a clear yes nor a clear no -- bounded retries of the
        # SAME confirmation (unlike a rejection, an unparseable reply
        # hasn't told us anything is actually wrong yet), then give up on
        # the flow same as every other awaiting-state below.
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None
            return False
        await _speak(session, booking_confirmation_prompt(pending["slots"], language=language))
        return True

    if awaiting == "confirm_correction":
        field = parse_correction_field(text)
        if field is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, booking_correction_prompt(language=language))
            return True
        # Drop just the disputed field and re-enter the ordinary
        # single-field flow at it -- NOT a restart of all five, which is
        # exactly the "repeating the prompt" the AC rules out.
        slots = dict(pending["slots"])
        slots.pop(field, None)
        session.pending = {
            "awaiting": field, "slots": slots, "candidates": None,
            "offered_date": pending.get("offered_date"), "retries": 0,
        }
        await _speak(session, missing_slot_prompt("book_appointment", field, language=language))
        return True

    # ADDED BY SOURAV -- report_status/report_send combined story's four new
    # pending states. Checked here, BEFORE the universal booking escape
    # hatch just below, for the same reason "confirm_booking"/
    # "confirm_correction" already are (see this function's docstring):
    # a "না" here means something specific to a report flow, not
    # "abandon the appointment" (there is no appointment in this flow).
    #
    # "report_phone", NOT "phone": the pre-existing booking flow already
    # uses the bare string "phone" as an awaiting value (see the shared
    # date/time_slot/patient_name/phone tail further down, and its
    # "confirm_correction" re-entry) -- a caller correcting a BOOKING's
    # phone number was briefly being routed into this report-flow handler
    # instead, because this check ran first and matched on the same
    # string. Caught by test_booking_readback.py's correction-round-trip
    # test failing once these states were wired in. See
    # agent/report_flow.py's AWAITING_REPORT_PHONE comment for the same
    # note from the other side of this collision.
    if awaiting == "report_phone":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        phone = parse_phone(text)
        if phone is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt(pending["flow"], "phone", language=language))
            return True
        await _handle_report_lookup(session, phone, pending.get("test_name"), pending["flow"],
                                     language=language)
        return True

    # ADDED BY SOURAV -- Phase 1: Database Schema & Policy Tables.
    # Outstanding Balance / Billing story. Own distinct string, NOT
    # "phone" or "report_phone" -- same collision reasoning as
    # "report_phone" above: a caller correcting a BOOKING's phone number,
    # or resuming a report flow's phone ask, must never be routed here
    # instead just because the bare string matched.
    if awaiting == "billing_phone":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        phone = parse_phone(text)
        if phone is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("billing_balance", "phone", language=language))
            return True
        # FIXED BY SOURAV -- Phase 2 end-to-end testing caught a real bug
        # here: this branch spoke the real answer but never cleared
        # session.pending, so the call stayed stuck in "awaiting a phone
        # number" afterward -- the caller's NEXT utterance, whatever it
        # was, would have been misinterpreted as another phone attempt
        # instead of a fresh question. Must be cleared BEFORE the tool
        # call, same ordering as the insurance_coverage_slot branch below,
        # so a slow/failing tool call never leaves pending in a stale
        # state either.
        session.pending = None
        result = await _tools.get_patient_billing(phone)
        await _speak(session, billing_balance_reply(result, language=language))
        return True

    # ADDED BY SOURAV -- Phase 1: Insurance Coverage Policy story. Unlike
    # phone/date/time_slot, "test_name" and "insurance_provider_name" are
    # free-text named entities with no local grammar to parse (per agent/
    # llm.py's own slot rule: copy the term as said, do not normalize --
    # the actual alias/fuzzy matching happens downstream in clinic-api's
    # _find_lab_test()/_find_insurance_provider()), so accepting the
    # caller's utterance verbatim for whichever field pending["missing_
    # field"] names IS the correct local equivalent of the LLM's own
    # extraction rule for these two slots, not a shortcut.
    if awaiting == "insurance_coverage_slot":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        value = text.strip()
        if not value:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("insurance_coverage", pending["missing_field"],
                                                        language=language))
            return True
        pending["slots"][pending["missing_field"]] = value
        pending["retries"] = 0
        still_missing = next(
            (f for f in ("test_name", "insurance_provider_name") if not pending["slots"].get(f)), None,
        )
        if still_missing:
            pending["missing_field"] = still_missing
            await _speak(session, missing_slot_prompt("insurance_coverage", still_missing, language=language))
            return True
        final_slots = pending["slots"]
        session.pending = None
        result = await _tools.get_insurance_coverage(
            final_slots["test_name"], final_slots["insurance_provider_name"],
        )
        await _speak(session, insurance_coverage_reply(final_slots, result, language=language))
        return True

    if awaiting == "compare_options_slot":
        # Mirrors "insurance_coverage_slot" immediately above exactly --
        # same two-free-text-slots shape, same accept-verbatim discipline
        # (agent/llm.py never classifies test-vs-package for these two
        # slots either; see this file's own _resolve_comparable_entity()
        # for where that resolution actually happens, only once both
        # names are known).
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        value = text.strip()
        if not value:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("compare_options", pending["missing_field"],
                                                        language=language))
            return True
        pending["slots"][pending["missing_field"]] = value
        pending["retries"] = 0
        still_missing = next(
            (f for f in ("compare_option_a", "compare_option_b") if not pending["slots"].get(f)), None,
        )
        if still_missing:
            pending["missing_field"] = still_missing
            await _speak(session, missing_slot_prompt("compare_options", still_missing, language=language))
            return True
        final_slots = pending["slots"]
        session.pending = None
        name_a, name_b = final_slots["compare_option_a"], final_slots["compare_option_b"]
        entity_a, entity_b = await _resolve_comparable_entity(name_a), await _resolve_comparable_entity(name_b)
        comparison = build_comparison(entity_a, entity_b)
        await _speak(session, compare_options_reply(name_a, name_b, entity_a, entity_b, comparison, language=language))
        _remember_compared_entities(session, entity_a, entity_b)
        return True

    if awaiting == "follow_up_clarification":
        # ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
        # previous answer" story. Unlike every OTHER pending state above,
        # this is not asking for a brand-new piece of information -- it
        # is asking the caller to pick which of two things they ALREADY
        # named a moment ago they meant (see agent/state.py's own
        # docstring on ambiguity, and this function's docstring above),
        # so the reply is matched against the small `candidates` list
        # rather than accepted verbatim.
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        matched = _match_candidate_name(text, pending["candidates"])
        if not matched:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, ambiguous_reference_reply(pending["kind"], pending["candidates"], language=language))
            return True
        session.pending = None
        # The caller just resolved the ambiguity themselves -- collapsing
        # the tracked state back down to this one name means the NEXT
        # follow-up (AC3's "three consecutive follow-ups without
        # re-prompting") does not need to ask again.
        state = _session_state(session)
        if state is not None:
            state.mark(pending["kind"], matched)
        resolved_slots = dict(pending["slots"])
        resolved_slots[primary_slot_for_intent(pending["intent"])] = matched
        # Reuses the exact same per-intent lookup-and-reply logic a solo
        # turn for this intent would use -- see that function's own
        # docstring; every intent agent/state.py can ever produce an
        # ambiguous_kind for is one _resolve_combinable_intent_fragment
        # already knows how to answer.
        reply = await _resolve_combinable_intent_fragment(pending["intent"], resolved_slots, language)
        if reply:
            await _speak(session, reply)
        return True

    # ADDED BY SOURAV -- "Caller asks to be called back" story. Three new
    # scoped states: "callback_time_window" and "callback_phone" (NOT
    # "phone" -- same collision reasoning as "report_phone"/"billing_phone"
    # above, since the pre-existing booking flow's own shared tail further
    # down already treats bare "phone" as ITS awaiting value) collect the
    # two fields this story needs, then "confirm_callback" reads them back
    # before the write -- same "every critical value is read back before
    # it is used" discipline as "confirm_booking" above.
    if awaiting == "callback_time_window":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        window = text.strip()
        if not window:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("request_callback", "callback_time_window", language=language))
            return True
        pending["slots"]["callback_time_window"] = window
        pending["retries"] = 0
        missing = _next_missing_callback(pending["slots"])
        if missing is None:
            pending["awaiting"] = "confirm_callback"
            await _speak(session, callback_confirmation_prompt(pending["slots"], language=language))
            return True
        pending["awaiting"] = missing
        await _speak(session, missing_slot_prompt("request_callback", missing, language=language))
        return True

    if awaiting == "callback_phone":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        phone = parse_phone(text)
        if phone is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("request_callback", "callback_phone", language=language))
            return True
        pending["slots"]["phone"] = phone
        pending["retries"] = 0
        missing = _next_missing_callback(pending["slots"])
        if missing is None:
            pending["awaiting"] = "confirm_callback"
            await _speak(session, callback_confirmation_prompt(pending["slots"], language=language))
            return True
        pending["awaiting"] = missing
        await _speak(session, missing_slot_prompt("request_callback", missing, language=language))
        return True

    if awaiting == "confirm_callback":
        if is_affirmative(text):
            await _finish_callback(session, pending["slots"], language=language)
            return True
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        # Neither a clear yes nor a clear no -- bounded retries of the
        # SAME confirmation, same posture as "confirm_booking" above.
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None
            return False
        await _speak(session, callback_confirmation_prompt(pending["slots"], language=language))
        return True

    if awaiting == "which_report":
        if is_negative(text):
            session.pending = None
            await _speak(session, "ঠিক আছে, তাহলে থাক। আর কিছু জানতে চান?")
            return True
        candidates = pending.get("candidates") or []
        report_number = match_candidate_report(text, candidates)
        if report_number is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, "দুঃখিত, কোন টেস্টের রিপোর্টের কথা বলছেন, আরেকটু স্পষ্ট করে বলবেন?")
            return True
        # Reconstruct a report_status-shaped result LOCALLY from the
        # candidate the caller just picked, rather than re-querying
        # clinic-api a second time -- the candidate list came from that
        # same lookup moments ago and already carries every field
        # interpret_report_status_result needs (status, delivery_enabled,
        # report_number, test_name). Considered tradeoff, not an oversight:
        # a report's status could in principle change in the few seconds
        # between the ambiguous listing and this answer, same as any
        # read-then-act gap; request_report_delivery() re-checks
        # eligibility server-side regardless (RULE 3/16 defense in depth),
        # so this can never cause an unauthorized delivery, only a stale
        # status read in an already-rare multi-report case.
        chosen = next(c for c in candidates if c["report_number"] == report_number)
        result = {"patient_found": True, "found": True, **chosen}
        await _finish_report_flow(session, pending.get("phone"), result, pending["flow"],
                                   language=language)
        return True

    if awaiting == "confirm_delivery":
        if is_affirmative(text):
            phone = pending.get("phone")
            report_number = pending["report_number"]
            try:
                delivery_result = await _tools.request_report_delivery(phone, report_number)
            except ToolCallError as e:
                logger.error("[%s] clinic API call failed: %s", session.call_id, e)
                session.pending = None
                await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                             fallback_reason="tool_failure")
                return True
            text_out, new_pending = interpret_delivery_request_result(
                delivery_result, report_number, language=language)
            if new_pending is not None:
                new_pending["phone"] = phone
            session.pending = new_pending
            await _speak(session, text_out)
            return True
        if is_negative(text):
            session.pending = None
            await _speak(session, delivery_declined_reply(language=language))
            return True
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None
            return False
        await _speak(session, "রিপোর্টটা কি আপনার ফোনে পাঠাব?")
        return True

    if awaiting == "otp_code":
        if is_negative(text):
            session.pending = None
            await _speak(session, delivery_declined_reply(language=language))
            return True
        otp = parse_otp(text)
        if otp is None:
            # RULE 9 / ATTACK 8: checked only AFTER parse_otp() already
            # failed on this same utterance, so "the otp is 482913" (which
            # DOES contain the word "otp" but is also a valid code) is
            # handled as a normal OTP attempt above, never misclassified
            # as a disclosure request.
            if looks_like_otp_disclosure_request(text):
                await _speak(session, otp_disclosure_refusal_reply(language=language))
                return True
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, "দুঃখিত, ওটিপিটা ঠিকমতো বুঝতে পারিনি, আবার বলবেন?")
            return True
        phone = pending.get("phone")
        report_number = pending["report_number"]
        try:
            result = await _tools.verify_report_otp(phone, report_number, otp)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            session.pending = None
            await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                         fallback_reason="tool_failure")
            return True
        text_out, new_pending = interpret_otp_verify_result(result, report_number, language=language)
        if new_pending is not None:
            new_pending["phone"] = phone
        session.pending = new_pending
        await _speak(session, text_out)
        return True

    # ADDED BY SOURAV -- "Caller asks something the agent does not cover"
    # story. Checked BEFORE the universal "না" escape hatch just below,
    # same reason confirm_booking/confirm_delivery/etc. all are: that
    # hatch's fixed "appointment bad thak" wording would be wrong here (no
    # appointment was ever in progress), and a plain "না" in this state
    # means "no, I'll contact the counter myself" -- a real, distinct
    # answer with its own reply, not an abandonment.
    if awaiting == "out_of_scope_choice":
        if is_affirmative(text):
            session.pending = None
            # Same honest "logged for a human, no real transfer capability"
            # handling as the "unclear" intent branch above -- see
            # agent/outcomes.record_human_handoff()'s own docstring.
            # intent="out_of_scope" here (not "unclear") so the two stay
            # distinguishable in the shared escalation ledger.
            record_human_handoff("out_of_scope", call_id=session.call_id)
            await _speak(session, human_fallback_reply(language=language))
            return True
        if is_negative(text):
            session.pending = None
            await _speak(session, out_of_scope_counter_reply(language=language))
            return True
        pending["retries"] += 1
        if pending["retries"] > 2:
            session.pending = None
            return False  # give a fresh LLM classification a chance instead
        await _speak(session, out_of_scope_reply(language=language))
        return True

    # Universal escape hatch, checked before any field-specific parsing:
    # a caller mid-flow who says "না" / "থাক" is abandoning the booking,
    # not answering whichever question was pending.
    if is_negative(text):
        session.pending = None
        await _speak(session, "ঠিক আছে, অ্যাপয়েন্টমেন্ট বাদ থাক। আর কিছু জানতে চান?")
        return True

    if awaiting == "doctor_choice":
        match = _match_candidate_doctor(text, pending.get("candidates") or [])
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

        await _speak(session, doctor_availability_reply({"doctor_name": match}, result, language=language))

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
        if value is None:
            pending["retries"] += 1
            if pending["retries"] > 2:
                session.pending = None
                return False
            await _speak(session, missing_slot_prompt("doctors_by_department", "date", language=language))
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

        await _speak(session, doctors_by_department_reply({"department": department}, result, language=language))

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
        await _speak(session, missing_slot_prompt("book_appointment", awaiting, language=language))
        return True

    pending["slots"][awaiting] = value
    pending["retries"] = 0
    missing = _next_missing(pending["slots"])
    if missing is None:
        # All 5 fields known -- read them back and wait for an explicit
        # affirmative (see the "confirm_booking" branch above) instead of
        # writing immediately.
        pending["awaiting"] = "confirm_booking"
        await _speak(session, booking_confirmation_prompt(pending["slots"], language=language))
        return True
    pending["awaiting"] = missing
    await _speak(session, missing_slot_prompt("book_appointment", missing, language=language))
    return True


async def _dispatch_turn(session: CallSession, utterance_wav: str):
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

        # ADDED BY SOURAV -- production bug: replies were spoken only in
        # Bengali no matter what language the caller actually used (see
        # detect_language()'s own docstring in agent/bn_normalize.py for
        # the full writeup). Detected fresh from THIS turn's own utterance
        # -- not carried over from a previous turn -- since a caller can
        # code-switch mid-call and every reply should reflect what they
        # just said. Every reply_templates.py function below already
        # accepted a language= argument; it was simply never passed.
        language = detect_language(text)

        # A booking (or the doctor-choice / date-confirm step just before
        # one) already in progress owns this turn -- see _continue_pending's
        # docstring for why intent extraction must NOT also run on top of it.
        if await _continue_pending(session, text):
            return

        try:
            data = await _resolve_intent(session, text)
        except ExtractionError as e:
            logger.error("[%s] intent extraction failed: %s", session.call_id, e)
            await _speak(session, "একটু সমস্যা হচ্ছে, একটু ধরুন।", fallback_reason="llm_failure")
            return

        # ADDED BY SOURAV -- "Caller asks two questions in one breath"
        # story. `data.get("intents")` is agent/llm.py's new ordered
        # array (see extract_intent()'s own docstring on
        # _apply_backward_compat_mirror for why "intent"/"slots" below
        # still work unchanged either way); it is ABSENT entirely on a
        # fast_path hit (agent/fast_path.py's as_llm_shape() never sets
        # it) and on every pre-this-story test mock that hands
        # _resolve_intent a plain {"intent":..., "slots":...} dict, so
        # `or [...]` safely wraps either of those into a one-item list.
        # Only when the caller asked more than one distinct question in
        # the same breath does `intents_list` actually have more than one
        # entry -- that is the ONLY new branch below; a length-1 list
        # falls straight through to the ORIGINAL if/elif chain, completely
        # unchanged by this story.
        intents_list = data.get("intents") or [{"intent": data["intent"], "slots": data["slots"]}]
        if len(intents_list) > 1:
            await _dispatch_multi_intent_turn(session, intents_list, language)
            return

        intent = data["intent"]
        slots = data["slots"]

        # ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
        # previous answer" story. Runs BEFORE any of the per-intent
        # branches below, for every intent (a no-op for the many intents
        # agent/state.py does not apply to at all -- see
        # primary_slot_for_intent()'s own comment). Two outcomes:
        #   - `slots` comes back with the intent's primary entity slot
        #     silently filled in from the last thing actually discussed,
        #     when the caller left it null this turn (a pronoun/elliptical
        #     follow-up) and exactly one candidate is tracked -- every
        #     branch below runs completely unaware anything was backfilled.
        #   - `ambiguous_kind` is set instead when that slot is null AND
        #     more than one different entity of that kind was discussed a
        #     moment ago (compare_options naming two tests, say) -- the
        #     turn stops HERE, asks which one was meant, and remembers
        #     enough (intent + already-known slots) to resume the exact
        #     same question once the caller answers (see
        #     _continue_pending's new "follow_up_clarification" state).
        _state = _session_state(session)
        ambiguous_kind = None
        if _state is not None:
            slots, ambiguous_kind = resolve_follow_up(_state, intent, slots)
        if ambiguous_kind:
            candidates = list(_state.slot_for(ambiguous_kind).names)
            session.pending = {
                "awaiting": "follow_up_clarification", "intent": intent, "slots": slots,
                "kind": ambiguous_kind, "candidates": candidates, "retries": 0,
            }
            await _speak(session, ambiguous_reference_reply(ambiguous_kind, candidates, language=language))
            return

        if intent == "smalltalk":
            await _speak(session, data.get("direct_reply_bn") or "নমস্কার, কী সাহায্য করতে পারি?")
            return

        if intent == "unclear":
            # UPDATED BY SOURAV -- wires up the business's own
            # human_fallback config (lab_tests_with_fallback_config sample
            # file's voice_agent_config.human_fallback block):
            # trigger_condition "query_unresolved_or_low_confidence" maps
            # onto this codebase's existing "unclear" intent (see
            # agent/llm.py's own docstring for exactly when the classifier
            # returns it) -- the one real, already-existing signal for
            # "the caller's query could not be resolved". This branch used
            # to speak a single fixed Bengali-only "sorry, please repeat"
            # line with no language selection at all; it now speaks the
            # business's own per-language "connecting you to an expert"
            # script instead, and records the handoff to the same
            # escalation ledger agent/outcomes.py already maintains -- see
            # human_fallback_reply()'s and record_human_handoff()'s own
            # docstrings for why the config's requested action
            # ("transfer_to_human_agent") is honestly logged rather than
            # literally transferred: this codebase has no telephony
            # transfer capability of any kind to actually do that.
            record_human_handoff(intent, call_id=session.call_id)
            await _speak(session, human_fallback_reply(language=language))
            return

        if intent == "out_of_scope":
            # ADDED BY SOURAV -- "Caller asks something the agent does not
            # cover" story. Distinct from "unclear" just above: the
            # classifier understood EXACTLY what the caller wants here
            # (see agent/llm.py's own "out_of_scope" vs "unclear"
            # distinction) -- it is simply not a service any intent above
            # covers. Rather than apologize-and-connect immediately (the
            # "unclear" path) or silently force it into a lookalike real
            # intent, this offers the caller an explicit choice and acts
            # on whichever they pick next turn -- see _continue_pending's
            # "out_of_scope_choice" branch below.
            await _speak(session, out_of_scope_reply(language=language))
            session.pending = {"awaiting": "out_of_scope_choice", "retries": 0}
            return

        try:
            if intent == "test_rate":
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_test_rate(slots["test_name"])
                await _speak(session, test_rate_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "test_sample":
                # UPDATED BY SOURAV -- restores parity with main_pcm.py,
                # which already had this branch (Story 5, "caller asks what
                # sample is needed") while main.py never did. Found while
                # wiring the report_status/report_send combined story:
                # main_pcm.py is a GENERATED file (see its own module
                # docstring and tools/make_pcm_variant.py) meant to be
                # produced FROM main.py, but this branch was added by hand
                # directly to main_pcm.py at some point without re-running
                # the generator off an updated main.py -- so a caller on
                # the WAV transport (main.py, ports 8080/8100 per
                # deploy/start_all.sh) asking only about sample type hit no
                # matching branch at all, even though agent/llm.py
                # classifies "test_sample" correctly on either transport.
                # Restoring it here BEFORE regenerating main_pcm.py from
                # this file closes that gap for good: from now on
                # main_pcm.py is only ever produced by re-running that
                # script against this file, so the two cannot drift apart
                # on this branch (or the three this story adds) again.
                # Same tool call as test_rate -- clinic-api's test lookup
                # already returns sample_type on every call, nothing new
                # was added to the API for this -- only the reply function
                # differs, so a caller who asked ONLY about the sample
                # hears just that, not the bundled rate+sample+duration
                # answer test_rate gives.
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_test_rate(slots["test_name"])
                await _speak(session, sample_type_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "test_duration":
                # ADDED BY SOURAV -- fixes a real production bug, reported
                # directly from a live call transcript:
                #   [User] How long does it take to get the urine test report?
                #   [AI]   Urine test rate is 200 taka.
                # A caller asking about REPORT TURNAROUND TIME was being
                # misclassified as "test_rate" and answered with the
                # test's PRICE instead. Root cause: an earlier story
                # ("Caller asks the price of a test") narrowed
                # test_rate_reply() to speak ONLY the price, but
                # agent/llm.py's intent prompt was never updated to match
                # -- it kept telling the classifier that "how long results
                # take" belongs to test_rate. See test_duration_reply()'s
                # own docstring for the full writeup, including a second,
                # related bug found and fixed in agent/fast_path.py.
                # Same tool call as test_rate/test_sample -- clinic-api's
                # test lookup already returns report_time_hours on every
                # call, nothing new was added to the API for this -- only
                # the reply function differs, so a caller who asked ONLY
                # about turnaround time hears just that, never the price
                # or the sample.
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_test_rate(slots["test_name"])
                await _speak(session, test_duration_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "test_preparation":
                # ADDED BY SOURAV -- "Caller asks how to prepare for a
                # test" story. Unlike test_rate/test_sample/test_duration
                # just above, this calls a DEDICATED new endpoint
                # (get_test_preparation -> GET /api/v1/tests/preparation)
                # rather than reusing get_test_rate's response, since
                # preparation data (fasting rules, medication holds,
                # per-language ready-to-speak scripts) is not part of that
                # payload at all -- see clinic-api/main.py's
                # _test_preparation_reply_dict() for the response shape.
                # Same test_name-required gate as those three: "how do I
                # prepare" has no "list every test's prep instructions"
                # analog the way health_package's bare "what packages do
                # you have" does, so a missing test_name always re-prompts
                # rather than trying to answer something unbounded.
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_test_preparation(slots["test_name"])
                await _speak(session, test_preparation_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "walkin_eligibility":
                # ADDED BY SOURAV -- Phase 1: Database Schema & Policy
                # Tables. Same single-required-slot gate as test_rate/
                # test_preparation above.
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_walkin_policy(slots["test_name"])
                await _speak(session, walkin_eligibility_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "prescription_requirements":
                if not slots.get("test_name"):
                    await _speak(session, missing_slot_prompt(intent, "test_name", language=language))
                    return
                result = await _tools.get_prescription_policy(slots["test_name"])
                await _speak(session, prescription_requirements_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "insurance_coverage":
                # Two required slots, not one -- ask for whichever is
                # still missing, mirroring book_appointment's own
                # merge-onto-pending pattern below (see agent/
                # semantic_cache.py's _is_l2_eligible for why this intent
                # is excluded from L2 entirely, the same reason
                # book_appointment is). A caller who names only the test
                # ("amar CBC insurance-e cover hobe?") gets asked for
                # their insurer next turn, and vice versa; either slot
                # already known this turn is kept.
                merged = dict(session.pending["slots"]) if session.pending else {}
                for field in ("test_name", "insurance_provider_name"):
                    if slots.get(field):
                        merged[field] = slots[field]
                missing = next((f for f in ("test_name", "insurance_provider_name") if not merged.get(f)), None)
                if missing:
                    session.pending = {
                        "awaiting": "insurance_coverage_slot", "slots": merged,
                        "missing_field": missing, "retries": 0,
                    }
                    await _speak(session, missing_slot_prompt(intent, missing, language=language))
                    return
                session.pending = None
                result = await _tools.get_insurance_coverage(merged["test_name"], merged["insurance_provider_name"])
                await _speak(session, insurance_coverage_reply(merged, result, language=language))

            elif intent == "compare_options":
                # ADDED BY SOURAV -- "Caller asks the agent to compare two
                # options" story. Same two-required-slots merge-onto-
                # pending pattern as insurance_coverage just above, for
                # the identical reason (either name can be missing this
                # turn). Once both names are in hand, EACH is resolved
                # independently via _resolve_comparable_entity() (tries
                # the test catalogue, then the package catalogue -- see
                # that function's own docstring) before
                # agent/compare_flow.py's build_comparison() computes the
                # actual price/component comparison in code -- this
                # branch itself does no arithmetic and makes no
                # recommendation; it only fetches both sides' live data
                # and hands it to compare_flow/reply_templates.
                merged = dict(session.pending["slots"]) if session.pending else {}
                for field in ("compare_option_a", "compare_option_b"):
                    if slots.get(field):
                        merged[field] = slots[field]
                missing = next((f for f in ("compare_option_a", "compare_option_b") if not merged.get(f)), None)
                if missing:
                    session.pending = {
                        "awaiting": "compare_options_slot", "slots": merged,
                        "missing_field": missing, "retries": 0,
                    }
                    await _speak(session, missing_slot_prompt(intent, missing, language=language))
                    return
                session.pending = None
                name_a, name_b = merged["compare_option_a"], merged["compare_option_b"]
                entity_a, entity_b = await _resolve_comparable_entity(name_a), await _resolve_comparable_entity(name_b)
                comparison = build_comparison(entity_a, entity_b)
                await _speak(session, compare_options_reply(name_a, name_b, entity_a, entity_b, comparison, language=language))
                _remember_compared_entities(session, entity_a, entity_b)

            elif intent == "billing_balance":
                # ADDED BY SOURAV -- Phase 1: Outstanding Balance / Billing
                # story. Identity resolved by PHONE (RULE 14/15), same
                # gate as report_status/report_send below -- deliberately
                # NOT behind OTP (see clinic-api/models.py's
                # PatientBilling docstring for that scoping decision).
                phone = parse_phone(slots.get("phone") or "")
                if not phone:
                    session.pending = {"awaiting": "billing_phone", "retries": 0}
                    await _speak(session, missing_slot_prompt(intent, "phone", language=language))
                    return
                result = await _tools.get_patient_billing(phone)
                await _speak(session, billing_balance_reply(result, language=language))

            elif intent == "report_status":
                # ADDED BY SOURAV -- "Lab Report Status & Secure Delivery"
                # combined story. Identity is resolved by PHONE, never by
                # name (RULE 14/15) -- if the caller's utterance didn't
                # carry one, ask for it and park in the "phone" pending
                # state above rather than guessing or proceeding without it.
                phone = parse_phone(slots.get("phone") or "")
                if not phone:
                    session.pending = {
                        "awaiting": "report_phone", "flow": "report_status",
                        "test_name": slots.get("test_name"), "retries": 0,
                    }
                    await _speak(session, missing_slot_prompt(intent, "phone", language=language))
                    return
                await _handle_report_lookup(session, phone, slots.get("test_name"), "report_status", language=language)

            elif intent == "report_send":
                # ADDED BY SOURAV -- same identity-by-phone gate as
                # report_status just above; the two intents share
                # _handle_report_lookup/_finish_report_flow and differ only
                # in `flow`, which agent/report_flow.py's
                # interpret_report_status_result() uses to decide whether a
                # READY+enabled report gets the "shall I send it?" offer
                # (report_status) or goes straight to requesting delivery
                # (report_send, since the caller already asked for it).
                phone = parse_phone(slots.get("phone") or "")
                if not phone:
                    session.pending = {
                        "awaiting": "report_phone", "flow": "report_send",
                        "test_name": slots.get("test_name"), "retries": 0,
                    }
                    await _speak(session, missing_slot_prompt(intent, "phone", language=language))
                    return
                await _handle_report_lookup(session, phone, slots.get("test_name"), "report_send", language=language)

            elif intent == "doctor_availability":
                if not slots.get("doctor_name"):
                    await _speak(session, missing_slot_prompt(intent, "doctor_name", language=language))
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
                await _speak(session, doctor_availability_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

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

            elif intent == "doctor_schedule":
                # ADDED BY SOURAV -- "Caller asks when a doctor sits" story.
                # Deliberately DATE-FREE, unlike doctor_availability just
                # above: this intent exists exactly for the caller who has
                # NOT named a day and wants the doctor's general recurring
                # weekly schedule instead (see agent/llm.py's SYSTEM_PROMPT
                # for how the two are told apart at classification time,
                # and clinic-api/main.py::doctor_schedule()'s docstring for
                # the response shape). No `date` slot is read or passed
                # here at all -- even if the LLM happened to also extract
                # one from the same utterance, it is not used, since a
                # date would silently turn this back into the OTHER
                # question this intent exists to be distinct from.
                #
                # Same known limitation as doctor_availability just above,
                # not introduced here: no pending state is opened when
                # doctor_name is missing, so the caller's next utterance
                # (e.g. a bare doctor's name in reply to the prompt) goes
                # through a fresh LLM classification rather than a
                # targeted single-slot fill. Flagged, not fixed -- fixing
                # it would mean touching doctor_availability's identical
                # gap too, which is out of this story's scope.
                if not slots.get("doctor_name"):
                    await _speak(session, missing_slot_prompt(intent, "doctor_name", language=language))
                    return
                result = await _tools.get_doctor_schedule(slots["doctor_name"])
                await _speak(session, doctor_schedule_reply(slots, result, language=language))
                _remember_primary_entity(session, intent, slots, result)

            elif intent == "doctors_by_department":
                if not slots.get("department"):
                    await _speak(session, missing_slot_prompt(intent, "department", language=language))
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
                await _speak(session, doctors_by_department_reply(slots, result, language=language))

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

                missing = _next_missing(merged)
                if missing is None:
                    # A caller who gave all 5 fields in one breath still
                    # gets the pre-write readback -- this is the SAME gap
                    # the multi-turn flow had (see _continue_pending's
                    # "confirm_booking" state): a single-shot utterance is
                    # exactly as capable of a misheard phone digit as one
                    # collected field-by-field.
                    session.pending = {
                        "awaiting": "confirm_booking", "slots": merged, "candidates": None,
                        "offered_date": (session.pending or {}).get("offered_date"), "retries": 0,
                    }
                    await _speak(session, booking_confirmation_prompt(merged, language=language))
                    return

                session.pending = {
                    "awaiting": missing, "slots": merged, "candidates": None,
                    "offered_date": (session.pending or {}).get("offered_date"), "retries": 0,
                }
                await _speak(session, missing_slot_prompt(intent, missing, language=language))

            elif intent == "request_callback":
                # ADDED BY SOURAV -- "Caller asks to be called back" story
                # (Evidence: "No outbound capability"). Availability is
                # checked FIRST, before asking for a single detail --
                # Acceptance Criterion 3: a caller is never walked through
                # collecting a time window and phone number only to be
                # refused at the very end. CALLBACKS_ENABLED is checked
                # BEFORE ever calling get_clinic_info() -- a deployment
                # that has turned the feature off entirely has no reason
                # to pay for that round-trip (cached or not) just to
                # decide something a config constant already answered.
                if not CALLBACKS_ENABLED:
                    await _speak(session, callback_unavailable_reply("disabled", language=language))
                    return

                # get_clinic_info() is the SAME already-cached call
                # clinic_info's own branch above makes (agent/
                # reference_data_cache.py) -- no new tool, no new network
                # round-trip pattern.
                hours_result = await _tools.get_clinic_info()
                hours = hours_result.get("hours") if hours_result.get("found") else None
                availability = check_callback_availability(
                    hours, datetime.date.today().weekday(),
                    datetime.datetime.now().strftime("%H:%M"), CALLBACKS_ENABLED,
                )
                if not availability["available"]:
                    await _speak(session, callback_unavailable_reply(availability["reason"], language=language))
                    return

                # Merge onto whatever session.pending already knows, same
                # "don't discard a field the caller already gave" reasoning
                # as book_appointment just above -- a caller who names a
                # time window AND a phone number in one breath should never
                # be asked for either again.
                merged = dict(session.pending["slots"]) if session.pending else {}
                for field in (*_CALLBACK_FIELDS, "callback_reason"):
                    if slots.get(field):
                        merged[field] = slots[field]

                # Acceptance Criterion 1's "preserving the conversation
                # context and reason" -- resolved ONCE, here, on the turn
                # that actually opens this flow (session.state reflects
                # whatever was discussed earlier in THIS call right now;
                # nothing about it changes while the rest of this flow
                # collects the remaining fields over the next turns, so
                # there is no benefit to re-resolving it later, only risk
                # of it drifting from what was true when the caller asked).
                # Always stored, even as null (build_callback_reason()'s own
                # "no reason given" case) -- see agent/callback_flow.py's
                # own docstring for why that null is never papered over
                # with an invented generic reason.
                if "callback_reason" not in merged:
                    state = _session_state(session)
                    merged["callback_reason"] = build_callback_reason(
                        slots.get("callback_reason"),
                        active_test=state.slot_for("test").primary if state else None,
                        active_doctor=state.slot_for("doctor").primary if state else None,
                        active_package=state.slot_for("package").primary if state else None,
                    )

                missing = _next_missing_callback(merged)
                if missing is None:
                    # Same "every critical value is read back before it is
                    # used" discipline as book_appointment's own
                    # confirm_booking state just above.
                    session.pending = {
                        "awaiting": "confirm_callback", "slots": merged, "candidates": None,
                        "offered_date": None, "retries": 0,
                    }
                    await _speak(session, callback_confirmation_prompt(merged, language=language))
                    return

                session.pending = {
                    "awaiting": missing, "slots": merged, "candidates": None,
                    "offered_date": None, "retries": 0,
                }
                await _speak(session, missing_slot_prompt("request_callback", missing, language=language))

            elif intent == "health_package":
                # ADDED BY SOURAV -- "Caller asks about a health package"
                # story. Deliberately NEVER re-prompts for a missing
                # "package_name" the way every single-entity intent above
                # does for its own required slot -- see agent/llm.py's own
                # comment on VALID_INTENTS: a caller who names no package
                # at all is asking a complete, different, equally valid
                # question ("what packages do you have"), backed by its
                # own clinic-api list endpoint, not an incomplete
                # extraction waiting on a re-prompt.
                if slots.get("package_name"):
                    result = await _tools.search_health_package(slots["package_name"])
                    await _speak(session, health_package_reply(slots, result, language=language))
                    _remember_primary_entity(session, intent, slots, result)
                else:
                    result = await _tools.get_health_packages()
                    await _speak(session, health_packages_list_reply(result, language=language))

            elif intent == "clinic_info":
                # ADDED BY SOURAV -- "Caller asks opening hours, address or
                # directions" story. No slot is required to call the tool
                # (clinic-api/models.py's ClinicInfo is a singleton table) --
                # "info_topic" only narrows which part of the already-
                # fetched answer gets SPOKEN, in clinic_info_reply() itself.
                # "today_weekday" resolves "which day" here, in dispatch,
                # the same way doctor_availability's date_iso default does
                # just above -- reply_templates.py never imports datetime
                # itself (see clinic_info_reply()'s own docstring).
                result = await _tools.get_clinic_info()
                info_slots = {
                    "info_topic": slots.get("info_topic"),
                    "today_weekday": datetime.date.today().weekday(),
                }
                await _speak(session, clinic_info_reply(info_slots, result, language=language))

        except ToolCallError as e:
            logger.error("[%s] clinic API call failed: %s", session.call_id, e)
            await _speak(session, "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
                         fallback_reason="tool_failure")


# ADDED BY SOURAV -- "Caller asks two questions in one breath" story. Sets
# of intents that never get a real inline answer in a COMBINED turn,
# regardless of slot completeness -- see _resolve_combinable_intent_fragment's
# own docstring just below for why each is excluded rather than composed.
_MULTI_INTENT_NEEDS_SEPARATE_FLOW = {"book_appointment", "report_status", "report_send"}
_MULTI_INTENT_NO_FRAGMENT = {"smalltalk", "unclear"}


async def _resolve_combinable_intent_fragment(intent: str, slots: dict, language: str) -> str | None:
    """ADDED BY SOURAV -- "Caller asks two questions in one breath" story.
    Resolves ONE intent (one entry of a multi-question turn's "intents"
    array) into its own spoken fragment for _dispatch_multi_intent_turn()
    below to join with the others, in order.

    Returns None ONLY for "smalltalk"/"unclear" -- see
    _MULTI_INTENT_NO_FRAGMENT above: neither is really a second QUESTION
    the caller needs an honest answer or acknowledgment for (a bare "ভালো
    আছেন?" tacked onto a real question is not something Criterion 2's
    "explicitly acknowledge the unanswerable one" was written for), so
    these two are the sole, deliberate exception to "nothing is silently
    dropped." Every other intent returns a real, non-empty fragment, one
    of:
      - The SAME reply_templates function a solo turn for that intent
        would use, called exactly the same way (a "found but not yet
        reviewed" policy row -- Criterion 2's "unreviewed" example --
        already gets an honest sentence for free from that same function,
        e.g. walkin_eligibility_reply(); zero new code needed for that
        case specifically).
      - multi_intent_missing_info_reply() when an otherwise-combinable
        intent is missing a slot it needs (test_name, doctor_name,
        department, or -- for insurance_coverage/billing_balance -- either
        of their two/one required fields). Deliberately generic rather
        than a targeted per-field re-prompt: a combined turn does not open
        a SECOND pending state on top of whatever the turn's other
        question may already need to report, so there is nowhere to
        attach a targeted follow-up question to (see
        _dispatch_multi_intent_turn's own docstring).
      - multi_intent_out_of_scope_reply() for "out_of_scope" -- a
        non-interactive acknowledgment, unlike out_of_scope_reply()'s
        solo interactive yes/no offer (again: no second pending state).
      - multi_intent_needs_separate_flow_reply() for book_appointment/
        report_status/report_send (_MULTI_INTENT_NEEDS_SEPARATE_FLOW) --
        ALWAYS, even when every slot they'd need already happens to be
        present. These three are multi-turn, sometimes security-sensitive
        (OTP) flows in their own right, not something to fold into a
        shared reply alongside an unrelated question.

    Deliberately narrower than the solo dispatch branches in _dispatch_turn's
    own if/elif chain above: this NEVER opens a follow-up pending state of
    any kind, even for an intent that would open one solo (doctor_
    availability's "book this day?" offer, doctors_by_department's
    candidate list, insurance_coverage's/billing_balance's slot-fill
    prompt) -- composing two independently-stateful sub-conversations into
    one reply is a different, larger problem than "answer both questions
    honestly in one turn."
    """
    if intent in _MULTI_INTENT_NO_FRAGMENT:
        return None

    if intent == "out_of_scope":
        return multi_intent_out_of_scope_reply(language=language)

    if intent in _MULTI_INTENT_NEEDS_SEPARATE_FLOW:
        return multi_intent_needs_separate_flow_reply(language=language)

    if intent == "test_rate":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_test_rate(slots["test_name"])
        return test_rate_reply(slots, result, language=language)

    if intent == "test_sample":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_test_rate(slots["test_name"])
        return sample_type_reply(slots, result, language=language)

    if intent == "test_duration":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_test_rate(slots["test_name"])
        return test_duration_reply(slots, result, language=language)

    if intent == "test_preparation":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_test_preparation(slots["test_name"])
        return test_preparation_reply(slots, result, language=language)

    if intent == "walkin_eligibility":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_walkin_policy(slots["test_name"])
        return walkin_eligibility_reply(slots, result, language=language)

    if intent == "prescription_requirements":
        if not slots.get("test_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_prescription_policy(slots["test_name"])
        return prescription_requirements_reply(slots, result, language=language)

    if intent == "insurance_coverage":
        if not slots.get("test_name") or not slots.get("insurance_provider_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_insurance_coverage(slots["test_name"], slots["insurance_provider_name"])
        return insurance_coverage_reply(slots, result, language=language)

    if intent == "billing_balance":
        phone = parse_phone(slots.get("phone") or "")
        if not phone:
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_patient_billing(phone)
        return billing_balance_reply(result, language=language)

    if intent == "doctor_availability":
        if not slots.get("doctor_name"):
            return multi_intent_missing_info_reply(language=language)
        date_iso = slots.get("date") or datetime.date.today().isoformat()
        result = await _tools.get_doctor_availability(slots["doctor_name"], date_iso)
        return doctor_availability_reply(slots, result, language=language)

    if intent == "doctor_schedule":
        if not slots.get("doctor_name"):
            return multi_intent_missing_info_reply(language=language)
        result = await _tools.get_doctor_schedule(slots["doctor_name"])
        return doctor_schedule_reply(slots, result, language=language)

    if intent == "doctors_by_department":
        if not slots.get("department"):
            return multi_intent_missing_info_reply(language=language)
        date_iso = slots.get("date") or datetime.date.today().isoformat()
        result = await _tools.get_doctors_by_department(slots["department"], date_iso)
        return doctors_by_department_reply(slots, result, language=language)

    if intent == "health_package":
        if slots.get("package_name"):
            result = await _tools.search_health_package(slots["package_name"])
            return health_package_reply(slots, result, language=language)
        result = await _tools.get_health_packages()
        return health_packages_list_reply(result, language=language)

    if intent == "clinic_info":
        result = await _tools.get_clinic_info()
        info_slots = {
            "info_topic": slots.get("info_topic"),
            "today_weekday": datetime.date.today().weekday(),
        }
        return clinic_info_reply(info_slots, result, language=language)

    # Defensive: every member of VALID_INTENTS (agent/llm.py) is handled
    # explicitly somewhere above -- this is unreachable in practice, but
    # falls back to the same honest "ask that one separately" fragment
    # book_appointment/report_status/report_send get, rather than ever
    # silently dropping an intent this function does not recognize.
    return multi_intent_needs_separate_flow_reply(language=language)


def _remember_multi_intent_entities(session: CallSession, intents_list: list[dict]) -> None:
    """ADDED BY SOURAV -- "Caller asks a follow-up that depends on the
    previous answer" story. A multi-intent turn naming two DIFFERENT
    tests in the same breath (e.g. "CBC-r rate koto, ar Lipid Profile-er
    sample ki lagbe?") is exactly the "multiple entities discussed
    previously" ambiguity Acceptance Criterion 2 describes -- handled
    here, once, after the whole turn's fragments are resolved, so a bare
    pronoun in the NEXT turn cannot silently be guessed as one or the
    other (see agent/state.py's own docstring).

    Deliberately uses the CALLER'S OWN WORDS for each name, not a tool
    result's canonical spelling: _resolve_combinable_intent_fragment()
    above returns only a rendered reply string, not the resolved entity
    data, so reconfirming a canonical name here would mean a second,
    duplicate tool call purely to remember it. This is a smaller
    guarantee than the single-intent path's _remember_primary_entity()
    (which only ever remembers a name a lookup just confirmed exists) --
    flagged, not silently equated with it -- but clinic-api's own lookups
    already tolerate the caller's raw phrasing fine on their own, so a
    follow-up backfilled from a multi-intent turn's memory is no worse
    off than the ORIGINAL turn's own lookup was.
    """
    state = _session_state(session)
    if state is None:
        return
    by_kind: dict[str, list[str]] = {}
    for item in intents_list:
        slot_key = primary_slot_for_intent(item.get("intent"))
        if slot_key is None:
            continue
        name = (item.get("slots") or {}).get(slot_key)
        if not name:
            continue
        by_kind.setdefault(kind_for_slot(slot_key), []).append(name)

    for kind, names in by_kind.items():
        distinct = list(dict.fromkeys(names))  # de-dupe, order-preserved
        if len(distinct) > 1:
            state.mark_ambiguous(kind, distinct)
        else:
            state.mark(kind, distinct[0])


async def _dispatch_multi_intent_turn(session: CallSession, intents_list: list[dict], language: str) -> None:
    """ADDED BY SOURAV -- "Caller asks two questions in one breath" story.
    Only ever called from _dispatch_turn above, and only when
    `len(intents_list) > 1` -- a single-entry list (the overwhelming
    majority of turns: a fast_path hit, a plain single-question LLM
    extraction, or any pre-this-story test mock) takes the ORIGINAL,
    completely untouched if/elif chain in _dispatch_turn instead. Nothing
    about that existing path changes for this story.

    Story Criterion 1 (Order Preservation): `intents_list` is iterated
    below in order, exactly as agent/llm.py's SYSTEM_PROMPT_TEMPLATE
    instructs the model to return it, and each fragment is appended to
    `fragments` in that same order -- no sorting or reordering anywhere.
    The joined reply is strictly in the order asked, by construction.

    Story Criterion 2 (Honest Partial Handling): every intent in
    `intents_list` produces either a real, grounded fragment or one of the
    three honest, non-fabricating fallback fragments -- see
    _resolve_combinable_intent_fragment() above for exactly which and why.
    Nothing is silently dropped; smalltalk/unclear are the sole,
    deliberate exception (see that function's own docstring).

    Tool-failure isolation ("individual tool calls executed independently
    without one tool failure short-circuiting the second"): each intent's
    fragment is resolved inside its OWN try/except ToolCallError in the
    loop below -- unlike the solo dispatch's single try/except wrapping
    its ENTIRE if/elif chain -- so intent #1's clinic-api call failing can
    never prevent intent #2's from running at all.
    """
    fragments: list[str] = []
    for item in intents_list:
        intent = item.get("intent")
        slots = item.get("slots") or {}
        try:
            fragment = await _resolve_combinable_intent_fragment(intent, slots, language)
        except ToolCallError as e:
            logger.error("[%s] clinic API call failed for intent %s (multi-intent turn): %s",
                         session.call_id, intent, e)
            fragment = "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।"
        if fragment:
            fragments.append(fragment)

    if not fragments:
        # Defensive: every entry was smalltalk/unclear. agent/llm.py's own
        # prompt instructs the model to never split one real question into
        # several entries, so a genuine multi-question turn should not
        # reach this -- but rather than speak nothing at all, fall back to
        # the same honest "connecting you to an expert" handling the solo
        # "unclear" path gives above.
        record_human_handoff("unclear", call_id=session.call_id)
        await _speak(session, human_fallback_reply(language=language))
        return

    await _speak(session, " ".join(fragments))
    _remember_multi_intent_entities(session, intents_list)


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

        sr = session.audio.sample_rate
        tail = session.audio.tail_tensor(session.processed_until_s)
        if tail.numel() < int(0.2 * sr):
            continue  # not enough new audio to judge yet -- not an error

        result = await asyncio.to_thread(_turn_detector.poll, tail, sr)
        if result.utterance_end_s is None:
            continue

        absolute_end_s = session.processed_until_s + result.utterance_end_s
        session.utt_seq += 1
        utterance_wav = await _slice_utterance(
            session, session.processed_until_s, absolute_end_s, session.utt_seq,
        )
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


app.mount("/", StaticFiles(directory="static/pcm", html=True), name="static")
