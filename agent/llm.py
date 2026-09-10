"""Node 2: intent + slot extraction via Qwen2.5, over Ollama.

Deliberately NOT vLLM, and deliberately NOT the model's native tool-calling
format. Two reasons, both grounded in what's already running on this
account rather than assumed:

1. VRAM. vLLM serving Qwen2.5-7B-Instruct at fp16 needs ~17GB just for the
   model (measured: https://www.spheron.network/tools/gpu-recommender/
   Qwen/Qwen2.5-7B-Instruct/). The already-deployed pipeline runs the same
   model through Ollama as a Q4_K_M GGUF -- 4.68GB on disk -- specifically
   so IndicConformer (~1.2GB VRAM) and this service's TTS model can share
   the same 24GB card comfortably. Introducing vLLM here would mean running
   two different serving stacks for the same model on the same box, for no
   benefit at this call volume.

2. voicerx/extract.py already hardened a JSON-mode + strict-schema pattern
   against a *reproduced* failure: a looser prompt caused the model to
   invent a drug name ("Naloxone") that was never said. Ollama's function
   tool_calls) is comparatively unproven on this stack. The pattern below
   -- classify intent, extract slots as literal spans, then have CODE (not
   the model) fill the reply from the API response -- is the same
   discipline, applied to prices and appointment slots instead of drugs.
   Quoting a wrong price with total confidence is this system's version of
   that bug, so the model is never allowed to state a number on its own;
   see main.py's _compose_reply().
"""
from __future__ import annotations

import datetime
import json
import os
import random
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

# PER-ATTEMPT ceiling, and the ceiling for the WHOLE turn across retries.
#
# These replace a bare 90s per-attempt timeout with no total cap. With
# max_retries=2 (three attempts) that arrangement allowed a single turn to
# occupy a worker for 270 SECONDS -- four and a half minutes of a live phone
# call spent in silence behind the half-duplex gate, long past the point any
# caller has hung up, while still holding its thread and its slot in Ollama's
# queue the entire time.
#
# The 90s figure was calibrated against a COLD model: llm.py measured a
# cold-loaded qwen2.5:7b at 47s just to answer "Say OK". That is the wrong
# thing to size for now. OLLAMA_KEEP_ALIVE=-1 (see deploy/env.vast.sh) pins
# the model resident, so cold starts are a startup-only event, and a warm
# generation for this prompt measures ~1-3s.
#
# What actually produces a slow turn today is QUEUEING, not loading:
# OLLAMA_NUM_PARALLEL=2 means only two generations run at once and the rest
# wait INSIDE Ollama where this process cannot see them. A caller stuck in
# that queue looks exactly like a cold start from here -- which is why a
# timeout sized for cold starts is precisely the wrong backstop for it. It
# lets the queue grow instead of shedding load, and every retry that fires
# adds another entry to the same queue it is already stuck behind.
#
# 20s is still ~7x the warm path, so it does not trip on normal slowness; it
# only fires when something is genuinely wrong. The 25s total budget is the
# real protection: it bounds the whole turn regardless of how the retries
# fall, so failing fast to the "একটু সমস্যা হচ্ছে" apology beats holding a
# caller in silence. Both are env-tunable so they can be tightened against
# real peak traffic without a redeploy.
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "20"))
OLLAMA_TURN_BUDGET_S = float(os.environ.get("OLLAMA_TURN_BUDGET_S", "25"))

# Retry backoff. Previously there was none: a failed attempt re-fired the
# instant it returned.
#
# That is the wrong reflex against this particular dependency. The failure
# being retried is usually not "the request was lost" but "Ollama is at
# OLLAMA_NUM_PARALLEL and everything else is queued behind it". Retrying
# immediately adds another entry to the very queue that caused the failure,
# and because every concurrent caller's retries fire on the same schedule
# they arrive together -- a thundering herd, synchronized by the shared
# outage that produced it. Load spikes hardest exactly when the service is
# least able to absorb it.
#
# Exponential growth spreads the retries out; the jitter de-synchronizes
# callers from each other so they stop arriving in a block. Both are small
# because the whole turn budget is only OLLAMA_TURN_BUDGET_S -- this is
# breathing room between attempts, not a real hold-off, and the sleep below
# is always clamped so it can never eat the budget the next attempt needs.
_BACKOFF_BASE_S = 0.25
_BACKOFF_CAP_S = 2.0
_BACKOFF_JITTER_S = 0.25


def _backoff_s(attempt: int) -> float:
    """Delay before the attempt AFTER this one. attempt is 1-based."""
    return (min(_BACKOFF_BASE_S * (2 ** (attempt - 1)), _BACKOFF_CAP_S)
            + random.uniform(0.0, _BACKOFF_JITTER_S))

VALID_INTENTS = {"test_rate", "doctor_availability", "book_appointment",
                 "doctors_by_department", "payment", "report_collection",
                 "patient_history", "smalltalk", "unclear"}

SYSTEM_PROMPT_TEMPLATE = """You are the intent-and-slot extractor for a diagnostic clinic's phone assistant. Callers speak BENGALI, HINDI or ENGLISH, and often mix them -- an English clinical term inside a Bengali sentence is normal, not an error. You will be given ONE caller utterance, transcribed by automatic speech recognition from live phone audio -- it may contain ASR errors, missing punctuation, or code-switched words written in another script.

Today's date is {today_iso} ({today_weekday}), Asia/Kolkata.

YOUR ONLY JOB is to classify intent and pull out slots that are LITERALLY present in the utterance. You do NOT know test prices, doctor schedules, or appointment availability -- do not guess or state any of those; that data comes from a separate lookup after you run.

INTENTS (exactly one):
- "test_rate": caller is asking the price/rate of a diagnostic test.
- "doctor_availability": caller is asking whether/when a named doctor is available.
- "doctors_by_department": caller is asking for doctors in a specific department (e.g., "ortho", "cardiology", "অর্থো").
- "book_appointment": caller wants to book, confirm, or reschedule an appointment.
- "payment": caller is asking HOW to pay, whether payment is needed in advance, or what payment methods are accepted (e.g. "কীভাবে টাকা দেব", "पैसे कैसे देने हैं", "do I need to pay online"). Asking only the PRICE of a test is "test_rate", not this.
- "report_collection": caller is asking when a report will be ready, or how to collect it (e.g. "রিপোর্ট কবে পাব", "रिपोर्ट कब मिलेगी", "how do I get my report").
- "patient_history": caller is asking about their OWN past tests or records (e.g. "আমার আগের টেস্টগুলো", "मेरी पिछली रिपोर्ट", "what tests have I had"). Asking when a report will be READY is "report_collection", not this.
- "smalltalk": greeting, thanks, or anything with no clinic-data lookup needed. You MAY write a short, warm reply yourself, IN THE CALLER'S OWN LANGUAGE, for this case only.
- "unclear": you cannot confidently tell what the caller wants, or the utterance is empty/garbled ASR noise.

SLOT RULES:
- Only fill a slot if the caller's words support it. Leave it null rather than inferring.
- "date": resolve relative time words in any of the three languages (আজ / आज / today = today, কাল / कल / tomorrow, পরশু / परसों / day after tomorrow, this/next weekday names) to an ISO yyyy-mm-dd using today's date above. If no date is mentioned for an availability/booking request, leave it null -- do not assume "today".
- "test_name" / "doctor_name": copy the term as the caller said it (Bengali or transliterated English), do not translate or normalize it -- the lookup service handles matching.
- "department": copy the department name as the caller said it (e.g., "ortho", "cardiology", "অর্থোপেডিক্স"), do not translate or normalize it -- the lookup service handles matching.
- "phone": only if a phone number is explicitly spoken, digits only.
- Never invent a patient name, phone number, or date that was not said.

Output ONLY a single valid JSON object, no other text, in exactly this shape:
{{
  "intent": "test_rate" | "doctor_availability" | "doctors_by_department" | "book_appointment" | "payment" | "report_collection" | "patient_history" | "smalltalk" | "unclear",
  "slots": {{
    "test_name": string or null,
    "doctor_name": string or null,
    "department": string or null,
    "date": string or null,
    "time_slot": string or null,
    "patient_name": string or null,
    "phone": string or null
  }},
  "direct_reply_bn": string or null
}}

"direct_reply_bn" must be null for every intent except "smalltalk" -- for every other intent, the reply is composed later from real clinic data, not from you."""


class ExtractionError(Exception):
    pass


def _call_ollama(prompt: str, timeout_s: float = OLLAMA_TIMEOUT_S) -> str:
    # See OLLAMA_TIMEOUT_S / OLLAMA_TURN_BUDGET_S above for why this is no
    # longer the old cold-start-sized 90s.
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


def _validate(data: dict) -> tuple[bool, list[str]]:
    errors = []
    if data.get("intent") not in VALID_INTENTS:
        errors.append(f"invalid intent: {data.get('intent')!r}")
    slots = data.get("slots")
    if not isinstance(slots, dict):
        errors.append("slots: expected object")
    else:
        for key in ("test_name", "doctor_name", "department", "date", "time_slot", "patient_name", "phone"):
            if key not in slots:
                errors.append(f"slots.{key}: missing")
    if data.get("intent") != "smalltalk" and data.get("direct_reply_bn") not in (None, ""):
        # Not fatal -- just strip it. The model overstepping here is the
        # exact failure mode this schema exists to prevent (see module
        # docstring), so we defend in code rather than trust a retry to fix it.
        data["direct_reply_bn"] = None
    return (len([e for e in errors if "missing" not in e or "intent" in e or "slots: expected" in e]) == 0
            and "slots" in data, errors)


def extract_intent(transcript_bn: str, max_retries: int = 2) -> tuple[dict, dict]:
    """Returns (parsed JSON dict, diagnostics dict)."""
    now = datetime.datetime.now()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        today_iso=now.strftime("%Y-%m-%d"),
        today_weekday=now.strftime("%A"),
    )
    prompt = f"{system_prompt}\n\nCALLER UTTERANCE (Bengali, ASR output):\n{transcript_bn}\n\nJSON:"

    diagnostics = {"attempts": 0, "total_time_s": 0.0, "errors": [], "budget_exhausted": False}
    last_error = None

    # monotonic, not time.time(): this measures a duration, and a wall-clock
    # step (NTP correction) must not be able to extend or collapse a live
    # caller's budget.
    deadline = time.monotonic() + OLLAMA_TURN_BUDGET_S

    for attempt in range(1, max_retries + 2):
        remaining = deadline - time.monotonic()
        # Never start an attempt that cannot meaningfully finish. Without
        # this the last retry fires with a sliver of budget, fails on
        # timeout, and buys nothing except one more entry in Ollama's queue
        # at the exact moment that queue is what is already hurting.
        if remaining <= 0.5:
            diagnostics["budget_exhausted"] = True
            diagnostics["errors"].append(
                f"turn budget {OLLAMA_TURN_BUDGET_S:.0f}s exhausted before attempt {attempt}",
            )
            break

        diagnostics["attempts"] = attempt
        t0 = time.monotonic()
        try:
            # Clamp to whatever budget is actually left, so no single
            # attempt can overrun the turn as a whole.
            raw = _call_ollama(prompt, timeout_s=min(OLLAMA_TIMEOUT_S, remaining))
            diagnostics["total_time_s"] += time.monotonic() - t0
            data = json.loads(raw)
            ok, errors = _validate(data)
            if not ok:
                raise ValueError(f"schema validation failed: {errors}")
            return data, diagnostics
        except Exception as e:  # noqa: BLE001 - retry on anything, log it
            diagnostics["total_time_s"] += time.monotonic() - t0
            last_error = e
            diagnostics["errors"].append(f"attempt {attempt}: {type(e).__name__}: {e}")

            # Back off before the next attempt, clamped so the wait can never
            # consume budget the retry itself needs. Blocking sleep is fine:
            # this runs on the dedicated HTTP pool (agent/executors.py), not
            # on the event loop and not on the audio path.
            budget_left = deadline - time.monotonic()
            if budget_left > 0.5:
                time.sleep(min(_backoff_s(attempt), budget_left - 0.5))

    raise ExtractionError(
        f"intent extraction failed after {diagnostics['attempts']} attempt(s) in "
        f"{diagnostics['total_time_s']:.1f}s"
        f"{' (turn budget exhausted)' if diagnostics['budget_exhausted'] else ''}: {last_error}",
    )
