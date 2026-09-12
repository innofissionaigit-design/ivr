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

import json
import logging
import re
import time
import urllib.request

logger = logging.getLogger("llm")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

VALID_INTENTS = {"test_rate", "doctor_availability", "book_appointment", "doctors_by_department", "smalltalk", "unclear"}

# story title: The model never originates a fact
# user story: As a clinical lead, I want every price, date and identifier to
#   come from a verified system response, so that a wrong answer is a data bug
#   rather than a model bug.
# acceptance criteria: Every factual sentence is a template substitution from a
#   validated tool response and the model is never shown a figure it could
#   restate. An automated assertion on every commit proves no model-composed
#   span reaches synthesis on a factual intent.
#
# The model names the MEANING of a date phrase; agent/date_calc.py turns that
# meaning into calendar dates. This set is the closed vocabulary it may use --
# imported rather than restated so the prompt and the calculator can never
# drift apart, which is the failure mode of every "keep these two lists in
# sync" comment ever written.
from agent.date_calc import VOCABULARY as DATE_EXPRESSIONS

_DATE_EXPR_LIST = ", ".join(sorted(DATE_EXPRESSIONS))

SYSTEM_PROMPT_TEMPLATE = """You are the intent-and-slot extractor for a diagnostic clinic's Bengali phone assistant. You will be given ONE caller utterance, transcribed by automatic speech recognition from live phone audio -- it may contain ASR errors, missing punctuation, or code-switched English words written in Bengali script.

You do NOT have a calendar and you are not given today's date. You never need one: when the caller mentions a day, you say WHICH EXPRESSION they used and a separate calculator works out the actual dates afterwards.

YOUR ONLY JOB is to classify intent and pull out slots that are LITERALLY present in the utterance. You do NOT know test prices, doctor schedules, or appointment availability -- do not guess or state any of those; that data comes from a separate lookup after you run.

INTENTS (one per part -- see MULTI-PART below):
- "test_rate": caller is asking the price/rate of a diagnostic test.
- "doctor_availability": caller is asking whether/when a named doctor is available.
- "doctors_by_department": caller is asking for doctors in a specific department (e.g., "ortho", "cardiology", "অর্থো").
- "book_appointment": caller wants to book, confirm, or reschedule an appointment.
- "smalltalk": greeting, thanks, or anything with no clinic-data lookup needed. You MAY write a short, warm Bengali reply yourself for this case only.
- "unclear": you cannot confidently tell what the caller wants, or the utterance is empty/garbled ASR noise.

SLOT RULES:
- Only fill a slot if the caller's words support it. Leave it null rather than inferring.
- "date_expr": if the caller referred to a day or a period, name it using EXACTLY ONE of these values and nothing else: {date_expr_list}. NEVER output a calendar date -- not in this field, not anywhere. You have no way to know what today is, so any yyyy-mm-dd you produced would be invented. Examples: আজ -> "today", কাল/আগামীকাল -> "tomorrow", পরশু -> "day_after_tomorrow", আগামী সপ্তাহে -> "next_week", আগামী শনিবার -> "next_saturday", শনিবার -> "saturday". If the caller DID refer to a day but none of the values above fit it, use "other" -- never guess a value that is merely close. If the caller mentioned no day at all, leave it null -- do not assume "today".
- "date": ONLY if the caller spoke a calendar date out loud ("১৫ তারিখ", "তেসরা সেপ্টেম্বর"), copy their words EXACTLY as they said them, the same way you copy a test name. Do not convert it to digits or to any date format. Otherwise null.
- "test_name" / "doctor_name": copy the term as the caller said it (Bengali or transliterated English), do not translate or normalize it -- the lookup service handles matching.
- "department": copy the department name as the caller said it (e.g., "ortho", "cardiology", "অর্থোপেডিক্স"), do not translate or normalize it -- the lookup service handles matching.
- "phone": only if a phone number is explicitly spoken, digits only.
- Never invent a patient name, phone number, or date that was not said.

MULTI-PART:
A caller may ask more than one thing in a single breath -- "uric acid test-er rate koto ar Dr Sen kobe boshen?" is TWO requests joined by "ar". List every request in "parts", IN THE ORDER THE CALLER SAID THEM, at most 3. The first element of "parts" MUST repeat the top-level "intent" and "slots" exactly. If the caller asked only one thing, "parts" has exactly one element. Do not invent a second part to fill the list, and do not add a part for a greeting, a thank-you, or noise you could not understand.

Output ONLY a single valid JSON object, no other text, in exactly this shape:
{{
  "intent": "test_rate" | "doctor_availability" | "doctors_by_department" | "book_appointment" | "smalltalk" | "unclear",
  "slots": {{
    "test_name": string or null,
    "doctor_name": string or null,
    "department": string or null,
    "date_expr": string or null,
    "date": string or null,
    "time_slot": string or null,
    "patient_name": string or null,
    "phone": string or null
  }},
  "parts": [
    {{ "intent": <as above>, "slots": {{ <same eight keys> }} }}
  ],
  "direct_reply_bn": string or null
}}

"direct_reply_bn" must be null for every intent except "smalltalk" -- for every other intent, the reply is composed later from real clinic data, not from you."""


# A yyyy-mm-dd anywhere in the `date` slot. See _validate().
_RE_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class ExtractionError(Exception):
    pass


def _call_ollama(prompt: str, timeout_s: int = 90) -> str:
    # 90s, not 20s: a cold-loaded Qwen2.5:7b (Ollama unloaded it after its
    # default 5-minute idle timeout) measured at 47s just to answer "Say
    # OK" on this pod. The real fix is OLLAMA_KEEP_ALIVE keeping the model
    # resident (see setup docs) so this path is rarely hit in practice --
    # this margin is a backstop for whenever it still is.
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
        for key in ("test_name", "doctor_name", "department", "date_expr", "date", "time_slot", "patient_name", "phone"):
            if key not in slots:
                errors.append(f"slots.{key}: missing")
    # story title: A multi-part question is answered in full
    # user story: As a caller who asked two things, I want both answered, so
    #   that I do not have to ask again.
    # acceptance criteria: Every answerable part of a turn is answered in the
    #   order asked, and any part that cannot be answered is explicitly
    #   addressed rather than dropped. Completeness is scored on a labelled
    #   multi-part set.
    #
    # `parts` is NOT validated into existence and its absence is NOT an error.
    # A model that ignores the field entirely -- an older checkpoint, a cold
    # prompt, a retry that came back terse -- produces exactly the agent that
    # shipped before this story, because turn_parts.normalise() falls back to
    # the top-level intent. Making it required would have turned a degraded
    # extraction into a failed turn, which is a worse trade on a phone line
    # than answering one question out of two.
    #
    # A malformed `parts` is dropped here rather than repaired, for the same
    # reason date_expr is: the log then says what the model produced, and the
    # fallback is a known-good single-part turn instead of a guess at what it
    # meant.
    if "parts" in data and not isinstance(data.get("parts"), list):
        logger.warning("model produced a non-list parts field %r -- dropped",
                       data.get("parts"))
        data.pop("parts", None)

    if data.get("intent") != "smalltalk" and data.get("direct_reply_bn") not in (None, ""):
        # Not fatal -- just strip it. The model overstepping here is the
        # exact failure mode this schema exists to prevent (see module
        # docstring), so we defend in code rather than trust a retry to fix it.
        data["direct_reply_bn"] = None

    # story title: The model never originates a fact
    # user story: As a clinical lead, I want every price, date and identifier
    #   to come from a verified system response, so that a wrong answer is a
    #   data bug rather than a model bug.
    # acceptance criteria: Every factual sentence is a template substitution
    #   from a validated tool response and the model is never shown a figure it
    #   could restate. An automated assertion on every commit proves no
    #   model-composed span reaches synthesis on a factual intent.
    #
    # Two guards on the date, defended in code for the same reason
    # direct_reply_bn is: the prompt asks, and the prompt is not a mechanism.
    if isinstance(slots, dict):
        # (a) An expression outside the closed vocabulary is not a near miss to
        # be salvaged -- date_calc.resolve_expression() would refuse it anyway,
        # and dropping it here means the log says which value the model made up.
        expr = slots.get("date_expr")
        if expr is not None and expr not in DATE_EXPRESSIONS:
            logger.warning("model invented date expression %r -- dropped", expr)
            slots["date_expr"] = None

        # (b) The model has no calendar any more, so a yyyy-mm-dd in `date` can
        # only have been invented. Stripped rather than passed along: this field
        # now carries the caller's own spoken words, and a fabricated date
        # sitting in it would be indistinguishable from one they actually said.
        if isinstance(slots.get("date"), str) and _RE_ISO_DATE.search(slots["date"]):
            logger.warning("model produced a calendar date %r -- dropped; it has no "
                           "calendar, so this was invented", slots["date"])
            slots["date"] = None
    return (len([e for e in errors if "missing" not in e or "intent" in e or "slots: expected" in e]) == 0
            and "slots" in data, errors)


def extract_intent(transcript_bn: str, max_retries: int = 2) -> tuple[dict, dict]:
    """Returns (parsed JSON dict, diagnostics dict)."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(date_expr_list=_DATE_EXPR_LIST)
    prompt = f"{system_prompt}\n\nCALLER UTTERANCE (Bengali, ASR output):\n{transcript_bn}\n\nJSON:"

    diagnostics = {"attempts": 0, "total_time_s": 0.0, "errors": []}
    last_error = None

    for attempt in range(1, max_retries + 2):
        diagnostics["attempts"] = attempt
        t0 = time.time()
        try:
            raw = _call_ollama(prompt)
            diagnostics["total_time_s"] += time.time() - t0
            data = json.loads(raw)
            ok, errors = _validate(data)
            if not ok:
                raise ValueError(f"schema validation failed: {errors}")
            return data, diagnostics
        except Exception as e:  # noqa: BLE001 - retry on anything, log it
            diagnostics["total_time_s"] += time.time() - t0
            last_error = e
            diagnostics["errors"].append(f"attempt {attempt}: {type(e).__name__}: {e}")

    raise ExtractionError(f"intent extraction failed after {diagnostics['attempts']} attempts: {last_error}")
