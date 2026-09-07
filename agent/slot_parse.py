"""Lightweight local parsers for slot values that fill an appointment
booking a field at a time, driven by main.py's CallSession.pending state
machine.

WHY LOCAL PARSING, NOT ANOTHER LLM CALL
----------------------------------------
Once a caller is inside a booking flow, main.py already knows exactly
which single field it just asked for. Re-running the LLM's full
intent+multi-slot extraction on a bare reply like "আজ" or "সাড়ে দশটা" is
both slower (a 7B-model round trip per field) and unreliable: llm.py's
SYSTEM_PROMPT_TEMPLATE classifies intent from cue words like "বুক" /
"অ্যাপয়েন্টমেন্ট", none of which appear in a bare "আজ" -- a caller who is
already three questions into booking is not going to repeat "আমি
অ্যাপয়েন্টমেন্ট করতে চাই" every turn just so the classifier has something
to key off. This is also the root cause of the original bug report: the
LLM sees each utterance in isolation with no memory of the conversation,
so a follow-up like "10 টায়" on its own has no doctor_name/date attached
to it and the LLM has nothing to extract them from.

So each of these functions answers one narrow question -- "does this
utterance look like a date/time/phone number, and if so, which one" --
using the same trust model as fast_path.py: return None whenever not
confident, and let main.py re-prompt (or give up and fall back to a
fresh LLM classification) rather than guess.
"""
from __future__ import annotations

import datetime
import re

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_WEEKDAYS_BN = {
    "সোমবার": 0, "সোম": 0,
    "মঙ্গলবার": 1, "মঙ্গল": 1,
    "বুধবার": 2, "বুধ": 2,
    "বৃহস্পতিবার": 3, "বৃহস্পতি": 3, "বিহস্পতি": 3,
    "শুক্রবার": 4, "শুক্র": 4,
    "শনিবার": 5, "শনি": 5,
    "রবিবার": 6, "রবি": 6,
}

_RELATIVE_DAYS = {
    "আজ": 0, "আজকে": 0, "আজকেই": 0,
    "কাল": 1, "আগামীকাল": 1, "কালকে": 1,
    "পরশু": 2, "পরশুদিন": 2,
}

# Kept deliberately small and exact-match only (see is_affirmative /
# is_negative below) -- these gate whole-utterance decisions like "abandon
# the booking flow", so a false hit on a substring inside an unrelated
# reply (e.g. a patient name that happens to contain "না") would be a much
# worse failure than occasionally not recognising a yes/no.
_AFFIRMATIVE = {"হ্যাঁ", "হ্যা", "হুম", "হুঁ", "ঠিক", "ঠিক আছে", "ওই দিন", "ওইদিন",
                "সেদিন", "সেদিনই", "সেই দিন", "চলবে", "ওকে", "হবে"}
_NEGATIVE = {"না", "নাহ", "না না", "লাগবে না", "থাক", "দরকার নেই", "ইচ্ছা নেই",
             "না থাক", "লাগবে নাহ"}


def _strip(text: str) -> str:
    return text.strip().strip("।!?., ")


def is_affirmative(text: str) -> bool:
    return _strip(text) in _AFFIRMATIVE


def is_negative(text: str) -> bool:
    return _strip(text) in _NEGATIVE


def parse_date(text: str, today: datetime.date | None = None,
                offered_date: str | None = None) -> str | None:
    """-> ISO date string, or None if not confident.

    `offered_date` is the ISO date main.py already spoke out loud (e.g.
    doctor_availability_reply's "next available: 2026-09-08, do you want
    that day or another one?") -- a bare affirmative reply ("হ্যাঁ", "ওই
    দিন") confirms THAT date, not literally "today"."""
    today = today or datetime.date.today()
    t = text.translate(_BN_DIGITS).strip()

    if offered_date and is_affirmative(t):
        return offered_date

    for word, offset in _RELATIVE_DAYS.items():
        if word in t:
            return (today + datetime.timedelta(days=offset)).isoformat()

    for word, weekday in _WEEKDAYS_BN.items():
        if word in t:
            days_ahead = (weekday - today.weekday()) % 7
            days_ahead = days_ahead or 7  # naming today's weekday means NEXT week's
            return (today + datetime.timedelta(days=days_ahead)).isoformat()

    # Explicit ISO date (e.g. carried over from an earlier LLM extraction).
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return None

    # "১৫ তারিখ" / "15 তারিখে" -- day-of-month in the current month,
    # rolling into next month if that day has already passed.
    #
    # (?<!\w) / (?!\w) here, NOT \b: Bengali vowel signs and the nukta
    # (e.g. the ে in তারিখে) are Unicode combining marks, which Python's
    # \w does NOT count as word characters. That makes \b fail to match
    # at the boundary right after them -- so "তারিখে" sitting at the very
    # end of an utterance (the normal case) silently never matched. The
    # lookaround forms only check "not a word character adjacent", which
    # is true at end-of-string and before whitespace/punctuation either
    # way, so they don't have this blind spot. See parse_time() below for
    # the same fix applied to টা/টার/টায়, where it mattered even more.
    m = re.search(r"(?<!\w)(\d{1,2})\s*(?:তারিখ|তারিখে|ই)(?!\w)", t)
    if m:
        day = int(m.group(1))
        if 1 <= day <= 31:
            try:
                candidate = today.replace(day=day)
            except ValueError:
                return None  # e.g. "31 তারিখ" in a 30-day month -- ask again
            if candidate < today:
                next_month = today.month % 12 + 1
                next_year = today.year + (1 if today.month == 12 else 0)
                try:
                    candidate = candidate.replace(year=next_year, month=next_month)
                except ValueError:
                    return None
            return candidate.isoformat()

    return None


_HOUR_WORD_TO_NUM = {
    "একটা": 1, "দুটো": 2, "দুইটা": 2, "তিনটে": 3, "তিনটা": 3, "চারটে": 4, "চারটা": 4,
    "পাঁচটা": 5, "ছটা": 6, "ছয়টা": 6, "সাতটা": 7, "আটটা": 8, "নটা": 9, "নয়টা": 9,
    "দশটা": 10, "এগারোটা": 11, "বারোটা": 12,
}

# Bengali day-part words -> the 24h hours they cover, used only to decide
# whether a bare 1-12 number means AM or PM.
_DAYPART_WORDS = ("সকাল", "দুপুর", "বিকেল", "সন্ধ্যা", "রাত")


def _to_24h(hour12: int, daypart: str | None) -> int:
    hour12 = hour12 % 12
    if daypart in ("দুপুর", "বিকেল", "সন্ধ্যা", "রাত"):
        return hour12 + 12 if hour12 != 0 else 12
    if daypart == "সকাল":
        return hour12 if hour12 != 0 else 12
    # No day-part cue spoken: this clinic's chamber hours run evenings
    # (see clinic-api/seed.py's DoctorSchedule rows, typically 17:00-20:00),
    # so treat a bare 1-7 as PM and 8-12 as AM -- the common case for "the
    # doctor's hours are 6 to 8" said without "সন্ধ্যা" in front of it.
    if 1 <= hour12 <= 7:
        return hour12 + 12
    return hour12 if hour12 != 0 else 12


def _extract_hour12_after(t: str, prefix: str) -> int | None:
    """Look immediately after `prefix` (সাড়ে/সোয়া/পৌনে) for the hour it
    modifies, in EITHER digit form ("সাড়ে ৭টা") or word form ("সাড়ে
    সাতটা") -- real callers and ASR output mix both freely, and the
    original version here only ever checked the digit form, so "সাড়ে
    দশটা" (half past ten, said as a word) silently lost its "half past"
    and was parsed as a bare 10:00."""
    idx = t.find(prefix)
    if idx == -1:
        return None
    rest = t[idx + len(prefix):].lstrip()
    # The digit and টা/টার/টায় are written with NO space between them
    # ("সাড়ে ৭টা"), so a plain (?!\w) right after the digit would wrongly
    # reject it -- ট is a word character. Consume that suffix as PART of
    # the match instead of asserting against it.
    m = re.match(r"(\d{1,2})(?:টা|টার|টায়)?(?!\w)", rest)
    if m:
        return int(m.group(1))
    for word, hour12 in _HOUR_WORD_TO_NUM.items():
        if rest.startswith(word):
            return hour12
    return None


def parse_time(text: str) -> str | None:
    """-> "HH:MM" in 24h, or None if not confident."""
    t = text.translate(_BN_DIGITS).strip()

    m = re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", t)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"

    daypart = next((w for w in _DAYPART_WORDS if w in t), None)

    hour12 = _extract_hour12_after(t, "সাড়ে")
    if hour12 is not None:
        return f"{_to_24h(hour12, daypart):02d}:30"
    hour12 = _extract_hour12_after(t, "সোয়া")
    if hour12 is not None:
        return f"{_to_24h(hour12, daypart):02d}:15"
    hour12 = _extract_hour12_after(t, "পৌনে")
    if hour12 is not None:
        h = _to_24h(hour12, daypart)
        return f"{(h - 1) % 24:02d}:45"

    # (?<!\w) / (?!\w), NOT \b -- see parse_date()'s তারিখ regex above for
    # why. This one matters even more: টা is the single most common way a
    # caller states an o'clock hour ("৭টা", "সন্ধ্যা ৭টা"), and it almost
    # always sits at the very end of the utterance -- exactly where \b
    # after a combining vowel sign (the া in টা) silently failed to match.
    # Confirmed by hand: "সন্ধ্যা ৭টা" and even a bare "৭টা" both returned
    # None under the old \b version.
    m = re.search(r"(?<!\w)(\d{1,2})\s*(?:টা|টার|টায়)(?!\w)", t)
    if m:
        return f"{_to_24h(int(m.group(1)), daypart):02d}:00"

    for word, hour12 in _HOUR_WORD_TO_NUM.items():
        if word in t:
            return f"{_to_24h(hour12, daypart):02d}:00"

    # Transliterated/English callers: "10 am", "10am", "10 pm".
    m = re.search(r"\b(\d{1,2})\s*([ap])\.?m\.?\b", t, re.IGNORECASE)
    if m:
        h = int(m.group(1)) % 12
        if m.group(2).lower() == "p":
            h += 12
        return f"{h:02d}:00"

    return None


def parse_phone(text: str) -> str | None:
    """-> a 10-digit phone number, or None if the utterance doesn't
    contain enough digits to be one."""
    digits = re.sub(r"\D", "", text.translate(_BN_DIGITS))
    if len(digits) < 10:
        return None
    return digits[-10:]  # tolerate a spoken +91 / leading 0 trunk prefix
