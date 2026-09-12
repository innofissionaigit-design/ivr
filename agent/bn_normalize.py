"""Verbalize a reply string into something the Bengali TTS can actually SAY.

This is not a cosmetic prettifier -- it fixes a proven, silent data-loss
bug. AI4Bharat's Bengali FastPitch tokenizer drops Latin digits entirely.
Measured on the live pod (22050Hz mono 16-bit, so bytes/2/22050 = seconds):

    "রেট টাকা।"            -> 102476 bytes   (no number at all)
    "রেট 250 টাকা।"        -> 100940 bytes   <- identical...
    "রেট 987654 টাকা।"     -> 100940 bytes   <- ...to a 6-digit number
    "রেট দুইশো পঞ্চাশ টাকা।" -> 143948 bytes   (spelled out: actually spoken)

Two different numbers producing the same audio, shorter than the sentence
with the number removed, is conclusive: every price, report time, chamber
hour and confirmation number this agent has ever "spoken" was silence.
Callers heard "রেট ___ টাকা" and reported it as the agent skipping words.

So every number reaching TTS gets spelled into Bengali words HERE, in
code, before synthesis. Deliberately not asked of the LLM: the model is
never allowed to restate a figure (see llm.py's module docstring), and
that rule does not get to be quietly relaxed just because the figure
needs reformatting.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- numbers

_ONES_TO_99 = [
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়",
    "দশ", "এগারো", "বারো", "তেরো", "চোদ্দো", "পনেরো", "ষোলো", "সতেরো", "আঠারো", "উনিশ",
    "কুড়ি", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আটাশ", "ঊনত্রিশ",
    "ত্রিশ", "একত্রিশ", "বত্রিশ", "তেত্রিশ", "চৌত্রিশ", "পঁয়ত্রিশ", "ছত্রিশ", "সাঁইত্রিশ", "আটত্রিশ", "ঊনচল্লিশ",
    "চল্লিশ", "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ", "চুয়াল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ", "ঊনপঞ্চাশ",
    "পঞ্চাশ", "একান্ন", "বাহান্ন", "তিপ্পান্ন", "চুয়ান্ন", "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "ঊনষাট",
    "ষাট", "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছেষট্টি", "সাতষট্টি", "আটষট্টি", "ঊনসত্তর",
    "সত্তর", "একাত্তর", "বাহাত্তর", "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর", "আটাত্তর", "ঊনআশি",
    "আশি", "একাশি", "বিরাশি", "তিরাশি", "চুরাশি", "পঁচাশি", "ছিয়াশি", "সাতাশি", "অষ্টাশি", "ঊননব্বই",
    "নব্বই", "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই", "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই", "নিরানব্বই",
]

_HUNDREDS = [
    "", "একশো", "দুইশো", "তিনশো", "চারশো", "পাঁচশো", "ছয়শো", "সাতশো", "আটশো", "নয়শো",
]

# Bengali digit glyphs -> ASCII, so ২৫০ and 250 take the same path.
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def number_to_bn_words(n: int) -> str:
    """Indian numbering system (হাজার / লাখ / কোটি), not the short scale."""
    if n < 0:
        return "মাইনাস " + number_to_bn_words(-n)
    if n < 100:
        return _ONES_TO_99[n]
    if n < 1000:
        head, rest = divmod(n, 100)
        out = _HUNDREDS[head]
        return out if rest == 0 else f"{out} {number_to_bn_words(rest)}"
    for divisor, word in ((10_000_000, "কোটি"), (100_000, "লাখ"), (1_000, "হাজার")):
        if n >= divisor:
            head, rest = divmod(n, divisor)
            out = f"{number_to_bn_words(head)} {word}"
            return out if rest == 0 else f"{out} {number_to_bn_words(rest)}"
    return str(n)  # unreachable


# Latin letters read aloud in Bengali. Confirmation IDs ("KCD-4471") are
# the only place these reach TTS, and without this the letters vanish the
# same way the digits did -- the caller hears four digits and no prefix.
_LETTER_BN = {
    "a": "এ", "b": "বি", "c": "সি", "d": "ডি", "e": "ই", "f": "এফ", "g": "জি",
    "h": "এইচ", "i": "আই", "j": "জে", "k": "কে", "l": "এল", "m": "এম",
    "n": "এন", "o": "ও", "p": "পি", "q": "কিউ", "r": "আর", "s": "এস",
    "t": "টি", "u": "ইউ", "v": "ভি", "w": "ডব্লিউ", "x": "এক্স", "y": "ওয়াই", "z": "জেড",
}


# story title: Figures are spoken at a pace a caller can write down
# user story: As a patient noting a price or a reference, I want it grouped
#   and slower, so that I do not have to ask twice.
# acceptance criteria: Prices, phone numbers and reference identifiers are
#   spoken with grouping and a reduced rate through the per-request speed
#   parameter. A listening test confirms callers transcribe correctly on
#   first hearing.
#
# GROUPING POLICY -- PROVISIONAL, PENDING THE LISTENING STUDY.
#
# None of the numbers below is an acceptance criterion. The story asks for
# "grouping"; how many digits go in a group is a design decision, and these
# are the ones a Kolkata caller is most likely to expect rather than ones
# anybody has measured. They are gathered here, in one table with a version
# on it, precisely so the study can move them without touching any logic --
# and so a result can say WHICH policy it validated. See
# docs/listening-test.md.
#
# THE SEPARATOR IS A COMMA, and that is a deliberate REUSE of an existing
# meaning rather than a free win. tts_server._split_for_prosody already
# treats "," as a clause boundary and splices PAUSE_S["clause"] of real
# silence there, so a comma inside a number produces an audible group break
# with no server change at all. But the server cannot tell a grouping comma
# from a grammatical one, and it gives both the same pause. Whether a
# clause-length gap is the right gap between digit groups is exactly the
# kind of question only a handset can answer.
#
# One consequence worth knowing before listening: a sentence that had no
# comma becomes clause-split as a whole once a grouped number is in it, so
# the words AROUND the number gain a boundary too. That is the intended
# effect -- a small breath before a number is what a person does -- but it
# is a prosody change to the carrier sentence, not only to the figure.
GROUPING_POLICY_VERSION = "2026-09-12.a"

GROUP_SEPARATOR = ","

# Digit-count -> group sizes, for lengths with a convention worth honouring.
# Ten digits is an Indian mobile number and 5+5 is how one is written on a
# form and read back at a counter.
_GROUPS_BY_LENGTH = {10: (5, 5)}

# Everything else is cut into near-equal groups no longer than this.
MAX_GROUP_DIGITS = 4


def _group_sizes(n: int) -> tuple[int, ...]:
    """-> how many characters go in each spoken group."""
    if n <= 0:
        return ()
    if n in _GROUPS_BY_LENGTH:
        return _GROUPS_BY_LENGTH[n]
    if n <= MAX_GROUP_DIGITS:
        return (n,)
    # Near-equal rather than "fours with a stray remainder": a 9-digit run
    # reads better as 3+3+3 than as 4+4+1, and a group of one is not a group.
    count = -(-n // MAX_GROUP_DIGITS)
    base, extra = divmod(n, count)
    return tuple(base + (1 if i < extra else 0) for i in range(count))


def _in_groups(chars: str) -> list[str]:
    out, index = [], 0
    for size in _group_sizes(len(chars)):
        out.append(chars[index:index + size])
        index += size
    return out


def _read_chars(chars: str) -> str:
    spoken = []
    for ch in chars:
        if ch.isdigit():
            spoken.append(_ONES_TO_99[int(ch)])
        elif ch.isalpha() and ch.lower() in _LETTER_BN:
            spoken.append(_LETTER_BN[ch.lower()])
    return " ".join(spoken)


def spell_out(s: str) -> str:
    """Character-by-character, the way an ID is read over a phone -- now in
    the groups the ID itself already has.

    CONTRACT CHANGE, stated rather than slipped in: this used to return a
    flat run of words and now returns a GROUPED one. The old behaviour threw
    away the hyphens in "KCD-20260914-4A2F" before speech, so a caller heard
    fifteen tokens in one breath with no boundary where their pen would
    pause -- the structure was in the string and was discarded on the way to
    the synthesiser.

    Group boundaries come from the identifier's own separators first, and
    any run longer than MAX_GROUP_DIGITS is cut again so no single group is
    too long to hold. Nothing is reordered and no character is dropped: the
    digit-fidelity guarantee in tests/test_number_fidelity.py is unchanged,
    and this function is still the one place that decides how an ID sounds.
    """
    groups = []
    for part in re.split(r"[^0-9A-Za-z]+", s):
        if not part:
            continue
        groups.extend(_read_chars(chunk) for chunk in _in_groups(part))
    return f"{GROUP_SEPARATOR} ".join(g for g in groups if g)


def digits_one_by_one(s: str) -> str:
    """Digit-by-digit, ungrouped, the way a person reads a short run aloud --
    "নয় আট সাত" not "নয়শো সাতাশি".

    Deliberately NOT grouped, and left exactly as it was. Its other caller is
    the fractional half of a decimal, where a group separator would be wrong
    -- "দুইশো পঞ্চাশ দশমিক পাঁচ, শূন্য" is not a thing anyone says. Grouping
    is opt-in through grouped_digits() below, so the decimal path stays
    byte-identical to what it produced before this story.
    """
    return " ".join(_ONES_TO_99[int(c)] if c.isdigit() else c for c in s if not c.isspace())


def grouped_digits(s: str) -> str:
    """Digit-by-digit, in groups, for a number the caller is writing down."""
    digits = "".join(c for c in s if not c.isspace())
    return f"{GROUP_SEPARATOR} ".join(_read_chars(chunk) for chunk in _in_groups(digits))


# ------------------------------------------------------------------ time

_HOUR_WORD = {
    1: "একটা", 2: "দুটো", 3: "তিনটে", 4: "চারটে", 5: "পাঁচটা", 6: "ছটা",
    7: "সাতটা", 8: "আটটা", 9: "নটা", 10: "দশটা", 11: "এগারোটা", 12: "বারোটা",
}


def _day_part(hour24: int) -> str:
    if 4 <= hour24 < 12:
        return "সকাল"
    if 12 <= hour24 < 16:
        return "দুপুর"
    if 16 <= hour24 < 18:
        return "বিকেল"
    if 18 <= hour24 < 20:
        return "সন্ধ্যা"
    return "রাত"


def time_to_bn_words(hh: int, mm: int) -> str:
    """Bengali speakers say সাড়ে/সোয়া/পৌনে for :30/:15/:45 -- reading
    "দশটা ত্রিশ মিনিট" instead is understandable but immediately marks the
    voice as a machine."""
    part = _day_part(hh)
    h12 = hh % 12 or 12
    if mm == 0:
        return f"{part} {_HOUR_WORD[h12]}"
    if mm == 30:
        return f"{part} সাড়ে {_HOUR_WORD[h12]}"
    if mm == 15:
        return f"{part} সোয়া {_HOUR_WORD[h12]}"
    if mm == 45:
        nxt = (h12 % 12) + 1
        return f"{_day_part((hh + 1) % 24)} পৌনে {_HOUR_WORD[nxt]}"
    return f"{part} {_HOUR_WORD[h12]} বেজে {number_to_bn_words(mm)} মিনিট"


# ------------------------------------------------------------------ date

_MONTHS_BN = [
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
    "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
]


def date_to_bn_words(y: int, m: int, d: int) -> str:
    if not 1 <= m <= 12:
        return f"{number_to_bn_words(d)} তারিখ"
    return f"{_MONTHS_BN[m - 1]} মাসের {number_to_bn_words(d)} তারিখ"


# ------------------------------------------------------------- the pass

_RE_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b(\s*তারিখে?)?")
_RE_TIME_RANGE = re.compile(r"\b(\d{1,2}):(\d{2})\s*[-–—to]{1,2}\s*(\d{1,2}):(\d{2})\b")
_RE_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_RE_PHONE = re.compile(r"\b(\d{10,})\b")
# story title: Numbers are never rounded, reordered or approximated
# user story: As a patient, I want the exact figure, so that what I am quoted
#   is what I pay.
# acceptance criteria: Figures pass from the validated response into the
#   template unchanged and are verbalised digit-faithfully. A test asserts
#   byte-level equality between the tool value and the spoken value for a
#   corpus of amounts, dates and identifiers.
#
# Was r"\b([A-Z]{2,}[-]?\d{3,})\b", which captured only the FIRST
# hyphen-separated group of a confirmation ID. clinic-api generates
# f"KCD-{date}-{uuid4().hex[:4].upper()}", so on this branch the old pattern
# produced two different corruptions and both were measured:
#
#   KCD-20260911-0031  ->  "কে সি ডি ...-একত্রিশ"   the trailing group fell
#                          through to the bare-integer sweep and was spoken as
#                          the WORD "thirty-one", leading zeros gone.
#
#   KCD-20260911-4A2F  ->  "কে সি ডি ...-চারAদুইF"  no match at all, because
#                          "4A2F" is not \d{3,}. The letters stay Latin, the
#                          tokenizer drops them, and agent/speakability.py
#                          BLOCKS the reply -- so every successful booking
#                          escalated to the counter instead of giving the
#                          caller their number.
#
# The second case is 100% of real IDs here, not an edge case. dev_sourav
# deferred it as "~85% of IDs, hex needs letter spelling"; that reasoning does
# not apply to this branch, because spell_out() has always handled A-F
# correctly via _LETTER_BN -- spell_out("KCD-4A2F") is "কে সি ডি চার এ দুই এফ".
# The letters were never the problem. The ID simply never reached spell_out.
#
# WHY THE DIGIT GUARD SITS ON THE PREFIX AND NOT ON EVERY GROUP: a bare
# [0-9A-F] run would match "DEADBEEF", and any all-caps word that happens to
# be hex. So the PREFIX still demands [A-Z]{2,} followed by \d{3,} -- that is
# what tells an identifier from a word, and it means nothing that matched the
# old pattern stops matching. Once the prefix has established this IS an ID,
# the trailing groups need no further proof and may be any hex, letters
# included: requiring a digit in them left the ~2% of IDs whose last four hex
# characters happen to be all A-F still broken, for no safety gained. Measured
# over 2000 generated IDs in the real format: 36 blocked before this line,
# 0 after.
_RE_CONF_ID = re.compile(r"\b([A-Z]{2,}-?\d{3,}(?:-[0-9A-F]+)*)\b")
_RE_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")
_RE_INT = re.compile(r"\d+")

# Latin fragments that survive in clinic data (test names like "Uric Acid",
# "CBC", sample types like "Blood"). The Bengali tokenizer drops these the
# same way it drops digits, so anything still in Latin script after
# verbalization is a word the caller will never hear. The lookup service
# owns the Bengali aliases (clinic-api seeds aliases_bn); this table is the
# last-resort spoken form for the handful of fields the API returns in
# English regardless of how the caller phrased the question.
#
# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
# THIS TABLE IS THE "REWRITE" HALF OF E12-S2.
# Every entry is hand-chosen for a CLOSED, enumerated set -- there are five
# sample types in the catalogue and eight departments, and they are known in
# advance. That is what separates this from transliterating arbitrary text at
# runtime, which this codebase does not do: an automatic transliteration turns
# a detectable hole into a confident mispronunciation, which is worse. When a
# span is not in this table the reply is BLOCKED, not guessed at -- see
# agent/speakability.py.
#
# The Bengali-script renderings of borrowed clinical words ("ইমেজিং",
# "কার্ডিয়াক") are how a Kolkata caller actually says them, rather than a
# translation into literary Bengali they would not use -- the same principle
# E2-S4E states for the caller's own borrowings. They are NOT yet signed off
# by a native listener; that sign-off is E2-S4F's criterion and is outstanding.
_LATIN_SPOKEN_BN = {
    "blood": "রক্ত",
    "urine": "মূত্র",
    "stool": "মল",
    "serum": "সিরাম",
    "saliva": "লালা",
    "swab": "সোয়াব",
    "plasma": "প্লাজমা",
    # Sample-type column values that are really modalities, not samples.
    # Reading them after a "স্যাম্পল:" label is awkward and E12-S3 rewrites
    # the sentence; this makes them AUDIBLE, which is a different job.
    "cardiac": "কার্ডিয়াক",
    "imaging": "ইমেজিং",
    "cervical smear": "সার্ভাইকাল স্মিয়ার",
}


def _sub_int(match: re.Match) -> str:
    return number_to_bn_words(int(match.group(0)))


def verbalize(text: str) -> str:
    """Rewrite `text` so every token in it is actually pronounceable by the
    Bengali FastPitch model. Order matters: the most specific patterns
    (dates, time ranges, long digit runs) must run before the bare-integer
    sweep, or "2026-08-25" gets read as three unrelated numbers."""
    if not text:
        return text

    text = text.translate(_BN_DIGITS)
    text = text.replace("₹", " টাকা ").replace("%", " শতাংশ ")

    # group(4) is a "তারিখ"/"তারিখে" the template already supplied. Absorb it:
    # date_to_bn_words ends in "তারিখ", so leaving it produces "... তারিখ তারিখে".
    text = _RE_DATE.sub(
        lambda m: date_to_bn_words(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        + ("ে" if (m.group(4) or "").strip().endswith("ে") else ""), text,
    )
    text = _RE_TIME_RANGE.sub(
        lambda m: (f"{time_to_bn_words(int(m.group(1)), int(m.group(2)))} থেকে "
                   f"{time_to_bn_words(int(m.group(3)), int(m.group(4)))} পর্যন্ত"), text,
    )
    text = _RE_TIME.sub(lambda m: time_to_bn_words(int(m.group(1)), int(m.group(2))), text)
    text = _RE_CONF_ID.sub(lambda m: spell_out(m.group(1)), text)
    # story title: Figures are spoken at a pace a caller can write down
    # A phone number is the figure a caller most often has to write down,
    # so it reads in groups. The decimal branch below deliberately keeps
    # the ungrouped reader -- see digits_one_by_one.
    text = _RE_PHONE.sub(lambda m: grouped_digits(m.group(1)), text)
    text = _RE_DECIMAL.sub(
        lambda m: f"{number_to_bn_words(int(m.group(1)))} দশমিক {digits_one_by_one(m.group(2))}", text,
    )
    text = _RE_INT.sub(_sub_int, text)

    # Whole-word, case-insensitive: only rewrites a Latin word we have a
    # spoken Bengali form for. Anything else Latin is left alone and
    # reported by `unspeakable_spans()` rather than silently mangled.
    #
    # STORY [Answer Quality and Grounding]
    # As a patient, I want to hear the whole sentence, so that I am
    # not left guessing what the agent tried to say.
    # Longest key first, and the key is escaped: entries may be PHRASES
    # ("cervical smear"), and a shorter key that is a prefix of a longer one
    # would otherwise consume half of it and leave the remainder stranded as
    # an unspeakable span -- a rewrite that manufactures the exact defect the
    # table exists to remove.
    for latin in sorted(_LATIN_SPOKEN_BN, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(latin)}\b", _LATIN_SPOKEN_BN[latin],
                      text, flags=re.IGNORECASE)

    return re.sub(r"\s{2,}", " ", text).strip()


_RE_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z .'-]*")


# STORY [Answer Quality and Grounding]
# As a patient, I want to hear the whole sentence, so that I am
# not left guessing what the agent tried to say.
def unspeakable_spans(text: str) -> list[str]:
    """Latin-script runs left after verbalize() -- these WILL be dropped
    silently by the tokenizer, exactly like the digits were.

    SINGLE LETTERS COUNT. This used to filter out anything one character
    long, which was a blind spot rather than noise reduction: the tokenizer
    drops a lone "T" exactly as completely as it drops "TSH", and the
    catalogue really does contain names shaped that way -- "Thyroid Profile
    (T3 T4 TSH)" verbalizes to "(Tতিন Tচার TSH)", of which the old filter
    reported only "TSH" and let both stray T's through unrecorded. Anything
    built on top of this function inherits its blind spots, so a gate that
    blocks unspeakable replies could not be trusted while this one existed.

    KNOWN LIMIT: this models Latin-versus-Bengali only, because the matrix
    language is currently hardcoded Bengali (agent/asr.py). It does not see
    Devanagari, symbols, or anything else outside the checkpoint's vocabulary
    -- those are caught, if at all, by tts_server.py's empty-chunk log.
    """
    return [s for s in (m.group(0).strip() for m in _RE_LATIN_RUN.finditer(text)) if s]
